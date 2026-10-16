"""
SQUADRONE — Corridor Navigation Module
File: corridor_nav.py

Uses 3x TF Mini S LiDAR (I2C) for obstacle avoidance
inside the 6-foot (183cm) wide AeroTHON corridor.

Drone dimensions (prop-to-prop):
    Front/back clearance: ~11cm from centre
    Left/right clearance: ~15cm from centre
    Usable space per side in corridor: ~76cm

Fixes applied:
    - Noise filter: ignore LiDAR reads < 10cm (eliminates glitch spikes)
    - WALL_PROXIMITY_CM raised to 60cm (corrects earlier, more margin)
    - Velocity uses MAV_FRAME_BODY_OFFSET_NED (body frame) so lateral
      corrections are always relative to drone nose — works regardless
      of which compass direction the corridor faces
    - set_mode uses msg.custom_mode (no deprecated mode_string_v10)
    - import cv2 at top of file

Integrates with:
    - mavlink_control.py  (send_velocity, set_mode, get_altitude)
    - banner_detect.py    (detect_green_banner, is_aligned)

States handled:
    S3 — Corridor Entry (forward lap)
    S4 — Corridor Navigation (forward lap)
    S8 — Corridor Entry (return lap)
    S9 — Corridor Navigation (return lap)
"""

import cv2
import time
import logging

from smbus2 import SMBus, i2c_msg
from pymavlink import mavutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ─── LIDAR CONFIG ─────────────────────────────────────────────────
BUS_NUMBER = 1

ADDR_LEFT  = 0x11
ADDR_FRONT = 0x12
ADDR_RIGHT = 0x14

LIDAR_MIN_CM = 10   # ignore reads below this — noise / glitch filter

# ─── CORRIDOR THRESHOLDS ──────────────────────────────────────────
# Corridor width: 6 feet = 183cm
# Drone width (prop-to-prop left/right): ~30cm
# Usable space per side: ~76cm
FRONT_OBSTACLE_CM   = 150   # stop if obstacle within 1.5m ahead
WALL_PROXIMITY_CM   = 60    # correct if wall within 60cm (was 50)
CORRIDOR_EXIT_CM    = 1000  # corridor exited when front clears 10m

# ─── FLIGHT PARAMETERS ────────────────────────────────────────────
CORRIDOR_ALTITUDE_M = 3.0   # fly at 3m inside corridor
FORWARD_SPEED       = 0.5   # m/s — max forward speed
CORRECTION_SPEED    = 0.3   # m/s — lateral correction speed
APPROACH_SPEED      = 0.3   # m/s — speed during banner approach

# Banner alignment
BANNER_ALIGN_THRESHOLD_PX = 30    # pixels
BANNER_ALIGN_HOLD_FRAMES  = 10    # frames to hold before entering

# ─── LIDAR READING ────────────────────────────────────────────────
OBTAIN_DATA_CMD = [0x5A, 0x05, 0x00, 0x01, 0x60]

def read_tfmini_s(bus, address):
    """
    Read distance from TF Mini S. Returns cm or None.
    Ignores reads below LIDAR_MIN_CM — these are glitch spikes
    (e.g. the 5cm spike seen in bench test at line 98).
    """
    try:
        bus.i2c_rdwr(i2c_msg.write(address, OBTAIN_DATA_CMD))
        time.sleep(0.002)
        read_msg = i2c_msg.read(address, 9)
        bus.i2c_rdwr(read_msg)
        data = list(read_msg)
        if (len(data) == 9 and
                data[0] == 0x59 and data[1] == 0x59 and
                (sum(data[0:8]) & 0xFF) == data[8]):
            dist = data[2] + (data[3] << 8) + 5
            if dist < LIDAR_MIN_CM:
                return None   # noise spike — treat as no reading
            return dist
    except Exception:
        pass
    return None

def read_all_lidars(bus):
    """
    Read all three LiDARs.
    Returns (front_cm, left_cm, right_cm).
    None means sensor error or noise spike for that unit.
    """
    front = read_tfmini_s(bus, ADDR_FRONT)
    left  = read_tfmini_s(bus, ADDR_LEFT)
    right = read_tfmini_s(bus, ADDR_RIGHT)
    return front, left, right

def log_lidar(front, left, right):
    f = f"{front:4d}" if front is not None else " ERR"
    l = f"{left:4d}"  if left  is not None else " ERR"
    r = f"{right:4d}" if right is not None else " ERR"
    logging.info(f"LiDAR | F:{f}cm  L:{l}cm  R:{r}cm")

# ─── MAVLINK HELPERS ──────────────────────────────────────────────
def send_velocity(mav, forward, right, down=0.0, duration_s=0.2):
    """
    Send BODY-frame velocity setpoint.
    forward = drone's nose direction (always forward regardless of heading)
    right   = drone's right side (always right regardless of heading)
    down    = positive = descend

    Using MAV_FRAME_BODY_OFFSET_NED so corrections work regardless of
    which compass direction the corridor faces.
    """
    end_time = time.time() + duration_s
    while time.time() < end_time:
        mav.mav.set_position_target_local_ned_send(
            0,
            mav.target_system,
            mav.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0000111111000111,
            0, 0, 0,
            forward, right, down,
            0, 0, 0,
            0, 0
        )
        time.sleep(0.1)

def stop_drone(mav):
    """Send zero velocity — drone holds position."""
    send_velocity(mav, 0, 0, 0, duration_s=0.3)

def set_mode(mav, mode_name):
    mode_id = mav.mode_mapping()[mode_name]
    mav.mav.set_mode_send(
        mav.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )
    start = time.time()
    while time.time() - start < 5:
        msg = mav.recv_match(
            type='HEARTBEAT', blocking=True, timeout=1
        )
        if msg and msg.custom_mode == mode_id:
            logging.info(f"Mode set: {mode_name}")
            return True
    logging.warning(f"Mode change to {mode_name} not confirmed")
    return False

def descend_to_altitude(mav, target_alt_m, tolerance=0.3):
    """Descend to target altitude and hold."""
    logging.info(f"Descending to {target_alt_m}m")
    set_mode(mav, 'GUIDED')

    mav.mav.set_position_target_local_ned_send(
        0,
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111111000,   # ignore XY, only Z
        0, 0, -target_alt_m,  # NED: negative Z = up
        0, 0, 0,
        0, 0, 0,
        0, 0
    )

    start = time.time()
    while time.time() - start < 15:
        msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
        if msg:
            current = msg.relative_alt / 1000.0
            logging.info(f"  Alt: {current:.2f}m")
            if abs(current - target_alt_m) < tolerance:
                logging.info(f"Reached {current:.2f}m")
                return True
    logging.warning("Altitude target not reached")
    return False

# ─── COMPUTE CORRIDOR CORRECTION ──────────────────────────────────
def compute_correction(front, left, right):
    """
    Given three LiDAR readings compute body-frame velocity commands.

    Returns (vforward, vright, should_stop, corridor_exited)
    vforward = forward speed along drone nose
    vright   = lateral speed (positive = drone's right)
    should_stop     = True if obstacle too close ahead
    corridor_exited = True if front > CORRIDOR_EXIT_CM
    """
    vforward = FORWARD_SPEED
    vright   = 0.0
    should_stop      = False
    corridor_exited  = False

    # Exit detection
    if front is not None and front > CORRIDOR_EXIT_CM:
        return 0.0, 0.0, False, True

    # Front obstacle check
    if front is not None and front < FRONT_OBSTACLE_CM:
        should_stop = True
        logging.info(f"  STOP — obstacle at {front}cm")
        return 0.0, 0.0, True, False

    # Lateral wall correction
    # Both walls close — bias toward side with more space
    if (left is not None and left < WALL_PROXIMITY_CM and
            right is not None and right < WALL_PROXIMITY_CM):
        if left > right:
            vright = -CORRECTION_SPEED   # more space on left — move left
        else:
            vright =  CORRECTION_SPEED   # more space on right — move right

    elif left is not None and left < WALL_PROXIMITY_CM:
        vright = CORRECTION_SPEED        # too close left — move right
        logging.info(f"  Left wall at {left}cm — correcting right")

    elif right is not None and right < WALL_PROXIMITY_CM:
        vright = -CORRECTION_SPEED       # too close right — move left
        logging.info(f"  Right wall at {right}cm — correcting left")

    return vforward, vright, False, False

# ─── STATE S3 / S8 — CORRIDOR ENTRY ──────────────────────────────
def state_corridor_entry(mav, cam_ai, detect_banner_fn,
                         is_aligned_fn, lap="forward"):
    """
    Approach corridor entrance and align with green banner.

    cam_ai           — picamera2 instance for AI Camera
    detect_banner_fn — function from banner_detect.py
    is_aligned_fn    — function from banner_detect.py
    lap              — "forward" or "return"

    Returns True when aligned and ready to enter.
    """
    logging.info(f"\n── S{'3' if lap=='forward' else '8'}: "
                 f"Corridor Entry ({lap}) ──")

    set_mode(mav, 'GUIDED')

    # Descend to corridor altitude
    descend_to_altitude(mav, CORRIDOR_ALTITUDE_M)

    aligned_count = 0

    while True:
        frame = cam_ai.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        detected, cx, cy, area, yaw_offset = detect_banner_fn(frame)

        if not detected:
            # No banner — move forward slowly to search
            send_velocity(mav, APPROACH_SPEED, 0, duration_s=0.3)
            aligned_count = 0
            logging.info("  Banner not detected — approaching")
            continue

        logging.info(f"  Banner detected | offset={yaw_offset}px")

        if is_aligned_fn(yaw_offset, threshold_px=BANNER_ALIGN_THRESHOLD_PX):
            aligned_count += 1
            logging.info(f"  Aligned {aligned_count}/{BANNER_ALIGN_HOLD_FRAMES}")

            if aligned_count >= BANNER_ALIGN_HOLD_FRAMES:
                stop_drone(mav)
                logging.info("  CORRIDOR ENTRY CONFIRMED")
                return True
        else:
            aligned_count = 0
            # Yaw correction toward banner centre
            yaw_rate = 0.3 if yaw_offset > 0 else -0.3
            mav.mav.command_long_send(
                mav.target_system,
                mav.target_component,
                mavutil.mavlink.MAV_CMD_CONDITION_YAW,
                0,
                abs(yaw_offset * 0.05),       # angle degrees (scaled)
                20,                             # speed deg/s
                1 if yaw_offset > 0 else -1,   # direction
                1,                              # relative
                0, 0, 0
            )
            time.sleep(0.15)

# ─── STATE S4 / S9 — CORRIDOR NAVIGATION ─────────────────────────
def state_corridor_navigation(mav, bus, lap="forward"):
    """
    Navigate through corridor using 3x LiDAR.
    Handles static obstacles.

    bus  — open SMBus instance
    lap  — "forward" or "return"

    Returns True when corridor exit detected.
    """
    logging.info(f"\n── S{'4' if lap=='forward' else '9'}: "
                 f"Corridor Navigation ({lap}) ──")

    set_mode(mav, 'GUIDED')

    obstacle_hold_start = None
    OBSTACLE_RETRY_S    = 2.0   # wait 2s before retrying after stop

    while True:
        front, left, right = read_all_lidars(bus)
        log_lidar(front, left, right)

        vforward, vright, should_stop, exited = compute_correction(
            front, left, right
        )

        if exited:
            stop_drone(mav)
            logging.info("  CORRIDOR EXIT DETECTED")
            return True

        if should_stop:
            stop_drone(mav)

            if obstacle_hold_start is None:
                obstacle_hold_start = time.time()
            elif time.time() - obstacle_hold_start > OBSTACLE_RETRY_S:
                logging.info("  Retrying after obstacle hold")
                obstacle_hold_start = None
                # Nudge toward side with more space
                if left is not None and right is not None:
                    if left > right:
                        send_velocity(mav, 0, -CORRECTION_SPEED,
                                      duration_s=0.5)
                    else:
                        send_velocity(mav, 0, CORRECTION_SPEED,
                                      duration_s=0.5)
        else:
            obstacle_hold_start = None
            send_velocity(mav, vforward, vright, duration_s=0.2)

        time.sleep(0.05)

# ─── STANDALONE TEST ──────────────────────────────────────────────
if __name__ == "__main__":
    """
    Standalone LiDAR bench test — no flight, no MAVLink needed.
    Run this on the bench to verify all 3 LiDARs and correction logic.

    How to test:
      1. python3 corridor_nav.py
      2. Hold hand in front  → should show STOP when < 150cm
      3. Hold hand on left   → should show 'correcting right' when < 60cm
      4. Hold hand on right  → should show 'correcting left'  when < 60cm
      5. Clear all sensors   → front reads 1000+cm → should show EXIT
      6. Hold < 10cm         → should return None (noise filter)
    """
    print("Corridor Navigation — LiDAR bench test (no flight)")
    print(f"Thresholds: FRONT_STOP={FRONT_OBSTACLE_CM}cm  "
          f"WALL_CORRECT={WALL_PROXIMITY_CM}cm  "
          f"EXIT={CORRIDOR_EXIT_CM}cm")
    print("Ctrl+C to quit\n")

    with SMBus(BUS_NUMBER) as bus:
        while True:
            front, left, right = read_all_lidars(bus)

            vf, vr, stop_flag, exited = compute_correction(
                front, left, right
            )

            f = f"{front:4d}cm" if front is not None else " ERR "
            l = f"{left:4d}cm"  if left  is not None else " ERR "
            r = f"{right:4d}cm" if right is not None else " ERR "

            status = "EXIT"  if exited    else \
                     "STOP"  if stop_flag else \
                     "GO"

            print(f"F:{f} L:{l} R:{r} | "
                  f"fwd={vf:.1f} right={vr:+.1f} | {status}    ",
                  end="\r", flush=True)

            time.sleep(0.05)