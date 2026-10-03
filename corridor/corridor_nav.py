"""
SQUADRONE — Corridor Navigation Module
File: corridor_nav.py

Uses 3x TF Mini S LiDAR (I2C) for obstacle avoidance
inside the 3.5m wide AeroTHON corridor.

Integrates with:
    - mavlink_control.py  (send_velocity, set_mode, get_altitude)
    - banner_detect.py    (detect_green_banner, is_aligned)

States handled:
    S3 — Corridor Entry (forward lap)
    S4 — Corridor Navigation (forward lap)
    S8 — Corridor Entry (return lap)
    S9 — Corridor Navigation (return lap)
"""

from smbus2 import SMBus, i2c_msg
from pymavlink import mavutil
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ─── LIDAR CONFIG ─────────────────────────────────────────────────
BUS_NUMBER = 1

ADDR_LEFT  = 0x11
ADDR_FRONT = 0x12
ADDR_RIGHT = 0x14

# ─── CORRIDOR THRESHOLDS ──────────────────────────────────────────
FRONT_OBSTACLE_CM   = 150   # stop if obstacle within 1.5m
WALL_PROXIMITY_CM   = 50    # correct if wall within 0.5m
CORRIDOR_EXIT_CM    = 1000  # corridor exited when front > 10m

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
    """Read distance from TF Mini S. Returns cm or None."""
    try:
        bus.i2c_rdwr(i2c_msg.write(address, OBTAIN_DATA_CMD))
        time.sleep(0.002)
        read_msg = i2c_msg.read(address, 9)
        bus.i2c_rdwr(read_msg)
        data = list(read_msg)
        if (len(data) == 9 and
                data[0] == 0x59 and data[1] == 0x59 and
                (sum(data[0:8]) & 0xFF) == data[8]):
            return data[2] + (data[3] << 8)
    except Exception:
        pass
    return None

def read_all_lidars(bus):
    """
    Read all three LiDARs.
    Returns (front_cm, left_cm, right_cm).
    None means sensor error for that unit.
    """
    front = read_tfmini_s(bus, ADDR_FRONT)
    left  = read_tfmini_s(bus, ADDR_LEFT)
    right = read_tfmini_s(bus, ADDR_RIGHT)
    return front, left, right

def log_lidar(front, left, right):
    f = f"{front:4d}" if front else " ERR"
    l = f"{left:4d}"  if left  else " ERR"
    r = f"{right:4d}" if right else " ERR"
    logging.info(f"LiDAR | F:{f}cm  L:{l}cm  R:{r}cm")

# ─── MAVLINK HELPERS ──────────────────────────────────────────────
def send_velocity(mav, vx, vy, vz=0.0, duration_s=0.2):
    """
    Send NED velocity setpoint.
    vx = North, vy = East, vz = Down (positive = down)
    Sends repeatedly for duration_s at 10Hz.
    """
    end_time = time.time() + duration_s
    while time.time() < end_time:
        mav.mav.set_position_target_local_ned_send(
            0,
            mav.target_system,
            mav.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111,
            0, 0, 0,
            vx, vy, vz,
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
        if msg and mavutil.mode_string_v10(msg) == mode_name:
            logging.info(f"Mode set: {mode_name}")
            return True
    logging.warning(f"Mode change to {mode_name} not confirmed")
    return False

def descend_to_altitude(mav, target_alt_m, tolerance=0.3):
    """Descend to target altitude and hold."""
    logging.info(f"Descending to {target_alt_m}m")
    set_mode(mav, 'GUIDED')

    # Send position hold at target altitude
    mav.mav.set_position_target_local_ned_send(
        0,
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        # Ignore XY position, only Z (altitude)
        0b0000111111111000,
        0, 0, -target_alt_m,   # NED: negative Z = up
        0, 0, 0,
        0, 0, 0,
        0, 0
    )

    start = time.time()
    while time.time() - start < 15:
        msg = mav.recv_match(
            type='VFR_HUD', blocking=True, timeout=1
        )
        if msg:
            current = msg.alt
            logging.info(f"  Alt: {current:.2f}m", )
            if abs(current - target_alt_m) < tolerance:
                logging.info(f"Reached {current:.2f}m")
                return True
    logging.warning("Altitude target not reached")
    return False

# ─── COMPUTE CORRIDOR CORRECTION ──────────────────────────────────
def compute_correction(front, left, right):
    """
    Given three LiDAR readings compute velocity commands.

    Returns (vx, vy, should_stop, corridor_exited)
    vx = forward velocity (North)
    vy = lateral velocity (East — positive = right)
    should_stop   = True if obstacle too close ahead
    corridor_exited = True if front > CORRIDOR_EXIT_CM
    """
    vx = FORWARD_SPEED
    vy = 0.0
    should_stop      = False
    corridor_exited  = False

    # Exit detection
    if front is not None and front > CORRIDOR_EXIT_CM:
        corridor_exited = True
        return 0.0, 0.0, False, True

    # Front obstacle check
    if front is not None and front < FRONT_OBSTACLE_CM:
        should_stop = True
        vx = 0.0
        logging.info(f"  STOP — obstacle at {front}cm")
        return vx, vy, should_stop, corridor_exited

    # Lateral correction
    # Both walls close — try to centre
    if (left is not None and left < WALL_PROXIMITY_CM and
            right is not None and right < WALL_PROXIMITY_CM):
        # Bias toward whichever side has more space
        if left > right:
            vy = -CORRECTION_SPEED   # move left (West)
        else:
            vy =  CORRECTION_SPEED   # move right (East)

    elif left is not None and left < WALL_PROXIMITY_CM:
        vy = CORRECTION_SPEED        # too close left — move right
        logging.info(f"  Left wall at {left}cm — correcting right")

    elif right is not None and right < WALL_PROXIMITY_CM:
        vy = -CORRECTION_SPEED       # too close right — move left
        logging.info(f"  Right wall at {right}cm — correcting left")

    return vx, vy, should_stop, corridor_exited

# ─── STATE S3 / S8 — CORRIDOR ENTRY ──────────────────────────────
def state_corridor_entry(mav, cam_ai, detect_banner_fn,
                         is_aligned_fn, lap="forward"):
    """
    Approach corridor entrance and align with green banner.

    cam_ai          — picamera2 instance for AI Camera
    detect_banner_fn — function from banner_detect.py
    is_aligned_fn   — function from banner_detect.py
    lap             — "forward" or "return"

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

        import cv2
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
            # Yaw correction
            # Positive offset = banner right = turn right = positive yaw rate
            yaw_rate = 0.3 if yaw_offset > 0 else -0.3
            mav.mav.command_long_send(
                mav.target_system,
                mav.target_component,
                mavutil.mavlink.MAV_CMD_CONDITION_YAW,
                0,
                abs(yaw_offset * 0.05),  # angle degrees (scaled)
                20,                        # speed deg/s
                1 if yaw_offset > 0 else -1,  # direction
                1,                         # relative
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

        vx, vy, should_stop, exited = compute_correction(
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
                # Try a small lateral nudge to find a way around
                if left is not None and right is not None:
                    if left > right:
                        send_velocity(mav, 0, -CORRECTION_SPEED,
                                      duration_s=0.5)
                    else:
                        send_velocity(mav, 0, CORRECTION_SPEED,
                                      duration_s=0.5)
        else:
            obstacle_hold_start = None
            send_velocity(mav, vx, vy, duration_s=0.2)

        time.sleep(0.05)

# ─── STANDALONE TEST ──────────────────────────────────────────────
if __name__ == "__main__":
    """
    Standalone LiDAR test — no flight.
    Run this on the bench to verify LiDAR logic and 
    correction values before any flight test.
    """
    print("Corridor Navigation — LiDAR logic test (no flight)")
    print("Hold objects at various distances to test corrections")
    print("Ctrl+C to quit\n")

    with SMBus(BUS_NUMBER) as bus:
        while True:
            front, left, right = read_all_lidars(bus)

            vx, vy, stop, exited = compute_correction(
                front, left, right
            )

            f = f"{front:4d}cm" if front else " ERR "
            l = f"{left:4d}cm"  if left  else " ERR "
            r = f"{right:4d}cm" if right else " ERR "

            status = "EXIT"  if exited else \
                     "STOP"  if stop   else \
                     "GO"

            print(f"F:{f} L:{l} R:{r} | "
                  f"vx={vx:.1f} vy={vy:+.1f} | {status}    ",
                  end="\r", flush=True)

            time.sleep(0.05)