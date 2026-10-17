"""
SQUADRONE — Corridor Flight Test
File: test_corridor_flight.py
Team: Squadrone AT2026RC60, PVGCOET Pune, SAEINDIA AeroTHON 2026

Full autonomous sequence:
  1. Connect & arm
  2. Take off to CORRIDOR_ALTITUDE_M
  3. Align to green banner on ground (downward-facing AI cam)
  4. Enter corridor (fly forward off banner)
  5. Navigate corridor with 3x TF Mini S LiDAR
  6. Detect corridor exit → trigger RTL

LiDAR I2C addresses (verified by user):
  ADDR_LEFT  = 0x15
  ADDR_FRONT = 0x14
  ADDR_RIGHT = 0x11

Banner detection:
  Uses detect_banner() from banner_detect.py
  Camera is downward-facing Picamera2 (ArduCam)
  Alignment: banner centroid within ALIGN_THRESHOLD_PX of frame centre
"""

import cv2
import time
import logging
import numpy as np
from collections import deque

from smbus2 import SMBus, i2c_msg
from picamera2 import Picamera2
from pymavlink import mavutil

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# MAVLink
SERIAL_PORT   = "/dev/ttyAMA0"
BAUD_RATE     = 921600

# Flight
TAKEOFF_ALT_M      = 3.0   # metres — takeoff + corridor cruise altitude
TAKEOFF_THRESHOLD  = 0.95  # fraction of target alt before takeoff declared done

# LiDAR (TF Mini S, I2C)
BUS_NUMBER  = 1
ADDR_LEFT   = 0x15
ADDR_FRONT  = 0x14
ADDR_RIGHT  = 0x11

LIDAR_MIN_CM        = 10    # ignore reads below — noise/glitch filter
FRONT_OBSTACLE_CM   = 150   # stop if obstacle within 1.5 m ahead
WALL_PROXIMITY_CM   = 60    # correct if wall within 60 cm
CORRIDOR_EXIT_CM    = 400   # NOT used for exit — kept for reference
# Exit is detected by BOTH side LiDARs going large simultaneously
# (corridor walls are green banners — when drone exits, left+right clear)
SIDE_CLEAR_CM       = 200   # both sides must read > this to declare exit
SIDE_CLEAR_FRAMES   = 5     # consecutive frames both sides clear before RTL

# Speeds (m/s)
FORWARD_SPEED    = 0.5
CORRECTION_SPEED = 0.3
APPROACH_SPEED   = 0.3   # creep forward when searching for banner

# Banner alignment
CAM_WIDTH           = 640
CAM_HEIGHT          = 480
ALIGN_THRESHOLD_PX  = 30    # pixels — centroid must be within this of centre
ALIGN_HOLD_FRAMES   = 10    # consecutive aligned frames before entry confirmed
BANNER_SEARCH_SPD   = 0.2   # m/s — slow creep when banner not visible

# Banner detection (from banner_detect.py)
LOWER_GREEN        = np.array([35,  80,  50])
UPPER_GREEN        = np.array([85, 255, 255])
MIN_AREA           = 500
MIN_ASPECT_RATIO   = 0.3
MAX_ASPECT_RATIO   = 3.5
MIN_RECTANGULARITY = 0.35

# Obstacle retry inside corridor
OBSTACLE_RETRY_S = 2.0

# ──────────────────────────────────────────────────────────────────────────────
# CAMERA
# ──────────────────────────────────────────────────────────────────────────────

def start_camera():
    cam = Picamera2(1)   # cam1 = ArduCam (downward-facing, rear-mounted)
                         # cam0 = AI Cam  (front-facing) — used in mission_align.py
    cfg = cam.create_preview_configuration(
        main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"},
        buffer_count=4
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(2)
    log.info("Camera ready")
    return cam

# ──────────────────────────────────────────────────────────────────────────────
# BANNER DETECTION  (from banner_detect.py — inlined so single-file deploy)
# ──────────────────────────────────────────────────────────────────────────────

_kernel = np.ones((5, 5), np.uint8)

def detect_banner(frame):
    """
    Detect green banner in frame (BGR).
    Returns dict with keys: center(cx,cy), box(x,y,w,h), area
             or None if not detected.
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        ar   = w / h
        rect = area / (w * h)
        if not (MIN_ASPECT_RATIO <= ar <= MAX_ASPECT_RATIO):
            continue
        if rect < MIN_RECTANGULARITY:
            continue
        if area > best_area:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            best_area = area
            best = {"center": (cx, cy), "box": (x, y, w, h), "area": area}

    return best

# ──────────────────────────────────────────────────────────────────────────────
# LIDAR (TF Mini S)
# ──────────────────────────────────────────────────────────────────────────────

OBTAIN_DATA_CMD = [0x5A, 0x05, 0x00, 0x01, 0x60]

def read_tfmini_s(bus, address):
    """Read distance in cm from one TF Mini S. Returns cm or None."""
    try:
        bus.i2c_rdwr(i2c_msg.write(address, OBTAIN_DATA_CMD))
        time.sleep(0.002)
        r = i2c_msg.read(address, 9)
        bus.i2c_rdwr(r)
        data = list(r)
        if (len(data) == 9
                and data[0] == 0x59 and data[1] == 0x59
                and (sum(data[0:8]) & 0xFF) == data[8]):
            dist = data[2] + (data[3] << 8) + 5
            if dist < LIDAR_MIN_CM:
                return None   # noise spike
            return dist
    except Exception:
        pass
    return None

def read_all_lidars(bus):
    """Returns (front_cm, left_cm, right_cm) — None means error/spike."""
    front = read_tfmini_s(bus, ADDR_FRONT)
    left  = read_tfmini_s(bus, ADDR_LEFT)
    right = read_tfmini_s(bus, ADDR_RIGHT)
    return front, left, right

def log_lidar(front, left, right):
    f = f"{front:4d}" if front is not None else " ERR"
    l = f"{left:4d}"  if left  is not None else " ERR"
    r = f"{right:4d}" if right is not None else " ERR"
    log.info(f"LiDAR | F:{f}cm  L:{l}cm  R:{r}cm")

# ──────────────────────────────────────────────────────────────────────────────
# MAVLINK HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def connect_mavlink():
    log.info(f"Connecting to {SERIAL_PORT} @ {BAUD_RATE}…")
    mav = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUD_RATE)
    mav.wait_heartbeat()
    log.info(f"Heartbeat from system {mav.target_system} "
             f"component {mav.target_component}")
    if mav.target_system == 0:
        raise RuntimeError("target_system is 0 — check connection")
    return mav

def set_mode(mav, mode_name):
    mode_id = mav.mode_mapping()[mode_name]
    mav.mav.set_mode_send(
        mav.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )
    start = time.time()
    while time.time() - start < 5:
        msg = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if msg and msg.custom_mode == mode_id:
            log.info(f"Mode: {mode_name}")
            return True
    log.warning(f"Mode change to {mode_name} not confirmed")
    return False

def get_altitude(mav):
    msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=2)
    if msg:
        return msg.relative_alt / 1000.0
    return 0.0

def send_velocity(mav, forward, right, down=0.0, duration_s=0.2):
    """
    Body-frame velocity setpoint.
    forward = drone nose direction
    right   = drone's right
    down    = positive = descend
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
    send_velocity(mav, 0, 0, 0, duration_s=0.3)

def arm_and_takeoff(mav, target_alt_m):
    """Arm, switch to GUIDED, take off to target_alt_m."""
    log.info("Setting GUIDED mode…")
    set_mode(mav, 'GUIDED')

    log.info("Arming…")
    mav.arducopter_arm()
    mav.motors_armed_wait()
    log.info("Armed")

    log.info(f"Takeoff to {target_alt_m}m…")
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0,
        target_alt_m
    )

    threshold = target_alt_m * TAKEOFF_THRESHOLD
    start = time.time()
    while time.time() - start < 30:
        alt = get_altitude(mav)
        log.info(f"  Alt: {alt:.2f}m / {target_alt_m}m")
        if alt >= threshold:
            log.info(f"Takeoff complete at {alt:.2f}m")
            return True
        time.sleep(0.5)

    log.warning("Takeoff altitude not reached in 30s")
    return False

def trigger_rtl(mav):
    log.info(">>> TRIGGERING RTL <<<")
    set_mode(mav, 'RTL')

# ──────────────────────────────────────────────────────────────────────────────
# STATE: BANNER ALIGNMENT
# ──────────────────────────────────────────────────────────────────────────────

def state_banner_align(mav, cam):
    """
    Hover and align downward camera over green banner.

    Strategy:
      - Capture frame, detect green banner
      - Compute pixel offset of banner centroid from frame centre
      - Convert pixel offset → lateral velocity correction
        (right pixel error  → move right,  left → move left)
        (forward pixel error → move forward, back → move back)
      - Hold ALIGN_HOLD_FRAMES consecutive aligned frames before returning

    Camera is REAR-mounted facing DOWN. Banner is placed IN FRONT of drone.
    So when hovering near the banner:
      - Banner appears near TOP of image  (drone nose = top of frame,
        rear camera sees what is ahead of the drone at top)
      - offset_x > 0 → banner RIGHT of frame centre → drone moves RIGHT
      - offset_y < 0 → banner ABOVE frame centre   → banner is AHEAD  → move FORWARD
      - offset_y > 0 → banner BELOW frame centre   → banner is BEHIND → move BACKWARD

    Returns True when banner is centred and stable.
    """
    log.info("\n── BANNER ALIGNMENT ──")
    log.info("  Camera: rear-mounted downward | Banner: on ground in front")
    set_mode(mav, 'GUIDED')

    cx_frame = CAM_WIDTH  // 2
    cy_frame = CAM_HEIGHT // 2

    # Pixel-to-velocity scale: 240px (half height) full-scale → APPROACH_SPEED
    PX_SCALE = APPROACH_SPEED / (CAM_HEIGHT / 2)

    align_count  = 0
    search_count = 0
    MAX_SEARCH   = 40   # frames without banner before slow creep forward

    while True:
        rgb   = cam.capture_array()
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        banner = detect_banner(frame)

        if banner is None:
            align_count  = 0
            search_count += 1
            if search_count >= MAX_SEARCH:
                # Banner not visible — creep forward: banner is in front,
                # so moving forward brings it into camera view (top of frame)
                log.info("  Banner not visible — creeping forward")
                send_velocity(mav, BANNER_SEARCH_SPD, 0, duration_s=0.5)
                search_count = 0
            else:
                stop_drone(mav)
                log.info("  No banner — holding")
            continue

        search_count = 0
        cx, cy = banner["center"]

        offset_x = cx - cx_frame   # +ve = banner right  → move right
        offset_y = cy - cy_frame   # +ve = banner below  → banner BEHIND → move back
                                   # -ve = banner above  → banner AHEAD  → move forward

        log.info(f"  Banner: ({cx},{cy})  offset: ({offset_x:+d},{offset_y:+d})px")

        aligned = (abs(offset_x) <= ALIGN_THRESHOLD_PX and
                   abs(offset_y) <= ALIGN_THRESHOLD_PX)

        if aligned:
            align_count += 1
            log.info(f"  ALIGNED {align_count}/{ALIGN_HOLD_FRAMES}")
            stop_drone(mav)

            if align_count >= ALIGN_HOLD_FRAMES:
                log.info("  BANNER CENTRED — CONFIRMED")
                return True
        else:
            align_count = 0
            # offset_x: banner right → vright positive (move right)
            vright = max(-APPROACH_SPEED,
                         min(APPROACH_SPEED, offset_x * PX_SCALE))
            # offset_y: banner below centre = banner BEHIND drone
            #   → move backward (negative forward)
            # offset_y: banner above centre = banner AHEAD of drone
            #   → move forward (positive forward)
            vforward = max(-APPROACH_SPEED,
                           min(APPROACH_SPEED, -offset_y * PX_SCALE))
            send_velocity(mav, vforward, vright, duration_s=0.2)

# ──────────────────────────────────────────────────────────────────────────────
# STATE: CORRIDOR ENTRY
# ──────────────────────────────────────────────────────────────────────────────

def state_corridor_entry(mav, bus):
    """
    Move forward from aligned banner position into corridor entrance.
    Stop when front LiDAR shows clear corridor ahead (< EXIT threshold
    but no obstacle), confirming drone is inside.

    Simple approach: fly forward for a fixed entry distance (2 s at
    APPROACH_SPEED), giving time for the banner to pass behind the drone
    and for the corridor walls to appear on left/right LiDARs.

    Returns True when entry complete.
    """
    log.info("\n── CORRIDOR ENTRY ──")
    set_mode(mav, 'GUIDED')

    ENTRY_DURATION_S = 2.5   # seconds forward at APPROACH_SPEED
    log.info(f"  Flying forward {ENTRY_DURATION_S}s to enter corridor")
    send_velocity(mav, APPROACH_SPEED, 0, duration_s=ENTRY_DURATION_S)
    stop_drone(mav)

    # Verify walls are now present (left or right LiDAR < 200cm)
    front, left, right = read_all_lidars(bus)
    log_lidar(front, left, right)

    walls_detected = (
        (left  is not None and left  < 200) or
        (right is not None and right < 200)
    )
    if walls_detected:
        log.info("  Corridor walls detected — ENTRY CONFIRMED")
    else:
        log.warning("  Walls not detected yet — proceeding anyway")

    return True

# ──────────────────────────────────────────────────────────────────────────────
# CORRIDOR NAVIGATION LOGIC
# ──────────────────────────────────────────────────────────────────────────────

def compute_correction(front, left, right):
    """
    Compute body-frame velocity from LiDAR readings.
    Returns (vforward, vright, should_stop, corridor_exited).

    Exit detection: BOTH side LiDARs go large simultaneously.
    Corridor walls are green banners — front LiDAR sees exit banner as
    a normal obstacle, so we cannot use front-clear for exit detection.
    When both sides read > SIDE_CLEAR_CM, the drone has exited the corridor.

    Note: caller tracks consecutive frames for SIDE_CLEAR_FRAMES debounce.
    This function returns corridor_exited=True each frame both sides are clear.
    """
    vforward = FORWARD_SPEED
    vright   = 0.0

    # Exit detection — both corridor walls disappear
    both_sides_clear = (
        (left  is None or left  > SIDE_CLEAR_CM) and
        (right is None or right > SIDE_CLEAR_CM)
    )
    if both_sides_clear:
        return 0.0, 0.0, False, True

    # Front obstacle (banner end wall or real obstacle)
    if front is not None and front < FRONT_OBSTACLE_CM:
        log.info(f"  STOP — obstacle at {front}cm")
        return 0.0, 0.0, True, False

    # Lateral wall correction
    both_close = (left  is not None and left  < WALL_PROXIMITY_CM and
                  right is not None and right < WALL_PROXIMITY_CM)

    if both_close:
        # Bias toward side with more space
        if left > right:
            vright = -CORRECTION_SPEED   # more space left → move left
        else:
            vright =  CORRECTION_SPEED   # more space right → move right

    elif left is not None and left < WALL_PROXIMITY_CM:
        vright = CORRECTION_SPEED
        log.info(f"  Left wall {left}cm — correcting right")

    elif right is not None and right < WALL_PROXIMITY_CM:
        vright = -CORRECTION_SPEED
        log.info(f"  Right wall {right}cm — correcting left")

    return vforward, vright, False, False

# ──────────────────────────────────────────────────────────────────────────────
# STATE: CORRIDOR NAVIGATION
# ──────────────────────────────────────────────────────────────────────────────

def state_corridor_navigation(mav, bus):
    """
    Fly through corridor using LiDAR until front > CORRIDOR_EXIT_CM.
    Returns True on exit detection.
    """
    log.info("\n── CORRIDOR NAVIGATION ──")
    log.info(f"  Exit condition: both sides > {SIDE_CLEAR_CM}cm "
             f"for {SIDE_CLEAR_FRAMES} consecutive frames")
    set_mode(mav, 'GUIDED')

    obstacle_hold_start = None
    exit_frame_count    = 0   # debounce: require N consecutive frames

    # Minimum travel time before exit can trigger (avoid false exit at entrance)
    MIN_TRAVEL_S = 2.0
    travel_start = time.time()

    while True:
        front, left, right = read_all_lidars(bus)
        log_lidar(front, left, right)

        vforward, vright, should_stop, exited = compute_correction(
            front, left, right
        )

        if exited and (time.time() - travel_start) > MIN_TRAVEL_S:
            exit_frame_count += 1
            log.info(f"  Exit candidate {exit_frame_count}/{SIDE_CLEAR_FRAMES} "
                     f"(L={left}, R={right})")
            if exit_frame_count >= SIDE_CLEAR_FRAMES:
                stop_drone(mav)
                log.info("  CORRIDOR EXIT CONFIRMED — both walls cleared")
                return True
            # Keep flying forward while counting
            send_velocity(mav, FORWARD_SPEED, 0, duration_s=0.2)
            time.sleep(0.05)
            continue
        else:
            exit_frame_count = 0   # reset if sides close in again

        if should_stop:
            stop_drone(mav)
            if obstacle_hold_start is None:
                obstacle_hold_start = time.time()
            elif time.time() - obstacle_hold_start > OBSTACLE_RETRY_S:
                log.info("  Retrying after obstacle hold")
                obstacle_hold_start = None
                # Nudge toward side with more space
                if left is not None and right is not None:
                    if left > right:
                        send_velocity(mav, 0, -CORRECTION_SPEED, duration_s=0.5)
                    else:
                        send_velocity(mav, 0,  CORRECTION_SPEED, duration_s=0.5)
        else:
            obstacle_hold_start = None
            send_velocity(mav, vforward, vright, duration_s=0.2)

        time.sleep(0.05)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("SQUADRONE — Corridor Flight Test")
    log.info("=" * 60)

    cam = None
    mav = None

    try:
        # ── 1. Hardware init ───────────────────────────────────────────────
        log.info("\n[1/6] Starting camera…")
        cam = start_camera()

        log.info("\n[2/6] Connecting MAVLink…")
        mav = connect_mavlink()

        log.info("\n[3/6] Opening I2C bus…")
        bus = SMBus(BUS_NUMBER)

        # Quick LiDAR sanity check
        front, left, right = read_all_lidars(bus)
        log_lidar(front, left, right)
        if front is None and left is None and right is None:
            log.warning("All LiDARs returned None — check wiring before flight")

        # ── 2. Arm & takeoff ───────────────────────────────────────────────
        log.info(f"\n[4/6] Arm & takeoff to {TAKEOFF_ALT_M}m…")
        if not arm_and_takeoff(mav, TAKEOFF_ALT_M):
            log.error("Takeoff failed — triggering RTL")
            trigger_rtl(mav)
            return

        time.sleep(1.0)   # brief hover to stabilise

        # ── 3. Banner alignment ────────────────────────────────────────────
        log.info("\n[5/6] Banner alignment phase…")
        state_banner_align(mav, cam)

        # ── 4. Corridor entry ──────────────────────────────────────────────
        log.info("\n[5b/6] Corridor entry…")
        state_corridor_entry(mav, bus)

        # ── 5. Corridor navigation ─────────────────────────────────────────
        log.info("\n[6/6] Corridor navigation…")
        state_corridor_navigation(mav, bus)

        # ── 6. RTL ─────────────────────────────────────────────────────────
        log.info("\nCorridor complete!")
        trigger_rtl(mav)

    except KeyboardInterrupt:
        log.warning("\n!!! KeyboardInterrupt — triggering RTL !!!")
        if mav:
            trigger_rtl(mav)

    except Exception as e:
        log.error(f"\n!!! EXCEPTION: {e} — triggering RTL !!!")
        if mav:
            trigger_rtl(mav)
        raise

    finally:
        if cam:
            cam.stop()
            log.info("Camera stopped")
        log.info("Script exited")


if __name__ == "__main__":
    main()