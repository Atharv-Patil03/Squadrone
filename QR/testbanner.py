"""
SQUADRONE — Banner Land Test
File: test_banner_land.py
Team: Squadrone AT2026RC60, PVGCOET Pune, SAEINDIA AeroTHON 2026

Test sequence:
  1. Connect MAVLink
  2. Arm & take off to TAKEOFF_ALT_M (3m)
  3. Hover and align downward ArduCam over green banner on ground
  4. Once centred and stable → descend to LAND_ALT_M (0.5m)
  5. Switch to LAND mode → drone lands on banner centre

Purpose:
  Validate banner detection + alignment logic in real flight
  BEFORE attempting corridor navigation.

Camera:
  ArduCam (downward-facing, rear-mounted) on cam1
  Auto-detected by model name — same technique as QR module.
  If auto-detect fails, set ARDUCAM_INDEX = 1 manually.

Green banner: 5ft x 5ft, placed on ground in front of drone.
ArduCam is rear-mounted facing down, so banner appears near
TOP of frame when drone is hovering forward of it.
"""

import cv2
import time
import logging
import os
import numpy as np
from collections import deque

from picamera2 import Picamera2
from pymavlink import mavutil

# ── DEBUG PREVIEW ─────────────────────────────────────────────────────────────
# Set SHOW_WINDOW=True only if Pi has a display connected (HDMI).
# DEBUG_SAVE_DIR: saves an annotated JPEG every DEBUG_SAVE_INTERVAL frames
#   → view on laptop:  scp pi@<PI_IP>:~/squadrone/debug_frame.jpg .
#   or watch live:     watch -n1 'scp pi@<PI_IP>:~/squadrone/debug_frame.jpg . && eog debug_frame.jpg'
SHOW_WINDOW          = False          # True = cv2 window (needs display)
DEBUG_SAVE_DIR       = os.path.expanduser("~/squadrone")
DEBUG_SAVE_INTERVAL  = 5             # save every N frames
_debug_frame_count   = 0

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
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE   = 921600

# Flight
TAKEOFF_ALT_M     = 5.0    # cruise altitude for alignment — higher so 5x5ft banner fits in frame
TAKEOFF_THRESHOLD = 0.95   # fraction of target alt = takeoff done
LAND_ALT_M        = 2.0    # descend to this before switching LAND mode
                           # Set above LAND_ALT_M ArduCopter param (your team set ~1.5m)
                           # so LAND mode starts its slow-descent phase correctly

# Camera
CAM_WIDTH        = 640
CAM_HEIGHT       = 480
ARDUCAM_INDEX    = None    # None = auto-detect ArduCam; set 0 or 1 to force

# Banner detection (HSV green)
LOWER_GREEN        = np.array([35,  80,  50])
UPPER_GREEN        = np.array([85, 255, 255])
MIN_AREA           = 300    # lowered — banner may be far away and small
MIN_ASPECT_RATIO   = 0.2    # relaxed — partial view at low altitude
MAX_ASPECT_RATIO   = 5.0    # relaxed — same
MIN_RECTANGULARITY = 0.20   # relaxed — partial frame fill at low alt

# Alignment
ALIGN_THRESHOLD_PX = 50    # centroid must be within this of frame centre
ALIGN_HOLD_FRAMES  = 5     # consecutive aligned frames → confirmed centred
ALIGN_TIMEOUT_S    = 60    # if not aligned within this many seconds → descend anyway
SEARCH_CREEP_SPD   = 0.3   # m/s — creep forward when banner not visible
ALIGN_SPEED        = 0.4   # m/s — max correction speed
MIN_MOVE_SPEED     = 0.15  # m/s — minimum speed sent (drone ignores < ~0.1 m/s)

# Detection history (same as banner_detect.py)
HISTORY_LENGTH     = 8     # frames in rolling window
REQUIRED_DETECTIONS= 5     # detections needed in window to confirm banner

# ──────────────────────────────────────────────────────────────────────────────
# CAMERA — auto-detect ArduCam (same technique as QR module)
# ──────────────────────────────────────────────────────────────────────────────

def resolve_arducam_index():
    """
    Find the ArduCam index. Pi5 has two cameras:
      cam0 = AI Cam (imx500)
      cam1 = ArduCam (imx519)
    Auto-detects by model name. Falls back to ARDUCAM_INDEX or index 1.
    """
    if ARDUCAM_INDEX is not None:
        log.info(f"ArduCam index forced to {ARDUCAM_INDEX}")
        return ARDUCAM_INDEX

    try:
        cameras = Picamera2.global_camera_info()
        log.info("Cameras: %s",
                 [(c.get("Num"), c.get("Model")) for c in cameras])

        # Prefer imx519 (ArduCam IMX519 Autofocus)
        for c in cameras:
            model = str(c.get("Model", "")).lower()
            if "imx519" in model or "arducam" in model:
                idx = c.get("Num", 1)
                log.info(f"ArduCam auto-detected at index {idx} ({model})")
                return idx

        # Fallback: use index 1 (cam1)
        log.warning("ArduCam not found by name — defaulting to index 1")
        return 1

    except Exception as e:
        log.warning(f"Camera info failed: {e} — defaulting to index 1")
        return 1


def start_camera():
    idx = resolve_arducam_index()
    cam = Picamera2(idx)
    cfg = cam.create_preview_configuration(
        main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"},
        buffer_count=4
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(2)

    # Enable continuous autofocus on IMX519 (ArduCam)
    try:
        cam.set_controls({"AfMode": 2, "AfSpeed": 1})  # 2=Continuous, 1=Fast
        log.info("Autofocus: continuous enabled")
    except Exception as e:
        log.warning(f"Autofocus not available: {e} — using fixed focus")

    time.sleep(1)  # let AF settle
    log.info(f"ArduCam started (index {idx}, {CAM_WIDTH}x{CAM_HEIGHT})")
    return cam

# ──────────────────────────────────────────────────────────────────────────────
# BANNER DETECTION
# ──────────────────────────────────────────────────────────────────────────────

_kernel = np.ones((5, 5), np.uint8)

# Rolling detection history — same approach as banner_detect.py
_detection_history = deque(maxlen=HISTORY_LENGTH)

def _find_banner_raw(frame):
    """Raw single-frame detection. Returns dict or None."""
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


def detect_banner(frame):
    """
    Stable detection using rolling history window (same as banner_detect.py).
    Returns averaged centroid dict if REQUIRED_DETECTIONS in last HISTORY_LENGTH frames,
    else None.
    """
    raw = _find_banner_raw(frame)
    _detection_history.append(raw)

    detections = [d for d in _detection_history if d is not None]
    if len(detections) < REQUIRED_DETECTIONS:
        return None

    # Average centroid and area over recent detections
    cx = int(sum(d["center"][0] for d in detections) / len(detections))
    cy = int(sum(d["center"][1] for d in detections) / len(detections))
    area = sum(d["area"] for d in detections) / len(detections)
    return {"center": (cx, cy), "area": area}


def reset_detection_history():
    """Call when re-entering a detection phase to clear stale frames."""
    _detection_history.clear()


# ──────────────────────────────────────────────────────────────────────────────
# DEBUG PREVIEW
# ──────────────────────────────────────────────────────────────────────────────

def draw_debug(frame, banner, offset_x=None, offset_y=None, label=""):
    """
    Draw detection overlay on frame and save/show.
    Call after detect_banner() with the same frame (BGR).
    """
    global _debug_frame_count
    _debug_frame_count += 1

    dbg = frame.copy()
    cx_frame = CAM_WIDTH  // 2
    cy_frame = CAM_HEIGHT // 2

    # Crosshair at frame centre
    cv2.line(dbg, (cx_frame - 20, cy_frame), (cx_frame + 20, cy_frame), (0, 255, 255), 1)
    cv2.line(dbg, (cx_frame, cy_frame - 20), (cx_frame, cy_frame + 20), (0, 255, 255), 1)

    if banner is not None:
        cx, cy = banner["center"]
        # Circle on detected centroid
        cv2.circle(dbg, (cx, cy), 10, (0, 255, 0), 2)
        # Line from frame centre to centroid
        cv2.line(dbg, (cx_frame, cy_frame), (cx, cy), (0, 200, 0), 1)
        txt = f"offset:({offset_x:+d},{offset_y:+d}) area:{int(banner['area'])}"
        cv2.putText(dbg, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(dbg, "NO BANNER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if label:
        cv2.putText(dbg, label, (10, CAM_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Save JPEG periodically
    if _debug_frame_count % DEBUG_SAVE_INTERVAL == 0:
        path = os.path.join(DEBUG_SAVE_DIR, "debug_frame.jpg")
        cv2.imwrite(path, dbg)

    # Optional live window
    if SHOW_WINDOW:
        cv2.imshow("Squadrone Camera", dbg)
        cv2.waitKey(1)

# ──────────────────────────────────────────────────────────────────────────────
# MAVLINK HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def connect_mavlink():
    log.info(f"Connecting {SERIAL_PORT} @ {BAUD_RATE}…")
    mav = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUD_RATE)
    # Wait until we get a heartbeat with a valid (non-zero) system ID.
    # The first heartbeat packet sometimes arrives before the FC has set
    # its sysid, giving target_system=0. Keep reading until it's non-zero.
    for attempt in range(20):
        mav.wait_heartbeat(timeout=3)
        if mav.target_system != 0:
            break
        log.info(f"  target_system=0 (attempt {attempt+1}/20), retrying…")
    else:
        raise RuntimeError("target_system=0 after 20 attempts — check connection — RTL !!!")
    log.info(f"Heartbeat — system {mav.target_system} "
             f"component {mav.target_component}")
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
    log.warning(f"Mode {mode_name} not confirmed")
    return False

def get_altitude(mav):
    """
    Return relative altitude in metres.
    Flushes stale queued messages to get the freshest reading.
    Returns None if no message arrives within timeout.
    """
    msg = None
    # Drain any queued GLOBAL_POSITION_INT messages to get the latest
    while True:
        m = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
        if m is None:
            break
        msg = m
    # If nothing was queued, wait briefly for a fresh one
    if msg is None:
        msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=2)
    if msg is not None:
        alt = msg.relative_alt / 1000.0
        return alt
    log.warning("get_altitude: no GLOBAL_POSITION_INT received!")
    return None   # explicit None so callers can detect failure

def send_velocity(mav, forward, right, down=0.0, duration_s=0.2):
    """Body-frame velocity — forward=nose, right=drone's right."""
    end = time.time() + duration_s
    while time.time() < end:
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
    log.info("Setting GUIDED…")
    set_mode(mav, 'GUIDED')

    log.info("Arming…")
    mav.arducopter_arm()
    mav.motors_armed_wait()
    log.info("Armed")

    log.info(f"Takeoff → {target_alt_m}m")
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, target_alt_m
    )

    threshold = target_alt_m * TAKEOFF_THRESHOLD
    start = time.time()
    while time.time() - start < 30:
        alt = get_altitude(mav)
        if alt is None:
            log.warning("  Alt: no fix yet…")
            time.sleep(0.5)
            continue
        log.info(f"  Alt: {alt:.2f}m  (target {threshold:.2f}m)")
        if alt >= threshold:
            log.info(f"Takeoff done at {alt:.2f}m")
            return True
        time.sleep(0.5)

    log.warning("Takeoff timeout")
    return False

def descend_to(mav, target_alt_m, timeout_s=30):
    """Descend to target_alt_m using downward velocity setpoints."""
    log.info(f"Descending to {target_alt_m}m…")
    DESCEND_SPEED = 0.4   # m/s downward — faster descent
    start = time.time()
    while time.time() - start < timeout_s:
        alt = get_altitude(mav)
        log.info(f"  Alt: {alt:.2f}m → target {target_alt_m}m")
        if alt <= target_alt_m + 0.3:
            log.info(f"  Reached {alt:.2f}m")
            stop_drone(mav)
            return True
        # Send downward velocity
        mav.mav.set_position_target_local_ned_send(
            0,
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0000111111000111,  # velocity only
            0, 0, 0,
            0, 0, DESCEND_SPEED,   # vx=0, vy=0, vz=down
            0, 0, 0,
            0, 0
        )
        time.sleep(0.3)
    log.warning("Descend timeout")
    return False

# ──────────────────────────────────────────────────────────────────────────────
# STATE: ALIGN TO BANNER
# ──────────────────────────────────────────────────────────────────────────────

def state_align_to_banner(mav, cam):
    """
    Hover at TAKEOFF_ALT_M and align downward camera over green banner.

    Camera is REAR-mounted facing DOWN.
    Banner is placed IN FRONT of drone on the ground.

    Frame geometry:
      Drone nose  → top of frame
      Drone tail  → bottom of frame  (camera is here)
      offset_x +ve → banner is RIGHT of centre → move right
      offset_y -ve → banner is near TOP        → banner is AHEAD  → move forward
      offset_y +ve → banner is near BOTTOM     → banner is BEHIND → move backward

    Returns True when banner is stably centred.
    """
    log.info("\n── ALIGN TO BANNER ──")
    log.info(f"  Threshold: ±{ALIGN_THRESHOLD_PX}px for {ALIGN_HOLD_FRAMES} frames")
    set_mode(mav, 'GUIDED')

    cx_frame = CAM_WIDTH  // 2
    cy_frame = CAM_HEIGHT // 2

    # Scale: half-frame offset → ALIGN_SPEED
    PX_SCALE_X = ALIGN_SPEED / (CAM_WIDTH  / 2)
    PX_SCALE_Y = ALIGN_SPEED / (CAM_HEIGHT / 2)

    align_count  = 0
    no_banner_ct = 0
    align_start  = time.time()

    while True:
        # Timeout — if we can't align in ALIGN_TIMEOUT_S, descend anyway
        if time.time() - align_start > ALIGN_TIMEOUT_S:
            log.warning(f"  Align timeout ({ALIGN_TIMEOUT_S}s) — descending anyway")
            return True
        rgb   = cam.capture_array()
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        banner = detect_banner(frame)

        # ── No banner visible ──────────────────────────────────────
        if banner is None:
            align_count  = 0
            no_banner_ct += 1
            action = "creeping forward" if no_banner_ct > 20 else "holding"
            log.info(f"  NO BANNER DETECTED (frame {no_banner_ct}) — {action}")
            draw_debug(frame, None, label=f"NO BANNER | {action}")
            if no_banner_ct > 20:
                # Banner is in front — creep forward to bring it into view
                send_velocity(mav, SEARCH_CREEP_SPD, 0, duration_s=0.5)
                no_banner_ct = 0
            else:
                stop_drone(mav)
            continue

        no_banner_ct = 0
        cx, cy = banner["center"]

        offset_x = cx - cx_frame   # +ve → right
        offset_y = cy - cy_frame   # +ve → below centre (banner behind) → move back

        aligned = (abs(offset_x) <= ALIGN_THRESHOLD_PX and
                   abs(offset_y) <= ALIGN_THRESHOLD_PX)

        status = f"ALIGNED {align_count}/{ALIGN_HOLD_FRAMES}" if aligned else "correcting"
        log.info(f"  Banner: ({cx},{cy})  "
                 f"offset: ({offset_x:+d},{offset_y:+d})px  "
                 f"area: {banner['area']}  {status}")
        draw_debug(frame, banner, offset_x, offset_y, label=f"ALIGN | {status}")

        if aligned:
            align_count += 1
            stop_drone(mav)
            log.info(f"  ALIGNED {align_count}/{ALIGN_HOLD_FRAMES}")
            if align_count >= ALIGN_HOLD_FRAMES:
                log.info("  ✓ BANNER CENTRED AND STABLE")
                return True
        else:
            align_count = 0
            # Proportional speed — but enforce minimum so drone actually moves
            raw_vx = +offset_y * PX_SCALE_Y   # forward/back
            raw_vy =  offset_x * PX_SCALE_X   # left/right

            # Apply minimum speed in the direction needed
            if abs(raw_vx) > 0.01:
                vforward = max(MIN_MOVE_SPEED, abs(raw_vx)) * (1 if raw_vx > 0 else -1)
                vforward = max(-ALIGN_SPEED, min(ALIGN_SPEED, vforward))
            else:
                vforward = 0.0

            if abs(raw_vy) > 0.01:
                vright = max(MIN_MOVE_SPEED, abs(raw_vy)) * (1 if raw_vy > 0 else -1)
                vright = max(-ALIGN_SPEED, min(ALIGN_SPEED, vright))
            else:
                vright = 0.0

            log.info(f"  Moving: fwd={vforward:+.2f} right={vright:+.2f} m/s")
            send_velocity(mav, vforward, vright, duration_s=0.3)

# ──────────────────────────────────────────────────────────────────────────────
# STATE: LAND ON BANNER
# ──────────────────────────────────────────────────────────────────────────────

def lock_yaw(mav):
    """
    Send a yaw command to hold current heading.
    This prevents yaw drift during the descent phase.
    """
    # Get current yaw from ATTITUDE message
    msg = mav.recv_match(type='ATTITUDE', blocking=True, timeout=2)
    if msg is None:
        log.warning("lock_yaw: no ATTITUDE message — skipping yaw lock")
        return
    import math
    yaw_deg = math.degrees(msg.yaw) % 360
    log.info(f"  Locking yaw at {yaw_deg:.1f}°")
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0,
        yaw_deg,  # target angle degrees
        10,       # yaw rate deg/s
        1,        # direction: 1=CW
        0,        # relative=0 (absolute heading)
        0, 0, 0
    )


def state_land_on_banner(mav, cam):
    """
    Descend slowly to LAND_ALT_M while keeping banner centred,
    then switch to LAND mode.

    During descent, banner grows in frame (area increases).
    Keep correcting lateral position as we descend.
    At LAND_ALT_M → switch LAND mode → ArduCopter handles final touchdown.
    """
    log.info("\n── LAND ON BANNER ──")
    reset_detection_history()   # clear history from align phase

    # Lock yaw so heading stays fixed during descent
    lock_yaw(mav)

    cx_frame = CAM_WIDTH  // 2
    cy_frame = CAM_HEIGHT // 2
    PX_SCALE_X = ALIGN_SPEED / (CAM_WIDTH  / 2)
    PX_SCALE_Y = ALIGN_SPEED / (CAM_HEIGHT / 2)

    DESCEND_SPEED = 0.4   # m/s downward
    descent_start = time.time()
    DESCENT_TIMEOUT_S = 60

    while True:
        # Safety: overall descent timeout
        if time.time() - descent_start > DESCENT_TIMEOUT_S:
            log.warning(f"  Descent timeout ({DESCENT_TIMEOUT_S}s) — triggering LAND anyway")
            stop_drone(mav)
            time.sleep(0.3)
            set_mode(mav, 'LAND')
            return True

        alt = get_altitude(mav)
        if alt is None:
            log.warning("  Alt: no fix — skipping loop iteration")
            time.sleep(0.2)
            continue

        log.info(f"  Alt: {alt:.2f}m  (LAND triggers at ≤{LAND_ALT_M:.1f}m)")

        if alt <= LAND_ALT_M:
            log.info(f"  ✓ Reached {alt:.2f}m → switching to LAND")
            stop_drone(mav)
            time.sleep(0.5)
            ok = set_mode(mav, 'LAND')
            log.info(f"  LAND mode {'set ✓' if ok else 'NOT CONFIRMED — retrying'}")
            if not ok:
                # Retry once
                time.sleep(0.5)
                set_mode(mav, 'LAND')

            log.info("  LAND mode active — waiting for touchdown…")
            # Wait for disarm OR altitude near zero
            land_start = time.time()
            while time.time() - land_start < 60:
                if not mav.motors_armed():
                    log.info("  ✓ Landed and disarmed automatically")
                    return True
                alt_now = get_altitude(mav)
                if alt_now is None:
                    alt_now = -1.0
                log.info(f"  Landing… alt={alt_now:.2f}m  armed={mav.motors_armed()}")
                if alt_now < 0.15:
                    log.info("  ✓ Altitude near zero — touchdown confirmed")
                    time.sleep(3)   # let ArduCopter disarm on its own
                    return True
                time.sleep(1)
            log.warning("  Land timeout (60s) — check drone manually")
            return True

        # Capture frame and keep centred while descending
        rgb   = cam.capture_array()
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        banner = detect_banner(frame)

        if banner is not None:
            cx, cy = banner["center"]
            offset_x = cx - cx_frame
            offset_y = cy - cy_frame
            log.info(f"  Banner: ({cx},{cy})  offset: ({offset_x:+d},{offset_y:+d})px  area:{banner['area']}")
            draw_debug(frame, banner, offset_x, offset_y, label=f"DESCEND | alt={alt:.2f}m")

            vright   = max(-ALIGN_SPEED,
                           min(ALIGN_SPEED,  offset_x * PX_SCALE_X))
            vforward = max(-ALIGN_SPEED,
                           min(ALIGN_SPEED, -offset_y * PX_SCALE_Y))
        else:
            log.info("  Banner NOT DETECTED during descent — holding lateral, continuing down")
            draw_debug(frame, None, label=f"DESCEND no-detect | alt={alt:.2f}m")
            vforward = 0.0
            vright   = 0.0

        # Descend + lateral correction combined
        send_velocity(mav, vforward, vright,
                      down=DESCEND_SPEED, duration_s=0.3)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("SQUADRONE — Banner Land Test")
    log.info("Place 5x5ft green banner on ground in FRONT of drone")
    log.info("=" * 60)

    cam = None
    mav = None

    try:
        # 1. Init hardware
        log.info("\n[1/4] Starting ArduCam…")
        cam = start_camera()

        log.info("\n[2/4] Connecting MAVLink…")
        mav = connect_mavlink()

        # Quick camera check — capture one frame and check for banner
        log.info("\nCamera check — looking for green banner…")
        rgb   = cam.capture_array()
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        b = detect_banner(frame)
        if b:
            log.info(f"  Banner visible at {b['center']} area={b['area']} ✓")
        else:
            log.warning("  Banner NOT visible in current frame — "
                        "check banner position and lighting before flying")

        input("\nPress ENTER to arm and take off, or Ctrl+C to abort…\n")

        # 2. Arm & takeoff
        log.info("\n[3/4] Arm & takeoff…")
        if not arm_and_takeoff(mav, TAKEOFF_ALT_M):
            log.error("Takeoff failed → RTL")
            set_mode(mav, 'RTL')
            return

        time.sleep(1.0)   # stabilise at cruise altitude

        # Lock yaw immediately after takeoff so heading holds through alignment
        log.info("  Locking yaw (hold current heading)…")
        lock_yaw(mav)
        time.sleep(0.5)

        # 3. Align to banner
        log.info("\n[4/4] Aligning to banner…")
        state_align_to_banner(mav, cam)

        # 4. Descend and land on banner
        state_land_on_banner(mav, cam)

        log.info("\n✓ Test complete — drone landed on banner")

    except KeyboardInterrupt:
        log.warning("\n!!! Ctrl+C — RTL !!!")
        if mav:
            set_mode(mav, 'RTL')

    except Exception as e:
        log.error(f"\n!!! EXCEPTION: {e} — RTL !!!")
        if mav:
            set_mode(mav, 'RTL')
        raise

    finally:
        if cam:
            cam.stop()
            log.info("Camera stopped")


if __name__ == "__main__":
    main()