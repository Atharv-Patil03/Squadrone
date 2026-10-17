"""
SQUADRONE — Banner Land Test
File: test_banner_land.py
Team: Squadrone AT2026RC60, PVGCOET Pune, SAEINDIA AeroTHON 2026

Test sequence:
  1. Connect MAVLink
  2. Arm & take off to TAKEOFF_ALT_M (5m)
  3. Auto-calibrate banner colour from ground (works with ANY green shade)
  4. Align downward ArduCam over green banner
     - Lateral search (right/left/fwd/back) — NO YAW, nose toward corridor
  5. Once centred → descend to LAND_ALT_M (2.0m) → LAND mode

Camera geometry (REAR-mounted, facing DOWN):
  Drone NOSE  → BOTTOM of frame
  Drone TAIL  → TOP of frame
  offset_y +ve → banner below centre → banner AHEAD  → move FORWARD
  offset_y -ve → banner above centre → banner BEHIND → move BACKWARD
  offset_x +ve → banner right of centre              → move RIGHT

BANNER vs GRASS/TREE rejection:
  [G1] MIN_AREA = 2000        — banner is large from 5m; grass blobs are tiny
  [G2] MIN_RECTANGULARITY=0.55 — banner is solid rect; grass/trees are irregular
  [G3] MIN_SOLIDITY = 0.80    — banner fills its convex hull; foliage does not
  [G4] Single dominant blob   — banner is ONE object; if largest < 3× second,
                                 it is scattered grass, not a banner
  [G5] Calibration locks on the LARGEST blob on ground — the banner itself

All fixes:
  [F1] get_altitude() drain loop
  [F2] lock_yaw() — nose stays toward corridor
  [F3] land state: None alt, retry, 60s timeout
  [F4] +offset_y for forward/back
  [F5] Relaxed HSV initially; tightened by calibration
  [F6] Ground check + diagnostic images
  [F7] --calibrate mode
  [F8] Honest landing result
  [F9] Immediate single-frame detection during descent
  [F10] Auto colour calibration — works with any shade of green
  [F11] Lateral search pattern, zero yaw
  [G1-G5] Banner-vs-grass/tree rejection
"""

import cv2
import time
import logging
import os
import sys
import math
import numpy as np
from collections import deque

from picamera2 import Picamera2
from pymavlink import mavutil

# ──────────────────────────────────────────────────────────────────────────────
# DEBUG
# ──────────────────────────────────────────────────────────────────────────────
SHOW_WINDOW          = True
DEBUG_SAVE_DIR       = os.path.expanduser("~/squadrone")
DEBUG_SAVE_INTERVAL  = 5
DEBUG_SAVE_NO_DETECT = True
os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)

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

SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE   = 921600

TAKEOFF_ALT_M     = 5.0
TAKEOFF_THRESHOLD = 0.95
LAND_ALT_M        = 2.0   # must be > ArduCopter LAND_ALT_LOW_M (1.5m)

CAM_WIDTH     = 640
CAM_HEIGHT    = 480
ARDUCAM_INDEX = None      # None = auto-detect

# ── HSV — broad initial range, overwritten by calibration ────────────────────
LOWER_GREEN = np.array([25, 40, 40])
UPPER_GREEN = np.array([95, 255, 255])

MORPH_KERNEL_SIZE = 5

# ── [G1-G4] Banner-vs-grass/tree shape filters ───────────────────────────────
#
#  From 5m altitude a 5×5 ft (1.5×1.5 m) banner covers roughly:
#    pixel_size ≈ (1.5 / 5.0) × (480 / VFOV_rad)
#  Conservatively: ~80×80 px = 6400 px² minimum expected area.
#  We use 2000 as the floor to handle partial views and oblique angles.
#
MIN_AREA           = 2000   # [G1] px² — rejects small grass blobs
MIN_ASPECT_RATIO   = 0.3    # banner is squarish (5×5 ft) — reject thin slivers
MAX_ASPECT_RATIO   = 3.5    # some tilt gives slight elongation — still limit
MIN_RECTANGULARITY = 0.55   # [G2] area / bounding-box — banner ≈0.7, grass ≈0.1
MIN_SOLIDITY       = 0.80   # [G3] area / convexHull area — banner≈0.9, foliage≈0.4
DOMINANT_RATIO     = 3.0    # [G4] largest must be ≥ this × second largest

# ── Alignment ────────────────────────────────────────────────────────────────
ALIGN_THRESHOLD_PX  = 50
ALIGN_HOLD_FRAMES   = 5
ALIGN_TIMEOUT_S     = 90
ALIGN_SPEED         = 0.4
MIN_MOVE_SPEED      = 0.15

HISTORY_LENGTH      = 8
REQUIRED_DETECTIONS = 5

# ── Search pattern (lateral only, zero yaw) ──────────────────────────────────
SEARCH_TRIGGER_FRAMES = 15  # no-detect frames before search kicks in

# (forward m/s, right m/s, duration s)
SEARCH_STEPS = [
    ( 0.0,  0.35, 1.2),   # slide right
    ( 0.0, -0.35, 2.4),   # slide left (crosses centre + overshoot)
    ( 0.0,  0.35, 1.2),   # back to centre
    ( 0.25, 0.0,  1.0),   # step forward
    ( 0.0,  0.45, 1.5),   # wider right
    ( 0.0, -0.45, 3.0),   # wider left
    ( 0.0,  0.45, 1.5),   # back to centre
    (-0.25, 0.0,  1.0),   # step back
    ( 0.0,  0.0,  1.0),   # pause — recheck
]

# ──────────────────────────────────────────────────────────────────────────────
# CAMERA
# ──────────────────────────────────────────────────────────────────────────────

def resolve_arducam_index():
    if ARDUCAM_INDEX is not None:
        return ARDUCAM_INDEX
    try:
        cameras = Picamera2.global_camera_info()
        log.info("Cameras: %s",
                 [(c.get("Num"), c.get("Model")) for c in cameras])
        for c in cameras:
            model = str(c.get("Model", "")).lower()
            if "imx519" in model or "arducam" in model:
                idx = c.get("Num", 1)
                log.info(f"ArduCam auto-detected at index {idx} ({model})")
                return idx
        log.warning("ArduCam not found by name — defaulting to index 1")
        return 1
    except Exception as e:
        log.warning(f"Camera info failed: {e} — defaulting to 1")
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
    try:
        cam.set_controls({"AfMode": 2, "AfSpeed": 1})
        log.info("Autofocus: continuous enabled")
    except Exception as e:
        log.warning(f"Autofocus not available: {e}")
    time.sleep(1)
    log.info(f"ArduCam started (index {idx}, {CAM_WIDTH}x{CAM_HEIGHT})")
    return cam

# ──────────────────────────────────────────────────────────────────────────────
# [F10] AUTO COLOUR CALIBRATION
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_banner_colour(cam, n_frames=20):
    """
    [F10][G5] Sample n_frames on the ground with banner in view.
    Finds the LARGEST single green blob (= the banner, not grass).
    Measures its exact H distribution and builds a tight range.
    Updates global LOWER_GREEN / UPPER_GREEN.
    """
    global LOWER_GREEN, UPPER_GREEN

    log.info("  Sampling banner colour (point camera at banner)…")
    h_vals = []
    s_vals = []

    broad_lower = np.array([20, 30, 30])
    broad_upper = np.array([100, 255, 255])
    kernel = np.ones((5, 5), np.uint8)

    for _ in range(n_frames):
        rgb   = cam.capture_array()
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, broad_lower, broad_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # [G5] Use ONLY the largest blob — that is the banner
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            continue

        # Sample H and S from the largest blob only
        blob_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(blob_mask, [largest], -1, 255, -1)

        h_vals.extend(hsv[:, :, 0][blob_mask == 255].tolist())
        s_vals.extend(hsv[:, :, 1][blob_mask == 255].tolist())
        time.sleep(0.05)

    if len(h_vals) < 30:
        log.warning("  Not enough colour samples — using defaults")
        log.warning("  Ensure banner is visible to camera during calibration")
        return LOWER_GREEN, UPPER_GREEN

    h_arr = np.array(h_vals)
    s_arr = np.array(s_vals)

    h_med  = int(np.median(h_arr))
    s_p10  = int(np.percentile(s_arr, 10))
    h_std  = int(np.std(h_arr))
    margin = max(14, h_std + 8)

    lower = np.array([max(0,   h_med - margin),
                      max(30,  s_p10  - 20),
                      30])
    upper = np.array([min(179, h_med + margin),
                      255, 255])

    log.info(f"  ✓ H_median={h_med}  H_std={h_std}  margin=±{margin}")
    log.info(f"  ✓ LOWER_GREEN = {lower}")
    log.info(f"  ✓ UPPER_GREEN = {upper}")

    # Save calibration diagnostic
    rgb   = cam.capture_array()
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv_f = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cal_mask = cv2.inRange(hsv_f, lower, upper)
    cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, "cal_frame.jpg"), frame)
    cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, "cal_mask.jpg"),
                cv2.cvtColor(cal_mask, cv2.COLOR_GRAY2BGR))
    log.info(f"  cal_frame.jpg + cal_mask.jpg → {DEBUG_SAVE_DIR}/")

    LOWER_GREEN = lower
    UPPER_GREEN = upper
    return lower, upper

# ──────────────────────────────────────────────────────────────────────────────
# BANNER DETECTION  [G1-G4 filters inside]
# ──────────────────────────────────────────────────────────────────────────────

_kernel      = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
_det_history = deque(maxlen=HISTORY_LENGTH)


def _find_banner_raw(frame):
    """
    Single-frame detection with full grass/tree rejection.

    Returns dict with center, area, green_px — or None.

    Rejection pipeline:
      HSV mask → morphology → find contours
      For each contour:
        [G1] area >= MIN_AREA
        [G2] rectangularity >= MIN_RECTANGULARITY
        [G3] solidity >= MIN_SOLIDITY
             (cv2.convexHull area vs contour area)
      Keep candidates, sort by area descending.
      [G4] If top-2 exist: largest must be >= DOMINANT_RATIO × second
           (rejects scattered grass even if individually large enough)
      Return largest surviving candidate.
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel)

    green_px = cv2.countNonZero(mask)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:                         # [G1]
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue

        ar   = w / h
        if not (MIN_ASPECT_RATIO <= ar <= MAX_ASPECT_RATIO):
            continue

        rect = area / (w * h)
        if rect < MIN_RECTANGULARITY:               # [G2]
            continue

        hull      = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < MIN_SOLIDITY:                 # [G3]
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx_c = int(M["m10"] / M["m00"])
        cy_c = int(M["m01"] / M["m00"])

        candidates.append({
            "center":       (cx_c, cy_c),
            "box":          (x, y, w, h),
            "area":         area,
            "solidity":     solidity,
            "rectangularity": rect,
            "green_px":     green_px,
        })

    if not candidates:
        return None

    # Sort largest first
    candidates.sort(key=lambda d: d["area"], reverse=True)

    # [G4] Dominant blob check — reject scattered grass
    if len(candidates) >= 2:
        if candidates[0]["area"] < DOMINANT_RATIO * candidates[1]["area"]:
            # Two similarly-sized blobs → likely grass patches, not a banner
            log.debug(f"  [G4] rejected: areas {candidates[0]['area']:.0f} "
                      f"vs {candidates[1]['area']:.0f} — looks like grass")
            return None

    best = candidates[0]
    log.debug(f"  [detect] area={best['area']:.0f} "
              f"rect={best['rectangularity']:.2f} "
              f"solid={best['solidity']:.2f} "
              f"gpx={green_px}")
    return best


def detect_banner_rolling(frame):
    """Rolling window — used during alignment for stability."""
    raw = _find_banner_raw(frame)
    _det_history.append(raw)
    detections = [d for d in _det_history if d is not None]
    if len(detections) < REQUIRED_DETECTIONS:
        return None
    cx   = int(sum(d["center"][0] for d in detections) / len(detections))
    cy   = int(sum(d["center"][1] for d in detections) / len(detections))
    area = sum(d["area"]           for d in detections) / len(detections)
    gp   = detections[-1].get("green_px", 0)
    return {"center": (cx, cy), "area": area, "green_px": gp}


def detect_banner_immediate(frame):
    """Single-frame — used during descent for fast corrections."""
    return _find_banner_raw(frame)


def reset_detection_history():
    _det_history.clear()


def get_green_px(frame):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    return cv2.countNonZero(mask)

# ──────────────────────────────────────────────────────────────────────────────
# DEBUG OVERLAY
# ──────────────────────────────────────────────────────────────────────────────

_dbg_count     = 0
_dbg_last_save = 0


def draw_debug(frame, banner, offset_x=None, offset_y=None,
               label="", force_save=False):
    global _dbg_count, _dbg_last_save
    _dbg_count += 1

    dbg  = frame.copy()
    cx_f = CAM_WIDTH  // 2
    cy_f = CAM_HEIGHT // 2

    cv2.line(dbg, (cx_f - 30, cy_f), (cx_f + 30, cy_f), (0, 255, 255), 1)
    cv2.line(dbg, (cx_f, cy_f - 30), (cx_f, cy_f + 30), (0, 255, 255), 1)
    cv2.putText(dbg, "TOP=TAIL", (5, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(dbg, "BOT=NOSE", (5, CAM_HEIGHT - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    if banner is not None:
        cx, cy = banner["center"]
        gp     = banner.get("green_px", "?")
        sol    = banner.get("solidity", 0)
        rect   = banner.get("rectangularity", 0)
        cv2.circle(dbg, (cx, cy), 14, (0, 255, 0), 2)
        cv2.line(dbg, (cx_f, cy_f), (cx, cy), (0, 200, 0), 1)
        if offset_x is not None:
            t1 = (f"off:({offset_x:+d},{offset_y:+d}) "
                  f"area:{int(banner['area'])}")
            t2 = f"solid:{sol:.2f} rect:{rect:.2f} gpx:{gp}"
            cv2.putText(dbg, t1, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
            cv2.putText(dbg, t2, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
    else:
        cv2.putText(dbg, "NO BANNER", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if label:
        cv2.putText(dbg, label, (10, CAM_HEIGHT - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 2)

    should_save = (
        force_save
        or (_dbg_count % DEBUG_SAVE_INTERVAL == 0)
        or (DEBUG_SAVE_NO_DETECT
            and banner is None
            and _dbg_count != _dbg_last_save)
    )
    if should_save:
        _dbg_last_save = _dbg_count
        cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, "debug_frame.jpg"), dbg)

    if SHOW_WINDOW:
        cv2.imshow("Squadrone", dbg)
        cv2.waitKey(1)

# ──────────────────────────────────────────────────────────────────────────────
# [F7] HSV CALIBRATION MODE  — python3 test_banner_land.py --calibrate
# ──────────────────────────────────────────────────────────────────────────────

def run_hsv_calibration(cam):
    log.info("=" * 60)
    log.info("HSV CALIBRATION — point camera at green banner")
    log.info(f"Output → {DEBUG_SAVE_DIR}/   |   Ctrl+C to stop")
    log.info("=" * 60)

    roi    = 100
    cx     = CAM_WIDTH  // 2
    cy     = CAM_HEIGHT // 2
    kernel = np.ones((5, 5), np.uint8)

    try:
        while True:
            rgb   = cam.capture_array()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            broad = cv2.inRange(hsv,
                                np.array([20, 30, 30]),
                                np.array([100, 255, 255]))
            broad = cv2.morphologyEx(broad, cv2.MORPH_OPEN, kernel)
            gp_b  = cv2.countNonZero(broad)

            tuned = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
            gp_t  = cv2.countNonZero(tuned)

            r    = hsv[cy-roi//2:cy+roi//2, cx-roi//2:cx+roi//2]
            hm   = r[:,:,0].mean()
            hmin = r[:,:,0].min()
            hmax = r[:,:,0].max()
            sm   = r[:,:,1].mean()
            vm   = r[:,:,2].mean()

            diag = frame.copy()
            cv2.rectangle(diag,
                          (cx-roi//2, cy-roi//2),
                          (cx+roi//2, cy+roi//2),
                          (0, 255, 255), 2)
            lines = [
                f"Centre H: {hmin:.0f}-{hmax:.0f}  mean={hm:.1f}",
                f"Centre S: mean={sm:.1f}",
                f"Centre V: mean={vm:.1f}",
                f"Broad green px: {gp_b}",
                f"Tuned green px: {gp_t}",
                f"Suggested LOWER: [{max(0,int(hm)-14)}, "
                f"{max(30,int(sm)-20)}, 30]",
                f"Suggested UPPER: [{min(179,int(hm)+14)}, 255, 255]",
            ]
            for i, t in enumerate(lines):
                cv2.putText(diag, t, (10, 24 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, "hsv_frame.jpg"),       diag)
            cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, "hsv_mask_broad.jpg"),
                        cv2.cvtColor(broad, cv2.COLOR_GRAY2BGR))
            cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, "hsv_mask_tuned.jpg"),
                        cv2.cvtColor(tuned, cv2.COLOR_GRAY2BGR))

            log.info(f"H={hm:.0f}[{hmin:.0f}-{hmax:.0f}] "
                     f"S={sm:.0f} V={vm:.0f} "
                     f"broad_gpx={gp_b} tuned_gpx={gp_t}")
            time.sleep(1.0)

    except KeyboardInterrupt:
        log.info("Calibration done.")

# ──────────────────────────────────────────────────────────────────────────────
# MAVLINK HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def connect_mavlink():
    log.info(f"Connecting {SERIAL_PORT} @ {BAUD_RATE}…")
    mav = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUD_RATE)
    for attempt in range(20):
        mav.wait_heartbeat(timeout=3)
        if mav.target_system != 0:
            break
        log.info(f"  target_system=0 (attempt {attempt+1}/20)")
    else:
        raise RuntimeError("target_system=0 after 20 attempts")
    log.info(f"Heartbeat — sys={mav.target_system} comp={mav.target_component}")
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
    """[F1] Drain stale queue, return freshest altitude. None on failure."""
    msg = None
    while True:
        m = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
        if m is None:
            break
        msg = m
    if msg is None:
        msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=2)
    if msg is not None:
        return msg.relative_alt / 1000.0
    log.warning("get_altitude: no GLOBAL_POSITION_INT!")
    return None


def send_velocity(mav, forward, right, down=0.0, duration_s=0.2):
    """Body-frame velocity. forward=nose, right=drone-right."""
    end = time.time() + duration_s
    while time.time() < end:
        mav.mav.set_position_target_local_ned_send(
            0,
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0000111111000111,
            0, 0, 0,
            forward, right, down,
            0, 0, 0,
            0, 0
        )
        time.sleep(0.05)


def stop_drone(mav):
    send_velocity(mav, 0, 0, 0, duration_s=0.3)


def lock_yaw(mav):
    """[F2] Lock current heading. Called after takeoff and never changed."""
    msg = mav.recv_match(type='ATTITUDE', blocking=True, timeout=2)
    if msg is None:
        log.warning("lock_yaw: no ATTITUDE — skipping")
        return
    yaw_deg = math.degrees(msg.yaw) % 360
    log.info(f"  Locking yaw at {yaw_deg:.1f}° (nose toward corridor)")
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0,
        yaw_deg, 10, 1, 0, 0, 0, 0
    )


def arm_and_takeoff(mav, target_alt_m):
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
            time.sleep(0.5)
            continue
        log.info(f"  Alt: {alt:.2f}m / {threshold:.2f}m")
        if alt >= threshold:
            log.info(f"  Takeoff done at {alt:.2f}m")
            return True
        time.sleep(0.5)
    log.warning("Takeoff timeout")
    return False

# ──────────────────────────────────────────────────────────────────────────────
# [F6] PRE-FLIGHT GROUND CHECK
# ──────────────────────────────────────────────────────────────────────────────

def ground_check(cam):
    log.info("\n── PRE-FLIGHT GROUND CHECK ──")
    rgb   = cam.capture_array()
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    broad = cv2.inRange(hsv, np.array([20,30,30]), np.array([100,255,255]))
    tuned = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    gp_b  = cv2.countNonZero(broad)
    gp_t  = cv2.countNonZero(tuned)

    cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, "precheck_frame.jpg"), frame)
    cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, "precheck_mask.jpg"),
                cv2.cvtColor(tuned, cv2.COLOR_GRAY2BGR))

    log.info(f"  Broad green px: {gp_b}")
    log.info(f"  Tuned green px: {gp_t}")

    raw = _find_banner_raw(frame)
    if raw:
        log.info(f"  ✓ Banner detected: center={raw['center']} "
                 f"area={raw['area']:.0f} "
                 f"solid={raw['solidity']:.2f} "
                 f"rect={raw['rectangularity']:.2f}")
    elif gp_b < 200:
        log.warning("  ⚠ Very few green pixels — camera not pointing at banner?")
    elif gp_t < 50:
        log.warning("  ⚠ Broad sees green but tuned does not — re-run calibration")
    else:
        log.warning("  ⚠ Green pixels OK but shape filters rejected all contours")
        log.warning(f"  MIN_AREA={MIN_AREA} MIN_SOLIDITY={MIN_SOLIDITY} "
                    f"MIN_RECTANGULARITY={MIN_RECTANGULARITY}")
        log.warning("  Try lowering MIN_AREA if banner is far away")

    log.info(f"  Saved → {DEBUG_SAVE_DIR}/precheck_*.jpg")
    return gp_t

# ──────────────────────────────────────────────────────────────────────────────
# [F11] SEARCH PATTERN — lateral translation only, ZERO YAW
# ──────────────────────────────────────────────────────────────────────────────

class BannerSearcher:
    """
    Executes SEARCH_STEPS one at a time.
    Drone nose ALWAYS toward corridor — no yaw change ever.
    Resets when banner is found.
    """
    def __init__(self, mav):
        self.mav   = mav
        self.step  = 0
        self.cycle = 0

    def execute_step(self):
        idx = self.step % len(SEARCH_STEPS)
        fwd, rgt, dur = SEARCH_STEPS[idx]
        log.info(f"  [SEARCH step {idx}/{len(SEARCH_STEPS)-1}] "
                 f"fwd={fwd:+.2f} right={rgt:+.2f} dur={dur:.1f}s  "
                 f"(nose stays toward corridor)")
        send_velocity(self.mav, fwd, rgt, duration_s=dur)
        self.step += 1
        if self.step % len(SEARCH_STEPS) == 0:
            self.cycle += 1
            log.info(f"  [SEARCH] full cycle {self.cycle} done — repeating")

    def reset(self):
        self.step  = 0
        self.cycle = 0

# ──────────────────────────────────────────────────────────────────────────────
# STATE: ALIGN TO BANNER
# ──────────────────────────────────────────────────────────────────────────────

def state_align_to_banner(mav, cam):
    log.info("\n── ALIGN TO BANNER ──")
    log.info(f"  Threshold ±{ALIGN_THRESHOLD_PX}px  hold {ALIGN_HOLD_FRAMES} frames")
    set_mode(mav, 'GUIDED')
    reset_detection_history()

    cx_frame = CAM_WIDTH  // 2
    cy_frame = CAM_HEIGHT // 2
    PX_SCALE_X = ALIGN_SPEED / (CAM_WIDTH  / 2)
    PX_SCALE_Y = ALIGN_SPEED / (CAM_HEIGHT / 2)

    searcher      = BannerSearcher(mav)
    align_count   = 0
    no_banner_ct  = 0
    align_start   = time.time()
    _last_gpx_log = 0

    while True:
        elapsed = time.time() - align_start
        if elapsed > ALIGN_TIMEOUT_S:
            log.warning(f"  Align timeout ({ALIGN_TIMEOUT_S}s) — descending anyway")
            return True

        rgb   = cam.capture_array()
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Periodic green-pixel diagnostic
        now = time.time()
        if now - _last_gpx_log > 3.0:
            gp = get_green_px(frame)
            log.info(f"  [diag] green_px={gp}  elapsed={elapsed:.0f}s  "
                     f"(0 = HSV wrong, low = banner not in frame)")
            _last_gpx_log = now

        banner = detect_banner_rolling(frame)

        # ── Banner not visible ─────────────────────────────────────────────
        if banner is None:
            align_count   = 0
            no_banner_ct += 1
            gp = get_green_px(frame)
            log.info(f"  NO BANNER (frame {no_banner_ct}) gpx={gp}")
            draw_debug(frame, None,
                       label=f"NO BANNER gpx={gp} t={elapsed:.0f}s")
            if no_banner_ct >= SEARCH_TRIGGER_FRAMES:
                no_banner_ct = 0
                searcher.execute_step()
            else:
                stop_drone(mav)
            continue

        # ── Banner found ───────────────────────────────────────────────────
        searcher.reset()
        no_banner_ct = 0

        cx, cy   = banner["center"]
        offset_x = cx - cx_frame
        offset_y = cy - cy_frame   # +ve = AHEAD = move FORWARD [F4]

        aligned = (abs(offset_x) <= ALIGN_THRESHOLD_PX and
                   abs(offset_y) <= ALIGN_THRESHOLD_PX)

        status = f"ALIGNED {align_count+1}/{ALIGN_HOLD_FRAMES}" if aligned else "correcting"
        log.info(f"  Banner ({cx},{cy})  "
                 f"off({offset_x:+d},{offset_y:+d})  "
                 f"area={int(banner['area'])}  {status}")
        draw_debug(frame, banner, offset_x, offset_y,
                   label=f"ALIGN | {status}")

        if aligned:
            align_count += 1
            stop_drone(mav)
            if align_count >= ALIGN_HOLD_FRAMES:
                log.info("  ✓ BANNER CENTRED AND STABLE")
                return True
        else:
            align_count = 0
            # [F4] +offset_y = forward
            raw_vx = +offset_y * PX_SCALE_Y
            raw_vy =  offset_x * PX_SCALE_X

            vforward = 0.0
            if abs(raw_vx) > 0.01:
                vforward = (max(MIN_MOVE_SPEED, abs(raw_vx))
                            * (1 if raw_vx > 0 else -1))
                vforward = max(-ALIGN_SPEED, min(ALIGN_SPEED, vforward))

            vright = 0.0
            if abs(raw_vy) > 0.01:
                vright = (max(MIN_MOVE_SPEED, abs(raw_vy))
                          * (1 if raw_vy > 0 else -1))
                vright = max(-ALIGN_SPEED, min(ALIGN_SPEED, vright))

            log.info(f"  Moving fwd={vforward:+.2f} right={vright:+.2f} m/s")
            send_velocity(mav, vforward, vright, duration_s=0.3)

# ──────────────────────────────────────────────────────────────────────────────
# STATE: LAND ON BANNER
# ──────────────────────────────────────────────────────────────────────────────

def state_land_on_banner(mav, cam):
    log.info("\n── LAND ON BANNER ──")
    reset_detection_history()
    lock_yaw(mav)

    cx_frame = CAM_WIDTH  // 2
    cy_frame = CAM_HEIGHT // 2
    PX_SCALE_X = ALIGN_SPEED / (CAM_WIDTH  / 2)
    PX_SCALE_Y = ALIGN_SPEED / (CAM_HEIGHT / 2)

    DESCEND_SPEED     = 0.4
    DESCENT_TIMEOUT_S = 60
    descent_start     = time.time()
    banner_seen       = False   # [F8]

    while True:
        if time.time() - descent_start > DESCENT_TIMEOUT_S:
            log.warning("  Descent timeout — triggering LAND")
            stop_drone(mav)
            time.sleep(0.3)
            set_mode(mav, 'LAND')
            _log_land_result(banner_seen)
            return True

        alt = get_altitude(mav)
        if alt is None:
            log.warning("  Alt: no fix — waiting")
            time.sleep(0.2)
            continue

        log.info(f"  Alt: {alt:.2f}m  (LAND at ≤{LAND_ALT_M:.1f}m)")

        # ── Trigger LAND ───────────────────────────────────────────────────
        if alt <= LAND_ALT_M:
            log.info(f"  ✓ {alt:.2f}m → LAND mode")
            stop_drone(mav)
            time.sleep(0.5)
            ok = set_mode(mav, 'LAND')
            if not ok:
                log.warning("  Retry LAND…")
                time.sleep(0.5)
                ok = set_mode(mav, 'LAND')
            log.info(f"  LAND {'✓' if ok else 'FAILED'}")

            land_start = time.time()
            while time.time() - land_start < 60:
                if not mav.motors_armed():
                    log.info("  ✓ Disarmed — landed")
                    _log_land_result(banner_seen)
                    return True
                alt_now = get_altitude(mav)
                if alt_now is None:
                    alt_now = 99.0
                log.info(f"  Landing… alt={alt_now:.2f}m  armed={mav.motors_armed()}")
                if alt_now < 0.15:
                    log.info("  ✓ Near zero — touchdown")
                    time.sleep(3)
                    _log_land_result(banner_seen)
                    return True
                time.sleep(1)
            log.warning("  Land wait timeout")
            _log_land_result(banner_seen)
            return True

        # ── Descend + lateral correction ───────────────────────────────────
        rgb    = cam.capture_array()
        frame  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        banner = detect_banner_immediate(frame)   # [F9]

        if banner is not None:
            banner_seen = True
            cx, cy   = banner["center"]
            offset_x = cx - cx_frame
            offset_y = cy - cy_frame
            log.info(f"  Banner ({cx},{cy}) "
                     f"off({offset_x:+d},{offset_y:+d}) "
                     f"area={int(banner['area'])} "
                     f"solid={banner.get('solidity',0):.2f}")
            draw_debug(frame, banner, offset_x, offset_y,
                       label=f"DESCEND alt={alt:.2f}m")

            vright   = max(-ALIGN_SPEED,
                           min(ALIGN_SPEED,  offset_x * PX_SCALE_X))
            vforward = max(-ALIGN_SPEED,
                           min(ALIGN_SPEED, +offset_y * PX_SCALE_Y))  # [F4]

            if 0 < abs(vforward) < MIN_MOVE_SPEED:
                vforward = MIN_MOVE_SPEED * (1 if vforward > 0 else -1)
            if 0 < abs(vright) < MIN_MOVE_SPEED:
                vright   = MIN_MOVE_SPEED * (1 if vright   > 0 else -1)
        else:
            log.info("  Banner NOT DETECTED — holding lateral, descending")
            draw_debug(frame, None,
                       label=f"DESCEND no-detect alt={alt:.2f}m",
                       force_save=True)
            vforward = 0.0
            vright   = 0.0

        send_velocity(mav, vforward, vright,
                      down=DESCEND_SPEED, duration_s=0.3)


def _log_land_result(banner_seen):
    """[F8] Honest result — did we actually confirm the banner?"""
    if banner_seen:
        log.info("\n✓ RESULT: Landed ON banner (banner confirmed during descent)")
    else:
        log.warning("\n⚠ RESULT: Landed WITHOUT banner confirmation")
        log.warning("  Drone landed at current GPS position — may not be on banner")
        log.warning(f"  Inspect {DEBUG_SAVE_DIR}/debug_frame.jpg for clues")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    calibrate_mode = "--calibrate" in sys.argv

    log.info("=" * 60)
    log.info("SQUADRONE — Banner Land Test")
    log.info("Banner vs grass/tree rejection: ENABLED")
    log.info("Search pattern: lateral only (zero yaw)")
    if calibrate_mode:
        log.info("MODE: HSV CALIBRATION (no flight)")
    log.info("=" * 60)

    cam = None
    mav = None

    try:
        log.info("\n[1] Starting ArduCam…")
        cam = start_camera()

        if calibrate_mode:
            run_hsv_calibration(cam)
            return

        # [F10] Auto colour calibration
        log.info("\n[2] Auto-calibrating banner colour…")
        log.info("  >>> Point camera AT THE GREEN BANNER now <<<")
        log.info("  Sampling for ~2 seconds…")
        calibrate_banner_colour(cam, n_frames=20)

        # [F6] Ground check
        log.info("\n[3] Ground check…")
        gp = ground_check(cam)
        if gp < 50:
            log.warning(f"  ⚠ Only {gp} tuned green pixels after calibration!")
            log.warning("  Run:  python3 test_banner_land.py --calibrate")
            ans = input("  Continue anyway? [y/N]: ").strip().lower()
            if ans != 'y':
                log.info("Aborted.")
                return

        log.info("\n[4] Connecting MAVLink…")
        mav = connect_mavlink()

        input("\nPress ENTER to arm and take off (Ctrl+C to abort)…\n")

        log.info("\n[5] Arm & takeoff…")
        if not arm_and_takeoff(mav, TAKEOFF_ALT_M):
            log.error("Takeoff failed → RTL")
            set_mode(mav, 'RTL')
            return

        time.sleep(1.0)

        log.info("\n[6] Locking yaw (nose toward corridor)…")
        lock_yaw(mav)
        time.sleep(0.5)

        log.info("\n[7] Aligning to banner…")
        state_align_to_banner(mav, cam)

        log.info("\n[8] Descending and landing…")
        state_land_on_banner(mav, cam)

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
        if SHOW_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()