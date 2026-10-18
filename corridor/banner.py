#!/usr/bin/env python3
"""
SQUADRONE — Banner Detect + Land Test  (v5.2 — GPS coordinate lock)
File: test_banner_land.py
Team: Squadrone AT2026RC60, PVGCOET Pune, SAEINDIA AeroTHON 2026

PURPOSE (TEST ONLY):
  Takeoff → find green banner → align above it → LAND on it.
  Landing = proof the detection and alignment worked.
  In the real mission the drone will hover at 3 m instead of landing.

CAMERA GEOMETRY (rear-mounted downward):
  Drone NOSE  → BOTTOM of frame  (+offset_y → banner ahead → move FORWARD)
  Drone TAIL  → TOP of frame
  +offset_x   → banner right      → move RIGHT

RUN MODES:
  python3 test_banner_land.py                 # normal flight
  python3 test_banner_land.py --calibrate     # HSV probe loop, no flight
  python3 test_banner_land.py --skip-calib    # skip auto-calibration step
  python3 test_banner_land.py --check-only    # ground check, no flight

v5 fixes over v4:
  [F1] LOSS_TIMEOUT_S bumped 0.6→1.5 — brief occlusions no longer break lock
  [F2] DOMINANT_RATIO dropped 2.5→1.6 — banner passes even with nearby greenery
  [F3] MIN_SOLIDITY dropped 0.80→0.70 — creased/folded banners still detected
  [F4] Rolling window uses WEIGHTED average (recent frames count more)
  [F5] Consecutive-detection streak tracked separately for robustness metric
  [F6] send_velocity default dur bumped 0.25→0.35 s — command reaches FC reliably
  [F7] Align loop sends explicit altitude hold every cycle (prevents drift)
  [F8] Descent uses rolling window (not immediate) — one bad frame can't yank drone
  [F9] All raw frames saved to ~/squadrone/frames/ during align (capped at 200)
  [F10] Ground check saves HSV mask + annotated frame for easy debugging
  [F11] Auto-calibration now triggered by ENTER prompt so user can point cam first
  [F12] SEARCH_STEPS expanded to cover 4-metre radius before giving up
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

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

DEBUG_DIR  = os.path.expanduser("~/squadrone")
FRAME_DIR  = os.path.join(DEBUG_DIR, "frames")
os.makedirs(FRAME_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(DEBUG_DIR, "flight.log"), mode='a')
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SERIAL_PORT     = "/dev/ttyAMA0"
BAUD_RATE       = 921600

TAKEOFF_ALT_M   = 5.0
TAKEOFF_THRESH  = 0.95
LAND_ALT_M      = 2.0          # trigger LAND mode once below this

CAM_WIDTH       = 640
CAM_HEIGHT      = 480

# ── HSV — from bannerdetect.py (team-confirmed) ───────────────────────────
LOWER_GREEN = np.array([35,  80,  50])
UPPER_GREEN = np.array([85, 255, 255])

MORPH_K     = 5

# ── Shape filters ─────────────────────────────────────────────────────────
MIN_AREA           = 400    # [G1] slightly lower for 5 m altitude
MIN_ASPECT_RATIO   = 0.25   # banner can look nearly square from directly above
MAX_ASPECT_RATIO   = 4.0
MIN_RECTANGULARITY = 0.30   # [G2] relaxed — banner may be partially occluded
MIN_SOLIDITY       = 0.70   # [G3] FIX [F3] — creased banner still passes
DOMINANT_RATIO     = 1.6    # [G4] FIX [F2] — allow nearby greenery

# ── Rolling window ────────────────────────────────────────────────────────
HISTORY_LENGTH       = 10
REQUIRED_DETECTIONS  = 6    # needs 6/10 — slightly easier than v4's 5/8
LOSS_TIMEOUT_S       = 1.5  # FIX [F1] — 1.5 s gap before detection resets

# ── Alignment ─────────────────────────────────────────────────────────────
ALIGN_THRESHOLD_PX   = 35   # ≈0.35 m real-world error at 5 m altitude
ALIGN_HOLD_FRAMES    = 6    # 6 consecutive centred frames
ALIGN_TIMEOUT_S      = 150

# Two-zone speed control
ALIGN_ZONE_FAR_PX    = 80   # > this → FAR speed
ALIGN_SPEED_FAR      = 0.55 # m/s when far off-centre
ALIGN_SPEED_NEAR     = 0.20 # m/s when close
MIN_ALIGN_SPEED      = 0.15 # absolute floor

MAX_ALIGN_SPEED      = ALIGN_SPEED_FAR  # kept for descent section

# ── GPS coordinate lock ────────────────────────────────────────────────────
# Once banner is confirmed, we save its GPS position and spam it every cycle.
# This anchors the drone against wind — camera only corrects the final pixels.
GPS_LOCK_AFTER_DETECTIONS = 3   # save GPS after this many rolling detections
GPS_SPAM_HZ = 10                # how often to re-send the locked position (per second)

# ── Search ────────────────────────────────────────────────────────────────
SEARCH_TRIGGER_FRAMES = 15  # wait longer before triggering search step

# FIX [F12]: Expanded search covers ~4m radius spiral
# Each tuple: (forward_mps, right_mps, duration_s)
SEARCH_STEPS = [
    # small initial sweep to find banner if just off-centre
    ( 0.0,  0.30, 1.2),
    ( 0.0, -0.30, 2.4),
    ( 0.0,  0.30, 1.2),
    ( 0.30, 0.0,  1.0),
    # wider sweep — 1 m radius
    ( 0.0,  0.50, 1.5),
    ( 0.0, -0.50, 3.0),
    ( 0.0,  0.50, 1.5),
    (-0.30, 0.0,  1.0),
    # even wider — 2 m radius
    ( 0.0,  0.60, 2.0),
    ( 0.0, -0.60, 4.0),
    ( 0.0,  0.60, 2.0),
    ( 0.40, 0.0,  1.5),
    ( 0.0,  0.60, 2.0),
    ( 0.0, -0.60, 4.0),
    ( 0.0,  0.60, 2.0),
    (-0.40, 0.0,  1.5),
    # return to start
    ( 0.0,  0.0,  1.0),
]

# ─────────────────────────────────────────────────────────────────────────────
# HEADING STATE  (body → NED — stops yaw-induced drift)
# ─────────────────────────────────────────────────────────────────────────────

_heading_rad = 0.0


def capture_heading(mav):
    global _heading_rad
    msg = mav.recv_match(type='ATTITUDE', blocking=True, timeout=3)
    if msg is None:
        log.warning("capture_heading: no ATTITUDE — using 0°")
        return
    _heading_rad = msg.yaw
    log.info(f"  Heading locked: {math.degrees(msg.yaw):.1f}°")


def body_to_ned(fwd, rgt):
    h = _heading_rad
    north = fwd * math.cos(h) - rgt * math.sin(h)
    east  = fwd * math.sin(h) + rgt * math.cos(h)
    return north, east

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────────────────────────────────────

def start_camera():
    idx = 1
    try:
        info = Picamera2.global_camera_info()
        for c in info:
            if "imx519" in str(c.get("Model", "")).lower():
                idx = c.get("Num", 1)
                log.info(f"IMX519 detected at index {idx}")
                break
    except Exception:
        pass
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
    except Exception:
        pass
    log.info(f"Camera started ({CAM_WIDTH}×{CAM_HEIGHT})")
    return cam

# ─────────────────────────────────────────────────────────────────────────────
# AUTO COLOUR CALIBRATION  [F11] — user points cam at banner first
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_banner_colour(cam, n_frames=25):
    """
    Samples n_frames with banner in view.
    Measures dominant hue + saturation → refines HSV range.
    Falls back to defaults on too few samples.
    """
    global LOWER_GREEN, UPPER_GREEN

    log.info("  Sampling banner colour…")
    broad_lo = np.array([20, 30, 30])
    broad_hi = np.array([100, 255, 255])
    kernel   = np.ones((5, 5), np.uint8)
    h_vals   = []
    s_vals   = []
    v_vals   = []

    for _ in range(n_frames):
        rgb   = cam.capture_array()
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask  = cv2.morphologyEx(
            cv2.inRange(hsv, broad_lo, broad_hi),
            cv2.MORPH_OPEN, kernel)
        cnts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        biggest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(biggest) < 300:
            continue
        bm = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(bm, [biggest], -1, 255, -1)
        h_vals.extend(hsv[:, :, 0][bm == 255].tolist())
        s_vals.extend(hsv[:, :, 1][bm == 255].tolist())
        v_vals.extend(hsv[:, :, 2][bm == 255].tolist())
        time.sleep(0.05)

    if len(h_vals) < 50:
        log.warning(f"  Too few samples ({len(h_vals)}) — keeping defaults "
                    f"{LOWER_GREEN}→{UPPER_GREEN}")
        return

    h_med   = int(np.median(h_vals))
    h_std   = int(np.std(h_vals))
    s_p10   = int(np.percentile(s_vals, 10))
    v_p10   = int(np.percentile(v_vals, 10))
    margin  = max(14, h_std + 8)

    LOWER_GREEN[:] = [max(0,   h_med - margin),
                      max(40,  s_p10 - 20),
                      max(30,  v_p10 - 20)]
    UPPER_GREEN[:] = [min(179, h_med + margin), 255, 255]

    log.info(f"  ✓ Calibrated HSV: LOWER={LOWER_GREEN}  UPPER={UPPER_GREEN}  "
             f"(h_med={h_med} margin=±{margin})")
    cv2.imwrite(os.path.join(DEBUG_DIR, "cal_frame.jpg"),
                cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR))

# ─────────────────────────────────────────────────────────────────────────────
# BANNER DETECTION
# ─────────────────────────────────────────────────────────────────────────────

_kernel        = np.ones((MORPH_K, MORPH_K), np.uint8)
_history       = deque(maxlen=HISTORY_LENGTH)
_last_seen_t   = 0.0
_streak        = 0          # consecutive detection frames (for logging)


def _detect_raw(frame):
    """
    Single-frame detection.
    Returns best-candidate dict or None.
    Filters: [G1] area, [G2] rectangularity, [G3] solidity, [G4] dominant blob.
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel)

    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA:                          # [G1]
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        ar = w / h
        if not (MIN_ASPECT_RATIO <= ar <= MAX_ASPECT_RATIO):
            continue
        rect_ratio = area / (w * h)
        if rect_ratio < MIN_RECTANGULARITY:          # [G2]
            continue
        hull      = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < MIN_SOLIDITY:                  # [G3]
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        candidates.append({
            "center": (cx, cy), "box": (x, y, w, h),
            "area": area, "solidity": solidity,
            "rectangularity": rect_ratio, "mask": mask
        })

    if not candidates:
        return None

    candidates.sort(key=lambda d: d["area"], reverse=True)

    # [G4] Dominant blob check — relaxed ratio
    if len(candidates) >= 2:
        ratio = candidates[0]["area"] / candidates[1]["area"]
        if ratio < DOMINANT_RATIO:
            log.debug(f"  [G4] rejected: ratio={ratio:.2f} < {DOMINANT_RATIO} "
                      f"({candidates[0]['area']:.0f} vs {candidates[1]['area']:.0f})")
            return None

    return candidates[0]


def detect_rolling(frame):
    """
    [F4] WEIGHTED rolling window — recent frames count more.
    [F5] Tracks consecutive-detection streak.
    [F1] LOSS_TIMEOUT_S = 1.5 s before resetting.
    Returns averaged detection dict or None.
    """
    global _last_seen_t, _streak
    raw = _detect_raw(frame)
    _history.append(raw)
    now = time.monotonic()
    if raw is not None:
        _last_seen_t = now
        _streak += 1
    else:
        _streak = 0

    if now - _last_seen_t > LOSS_TIMEOUT_S:
        return None

    detections = [d for d in _history if d is not None]
    if len(detections) < REQUIRED_DETECTIONS:
        return None

    # Weighted average — most recent detection has weight HISTORY_LENGTH,
    # oldest has weight 1.  Indices in detections[] are oldest→newest.
    weights = [float(i + 1) for i in range(len(detections))]
    total_w = sum(weights)
    cx   = int(sum(w * d["center"][0] for w, d in zip(weights, detections)) / total_w)
    cy   = int(sum(w * d["center"][1] for w, d in zip(weights, detections)) / total_w)
    area = sum(w * d["area"]          for w, d in zip(weights, detections)) / total_w
    return {
        "center": (cx, cy),
        "area":   area,
        "box":    detections[-1]["box"],
        "rectangularity": detections[-1]["rectangularity"],
        "solidity":       detections[-1]["solidity"],
        "streak":         _streak,
    }


def detect_immediate(frame):
    """Single-frame, no window — used only for landing correction."""
    return _detect_raw(frame)


def reset_history():
    global _last_seen_t, _streak
    _history.clear()
    _last_seen_t = 0.0
    _streak      = 0

# ─────────────────────────────────────────────────────────────────────────────
# DEBUG SAVE  [F9] — save all frames during align, capped at 200
# ─────────────────────────────────────────────────────────────────────────────

_save_n        = 0
_frame_save_n  = 0
_MAX_FRAMES    = 200


def save_debug(frame, banner, label="", force=False, save_frame=False):
    global _save_n, _frame_save_n
    _save_n += 1

    cx_f = CAM_WIDTH  // 2
    cy_f = CAM_HEIGHT // 2

    dbg = frame.copy()
    cv2.line(dbg, (cx_f-30, cy_f), (cx_f+30, cy_f), (0, 255, 255), 1)
    cv2.line(dbg, (cx_f, cy_f-30), (cx_f, cy_f+30), (0, 255, 255), 1)
    cv2.putText(dbg, "BOT=NOSE", (5, CAM_HEIGHT-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    if banner:
        cx, cy = banner["center"]
        x, y, w, h = banner.get("box", (0, 0, 0, 0))
        cv2.rectangle(dbg, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(dbg,   (cx, cy), 8, (0, 255, 0), 2)
        cv2.line(dbg, (cx_f, cy_f), (cx, cy), (0, 200, 0), 1)
        streak = banner.get("streak", 0)
        cv2.putText(dbg,
            f"off({cx-cx_f:+d},{cy-cy_f:+d}) "
            f"area={int(banner['area'])} "
            f"sol={banner.get('solidity',0):.2f} "
            f"str={streak}",
            (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    else:
        cv2.putText(dbg, "NO BANNER", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if label:
        cv2.putText(dbg, label, (8, CAM_HEIGHT-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1)

    # Always update the live debug frame
    cv2.imwrite(os.path.join(DEBUG_DIR, "debug_frame.jpg"), dbg)

    # [F9] Save individual frames during align phase
    if save_frame and _frame_save_n < _MAX_FRAMES:
        fn = os.path.join(FRAME_DIR, f"f{_frame_save_n:04d}.jpg")
        cv2.imwrite(fn, dbg)
        _frame_save_n += 1

# ─────────────────────────────────────────────────────────────────────────────
# MAVLINK
# ─────────────────────────────────────────────────────────────────────────────

def connect_mavlink():
    log.info(f"Connecting {SERIAL_PORT} @ {BAUD_RATE}…")
    mav = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUD_RATE)
    for i in range(25):
        mav.wait_heartbeat(timeout=3)
        if mav.target_system != 0:
            break
        log.info(f"  target_system=0 ({i+1}/25)")
    else:
        raise RuntimeError("MAVLink: target_system=0 after 25 attempts")
    log.info(f"Connected sys={mav.target_system} comp={mav.target_component}")
    return mav


def set_mode(mav, mode):
    mid = mav.mode_mapping()[mode]
    mav.mav.set_mode_send(
        mav.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mid)
    t0 = time.time()
    while time.time() - t0 < 5:
        m = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if m and m.custom_mode == mid:
            log.info(f"Mode → {mode}")
            return True
    log.warning(f"Mode {mode} not confirmed")
    return False


def get_alt(mav):
    """Drain queue, return latest relative altitude in metres."""
    msg = None
    while True:
        m = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
        if m is None:
            break
        msg = m
    if msg is None:
        msg = mav.recv_match(
            type='GLOBAL_POSITION_INT', blocking=True, timeout=2)
    return (msg.relative_alt / 1000.0) if msg else None


def get_gps(mav):
    """Return (lat, lon) in degrees, or (None, None)."""
    msg = None
    while True:
        m = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
        if m is None:
            break
        msg = m
    if msg is None:
        msg = mav.recv_match(
            type='GLOBAL_POSITION_INT', blocking=True, timeout=2)
    if msg is None:
        return None, None
    return msg.lat / 1e7, msg.lon / 1e7


def spam_position(mav, lat, lon, alt_m, dur=0.0):
    """
    Spam SET_POSITION_TARGET_GLOBAL_INT at (lat, lon, alt_m) for dur seconds.
    If dur=0 sends exactly once.
    This is what locks the drone to a GPS coordinate — FC actively holds it.
    """
    mask = 0b0000111111111000   # position XYZ only, ignore vel/acc/yaw
    t_end = time.time() + max(dur, 0.001)
    while time.time() < t_end:
        mav.mav.set_position_target_global_int_send(
            0,
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            mask,
            int(lat * 1e7), int(lon * 1e7), int(alt_m),
            0, 0, 0,
            0, 0, 0,
            0, 0
        )
        time.sleep(0.1)   # 10 Hz


def send_velocity(mav, fwd_body, rgt_body, down=0.0, dur=0.35):
    """
    [F6] dur default bumped to 0.35 s — command reliably reaches FC.
    Sends repeated SET_POSITION_TARGET_LOCAL_NED for the duration.
    """
    north, east = body_to_ned(fwd_body, rgt_body)
    t_end = time.time() + dur
    while time.time() < t_end:
        mav.mav.set_position_target_local_ned_send(
            0,
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111,   # velocity only
            0, 0, 0,
            north, east, down,
            0, 0, 0,
            0, 0
        )
        time.sleep(0.05)


def stop_drone(mav, dur=0.5):
    """Brake. Slightly longer than v4 for more reliable stop."""
    send_velocity(mav, 0, 0, 0, dur=dur)


def hold_alt(mav, target_m):
    """
    [F7] Send explicit altitude hold to prevent slow descent during align.
    Uses MAV_FRAME_GLOBAL_RELATIVE_ALT — position control only.
    """
    mav.mav.set_position_target_global_int_send(
        0,
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        0b0000111111111000,   # position XYZ only (use current lat/lon)
        0, 0, int(target_m),
        0, 0, 0,
        0, 0, 0,
        0, 0
    )


def lock_yaw(mav):
    msg = mav.recv_match(type='ATTITUDE', blocking=True, timeout=2)
    if msg is None:
        return
    yaw_deg = math.degrees(msg.yaw) % 360
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0, yaw_deg, 10, 1, 0, 0, 0, 0)
    log.info(f"  Yaw locked @ {yaw_deg:.1f}°")


def arm_and_takeoff(mav, target_m):
    set_mode(mav, 'GUIDED')
    log.info("Arming…")
    mav.arducopter_arm()
    mav.motors_armed_wait()
    log.info("Armed ✓")
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, target_m)
    thresh = target_m * TAKEOFF_THRESH
    t0 = time.time()
    while time.time() - t0 < 35:
        alt = get_alt(mav)
        if alt is not None:
            log.info(f"  Alt: {alt:.2f}m / {thresh:.2f}m")
            if alt >= thresh:
                log.info("  Takeoff done ✓")
                return True
        time.sleep(0.5)
    log.error("Takeoff timeout!")
    return False

# ─────────────────────────────────────────────────────────────────────────────
# SEARCH PATTERN
# ─────────────────────────────────────────────────────────────────────────────

class BannerSearcher:
    def __init__(self, mav):
        self.mav  = mav
        self.step = 0

    def execute_step(self):
        idx = self.step % len(SEARCH_STEPS)
        fwd, rgt, dur = SEARCH_STEPS[idx]
        log.info(f"  [SEARCH {idx+1}/{len(SEARCH_STEPS)}] "
                 f"fwd={fwd:+.2f} rgt={rgt:+.2f} dur={dur:.1f}s")
        stop_drone(self.mav, 0.4)
        send_velocity(self.mav, fwd, rgt, dur=dur)
        stop_drone(self.mav, 0.4)
        self.step += 1

    def reset(self):
        self.step = 0

# ─────────────────────────────────────────────────────────────────────────────
# STATE 1: ALIGN ABOVE BANNER
# ─────────────────────────────────────────────────────────────────────────────

def state_align(mav, cam):
    """
    GPS-LOCK + camera fine-correction alignment.

    Phase 1 — SEARCH: no banner visible, execute search steps while
              spamming last-known GPS position (or takeoff position).
    Phase 2 — ACQUIRING: banner rolling-window filling up, drone holds GPS.
    Phase 3 — CORRECTING: banner confirmed, GPS locked to banner position,
              camera offsets nudge the drone left/right/fwd/back.
    Phase 4 — HOLDING: centred for ALIGN_HOLD_FRAMES consecutive frames → done.

    The GPS spam is the key change: even during corrections the FC is told
    "stay at this lat/lon at 5 m" every 0.1 s.  Wind cannot blow the drone
    away between correction pulses.
    """
    log.info("\n── STATE: ALIGN ABOVE BANNER (GPS-lock mode) ──")
    set_mode(mav, 'GUIDED')
    reset_history()

    cx_f = CAM_WIDTH  // 2
    cy_f = CAM_HEIGHT // 2

    # Grab takeoff GPS as starting anchor — used during search phase
    lat0, lon0 = get_gps(mav)
    if lat0 is None:
        log.warning("  No GPS at align start — will use first available fix")
        lat0, lon0 = 0.0, 0.0

    lock_lat   = lat0          # GPS anchor, updated when banner confirmed
    lock_lon   = lon0
    gps_locked = False         # True once we've saved banner GPS position
    lock_count = 0             # how many rolling detections since lock

    searcher      = BannerSearcher(mav)
    align_hold    = 0
    no_banner_ct  = 0
    t_start       = time.time()
    t_last_yaw    = time.time()
    t_last_diag   = time.monotonic()
    t_last_gps_spam = 0.0
    first_seen    = False

    log.info(f"  Takeoff anchor: {lat0:.7f}, {lon0:.7f}")

    while True:
        elapsed = time.time() - t_start
        if elapsed > ALIGN_TIMEOUT_S:
            log.warning("  Align timeout — proceeding to land")
            return lock_lat, lock_lon

        # ── Spam GPS position every 0.1 s (10 Hz) ─────────────────────────
        # This is the anchor — FC holds this coordinate regardless of camera state
        now_t = time.time()
        if now_t - t_last_gps_spam >= 0.1:
            spam_position(mav, lock_lat, lock_lon, TAKEOFF_ALT_M)
            t_last_gps_spam = now_t

        # Re-lock yaw every 15 s
        if now_t - t_last_yaw > 15:
            lock_yaw(mav)
            t_last_yaw = now_t

        rgb   = cam.capture_array()
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Periodic diagnostic
        now_m = time.monotonic()
        if now_m - t_last_diag > 4.0:
            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
            gp   = cv2.countNonZero(mask)
            alt  = get_alt(mav)
            alt_s = f"{alt:.2f}m" if alt else "?"
            log.info(f"  [diag] gpx={gp}  alt={alt_s}  streak={_streak}  "
                     f"elapsed={elapsed:.0f}s  "
                     f"locked={gps_locked}  anchor=({lock_lat:.6f},{lock_lon:.6f})")
            t_last_diag = now_m

        # First raw detection → stop lateral, let rolling window fill
        any_raw = _detect_raw(frame)
        if any_raw and not first_seen:
            first_seen = True
            log.info("  ↓ Banner first seen — stopping to fill rolling window")
            stop_drone(mav, dur=0.6)
            reset_history()

        banner = detect_rolling(frame)
        save_debug(frame, banner,
                   label=f"ALIGN t={elapsed:.0f}s hold={align_hold} "
                         f"{'GPS✓' if gps_locked else 'search'}",
                   save_frame=True)

        # ── No confirmed detection ─────────────────────────────────────────
        if banner is None:
            align_hold   = 0
            no_banner_ct += 1
            first_seen   = False
            # GPS spam already holding position — no extra stop_drone needed

            if no_banner_ct >= SEARCH_TRIGGER_FRAMES:
                no_banner_ct = 0
                # Move to next search step (velocity override), then GPS re-anchors
                searcher.execute_step()
                # Update anchor to current position after search move
                nlat, nlon = get_gps(mav)
                if nlat is not None:
                    lock_lat, lock_lon = nlat, nlon
                    log.info(f"  Search anchor updated: {lock_lat:.6f},{lock_lon:.6f}")
                reset_history()
            continue

        # ── Banner confirmed ───────────────────────────────────────────────
        searcher.reset()
        no_banner_ct = 0
        first_seen   = True
        lock_count  += 1

        # Save GPS lock after enough consecutive detections
        if not gps_locked and lock_count >= GPS_LOCK_AFTER_DETECTIONS:
            nlat, nlon = get_gps(mav)
            if nlat is not None:
                lock_lat, lock_lon = nlat, nlon
                gps_locked = True
                log.info(f"  ★ GPS LOCKED to banner: {lock_lat:.7f}, {lock_lon:.7f}")

        cx, cy   = banner["center"]
        offset_x = cx - cx_f    # + → right
        offset_y = cy - cy_f    # + → forward

        aligned = (abs(offset_x) <= ALIGN_THRESHOLD_PX and
                   abs(offset_y) <= ALIGN_THRESHOLD_PX)

        tag = f"HOLD {align_hold+1}/{ALIGN_HOLD_FRAMES}" if aligned else "correcting"
        log.info(f"  ({cx},{cy}) off({offset_x:+d},{offset_y:+d}) "
                 f"area={int(banner['area'])} "
                 f"sol={banner['solidity']:.2f} "
                 f"str={banner.get('streak',0)} [{tag}]")

        if aligned:
            align_hold += 1
            # Keep spamming GPS — don't send velocity while holding
            if align_hold >= ALIGN_HOLD_FRAMES:
                log.info("  ✓ CENTRED OVER BANNER")
                # Final GPS snap to exact position
                cur_lat, cur_lon = get_gps(mav)
                if cur_lat:
                    lock_lat, lock_lon = cur_lat, cur_lon
                stop_drone(mav, dur=0.5)
                return lock_lat, lock_lon
        else:
            align_hold = 0

            # ── Two-zone velocity correction on top of GPS hold ───────────
            # GPS spam keeps the drone from drifting.
            # Velocity nudge corrects the remaining pixel offset.
            err_px = math.hypot(offset_x, offset_y)

            if err_px > ALIGN_ZONE_FAR_PX:
                scale = ALIGN_SPEED_FAR / err_px
            else:
                scale = ALIGN_SPEED_NEAR / max(err_px, 1.0)

            vf = offset_y * scale
            vr = offset_x * scale

            if 0 < abs(vf) < MIN_ALIGN_SPEED:
                vf = math.copysign(MIN_ALIGN_SPEED, vf)
            if 0 < abs(vr) < MIN_ALIGN_SPEED:
                vr = math.copysign(MIN_ALIGN_SPEED, vr)
            vf = max(-ALIGN_SPEED_FAR, min(ALIGN_SPEED_FAR, vf))
            vr = max(-ALIGN_SPEED_FAR, min(ALIGN_SPEED_FAR, vr))

            zone = "FAR" if err_px > ALIGN_ZONE_FAR_PX else "NEAR"
            log.info(f"  [{zone} err={err_px:.0f}px] → fwd={vf:+.2f} rgt={vr:+.2f} m/s")

            dur = 0.45 if err_px > ALIGN_ZONE_FAR_PX else 0.30
            send_velocity(mav, vf, vr, dur=dur)

            # After velocity nudge — update GPS anchor to new position
            # so FC holds the corrected spot, not the old one
            if gps_locked:
                nlat, nlon = get_gps(mav)
                if nlat is not None:
                    lock_lat, lock_lon = nlat, nlon

# ─────────────────────────────────────────────────────────────────────────────
# STATE 2: DESCEND AND LAND
# ─────────────────────────────────────────────────────────────────────────────

# [F8] Descent uses a short rolling window (5-frame) rather than single frame
_desc_history = deque(maxlen=5)
_desc_last_t  = 0.0


def _detect_descent(frame):
    """5-frame rolling window for descent correction — smoother than immediate."""
    global _desc_last_t
    raw = _detect_raw(frame)
    _desc_history.append(raw)
    now = time.monotonic()
    if raw is not None:
        _desc_last_t = now
    if now - _desc_last_t > 0.8:       # shorter loss timeout during descent
        return None
    dets = [d for d in _desc_history if d is not None]
    if len(dets) < 2:
        return None
    cx = int(sum(d["center"][0] for d in dets) / len(dets))
    cy = int(sum(d["center"][1] for d in dets) / len(dets))
    return {"center": (cx, cy), "area": dets[-1]["area"],
            "box": dets[-1]["box"],
            "solidity": dets[-1]["solidity"],
            "rectangularity": dets[-1]["rectangularity"]}


def state_land(mav, cam, lock_lat=None, lock_lon=None):
    """
    Descend and land on the banner.

    lock_lat / lock_lon — GPS coordinates returned by state_align().
    While set, they are spammed every 0.1 s at decreasing altitude so
    the FC holds position against wind even as the drone descends.
    When the banner is visible, small lateral corrections are also sent.
    """
    log.info("\n── STATE: DESCEND + LAND ──")
    reset_history()
    _desc_history.clear()

    cx_f = CAM_WIDTH  // 2
    cy_f = CAM_HEIGHT // 2
    # During descent use near-zone speed only — gentle corrections
    KPX  = ALIGN_SPEED_NEAR / (CAM_WIDTH  / 2)
    KPY  = ALIGN_SPEED_NEAR / (CAM_HEIGHT / 2)

    DESCEND_SPEED   = 0.30    # m/s downward — gentle
    GPS_SPAM_PERIOD = 0.1     # 10 Hz GPS spam
    t_start         = time.time()
    t_last_gps_spam = 0.0

    if lock_lat is not None:
        log.info(f"  GPS lock: {lock_lat:.7f}, {lock_lon:.7f} — spamming during descent")
    else:
        # Grab current position as best-effort anchor
        lock_lat, lock_lon = get_gps(mav)
        if lock_lat is not None:
            log.info(f"  No lock passed — using current GPS: {lock_lat:.7f}, {lock_lon:.7f}")

    while True:
        now_t = time.time()
        if now_t - t_start > 60:
            log.warning("  Descent timeout → LAND mode")
            stop_drone(mav)
            set_mode(mav, 'LAND')
            return

        alt = get_alt(mav)
        if alt is None:
            time.sleep(0.2)
            continue

        # ── Spam GPS position at current descent altitude ─────────────────
        # This keeps FC fighting wind — it doesn't fight downward motion
        # because we then override with a velocity command anyway.
        if lock_lat is not None and now_t - t_last_gps_spam >= GPS_SPAM_PERIOD:
            spam_position(mav, lock_lat, lock_lon, max(alt, 0.5))
            t_last_gps_spam = now_t

        log.info(f"  Alt: {alt:.2f}m")

        if alt <= LAND_ALT_M:
            log.info(f"  ✓ {alt:.2f}m ≤ {LAND_ALT_M}m → LAND mode")
            stop_drone(mav, dur=0.5)
            set_mode(mav, 'LAND')
            # Wait for disarm = on ground
            t_land = time.time()
            while time.time() - t_land < 60:
                if not mav.motors_armed():
                    log.info("  ✓ Disarmed — landed!")
                    return
                a = get_alt(mav)
                if a is not None:
                    log.info(f"  Landing… alt={a:.2f}m")
                    if a < 0.15:
                        time.sleep(2)
                        return
                else:
                    log.info("  Landing…")
                time.sleep(1)
            return

        rgb    = cam.capture_array()
        frame  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        banner = _detect_descent(frame)    # [F8] rolling window during descent

        if banner:
            cx, cy   = banner["center"]
            offset_x = cx - cx_f
            offset_y = cy - cy_f
            log.info(f"  Banner ({cx},{cy}) off({offset_x:+d},{offset_y:+d}) "
                     f"alt={alt:.2f}m")
            save_debug(frame, banner, label=f"DESCEND alt={alt:.2f}m")

            vf = max(-MAX_ALIGN_SPEED,
                     min(MAX_ALIGN_SPEED, offset_y * KPY))
            vr = max(-MAX_ALIGN_SPEED,
                     min(MAX_ALIGN_SPEED, offset_x * KPX))
            if 0 < abs(vf) < MIN_ALIGN_SPEED:
                vf = math.copysign(MIN_ALIGN_SPEED, vf)
            if 0 < abs(vr) < MIN_ALIGN_SPEED:
                vr = math.copysign(MIN_ALIGN_SPEED, vr)

            # After lateral correction, update GPS lock to new position
            if lock_lat is not None:
                nlat, nlon = get_gps(mav)
                if nlat is not None:
                    lock_lat, lock_lon = nlat, nlon
        else:
            log.info(f"  Banner not visible — GPS holding, descending  alt={alt:.2f}m")
            save_debug(frame, None,
                       label=f"DESCEND no-detect alt={alt:.2f}m", force=True)
            vf, vr = 0.0, 0.0

        send_velocity(mav, vf, vr, down=DESCEND_SPEED, dur=0.3)

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION MODE  — python3 test_banner_land.py --calibrate
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(cam):
    log.info("=" * 60)
    log.info("HSV CALIBRATION — point at banner, Ctrl+C to stop")
    log.info(f"Saves to {DEBUG_DIR}/hsv_*.jpg  (refresh every 1 s)")
    log.info("=" * 60)
    kernel = np.ones((5, 5), np.uint8)
    cx = CAM_WIDTH  // 2
    cy = CAM_HEIGHT // 2
    roi = 80
    n = 0
    try:
        while True:
            rgb   = cam.capture_array()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            tuned = cv2.morphologyEx(
                cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN),
                cv2.MORPH_OPEN, kernel)
            broad = cv2.morphologyEx(
                cv2.inRange(hsv, np.array([20, 30, 30]),
                            np.array([100, 255, 255])),
                cv2.MORPH_OPEN, kernel)

            r  = hsv[cy-roi//2:cy+roi//2, cx-roi//2:cx+roi//2]
            hm = r[:, :, 0].mean()
            sm = r[:, :, 1].mean()
            vm = r[:, :, 2].mean()

            diag = frame.copy()
            cv2.rectangle(diag, (cx-roi//2, cy-roi//2),
                          (cx+roi//2, cy+roi//2), (0, 255, 255), 2)

            # Run detection on this frame
            raw = _detect_raw(frame)
            det_label = (f"DETECTED area={raw['area']:.0f} "
                         f"sol={raw['solidity']:.2f}"
                         if raw else "NO DETECT")
            lines = [
                f"ROI  H: {r[:,:,0].min():.0f}-{r[:,:,0].max():.0f}  mean={hm:.1f}",
                f"     S: {sm:.1f}   V: {vm:.1f}",
                f"Broad px: {cv2.countNonZero(broad)}   "
                f"Tuned px: {cv2.countNonZero(tuned)}",
                f"Suggested LOWER = [{max(0,int(hm)-14)},"
                f"{max(40,int(sm)-20)},{max(30,int(vm)-20)}]",
                f"Suggested UPPER = [{min(179,int(hm)+14)},255,255]",
                det_label,
            ]
            for i, t in enumerate(lines):
                col = (0, 255, 0) if "DETECT" in t and "NO" not in t else (0, 200, 255)
                cv2.putText(diag, t, (8, 22+i*22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
            cv2.imwrite(os.path.join(DEBUG_DIR, "hsv_frame.jpg"), diag)
            cv2.imwrite(os.path.join(DEBUG_DIR, "hsv_mask.jpg"),
                        cv2.cvtColor(tuned, cv2.COLOR_GRAY2BGR))
            n += 1
            log.info(f"[{n}] H={hm:.0f} S={sm:.0f} V={vm:.0f} "
                     f"tuned={cv2.countNonZero(tuned)}  {det_label}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        log.info("Calibration done.")

# ─────────────────────────────────────────────────────────────────────────────
# GROUND CHECK  [F10] — saves detailed HSV mask for debugging
# ─────────────────────────────────────────────────────────────────────────────

def ground_check(cam):
    log.info("\n── GROUND CHECK ──")
    rgb   = cam.capture_array()
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    mask_morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _kernel)
    mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, _kernel)
    gp    = cv2.countNonZero(mask)

    # Annotated colour frame
    ann = frame.copy()
    cnts, _ = cv2.findContours(
        mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        a = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        col = (0, 255, 0) if a >= MIN_AREA else (0, 100, 255)
        cv2.rectangle(ann, (x, y), (x+w, y+h), col, 2)
        cv2.putText(ann, f"{a:.0f}", (x, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    cv2.imwrite(os.path.join(DEBUG_DIR, "check_frame.jpg"), ann)
    cv2.imwrite(os.path.join(DEBUG_DIR, "check_mask.jpg"),
                cv2.cvtColor(mask_morph, cv2.COLOR_GRAY2BGR))

    raw = _detect_raw(frame)
    if raw:
        log.info(f"  ✓ DETECTED  center={raw['center']}  "
                 f"area={raw['area']:.0f}  "
                 f"sol={raw['solidity']:.2f}  "
                 f"rect={raw['rectangularity']:.2f}")
    else:
        log.info(f"  ✗ NOT DETECTED  raw_gpx={gp}")
        if gp < 100:
            log.warning("    → Very few green pixels. "
                        "Run --calibrate, or fix lighting / HSV values.")
        else:
            log.warning(
                f"    → Green found ({gp}px) but shape filter rejected it. "
                f"Check: MIN_AREA={MIN_AREA} MIN_RECT={MIN_RECTANGULARITY} "
                f"MIN_SOL={MIN_SOLIDITY}  "
                f"  Saved: {DEBUG_DIR}/check_mask.jpg")
    log.info(f"  Debug images: {DEBUG_DIR}/check_*.jpg")
    return raw is not None

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def preview_thread(cam_ref, stop_event):
    """
    Background thread: grabs frames, applies HSV mask, shows live window.
    Uses the global LOWER_GREEN / UPPER_GREEN (updated by in-flight calibration).
    Press 'q' in the preview window to quit (on a display-connected machine).
    """
    win = "SQUADRONE — Live Preview"
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 640, 360)
    except Exception:
        return   # headless — no display available, skip silently

    kernel = np.ones((5, 5), np.uint8)
    while not stop_event.is_set():
        try:
            rgb   = cam_ref[0].capture_array()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask  = cv2.morphologyEx(
                cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN),
                cv2.MORPH_OPEN, kernel)

            # Green tint overlay where mask is active
            overlay = frame.copy()
            overlay[mask > 0] = [0, 200, 0]
            display = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

            # Draw crosshair
            cx, cy = CAM_WIDTH // 2, CAM_HEIGHT // 2
            cv2.line(display, (cx-20, cy), (cx+20, cy), (0,255,255), 1)
            cv2.line(display, (cx, cy-20), (cx, cy+20), (0,255,255), 1)

            # Pixel count
            gp = cv2.countNonZero(mask)
            col = (0,255,0) if gp > MIN_AREA else (0,100,255)
            cv2.putText(display, f"green_px={gp}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
            cv2.putText(display,
                        f"HSV [{LOWER_GREEN[0]}-{UPPER_GREEN[0]}]",
                        (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200,200,200), 1)

            cv2.imshow(win, display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        except Exception:
            time.sleep(0.1)
    cv2.destroyAllWindows()


def auto_calibrate_inflight(cam):
    """
    Called after takeoff, while hovering at TAKEOFF_ALT_M.
    Camera is now pointing straight down — banner may be visible.
    Samples 3 seconds of frames (60 frames @ 20fps) and refines HSV.
    Safe to call even if banner isn't in view — keeps defaults.
    """
    log.info("  [calib] Auto-calibrating at altitude — sampling 3 s…")
    calibrate_banner_colour(cam, n_frames=60)


def main():
    import threading

    calibrate_only = "--calibrate"  in sys.argv
    skip_calib     = "--skip-calib" in sys.argv
    check_only     = "--check-only" in sys.argv

    log.info("=" * 60)
    log.info("SQUADRONE — Banner Land Test v5.2")
    log.info("Takeoff → auto-calibrate → find banner → align → LAND")
    log.info("=" * 60)

    cam = mav = None
    stop_preview = threading.Event()
    cam_ref = [None]   # mutable ref for preview thread

    try:
        log.info("\n[1] Camera…")
        cam = start_camera()
        cam_ref[0] = cam

        if calibrate_only:
            run_calibration(cam)
            return

        # ── Live preview (background thread — non-blocking) ───────────────
        prev_t = threading.Thread(
            target=preview_thread, args=(cam_ref, stop_preview),
            daemon=True)
        prev_t.start()
        log.info("  Live preview started (headless = silently skipped)")

        # ── Skip the ground-check banner gate entirely ────────────────────
        # Camera is pointing sideways / down on the ground, not at the banner.
        # Just verify the camera is alive and saving frames.
        log.info("\n[2] Camera warmup + ground snapshot…")
        ground_check(cam)    # saves check_*.jpg but does NOT gate flight
        log.info("  (Ground check is informational only — flight proceeds regardless)")

        if check_only:
            log.info("  --check-only: done.")
            stop_preview.set()
            return

        log.info("\n[3] MAVLink…")
        mav = connect_mavlink()

        input("\n  >>> ENTER to arm + takeoff   (Ctrl+C = abort) <<<\n")

        log.info("\n[4] Arm + takeoff…")
        if not arm_and_takeoff(mav, TAKEOFF_ALT_M):
            set_mode(mav, 'RTL')
            stop_preview.set()
            return

        time.sleep(1.5)   # settle at altitude

        # ── Auto-calibrate now — camera is pointing down at the field ─────
        if not skip_calib:
            log.info("\n[5] In-flight auto-calibration…")
            auto_calibrate_inflight(cam)
        else:
            log.info(f"\n[5] Calibration skipped — HSV={LOWER_GREEN}→{UPPER_GREEN}")

        log.info("\n[6] Lock heading + yaw…")
        capture_heading(mav)
        lock_yaw(mav)
        time.sleep(0.5)

        log.info("\n[7] Align above banner…")
        lock_lat, lock_lon = state_align(mav, cam)

        log.info("\n[8] Descend + land on banner…")
        state_land(mav, cam, lock_lat=lock_lat, lock_lon=lock_lon)

        log.info("\n✓ MISSION COMPLETE")
        log.info(f"  Debug frames: {FRAME_DIR}/")
        log.info(f"  Live frame:   {DEBUG_DIR}/debug_frame.jpg")
        stop_preview.set()

    except KeyboardInterrupt:
        log.warning("\n!!! Ctrl+C — RTL !!!")
        stop_preview.set()
        if mav:
            stop_drone(mav)
            set_mode(mav, 'RTL')

    except Exception as e:
        log.error(f"\n!!! EXCEPTION: {e} !!!")
        import traceback; traceback.print_exc()
        stop_preview.set()
        if mav:
            stop_drone(mav)
            set_mode(mav, 'RTL')
        raise

    finally:
        if cam:
            cam.stop()
        log.info("Camera stopped.")


if __name__ == "__main__":
    main()