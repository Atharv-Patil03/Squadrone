#!/usr/bin/env python3
"""
SQUADRONE — Banner Search + Align   (v2)
File: banner_search_dual.py
Team: Squadrone AT2026RC60, PVGCOET Pune, SAEINDIA AeroTHON 2026

WHAT IT DOES
  1. Heading on the ground = CORRIDOR heading. Put the drone NOSE towards the
     corridor before running. Yaw stays locked to it for the whole flight.
  2. Takes off to SEARCH_ALT_M.
  3. FORWARD SWEEP with the DOWN camera (banner is always ahead):
        try 1: forward to 2 m -> left 2 m -> centre -> right 2 m -> centre
        try 2: forward to 4 m -> left -> centre -> right -> centre
        ... up to MAX_TRIES (5 tries = 10 m ahead). Never flies backwards.
     The down camera watches the whole time, also while moving.
  4. AI camera (forward) is a FALLBACK only — "banner too far off":
        - one look at the start (banner far ahead -> fly straight to it)
        - one look before giving up
     If it sees the banner, the drone flies to its estimated spot and the
     DOWN camera takes over. Nothing found at all -> RTL.
  5. GLIMPSE -> LOCK -> CHASE: the moment the DOWN cam sees the banner, even in a
     single frame, the BANNER's coordinate is computed
        (drone position + pixel offset converted to metres, rear-cam offset
         compensated) and locked. The drone flies to that coordinate and every
     new sighting pushes the lock towards the latest view. Arrives and confirms
     -> align. Arrives and nothing there -> false sighting, that spot is
     remembered and ignored, search resumes at the same waypoint.
     Pixel->metres is skipped while the drone is tilted > MAX_TILT_DEG
     (camera not looking straight down).
  6. Align: hold the locked coordinate (FC position control fights wind),
     keep pushing it with each sighting until centred for ALIGN_HOLD_S.
  7. Centred + facing corridor -> FINAL_ACTION ("land" for test, "hover").

RUN
  python3 banner_search_dual.py              # fly
  python3 banner_search_dual.py --bench      # NO flight: cameras + detection
  python3 banner_search_dual.py --ai-cam     # also use the AI camera as fallback

BENCH CHECKS (props OFF)
  - Banner under the DOWN cam, move it towards the NOSE: "fwd" must go +.
    Move it to the drone's RIGHT: "rgt" must go +. Else flip DOWN_FLIP_Y/X.
  - Bench also writes ~/squadrone/mask_down.jpg (what the detector sees).
"""

import os
import sys
import math
import time
import logging
import threading
from collections import deque

import cv2
import numpy as np

Picamera2 = None   # imported in main() so this file can be tested off the Pi
mavutil = None

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE   = 921600

# Flight
SEARCH_ALT_M     = 5.0
TAKEOFF_ACCEPT   = 0.90
TAKEOFF_TIMEOUT  = 45
SEARCH_SPEED_MS  = 1.0
ALIGN_SPEED_MS   = 0.7
CORRIDOR_HEADING_DEG = None   # None = heading on the ground at start

# Forward sweep
SEARCH_STEP_FWD_M = 2.0   # each try goes this much further forward
SEARCH_SIDE_M     = 2.0   # left / right leg length
MAX_TRIES         = 5
DWELL_S           = 1.2   # hover + look at each waypoint
ARRIVE_TOL_M      = 0.35

# AI camera fallback (horizontal camera: at 5 m it only sees ground > ~11 m away,
# and distance from a shallow angle is unreliable -> we trust the DIRECTION and
# move in steps, re-checking each time)
FWD_LOOK_S        = 2.0   # how long to look with the AI cam
FWD_MAX_RANGE_M   = 30.0  # ignore estimates further than this
FWD_STEP_M        = 4.0   # move this far towards the banner per step
FWD_MAX_STEPS     = 6
FWD_EXTRA_TRIES   = 2     # down-cam sweep tries after AI-cam steps

# Glimpse -> coordinate lock -> chase
GLIMPSE_FRESH_S     = 0.3   # a single-frame sighting younger than this counts
GLIMPSE_MIN_SCORE   = 0.45  # single frames must look clearly like a banner
CHASE_CONFIRM_S     = 3.0   # at the coordinate: no confirm within this -> reject
CHASE_TIMEOUT_S     = 25
REJECT_RADIUS_M     = 1.5   # ignore new glimpses near a rejected spot
MAX_TILT_DEG        = 8.0   # don't trust pixel->metres while tilted more than this

# Alignment
ALIGN_TOL_M      = 0.20
ALIGN_HOLD_S     = 1.5
ALIGN_TIMEOUT_S  = 90
LOCK_EMA         = 0.35
LOCK_MAX_SPEED   = 0.6
LOCK_GATE_M      = 1.2     # sightings further than this from the lock = other object
LOST_GIVEUP_S    = 8.0

# Final action
FINAL_ACTION   = "land"    # "land" (test) or "hover" (mission)
LAND_SWITCH_M  = 1.5
DESCENT_MS     = 0.35
HOVER_ALT_M    = 3.0
HOVER_TIME_S   = 10

# Cameras
CAM_W, CAM_H = 640, 480
AWB_DAYLIGHT = True        # fixed daylight white balance (see detection notes)

DOWN_MODEL        = "imx519"   # ArduCam, rear, straight down
DOWN_HFOV_DEG     = 66.0
DOWN_VFOV_DEG     = 51.0
DOWN_OFFSET_FWD_M = -0.10      # camera vs drone centre (negative = behind) — measure
DOWN_FLIP_X       = False      # image right  = drone right
DOWN_FLIP_Y       = True       # survey flight 20-10 + banner drift: fwd/back reversed

FWD_MODEL         = "imx500"   # Pi AI Camera, nose, ahead + down
FWD_HFOV_DEG      = 66.0
FWD_VFOV_DEG      = 51.0
FWD_TILT_DEG      = 0.0        # axis angle below horizontal (photo: ~level)
FWD_OFFSET_FWD_M  = 0.15
FWD_FLIP_X        = False

# ── Detection ────────────────────────────────────────────────────────────────
# Colour gate = HSV range (team-confirmed) AND Lab a* (green-red axis).
# Lab a* barely changes with brightness/shadow, and rejects dull olive grass.
LOWER_GREEN = np.array([35,  80,  50])
UPPER_GREEN = np.array([85, 255, 255])
LAB_A_MAX   = 120          # OpenCV a*: 128 = neutral, lower = greener
# Shape / texture (all rotation-invariant)
MIN_FILL      = 0.60       # contour area / min-area rotated rectangle
MIN_SOLIDITY  = 0.75       # contour area / convex hull
MAX_ELONG     = 5.0        # long side / short side of rotated rectangle
TEX_PIX_MAX   = 18         # per-pixel texture gate (grass ~28, creased banner 11-14)
MAX_TEXTURE   = 18.0       # mean texture inside a blob (same scale)
MAX_AREA_FRAC = 1.01       # no limit: up close the banner may fill the whole frame
AMBIGUOUS     = 0.85       # 2nd-best score >= 85 % of best -> no decision
DOWN_MIN_AREA = 400
FWD_MIN_AREA  = 120
# Optional: real banner size (m). Rejects blobs far too big/small for altitude.
BANNER_SIZE_M = None       # e.g. (1.0, 0.6) — set once you know it
SIZE_TOL      = 3.0        # accept 1/3x .. 3x the expected pixel area

DOWN_WINDOW, DOWN_NEED = 8, 5
FWD_WINDOW,  FWD_NEED  = 6, 4
CONFIRM_STALE_S        = 1.0

DEBUG_DIR  = os.path.expanduser("~/squadrone")
FRAME_DIR  = os.path.join(DEBUG_DIR, "frames_dual")
MAX_FRAMES = 400

os.makedirs(FRAME_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(os.path.join(DEBUG_DIR, "dual_search.log"))])
log = logging.getLogger("dual")

_k5 = np.ones((5, 5), np.uint8)
HEAL_PX = 17                                   # closes cuts up to ~15 px wide
_k_heal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (HEAL_PX, HEAL_PX))

# ═════════════════════════════════════════════════════════════════════════════
# GEOMETRY
# ═════════════════════════════════════════════════════════════════════════════

def body_to_ned(fwd, rgt, yaw):
    return (fwd * math.cos(yaw) - rgt * math.sin(yaw),
            fwd * math.sin(yaw) + rgt * math.cos(yaw))


def ned_to_body(n, e, yaw):
    return (n * math.cos(yaw) + e * math.sin(yaw),
            -n * math.sin(yaw) + e * math.cos(yaw))


def down_m_per_px(alt):
    return 2 * alt * math.tan(math.radians(DOWN_HFOV_DEG / 2)) / CAM_W


def down_pixel_to_body(px, py, alt):
    """DOWN cam pixel -> banner (fwd, rgt) metres from DRONE CENTRE."""
    dx = (px - CAM_W / 2) / (CAM_W / 2)
    dy = (py - CAM_H / 2) / (CAM_H / 2)
    if DOWN_FLIP_X:
        dx = -dx
    if DOWN_FLIP_Y:
        dy = -dy
    rgt = dx * alt * math.tan(math.radians(DOWN_HFOV_DEG / 2))
    fwd = dy * alt * math.tan(math.radians(DOWN_VFOV_DEG / 2))
    return fwd + DOWN_OFFSET_FWD_M, rgt


def fwd_pixel_to_body(px, py, alt):
    """AI cam pixel -> rough banner (fwd, rgt) metres, or None above horizon."""
    dx = (px - CAM_W / 2) / (CAM_W / 2)
    dy = (py - CAM_H / 2) / (CAM_H / 2)
    if FWD_FLIP_X:
        dx = -dx
    depress = math.radians(FWD_TILT_DEG) + math.atan(
        dy * math.tan(math.radians(FWD_VFOV_DEG / 2)))
    if depress <= math.radians(3):
        return None
    ground = alt / math.tan(depress)
    bearing = math.atan(dx * math.tan(math.radians(FWD_HFOV_DEG / 2)))
    return ground + FWD_OFFSET_FWD_M, ground / max(math.cos(bearing), 1e-3) * math.sin(bearing)


def sweep_waypoints(try_idx):
    """Try k (0-based): forward to depth, left, centre, right, centre."""
    d = SEARCH_STEP_FWD_M * (try_idx + 1)
    s = SEARCH_SIDE_M
    return [(d, 0.0), (d, -s), (d, 0.0), (d, s), (d, 0.0)]

# ═════════════════════════════════════════════════════════════════════════════
# DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def texture_map(bgr):
    """Local high-frequency energy. Grass/leaves = high, printed banner = low.
    (Laplacian ignores smooth shading, so shadows don't count as texture.)"""
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    lap = cv2.convertScaleAbs(cv2.Laplacian(g, cv2.CV_16S, ksize=3))
    return cv2.blur(lap, (9, 9))


def green_mask(bgr):
    """Green (HSV AND Lab a*) AND smooth (low texture), then morphology."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    a = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 1]
    tex = texture_map(bgr)
    m = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    m = cv2.bitwise_and(m, cv2.inRange(a, 0, LAB_A_MAX))
    m = cv2.bitwise_and(m, cv2.inRange(tex, 0, TEX_PIX_MAX))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _k5)
    # heal thin cuts (creases, folds, streaks, shadow edges) so the banner is
    # ONE blob again; grass stays out because it was removed pixel-by-pixel
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _k_heal)
    return m, tex


def detect_raw(bgr, min_area, alt=None, why=None):
    """
    Score every green blob, return the best banner candidate (or None).
      fill     = area / rotated-rectangle area   (works at ANY rotation)
      solidity = area / convex hull              (grass/trees are ragged)
      texture  = local detail inside blob        (printed banner is flat)
    why: pass a list to get a plain-English reason for every rejection.
    """
    mask, tex = green_mask(bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = MAX_AREA_FRAC * CAM_W * CAM_H
    if why is not None:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        a = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 1]
        col = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
        col_lab = cv2.bitwise_and(col, cv2.inRange(a, 0, LAB_A_MAX))
        n = CAM_W * CAM_H / 100.0
        why.append(f"green(HSV) {cv2.countNonZero(col) / n:.0f}%  "
                   f"+Lab {cv2.countNonZero(col_lab) / n:.0f}%  "
                   f"+smooth(tex<{TEX_PIX_MAX}) {cv2.countNonZero(mask) / n:.0f}%  "
                   f"blobs={len(cnts)}")
        if cv2.countNonZero(col):
            why.append(f"  green pixels: median H={np.median(hsv[:, :, 0][col > 0]):.0f} "
                       f"S={np.median(hsv[:, :, 1][col > 0]):.0f} "
                       f"V={np.median(hsv[:, :, 2][col > 0]):.0f}  "
                       f"Lab a*={np.median(a[col > 0]):.0f} (max {LAB_A_MAX})  "
                       f"texture median={np.median(tex[col > 0]):.0f} (max {TEX_PIX_MAX})")
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    def reject(c, msg):
        if why is not None and len(why) < 6:
            why.append(f"  blob area={cv2.contourArea(c):.0f}px REJECTED: {msg}")

    exp_area = None
    if BANNER_SIZE_M and alt and alt > 0.5:
        mpp = down_m_per_px(alt)
        exp_area = (BANNER_SIZE_M[0] / mpp) * (BANNER_SIZE_M[1] / mpp)

    cands = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            if why is not None and area > 50:
                reject(c, f"too small (< {min_area})")
            continue
        if area > max_area:
            reject(c, f"too big (> {MAX_AREA_FRAC:.0%} of frame)")
            continue
        (rcx, rcy), (rw, rh), ang = cv2.minAreaRect(c)
        if rw < 2 or rh < 2:
            continue
        fill = area / (rw * rh)
        elong = max(rw, rh) / min(rw, rh)
        hull_a = cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull_a if hull_a else 0
        if fill < MIN_FILL:
            reject(c, f"not rectangular: fill={fill:.2f} < {MIN_FILL}")
            continue
        if solidity < MIN_SOLIDITY:
            reject(c, f"ragged edge: solidity={solidity:.2f} < {MIN_SOLIDITY}")
            continue
        if elong > MAX_ELONG:
            reject(c, f"too long/thin: {elong:.1f} > {MAX_ELONG}")
            continue
        if exp_area and not (exp_area / SIZE_TOL <= area <= exp_area * SIZE_TOL):
            reject(c, f"wrong size for altitude (expected ~{exp_area:.0f}px)")
            continue
        x, y, w, h = cv2.boundingRect(c)
        blob = np.zeros((h, w), np.uint8)
        cv2.drawContours(blob, [c - [x, y]], -1, 255, -1)
        texture = float(cv2.mean(tex[y:y + h, x:x + w], mask=blob)[0])
        if texture > MAX_TEXTURE:
            reject(c, f"too textured: {texture:.1f} > {MAX_TEXTURE}")
            continue
        if why is not None:
            why.append(f"  blob area={area:.0f}px OK  fill={fill:.2f} sol={solidity:.2f} "
                       f"tex={texture:.1f}")
        m = cv2.moments(c)
        score = (fill * solidity * (1.0 - 0.5 * texture / MAX_TEXTURE)
                 * min(1.0, math.sqrt(area / (min_area * 6))))
        cands.append({"center": (m["m10"] / m["m00"], m["m01"] / m["m00"]),
                      "box": (x, y, w, h), "rbox": ((rcx, rcy), (rw, rh), ang),
                      "area": area, "fill": fill, "solidity": solidity,
                      "texture": texture, "score": score})
    if not cands:
        return None, mask
    cands.sort(key=lambda d: d["score"], reverse=True)
    if len(cands) >= 2 and cands[1]["score"] >= AMBIGUOUS * cands[0]["score"]:
        if why is not None:
            why.append(f"  REJECTED: two equally good blobs (scores {cands[0]['score']:.2f} "
                       f"/ {cands[1]['score']:.2f}) — e.g. banner split by a shadow/crease")
        return None, mask        # two equally good blobs — don't guess
    return cands[0], mask


class CamWorker(threading.Thread):
    """One camera: capture + detect in its own thread. `active=False` pauses it."""

    def __init__(self, name, model, min_area, window, need, active=True):
        super().__init__(daemon=True)
        self.name, self.model = name, model
        self.min_area, self.need = min_area, need
        self.hist = deque(maxlen=window)
        self.lk = threading.Lock()
        self.frame = self.raw = self.confirmed = self.mask = None
        self.t_last_det = 0.0
        self.fps = 0.0
        self.alt = None            # mission sets this (for size gate)
        self.active = active
        self._stop_evt = threading.Event()
        self.cam = None

    def open(self, index):
        self.cam = Picamera2(index)
        cfg = self.cam.create_preview_configuration(
            main={"size": (CAM_W, CAM_H), "format": "RGB888"}, buffer_count=4)
        self.cam.configure(cfg)
        self.cam.start()
        time.sleep(1.5)
        ctrls = {}
        if AWB_DAYLIGHT:
            ctrls.update({"AwbEnable": True, "AwbMode": 5})   # 5 = Daylight
        try:
            self.cam.set_controls(ctrls)
        except Exception as e:
            log.warning(f"  {self.name}: AWB control not applied ({e})")
        try:
            self.cam.set_controls({"AfMode": 2, "AfSpeed": 1})
        except Exception:
            pass
        log.info(f"  {self.name} cam ({self.model}) on index {index}")

    def set_active(self, on):
        if on and not self.active:
            self.hist.clear()
        self.active = on

    def run(self):
        n, t0 = 0, time.time()
        while not self._stop_evt.is_set():
            if not self.active:
                time.sleep(0.1)
                continue
            try:
                # Picamera2 "RGB888" is already B,G,R byte order = OpenCV BGR.
                frame = self.cam.capture_array()
            except Exception as e:
                log.warning(f"  {self.name} capture error: {e}")
                time.sleep(0.2)
                continue
            det, mask = detect_raw(frame, self.min_area, self.alt)
            now = time.time()
            if det is not None:
                det["t"] = now
            self.hist.append(det)
            if det is not None:
                self.t_last_det = now
            hits = [d for d in self.hist if d is not None]
            conf = None
            if len(hits) >= self.need and now - self.t_last_det < CONFIRM_STALE_S:
                ws = np.arange(1, len(hits) + 1, dtype=float)
                conf = dict(hits[-1])
                conf["center"] = (
                    float(np.average([h["center"][0] for h in hits], weights=ws)),
                    float(np.average([h["center"][1] for h in hits], weights=ws)))
                conf["t"] = now
            with self.lk:
                self.frame, self.raw, self.confirmed, self.mask = frame, det, conf, mask
            n += 1
            if now - t0 >= 2.0:
                self.fps, n, t0 = n / (now - t0), 0, now

    def get(self):
        with self.lk:
            return self.frame, self.raw, self.confirmed

    def get_mask(self):
        with self.lk:
            return self.mask

    def reset(self):
        with self.lk:
            self.hist.clear()
            self.confirmed = None

    def stop(self):
        self._stop_evt.set()
        try:
            if self.cam:
                self.cam.stop()
                self.cam.close()
        except Exception:
            pass


def open_cameras(use_fwd=True):
    info = Picamera2.global_camera_info()
    log.info(f"  Cameras found: {[c.get('Model') for c in info]}")
    idx = {}
    for i, c in enumerate(info):
        idx.setdefault(str(c.get("Model", "")).lower(), c.get("Num", i))
    if DOWN_MODEL not in idx:
        raise RuntimeError(f"DOWN camera '{DOWN_MODEL}' not found")
    down = CamWorker("DOWN", DOWN_MODEL, DOWN_MIN_AREA, DOWN_WINDOW, DOWN_NEED)
    down.open(idx[DOWN_MODEL])
    down.start()
    fwd = None
    if use_fwd and FWD_MODEL in idx:
        try:
            fwd = CamWorker("FWD", FWD_MODEL, FWD_MIN_AREA, FWD_WINDOW, FWD_NEED,
                            active=False)        # fallback: paused until needed
            fwd.open(idx[FWD_MODEL])
            fwd.start()
        except Exception as e:
            log.warning(f"  AI cam failed ({e}) — DOWN cam only")
            fwd = None
    elif use_fwd:
        log.warning(f"  AI cam '{FWD_MODEL}' not found — DOWN cam only")
    return down, fwd

# ═════════════════════════════════════════════════════════════════════════════
# PREVIEW — plain camera image, thin box, crosshair, one status line
# ═════════════════════════════════════════════════════════════════════════════

class Preview:
    def __init__(self):
        self.gui = bool(os.environ.get("DISPLAY"))
        self.t_save = 0.0
        self.n_saved = 0
        self.status = ""
        if self.gui:
            try:
                cv2.namedWindow("SQUADRONE", cv2.WINDOW_NORMAL)
            except Exception:
                self.gui = False

    @staticmethod
    def _draw(frame, raw, conf, label):
        if frame is None:
            return None
        img = frame.copy()
        cx, cy = CAM_W // 2, CAM_H // 2
        cv2.drawMarker(img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 24, 1)
        det = conf or raw
        if det:
            col = (0, 255, 0) if conf else (0, 200, 255)
            box = cv2.boxPoints(det["rbox"]).astype(np.int32)
            cv2.polylines(img, [box], True, col, 2)
            if conf:
                cv2.circle(img, (int(det["center"][0]), int(det["center"][1])), 4, col, -1)
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def update(self, down, fwd, status=None):
        if status is not None:
            self.status = status
        now = time.time()
        if not self.gui and now - self.t_save < 0.5:
            return
        tiles = [self._draw(*down.get(), "DOWN")]
        if fwd is not None and fwd.active:
            tiles.append(self._draw(*fwd.get(), "AI cam"))
        tiles = [t for t in tiles if t is not None]
        if not tiles:
            return
        img = np.hstack(tiles)
        cv2.putText(img, self.status[:110], (8, CAM_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if self.gui:
            cv2.imshow("SQUADRONE", img)
            cv2.waitKey(1)
        if now - self.t_save >= 0.5:
            self.t_save = now
            cv2.imwrite(os.path.join(DEBUG_DIR, "live.jpg"), img)
            if self.n_saved < MAX_FRAMES:
                cv2.imwrite(os.path.join(FRAME_DIR, f"d{self.n_saved:04d}.jpg"), img)
                self.n_saved += 1

    def close(self):
        if self.gui:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

# ═════════════════════════════════════════════════════════════════════════════
# MAVLINK — one reader thread owns all incoming messages
# ═════════════════════════════════════════════════════════════════════════════

class Drone:
    def __init__(self):
        self.mav = None
        self.state = {}
        self.lk = threading.Lock()
        self._stop_evt = threading.Event()
        self.ground_z = None

    def connect(self):
        log.info(f"Connecting {SERIAL_PORT} @ {BAUD_RATE}…")
        self.mav = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUD_RATE)
        # Same as survey.py: wait for the FLIGHT CONTROLLER's own heartbeat,
        # not a radio / GCS one (those caused the false "disarmed -> RTL").
        t0 = time.time()
        while True:
            hb = self.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
            if (hb and hb.get_srcSystem() != 0
                    and hb.get_srcComponent() == mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1
                    and hb.autopilot != mavutil.mavlink.MAV_AUTOPILOT_INVALID):
                self.mav.target_system = hb.get_srcSystem()
                self.mav.target_component = hb.get_srcComponent()
                break
            if time.time() - t0 > 30:
                mv = self.mav.mav
                nb = getattr(mv, "total_bytes_received", 0)
                npk = getattr(mv, "total_packets_received", 0)
                why = ("NOTHING received -> Pixhawk not powered / cable / port busy "
                       f"(sudo fuser {SERIAL_PORT})" if nb == 0 else
                       f"{nb} bytes, 0 packets -> wrong baud" if npk == 0 else
                       f"{npk} packets but no flight-controller heartbeat")
                raise RuntimeError(f"No heartbeat from FC in 30 s: {why}")
        log.info(f"  Connected sys={self.mav.target_system}")
        threading.Thread(target=self._reader, daemon=True).start()
        for msg_id, hz in ((30, 20), (32, 20), (33, 10), (24, 2)):
            self.mav.mav.command_long_send(
                self.mav.target_system, self.mav.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                msg_id, int(1e6 / hz), 0, 0, 0, 0, 0)
        t0 = time.time()
        while time.time() - t0 < 5:
            if self.get("LOCAL_POSITION_NED") and self.get("ATTITUDE"):
                return
            time.sleep(0.1)
        log.warning("  No LOCAL_POSITION_NED yet — wait for GPS lock / EKF origin")

    def _reader(self):
        while not self._stop_evt.is_set():
            try:
                m = self.mav.recv_match(blocking=True, timeout=0.5)
            except Exception:
                continue
            if m is None or m.get_type() == "BAD_DATA":
                continue
            # Only trust the FLIGHT CONTROLLER (our sysid, autopilot component).
            # The FC also forwards heartbeats from Mission Planner / ELRS /
            # radios — those say "disarmed, mode 0" and must be ignored.
            if m.get_srcSystem() != self.mav.target_system:
                continue
            if m.get_type() == "HEARTBEAT" and (
                    m.get_srcComponent() != mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1
                    or m.type in (mavutil.mavlink.MAV_TYPE_GCS,
                                  mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER)
                    or m.autopilot == mavutil.mavlink.MAV_AUTOPILOT_INVALID):
                continue
            with self.lk:
                self.state[m.get_type()] = m
            if m.get_type() == "STATUSTEXT":
                log.info(f"  [FC] {m.text}")

    def get(self, t):
        with self.lk:
            return self.state.get(t)

    def pos_ned(self):
        m = self.get("LOCAL_POSITION_NED")
        return (m.x, m.y, m.z) if m else None

    def set_ground(self):
        """Remember the ground level at the takeoff spot (local z drifts and is
        relative to where the EKF started, not to the ground)."""
        zs = []
        t0 = time.time()
        while time.time() - t0 < 1.0:
            m = self.get("LOCAL_POSITION_NED")
            if m:
                zs.append(m.z)
            time.sleep(0.05)
        self.ground_z = float(np.median(zs)) if zs else None
        log.info(f"  Ground level set (local z = {self.ground_z})")

    def alt(self):
        """Height above the takeoff spot, metres."""
        m = self.get("LOCAL_POSITION_NED")
        if m and self.ground_z is not None:
            return self.ground_z - m.z
        g = self.get("GLOBAL_POSITION_INT")
        return g.relative_alt / 1000.0 if g else None

    def speed(self):
        m = self.get("LOCAL_POSITION_NED")
        return math.hypot(m.vx, m.vy) if m else 0.0

    def yaw(self):
        m = self.get("ATTITUDE")
        return m.yaw if m else None

    def tilt_deg(self):
        m = self.get("ATTITUDE")
        return math.degrees(max(abs(m.roll), abs(m.pitch))) if m else 0.0

    def armed(self):
        hb = self.get("HEARTBEAT")
        return bool(hb and hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def mode(self):
        hb = self.get("HEARTBEAT")
        if not hb:
            return None
        for name, mid in self.mav.mode_mapping().items():
            if mid == hb.custom_mode:
                return name
        return None

    def set_mode(self, name, timeout=5):
        mid = self.mav.mode_mapping()[name]
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.mav.mav.set_mode_send(
                self.mav.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mid)
            time.sleep(0.5)
            if self.mode() == name:
                log.info(f"  Mode → {name}")
                return True
        log.warning(f"  Mode {name} not confirmed")
        return False

    def set_speed(self, ms):
        self.mav.mav.command_long_send(
            self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, 0, 1, ms, -1, 0, 0, 0, 0)

    def goto_ned(self, n, e, alt, yaw):
        """Absolute local-NED position + absolute yaw (FC holds it vs wind).
        alt = metres above the takeoff spot."""
        z = (self.ground_z if self.ground_z is not None else 0.0) - alt
        self.mav.mav.set_position_target_local_ned_send(
            0, self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, 0b0000100111111000,
            n, e, z, 0, 0, 0, 0, 0, 0, yaw, 0)

    def arm_and_takeoff(self, alt):
        if not self.set_mode("GUIDED"):
            return False
        self.set_ground()
        log.info("  Arming…")
        t0 = time.time()
        while not self.armed() and time.time() - t0 < 15:
            self.mav.mav.command_long_send(
                self.mav.target_system, self.mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
            time.sleep(1.0)
        if not self.armed():
            log.error("  Arming failed — check pre-arm messages")
            return False
        log.info("  Armed ✓ — taking off")
        time.sleep(0.5)
        self.mav.mav.command_long_send(
            self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, alt)
        t0, t_log, t_disarm = time.time(), 0, None
        while time.time() - t0 < TAKEOFF_TIMEOUT:
            a = self.alt() or 0.0
            if time.time() - t_log > 1.0:
                log.info(f"  Alt {a:.2f} / {alt:.1f} m   mode={self.mode()}")
                t_log = time.time()
            if a >= alt * TAKEOFF_ACCEPT:
                log.info("  Takeoff done ✓")
                return True
            # disarmed must persist 1.5 s — one stray message can't abort a takeoff
            if not self.armed():
                t_disarm = t_disarm or time.time()
                if time.time() - t_disarm > 1.5:
                    log.error("  FC disarmed during takeoff (check Mission Planner messages)")
                    return False
            else:
                t_disarm = None
            time.sleep(0.2)
        a = self.alt() or 0.0
        if a >= alt * 0.7:
            log.warning(f"  Takeoff slow but at {a:.2f} m — continuing")
            return True
        log.error(f"  Takeoff timeout at {a:.2f} m")
        return False

    def close(self):
        self._stop_evt.set()

# ═════════════════════════════════════════════════════════════════════════════
# MISSION
# ═════════════════════════════════════════════════════════════════════════════

class PilotTookOver(Exception):
    pass


class Mission:
    def __init__(self, drone, down, fwd, preview):
        self.d, self.down, self.fwd, self.pv = drone, down, fwd, preview
        self.yaw = 0.0
        self.lock_ne = None       # locked BANNER coordinate (local NED m)
        self.glimpse_ne = None
        self.rejects = []         # coordinates of false sightings
        self.axis_ref = None      # in-flight camera direction check
        self.axis_votes = {"X": [], "Y": []}
        self.axis_good = set()
        self.axes_ok = False
        self.t_alog = 0.0
        self.releasing = False    # True once the script itself hands over (LAND/RTL)
        self.t_slog = 0.0

    def tick(self, status):
        # Pilot flipped the mode switch (Loiter, RTL...) -> script lets go at once
        m = self.d.mode()
        if not self.releasing and m not in (None, "GUIDED") and self.d.armed():
            raise PilotTookOver(m)
        now = time.time()
        if now - self.t_slog > 2.0:              # analytics: 1 line every 2 s
            self.t_slog = now
            raw, conf = self.down.get()[1], self.down.get()[2]
            log.info(f"  [{status[:60]}]  alt {self.d.alt() or 0:.1f} m  tilt "
                     f"{self.d.tilt_deg():.0f}°  down fps {self.down.fps:.0f}  banner "
                     + ("CONFIRMED" if conf else (f"glimpse score {raw['score']:.2f}"
                                                   if raw else "--")))
        self.down.alt = self.d.alt()
        self.pv.update(self.down, self.fwd, status)
        time.sleep(0.05)

    def here(self):
        p = self.d.pos_ned()
        return (p[0], p[1])

    def offset(self, fwd, rgt, origin):
        dn, de = body_to_ned(fwd, rgt, self.yaw)
        return origin[0] + dn, origin[1] + de

    def cur_yaw(self):
        y = self.d.yaw()
        return self.yaw if y is None else y

    def down_conf(self):
        return self.down.get()[2]

    def down_glimpse(self):
        """Fresh single-frame DOWN detection (even a brief one), or None."""
        raw = self.down.get()[1]
        if raw is None or time.time() - raw.get("t", 0) > GLIMPSE_FRESH_S:
            return None
        return raw if raw["score"] >= GLIMPSE_MIN_SCORE else None

    def down_banner_ne(self, det):
        """Detection -> BANNER coordinate (local NED, metres; GPS-fused by EKF).
        Drone position + pixel offset converted to metres. None if tilted."""
        alt, p = self.d.alt(), self.d.pos_ned()
        if alt is None or p is None or alt < 0.5:
            return None
        if self.d.tilt_deg() > MAX_TILT_DEG:
            return None          # camera not pointing straight down right now
        f, r = down_pixel_to_body(det["center"][0], det["center"][1], alt)
        dn, de = body_to_ned(f, r, self.cur_yaw())
        return p[0] + dn, p[1] + de

    def rejected(self, ne):
        return any(math.hypot(ne[0] - q[0], ne[1] - q[1]) < REJECT_RADIUS_M
                   for q in self.rejects)

    def push_lock(self, meas, weight, gate=None):
        """Move the locked banner coordinate towards a new measurement.
        Sightings further than `gate` m from the lock are a DIFFERENT green
        object (grass patch, tarp…) and are ignored."""
        if self.lock_ne is None:
            self.lock_ne = meas
            return True
        gate = LOCK_GATE_M if gate is None else gate
        if math.hypot(meas[0] - self.lock_ne[0], meas[1] - self.lock_ne[1]) > gate:
            return False
        self.lock_ne = (self.lock_ne[0] * (1 - weight) + meas[0] * weight,
                        self.lock_ne[1] * (1 - weight) + meas[1] * weight)
        return True

    # ── fly while the DOWN cam watches ────────────────────────────────────
    def fly_watch(self, n, e, label, dwell=DWELL_S):
        """'glimpse' the moment the DOWN cam sees a banner (even 1 frame) at a
        coordinate not already rejected; 'arrived' otherwise."""
        p = self.here()
        timeout = math.hypot(n - p[0], e - p[1]) / SEARCH_SPEED_MS * 2.5 + 8
        t0, t_arr = time.time(), None
        while True:
            self.d.goto_ned(n, e, SEARCH_ALT_M, self.yaw)
            g = self.down_glimpse() or self.down_conf()
            if g is not None:
                ne = self.down_banner_ne(g)
                if ne is not None and not self.rejected(ne):
                    self.glimpse_ne = ne
                    return "glimpse"
            p = self.here()
            err = math.hypot(n - p[0], e - p[1])
            if err < ARRIVE_TOL_M and t_arr is None:
                t_arr = time.time()
            if t_arr and time.time() - t_arr >= dwell:
                return "arrived"
            if time.time() - t0 > timeout:
                log.warning(f"  {label}: waypoint timeout (err {err:.2f} m)")
                return "arrived"
            self.tick(f"{label}  err={err:.2f}m alt={self.d.alt() or 0:.1f}m")

    def check_axes(self, det):
        """
        In-flight direction self-check. When the drone really moves forward by
        D metres (telemetry), a fixed banner must move BACK by D in the camera
        reading. If it moves the same way, that camera axis is reversed:
        flip it, re-lock, carry on. Same for left/right.
        """
        if self.axes_ok or det is None or self.d.tilt_deg() > MAX_TILT_DEG:
            return False
        alt, p = self.d.alt(), self.d.pos_ned()
        if alt is None or p is None or alt < 1.0:
            return False
        f, r = down_pixel_to_body(det["center"][0], det["center"][1], alt)
        y = self.cur_yaw()
        if self.axis_ref is None or abs(y - self.axis_ref[4]) > math.radians(10):
            self.axis_ref = (p[0], p[1], f, r, y)
            return False
        n0, e0, f0, r0, _ = self.axis_ref
        dfwd, drgt = ned_to_body(p[0] - n0, p[1] - e0, y)
        if math.hypot(dfwd, drgt) < 0.4:
            return False
        self.axis_ref = (p[0], p[1], f, r, y)
        flipped = False
        for axis, moved, seen in (("Y", dfwd, f - f0), ("X", drgt, r - r0)):
            if abs(moved) < 0.3 or abs(seen) < 0.15:
                continue
            votes = self.axis_votes[axis]
            votes.append(seen * moved > 0)          # True = moved the WRONG way
            del votes[:-4]
            if votes.count(True) >= 3:
                global DOWN_FLIP_X, DOWN_FLIP_Y
                if axis == "Y":
                    DOWN_FLIP_Y = not DOWN_FLIP_Y
                else:
                    DOWN_FLIP_X = not DOWN_FLIP_X
                log.warning(f"  ⚠ CAMERA {'FORWARD/BACK' if axis == 'Y' else 'LEFT/RIGHT'} "
                            f"WAS REVERSED — auto-flipped. Set DOWN_FLIP_{axis} = "
                            f"{DOWN_FLIP_Y if axis == 'Y' else DOWN_FLIP_X} permanently.")
                votes.clear()
                self.axis_ref = None
                flipped = True
            elif votes.count(False) >= 3:
                self.axis_good.add(axis)
        if self.axis_good >= {"X", "Y"} and not self.axes_ok:
            self.axes_ok = True
            log.info("  ✓ Camera directions verified in flight")
        if flipped:                                 # lock was built on wrong maths
            fresh = self.down_banner_ne(det)
            if fresh is not None:
                self.lock_ne = fresh
        return flipped

    def chase(self):
        """
        Glimpse -> lock that coordinate -> fly to it, pushing the lock on every
        new sighting. Returns True once the DOWN cam CONFIRMS the banner there
        (then align takes over from the same lock), False if it was a false
        sighting (coordinate is remembered and ignored from now on).
        """
        self.lock_ne = self.glimpse_ne
        log.info(f"  ◎ Glimpse → coordinate locked N={self.lock_ne[0]:.2f} "
                 f"E={self.lock_ne[1]:.2f}, chasing")
        self.d.set_speed(ALIGN_SPEED_MS)
        t0 = time.time()
        t_arr = None
        sightings = 1
        while time.time() - t0 < CHASE_TIMEOUT_S:
            g = self.down_glimpse()
            if g is not None and not self.check_axes(g):
                meas = self.down_banner_ne(g)
                if meas is not None and self.push_lock(meas, 0.5, gate=2.0):
                    sightings += 1                   # lock pulled to latest view
            c = self.down_conf()
            if c is not None and (self.down_banner_ne(c) is None or math.hypot(
                    *(x - y for x, y in zip(self.down_banner_ne(c), self.lock_ne))) < 2.0):
                log.info(f"  ★ Confirmed after {sightings} sightings")
                return True
            self.d.goto_ned(self.lock_ne[0], self.lock_ne[1], SEARCH_ALT_M, self.yaw)
            p = self.here()
            err = math.hypot(self.lock_ne[0] - p[0], self.lock_ne[1] - p[1])
            if err < ARRIVE_TOL_M and t_arr is None:
                t_arr = time.time()
            if t_arr and time.time() - t_arr > CHASE_CONFIRM_S:
                break
            self.tick(f"CHASE err={err:.2f}m sightings={sightings}")
        log.info("  ✗ Not confirmed at the locked coordinate — false sighting, "
                 "ignoring this spot")
        self.rejects.append(self.lock_ne)
        self.lock_ne = None
        self.d.set_speed(SEARCH_SPEED_MS)
        return False

    def sweep(self, origin, tries):
        """Forward sweep from origin. True when a banner is confirmed."""
        for t in range(tries):
            wps = sweep_waypoints(t)
            log.info(f"\n── SWEEP try {t + 1}/{tries}  depth {wps[0][0]:.0f} m ──")
            i = 0
            while i < len(wps):
                f, r = wps[i]
                n, e = self.offset(f, r, origin)
                lbl = f"try {t + 1}/{tries}  wp {i + 1}/5  ({f:.0f}F {r:+.0f}R)"
                if self.fly_watch(n, e, lbl) == "glimpse":
                    if self.chase():
                        return True
                    continue              # false sighting: redo this waypoint
                i += 1
        return False

    # ── AI camera fallback ───────────────────────────────────────────────
    def fwd_look(self, label):
        """Hover and look ahead. Returns (fwd, rgt) metres estimate or None."""
        if self.fwd is None:
            return None
        log.info(f"  AI cam look ({label})…")
        self.fwd.set_active(True)
        hold = self.here()
        t0, est = time.time(), None
        while time.time() - t0 < FWD_LOOK_S + 1.0:
            self.d.goto_ned(hold[0], hold[1], SEARCH_ALT_M, self.yaw)
            c = self.fwd.get()[2]
            if c is not None:
                b = fwd_pixel_to_body(c["center"][0], c["center"][1],
                                      self.d.alt() or SEARCH_ALT_M)
                if b is not None and math.hypot(*b) <= FWD_MAX_RANGE_M:
                    est = b
                    break
            self.tick(f"AI cam look ({label})")
        self.fwd.set_active(False)
        if est:
            log.info(f"  ★ AI cam: banner ~{est[0]:.0f} m ahead, {est[1]:+.1f} m right")
        return est

    def fwd_steps(self, est):
        """Step towards the AI-cam direction, re-looking after each step, with
        the DOWN cam watching all the way. True when confirmed."""
        for s in range(FWD_MAX_STEPS):
            dist = math.hypot(*est)
            step = min(FWD_STEP_M, max(1.0, dist - 1.0))
            f, r = est[0] / dist * step, est[1] / dist * step
            dn, de = body_to_ned(f, r, self.cur_yaw())
            p = self.here()
            res = self.fly_watch(p[0] + dn, p[1] + de, f"AI-cam step {s + 1}")
            if res == "glimpse" and self.chase():
                return True
            nxt = self.fwd_look(f"step {s + 1}")
            if nxt is None:
                break                    # too close for the AI cam -> sweep here
            est = nxt
        return self.sweep(self.here(), FWD_EXTRA_TRIES)

    # ── align ────────────────────────────────────────────────────────────
    def align(self):
        log.info("\n── ALIGN (position lock) ──")
        self.d.set_speed(ALIGN_SPEED_MS)
        # keep the lock from the chase — it already points at the banner
        t0 = t_seen = time.time()
        t_good = None
        while time.time() - t0 < ALIGN_TIMEOUT_S:
            conf, now = self.down_conf(), time.time()
            g = self.down_glimpse()
            seen = conf or g
            if seen is not None:
                t_seen = now
                if not self.check_axes(g):
                    meas = self.down_banner_ne(seen)
                    if meas is not None and self.d.speed() < LOCK_MAX_SPEED:
                        # every sighting pushes the lock; confirmed ones count more
                        self.push_lock(meas, LOCK_EMA if conf else LOCK_EMA * 0.5)
                if now - self.t_alog > 1.0:            # 1 Hz diagnostics
                    self.t_alog = now
                    a = self.d.alt() or SEARCH_ALT_M
                    f, r = down_pixel_to_body(seen["center"][0], seen["center"][1], a)
                    p = self.here()
                    log.info(f"    banner px=({seen['center'][0]:.0f},{seen['center'][1]:.0f}) "
                             f"→ fwd={f:+.2f} rgt={r:+.2f} m | drone N={p[0]:.2f} E={p[1]:.2f} "
                             f"| lock N={self.lock_ne[0]:.2f} E={self.lock_ne[1]:.2f}"
                             if self.lock_ne else "")
            lost = now - t_seen
            if lost > LOST_GIVEUP_S:
                log.warning("  Banner lost too long")
                return False
            if self.lock_ne is None:
                p = self.here()
                self.d.goto_ned(p[0], p[1], SEARCH_ALT_M, self.yaw)
                self.tick("ALIGN  waiting for lock")
                continue
            self.d.goto_ned(self.lock_ne[0], self.lock_ne[1], SEARCH_ALT_M, self.yaw)
            p = self.here()
            err = math.hypot(self.lock_ne[0] - p[0], self.lock_ne[1] - p[1])
            yaw_err = abs((math.degrees(self.cur_yaw() - self.yaw) + 180) % 360 - 180)
            cam_err = None
            if conf is not None:
                f, r = down_pixel_to_body(conf["center"][0], conf["center"][1],
                                          self.d.alt() or SEARCH_ALT_M)
                cam_err = math.hypot(f, r)
            good = (err < ALIGN_TOL_M and cam_err is not None
                    and cam_err < ALIGN_TOL_M * 1.5 and yaw_err < 8)
            if good:
                t_good = t_good or now
                if now - t_good >= ALIGN_HOLD_S:
                    log.info(f"  ✓ CENTRED, facing corridor "
                             f"(pos {err:.2f} m, cam {cam_err:.2f} m)")
                    return True
            else:
                t_good = None
            ce = f"{cam_err:.2f}" if cam_err is not None else "--"
            self.tick(f"ALIGN pos={err:.2f}m cam={ce}m yaw={yaw_err:.0f}° lost={lost:.1f}s")
        log.warning("  Align timeout")
        return self.lock_ne is not None

    # ── finish ───────────────────────────────────────────────────────────
    def descend_on_lock(self, final_alt):
        tgt = self.d.alt() or SEARCH_ALT_M
        t_prev = time.time()
        while True:
            now = time.time()
            dt, t_prev = now - t_prev, now
            conf = self.down_conf()
            if conf is not None and self.d.speed() < LOCK_MAX_SPEED:
                meas = self.down_banner_ne(conf)
                if meas is not None:
                    a = LOCK_EMA * 0.6
                    self.lock_ne = (self.lock_ne[0] * (1 - a) + meas[0] * a,
                                    self.lock_ne[1] * (1 - a) + meas[1] * a)
            p = self.here()
            err = math.hypot(self.lock_ne[0] - p[0], self.lock_ne[1] - p[1])
            if err < 0.35:
                tgt = max(final_alt, tgt - DESCENT_MS * dt)
            self.d.goto_ned(self.lock_ne[0], self.lock_ne[1], tgt, self.yaw)
            a_now = self.d.alt() or tgt
            self.tick(f"DESCEND alt={a_now:.2f}m tgt={tgt:.2f}m err={err:.2f}m "
                      f"{'cam' if conf else 'lock'}")
            if a_now <= final_alt + 0.15:
                return

    def finish(self):
        if FINAL_ACTION == "land":
            log.info("\n── DESCEND + LAND on banner ──")
            self.descend_on_lock(LAND_SWITCH_M)
            self.releasing = True
            self.d.set_mode("LAND")
            t0 = time.time()
            while self.d.armed() and time.time() - t0 < 60:
                self.tick(f"LANDING alt={self.d.alt() or 0:.2f}m")
            log.info("  ✓ Landed" if not self.d.armed() else "  Landing timeout")
        else:
            log.info(f"\n── HOVER {HOVER_ALT_M} m over banner, facing corridor ──")
            self.descend_on_lock(HOVER_ALT_M)
            t0 = time.time()
            while time.time() - t0 < HOVER_TIME_S:
                self.d.goto_ned(self.lock_ne[0], self.lock_ne[1], HOVER_ALT_M, self.yaw)
                self.tick(f"HOVER {time.time() - t0:.0f}/{HOVER_TIME_S}s")
            self.releasing = True
            self.d.set_mode("RTL")

    # ── flow ─────────────────────────────────────────────────────────────
    def find_banner(self):
        """True once the DOWN cam has confirmed a banner (lock_ne is set)."""
        start = self.here()
        g = self.down_conf() or self.down_glimpse()
        if g is not None:
            ne = self.down_banner_ne(g)
            if ne is not None:
                self.glimpse_ne = ne
                if self.chase():
                    return True
        est = self.fwd_look("start")                 # far banner? head that way
        if est is not None and self.fwd_steps(est):
            return True
        if self.sweep(start, MAX_TRIES):
            return True
        est = self.fwd_look("last chance")           # before giving up
        if est is not None and self.fwd_steps(est):
            return True
        return False

    def run(self):
        if CORRIDOR_HEADING_DEG is not None:
            self.yaw = math.radians(CORRIDOR_HEADING_DEG)
        elif self.d.yaw() is not None:
            self.yaw = self.d.yaw()
        log.info(f"  Corridor heading = {math.degrees(self.yaw) % 360:.1f}°")

        if not self.d.arm_and_takeoff(SEARCH_ALT_M):
            self.d.set_mode("RTL")
            return
        self.d.set_speed(SEARCH_SPEED_MS)
        hold = self.here()
        t0 = time.time()
        while time.time() - t0 < 2.0:
            self.d.goto_ned(hold[0], hold[1], SEARCH_ALT_M, self.yaw)
            self.tick("SETTLE + face corridor")

        for _ in range(2):
            if not self.find_banner():
                log.warning("\n✗ Banner not found → RTL")
                self.d.set_mode("RTL")
                return
            if self.align():
                self.finish()
                log.info("\n✓ DONE")
                return
            self.down.reset()
            self.lock_ne = None
        log.warning("  Could not hold alignment → RTL")
        self.d.set_mode("RTL")

# ═════════════════════════════════════════════════════════════════════════════
# BENCH MODE
# ═════════════════════════════════════════════════════════════════════════════

def bench(down, fwd, pv, alt=SEARCH_ALT_M):
    log.info(f"BENCH — no flight. Geometry assumes alt {alt} m. Ctrl+C to stop.")
    if fwd:
        fwd.set_active(True)
    down.alt = alt
    t_log = t_mask = 0
    while True:
        _, raw, dc = down.get()
        parts = []
        if dc:
            f, r = down_pixel_to_body(dc["center"][0], dc["center"][1], alt)
            parts.append(f"DOWN fwd={f:+.2f} rgt={r:+.2f}m  fill={dc['fill']:.2f} "
                         f"sol={dc['solidity']:.2f} tex={dc['texture']:.0f}")
        else:
            parts.append("DOWN " + ("seen (confirming)" if raw else "--"))
        if fwd:
            fc = fwd.get()[2]
            b = fwd_pixel_to_body(fc["center"][0], fc["center"][1], alt) if fc else None
            parts.append(f"AI fwd={b[0]:.1f} rgt={b[1]:+.1f}m" if b
                         else ("AI above horizon" if fc else "AI --"))
        status = " | ".join(parts)
        pv.update(down, fwd, status)
        now = time.time()
        if now - t_log > (0.5 if dc else 1.5):
            log.info(status + f"   fps {down.fps:.0f}")
            if not dc:                                 # explain WHY nothing is found
                fr = down.get()[0]
                if fr is not None:
                    why = []
                    detect_raw(fr, DOWN_MIN_AREA, None, why)
                    for line in why:
                        log.info("    " + line)
            t_log = now
        if now - t_mask > 1.0:
            m = down.get_mask()
            if m is not None:
                cv2.imwrite(os.path.join(DEBUG_DIR, "mask_down.jpg"), m)
            t_mask = now
        time.sleep(0.05)

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    global Picamera2, mavutil
    from picamera2 import Picamera2 as _P
    Picamera2 = _P

    global TEX_PIX_MAX, MAX_TEXTURE, DOWN_FLIP_X, DOWN_FLIP_Y, FINAL_ACTION, SEARCH_ALT_M
    bench_mode = "--bench" in sys.argv
    if "--hover" in sys.argv:
        FINAL_ACTION = "hover"       # hover HOVER_ALT_M over the banner, then RTL
    if "--alt" in sys.argv:
        SEARCH_ALT_M = float(sys.argv[sys.argv.index("--alt") + 1])
    if "--flip-y" in sys.argv:
        DOWN_FLIP_Y = not DOWN_FLIP_Y
    if "--flip-x" in sys.argv:
        DOWN_FLIP_X = not DOWN_FLIP_X
    log.info(f"  camera directions: DOWN_FLIP_X={DOWN_FLIP_X} DOWN_FLIP_Y={DOWN_FLIP_Y} "
             "(auto-corrected in flight if wrong)")
    use_fwd = "--ai-cam" in sys.argv          # ArduCam (down) only by default
    # field tuning without editing the file:
    #   --tex 20   allow more texture (creased / shiny banner)
    #   --no-tex   switch the texture filter off completely (diagnosis)
    if "--no-tex" in sys.argv:
        TEX_PIX_MAX, MAX_TEXTURE = 255, 255.0
    elif "--tex" in sys.argv:
        v = float(sys.argv[sys.argv.index("--tex") + 1])
        TEX_PIX_MAX, MAX_TEXTURE = int(v), v * 0.75
    log.info(f"  texture limits: pixel {TEX_PIX_MAX}, blob {MAX_TEXTURE}")
    log.info("=" * 64)
    log.info(f"SQUADRONE — banner search v2  ({'BENCH' if bench_mode else 'FLIGHT'}, "
             f"final={FINAL_ACTION}, AI cam={'fallback' if use_fwd else 'off'})")
    log.info("=" * 64)

    down = fwd = drone = None
    pv = Preview()
    try:
        down, fwd = open_cameras(use_fwd)
        if bench_mode:
            bench(down, fwd, pv)
            return
        from pymavlink import mavutil as _m
        mavutil = _m
        drone = Drone()
        drone.connect()
        gps = drone.get("GPS_RAW_INT")
        if gps is not None and gps.fix_type < 3:
            log.warning(f"  GPS fix_type={gps.fix_type} — position lock needs 3D fix")
        log.info(f"  Heading now {math.degrees(drone.yaw() or 0) % 360:.1f}° — "
                 "nose must point DOWN THE CORRIDOR")
        input("\n  >>> ENTER to arm + take off   (Ctrl+C = abort / RTL) <<<\n")
        Mission(drone, down, fwd, pv).run()
    except PilotTookOver as e:
        log.warning(f"\n!!! Pilot took over (mode {e}) — script stops, you have control")
    except KeyboardInterrupt:
        log.warning("\n!!! Ctrl+C !!!")
        if drone and drone.mav and drone.armed():
            drone.set_mode("RTL")
    except Exception as e:
        log.error(f"!!! EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        if drone and drone.mav and drone.armed() and drone.mode() == "GUIDED":
            drone.set_mode("RTL")
    finally:
        for c in (down, fwd):
            if c:
                c.stop()
        if drone:
            drone.close()
        pv.close()


if __name__ == "__main__":
    main()