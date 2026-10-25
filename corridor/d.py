#!/usr/bin/env python3
"""
SQUADRONE — CORRIDOR (entry = green, exit = blue)          File: corridor.py
Team: Squadrone AT2026RC60, PVGCOET Pune, SAEINDIA AeroTHON 2026
Built on banner_search_dual.py (search + step-and-settle centring, field-tested).

FIELD (PDF 25-10): walls 8 ft (2.44 m) high, 22 ft (6.7 m) long, 11.5 ft (3.5 m)
apart. Ground banner 11.5 x 1 ft (3.5 x 0.30 m) ACROSS the corridor mouth,
in front of the entry arch (arch banner above it). Green = entry, blue = exit.

SEQUENCE
  1. Take off to 5 m, nose roughly towards the corridor.
  2. Find the ground banner (forward/left/right sweep — --tries/--step/--side).
  3. Centre on it (step-and-settle, scale self-calibrating).
  4. CORRIDOR HEADING FROM THE BANNER: it lies across the corridor, so the
     corridor runs at 90° to its long axis. Turn to that heading, re-centre.
  5. Back off BACKOFF_M (so we don't descend into the arch), descend to
     CORRIDOR_ALT_M (1.3 m, walls are 2.44 m).
  6. Fly through at CORRIDOR_SPEED: LEFT/RIGHT TFmini-S keep the drone centred,
     FRONT TFmini-S is a safety stop. Done when past the far end + margin.
     Any LiDAR dead / prop closer than SIDE_ABORT_M to a wall -> climb out
     to ABORT_ALT_M and RTL.
  7. After the corridor (--after): land (default for tests) / hover / rtl /
     climb (FSM: continue at 10 m).

RUN
  python3 corridor.py --lidar-scan          # find + name the 3 TFminis (once)
  python3 corridor.py --lidar-bench         # live distances, centring error
  python3 corridor.py --bench               # camera: banner + corridor heading
  python3 corridor.py                       # green corridor, land after
  python3 corridor.py --blue                # blue banner (no obstacle logic yet)
  python3 corridor.py --tries 6 --step 3 --side 2   # search pattern
"""

import os
import sys
import json
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
# Re-lock: flight 24-10 hovered 50 s on a stale lock while the camera saw the
# banner steadily 2.2 m away (every sighting thrown out by LOCK_GATE_M). If the
# sightings OUTSIDE the gate agree with each other for RELOCK_S, and none fall
# inside the gate, the lock was wrong -> move it to where the banner really is.
RELOCK_S         = 2.0
RELOCK_MIN_N     = 5
RELOCK_SPREAD_M  = 0.8

# ── Centring + landing: STEP-AND-SETTLE visual servo ─────────────────────────
# Stop -> measure the banner offset over several frames -> move PART of the
# way (gain) -> stop -> measure again. After every move the script compares
# how far the drone REALLY moved (telemetry) with how far the banner moved in
# the camera, and corrects its metres-per-pixel scale (per axis) and axis
# directions. A 2x scale error (flight 24-10) is fixed after 1-2 steps.
SERVO_GAIN        = 0.7     # move this fraction of the measured offset
SERVO_MAX_STEP_M  = 2.0
PROBE_GAIN        = 0.4     # until an axis is calibrated in flight: small careful
PROBE_MAX_M       = 1.0     # moves, so a wrong scale/direction can't throw the
                            # banner out of view (it is measured after each move)
SERVO_MAX_STEPS   = 20
SETTLE_SPEED_MS   = 0.15    # "stopped" = slower than this
SETTLE_TIMEOUT_S  = 8.0
MEASURE_S         = 0.8     # look this long while stopped
MEASURE_MIN_N     = 3       # need this many detections in that time
CENTRE_TOL_5M     = 0.30    # centred within this at search altitude
CENTRE_TOL_LOW    = 0.20    # ... and at the 3 m stage
CENTRED_NEEDED    = 2       # consecutive centred measurements
SCALE_MIN, SCALE_MAX = 0.6, 1.6   # FOV is now measured at start-up -> scale ~1
RATIO_OK = (0.45, 2.2)            # a move/shift ratio outside this = bad measurement
FLIP_VOTES = 2                    # reversed this many times IN A ROW before flipping
LAND_STAGE_M      = 3.0     # centre again here before the final descent
FILL_FRAME_FRAC   = 0.55    # banner covers this much of frame = surely over it

# Final action
FINAL_ACTION   = "land"    # "land" (test) or "hover" (mission)
LAND_SWITCH_M  = 1.2       # GUIDED descent to here (centred), then LAND
DESCENT_MS     = 0.35
HOVER_ALT_M    = 3.0
HOVER_TIME_S   = 10

# ── Corridor (PDF 25-10) ─────────────────────────────────────────────────────
CORRIDOR_LEN_M    = 6.7      # 22 ft  (older slide said 10 m — --corridor-len 10)
CORRIDOR_W_M      = 3.5      # 11.5 ft between the walls
WALL_H_M          = 2.44     # 8 ft
CORRIDOR_ALT_M    = 1.3      # fly height inside (walls 2.44 m)
CORRIDOR_SPEED    = 0.5      # m/s inside
BACKOFF_M         = 2.5      # descend this far BEFORE the banner (arch is above it)
EXIT_MARGIN_M     = 1.5      # keep going this far past the far end
LOOKAHEAD_M       = 1.0      # position carrot ahead of the drone
CENTRE_GAIN       = 0.35     # EMA weight of each LiDAR centre estimate
MAX_YAW_FIX_DEG   = 60       # banner says corridor is further than this from the nose -> don't enter
ABORT_ALT_M       = 4.0      # climb out of the corridor to this, then RTL
AFTER_CORRIDOR    = "land"   # land | hover | rtl | climb

# ── LiDARs: 3x TFmini-S on the Pi's I2C bus ──────────────────────────────────
I2C_BUS           = 1
LIDAR_FILE        = os.path.expanduser("~/squadrone/lidars.json")   # --lidar-scan
LIDAR_ADDR        = {"front": 0x14, "left": 0x15, "right": 0x11}     # team's wiring (25-10)
LIDAR_BIAS_M      = -0.04    # subtracted -> ADDS 4 cm, same as the team's test script
FRONT_TIP_M       = 0.15     # front prop tip is this far AHEAD of the front lidar
SIDE_TIP_M        = 0.18     # side prop tips are this far OUTSIDE the side lidars
LIDAR_MIN_STRENGTH = 100     # TFmini-S: below = unreliable
LIDAR_MAX_M       = 12.0
WALL_SEEN_MAX_M   = 3.2      # side reading above this = no wall there
FRONT_STOP_M      = 0.70     # front prop-to-obstacle clearance -> stop
SIDE_PUSH_M       = 0.35     # side prop-to-wall clearance -> stop + push to centre
SIDE_ABORT_M      = 0.15     # -> climb out + RTL
LIDAR_STALE_S     = 0.5

# Cameras
CAM_W, CAM_H = 640, 480
AWB_DAYLIGHT = True        # fixed daylight white balance (see detection notes)

DOWN_MODEL        = "imx519"   # ArduCam, rear, straight down
DOWN_HFOV_DEG     = 66.0       # FULL-sensor FOV. The actual FOV is recomputed at
DOWN_VFOV_DEG     = 51.0       # camera start from the sensor crop (see full_fov_config)
FULL_HFOV_DEG, FULL_VFOV_DEG = DOWN_HFOV_DEG, DOWN_VFOV_DEG
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
LAB_CH      = 1            # Lab channel used as 2nd colour gate (1 = a*, 2 = b*)
BANNER_COLOR = "green"
COLORS = {   # name: (HSV low, HSV high, Lab channel, Lab max)
    "green": (np.array([35, 80, 50]), np.array([85, 255, 255]), 1, 120),   # #00B140
    "blue":  (np.array([100, 120, 40]), np.array([130, 255, 255]), 2, 110), # #0000FF (b* low = blue)
}


def set_banner_color(name):
    global LOWER_GREEN, UPPER_GREEN, LAB_CH, LAB_A_MAX, BANNER_COLOR
    LOWER_GREEN, UPPER_GREEN, LAB_CH, LAB_A_MAX = COLORS[name]
    BANNER_COLOR = name
# Shape / texture (all rotation-invariant)
MIN_FILL      = 0.60       # contour area / min-area rotated rectangle
MIN_SOLIDITY  = 0.75       # contour area / convex hull
MAX_ELONG     = 25.0       # corridor ground banner 3.5 x 0.30 m = 11.7:1 (edges/angle make it more)
TEX_PIX_MAX   = 18         # per-pixel texture gate (grass ~28, creased banner 11-14)
MAX_TEXTURE   = 18.0       # mean texture inside a blob (same scale)
MAX_AREA_FRAC = 1.01       # no limit: up close the banner may fill the whole frame
AMBIGUOUS     = 1.01       # off: ground + arch banner can both be visible (same strip)
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
              logging.FileHandler(os.path.join(DEBUG_DIR, "corridor.log"))])
log = logging.getLogger("dual")

_k5 = np.ones((5, 5), np.uint8)
HEAL_PX = 17                                   # closes cuts up to ~15 px wide
_k_heal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (HEAL_PX, HEAL_PX))
_k_regrow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
_k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

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
    a = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, LAB_CH]
    tex = texture_map(bgr)
    col = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    col = cv2.bitwise_and(col, cv2.inRange(a, 0, LAB_A_MAX))
    core = cv2.bitwise_and(col, cv2.inRange(tex, 0, TEX_PIX_MAX))
    # The texture map is high for ~8 px along every sharp edge, which ate half
    # of a 0.3 m strip (25 px -> 14 px at 5 m). Grow the smooth core back out
    # over pixels that have the right colour: edges return, grass doesn't.
    m = cv2.bitwise_and(col, cv2.dilate(core, _k_regrow))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _k5)
    # heal thin cuts (creases, folds, streaks, shadow edges) so the banner is
    # ONE blob again; grass stays out because it was removed pixel-by-pixel
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _k_heal)
    return m, tex


def merge_strip_pieces(cnts, min_area):
    """A long banner often comes out as 2+ pieces (a join between two sheets,
    a crease, glare across it). Picking one piece moves the 'centre' along the
    strip by up to half its length (flight 25-10: 'banner shifted 2.4 m'
    while the drone moved 0.6 m). Join pieces that are parallel, on the same
    line and close end-to-end into one contour."""
    info = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area / 4:
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        if min(w, h) < 2:
            continue
        L, T = max(w, h), min(w, h)
        bp = cv2.boxPoints(((cx, cy), (w, h), ang))
        e1, e2 = bp[1] - bp[0], bp[2] - bp[1]
        ax = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
        ax = ax / max(np.hypot(*ax), 1e-6)
        info.append([c, np.array([cx, cy]), ax, L, T])
    if len(info) < 2:
        return cnts, 0
    parent = list(range(len(info)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(info)):
        for j in range(i + 1, len(info)):
            ci, cj = info[i], info[j]
            if ci[3] / ci[4] < cj[3] / cj[4]:
                ci, cj = cj, ci                            # ci = the more strip-like one
            if ci[3] / ci[4] < 2.5:
                continue                                   # neither is a strip
            if cj[3] / cj[4] >= 2.5 and abs(float(np.dot(ci[2], cj[2]))) < math.cos(math.radians(12)):
                continue                                   # two strips, not parallel
            # the shorter piece must be as thick as the strip (seen across it)
            thick_j = min(cj[3], cj[4])
            if not (0.5 * ci[4] <= thick_j <= 1.8 * ci[4]):
                continue
            d = cj[1] - ci[1]
            perp = abs(d[0] * ci[2][1] - d[1] * ci[2][0])  # off the line
            along = abs(float(np.dot(d, ci[2])))
            ext_j = cj[3] if cj[3] / cj[4] >= 2.5 else max(cj[3], cj[4])
            gap = along - (ci[3] + ext_j) / 2              # end-to-end gap
            T = max(ci[4], cj[4])
            if perp < 1.0 * T and gap < 3.0 * T:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(info)):
        groups.setdefault(find(i), []).append(i)
    merged_ids, out, n = set(), [], 0
    for g in groups.values():
        if len(g) > 1:
            pts = np.vstack([info[k][0] for k in g])
            out.append(cv2.convexHull(pts))
            merged_ids.update(id(info[k][0]) for k in g)
            n += len(g)
    out += [c for c in cnts if id(c) not in merged_ids]
    return out, n


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
    cnts, n_merged = merge_strip_pieces(list(cnts), min_area)
    if why is not None and n_merged:
        why.append(f"  joined {n_merged} pieces of a strip into one")
    max_area = MAX_AREA_FRAC * CAM_W * CAM_H
    if why is not None:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        a = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, LAB_CH]
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
        # texture INSIDE the blob, away from its edges (edges are always
        # "textured" against the white border / ground)
        inner = cv2.erode(blob, _k_inner)
        if cv2.countNonZero(inner) < 0.15 * area:
            inner = blob
        # MEDIAN, not mean: glare spots / creases on a glossy banner are local
        # and must not outvote the smooth rest of it
        tv = tex[y:y + h, x:x + w][inner > 0]
        texture = float(np.median(tv)) if tv.size else 255.0
        if texture > MAX_TEXTURE:
            reject(c, f"too textured: {texture:.1f} > {MAX_TEXTURE}")
            continue
        if why is not None:
            why.append(f"  blob area={area:.0f}px OK  fill={fill:.2f} sol={solidity:.2f} "
                       f"tex={texture:.1f}")
        m = cv2.moments(c)
        score = (fill * solidity * (1.0 - 0.5 * texture / MAX_TEXTURE)
                 * min(1.0, math.sqrt(area / (min_area * 6))))
        bp = cv2.boxPoints(((rcx, rcy), (rw, rh), ang))
        e1, e2 = bp[1] - bp[0], bp[2] - bp[1]
        ax = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
        ax = ax / max(np.hypot(*ax), 1e-6)
        cands.append({"center": (m["m10"] / m["m00"], m["m01"] / m["m00"]),
                      "axis": (float(ax[0]), float(ax[1])), "elong": elong,
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


def full_fov_mode(cam):
    """Smallest sensor mode that still covers the WHOLE sensor (no crop)."""
    try:
        modes = cam.sensor_modes
        wmax = max(m["crop_limits"][2] for m in modes)
        full = [m for m in modes if m["crop_limits"][2] >= 0.98 * wmax]
        best = min(full, key=lambda m: m["size"][0] * m["size"][1])
        log.info(f"  sensor modes: " + ", ".join(
            f"{m['size'][0]}x{m['size'][1]}{'(crop)' if m not in full else ''}" for m in modes)
            + f"  -> using {best['size'][0]}x{best['size'][1]} (full field of view)")
        return best
    except Exception as e:
        log.warning(f"  could not list sensor modes ({e})")
        return None


def measure_fov(cam):
    """Set DOWN_HFOV/VFOV from the crop libcamera is ACTUALLY using."""
    global DOWN_HFOV_DEG, DOWN_VFOV_DEG
    try:
        full = getattr(cam, "sensor_resolution", None)
        props = getattr(cam, "camera_properties", {}) or {}
        if not full:
            full = props.get("PixelArraySize")
        if not full and props.get("PixelArrayActiveAreas"):
            full = props["PixelArrayActiveAreas"][0][2:]
        if not full:
            sc = (getattr(cam, "camera_controls", {}) or {}).get("ScalerCrop")
            full = sc[1][2:] if sc else None
        if not full:
            raise RuntimeError("sensor size unknown")
        full_w, full_h = full
        crop = cam.capture_metadata().get("ScalerCrop")
        if not crop:
            log.warning("  no ScalerCrop in metadata — assuming full FOV")
            return
        fx, fy = crop[2] / full_w, crop[3] / full_h
        DOWN_HFOV_DEG = 2 * math.degrees(math.atan(fx * math.tan(math.radians(FULL_HFOV_DEG / 2))))
        DOWN_VFOV_DEG = 2 * math.degrees(math.atan(fy * math.tan(math.radians(FULL_VFOV_DEG / 2))))
        msg = (f"  DOWN cam crop {crop[2]}x{crop[3]} of {full_w}x{full_h} "
               f"({fx * 100:.0f}% x {fy * 100:.0f}%) -> FOV {DOWN_HFOV_DEG:.1f}° x "
               f"{DOWN_VFOV_DEG:.1f}° (at 5 m the frame covers "
               f"{2 * 5 * math.tan(math.radians(DOWN_HFOV_DEG / 2)):.1f} x "
               f"{2 * 5 * math.tan(math.radians(DOWN_VFOV_DEG / 2)):.1f} m)")
        (log.info if fx > 0.95 and fy > 0.95 else log.warning)(
            msg + ("" if fx > 0.95 and fy > 0.95 else "  ⚠ CROPPED view — maths adjusted"))
    except Exception as e:
        log.warning(f"  FOV measurement failed ({e}) — using {DOWN_HFOV_DEG}° x {DOWN_VFOV_DEG}°")


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
        # Flight 24-10: for a 640x480 output libcamera picked the 1280x720 sensor
        # mode = a CROPPED (zoomed-in) view. The maths assumed the full 66° FOV,
        # so every distance came out ~2x too big and the drone swung back and
        # forth across the banner. Force the full-field mode.
        mode = full_fov_mode(self.cam)
        if mode is not None:
            # keep the raw 'format' Picamera2 filled in, only change the size
            # (flight 24-10 #2: replacing the whole dict -> KeyError 'format')
            cfg.setdefault("raw", {})
            cfg["raw"]["size"] = tuple(mode["size"])
        try:
            self.cam.configure(cfg)
        except Exception as e:
            log.warning(f"  {self.name}: full-FOV sensor mode refused ({e}) — default mode")
            cfg = self.cam.create_preview_configuration(
                main={"size": (CAM_W, CAM_H), "format": "RGB888"}, buffer_count=4)
            self.cam.configure(cfg)
        self.cam.start()
        time.sleep(1.5)
        if self.name == "DOWN":
            measure_fov(self.cam)
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
# LIDARS — 3x Benewake TFmini-S on I2C
# ═════════════════════════════════════════════════════════════════════════════
_TF_CMD = [0x5A, 0x05, 0x00, 0x01, 0x60]      # "get frame, distance in cm"


def tf_read(bus, addr):
    """One TFmini-S frame -> (distance_m, strength) or None."""
    from smbus2 import i2c_msg
    try:
        bus.i2c_rdwr(i2c_msg.write(addr, _TF_CMD))
        time.sleep(0.001)
        r = i2c_msg.read(addr, 9)
        bus.i2c_rdwr(r)
        d = list(r)
    except OSError:
        return None
    if d[0] != 0x59 or d[1] != 0x59 or (sum(d[:8]) & 0xFF) != d[8]:
        return None
    return (d[2] | d[3] << 8) / 100.0, d[4] | d[5] << 8


class Lidars(threading.Thread):
    """Reads front/left/right ~50 Hz each. get(name) = filtered metres (bias
    removed) or None when stale/unreliable/out of range."""

    def __init__(self):
        super().__init__(daemon=True)
        self.addr = dict(LIDAR_ADDR)
        if os.path.isfile(LIDAR_FILE):
            with open(LIDAR_FILE) as f:
                self.addr = {k: int(v, 16) if isinstance(v, str) else int(v)
                             for k, v in json.load(f).items()}
        self.hist = {k: deque(maxlen=5) for k in self.addr}
        self.n_ok = {k: 0 for k in self.addr}
        self.n_all = {k: 0 for k in self.addr}
        self.last_raw = {k: None for k in self.addr}
        self.lk = threading.Lock()
        self._stop_evt = threading.Event()
        from smbus2 import SMBus
        self.bus = SMBus(I2C_BUS)
        log.info("  LiDARs: " + ", ".join(f"{k}=0x{v:02X}" for k, v in self.addr.items())
                 + ("" if os.path.isfile(LIDAR_FILE) else "  (built-in addresses)"))

    def run(self):
        while not self._stop_evt.is_set():
            for k, a in self.addr.items():
                r = tf_read(self.bus, a)
                now = time.time()
                with self.lk:
                    self.n_all[k] += 1
                    self.last_raw[k] = r
                    if r and r[1] >= LIDAR_MIN_STRENGTH and r[1] != 65535 and 0.05 < r[0] <= LIDAR_MAX_M:
                        self.hist[k].append((now, r[0] - LIDAR_BIAS_M))
                        self.n_ok[k] += 1
            time.sleep(0.015)

    def get(self, k):
        with self.lk:
            h = [d for t, d in self.hist[k] if time.time() - t < LIDAR_STALE_S]
        return float(np.median(h)) if h else None

    def alive(self, k):
        """True if the sensor answered recently (even 'out of range' counts)."""
        with self.lk:
            return self.last_raw[k] is not None

    def stats(self):
        with self.lk:
            out = {k: (self.n_ok[k], self.n_all[k]) for k in self.addr}
            for k in self.addr:
                self.n_ok[k] = self.n_all[k] = 0
        return out

    def stop(self):
        self._stop_evt.set()


def lidar_scan():
    """Find every TFmini-S on the bus, then name them by covering each one."""
    from smbus2 import SMBus
    bus = SMBus(I2C_BUS)
    found = []
    for a in range(0x08, 0x78):
        if tf_read(bus, a) is not None:
            found.append(a)
    log.info(f"TFmini-S found at: {[f'0x{a:02X}' for a in found] or 'NONE'}")
    if len(found) < 3:
        log.error("Need 3. Check wiring (SDA/SCL/5V/GND), 'sudo raspi-config' -> I2C on, "
                  "and that each TFmini has its OWN address (all ship as 0x10).")
        if not found:
            return
    names = {}
    for name in ("front", "left", "right"):
        input(f"\n>>> Hold your hand ~15 cm in front of the {name.upper()} lidar, press ENTER")
        best = None
        for a in found:
            vals = [tf_read(bus, a) for _ in range(5)]
            vals = [v[0] for v in vals if v]
            d = float(np.median(vals)) if vals else 99
            log.info(f"    0x{a:02X}: {d:.2f} m")
            if a not in names.values() and (best is None or d < best[1]):
                best = (a, d)
        names[name] = best[0]
        log.info(f"  {name} = 0x{best[0]:02X} ({best[1]:.2f} m)")
    with open(LIDAR_FILE, "w") as f:
        json.dump({k: f"0x{v:02X}" for k, v in names.items()}, f, indent=2)
    log.info(f"Saved {LIDAR_FILE}")


def lidar_bench():
    li = Lidars()
    li.start()
    log.info("LIDAR BENCH — Ctrl+C to stop. Check each against a tape measure "
             f"({-LIDAR_BIAS_M * 100:+.0f} cm already added to every reading).")
    t_st = time.time()
    try:
        while True:
            f, l, r = li.get("front"), li.get("left"), li.get("right")
            fmt = lambda v: f"{v:5.2f}" if v is not None else "  -- "
            raw = {k: li.last_raw[k] for k in li.addr}
            line = (f"F {fmt(f)} (prop {fmt(f - FRONT_TIP_M if f else None)})  "
                    f"L {fmt(l)} (prop {fmt(l - SIDE_TIP_M if l else None)})  "
                    f"R {fmt(r)} (prop {fmt(r - SIDE_TIP_M if r else None)})")
            if l is not None and r is not None:
                line += f"  | centre err {(r - l) / 2:+.2f} m ({'move RIGHT' if r > l else 'move LEFT'})"
            line += "  | strength " + " ".join(
                f"{k[0].upper()}{raw[k][1] if raw[k] else '?'}" for k in raw)
            log.info(line)
            if time.time() - t_st > 5:
                st = li.stats()
                log.info("    rate: " + "  ".join(
                    f"{k} {n_all / (time.time() - t_st):.0f} Hz ({100 * n_ok / max(n_all, 1):.0f}% valid)"
                    for k, (n_ok, n_all) in st.items()))
                t_st = time.time()
            time.sleep(0.3)
    finally:
        li.stop()

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
        self.outside = deque(maxlen=40)   # (t, ne) sightings rejected by the gate
        self.t_inside = 0.0               # last sighting accepted by the gate
        self.n_relock = 0
        self.scale = [1.0, 1.0]           # [fwd, rgt] correction found in flight
        self.calibrated = [False, False]
        self.flip_votes = [0, 0]
        self.lidar = None                  # Lidars() — set by main
        self.corr_log = []                 # (s, lat, L, R, F) for the report
        self.servo_log = []
        self.relock_hist = []

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

    def maybe_relock(self, now):
        """Steady sightings outside the gate + nothing inside it = wrong lock."""
        recent = [m for t, m in self.outside if now - t <= RELOCK_S and t > self.t_inside]
        if len(recent) < RELOCK_MIN_N or now - self.t_inside < RELOCK_S:
            return
        med = (float(np.median([m[0] for m in recent])), float(np.median([m[1] for m in recent])))
        spread = max(math.hypot(m[0] - med[0], m[1] - med[1]) for m in recent)
        if spread > RELOCK_SPREAD_M:
            return                               # scattered = noise, keep the lock
        jump = math.hypot(med[0] - self.lock_ne[0], med[1] - self.lock_ne[1])
        self.n_relock += 1
        # SWING guard: lock A -> B -> back near A = flying across the banner
        # and overshooting (distance scale wrong). The banner is in between.
        prev = self.relock_hist[-1] if self.relock_hist else None
        self.relock_hist.append(self.lock_ne)
        if prev is not None and math.hypot(med[0] - prev[0], med[1] - prev[1]) < 1.0:
            mid = ((med[0] + self.lock_ne[0]) / 2, (med[1] + self.lock_ne[1]) / 2)
            log.warning(f"  ⚠ SWINGING across the banner (re-lock went back to where it "
                        f"was before) — camera distance scale is wrong. Using the MIDPOINT "
                        f"N={mid[0]:.2f},E={mid[1]:.2f}. Check the 'DOWN cam crop' line!")
            med = mid
        log.warning(f"  ⟲ RE-LOCK #{self.n_relock}: {len(recent)} sightings in {RELOCK_S:.0f} s "
                    f"agree (spread {spread:.2f} m) but are {jump:.2f} m from the lock -> "
                    f"lock moved N={self.lock_ne[0]:.2f},E={self.lock_ne[1]:.2f} → "
                    f"N={med[0]:.2f},E={med[1]:.2f}")
        self.lock_ne = med
        self.outside.clear()
        self.t_inside = now

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
        # 1) STOP right here and look properly. The banner is usually still in
        #    view; centring (step-and-settle) then works from here and does not
        #    depend on the pixel->metres scale being right.
        log.info("  ◎ Glimpse — stopping to look")
        self.d.set_speed(ALIGN_SPEED_MS)
        here = self.here()
        self.settle(here, SEARCH_ALT_M, "GLIMPSE")
        m = self.measure(here, SEARCH_ALT_M)
        if m is not None:
            log.info(f"  ★ Confirmed from a standstill ({m['n']} looks)")
            self.lock_ne = here
            return True
        # 2) not in view any more: fly to where the glimpse said it was
        self.lock_ne = self.glimpse_ne
        log.info(f"  ◎ Not in view after stopping → coordinate locked N={self.lock_ne[0]:.2f} "
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
    def align_old(self):
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
                        if self.push_lock(meas, LOCK_EMA if conf else LOCK_EMA * 0.5):
                            self.t_inside = now
                        else:
                            self.outside.append((now, meas))
                            self.maybe_relock(now)
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
        # Never land/hover on an unconfirmed spot (before: timeout -> True ->
        # would have landed 2 m off the banner).
        log.warning(f"  Align timeout after {ALIGN_TIMEOUT_S:.0f} s — NOT centred, will not "
                    f"land here (re-locks this attempt: {self.n_relock})")
        return False


    # ═════ STEP-AND-SETTLE VISUAL SERVO ═══════════════════════════════════
    def raw_vec(self, det, alt):
        """Camera part of the banner offset (fwd, rgt) metres from the nominal
        FOV — NOT scale-corrected, camera mounting offset NOT included."""
        f, r = down_pixel_to_body(det["center"][0], det["center"][1], alt)
        return f - DOWN_OFFSET_FWD_M, r

    def offset_of(self, m):
        """Banner offset from the DRONE CENTRE, with the in-flight scale."""
        return (m["raw"][0] * self.scale[0] + DOWN_OFFSET_FWD_M, m["raw"][1] * self.scale[1])

    def settle(self, tgt, alt, label):
        """Fly to tgt (N,E) at alt and wait until really stopped there."""
        t0, t_ok = time.time(), None
        while time.time() - t0 < SETTLE_TIMEOUT_S:
            self.d.goto_ned(tgt[0], tgt[1], alt, self.yaw)
            p = self.here()
            err = math.hypot(tgt[0] - p[0], tgt[1] - p[1])
            ok = (err < 0.25 and self.d.speed() < SETTLE_SPEED_MS
                  and abs((self.d.alt() or alt) - alt) < 0.3 and self.d.tilt_deg() < 6)
            if ok:
                t_ok = t_ok or time.time()
                if time.time() - t_ok > 0.3:
                    return True
            else:
                t_ok = None
            self.tick(f"{label} settle err={err:.2f}m v={self.d.speed():.2f}")
        return False

    def measure(self, tgt, alt):
        """Hold still, collect detections for MEASURE_S. Returns dict with
        median camera vector + full offset, or None if the banner isn't seen."""
        vecs, fracs, t0, last_t = [], [], time.time(), 0.0
        while time.time() - t0 < MEASURE_S or (len(vecs) < MEASURE_MIN_N
                                               and time.time() - t0 < MEASURE_S * 2.5):
            self.d.goto_ned(tgt[0], tgt[1], alt, self.yaw)
            # CONFIRMED detections only (seen in >= DOWN_NEED of the last
            # DOWN_WINDOW frames): a one-frame green patch can't steer the drone
            det = self.down_conf()
            if det is not None and det.get("t", 0) != last_t and self.d.tilt_deg() < MAX_TILT_DEG:
                last_t = det.get("t", 0)
                a = self.d.alt() or alt
                vecs.append(self.raw_vec(det, a))
                fracs.append(det.get("area", 0) / float(CAM_W * CAM_H))
            self.tick(f"measure {len(vecs)} looks")
        if len(vecs) < MEASURE_MIN_N:
            return None
        return {"raw": (float(np.median([v[0] for v in vecs])),
                        float(np.median([v[1] for v in vecs]))),
                "n": len(vecs), "frac": float(np.median(fracs)), "pos": self.here(),
                "flips": (DOWN_FLIP_X, DOWN_FLIP_Y)}

    def calibrate(self, m0, m1):
        """Drone moved between two measurements: a fixed banner must shift in
        the camera by exactly -move. Fix scale / direction per axis."""
        global DOWN_FLIP_X, DOWN_FLIP_Y
        if m0["flips"] != m1["flips"]:
            return                          # measured with different maths
        mf, mr = ned_to_body(m1["pos"][0] - m0["pos"][0], m1["pos"][1] - m0["pos"][1], self.yaw)
        df = (m1["raw"][0] - m0["raw"][0]) * self.scale[0]
        dr = (m1["raw"][1] - m0["raw"][1]) * self.scale[1]
        for i, (axis, moved, seen) in enumerate((("fwd", mf, df), ("rgt", mr, dr))):
            if abs(moved) < 0.2:
                continue
            ratio = seen / -moved           # 1.0 = perfect; 2.0 = reads 2x too far
            if ratio < -0.3:                # banner moved WITH the drone = axis reversed
                self.flip_votes[i] += 1
                if self.flip_votes[i] < FLIP_VOTES:
                    log.info(f"    {axis}: banner moved WITH the drone (moved {moved:+.2f} m, "
                             f"banner {seen:+.2f} m) — vote {self.flip_votes[i]}/{FLIP_VOTES} "
                             f"to flip, not yet")
                    continue
                self.flip_votes[i] = 0
                if axis == "fwd":
                    DOWN_FLIP_Y = not DOWN_FLIP_Y
                else:
                    DOWN_FLIP_X = not DOWN_FLIP_X
                self.calibrated[i] = False      # re-check after the flip
                log.warning(f"    ⚠ {axis} axis REVERSED (moved {moved:+.2f} m, banner "
                            f"{seen:+.2f} m) — flipped DOWN_FLIP_{'Y' if axis == 'fwd' else 'X'}")
                continue
            self.flip_votes[i] = 0
            if not (RATIO_OK[0] <= ratio <= RATIO_OK[1]):
                log.info(f"    scale {axis}: ratio {ratio:.2f} (moved {moved:+.2f}, banner "
                         f"{seen:+.2f}) looks like a bad measurement — ignored")
                continue
            old = self.scale[i]
            new = min(SCALE_MAX, max(SCALE_MIN, old / ratio))
            self.scale[i] = old * 0.3 + new * 0.7 if self.calibrated[i] else new
            self.calibrated[i] = True
            log.info(f"    scale {axis}: moved {moved:+.2f} m, banner shifted {seen:+.2f} m "
                     f"(ratio {ratio:.2f}) -> scale {old:.2f} → {self.scale[i]:.2f}")

    def servo_centre(self, alt, tol, label):
        """Centre over the banner at `alt`. True when centred CENTRED_NEEDED times
        in a row (or the banner fills the frame at low altitude).
        Axes whose scale/direction is not yet verified in flight are PROBED one
        at a time with a short move; if a probe loses the banner the drone goes
        back and probes that axis the other way (direction was wrong)."""
        tgt = self.here()
        prev, good, lost = None, 0, 0
        probe_sign = [1.0, 1.0]           # -1 = try the opposite of what the camera says
        probe_size = [PROBE_MAX_M * 0.6, PROBE_MAX_M * 0.6]
        last_probe = None                 # axis probed by the last move
        names = ("fwd", "rgt")
        for step in range(1, SERVO_MAX_STEPS + 1):
            self.settle(tgt, alt, f"{label} step {step}")
            m = self.measure(tgt, alt)
            if m is None:
                if prev is not None:
                    # the last move threw the banner out of view -> back
                    tgt = prev["pos"]
                    if last_probe is not None:
                        i = last_probe
                        probe_sign[i] = -probe_sign[i]
                        if probe_sign[i] > 0:          # tried both ways: smaller
                            probe_size[i] = max(0.25, probe_size[i] * 0.5)
                        log.info(f"    {label} step {step}: lost it after the {names[i]} "
                                 f"probe — back, next {names[i]} probe "
                                 f"{'OPPOSITE way' if probe_sign[i] < 0 else 'smaller'}")
                    else:
                        self.calibrated = [False, False]
                        log.info(f"    {label} step {step}: lost it after the move — back, "
                                 f"probing axes again")
                    prev, last_probe = None, None
                    continue
                lost += 1
                log.info(f"    {label} step {step}: banner NOT seen ({lost}/3)")
                if lost >= 3:
                    return False
                continue
            lost = 0
            if prev is not None:
                was = list(self.calibrated)
                self.calibrate(prev, m)
                if m["flips"] != (DOWN_FLIP_X, DOWN_FLIP_Y):   # an axis was flipped
                    probe_sign = [1.0, 1.0]
                    m = self.measure(tgt, alt) or m
                for i in (0, 1):
                    if self.calibrated[i] and not was[i]:
                        probe_sign[i] = 1.0
            m["off"] = self.offset_of(m)
            dist = math.hypot(*m["off"])
            big = m["frac"] >= FILL_FRAME_FRAC and alt <= LAND_STAGE_M + 0.5
            log.info(f"    {label} step {step}: banner fwd={m['off'][0]:+.2f} rgt={m['off'][1]:+.2f} m "
                     f"(dist {dist:.2f}, tol {tol:.2f}, {m['n']} looks, fills "
                     f"{m['frac'] * 100:.0f}% of frame, scale f{self.scale[0]:.2f}/r{self.scale[1]:.2f}"
                     f"{'' if all(self.calibrated) else ', verified: ' + '+'.join(n for n, c in zip(names, self.calibrated) if c) or ', verified: none'})")
            self.servo_log.append((label, step, dist))
            if dist < tol or big:
                good += 1
                if good >= CENTRED_NEEDED:
                    self.lock_ne = tgt
                    log.info(f"  ✓ CENTRED at {alt:.1f} m: off {dist:.2f} m "
                             f"({'banner fills the frame' if big else 'within tolerance'})")
                    return True
                prev, last_probe = m, None
                continue                       # measure again from the same spot
            good = 0
            step_b = [0.0, 0.0]
            for i in (0, 1):
                if self.calibrated[i]:
                    step_b[i] = max(-SERVO_MAX_STEP_M, min(SERVO_MAX_STEP_M, m["off"][i] * SERVO_GAIN))
            # probe ONE unverified axis (the one with the bigger offset)
            unver = [i for i in (0, 1) if not self.calibrated[i] and abs(m["off"][i]) > tol * 0.5]
            last_probe = None
            if unver:
                i = max(unver, key=lambda k: abs(m["off"][k]))
                size = min(probe_size[i], abs(m["off"][i]) * PROBE_GAIN + 0.2)
                step_b[i] = math.copysign(size, m["off"][i]) * probe_sign[i]
                last_probe = i
            elif not any(self.calibrated):      # offsets tiny but nothing verified
                for i in (0, 1):
                    step_b[i] = m["off"][i] * PROBE_GAIN
            dn, de = body_to_ned(step_b[0], step_b[1], self.yaw)
            tgt = (m["pos"][0] + dn, m["pos"][1] + de)
            prev = m
        log.warning(f"  {label}: not centred after {SERVO_MAX_STEPS} steps")
        return False

    def descend_to(self, alt, label):
        tgt = self.lock_ne or self.here()
        t0 = time.time()
        while abs((self.d.alt() or alt) - alt) > 0.2 and time.time() - t0 < 25:
            self.d.goto_ned(tgt[0], tgt[1], alt, self.yaw)
            self.tick(f"{label} descend {self.d.alt() or 0:.1f} → {alt:.1f} m")

    def align(self):
        log.info(f"\n── ALIGN at {SEARCH_ALT_M:.0f} m (step-and-settle) ──")
        self.d.set_speed(ALIGN_SPEED_MS)
        return self.servo_centre(SEARCH_ALT_M, CENTRE_TOL_5M, "ALIGN")


    # ═════ CORRIDOR ═══════════════════════════════════════════════════════
    def heading_from_banner(self):
        """The ground banner lies ACROSS the corridor -> corridor heading is 90°
        to its long axis (the one closer to the current nose). None if unsure."""
        t0, dirs, last_t = time.time(), [], 0
        here = self.here()
        while time.time() - t0 < 2.0:
            self.d.goto_ned(here[0], here[1], SEARCH_ALT_M, self.yaw)
            c = self.down_conf()
            if c is not None and c.get("axis") and c.get("t") != last_t \
                    and c.get("elong", 0) >= 3.0 and self.d.tilt_deg() < MAX_TILT_DEG:
                last_t = c["t"]
                a = self.d.alt() or SEARCH_ALT_M
                cx, cy = c["center"]
                ax, ay = c["axis"]
                f1, r1 = down_pixel_to_body(cx - 60 * ax, cy - 60 * ay, a)
                f2, r2 = down_pixel_to_body(cx + 60 * ax, cy + 60 * ay, a)
                bf, br = (f2 - f1) * self.scale[0], (r2 - r1) * self.scale[1]
                # banner axis angle in body frame (0 = nose); corridor = +90°
                dirs.append(math.atan2(br, bf) + math.pi / 2)
            self.tick(f"HEADING from banner ({len(dirs)} looks)")
        if len(dirs) < 4:
            log.warning(f"  Banner heading: only {len(dirs)} usable looks "
                        "(is it a long strip?) — keeping the current heading")
            return None
        # average a direction that is only known modulo 180°: double the angles
        c2 = float(np.mean([math.cos(2 * d) for d in dirs]))
        s2 = float(np.mean([math.sin(2 * d) for d in dirs]))
        spread = math.degrees(math.sqrt(max(0.0, -2 * math.log(max(math.hypot(c2, s2), 1e-6)))) / 2)
        rel = math.atan2(s2, c2) / 2                  # in (-90°, 90°] = closest to nose
        log.info(f"  Banner says corridor is {math.degrees(rel):+.1f}° from the nose "
                 f"({len(dirs)} looks, spread {spread:.1f}°)")
        if spread > 10:
            log.warning("  heading spread too large — keeping the current heading")
            return None
        if abs(math.degrees(rel)) > MAX_YAW_FIX_DEG:
            log.error(f"  corridor is {math.degrees(rel):+.0f}° from the nose (> {MAX_YAW_FIX_DEG}°): "
                      "can't tell which way to go in — NOT entering")
            return "refuse"
        return self.yaw + rel

    def lidar_ok(self):
        if self.lidar is None:
            return False, "no LiDAR object"
        dead = [k for k in ("front", "left", "right") if not self.lidar.alive(k)]
        return (not dead), ("all 3 answering" if not dead else f"NOT answering: {dead}")

    def abort_climb(self, why):
        log.error(f"\n!!! CORRIDOR ABORT: {why} — climbing straight up to {ABORT_ALT_M} m, then RTL")
        p = self.here()
        t0 = time.time()
        while (self.d.alt() or 0) < ABORT_ALT_M - 0.3 and time.time() - t0 < 15:
            self.d.goto_ned(p[0], p[1], ABORT_ALT_M, self.yaw)
            self.tick(f"ABORT climb {self.d.alt() or 0:.1f} m")
        self.releasing = True
        self.d.set_mode("RTL")
        return False

    def corridor(self):
        """Banner centred + facing the corridor (self.yaw). Back off, descend,
        fly through with LiDAR centring. True when out the far end."""
        u = (math.cos(self.yaw), math.sin(self.yaw))        # along corridor (N, E)
        v = (-math.sin(self.yaw), math.cos(self.yaw))       # to the right
        B = self.here()                                     # banner = corridor mouth centre
        E0 = (B[0] - u[0] * BACKOFF_M, B[1] - u[1] * BACKOFF_M)
        s_exit = BACKOFF_M + CORRIDOR_LEN_M + EXIT_MARGIN_M
        ok, msg = self.lidar_ok()
        log.info(f"\n── CORRIDOR ({BANNER_COLOR}) heading {math.degrees(self.yaw) % 360:.1f}°, "
                 f"len {CORRIDOR_LEN_M} m, width {CORRIDOR_W_M} m — LiDARs: {msg} ──")
        if not ok:
            log.error("  ✗ LiDARs not healthy — NOT entering the corridor → RTL")
            self.releasing = True
            self.d.set_mode("RTL")
            return False
        # back off at 5 m, then down to corridor height
        self.d.set_speed(ALIGN_SPEED_MS)
        self.settle(E0, SEARCH_ALT_M, "BACKOFF")
        log.info(f"  backed off {BACKOFF_M} m; descending to {CORRIDOR_ALT_M} m")
        t0 = time.time()
        while abs((self.d.alt() or 0) - CORRIDOR_ALT_M) > 0.15 and time.time() - t0 < 25:
            self.d.goto_ned(E0[0], E0[1], CORRIDOR_ALT_M, self.yaw)
            self.tick(f"DESCEND to corridor {self.d.alt() or 0:.1f} m")
        self.settle(E0, CORRIDOR_ALT_M, "ENTRY")
        self.d.set_speed(CORRIDOR_SPEED)
        lat_ref = 0.0                       # wanted lateral position (m right of the E0 line)
        t0 = t_log = time.time()
        t_front = None
        c0 = CORRIDOR_W_M / 2 - (0.225 - SIDE_TIP_M)   # side reading when centred
        n_walls = 0
        while True:
            now = time.time()
            if now - t0 > 90:
                return self.abort_climb("corridor timeout (90 s)")
            p = self.here()
            rel = (p[0] - E0[0], p[1] - E0[1])
            s_ = rel[0] * u[0] + rel[1] * u[1]              # progress
            lat = rel[0] * v[0] + rel[1] * v[1]             # right of the line
            if s_ >= s_exit - 0.3:
                log.info(f"  ✓ THROUGH the corridor (progress {s_:.1f} m, walls seen in "
                         f"{n_walls} readings, max |lat corr| "
                         f"{max([abs(x[1]) for x in self.corr_log] or [0]):.2f} m)")
                return True
            for k in ("front", "left", "right"):
                if not self.lidar.alive(k):
                    return self.abort_climb(f"{k} LiDAR stopped answering")
            F, L, R = self.lidar.get("front"), self.lidar.get("left"), self.lidar.get("right")
            Lw = L if (L is not None and L < WALL_SEEN_MAX_M) else None
            Rw = R if (R is not None and R < WALL_SEEN_MAX_M) else None
            # ── centring: where is the corridor centre, relative to us? ──
            # err = how far the corridor centre is to our RIGHT (m)
            err = None
            if Lw is not None and Rw is not None:
                err = (Rw - Lw) / 2
            elif Lw is not None:                    # only the left wall in view
                err = c0 - Lw
            elif Rw is not None:                    # only the right wall in view
                err = Rw - c0
            if err is not None:
                n_walls += 1
                lat_ref = lat_ref * (1 - CENTRE_GAIN) + (lat + err) * CENTRE_GAIN
            # ── safety ──
            fc = F - FRONT_TIP_M if F is not None else None
            lc = Lw - SIDE_TIP_M if Lw is not None else None
            rc = Rw - SIDE_TIP_M if Rw is not None else None
            side_min = min([c for c in (lc, rc) if c is not None] or [9])
            if side_min < SIDE_ABORT_M:
                return self.abort_climb(f"prop {side_min:.2f} m from a wall")
            hold = False
            if fc is not None and fc < FRONT_STOP_M:
                t_front = t_front or now
                hold = True
                if now - t_front > 5:
                    return self.abort_climb(f"blocked ahead for 5 s (front prop clearance {fc:.2f} m)")
            else:
                t_front = None
            if side_min < SIDE_PUSH_M:
                hold = True                          # stop going forward, recentre first
                if err is not None:
                    lat_ref = lat + err
            fwd = s_ if hold else min(s_ + LOOKAHEAD_M, s_exit)
            tgt = (E0[0] + u[0] * fwd + v[0] * lat_ref, E0[1] + u[1] * fwd + v[1] * lat_ref)
            self.d.goto_ned(tgt[0], tgt[1], CORRIDOR_ALT_M, self.yaw)
            self.corr_log.append((round(s_, 2), round(lat_ref - lat, 2), L, R, F))
            if now - t_log > 0.5:
                t_log = now
                fm = lambda x: f"{x:.2f}" if x is not None else "--"
                log.info(f"    s={s_:4.1f}/{s_exit:.1f} m  lat={lat:+.2f} ref={lat_ref:+.2f}  "
                         f"L={fm(L)} R={fm(R)} F={fm(F)}  prop-clear L={fm(lc)} R={fm(rc)} F={fm(fc)}"
                         f"  alt {self.d.alt() or 0:.2f}{'  HOLD' if hold else ''}")
            self.tick(f"CORRIDOR s={s_:.1f}")

    def after_corridor(self):
        if AFTER_CORRIDOR == "land":
            log.info("  → LAND after the corridor")
            self.releasing = True
            self.d.set_mode("LAND")
            t0 = time.time()
            while self.d.armed() and time.time() - t0 < 40:
                self.tick(f"LANDING {self.d.alt() or 0:.2f} m")
        elif AFTER_CORRIDOR == "hover":
            p = self.here()
            t0 = time.time()
            while time.time() - t0 < 10:
                self.d.goto_ned(p[0], p[1], CORRIDOR_ALT_M, self.yaw)
                self.tick("HOVER after corridor")
            self.releasing = True
            self.d.set_mode("RTL")
        elif AFTER_CORRIDOR == "climb":
            p = self.here()
            t0 = time.time()
            while (self.d.alt() or 0) < SEARCH_ALT_M - 0.3 and time.time() - t0 < 15:
                self.d.goto_ned(p[0], p[1], SEARCH_ALT_M, self.yaw)
                self.tick("CLIMB after corridor")
        else:
            self.releasing = True
            self.d.set_mode("RTL")

    def run_corridor(self):
        if CORRIDOR_HEADING_DEG is not None:
            self.yaw = math.radians(CORRIDOR_HEADING_DEG)
        elif self.d.yaw() is not None:
            self.yaw = self.d.yaw()
        log.info(f"  Start heading = {math.degrees(self.yaw) % 360:.1f}°  "
                 f"(banner {BANNER_COLOR})")
        ok, msg = self.lidar_ok()
        log.info(f"  LiDARs before takeoff: {msg}")
        if not self.d.arm_and_takeoff(SEARCH_ALT_M):
            self.d.set_mode("RTL")
            return
        self.d.set_speed(SEARCH_SPEED_MS)
        hold = self.here()
        t0 = time.time()
        while time.time() - t0 < 2.0:
            self.d.goto_ned(hold[0], hold[1], SEARCH_ALT_M, self.yaw)
            self.tick("SETTLE")
        for attempt in range(2):
            if not self.find_banner():
                log.warning(f"\n✗ {BANNER_COLOR} banner not found → RTL")
                self.releasing = True
                self.d.set_mode("RTL")
                return
            if not self.align():
                self.down.reset()
                continue
            new = self.heading_from_banner()
            if new == "refuse":
                log.warning("  → RTL (point the nose at the corridor before takeoff)")
                self.releasing = True
                self.d.set_mode("RTL")
                return
            if new is None:
                log.warning("  using the take-off heading as the corridor heading")
            else:
                turn = (math.degrees(new - self.yaw) + 180) % 360 - 180
                log.info(f"  Turning {turn:+.1f}° to face the corridor "
                         f"({math.degrees(new) % 360:.1f}°)")
                self.yaw = new
                self.calibrated = [False, False]          # body axes moved
                if not self.servo_centre(SEARCH_ALT_M, CENTRE_TOL_5M, "RE-CENTRE"):
                    continue
            if self.corridor():
                self.after_corridor()
                log.info("\n✓ CORRIDOR DONE")
            return
        log.warning("  Could not centre on the banner → RTL")
        self.releasing = True
        self.d.set_mode("RTL")

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
        # stage: down to 3 m, centre again (closer = more accurate)
        log.info(f"\n── DESCEND to {LAND_STAGE_M:.0f} m and re-centre ──")
        self.descend_to(LAND_STAGE_M, "STAGE")
        if not self.servo_centre(LAND_STAGE_M, CENTRE_TOL_LOW, "STAGE"):
            log.warning("  Lost the banner at 3 m — climbing back to re-centre")
            self.descend_to(SEARCH_ALT_M, "CLIMB")
            if not self.servo_centre(SEARCH_ALT_M, CENTRE_TOL_5M, "RE-ALIGN"):
                log.warning("  ✗ Banner lost — NOT landing blind → RTL")
                self.releasing = True
                self.d.set_mode("RTL")
                return
            self.descend_to(LAND_STAGE_M, "STAGE")
            if not self.servo_centre(LAND_STAGE_M, CENTRE_TOL_LOW, "STAGE"):
                log.warning("  ✗ Still can't centre at 3 m → RTL")
                self.releasing = True
                self.d.set_mode("RTL")
                return
        if FINAL_ACTION == "land":
            log.info(f"\n── FINAL DESCENT to {LAND_SWITCH_M} m over N={self.lock_ne[0]:.2f} "
                     f"E={self.lock_ne[1]:.2f}, then LAND ──")
            self.descend_to(LAND_SWITCH_M, "FINAL")
            p = self.here()
            log.info(f"  at {self.d.alt() or 0:.2f} m, {math.hypot(p[0] - self.lock_ne[0], p[1] - self.lock_ne[1]):.2f} m "
                     f"from the centred spot → LAND")
            self.releasing = True
            self.d.set_mode("LAND")
            t0 = time.time()
            while self.d.armed() and time.time() - t0 < 60:
                self.tick(f"LANDING alt={self.d.alt() or 0:.2f}m")
            log.info("  ✓ Landed" if not self.d.armed() else "  Landing timeout")
        else:
            log.info(f"\n── HOVER {HOVER_ALT_M} m over banner, facing corridor ──")
            self.descend_to(HOVER_ALT_M, "HOVER")
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
            ang = ""
            if dc.get("axis"):
                cx, cy = dc["center"]
                ax, ay = dc["axis"]
                f1, r1 = down_pixel_to_body(cx - 60 * ax, cy - 60 * ay, alt)
                f2, r2 = down_pixel_to_body(cx + 60 * ax, cy + 60 * ay, alt)
                rel = math.degrees(math.atan2(r2 - r1, f2 - f1) + math.pi / 2)
                rel = (rel + 90) % 180 - 90
                ang = f"  strip {dc.get('elong', 0):.1f}:1 → corridor {rel:+.0f}° from nose"
            parts.append(f"DOWN fwd={f:+.2f} rgt={r:+.2f}m  fill={dc['fill']:.2f} "
                         f"sol={dc['solidity']:.2f} tex={dc['texture']:.0f}{ang}")
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
    global MAX_TRIES, SEARCH_STEP_FWD_M, SEARCH_SIDE_M, CORRIDOR_LEN_M, CORRIDOR_ALT_M
    global CORRIDOR_W_M, AFTER_CORRIDOR, CORRIDOR_SPEED
    a = sys.argv

    def arg(name, typ=float):
        return typ(a[a.index(name) + 1]) if name in a else None

    if "--lidar-scan" in a:
        lidar_scan()
        return
    if "--lidar-bench" in a:
        lidar_bench()
        return
    set_banner_color("blue" if "--blue" in a else "green")
    MAX_TRIES = arg("--tries", int) or MAX_TRIES
    SEARCH_STEP_FWD_M = arg("--step") or SEARCH_STEP_FWD_M
    SEARCH_SIDE_M = arg("--side") or SEARCH_SIDE_M
    CORRIDOR_LEN_M = arg("--corridor-len") or CORRIDOR_LEN_M
    CORRIDOR_W_M = arg("--corridor-width") or CORRIDOR_W_M
    CORRIDOR_ALT_M = arg("--corridor-alt") or CORRIDOR_ALT_M
    CORRIDOR_SPEED = arg("--speed") or CORRIDOR_SPEED
    AFTER_CORRIDOR = arg("--after", str) or AFTER_CORRIDOR
    if CORRIDOR_ALT_M > WALL_H_M - 0.6:
        raise SystemExit(f"--corridor-alt {CORRIDOR_ALT_M} too close to the wall top ({WALL_H_M} m)")
    log.info(f"  banner={BANNER_COLOR}  search: {MAX_TRIES} tries x {SEARCH_STEP_FWD_M} m fwd, "
             f"±{SEARCH_SIDE_M} m side | corridor {CORRIDOR_LEN_M} x {CORRIDOR_W_M} m at "
             f"{CORRIDOR_ALT_M} m, {CORRIDOR_SPEED} m/s, after: {AFTER_CORRIDOR}")
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
    log.info(f"SQUADRONE — CORRIDOR ({BANNER_COLOR})  ({'BENCH' if bench_mode else 'FLIGHT'})")
    log.info("=" * 64)

    down = fwd = drone = lidar = None
    pv = Preview()
    try:
        down, fwd = open_cameras(use_fwd)
        if bench_mode:
            bench(down, fwd, pv)
            return
        from pymavlink import mavutil as _m
        mavutil = _m
        lidar = Lidars()
        lidar.start()
        time.sleep(0.5)
        log.info("  LiDAR now: " + "  ".join(
            f"{k}={lidar.get(k):.2f}m" if lidar.get(k) is not None else f"{k}=--"
            for k in ("front", "left", "right")))
        drone = Drone()
        drone.connect()
        gps = drone.get("GPS_RAW_INT")
        if gps is not None and gps.fix_type < 3:
            log.warning(f"  GPS fix_type={gps.fix_type} — position lock needs 3D fix")
        log.info(f"  Heading now {math.degrees(drone.yaw() or 0) % 360:.1f}° — "
                 "nose must point DOWN THE CORRIDOR")
        input("\n  >>> ENTER to arm + take off   (Ctrl+C = abort / RTL) <<<\n")
        m = Mission(drone, down, fwd, pv)
        m.lidar = lidar
        m.run_corridor()
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
        for c in (down, fwd, lidar):
            if c:
                c.stop()
        if drone:
            drone.close()
        pv.close()


if __name__ == "__main__":
    main()