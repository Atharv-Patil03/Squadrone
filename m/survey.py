#!/usr/bin/env python3
"""
SQUADRONE — Lawnmower Survey + QR scan & align   (QR only)
File: survey_qr.py      (built on lawnmower_trial.py)
Team: Squadrone AT2026RC60, PVGCOET Pune, SAEINDIA AeroTHON 2026

WHAT IT DOES
  1. Takeoff -> hover at REF_ALT_M (5 m) over the takeoff spot -> read the
     START QR: its text (may be empty) AND its image (perspective-corrected
     200x200 crop). Saved to ~/squadrone/reference_qr.png/.json.
  2. Climb to SURVEY_ALT_M (10 m, rulebook) and fly the lawnmower inside the
     field A -> B ... -> D WITHOUT stopping. The camera (ArduCam IMX519, rear,
     2328x1748, WeChat + pyzbar) logs every QR with its ground position and
     compares it with the start QR:
        both have text -> text decides;  otherwise -> image similarity
        (best of 4 rotations, normalised correlation, threshold 0.60).
  3. After the survey: fly to the best-matching QR, descend to DELIVERY_ALT_M
     (5 m), centre on it and ASK the operator to confirm (drone holds position
     while waiting): 'yes' -> RTL home; 'no' -> next candidate (or RTL).
  Results (~/squadrone): survey.log, survey_results.json (match + scores per
  QR), survey_map.png, match_report.png (start QR next to every QR with its
  score/verdict), qr_crops/qrN.png, reference_qr.png, TIMING per phase in the log.
  SAFETY: >3 m outside the field (after reaching A) = automatic RTL. Mode
  change by the pilot (e.g. Loiter) = script stops immediately.

RUN
  python3 survey.py --plan-only                 # path + map, no drone
  python3 survey.py --record A                  # save a corner from drone GPS
  python3 survey.py --bench --save-ref          # save the START QR on the ground
  python3 survey.py --bench                     # live match scores vs saved start QR
  python3 survey.py --alt 10 --speed 1.0        # full mission (start QR read at takeoff)
  python3 survey.py --alt 10 --ref saved        # use the start QR saved on the bench
  options: --match text|image|both  --threshold 0.6  --ref-alt 5  --delivery-alt 5
"""

import os
import sys
import json
import math
import time
import signal
import logging
import argparse
import threading
from collections import deque

import cv2
import numpy as np

Picamera2 = None
mavutil = None

# ═════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ═════════════════════════════════════════════════════════════════════════════

# Field — keep A, B, C, D order:  A = start, A->B = first pass, D = finish
# COMPETITION VENUE (Bangalore), waypoint map 30-10-2026: field ~16 x 23 m.
# (old Pune practice field: A 18.4897888, 73.8516870 ...)
POINT_A = (13.1705094, 77.5330114)   # start
POINT_B = (13.1705518, 77.5328726)   # end of first pass
POINT_C = (13.1707492, 77.5329336)
POINT_D = (13.1707059, 77.5330857)   # finish
VERTICES = [POINT_A, POINT_B, POINT_C, POINT_D]
# Corners recorded BY THE DRONE (python survey.py --record A ... D) are saved
# here and override the values above. Delete the file to go back to them.
CORNERS_FILE = os.path.expanduser("~/squadrone/field_corners.json")
INSET_M = 2.0

SURVEY_ALT_M    = 10.0
SPACING_M       = 8.0
SPEED_MPS       = 1.0
ACCEPT_RADIUS_M = 1.0
WAYPOINT_TIMEOUT = 120
GEOFENCE_MARGIN_M = 3.0    # further than this outside the field -> RTL
MAX_START_DIST_M  = 60.0   # refuse to start further than this from A
WARN_START_DIST_M = 15.0   # warn (but allow) beyond this

# QR
QR_ALIGN_EACH     = False  # True = stop and centre over every QR (old behaviour)
QR_DECODE_ALT_M   = None   # None = never leave survey alt (rulebook: 10 m).
                           # --decode-alt 8 allows dropping to 8 m over a hard QR
EXPOSURE_EV       = -1.0   # darker exposure: glossy QR in sun glare stays readable
QR_MERGE_M        = 2.0    # undecoded sightings closer than this = same QR
QR_TEXT_MERGE_M   = 8.0    # same text within this = same QR
QR_VISIT_TIMEOUT  = 30
QR_CENTRE_TOL_M   = 0.30
QR_HOLD_S         = 1.0
QR_SIZE_M         = 1.0    # real QR size (rulebook: 1 m x 1 m)
QR_SIZE_TOL       = 2.5    # accept 1/2.5 .. 2.5 x the expected size
TARGET_QR         = None   # text (or part of it) of the QR to finish over
HOVER_S           = 10

# Reference (start) QR matching — the QRs may carry NO data, so the QR IMAGE
# is compared too (team QR script method: 200x200 perspective crop, normalised
# correlation, best of 4 rotations).
MATCH_MODE        = "both"   # "text" | "image" | "both" (text decides when both have text)
IMAGE_MATCH_THRESHOLD = 0.60 # tune on the bench: --bench prints similarity scores
CROP_SIZE         = 200
REF_ALT_M         = 5.0      # hover here over the takeoff spot to read the start QR
REF_TIMEOUT_S     = 25
REF_SIGHTINGS     = 3
REF_EXCLUDE_M     = 2.5      # never treat the start QR itself as a field QR
DELIVERY_ALT_M    = 5.0      # descend to this over the matching QR
REF_PNG  = os.path.expanduser("~/squadrone/reference_qr.png")
REF_JSON = os.path.expanduser("~/squadrone/reference_qr.json")
CROP_DIR = os.path.expanduser("~/squadrone/qr_crops")

# Camera (DOWN, ArduCam IMX519, rear, looking straight down)
CAM_MODEL        = "imx519"       # --cam imx500 = Pi AI Camera (FIXED focus, no motor)
CAMS = {   # model: (resolution, HFOV, VFOV) — full field of view modes
    "imx519": ((2328, 1748), 66.0, 51.0),   # ArduCam 16 MP autofocus (2x2 binned)
    "imx500": ((2028, 1520), 67.1, 52.9),   # Pi AI Camera 12 MP, 4.74 mm lens (2x2 binned)
}
CAM_RES          = (2328, 1748)   # --res 4656x3496 = full sensor (sharper, slower)
# ── PAYLOAD (same servos as payload_test.py, field-verified) ──────────────────
# SERVO5_FUNCTION and SERVO6_FUNCTION must be 0 so DO_SET_SERVO controls them.
PAYLOAD          = True      # --no-payload: confirm + RTL only
LATCH_SERVO      = 6
LATCH_CLOSED_PWM = 2000      # field-tested values (your payload_test.py)
LATCH_OPEN_PWM   = 1000
LATCH_WAIT_S     = 2.0
WINCH_SERVO      = 5
WINCH_STOP_PWM   = 1500
WINCH_UNWIND_PWM = 2000      # lowers the payload
WINCH_REWIND_PWM = 1000      # pulls it back up
UNWIND_S         = 50.0
REWIND_BEFORE_RTL = False    # --rewind: wind the line back up before RTL

# GO_ON_MATCH: the moment a QR is a CONFIDENT match (same text, or image score
# above threshold on 2+ looks) the survey is interrupted and the drone goes
# straight to it (5 m, confirm, payload). 'no' from the operator = climb back
# and resume the survey where it left off. --full-survey = old behaviour
# (finish the whole lawnmower first, then go to the best match).
GO_ON_MATCH = True
AUTO_DROP = False           # True = no operator prompt: drop by itself when SURE
                            # (same AprilTag ID, or close-up image >= CONFIDENT_IMAGE)
CONFIDENT_IMAGE = 0.85      # image score (median of 2+ looks) to interrupt the survey.
                            # Between IMAGE_MATCH_THRESHOLD and this = "possible match":
                            # the survey finishes, then the best-scoring QR is visited
                            # first (operator confirms each).

FOCUS_M          = None           # None = continuous AF (field-proven clear feed, 23-10)
                                  # --focus-alt N locks the lens at N m (was blurry 23-10)
PREVIEW_RES      = (640, 480)
HFOV_DEG         = 66.0
VFOV_DEG         = 51.0
CAM_OFFSET_FWD_M = -0.10          # camera behind drone centre
# FLIP_Y = True: flight 20-10 showed QR positions sliding WITH the drone along
# track (and the banner drifting back) = forward/back was reversed.
FLIP_X, FLIP_Y   = False, True
MAX_TILT_DEG     = 10.0
# Motion blur: the drone no longer stops over QRs, so frames are taken while
# moving. Cap the shutter time (brightness kept with sensor gain instead).
#   blur (pixels) = speed x shutter / mm-per-pixel:  1.5 m/s x 1 ms = 1.5 mm
#   = 0.3 px at 10 m (sharp);  x 10 ms = 2.7 px (soft QR modules)
MAX_SHUTTER_US   = 0              # 0 = fully automatic exposure (like the clear-feed
                                  # script). --shutter-us 1000 = old capped shutter
MAX_GAIN         = 16.0
QR_DWELL_S       = 3.0            # look this long at a NEW QR (sharp frames)
# LOOK-OVER: a new QR is first seen at the EDGE of the frame (tilted, often cut
# off -> no crop -> no image score). Instead of pausing where the drone is, fly
# OVER the QR and hold there so it sits in the middle of the frame, whole and
# sharp. Then every frame gives an image score. Skipped once the QR is decided
# (text decoded / confident image match).
QR_LOOK_OVER     = True
# RE-CHECK PASS: survey done and no QR confirmed -> fly back to EVERY QR found
# (best score first), hold over it at 10 m, and if still not sure drop to
# RECHECK_LOW_ALT_M for a bigger, sharper view. A score >= the threshold there
# -> descend to 5 m and ask the operator as usual.
RECHECK          = True
RECHECK_HOLD_S   = 3.0
RECHECK_LOW_ALT_M = 6.0
RECHECK_MAX_S    = 240           # whole re-check pass gives up after this
LOOK_ARRIVE_M    = 0.8            # "over it" = within this of the QR estimate
LOOK_MAX_S       = 12.0           # give up the look after this long in total
# Competition QRs carry NO data and have NO white border (organisers 25/26-10):
# they are found as SQUARES of high black/white contrast (find_qr_squares) and
# matched by IMAGE only. Partly-visible QRs (touching the frame edge) give bad
# crops -> not used for matching.
EDGE_MARGIN_FRAC = 0.03
# 29-10: the "QRs" are AprilTags (family 36h11, IDs 100/200/300/400/500 on the
# boards we got). Each tag carries an ID that survives rotation, shadows and the
# net pattern -> the match is simply "same ID as the start tag". Tested on the
# 10 photos: every one read, down to 50 px across (= a 1 m board from ~45 m).
MARKERS          = "apriltag"       # "apriltag" (competition) | "qr" (old QR pipeline)
TAG_FAMILY       = "DICT_APRILTAG_36h11"
DECODE_QR        = True             # original pipeline: READ QRs (WeChat -> pyzbar) and
                                    # match by text when both have data. The square
                                    # finder + image matching run ON TOP, always.
                                    # --image-only: skip reading (~13 ms/frame instead
                                    # of ~200-500 ms) if the QRs really carry no data.
INVERT_PASS      = True             # organisers 26-10: QR may be WHITE modules on BLACK
                                    # -> also try reading the inverted image
SHAPE_SIZE_TOL   = 1.6              # decode-free squares must be 1/1.6 .. 1.6 x 1 m
FRAME_EVERY_S    = 2.0            # save a flight frame this often (diagnostics)
MAX_SAVED_FRAMES = 400

SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE   = 921600
EARTH_R     = 6_378_137.0

DEBUG_DIR = os.path.expanduser("~/squadrone")
os.makedirs(DEBUG_DIR, exist_ok=True)
FRAME_DIR = os.path.join(DEBUG_DIR, "flight_frames")
class _NonBlockingConsole(logging.Handler):
    """Console output that can NEVER stall the flight code. If the terminal
    stops reading (VNC/SSH/Wi-Fi dropout when the drone flies away), a normal
    StreamHandler blocks inside print and freezes EVERY thread that logs —
    camera, MAVLink reader, mission loop (flight 26-10: 4 min frozen, then
    'FC heartbeat lost'). Here lines go into a queue; a background thread
    writes them; if the queue is full, lines are dropped (still in the file)."""

    def __init__(self):
        super().__init__()
        import queue
        self.q = queue.Queue(maxsize=2000)
        self.dropped = 0
        threading.Thread(target=self._pump, daemon=True).start()

    def emit(self, record):
        try:
            self.q.put_nowait(self.format(record))
        except Exception:
            self.dropped += 1

    def _pump(self):
        while True:
            line = self.q.get()
            try:
                if self.dropped:
                    sys.stdout.write(f"[console dropped {self.dropped} lines - see the log file]\n")
                    self.dropped = 0
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
            except Exception:
                pass


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[_NonBlockingConsole(),
                              logging.FileHandler(os.path.join(DEBUG_DIR, "survey.log"))])
log = logging.getLogger("survey")

# ═════════════════════════════════════════════════════════════════════════════
# GEOMETRY  (field frame: x = east, y = north, metres from POINT_A)
# ═════════════════════════════════════════════════════════════════════════════

def latlon_to_xy(lat, lon, lat0, lon0):
    x = math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * EARTH_R
    return x, y


def xy_to_latlon(x, y, lat0, lon0):
    return (lat0 + math.degrees(y / EARTH_R),
            lon0 + math.degrees(x / (EARTH_R * math.cos(math.radians(lat0)))))


def inset_quad(quad, d):
    """Move every edge of a convex quad inward by d metres (order preserved)."""
    if d <= 0:
        return list(quad)
    n = len(quad)
    area2 = sum(quad[i][0] * quad[(i + 1) % n][1] - quad[(i + 1) % n][0] * quad[i][1]
                for i in range(n))
    s = 1 if area2 > 0 else -1
    lines = []
    for i in range(n):
        a, b = quad[i], quad[(i + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        nx, ny = -dy / L * s, dx / L * s
        lines.append(((a[0] + nx * d, a[1] + ny * d), (dx, dy)))
    out = []
    for i in range(n):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        den = d1[0] * d2[1] - d1[1] * d2[0]
        t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
        out.append((p1[0] + t * d1[0], p1[1] + t * d1[1]))
    return out


def lerp(p, q, f):
    return (p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f)


def build_lawnmower(A, B, C, D, spacing):
    """Passes parallel to A-B, stepping towards D-C. Always ends at D.
    Returns [(x, y, label)] (unchanged logic from lawnmower_trial.py)."""
    width = max(math.dist(A, D), math.dist(B, C))
    n_steps = max(1, math.ceil(width / spacing - 1e-6))
    log.info(f"Field width {width:.1f} m -> {n_steps + 1} passes, "
             f"actual gap {width / n_steps:.1f} m (asked {spacing} m)")
    path = []
    for k in range(n_steps + 1):
        f = k / n_steps
        a_side, b_side = lerp(A, D, f), lerp(B, C, f)
        seg = ([(a_side, "A-side"), (b_side, "B-side")] if k % 2 == 0
               else [(b_side, "B-side"), (a_side, "A-side")])
        for p, lab in seg:
            path.append((p[0], p[1], lab))
    path[0], path[1] = (A[0], A[1], "A"), (B[0], B[1], "B")
    if math.dist(path[-1][:2], D) < 0.01:
        path[-1], path[-2] = (D[0], D[1], "D"), (C[0], C[1], "C")
    else:
        path[-1], path[-2] = (C[0], C[1], "C"), (D[0], D[1], "D")
        path.append((D[0], D[1], "D"))
    return path


def bearing(p, q):
    """Yaw (rad, 0 = north, clockwise) to fly from p to q in the field frame."""
    return math.atan2(q[0] - p[0], q[1] - p[1])


def pixel_to_ground(px, py, w, h, alt, yaw):
    """DOWN-cam pixel -> (dx_east, dy_north) metres from the DRONE CENTRE."""
    nx = (px - w / 2) / (w / 2)
    ny = (py - h / 2) / (h / 2)
    if FLIP_X:
        nx = -nx
    if FLIP_Y:
        ny = -ny
    rgt = nx * alt * math.tan(math.radians(HFOV_DEG / 2))
    fwd = ny * alt * math.tan(math.radians(VFOV_DEG / 2)) + CAM_OFFSET_FWD_M
    north = fwd * math.cos(yaw) - rgt * math.sin(yaw)
    east = fwd * math.sin(yaw) + rgt * math.cos(yaw)
    return east, north


def dist_outside(x, y, poly):
    """0 if (x, y) is inside poly, else distance (m) to the nearest edge."""
    if point_in_poly(x, y, poly):
        return 0.0
    best = 1e9
    for i in range(len(poly)):
        (ax, ay), (bx, by) = poly[i], poly[(i + 1) % len(poly)]
        L2 = (bx - ax) ** 2 + (by - ay) ** 2
        t = max(0.0, min(1.0, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / L2)) if L2 else 0
        best = min(best, math.hypot(x - ax - t * (bx - ax), y - ay - t * (by - ay)))
    return best


def compass(yaw):
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    deg = math.degrees(yaw) % 360
    return f"{deg:.0f}° ({names[int((deg + 22.5) // 45) % 8]})"


class GeofenceBreach(Exception):
    pass


class MatchFound(Exception):
    def __init__(self, q):
        super().__init__(f"QR #{q.id}")
        self.q = q


def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside

# ═════════════════════════════════════════════════════════════════════════════
# FIELD MAP IMAGE (field outline, plan, flown track, QRs)
# ═════════════════════════════════════════════════════════════════════════════

def render_map(poly, path=None, track=None, qrs=None, drone=None, ppm=12):
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    x0, y0 = min(xs) - 4, min(ys) - 4
    W = int((max(xs) + 4 - x0) * ppm)
    H = int((max(ys) + 4 - y0) * ppm)
    img = np.full((H, W, 3), 40, np.uint8)
    px = lambda p: (int((p[0] - x0) * ppm), H - 1 - int((p[1] - y0) * ppm))   # north up
    cv2.fillPoly(img, [np.array([px(p) for p in poly], np.int32)], (70, 90, 70))
    for lab, p in zip("ABCD", poly):
        cv2.putText(img, lab, px(p), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if path:
        for i in range(len(path) - 1):
            cv2.line(img, px(path[i]), px(path[i + 1]), (200, 200, 200), 1)
    if track and len(track) > 1:
        cv2.polylines(img, [np.array([px(p) for p in track], np.int32)], False, (255, 200, 0), 2)
    for q in (qrs or []):
        if q.rejected:
            continue
        col = (0, 255, 0) if q.aligned else (0, 200, 255)
        cv2.circle(img, px((q.x, q.y)), 6, col, 2)
        cv2.putText(img, (q.text or "?")[:14], (px((q.x, q.y))[0] + 8, px((q.x, q.y))[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    if drone:
        cv2.circle(img, px(drone), 5, (255, 255, 255), -1)
    cv2.putText(img, "N ^", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img

# ═════════════════════════════════════════════════════════════════════════════
# QR DETECTION
# ═════════════════════════════════════════════════════════════════════════════

try:
    from pyzbar import pyzbar as _zbar
    _HAVE_ZBAR = True
except Exception:
    _HAVE_ZBAR = False
_cv_qr = cv2.QRCodeDetector()

# ── WeChatQRCode (team's onboard QR algorithm): CNN detector + super-resolution,
#    far better than pyzbar on small / soft QRs from 10 m. Primary decoder;
#    pyzbar is the fallback. Needs opencv-contrib + the 4 model files.
WECHAT_FILES = ("detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel")
WECHAT_MODEL_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
    os.path.expanduser("~/squadrone/models"),
    os.path.expanduser("~/Downloads/Squadrone/Squadrone/models"),
]
_wechat = None
_wechat_dir = None


def load_wechat(extra_dir=None):
    """Find the model folder and create the detector. Returns True if loaded."""
    global _wechat, _wechat_dir
    if not hasattr(cv2, "wechat_qrcode_WeChatQRCode"):
        log.warning("  WeChatQRCode not in this OpenCV (need opencv-contrib) — pyzbar only")
        return False
    import glob
    dirs = ([extra_dir] if extra_dir else []) + WECHAT_MODEL_DIRS + \
        glob.glob(os.path.expanduser("~/Downloads/Squadrone/Squadrone/*/models"))
    for d in dirs:
        paths = [os.path.join(d, f) for f in WECHAT_FILES]
        if not all(os.path.isfile(p) for p in paths):
            continue
        for args in (paths, [paths[0], paths[1]], []):      # 4.x API, then newer ones
            try:
                _wechat = cv2.wechat_qrcode_WeChatQRCode(*args)
                _wechat_dir = d
                log.info(f"  WeChatQRCode loaded from {d}")
                return True
            except Exception:                   # any failure -> try next / pyzbar
                continue
    log.warning("  WeChatQRCode model files not found in " + ", ".join(dirs[:4]) +
                " — pyzbar only (copy the 'models' folder next to this script)")
    return False


def _decode_wechat(bgr, ox=0, oy=0, by="wechat"):
    out = []
    try:
        texts, points = _wechat.detectAndDecode(bgr)
    except cv2.error:
        return out
    for t, p in zip(texts or [], points if points is not None else []):
        quad = np.asarray(p, np.float32).reshape(-1, 2)[:4] + (ox, oy)
        out.append({"text": t or None, "c": tuple(float(v) for v in quad.mean(axis=0)),
                    "quad": quad, "by": by})
    return out


# Tile scan: WeChat shrinks the whole frame before detecting, so a small QR
# (far away) vanishes. Scanning a tile keeps it big relative to the input.
# Bench: 85 px QR -> 0/5 full frame, 5/5 tiled. All tiles every frame is too
# slow, so each frame = full frame + ONE tile, round robin.
SCAN_TILES = 3            # 3x3 tiles, 25 % overlap (0 = off)
GLARE_PASS = True         # retry with contrast equalisation when nothing decodes
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
_glare_i = 0
_extra_i = 0
_tile_i = 0


def _tile_rects(W, H, n=SCAN_TILES, ov=0.25):
    tw, th = int(W / (n - (n - 1) * ov)), int(H / (n - (n - 1) * ov))
    return [(int(i * tw * (1 - ov)), int(j * th * (1 - ov)), tw, th)
            for j in range(n) for i in range(n)]


def _decode_next_tile(bgr):
    global _tile_i
    rects = _tile_rects(bgr.shape[1], bgr.shape[0])
    x, y, w, h = rects[_tile_i % len(rects)]
    _tile_i += 1
    return _decode_wechat(bgr[y:y + h, x:x + w], x, y, by=f"wechat-tile{_tile_i % len(rects)}")


_tag_det = None
_tag_i = 0


def scan_tags(bgr):
    """AprilTags -> [{'text': 'TAG 200', 'id', 'c', 'quad' (4x2 full-res px), 'by'}].
    Half-size image first (~12 ms); full size every other frame when the half
    found nothing (small/far tags)."""
    global _tag_det, _tag_i
    if _tag_det is None:
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        _tag_det = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, TAG_FAMILY)), p)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    _tag_i += 1
    out = []
    passes = [(cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2),
                          interpolation=cv2.INTER_AREA), 2.0, "apriltag")]
    if _tag_i % 2 == 0:
        passes.append((gray, 1.0, "apriltag-full"))
    for img, k, by in passes:
        corners, ids, _ = _tag_det.detectMarkers(img)
        if ids is not None:
            for c, i in zip(corners, ids.ravel()):
                quad = c.reshape(4, 2).astype(np.float32) * k
                cc = tuple(float(v) for v in quad.mean(axis=0))
                if all(math.dist(cc, o["c"]) > 40 for o in out):
                    out.append({"text": f"TAG {int(i)}", "id": int(i), "c": cc,
                                "quad": quad, "by": by})
        if out:
            break
    return out


def scan_qr(bgr):
    """Return [{'text': str|None, 'c': (cx, cy), 'quad': np.array}] in full-res px.
    text None = a QR shape was found but could not be decoded (too small/blurred).
    Order: WeChatQRCode -> pyzbar -> OpenCV (same as the team's QR script)."""
    if MARKERS == "apriltag":
        return scan_tags(bgr)
    if not DECODE_QR:
        out = []
        for d in locate_qrs(bgr) + find_qr_squares(bgr):    # accurate corners first
            if all(math.dist(d["c"], o["c"]) > 40 for o in out):
                out.append(d)
        return out
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = []
    if _wechat is not None:
        # Full frame at HALF size: big QRs decode fine and it is always fast
        # (bench: max 49 ms). Full-res full frame sometimes took 30 s (SR
        # chewing on soil texture) — small QRs are covered by the tiles.
        half = cv2.resize(bgr, (bgr.shape[1] // 2, bgr.shape[0] // 2),
                          interpolation=cv2.INTER_AREA)
        # Light denoise: on busy texture (soil, grass) WeChat's CNN finds dozens
        # of false candidates and took 4-5 s per frame; sigma 1 px removes that
        # (-> ~11 ms) while QR modules (several px) are untouched.
        half = cv2.GaussianBlur(half, (0, 0), 1.0)
        found = _decode_wechat(half)
        global _glare_i
        _glare_i += 1
        if not found and GLARE_PASS and _glare_i % 2 == 0:     # every other frame
            # Glare washes the QR out (dark modules turn light grey). Local
            # contrast equalisation brings the pattern back.
            g = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
            found = _decode_wechat(cv2.cvtColor(_clahe.apply(g), cv2.COLOR_GRAY2BGR),
                                   by="wechat-glare")
        elif not found and INVERT_PASS and _glare_i % 2 == 1:  # the other frames
            # white-on-black QR: readers expect dark modules on light -> invert
            found = _decode_wechat(255 - half, by="wechat-inverted")
        for d in found:
            d["quad"] = d["quad"] * 2.0
            d["c"] = (d["c"][0] * 2.0, d["c"][1] * 2.0)
            out.append(d)
    if _wechat is not None and SCAN_TILES:
        for d in _decode_next_tile(bgr):            # adds far / small QRs
            if all(math.dist(d["c"], o["c"]) > 40 for o in out):
                out.append(d)
    if not out and _HAVE_ZBAR:
        for img_z, tag in ((gray, "pyzbar"),) + (((255 - gray), "pyzbar-inverted"),) * INVERT_PASS:
            for o in _zbar.decode(img_z, symbols=[_zbar.ZBarSymbol.QRCODE]):
                quad = np.array([(p.x, p.y) for p in o.polygon], np.float32)
                out.append({"text": o.data.decode("utf-8", "replace"),
                            "c": tuple(float(v) for v in quad.mean(axis=0)),
                            "quad": quad, "by": tag})
            if out:
                break
    elif not out and _wechat is None:
        try:
            ok, texts, pts, _ = _cv_qr.detectAndDecodeMulti(gray)
            if ok:
                for t, q in zip(texts, pts):
                    if t:
                        out.append({"text": t, "c": tuple(q.mean(axis=0)), "quad": q})
        except cv2.error:
            pass
    # undecoded QR shapes (finder patterns) on a half-size image
    half = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2),
                      interpolation=cv2.INTER_AREA)
    try:
        ok, pts = _cv_qr.detectMulti(half)
    except cv2.error:
        ok, pts = False, None
    # EXTRA (on top of the original readers above, which are unchanged): QRs
    # that could not be READ — team-mate's locators, then the contrast-square
    # finder. When this frame already READ a QR, the extras only run every 3rd
    # frame, so the original pipeline keeps its full speed on readable QRs.
    global _extra_i
    _extra_i += 1
    if not any(o.get("text") for o in out) or _extra_i % 3 == 0:
        for d in locate_qrs(bgr) + find_qr_squares(bgr):
            if all(math.dist(d["c"], o["c"]) > 40 for o in out):
                out.append(d)
    if ok and pts is not None:
        for q in pts:
            q = q * 2.0
            c = tuple(q.mean(axis=0))
            if all(math.dist(c, o["c"]) > 40 for o in out):
                out.append({"text": None, "c": c, "quad": q, "by": "cv-detect"})
    return out


# ── Decode-free QR finder ────────────────────────────────────────────────────
SHAPE_STD_MIN = 38      # local grey std (7x7 at half res) inside a QR
SHAPE_ASPECT_MAX = 1.45
SHAPE_FILL_MIN = 0.70
SHAPE_DENSE_MIN = 0.60  # share of the box that is high-contrast (rejects rings/borders)
SHAPE_SIDE_PX = (60, 2400)


def find_qr_squares(bgr, why=None):
    """A 1 m QR seen from above = a SQUARE of very high local black/white
    contrast; soil/grass have far less. Needs no data, no white border.
    Synthetic test (24 scenes, soil/grass, 5-12 m, glare, blur): 37/39 found."""
    H, W = bgr.shape[:2]
    s = 2 if W > 1400 else 1
    small = cv2.resize(bgr, (W // s, H // s), interpolation=cv2.INTER_AREA) if s > 1 else bgr
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = cv2.blur(g, (7, 7))
    std = np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - m * m, 0))
    raw = (std > SHAPE_STD_MIN).astype(np.uint8) * 255
    hi = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    hi = cv2.morphologyEx(hi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    cnts, _ = cv2.findContours(hi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < (SHAPE_SIDE_PX[0] / s) ** 2 * 0.5:
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        if min(w, h) < 2:
            continue
        side, asp, fill = math.sqrt(w * h) * s, max(w, h) / min(w, h), area / (w * h)
        bx, by, bw, bh = cv2.boundingRect(c)
        blob = np.zeros((bh, bw), np.uint8)
        cv2.drawContours(blob, [c - [bx, by]], -1, 255, -1)
        dense = cv2.countNonZero(cv2.bitwise_and(raw[by:by + bh, bx:bx + bw], blob)) / max(area, 1)
        ok = (SHAPE_SIDE_PX[0] <= side <= SHAPE_SIDE_PX[1] and asp <= SHAPE_ASPECT_MAX
              and fill >= SHAPE_FILL_MIN and dense >= SHAPE_DENSE_MIN)
        if why is not None:
            why.append(f"square? side {side:.0f}px aspect {asp:.2f} fill {fill:.2f} "
                       f"contrast-density {dense:.2f} -> {'QR' if ok else 'no'}")
        if ok:
            box = (cv2.boxPoints(((cx, cy), (w, h), ang)) * s).astype(np.float32)
            out.append({"text": None, "c": (cx * s, cy * s), "quad": box, "by": "shape"})
    return out

# ── Team-mate's locators (find a QR WITHOUT reading it) ─────────────────────
# a) OpenCV QRCodeDetectorAruco.detectMulti — locates codes, decoding not needed
# b) finder-square geometry: 3 corner squares at a right angle -> full quad
# Both give accurate corners (better crops than the contrast-square finder).
_QR_LOC = cv2.QRCodeDetectorAruco() if hasattr(cv2, "QRCodeDetectorAruco") else cv2.QRCodeDetector()
FINDER_MIN_SIDE_PX, FINDER_MAX_SIDE_PX = 5, 250       # at half resolution
F_MIN_ASPECT, F_MIN_FILL = 0.75, 0.80
F_MAX_SIZE_RATIO, F_MAX_LEG_RATIO, F_MAX_ABS_COS = 1.40, 1.30, 0.30
F_SPACING = (1.5, 12.0)


def _finder_candidates(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    out = []
    for inv in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):   # dark-on-light AND white-on-black
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, inv, 51, 7)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cnts, hier = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            continue
        raw = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            side = math.sqrt(area)
            if not (FINDER_MIN_SIDE_PX <= side <= FINDER_MAX_SIDE_PX):
                continue
            ap = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            if len(ap) != 4 or not cv2.isContourConvex(ap):
                continue
            (cx, cy), (w, h), _ = cv2.minAreaRect(cnt)
            if w < 1 or h < 1 or min(w, h) / max(w, h) < F_MIN_ASPECT or area / (w * h) < F_MIN_FILL:
                continue
            raw.append({"center": np.array([cx, cy], np.float32), "side": side, "area": area})
        raw.sort(key=lambda c: c["area"], reverse=True)
        kept = []
        for c in raw:                          # drop concentric duplicates
            if all(np.linalg.norm(c["center"] - k["center"]) >= 0.5 * k["side"] for k in kept):
                kept.append(c)
        out.append(kept)
    return out


def _best_triple(cands):
    from itertools import combinations
    best, best_score = None, float("inf")
    for trio in combinations(cands[:15], 3):
        sides = [c["side"] for c in trio]
        size_ratio = max(sides) / min(sides)
        if size_ratio > F_MAX_SIZE_RATIO:
            continue
        ms = float(np.mean(sides))
        pts = [c["center"] for c in trio]
        for v in range(3):
            p, a, b = pts[v], pts[(v + 1) % 3], pts[(v + 2) % 3]
            pa, pb = a - p, b - p
            la, lb = float(np.linalg.norm(pa)), float(np.linalg.norm(pb))
            if la < 1e-6 or lb < 1e-6:
                continue
            leg = max(la, lb) / min(la, lb)
            cosv = float(np.dot(pa, pb) / (la * lb))
            if leg > F_MAX_LEG_RATIO or abs(cosv) > F_MAX_ABS_COS:
                continue
            if not (F_SPACING[0] <= ((la + lb) / 2) / ms <= F_SPACING[1]):
                continue
            sc = abs(cosv) + (leg - 1) + (size_ratio - 1)
            if sc < best_score:
                best_score, best = sc, {"trio": trio, "p": p, "a": a, "b": b, "ms": ms}
    return best


def _triple_to_quad(m):
    p, a, b = m["p"], m["a"], m["b"]
    d = a + b - p
    u = (a - p) / np.linalg.norm(a - p)
    v = (b - p) / np.linalg.norm(b - p)
    h = m["ms"] / 2.0
    return np.array([p - h * u - h * v, a + h * u - h * v, d + h * u + h * v, b - h * u + h * v],
                    np.float32)


def locate_qrs(bgr):
    """Team-mate's locators on a half-size image -> [{'text': None, 'quad', 'c', 'by'}]."""
    H, W = bgr.shape[:2]
    s = 2 if W > 1400 else 1
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if s > 1:
        g = cv2.resize(g, (W // s, H // s), interpolation=cv2.INTER_AREA)
    out = []
    try:
        ok, pts = _QR_LOC.detectMulti(g)
        if ok and pts is not None:
            for q in np.asarray(pts, np.float32).reshape(-1, 4, 2):
                out.append({"text": None, "quad": q * s, "c": tuple(q.mean(axis=0) * s),
                            "by": "aruco-locate"})
    except cv2.error:
        pass
    for cands in _finder_candidates(g):
        for _ in range(3):
            m = _best_triple(cands)
            if m is None:
                break
            q = _triple_to_quad(m) * s
            out.append({"text": None, "quad": q, "c": tuple(q.mean(axis=0)), "by": "finders"})
            used = {id(c) for c in m["trio"]}
            cands = [c for c in cands if id(c) not in used]
    return out


def _ordered_quad(quad):
    """4 corners in one consistent winding (so crops are never mirrored)."""
    q = np.asarray(quad, np.float32).reshape(-1, 2)
    if len(q) != 4:
        q = cv2.boxPoints(cv2.minAreaRect(q)).astype(np.float32)
    c = q.mean(axis=0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]
    k = int(np.argmin(q[:, 0] + q[:, 1]))           # start top-left
    return np.roll(q, -k, axis=0)


def qr_crop(gray, quad, margin=0.06):
    """Perspective-corrected CROP_SIZE x CROP_SIZE equalised crop of a QR, with
    a small margin (a quad from a blob may be a module off on any side)."""
    try:
        dst = np.array([[0, 0], [CROP_SIZE, 0], [CROP_SIZE, CROP_SIZE], [0, CROP_SIZE]],
                       np.float32)
        q = _ordered_quad(quad)
        c = q.mean(axis=0)
        q = (c + (q - c) * (1 + 2 * margin)).astype(np.float32)
        M = cv2.getPerspectiveTransform(q, dst)
        return cv2.equalizeHist(cv2.warpPerspective(gray, M, (CROP_SIZE, CROP_SIZE)))
    except Exception:
        return None


def image_similarity(ref_gray, cand_gray, scales=(0.94, 1.0, 1.06), corner=0.30):
    """1) find the best alignment: 4 rotations x 3 scales x small shifts (template
    = middle 80 % of the reference slid over the candidate);
    2) score ONLY the data area of that alignment: the 4 corners (finder
    patterns — identical in every QR) are left out. Otherwise two different
    QRs look ~0.8 alike just because of their finder squares.
    Synthetic no-data QRs, reference 5 m vs 9-11 m: same 0.89-0.98 (median 0.95),
    different median 0.36 (worst look-alike with the same overlay box: 0.83)."""
    ref = cv2.GaussianBlur(ref_gray, (0, 0), 2.0).astype(np.float32)
    cand = cv2.GaussianBlur(cand_gray, (0, 0), 2.0).astype(np.float32)
    best = (-2.0, None)
    for sc in scales:
        t = cv2.resize(ref, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        k = int(t.shape[0] * 0.80)
        o = (t.shape[0] - k) // 2
        t = t[o:o + k, o:o + k]
        c = cand
        for _ in range(4):
            _, mx, _, loc = cv2.minMaxLoc(cv2.matchTemplate(c, t, cv2.TM_CCOEFF_NORMED))
            if mx > best[0]:
                best = (mx, (t, c, loc))
            c = np.ascontiguousarray(np.rot90(c, -1))
    _, (t, c, (x, y)) = best
    win = c[y:y + t.shape[0], x:x + t.shape[1]]
    n = t.shape[0]
    m = np.ones(t.shape, bool)
    q = max(1, int(n * (corner - 0.10) / 0.80))
    m[:q, :q] = m[:q, n - q:] = m[n - q:, :q] = m[n - q:, n - q:] = False
    a, b = t[m] - t[m].mean(), win[m] - win[m].mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


class Reference:
    """The start QR: its text (may be empty) and its image."""

    def __init__(self):
        self.text, self.crop, self.xy = None, None, None

    def ready(self):
        return self.crop is not None or bool(self.text)

    def compare(self, text, crop):
        """-> (match True/False or None if undecidable, method, score)"""
        rt, ct = (self.text or "").strip(), (text or "").strip()
        if MATCH_MODE in ("text", "both") and rt and ct:
            return ct == rt, "text", 1.0 if ct == rt else 0.0
        if MATCH_MODE == "text":
            return None, "none", None
        if crop is not None and self.crop is not None:
            sc = image_similarity(self.crop, crop)
            return sc >= IMAGE_MATCH_THRESHOLD, "image", sc
        return None, "none", None

    def save(self):
        os.makedirs(os.path.dirname(REF_PNG), exist_ok=True)
        if self.crop is not None:
            cv2.imwrite(REF_PNG, self.crop)
        with open(REF_JSON, "w") as f:
            json.dump({"text": self.text or ""}, f)

    def load(self):
        if os.path.isfile(REF_PNG):
            self.crop = cv2.imread(REF_PNG, cv2.IMREAD_GRAYSCALE)
        if os.path.isfile(REF_JSON):
            with open(REF_JSON) as f:
                self.text = json.load(f).get("text") or None
        return self.ready()

    def describe(self):
        return (f"text={self.text!r} " if self.text else "text=(none) ") + \
               ("+ image" if self.crop is not None else "(no image)")


# ── LENS FOCUS, set directly on the lens motor ──────────────────────────────
# This Pi's libcamera tuning for the IMX519 has NO autofocus ("no AF
# algorithm"), so AfMode / LensPosition are ignored. The lens motor (VCM)
# driver still exposes `focus_absolute` on a /dev/v4l-subdev*, which v4l2-ctl
# can set. --focus-sweep finds the sharpest value at 10 m and saves it here;
# every camera start applies it.
FOCUS_FILE = os.path.expanduser("~/squadrone/focus.json")
FOCUS_VALUE = None          # --focus-value N overrides focus.json


def _v4l2(args, timeout=3):
    import subprocess
    try:
        r = subprocess.run(["v4l2-ctl"] + args, capture_output=True, text=True,
                           timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except FileNotFoundError:
        return 127, "v4l2-ctl not installed (sudo apt install v4l-utils)"
    except Exception as e:
        return 1, str(e)


def find_lens():
    """-> dict(dev, min, max, step, default, value) for the lens motor, or None."""
    import glob, re
    for dev in sorted(glob.glob("/dev/v4l-subdev*")):
        rc, out = _v4l2(["-d", dev, "-l"])
        m = re.search(r"focus_absolute.*?min=(-?\d+)\s+max=(-?\d+)\s+step=(\d+)\s+"
                      r"default=(-?\d+)\s+value=(-?\d+)", out)
        if m:
            mn, mx, st, df, val = map(int, m.groups())
            return {"dev": dev, "min": mn, "max": mx, "step": max(1, st),
                    "default": df, "value": val}
    return None


def set_lens(lens, value):
    v = int(max(lens["min"], min(lens["max"], value)))
    rc, out = _v4l2(["-d", lens["dev"], "-c", f"focus_absolute={v}"])
    if rc != 0:
        log.warning(f"  lens: could not set focus_absolute={v}: {out.strip()[:120]}")
        return None
    return v


def get_lens_value(lens):
    import re
    rc, out = _v4l2(["-d", lens["dev"], "-C", "focus_absolute"])
    m = re.search(r"focus_absolute:\s*(-?\d+)", out)
    return int(m.group(1)) if m else None


def apply_saved_focus():
    """Called after the camera starts. Returns a one-line description."""
    lens = find_lens()
    if lens is None:
        return ("lens motor NOT found (no /dev/v4l-subdev with focus_absolute) — focus "
                "is wherever the lens rests. Check: ls /dev/v4l-subdev*")
    val, src = FOCUS_VALUE, "--focus-value"
    if val is None and os.path.isfile(FOCUS_FILE):
        with open(FOCUS_FILE) as f:
            fj = json.load(f)
        val, src = fj.get("focus_absolute"), (f"focus.json (swept at {fj.get('distance_m', '?')} m, "
                                              f"{fj.get('date', '?')})")
    if val is None:
        return (f"lens on {lens['dev']} at its resting value {lens['value']} "
                f"(range {lens['min']}..{lens['max']}) — NOT calibrated, run --focus-sweep")
    got = set_lens(lens, val)
    back = get_lens_value(lens)
    return (f"lens focus_absolute={got} (read back {back}) on {lens['dev']} from {src}"
            if got is not None else "lens focus could not be set")


_AF_STATES = {0: "idle", 1: "scanning", 2: "FOCUSED", 3: "FAILED"}


def _lens_txt(md):
    """'0.112dpt≈8.9m FOCUSED' — shows what autofocus is doing in every [cam] line."""
    lp, st = md.get("LensPosition"), md.get("AfState")
    if lp is None:
        return "n/a (lens not controlled!)"
    dist = "inf" if lp < 0.02 else f"{1 / lp:.1f}m"
    return f"{lp:.3f}dpt≈{dist} {_AF_STATES.get(st, st) if st is not None else ''}".strip()


class SurveyCam(threading.Thread):
    """Captures high-res frames and scans for QRs. Every observation is queued
    with the frame time so the mission can use the drone pose AT THAT MOMENT."""

    def __init__(self):
        super().__init__(daemon=True)
        self.cam = None
        self.obs = deque(maxlen=60)
        self.lk = threading.Lock()
        self.preview = None
        self.fps = 0.0
        self.scan_ms = 0.0
        self._stop_evt = threading.Event()
        self.sharp, self.md = 0.0, {}
        self.ae_locked, self.gain, self.target_mean = False, 1.0, None
        self.t_open = self.t_gain = 0.0
        self.overlay = ""                 # mission writes alt/speed/phase here
        self.focus_info = ""
        self.scan_enabled = True
        self.last_frame = None
        self.n_saved = 0

    def open(self):
        info = Picamera2.global_camera_info()
        idx = next((c.get("Num", i) for i, c in enumerate(info)
                    if CAM_MODEL in str(c.get("Model", "")).lower()), None)
        if idx is None:
            raise RuntimeError(f"camera {CAM_MODEL} not found: {info}")
        self.cam = Picamera2(idx)
        cfg = self.cam.create_video_configuration(
            main={"size": CAM_RES, "format": "RGB888"}, buffer_count=3)
        self.cam.configure(cfg)
        self.cam.start()
        time.sleep(1.5)
        self.t_open = time.time()
        for ctrl in ({"AwbEnable": True, "AwbMode": 5},
                     {"AeExposureMode": 1},           # short exposure = less motion blur
                     {"ExposureValue": EXPOSURE_EV}): # glare: keep whites from blowing out
            try:
                self.cam.set_controls(ctrl)
            except Exception as e:
                log.warning(f"  camera control {list(ctrl)} not applied: {e}")
        self.focus_info = self._set_focus()
        log.info(f"  Focus: {self.focus_info}")
        log.info(f"  Camera {CAM_MODEL} @ {CAM_RES}  "
                 f"(QR decoders: {'WeChat + ' if _wechat is not None else ''}"
                 f"{'pyzbar' if _HAVE_ZBAR else 'OpenCV'})")

    def _set_focus(self):
        """1) libcamera AF (AfMode advertised) -> lock LensPosition = 1/FOCUS_M,
           2) else drive the lens motor directly (v4l2 focus_absolute)."""
        has_af = "AfMode" in (getattr(self.cam, "camera_controls", {}) or {})
        if not has_af and FOCUS_VALUE is None and CAM_MODEL != "imx519":
            return f"{CAM_MODEL}: FIXED-focus lens (no motor) — nothing to set, can't drift"
        if FOCUS_VALUE is not None or not has_af:
            why = "--focus-value given" if FOCUS_VALUE is not None else \
                  "libcamera has NO AfMode for this camera"
            return f"[direct lens] ({why}) " + apply_saved_focus()
        try:
            if FOCUS_M:
                self.cam.set_controls({"AfMode": 0, "LensPosition": 1.0 / FOCUS_M})
                want = f"LOCKED at {FOCUS_M:g} m (LensPosition {1.0 / FOCUS_M:.3f} dpt)"
            else:
                self.cam.set_controls({"AfMode": 2, "AfSpeed": 1})
                want = "continuous AF"
        except Exception as e:
            return f"libcamera focus FAILED ({e}) -> [direct lens] " + apply_saved_focus()
        time.sleep(0.6)
        try:
            lp = self.cam.capture_metadata().get("LensPosition")
        except Exception:
            lp = None
        chk = "n/a" if lp is None else f"{lp:.3f} dpt (= {1 / lp:.1f} m)" if lp > 0.01 \
            else f"{lp:.3f} dpt (infinity)"
        return f"[libcamera AF] {want}; lens reports {chk}"

    def _shutter_control(self, md, small_gray):
        """Mini auto-exposure with a capped shutter: once AE wants a shutter
        longer than MAX_SHUTTER_US, take over: fixed short shutter, and keep
        the image brightness where it was by adjusting gain once a second."""
        if not MAX_SHUTTER_US:
            return
        md = md or {}
        now = time.time()
        exp, gain = md.get("ExposureTime"), md.get("AnalogueGain")
        mean = float(small_gray.mean())
        if not self.ae_locked:
            if exp and gain and exp > MAX_SHUTTER_US and now - self.t_open > 1.5:
                self.gain = float(min(MAX_GAIN, gain * exp / MAX_SHUTTER_US))
                self.target_mean = mean
                try:
                    self.cam.set_controls({"AeEnable": False,
                                           "ExposureTime": int(MAX_SHUTTER_US),
                                           "AnalogueGain": self.gain})
                    self.ae_locked = True
                    log.info(f"  Shutter capped: AE wanted {exp} us x{gain:.1f} → "
                             f"{MAX_SHUTTER_US} us x{self.gain:.1f} (target brightness "
                             f"{mean:.0f})")
                except Exception as e:
                    log.warning(f"  could not cap shutter: {e}")
                    self.ae_locked = None          # don't try again
            return
        if self.ae_locked and now - self.t_gain > 1.0 and mean > 1:
            self.t_gain = now
            new = float(min(MAX_GAIN, max(1.0, self.gain * (self.target_mean / mean) ** 0.5)))
            if abs(new - self.gain) / self.gain > 0.05:
                self.gain = new
                try:
                    self.cam.set_controls({"AnalogueGain": self.gain})
                except Exception:
                    pass

    def run(self):
        n, t0 = 0, time.time()
        t_diag = t_frame = 0.0
        while not self._stop_evt.is_set():
            try:
                req = self.cam.capture_request()
                try:
                    frame = req.make_array("main")          # BGR order already
                    md = req.get_metadata()
                finally:
                    req.release()
            except Exception as e:
                log.warning(f"  capture: {e}")
                time.sleep(0.2)
                continue
            t = time.time()
            small = cv2.resize(frame, PREVIEW_RES, interpolation=cv2.INTER_AREA)
            sg = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            self.sharp = float(cv2.Laplacian(sg, cv2.CV_64F).var())   # blur meter
            self.md = md
            self._shutter_control(md, sg)
            t_scan = time.time()
            qrs = scan_qr(frame) if self.scan_enabled else []
            self.last_frame = frame
            if qrs:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fh, fw = gray.shape
                mx, my = EDGE_MARGIN_FRAC * fw, EDGE_MARGIN_FRAC * fh
                for q in qrs:
                    if q.get("quad") is not None:
                        qq = np.asarray(q["quad"], np.float32).reshape(-1, 2)
                        q["full"] = bool((qq[:, 0] > mx).all() and (qq[:, 0] < fw - mx).all()
                                         and (qq[:, 1] > my).all() and (qq[:, 1] < fh - my).all())
                        q["crop"] = qr_crop(gray, q["quad"]) if q["full"] else None
                        q["area"] = float(cv2.contourArea(
                            np.asarray(q["quad"], np.float32).reshape(-1, 2)))
            self.scan_ms = (time.time() - t_scan) * 1000
            o = {"t": t, "qrs": qrs, "w": frame.shape[1], "h": frame.shape[0],
                 "sharp": self.sharp}
            with self.lk:
                self.obs.append(o)
                self.preview = (small, qrs, frame.shape[1] / PREVIEW_RES[0])
            n += 1
            if t - t0 > 2:
                self.fps, n, t0 = n / (t - t0), 0, t
            if t - t_diag >= 2.0:                  # camera health line
                t_diag = t
                log.info(f"  [cam] fps {self.fps:.1f}  scan {self.scan_ms:.0f} ms  "
                         f"sharpness {self.sharp:.0f}  shutter {md.get('ExposureTime', '?')} us  "
                         f"gain {md.get('AnalogueGain', 0):.1f}  "
                         f"lens {_lens_txt(md)}  QRs {len(qrs)}  "
                         f"| {self.overlay}")
            if FRAME_EVERY_S and t - t_frame >= FRAME_EVERY_S and self.n_saved < MAX_SAVED_FRAMES:
                t_frame = t
                fr = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2),
                                interpolation=cv2.INTER_AREA)
                for q in qrs:
                    cv2.polylines(fr, [(np.asarray(q["quad"]) / 2).astype(np.int32)], True,
                                  (0, 255, 0) if q["text"] else (0, 200, 255), 3)
                cv2.putText(fr, f"sharp {self.sharp:.0f}  shutter {md.get('ExposureTime', '?')}us "
                                f"gain {md.get('AnalogueGain', 0):.1f}  {self.overlay}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                os.makedirs(FRAME_DIR, exist_ok=True)
                cv2.imwrite(os.path.join(FRAME_DIR, f"f{self.n_saved:04d}.jpg"), fr,
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
                self.n_saved += 1

    def drain(self):
        with self.lk:
            out = list(self.obs)
            self.obs.clear()
        return out

    def stop(self):
        self._stop_evt.set()
        try:
            self.cam.stop()
            self.cam.close()
        except Exception:
            pass

# ═════════════════════════════════════════════════════════════════════════════
# DRONE (MAVLink reader thread, filtered heartbeats, pose history)
# ═════════════════════════════════════════════════════════════════════════════

class PilotTookOver(Exception):
    pass


class Drone:
    def __init__(self, conn, baud, origin):
        self.origin = origin
        log.info(f"Connecting {conn} @ {baud}…")
        self.mav = mavutil.mavlink_connection(conn, baud=baud)
        # The first heartbeat can come from a radio / other device (system 0).
        # Keep waiting until we see the FLIGHT CONTROLLER itself.
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
                nerr = getattr(mv, "total_receive_errors", 0)
                if nb == 0:
                    why = ("NOTHING received on the port -> Pixhawk not powered (battery!), "
                           "cable loose/TX-RX swapped, or another program holds the port "
                           f"(check: sudo fuser {conn})")
                elif npk == 0:
                    why = (f"{nb} bytes but 0 valid packets ({nerr} errors) -> WRONG BAUD: "
                           f"Pixhawk SERIALx_BAUD must match {baud}")
                else:
                    why = (f"{npk} packets but none is the flight controller's heartbeat "
                           "(only radio/GCS) -> FC still booting or not on this link")
                raise RuntimeError(f"No heartbeat from the flight controller on {conn} @ "
                                   f"{baud} in 30 s: {why}")
        log.info(f"  Heartbeat ✓ FC system={self.mav.target_system} "
                 f"component={self.mav.target_component}")
        self.lk = threading.Lock()
        self.state = {}
        self.poses = deque(maxlen=400)     # (t, x, y, alt, yaw, tilt)
        self._stop_evt = threading.Event()
        threading.Thread(target=self._reader, daemon=True).start()
        for msg_id, hz in ((30, 20), (33, 20), (24, 2)):
            self.mav.mav.command_long_send(
                self.mav.target_system, self.mav.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                msg_id, int(1e6 / hz), 0, 0, 0, 0, 0)

    def _reader(self):
        L = mavutil.mavlink
        while not self._stop_evt.is_set():
            try:
                m = self.mav.recv_match(blocking=True, timeout=0.5)
            except Exception:
                continue
            if m is None or m.get_type() == "BAD_DATA":
                continue
            if m.get_srcSystem() != self.mav.target_system:
                continue
            t = m.get_type()
            if t == "HEARTBEAT" and (m.get_srcComponent() != L.MAV_COMP_ID_AUTOPILOT1
                                     or m.type in (L.MAV_TYPE_GCS, L.MAV_TYPE_ONBOARD_CONTROLLER)
                                     or m.autopilot == L.MAV_AUTOPILOT_INVALID):
                continue                     # radios / GCS / companions — ignore
            with self.lk:
                self.state[t] = m
                if t == "HEARTBEAT":
                    self.state["_hb_t"] = time.time()
                if t == "GLOBAL_POSITION_INT":
                    att = self.state.get("ATTITUDE")
                    x, y = latlon_to_xy(m.lat / 1e7, m.lon / 1e7, *self.origin)
                    yaw = att.yaw if att else 0.0
                    tilt = math.degrees(max(abs(att.roll), abs(att.pitch))) if att else 0.0
                    self.poses.append((time.time(), x, y, m.relative_alt / 1000.0, yaw, tilt))
            if t == "STATUSTEXT":
                log.info(f"  [FC] {m.text}")
            if t == "COMMAND_ACK":
                with self.lk:
                    self.state[f"_ack_{m.command}"] = (m.result, time.time())

    def get(self, t):
        with self.lk:
            return self.state.get(t)

    def pose(self):
        with self.lk:
            return self.poses[-1] if self.poses else None

    def pose_at(self, t):
        with self.lk:
            if not self.poses:
                return None
            return min(self.poses, key=lambda p: abs(p[0] - t))

    def xy(self):
        p = self.pose()
        return (p[1], p[2]) if p else None

    def alt(self):
        p = self.pose()
        return p[3] if p else None

    def armed(self):
        hb = self.get("HEARTBEAT")
        return bool(hb and hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def mode(self):
        hb = self.get("HEARTBEAT")
        if not hb:
            return None
        return next((n for n, i in self.mav.mode_mapping().items() if i == hb.custom_mode), None)

    def check(self):
        if time.time() - (self.get("_hb_t") or time.time()) > 5:
            raise RuntimeError("FC heartbeat lost")
        if self.mode() not in (None, "GUIDED"):
            raise PilotTookOver()

    def set_mode(self, name, timeout=5):
        mid = self.mav.mode_mapping()[name]
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.mav.mav.set_mode_send(self.mav.target_system,
                                       mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mid)
            time.sleep(0.5)
            if self.mode() == name:
                log.info(f"  Mode → {name}")
                return True
        log.warning(f"  Mode {name} not confirmed")
        return False

    def cmd(self, c, *p):
        p = list(p) + [0] * (7 - len(p))
        self.mav.mav.command_long_send(self.mav.target_system, self.mav.target_component,
                                       c, 0, *p)

    def servo(self, num, pwm, label=""):
        """MAV_CMD_DO_SET_SERVO, retried until the FC acknowledges it."""
        L = mavutil.mavlink
        for attempt in range(5):
            t_send = time.time()
            self.cmd(L.MAV_CMD_DO_SET_SERVO, num, pwm)
            while time.time() - t_send < 0.6:
                ack = self.get(f"_ack_{L.MAV_CMD_DO_SET_SERVO}")
                if ack and ack[1] >= t_send:
                    if ack[0] == L.MAV_RESULT_ACCEPTED:
                        log.info(f"  SERVO{num} = {pwm} us  {label}  ✓")
                        return True
                    log.error(f"  SERVO{num} = {pwm} REJECTED by FC (result {ack[0]}) — "
                              f"is SERVO{num}_FUNCTION = 0?")
                    return False
                time.sleep(0.05)
        log.warning(f"  SERVO{num} = {pwm} sent, but NO acknowledgement from the FC")
        return False

    def set_param(self, name, value):
        self.mav.mav.param_set_send(self.mav.target_system, self.mav.target_component,
                                    name.encode(), float(value),
                                    mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

    def goto_xy(self, x, y, alt, yaw):
        lat, lon = xy_to_latlon(x, y, *self.origin)
        self.mav.mav.set_position_target_global_int_send(
            0, self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, 0b0000100111111000,
            int(lat * 1e7), int(lon * 1e7), float(alt), 0, 0, 0, 0, 0, 0, yaw, 0)

    def wait_gps(self, timeout=300):
        log.info("Waiting for GPS 3D fix…")
        t0 = time.time()
        while time.time() - t0 < timeout:
            g = self.get("GPS_RAW_INT")
            if g and g.fix_type >= 3 and self.pose():
                log.info(f"  GPS ✓ sats={g.satellites_visible} hdop={g.eph / 100:.2f}")
                return
            time.sleep(0.5)
        raise RuntimeError("No GPS fix")

    def arm_and_takeoff(self, alt):
        if not self.set_mode("GUIDED"):
            raise RuntimeError("GUIDED not accepted")
        log.info("  Arming…")
        t0 = time.time()
        while not self.armed():
            if time.time() - t0 > 15:
                raise RuntimeError("Arm failed — check pre-arm messages")
            self.cmd(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            time.sleep(1.0)
        log.info("  Armed ✓ — taking off")
        time.sleep(0.5)
        self.cmd(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, alt)
        t0, t_dis, t_log = time.time(), None, 0
        while (self.alt() or 0) < 0.9 * alt:
            if time.time() - t0 > 45:
                raise RuntimeError(f"Takeoff timed out at {self.alt():.1f} m")
            if not self.armed():
                t_dis = t_dis or time.time()
                if time.time() - t_dis > 1.5:
                    raise RuntimeError("FC disarmed during takeoff")
            else:
                t_dis = None
            if time.time() - t_log > 1:
                log.info(f"  alt {self.alt() or 0:.1f} / {alt} m  mode={self.mode()}")
                t_log = time.time()
            time.sleep(0.2)
        log.info("  Takeoff done ✓")
        self.cmd(mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, 1, SPEED_MPS, -1)

    def close(self):
        self._stop_evt.set()

# ═════════════════════════════════════════════════════════════════════════════
# QR REGISTRY
# ═════════════════════════════════════════════════════════════════════════════

class QR:
    _n = 0

    def __init__(self, text, x, y):
        QR._n += 1
        self.id, self.text, self.x, self.y = QR._n, text, x, y
        self.n, self.aligned, self.rejected, self.visits = 1, False, False, 0
        self.sims, self.sims_low = [], []    # image similarity to the reference
        self.text_match = None               # True/False when both have text
        self.best_crop, self.best_area = None, 0.0
        self.announced = False
        self.tried = False                   # already flown to for confirmation
        self.op_rejected = False             # operator said "no"

    def confident(self):
        """Match strong enough to interrupt the survey for."""
        if self.text_match:
            return True
        return bool(self.sims) and len(self.sims) >= 2 and \
            float(np.median(self.sims)) >= CONFIDENT_IMAGE

    def verdict(self):
        """-> (True/False/None, short reason)"""
        if self.text_match is not None:
            return self.text_match, "text"
        if self.sims:
            med = float(np.median(self.sims))
            if len(self.sims) >= 2 or med >= IMAGE_MATCH_THRESHOLD + 0.1:
                return med >= IMAGE_MATCH_THRESHOLD, f"image {med:.2f}"
            return None, f"image {med:.2f} (1 look)"
        return None, "no evidence"

    def score(self):
        if self.text_match:
            return 2.0
        return float(np.median(self.sims)) if self.sims else -1.0

    def update(self, text, x, y, w=None):
        w = w if w is not None else 1.0 / (self.n + 1)
        self.x, self.y = self.x * (1 - w) + x * w, self.y * (1 - w) + y * w
        self.n += 1
        if text and not self.text:
            self.text = text
            log.info(f"  ▣ QR #{self.id} decoded: '{text}'")

    def __repr__(self):
        return f"QR#{self.id}('{self.text}', {self.x:.1f}, {self.y:.1f})"


class Registry:
    def __init__(self):
        self.items = []

    def add(self, text, x, y):
        x, y = float(x), float(y)
        best = None
        for q in self.items:
            d = math.hypot(q.x - x, q.y - y)
            if text and q.text == text:
                limit = QR_TEXT_MERGE_M          # same text = same QR (unless far)
            elif text is None or q.text is None:
                limit = QR_MERGE_M
            else:
                continue                         # different texts never merge
            if d < limit and (best is None or d < best[0]):
                best = (d, q)
        if best:
            q = best[1]
            if not q.aligned:               # aligned coordinate is final
                q.update(text, x, y)
            elif text and not q.text:
                q.text = text
            return q, False
        q = QR(text, x, y)
        self.items.append(q)
        log.info(f"  ◎ New QR #{q.id} at ({x:.1f}, {y:.1f})  "
                 f"{repr(text) if text else '(not decoded yet)'}")
        return q, True

    def needs_visit(self):
        for q in self.items:
            if q.aligned or q.rejected or q.visits >= 2:
                continue
            if QR_ALIGN_EACH:
                return q
        return None

    def find(self, target):
        return next((q for q in self.items if q.text and target in q.text), None)

# ═════════════════════════════════════════════════════════════════════════════
# MISSION
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# MISSION
# ═════════════════════════════════════════════════════════════════════════════

class Survey:
    def __init__(self, drone, cam, field, path, origin, show=True):
        self.d, self.cam, self.field, self.path, self.origin = drone, cam, field, path, origin
        self.reg = Registry()
        self.track = []
        self.t_track = self.t_map = 0
        self.show = show and bool(os.environ.get("DISPLAY"))
        self.status = ""
        self.yaw = 0.0
        xy = drone.xy()
        start_out = dist_outside(*xy, field) if xy else 0.0
        self.fence_m = max(GEOFENCE_MARGIN_M, start_out + 2.0)
        self.ref = Reference()
        self.phases = []          # [name, start, end] — timing summary
        self.dwell_until, self.dwell_xy, self.no_dwell = 0.0, None, False
        self.look_q, self.look_t0, self.look_arr, self.look_n0 = None, 0.0, None, 0
        self._sharp = 0.0
        self.confirmed = None
        self.payload_log, self.payload_done, self.winch_running = [], False, False

    # ── camera observations → QR registry ────────────────────────────────
    def process(self):
        for o in self.cam.drain():
            p = self.d.pose_at(o["t"])
            if p is None:
                continue
            _, x, y, alt, yaw, tilt = p
            if alt < 2.0 or tilt > MAX_TILT_DEG:
                continue
            mpp = 2 * alt * math.tan(math.radians(HFOV_DEG / 2)) / o["w"]
            for q in o["qrs"]:
                if q.get("quad") is not None:          # size check (QR is 1 m)
                    quad = np.asarray(q["quad"], np.float32).reshape(-1, 2)
                    side_m = math.sqrt(max(cv2.contourArea(quad), 1.0)) * mpp
                    tol = SHAPE_SIZE_TOL if q.get("by") == "shape" else QR_SIZE_TOL
                    if not (QR_SIZE_M / tol <= side_m <= QR_SIZE_M * tol):
                        continue
                e, n = pixel_to_ground(q["c"][0], q["c"][1], o["w"], o["h"], alt, yaw)
                qx, qy = x + e, y + n
                if self.ref.xy and math.dist((qx, qy), self.ref.xy) < REF_EXCLUDE_M:
                    continue                      # that's the start QR itself
                if point_in_poly(qx, qy, self.field):
                    rec, is_new = self.reg.add(q["text"], qx, qy)
                    if is_new and QR_DWELL_S and not self.no_dwell:
                        if QR_LOOK_OVER and self.look_q is None:
                            self.look_q, self.look_t0, self.look_arr = rec, time.time(), None
                            self.look_n0 = 0
                            log.info(f"    ◎ QR #{rec.id}: flying OVER it for a centred look "
                                     f"({QR_DWELL_S:.0f} s; sharpness now {o.get('sharp', 0):.0f})")
                        elif not QR_LOOK_OVER:
                            self.dwell_until = time.time() + QR_DWELL_S
                            self.dwell_xy = (x, y)
                            log.info(f"    pausing {QR_DWELL_S:.0f} s for sharp frames of QR "
                                     f"#{rec.id} (sharpness now {o.get('sharp', 0):.0f})")
                    rec.last_seen = o["t"]
                    rec.last_off = math.hypot(e, n)
                    self._sharp = o.get("sharp", 0) or 0
                    self.judge(rec, q, alt)

    def judge(self, rec, q, alt):
        """Compare one sighting with the reference and keep the evidence."""
        crop = q.get("crop")
        rec.n_seen = getattr(rec, "n_seen", 0) + 1
        if crop is not None:
            rec.n_crop = getattr(rec, "n_crop", 0) + 1
        rec.sharp_max = max(getattr(rec, "sharp_max", 0.0), self._sharp)
        if crop is not None and q.get("area", 0) > rec.best_area:
            rec.best_crop, rec.best_area = crop, q.get("area", 0)
        if not self.ref.ready():
            return
        m, method, sc = self.ref.compare(q["text"], crop)
        if method == "text":
            rec.text_match = m
        elif method == "image":
            rec.sims.append(sc)
            if alt < SURVEY_ALT_M - 2:
                rec.sims_low.append(sc)
        v, why = rec.verdict()
        if v and not rec.announced:
            rec.announced = True
            log.info(f"  ★ QR #{rec.id} MATCHES the start QR ({why})")

    def tick(self, status):
        self.status = status
        a = self.d.alt()
        self.cam.overlay = f"alt {a or 0:.1f} m  {status[:60]}"
        self.process()
        now = time.time()
        xy = self.d.xy()
        if xy and dist_outside(*xy, self.field) > self.fence_m:      # hard guard
            raise GeofenceBreach(f"drone is {dist_outside(*xy, self.field):.1f} m "
                                 f"outside the field (limit {self.fence_m:.1f} m)")
        if xy and now - self.t_track > 0.5:
            self.track.append(xy)
            self.t_track = now
        if now - self.t_map > (0.3 if self.show else 3.0):
            self.t_map = now
            img = render_map(self.field, [p[:2] for p in self.path], self.track,
                             self.reg.items, xy)
            cv2.imwrite(os.path.join(DEBUG_DIR, "survey_map.png"), img)
            if self.show:
                self._show(img)
        time.sleep(0.1)

    def _show(self, mapimg):
        pv = self.cam.preview
        if pv is None:
            return
        small, qrs, s = pv
        cam = small.copy()
        for q in qrs:
            cv2.polylines(cam, [(np.asarray(q["quad"]) / s).astype(np.int32)], True,
                          (0, 255, 0) if q["text"] else (0, 200, 255), 2)
        cv2.putText(cam, self.status[:80], (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        h = cam.shape[0]
        m = cv2.resize(mapimg, (int(mapimg.shape[1] * h / mapimg.shape[0]), h))
        cv2.imshow("SQUADRONE survey", np.hstack([cam, m]))
        cv2.waitKey(1)

    # ── movement ─────────────────────────────────────────────────────────
    def wait_alt(self, x, y, alt, tol=0.7, timeout=20):
        t0 = time.time()
        while abs((self.d.alt() or 0) - alt) > tol and time.time() - t0 < timeout:
            self.d.check()
            self.d.goto_xy(x, y, alt, self.yaw)
            self.tick(f"altitude → {alt:.0f} m")

    def goto(self, target, alt, yaw=None, handle_qr=True, accept=ACCEPT_RADIUS_M):
        """Fly straight to target (field xy). Pauses for new QRs, then resumes."""
        if yaw is not None:
            self.yaw = yaw
        t0 = time.time()
        while True:
            self.d.check()
            if time.time() - t0 > WAYPOINT_TIMEOUT:
                log.warning("  Waypoint timeout — skipping")
                return False
            if handle_qr and GO_ON_MATCH and self.ref.ready():
                m = next((q for q in self.reg.items if not q.rejected and not q.tried
                          and q.confident()), None)
                if m is not None:
                    self.look_q = None
                    raise MatchFound(m)
            if handle_qr:
                q = self.reg.needs_visit()
                if q is not None:
                    self.visit_qr(q)
                    continue
            cur = self.d.xy()
            if handle_qr and self.look_q is not None and self.look_step(alt):
                continue
            if handle_qr and time.time() < self.dwell_until and self.dwell_xy:
                self.d.goto_xy(self.dwell_xy[0], self.dwell_xy[1], alt, self.yaw)
                self.tick(f"dwell {self.dwell_until - time.time():.1f} s (sharp frames)")
                continue
            dist = math.dist(cur, target)
            if dist < accept:
                return True
            self.d.goto_xy(target[0], target[1], alt, self.yaw)
            self.tick(f"→ ({target[0]:.1f},{target[1]:.1f}) {dist:.1f} m  "
                      f"alt {self.d.alt() or 0:.1f}  QRs {len(self.reg.items)}")

    def look_step(self, alt):
        """One loop step of the LOOK-OVER. True while still looking."""
        q, now = self.look_q, time.time()
        cur = self.d.xy()
        decided = q.rejected or q.confident() or q.text_match is not None
        if cur and self.look_arr is None and math.dist(cur, (q.x, q.y)) < LOOK_ARRIVE_M:
            self.look_arr = now
            self.look_n0 = len(q.sims)
            self._look_c0 = (getattr(q, "n_seen", 0), getattr(q, "n_crop", 0))
        held = self.look_arr is not None and now - self.look_arr >= QR_DWELL_S
        if decided or held or now - self.look_t0 > LOOK_MAX_S:
            sims = q.sims[self.look_n0:] if self.look_arr is not None else []
            c0 = getattr(self, "_look_c0", (0, 0))
            seen = getattr(q, "n_seen", 0) - c0[0]
            crops = getattr(q, "n_crop", 0) - c0[1]
            why = ("decided" if decided else "held" if held else
                   "timeout (never got over it)" if self.look_arr is None else "timeout")
            sc = (f"image scores over it: median {np.median(sims):.2f}, best {max(sims):.2f} "
                  f"({len(sims)})" if sims else "NO image score over it")
            log.info(f"    ◎ look at QR #{q.id} done ({why}): seen {seen}x over it, {crops} "
                     f"whole crops, {sc}, sharpness max {getattr(q, 'sharp_max', 0):.0f}"
                     + ("  ← blurry/dark: focus or light problem" if getattr(q, "sharp_max", 0) < 30
                        else "")
                     + (f"  text={q.text!r}" if q.text else ""))
            self.look_q = None
            return False
        self.d.goto_xy(q.x, q.y, alt, self.yaw)
        self.tick(f"LOOK at QR #{q.id} "
                  + (f"{now - self.look_arr:.1f}/{QR_DWELL_S:.0f} s" if self.look_arr else "→ over it"))
        return True

    def visit_qr(self, q, hover_s=0.0, alt=None, need_text=True, climb_back=True):
        """Fly over QR, centre on it (at `alt`, default survey alt), refine coord."""
        q.visits += 1
        log.info(f"\n  ▶ Visiting {q}")
        self.goto((q.x, q.y), SURVEY_ALT_M, handle_qr=False, accept=0.6)
        alt_cmd = alt or SURVEY_ALT_M
        t0, t_good = time.time(), None
        seen0 = getattr(q, "last_seen", 0)
        while time.time() - t0 < QR_VISIT_TIMEOUT:
            self.d.check()
            self.tick(f"QR #{q.id} centring  alt {self.d.alt() or 0:.1f}  "
                      f"text={q.text or '?'}")
            fresh = time.time() - getattr(q, "last_seen", 0) < 1.0
            # Drop to QR_DECODE_ALT_M if it won't decode (4 s) OR won't hold
            # steady enough to centre (8 s) — e.g. sun glare at 10 m.
            el = time.time() - t0
            if QR_DECODE_ALT_M and alt_cmd > QR_DECODE_ALT_M and ((q.text is None and el > 4)
                                              or (el > 8 and t_good is None)):
                alt_cmd = QR_DECODE_ALT_M
                why = "not decoded" if q.text is None else "can't hold it in view"
                log.info(f"    {why} from {SURVEY_ALT_M:.0f} m → descending to "
                         f"{QR_DECODE_ALT_M:.0f} m")
            self.d.goto_xy(q.x, q.y, alt_cmd, self.yaw)
            cur = self.d.xy()
            centred = (fresh and math.dist(cur, (q.x, q.y)) < QR_CENTRE_TOL_M
                       and getattr(q, "last_off", 9) < QR_CENTRE_TOL_M * 1.5
                       and abs((self.d.alt() or 0) - alt_cmd) < 0.8)
            if centred and (q.text or not need_text):
                t_good = t_good or time.time()
                if time.time() - t_good >= QR_HOLD_S:
                    q.aligned = True
                    lat, lon = xy_to_latlon(q.x, q.y, *self.origin)
                    log.info(f"    ✓ Centred over QR #{q.id} '{q.text}' at "
                             f"{self.d.alt() or 0:.1f} m  {lat:.7f}, {lon:.7f}")
                    break
            else:
                t_good = None
        else:
            if getattr(q, "last_seen", 0) <= seen0:
                log.info("    ✗ Never seen again — false detection, ignoring")
                q.rejected = True
            else:
                log.warning("    Could not centre / decode in time")
        t_h = time.time()
        while time.time() - t_h < hover_s:
            self.d.check()
            self.d.goto_xy(q.x, q.y, alt_cmd, self.yaw)
            self.tick(f"HOVER over QR '{q.text}' {time.time() - t_h:.0f}/{hover_s:.0f} s")
        if climb_back and alt_cmd != SURVEY_ALT_M:
            self.wait_alt(q.x, q.y, SURVEY_ALT_M)
        return q.aligned

    # ── start QR ─────────────────────────────────────────────────────────
    def capture_reference(self, home, fallback=True, timeout=None):
        """Hover at REF_ALT_M over the takeoff spot and store the start QR."""
        self.phase("read start QR (5 m)")
        log.info(f"\n══ READING START QR at {REF_ALT_M:.0f} m ══")
        seen = []
        t0 = time.time()
        while time.time() - t0 < (timeout or REF_TIMEOUT_S) and len(seen) < REF_SIGHTINGS + 2:
            self.d.check()
            self.d.goto_xy(home[0], home[1], REF_ALT_M, self.yaw)
            for o in self.cam.drain():
                p = self.d.pose_at(o["t"])
                if p is None or p[3] < REF_ALT_M - 1.5 or p[5] > MAX_TILT_DEG:
                    continue
                cands = [q for q in o["qrs"] if q.get("crop") is not None]
                if not cands:
                    continue
                cx, cy = o["w"] / 2, o["h"] / 2      # the one nearest the centre
                q = min(cands, key=lambda d: math.dist(d["c"], (cx, cy)))
                e, n = pixel_to_ground(q["c"][0], q["c"][1], o["w"], o["h"], p[3], p[4])
                sharp = cv2.Laplacian(q["crop"], cv2.CV_64F).var()
                seen.append((q["text"], q["crop"], sharp, (p[1] + e, p[2] + n)))
            time.sleep(0.1)
        if len(seen) < REF_SIGHTINGS:
            if not fallback:
                log.warning(f"  ✗ start tag not seen here ({len(seen)} sightings)")
                return False
            log.warning(f"  ✗ Start QR not seen ({len(seen)} sightings) — trying the "
                        "saved reference")
            if self.ref.load():
                log.info(f"  Using saved reference: {self.ref.describe()}")
            else:
                log.warning("  No reference at all — survey will only LOG QRs")
            return self.ref.ready()
        texts = [t for t, *_ in seen if t]
        self.ref.text = max(set(texts), key=texts.count) if texts else None
        self.ref.crop = max(seen, key=lambda s: s[2])[1]       # sharpest look
        xs = [s[3] for s in seen]
        self.ref.xy = (float(np.median([v[0] for v in xs])), float(np.median([v[1] for v in xs])))
        self.ref.save()
        sims = [image_similarity(self.ref.crop, s[1]) for s in seen]
        log.info(f"  ✓ Start QR stored: {self.ref.describe()}  "
                 f"({len(seen)} looks, self-similarity {np.median(sims):.2f})")
        return True

    # ── delivery ─────────────────────────────────────────────────────────
    def ask_holding(self, xy, alt, prompt):
        """Ask the operator while a background thread keeps the drone on the
        spot (input() blocks the main loop)."""
        stop = threading.Event()

        def keep():
            while not stop.is_set():
                if self.d.mode() == "GUIDED":
                    self.d.goto_xy(xy[0], xy[1], alt, self.yaw)
                time.sleep(0.5)
        threading.Thread(target=keep, daemon=True).start()
        try:
            ans = input(prompt).strip().lower()
        finally:
            stop.set()
        self.d.check()
        return ans

    def hold(self, xy, alt, seconds, label):
        """Stay on the spot for `seconds`, logging alt + drift every 2 s."""
        t0, t_log, drift = time.time(), 0, []
        while time.time() - t0 < seconds:
            self.d.check()
            self.d.goto_xy(xy[0], xy[1], alt, self.yaw)
            p = self.d.xy()
            if p:
                drift.append(math.dist(p, xy))
            if time.time() - t_log >= 2:
                log.info(f"    {label} {time.time() - t0:4.0f}/{seconds:.0f} s   "
                         f"alt {self.d.alt() or 0:.1f} m   off-spot {drift[-1] if drift else 0:.1f} m")
                t_log = time.time()
            time.sleep(0.2)
        return max(drift) if drift else 0.0

    def drop_payload(self, q):
        """Unlatch, then unwind the winch for UNWIND_S while holding over the QR."""
        xy, alt = (q.x, q.y), DELIVERY_ALT_M
        self.phase("payload: unlatch")
        log.info(f"\n══ PAYLOAD over QR #{q.id} at {self.d.alt() or 0:.1f} m ══")
        log.info(f"  UNLATCH: SERVO{LATCH_SERVO} = {LATCH_OPEN_PWM}")
        self.payload_log.append(("latch open", self.d.servo(LATCH_SERVO, LATCH_OPEN_PWM, "latch OPEN")))
        self.hold(xy, alt, LATCH_WAIT_S, "latch opening")
        self.phase(f"payload: unwind {UNWIND_S:.0f} s")
        log.info(f"  UNWIND: SERVO{WINCH_SERVO} = {WINCH_UNWIND_PWM} for {UNWIND_S:.0f} s")
        self.payload_log.append(("winch unwind", self.d.servo(WINCH_SERVO, WINCH_UNWIND_PWM, "UNWIND")))
        self.winch_running = True
        try:
            worst = self.hold(xy, alt, UNWIND_S, "unwinding")
        finally:
            self.payload_log.append(("winch stop", self.d.servo(WINCH_SERVO, WINCH_STOP_PWM, "winch STOP")))
            self.winch_running = False
        log.info(f"  ✓ Payload lowered (max drift while unwinding {worst:.1f} m)")
        if REWIND_BEFORE_RTL:
            self.phase(f"payload: rewind {UNWIND_S:.0f} s")
            self.payload_log.append(("winch rewind", self.d.servo(WINCH_SERVO, WINCH_REWIND_PWM, "REWIND")))
            self.winch_running = True
            try:
                self.hold(xy, alt, UNWIND_S, "rewinding")
            finally:
                self.payload_log.append(("winch stop", self.d.servo(WINCH_SERVO, WINCH_STOP_PWM, "winch STOP")))
                self.winch_running = False
        self.payload_done = True

    def hold_over(self, q, alt, secs):
        """Fly over QR q at alt and hold `secs` once there. Returns the number of
        sightings of q while holding (0 = not seen again)."""
        t0 = time.time()
        while time.time() - t0 < 25:
            self.d.check()
            cur = self.d.xy()
            if cur and math.dist(cur, (q.x, q.y)) < LOOK_ARRIVE_M and \
                    abs((self.d.alt() or 0) - alt) < 0.5:
                break
            self.d.goto_xy(q.x, q.y, alt, self.yaw)
            self.tick(f"RE-CHECK → QR #{q.id} at {alt:.0f} m")
        seen0, t_h = getattr(q, "n_seen", 0), time.time()
        crop0 = getattr(q, "n_crop", 0)
        while time.time() - t_h < secs:
            self.d.check()
            self.d.goto_xy(q.x, q.y, alt, self.yaw)
            self.tick(f"RE-CHECK QR #{q.id} at {alt:.0f} m  {time.time() - t_h:.1f}/{secs:.0f} s")
        return getattr(q, "n_seen", 0) - seen0, getattr(q, "n_crop", 0) - crop0

    def recheck(self):
        """Second chance for every unconfirmed QR. True = confirmed (+ payload)."""
        left = [q for q in self.reg.items
                if not (q.rejected or q.op_rejected or q.tried or q.text_match is False)]
        if not RECHECK or not left or not self.ref.ready():
            return False
        left.sort(key=lambda q: q.score(), reverse=True)
        self.phase("re-check QRs")
        log.info(f"\n══ RE-CHECK: no match confirmed — visiting {len(left)} QR(s) again "
                 f"(best first): " + ", ".join(f"#{q.id} {q.verdict()[1]}" for q in left) + " ══")
        self.no_dwell = True
        t_all = time.time()
        for q in left:
            if time.time() - t_all > RECHECK_MAX_S:
                log.warning(f"  RE-CHECK: {RECHECK_MAX_S} s used up — stopping")
                break
            res, best_new = [], None
            for alt in (SURVEY_ALT_M, RECHECK_LOW_ALT_M):
                n0 = len(q.sims)
                seen, crops = self.hold_over(q, alt, RECHECK_HOLD_S)
                new = q.sims[n0:]
                med = float(np.median(new)) if new else None
                best_new = new or best_new           # closest look that gave scores
                res.append(f"{alt:.0f} m: seen {seen}x, {crops} whole crops, "
                           + (f"score median {med:.2f} best {max(new):.2f}" if new else "NO score"))
                if q.text_match or (med is not None and med >= CONFIDENT_IMAGE):
                    break                      # clear already, no need to go lower
            # decide on the closest look that produced scores
            score = float(np.median(best_new)) if best_new else None
            match = bool(q.text_match) or (score is not None and score >= IMAGE_MATCH_THRESHOLD)
            log.info(f"  ◎ RE-CHECK QR #{q.id}: " + " | ".join(res)
                     + f" | sharpness max {getattr(q, 'sharp_max', 0):.0f}"
                     + (f" → {'MATCH?' if match else 'no match'}"))
            if not match and getattr(q, "sharp_max", 0) < 30:
                log.warning(f"    QR #{q.id} never sharp (< 30) — camera focus/light, "
                            "not proof that it doesn't match")
            self.match_report(highlight=q.id)
            if match:
                if self.deliver([q], mid_survey=True, no_text="next QR"):
                    return True
            self.wait_alt(q.x, q.y, SURVEY_ALT_M)
        log.warning("  RE-CHECK finished — no QR confirmed")
        return False

    def deliver(self, cands=None, mid_survey=False, no_text=None):
        """Go to the matching QR(s), descend to DELIVERY_ALT_M, centre, then ASK
        the operator to confirm. Returns True when confirmed (caller RTLs).
        mid_survey: a single match found during the survey; 'no' = resume."""
        self.no_dwell = True
        if cands is None:
            cands = [q for q in self.reg.items
                     if not q.rejected and not q.op_rejected and q.verdict()[0]]
            cands.sort(key=lambda q: q.score(), reverse=True)
        if not cands:
            log.warning("  ✗ No QR matched the start QR — RTL")
            return False
        log.info(f"  Candidates (best first): " + ", ".join(
            f"#{q.id} {q.verdict()[1]}" for q in cands))
        for i, q in enumerate(cands):
            self.phase(f"descend to match #{q.id}")
            log.info(f"\n══ MATCH: QR #{q.id} ({q.verdict()[1]}) — descending to "
                     f"{DELIVERY_ALT_M:.0f} m ══")
            q.aligned, q.visits = False, 0
            n_low = len(q.sims_low)
            ok = self.visit_qr(q, alt=DELIVERY_ALT_M, need_text=False, climb_back=False)
            low = q.sims_low[n_low:]
            close = f"{np.median(low):.2f}" if low else "n/a"
            if low and float(np.median(low)) < IMAGE_MATCH_THRESHOLD:
                log.warning(f"    Close-up image score {close} is below the threshold "
                            f"{IMAGE_MATCH_THRESHOLD} — look at the match report before confirming")
            log.info(f"    {'✓ Centred' if ok else '⚠ NOT centred'} over QR #{q.id} at "
                     f"{self.d.alt() or 0:.1f} m | survey score {q.verdict()[1]} | "
                     f"close-up {close} | offset {getattr(q, 'last_off', 0):.2f} m")
            self.match_report(highlight=q.id)
            q.tried = True
            more = i + 1 < len(cands)
            self.phase(f"operator confirm #{q.id}" if not AUTO_DROP else f"auto decision #{q.id}")
            while not AUTO_DROP:
                ans = self.ask_holding(
                    (q.x, q.y), DELIVERY_ALT_M,
                    f"\n>>> Over QR #{q.id} at {self.d.alt() or 0:.1f} m "
                    f"(score {q.verdict()[1]}, close-up {close}). "
                    f"CONFIRM this is the match? 'yes' = "
                    + (f"DROP PAYLOAD (unlatch + unwind {UNWIND_S:.0f} s) then RTL"
                       if PAYLOAD else "confirm + RTL")
                    + (f", 'no' = {no_text}" if no_text else
                       ", 'no' = resume the survey" if mid_survey else
                       ", 'no' = try next candidate" if more else
                       ", 'no' = re-check the other QRs" if RECHECK else ", 'no' = RTL")
                    + ": ")
                if ans in ("yes", "no"):
                    break
                print("  type yes or no")
            if AUTO_DROP:
                # NO operator: drop only when SURE — same AprilTag ID as the start
                # tag, or (no ID read) a confident image match in the close-up
                low_med = float(np.median(low)) if low else None
                sure = bool(q.text_match) or (low_med is not None and low_med >= CONFIDENT_IMAGE)
                why = ("same ID as the start tag" if q.text_match else
                       f"close-up image score {close} (need {CONFIDENT_IMAGE})")
                log.info(f"  AUTO (no operator): QR #{q.id} {'SURE' if sure else 'NOT sure'} — {why}")
                ans = "yes" if sure else "no"
            if ans == "yes":
                log.info(f"  ✓ {'AUTO-' if AUTO_DROP else 'Operator '}CONFIRMED QR #{q.id} as the match")
                self.confirmed = q
                if PAYLOAD:
                    self.drop_payload(q)
                return True
            log.info(f"  ✗ Operator rejected QR #{q.id}")
            q.op_rejected = True
            if more or mid_survey:
                self.wait_alt(q.x, q.y, SURVEY_ALT_M)
        if mid_survey:
            self.no_dwell = False
            return False
        log.warning("  No candidate confirmed — RTL")
        return False

    # ── analytics ────────────────────────────────────────────────────────
    def phase(self, name):
        """Mark the start of a mission phase (timing summary at the end)."""
        now = time.time()
        if self.phases:
            self.phases[-1][2] = now
        self.phases.append([name, now, None])

    def phase_summary(self):
        if not self.phases:
            return
        self.phases[-1][2] = self.phases[-1][2] or time.time()
        log.info("\n══ TIMING ══")
        for name, t0, t1 in self.phases:
            log.info(f"  {name:28s} {t1 - t0:6.1f} s")
        log.info(f"  {'TOTAL':28s} {self.phases[-1][2] - self.phases[0][1]:6.1f} s")

    def match_report(self, highlight=None):
        """One image: start QR + every field QR crop with its score and verdict.
        ~/squadrone/match_report.png — open it to see WHY something matched."""
        tiles = []
        def tile(img, lines, col):
            t = np.full((CROP_SIZE + 64, CROP_SIZE + 10, 3), 30, np.uint8)
            if img is not None:
                t[5:5 + CROP_SIZE, 5:5 + CROP_SIZE] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                cv2.putText(t, "no image", (55, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
            cv2.rectangle(t, (2, 2), (t.shape[1] - 3, CROP_SIZE + 8), col, 3)
            for k, ln in enumerate(lines):
                cv2.putText(t, ln[:26], (6, CROP_SIZE + 26 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, col, 1, cv2.LINE_AA)
            return t
        tiles.append(tile(self.ref.crop, ["START QR (reference)",
                                          f"text={self.ref.text!r}"[:26]], (255, 255, 255)))
        for q in self.reg.items:
            if q.rejected:
                continue
            v, why = q.verdict()
            col = (0, 220, 0) if v else ((0, 0, 230) if v is False else (0, 200, 255))
            if q.id == highlight:
                col = (255, 200, 0)
            tag = "MATCH" if v else ("no match" if v is False else "undecided")
            tiles.append(tile(q.best_crop, [f"QR #{q.id}  {tag}", f"{why}  ({q.n} looks)"], col))
        cols = 4
        rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
        blank = np.full_like(tiles[0], 30)
        img = np.vstack([np.hstack(r + [blank] * (cols - len(r))) for r in rows])
        path = os.path.join(DEBUG_DIR, "match_report.png")
        cv2.imwrite(path, img)
        log.info(f"  Match report: {path}")

    # ── main flow ────────────────────────────────────────────────────────
    def run(self, target_text=None):
        self.phase("survey (10 m lawnmower)")
        prev = self.d.xy()
        i = 0
        while i < len(self.path):
            x, y, lab = self.path[i]
            i += 1
            prev = self.d.xy() or prev
            yaw = bearing(prev, (x, y)) if math.dist(prev, (x, y)) > 1 else self.yaw
            log.info(f"\nWP {i}/{len(self.path)} [{lab}]  ({x:.1f}, {y:.1f})  "
                     f"heading {compass(yaw)}")
            try:
                self.goto((x, y), SURVEY_ALT_M, yaw=yaw)
            except MatchFound as mf:
                q = mf.q
                v, why = q.verdict()
                log.info(f"\n★★ CONFIDENT MATCH during the survey: QR #{q.id} ({why}, "
                         f"{q.n} looks) — interrupting the survey at WP {i} to go there")
                self.save()
                self.match_report(highlight=q.id)
                try:
                    if self.deliver([q], mid_survey=True):
                        return               # confirmed (+ payload) -> RTL
                finally:
                    if self.confirmed:
                        self.phase("RTL")
                        self.phase_summary()
                log.info(f"  Resuming the survey: WP {i}/{len(self.path)} [{lab}]")
                self.phase("survey (resumed)")
                i -= 1                       # fly to the same waypoint again
                continue
            if i == 1 and self.fence_m > GEOFENCE_MARGIN_M:
                # Reached A: the trip from takeoff is over — tighten the guard.
                self.fence_m = GEOFENCE_MARGIN_M
                log.info(f"  At A — geofence guard now {GEOFENCE_MARGIN_M:.0f} m "
                         "outside the field")
        while self.reg.needs_visit():
            self.visit_qr(self.reg.needs_visit())

        log.info("\n══ SURVEY COMPLETE ══")
        log.info(f"  Start QR: {self.ref.describe() if self.ref.ready() else 'NONE'}")
        for q in self.reg.items:
            if not q.rejected:
                lat, lon = xy_to_latlon(q.x, q.y, *self.origin)
                v, why = q.verdict()
                tag = "MATCH" if v else ("no match" if v is False else "undecided")
                log.info(f"  QR #{q.id}: text={q.text!r:<28} {lat:.7f}, {lon:.7f}  "
                         f"{tag:9s} ({why}, {q.n} looks)")
        self.save()
        self.match_report()

        if target_text is None and self.ref.ready():
            if not any(q.verdict()[0] and not q.op_rejected and not q.rejected
                       for q in self.reg.items):
                log.warning("  No (remaining) QR matches the start QR — RTL")
            try:
                if not self.deliver():
                    self.recheck()
            finally:
                self.phase("RTL")
                self.phase_summary()
            return
        if target_text:
            q = self.reg.find(target_text)
            if q is None:
                log.warning(f"  Target QR '{target_text}' not found")
            else:
                log.info(f"\n══ TARGET '{q.text}' — going there ══")
                q.aligned, q.visits = False, 0
                self.visit_qr(q, hover_s=HOVER_S)

    def save(self):
        cv2.imwrite(os.path.join(DEBUG_DIR, "survey_map.png"),      # map first
                    render_map(self.field, [p[:2] for p in self.path], self.track,
                               self.reg.items, self.d.xy()))
        res = {"qrs": []}
        for q in self.reg.items:
            if q.rejected:
                continue
            lat, lon = xy_to_latlon(q.x, q.y, *self.origin)
            res["qrs"].append({"id": int(q.id), "text": q.text,
                               "lat": round(float(lat), 7), "lon": round(float(lon), 7),
                               "x": round(float(q.x), 2), "y": round(float(q.y), 2),
                               "sightings": int(q.n), "centred": bool(q.aligned),
                               "match": q.verdict()[0], "match_reason": q.verdict()[1],
                               "image_scores": [round(float(v), 3) for v in q.sims]})
            if q.best_crop is not None:
                os.makedirs(CROP_DIR, exist_ok=True)
                cv2.imwrite(os.path.join(CROP_DIR, f"qr{q.id}.png"), q.best_crop)
        try:
            with open(os.path.join(DEBUG_DIR, "survey_results.json"), "w") as f:
                json.dump(res, f, indent=2)
        except Exception as e:                      # never let saving stop the flight
            log.error(f"  could not save results: {e}")
        log.info(f"  Saved {DEBUG_DIR}/survey_results.json + survey_map.png")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def load_corners():
    """Use drone-recorded corners if the file exists (all four required)."""
    global POINT_A, POINT_B, POINT_C, POINT_D, VERTICES
    if not os.path.isfile(CORNERS_FILE):
        log.info("  Corners: built-in values (no field_corners.json)")
        return
    with open(CORNERS_FILE) as f:
        c = json.load(f)
    if "A" in c and math.dist((0, 0), latlon_to_xy(*c["A"], *POINT_A)) > 500:
        log.warning(f"  ⚠ {CORNERS_FILE} holds corners of ANOTHER site (A is "
                    f"{math.dist((0, 0), latlon_to_xy(*c['A'], *POINT_A)) / 1000:.0f} km away) — "
                    "IGNORED, using the built-in venue corners. Delete that file.")
        return
    missing = [k for k in "ABCD" if k not in c]
    if missing:
        raise SystemExit(f"field_corners.json is missing {', '.join(missing)} — record "
                         f"them with --record, or delete {CORNERS_FILE}")
    POINT_A, POINT_B, POINT_C, POINT_D = (tuple(c[k]) for k in "ABCD")
    VERTICES = [POINT_A, POINT_B, POINT_C, POINT_D]
    log.info(f"  Corners: from {CORNERS_FILE} (recorded by the drone)")


def record_corner(label, conn, baud):
    """Stand the drone ON the corner, then run this. Averages 10 s of GPS."""
    drone = Drone(conn, baud, POINT_A)
    drone.wait_gps()
    log.info(f"Recording corner {label}: keep the drone still for 10 s…")
    pts, t0, last = [], time.time(), None
    while time.time() - t0 < 10:
        m = drone.get("GLOBAL_POSITION_INT")
        if m is not None and m is not last:
            pts.append((m.lat / 1e7, m.lon / 1e7))
            last = m
        time.sleep(0.1)
    drone.close()
    if len(pts) < 10:
        raise SystemExit("Too few GPS samples — check telemetry")
    lat = float(np.median([p[0] for p in pts]))
    lon = float(np.median([p[1] for p in pts]))
    spread = max(math.dist((0, 0), latlon_to_xy(p[0], p[1], lat, lon)) for p in pts)
    g = drone.get("GPS_RAW_INT")
    c = {}
    if os.path.isfile(CORNERS_FILE):
        with open(CORNERS_FILE) as f:
            c = json.load(f)
    c[label] = [round(lat, 7), round(lon, 7)]
    with open(CORNERS_FILE, "w") as f:
        json.dump(c, f, indent=2)
    log.info(f"  ✓ Corner {label} = {lat:.7f}, {lon:.7f}   ({len(pts)} samples, "
             f"wander {spread:.1f} m, sats {g.satellites_visible if g else '?'}, "
             f"hdop {g.eph / 100 if g else 0:.2f})")
    if spread > 2.0:
        log.warning("  GPS wandered > 2 m while recording — wait for better GPS and redo")
    order = [k for k in "ABCD" if k in c]
    for k1, k2 in zip(order, order[1:]):
        x, y = latlon_to_xy(*c[k2], *c[k1])
        log.info(f"    {k1} → {k2}: {math.hypot(x, y):.1f} m, heading "
                 f"{compass(math.atan2(x, y))}")
    left = [k for k in "ABCD" if k not in c]
    log.info(f"  Saved to {CORNERS_FILE}. " +
             (f"Still to record: {', '.join(left)}" if left else "All four corners recorded."))


def plan(origin):
    field = [latlon_to_xy(la, lo, *origin) for la, lo in VERTICES]
    A, B, C, D = inset_quad(field, INSET_M)
    return field, build_lawnmower(A, B, C, D, SPACING_M)


def focus_sweep(dist):
    """Point the camera at a textured target `dist` m away (a QR on a wall, a
    building, the far side of the field). Steps the lens through its range,
    measures sharpness, saves the best value to focus.json + a plot."""
    lens = find_lens()
    if lens is None:
        log.error("Lens motor not found. Run:  ls /dev/v4l-subdev*   and   "
                  "for d in /dev/v4l-subdev*; do v4l2-ctl -d $d -l; done   — send me the output")
        return
    log.info(f"Lens motor: {lens['dev']}  focus_absolute range {lens['min']}..{lens['max']} "
             f"(now {lens['value']})")
    cam = SurveyCam()
    cam.scan_enabled = False                  # sharpness only — fast
    cam.open()
    cam.start()
    time.sleep(1.0)

    def measure(v):
        set_lens(lens, v)
        time.sleep(0.35)                      # lens settles, flush old frames
        vals, t0 = [], time.time()
        while len(vals) < 3 and time.time() - t0 < 2:
            vals.append(cam.sharp)
            time.sleep(0.12)
        return float(np.median(vals))

    results = {}
    try:
        lo, hi = lens["min"], lens["max"]
        coarse = np.linspace(lo, hi, 25).astype(int)
        log.info(f"Coarse sweep ({len(coarse)} steps), keep the camera STILL…")
        for v in coarse:
            results[int(v)] = measure(v)
            log.info(f"  focus {v:5d}  sharpness {results[int(v)]:8.1f}  "
                     + "█" * int(40 * results[int(v)] / max(max(results.values()), 1)))
        best = max(results, key=results.get)
        step = max(lens["step"], int((hi - lo) / 24))
        fine = np.linspace(max(lo, best - step), min(hi, best + step), 13).astype(int)
        log.info(f"Fine sweep around {best}…")
        for v in fine:
            results[int(v)] = measure(v)
        best = max(results, key=results.get)
        set_lens(lens, best)
    finally:
        cam.stop()
    ks = sorted(results)
    peak, low = results[best], min(results.values())
    log.info("\n══ FOCUS SWEEP RESULT ══")
    for v in ks:
        log.info(f"  focus {v:5d}  sharpness {results[v]:8.1f}  " +
                 "█" * int(40 * results[v] / max(peak, 1)) + ("   ← BEST" if v == best else ""))
    log.info(f"  Best focus_absolute = {best}  (sharpness {peak:.0f}, worst {low:.0f}, "
             f"contrast ×{peak / max(low, 1):.1f})")
    if peak < 1.5 * max(low, 1):
        log.warning("  The curve is FLAT — the lens may not be moving, or the scene has no "
                    "detail. Point at something textured and retry.")
    # plot
    W, H = 900, 360
    img = np.full((H, W, 3), 30, np.uint8)
    xs = lambda v: int(40 + (v - ks[0]) / max(1, ks[-1] - ks[0]) * (W - 80))
    ys = lambda sv: int(H - 40 - sv / max(peak, 1) * (H - 80))
    pts = np.array([(xs(v), ys(results[v])) for v in ks], np.int32)
    cv2.polylines(img, [pts], False, (0, 220, 255), 2)
    cv2.line(img, (xs(best), 20), (xs(best), H - 40), (0, 255, 0), 1)
    cv2.putText(img, f"best {best}", (xs(best) + 5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(img, f"focus sweep at {dist} m  (x: focus_absolute {ks[0]}..{ks[-1]}, y: sharpness)",
                (40, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.imwrite(os.path.join(DEBUG_DIR, "focus_sweep.png"), img)
    with open(FOCUS_FILE, "w") as f:
        json.dump({"focus_absolute": int(best), "distance_m": dist, "device": lens["dev"],
                   "sharpness": round(peak, 1), "date": time.strftime("%Y-%m-%d %H:%M"),
                   "curve": {str(k): round(results[k], 1) for k in ks}}, f, indent=2)
    log.info(f"  Saved {FOCUS_FILE} + {DEBUG_DIR}/focus_sweep.png — every camera start now "
             f"uses focus_absolute={best}")


def bench(dist=None, save_ref=False):
    """dist = distance camera->QR in metres (for FOV measurement + direction check).
    save_ref: store the QR in view as the start-QR reference.
    If a reference exists, every QR in view gets a MATCH / no-match score."""
    cam = SurveyCam()
    cam.open()
    cam.start()
    log.info("BENCH — no flight. Ctrl+C to stop.")
    ref = Reference()
    ref_looks = []
    if save_ref:
        log.info("  SAVE-REF: hold the START QR in the middle of the view…")
    elif ref.load():
        log.info(f"  Reference loaded ({ref.describe()}) — showing match scores. "
                 f"Threshold {IMAGE_MATCH_THRESHOLD}")
    if dist:
        log.info(f"  FOV mode: QR ({QR_SIZE_M} m) at {dist} m from the lens")
    show = bool(os.environ.get("DISPLAY"))
    t_log = 0
    fovs = []
    t_why = 0
    try:
        while True:
            obs = cam.drain()
            pv = cam.preview
            if obs and time.time() - t_log > 0.5:
                o = obs[-1]
                qs = [f"{q['text']} [{q.get('by', '?')}]" if q["text"]
                      else "(QR shape, not decoded)" for q in o["qrs"]]
                log.info(f"fps {cam.fps:.1f}  scan {cam.scan_ms:.0f} ms  QRs {qs}")
                part = [q for q in o["qrs"] if q.get("quad") is not None and not q.get("full", True)]
                if part:
                    log.info(f"    {len(part)} QR(s) touch the frame edge — not used for matching; "
                             "hold the WHOLE QR inside the view")
                if not o["qrs"] and time.time() - t_why > 3 and cam.last_frame is not None:
                    t_why = time.time()
                    why = []
                    find_qr_squares(cam.last_frame, why)
                    log.info(f"    why no QR: sharpness {cam.sharp:.0f} (blurry < 30?), "
                             f"{len(why)} contrast blob(s): " + ("; ".join(why[:3]) or "none"))
                crops = [q for q in o["qrs"] if q.get("crop") is not None]
                if save_ref and crops:
                    q = min(crops, key=lambda d: math.dist(d["c"], (o["w"] / 2, o["h"] / 2)))
                    ref_looks.append((q["text"], q["crop"],
                                      cv2.Laplacian(q["crop"], cv2.CV_64F).var()))
                    if len(ref_looks) >= REF_SIGHTINGS + 2:
                        texts = [t for t, _, _ in ref_looks if t]
                        ref.text = max(set(texts), key=texts.count) if texts else None
                        ref.crop = max(ref_looks, key=lambda r: r[2])[1]
                        ref.save()
                        log.info(f"  ✓ Reference saved: {ref.describe()} → {REF_PNG}")
                        save_ref = False
                elif ref.ready():
                    for q in crops:
                        m, meth, sc = ref.compare(q["text"], q["crop"])
                        log.info(f"    vs start QR: {'MATCH' if m else 'no match'} "
                                 f"({meth}{'' if sc is None else f' {sc:.2f}'})  "
                                 f"text={q['text']!r}")
                if dist:
                    for q in o["qrs"]:
                        if not q["text"]:
                            continue
                        quad = np.asarray(q["quad"], np.float32).reshape(-1, 2)
                        side_px = math.sqrt(cv2.contourArea(quad))
                        # QR spans side_px of W pixels -> real HFOV
                        hfov = 2 * math.degrees(math.atan(
                            (o["w"] / side_px) * QR_SIZE_M / 2 / dist))
                        fovs.append(hfov)
                        e, n = pixel_to_ground(q["c"][0], q["c"][1], o["w"], o["h"], dist, 0.0)
                        log.info(f"    QR side {side_px:.0f}px → camera HFOV ≈ {hfov:.1f}° "
                                 f"(median {np.median(fovs):.1f}°, script uses {HFOV_DEG}°)  "
                                 f"| QR is fwd={n - CAM_OFFSET_FWD_M:+.2f} m, rgt={e:+.2f} m")
                t_log = time.time()
            if pv is not None:
                small, qrs, s = pv
                img = small.copy()
                for q in qrs:
                    cv2.polylines(img, [(np.asarray(q["quad"]) / s).astype(np.int32)], True,
                                  (0, 255, 0) if q["text"] else (0, 200, 255), 2)
                    if q["text"]:
                        c = (int(q["c"][0] / s), int(q["c"][1] / s))
                        cv2.putText(img, q["text"][:20], c, cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imwrite(os.path.join(DEBUG_DIR, "bench_survey.jpg"), img)
                if show:
                    cv2.imshow("bench", img)
                    cv2.waitKey(1)
            time.sleep(0.1)
    finally:
        cam.stop()


def compare_files(paths):
    """--compare: score photos/crops against the start QR WITHOUT flying.
    First path = the START QR (a photo, reference_qr.png, ...), the rest =
    candidates (phone photos, ~/squadrone/qr_crops/qrN.png ...). A full photo is
    searched for the QR (same locators as in flight) and perspective-cropped."""
    def load(pth):
        img = cv2.imread(pth)
        if img is None:
            return None, "can't read"
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if max(g.shape) <= CROP_SIZE * 1.5:              # already a crop
            return cv2.equalizeHist(cv2.resize(g, (CROP_SIZE, CROP_SIZE))), "crop file"
        if MARKERS == "apriltag":
            tags = scan_tags(img) or scan_tags(img)       # 2nd call includes full res
            if tags:
                t = max(tags, key=lambda q: cv2.contourArea(q["quad"]))
                return (qr_crop(g, t["quad"]), t["text"]), f"{t['text']} read"
            return None, "NO AprilTag read in the photo"
        found = locate_qrs(img) + [q for q in find_qr_squares(img) if q.get("quad") is not None]
        best = None
        for q in found:
            c = qr_crop(g, q["quad"])
            a = cv2.contourArea(np.asarray(q["quad"], np.float32).reshape(-1, 2))
            if c is not None and (best is None or a > best[0]):
                best = (a, c, q.get("by", "?"))
        if best is None:
            return None, "NO QR found in the photo (try a closer/straighter photo)"
        return best[1], f"QR found by {best[2]}, {math.sqrt(best[0]):.0f}px side"
    ref, how = load(paths[0])
    print(f"START  {paths[0]}: {how}")
    if ref is None:
        return
    ref_text = None
    if isinstance(ref, tuple):
        ref, ref_text = ref
    os.makedirs(DEBUG_DIR, exist_ok=True)
    tiles = [ref]
    for pth in paths[1:]:
        c, how = load(pth)
        if c is None:
            print(f"  {pth}: {how}")
            continue
        if isinstance(c, tuple):
            c, txt = c
            sc = image_similarity(ref, c)
            print(f"  {pth}: {txt} → {'MATCH (same ID)' if txt == ref_text else 'no match'}"
                  f"   (image score {sc:.2f} for reference)")
            tiles.append(c)
            continue
        sc = image_similarity(ref, c)
        v = ("MATCH — would interrupt the survey" if sc >= CONFIDENT_IMAGE else
             "possible match (asks at the end)" if sc >= IMAGE_MATCH_THRESHOLD else "no match")
        print(f"  {pth}: score {sc:.2f} → {v}   ({how})")
        tiles.append(c)
    out = os.path.join(DEBUG_DIR, "compare.png")
    cv2.imwrite(out, np.hstack([cv2.resize(t, (200, 200)) for t in tiles]))
    print(f"crops side by side (start QR first): {out}")
    print(f"thresholds: >= {IMAGE_MATCH_THRESHOLD} possible, >= {CONFIDENT_IMAGE} confident")


def main():
    global SPACING_M, SURVEY_ALT_M, SPEED_MPS, INSET_M, TARGET_QR, QR_ALIGN_EACH
    global PAYLOAD, UNWIND_S, REWIND_BEFORE_RTL, GO_ON_MATCH, DECODE_QR
    global CAM_MODEL, HFOV_DEG, VFOV_DEG, CAM_OFFSET_FWD_M, FLIP_X, FLIP_Y
    global FOCUS_M, FOCUS_VALUE, CAM_RES, SCAN_TILES, EXPOSURE_EV, QR_DECODE_ALT_M
    global Picamera2, mavutil
    global MATCH_MODE, IMAGE_MATCH_THRESHOLD, REF_ALT_M, DELIVERY_ALT_M, MARKERS
    global MAX_SHUTTER_US, QR_DWELL_S
    ap = argparse.ArgumentParser(description="Squadrone lawnmower survey with QR scan")
    ap.add_argument("--conn", default=SERIAL_PORT)
    ap.add_argument("--baud", type=int, default=BAUD_RATE)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--record", choices=list("ABCD") + ["GREEN_IN", "GREEN_OUT", "BLUE_IN", "BLUE_OUT"],
                    help="stand the drone on that corner and save its GPS position")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--markers", choices=["apriltag", "qr"], default=MARKERS,
                    help="apriltag = AprilTag 36h11 IDs (competition), qr = old QR pipeline")
    ap.add_argument("--compare", nargs="+", metavar="IMG",
                    help="no flight: score images vs the FIRST one (start QR), e.g. "
                         "--compare start.jpg field1.jpg field2.jpg")
    ap.add_argument("--spacing", type=float, default=SPACING_M)
    ap.add_argument("--alt", type=float, default=SURVEY_ALT_M)
    ap.add_argument("--speed", type=float, default=SPEED_MPS)
    ap.add_argument("--inset", type=float, default=INSET_M)
    ap.add_argument("--target", default=TARGET_QR, help="QR text to finish over")
    ap.add_argument("--no-align", action="store_true", help="(default now) don't stop at each QR")
    ap.add_argument("--align-each", action="store_true", help="stop and centre over every QR")
    ap.add_argument("--ref", choices=["takeoff", "saved"], default="takeoff",
                    help="start QR: read it after takeoff (default) or use the one saved "
                         "with --bench --save-ref")
    ap.add_argument("--save-ref", action="store_true", help="bench: save the QR in view as start QR")
    ap.add_argument("--match", choices=["text", "image", "both"], default=MATCH_MODE)
    ap.add_argument("--threshold", type=float, default=IMAGE_MATCH_THRESHOLD,
                    help="image similarity needed for a match (tune with --bench)")
    ap.add_argument("--ref-alt", type=float, default=REF_ALT_M)
    ap.add_argument("--delivery-alt", type=float, default=DELIVERY_ALT_M)
    ap.add_argument("--dist", type=float, help="bench: camera-to-QR distance (m) to measure FOV")
    ap.add_argument("--models", help="folder with the WeChatQRCode model files")
    ap.add_argument("--no-wechat", action="store_true", help="use pyzbar only")
    ap.add_argument("--image-only", action="store_true",
                    help="skip reading QR data, match by image only (fast)")
    ap.add_argument("--decode", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--decode-alt", type=float, default=None,
                    help="allow descending to this altitude over a QR that won't read "
                         "(off by default: rulebook requires 10 m)")
    ap.add_argument("--ev", type=float, default=EXPOSURE_EV,
                    help="exposure compensation (-2 darker .. +2 brighter), default -1 for glare")
    ap.add_argument("--focus-alt", type=float, nargs="?", const=SURVEY_ALT_M, default=None,
                    help="lock focus at N metres (bare flag = survey alt). Default: continuous AF")
    ap.add_argument("--no-payload", action="store_true", help="confirm + RTL, no drop")
    ap.add_argument("--full-survey", action="store_true",
                    help="finish the whole survey before going to the match")
    ap.add_argument("--unwind-s", type=float, default=UNWIND_S, help="winch unwind time (s)")
    ap.add_argument("--rewind", action="store_true", help="rewind the line before RTL")
    ap.add_argument("--af", action="store_true", help="continuous autofocus (the default)")
    ap.add_argument("--res", help="camera resolution WxH, e.g. 4656x3496")
    ap.add_argument("--cam", choices=list(CAMS), default=CAM_MODEL,
                    help="down camera: imx519 (ArduCam AF) or imx500 (AI cam, fixed focus)")
    ap.add_argument("--cam-offset", type=float, help="camera position, m ahead of drone centre (- = behind)")
    ap.add_argument("--flip-x", action="store_true", help="camera left/right reversed")
    ap.add_argument("--flip-y", action="store_true", help="toggle camera fwd/back direction")
    ap.add_argument("--tiles", type=int, default=SCAN_TILES, help="tile grid N (0 = off)")
    ap.add_argument("--focus-sweep", action="store_true",
                    help="ground: find the sharpest lens position at --dist m (default 10)")
    ap.add_argument("--focus-value", type=int, help="set the lens focus_absolute directly")
    ap.add_argument("--shutter-us", type=int, default=MAX_SHUTTER_US,
                    help="max shutter time in microseconds (0 = fully automatic)")
    ap.add_argument("--dwell", type=float, default=QR_DWELL_S,
                    help="pause this many seconds when a new QR appears (0 = never)")
    a = ap.parse_args()
    MARKERS = a.markers
    if a.compare:                      # offline: no camera, no drone
        IMAGE_MATCH_THRESHOLD = a.threshold
        compare_files(a.compare)
        return
    if not (1 <= a.spacing <= 30 and 3 <= a.alt <= 30 and 0.5 <= a.speed <= 5
            and 0 <= a.inset <= 5):
        ap.error("spacing 1-30, alt 3-30, speed 0.5-5, inset 0-5")
    SPACING_M, SURVEY_ALT_M, SPEED_MPS, INSET_M = a.spacing, a.alt, a.speed, a.inset
    TARGET_QR, QR_ALIGN_EACH = a.target, a.align_each
    MATCH_MODE, IMAGE_MATCH_THRESHOLD = a.match, a.threshold
    REF_ALT_M, DELIVERY_ALT_M = a.ref_alt, a.delivery_alt
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    log.info("=" * 60)
    log.info(f"SQUADRONE QR survey  alt={SURVEY_ALT_M} spacing={SPACING_M} "
             f"speed={SPEED_MPS} inset={INSET_M} target={TARGET_QR!r} "
             f"align_each={QR_ALIGN_EACH}")
    log.info(f"Start-QR matching: ref={a.ref} mode={MATCH_MODE} "
             f"threshold={IMAGE_MATCH_THRESHOLD} → deliver at {DELIVERY_ALT_M:.0f} m")
    log.info("=" * 60)

    if a.record:
        from pymavlink import mavutil as _m
        mavutil = _m
        record_corner(a.record, a.conn, a.baud)
        return
    load_corners()
    origin = POINT_A
    field, path = plan(origin)
    length = sum(math.dist(path[i][:2], path[i + 1][:2]) for i in range(len(path) - 1))
    log.info(f"Planned {len(path)} waypoints, {length:.0f} m (~{length / SPEED_MPS:.0f} s "
             f"+ QR stops)")
    for i, (x, y, lab) in enumerate(path, 1):
        la, lo = xy_to_latlon(x, y, *origin)
        log.info(f"  WP{i:02d} [{lab:6s}] {la:.7f}, {lo:.7f}")
    cv2.imwrite(os.path.join(DEBUG_DIR, "survey_plan.png"),
                render_map(field, [p[:2] for p in path]))
    log.info(f"  Plan image: {DEBUG_DIR}/survey_plan.png")
    if a.plan_only:
        return

    from picamera2 import Picamera2 as _P
    Picamera2 = _P
    SCAN_TILES = a.tiles
    MAX_SHUTTER_US, QR_DWELL_S = a.shutter_us, a.dwell
    EXPOSURE_EV = a.ev
    QR_DECODE_ALT_M = a.decode_alt
    FOCUS_M = None if a.af else a.focus_alt
    PAYLOAD, UNWIND_S, REWIND_BEFORE_RTL = not a.no_payload, a.unwind_s, a.rewind
    GO_ON_MATCH = not a.full_survey
    CAM_MODEL = a.cam
    CAM_RES, HFOV_DEG, VFOV_DEG = CAMS[CAM_MODEL]
    if a.cam_offset is not None:
        CAM_OFFSET_FWD_M = a.cam_offset
    if a.flip_x:
        FLIP_X = not FLIP_X
    if a.flip_y:
        FLIP_Y = not FLIP_Y
    log.info(f"  Down camera: {CAM_MODEL} {CAM_RES} HFOV {HFOV_DEG}° offset {CAM_OFFSET_FWD_M:+.2f} m "
             f"flips X={FLIP_X} Y={FLIP_Y}")
    if a.res:
        CAM_RES = tuple(int(v) for v in a.res.lower().split("x"))
    DECODE_QR = not a.image_only
    log.info(f"  QR mode: {'READ (WeChat/pyzbar, normal + inverted) + IMAGE match on top' if DECODE_QR else 'IMAGE ONLY (found as squares, no reading)'}")
    if DECODE_QR and not a.no_wechat:
        load_wechat(a.models)
    FOCUS_VALUE = a.focus_value
    if a.focus_sweep:
        focus_sweep(a.dist or 10.0)
        return
    if a.bench:
        bench(a.dist, a.save_ref)
        return
    from pymavlink import mavutil as _m
    mavutil = _m

    cam = SurveyCam()
    cam.open()
    cam.start()
    drone = Drone(a.conn, a.baud, origin)
    took_over = False
    sv = None
    try:
        drone.wait_gps()
        xy = drone.xy()
        dlat, dlon = xy_to_latlon(*xy, *origin)
        g = drone.get("GPS_RAW_INT")
        log.info(f"Drone GPS position: {dlat:.7f}, {dlon:.7f}   "
                 f"(sats {g.satellites_visible if g else '?'}, "
                 f"hdop {g.eph / 100 if g else 0:.2f})")
        log.info(f"Corner A (recorded): {POINT_A[0]:.7f}, {POINT_A[1]:.7f}")
        log.info(f"Drone is {math.dist(xy, path[0][:2]):.1f} m from start point A, "
                 f"{dist_outside(*xy, field):.1f} m outside the field")
        log.info(f"FIRST MOVE after takeoff: {math.dist(xy, path[0][:2]):.1f} m towards "
                 f"{compass(bearing(xy, path[0][:2]))} — check this matches where A is!")
        log.info(f"Then pass 1: A → B towards {compass(bearing(path[0][:2], path[1][:2]))}")
        if math.dist(xy, path[0][:2]) > WARN_START_DIST_M:
            log.warning(f"  Starting {math.dist(xy, path[0][:2]):.0f} m from A — the drone "
                        "will fly straight there first. Make sure that line is clear.")
        if math.dist(xy, path[0][:2]) > MAX_START_DIST_M:
            raise RuntimeError(f"More than {MAX_START_DIST_M:.0f} m from A — "
                               "move the drone near A first")
        drone.set_param("RTL_ALT", SURVEY_ALT_M * 100)
        if PAYLOAD:
            log.info(f"Payload: on confirm -> unlatch SERVO{LATCH_SERVO}, unwind SERVO"
                     f"{WINCH_SERVO} {UNWIND_S:.0f} s, {'rewind, ' if REWIND_BEFORE_RTL else ''}RTL")
            ok1 = drone.servo(LATCH_SERVO, LATCH_CLOSED_PWM, "latch CLOSED")
            ok2 = drone.servo(WINCH_SERVO, WINCH_STOP_PWM, "winch STOP")
            if not (ok1 and ok2):
                log.warning("  Servo check FAILED — the drop may not work. Check "
                            "SERVO5/6_FUNCTION = 0 before flying (Ctrl+C to abort).")
        input("\n>>> " + ("Payload LATCHED? " if PAYLOAD else "")
              + "ENTER to arm and take off (area clear!) <<<\n")
        first_alt = REF_ALT_M if a.ref == "takeoff" else SURVEY_ALT_M
        drone.arm_and_takeoff(first_alt)
        sv = Survey(drone, cam, field, path, origin)
        home = drone.xy()
        if a.ref == "takeoff":
            sv.capture_reference(home)
        elif sv.ref.load():
            log.info(f"  Start QR from file: {sv.ref.describe()}")
        else:
            log.warning("  --ref saved but no saved reference — survey will only LOG QRs")
        sv.phase("climb to survey alt")
        sv.wait_alt(home[0], home[1], SURVEY_ALT_M)
        sv.run(TARGET_QR)
        log.info("\n✓ Done — RTL")
    except KeyboardInterrupt:
        log.warning("Ctrl+C — aborting")
    except PilotTookOver:
        took_over = True
        log.warning(f"Pilot / failsafe took over (mode {drone.mode()}) — script exits")
    except GeofenceBreach as e:
        log.error(f"!!! GEOFENCE: {e} — RTL")
    except Exception as e:
        log.error(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if PAYLOAD:
            drone.servo(WINCH_SERVO, WINCH_STOP_PWM, "winch STOP (safety)")
        if not took_over and drone.armed():
            drone.set_mode("RTL")
        try:
            if PAYLOAD and sv is not None:
                log.info("\n══ PAYLOAD ══")
                log.info(f"  delivered: {'YES over QR #%d' % sv.confirmed.id if sv.payload_done else 'NO'}")
                for what, ok in sv.payload_log:
                    log.info(f"  {what:14s} {'ACK ✓' if ok else 'NO ACK ✗'}")
        except Exception:
            pass
        cam.stop()
        drone.close()


if __name__ == "__main__":
    main()