#!/usr/bin/env python3
"""
SQUADRONE — Lawnmower Survey: QR scan + align, red-zone avoidance   (v1)
File: survey_qr_redzone.py      (built on lawnmower_trial.py)
Team: Squadrone AT2026RC60, PVGCOET Pune, SAEINDIA AeroTHON 2026

WHAT IT DOES
  1. Same field + lawnmower path as lawnmower_trial.py (A -> B ... -> D, inset).
     The drone faces its direction of travel on every pass.
  2. DOWN camera (ArduCam IMX519, rear) captures at HIGH resolution
     (2328x1748 ≈ 5.6 mm/pixel at 10 m) so QR codes decode from altitude.
  3. QR: every QR seen is put on a list with its ground coordinate
     (drone position at the moment of the frame + pixel offset in metres).
     A new QR -> pause the pass, fly over it, centre on it (descend to
     QR_DECODE_ALT only if it won't decode from survey height), record the
     refined coordinate + text, climb back, resume the pass.
  4. RED ZONES: red seen by the camera is projected onto a grid map of the
     field (+ safety margin). Every move is planned with A* on that map, so
     the drone goes AROUND red areas; passes are re-planned the moment new
     red appears. Known red zones can be entered before flight
     (RED_ZONES_LATLON) and are avoided from the start.
  5. End: summary + ~/squadrone/survey_results.json + survey_map.png.
     If TARGET_QR is set and found -> fly to it, centre, hover HOVER_S.
     Then RTL.

RUN
  python3 survey_qr_redzone.py --plan-only          # path + map, no drone
  python3 survey_qr_redzone.py --bench              # camera + QR/red detection only
  python3 survey_qr_redzone.py --alt 5 --spacing 8  # trial
  python3 survey_qr_redzone.py --alt 10 --spacing 8 --target "ZONE-3"
  python3 survey_qr_redzone.py --no-align           # just log QRs, don't stop

INSTALL (once):  sudo apt install python3-pyzbar libzbar0
                 (pyzbar decodes QRs better + faster; OpenCV is the fallback)
"""

import os
import sys
import json
import math
import time
import heapq
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

# Field — keep A, B, C, D order (recorded 17-10-2026)
POINT_A = (18.4898160, 73.8516559)   # start (SE)
POINT_B = (18.4898089, 73.8513138)   # end of first pass (SW)
POINT_C = (18.4899799, 73.8512960)   # NW
POINT_D = (18.4900219, 73.8516538)   # finish (NE)
VERTICES = [POINT_A, POINT_B, POINT_C, POINT_D]
INSET_M = 2.0

# Known red zones (optional) — list of polygons, each a list of (lat, lon)
RED_ZONES_LATLON = []

SURVEY_ALT_M    = 10.0
SPACING_M       = 8.0     # 10 m alt sees ~13 m across -> 8 m spacing overlaps
SPEED_MPS       = 1.5
ACCEPT_RADIUS_M = 1.0
VIA_RADIUS_M    = 0.8     # for detour points
WAYPOINT_TIMEOUT = 120

# QR
QR_ALIGN_EACH     = True   # stop and centre over every QR (False = just log)
QR_DECODE_ALT_M   = 5.0    # descend to this if it won't decode from survey alt
QR_MERGE_M        = 2.0    # sightings closer than this = same QR
QR_VISIT_TIMEOUT  = 30
QR_CENTRE_TOL_M   = 0.30
QR_HOLD_S         = 1.0
TARGET_QR         = None   # text (or part of it) of the QR to finish over
HOVER_S           = 10
QR_SIZE_M         = 1.0    # real QR size (rulebook: 1 m x 1 m)
QR_SIZE_TOL       = 2.5    # accept 1/2.5 .. 2.5 x the expected size in pixels

# Red zone (team reading of the rules: > 10 s over red = -5 points)
RED_PENALTY_S  = 10.0
RED_MAX_S      = 5.0       # never stay over red longer than this (half the limit)
RED_MARGIN_M   = 1.5       # normal planning keeps this far from any red
RED_MIN_PX     = 150       # red pixels (in the 640x480 analysis image) to count
LAB_A_MIN_RED  = 150       # Lab a* above this = clearly red (128 = neutral)
MAP_RES_M      = 0.5

# Camera (DOWN, ArduCam IMX519, rear, looking straight down)
CAM_MODEL        = "imx519"
CAM_RES          = (2328, 1748)   # full field of view, 2x2 binned
ANA_RES          = (640, 480)     # analysis size for red detection
HFOV_DEG         = 66.0
VFOV_DEG         = 51.0
CAM_OFFSET_FWD_M = -0.10          # camera behind drone centre
FLIP_X, FLIP_Y   = False, False   # image right = drone right, bottom = nose
MAX_TILT_DEG     = 10.0           # skip geometry while tilted more than this

SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE   = 921600
EARTH_R     = 6_378_137.0

DEBUG_DIR = os.path.expanduser("~/squadrone")
os.makedirs(DEBUG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
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


def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside

# ═════════════════════════════════════════════════════════════════════════════
# GRID MAP + A*  (red zones -> blocked cells, plan around them)
# ═════════════════════════════════════════════════════════════════════════════

class GridMap:
    def __init__(self, field_poly, res=MAP_RES_M, margin=RED_MARGIN_M):
        self.poly, self.res = field_poly, res
        xs, ys = [p[0] for p in field_poly], [p[1] for p in field_poly]
        self.x0, self.y0 = min(xs) - 1, min(ys) - 1
        self.nx = int(math.ceil((max(xs) + 1 - self.x0) / res))
        self.ny = int(math.ceil((max(ys) + 1 - self.y0) / res))
        self.inside = np.zeros((self.ny, self.nx), np.uint8)
        pts = np.array([[(x - self.x0) / res, (y - self.y0) / res] for x, y in field_poly])
        cv2.fillPoly(self.inside, [np.round(pts).astype(np.int32)], 1)
        self.red = np.zeros_like(self.inside)
        r = max(1, int(math.ceil(margin / res)))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        self.version = 0
        self._update()

    def _update(self):
        self.blocked = (cv2.dilate(self.red, self.kernel) > 0) | (self.inside == 0)
        self.version += 1

    def cell(self, x, y):
        return (int(math.floor((y - self.y0) / self.res)),
                int(math.floor((x - self.x0) / self.res)))

    def centre(self, rc):
        return (self.x0 + (rc[1] + 0.5) * self.res, self.y0 + (rc[0] + 0.5) * self.res)

    def valid(self, rc):
        return 0 <= rc[0] < self.ny and 0 <= rc[1] < self.nx

    def is_blocked(self, x, y):
        rc = self.cell(x, y)
        return (not self.valid(rc)) or bool(self.blocked[rc])

    def is_red(self, x, y):
        """Inside red OR its safety margin."""
        rc = self.cell(x, y)
        return self.valid(rc) and bool(cv2.dilate(self.red, self.kernel)[rc])

    def is_raw_red(self, x, y):
        """Actually over red (what the penalty counts)."""
        rc = self.cell(x, y)
        return self.valid(rc) and bool(self.red[rc])

    def red_length(self, p, q):
        """Metres of the straight line p->q that lie over raw red."""
        d = math.dist(p, q)
        n = max(2, int(d / (self.res / 2)))
        return sum(self.is_raw_red(*lerp(p, q, (i + 0.5) / n)) for i in range(n)) * d / n

    def inside_field(self, x, y):
        rc = self.cell(x, y)
        return self.valid(rc) and bool(self.inside[rc])

    def mark_red(self, pts):
        changed = False
        for x, y in pts:
            rc = self.cell(x, y)
            if self.valid(rc) and not self.red[rc]:
                self.red[rc] = 1
                changed = True
        if changed:
            self._update()
        return changed

    def mark_polygon(self, poly_xy):
        pts = np.array([[(x - self.x0) / self.res, (y - self.y0) / self.res]
                        for x, y in poly_xy])
        cv2.fillPoly(self.red, [np.round(pts).astype(np.int32)], 1)
        self._update()

    def nearest_free(self, x, y):
        start = self.cell(x, y)
        if self.valid(start) and not self.blocked[start]:
            return (x, y)
        seen, q = {start}, deque([start])
        while q:
            rc = q.popleft()
            if self.valid(rc) and not self.blocked[rc]:
                return self.centre(rc)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (rc[0] + d[0], rc[1] + d[1])
                if n not in seen and -5 <= n[0] < self.ny + 5 and -5 <= n[1] < self.nx + 5:
                    seen.add(n)
                    q.append(n)
        return None

    def segment_clear(self, p, q):
        d = math.dist(p, q)
        n = max(2, int(d / (self.res / 2)))
        return all(not self.is_blocked(*lerp(p, q, i / n)) for i in range(n + 1))

    def plan(self, a, b, exact=False):
        """A* from a to b avoiding blocked cells; returns [xy...] or None.
        Goal inside a blocked area -> nearest free spot instead, unless
        exact=True and b is only in the safety MARGIN (not on red): then a
        short final hop to b is added (e.g. a QR right next to a red zone)."""
        r = self._plan(a, b)
        if (r is not None and exact and math.dist(r[-1], b) > 0.05
                and not self.is_raw_red(*b) and self.inside_field(*b)):
            r = r + [b]
        return r

    def _plan(self, a, b):
        prefix = []
        if self.is_blocked(*a):
            esc = self.nearest_free(*a)
            if esc is None:
                return None
            prefix, a = [esc], esc
        goal = self.nearest_free(*b)
        if goal is None:
            return None
        if self.segment_clear(a, goal):
            return prefix + [goal]
        s, g = self.cell(*a), self.cell(*goal)
        openq = [(0.0, s)]
        came, cost = {s: None}, {s: 0.0}
        found = False
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == g:
                found = True
                break
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    n = (cur[0] + dr, cur[1] + dc)
                    if not self.valid(n) or self.blocked[n]:
                        continue
                    c = cost[cur] + (1.414 if dr and dc else 1.0)
                    if c < cost.get(n, 1e9):
                        cost[n], came[n] = c, cur
                        heapq.heappush(openq, (c + math.hypot(n[0] - g[0], n[1] - g[1]), n))
        if not found:
            return None
        cells = []
        while cur is not None:
            cells.append(cur)
            cur = came[cur]
        pts = [self.centre(c) for c in reversed(cells)]
        pts[0], pts[-1] = a, goal
        out, i = [], 0                  # line-of-sight smoothing
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.segment_clear(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return prefix + out

    def route_blocked(self, cur, route):
        pts = [cur] + list(route)
        return any(not self.segment_clear(pts[i], pts[i + 1]) for i in range(len(pts) - 1))

    def render(self, scale=10, path=None, track=None, qrs=None, drone=None):
        img = np.full((self.ny * scale, self.nx * scale, 3), 40, np.uint8)
        big = lambda m: cv2.resize(m.astype(np.uint8), (self.nx * scale, self.ny * scale),
                                   interpolation=cv2.INTER_NEAREST) > 0
        img[big(self.inside)] = (70, 90, 70)
        img[big(self.blocked & (self.inside > 0))] = (60, 60, 150)
        img[big(self.red)] = (0, 0, 230)
        ppm = scale / self.res          # image pixels per metre
        to_px = lambda p: (int((p[0] - self.x0) * ppm), int((p[1] - self.y0) * ppm))
        if path:
            for i in range(len(path) - 1):
                cv2.line(img, to_px(path[i]), to_px(path[i + 1]), (200, 200, 200), 1)
        if track and len(track) > 1:
            cv2.polylines(img, [np.array([to_px(p) for p in track], np.int32)], False,
                          (255, 200, 0), 2)
        img = cv2.flip(img, 0)          # north up
        H = img.shape[0]
        fl = lambda p: (to_px(p)[0], H - 1 - to_px(p)[1])
        for q in (qrs or []):
            col = (0, 255, 0) if q.aligned else (0, 200, 255)
            cv2.circle(img, fl((q.x, q.y)), 6, col, 2)
            cv2.putText(img, (q.text or "?")[:12], (fl((q.x, q.y))[0] + 8, fl((q.x, q.y))[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        if drone:
            cv2.circle(img, fl(drone), 5, (255, 255, 255), -1)
        return img

# ═════════════════════════════════════════════════════════════════════════════
# DETECTION
# ═════════════════════════════════════════════════════════════════════════════

try:
    from pyzbar import pyzbar as _zbar
    _HAVE_ZBAR = True
except Exception:
    _HAVE_ZBAR = False
_cv_qr = cv2.QRCodeDetector()


def red_mask(small_bgr):
    """Red = HSV hue near 0/180 with good saturation AND Lab a* clearly red."""
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    a = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB)[:, :, 1]
    m = cv2.inRange(hsv, (0, 100, 60), (10, 255, 255)) | \
        cv2.inRange(hsv, (170, 100, 60), (180, 255, 255))
    m &= cv2.inRange(a, LAB_A_MIN_RED, 255)
    k = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


def red_sample_points(mask, step=8):
    """Sparse pixel samples of the red mask (for projecting onto the map)."""
    if cv2.countNonZero(mask) < RED_MIN_PX:
        return []
    ys, xs = np.nonzero(mask[::step, ::step])
    return list(zip((xs * step).tolist(), (ys * step).tolist()))


def scan_qr(gray):
    """Return [{'text': str|None, 'c': (cx, cy), 'quad': np.array}] in full-res px.
    text None = a QR shape was found but could not be decoded (too small/blurred)."""
    out = []
    if _HAVE_ZBAR:
        for o in _zbar.decode(gray, symbols=[_zbar.ZBarSymbol.QRCODE]):
            quad = np.array([(p.x, p.y) for p in o.polygon], np.float32)
            out.append({"text": o.data.decode("utf-8", "replace"),
                        "c": tuple(quad.mean(axis=0)), "quad": quad})
    else:
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
    if ok and pts is not None:
        for q in pts:
            q = q * 2.0
            c = tuple(q.mean(axis=0))
            if all(math.dist(c, o["c"]) > 40 for o in out):
                out.append({"text": None, "c": c, "quad": q})
    return out


class SurveyCam(threading.Thread):
    """Captures high-res frames; finds red (on a small copy) and QRs (full res).
    Every observation is queued with the frame time so the mission can use the
    drone pose AT THAT MOMENT."""

    def __init__(self):
        super().__init__(daemon=True)
        self.cam = None
        self.obs = deque(maxlen=60)
        self.lk = threading.Lock()
        self.preview = None
        self.fps = 0.0
        self._stop_evt = threading.Event()

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
        for ctrl in ({"AwbEnable": True, "AwbMode": 5}, {"AfMode": 2, "AfSpeed": 1}):
            try:
                self.cam.set_controls(ctrl)
            except Exception:
                pass
        log.info(f"  Camera {CAM_MODEL} @ {CAM_RES}  "
                 f"(QR decoder: {'pyzbar' if _HAVE_ZBAR else 'OpenCV'})")

    def run(self):
        n, t0 = 0, time.time()
        while not self._stop_evt.is_set():
            try:
                frame = self.cam.capture_array()        # BGR order already
            except Exception as e:
                log.warning(f"  capture: {e}")
                time.sleep(0.2)
                continue
            t = time.time()
            small = cv2.resize(frame, ANA_RES, interpolation=cv2.INTER_AREA)
            rm = red_mask(small)
            qrs = scan_qr(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            o = {"t": t, "red": red_sample_points(rm), "qrs": qrs,
                 "w": frame.shape[1], "h": frame.shape[0],
                 "aw": ANA_RES[0], "ah": ANA_RES[1]}
            with self.lk:
                self.obs.append(o)
                self.preview = (small, rm, qrs, frame.shape[1] / ANA_RES[0])
            n += 1
            if t - t0 > 2:
                self.fps, n, t0 = n / (t - t0), 0, t

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
        self.mav.wait_heartbeat()
        log.info(f"  Heartbeat ✓ system={self.mav.target_system}")
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
        best = None
        for q in self.items:
            d = math.hypot(q.x - x, q.y - y)
            same = (text is None or q.text is None or q.text == text)
            if d < QR_MERGE_M and same and (best is None or d < best[0]):
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
            if QR_ALIGN_EACH or q.text is None:
                return q
        return None

    def find(self, target):
        return next((q for q in self.items if q.text and target in q.text), None)

# ═════════════════════════════════════════════════════════════════════════════
# MISSION
# ═════════════════════════════════════════════════════════════════════════════

class Survey:
    def __init__(self, drone, cam, gmap, path, origin, show=True):
        self.d, self.cam, self.map, self.path, self.origin = drone, cam, gmap, path, origin
        self.reg = Registry()
        self.track = []
        self.t_track = 0
        self.t_map = 0
        self.show = show and bool(os.environ.get("DISPLAY"))
        self.status = ""
        self.yaw = 0.0
        self.red_since = None      # when the drone started being over red
        self.red_total = 0.0
        self.red_worst = 0.0       # longest continuous time over red (penalty metric)
        self.t_prev = time.time()

    def red_time(self):
        """Seconds continuously over red right now (0 if not over red)."""
        return time.time() - self.red_since if self.red_since else 0.0

    # ── observations → map + registry ────────────────────────────────────
    def process(self):
        for o in self.cam.drain():
            p = self.d.pose_at(o["t"])
            if p is None:
                continue
            _, x, y, alt, yaw, tilt = p
            if alt < 2.0 or tilt > MAX_TILT_DEG:
                continue
            if o["red"]:
                pts = [pixel_to_ground(px, py, o["aw"], o["ah"], alt, yaw) for px, py in o["red"]]
                if self.map.mark_red([(x + e, y + n) for e, n in pts]):
                    if time.time() - getattr(self, "t_redlog", 0) > 3:
                        self.t_redlog = time.time()
                        log.info(f"  ■ Red zone mapped near ({x:.1f}, {y:.1f}) — "
                                 f"{self.map.red.sum() * MAP_RES_M ** 2:.0f} m² so far")
            mpp = 2 * alt * math.tan(math.radians(HFOV_DEG / 2)) / o["w"]
            for q in o["qrs"]:
                if q.get("quad") is not None:          # size check (QR is 1 m)
                    quad = np.asarray(q["quad"], np.float32).reshape(-1, 2)
                    side_m = math.sqrt(max(cv2.contourArea(quad), 1.0)) * mpp
                    if not (QR_SIZE_M / QR_SIZE_TOL <= side_m <= QR_SIZE_M * QR_SIZE_TOL):
                        continue
                e, n = pixel_to_ground(q["c"][0], q["c"][1], o["w"], o["h"], alt, yaw)
                qx, qy = x + e, y + n
                if point_in_poly(qx, qy, self.map.poly):
                    rec, _ = self.reg.add(q["text"], qx, qy)
                    rec.last_seen = o["t"]
                    rec.last_off = math.hypot(e, n)

    def tick(self, status):
        self.status = status
        self.process()
        now = time.time()
        xy = self.d.xy()
        dt, self.t_prev = now - self.t_prev, now
        if xy and self.map.is_raw_red(*xy):          # penalty timer
            self.red_since = self.red_since or now
            self.red_total += dt
            self.red_worst = max(self.red_worst, self.red_time())
            if self.red_time() > RED_PENALTY_S * 0.8:
                log.error(f"  !!! {self.red_time():.1f} s over red zone")
        else:
            self.red_since = None
        if xy and now - self.t_track > 0.5:
            self.track.append(xy)
            self.t_track = now
        if now - self.t_map > (0.3 if self.show else 3.0):
            self.t_map = now
            img = self.map.render(path=[p[:2] for p in self.path], track=self.track,
                                  qrs=self.reg.items, drone=xy)
            cv2.imwrite(os.path.join(DEBUG_DIR, "survey_map.png"), img)
            if self.show:
                self._show(img)
        time.sleep(0.1)

    def _show(self, mapimg):
        pv = self.cam.preview
        if pv is None:
            return
        small, rm, qrs, s = pv
        cam = small.copy()
        cnts, _ = cv2.findContours(rm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cam, cnts, -1, (0, 0, 255), 2)
        for q in qrs:
            quad = (q["quad"] / s).astype(np.int32)
            cv2.polylines(cam, [quad], True, (0, 255, 0) if q["text"] else (0, 200, 255), 2)
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

    def quick_cross(self, cur, target):
        """No route around the red: allowed to fly STRAIGHT over it only if the
        time over red stays under RED_MAX_S (penalty is for > RED_PENALTY_S)."""
        red_m = self.map.red_length(cur, target)
        t_red = red_m / SPEED_MPS
        if t_red < RED_MAX_S:
            log.warning(f"  No way around — crossing red directly "
                        f"({red_m:.1f} m ≈ {t_red:.1f} s over red, limit {RED_PENALTY_S:.0f} s)")
            return [target]
        log.warning(f"  No way around and crossing would take {t_red:.1f} s over red — skipping")
        return None

    def goto(self, target, alt, yaw=None, handle_qr=True, accept=ACCEPT_RADIUS_M, exact=False):
        """Fly to target (field xy) around red zones. Pauses for new QRs."""
        if yaw is not None:
            self.yaw = yaw
        cur = self.d.xy()
        route = self.map.plan(cur, target, exact)
        crossing = False
        if route is None:
            route, crossing = self.quick_cross(cur, target), True
            if route is None:
                return False
        ver = self.map.version
        t0 = time.time()
        while route:
            self.d.check()
            if time.time() - t0 > WAYPOINT_TIMEOUT:
                log.warning("  Waypoint timeout — skipping")
                return False
            cur = self.d.xy()
            if not crossing and self.red_time() > RED_MAX_S * 0.5:
                log.warning(f"  Over red for {self.red_time():.1f} s — leaving it now")
                route = self.map.plan(cur, target, exact)   # starts with an escape
                if route is None:
                    route, crossing = self.quick_cross(cur, target), True
                    if route is None:
                        return False
            if not crossing and self.map.version != ver:   # new red -> re-plan if needed
                ver = self.map.version
                check = route[:-1] if exact else route
                if check and self.map.route_blocked(cur, check):
                    log.info("  ↺ Route crosses red zone — re-planning")
                    route = self.map.plan(cur, target, exact)
                    if route is None:
                        route, crossing = self.quick_cross(cur, target), True
                        if route is None:
                            return False
            if handle_qr:
                q = self.reg.needs_visit()
                if q is not None:
                    self.visit_qr(q)
                    route = self.map.plan(self.d.xy(), target)
                    if route is None:
                        log.warning("  Target unreachable after QR visit — skipping")
                        return False
                    ver = self.map.version
                    continue
            wp = route[0]
            self.d.goto_xy(wp[0], wp[1], alt, self.yaw)
            last = len(route) == 1
            dist = math.dist(cur, wp)
            if dist < (accept if last else VIA_RADIUS_M):
                route.pop(0)
            self.tick(f"→ ({wp[0]:.1f},{wp[1]:.1f}) {dist:.1f} m  alt {self.d.alt() or 0:.1f}"
                      f"  QRs {len(self.reg.items)}"
                      + ("  [detour]" if not last else ""))
        return True

    def visit_qr(self, q, hover_s=0.0):
        """Fly over QR, centre on it, decode (descending if needed), refine coord."""
        q.visits += 1
        log.info(f"\n  ▶ Visiting {q}")
        if self.map.is_raw_red(q.x, q.y):
            log.warning("    QR is ON a red zone — not hovering there (penalty)")
            q.rejected = True
            return False
        self.goto((q.x, q.y), SURVEY_ALT_M, handle_qr=False, accept=0.6, exact=True)
        alt_cmd = SURVEY_ALT_M
        t0, t_good = time.time(), None
        seen0 = getattr(q, "last_seen", 0)
        while time.time() - t0 < QR_VISIT_TIMEOUT:
            self.d.check()
            self.tick(f"QR #{q.id} centring  alt {self.d.alt() or 0:.1f}  "
                      f"text={q.text or '?'}")
            fresh = time.time() - getattr(q, "last_seen", 0) < 1.0
            if (q.text is None and time.time() - t0 > 4 and alt_cmd > QR_DECODE_ALT_M):
                alt_cmd = QR_DECODE_ALT_M
                log.info(f"    not decoded from {SURVEY_ALT_M:.0f} m → descending to "
                         f"{QR_DECODE_ALT_M:.0f} m")
            self.d.goto_xy(q.x, q.y, alt_cmd, self.yaw)
            cur = self.d.xy()
            centred = (fresh and math.dist(cur, (q.x, q.y)) < QR_CENTRE_TOL_M
                       and getattr(q, "last_off", 9) < QR_CENTRE_TOL_M * 1.5
                       and abs((self.d.alt() or 0) - alt_cmd) < 0.8)
            if centred and q.text:
                t_good = t_good or time.time()
                if time.time() - t_good >= QR_HOLD_S:
                    q.aligned = True
                    lat, lon = xy_to_latlon(q.x, q.y, *self.origin)
                    log.info(f"    ✓ Centred over QR #{q.id} '{q.text}'  "
                             f"{lat:.7f}, {lon:.7f}")
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
        if alt_cmd != SURVEY_ALT_M:
            self.wait_alt(q.x, q.y, SURVEY_ALT_M)
        return q.aligned

    # ── main flow ────────────────────────────────────────────────────────
    def run(self, target_text=None):
        prev = self.d.xy()
        for i, (x, y, lab) in enumerate(self.path, 1):
            yaw = bearing(prev, (x, y)) if math.dist(prev, (x, y)) > 1 else self.yaw
            log.info(f"\nWP {i}/{len(self.path)} [{lab}]  ({x:.1f}, {y:.1f})")
            self.goto((x, y), SURVEY_ALT_M, yaw=yaw)
            prev = (x, y)
        # mop up anything still unvisited
        while self.reg.needs_visit():
            self.visit_qr(self.reg.needs_visit())

        log.info("\n══ SURVEY COMPLETE ══")
        for q in self.reg.items:
            if not q.rejected:
                lat, lon = xy_to_latlon(q.x, q.y, *self.origin)
                log.info(f"  QR #{q.id}: '{q.text}'  {lat:.7f}, {lon:.7f}  "
                         f"{'centred' if q.aligned else 'seen'}")
        log.info(f"  Red mapped: ~{self.map.red.sum() * MAP_RES_M ** 2:.0f} m²   "
                 f"time over red: total {self.red_total:.1f} s, "
                 f"longest {self.red_worst:.1f} s (penalty at {RED_PENALTY_S:.0f} s)")
        self.save()

        if target_text:
            q = self.reg.find(target_text)
            if q is None:
                log.warning(f"  Target QR '{target_text}' not found")
            else:
                log.info(f"\n══ TARGET '{q.text}' — going there ══")
                q.aligned = False
                q.visits = 0
                self.visit_qr(q, hover_s=HOVER_S)

    def save(self):
        res = {"qrs": [], "red_cells_xy": []}
        for q in self.reg.items:
            if q.rejected:
                continue
            lat, lon = xy_to_latlon(q.x, q.y, *self.origin)
            res["qrs"].append({"id": q.id, "text": q.text, "lat": lat, "lon": lon,
                               "x": round(q.x, 2), "y": round(q.y, 2), "centred": q.aligned})
        rr, cc = np.nonzero(self.map.red)
        res["red_cells_xy"] = [list(map(lambda v: round(v, 2), self.map.centre((r, c))))
                               for r, c in zip(rr, cc)]
        with open(os.path.join(DEBUG_DIR, "survey_results.json"), "w") as f:
            json.dump(res, f, indent=2)
        cv2.imwrite(os.path.join(DEBUG_DIR, "survey_map.png"),
                    self.map.render(path=[p[:2] for p in self.path], track=self.track,
                                    qrs=self.reg.items, drone=self.d.xy()))
        log.info(f"  Saved {DEBUG_DIR}/survey_results.json + survey_map.png")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def plan(origin):
    quad = [latlon_to_xy(la, lo, *origin) for la, lo in VERTICES]
    A, B, C, D = inset_quad(quad, INSET_M)
    path = build_lawnmower(A, B, C, D, SPACING_M)
    gmap = GridMap(quad)
    for poly in RED_ZONES_LATLON:
        gmap.mark_polygon([latlon_to_xy(la, lo, *origin) for la, lo in poly])
    return quad, path, gmap


def bench():
    cam = SurveyCam()
    cam.open()
    cam.start()
    log.info("BENCH — no flight. Ctrl+C to stop.")
    show = bool(os.environ.get("DISPLAY"))
    t_log = 0
    try:
        while True:
            obs = cam.drain()
            pv = cam.preview
            if obs and time.time() - t_log > 0.5:
                o = obs[-1]
                qs = [q["text"] or "(QR shape, not decoded)" for q in o["qrs"]]
                log.info(f"fps {cam.fps:.1f}  red samples {len(o['red'])}  QRs {qs}")
                t_log = time.time()
            if pv is not None:
                small, rm, qrs, s = pv
                img = small.copy()
                cnts, _ = cv2.findContours(rm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnts, -1, (0, 0, 255), 2)
                for q in qrs:
                    cv2.polylines(img, [(q["quad"] / s).astype(np.int32)], True,
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


def main():
    global SPACING_M, SURVEY_ALT_M, SPEED_MPS, INSET_M, TARGET_QR, QR_ALIGN_EACH
    global Picamera2, mavutil
    ap = argparse.ArgumentParser(description="Squadrone survey: QR + red zones")
    ap.add_argument("--conn", default=SERIAL_PORT)
    ap.add_argument("--baud", type=int, default=BAUD_RATE)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--spacing", type=float, default=SPACING_M)
    ap.add_argument("--alt", type=float, default=SURVEY_ALT_M)
    ap.add_argument("--speed", type=float, default=SPEED_MPS)
    ap.add_argument("--inset", type=float, default=INSET_M)
    ap.add_argument("--target", default=TARGET_QR, help="QR text to finish over")
    ap.add_argument("--no-align", action="store_true", help="log QRs, don't stop at each")
    a = ap.parse_args()
    if not (1 <= a.spacing <= 30 and 3 <= a.alt <= 30 and 0.5 <= a.speed <= 5
            and 0 <= a.inset <= 5):
        ap.error("spacing 1-30, alt 3-30, speed 0.5-5, inset 0-5")
    SPACING_M, SURVEY_ALT_M, SPEED_MPS, INSET_M = a.spacing, a.alt, a.speed, a.inset
    TARGET_QR, QR_ALIGN_EACH = a.target, not a.no_align
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    log.info("=" * 60)
    log.info(f"SQUADRONE survey  alt={SURVEY_ALT_M} spacing={SPACING_M} speed={SPEED_MPS} "
             f"inset={INSET_M} target={TARGET_QR!r} align_each={QR_ALIGN_EACH}")
    log.info("=" * 60)

    origin = POINT_A
    quad, path, gmap = plan(origin)
    length = sum(math.dist(path[i][:2], path[i + 1][:2]) for i in range(len(path) - 1))
    log.info(f"Planned {len(path)} waypoints, {length:.0f} m (~{length / SPEED_MPS:.0f} s "
             f"+ QR stops)")
    for i, (x, y, lab) in enumerate(path, 1):
        la, lo = xy_to_latlon(x, y, *origin)
        log.info(f"  WP{i:02d} [{lab:6s}] {la:.7f}, {lo:.7f}")
    cv2.imwrite(os.path.join(DEBUG_DIR, "survey_plan.png"),
                gmap.render(path=[p[:2] for p in path]))
    log.info(f"  Plan image: {DEBUG_DIR}/survey_plan.png")
    if a.plan_only:
        return

    from picamera2 import Picamera2 as _P
    Picamera2 = _P
    if a.bench:
        bench()
        return
    from pymavlink import mavutil as _m
    mavutil = _m

    cam = SurveyCam()
    cam.open()
    cam.start()
    drone = Drone(a.conn, a.baud, origin)
    took_over = False
    try:
        drone.wait_gps()
        xy = drone.xy()
        log.info(f"Drone is {math.dist(xy, path[0][:2]):.1f} m from start point A")
        drone.set_param("RTL_ALT", SURVEY_ALT_M * 100)
        input("\n>>> ENTER to arm and take off (area clear!) <<<\n")
        drone.arm_and_takeoff(SURVEY_ALT_M)
        Survey(drone, cam, gmap, path, origin).run(TARGET_QR)
        log.info("\n✓ Done — RTL")
    except KeyboardInterrupt:
        log.warning("Ctrl+C — aborting")
    except PilotTookOver:
        took_over = True
        log.warning(f"Pilot / failsafe took over (mode {drone.mode()}) — script exits")
    except Exception as e:
        log.error(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if not took_over and drone.armed():
            drone.set_mode("RTL")
        cam.stop()
        drone.close()


if __name__ == "__main__":
    main()