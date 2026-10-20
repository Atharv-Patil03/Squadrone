#!/usr/bin/env python3
"""
SQUADRONE — Lawnmower Survey + QR scan & align   (QR only)
File: survey_qr.py      (built on lawnmower_trial.py)
Team: Squadrone AT2026RC60, PVGCOET Pune, SAEINDIA AeroTHON 2026

WHAT IT DOES
  1. Lawnmower path inside the field A -> B ... -> D (same logic as
     lawnmower_trial.py, with an inset). Drone faces its direction of travel.
  2. DOWN camera (ArduCam IMX519, rear) captures at 2328x1748 (~5.6 mm/pixel
     at 10 m) so 1 m QR codes decode from survey height.
  3. Every QR seen gets a ground coordinate (drone position at that frame +
     pixel offset in metres). New QR -> pause the pass, fly over it, centre on
     it (descend to QR_DECODE_ALT_M only if it will not decode), record text +
     refined coordinate, climb back, resume the pass.
  4. End: summary + ~/squadrone/survey_results.json + survey_map.png.
     --target TEXT -> fly to that QR, centre, hover HOVER_S. Then RTL.
  SAFETY: >3 m outside the field = automatic RTL. Mode change by the pilot
  (e.g. Loiter) = script stops immediately.

RUN
  python3 survey_qr.py --plan-only            # path + map, no drone
  python3 survey_qr.py --bench                # camera + QR detection only
  python3 survey_qr.py --alt 5 --no-align     # fly the path, just log QRs
  python3 survey_qr.py --alt 5                # stop + centre on each QR
  python3 survey_qr.py --alt 10 --target "ZONE-3"
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
# Recorded with the drone on 20-10-2026
POINT_A = (18.4898362, 73.8515125)   # start (SE)
POINT_B = (18.4898302, 73.8511694)   # end of first pass (SW)
POINT_C = (18.4900060, 73.8511663)   # NW
POINT_D = (18.4900276, 73.8515075)   # finish (NE)
VERTICES = [POINT_A, POINT_B, POINT_C, POINT_D]
INSET_M = 2.0

SURVEY_ALT_M    = 10.0
SPACING_M       = 8.0
SPEED_MPS       = 1.5
ACCEPT_RADIUS_M = 1.0
WAYPOINT_TIMEOUT = 120
GEOFENCE_MARGIN_M = 3.0    # further than this outside the field -> RTL
MAX_START_DIST_M  = 15.0   # refuse to start further than this from A

# QR
QR_ALIGN_EACH     = True   # stop and centre over every QR (False = just log)
QR_DECODE_ALT_M   = 5.0    # descend to this if it will not decode from survey alt
QR_MERGE_M        = 2.0    # undecoded sightings closer than this = same QR
QR_TEXT_MERGE_M   = 8.0    # same text within this = same QR
QR_VISIT_TIMEOUT  = 30
QR_CENTRE_TOL_M   = 0.30
QR_HOLD_S         = 1.0
QR_SIZE_M         = 1.0    # real QR size (rulebook: 1 m x 1 m)
QR_SIZE_TOL       = 2.5    # accept 1/2.5 .. 2.5 x the expected size
TARGET_QR         = None   # text (or part of it) of the QR to finish over
HOVER_S           = 10

# Camera (DOWN, ArduCam IMX519, rear, looking straight down)
CAM_MODEL        = "imx519"
CAM_RES          = (2328, 1748)   # --res 4656x3496 = full sensor (sharper, slower)
FOCUS_M          = None           # None = continuous AF; --focus-alt locks it
PREVIEW_RES      = (640, 480)
HFOV_DEG         = 66.0
VFOV_DEG         = 51.0
CAM_OFFSET_FWD_M = -0.10          # camera behind drone centre
# FLIP_Y = True: flight 20-10 showed QR positions sliding WITH the drone along
# track (and the banner drifting back) = forward/back was reversed.
FLIP_X, FLIP_Y   = False, True
MAX_TILT_DEG     = 10.0

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
        if not t:
            continue
        quad = np.asarray(p, np.float32).reshape(-1, 2)[:4] + (ox, oy)
        out.append({"text": t, "c": tuple(float(v) for v in quad.mean(axis=0)),
                    "quad": quad, "by": by})
    return out


# Tile scan: WeChat shrinks the whole frame before detecting, so a small QR
# (far away) vanishes. Scanning a tile keeps it big relative to the input.
# Bench: 85 px QR -> 0/5 full frame, 5/5 tiled. All tiles every frame is too
# slow, so each frame = full frame + ONE tile, round robin.
SCAN_TILES = 3            # 3x3 tiles, 25 % overlap (0 = off)
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


def scan_qr(bgr):
    """Return [{'text': str|None, 'c': (cx, cy), 'quad': np.array}] in full-res px.
    text None = a QR shape was found but could not be decoded (too small/blurred).
    Order: WeChatQRCode -> pyzbar -> OpenCV (same as the team's QR script)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = []
    if _wechat is not None:
        # Full frame at HALF size: big QRs decode fine and it is always fast
        # (bench: max 49 ms). Full-res full frame sometimes took 30 s (SR
        # chewing on soil texture) — small QRs are covered by the tiles.
        half = cv2.resize(bgr, (bgr.shape[1] // 2, bgr.shape[0] // 2),
                          interpolation=cv2.INTER_AREA)
        for d in _decode_wechat(half):
            d["quad"] = d["quad"] * 2.0
            d["c"] = (d["c"][0] * 2.0, d["c"][1] * 2.0)
            out.append(d)
    if _wechat is not None and SCAN_TILES:
        for d in _decode_next_tile(bgr):            # adds far / small QRs
            if all(math.dist(d["c"], o["c"]) > 40 for o in out):
                out.append(d)
    if not out and _HAVE_ZBAR:
        for o in _zbar.decode(gray, symbols=[_zbar.ZBarSymbol.QRCODE]):
            quad = np.array([(p.x, p.y) for p in o.polygon], np.float32)
            out.append({"text": o.data.decode("utf-8", "replace"),
                        "c": tuple(float(v) for v in quad.mean(axis=0)),
                        "quad": quad, "by": "pyzbar"})
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
    if ok and pts is not None:
        for q in pts:
            q = q * 2.0
            c = tuple(q.mean(axis=0))
            if all(math.dist(c, o["c"]) > 40 for o in out):
                out.append({"text": None, "c": c, "quad": q})
    return out


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
        focus = ({"AfMode": 0, "LensPosition": 1.0 / FOCUS_M} if FOCUS_M     # manual
                 else {"AfMode": 2, "AfSpeed": 1})                          # continuous
        for ctrl in ({"AwbEnable": True, "AwbMode": 5}, focus,
                     {"AeExposureMode": 1}):          # short exposure = less motion blur
            try:
                self.cam.set_controls(ctrl)
            except Exception as e:
                log.warning(f"  camera control {list(ctrl)} not applied: {e}")
        log.info(f"  Focus: {'fixed at %.1f m' % FOCUS_M if FOCUS_M else 'continuous AF'}")
        log.info(f"  Camera {CAM_MODEL} @ {CAM_RES}  "
                 f"(QR decoders: {'WeChat + ' if _wechat is not None else ''}"
                 f"{'pyzbar' if _HAVE_ZBAR else 'OpenCV'})")

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
            t_scan = time.time()
            qrs = scan_qr(frame)
            self.scan_ms = (time.time() - t_scan) * 1000
            o = {"t": t, "qrs": qrs, "w": frame.shape[1], "h": frame.shape[0]}
            small = cv2.resize(frame, PREVIEW_RES, interpolation=cv2.INTER_AREA)
            with self.lk:
                self.obs.append(o)
                self.preview = (small, qrs, frame.shape[1] / PREVIEW_RES[0])
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
                raise RuntimeError("No heartbeat from the flight controller")
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
            if QR_ALIGN_EACH or q.text is None:
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
                    if not (QR_SIZE_M / QR_SIZE_TOL <= side_m <= QR_SIZE_M * QR_SIZE_TOL):
                        continue
                e, n = pixel_to_ground(q["c"][0], q["c"][1], o["w"], o["h"], alt, yaw)
                qx, qy = x + e, y + n
                if point_in_poly(qx, qy, self.field):
                    rec, _ = self.reg.add(q["text"], qx, qy)
                    rec.last_seen = o["t"]
                    rec.last_off = math.hypot(e, n)

    def tick(self, status):
        self.status = status
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
            if handle_qr:
                q = self.reg.needs_visit()
                if q is not None:
                    self.visit_qr(q)
                    continue
            cur = self.d.xy()
            dist = math.dist(cur, target)
            if dist < accept:
                return True
            self.d.goto_xy(target[0], target[1], alt, self.yaw)
            self.tick(f"→ ({target[0]:.1f},{target[1]:.1f}) {dist:.1f} m  "
                      f"alt {self.d.alt() or 0:.1f}  QRs {len(self.reg.items)}")

    def visit_qr(self, q, hover_s=0.0):
        """Fly over QR, centre on it, decode (descending if needed), refine coord."""
        q.visits += 1
        log.info(f"\n  ▶ Visiting {q}")
        self.goto((q.x, q.y), SURVEY_ALT_M, handle_qr=False, accept=0.6)
        alt_cmd = SURVEY_ALT_M
        t0, t_good = time.time(), None
        seen0 = getattr(q, "last_seen", 0)
        while time.time() - t0 < QR_VISIT_TIMEOUT:
            self.d.check()
            self.tick(f"QR #{q.id} centring  alt {self.d.alt() or 0:.1f}  "
                      f"text={q.text or '?'}")
            fresh = time.time() - getattr(q, "last_seen", 0) < 1.0
            if q.text is None and time.time() - t0 > 4 and alt_cmd > QR_DECODE_ALT_M:
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
                    log.info(f"    ✓ Centred over QR #{q.id} '{q.text}'  {lat:.7f}, {lon:.7f}")
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
            log.info(f"\nWP {i}/{len(self.path)} [{lab}]  ({x:.1f}, {y:.1f})  "
                     f"heading {compass(yaw)}")
            self.goto((x, y), SURVEY_ALT_M, yaw=yaw)
            prev = (x, y)
        while self.reg.needs_visit():
            self.visit_qr(self.reg.needs_visit())

        log.info("\n══ SURVEY COMPLETE ══")
        for q in self.reg.items:
            if not q.rejected:
                lat, lon = xy_to_latlon(q.x, q.y, *self.origin)
                log.info(f"  QR #{q.id}: '{q.text}'  {lat:.7f}, {lon:.7f}  "
                         f"{'centred' if q.aligned else 'seen'}")
        self.save()

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
                               "sightings": int(q.n), "centred": bool(q.aligned)})
        try:
            with open(os.path.join(DEBUG_DIR, "survey_results.json"), "w") as f:
                json.dump(res, f, indent=2)
        except Exception as e:                      # never let saving stop the flight
            log.error(f"  could not save results: {e}")
        log.info(f"  Saved {DEBUG_DIR}/survey_results.json + survey_map.png")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def plan(origin):
    field = [latlon_to_xy(la, lo, *origin) for la, lo in VERTICES]
    A, B, C, D = inset_quad(field, INSET_M)
    return field, build_lawnmower(A, B, C, D, SPACING_M)


def bench(dist=None):
    """dist = distance camera->QR in metres (for FOV measurement + direction check)."""
    cam = SurveyCam()
    cam.open()
    cam.start()
    log.info("BENCH — no flight. Ctrl+C to stop.")
    if dist:
        log.info(f"  FOV mode: QR ({QR_SIZE_M} m) at {dist} m from the lens")
    show = bool(os.environ.get("DISPLAY"))
    t_log = 0
    fovs = []
    try:
        while True:
            obs = cam.drain()
            pv = cam.preview
            if obs and time.time() - t_log > 0.5:
                o = obs[-1]
                qs = [f"{q['text']} [{q.get('by', '?')}]" if q["text"]
                      else "(QR shape, not decoded)" for q in o["qrs"]]
                log.info(f"fps {cam.fps:.1f}  scan {cam.scan_ms:.0f} ms  QRs {qs}")
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


def main():
    global SPACING_M, SURVEY_ALT_M, SPEED_MPS, INSET_M, TARGET_QR, QR_ALIGN_EACH
    global FOCUS_M, CAM_RES, SCAN_TILES
    global Picamera2, mavutil
    ap = argparse.ArgumentParser(description="Squadrone lawnmower survey with QR scan")
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
    ap.add_argument("--dist", type=float, help="bench: camera-to-QR distance (m) to measure FOV")
    ap.add_argument("--models", help="folder with the WeChatQRCode model files")
    ap.add_argument("--no-wechat", action="store_true", help="use pyzbar only")
    ap.add_argument("--focus-alt", action="store_true",
                    help="lock focus at the survey altitude (no AF hunting)")
    ap.add_argument("--res", help="camera resolution WxH, e.g. 4656x3496")
    ap.add_argument("--tiles", type=int, default=SCAN_TILES, help="tile grid N (0 = off)")
    a = ap.parse_args()
    if not (1 <= a.spacing <= 30 and 3 <= a.alt <= 30 and 0.5 <= a.speed <= 5
            and 0 <= a.inset <= 5):
        ap.error("spacing 1-30, alt 3-30, speed 0.5-5, inset 0-5")
    SPACING_M, SURVEY_ALT_M, SPEED_MPS, INSET_M = a.spacing, a.alt, a.speed, a.inset
    TARGET_QR, QR_ALIGN_EACH = a.target, not a.no_align
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    log.info("=" * 60)
    log.info(f"SQUADRONE QR survey  alt={SURVEY_ALT_M} spacing={SPACING_M} "
             f"speed={SPEED_MPS} inset={INSET_M} target={TARGET_QR!r} "
             f"align_each={QR_ALIGN_EACH}")
    log.info("=" * 60)

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
    if a.focus_alt:
        FOCUS_M = a.dist if (a.bench and a.dist) else SURVEY_ALT_M
    if a.res:
        CAM_RES = tuple(int(v) for v in a.res.lower().split("x"))
    if not a.no_wechat:
        load_wechat(a.models)
    if a.bench:
        bench(a.dist)
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
        log.info(f"Drone is {math.dist(xy, path[0][:2]):.1f} m from start point A, "
                 f"{dist_outside(*xy, field):.1f} m outside the field")
        log.info(f"FIRST MOVE after takeoff: {math.dist(xy, path[0][:2]):.1f} m towards "
                 f"{compass(bearing(xy, path[0][:2]))} — check this matches where A is!")
        log.info(f"Then pass 1: A → B towards {compass(bearing(path[0][:2], path[1][:2]))}")
        if math.dist(xy, path[0][:2]) > MAX_START_DIST_M:
            raise RuntimeError(f"More than {MAX_START_DIST_M:.0f} m from A — "
                               "move the drone near A first")
        drone.set_param("RTL_ALT", SURVEY_ALT_M * 100)
        input("\n>>> ENTER to arm and take off (area clear!) <<<\n")
        drone.arm_and_takeoff(SURVEY_ALT_M)
        Survey(drone, cam, field, path, origin).run(TARGET_QR)
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
        if not took_over and drone.armed():
            drone.set_mode("RTL")
        cam.stop()
        drone.close()


if __name__ == "__main__":
    main()