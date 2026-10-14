"""
SQUADRONE - ONBOARD QR DETECTION (mission ready)

Hardware:
    Raspberry Pi 5
    Arducam IMX519 Autofocus (SKU B0371)
    NOTE: this Pi also has a second camera (e.g. imx500) attached.
          Picamera2() with no index can open the WRONG camera, which
          then fails on AfMode/AfSpeed. See resolve_camera_index().

Primary detector : OpenCV WeChatQRCode
Fallback         : pyzbar

Changes from perception prototype:
    - get_confirmed_target() returns confirmed string for FSM
    - get_alignment_offset() returns (north_m, east_m) for velocity control
    - altitude can be updated live via set_altitude()
    - module can be imported by FSM or run standalone
    - Auto-detects and opens the IMX519 specifically
    - update_tracks() uses the actual captured frame height

NEW - dual matching (text / image / both):
    - store_reference()  : saves start-QR text + image crop (auto in S2)
    - is_target_qr()     : text match, image match, or both
    - start_delivery_search() : switch to delivery phase; offsets are then
                                computed ONLY for the QR matching the reference
    - is_target_visible(): FSM check that the target is in view this frame
    - stale offsets are zeroed when the target is not visible

FSM USAGE
    S2 (start QR):   process_frame(frame)  -> poll get_confirmed_target()
                     reference is stored automatically on confirmation
    S5 (delivery):   start_delivery_search()   (keeps the reference)
                     process_frame(frame) -> is_target_visible(),
                                             get_alignment_offset()
"""

import cv2
import numpy as np
import time
import logging
import math

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from picamera2 import Picamera2
from libcamera import controls


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CAMERA_WIDTH    = 1920
CAMERA_HEIGHT   = 1080
CAMERA_FORMAT   = "RGB888"

# None = auto-detect IMX519 by model name; set an int to force it
CAMERA_INDEX = None

LENS_POSITION   = None
EXPOSURE_US     = None
ANALOGUE_GAIN   = None

MAX_PROCESS_WIDTH  = 1920
PREPROCESS_MODE    = "original"

BLUR_GATE_ENABLED  = False
BLUR_THRESHOLD     = 30.0

CONSENSUS_REQUIRED    = 3
MAX_MISSED_FRAMES     = 5
TRACK_MATCH_DISTANCE  = 100

SHOW_PREVIEW   = True
DISPLAY_WIDTH  = 1280

# GSD - altitude updated live via set_altitude()
ALTITUDE_M          = 5.0
FOCAL_LENGTH_MM     = 4.28
PIXEL_SIZE_UM       = 1.22
SENSOR_FULL_WIDTH_PX = 4656

PRINT_ON_CONFIRMATION = True
FPS_LOG_INTERVAL      = 2.0

MODEL_DIR              = BASE_DIR / "models"
WECHAT_DETECT_PROTOTXT = str(MODEL_DIR / "detect.prototxt")
WECHAT_DETECT_MODEL    = str(MODEL_DIR / "detect.caffemodel")
WECHAT_SR_PROTOTXT     = str(MODEL_DIR / "sr.prototxt")
WECHAT_SR_MODEL        = str(MODEL_DIR / "sr.caffemodel")

# ─── DETECTION MODE ───────────────────────────────────────────────
# Set this on the day, based on what the organisers tell you.
# "text"  - compare decoded text only (QR has readable content)
# "image" - compare QR visual pattern only (content unknown/unreadable)
# "both"  - text decides when BOTH QRs decode to non-empty text,
#           otherwise falls back to image matching
DETECTION_MODE = "both"

TEXT_MATCH_ENABLED  = True
IMAGE_MATCH_ENABLED = True

# Image similarity (normalised correlation, -1..1, higher = more alike).
# TUNE THIS ON THE BENCH: print scores for same-QR vs different-QR and
# pick a value between the two clusters.
IMAGE_MATCH_THRESHOLD = 0.60
CROP_SIZE             = 200
SAVE_REFERENCE_CROP   = True


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ============================================================
# OPTIONAL PYZBAR
# ============================================================

try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except Exception as exc:
    pyzbar = None
    PYZBAR_AVAILABLE = False
    logging.warning("pyzbar unavailable: %s", exc)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Detection:
    text: str
    cx: float
    cy: float
    corners: np.ndarray
    method: str


@dataclass
class QRTrack:
    track_id: int
    text: str
    cx: float
    cy: float
    history: deque = field(default_factory=deque)
    misses: int = 0
    confirmed: bool = False
    printed: bool = False


tracks        = {}
next_track_id = 1

sensor_crop_width_px = SENSOR_FULL_WIDTH_PX

# ── FSM-FACING STATE ──────────────────────────────────────────────
_confirmed_target   = None      # str  - set when first QR confirmed (S2)
_last_north_m       = 0.0       # float - NED offset to target QR
_last_east_m        = 0.0
_live_altitude_m    = ALTITUDE_M

# ── REFERENCE + DELIVERY STATE ────────────────────────────────────
reference_text  = None          # decoded text of start QR ("" if none)
reference_crop  = None          # 200x200 equalised grayscale of start QR
_delivery_mode  = False         # True after start_delivery_search()
_target_visible = False         # delivery target matched this frame
_last_match_info = ""           # "method: info" for overlay/debug


def set_altitude(alt_m: float):
    """Call this from FSM with live Pixhawk altitude each frame."""
    global _live_altitude_m
    _live_altitude_m = alt_m


def get_confirmed_target() -> str | None:
    """
    Returns the confirmed start-QR string once consensus is reached.
    Returns None if not yet confirmed.
    NOTE: in "image"/"both" mode this can be "" (empty string) for a QR
    with no readable text - check `is None`, not truthiness.
    """
    return _confirmed_target


def get_alignment_offset() -> tuple[float, float]:
    """
    Returns (north_m, east_m) offset from image centre to the target QR.
    Returns (0.0, 0.0) if the target QR is not visible this frame.
    """
    return _last_north_m, _last_east_m


def is_target_visible() -> bool:
    """
    Delivery phase: True if the QR matching the reference was seen in the
    most recent processed frame.
    """
    return _target_visible


def _zero_offsets():
    global _last_north_m, _last_east_m
    _last_north_m = 0.0
    _last_east_m  = 0.0


def reset_detection(keep_reference: bool = False):
    """
    Call between missions or on FSM state reset.
    keep_reference=True keeps the stored start-QR reference (text + image).
    """
    global _confirmed_target, tracks, next_track_id
    global _delivery_mode, _target_visible, _last_match_info
    _confirmed_target = None
    _zero_offsets()
    tracks            = {}
    next_track_id     = 1
    _delivery_mode    = False
    _target_visible   = False
    _last_match_info  = ""
    if not keep_reference:
        clear_reference()
    logging.info("Detection state reset (keep_reference=%s)", keep_reference)


def start_delivery_search():
    """
    Call when the FSM moves to the delivery phase (S5).
    Keeps the reference, clears tracks/offsets, and from now on offsets are
    computed only for the QR that matches the reference.
    """
    global _delivery_mode, _target_visible, tracks, next_track_id
    if not reference_ready():
        logging.warning("start_delivery_search() with NO reference stored")
    tracks          = {}
    next_track_id   = 1
    _delivery_mode  = True
    _target_visible = False
    _zero_offsets()
    logging.info("Delivery search started (mode=%s)", DETECTION_MODE)


# ============================================================
# WECHAT DETECTOR
# ============================================================

detector      = None
wechat_failed = False


def get_detector():
    global detector, wechat_failed
    if detector is None and not wechat_failed:
        try:
            detector = cv2.wechat_qrcode_WeChatQRCode(
                WECHAT_DETECT_PROTOTXT,
                WECHAT_DETECT_MODEL,
                WECHAT_SR_PROTOTXT,
                WECHAT_SR_MODEL
            )
            logging.info("WeChatQRCode loaded")
        except Exception as exc:
            wechat_failed = True
            logging.error("WeChatQRCode load failed: %s", exc)
    return detector


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_frame(frame):
    original_h, original_w = frame.shape[:2]
    processed = frame

    if MAX_PROCESS_WIDTH and original_w > MAX_PROCESS_WIDTH:
        scale     = MAX_PROCESS_WIDTH / original_w
        processed = cv2.resize(frame,
                               (int(original_w * scale),
                                int(original_h * scale)),
                               interpolation=cv2.INTER_AREA)

    processed_h, processed_w = processed.shape[:2]
    scale_x = original_w / processed_w
    scale_y = original_h / processed_h

    if PREPROCESS_MODE == "original":
        return processed, scale_x, scale_y

    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    if PREPROCESS_MODE in ("clahe", "sharpen"):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

    if PREPROCESS_MODE == "sharpen":
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        gray   = cv2.filter2D(gray, -1, kernel)
        gray   = cv2.GaussianBlur(gray, (3, 3), 0)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), scale_x, scale_y


# ============================================================
# BLUR DETECTION
# ============================================================

def sharpness_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ============================================================
# GSD AND GROUND OFFSET
# ============================================================

def calculate_gsd(altitude_m, image_width_px):
    used_sensor_width_mm = sensor_crop_width_px * (PIXEL_SIZE_UM / 1000.0)
    ground_width_m       = altitude_m * used_sensor_width_mm / FOCAL_LENGTH_MM
    return ground_width_m / image_width_px


def pixel_to_ned_offset(offset_x_px, offset_y_px,
                        altitude_m, image_width_px):
    gsd     = calculate_gsd(altitude_m, image_width_px)
    east_m  =  offset_x_px * gsd
    north_m = -offset_y_px * gsd
    return round(north_m, 4), round(east_m, 4)


# ============================================================
# QR CORNER HELPERS
# ============================================================

def normalize_corners(points):
    if points is None:
        return None
    try:
        corners = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        return corners[:4] if len(corners) >= 4 else None
    except (ValueError, TypeError):
        return None


def calculate_center(corners):
    return float(np.mean(corners[:, 0])), float(np.mean(corners[:, 1]))


# ============================================================
# REFERENCE STORAGE + MATCHING  (text / image / both)
# ============================================================

def extract_qr_crop(frame, corners):
    """
    Perspective-correct crop of the QR region.
    `frame` must be in the SAME coordinate system as `corners`
    (i.e. the original camera frame, since corners are scaled back).
    Returns CROP_SIZE x CROP_SIZE equalised grayscale, or None.
    """
    try:
        corners = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        if len(corners) < 4:
            return None

        dst = np.array([[0, 0],
                        [CROP_SIZE, 0],
                        [CROP_SIZE, CROP_SIZE],
                        [0, CROP_SIZE]], dtype=np.float32)

        M    = cv2.getPerspectiveTransform(corners[:4], dst)
        crop = cv2.warpPerspective(frame, M, (CROP_SIZE, CROP_SIZE))
        if crop.ndim == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(crop)

    except Exception as exc:
        logging.warning("Crop error: %s", exc)
        return None


def image_similarity(ref_gray, cand_gray):
    """
    Normalised correlation between two same-size crops, taking the best of
    4 rotations (corner order from the detector can rotate the crop).
    Returns a float in about -1..1; higher = more similar.
    """
    ref  = cv2.GaussianBlur(ref_gray, (5, 5), 0)
    cand = cv2.GaussianBlur(cand_gray, (5, 5), 0)
    best = -1.0
    for _ in range(4):
        score = float(cv2.matchTemplate(
            cand, ref, cv2.TM_CCOEFF_NORMED)[0][0])
        best = max(best, score)
        cand = cv2.rotate(cand, cv2.ROTATE_90_CLOCKWISE)
    return best


def store_reference(frame, corners, decoded_text):
    """
    Store the start QR as the reference (text + image crop).
    Called automatically from process_frame() when the start QR confirms,
    or call it manually.
    """
    global reference_text, reference_crop

    reference_text = decoded_text if decoded_text else ""
    logging.info("Reference text stored: '%s'", reference_text)

    crop = extract_qr_crop(frame, corners)
    if crop is not None:
        reference_crop = crop
        if SAVE_REFERENCE_CROP:
            cv2.imwrite(str(BASE_DIR / "reference_qr_crop.png"), crop)
        logging.info("Reference QR image stored")
    else:
        reference_crop = None
        logging.warning("Could not extract QR crop for image matching")


def clear_reference():
    global reference_text, reference_crop
    reference_text = None
    reference_crop = None


def reference_ready():
    """True if at least one reference method is available."""
    return reference_text is not None or reference_crop is not None


def is_target_qr(frame, corners, decoded_text):
    """
    Returns (is_match, method, info).

    "text"  : text only. Needs non-empty reference text.
    "image" : image similarity only.
    "both"  : if reference AND candidate both have non-empty text, text
              decides (a text mismatch is a rejection - no image fallback).
              Otherwise image similarity decides.
    """
    if not reference_ready():
        return False, "none", "no reference stored"

    ref_text  = (reference_text or "").strip()
    cand_text = (decoded_text or "").strip()

    # ── Text matching ──────────────────────────────────────────────
    if TEXT_MATCH_ENABLED and DETECTION_MODE in ("text", "both"):
        if DETECTION_MODE == "text":
            if not ref_text:
                return False, "text", "reference has no text"
            return cand_text == ref_text, "text", cand_text
        if ref_text and cand_text:
            return cand_text == ref_text, "text", cand_text

    # ── Image matching ─────────────────────────────────────────────
    if (IMAGE_MATCH_ENABLED and DETECTION_MODE in ("image", "both")
            and reference_crop is not None):
        crop = extract_qr_crop(frame, corners)
        if crop is not None:
            score = image_similarity(reference_crop, crop)
            match = score >= IMAGE_MATCH_THRESHOLD
            logging.debug("Image similarity %.3f (%s)", score,
                          "MATCH" if match else "no match")
            return match, "image", f"sim={score:.2f}"

    return False, "none", "no match method succeeded"


# ============================================================
# DETECTORS
# ============================================================

def detect_with_wechat(frame):
    qr_detector = get_detector()
    detections  = []
    if qr_detector is None:
        return detections
    try:
        texts, points = qr_detector.detectAndDecode(frame)
        if texts is None or points is None:
            return detections
        points = np.asarray(points)
        for i, text in enumerate(texts):
            # Empty text is only usable when image matching is allowed
            if not text and DETECTION_MODE == "text":
                continue
            if i >= len(points):
                continue
            corners = normalize_corners(points[i])
            if corners is None:
                continue
            cx, cy = calculate_center(corners)
            detections.append(
                Detection(text=text or "", cx=cx, cy=cy,
                          corners=corners, method="WeChat"))
    except cv2.error as exc:
        logging.warning("WeChat error: %s", exc)
    return detections


def detect_with_pyzbar(frame):
    if not PYZBAR_AVAILABLE:
        return []
    detections = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        for obj in pyzbar.decode(gray):
            text = obj.data.decode("utf-8", errors="replace")
            x, y, w, h = (obj.rect.left, obj.rect.top,
                          obj.rect.width, obj.rect.height)
            corners = np.array([[x, y], [x+w, y],
                                 [x+w, y+h], [x, y+h]],
                                dtype=np.float32)
            detections.append(
                Detection(text=text, cx=x+w/2, cy=y+h/2,
                          corners=corners, method="pyzbar"))
    except Exception as exc:
        logging.warning("pyzbar error: %s", exc)
    return detections


def detect_qrs(frame):
    detections = detect_with_wechat(frame)
    return detections if detections else detect_with_pyzbar(frame)


# ============================================================
# TRACKING  +  FSM STATE UPDATE
# ============================================================

def find_track(detection):
    for track in tracks.values():
        if track.text != detection.text:
            continue
        if (math.hypot(track.cx - detection.cx,
                       track.cy - detection.cy) < TRACK_MATCH_DISTANCE):
            return track
    return None


def update_tracks(detections, image_width_px, image_height_px,
                  update_fsm_state=True):
    """
    Update tracking state. When update_fsm_state is True (start-QR phase)
    also writes _confirmed_target and the offsets; the offsets are zeroed
    on frames where the confirmed target is not seen.
    In the delivery phase update_fsm_state is False and offsets come from
    _update_delivery_offset() instead.
    """
    global next_track_id, _confirmed_target
    global _last_north_m, _last_east_m

    for track in tracks.values():
        track.misses += 1

    unmatched   = set(tracks.keys())
    seen_target = False

    for det in detections:
        best_id   = None
        best_dist = TRACK_MATCH_DISTANCE

        for tid in unmatched:
            t = tracks[tid]
            if t.text != det.text:
                continue
            d = math.hypot(t.cx - det.cx, t.cy - det.cy)
            if d < best_dist:
                best_dist = d
                best_id   = tid

        if best_id is None:
            t = QRTrack(
                track_id=next_track_id,
                text=det.text,
                cx=det.cx, cy=det.cy,
                history=deque(maxlen=CONSENSUS_REQUIRED)
            )
            t.history.append(det.text)
            tracks[next_track_id] = t
            next_track_id += 1
        else:
            t = tracks[best_id]
            unmatched.remove(best_id)
            t.cx, t.cy = det.cx, det.cy
            t.misses   = 0
            t.history.append(det.text)

        # Consensus check
        if (len(t.history) >= CONSENSUS_REQUIRED
                and len(set(t.history)) == 1
                and not t.confirmed):
            t.confirmed = True
            if PRINT_ON_CONFIRMATION and not t.printed:
                print(f"\nQR CONFIRMED | ID={t.track_id} | Text={t.text!r}")
                t.printed = True

        # ── Update FSM globals (start-QR phase only) ──────────────
        if update_fsm_state and t.confirmed:
            if _confirmed_target is None:
                _confirmed_target = t.text

            if t.text == _confirmed_target:
                seen_target = True
                off_x = t.cx - image_width_px  // 2
                off_y = t.cy - image_height_px // 2
                n, e  = pixel_to_ned_offset(
                    off_x, off_y, _live_altitude_m, image_width_px)
                _last_north_m = n
                _last_east_m  = e

    # Target not visible this frame -> don't leave stale offsets
    if update_fsm_state and _confirmed_target is not None and not seen_target:
        _zero_offsets()

    # Remove stale tracks
    for tid in [tid for tid, t in tracks.items()
                if t.misses > MAX_MISSED_FRAMES]:
        del tracks[tid]


def _update_delivery_offset(frame, detections, image_w, image_h):
    """
    Delivery phase: find the detection that matches the reference and
    compute its offset. Zero everything if no match this frame.
    """
    global _last_north_m, _last_east_m, _target_visible, _last_match_info

    for det in detections:
        ok, method, info = is_target_qr(frame, det.corners, det.text)
        if ok:
            off_x = det.cx - image_w // 2
            off_y = det.cy - image_h // 2
            n, e  = pixel_to_ned_offset(
                off_x, off_y, _live_altitude_m, image_w)
            _last_north_m    = n
            _last_east_m     = e
            _target_visible  = True
            _last_match_info = f"{method}: {info}"
            return

    _zero_offsets()
    _target_visible  = False
    _last_match_info = "no match"


def process_frame(frame):
    """
    Process one camera frame (call once per captured frame from the FSM).

    Start-QR phase    : confirms the start QR, stores the reference
                        automatically, updates get_alignment_offset().
    Delivery phase    : (after start_delivery_search()) matches detections
                        against the reference; read is_target_visible()
                        and get_alignment_offset().

    Returns the list of detections (None if the frame was skipped by the
    blur gate).
    """
    original_h, original_w = frame.shape[:2]

    if BLUR_GATE_ENABLED:
        if sharpness_score(frame) < BLUR_THRESHOLD:
            return None

    processed, sx, sy = preprocess_frame(frame)
    detections        = detect_qrs(processed)

    # Back to original camera-frame resolution
    for det in detections:
        det.cx *= sx
        det.cy *= sy
        det.corners[:, 0] *= sx
        det.corners[:, 1] *= sy

    update_tracks(detections, original_w, original_h,
                  update_fsm_state=not _delivery_mode)

    if _delivery_mode:
        # corners are in original-frame coords -> pass the ORIGINAL frame
        _update_delivery_offset(frame, detections, original_w, original_h)

    elif _confirmed_target is not None and not reference_ready():
        for det in detections:
            if det.text == _confirmed_target:
                store_reference(frame, det.corners, det.text)
                break

    return detections


# ============================================================
# DISPLAY
# ============================================================

def put_text(img, text, org, colour=(255,255,255), scale=0.6, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0,0,0), thickness+2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, colour, thickness)


def draw_overlay(display, detections, blur_score, sharp, fps,
                 processing_ms, focus_mode, lens_pos,
                 original_width, original_height):

    image_cx = original_width  // 2
    image_cy = original_height // 2

    cv2.drawMarker(display, (image_cx, image_cy),
                   (255,255,0), cv2.MARKER_CROSS, 30, 2)

    put_text(display,
             f"FPS: {fps:.1f}  Processing: {processing_ms:.0f} ms",
             (10, 30))
    put_text(display,
             f"Blur: {blur_score:.1f}"
             + ("" if sharp else "  BLURRY"),
             (10, 60),
             (255,255,255) if sharp else (0,165,255))
    put_text(display,
             f"Focus: {focus_mode}  Alt: {_live_altitude_m:.1f}m  "
             f"Mode: {DETECTION_MODE}",
             (10, 90))

    if _delivery_mode:
        if _target_visible:
            put_text(display,
                     f"DELIVERY MATCH ({_last_match_info})  "
                     f"N={_last_north_m:.3f}m E={_last_east_m:.3f}m",
                     (10, 120), (0, 255, 0))
        else:
            put_text(display, f"DELIVERY: no match ({_last_match_info})",
                     (10, 120), (0, 0, 255))
    elif _confirmed_target is not None:
        put_text(display,
                 f"TARGET: {_confirmed_target!r}  "
                 f"N={_last_north_m:.3f}m E={_last_east_m:.3f}m  "
                 f"ref={'yes' if reference_ready() else 'no'}",
                 (10, 120), (0, 255, 0))
    else:
        put_text(display, "Searching...", (10, 120), (0, 0, 255))

    info_y = 150

    for det in detections:
        cx, cy  = int(det.cx), int(det.cy)
        corners = det.corners.astype(np.int32)
        t       = find_track(det)

        colour = (0,255,0) if (t and t.confirmed) else (0,165,255)
        status = "CONFIRMED" if (t and t.confirmed) else "DETECTING"

        cv2.polylines(display, [corners.reshape(-1,1,2)], True, colour, 3)
        cv2.circle(display, (cx, cy), 7, (0,0,255), -1)
        cv2.line(display, (image_cx, image_cy), (cx, cy), (255,0,255), 2)

        tid = t.track_id if t else -1
        put_text(display,
                 f"ID {tid} [{det.method}]: {det.text[:25]} {status}",
                 (max(10, cx+10), max(20, cy-10)), colour, 0.55)

        off_x = cx - image_cx
        off_y = cy - image_cy
        put_text(display,
                 f"ID {tid}: X={off_x}px Y={off_y}px",
                 (10, info_y), (220,220,220), 0.5, 1)
        info_y += 22

    put_text(display, f"Tracks: {len(tracks)}",
             (10, original_height - 20))

    return display


# ============================================================
# CAMERA INITIALIZATION
# ============================================================

def resolve_camera_index():
    """Pick the IMX519's camera index (Pi has multiple cameras)."""

    if CAMERA_INDEX is not None:
        return CAMERA_INDEX

    cameras = Picamera2.global_camera_info()
    logging.info("Cameras detected: %s",
                 [(c.get("Num"), c.get("Model")) for c in cameras])

    for cam_info in cameras:
        if "imx519" in str(cam_info.get("Model", "")).lower():
            logging.info("Selected imx519 at index %s", cam_info.get("Num"))
            return cam_info.get("Num", 0)

    logging.warning(
        "No camera with 'imx519' in its model name was found; "
        "defaulting to index 0. Set CAMERA_INDEX manually if this is wrong."
    )
    return 0


def apply_focus_mode(camera, mode):
    if isinstance(mode, (int, float)):
        camera.set_controls({
            "AfMode":       controls.AfModeEnum.Manual,
            "LensPosition": float(mode),
        })
        return f"manual ({mode:.2f} dpt)"
    if mode == "auto":
        camera.set_controls({"AfMode": controls.AfModeEnum.Auto})
        try:
            camera.autofocus_cycle()
        except Exception as exc:
            logging.warning("AF cycle failed: %s", exc)
        return "auto (one shot)"
    camera.set_controls({
        "AfMode":  controls.AfModeEnum.Continuous,
        "AfSpeed": controls.AfSpeedEnum.Fast,
    })
    return "continuous"


def start_camera():
    global sensor_crop_width_px

    camera_index = resolve_camera_index()
    camera = Picamera2(camera_num=camera_index)

    config = camera.create_video_configuration(
        main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT),
              "format": CAMERA_FORMAT},
        buffer_count=4
    )
    camera.configure(config)
    camera.start()

    focus_mode = apply_focus_mode(
        camera,
        LENS_POSITION if LENS_POSITION is not None else "continuous"
    )

    if EXPOSURE_US is not None:
        camera.set_controls({
            "AeEnable":    False,
            "ExposureTime": int(EXPOSURE_US),
            "AnalogueGain": float(ANALOGUE_GAIN or 4.0),
        })

    time.sleep(2)

    try:
        crop = camera.capture_metadata().get("ScalerCrop")
        if crop:
            sensor_crop_width_px = crop[2]
    except Exception as exc:
        logging.warning("ScalerCrop read failed: %s", exc)

    logging.info("Camera: %dx%d | focus=%s | crop_w=%d px",
                 CAMERA_WIDTH, CAMERA_HEIGHT,
                 focus_mode, sensor_crop_width_px)

    return camera, focus_mode


# ============================================================
# MAIN - standalone bench test
# ============================================================

def main():
    camera, focus_mode = start_camera()

    fps_counter        = 0
    fps                = 0.0
    fps_start          = last_fps_log = time.perf_counter()
    lens_pos           = 0.0
    lens_fetch_counter = 0

    logging.info("Keys: q quit | a AF once | c continuous | s save | "
                 "d delivery search (match vs reference) | x full reset")
    logging.info("Bench test: show QR A -> confirm -> press d -> show QR A "
                 "(MATCH) then QR B (no match)")

    try:
        while True:
            loop_start = time.perf_counter()

            frame = camera.capture_array()
            original_h, original_w = frame.shape[:2]

            blur_score = sharpness_score(frame)
            sharp      = (not BLUR_GATE_ENABLED or
                          blur_score > BLUR_THRESHOLD)

            detections = process_frame(frame) or []

            processing_ms = (time.perf_counter() - loop_start) * 1000.0

            fps_counter += 1
            now          = time.perf_counter()
            if now - fps_start >= 1.0:
                fps         = fps_counter / (now - fps_start)
                fps_counter = 0
                fps_start   = now

            if now - last_fps_log >= FPS_LOG_INTERVAL:
                logging.info("FPS=%.1f | proc=%.0f ms | blur=%.1f | "
                             "tracks=%d | target=%r | delivery=%s | "
                             "visible=%s | %s",
                             fps, processing_ms, blur_score,
                             len(tracks), _confirmed_target,
                             _delivery_mode, _target_visible,
                             _last_match_info)
                last_fps_log = now

            if SHOW_PREVIEW:
                display            = frame.copy()
                lens_fetch_counter += 1
                if lens_fetch_counter % 10 == 0:
                    lens_pos = (camera.capture_metadata()
                                .get("LensPosition", 0.0))
                display = draw_overlay(
                    display, detections, blur_score, sharp,
                    fps, processing_ms, focus_mode, lens_pos,
                    original_w, original_h
                )
                scale = DISPLAY_WIDTH / original_w
                if scale < 1.0:
                    display = cv2.resize(
                        display,
                        (DISPLAY_WIDTH, int(original_h * scale))
                    )
                cv2.imshow("Squadrone QR", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("a"):
                    focus_mode = apply_focus_mode(camera, "auto")
                elif key == ord("c"):
                    focus_mode = apply_focus_mode(camera, "continuous")
                elif key == ord("s"):
                    path = BASE_DIR / "capture.jpg"
                    cv2.imwrite(str(path), frame)
                    logging.info("Saved %s", path)
                elif key == ord("d"):
                    reset_detection(keep_reference=True)
                    start_delivery_search()
                elif key == ord("x"):
                    reset_detection(keep_reference=False)

    except KeyboardInterrupt:
        logging.info("Stopped by user")

    finally:
        camera.stop()
        camera.close()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()
        logging.info("Shutdown complete")
        if tracks:
            for t in tracks.values():
                logging.info("Track %d | %r | confirmed=%s",
                             t.track_id, t.text, t.confirmed)


if __name__ == "__main__":
    main()