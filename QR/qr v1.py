"""
SQUADRONE - ONBOARD QR DETECTION (fixed)

Hardware:
    Raspberry Pi 5
    Arducam IMX519 Autofocus (SKU B0371)

Primary detector : OpenCV WeChatQRCode
Fallback         : pyzbar

Changes from the previous version:
    - Model paths are absolute (relative to this file)
    - Autofocus is enabled correctly (camera.start() was called twice before);
      optional fixed LensPosition and exposure cap for flight
    - Removed the RGB->BGR conversion (Picamera2 "RGB888" is already BGR)
    - Default preprocessing is "original" (CLAHE/sharpen hurt WeChat)
    - Blur score is measured on the RAW frame, not the sharpened one
    - Blur gate can be disabled while calibrating
    - GSD uses the real sensor crop (ScalerCrop) and the output frame width
    - Overlay text no longer overlaps
    - Keys: q quit | a autofocus once | c continuous AF | s save frame

NOTE:
    This is a perception prototype, not flight-control software.
    Ground offsets are approximate and must be validated.
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

# Camera configuration
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FORMAT = "RGB888"          # NOTE: arrays are BGR-ordered already

# Focus:
#   None  -> continuous autofocus
#   float -> fixed focus in dioptres (1 / distance in metres)
#            e.g. 0.2 = 5 m, 1.0 = 1 m, 2.5 = 40 cm
LENS_POSITION = None

# Exposure (reduces motion blur). None = automatic.
# Example for flight: EXPOSURE_US = 4000, ANALOGUE_GAIN = 4.0
EXPOSURE_US = None
ANALOGUE_GAIN = None

# Optional image resizing before QR detection. None = disabled.
MAX_PROCESS_WIDTH = 1920

# Preprocessing options:
# "original" = colour frame untouched (recommended for WeChat)
# "clahe"    = grayscale + CLAHE
# "sharpen"  = grayscale + CLAHE + sharpening + blur
PREPROCESS_MODE = "original"

# Blur gate. Set BLUR_GATE_ENABLED = False while calibrating; watch the
# "Blur score" on screen for sharp vs blurry views, then pick a threshold.
BLUR_GATE_ENABLED = False
BLUR_THRESHOLD = 30.0

# QR confirmation and tracking
CONSENSUS_REQUIRED = 3
MAX_MISSED_FRAMES = 5
TRACK_MATCH_DISTANCE = 100

# Camera preview
SHOW_PREVIEW = True
DISPLAY_WIDTH = 1280              # preview window width (display only)

# GSD settings
ALTITUDE_M = 5.0                  # replace with live telemetry later
FOCAL_LENGTH_MM = 4.28
PIXEL_SIZE_UM = 1.22              # native IMX519 pixel pitch
SENSOR_FULL_WIDTH_PX = 4656       # native IMX519 active width

PRINT_ON_CONFIRMATION = True
FPS_LOG_INTERVAL = 2.0

# Camera selection. This Pi has multiple cameras attached (e.g. an imx500
# AI camera alongside the imx519). Picamera2() with no argument opens
# camera index 0, which may NOT be the IMX519 and will fail on AfMode.
# Set this to the correct index, found by running on the Pi:
#   python -c "from picamera2 import Picamera2; print(Picamera2.global_camera_info())"
# and looking for the entry with Model containing "imx519".
CAMERA_INDEX = None   # None = auto-detect by model name; set an int to force it

# WeChat model files
MODEL_DIR = BASE_DIR / "models"
WECHAT_DETECT_PROTOTXT = str(MODEL_DIR / "detect.prototxt")
WECHAT_DETECT_MODEL = str(MODEL_DIR / "detect.caffemodel")
WECHAT_SR_PROTOTXT = str(MODEL_DIR / "sr.prototxt")
WECHAT_SR_MODEL = str(MODEL_DIR / "sr.caffemodel")


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ============================================================
# OPTIONAL PYZBAR IMPORT
# ============================================================

try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except Exception as exc:
    pyzbar = None
    PYZBAR_AVAILABLE = False
    logging.warning("pyzbar unavailable; fallback disabled: %s", exc)


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


tracks = {}
next_track_id = 1

# Width (in native sensor pixels) covered by the output frame.
# Updated from camera metadata at startup.
sensor_crop_width_px = SENSOR_FULL_WIDTH_PX


# ============================================================
# WECHAT DETECTOR
# ============================================================

detector = None
wechat_failed = False


def get_detector():
    """Load WeChatQRCode once. Returns None if the models can't be loaded."""
    global detector, wechat_failed

    if detector is None and not wechat_failed:
        try:
            detector = cv2.wechat_qrcode_WeChatQRCode(
                WECHAT_DETECT_PROTOTXT,
                WECHAT_DETECT_MODEL,
                WECHAT_SR_PROTOTXT,
                WECHAT_SR_MODEL
            )
            logging.info("WeChatQRCode detector loaded successfully")
        except Exception as exc:
            wechat_failed = True
            logging.error(
                "WeChatQRCode could not load (%s). Using pyzbar only. "
                "Check %s", exc, MODEL_DIR
            )

    return detector


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_frame(frame):
    """
    Returns (processed BGR frame, scale_x, scale_y).
    scale_x/scale_y convert processed coordinates back to the
    original camera-frame coordinates.
    """

    original_h, original_w = frame.shape[:2]
    processed = frame

    if MAX_PROCESS_WIDTH is not None and original_w > MAX_PROCESS_WIDTH:
        scale = MAX_PROCESS_WIDTH / original_w
        processed = cv2.resize(
            frame,
            (int(original_w * scale), int(original_h * scale)),
            interpolation=cv2.INTER_AREA
        )

    processed_h, processed_w = processed.shape[:2]
    scale_x = original_w / processed_w
    scale_y = original_h / processed_h

    if PREPROCESS_MODE == "original":
        return processed, scale_x, scale_y

    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    if PREPROCESS_MODE in ("clahe", "sharpen"):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    if PREPROCESS_MODE == "sharpen":
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        gray = cv2.filter2D(gray, -1, kernel)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), scale_x, scale_y


# ============================================================
# BLUR DETECTION
# ============================================================

def sharpness_score(frame):
    """Laplacian variance of the raw frame (higher = sharper)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ============================================================
# GSD AND GROUND OFFSET
# ============================================================

def calculate_gsd(altitude_m, image_width_px):
    """
    Metres per output-frame pixel.

    ground width  = altitude * (physical width of sensor area used) / focal
    GSD           = ground width / output frame width in pixels

    The sensor area used comes from ScalerCrop (native pixels), so binned
    or cropped sensor modes are handled correctly.
    """

    used_sensor_width_mm = sensor_crop_width_px * (PIXEL_SIZE_UM / 1000.0)
    ground_width_m = altitude_m * used_sensor_width_mm / FOCAL_LENGTH_MM

    return ground_width_m / image_width_px


def pixel_to_ned_offset(offset_x_px, offset_y_px, altitude_m, image_width_px):
    """Approximate (north_m, east_m). Assumes nadir camera, image-up = north."""

    gsd = calculate_gsd(altitude_m, image_width_px)

    east_m = offset_x_px * gsd
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
        if len(corners) < 4:
            return None
        return corners[:4]
    except (ValueError, TypeError):
        return None


def calculate_center(corners):
    return float(np.mean(corners[:, 0])), float(np.mean(corners[:, 1]))


# ============================================================
# DETECTORS
# ============================================================

def detect_with_wechat(frame):
    qr_detector = get_detector()
    detections = []

    if qr_detector is None:
        return detections

    try:
        texts, points = qr_detector.detectAndDecode(frame)

        if texts is None or points is None:
            return detections

        points = np.asarray(points)

        for index, text in enumerate(texts):
            if not text or index >= len(points):
                continue

            corners = normalize_corners(points[index])
            if corners is None:
                continue

            cx, cy = calculate_center(corners)
            detections.append(
                Detection(text=text, cx=cx, cy=cy,
                          corners=corners, method="WeChat")
            )

    except cv2.error as exc:
        logging.warning("WeChat detection error: %s", exc)

    return detections


def detect_with_pyzbar(frame):
    if not PYZBAR_AVAILABLE:
        return []

    detections = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try:
        for obj in pyzbar.decode(gray):
            text = obj.data.decode("utf-8", errors="replace")

            x, y = obj.rect.left, obj.rect.top
            w, h = obj.rect.width, obj.rect.height

            corners = np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                dtype=np.float32
            )

            detections.append(
                Detection(text=text, cx=x + w / 2, cy=y + h / 2,
                          corners=corners, method="pyzbar")
            )

    except Exception as exc:
        logging.warning("pyzbar decode failed: %s", exc)

    return detections


def detect_qrs(frame):
    """WeChat first; pyzbar only if WeChat decoded nothing."""
    detections = detect_with_wechat(frame)

    if detections:
        return detections

    return detect_with_pyzbar(frame)


# ============================================================
# TRACKING
# ============================================================

def calculate_distance(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def update_tracks(detections):
    global next_track_id

    for track in tracks.values():
        track.misses += 1

    unmatched_track_ids = set(tracks.keys())

    for detection in detections:

        best_track_id = None
        best_distance = TRACK_MATCH_DISTANCE

        for track_id in unmatched_track_ids:
            track = tracks[track_id]

            if track.text != detection.text:
                continue

            d = calculate_distance(track.cx, track.cy,
                                   detection.cx, detection.cy)

            if d < best_distance:
                best_distance = d
                best_track_id = track_id

        if best_track_id is None:
            track = QRTrack(
                track_id=next_track_id,
                text=detection.text,
                cx=detection.cx,
                cy=detection.cy,
                history=deque(maxlen=CONSENSUS_REQUIRED)
            )
            track.history.append(detection.text)

            tracks[next_track_id] = track
            next_track_id += 1

        else:
            track = tracks[best_track_id]
            unmatched_track_ids.remove(best_track_id)

            track.cx = detection.cx
            track.cy = detection.cy
            track.misses = 0
            track.history.append(detection.text)

        if (
            len(track.history) >= CONSENSUS_REQUIRED
            and len(set(track.history)) == 1
            and not track.confirmed
        ):
            track.confirmed = True

            if PRINT_ON_CONFIRMATION and not track.printed:
                print(
                    f"\nQR CONFIRMED | ID={track.track_id} | "
                    f"Text={track.text}"
                )
                track.printed = True

    expired_ids = [
        track_id for track_id, track in tracks.items()
        if track.misses > MAX_MISSED_FRAMES
    ]

    for track_id in expired_ids:
        del tracks[track_id]


def find_track(detection):
    for track in tracks.values():
        if track.text != detection.text:
            continue

        if calculate_distance(track.cx, track.cy,
                              detection.cx, detection.cy) < TRACK_MATCH_DISTANCE:
            return track

    return None


# ============================================================
# DISPLAY
# ============================================================

def put_text(img, text, org, colour=(255, 255, 255), scale=0.6, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, colour, thickness)


def draw_overlay(display, detections, blur_score, sharp, fps,
                 processing_ms, focus_mode, lens_pos,
                 original_width, original_height):

    image_cx = original_width // 2
    image_cy = original_height // 2

    cv2.drawMarker(display, (image_cx, image_cy), (255, 255, 0),
                   cv2.MARKER_CROSS, 30, 2)

    # Status block, top-left (fixed lines so nothing overlaps)
    put_text(display, f"FPS: {fps:.1f}  Processing: {processing_ms:.0f} ms",
             (10, 30))
    put_text(display, f"Blur score: {blur_score:.1f}"
                      + ("" if sharp else "  (BELOW THRESHOLD)"),
             (10, 60), (255, 255, 255) if sharp else (0, 165, 255))
    put_text(display, f"Focus: {focus_mode}  LensPos: {lens_pos:.2f}",
             (10, 90))

    if not detections:
        put_text(display, "Searching...", (10, 120), (0, 0, 255))

    info_y = 150

    for detection in detections:

        cx = int(detection.cx)
        cy = int(detection.cy)
        corners = detection.corners.astype(np.int32)

        matched_track = find_track(detection)

        if matched_track and matched_track.confirmed:
            colour = (0, 255, 0)
            status = "CONFIRMED"
        else:
            colour = (0, 165, 255)
            status = "DETECTING"

        cv2.polylines(display, [corners.reshape(-1, 1, 2)], True, colour, 3)
        cv2.circle(display, (cx, cy), 7, (0, 0, 255), -1)
        cv2.line(display, (image_cx, image_cy), (cx, cy), (255, 0, 255), 2)

        offset_x = cx - image_cx
        offset_y = cy - image_cy

        north, east = pixel_to_ned_offset(
            offset_x, offset_y, ALTITUDE_M, original_width
        )

        track_id = matched_track.track_id if matched_track else -1

        put_text(display,
                 f"ID {track_id} [{detection.method}]: "
                 f"{detection.text[:25]} {status}",
                 (max(10, cx + 10), max(20, cy - 10)), colour, 0.6)

        put_text(display,
                 f"ID {track_id}: X={offset_x}px Y={offset_y}px",
                 (10, info_y), (220, 220, 220), 0.55, 1)
        info_y += 25

        if matched_track and matched_track.confirmed:
            put_text(display,
                     f"  Ground approx N={north}m E={east}m",
                     (10, info_y), (0, 255, 0), 0.55, 1)
            info_y += 25

    put_text(display, f"QR tracks: {len(tracks)}",
             (10, original_height - 20))

    return display


# ============================================================
# CAMERA INITIALIZATION
# ============================================================

def apply_focus_mode(camera, mode):
    """mode: 'continuous', 'auto' (one cycle) or a float LensPosition."""

    if isinstance(mode, (int, float)):
        camera.set_controls({
            "AfMode": controls.AfModeEnum.Manual,
            "LensPosition": float(mode),
        })
        return f"manual ({mode:.2f} dpt)"

    if mode == "auto":
        camera.set_controls({"AfMode": controls.AfModeEnum.Auto})
        try:
            ok = camera.autofocus_cycle()
            logging.info("Autofocus cycle success: %s", ok)
        except Exception as exc:
            logging.warning("Autofocus cycle failed: %s", exc)
        return "auto (one shot)"

    camera.set_controls({
        "AfMode": controls.AfModeEnum.Continuous,
        "AfSpeed": controls.AfSpeedEnum.Fast,
    })
    return "continuous"


def resolve_camera_index():
    """
    Pick the IMX519's camera index. Multiple cameras may be attached
    (e.g. an imx500 AI camera alongside the imx519), and Picamera2()
    with no argument opens index 0, which may be the wrong sensor.
    """

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


def start_camera():
    """Initialize the Arducam IMX519 through Picamera2."""

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
            "AeEnable": False,
            "ExposureTime": int(EXPOSURE_US),
            "AnalogueGain": float(ANALOGUE_GAIN or 4.0),
        })

    # Allow exposure and autofocus to settle.
    time.sleep(2)

    # Read how much of the sensor the output frame actually covers.
    try:
        crop = camera.capture_metadata().get("ScalerCrop")
        if crop:
            sensor_crop_width_px = crop[2]
    except Exception as exc:
        logging.warning("Could not read ScalerCrop: %s", exc)

    logging.info("Camera started: %dx%d | focus=%s | sensor crop width=%d px",
                 CAMERA_WIDTH, CAMERA_HEIGHT, focus_mode,
                 sensor_crop_width_px)

    return camera, focus_mode


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    camera, focus_mode = start_camera()

    frame_counter = 0
    fps_counter = 0
    fps = 0.0
    fps_start = time.perf_counter()
    last_fps_log = time.perf_counter()

    logging.info("QR detection started. Keys: q quit | a AF once | "
                 "c continuous AF | s save frame")

    try:
        while True:

            loop_start = time.perf_counter()

            # Picamera2 "RGB888" arrays are already BGR-ordered for OpenCV.
            frame = camera.capture_array()

            original_height, original_width = frame.shape[:2]

            # Blur score from the RAW frame.
            blur_score = sharpness_score(frame)
            sharp = (not BLUR_GATE_ENABLED) or (blur_score > BLUR_THRESHOLD)

            processed, scale_x, scale_y = preprocess_frame(frame)

            detections = []

            if sharp:
                detections = detect_qrs(processed)

                # Convert coordinates back to original frame.
                for detection in detections:
                    detection.cx *= scale_x
                    detection.cy *= scale_y
                    detection.corners[:, 0] *= scale_x
                    detection.corners[:, 1] *= scale_y

                update_tracks(detections)
            else:
                update_tracks([])

            processing_ms = (time.perf_counter() - loop_start) * 1000.0

            frame_counter += 1
            fps_counter += 1

            now = time.perf_counter()
            elapsed = now - fps_start

            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = now

            if now - last_fps_log >= FPS_LOG_INTERVAL:
                logging.info(
                    "FPS=%.1f | Processing=%.1f ms | Blur=%.1f | Tracks=%d",
                    fps, processing_ms, blur_score, len(tracks)
                )
                last_fps_log = now

            if SHOW_PREVIEW:
                display = frame.copy()

                lens_pos = camera.capture_metadata().get("LensPosition", 0.0)

                display = draw_overlay(
                    display, detections, blur_score, sharp, fps,
                    processing_ms, focus_mode, lens_pos,
                    original_width, original_height
                )

                scale = DISPLAY_WIDTH / original_width
                if scale < 1.0:
                    display = cv2.resize(
                        display,
                        (DISPLAY_WIDTH, int(original_height * scale))
                    )

                cv2.imshow("Squadrone Onboard QR Detection", display)

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

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received.")

    finally:
        camera.stop()
        camera.close()

        if SHOW_PREVIEW:
            cv2.destroyAllWindows()

        logging.info("Camera stopped.")

        if tracks:
            logging.info("Final QR tracks:")

            for track in tracks.values():
                logging.info("ID=%d | Text=%s | Confirmed=%s",
                             track.track_id, track.text, track.confirmed)


if __name__ == "__main__":
    main()