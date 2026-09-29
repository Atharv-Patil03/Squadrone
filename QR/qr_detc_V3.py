
"""
SQUADRONE - ONBOARD QR DETECTION
Hardware:
    Raspberry Pi 5
    Arducam IMX519 Autofocus (SKU B0371)

Primary detector:
    OpenCV WeChatQRCode

Fallback:
    pyzbar

Features:
    - Picamera2 camera capture
    - Configurable resolution
    - Optional resizing for processing
    - CLAHE, sharpening and Gaussian blur
    - Blur detection
    - WeChatQRCode primary detection
    - pyzbar fallback
    - Multiple QR detection and tracking
    - Consecutive-frame confirmation
    - Missed-frame tolerance
    - GSD-based approximate ground offset
    - Optional preview
    - FPS and processing-time monitoring

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

from picamera2 import Picamera2


# ============================================================
# CONFIGURATION
# ============================================================

# Camera configuration
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FORMAT = "RGB888"

# Optional image resizing before QR detection.
# Set to None to disable resizing.
MAX_PROCESS_WIDTH = 1920

# Preprocessing options:
# "original" = grayscale only
# "clahe"    = grayscale + CLAHE
# "sharpen"  = grayscale + CLAHE + sharpening + blur
PREPROCESS_MODE = "sharpen"

# Blur detection
BLUR_THRESHOLD = 100.0

# QR confirmation and tracking
CONSENSUS_REQUIRED = 3
MAX_MISSED_FRAMES = 5
TRACK_MATCH_DISTANCE = 100

# Camera preview
SHOW_PREVIEW = True

# GSD settings
# Fixed test altitude; replace with live telemetry later.
ALTITUDE_M = 5.0

# Arducam IMX519 specifications
FOCAL_LENGTH_MM = 4.28
PIXEL_SIZE_UM = 1.22

# Print confirmed QR only once per track
PRINT_ON_CONFIRMATION = True

# Performance monitoring
FPS_LOG_INTERVAL = 2.0

# WeChat model files
WECHAT_DETECT_PROTOTXT = "models/detect.prototxt"
WECHAT_DETECT_MODEL = "models/detect.caffemodel"
WECHAT_SR_PROTOTXT = "models/sr.prototxt"
WECHAT_SR_MODEL = "models/sr.caffemodel"


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
    logging.warning(
        "pyzbar unavailable; fallback disabled: %s", exc
    )


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


# ============================================================
# INITIALIZE WECHAT DETECTOR
# ============================================================

detector = None


def get_detector():
    global detector

    if detector is None:
        detector = cv2.wechat_qrcode_WeChatQRCode(
            WECHAT_DETECT_PROTOTXT,
            WECHAT_DETECT_MODEL,
            WECHAT_SR_PROTOTXT,
            WECHAT_SR_MODEL
        )

        logging.info("WeChatQRCode detector loaded successfully")

    return detector


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_frame(frame):
    """
    Preprocess the captured frame.

    Returns:
        processed frame
        scale_x
        scale_y

    scale_x and scale_y convert processed coordinates
    back to original camera-frame coordinates.
    """

    original_h, original_w = frame.shape[:2]

    # Optional resizing to reduce processing load.
    if MAX_PROCESS_WIDTH is not None:
        if original_w > MAX_PROCESS_WIDTH:
            scale = MAX_PROCESS_WIDTH / original_w

            new_w = int(original_w * scale)
            new_h = int(original_h * scale)

            frame = cv2.resize(
                frame,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

    processed_h, processed_w = frame.shape[:2]

    scale_x = original_w / processed_w
    scale_y = original_h / processed_h

    # Convert to grayscale.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if PREPROCESS_MODE == "clahe" or PREPROCESS_MODE == "sharpen":
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        gray = clahe.apply(gray)

    if PREPROCESS_MODE == "sharpen":
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)

        gray = cv2.filter2D(gray, -1, kernel)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    processed = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )

    return processed, scale_x, scale_y


# ============================================================
# BLUR DETECTION
# ============================================================

def is_sharp(frame, threshold=BLUR_THRESHOLD):
    """
    Laplacian variance sharpness measure.

    Higher score generally means a sharper image.
    The threshold must be calibrated using real footage.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return score > threshold, round(score, 1)


# ============================================================
# GSD AND GROUND OFFSET
# ============================================================

def calculate_gsd(altitude_m, image_width_px):
    """
    Calculate approximate ground sampling distance.

    Uses:
        Pixel pitch = 1.22 micrometres
        Focal length = 4.28 mm

    Assumes:
        - Camera points vertically downward
        - Ground is flat
        - Image uses the full sensor width
        - Altitude is above the ground

    Returns:
        GSD in metres per original image pixel
    """

    pixel_size_mm = PIXEL_SIZE_UM / 1000.0

    gsd = (
        altitude_m * pixel_size_mm
    ) / FOCAL_LENGTH_MM

    # Convert to metres per pixel.
    return gsd


def pixel_to_ned_offset(
    offset_x_px,
    offset_y_px,
    altitude_m,
    image_width_px
):
    """
    Convert pixel offsets into approximate ground offsets.

    The GSD is based on the native pixel pitch. When using
    a resized image, coordinates must first be converted
    back to the original camera-frame coordinates.

    Returns:
        north_m, east_m
    """

    gsd = calculate_gsd(
        altitude_m,
        image_width_px
    )

    east_m = offset_x_px * gsd
    north_m = -offset_y_px * gsd

    return round(north_m, 4), round(east_m, 4)


# ============================================================
# QR CORNER NORMALIZATION
# ============================================================

def normalize_corners(points):
    """
    Normalize detector corner data into a 4x2 array.
    Handles different OpenCV output shapes.
    """

    if points is None:
        return None

    try:
        corners = np.asarray(
            points,
            dtype=np.float32
        ).reshape(-1, 2)

        if len(corners) < 4:
            return None

        return corners[:4]

    except (ValueError, TypeError):
        return None


def calculate_center(corners):
    cx = float(np.mean(corners[:, 0]))
    cy = float(np.mean(corners[:, 1]))

    return cx, cy


# ============================================================
# WECHAT QR DETECTION - PRIMARY
# ============================================================

def detect_with_wechat(frame):
    """
    Primary QR detector.

    Returns a list of decoded QR detections.
    """

    qr_detector = get_detector()
    detections = []

    try:
        texts, points = qr_detector.detectAndDecode(frame)

        if texts is None or points is None:
            return detections

        points = np.asarray(points)

        for index, text in enumerate(texts):
            if not text:
                continue

            if index >= len(points):
                continue

            corners = normalize_corners(points[index])

            if corners is None:
                continue

            cx, cy = calculate_center(corners)

            detections.append(
                Detection(
                    text=text,
                    cx=cx,
                    cy=cy,
                    corners=corners,
                    method="WeChat"
                )
            )

    except cv2.error as exc:
        logging.warning("WeChat detection error: %s", exc)

    return detections


# ============================================================
# PYZBAR FALLBACK
# ============================================================

def detect_with_pyzbar(frame):
    """
    Fallback detector.

    Called only if WeChatQRCode returns no decoded QR.
    """

    if not PYZBAR_AVAILABLE:
        return []

    detections = []

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    try:
        decoded_objects = pyzbar.decode(gray)

        for obj in decoded_objects:
            text = obj.data.decode(
                "utf-8",
                errors="replace"
            )

            x = obj.rect.left
            y = obj.rect.top
            w = obj.rect.width
            h = obj.rect.height

            cx = x + w / 2
            cy = y + h / 2

            corners = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ], dtype=np.float32)

            detections.append(
                Detection(
                    text=text,
                    cx=cx,
                    cy=cy,
                    corners=corners,
                    method="pyzbar"
                )
            )

    except Exception as exc:
        logging.warning("pyzbar decode failed: %s", exc)

    return detections


# ============================================================
# PRIMARY + FALLBACK DETECTION
# ============================================================

def detect_qrs(frame):
    """
    WeChatQRCode is always tried first.
    pyzbar is used only if WeChat returns no decoded QR.
    """

    detections = detect_with_wechat(frame)

    if detections:
        return detections

    return detect_with_pyzbar(frame)


# ============================================================
# TRACKING
# ============================================================

def calculate_distance(x1, y1, x2, y2):
    return math.hypot(
        x1 - x2,
        y1 - y2
    )


def update_tracks(detections):
    """
    Match QR detections to existing tracks.

    Matching uses:
        - Decoded text
        - Distance between QR centres

    Each track has its own confirmation history.
    """

    global next_track_id

    # Mark all tracks as missed before matching this frame.
    for track in tracks.values():
        track.misses += 1

    unmatched_track_ids = set(tracks.keys())

    for detection in detections:

        best_track_id = None
        best_distance = TRACK_MATCH_DISTANCE

        # Match only tracks with the same decoded text.
        for track_id in unmatched_track_ids:
            track = tracks[track_id]

            if track.text != detection.text:
                continue

            d = calculate_distance(
                track.cx,
                track.cy,
                detection.cx,
                detection.cy
            )

            if d < best_distance:
                best_distance = d
                best_track_id = track_id

        if best_track_id is None:
            # Create a new track.
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
            # Update existing track.
            track = tracks[best_track_id]
            unmatched_track_ids.remove(best_track_id)

            track.cx = detection.cx
            track.cy = detection.cy
            track.misses = 0

            track.history.append(detection.text)

        # Confirm when enough matching detections have arrived.
        if (
            len(track.history) >= CONSENSUS_REQUIRED
            and len(set(track.history)) == 1
        ):
            if not track.confirmed:
                track.confirmed = True

                if PRINT_ON_CONFIRMATION and not track.printed:
                    print(
                        f"\nQR CONFIRMED | "
                        f"ID={track.track_id} | "
                        f"Text={track.text}"
                    )
                    track.printed = True

    # Remove tracks that have been missing too long.
    expired_ids = [
        track_id
        for track_id, track in tracks.items()
        if track.misses > MAX_MISSED_FRAMES
    ]

    for track_id in expired_ids:
        del tracks[track_id]


# ============================================================
# DISPLAY
# ============================================================

def draw_overlay(
    display,
    detections,
    scale_x,
    scale_y,
    blur_score,
    fps,
    processing_ms,
    original_width,
    original_height
):
    """
    Draw camera centre, QR boxes, labels and status.
    """

    image_cx = original_width // 2
    image_cy = original_height // 2

    cv2.drawMarker(
        display,
        (image_cx, image_cy),
        (255, 255, 0),
        cv2.MARKER_CROSS,
        20,
        2
    )

    for detection in detections:

        # Detection coordinates are in original-frame pixels.
        cx = int(detection.cx)
        cy = int(detection.cy)

        corners = detection.corners.astype(np.int32)

        # Find corresponding track.
        matched_track = None

        for track in tracks.values():
            if track.text != detection.text:
                continue

            d = calculate_distance(
                track.cx,
                track.cy,
                detection.cx,
                detection.cy
            )

            if d < TRACK_MATCH_DISTANCE:
                matched_track = track
                break

        if matched_track and matched_track.confirmed:
            colour = (0, 255, 0)
            status = "CONFIRMED"
        else:
            colour = (0, 165, 255)
            status = "DETECTING"

        cv2.polylines(
            display,
            [corners.reshape(-1, 1, 2)],
            True,
            colour,
            2
        )

        cv2.circle(
            display,
            (cx, cy),
            7,
            (0, 0, 255),
            -1
        )

        cv2.line(
            display,
            (image_cx, image_cy),
            (cx, cy),
            (255, 0, 255),
            2
        )

        offset_x = cx - image_cx
        offset_y = cy - image_cy

        north, east = pixel_to_ned_offset(
            offset_x,
            offset_y,
            ALTITUDE_M,
            original_width
        )

        track_id = (
            matched_track.track_id
            if matched_track
            else -1
        )

        label = (
            f"ID {track_id}: "
            f"{detection.text[:25]} "
            f"{status}"
        )

        cv2.putText(
            display,
            label,
            (max(10, cx + 10), max(20, cy - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            2
        )

        cv2.putText(
            display,
            f"Offset X={offset_x}px Y={offset_y}px",
            (10, 105 + 25 * track_id if track_id >= 0 else 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1
        )

        if matched_track and matched_track.confirmed:
            cv2.putText(
                display,
                f"Ground approx N={north}m E={east}m",
                (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1
            )

    cv2.putText(
        display,
        f"FPS: {fps:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Blur score: {blur_score:.1f}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Processing: {processing_ms:.1f} ms",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"QR tracks: {len(tracks)}",
        (10, original_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    return display


# ============================================================
# CAMERA INITIALIZATION
# ============================================================

def start_camera():
    """
    Initialize the Arducam IMX519 through Picamera2.
    """

    camera = Picamera2()

    config = camera.create_video_configuration(
        main={
            "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
            "format": CAMERA_FORMAT
        },
        buffer_count=4
    )

    camera.configure(config)
    from libcamera import controls

    camera.start()
    camera.set_controls({
        "AfMode": controls.AfModeEnum.Continuous,
        "AfSpeed": controls.AfSpeedEnum.Fast,
    })
    
    camera.start()

    # Allow exposure and autofocus to settle.
    time.sleep(2)

    logging.info(
        "Camera started: %dx%d",
        CAMERA_WIDTH,
        CAMERA_HEIGHT
    )

    return camera


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    camera = start_camera()

    frame_counter = 0
    fps_counter = 0

    fps = 0.0
    fps_start = time.perf_counter()

    last_fps_log = time.perf_counter()

    logging.info("QR detection started. Press Q to quit.")

    try:
        while True:

            loop_start = time.perf_counter()

            # Capture RGB frame from Picamera2.
            rgb_frame = camera.capture_array()

            # Convert RGB to BGR for OpenCV.
            frame = cv2.cvtColor(
                rgb_frame,
                cv2.COLOR_RGB2BGR
            )

            original_height, original_width = frame.shape[:2]

            display = frame.copy()

            image_cx = original_width // 2
            image_cy = original_height // 2

            # Preprocessing.
            processed, scale_x, scale_y = preprocess_frame(frame)

            # Blur check on the processed image.
            sharp, blur_score = is_sharp(processed)

            detections = []

            if sharp:
                # WeChat first, pyzbar only if needed.
                detections = detect_qrs(processed)

                # Convert coordinates back to original frame.
                for detection in detections:
                    detection.cx *= scale_x
                    detection.cy *= scale_y
                    detection.corners[:, 0] *= scale_x
                    detection.corners[:, 1] *= scale_y

                update_tracks(detections)

            else:
                # A blurry frame is treated as a missed detection.
                update_tracks([])

            processing_ms = (
                time.perf_counter() - loop_start
            ) * 1000.0

            # FPS calculation.
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
                    "FPS=%.1f | Processing=%.1f ms | Tracks=%d",
                    fps,
                    processing_ms,
                    len(tracks)
                )
                last_fps_log = now

            # Display status.
            if SHOW_PREVIEW:
                if not sharp:
                    cv2.putText(
                        display,
                        f"BLURRY ({blur_score})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 165, 255),
                        2
                    )

                else:
                    if not detections:
                        cv2.putText(
                            display,
                            "Searching...",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                display = draw_overlay(
                    display,
                    detections,
                    scale_x,
                    scale_y,
                    blur_score,
                    fps,
                    processing_ms,
                    original_width,
                    original_height
                )

                cv2.imshow(
                    "Squadrone Onboard QR Detection",
                    display
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

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
                logging.info(
                    "ID=%d | Text=%s | Confirmed=%s",
                    track.track_id,
                    track.text,
                    track.confirmed
                )


if __name__ == "__main__":
    main()