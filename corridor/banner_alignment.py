
import cv2
import numpy as np
import time
from collections import deque
from picamera2 import Picamera2

# ============================================================
# CONFIGURATION
# ============================================================

WIDTH = 640
HEIGHT = 480

# Aerothon methodology HSV range
LOWER_GREEN = np.array([40, 80, 80], dtype=np.uint8)
UPPER_GREEN = np.array([80, 255, 255], dtype=np.uint8)

MIN_AREA = 500
MIN_RECTANGULARITY = 0.35
MIN_ASPECT_RATIO = 0.30
MAX_ASPECT_RATIO = 3.50

HISTORY_LENGTH = 10
REQUIRED_DETECTIONS = 7
DETECTION_LOSS_TIMEOUT = 0.5

# Image-centre tolerance in pixels.
# This is a test threshold, not a validated flight threshold.
ALIGNMENT_TOLERANCE_PX = 20

# ============================================================
# BANNER DETECTION
# ============================================================

def detect_banner(frame, kernel):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        LOWER_GREEN,
        UPPER_GREEN
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            continue

        aspect_ratio = w / h
        rectangularity = area / (w * h)

        if not MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO:
            continue

        if rectangularity < MIN_RECTANGULARITY:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        # Contour centroid
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        # Apparent orientation of the detected region.
        # This is not automatically the drone heading error.
        rect = cv2.minAreaRect(contour)
        angle = rect[2]

        candidates.append({
            "contour": contour,
            "box": (x, y, w, h),
            "center": (cx, cy),
            "area": area,
            "aspect_ratio": aspect_ratio,
            "rectangularity": rectangularity,
            "angle": angle
        })

    if not candidates:
        return None, mask

    # Select the largest region that passes the filters
    return max(candidates, key=lambda item: item["area"]), mask


# ============================================================
# ALIGNMENT ESTIMATE
# ============================================================

def calculate_alignment(banner, frame_width):
    if banner is None:
        return {
            "valid": False,
            "offset_x": None,
            "alignment_status": "NO_BANNER",
            "yaw_error": None
        }

    image_center_x = frame_width // 2
    banner_center_x = banner["center"][0]

    offset_x = banner_center_x - image_center_x

    if abs(offset_x) <= ALIGNMENT_TOLERANCE_PX:
        status = "IMAGE_CENTERED"
    elif offset_x > 0:
        status = "BANNER_RIGHT"
    else:
        status = "BANNER_LEFT"

    return {
        "valid": True,
        "offset_x": offset_x,
        "alignment_status": status,
        # Not a calibrated heading estimate.
        "yaw_error": None
    }


# ============================================================
# MAIN
# ============================================================

def main():
    camera = Picamera2()

    config = camera.create_preview_configuration(
        main={
            "size": (WIDTH, HEIGHT),
            "format": "RGB888"
        },
        buffer_count=4
    )

    camera.configure(config)
    camera.start()
    time.sleep(2)

    kernel = np.ones((5, 5), dtype=np.uint8)
    history = deque(maxlen=HISTORY_LENGTH)

    last_detection_time = 0.0
    previous_time = time.monotonic()
    fps = 0.0

    print("Banner detection and alignment test started.")
    print("Press q or Esc to exit.")

    try:
        while True:
            rgb = camera.capture_array()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            height, width = frame.shape[:2]
            image_center = (width // 2, height // 2)

            banner, mask = detect_banner(frame, kernel)
            detected = banner is not None

            now = time.monotonic()

            history.append(detected)

            if detected:
                last_detection_time = now

            stable = (
                len(history) == HISTORY_LENGTH
                and sum(history) >= REQUIRED_DETECTIONS
                and now - last_detection_time <= DETECTION_LOSS_TIMEOUT
            )

            alignment = calculate_alignment(banner, width)

            # Draw image centre
            cv2.drawMarker(
                frame,
                image_center,
                (255, 0, 0),
                cv2.MARKER_CROSS,
                24,
                2
            )

            if banner is not None:
                x, y, w, h = banner["box"]
                cx, cy = banner["center"]

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    frame, (cx, cy), 6, (0, 0, 255), -1
                )

                cv2.putText(
                    frame,
                    f"Offset X: {alignment['offset_x']} px",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Area: {int(banner['area'])} px",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Aspect: {banner['aspect_ratio']:.2f}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Apparent angle: {banner['angle']:.1f}",
                    (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

            # Stable detection and image-centering status
            if stable and alignment["valid"]:
                if alignment["alignment_status"] == "IMAGE_CENTERED":
                    status = "STABLE - IMAGE CENTERED"
                    color = (0, 255, 0)
                else:
                    status = "STABLE - ALIGNMENT NEEDED"
                    color = (0, 255, 255)
            elif detected:
                status = "VERIFYING BANNER"
                color = (0, 255, 255)
            else:
                status = "NO VALID BANNER"
                color = (0, 0, 255)

            cv2.putText(
                frame,
                status,
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

            # FPS estimate
            current_time = time.monotonic()
            dt = current_time - previous_time
            previous_time = current_time

            if dt > 0:
                instantaneous_fps = 1.0 / dt
                fps = (
                    instantaneous_fps if fps == 0
                    else 0.9 * fps + 0.1 * instantaneous_fps
                )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (width - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("Banner Detection and Alignment", frame)
            cv2.imshow("Green HSV Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        print("Stopped by user.")

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Camera stopped.")


if __name__ == "__main__":
    main()