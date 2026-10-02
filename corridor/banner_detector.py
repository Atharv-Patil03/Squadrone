
import cv2
import numpy as np
import time
from collections import deque
from picamera2 import Picamera2

# ==================================================
# CONFIGURATION
# ==================================================

WIDTH = 640
HEIGHT = 480

LOWER_GREEN = np.array([35, 80, 50])
UPPER_GREEN = np.array([85, 255, 255])

MIN_AREA = 500
MIN_ASPECT_RATIO = 0.3
MAX_ASPECT_RATIO = 3.5
MIN_RECTANGULARITY = 0.35

HISTORY_LENGTH = 10
REQUIRED_DETECTIONS = 7
LOSS_TIMEOUT = 0.5

# ==================================================
# CAMERA SETUP
# ==================================================

def start_camera():
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
    return camera

# ==================================================
# BANNER DETECTION
# ==================================================

def detect_banner(frame, kernel):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv, LOWER_GREEN, UPPER_GREEN
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

    best = None
    best_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w == 0 or h == 0:
            continue

        aspect_ratio = w / h
        rectangularity = area / (w * h)

        if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
            continue

        if rectangularity < MIN_RECTANGULARITY:
            continue

        if area > best_area:
            moments = cv2.moments(contour)

            if moments["m00"] == 0:
                continue

            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            best_area = area
            best = {
                "box": (x, y, w, h),
                "center": (cx, cy),
                "area": area,
                "aspect_ratio": aspect_ratio,
                "rectangularity": rectangularity
            }

    return best, mask

# ==================================================
# MAIN
# ==================================================

def main():
    camera = start_camera()
    kernel = np.ones((5, 5), np.uint8)

    detection_history = deque(maxlen=HISTORY_LENGTH)
    last_detection_time = 0
    fps = 0
    previous_time = time.monotonic()

    print("Banner detector started. Press q to quit.")

    try:
        while True:
            rgb = camera.capture_array()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            height, width = frame.shape[:2]
            center_x = width // 2
            center_y = height // 2

            banner, mask = detect_banner(frame, kernel)
            detected = banner is not None

            detection_history.append(detected)

            if detected:
                last_detection_time = time.monotonic()

            stable = (
                len(detection_history) == HISTORY_LENGTH
                and sum(detection_history) >= REQUIRED_DETECTIONS
                and time.monotonic() - last_detection_time <= LOSS_TIMEOUT
            )

            # Draw image centre
            cv2.drawMarker(
                frame,
                (center_x, center_y),
                (255, 0, 0),
                cv2.MARKER_CROSS,
                20,
                2
            )

            if detected:
                x, y, w, h = banner["box"]
                cx, cy = banner["center"]

                offset_x = cx - center_x
                offset_y = cy - center_y

                cv2.rectangle(
                    frame, (x, y), (x + w, y + h),
                    (0, 255, 0), 2
                )
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                cv2.putText(
                    frame,
                    f"Offset: X={offset_x} Y={offset_y}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2
                )
                cv2.putText(
                    frame,
                    f"Area: {int(banner['area'])} px",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2
                )
                cv2.putText(
                    frame,
                    f"Shape: {banner['rectangularity']:.2f}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2
                )

            # Detection status
            if stable:
                status = "BANNER DETECTED - STABLE"
                color = (0, 255, 0)
            elif detected:
                status = "VERIFYING DETECTION"
                color = (0, 255, 255)
            else:
                status = "NO VALID BANNER"
                color = (0, 0, 255)

            cv2.putText(
                frame, status, (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
            )

            # FPS calculation
            current_time = time.monotonic()
            elapsed = current_time - previous_time
            previous_time = current_time

            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0 else 0.9 * fps + 0.1 * instant_fps

            cv2.putText(
                frame, f"FPS: {fps:.1f}",
                (width - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )

            cv2.imshow("Banner Detection", frame)
            cv2.imshow("HSV Mask", mask)

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