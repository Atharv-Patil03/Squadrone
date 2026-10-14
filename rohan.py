import cv2
import numpy as np
import math
import time
from collections import deque

# HSV ranges
LOWER_GREEN = np.array([35, 80, 50])
UPPER_GREEN = np.array([85, 255, 255])

# Red wraps around both ends of OpenCV HSV hue; thresholds are relaxed to recover red detections.
LOWER_RED_1 = np.array([0, 100, 50])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 100, 50])
UPPER_RED_2 = np.array([180, 255, 255])

MIN_CONTOUR_AREA = 500

# Tracking settings (no centre-history trail is drawn).
TRAIL_LENGTH = 150
MOVEMENT_THRESHOLD = 5
STATIONARY_TIME = 0.7

# Raspberry Pi Camera Module / Raspberry Pi AI Camera (IMX500)
# Requires Picamera2 installed on the Raspberry Pi OS system Python.
try:
    from picamera2 import Picamera2
except ImportError:
    print("ERROR: Picamera2 is not installed. Install it on Raspberry Pi OS with:")
    print("sudo apt update && sudo apt install -y python3-picamera2")
    raise SystemExit(1)

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(camera_config)
picam2.start()
time.sleep(2.0)

print("Picamera2 started. Waiting for frames...")
frame = None
for i in range(10):
    try:
        frame_rgb = picam2.capture_array()
        if frame_rgb is not None and frame_rgb.size > 0:
            # Convert the Picamera2 RGB frame to OpenCV's BGR convention.
            frame = frame_rgb  # Picamera2 RGB888 array is BGR-ordered for OpenCV
            print("First valid frame received!")
            break
    except Exception as exc:
        print(f"Frame {i + 1} capture failed: {exc}")
    time.sleep(0.3)

if frame is None:
    print("ERROR: No frames received after 10 attempts.")
    picam2.stop()
    raise SystemExit(1)

# Use the actual captured frame size; don't reconfigure the active stream.
frame_height, frame_width = frame.shape[:2]
print("--------------------------------------")
print("Camera started")
print(f"Resolution: {frame_width} x {frame_height}")
print("--------------------------------------")

kernel = np.ones((5, 5), np.uint8)

tracking = {
    "GREEN": {
        "history": deque(maxlen=TRAIL_LENGTH),
        "previous_cx": None,
        "previous_cy": None,
        "last_movement_time": time.time(),
        "stationary": False,
    },
    "RED": {
        "history": deque(maxlen=TRAIL_LENGTH),
        "previous_cx": None,
        "previous_cy": None,
        "last_movement_time": time.time(),
        "stationary": False,
    },
}

# OpenCV drawing colours are BGR.
OBJECT_COLORS = {"GREEN": (0, 255, 0), "RED": (0, 0, 255)}

try:
    consecutive_read_failures = 0
    while True:
        try:
            frame_rgb = picam2.capture_array()
            if frame_rgb is None or frame_rgb.size == 0:
                raise RuntimeError("Empty frame received")
            # Picamera2 supplies RGB888; OpenCV processing below uses BGR.
            frame = frame_rgb  # Picamera2 RGB888 array is BGR-ordered for OpenCV
        except Exception as exc:
            consecutive_read_failures += 1
            print(
                f"WARNING: Could not capture frame "
                f"({consecutive_read_failures}/10): {exc}"
            )
            if consecutive_read_failures >= 10:
                print("ERROR: Camera stream failed repeatedly; stopping.")
                break
            time.sleep(0.1)
            continue

        consecutive_read_failures = 0
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Green segmentation
        green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

        # Red segmentation (red occupies both ends of the HSV hue scale)
        red_mask_1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
        red_mask_2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

        # Reduce small noise in both masks.
        green_mask = cv2.erode(green_mask, kernel, iterations=1)
        green_mask = cv2.dilate(green_mask, kernel, iterations=2)
        red_mask = cv2.erode(red_mask, kernel, iterations=1)
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)

        # Colour-coded mask window.
        combined_mask = np.zeros_like(frame)
        combined_mask[green_mask > 0] = OBJECT_COLORS["GREEN"]
        combined_mask[red_mask > 0] = OBJECT_COLORS["RED"]

        detections = {"GREEN": green_mask, "RED": red_mask}
        hud_info = {}

        # Process EVERY contour above the minimum area, not just the largest.
        for color_name, mask in detections.items():
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            draw_color = OBJECT_COLORS[color_name]
            valid_contours = sorted(
                (c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA),
                key=cv2.contourArea,
                reverse=True,
            )

            objects = []
            for object_index, contour in enumerate(valid_contours, start=1):
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                moments = cv2.moments(contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                else:
                    cx, cy = x + w // 2, y + h // 2

                offset_x = cx - frame_center_x
                offset_y = cy - frame_center_y

                cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
                cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
                cv2.line(frame, (cx - 10, cy), (cx + 10, cy), draw_color, 2)
                cv2.line(frame, (cx, cy - 10), (cx, cy + 10), draw_color, 2)
                cv2.putText(
                    frame, f"{color_name} #{object_index}",
                    (x, max(18, y - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, draw_color, 2, cv2.LINE_AA
                )

                if color_name == "RED":
                    danger_padding = 18
                    dx1 = max(0, x - danger_padding)
                    dy1 = max(0, y - danger_padding)
                    dx2 = min(frame_width - 1, x + w + danger_padding)
                    dy2 = min(frame_height - 1, y + h + danger_padding)
                    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
                    cv2.putText(
                        frame, "DANGER ZONE", (dx1, max(18, dy1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2,
                        cv2.LINE_AA
                    )

                objects.append({
                    "index": object_index,
                    "centre": (cx, cy),
                    "offset_x": offset_x,
                    "offset_y": offset_y,
                    "area": int(area),
                })

            hud_info[color_name] = objects

        # HUD shows the number of detections and details for up to four objects per colour.
        hud_y = 25
        for color_name in ("GREEN", "RED"):
            draw_color = OBJECT_COLORS[color_name]
            objects = hud_info[color_name]
            lines = [f"{color_name}: {len(objects)} object(s)"]
            for info in objects[:4]:
                lines.append(
                    f"#{info['index']} C{info['centre']} "
                    f"XY({info['offset_x']:+d},{info['offset_y']:+d}) "
                    f"A{info['area']}px2"
                )
            if len(objects) > 4:
                lines.append(f"... +{len(objects) - 4} more")
            for line in lines:
                (text_w, text_h), baseline = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
                )
                cv2.rectangle(
                    frame, (8, hud_y - text_h - 3),
                    (16 + text_w, hud_y + baseline), (0, 0, 0), -1
                )
                cv2.putText(
                    frame, line, (12, hud_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, draw_color, 1, cv2.LINE_AA
                )
                hud_y += 20
            hud_y += 5

        # Camera centre marker.
        cv2.circle(frame, (frame_center_x, frame_center_y), 5, (255, 0, 0), -1)
        cv2.line(
            frame, (frame_center_x - 20, frame_center_y),
            (frame_center_x + 20, frame_center_y), (255, 0, 0), 1
        )
        cv2.line(
            frame, (frame_center_x, frame_center_y - 20),
            (frame_center_x, frame_center_y + 20), (255, 0, 0), 1
        )

        cv2.imshow("1 - HSV Color Identification", combined_mask)
        cv2.imshow("2 - Object Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Picamera2 stopped.")
