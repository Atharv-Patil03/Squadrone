import cv2
import numpy as np

try:
    from pyzbar import pyzbar
except Exception as exc:
    pyzbar = None
    print(f"pyzbar unavailable, QR fallback disabled: {exc}")


# ─── INITIALISE DETECTOR ──────────────────────────────────────────

detector = None


def get_detector():
    global detector

    if detector is None:
        detector = cv2.wechat_qrcode_WeChatQRCode(
            "models/detect.prototxt",
            "models/detect.caffemodel",
            "models/sr.prototxt",
            "models/sr.caffemodel"
        )
        print("Detector loaded successfully")

    return detector


# ─── PREPROCESSING ────────────────────────────────────────────────

def preprocess_frame(frame):
    h, w = frame.shape[:2]

    if w > 1920:
        scale = 1920 / w
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply(gray)

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    gray = cv2.filter2D(gray, -1, kernel)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return processed


# ─── BLUR DETECTION ───────────────────────────────────────────────

def is_sharp(frame, threshold=100):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()

    return score > threshold, round(score, 1)


# ─── PIXEL TO GROUND OFFSET (GSD) ──────────────────────────────────

def pixel_to_ned_offset(offset_x_px, offset_y_px, altitude_m, image_width_px):
    """
    Convert pixel offsets to approximate ground offsets.

    Assumes:
      - Arducam IMX519
      - Focal length = 4.74 mm
      - Sensor width = 5.6 mm
      - Camera points straight down
      - Ground is flat

    Returns:
      north_m, east_m
    """

    focal_length_mm = 4.74
    sensor_width_mm = 5.6

    gsd = (
        (sensor_width_mm / 1000) * altitude_m
    ) / (
        (focal_length_mm / 1000) * image_width_px
    )

    east_m = offset_x_px * gsd
    north_m = -offset_y_px * gsd

    return round(north_m, 4), round(east_m, 4)


# ─── QR DETECTION ─────────────────────────────────────────────────

def detect_qr(frame):
    """
    Try WeChatQRCode first.
    Fall back to pyzbar if WeChatQRCode fails.

    Returns:
        text, center_x, center_y, method
    """

    qr_detector = get_detector()

    texts, points = qr_detector.detectAndDecode(frame)

    if texts and points is not None and len(points) > 0:
        cx = int(np.mean(points[0][:, 0]))
        cy = int(np.mean(points[0][:, 1]))

        return texts[0], cx, cy, "WeChat"

    if pyzbar is None:
        return None, None, None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try:
        decoded_objects = pyzbar.decode(gray)
    except Exception as exc:
        print(f"pyzbar decode failed: {exc}")
        return None, None, None, None

    if decoded_objects:
        obj = decoded_objects[0]
        text = obj.data.decode("utf-8")

        x, y, w, h = obj.rect
        cx = x + w // 2
        cy = y + h // 2

        return text, cx, cy, "pyzbar"

    return None, None, None, None


# ─── CAMERA ───────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit()

print("Camera running. Point at QR code. Press Q to quit.")


# from picamera2 import Picamera2
# import time

# picam2 = Picamera2()

# camera_config = picam2.create_video_configuration(
#     main={
#         "size": (1920, 1080),
#         "format": "RGB888"
#     }
# )

# picam2.configure(camera_config)
# picam2.start()

# # Allow camera exposure and white balance to settle
# time.sleep(2)

# # Trigger autofocus once before QR scanning
# picam2.set_controls({
#     "AfMode": 1,
#     "AfTrigger": 0
# })

# time.sleep(2)

# print("Arducam IMX519 started successfully")


# ─── CONSENSUS TRACKING ───────────────────────────────────────────

decode_history = []
CONSENSUS_REQUIRED = 3
confirmed_target = None

# Fixed test altitude. Replace with actual altitude later.
altitude_m = 5.0


# ─── MAIN LOOP ────────────────────────────────────────────────────

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Cannot read frame")
        break

    display = frame.copy()

    original_h, original_w = frame.shape[:2]
    img_cx = original_w // 2
    img_cy = original_h // 2

    # Draw image centre crosshair
    cv2.drawMarker(
        display,
        (img_cx, img_cy),
        (255, 255, 0),
        cv2.MARKER_CROSS,
        20,
        1
    )

    # Blur check
    sharp, blur_score = is_sharp(frame)

    if not sharp:
        decode_history.clear()

        cv2.putText(
            display,
            f"BLURRY ({blur_score})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2
        )

        cv2.imshow("QR Detection", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue

    # Preprocess and detect
    processed = preprocess_frame(frame)

    processed_h, processed_w = processed.shape[:2]

    text, qr_cx, qr_cy, method = detect_qr(processed)

    if text:
        decode_history.append(text)

        if len(decode_history) > CONSENSUS_REQUIRED:
            decode_history.pop(0)

        # Convert coordinates to original image dimensions
        scale_x = original_w / processed_w
        scale_y = original_h / processed_h

        qr_cx_original = int(qr_cx * scale_x)
        qr_cy_original = int(qr_cy * scale_y)

        # Calculate pixel offset from original image centre
        offset_x = qr_cx_original - img_cx
        offset_y = qr_cy_original - img_cy

        # Draw QR centre
        cv2.circle(
            display,
            (qr_cx_original, qr_cy_original),
            8,
            (0, 0, 255),
            -1
        )

        # Draw line from image centre to QR centre
        cv2.line(
            display,
            (img_cx, img_cy),
            (qr_cx_original, qr_cy_original),
            (255, 0, 255),
            2
        )

        # Display information
        cv2.putText(
            display,
            f"Decoded: {text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"Method: {method}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 0),
            1
        )

        cv2.putText(
            display,
            f"QR centre: ({qr_cx_original}, {qr_cy_original})",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )

        cv2.putText(
            display,
            f"Offset X: {offset_x}px  Y: {offset_y}px",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )

        cv2.putText(
            display,
            f"Consensus: {len(decode_history)}/{CONSENSUS_REQUIRED}",
            (10, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            1
        )

        # Check consensus
        if (
            len(decode_history) == CONSENSUS_REQUIRED
            and len(set(decode_history)) == 1
        ):
            confirmed_target = text

            # Calculate approximate ground offset
            north, east = pixel_to_ned_offset(
                offset_x,
                offset_y,
                altitude_m,
                original_w
            )

            cv2.putText(
                display,
                f"CONFIRMED: {confirmed_target}",
                (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                3
            )

            print(f"\nTARGET CONFIRMED: {confirmed_target}")
            print(f"QR Centre: ({qr_cx_original}, {qr_cy_original})")
            print(f"Image Centre: ({img_cx}, {img_cy})")
            print(f"Pixel Offset: X={offset_x}px, Y={offset_y}px")
            print(f"Decoded by: {method}")
            print(f"Test altitude: {altitude_m} m")
            print("Approximate ground offset:")
            print(f"  North: {north} m")
            print(f"  East:  {east} m")

    else:
        decode_history.clear()

        cv2.putText(
            display,
            "Searching...",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow("QR Detection", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ─── CLEANUP ──────────────────────────────────────────────────────

cap.release()
cv2.destroyAllWindows()

if confirmed_target:
    print(f"\nFinal confirmed target: {confirmed_target}")
else:
    print("\nNo target confirmed in this session")

