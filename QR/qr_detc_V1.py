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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    kernel = np.array([[0, -1, 0],
                       [-1,  5, -1],
                       [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return processed

def is_sharp(frame, threshold=100):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score > threshold, round(score, 1)

# ─── DUAL DECODER ─────────────────────────────────────────────────
def detect_qr(frame):
    """
    Try WeChatQRCode first.
    Fall back to pyzbar if WeChatQRCode fails.
    Returns (text, center_x, center_y, method) or (None, None, None, None)
    """
    # Primary — WeChatQRCode
    qr_detector = get_detector()
    texts, points = qr_detector.detectAndDecode(frame)
    if texts:
        cx = int(np.mean(points[0][:, 0]))
        cy = int(np.mean(points[0][:, 1]))
        return texts[0], cx, cy, "WeChat"

    # Fallback — pyzbar
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
        text = obj.data.decode('utf-8')
        x, y, w, h = obj.rect
        cx = x + w // 2
        cy = y + h // 2
        return text, cx, cy, "pyzbar"

    return None, None, None, None

# ─── MAIN LOOP ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit()

print("Camera running. Point at QR code. Press Q to quit.")

# from picamera2 import Picamera2

# cam = Picamera2()
# cam.configure(cam.create_video_configuration(
#     main={"size": (1920, 1080), "format": "RGB888"}
# ))
# cam.start()

# # In loop:
# frame = cam.capture_array()
# # Convert RGB to BGR for OpenCV
# frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


# Consensus tracking
decode_history = []
CONSENSUS_REQUIRED = 3
confirmed_target = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame")
        break

    display = frame.copy()
    img_h, img_w = frame.shape[:2]
    img_cx = img_w // 2
    img_cy = img_h // 2

    # Draw image centre crosshair
    cv2.drawMarker(display, (img_cx, img_cy),
                   (255, 255, 0), cv2.MARKER_CROSS, 20, 1)

    # Blur check
    sharp, blur_score = is_sharp(frame)
    if not sharp:
        decode_history.clear()
        cv2.putText(display, f"BLURRY ({blur_score})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 165, 255), 2)
        cv2.imshow("QR Detection", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Preprocess and detect
    processed = preprocess_frame(frame)
    text, qr_cx, qr_cy, method = detect_qr(processed)

    if text:
        # Add to consensus history
        decode_history.append(text)
        if len(decode_history) > CONSENSUS_REQUIRED:
            decode_history.pop(0)

        # Calculate offset from image centre
        offset_x = qr_cx - img_cx
        offset_y = qr_cy - img_cy

        # Draw QR centre
        cv2.circle(display, (qr_cx, qr_cy), 8, (0, 0, 255), -1)

        # Draw line from image centre to QR centre
        cv2.line(display, (img_cx, img_cy),
                 (qr_cx, qr_cy), (255, 0, 255), 2)

        # Display info
        cv2.putText(display, f"Decoded: {text}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Method: {method}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 0), 1)
        cv2.putText(display, f"QR centre: ({qr_cx}, {qr_cy})",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)
        cv2.putText(display, f"Offset X: {offset_x}px  Y: {offset_y}px",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)
        cv2.putText(display,
                    f"Consensus: {len(decode_history)}/{CONSENSUS_REQUIRED}",
                    (10, 135), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 200, 255), 1)

        # Check consensus
        if (len(decode_history) == CONSENSUS_REQUIRED and
                len(set(decode_history)) == 1):
            confirmed_target = text
            cv2.putText(display, f"CONFIRMED: {confirmed_target}",
                        (10, 170), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 3)
            print(f"\n TARGET CONFIRMED: {confirmed_target}")
            print(f" QR Centre: ({qr_cx}, {qr_cy})")
            print(f" Image Centre: ({img_cx}, {img_cy})")
            print(f" Offset: X={offset_x}px, Y={offset_y}px")
            print(f" Decoded by: {method}")

    else:
        decode_history.clear()
        cv2.putText(display, "Searching...",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

    cv2.imshow("QR Detection", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if confirmed_target:
    print(f"\nFinal confirmed target: {confirmed_target}")
else:
    print("\nNo target confirmed in this session")