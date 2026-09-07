import cv2
import numpy as np

# Initialize detector
detector = cv2.wechat_qrcode_WeChatQRCode(
    "models/detect.prototxt",
    "models/detect.caffemodel",
    "models/sr.prototxt",
    "models/sr.caffemodel"
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit()

print("Camera opened. Point it at a QR code. Press Q to quit.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("ERROR: Cannot read frame")
        break
    
    # Run detection on every frame
    texts, points = detector.detectAndDecode(frame)
    
    if texts:
        # QR detected
        decoded_text = texts[0]
        print(f"Decoded: {decoded_text}")
        
        # Draw green border around QR
        if points is not None:
            corners = points[0].astype(int)
            for i in range(4):
                cv2.line(frame,
                        tuple(corners[i]),
                        tuple(corners[(i+1) % 4]),
                        (0, 255, 0), 3)
            
            # Calculate and draw centre point
            center_x = int(np.mean(corners[:, 0]))
            center_y = int(np.mean(corners[:, 1]))
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Show decoded text on frame
            cv2.putText(frame, decoded_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        # No QR detected
        cv2.putText(frame, "No QR detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow("WeChatQRCode Test", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
