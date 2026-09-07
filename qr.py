import cv2

# Load the WeChatQRCode detector with model files
detector = cv2.wechat_qrcode_WeChatQRCode(
    "models/detect.prototxt",
    "models/detect.caffemodel",
    "models/sr.prototxt",
    "models/sr.caffemodel"
)

# Load your test image
image = cv2.imread("test_qr.png")

if image is None:
    print("ERROR: Could not load image. Check the file path.")
    exit()

# Run detection
texts, points = detector.detectAndDecode(image)

# Check result
if texts:
    print(f"SUCCESS: QR decoded")
    print(f"Content: {texts[0]}")
    print(f"Number of QR codes found: {len(texts)}")
else:
    print("FAILED: No QR code detected")

# Draw the detected QR boundary on the image
result_image = image.copy()
if points is not None:
    for point_set in points:
        pts = point_set.astype(int)
        for i in range(len(pts)):
            cv2.line(result_image, 
                    tuple(pts[i]), 
                    tuple(pts[(i+1) % len(pts)]), 
                    (0, 255, 0), 2)

cv2.imshow("Detection Result", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()