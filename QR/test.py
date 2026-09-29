from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()

# Configure a preview stream
config = picam2.create_preview_configuration(
    main={"size": (1920, 1080), "format": "RGB888"}
)

picam2.configure(config)

picam2.start_preview(Preview.QTGL)
picam2.start()

print("Camera feed started. Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    picam2.stop_preview()
    picam2.stop()
