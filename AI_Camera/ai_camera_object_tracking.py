import cv2
import numpy as np
import time
import math
from collections import deque

from picamera2 import Picamera2


# ============================================================
# SETTINGS
# ============================================================

# Green HSV range
LOWER_GREEN = np.array([35, 80, 50])
UPPER_GREEN = np.array([85, 255, 255])

# Ignore small green blobs
MIN_CONTOUR_AREA = 500

# Movement detection
MOVEMENT_THRESHOLD = 5
STATIONARY_TIME = 0.7

# Tracking trail
TRAIL_LENGTH = 30


# ============================================================
# CAMERA SETUP
# ============================================================

picam2 = Picamera2()

camera_config = picam2.create_preview_configuration(
    main={
        "size": (1280, 720),
        "format": "RGB888"
    }
)

picam2.configure(camera_config)

picam2.start()

time.sleep(2)

print("------------------------------------------")
print("Raspberry Pi AI Camera started")
print("Resolution: 1280 x 720")
print("------------------------------------------")


# ============================================================
# FRAME PARAMETERS
# ============================================================

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

FRAME_CENTER_X = FRAME_WIDTH // 2
FRAME_CENTER_Y = FRAME_HEIGHT // 2


# ============================================================
# IMAGE PROCESSING
# ============================================================

kernel = np.ones((5, 5), np.uint8)


# ============================================================
# TRACKING VARIABLES
# ============================================================

centre_history = deque(
    maxlen=TRAIL_LENGTH
)

previous_cx = None
previous_cy = None

last_movement_time = time.time()

object_stationary = False


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # CAPTURE IMAGE FROM AI CAMERA
    # --------------------------------------------------------

    frame = picam2.capture_array()

    # Picamera2 gives RGB
    # OpenCV uses BGR
    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )


    # --------------------------------------------------------
    # CREATE HSV IMAGE
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )


    # --------------------------------------------------------
    # GREEN SEGMENTATION
    # --------------------------------------------------------

    mask = cv2.inRange(
        hsv,
        LOWER_GREEN,
        UPPER_GREEN
    )


    # --------------------------------------------------------
    # REMOVE NOISE
    # --------------------------------------------------------

    mask = cv2.erode(
        mask,
        kernel,
        iterations=1
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=2
    )


    # ========================================================
    # FIND CONTOURS
    # ========================================================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    object_detected = False


    # ========================================================
    # FIND LARGEST GREEN OBJECT
    # ========================================================

    if len(contours) > 0:

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(
            largest_contour
        )


        if area > MIN_CONTOUR_AREA:

            object_detected = True


            # =================================================
            # BOUNDING RECTANGLE
            # =================================================

            x, y, w, h = cv2.boundingRect(
                largest_contour
            )


            # =================================================
            # GEOMETRIC CENTRE
            # =================================================

            M = cv2.moments(
                largest_contour
            )

            if M["m00"] != 0:

                cx = int(
                    M["m10"] / M["m00"]
                )

                cy = int(
                    M["m01"] / M["m00"]
                )

            else:

                cx = x + w // 2
                cy = y + h // 2


            # =================================================
            # SAVE CENTRE FOR TRACKING
            # =================================================

            centre_history.append(
                (cx, cy)
            )


            # =================================================
            # MOVEMENT DETECTION
            # =================================================

            if previous_cx is not None:

                movement = math.sqrt(
                    (cx - previous_cx) ** 2 +
                    (cy - previous_cy) ** 2
                )

                if movement > MOVEMENT_THRESHOLD:

                    last_movement_time = time.time()

                    object_stationary = False

                else:

                    if (
                        time.time()
                        - last_movement_time
                        > STATIONARY_TIME
                    ):

                        object_stationary = True


            previous_cx = cx
            previous_cy = cy


            # =================================================
            # DRAW BOUNDARY
            # =================================================

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )


            # =================================================
            # DRAW GEOMETRIC CENTRE
            # =================================================

            cv2.circle(
                frame,
                (cx, cy),
                7,
                (0, 0, 255),
                -1
            )


            # Centre cross

            cv2.line(
                frame,
                (cx - 15, cy),
                (cx + 15, cy),
                (0, 0, 255),
                2
            )

            cv2.line(
                frame,
                (cx, cy - 15),
                (cx, cy + 15),
                (0, 0, 255),
                2
            )


            # =================================================
            # TRACKING TRAIL
            # =================================================

            for i in range(
                1,
                len(centre_history)
            ):

                p1 = centre_history[i - 1]
                p2 = centre_history[i]

                cv2.line(
                    frame,
                    p1,
                    p2,
                    (255, 0, 255),
                    3
                )


            # =================================================
            # X DIMENSION
            # =================================================

            dimension_y = max(
                y - 20,
                25
            )

            cv2.line(
                frame,
                (x, dimension_y),
                (x + w, dimension_y),
                (255, 255, 0),
                2
            )

            # End ticks

            cv2.line(
                frame,
                (x, dimension_y - 7),
                (x, dimension_y + 7),
                (255, 255, 0),
                2
            )

            cv2.line(
                frame,
                (x + w, dimension_y - 7),
                (x + w, dimension_y + 7),
                (255, 255, 0),
                2
            )


            # =================================================
            # Y DIMENSION
            # =================================================

            dimension_x = min(
                x + w + 20,
                FRAME_WIDTH - 25
            )

            cv2.line(
                frame,
                (dimension_x, y),
                (dimension_x, y + h),
                (255, 255, 0),
                2
            )

            # End ticks

            cv2.line(
                frame,
                (dimension_x - 7, y),
                (dimension_x + 7, y),
                (255, 255, 0),
                2
            )

            cv2.line(
                frame,
                (dimension_x - 7, y + h),
                (dimension_x + 7, y + h),
                (255, 255, 0),
                2
            )


            # =================================================
            # OBJECT DIMENSIONS
            # =================================================

            cv2.putText(
                frame,
                f"X = {w} px",
                (x, dimension_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Y = {h} px",
                (dimension_x + 10, y + h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )


            # =================================================
            # CENTRE COORDINATES
            # =================================================

            cv2.putText(
                frame,
                f"Centre = ({cx}, {cy})",
                (x, y + h + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )


            # =================================================
            # CONTOUR AREA
            # =================================================

            cv2.putText(
                frame,
                f"Area = {int(area)} px",
                (x, y + h + 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )


            # =================================================
            # MOVEMENT STATUS
            # =================================================

            if object_stationary:

                status = "STATIONARY"
                status_color = (0, 255, 255)

            else:

                status = "MOVING"
                status_color = (0, 255, 0)


            cv2.putText(
                frame,
                status,
                (x, max(y - 45, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2
            )


            # =================================================
            # CAMERA CENTRE OFFSET
            # =================================================

            dx = cx - FRAME_CENTER_X
            dy = cy - FRAME_CENTER_Y

            cv2.putText(
                frame,
                f"dx = {dx}   dy = {dy}",
                (20, FRAME_HEIGHT - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


    else:

        # =====================================================
        # NO OBJECT
        # =====================================================

        cv2.putText(
            frame,
            "NO GREEN OBJECT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ========================================================
    # CAMERA FRAME CENTRE
    # ========================================================

    cv2.circle(
        frame,
        (
            FRAME_CENTER_X,
            FRAME_CENTER_Y
        ),
        6,
        (255, 0, 0),
        -1
    )


    # Horizontal line

    cv2.line(
        frame,
        (
            FRAME_CENTER_X - 25,
            FRAME_CENTER_Y
        ),
        (
            FRAME_CENTER_X + 25,
            FRAME_CENTER_Y
        ),
        (255, 0, 0),
        2
    )


    # Vertical line

    cv2.line(
        frame,
        (
            FRAME_CENTER_X,
            FRAME_CENTER_Y - 25
        ),
        (
            FRAME_CENTER_X,
            FRAME_CENTER_Y + 25
        ),
        (255, 0, 0),
        2
    )


    # ========================================================
    # FEED 1
    # HSV MASK
    # ========================================================

    cv2.imshow(
        "1 - HSV Identification",
        mask
    )


    # ========================================================
    # FEED 2
    # ORIGINAL CAMERA + TRACKING
    # ========================================================

    cv2.imshow(
        "2 - Object Tracking",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


# ============================================================
# CLEANUP
# ============================================================

picam2.stop()

cv2.destroyAllWindows()

print("Camera stopped.")