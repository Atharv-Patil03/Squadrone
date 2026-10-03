import cv2
import numpy as np
import time
from pymavlink import mavutil
from picamera2 import Picamera2
from pyzbar import pyzbar

# ─── CONFIGURATION ────────────────────────────────────────────────
SERIAL_PORT     = '/dev/ttyAMA0'
BAUD_RATE       = 921600
SCAN_ALTITUDE   = 5.0       # metres — initial QR scan
ALIGN_THRESHOLD = 0.15      # metres — alignment success radius
MAX_SPEED       = 0.3       # m/s   — max alignment speed
KP              = 0.4       # proportional gain
CONSENSUS_REQ   = 3         # frames before confirming decode
SEARCH_TIMEOUT  = 8.0       # seconds before search pattern triggers

# IMX519 camera specs
FOCAL_LENGTH_MM = 4.74
SENSOR_WIDTH_MM = 5.6
IMAGE_WIDTH_PX  = 1920

# ─── MAVLink HELPERS ──────────────────────────────────────────────
def connect(port, baud):
    print(f"Connecting on {port}...")
    mav = mavutil.mavlink_connection(port, baud=baud)
    mav.wait_heartbeat()
    print(f"Connected — System {mav.target_system}")
    return mav

def set_mode(mav, mode_name):
    mode_id = mav.mode_mapping()[mode_name]
    mav.mav.set_mode_send(
        mav.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )
    start = time.time()
    while time.time() - start < 5:
        msg = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if msg and mavutil.mode_string_v10(msg) == mode_name:
            print(f"Mode: {mode_name}")
            return True
    return False

def get_altitude(mav):
    msg = mav.recv_match(type='VFR_HUD', blocking=True, timeout=2)
    return msg.alt if msg else 0.0

def send_velocity(mav, vx, vy, vz=0.0, duration_s=0.5):
    start = time.time()
    while time.time() - start < duration_s:
        mav.mav.set_position_target_local_ned_send(
            0,
            mav.target_system,
            mav.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, 0
        )
        time.sleep(0.1)

def stop(mav):
    send_velocity(mav, 0, 0, 0, duration_s=0.3)

def arm_and_takeoff(mav, altitude_m):
    set_mode(mav, 'GUIDED')
    time.sleep(1)

    # Arm
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0
    )
    time.sleep(2)

    # Takeoff
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude_m
    )
    print(f"Taking off to {altitude_m}m...")

    # Wait for altitude
    start = time.time()
    while time.time() - start < 30:
        alt = get_altitude(mav)
        print(f"  Altitude: {alt:.2f}m", end='\r')
        if alt >= altitude_m * 0.95:
            print(f"\nReached {alt:.2f}m")
            return True
    return False

def land_and_disarm(mav):
    set_mode(mav, 'LAND')
    print("Landing...")
    start = time.time()
    while time.time() - start < 30:
        alt = get_altitude(mav)
        print(f"  Altitude: {alt:.2f}m", end='\r')
        if alt < 0.3:
            print("\nLanded")
            break
    time.sleep(2)
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 0, 0, 0, 0, 0, 0, 0
    )
    print("Disarmed")

# ─── CAMERA SETUP ─────────────────────────────────────────────────
def init_camera():
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (1920, 1080), "format": "RGB888"}
    ))
    cam.start()
    time.sleep(1)
    print("Camera ready")
    return cam

# ─── QR DETECTOR ──────────────────────────────────────────────────
wechat = cv2.wechat_qrcode_WeChatQRCode(
    "models/detect.prototxt",
    "models/detect.caffemodel",
    "models/sr.prototxt",
    "models/sr.caffemodel"
)

def preprocess(frame):
    # WeChatQRCode CNN works better on natural colour frames
    # Only resize if needed
    if frame.shape[1] > 1920:
        scale = 1920 / frame.shape[1]
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    return frame

def is_sharp(frame, threshold=100):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() > threshold

def detect_qr(frame):
    texts, points = wechat.detectAndDecode(frame)
    if texts and texts[0]:
        cx = int(np.mean(points[0][:, 0]))
        cy = int(np.mean(points[0][:, 1]))
        return texts[0], cx, cy, "WeChat"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    objs = pyzbar.decode(gray)
    if objs:
        obj = objs[0]
        text = obj.data.decode('utf-8')
        x, y, w, h = obj.rect
        return text, x + w//2, y + h//2, "pyzbar"

    return None, None, None, None

# ─── GSD CALCULATION ──────────────────────────────────────────────
# NOTE: This uses fixed sensor constants.
# Final FSM uses qr_module.get_alignment_offset() which
# reads actual ScalerCrop from camera metadata — more accurate.
def pixel_to_ned(offset_x_px, offset_y_px, altitude_m):
    gsd = (SENSOR_WIDTH_MM / 1000 * altitude_m) / \
          (FOCAL_LENGTH_MM  / 1000 * IMAGE_WIDTH_PX)
    east_m  =  offset_x_px * gsd
    north_m = -offset_y_px * gsd
    return round(north_m, 4), round(east_m, 4)

# ─── SEARCH PATTERN ───────────────────────────────────────────────
def execute_search_pattern(mav):
    """
    Move in a plus pattern to find QR if not detected.
    Each move is 1m at 0.3m/s — takes about 3s per leg.
    """
    print("Executing search pattern...")
    moves = [
        ( 0.3,  0.0),   # North
        (-0.3,  0.0),   # South (back to centre)
        ( 0.0,  0.3),   # East
        ( 0.0, -0.3),   # West (back to centre)
    ]
    for vx, vy in moves:
        send_velocity(mav, vx, vy, duration_s=3)
        stop(mav)
        time.sleep(1)

# ─── STATE: S2 — INITIAL QR SCAN ─────────────────────────────────
def state_qr_scan(mav, cam):
    """
    Hover at 5m in LOITER.
    Run QR detection until 3-frame consensus confirmed.
    Return confirmed delivery target string.
    """
    print("\n── S2: Initial QR Scan ──")
    set_mode(mav, 'LOITER')
    time.sleep(1.5)   # Stabilise

    decode_history = []
    search_retries  = 0
    MAX_RETRIES     = 3
    scan_start      = time.time()

    while True:
        frame = cam.capture_array()
        

        if not is_sharp(frame):
            decode_history.clear()
            continue

        processed       = preprocess(frame)
        text, cx, cy, _ = detect_qr(processed)

        if text:
            decode_history.append(text)
            if len(decode_history) > CONSENSUS_REQ:
                decode_history.pop(0)

            print(f"  Candidate: {text} "
                  f"({len(decode_history)}/{CONSENSUS_REQ})")

            if (len(decode_history) == CONSENSUS_REQ and
                    len(set(decode_history)) == 1):
                confirmed = text
                print(f"  CONFIRMED TARGET: {confirmed}")
                return confirmed
        else:
            decode_history.clear()

            # Check if search pattern needed
            if time.time() - scan_start > SEARCH_TIMEOUT:
                if search_retries >= MAX_RETRIES:
                    print("  ERROR: QR not found after max retries")
                    return None
                set_mode(mav, 'GUIDED')
                execute_search_pattern(mav)
                set_mode(mav, 'LOITER')
                time.sleep(1.5)
                search_retries += 1
                scan_start = time.time()

# ─── STATE: S6 — TARGET ALIGNMENT ────────────────────────────────
def state_alignment(mav, cam, delivery_target):
    """
    Scan delivery zone QR codes at 10m.
    Find matching QR.
    Align drone within 15cm of its centre.
    """
    print("\n── S6: Target Alignment ──")
    set_mode(mav, 'GUIDED')

    aligned_count = 0
    ALIGNED_REQUIRED = 3

    while True:
        frame = cam.capture_array()
        

        if not is_sharp(frame):
            aligned_count = 0
            continue

        processed        = preprocess(frame)
        text, cx, cy, _  = detect_qr(processed)

        if text != delivery_target:
            aligned_count = 0
            if text:
                print(f"  Wrong QR: {text} — ignoring")
            else:
                print("  No QR detected")
            continue

        # Correct QR found — calculate offset
        img_cx = IMAGE_WIDTH_PX  // 2
        img_cy = 1080            // 2

        offset_x_px = cx - img_cx
        offset_y_px = cy - img_cy

        altitude = get_altitude(mav)
        north_m, east_m = pixel_to_ned(
            offset_x_px, offset_y_px, altitude
        )

        distance = (north_m**2 + east_m**2) ** 0.5

        print(f"  Offset: N={north_m:.3f}m  E={east_m:.3f}m  "
              f"dist={distance:.3f}m")

        if distance < ALIGN_THRESHOLD:
            aligned_count += 1
            print(f"  Within threshold "
                  f"({aligned_count}/{ALIGNED_REQUIRED})")

            if aligned_count >= ALIGNED_REQUIRED:
                print("  ALIGNED")
                stop(mav)
                return True
        else:
            aligned_count = 0

            # Proportional velocity command
            vx = max(-MAX_SPEED, min(MAX_SPEED, KP * north_m))
            vy = max(-MAX_SPEED, min(MAX_SPEED, KP * east_m))
            send_velocity(mav, vx, vy, duration_s=0.3)

# ─── MAIN ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    mav = connect(SERIAL_PORT, BAUD_RATE)
    cam = init_camera()

    try:
        # Takeoff to 5m
        if not arm_and_takeoff(mav, SCAN_ALTITUDE):
            print("Takeoff failed")
            exit()

        # S2 — Scan start QR
        delivery_target = state_qr_scan(mav, cam)
        if not delivery_target:
            print("Could not determine delivery target — landing")
            land_and_disarm(mav)
            exit()

        print(f"\nDelivery target stored: {delivery_target}")

        # ── At this point in full mission you would
        #    proceed to S3 corridor entry.
        #    For this test we go straight to S6
        #    to test alignment only. ──

        # S6 — Align over target QR at delivery zone
        # For testing: place the matching QR on ground
        # directly below the drone at current position
        aligned = state_alignment(mav, cam, delivery_target)

        if aligned:
            print("\nAlignment confirmed — ready for payload delivery")
            # S7 payload delivery will go here next
        
        # Land for now
        land_and_disarm(mav)

    except KeyboardInterrupt:
        print("\nInterrupt — landing")
        set_mode(mav, 'LAND')
        time.sleep(5)