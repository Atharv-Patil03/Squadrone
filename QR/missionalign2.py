"""
SQUADRONE - MISSION TEST: S2 (initial QR scan) + S6 (target alignment)

Fixes vs previous version:
    - Absolute model paths (no more "exists(detector_prototxt_path)" error)
    - Opens the IMX519 explicitly (this Pi also has an imx500 at index 0)
    - Enables autofocus
    - GSD uses the real IMX519 values and the actual ScalerCrop width
    - Velocity commands are sent in the BODY frame (forward/right), because
      image offsets are relative to the drone, not to true north.
      Assumes the top of the image points to the drone's nose; if the
      camera is mounted rotated, adjust CAMERA_ROTATION_DEG.
    - Altitude from GLOBAL_POSITION_INT.relative_alt (relative to home)
    - Blurry/no-detection frames no longer skip the search timeout
    - Alignment has a timeout and holds position when the target is lost
    - Every failure path lands instead of exit()-ing in the air
    - Mode check compares custom_mode (no deprecated mode_string_v10)
    - S2 scan uses GUIDED + zero velocity hold (not LOITER — avoids sink)
    - land_and_disarm waits for baro settle after mode switch
    - arm_and_takeoff streams position data before takeoff command

TEST WITH PROPS OFF / TETHERED FIRST.
"""

import math
import time
import logging
from pathlib import Path

import cv2
import numpy as np
from pymavlink import mavutil
from picamera2 import Picamera2
from libcamera import controls

try:
    from pyzbar import pyzbar
except Exception:
    pyzbar = None

# ─── CONFIGURATION ────────────────────────────────────────────────
SERIAL_PORT      = '/dev/ttyAMA0'
BAUD_RATE        = 921600
SCAN_ALTITUDE    = 5.0      # metres
ALIGN_THRESHOLD  = 0.15     # metres
MAX_SPEED        = 0.3      # m/s
KP               = 0.4
CONSENSUS_REQ    = 3
SEARCH_TIMEOUT   = 8.0      # s before a search move
MAX_RETRIES      = 3
ALIGN_TIMEOUT    = 60.0     # s total for S6
ALIGNED_REQUIRED = 3

# Camera
CAMERA_WIDTH     = 1920
CAMERA_HEIGHT    = 1080
CAMERA_INDEX     = None     # None = auto-pick imx519
LENS_POSITION    = None     # None = continuous AF; float = fixed (dioptres)
FOCAL_LENGTH_MM  = 4.28
PIXEL_SIZE_UM    = 1.22
SENSOR_FULL_WIDTH_PX = 4656
CAMERA_ROTATION_DEG = 0

BLUR_GATE_ENABLED = False
BLUR_THRESHOLD    = 30.0

MODEL_DIR = Path(__file__).resolve().parent / "models"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

sensor_crop_width_px = SENSOR_FULL_WIDTH_PX
_last_alt_m = 0.0


# ─── MAVLink HELPERS ──────────────────────────────────────────────
def connect(port, baud):
    print(f"Connecting on {port}...")
    mav = mavutil.mavlink_connection(port, baud=baud)
    mav.wait_heartbeat()
    print(f"Connected — System {mav.target_system}")

    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0, 33, 100000, 0, 0, 0, 0, 0
    )
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
        if msg and msg.custom_mode == mode_id:
            print(f"Mode: {mode_name}")
            return True
    print(f"WARNING: mode {mode_name} not confirmed")
    return False


def get_altitude(mav):
    global _last_alt_m
    msg = mav.recv_match(type='GLOBAL_POSITION_INT',
                         blocking=True, timeout=1)
    if msg:
        _last_alt_m = msg.relative_alt / 1000.0
    return _last_alt_m


def send_velocity(mav, forward, right, down=0.0, duration_s=0.5):
    """Body-frame velocity: forward / right / down in m/s."""
    start = time.time()
    while time.time() - start < duration_s:
        mav.mav.set_position_target_local_ned_send(
            0,
            mav.target_system,
            mav.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0000111111000111,
            0, 0, 0,
            forward, right, down,
            0, 0, 0,
            0, 0
        )
        time.sleep(0.1)


def stop(mav, duration_s=0.3):
    send_velocity(mav, 0, 0, 0, duration_s=duration_s)


def arm_and_takeoff(mav, altitude_m):
    set_mode(mav, 'GUIDED')
    time.sleep(1)

    print("\nChecks: props secure? area clear? RC on?")
    if input("Type YES to arm: ").strip().upper() != "YES":
        print("Cancelled")
        return False

    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0
    )
    time.sleep(2)

    # FIXED — stream position data before takeoff command
    mav.mav.request_data_stream_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1
    )
    time.sleep(0.5)

    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude_m
    )
    print(f"Taking off to {altitude_m}m...")

    start = time.time()
    while time.time() - start < 30:
        alt = get_altitude(mav)
        print(f"  Altitude: {alt:.2f}m", end='\r')
        if alt >= altitude_m * 0.95:
            print(f"\nReached {alt:.2f}m")
            time.sleep(2)   # FIXED — stabilise after takeoff
            return True
    return False


def land_and_disarm(mav):
    set_mode(mav, 'LAND')
    time.sleep(2)           # FIXED — wait for baro to settle
    print("Landing...")
    start = time.time()
    while time.time() - start < 40:
        alt = get_altitude(mav)
        print(f"  Altitude: {alt:.2f}m", end='\r')
        if alt < 0.5:
            print("\nLanded")
            break
    time.sleep(3)           # FIXED — confirm fully settled
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 0, 0, 0, 0, 0, 0, 0
    )
    print("Disarmed")


# ─── CAMERA ───────────────────────────────────────────────────────
def resolve_camera_index():
    if CAMERA_INDEX is not None:
        return CAMERA_INDEX
    cams = Picamera2.global_camera_info()
    logging.info("Cameras detected: %s",
                 [(c.get("Num"), c.get("Model")) for c in cams])
    for c in cams:
        if "imx519" in str(c.get("Model", "")).lower():
            logging.info("Selected imx519 at index %s", c.get("Num"))
            return c.get("Num", 0)
    logging.warning("imx519 not found; using index 0")
    return 0


def init_camera():
    global sensor_crop_width_px

    cam = Picamera2(camera_num=resolve_camera_index())
    cam.configure(cam.create_video_configuration(
        main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"},
        buffer_count=4
    ))
    cam.start()

    if LENS_POSITION is None:
        cam.set_controls({"AfMode": controls.AfModeEnum.Continuous,
                          "AfSpeed": controls.AfSpeedEnum.Fast})
    else:
        cam.set_controls({"AfMode": controls.AfModeEnum.Manual,
                          "LensPosition": float(LENS_POSITION)})
    time.sleep(2)

    try:
        crop = cam.capture_metadata().get("ScalerCrop")
        if crop:
            sensor_crop_width_px = crop[2]
    except Exception as exc:
        logging.warning("ScalerCrop read failed: %s", exc)

    print(f"Camera ready (sensor crop width {sensor_crop_width_px}px)")
    return cam


# ─── QR DETECTION ─────────────────────────────────────────────────
wechat = None
try:
    wechat = cv2.wechat_qrcode_WeChatQRCode(
        str(MODEL_DIR / "detect.prototxt"),
        str(MODEL_DIR / "detect.caffemodel"),
        str(MODEL_DIR / "sr.prototxt"),
        str(MODEL_DIR / "sr.caffemodel"),
    )
    logging.info("WeChatQRCode loaded")
except Exception as exc:
    logging.error("WeChatQRCode failed to load (%s). "
                  "Using pyzbar only. Models expected in %s", exc, MODEL_DIR)


def is_sharp(frame):
    if not BLUR_GATE_ENABLED:
        return True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() > BLUR_THRESHOLD


def detect_qr(frame):
    if wechat is not None:
        try:
            texts, points = wechat.detectAndDecode(frame)
            if texts is not None and points is not None:
                for text, pts in zip(texts, points):
                    if text:
                        pts = np.asarray(pts).reshape(-1, 2)
                        return (text, float(pts[:, 0].mean()),
                                float(pts[:, 1].mean()))
        except cv2.error as exc:
            logging.warning("WeChat error: %s", exc)

    if pyzbar is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objs = pyzbar.decode(gray)
        if objs:
            o = objs[0]
            x, y, w, h = o.rect
            return (o.data.decode('utf-8', errors='replace'),
                    x + w / 2, y + h / 2)

    return None, None, None


# ─── GROUND OFFSET ────────────────────────────────────────────────
def pixel_to_body_offset(offset_x_px, offset_y_px, altitude_m, frame_w_px):
    used_sensor_width_mm = sensor_crop_width_px * (PIXEL_SIZE_UM / 1000.0)
    ground_width_m = altitude_m * used_sensor_width_mm / FOCAL_LENGTH_MM
    gsd = ground_width_m / frame_w_px

    img_right = offset_x_px * gsd
    img_down = offset_y_px * gsd

    a = math.radians(CAMERA_ROTATION_DEG)
    forward = -img_down * math.cos(a) + img_right * math.sin(a)
    right = img_right * math.cos(a) + img_down * math.sin(a)

    return round(forward, 4), round(right, 4)


# ─── SEARCH PATTERN ───────────────────────────────────────────────
def execute_search_pattern(mav):
    print("Executing search pattern...")
    for fwd, rgt in [(0.3, 0.0), (-0.3, 0.0), (0.0, 0.3), (0.0, -0.3)]:
        send_velocity(mav, fwd, rgt, duration_s=3)
        stop(mav)
        time.sleep(1)


# ─── S2: INITIAL QR SCAN ──────────────────────────────────────────
def state_qr_scan(mav, cam):
    print("\n── S2: Initial QR Scan ──")
    # FIXED — stay in GUIDED, hold position with zero velocity
    # LOITER was causing altitude sink on mode switch
    set_mode(mav, 'GUIDED')
    time.sleep(1.5)

    history = []
    retries = 0
    scan_start = time.time()

    while True:
        frame = cam.capture_array()

        # Hold position while scanning
        send_velocity(mav, 0, 0, duration_s=0.1)

        text = None
        if is_sharp(frame):
            text, _, _ = detect_qr(frame)

        if text:
            history.append(text)
            if len(history) > CONSENSUS_REQ:
                history.pop(0)

            print(f"  Candidate: {text} ({len(history)}/{CONSENSUS_REQ})")

            if len(history) == CONSENSUS_REQ and len(set(history)) == 1:
                print(f"  CONFIRMED TARGET: {text}")
                return text
            continue

        history.clear()

        if time.time() - scan_start > SEARCH_TIMEOUT:
            if retries >= MAX_RETRIES:
                print("  ERROR: QR not found after max retries")
                return None
            execute_search_pattern(mav)
            time.sleep(1.5)
            retries += 1
            scan_start = time.time()


# ─── S6: TARGET ALIGNMENT ─────────────────────────────────────────
def state_alignment(mav, cam, delivery_target):
    if not delivery_target:
        print("ERROR: no delivery target set")
        return False
    print("\n── S6: Target Alignment ──")
    set_mode(mav, 'GUIDED')

    aligned_count = 0
    start = time.time()

    while time.time() - start < ALIGN_TIMEOUT:
        frame = cam.capture_array()
        frame_h, frame_w = frame.shape[:2]

        text, cx, cy = (None, None, None)
        if is_sharp(frame):
            text, cx, cy = detect_qr(frame)

        if text != delivery_target:
            aligned_count = 0
            print(f"  Wrong QR: {text} — holding" if text
                  else "  No QR detected — holding")
            send_velocity(mav, 0, 0, duration_s=0.2)
            continue

        offset_x_px = cx - frame_w / 2
        offset_y_px = cy - frame_h / 2

        altitude = get_altitude(mav)
        forward_m, right_m = pixel_to_body_offset(
            offset_x_px, offset_y_px, altitude, frame_w)

        distance = math.hypot(forward_m, right_m)
        print(f"  Offset: fwd={forward_m:.3f}m right={right_m:.3f}m "
              f"dist={distance:.3f}m alt={altitude:.2f}m")

        if distance < ALIGN_THRESHOLD:
            aligned_count += 1
            print(f"  Within threshold ({aligned_count}/{ALIGNED_REQUIRED})")
            if aligned_count >= ALIGNED_REQUIRED:
                print("  ALIGNED")
                stop(mav)
                return True
        else:
            aligned_count = 0
            vf = max(-MAX_SPEED, min(MAX_SPEED, KP * forward_m))
            vr = max(-MAX_SPEED, min(MAX_SPEED, KP * right_m))
            send_velocity(mav, vf, vr, duration_s=0.3)

    print("  ERROR: alignment timed out")
    stop(mav)
    return False


# ─── MAIN ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    mav = connect(SERIAL_PORT, BAUD_RATE)
    cam = init_camera()

    try:
        if not arm_and_takeoff(mav, SCAN_ALTITUDE):
            print("Takeoff failed — landing")
            land_and_disarm(mav)
            raise SystemExit(1)

        delivery_target = state_qr_scan(mav, cam)
        if not delivery_target:
            print("Could not determine delivery target — landing")
            land_and_disarm(mav)
            raise SystemExit(1)

        print(f"\nDelivery target stored: {delivery_target}")

        aligned = state_alignment(mav, cam, delivery_target)

        if aligned:
            print("\nAlignment confirmed — ready for payload delivery")

        land_and_disarm(mav)

    except KeyboardInterrupt:
        print("\nInterrupt — landing")
        set_mode(mav, 'LAND')
        time.sleep(5)

    except SystemExit:
        raise

    except Exception as exc:
        print(f"\nUnexpected error: {exc} — landing")
        set_mode(mav, 'LAND')
        raise

    finally:
        cam.stop()
        cam.close()