"""
SQUADRONE - BENCH TEST WITH PIXHAWK CONNECTED (PROPS OFF)

This script NEVER arms, NEVER takes off and NEVER changes flight mode.
It only:
    - connects to the Pixhawk and shows live telemetry (mode, armed
      state, altitude, link health)
    - Phase 1: scans the START QR and stores it as the target
    - Phase 2: matches every QR in view against the target and shows the
      body-frame velocity command the real mission would send
      (nothing is sent unless SEND_COMMANDS = True)

Safety: if the vehicle reports ARMED, the script refuses to start and
(if it becomes armed while running) stops sending anything.

Keys (click the preview window first):
    r reset to Phase 1 | a autofocus | c continuous AF | s save | q quit
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

# False = only DISPLAY the velocity command (recommended).
# True  = also send it, but only while the vehicle is DISARMED
#         (the autopilot ignores it; this just exercises the link).
SEND_COMMANDS    = False

CAMERA_WIDTH     = 1920
CAMERA_HEIGHT    = 1080
CAMERA_INDEX     = None
DISPLAY_WIDTH    = 1280

CONSENSUS_REQ    = 3
ALIGN_THRESHOLD  = 0.15
ALIGNED_REQUIRED = 3
MAX_SPEED        = 0.3
KP               = 0.4

# On the bench the real altitude is ~0 (so offsets would be 0).
# Use the real camera-to-paper distance instead, in metres.
BENCH_ALTITUDE_M = 0.5

FOCAL_LENGTH_MM  = 4.28
PIXEL_SIZE_UM    = 1.22
SENSOR_FULL_WIDTH_PX = 4656
CAMERA_ROTATION_DEG = 0   # 0 = image top points to the drone nose

MODEL_DIR = Path(__file__).resolve().parent / "models"
BASE_DIR = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

sensor_crop_width_px = SENSOR_FULL_WIDTH_PX


# ─── MAVLINK ──────────────────────────────────────────────────────
class Telemetry:
    """Non-blocking reader of the few messages we care about."""

    def __init__(self, mav):
        self.mav = mav
        self.mode = "?"
        self.armed = False
        self.alt_m = 0.0
        self.last_heartbeat = time.time()
        self.has_position = False

    def update(self):
        while True:
            msg = self.mav.recv_match(blocking=False)
            if msg is None:
                break
            t = msg.get_type()
            if t == "HEARTBEAT":
                self.last_heartbeat = time.time()
                self.armed = bool(
                    msg.base_mode
                    & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                try:
                    self.mode = mavutil.mode_string_v10(msg)
                except Exception:
                    self.mode = str(msg.custom_mode)
            elif t == "GLOBAL_POSITION_INT":
                self.alt_m = msg.relative_alt / 1000.0
                self.has_position = True

    @property
    def link_ok(self):
        return (time.time() - self.last_heartbeat) < 3.0


def connect():
    print(f"Connecting to Pixhawk on {SERIAL_PORT} @ {BAUD_RATE}...")
    mav = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUD_RATE)
    mav.wait_heartbeat(timeout=10)
    print(f"Connected — system {mav.target_system}, "
          f"component {mav.target_component}")

    # Ask for 10 Hz GLOBAL_POSITION_INT (altitude)
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0, 33, 100000, 0, 0, 0, 0, 0)
    return mav


def send_velocity(mav, forward, right):
    mav.mav.set_position_target_local_ned_send(
        0, mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111000111,
        0, 0, 0,
        forward, right, 0.0,
        0, 0, 0,
        0, 0)


# ─── CAMERA ───────────────────────────────────────────────────────
def resolve_camera_index():
    if CAMERA_INDEX is not None:
        return CAMERA_INDEX
    for c in Picamera2.global_camera_info():
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
        buffer_count=4))
    cam.start()
    cam.set_controls({"AfMode": controls.AfModeEnum.Continuous,
                      "AfSpeed": controls.AfSpeedEnum.Fast})
    time.sleep(2)
    try:
        crop = cam.capture_metadata().get("ScalerCrop")
        if crop:
            sensor_crop_width_px = crop[2]
    except Exception as exc:
        logging.warning("ScalerCrop read failed: %s", exc)
    return cam


def apply_af(cam, mode):
    if mode == "auto":
        cam.set_controls({"AfMode": controls.AfModeEnum.Auto})
        try:
            logging.info("AF cycle success: %s", cam.autofocus_cycle())
        except Exception as exc:
            logging.warning("AF cycle failed: %s", exc)
        return "auto"
    cam.set_controls({"AfMode": controls.AfModeEnum.Continuous,
                      "AfSpeed": controls.AfSpeedEnum.Fast})
    return "continuous"


# ─── DETECTION ────────────────────────────────────────────────────
wechat = None
try:
    wechat = cv2.wechat_qrcode_WeChatQRCode(
        str(MODEL_DIR / "detect.prototxt"),
        str(MODEL_DIR / "detect.caffemodel"),
        str(MODEL_DIR / "sr.prototxt"),
        str(MODEL_DIR / "sr.caffemodel"))
    logging.info("WeChatQRCode loaded")
except Exception as exc:
    logging.error("WeChatQRCode failed to load (%s). Using pyzbar only. "
                  "Models expected in %s", exc, MODEL_DIR)


def detect_all(frame):
    results = []
    if wechat is not None:
        try:
            texts, points = wechat.detectAndDecode(frame)
            if texts is not None and points is not None:
                for text, pts in zip(texts, points):
                    if text:
                        c = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
                        if len(c) >= 4:
                            results.append((text, c[:4], "WeChat"))
        except cv2.error as exc:
            logging.warning("WeChat error: %s", exc)

    if not results and pyzbar is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for o in pyzbar.decode(gray):
            x, y, w, h = o.rect
            c = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                         dtype=np.float32)
            results.append((o.data.decode("utf-8", errors="replace"),
                            c, "pyzbar"))
    return results


def polygon_area(c):
    x, y = c[:, 0], c[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def pixel_to_body_offset(offset_x_px, offset_y_px, distance_m, frame_w_px):
    used_mm = sensor_crop_width_px * (PIXEL_SIZE_UM / 1000.0)
    gsd = (distance_m * used_mm / FOCAL_LENGTH_MM) / frame_w_px
    img_right = offset_x_px * gsd
    img_down = offset_y_px * gsd
    a = math.radians(CAMERA_ROTATION_DEG)
    forward = -img_down * math.cos(a) + img_right * math.sin(a)
    right = img_right * math.cos(a) + img_down * math.sin(a)
    return forward, right


# ─── DRAWING ──────────────────────────────────────────────────────
def put(img, text, org, colour=(255, 255, 255), scale=0.7, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thick + 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, colour, thick)


# ─── MAIN ─────────────────────────────────────────────────────────
def main():
    mav = connect()
    tel = Telemetry(mav)

    # Let a couple of heartbeats arrive, then check the armed state
    t0 = time.time()
    while time.time() - t0 < 2.0:
        tel.update()
        time.sleep(0.05)

    if tel.armed:
        print("\nVehicle is ARMED. Refusing to run the bench test.")
        print("Disarm it (and keep propellers off), then run again.")
        return

    print(f"Pixhawk: mode={tel.mode} armed={tel.armed}")
    print("This script never arms or changes mode.\n")

    cam = init_camera()
    af_mode = "continuous"

    phase = "SCAN"
    target = None
    history = []
    aligned_count = 0
    aligned_done = False
    last_event = None
    fps, fps_n, fps_t = 0.0, 0, time.perf_counter()

    print("Phase 1: show the START QR to the camera.")
    print("Keys: r reset | a autofocus | c continuous | s save | q quit\n")

    try:
        while True:
            tel.update()

            frame = cam.capture_array()        # already BGR
            fh, fw = frame.shape[:2]
            cx0, cy0 = fw // 2, fh // 2

            dets = detect_all(frame)
            display = frame.copy()
            cv2.drawMarker(display, (cx0, cy0), (255, 255, 0),
                           cv2.MARKER_CROSS, 40, 3)

            cmd_text = "cmd: -"

            # ───── Phase 1 ─────
            if phase == "SCAN":
                if dets:
                    text, corners, method = max(
                        dets, key=lambda d: polygon_area(d[1]))
                    history = (history + [text])[-CONSENSUS_REQ:]

                    cv2.polylines(display,
                                  [corners.astype(np.int32).reshape(-1, 1, 2)],
                                  True, (0, 165, 255), 4)
                    put(display, f"{text[:40]} [{method}]",
                        (int(corners[0][0]),
                         max(30, int(corners[0][1]) - 15)),
                        (0, 165, 255))

                    if (len(history) == CONSENSUS_REQ
                            and len(set(history)) == 1):
                        target = text
                        phase = "MATCH"
                        history = []
                        aligned_count, aligned_done = 0, False
                        last_event = None
                        print(f"START QR CONFIRMED -> target = {target!r}")
                        print("Phase 2: now show the FINAL QR(s).\n")
                else:
                    history = []

                put(display, "PHASE 1: SCAN START QR", (10, 40),
                    (0, 165, 255), 0.8)
                put(display, f"Consensus: {len(history)}/{CONSENSUS_REQ}",
                    (10, 75))

            # ───── Phase 2 ─────
            else:
                any_match = False

                for text, corners, method in dets:
                    is_match = (text == target)
                    colour = (0, 255, 0) if is_match else (0, 0, 255)
                    cv2.polylines(display,
                                  [corners.astype(np.int32).reshape(-1, 1, 2)],
                                  True, colour, 4)
                    mx, my = corners.mean(axis=0)
                    put(display,
                        f"{'MATCH' if is_match else 'WRONG'}: "
                        f"{text[:30]} [{method}]",
                        (int(corners[0][0]),
                         max(30, int(corners[0][1]) - 15)), colour)

                    if is_match:
                        any_match = True
                        cv2.line(display, (cx0, cy0), (int(mx), int(my)),
                                 (255, 0, 255), 2)

                        fwd, rgt = pixel_to_body_offset(
                            mx - cx0, my - cy0, BENCH_ALTITUDE_M, fw)
                        dist = math.hypot(fwd, rgt)
                        vf = max(-MAX_SPEED, min(MAX_SPEED, KP * fwd))
                        vr = max(-MAX_SPEED, min(MAX_SPEED, KP * rgt))

                        put(display,
                            f"offset fwd={fwd:+.3f}m right={rgt:+.3f}m "
                            f"dist={dist:.3f}m", (10, 155), (0, 255, 0))

                        if dist < ALIGN_THRESHOLD:
                            aligned_count += 1
                            vf = vr = 0.0
                        else:
                            aligned_count = 0

                        cmd_text = (f"cmd (not sent): fwd={vf:+.2f} "
                                    f"right={vr:+.2f} m/s")

                        if aligned_count >= ALIGNED_REQUIRED \
                                and not aligned_done:
                            aligned_done = True
                            print("ALIGNED (matching QR centred)")

                        # Optional sending — never while armed
                        if SEND_COMMANDS and not tel.armed:
                            send_velocity(mav, vf, vr)
                            cmd_text = (f"cmd SENT (disarmed): fwd={vf:+.2f} "
                                        f"right={vr:+.2f} m/s")

                if any_match:
                    event = "MATCH"
                elif dets:
                    event = "WRONG:" + ",".join(sorted({d[0] for d in dets}))
                else:
                    event = "NONE"

                if event != last_event:
                    if event == "MATCH":
                        print(f"MATCH: QR equals target {target!r}")
                    elif event.startswith("WRONG"):
                        print(f"WRONG QR: {event[6:]} "
                              f"(target is {target!r})")
                    last_event = event

                if not any_match:
                    aligned_count, aligned_done = 0, False

                status = ("ALIGNED" if aligned_done
                          else "MATCH" if any_match
                          else "WRONG QR" if dets
                          else "no QR in view")
                put(display, "PHASE 2: MATCH FINAL QR", (10, 40),
                    (0, 255, 0), 0.8)
                put(display, f"TARGET: {target[:40]}", (10, 75))
                put(display, f"Status: {status}", (10, 115),
                    (0, 255, 0) if any_match else (0, 0, 255))
                put(display, cmd_text, (10, 190), (255, 255, 0), 0.6)

            # Pixhawk status line (top right area, bottom-left block)
            link_col = (0, 255, 0) if tel.link_ok else (0, 0, 255)
            put(display,
                f"PIXHAWK {'OK' if tel.link_ok else 'NO LINK'} | "
                f"mode {tel.mode} | "
                f"{'ARMED!' if tel.armed else 'disarmed'} | "
                f"alt {tel.alt_m:.2f}m",
                (10, fh - 55), link_col, 0.7)

            if tel.armed:
                put(display, "ARMED - STOP: DISARM BEFORE BENCH TESTING",
                    (10, fh - 90), (0, 0, 255), 0.9, 3)

            fps_n += 1
            now = time.perf_counter()
            if now - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (now - fps_t), 0, now
            put(display,
                f"FPS {fps:.1f} | AF {af_mode} | "
                f"r reset  a AF  c cont  s save  q quit",
                (10, fh - 20), (255, 255, 255), 0.6)

            scale = DISPLAY_WIDTH / fw
            if scale < 1.0:
                display = cv2.resize(display,
                                     (DISPLAY_WIDTH, int(fh * scale)))
            cv2.imshow("Squadrone bench test (Pixhawk, props off)", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                phase, target, history = "SCAN", None, []
                aligned_count, aligned_done, last_event = 0, False, None
                print("\nReset. Phase 1: show the START QR.\n")
            elif key == ord("a"):
                af_mode = apply_af(cam, "auto")
            elif key == ord("c"):
                af_mode = apply_af(cam, "continuous")
            elif key == ord("s"):
                path = BASE_DIR / "bench_capture.jpg"
                cv2.imwrite(str(path), frame)
                print("Saved", path)

    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()