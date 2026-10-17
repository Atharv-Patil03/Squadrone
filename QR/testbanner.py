"""
SQUADRONE — Banner Land Test
File: test_banner_land.py
Team: Squadrone AT2026RC60, PVGCOET Pune, SAEINDIA AeroTHON 2026

Test sequence:
  1. Connect MAVLink
  2. Arm & take off to TAKEOFF_ALT_M (3m)
  3. Hover and align downward ArduCam over green banner on ground
  4. Once centred and stable → descend to LAND_ALT_M (0.5m)
  5. Switch to LAND mode → drone lands on banner centre

Purpose:
  Validate banner detection + alignment logic in real flight
  BEFORE attempting corridor navigation.

Camera:
  ArduCam (downward-facing, rear-mounted) on cam1
  Auto-detected by model name — same technique as QR module.
  If auto-detect fails, set ARDUCAM_INDEX = 1 manually.

Green banner: 5ft x 5ft, placed on ground in front of drone.
ArduCam is rear-mounted facing down, so banner appears near
TOP of frame when drone is hovering forward of it.
"""

import cv2
import time
import logging
import numpy as np

from picamera2 import Picamera2
from pymavlink import mavutil

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# MAVLink
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE   = 921600

# Flight
TAKEOFF_ALT_M     = 3.0    # cruise altitude for alignment
TAKEOFF_THRESHOLD = 0.95   # fraction of target alt = takeoff done
LAND_ALT_M        = 0.5    # descend to this before switching LAND mode

# Camera
CAM_WIDTH        = 640
CAM_HEIGHT       = 480
ARDUCAM_INDEX    = None    # None = auto-detect ArduCam; set 0 or 1 to force

# Banner detection (HSV green)
LOWER_GREEN        = np.array([35,  80,  50])
UPPER_GREEN        = np.array([85, 255, 255])
MIN_AREA           = 300    # lowered — banner may be far away and small
MIN_ASPECT_RATIO   = 0.2    # relaxed — partial view at low altitude
MAX_ASPECT_RATIO   = 5.0    # relaxed — same
MIN_RECTANGULARITY = 0.20   # relaxed — partial frame fill at low alt

# Alignment
ALIGN_THRESHOLD_PX = 30    # centroid must be within this of frame centre
ALIGN_HOLD_FRAMES  = 15    # consecutive aligned frames → confirmed centred
SEARCH_CREEP_SPD   = 0.2   # m/s — creep forward when banner not visible
ALIGN_SPEED        = 0.25  # m/s — max correction speed during alignment

# ──────────────────────────────────────────────────────────────────────────────
# CAMERA — auto-detect ArduCam (same technique as QR module)
# ──────────────────────────────────────────────────────────────────────────────

def resolve_arducam_index():
    """
    Find the ArduCam index. Pi5 has two cameras:
      cam0 = AI Cam (imx500)
      cam1 = ArduCam (imx519)
    Auto-detects by model name. Falls back to ARDUCAM_INDEX or index 1.
    """
    if ARDUCAM_INDEX is not None:
        log.info(f"ArduCam index forced to {ARDUCAM_INDEX}")
        return ARDUCAM_INDEX

    try:
        cameras = Picamera2.global_camera_info()
        log.info("Cameras: %s",
                 [(c.get("Num"), c.get("Model")) for c in cameras])

        # Prefer imx519 (ArduCam IMX519 Autofocus)
        for c in cameras:
            model = str(c.get("Model", "")).lower()
            if "imx519" in model or "arducam" in model:
                idx = c.get("Num", 1)
                log.info(f"ArduCam auto-detected at index {idx} ({model})")
                return idx

        # Fallback: use index 1 (cam1)
        log.warning("ArduCam not found by name — defaulting to index 1")
        return 1

    except Exception as e:
        log.warning(f"Camera info failed: {e} — defaulting to index 1")
        return 1


def start_camera():
    idx = resolve_arducam_index()
    cam = Picamera2(idx)
    cfg = cam.create_preview_configuration(
        main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"},
        buffer_count=4
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(2)
    log.info(f"ArduCam started (index {idx}, {CAM_WIDTH}x{CAM_HEIGHT})")
    return cam

# ──────────────────────────────────────────────────────────────────────────────
# BANNER DETECTION
# ──────────────────────────────────────────────────────────────────────────────

_kernel = np.ones((5, 5), np.uint8)

def detect_banner(frame):
    """
    Detect largest green banner in BGR frame.
    Returns dict {center:(cx,cy), box:(x,y,w,h), area} or None.
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        ar   = w / h
        rect = area / (w * h)
        if not (MIN_ASPECT_RATIO <= ar <= MAX_ASPECT_RATIO):
            continue
        if rect < MIN_RECTANGULARITY:
            continue
        if area > best_area:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            best_area = area
            best = {"center": (cx, cy), "box": (x, y, w, h), "area": area}

    return best

# ──────────────────────────────────────────────────────────────────────────────
# MAVLINK HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def connect_mavlink():
    log.info(f"Connecting {SERIAL_PORT} @ {BAUD_RATE}…")
    mav = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUD_RATE)
    # Wait until we get a heartbeat with a valid (non-zero) system ID.
    # The first heartbeat packet sometimes arrives before the FC has set
    # its sysid, giving target_system=0. Keep reading until it's non-zero.
    for attempt in range(20):
        mav.wait_heartbeat(timeout=3)
        if mav.target_system != 0:
            break
        log.info(f"  target_system=0 (attempt {attempt+1}/20), retrying…")
    else:
        raise RuntimeError("target_system=0 after 20 attempts — check connection — RTL !!!")
    log.info(f"Heartbeat — system {mav.target_system} "
             f"component {mav.target_component}")
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
            log.info(f"Mode: {mode_name}")
            return True
    log.warning(f"Mode {mode_name} not confirmed")
    return False

def get_altitude(mav):
    msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=2)
    if msg:
        return msg.relative_alt / 1000.0
    return 0.0

def send_velocity(mav, forward, right, down=0.0, duration_s=0.2):
    """Body-frame velocity — forward=nose, right=drone's right."""
    end = time.time() + duration_s
    while time.time() < end:
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

def stop_drone(mav):
    send_velocity(mav, 0, 0, 0, duration_s=0.3)

def arm_and_takeoff(mav, target_alt_m):
    log.info("Setting GUIDED…")
    set_mode(mav, 'GUIDED')

    log.info("Arming…")
    mav.arducopter_arm()
    mav.motors_armed_wait()
    log.info("Armed")

    log.info(f"Takeoff → {target_alt_m}m")
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, target_alt_m
    )

    threshold = target_alt_m * TAKEOFF_THRESHOLD
    start = time.time()
    while time.time() - start < 30:
        alt = get_altitude(mav)
        log.info(f"  Alt: {alt:.2f}m")
        if alt >= threshold:
            log.info(f"Takeoff done at {alt:.2f}m")
            return True
        time.sleep(0.5)

    log.warning("Takeoff timeout")
    return False

def descend_to(mav, target_alt_m, timeout_s=15):
    """Descend to target_alt_m using NED position setpoint."""
    log.info(f"Descending to {target_alt_m}m…")
    mav.mav.set_position_target_local_ned_send(
        0,
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111111000,   # only Z
        0, 0, -target_alt_m,
        0, 0, 0,
        0, 0, 0,
        0, 0
    )
    start = time.time()
    while time.time() - start < timeout_s:
        alt = get_altitude(mav)
        log.info(f"  Alt: {alt:.2f}m → target {target_alt_m}m")
        if abs(alt - target_alt_m) < 0.3:
            log.info(f"  Reached {alt:.2f}m")
            return True
        time.sleep(0.5)
    log.warning("Descend timeout")
    return False

# ──────────────────────────────────────────────────────────────────────────────
# STATE: ALIGN TO BANNER
# ──────────────────────────────────────────────────────────────────────────────

def state_align_to_banner(mav, cam):
    """
    Hover at TAKEOFF_ALT_M and align downward camera over green banner.

    Camera is REAR-mounted facing DOWN.
    Banner is placed IN FRONT of drone on the ground.

    Frame geometry:
      Drone nose  → top of frame
      Drone tail  → bottom of frame  (camera is here)
      offset_x +ve → banner is RIGHT of centre → move right
      offset_y -ve → banner is near TOP        → banner is AHEAD  → move forward
      offset_y +ve → banner is near BOTTOM     → banner is BEHIND → move backward

    Returns True when banner is stably centred.
    """
    log.info("\n── ALIGN TO BANNER ──")
    log.info(f"  Threshold: ±{ALIGN_THRESHOLD_PX}px for {ALIGN_HOLD_FRAMES} frames")
    set_mode(mav, 'GUIDED')

    cx_frame = CAM_WIDTH  // 2
    cy_frame = CAM_HEIGHT // 2

    # Scale: half-frame offset → ALIGN_SPEED
    PX_SCALE_X = ALIGN_SPEED / (CAM_WIDTH  / 2)
    PX_SCALE_Y = ALIGN_SPEED / (CAM_HEIGHT / 2)

    align_count  = 0
    no_banner_ct = 0

    while True:
        rgb   = cam.capture_array()
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        banner = detect_banner(frame)

        # ── No banner visible ──────────────────────────────────────
        if banner is None:
            align_count  = 0
            no_banner_ct += 1
            action = "creeping forward" if no_banner_ct > 20 else "holding"
            log.info(f"  NO BANNER DETECTED (frame {no_banner_ct}) — {action}")
            if no_banner_ct > 20:
                # Banner is in front — creep forward to bring it into view
                send_velocity(mav, SEARCH_CREEP_SPD, 0, duration_s=0.5)
                no_banner_ct = 0
            else:
                stop_drone(mav)
            continue

        no_banner_ct = 0
        cx, cy = banner["center"]

        offset_x = cx - cx_frame   # +ve → right
        offset_y = cy - cy_frame   # +ve → below centre (banner behind) → move back

        aligned = (abs(offset_x) <= ALIGN_THRESHOLD_PX and
                   abs(offset_y) <= ALIGN_THRESHOLD_PX)

        log.info(f"  Banner: ({cx},{cy})  "
                 f"offset: ({offset_x:+d},{offset_y:+d})px  "
                 f"area: {banner['area']}  "
                 f"{'ALIGNED' if aligned else 'correcting'}")

        if aligned:
            align_count += 1
            stop_drone(mav)
            if align_count >= ALIGN_HOLD_FRAMES:
                log.info("  ✓ BANNER CENTRED AND STABLE")
                return True
        else:
            align_count = 0
            # offset_x +ve → banner right → vright positive
            vright   = max(-ALIGN_SPEED,
                           min(ALIGN_SPEED,  offset_x * PX_SCALE_X))
            # offset_y +ve → banner behind → move backward (vforward negative)
            # offset_y -ve → banner ahead  → move forward  (vforward positive)
            vforward = max(-ALIGN_SPEED,
                           min(ALIGN_SPEED, -offset_y * PX_SCALE_Y))
            send_velocity(mav, vforward, vright, duration_s=0.2)

# ──────────────────────────────────────────────────────────────────────────────
# STATE: LAND ON BANNER
# ──────────────────────────────────────────────────────────────────────────────

def state_land_on_banner(mav, cam):
    """
    Descend slowly to LAND_ALT_M while keeping banner centred,
    then switch to LAND mode.

    During descent, banner grows in frame (area increases).
    Keep correcting lateral position as we descend.
    At LAND_ALT_M → switch LAND mode → ArduCopter handles final touchdown.
    """
    log.info("\n── LAND ON BANNER ──")

    cx_frame = CAM_WIDTH  // 2
    cy_frame = CAM_HEIGHT // 2
    PX_SCALE_X = ALIGN_SPEED / (CAM_WIDTH  / 2)
    PX_SCALE_Y = ALIGN_SPEED / (CAM_HEIGHT / 2)

    DESCEND_SPEED = 0.15   # m/s downward — slow descent

    while True:
        alt = get_altitude(mav)
        log.info(f"  Alt: {alt:.2f}m")

        if alt <= LAND_ALT_M:
            log.info(f"  Reached {alt:.2f}m → switching to LAND")
            stop_drone(mav)
            time.sleep(0.5)
            set_mode(mav, 'LAND')
            log.info("  LAND mode active — waiting for touchdown…")
            # Wait for disarm OR altitude near zero
            start = time.time()
            while time.time() - start < 45:
                if not mav.motors_armed():
                    log.info("  ✓ Landed and disarmed automatically")
                    return True
                alt_now = get_altitude(mav)
                log.info(f"  Landing… alt={alt_now:.2f}m  armed={mav.motors_armed()}")
                if alt_now < 0.15:
                    log.info("  ✓ Altitude near zero — touchdown confirmed")
                    time.sleep(3)   # let ArduCopter disarm on its own
                    return True
                time.sleep(1)
            log.warning("  Land timeout (45s) — check drone manually")
            return True

        # Capture frame and keep centred while descending
        rgb   = cam.capture_array()
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        banner = detect_banner(frame)

        if banner is not None:
            cx, cy = banner["center"]
            offset_x = cx - cx_frame
            offset_y = cy - cy_frame
            log.info(f"  Banner: ({cx},{cy})  offset: ({offset_x:+d},{offset_y:+d})px  area:{banner['area']}")

            vright   = max(-ALIGN_SPEED,
                           min(ALIGN_SPEED,  offset_x * PX_SCALE_X))
            vforward = max(-ALIGN_SPEED,
                           min(ALIGN_SPEED, -offset_y * PX_SCALE_Y))
        else:
            log.info("  Banner NOT DETECTED during descent (may fill frame) — holding lateral, continuing down")
            vforward = 0.0
            vright   = 0.0

        # Descend + lateral correction combined
        send_velocity(mav, vforward, vright,
                      down=DESCEND_SPEED, duration_s=0.3)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("SQUADRONE — Banner Land Test")
    log.info("Place 5x5ft green banner on ground in FRONT of drone")
    log.info("=" * 60)

    cam = None
    mav = None

    try:
        # 1. Init hardware
        log.info("\n[1/4] Starting ArduCam…")
        cam = start_camera()

        log.info("\n[2/4] Connecting MAVLink…")
        mav = connect_mavlink()

        # Quick camera check — capture one frame and check for banner
        log.info("\nCamera check — looking for green banner…")
        rgb   = cam.capture_array()
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        b = detect_banner(frame)
        if b:
            log.info(f"  Banner visible at {b['center']} area={b['area']} ✓")
        else:
            log.warning("  Banner NOT visible in current frame — "
                        "check banner position and lighting before flying")

        input("\nPress ENTER to arm and take off, or Ctrl+C to abort…\n")

        # 2. Arm & takeoff
        log.info("\n[3/4] Arm & takeoff…")
        if not arm_and_takeoff(mav, TAKEOFF_ALT_M):
            log.error("Takeoff failed → RTL")
            set_mode(mav, 'RTL')
            return

        time.sleep(1.0)   # stabilise

        # 3. Align to banner
        log.info("\n[4/4] Aligning to banner…")
        state_align_to_banner(mav, cam)

        # 4. Descend and land on banner
        state_land_on_banner(mav, cam)

        log.info("\n✓ Test complete — drone landed on banner")

    except KeyboardInterrupt:
        log.warning("\n!!! Ctrl+C — RTL !!!")
        if mav:
            set_mode(mav, 'RTL')

    except Exception as e:
        log.error(f"\n!!! EXCEPTION: {e} — RTL !!!")
        if mav:
            set_mode(mav, 'RTL')
        raise

    finally:
        if cam:
            cam.stop()
            log.info("Camera stopped")


if __name__ == "__main__":
    main()