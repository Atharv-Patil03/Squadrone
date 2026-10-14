"""
SQUADRONE — Autonomous Takeoff and Land Test
File: test_takeoff_land.py

Tests: GUIDED mode → Arm → Takeoff → Hold → Land → Disarm
"""

from pymavlink import mavutil
import time

# ─── CONFIG — change only these ───────────────────────────────────
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 921600
TARGET_ALT = 2.0  # metres — keep at 2m for first test
HOLD_TIME = 5  # seconds to hover before landing


# ─── CONNECT ──────────────────────────────────────────────────────
def connect():
    print("Connecting...")
    mav = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUD_RATE)
    mav.wait_heartbeat()
    print(f"Connected — System {mav.target_system}")
    return mav


# ─── HELPERS ──────────────────────────────────────────────────────
def get_alt(mav):
    msg = mav.recv_match(type="VFR_HUD", blocking=True, timeout=2)
    return round(msg.alt, 2) if msg else 0.0


def is_armed(mav):
    msg = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
    if msg:
        return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    return False


def set_mode(mav, mode):
    print(f"Setting mode: {mode}")
    mode_id = mav.mode_mapping()[mode]
    mav.mav.set_mode_send(
        mav.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id
    )
    start = time.time()
    while time.time() - start < 10:
        msg = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if msg and mavutil.mode_string_v10(msg) == mode:
            print(f"Mode confirmed: {mode}")
            return True
    print(f"WARNING: {mode} not confirmed")
    return False


# ─── ACTIONS ──────────────────────────────────────────────────────
def arm(mav):
    print("\nAbout to ARM. Final checks:")
    print("  1. Props secure?")
    print("  2. Area clear?")
    print("  3. RC transmitter ON?")
    print("  4. You are ready to take manual control?")
    if input("\nType YES to arm: ").strip().upper() != "YES":
        print("Cancelled")
        return False

    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    start = time.time()
    while time.time() - start < 10:
        if is_armed(mav):
            print("ARMED")
            return True
        time.sleep(0.3)
    print("ARM FAILED — check pre-arm errors in Mission Planner")
    return False


def disarm(mav):
    print("Disarming...")
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    time.sleep(2)
    print("Disarmed")


def takeoff(mav, alt):
    print(f"Taking off to {alt}m...")
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        alt,
    )
    start = time.time()
    while time.time() - start < 30:
        a = get_alt(mav)
        print(f"  Alt: {a:.2f}m", end="\r")
        if a >= alt * 0.90:
            print(f"\nReached {a:.2f}m")
            return True
        time.sleep(0.2)
    print("\nWARNING: Target altitude not reached in 30s")
    return False


def hold(mav, seconds):
    print(f"Holding for {seconds}s in LOITER...")
    set_mode(mav, "LOITER")
    for i in range(seconds, 0, -1):
        print(f"  {i}s remaining  Alt:{get_alt(mav):.1f}m", end="\r")
        time.sleep(1)
    print("\nHold done")


def land(mav):
    print("Landing...")
    set_mode(mav, "LAND")
    start = time.time()
    while time.time() - start < 30:
        a = get_alt(mav)
        print(f"  Alt: {a:.2f}m", end="\r")
        if a < 0.3:
            print("\nLanded")
            return True
        time.sleep(0.2)
    print("\nWARNING: Landing not confirmed")
    return False


# ─── MAIN ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    mav = connect()

    try:
        # Step 1 — GUIDED mode
        if not set_mode(mav, "GUIDED"):
            print("Cannot set GUIDED — stopping")
            exit()
        time.sleep(1)

        # Step 2 — Arm (requires manual YES confirmation)
        if not arm(mav):
            exit()
        time.sleep(2)

        # Step 3 — Takeoff
        if not takeoff(mav, TARGET_ALT):
            print("Takeoff issue — landing")
            land(mav)
            disarm(mav)
            exit()

        # Step 4 — Hold position
        hold(mav, HOLD_TIME)

        # Step 5 — Land
        land(mav)

        # Step 6 — Disarm
        disarm(mav)

        print("\nTest complete")

    except KeyboardInterrupt:
        print("\nCtrl+C detected — LANDING NOW")
        set_mode(mav, "LAND")
        time.sleep(8)
        disarm(mav)
        print("Safe")
