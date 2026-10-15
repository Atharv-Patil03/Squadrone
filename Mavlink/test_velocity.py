"""
SQUADRONE — Velocity Command Test
Run AFTER takeoff test passes successfully.
Tests NED velocity commands — move North, hold, return.
"""

from pymavlink import mavutil
import time

SERIAL_PORT = '/dev/ttyAMA0'
BAUD_RATE   = 921600
TEST_ALT    = 2.0
SPEED       = 0.3    # m/s — slow for first test

def connect():
    print("Connecting...")
    mav = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUD_RATE)
    mav.wait_heartbeat()
    if mav.target_system == 0:
        print("System 0 — waiting for reboot...")
        time.sleep(3)
        mav.wait_heartbeat()
    print(f"Connected — System {mav.target_system}")
    return mav

def set_mode(mav, mode):
    mode_id = mav.mode_mapping()[mode]
    mav.mav.set_mode_send(
        mav.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )
    start = time.time()
    while time.time() - start < 10:
        msg = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if msg and msg.custom_mode == mode_id:   # FIXED
            print(f"Mode: {mode}")
            return True
    print(f"WARNING: {mode} not confirmed")
    return False

def get_alt(mav):
    mav.mav.request_data_stream_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1
    )
    msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=2)
    return round(msg.relative_alt / 1000.0, 2) if msg else 0.0

def send_velocity(mav, vx, vy, vz=0.0, duration_s=2.0):
    """
    vx = North (positive = move North)
    vy = East  (positive = move East)
    vz = Down  (positive = move Down — use negative to go up)
    """
    print(f"Moving: vx={vx} vy={vy} for {duration_s}s")
    end = time.time() + duration_s
    while time.time() < end:
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
    send_velocity(mav, 0, 0, 0, duration_s=0.5)
    print("Stopped")

def arm(mav):
    if input("Type YES to arm: ").strip().upper() != "YES":
        return False
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0
    )
    start = time.time()
    while time.time() - start < 10:
        msg = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if msg and msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            print("Armed")
            return True
    return False

def takeoff(mav, alt):
    print(f"Taking off to {alt}m...")
    mav.mav.request_data_stream_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1
    )
    time.sleep(0.5)
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, alt
    )
    start = time.time()
    while time.time() - start < 30:
        a = get_alt(mav)
        print(f"  Alt: {a:.2f}m", end='\r')
        if a >= alt * 0.95:    # FIXED — was 0.90
            print(f"\nReached {a:.2f}m")
            return True
        time.sleep(0.2)
    print(f"\nWARNING: Did not reach {alt}m")
    return False

def land(mav):
    print("Landing...")
    set_mode(mav, 'LAND')
    time.sleep(2)              # FIXED — wait for baro to settle
    start = time.time()
    while time.time() - start < 30:
        a = get_alt(mav)
        print(f"  Alt: {a:.2f}m", end='\r')
        if a < 0.5:            # FIXED — was no threshold check
            time.sleep(3)      # FIXED — confirm fully settled
            print("\nLanded")
            return True
        time.sleep(0.2)
    print("\nWARNING: Landing not confirmed")
    return False

def disarm(mav):
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 0, 0, 0, 0, 0, 0, 0
    )
    time.sleep(2)
    print("Disarmed")

if __name__ == "__main__":
    mav = connect()

    try:
        set_mode(mav, 'GUIDED')
        time.sleep(1)

        if not arm(mav):
            exit()
        time.sleep(2)

        if not takeoff(mav, TEST_ALT):
            land(mav)
            disarm(mav)
            exit()

        time.sleep(2)          # FIXED — stabilise after takeoff

        # Test 1 — Move North
        print("\nTest 1: Moving North")
        send_velocity(mav, SPEED, 0, duration_s=3)
        stop(mav)
        time.sleep(2)

        # Test 2 — Move South back
        print("Test 2: Moving South (back)")
        send_velocity(mav, -SPEED, 0, duration_s=3)
        stop(mav)
        time.sleep(2)

        # Test 3 — Move East
        print("Test 3: Moving East")
        send_velocity(mav, 0, SPEED, duration_s=3)
        stop(mav)
        time.sleep(2)

        # Test 4 — Move West back
        print("Test 4: Moving West (back)")
        send_velocity(mav, 0, -SPEED, duration_s=3)
        stop(mav)
        time.sleep(2)

        print("\nAll velocity tests done — landing")
        land(mav)
        disarm(mav)
        print("Complete")

    except KeyboardInterrupt:
        print("\nCtrl+C — landing")
        set_mode(mav, 'LAND')
        time.sleep(8)
        disarm(mav)