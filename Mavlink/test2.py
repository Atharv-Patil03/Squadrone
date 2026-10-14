from pymavlink import mavutil
import time

SERIAL_PORT  = '/dev/ttyAMA0'
BAUD_RATE    = 921600
TARGET_ALT   = 2.0
HOLD_TIME    = 5

def connect():
    print("Connecting...")
    mav = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUD_RATE)
    mav.wait_heartbeat()
    print(f"Connected — System {mav.target_system}")
    return mav

def get_alt(mav):
    mav.mav.request_data_stream_send(mav.target_system, mav.target_component, mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)
    msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=2)
    if msg:
        return round(msg.relative_alt / 1000.0, 2)
    return 0.0

def set_mode(mav, mode):
    print(f"Setting mode: {mode}")
    mode_id = mav.mode_mapping()[mode]
    mav.mav.set_mode_send(mav.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)
    start = time.time()
    while time.time() - start < 10:
        msg = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if msg and mavutil.mode_string_v10(msg) == mode:
            print(f"Mode confirmed: {mode}")
            return True
    print(f"WARNING: {mode} not confirmed")
    return False

def arm(mav):
    print("\nFinal checks:")
    print("  1. Props secure?")
    print("  2. Area clear?")
    print("  3. RC transmitter ON?")
    if input("\nType YES to arm: ").strip().upper() != "YES":
        print("Cancelled")
        return False
    mav.mav.command_long_send(mav.target_system, mav.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
    start = time.time()
    while time.time() - start < 10:
        msg = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if msg and msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            print("ARMED")
            return True
        time.sleep(0.2)
    print("ARM FAILED")
    return False

def disarm(mav):
    mav.mav.command_long_send(mav.target_system, mav.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 0, 0, 0, 0, 0, 0)
    time.sleep(2)
    print("Disarmed")

def takeoff(mav, alt):
    print(f"Taking off to {alt}m...")
    mav.mav.request_data_stream_send(mav.target_system, mav.target_component, mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)
    time.sleep(0.5)
    mav.mav.command_long_send(mav.target_system, mav.target_component, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, alt)
    start = time.time()
    while time.time() - start < 30:
        msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=2)
        if msg:
            a = round(msg.relative_alt / 1000.0, 2)
            print(f"  Alt: {a:.2f}m / {alt}m", end='\r')
            if a >= alt * 0.90:
                print(f"\nReached {a:.2f}m")
                return True
        time.sleep(0.1)
    print(f"\nWARNING: Did not reach {alt}m")
    return False

def hold(mav, seconds):
    print(f"Holding {seconds}s in LOITER...")
    set_mode(mav, 'LOITER')
    for i in range(seconds, 0, -1):
        msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
        a = round(msg.relative_alt / 1000.0, 2) if msg else 0.0
        print(f"  {i}s  Alt:{a:.2f}m", end='\r')
        time.sleep(1)
    print("\nHold done")

def land(mav):
    print("Landing...")
    set_mode(mav, 'LAND')
    start = time.time()
    while time.time() - start < 30:
        msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=2)
        if msg:
            a = round(msg.relative_alt / 1000.0, 2)
            print(f"  Alt: {a:.2f}m", end='\r')
            if a < 0.3:
                print("\nLanded")
                return True
        time.sleep(0.1)
    print("\nWARNING: Landing not confirmed")
    return False

if __name__ == "__main__":
    mav = connect()
    try:
        if not set_mode(mav, 'GUIDED'):
            exit()
        time.sleep(1)
        if not arm(mav):
            exit()
        time.sleep(2)
        if not takeoff(mav, TARGET_ALT):
            land(mav)
            disarm(mav)
            exit()
        hold(mav, HOLD_TIME)
        land(mav)
        disarm(mav)
        print("\nTest complete")
    except KeyboardInterrupt:
        print("\nCtrl+C — LANDING")
        set_mode(mav, 'LAND')
        time.sleep(8)
        disarm(mav)
