from pymavlink import mavutil
import time

# ─── CONNECTION ───────────────────────────────────────────────────
def connect(port='/dev/ttyAMA0', baud=921600):
    mav = mavutil.mavlink_connection(port, baud=baud)
    mav.wait_heartbeat()
    print(f"Connected — System {mav.target_system}")
    return mav

# ─── MODE SETTING ─────────────────────────────────────────────────
def set_mode(mav, mode_name):
    """Set ArduPilot flight mode by name."""
    mode_id = mav.mode_mapping()[mode_name]
    mav.mav.set_mode_send(
        mav.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )
    # Verify mode was set
    start = time.time()
    while time.time() - start < 5:
        msg = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if msg:
            current_mode = mavutil.mode_string_v10(msg)
            if current_mode == mode_name:
                print(f"Mode set to: {mode_name}")
                return True
    print(f"WARNING: Mode change to {mode_name} not confirmed")
    return False

# ─── ARM ──────────────────────────────────────────────────────────
def arm(mav):
    """Arm the drone."""
    print("Arming...")
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,    # confirmation
        1,    # arm (0 = disarm)
        0, 0, 0, 0, 0, 0
    )
    # Wait for arm confirmation
    start = time.time()
    while time.time() - start < 10:
        msg = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if msg and msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            print("Armed successfully")
            return True
    print("ERROR: Arming failed")
    return False

# ─── DISARM ───────────────────────────────────────────────────────
def disarm(mav):
    """Disarm the drone."""
    print("Disarming...")
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        0,    # disarm
        0, 0, 0, 0, 0, 0
    )
    time.sleep(2)
    print("Disarmed")

# ─── TAKEOFF ──────────────────────────────────────────────────────
def takeoff(mav, altitude_m=2.0):
    """
    Command takeoff to specified altitude.
    Drone must be armed and in GUIDED mode first.
    """
    print(f"Taking off to {altitude_m}m...")
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,          # confirmation
        0,          # min pitch
        0, 0, 0,    # empty
        0, 0,       # lat, lon (0 = current position)
        altitude_m  # target altitude in metres
    )
    # Wait until altitude is reached
    print("Waiting to reach altitude...")
    start = time.time()
    while time.time() - start < 30:
        msg = mav.recv_match(type='VFR_HUD', blocking=True, timeout=1)
        if msg:
            current_alt = msg.alt
            print(f"  Current altitude: {current_alt:.2f}m", end='\r')
            if current_alt >= altitude_m * 0.95:
                print(f"\nTarget altitude reached: {current_alt:.2f}m")
                return True
    print("\nWARNING: Altitude not reached within 30 seconds")
    return False

# ─── LAND ─────────────────────────────────────────────────────────
def land(mav):
    """Command drone to land at current position."""
    print("Landing...")
    set_mode(mav, 'LAND')
    # Wait for landing
    start = time.time()
    while time.time() - start < 30:
        msg = mav.recv_match(type='VFR_HUD', blocking=True, timeout=1)
        if msg:
            print(f"  Altitude: {msg.alt:.2f}m", end='\r')
            if msg.alt < 0.3:
                print("\nLanded successfully")
                return True
    print("\nWARNING: Landing not confirmed")
    return False

# ─── HOLD POSITION ────────────────────────────────────────────────
def hold_position(mav, duration_s=5):
    """Switch to LOITER mode to hold position."""
    print(f"Holding position for {duration_s} seconds...")
    set_mode(mav, 'LOITER')
    time.sleep(duration_s)
    set_mode(mav, 'GUIDED')
    print("Position hold complete")

# ─── EMERGENCY STOP ───────────────────────────────────────────────
def emergency_stop(mav):
    """Force immediate land — use only in emergency."""
    print("EMERGENCY STOP")
    set_mode(mav, 'LAND')

# ─── FULL TEST SEQUENCE ───────────────────────────────────────────
if __name__ == "__main__":
    mav = connect(port='/dev/ttyAMA0', baud=921600)
    
    try:
        # Step 1: Set GUIDED mode
        set_mode(mav, 'GUIDED')
        time.sleep(1)
        
        # Step 2: Arm
        if not arm(mav):
            print("Arming failed. Stopping.")
            exit()
        time.sleep(2)
        
        # Step 3: Takeoff to 2m
        if not takeoff(mav, altitude_m=2.0):
            print("Takeoff failed. Landing.")
            land(mav)
            disarm(mav)
            exit()
        
        # Step 4: Hold for 5 seconds
        hold_position(mav, duration_s=5)
        
        # Step 5: Land
        land(mav)
        
        # Step 6: Disarm
        disarm(mav)
        
        print("\nTest sequence complete.")
        
    except KeyboardInterrupt:
        print("\nKeyboard interrupt — landing now")
        emergency_stop(mav)
        time.sleep(3)
        disarm(mav)