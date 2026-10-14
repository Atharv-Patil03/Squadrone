from pymavlink import mavutil
import time

PORT = '/dev/ttyAMA0'
BAUD = 921600

print("Connecting to Pixhawk...")

mav = mavutil.mavlink_connection(
    PORT,
    baud=BAUD
)

print("Waiting for heartbeat...")
mav.wait_heartbeat()

print("\nCONNECTED")
print("System ID:", mav.target_system)
print("Component ID:", mav.target_component)

# --------------------------------------------------
# SHOW CURRENT MODE
# --------------------------------------------------

msg = mav.recv_match(
    type='HEARTBEAT',
    blocking=True,
    timeout=3
)

if msg:
    print("Current mode:", mavutil.mode_string_v10(msg))


# --------------------------------------------------
# SET GUIDED MODE
# --------------------------------------------------

print("\nSetting GUIDED mode...")

mode_mapping = mav.mode_mapping()

if mode_mapping is None:
    print("ERROR: Flight controller did not provide mode mapping.")
else:
    print("Available modes:")
    print(mode_mapping)

    if 'GUIDED' not in mode_mapping:
        print("\nERROR: GUIDED mode is not available.")
    else:

        guided_mode = mode_mapping['LOITER']

        mav.mav.set_mode_send(
            mav.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            guided_mode
        )

        time.sleep(2)

        msg = mav.recv_match(
            type='HEARTBEAT',
            blocking=True,
            timeout=3
        )

        if msg:
            current_mode = mavutil.mode_string_v10(msg)
            print("Mode reported by Pixhawk:", current_mode)

# --------------------------------------------------
# TAKEOFF COMMAND TEST
# --------------------------------------------------

print("\nSending TAKEOFF command...")
print("NOTE: Drone is NOT armed.")

mav.mav.command_long_send(
    mav.target_system,
    mav.target_component,

    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,

    0,

    0,      # pitch
    0,      # empty
    0,
    0,
    0,      # latitude = current
    0,      # longitude = current
    2.0     # altitude = 2 metres
)

print("TAKEOFF command sent.")

# --------------------------------------------------
# CHECK ACK
# --------------------------------------------------

print("\nWaiting for COMMAND_ACK...")

start = time.time()

while time.time() - start < 5:

    msg = mav.recv_match(
        type='COMMAND_ACK',
        blocking=True,
        timeout=1
    )

    if msg:

        print("\nCOMMAND_ACK received")
        print("Command:", msg.command)
        print("Result:", msg.result)

        if msg.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:

            if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                print("TAKEOFF COMMAND ACCEPTED")

            elif msg.result == mavutil.mavlink.MAV_RESULT_DENIED:
                print("TAKEOFF COMMAND DENIED")

            elif msg.result == mavutil.mavlink.MAV_RESULT_TEMPORARILY_REJECTED:
                print("TAKEOFF TEMPORARILY REJECTED")

            else:
                print("TAKEOFF RESULT CODE:", msg.result)

            break

else:
    print("No COMMAND_ACK received.")

print("\nTest finished.")
mav.close()