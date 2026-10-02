from pymavlink import mavutil
import time

# Connect to Pixhawk
connection = mavutil.mavlink_connection(
    '/dev/ttyAMA0',
    baud=921600
)

print("Waiting for Pixhawk heartbeat...")
connection.wait_heartbeat()
print("Pixhawk connected!")

# Display system information
print("System ID:", connection.target_system)
print("Component ID:", connection.target_component)

# ARM
print("\nSending ARM command...")
connection.mav.command_long_send(
    connection.target_system,
    connection.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,
    1,  # 1 = ARM
    0, 0, 0, 0, 0, 0
)

# Wait for response
msg = connection.recv_match(
    type='COMMAND_ACK',
    blocking=True,
    timeout=5
)

if msg:
    print("ARM ACK:", msg.result)
else:
    print("No ARM acknowledgement received")

time.sleep(3)

# Check armed status
msg = connection.recv_match(
    type='HEARTBEAT',
    blocking=True,
    timeout=5
)

if msg:
    armed = bool(
        msg.base_mode &
        mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
    )
    print("Armed:", armed)

# DISARM
print("\nSending DISARM command...")
connection.mav.command_long_send(
    connection.target_system,
    connection.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,
    0,  # 0 = DISARM
    0, 0, 0, 0, 0, 0
)

msg = connection.recv_match(
    type='COMMAND_ACK',
    blocking=True,
    timeout=5
)

if msg:
    print("DISARM ACK:", msg.result)
else:
    print("No DISARM acknowledgement received")

time.sleep(2)

print("Arm/disarm test completed.")
connection.close()