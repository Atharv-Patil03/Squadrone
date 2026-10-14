from pymavlink import mavutil
import time

mav = mavutil.mavlink_connection('/dev/ttyAMA0', baud=921600)
mav.wait_heartbeat()
print("Connected")

# Request all data streams
mav.mav.request_data_stream_send(
    mav.target_system,
    mav.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL,
    10,  # 10 Hz
    1    # start
)

print("Waiting for GPS data (15 seconds)...")
start = time.time()
while time.time() - start < 15:
    msg = mav.recv_match(
        type=['GPS_RAW_INT', 'GLOBAL_POSITION_INT'],
        blocking=True,
        timeout=1
    )
    if msg:
        if msg.get_type() == 'GPS_RAW_INT':
            print(f"Satellites: {msg.satellites_visible}")
            print(f"Fix type:   {msg.fix_type}")
            break
        elif msg.get_type() == 'GLOBAL_POSITION_INT':
            print(f"Got position message — GPS working")
            break
else:
    print("Still no GPS — check physical connection")