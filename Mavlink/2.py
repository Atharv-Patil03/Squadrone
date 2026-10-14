from pymavlink import mavutil

mav = mavutil.mavlink_connection('/dev/ttyAMA0', baud=921600)
mav.wait_heartbeat()

msg = mav.recv_match(type='GPS_RAW_INT', blocking=True, timeout=5)
if msg:
    print(f"Satellites: {msg.satellites_visible}")
    print(f"Fix type: {msg.fix_type}  (3 = 3D fix, need minimum 3)")
else:
    print("No GPS message received")