from pymavlink import mavutil
import time

def connect_to_pixhawk(port='/dev/ttyAMA0', baud=921600):
    """
    Connect to Pixhawk via UART.
    On Windows for testing use 'COM3' or whichever port.
    On Raspberry Pi use '/dev/ttyAMA0'
    """
    print(f"Connecting to Pixhawk on {port} at {baud} baud...")
    
    connection = mavutil.mavlink_connection(port, baud=baud)
    
    print("Waiting for heartbeat...")
    connection.wait_heartbeat()
    
    print(f"Heartbeat received!")
    print(f"System ID:    {connection.target_system}")
    print(f"Component ID: {connection.target_component}")
    
    return connection

def get_telemetry(connection):
    """Read basic telemetry — altitude, mode, battery."""
    
    # Request data streams
    connection.mav.request_data_stream_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL,
        10,  # 10 Hz
        1    # Start
    )
    
    print("\nReading telemetry for 5 seconds...")
    start = time.time()
    
    while time.time() - start < 5:
        msg = connection.recv_match(
            type=['VFR_HUD', 'BATTERY_STATUS', 'HEARTBEAT'],
            blocking=True,
            timeout=1
        )
        
        if msg is None:
            continue
            
        if msg.get_type() == 'VFR_HUD':
            print(f"Altitude: {msg.alt:.2f}m  "
                  f"Airspeed: {msg.airspeed:.1f}m/s  "
                  f"Groundspeed: {msg.groundspeed:.1f}m/s")
            
        elif msg.get_type() == 'BATTERY_STATUS':
            voltage = msg.voltages[0] / 1000.0
            print(f"Battery: {voltage:.2f}V")
            
        elif msg.get_type() == 'HEARTBEAT':
            mode = mavutil.mode_string_v10(msg)
            print(f"Flight mode: {mode}")

if __name__ == "__main__":
    # Change port based on your setup:
    # Windows testing via USB:  'COM3'  (check Device Manager)
    # Raspberry Pi via UART:    '/dev/ttyAMA0'
    
    mav = connect_to_pixhawk(port='/dev/ttyAMA0', baud=921600)
    get_telemetry(mav)
    print("\nConnection test complete.")