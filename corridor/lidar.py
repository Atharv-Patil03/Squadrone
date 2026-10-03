from smbus2 import SMBus, i2c_msg
import time

BUS_NUMBER = 1

# Defined Addresses
ADDR_LEFT = 0x11
ADDR_FRONT = 0x12
ADDR_RIGHT = 0x14

def read_tfmini_s(bus, address):
    OBTAIN_DATA_CMD = [0x5A, 0x05, 0x00, 0x01, 0x60]
    try:
        bus.i2c_rdwr(i2c_msg.write(address, OBTAIN_DATA_CMD))
        time.sleep(0.002)
       
        read_msg = i2c_msg.read(address, 9)
        bus.i2c_rdwr(read_msg)
        data = list(read_msg)
       
        if len(data) == 9 and data[0] == 0x59 and data[1] == 0x59:
            if (sum(data[0:8]) & 0xFF) == data[8]:
                return data[2] + (data[3] << 8)
    except Exception:
        pass
    return None

if __name__ == "__main__":
    print("Starting 3x TFmini-S LiDAR output...")
    with SMBus(BUS_NUMBER) as bus:
        while True:
            left_dist = read_tfmini_s(bus, ADDR_LEFT)
            front_dist = read_tfmini_s(bus, ADDR_FRONT)
            right_dist = read_tfmini_s(bus, ADDR_RIGHT)
           
            l_str = f"{left_dist:4d} cm" if left_dist is not None else " Error "
            f_str = f"{front_dist:4d} cm" if front_dist is not None else " Error "
            r_str = f"{right_dist:4d} cm" if right_dist is not None else " Error "
           
            print(f"Left: {l_str}  |  Front: {f_str}  |  Right: {r_str}    ", end="\r", flush=True)
            time.sleep(0.05)