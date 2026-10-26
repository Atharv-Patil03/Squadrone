#!/usr/bin/env python3
"""
Simple GUI: click a switch position -> PWM written directly to Pixhawk SERVO5 / SERVO6
(MAV_CMD_DO_SET_SERVO over MAVLink2 on TELEM2). No RC override involved.

    python3 pixhawk_servo_gui.py --device /dev/serial0 --baud 57600
"""
import os
os.environ["MAVLINK20"] = "1"

import argparse
import threading
import tkinter as tk
from pymavlink import mavutil

# --- edit these PWM values if needed ---
SERVO5_POS = [("LOW", 1000), ("MID", 1500), ("HIGH", 2000)]   # 3-position switch
SERVO6_POS = [("OFF", 1000), ("ON", 2000)]                    # 2-position switch

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="/dev/ttyAMA0")
ap.add_argument("--baud", type=int, default=921600)
args = ap.parse_args()

master = None


def connect():
    global master
    master = mavutil.mavlink_connection(args.device, baud=args.baud, source_system=255)
    master.wait_heartbeat()
    status.config(text="Connected", fg="#4cd964")


def set_servo(servo_no, pwm):
    if master is None:
        status.config(text="Not connected yet", fg="#ff5555")
        return
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0,
        servo_no, pwm, 0, 0, 0, 0, 0)
    status.config(text=f"SERVO{servo_no} = {pwm} us", fg="#4cd964")


class Switch(tk.Canvas):
    """Vertical switch; click a position to move the lever there."""
    def __init__(self, parent, servo_no, positions, initial):
        self.servo_no, self.pos, self.idx = servo_no, positions, initial
        n = len(positions)
        super().__init__(parent, width=170, height=60 * n + 30, bg="#1e1f24", highlightthickness=0)
        self.bind("<Button-1>", self.click)
        self.draw()

    def y(self, i):
        return 35 + i * 60

    def draw(self):
        self.delete("all")
        n = len(self.pos)
        self.create_rectangle(40, self.y(0) - 25, 80, self.y(n - 1) + 25, fill="#2d2f36", outline="#666", width=2)
        for i, (name, pwm) in enumerate(self.pos):
            c = "#4cd964" if i == self.idx else "#aaa"
            self.create_text(95, self.y(i) - 8, text=name, anchor="w", fill=c, font=("Arial", 11, "bold"))
            self.create_text(95, self.y(i) + 9, text=f"{pwm} us", anchor="w", fill=c, font=("Arial", 9))
        y = self.y(self.idx)
        self.create_line(60, y, 60, (self.y(0) + self.y(n - 1)) / 2, width=8, fill="#bbb", capstyle="round")
        self.create_oval(46, y - 14, 74, y + 14, fill="#4cd964", outline="white", width=2)

    def click(self, ev):
        self.idx = min(range(len(self.pos)), key=lambda k: abs(self.y(k) - ev.y))
        self.draw()
        set_servo(self.servo_no, self.pos[self.idx][1])


root = tk.Tk()
root.title("Pixhawk Servo Switches")
root.configure(bg="#14151a")
root.resizable(False, False)

for col, (title, no, pos, init) in enumerate([("SERVO5\n(3-position)", 5, SERVO5_POS, 1),
                                               ("SERVO6\n(2-position)", 6, SERVO6_POS, 0)]):
    tk.Label(root, text=title, bg="#14151a", fg="white", font=("Arial", 11, "bold")).grid(row=0, column=col, pady=(10, 0))
    Switch(root, no, pos, init).grid(row=1, column=col, padx=10, pady=5, sticky="n")

status = tk.Label(root, text="Connecting...", bg="#14151a", fg="#ffcc00")
status.grid(row=2, column=0, columnspan=2, pady=8)

threading.Thread(target=connect, daemon=True).start()
root.mainloop()
