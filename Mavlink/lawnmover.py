#!/usr/bin/env python3
"""
SQUADRONE — Lawnmower Trial Run
File: lawnmower_trial.py

TRIAL SETTINGS (safe for first test):
  Alt   = 5 m   (low, easy to watch)
  Spacing = 8 m  (fewer passes, faster)
  Speed = 1.5 m/s (slow, safe)

Coordinates (delivery zone corners, in sequence from Mission Planner):
  (18.4897992, 73.8514467)
  (18.4898017, 73.8511761)
  (18.4900176, 73.8511614)
  (18.4900281, 73.8514457)

Run:
  python3 lawnmower_trial.py
  python3 lawnmower_trial.py --plan-only    # just print waypoints, no flight
  python3 lawnmower_trial.py --conn udpin:0.0.0.0:14550  # SITL test
"""

import argparse
import math
import signal
import time

from pymavlink import mavutil

# ═══════════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════════

VERTICES = [
    (18.4897992, 73.8514467),
    (18.4898017, 73.8511761),
    (18.4900176, 73.8511614),
    (18.4900281, 73.8514457),
]

ALTITUDE_M      =  5.0    # TRIAL: 5m (change to 10m for real mission)
SPACING_M       =  8.0    # TRIAL: 8m spacing (fewer passes)
SPEED_MPS       =  1.5    # m/s — slow and safe
ACCEPT_RADIUS_M =  2.0    # waypoint "reached" threshold
WAYPOINT_TIMEOUT = 120    # seconds per waypoint before RTL

SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE   = 921600

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

EARTH_R         = 6_378_137.0
LOOP_HZ         = 10
GUIDED_MODE_ID  = 4
POSITION_ONLY_MASK = 0b0000_1101_1111_1000


# ═══════════════════════════════════════════════════════════════════
# PATH PLANNING
# ═══════════════════════════════════════════════════════════════════

def latlon_to_xy(lat, lon, lat0, lon0):
    x = math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * EARTH_R
    return x, y

def xy_to_latlon(x, y, lat0, lon0):
    lat = lat0 + math.degrees(y / EARTH_R)
    lon = lon0 + math.degrees(x / (EARTH_R * math.cos(math.radians(lat0))))
    return lat, lon

def order_corners(points):
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1]-cy, p[0]-cx))

def build_lawnmower(corners, spacing, drone_xy):
    best_path, best_dist = None, float("inf")
    for i in range(4):
        for step in (1, -1):
            start  = corners[i]
            along  = corners[(i + step) % 4]
            across = corners[(i - step) % 4]
            length = math.hypot(along[0]-start[0], along[1]-start[1])
            width  = math.hypot(across[0]-start[0], across[1]-start[1])
            ux = (along[0]-start[0]) / length
            uy = (along[1]-start[1]) / length
            vx = (across[0]-start[0]) / width
            vy = (across[1]-start[1]) / width
            offsets, s = [], 0.0
            while s < width - 1e-6:
                offsets.append(s)
                s += spacing
            if width - offsets[-1] > 0.5:
                offsets.append(width)
            else:
                offsets[-1] = width
            path = []
            for k, s in enumerate(offsets):
                ends = (0.0, length) if k % 2 == 0 else (length, 0.0)
                for t in ends:
                    path.append((start[0] + t*ux + s*vx,
                                 start[1] + t*uy + s*vy))
            d = math.hypot(path[0][0]-drone_xy[0], path[0][1]-drone_xy[1])
            if d < best_dist:
                best_path, best_dist = path, d
    return best_path


# ═══════════════════════════════════════════════════════════════════
# DRONE CLASS
# ═══════════════════════════════════════════════════════════════════

class Drone:
    def __init__(self, conn, baud):
        print(f"Connecting {conn} @ {baud}…")
        self.master = mavutil.mavlink_connection(conn, baud=baud)
        self.master.wait_heartbeat()
        print(f"Heartbeat ✓  system={self.master.target_system}")
        self.lat = self.lon = None
        self.alt = 0.0
        self.armed = False
        self.mode_id = None
        self.last_hb = time.time()
        self.master.mav.request_data_stream_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)

    def update(self):
        while True:
            msg = self.master.recv_match(blocking=False)
            if msg is None:
                return
            t = msg.get_type()
            if t == "GLOBAL_POSITION_INT":
                self.lat = msg.lat / 1e7
                self.lon = msg.lon / 1e7
                self.alt = msg.relative_alt / 1000.0
            elif t == "HEARTBEAT" and msg.get_srcSystem() == self.master.target_system:
                self.armed   = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self.mode_id = msg.custom_mode
                self.last_hb = time.time()
            elif t == "STATUSTEXT":
                print(f"  [FC] {msg.text}")

    def wait_gps(self, timeout=120):
        print("Waiting for GPS 3D fix…")
        t0 = time.time()
        while time.time()-t0 < timeout:
            self.update()
            try:
                gps = self.master.recv_match(type="GPS_RAW_INT", blocking=False)
            except Exception:
                gps = None
            if gps and gps.fix_type >= 3 and self.lat is not None:
                print(f"GPS ✓  {self.lat:.7f}, {self.lon:.7f}  "
                      f"HDOP={gps.eph/100:.2f}")
                return
            elapsed = int(time.time()-t0)
            if elapsed % 5 == 0:
                print(f"  waiting… {elapsed}s", end="\r")
            time.sleep(0.5)
        raise RuntimeError("No GPS fix — check antenna")

    def set_param(self, name, value):
        self.master.mav.param_set_send(
            self.master.target_system, self.master.target_component,
            name.encode(), float(value), mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        time.sleep(0.5)

    def send_cmd(self, cmd, *p):
        pp = list(p) + [0]*(7-len(p))
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            cmd, 0, *pp)

    def set_mode(self, name):
        self.master.set_mode(name)
        target = self.master.mode_mapping()[name]
        t0 = time.time()
        while time.time()-t0 < 5:
            self.update()
            if self.mode_id == target:
                print(f"Mode → {name} ✓")
                return
            time.sleep(0.1)
        raise RuntimeError(f"Mode {name} not confirmed")

    def goto(self, lat, lon, alt):
        self.master.mav.set_position_target_global_int_send(
            0, self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            POSITION_ONLY_MASK,
            int(lat*1e7), int(lon*1e7), alt,
            0, 0, 0, 0, 0, 0, 0, 0)

    def check_link(self):
        if time.time() - self.last_hb > 5:
            raise RuntimeError("FC heartbeat lost")


class PilotTookOver(Exception):
    pass


# ═══════════════════════════════════════════════════════════════════
# FLIGHT
# ═══════════════════════════════════════════════════════════════════

def arm_and_takeoff(drone, alt):
    drone.set_mode("GUIDED")
    print("Arming…")
    drone.master.arducopter_arm()
    t0 = time.time()
    while not drone.armed:
        if time.time()-t0 > 15:
            raise RuntimeError("Arm failed — check pre-arm warnings above")
        drone.update()
        time.sleep(0.2)
    print("Armed ✓ — taking off…")
    drone.send_cmd(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,0,0,0,0,0, alt)
    t0 = time.time()
    while drone.alt < 0.9*alt:
        if time.time()-t0 > 35:
            raise RuntimeError("Takeoff timed out")
        drone.update()
        print(f"  alt={drone.alt:.1f}m…", end="\r")
        time.sleep(0.3)
    print(f"\nTakeoff done — {drone.alt:.1f}m ✓")
    drone.send_cmd(mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, 1, SPEED_MPS, -1, 0)
    print(f"Speed → {SPEED_MPS} m/s")


def fly_lawnmower(drone, waypoints, origin):
    lat0, lon0 = origin
    total = len(waypoints)
    for idx, (wx, wy) in enumerate(waypoints, 1):
        wlat, wlon = xy_to_latlon(wx, wy, lat0, lon0)
        print(f"\nWP {idx}/{total}  →  {wlat:.7f}, {wlon:.7f}")
        t0 = time.time()
        while True:
            drone.update()
            drone.check_link()
            if drone.mode_id != GUIDED_MODE_ID:
                raise PilotTookOver()
            if time.time()-t0 > WAYPOINT_TIMEOUT:
                raise RuntimeError(f"WP {idx} timed out")
            drone.goto(wlat, wlon, ALTITUDE_M)
            if drone.lat is None:
                time.sleep(1/LOOP_HZ)
                continue
            dx, dy = latlon_to_xy(drone.lat, drone.lon, lat0, lon0)
            dist = math.hypot(wx-dx, wy-dy)
            dalt = abs(drone.alt - ALTITUDE_M)
            print(f"  dist={dist:.1f}m  alt={drone.alt:.1f}m", end="\r")
            if dist < ACCEPT_RADIUS_M and dalt < 1.0:
                print(f"\n  ✓ WP {idx} reached")
                break
            time.sleep(1/LOOP_HZ)


def return_home(drone):
    drone.update()
    if drone.alt < 1.0:
        print("Very low → LAND")
        drone.set_mode("LAND")
    else:
        print("RTL…")
        drone.set_mode("RTL")
    while drone.armed:
        drone.update()
        time.sleep(0.5)
    print("Landed + disarmed ✓")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Squadrone lawnmower trial")
    ap.add_argument("--conn",      default=SERIAL_PORT)
    ap.add_argument("--baud",      type=int, default=BAUD_RATE)
    ap.add_argument("--plan-only", action="store_true",
                    help="Print waypoints and exit (no flight)")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM,
                  lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    print("=" * 55)
    print("SQUADRONE — Lawnmower Trial")
    print(f"Alt={ALTITUDE_M}m  Spacing={SPACING_M}m  Speed={SPEED_MPS}m/s")
    print("=" * 55)

    # ── Plan (no drone needed) ─────────────────────────────────────
    origin  = VERTICES[0]
    corners = order_corners(
        [latlon_to_xy(la, lo, *origin) for la, lo in VERTICES])
    dummy   = latlon_to_xy(*VERTICES[0], *origin)
    path_xy = build_lawnmower(corners, SPACING_M, dummy)

    print(f"\nPlanned {len(path_xy)} waypoints:")
    for i, (x, y) in enumerate(path_xy, 1):
        la, lo = xy_to_latlon(x, y, *origin)
        print(f"  WP{i:02d}: {la:.7f}, {lo:.7f}")
    print()

    if args.plan_only:
        print("--plan-only: done. No flight.")
        return

    # ── Connect ────────────────────────────────────────────────────
    drone = Drone(args.conn, args.baud)
    drone.wait_gps()

    # Re-plan from actual drone position
    drone_xy = latlon_to_xy(drone.lat, drone.lon, *origin)
    path_xy  = build_lawnmower(corners, SPACING_M, drone_xy)
    print(f"Re-planned from drone pos — {len(path_xy)} WPs\n")

    drone.set_param("RTL_ALT", ALTITUDE_M * 100)

    pilot_took_over = False
    try:
        arm_and_takeoff(drone, ALTITUDE_M)
        fly_lawnmower(drone, path_xy, origin)
        print("\n✓ Lawnmower trial complete!")

    except KeyboardInterrupt:
        print("\nCtrl+C — aborting")

    except PilotTookOver:
        pilot_took_over = True
        print("Pilot/failsafe took over — script exits")

    except Exception as e:
        print(f"\nERROR: {e}")

    if not pilot_took_over and drone.armed:
        try:
            return_home(drone)
        except KeyboardInterrupt:
            print("RTL/LAND already commanded.")


if __name__ == "__main__":
    main()