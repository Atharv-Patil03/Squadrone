#!/usr/bin/env python3
"""
SQUADRONE — Lawnmower Trial Run
File: lawnmower_trial.py

TRIAL SETTINGS (safe for first test):
  Alt   = 5 m   (low, easy to watch)
  Spacing = 8 m  (fewer passes, faster)
  Speed = 1.5 m/s (slow, safe)

Field corners (recorded with the drone on 17-10-2026):
  A  (18.4898160, 73.8516559)   SE
  B  (18.4898089, 73.8513138)   SW
  C  (18.4899799, 73.8512960)   NW
  D  (18.4900219, 73.8516538)   NE

Path order:
  Takeoff -> A -> B -> zig-zag towards C/D side -> ... -> D -> RTL home
  First pass is always A -> B. The path always ends at D:
  if the last pass ends at C, the drone flies C -> D before going home.

Run:
  python3 lawnmower_trial.py
  python3 lawnmower_trial.py --plan-only    # just print waypoints, no flight
  python3 lawnmower_trial.py --spacing 5 --plan-only   # change pass spacing
  python3 lawnmower_trial.py --spacing 5 --alt 8 --speed 2 --inset 2
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

# Keep this exact order: A, B, C, D
POINT_A = (18.4898160, 73.8516559)   # start of lawnmower (SE)
POINT_B = (18.4898089, 73.8513138)   # end of first pass (SW)
POINT_C = (18.4899799, 73.8512960)   # NW
POINT_D = (18.4900219, 73.8516538)   # finish point (NE)
VERTICES = [POINT_A, POINT_B, POINT_C, POINT_D]

# Pull the whole path this many metres inside the corners.
# 0.0 = fly exactly through A, B, ..., D (as recorded).
# There are obstacles just outside the field and GPS error is 1-2 m,
# so 2.0 is recommended once the basic pattern has been checked.
INSET_M = 0.0

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

def inset_quad(quad, d):
    """Move every edge of a convex quad inward by d metres.
    Vertex order (A, B, C, D) is preserved."""
    if d <= 0:
        return list(quad)
    n = len(quad)
    area2 = sum(quad[i][0]*quad[(i+1) % n][1] - quad[(i+1) % n][0]*quad[i][1]
                for i in range(n))
    s = 1 if area2 > 0 else -1
    lines = []
    for i in range(n):
        a, b = quad[i], quad[(i+1) % n]
        dx, dy = b[0]-a[0], b[1]-a[1]
        L = math.hypot(dx, dy)
        nx, ny = -dy/L*s, dx/L*s
        lines.append(((a[0]+nx*d, a[1]+ny*d), (dx, dy)))
    out = []
    for i in range(n):
        (p1, d1), (p2, d2) = lines[i-1], lines[i]
        den = d1[0]*d2[1] - d1[1]*d2[0]
        t = ((p2[0]-p1[0])*d2[1] - (p2[1]-p1[1])*d2[0]) / den
        out.append((p1[0]+t*d1[0], p1[1]+t*d1[1]))
    return out


def lerp(p, q, f):
    return (p[0] + (q[0]-p[0])*f, p[1] + (q[1]-p[1])*f)


def build_lawnmower(A, B, C, D, spacing):
    """
    Passes run parallel to A-B and step from the A-B side to the D-C side.
      pass 0 : A -> B
      pass 1 : (B-side) -> (A-side)
      ...
      last   : the D-C edge
    Returns list of (x, y, label). Always ends at D.
    """
    # Passes are spread evenly, so the real gap is <= the requested spacing
    # (e.g. 23 m field, 8 m asked -> 3 gaps of ~7.6 m).
    width = max(math.dist(A, D), math.dist(B, C))
    n_steps = max(1, math.ceil(width / spacing - 1e-6))
    print(f"Field width {width:.1f} m -> {n_steps + 1} passes, "
          f"actual gap {width / n_steps:.1f} m (asked {spacing} m)")
    path = []
    for k in range(n_steps + 1):
        f = k / n_steps
        a_side = lerp(A, D, f)          # point on edge A-D
        b_side = lerp(B, C, f)          # point on edge B-C
        if k % 2 == 0:
            seg = [(a_side, "A-side"), (b_side, "B-side")]
        else:
            seg = [(b_side, "B-side"), (a_side, "A-side")]
        for p, lab in seg:
            path.append((p[0], p[1], lab))

    # Labels for the real corners
    path[0] = (A[0], A[1], "A")
    path[1] = (B[0], B[1], "B")
    last_x, last_y, _ = path[-1]
    if math.dist((last_x, last_y), D) < 0.01:
        path[-1] = (D[0], D[1], "D")
        path[-2] = (C[0], C[1], "C")
    else:
        # finished at C -> fly along to D
        path[-1] = (C[0], C[1], "C")
        path[-2] = (D[0], D[1], "D")
        path.append((D[0], D[1], "D"))
    return path


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
        self.gps_fix = 0
        self.gps_sats = 0
        self.gps_hdop = 99.0
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
            elif t == "GPS_RAW_INT":
                self.gps_fix  = msg.fix_type
                self.gps_sats = msg.satellites_visible
                self.gps_hdop = msg.eph / 100.0
            elif t == "STATUSTEXT":
                print(f"  [FC] {msg.text}")

    def wait_gps(self, timeout=300):
        print("Waiting for GPS 3D fix… (press Ctrl+C to skip if FC already has position)")
        t0 = time.time()
        last_print = 0
        while time.time()-t0 < timeout:
            self.update()   # GPS_RAW_INT is now read inside update()
            elapsed = int(time.time()-t0)
            if elapsed - last_print >= 3:
                print(f"  fix={self.gps_fix}  sats={self.gps_sats}  "
                      f"hdop={self.gps_hdop:.2f}  lat={self.lat}  {elapsed}s elapsed")
                last_print = elapsed
            if self.gps_fix >= 3 and self.lat is not None:
                print(f"GPS ✓  {self.lat:.7f}, {self.lon:.7f}  "
                      f"sats={self.gps_sats}  HDOP={self.gps_hdop:.2f}")
                return
            time.sleep(0.5)
        raise RuntimeError("No GPS fix after 5 min — check antenna")

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
    for idx, (wx, wy, label) in enumerate(waypoints, 1):
        wlat, wlon = xy_to_latlon(wx, wy, lat0, lon0)
        print(f"\nWP {idx}/{total} [{label}]  →  {wlat:.7f}, {wlon:.7f}")
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
    global SPACING_M, ALTITUDE_M, SPEED_MPS, INSET_M
    ap = argparse.ArgumentParser(description="Squadrone lawnmower trial")
    ap.add_argument("--conn",      default=SERIAL_PORT)
    ap.add_argument("--baud",      type=int, default=BAUD_RATE)
    ap.add_argument("--plan-only", action="store_true",
                    help="Print waypoints and exit (no flight)")
    ap.add_argument("--spacing", type=float, default=SPACING_M,
                    help=f"distance between passes in m (default {SPACING_M})")
    ap.add_argument("--alt", type=float, default=ALTITUDE_M,
                    help=f"flight altitude in m (default {ALTITUDE_M})")
    ap.add_argument("--speed", type=float, default=SPEED_MPS,
                    help=f"speed in m/s (default {SPEED_MPS})")
    ap.add_argument("--inset", type=float, default=INSET_M,
                    help=f"keep path this many m inside the corners (default {INSET_M})")
    args = ap.parse_args()

    if not 1.0 <= args.spacing <= 30.0:
        ap.error("--spacing must be between 1 and 30 m")
    if not 2.0 <= args.alt <= 30.0:
        ap.error("--alt must be between 2 and 30 m")
    if not 0.5 <= args.speed <= 5.0:
        ap.error("--speed must be between 0.5 and 5 m/s")
    if not 0.0 <= args.inset <= 5.0:
        ap.error("--inset must be between 0 and 5 m")

    # Apply command-line values (used by the flight functions below)
    SPACING_M, ALTITUDE_M, SPEED_MPS, INSET_M = (
        args.spacing, args.alt, args.speed, args.inset)

    signal.signal(signal.SIGTERM,
                  lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    print("=" * 55)
    print("SQUADRONE — Lawnmower Trial")
    print(f"Alt={ALTITUDE_M}m  Spacing={SPACING_M}m  Speed={SPEED_MPS}m/s")
    print("=" * 55)

    # ── Plan (no drone needed) ─────────────────────────────────────
    origin  = POINT_A
    quad    = [latlon_to_xy(la, lo, *origin) for la, lo in VERTICES]  # A,B,C,D
    A, B, C, D = inset_quad(quad, INSET_M)
    path_xy = build_lawnmower(A, B, C, D, SPACING_M)

    length = sum(math.dist(path_xy[i][:2], path_xy[i+1][:2])
                 for i in range(len(path_xy)-1))
    print(f"\nPlanned {len(path_xy)} waypoints  (inset {INSET_M} m, "
          f"path {length:.0f} m, ~{length/SPEED_MPS:.0f} s):")
    for i, (x, y, label) in enumerate(path_xy, 1):
        la, lo = xy_to_latlon(x, y, *origin)
        print(f"  WP{i:02d} [{label:6s}]: {la:.7f}, {lo:.7f}")
    print("  then RTL -> home\n")

    if args.plan_only:
        print("--plan-only: done. No flight.")
        return

    # ── Connect ────────────────────────────────────────────────────
    drone = Drone(args.conn, args.baud)
    try:
        drone.wait_gps()
    except KeyboardInterrupt:
        print("\nGPS wait skipped by user — using FC position")
        if drone.lat is None:
            # Force-read one position message
            for _ in range(20):
                drone.update()
                time.sleep(0.1)
                if drone.lat is not None:
                    break
        if drone.lat is None:
            raise RuntimeError("No position from FC — cannot plan path")
        print(f"Position: {drone.lat:.7f}, {drone.lon:.7f}")

    # Path is fixed (always A -> B ... -> D), no re-planning.
    dx, dy = latlon_to_xy(drone.lat, drone.lon, *origin)
    print(f"Drone is {math.hypot(dx - A[0], dy - A[1]):.1f} m from start point A")

    drone.set_param("RTL_ALT", ALTITUDE_M * 100)

    input("\n>>> Press ENTER to arm and takeoff (make sure area is clear!) <<<\n")

    pilot_took_over = False
    try:
        arm_and_takeoff(drone, ALTITUDE_M)
        fly_lawnmower(drone, path_xy, origin)
        print("\n✓ Lawnmower trial complete — at D, returning home")

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