#!/usr/bin/env python3
"""
SQUADRONE — FULL MISSION (one flight)                       File: mission.py
Team: Squadrone AT2026RC60, PVGCOET Pune, SAEINDIA AeroTHON 2026

Glues the two field-tested scripts together — they must be in the SAME folder
with exactly these names:   survey.py   corridor.py   mission.py

DEFAULT FLOW (30-10, GPS corridor points — see CORRIDOR_POINTS below)
  takeoff -> start tag at 5 m -> fly to GREEN_IN at 5 m facing GREEN_OUT -> banner
  camera LOOKS (shown + logged, does not steer) -> back off, descend, LiDAR corridor
  -> 10 m -> lawnmower -> AUTO drop when SURE (same AprilTag ID) -> rewind line
  -> BLUE_IN (attempt: stops + climbs out at an obstacle) or --blue skip -> RTL.
  Any corridor trouble (LiDAR dead/unsure, wall too close, blocked) = climb
  straight up and CONTINUE with the next phase, no hovering, no RTL.
  Record the points with the drone standing on each spot:
     python3 survey.py --record GREEN_IN   (GREEN_OUT, BLUE_IN, BLUE_OUT)
  python3 mission.py                         # this flow
  python3 mission.py --blue skip             # RTL after the payload
  python3 mission.py --ask                   # operator types yes/no before the drop
  python3 mission.py --vision-search         # OLD flow below (banner search, no GPS points)

OLD FLOW (--vision-search)
  1. PREFLIGHT  LiDARs, camera, GPS, servos (latch closed, winch stop), RTL_ALT.
  2. TAKEOFF    to 5 m over the take-off spot (nose roughly towards the corridor).
  3. START QR   read + store the start QR at 5 m          (survey.py, high-res cam)
  4. CORRIDOR   find the green banner (forward / left / right pattern), centre,
                heading from the banner, back off, descend, fly through with the
                LiDARs, end = walls gone                   (corridor.py, banner cam)
  5. CLIMB      straight up to 10 m past the corridor exit.
  6. SURVEY     lawnmower A -> ... -> D at 10 m, match every QR with the start QR,
                confirmed match -> 5 m -> operator 'yes' -> payload -> RTL
                                                           (survey.py)
  7. RTL        (also on any failure; the pilot's mode switch always wins)
  The blue/exit corridor is NOT flown yet: the drone returns home at RTL height.

Every phase is logged with its duration; the summary at the end shows
which phase passed/failed. Logs: ~/squadrone/mission.log (+ survey.log).

RUN (inside tmux!)
  python3 mission.py                         # everything
  python3 mission.py --skip-corridor         # start QR + survey + payload only
  python3 mission.py --skip-survey           # start QR + corridor only (then RTL)
  python3 mission.py --cam imx500            # AI camera as the down camera
  options: --tries --step --side (banner search)  --corridor-alt --corridor-len
           --corridor-width --corridor-speed  --no-payload  --image-only
"""
import os
import sys
import math
import time
import logging
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import survey as S          # noqa: E402  (sets up logging first)
import corridor as C        # noqa: E402

log = logging.getLogger("mission")
logging.getLogger().addHandler(logging.FileHandler(os.path.join(S.DEBUG_DIR, "mission.log")))
for h in logging.getLogger().handlers:
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))


class Abort(Exception):
    pass


# ═════════════════════════════════════════════════════════════════════════════
# CORRIDOR POINTS (GPS)  — the drone flies to these instead of searching.
#   GREEN_IN  = centre of the green GROUND banner (corridor mouth)
#   GREEN_OUT = centre of the far (exit) end of the green corridor, between the walls
#   BLUE_IN / BLUE_OUT = same for the blue corridor (fly IN -> OUT)
# Either type them here (lat, lon) or record them with the drone standing on the
# spot:  python3 survey.py --record GREEN_IN   (saved in ~/squadrone/field_corners.json,
# which wins over the values below).
# ═════════════════════════════════════════════════════════════════════════════
# Venue waypoint map 30-10 (A..D field, corridors outside the A-D edge):
#   green walls N-M / P-K, entry L (ground banner), exit O;
#   blue  walls G-H / E-I, entry F (ground banner), exit J/J2.
VENUE = {
    "A": (13.1705094, 77.5330114), "B": (13.1705518, 77.5328726),
    "C": (13.1707492, 77.5329336), "D": (13.1707059, 77.5330857),
    "E": (13.1706642, 77.5330787), "F": (13.1706544, 77.5330581),
    "G": (13.1706389, 77.5330605), "H": (13.1706007, 77.5331004),
    "I": (13.1706414, 77.5331319), "J": (13.1706311, 77.5331340),
    "J2": (13.1706273, 77.5331254), "K": (13.1705882, 77.5331196),
    "L": (13.1705764, 77.5331010), "M": (13.1705573, 77.5330965),
    "N": (13.1705864, 77.5330484), "O": (13.1705964, 77.5330516),
    "P": (13.1706151, 77.5330615), "QR": (13.1705856, 77.5331492),
    "Takeoff": (13.1705787, 77.5331730),
}
# Each corridor centre = average of the measured centre point and the midpoint
# of the two wall ends (GPS noise ~1 m per point; L is 0.9 m and O 0.6 m from
# their wall midpoints). Green: 6.2 m long, heading 296°. Blue: 7.0 m, 117°.
CORRIDOR_POINTS = {
    "GREEN_IN":  (13.1705746, 77.5331045),   # ~ L
    "GREEN_OUT": (13.1705986, 77.5330533),   # ~ O
    "BLUE_IN":   (13.1706530, 77.5330639),   # ~ F
    "BLUE_OUT":  (13.1706242, 77.5331208),   # ~ J2 / mid(H, I)
    "START_QR":  (13.1705856, 77.5331492),   # QR (start tag), read at 5 m
}
VISION_LOOK_S = 4.0         # at the entry point: look for the banner this long (logged/shown)
POINT_TOL_M   = 0.5         # "arrived" at a GPS point
OVER_ALT_M    = 6.0         # fly OVER a corridor at this (walls 2.44 m, poles ~4 m)
QR_RETRY_M    = 1.5         # start tag not seen at QR: move this far forward, read again


def load_points():
    pts = dict(CORRIDOR_POINTS)
    if os.path.isfile(S.CORNERS_FILE):
        import json
        with open(S.CORNERS_FILE) as f:
            for k, v in json.load(f).items():
                if k in pts and math.dist((0, 0), S.latlon_to_xy(*v, *VENUE["A"])) < 500:
                    pts[k] = tuple(v)          # drone-recorded point of THIS venue wins
    return pts


# ═════════════════════════════════════════════════════════════════════════════
# One MAVLink link (survey.Drone) seen by corridor.Mission through this adapter
# ═════════════════════════════════════════════════════════════════════════════
class CorridorDrone:
    """corridor.py wants local-NED style calls; survey.Drone works in the field
    frame (x = east, y = north, metres from POINT_A, relative altitude)."""

    def __init__(self, sd):
        self.sd = sd
        self.mav = sd.mav
        self.ground_z = 0.0

    def pos_ned(self):
        p = self.sd.pose()
        return (p[2], p[1], -p[3]) if p else None

    def alt(self):
        return self.sd.alt()

    def speed(self):
        m = self.sd.get("GLOBAL_POSITION_INT")
        return math.hypot(m.vx, m.vy) / 100.0 if m else 0.0

    def yaw(self):
        m = self.sd.get("ATTITUDE")
        return m.yaw if m else None

    def tilt_deg(self):
        m = self.sd.get("ATTITUDE")
        return math.degrees(max(abs(m.roll), abs(m.pitch))) if m else 0.0

    def armed(self):
        return self.sd.armed()

    def mode(self):
        return self.sd.mode()

    def set_mode(self, name, timeout=5):
        return self.sd.set_mode(name, timeout)

    def set_speed(self, ms):
        self.sd.cmd(S.mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, 1, ms, -1)

    def goto_ned(self, n, e, alt, yaw):
        self.sd.goto_xy(e, n, alt, yaw)


# ═════════════════════════════════════════════════════════════════════════════
class MissionRun:
    def __init__(self, a):
        self.a = a
        self.results = []          # (phase, ok, seconds, note)
        self.t_phase = None
        self.drone = self.scam = self.lidar = self.down = self.pv = None
        self.sv = None
        self.lidar_bad = False
        self.pts = load_points()

    # ── bookkeeping ──────────────────────────────────────────────────────
    def begin(self, name):
        self.cur = name
        self.t_phase = time.time()
        log.info("\n" + "█" * 64 + f"\n█  PHASE: {name}\n" + "█" * 64)

    def end(self, ok, note=""):
        dt = time.time() - self.t_phase
        self.results.append((self.cur, ok, dt, note))
        log.info(f"█  {self.cur}: {'✓ OK' if ok else '✗ FAILED'} in {dt:.0f} s  {note}")
        if not ok:
            raise Abort(f"{self.cur} failed: {note}")

    def summary(self):
        log.info("\n══ MISSION SUMMARY ══")
        for name, ok, dt, note in self.results:
            log.info(f"  {'✓' if ok else '✗'} {name:12s} {dt:6.0f} s  {note}")

    # ── cameras: one physical camera, two users -> open one at a time ───
    def survey_cam_on(self):
        if self.down is not None:
            self.down.stop()
            self.down = None
            time.sleep(1.0)
        if self.scam is None:
            self.scam = S.SurveyCam()
            self.scam.open()
            self.scam.start()
            time.sleep(1.0)
        return self.scam

    def banner_cam_on(self):
        if self.scam is not None:
            self.scam.stop()
            self.scam = None
            time.sleep(1.0)
        if self.down is None:
            self.down, _ = C.open_cameras(use_fwd=False)
            time.sleep(1.0)
        return self.down

    # ── phases ───────────────────────────────────────────────────────────
    def preflight(self):
        self.begin("PREFLIGHT")
        S.load_corners()
        self.origin = S.POINT_A
        self.field, self.path = S.plan(self.origin)
        need_lidar = (not self.a.skip_corridor) or (self.a.gps_corridor and self.a.blue != "skip")
        if need_lidar:
            self.lidar = C.Lidars()
            self.lidar.start()
            time.sleep(0.6)
            vals = {k: self.lidar.get(k) for k in ("front", "left", "right")}
            alive = {k: self.lidar.alive(k) for k in vals}
            log.info("  LiDARs: " + "  ".join(
                f"{k}={'%.2fm' % v if v is not None else '--'}{'' if alive[k] else ' (NOT ANSWERING)'}"
                for k, v in vals.items()))
            if not all(alive.values()):
                if self.a.gps_corridor:
                    log.warning("  ⚠ a LiDAR is not answering — the GREEN corridor will be SKIPPED, "
                                "mission goes straight to the lawnmower")
                    self.lidar_bad = True
                else:
                    self.end(False, "a LiDAR is not answering")
        self.drone = S.Drone(self.a.conn, self.a.baud, self.origin)
        self.drone.wait_gps()
        self.start_yaw = CorridorDrone(self.drone).yaw() or 0.0
        xy = self.drone.xy()
        log.info(f"  Heading now {math.degrees(self.start_yaw) % 360:.0f}° (nose must point "
                 f"roughly at the GREEN corridor) | {math.dist(xy, self.path[0][:2]):.0f} m from A, "
                 f"{S.dist_outside(*xy, self.field):.0f} m outside the field")
        self.drone.set_param("RTL_ALT", S.SURVEY_ALT_M * 100)
        if S.PAYLOAD:
            ok1 = self.drone.servo(S.LATCH_SERVO, S.LATCH_CLOSED_PWM, "latch CLOSED")
            ok2 = self.drone.servo(S.WINCH_SERVO, S.WINCH_STOP_PWM, "winch STOP")
            if not (ok1 and ok2):
                log.warning("  Servo check FAILED — the drop may not work (SERVO5/6_FUNCTION = 0?)")
        self.survey_cam_on()
        log.info(f"  Survey camera: {self.scam.focus_info}")
        self.end(True)

    def takeoff(self):
        self.begin("TAKEOFF")
        input("\n>>> Payload latched? Nose towards the green corridor? Area clear? "
              "ENTER to arm + take off <<<\n")
        self.drone.arm_and_takeoff(S.REF_ALT_M)
        self.home = self.drone.xy()
        self.end(True, f"at {self.drone.alt():.1f} m")

    def start_qr(self):
        self.begin("START QR")
        self.sv = S.Survey(self.drone, self.survey_cam_on(), self.field, self.path, self.origin)
        self.sv.yaw = self.start_yaw                      # don't spin while reading
        qr_ll = self.pts.get("START_QR") if self.a.gps_corridor else None
        spot = S.latlon_to_xy(*qr_ll, *self.origin) if qr_ll else self.home
        if qr_ll:
            log.info(f"  start tag at point QR, {math.dist(self.home, spot):.1f} m from take-off")
            ok = self.sv.capture_reference(spot, fallback=False, timeout=12)
            if not ok:
                # not seen: 1.5 m further in the take-off -> QR direction, look again
                d = math.dist(self.home, spot) or 1.0
                spot2 = (spot[0] + (spot[0] - self.home[0]) / d * QR_RETRY_M,
                         spot[1] + (spot[1] - self.home[1]) / d * QR_RETRY_M)
                log.info(f"  → moving {QR_RETRY_M} m forward and reading again")
                ok = self.sv.capture_reference(spot2)        # this one may use the saved tag
        else:
            ok = self.sv.capture_reference(spot)
        self.end(ok, self.sv.ref.describe() if ok else "start QR not seen")

    def corridor(self):
        self.begin("CORRIDOR")
        if self.sv is not None:
            self.sv.phase("corridor + climb (mission)")   # keeps survey's timing table honest
        down = self.banner_cam_on()
        self.pv = C.Preview()
        cd = CorridorDrone(self.drone)
        m = C.Mission(cd, down, None, self.pv)
        m.lidar = self.lidar
        m.yaw = self.start_yaw
        cd.set_speed(C.SEARCH_SPEED_MS)
        for attempt in range(2):
            if not m.find_banner():
                self.end(False, "green banner not found")
            if not m.align():
                m.down.reset()
                continue
            new = m.heading_from_banner()
            if new == "refuse":
                self.end(False, "corridor more than 60° off the nose")
            if new is not None:
                log.info(f"  turning {(math.degrees(new - m.yaw) + 180) % 360 - 180:+.1f}° "
                         "to face the corridor")
                m.yaw = new
                m.calibrated = [False, False]
                if not m.servo_centre(C.SEARCH_ALT_M, C.CENTRE_TOL_5M, "RE-CENTRE"):
                    continue
            if m.corridor():
                self.corr_yaw = m.yaw
                self.end(True, f"walls seen over {len(m.corr_log)} readings")
                return
            raise Abort("corridor aborted (drone already climbing / RTL)")
        self.end(False, "could not centre on the banner")

    # ── GPS-guided corridor (green or blue) ─────────────────────────────
    def fly_to(self, x, y, alt, yaw, label, timeout=90):
        """Field-frame point (x east, y north); climbs/descends on the way."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.drone.check()
            p = self.drone.xy()
            if p and math.dist(p, (x, y)) < POINT_TOL_M and abs((self.drone.alt() or 0) - alt) < 0.4:
                return True
            self.drone.goto_xy(x, y, alt, yaw)
            if int(time.time() * 2) % 4 == 0:
                log.info(f"  [{label}] {math.dist(p, (x, y)) if p else -1:.1f} m to go, "
                         f"alt {self.drone.alt() or 0:.1f} m")
            time.sleep(0.25)
        log.warning(f"  [{label}] not reached in {timeout} s")
        return False

    def gps_corridor(self, color):
        """Fly to <COLOR>_IN at 5 m facing <COLOR>_OUT, LOOK for the banner (logged
        + shown, not used to move), then back off / descend / LiDAR corridor.
        Returns True through, False skipped/aborted (the drone is then at
        ABORT_ALT_M or higher, mission continues)."""
        a_ll, b_ll = self.pts.get(f"{color.upper()}_IN"), self.pts.get(f"{color.upper()}_OUT")
        if not a_ll or not b_ll:
            log.warning(f"  {color.upper()}_IN / _OUT not set — {color} corridor SKIPPED")
            return False
        ax, ay = S.latlon_to_xy(*a_ll, *self.origin)
        bx, by = S.latlon_to_xy(*b_ll, *self.origin)
        yaw = S.bearing((ax, ay), (bx, by))
        length = math.dist((ax, ay), (bx, by))
        self.corr_yaw = yaw
        C.set_banner_color(color, loose=self.a.no_border)
        C.OBST_MODE = "gap" if (color == "blue" and self.a.blue == "gap") else None
        if C.OBST_MODE:
            log.warning("  ⚠ BLUE: GAP obstacle navigation ON (experimental — sim: some runs "
                        "came within a few cm of a board)")
        if self.a.corridor_len is None:
            C.CORRIDOR_LEN_M = length
        log.info(f"  {color.upper()} corridor from GPS: entry ({ax:.1f}, {ay:.1f}) -> exit "
                 f"({bx:.1f}, {by:.1f}), heading {math.degrees(yaw) % 360:.0f}°, length "
                 f"{length:.1f} m (using {C.CORRIDOR_LEN_M:.1f} m)")
        if length < 2 or length > 20:
            log.error(f"  ✗ {color} entry/exit {length:.1f} m apart — points look wrong, SKIPPED")
            return False
        # 1) to the entry point at search height, nose along the corridor
        here = self.drone.xy()
        self.drone.cmd(S.mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, 1, C.SEARCH_SPEED_MS, -1)
        if (self.drone.alt() or 0) < C.SEARCH_ALT_M - 0.5:
            self.fly_to(here[0], here[1], C.SEARCH_ALT_M, yaw, "climb", 30)
        if not self.fly_to(ax, ay, C.SEARCH_ALT_M, yaw, f"to {color.upper()}_IN", 120):
            return False
        # 2) banner camera ON: look + log (vision does NOT move the drone)
        down = self.banner_cam_on()
        self.pv = self.pv or C.Preview()
        cd = CorridorDrone(self.drone)
        m = C.Mission(cd, down, None, self.pv)
        m.lidar = self.lidar
        m.yaw = yaw
        self.vision_check(m, color, yaw)
        # 3) corridor: back off, descend, LiDARs (unchanged code)
        if self.lidar is None or self.lidar_bad:
            log.warning("  no healthy LiDARs — corridor not entered")
            return False
        self.last_m = m
        if color == "blue" and self.a.blue == "hug":
            # wall-hug: left lane, stick to a wall, slide across at each board
            ok = m.wall_hug((ay, ax), (by, bx))
            self.corr_log_n = len(getattr(m, "hug_log", []))
            if not ok:
                log.warning(f"  blue wall-hug: {getattr(m, 'abort_reason', 'aborted')}")
            return ok
        ok = m.corridor()
        self.corr_log_n = len(m.corr_log)
        if not ok:
            log.warning(f"  {color} corridor: {getattr(m, 'abort_reason', 'aborted')}")
        return ok

    def vision_check(self, m, color, yaw):
        t0, looks = time.time(), []
        while time.time() - t0 < VISION_LOOK_S:
            p = self.drone.xy()
            self.drone.goto_xy(p[0], p[1], C.SEARCH_ALT_M, yaw)
            det = m.down_conf()
            if det is not None and det.get("t", 0) and (not looks or det["t"] != looks[-1][2]):
                f, r = C.down_pixel_to_body(det["center"][0], det["center"][1],
                                            self.drone.alt() or C.SEARCH_ALT_M)
                ang = None
                if det.get("axis") and det.get("elong", 0) >= 3:
                    ax_, ay_ = det["axis"]
                    cx, cy = det["center"]
                    alt = self.drone.alt() or C.SEARCH_ALT_M
                    f1, r1 = C.down_pixel_to_body(cx - 60 * ax_, cy - 60 * ay_, alt)
                    f2, r2 = C.down_pixel_to_body(cx + 60 * ax_, cy + 60 * ay_, alt)
                    ang = (math.degrees(math.atan2(r2 - r1, f2 - f1) + math.pi / 2) + 90) % 180 - 90
                looks.append((f, r, det["t"], ang))
            self.pv.update(m.down, None, f"VISION CHECK {color}: {len(looks)} looks")
            time.sleep(0.1)
        if looks:
            f = float(np.median([l[0] for l in looks]))
            r = float(np.median([l[1] for l in looks]))
            angs = [l[3] for l in looks if l[3] is not None]
            log.info(f"  ★ VISION: {color} banner SEEN ({len(looks)} looks) — it is {f:+.2f} m fwd, "
                     f"{r:+.2f} m right of the GPS entry point"
                     + (f"; corridor {np.median(angs):+.0f}° from the GPS heading" if angs else "")
                     + "  (shown/logged only — GPS + LiDARs steer)")
        else:
            log.info(f"  VISION: {color} banner NOT seen in {VISION_LOOK_S:.0f} s at the GPS entry "
                     "point (GPS + LiDARs steer anyway)")

    def green_gps(self):
        self.begin("GREEN CORRIDOR (GPS)")
        if self.sv is not None:
            self.sv.phase("green corridor (mission)")
        if self.lidar_bad:
            # no LiDARs: can't fly THROUGH safely; go L -> O by GPS ABOVE the walls
            ok = False
            if self.pts.get("GREEN_IN") and self.pts.get("GREEN_OUT"):
                ax, ay = S.latlon_to_xy(*self.pts["GREEN_IN"], *self.origin)
                ox, oy = S.latlon_to_xy(*self.pts["GREEN_OUT"], *self.origin)
                yaw = S.bearing((ax, ay), (ox, oy))
                self.corr_yaw = yaw
                log.warning(f"  LiDAR not answering → GPS only: L → O at {C.SEARCH_ALT_M:.0f} m "
                            "(above walls + arch)")
                self.fly_to(ax, ay, C.SEARCH_ALT_M, yaw, "to L", 120)
                self.fly_to(ox, oy, C.SEARCH_ALT_M, yaw, "L → O by GPS", 90)
            self.results.append((self.cur, False, time.time() - self.t_phase,
                                 "LiDAR not answering — crossed L→O by GPS at 5 m"))
            return
        ok = self.gps_corridor("green")
        if not ok and self.pts.get("GREEN_OUT"):
            # couldn't get through with the LiDARs -> go to O at the height it is at
            # now, then 10 m. Exception: something PHYSICAL blocked it (wall too
            # close / board ahead) -> it already climbed out; cross ABOVE the walls.
            ox, oy = S.latlon_to_xy(*self.pts["GREEN_OUT"], *self.origin)
            m = getattr(self, "last_m", None)
            sensor = bool(m is not None and getattr(m, "abort_sensor", False))
            alt_now = self.drone.alt() or C.CORRIDOR_ALT_M
            alt = alt_now if (sensor or alt_now >= OVER_ALT_M - 1.5) else OVER_ALT_M
            log.warning(f"  → fallback: flying to O (green exit) at {alt:.1f} m "
                        + ("(LiDAR problem: same height, by GPS)" if sensor else
                           "(blocked: above the walls)" if alt != alt_now else "(same height)"))
            p = self.drone.xy()
            if abs(alt - alt_now) > 0.3:
                self.fly_to(p[0], p[1], alt, self.corr_yaw, "height", 30)
            self.drone.cmd(S.mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, 1,
                           C.CORRIDOR_SPEED if sensor else C.ALIGN_SPEED_MS, -1)
            self.fly_to(ox, oy, alt, self.corr_yaw, "to O", 90)
        dt = time.time() - self.t_phase
        self.results.append((self.cur, ok, dt, "through" if ok else "skipped/aborted → lawnmower"))
        log.info(f"█  {self.cur}: {'✓ THROUGH' if ok else '✗ not through — continuing to the lawnmower'} "
                 f"in {dt:.0f} s")

    def blue(self):
        self.begin("BLUE CORRIDOR")
        if self.a.blue == "skip":
            self.results.append((self.cur, True, 0.0, "skipped (--blue skip)"))
            log.info("  skipped (--blue skip) → RTL")
            return
        if S.PAYLOAD and self.sv is not None and self.sv.payload_done:
            log.info("  (rope left with the payload — no rewind, flying blue)")
        if self.scam is not None:
            self.drone.cmd(S.mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, 1, S.SPEED_MPS, -1)
        bi = self.pts.get("BLUE_IN")
        if bi:                                         # transit at survey height first
            bx, by = S.latlon_to_xy(*bi, *self.origin)
            p = self.drone.xy()
            yaw = S.bearing(p, (bx, by)) if math.dist(p, (bx, by)) > 1 else self.start_yaw
            self.fly_to(p[0], p[1], max(self.drone.alt() or 0, S.SURVEY_ALT_M), yaw, "up", 30)
            self.fly_to(bx, by, S.SURVEY_ALT_M, yaw, "to BLUE_IN at 10 m", 150)
        ok = self.gps_corridor("blue")
        dt = time.time() - self.t_phase
        self.results.append((self.cur, ok, dt, "through" if ok else "attempted — stopped (obstacle/LiDAR)"))
        log.info(f"█  {self.cur}: {'✓ THROUGH' if ok else '✗ attempted, not through'} in {dt:.0f} s → RTL")

    def climb_soft(self):
        """Up to survey height, never aborts the mission."""
        p = self.drone.xy()
        t0 = time.time()
        while (self.drone.alt() or 0) < S.SURVEY_ALT_M - 0.5 and time.time() - t0 < 25:
            self.drone.check()
            self.drone.goto_xy(p[0], p[1], S.SURVEY_ALT_M, getattr(self, "corr_yaw", self.start_yaw))
            time.sleep(0.2)

    def climb(self):
        self.begin("CLIMB")
        p = self.drone.xy()
        t0 = time.time()
        while (self.drone.alt() or 0) < S.SURVEY_ALT_M - 0.5 and time.time() - t0 < 25:
            self.drone.check()
            self.drone.goto_xy(p[0], p[1], S.SURVEY_ALT_M, getattr(self, "corr_yaw", self.start_yaw))
            time.sleep(0.2)
        self.end((self.drone.alt() or 0) >= S.SURVEY_ALT_M - 1.0, f"at {self.drone.alt():.1f} m")

    def survey(self):
        self.begin("SURVEY + PAYLOAD")
        cam = self.survey_cam_on()
        sv = self.sv
        sv.cam = cam
        # the corridor left the FC at its slow corridor speed (DO_CHANGE_SPEED
        # persists) — restore the survey speed (flight 26-10: survey crawled at 0.3 m/s)
        self.drone.cmd(S.mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, 1, S.SPEED_MPS, -1)
        log.info(f"  speed set back to {S.SPEED_MPS} m/s for the survey")
        xy = self.drone.xy()
        sv.fence_m = max(S.GEOFENCE_MARGIN_M, S.dist_outside(*xy, self.field) + 2.0)
        sv.phase("climb to survey alt")
        sv.wait_alt(xy[0], xy[1], S.SURVEY_ALT_M)
        sv.run(S.TARGET_QR)
        q = sv.confirmed
        p = self.drone.xy()
        if (self.drone.alt() or 0) < S.SURVEY_ALT_M - 0.5:
            log.info("  back up to 10 m after the payload")
            self.fly_to(p[0], p[1], S.SURVEY_ALT_M, sv.yaw, "up to 10 m", 30)
        if self.a.gps_corridor:
            self.results.append((self.cur, True, time.time() - self.t_phase,
                                 (f"payload delivered over QR #{q.id}" if sv.payload_done else
                                  f"QR #{q.id}, no payload") if q else "no QR matched"))
            return
        self.end(True, (f"payload delivered over QR #{q.id}" if sv.payload_done else
                        f"confirmed QR #{q.id}, no payload") if q else "no QR confirmed")

    # ── main ─────────────────────────────────────────────────────────────
    def run(self):
        took_over = False
        try:
            self.preflight()
            self.takeoff()
            if not getattr(self.a, "only", None):
                self.start_qr()
            else:
                log.info(f"  TEST --only {self.a.only}: no start tag, no lawnmower, no payload")
            if self.a.gps_corridor:
                if not self.a.skip_corridor:
                    self.green_gps()
                    self.climb_soft()
                if not self.a.skip_survey and not getattr(self.a, "only", None):
                    self.survey()
                self.blue()
            else:
                if not self.a.skip_corridor:
                    self.corridor()
                    self.climb()
                if not self.a.skip_survey:
                    self.survey()
            log.info("\n✓ MISSION COMPLETE → RTL")
        except (S.PilotTookOver, C.PilotTookOver):
            took_over = True
            log.warning(f"\n!!! Pilot / failsafe took over (mode {self.drone.mode() if self.drone else '?'}) "
                        "— script stops, you have control")
        except Abort as e:
            log.error(f"\n!!! ABORT: {e} → RTL")
        except S.GeofenceBreach as e:
            log.error(f"\n!!! GEOFENCE: {e} → RTL")
        except KeyboardInterrupt:
            log.warning("\n!!! Ctrl+C → RTL")
        except Exception as e:
            log.error(f"\n!!! ERROR: {e} → RTL")
            import traceback
            traceback.print_exc()
        finally:
            try:
                if self.drone and S.PAYLOAD:
                    self.drone.servo(S.WINCH_SERVO, S.WINCH_STOP_PWM, "winch STOP (safety)")
                if self.drone and not took_over and self.drone.armed() and self.drone.mode() != "LAND":
                    self.drone.set_mode("RTL")
            except Exception:
                pass
            if self.sv is not None:
                try:
                    self.sv.save()
                    self.sv.match_report()
                except Exception:
                    pass
            self.summary()
            for c in (self.scam, self.down, self.lidar):
                try:
                    if c:
                        c.stop()
                except Exception:
                    pass
            if self.pv:
                self.pv.close()
            if self.drone:
                self.drone.close()


def main():
    ap = argparse.ArgumentParser(description="Squadrone full mission")
    ap.add_argument("--conn", default=S.SERIAL_PORT)
    ap.add_argument("--baud", type=int, default=S.BAUD_RATE)
    ap.add_argument("--cam", choices=list(S.CAMS), default="imx519")
    ap.add_argument("--cam-offset", type=float)
    ap.add_argument("--skip-corridor", action="store_true")
    ap.add_argument("--skip-survey", action="store_true")
    ap.add_argument("--no-payload", action="store_true")
    ap.add_argument("--image-only", action="store_true")
    ap.add_argument("--no-border", action="store_true",
                    help="practice strip without the white border (wide colour gate)")
    ap.add_argument("--markers", choices=["apriltag", "qr"], default="apriltag",
                    help="apriltag = AprilTag 36h11 IDs (competition), qr = old QR pipeline")
    ap.add_argument("--vision-search", dest="gps_corridor", action="store_false",
                    help="old flow: SEARCH for the green banner (no GPS corridor points)")
    ap.add_argument("--survey-only", action="store_true",
                    help="skip BOTH corridors: start tag, lawnmower, payload, 10 m, RTL")
    ap.add_argument("--blue", choices=["hug", "attempt", "gap", "skip"], default="hug",
                    help="exit corridor: hug = left lane + wall-hug, slide across at each board, "
                         "climb to 10 m at the end (default); attempt = stop at the first board, "
                         "climb out; gap = old experimental avoidance; skip -> RTL")
    ap.add_argument("--only", choices=["green", "blue", "both"],
                    help="TEST: corridors only (no start tag, no lawnmower, no payload)")
    ap.add_argument("--ask", action="store_true",
                    help="operator confirms the drop (old behaviour); default: AUTO when sure")
    ap.add_argument("--tries", type=int)
    ap.add_argument("--step", type=float)
    ap.add_argument("--side", type=float)
    ap.add_argument("--corridor-alt", type=float)
    ap.add_argument("--corridor-len", type=float)
    ap.add_argument("--corridor-width", type=float)
    ap.add_argument("--corridor-speed", type=float)
    a = ap.parse_args()

    from picamera2 import Picamera2
    from pymavlink import mavutil
    S.Picamera2 = C.Picamera2 = Picamera2
    S.mavutil = C.mavutil = mavutil

    # survey settings
    S.CAM_MODEL = a.cam
    S.CAM_RES, S.HFOV_DEG, S.VFOV_DEG = S.CAMS[a.cam]
    S.DECODE_QR = not a.image_only
    S.MARKERS = a.markers
    S.PAYLOAD = not a.no_payload
    if S.DECODE_QR:
        S.load_wechat(None)
    # corridor settings
    C.WHITE_BORDER = not a.no_border
    C.set_banner_color("green", loose=a.no_border)
    C.OBSTACLES = False
    if a.survey_only:
        a.skip_corridor, a.blue = True, "skip"
    if a.only:
        S.PAYLOAD = False
        if a.only == "green":
            a.blue = "skip"
        elif a.only == "blue":
            a.skip_corridor = True
            if a.blue == "skip":
                a.blue = "hug"
    if a.gps_corridor:
        C.ON_ABORT = "continue"            # corridor trouble -> climb out, mission goes on
        S.AUTO_DROP = not a.ask
        # 31-10: the rope is LEFT with the payload -> no rewind (REWIND_BEFORE_RTL stays False)
    if a.cam == "imx500":
        C.DOWN_MODEL = "imx500"
        C.FULL_HFOV_DEG = C.DOWN_HFOV_DEG = 67.1
        C.FULL_VFOV_DEG = C.DOWN_VFOV_DEG = 52.9
    if a.cam_offset is not None:
        S.CAM_OFFSET_FWD_M = C.DOWN_OFFSET_FWD_M = a.cam_offset
    for opt, mod, name in (("tries", C, "MAX_TRIES"), ("step", C, "SEARCH_STEP_FWD_M"),
                           ("side", C, "SEARCH_SIDE_M"), ("corridor_alt", C, "CORRIDOR_ALT_M"),
                           ("corridor_len", C, "CORRIDOR_LEN_M"), ("corridor_width", C, "CORRIDOR_W_M"),
                           ("corridor_speed", C, "CORRIDOR_SPEED")):
        v = getattr(a, opt)
        if v is not None:
            setattr(mod, name, v)
    log.info("=" * 64)
    log.info(f"SQUADRONE FULL MISSION  cam={a.cam}  corridor={'skip' if a.skip_corridor else 'green'}  "
             f"survey={'skip' if a.skip_survey else 'yes'}  payload={'yes' if S.PAYLOAD else 'no'}  "
             f"banner={'practice (no border)' if a.no_border else 'venue (white border)'}  markers={a.markers if a.markers == 'apriltag' else ('QR image only' if a.image_only else 'QR read + image')}")
    log.info(f"  banner search {C.MAX_TRIES} tries x {C.SEARCH_STEP_FWD_M} m, ±{C.SEARCH_SIDE_M} m | "
             f"corridor {C.CORRIDOR_LEN_M} x {C.CORRIDOR_W_M} m at {C.CORRIDOR_ALT_M} m, "
             f"{C.CORRIDOR_SPEED} m/s | survey {S.SURVEY_ALT_M} m, delivery {S.DELIVERY_ALT_M} m")
    if a.gps_corridor:
        pts = load_points()
        log.info("  FLOW: start tag → GREEN corridor by GPS (vision shown/logged) + LiDARs → "
                 "lawnmower → " + ("AUTO drop when sure" if S.AUTO_DROP else "operator confirms drop")
                 + f" → BLUE {a.blue} → RTL.  Corridor trouble = climb out + continue.")
        for k, v in pts.items():
            log.info(f"    {k:9s} {v if v else 'NOT SET'}")
    log.info("=" * 64)
    if a.blue == "gap":
        ans = input("\n>>> --blue gap = EXPERIMENTAL obstacle avoidance (single front beam; "
                    "in simulation some runs came within a few cm of a board).\n"
                    "    Type GAP to use it, anything else = option 1 (stop at the board): ").strip()
        if ans != "GAP":
            a.blue = "attempt"
            log.info("  → blue = option 1 (attempt, stop at the board)")
    MissionRun(a).run()


if __name__ == "__main__":
    main()