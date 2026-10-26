#!/usr/bin/env python3
"""
SQUADRONE — FULL MISSION (one flight)                       File: mission.py
Team: Squadrone AT2026RC60, PVGCOET Pune, SAEINDIA AeroTHON 2026

Glues the two field-tested scripts together — they must be in the SAME folder
with exactly these names:   survey.py   corridor.py   mission.py

PHASES
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
        if not self.a.skip_corridor:
            self.lidar = C.Lidars()
            self.lidar.start()
            time.sleep(0.6)
            vals = {k: self.lidar.get(k) for k in ("front", "left", "right")}
            alive = {k: self.lidar.alive(k) for k in vals}
            log.info("  LiDARs: " + "  ".join(
                f"{k}={'%.2fm' % v if v is not None else '--'}{'' if alive[k] else ' (NOT ANSWERING)'}"
                for k, v in vals.items()))
            if not all(alive.values()):
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
        ok = self.sv.capture_reference(self.home)
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
        xy = self.drone.xy()
        sv.fence_m = max(S.GEOFENCE_MARGIN_M, S.dist_outside(*xy, self.field) + 2.0)
        sv.phase("climb to survey alt")
        sv.wait_alt(xy[0], xy[1], S.SURVEY_ALT_M)
        sv.run(S.TARGET_QR)
        q = sv.confirmed
        self.end(True, (f"payload delivered over QR #{q.id}" if sv.payload_done else
                        f"confirmed QR #{q.id}, no payload") if q else "no QR confirmed")

    # ── main ─────────────────────────────────────────────────────────────
    def run(self):
        took_over = False
        try:
            self.preflight()
            self.takeoff()
            self.start_qr()
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
    S.PAYLOAD = not a.no_payload
    if S.DECODE_QR:
        S.load_wechat(None)
    # corridor settings
    C.set_banner_color("green")
    C.OBSTACLES = False
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
             f"QR={'image only' if a.image_only else 'read + image'}")
    log.info(f"  banner search {C.MAX_TRIES} tries x {C.SEARCH_STEP_FWD_M} m, ±{C.SEARCH_SIDE_M} m | "
             f"corridor {C.CORRIDOR_LEN_M} x {C.CORRIDOR_W_M} m at {C.CORRIDOR_ALT_M} m, "
             f"{C.CORRIDOR_SPEED} m/s | survey {S.SURVEY_ALT_M} m, delivery {S.DELIVERY_ALT_M} m")
    log.info("=" * 64)
    MissionRun(a).run()


if __name__ == "__main__":
    main()