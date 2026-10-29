#!/usr/bin/env python3
"""
SQUADRONE — Payload delivery test
File: payload_test.py
Team: Squadrone AT2026RC60, PVGCOET Pune, SAEINDIA AeroTHON 2026

SEQUENCE (flight)
  1. Arm, take off to TAKEOFF_ALT_M (10 m), hold position.
  2. Wait for the operator to type "yes".
  3. Descend to DROP_ALT_M (5 m) over the same spot, settle.
  3b. CONFIRM DELIVERY: operator types "yes" (or "up" / "rtl") — the drone
      holds position while waiting. Nothing moves on the payload before this.
  4. UNLATCH  : SERVO6 -> LATCH_OPEN_PWM, wait LATCH_WAIT_S.
  5. UNWIND   : SERVO5 -> WINCH_UNWIND_PWM for UNWIND_S (40 s), then WINCH_STOP_PWM.
  6. Ask: [w] wind back up (same time), [r] RTL, [l] land here.
  Ctrl+C at ANY moment: winch stops, latch is left as is, drone RTLs.
  Pilot switches mode (e.g. Loiter): winch stops, script lets go.

GROUND CHECK FIRST — PROPS OFF (no arming, servos only):
  python3 payload_test.py --bench               # full servo sequence, 40 s unwind
  python3 payload_test.py --bench --unwind-s 5  # short version
  python3 payload_test.py --servo 6 2000        # move one servo, then exit
  -> check that LATCH_OPEN_PWM really opens the latch and WINCH_UNWIND_PWM
     really unwinds (lowers). If not, swap the values below.

FLIGHT
  python3 payload_test.py
  python3 payload_test.py --alt 10 --drop-alt 5 --unwind-s 40

ArduPilot: SERVO5_FUNCTION and SERVO6_FUNCTION must be 0 (Disabled) so that
MAV_CMD_DO_SET_SERVO controls them (same as the GUI script).
"""
import os
os.environ["MAVLINK20"] = "1"

import sys
import math
import time
import logging
import argparse
import threading

from pymavlink import mavutil

# ═════════════════════════════════════════════════════════════════════════════
# SETTINGS — CHECK THESE ON THE GROUND (props off) WITH --bench
# ═════════════════════════════════════════════════════════════════════════════

SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE   = 921600

LATCH_SERVO      = 6       # 2-position switch in the GUI
LATCH_CLOSED_PWM = 2000    # GUI "OFF"  (assumed: holds the payload)
LATCH_OPEN_PWM   = 1000    # GUI "ON"   (assumed: releases the latch)
LATCH_WAIT_S     = 2.0

WINCH_SERVO       = 5      # 3-position switch in the GUI
WINCH_STOP_PWM    = 1500   # GUI "MID"
WINCH_UNWIND_PWM  = 2000   # GUI "HIGH" (assumed: lowers the payload)
WINCH_REWIND_PWM  = 1000   # GUI "LOW"  (assumed: pulls it back up)
UNWIND_S          = 40.0

TAKEOFF_ALT_M = 10.0
DROP_ALT_M    = 5.0
ALT_TOL_M     = 0.4
SETTLE_S      = 3.0

LOG_DIR = os.path.expanduser("~/squadrone")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(os.path.join(LOG_DIR, "payload_test.log"))])
log = logging.getLogger("payload")


class PilotTookOver(Exception):
    pass

# ═════════════════════════════════════════════════════════════════════════════
# DRONE LINK  (same reader / heartbeat filter as the survey script)
# ═════════════════════════════════════════════════════════════════════════════

class Drone:
    def __init__(self, conn, baud):
        log.info(f"Connecting {conn} @ {baud}…")
        self.mav = mavutil.mavlink_connection(conn, baud=baud)
        t0 = time.time()
        while True:          # wait for the FLIGHT CONTROLLER's own heartbeat
            hb = self.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
            if (hb and hb.get_srcSystem() != 0
                    and hb.get_srcComponent() == mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1
                    and hb.autopilot != mavutil.mavlink.MAV_AUTOPILOT_INVALID):
                self.mav.target_system = hb.get_srcSystem()
                self.mav.target_component = hb.get_srcComponent()
                break
            if time.time() - t0 > 30:
                raise RuntimeError("No heartbeat from the flight controller")
        log.info(f"  Heartbeat ✓ FC system={self.mav.target_system}")
        self.lk = threading.Lock()
        self.state = {}
        self.acks = {}
        self._stop = threading.Event()
        threading.Thread(target=self._reader, daemon=True).start()
        for msg_id, hz in ((33, 10), (24, 2)):
            self.cmd(mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, int(1e6 / hz))

    def _reader(self):
        L = mavutil.mavlink
        while not self._stop.is_set():
            try:
                m = self.mav.recv_match(blocking=True, timeout=0.5)
            except Exception:
                continue
            if m is None or m.get_type() == "BAD_DATA":
                continue
            if m.get_srcSystem() != self.mav.target_system:
                continue
            t = m.get_type()
            if t == "HEARTBEAT" and (m.get_srcComponent() != L.MAV_COMP_ID_AUTOPILOT1
                                     or m.autopilot == L.MAV_AUTOPILOT_INVALID):
                continue
            with self.lk:
                self.state[t] = m
                if t == "COMMAND_ACK":
                    self.acks[m.command] = (m.result, time.time())
            if t == "STATUSTEXT":
                log.info(f"  [FC] {m.text}")

    def get(self, t):
        with self.lk:
            return self.state.get(t)

    def cmd(self, c, *p):
        p = list(p) + [0] * (7 - len(p))
        self.mav.mav.command_long_send(self.mav.target_system, self.mav.target_component,
                                       c, 0, *p)

    # ── servos ───────────────────────────────────────────────────────────
    def servo(self, num, pwm, label=""):
        """MAV_CMD_DO_SET_SERVO, retried until the FC acknowledges it."""
        L = mavutil.mavlink
        for attempt in range(5):
            t_send = time.time()
            self.cmd(L.MAV_CMD_DO_SET_SERVO, num, pwm)
            while time.time() - t_send < 0.6:
                ack = self.acks.get(L.MAV_CMD_DO_SET_SERVO)
                if ack and ack[1] >= t_send:
                    if ack[0] == L.MAV_RESULT_ACCEPTED:
                        log.info(f"  SERVO{num} = {pwm} us  {label}  ✓")
                        return True
                    log.error(f"  SERVO{num} = {pwm} REJECTED by FC (result {ack[0]}) — "
                              f"is SERVO{num}_FUNCTION = 0?")
                    return False
                time.sleep(0.05)
        log.warning(f"  SERVO{num} = {pwm} sent, but no acknowledgement from the FC")
        return False

    # ── state ────────────────────────────────────────────────────────────
    def pos(self):
        m = self.get("GLOBAL_POSITION_INT")
        return (m.lat / 1e7, m.lon / 1e7, m.relative_alt / 1000.0) if m else None

    def alt(self):
        p = self.pos()
        return p[2] if p else None

    def armed(self):
        hb = self.get("HEARTBEAT")
        return bool(hb and hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def mode(self):
        hb = self.get("HEARTBEAT")
        if not hb:
            return None
        return next((n for n, i in self.mav.mode_mapping().items() if i == hb.custom_mode), None)

    def check(self):
        if self.mode() not in (None, "GUIDED"):
            raise PilotTookOver()

    def set_mode(self, name, timeout=5):
        mid = self.mav.mode_mapping()[name]
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.mav.mav.set_mode_send(self.mav.target_system,
                                       mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mid)
            time.sleep(0.5)
            if self.mode() == name:
                log.info(f"  Mode → {name}")
                return True
        log.warning(f"  Mode {name} not confirmed")
        return False

    def goto(self, lat, lon, alt):
        self.mav.mav.set_position_target_global_int_send(
            0, self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, 0b0000110111111000,
            int(lat * 1e7), int(lon * 1e7), float(alt), 0, 0, 0, 0, 0, 0, 0, 0)

    def wait_gps(self, timeout=300):
        log.info("Waiting for GPS 3D fix…")
        t0 = time.time()
        while time.time() - t0 < timeout:
            g = self.get("GPS_RAW_INT")
            if g and g.fix_type >= 3 and self.pos():
                log.info(f"  GPS ✓ sats={g.satellites_visible} hdop={g.eph / 100:.2f}")
                return
            time.sleep(0.5)
        raise RuntimeError("No GPS fix")

    def arm_and_takeoff(self, alt):
        if not self.set_mode("GUIDED"):
            raise RuntimeError("GUIDED not accepted")
        log.info("  Arming…")
        t0 = time.time()
        while not self.armed():
            if time.time() - t0 > 15:
                raise RuntimeError("Arm failed — check pre-arm messages")
            self.cmd(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            time.sleep(1.0)
        log.info("  Armed ✓ — taking off")
        time.sleep(0.5)
        self.cmd(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, alt)
        t0, t_dis, t_log = time.time(), None, 0
        while (self.alt() or 0) < 0.9 * alt:
            if time.time() - t0 > 45:
                raise RuntimeError(f"Takeoff timed out at {self.alt():.1f} m")
            if not self.armed():
                t_dis = t_dis or time.time()
                if time.time() - t_dis > 1.5:
                    raise RuntimeError("FC disarmed during takeoff")
            else:
                t_dis = None
            if time.time() - t_log > 1:
                log.info(f"  alt {self.alt() or 0:.1f} / {alt} m  mode={self.mode()}")
                t_log = time.time()
            time.sleep(0.2)
        log.info("  Takeoff done ✓")

    def close(self):
        self._stop.set()

# ═════════════════════════════════════════════════════════════════════════════
# STEPS
# ═════════════════════════════════════════════════════════════════════════════

def hold(drone, spot, alt, seconds, label, flying=True):
    """Hold position (when flying) for `seconds`, logging once a second."""
    t0, t_log = time.time(), 0
    while time.time() - t0 < seconds:
        if flying:
            drone.check()
            drone.goto(spot[0], spot[1], alt)
        if time.time() - t_log >= 1:
            a = drone.alt()
            log.info(f"  {label}  {time.time() - t0:4.0f}/{seconds:.0f} s"
                     + (f"   alt {a:.1f} m" if (flying and a is not None) else ""))
            t_log = time.time()
        time.sleep(0.2)


def go_to_alt(drone, spot, alt, timeout=30):
    t0, t_log = time.time(), 0
    while abs((drone.alt() or 0) - alt) > ALT_TOL_M:
        drone.check()
        if time.time() - t0 > timeout:
            raise RuntimeError(f"Could not reach {alt} m (at {drone.alt():.1f} m)")
        drone.goto(spot[0], spot[1], alt)
        if time.time() - t_log >= 1:
            log.info(f"  alt {drone.alt() or 0:.1f} → {alt} m")
            t_log = time.time()
        time.sleep(0.2)


def winch(drone, spot, alt, pwm, seconds, label, flying=True):
    """Run the winch for `seconds`, ALWAYS stopping it afterwards."""
    log.info(f"\n── {label}: SERVO{WINCH_SERVO} = {pwm} for {seconds:.0f} s ──")
    drone.servo(WINCH_SERVO, pwm, label)
    try:
        hold(drone, spot, alt, seconds, label, flying)
    finally:
        drone.servo(WINCH_SERVO, WINCH_STOP_PWM, "winch STOP")


def delivery_sequence(drone, spot, alt, unwind_s, flying=True):
    log.info(f"\n── UNLATCH: SERVO{LATCH_SERVO} = {LATCH_OPEN_PWM} ──")
    drone.servo(LATCH_SERVO, LATCH_OPEN_PWM, "latch OPEN")
    hold(drone, spot, alt, LATCH_WAIT_S, "latch opening", flying)
    winch(drone, spot, alt, WINCH_UNWIND_PWM, unwind_s, "UNWIND", flying)
    log.info("  ✓ Payload lowered")


def ask_while_holding(drone, spot, alt, prompt):
    """Keep the position target fresh in the background while the operator
    types an answer (input() blocks the main thread)."""
    stop = threading.Event()

    def keep():
        while not stop.is_set():
            if drone.mode() == "GUIDED":
                drone.goto(spot[0], spot[1], alt)
            time.sleep(0.5)
    threading.Thread(target=keep, daemon=True).start()
    try:
        ans = input(prompt).strip().lower()
    finally:
        stop.set()
    drone.check()
    return ans


def ask(prompt, choices):
    while True:
        a = input(prompt).strip().lower()
        if a in choices:
            return a
        print(f"  type one of: {', '.join(choices)}")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Squadrone payload delivery test")
    ap.add_argument("--conn", default=SERIAL_PORT)
    ap.add_argument("--baud", type=int, default=BAUD_RATE)
    ap.add_argument("--alt", type=float, default=TAKEOFF_ALT_M)
    ap.add_argument("--drop-alt", type=float, default=DROP_ALT_M)
    ap.add_argument("--unwind-s", type=float, default=UNWIND_S)
    ap.add_argument("--bench", action="store_true", help="servo sequence only, NO flight")
    ap.add_argument("--servo", nargs=2, type=int, metavar=("NUM", "PWM"),
                    help="set one servo and exit (ground test)")
    a = ap.parse_args()
    if not (3 <= a.drop_alt < a.alt <= 30 and 1 <= a.unwind_s <= 120):
        ap.error("need 3 <= drop-alt < alt <= 30 and 1 <= unwind-s <= 120")

    drone = Drone(a.conn, a.baud)
    try:
        # ── single servo ────────────────────────────────────────────────
        if a.servo:
            drone.servo(a.servo[0], a.servo[1], "manual")
            return
        # ── bench: servos only, no arming ───────────────────────────────
        if a.bench:
            if drone.armed():
                raise RuntimeError("Drone is ARMED — bench mode refuses to run")
            log.info("BENCH — PROPS OFF. Servo sequence only, no flight.")
            drone.servo(LATCH_SERVO, LATCH_CLOSED_PWM, "latch CLOSED")
            drone.servo(WINCH_SERVO, WINCH_STOP_PWM, "winch STOP")
            if ask("\n>>> Latch should be CLOSED, winch still. CONFIRM delivery sequence? "
                   "[yes/no]: ", ["yes", "no"]) != "yes":
                log.info("Cancelled.")
                return
            delivery_sequence(drone, None, None, a.unwind_s, flying=False)
            if ask("\nRewind the same time? [y/n]: ", ["y", "n"]) == "y":
                winch(drone, None, None, WINCH_REWIND_PWM, a.unwind_s, "REWIND", flying=False)
            if ask("Close the latch? [y/n]: ", ["y", "n"]) == "y":
                drone.servo(LATCH_SERVO, LATCH_CLOSED_PWM, "latch CLOSED")
            log.info("Bench done.")
            return

        # ── flight ──────────────────────────────────────────────────────
        drone.wait_gps()
        log.info(f"Plan: takeoff {a.alt:.0f} m → (you type yes) → {a.drop_alt:.0f} m → "
                 f"unlatch → unwind {a.unwind_s:.0f} s")
        drone.servo(LATCH_SERVO, LATCH_CLOSED_PWM, "latch CLOSED")
        drone.servo(WINCH_SERVO, WINCH_STOP_PWM, "winch STOP")
        input("\n>>> Payload latched? Area clear? ENTER to arm and take off <<<\n")
        drone.arm_and_takeoff(a.alt)
        p = drone.pos()
        spot = (p[0], p[1])
        log.info(f"  Holding at {spot[0]:.7f}, {spot[1]:.7f}")
        go_to_alt(drone, spot, a.alt)

        while ask_while_holding(drone, spot, a.alt,
                                f"\n>>> Holding at {a.alt:.0f} m. Type 'yes' to descend to "
                                f"{a.drop_alt:.0f} m: ") != "yes":
            pass

        log.info(f"\n── DESCEND to {a.drop_alt:.0f} m ──")
        go_to_alt(drone, spot, a.drop_alt)
        hold(drone, spot, a.drop_alt, SETTLE_S, "settling")
        while True:
            ans = ask_while_holding(
                drone, spot, a.drop_alt,
                f"\n>>> At {drone.alt() or 0:.1f} m. CONFIRM PAYLOAD DELIVERY "
                f"(unlatch + unwind {a.unwind_s:.0f} s)? Type 'yes' "
                f"(or 'up' = back to {a.alt:.0f} m, 'rtl' = abort): ")
            if ans == "yes":
                break
            if ans == "rtl":
                log.warning("Delivery cancelled — RTL")
                drone.set_mode("RTL")
                return
            if ans == "up":
                go_to_alt(drone, spot, a.alt)
                while ask_while_holding(drone, spot, a.alt,
                                        f">>> Back at {a.alt:.0f} m. 'yes' to descend again: ") != "yes":
                    pass
                go_to_alt(drone, spot, a.drop_alt)
                hold(drone, spot, a.drop_alt, SETTLE_S, "settling")
        log.info("  Delivery CONFIRMED by operator")
        delivery_sequence(drone, spot, a.drop_alt, a.unwind_s)

        choice = ask("\nNext: [w] wind back up, [r] RTL, [l] land here: ", ["w", "r", "l"])
        if choice == "w":
            winch(drone, spot, a.drop_alt, WINCH_REWIND_PWM, a.unwind_s, "REWIND")
            choice = ask("Next: [r] RTL, [l] land here: ", ["r", "l"])
        drone.set_mode("RTL" if choice == "r" else "LAND")
        log.info("✓ Done")

    except PilotTookOver:
        log.warning(f"Pilot took over (mode {drone.mode()}) — stopping winch, script exits")
        drone.servo(WINCH_SERVO, WINCH_STOP_PWM, "winch STOP")
    except KeyboardInterrupt:
        log.warning("Ctrl+C — stopping winch")
        drone.servo(WINCH_SERVO, WINCH_STOP_PWM, "winch STOP")
        if drone.armed() and drone.mode() == "GUIDED":
            drone.set_mode("RTL")
    except Exception as e:
        log.error(f"ERROR: {e}")
        drone.servo(WINCH_SERVO, WINCH_STOP_PWM, "winch STOP")
        if drone.armed() and drone.mode() == "GUIDED":
            drone.set_mode("RTL")
    finally:
        drone.close()


if __name__ == "__main__":
    main()
