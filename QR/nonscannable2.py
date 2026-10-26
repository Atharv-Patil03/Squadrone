#!/usr/bin/env python3
"""
Find one specific (possibly non-scannable) QR-like marker among other QR codes.

Platform : Raspberry Pi 5 + Arducam IMX519 (Picamera2 / libcamera)

Flow
  1. Camera init, continuous autofocus.
  2. 15 s window: detect the QR, highlight it, confirm 3x, store its pattern.
  3. (optional) terminal 'y' confirmation.
  4. Live search: every QR-like object in view is highlighted
       green  = matches the stored pattern
       red    = "NOT MATCHING QR"
     A match must be seen 3 times in a row -> prints "FOUND EXACT MATCH" and exits.

Detection (any of these can find a QR, results are merged)
  a) OpenCV QR localiser (QRCodeDetectorAruco if available, else QRCodeDetector).
     It only *locates* the code, so it still works when decoding fails.
  b) Finder-square geometry (3 corner squares) as a fallback for markers the
     OpenCV detector cannot see.

Identity check
  Every candidate is perspective-rectified to a square patch and compared with
  the stored reference patch (4 rotations, small shift tolerance). This is
  scale- and perspective-invariant, so a 5 m reference matches a 10 m view,
  and unlike ORB it tells apart two QRs that share the same finder squares.
"""

import sys
import time
import threading
from itertools import combinations

import cv2
import numpy as np
from picamera2 import Picamera2

try:
    from libcamera import controls
except ImportError:
    controls = None

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
CAMERA_NUM = 0
FRAME_SIZE = (1280, 720)
FRAME_FORMAT = "BGR888"

# Capture phase
CAPTURE_TIMEOUT_S = 15.0
REQUIRED_CONFIRMATIONS = 3          # used for both capture and final match
STABILITY_DELAY_S = 0.30            # min gap between two counted confirmations
STABILITY_TOL_PX = 15.0             # max target drift between consecutive frames
ROI_PADDING_PX = 20
ASK_USER_CONFIRMATION = True        # terminal 'y' prompt after capture

# Candidate filtering
MIN_QUAD_SIDE_PX = 18               # smallest QR (sqrt of area) to consider, ~10 m
MAX_FINDER_MARKERS = 3              # finder-geometry markers per frame

# Finder-square filter (side = sqrt(contour area), pixels)
FINDER_MIN_SIDE_PX = 6
FINDER_MAX_SIDE_PX = 250
MIN_ASPECT = 0.75
MIN_FILL = 0.80
REQUIRE_NESTED = False
MAX_SIZE_RATIO = 1.40
MAX_LEG_RATIO = 1.30
MAX_ABS_COS = 0.30
MIN_SPACING_RATIO = 1.5
MAX_SPACING_RATIO = 12.0

# Pattern comparison
PATCH_SIZE = 96                     # rectified patch is PATCH_SIZE x PATCH_SIZE
SHIFT_TOL_PX = 8                    # tolerated misalignment inside the patch
MATCH_SCORE_THRESHOLD = 0.60        # CALIBRATE: see notes (score is shown on screen)

VERIFY_PROMPT = ("\nNon-scannable QR pattern captured. "
                 "Proceed to live matching? (Enter 'y' to confirm): ")

GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 215, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# OpenCV QR localiser: Aruco-based one (OpenCV >= 4.8) is faster and more robust.
if hasattr(cv2, "QRCodeDetectorAruco"):
    _QR_DETECTOR = cv2.QRCodeDetectorAruco()
    _QR_DETECTOR_NAME = "QRCodeDetectorAruco"
else:
    _QR_DETECTOR = cv2.QRCodeDetector()
    _QR_DETECTOR_NAME = "QRCodeDetector"


# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------
def init_camera():
    print("[INFO] Cameras found:")
    for i, info in enumerate(Picamera2.global_camera_info()):
        print(f"       index {i}: {info.get('Model', 'unknown')}")
    print(f"[INFO] QR localiser: {_QR_DETECTOR_NAME} + finder-square fallback")

    cam = Picamera2(camera_num=CAMERA_NUM)
    cam.configure(cam.create_video_configuration(
        main={"size": FRAME_SIZE, "format": FRAME_FORMAT}))
    cam.start()

    try:
        cam.set_controls({
            "AfMode": controls.AfModeEnum.Continuous,
            "AfSpeed": controls.AfSpeedEnum.Fast,
        })
        print("[INFO] Continuous fast autofocus enabled.")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Autofocus controls not supported ({exc}); continuing without.")

    time.sleep(1.5)
    return cam


def grab_frame(cap):
    """Picamera2 'BGR888' arrays are in R,G,B order; convert to OpenCV's BGR."""
    return cv2.cvtColor(cap.capture_array("main"), cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def _clockwise(quad):
    """Force a consistent winding so rectification never mirrors the pattern."""
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    x, y = q[:, 0], q[:, 1]
    signed = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return (q[::-1].copy() if signed < 0 else q), abs(signed) / 2.0


def _make_candidate(quad, method):
    q, area = _clockwise(quad)
    size = float(np.sqrt(area))
    if size < MIN_QUAD_SIDE_PX or size > FRAME_SIZE[1] * 1.2:
        return None
    if not cv2.isContourConvex(np.round(q).astype(np.int32).reshape(-1, 1, 2)):
        return None
    return {"quad": q, "center": q.mean(axis=0), "size": size, "method": method}


def _dedupe(cands):
    kept = []
    for c in cands:  # earlier entries (OpenCV localiser) win over finder geometry
        if any(np.linalg.norm(c["center"] - k["center"]) < 0.4 * max(c["size"], k["size"])
               for k in kept):
            continue
        kept.append(c)
    return kept


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------
def _opencv_quads(gray):
    try:
        ok, pts = _QR_DETECTOR.detectMulti(gray)
    except cv2.error:
        return []
    if not ok or pts is None:
        return []
    return [q for q in np.asarray(pts, dtype=np.float32).reshape(-1, 4, 2)]


def _nesting_depth(hierarchy, idx):
    child = hierarchy[0][idx][2]
    if child == -1:
        return 0
    return 1 if hierarchy[0][child][2] == -1 else 2


def _finder_candidates(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 51, 7)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    raw = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        side = np.sqrt(area)
        if not (FINDER_MIN_SIDE_PX <= side <= FINDER_MAX_SIDE_PX):
            continue
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), _ = rect
        if w < 1 or h < 1 or min(w, h) / max(w, h) < MIN_ASPECT:
            continue
        if area / (w * h) < MIN_FILL:
            continue
        if REQUIRE_NESTED and _nesting_depth(hierarchy, i) < 2:
            continue
        raw.append({"center": np.array([cx, cy], np.float32), "side": float(side),
                    "area": float(area), "box": cv2.boxPoints(rect)})

    raw.sort(key=lambda c: c["area"], reverse=True)
    kept = []
    for cand in raw:  # drop concentric duplicates (outer ring vs inner core)
        if any(np.linalg.norm(cand["center"] - k["center"]) < 0.5 * k["side"] for k in kept):
            continue
        kept.append(cand)
    return kept


def _best_triple(cands):
    best, best_score = None, float("inf")
    for trio in combinations(cands[:15], 3):
        sides = [c["side"] for c in trio]
        size_ratio = max(sides) / min(sides)
        if size_ratio > MAX_SIZE_RATIO:
            continue
        mean_side = float(np.mean(sides))
        pts = [c["center"] for c in trio]
        for v in range(3):
            p, a, b = pts[v], pts[(v + 1) % 3], pts[(v + 2) % 3]
            pa, pb = a - p, b - p
            la, lb = float(np.linalg.norm(pa)), float(np.linalg.norm(pb))
            if la < 1e-6 or lb < 1e-6:
                continue
            leg_ratio = max(la, lb) / min(la, lb)
            if leg_ratio > MAX_LEG_RATIO:
                continue
            cos_ang = float(np.dot(pa, pb) / (la * lb))
            if abs(cos_ang) > MAX_ABS_COS:
                continue
            if not (MIN_SPACING_RATIO <= ((la + lb) / 2.0) / mean_side <= MAX_SPACING_RATIO):
                continue
            score = abs(cos_ang) + (leg_ratio - 1.0) + (size_ratio - 1.0)
            if score < best_score:
                best_score = score
                best = {"trio": trio, "corner": p, "a": a, "b": b,
                        "fourth": a + b - p, "mean_side": mean_side}
    return best


def _triple_to_quad(m):
    """Outer corners of the full QR from 3 finder centres (the 4th is implied)."""
    p, a, b, d = m["corner"], m["a"], m["b"], m["fourth"]
    u = (a - p) / np.linalg.norm(a - p)
    v = (b - p) / np.linalg.norm(b - p)
    h = m["mean_side"] / 2.0
    return np.array([p - h * u - h * v, a + h * u - h * v,
                     d + h * u + h * v, b - h * u + h * v], dtype=np.float32)


def _finder_quads(gray):
    remaining = _finder_candidates(gray)
    quads = []
    for _ in range(MAX_FINDER_MARKERS):
        m = _best_triple(remaining)
        if m is None:
            break
        quads.append(_triple_to_quad(m))
        used = {id(c) for c in m["trio"]}
        remaining = [c for c in remaining if id(c) not in used]
    return quads


def detect_qr_candidates(gray):
    """All QR-like objects in the frame (OpenCV localiser + finder geometry)."""
    cands = [_make_candidate(q, "opencv") for q in _opencv_quads(gray)]
    cands += [_make_candidate(q, "finders") for q in _finder_quads(gray)]
    return _dedupe([c for c in cands if c is not None])


# ----------------------------------------------------------------------------
# Pattern comparison
# ----------------------------------------------------------------------------
def rectify(gray, quad, size=PATCH_SIZE):
    """Perspective-warp the quad to a size x size grayscale patch."""
    big = size * 2
    dst = np.float32([[0, 0], [big - 1, 0], [big - 1, big - 1], [0, big - 1]])
    m = cv2.getPerspectiveTransform(np.asarray(quad, np.float32), dst)
    warped = cv2.warpPerspective(gray, m, (big, big), flags=cv2.INTER_LINEAR)
    return cv2.resize(warped, (size, size), interpolation=cv2.INTER_AREA)


def _prep(patch):
    """Binarise (Otsu) then soften, so lighting/blur differences matter less."""
    blur = cv2.GaussianBlur(patch, (0, 0), 1.2)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.GaussianBlur(bw.astype(np.float32) / 255.0, (0, 0), 2.0)


def pattern_score(ref_prep, cand_patch):
    """Best normalised correlation over 4 rotations and +/-SHIFT_TOL_PX shifts."""
    cand = _prep(cand_patch)
    m = SHIFT_TOL_PX
    templ = np.ascontiguousarray(ref_prep[m:PATCH_SIZE - m, m:PATCH_SIZE - m])
    best = -1.0
    for k in range(4):
        res = cv2.matchTemplate(np.ascontiguousarray(np.rot90(cand, k)),
                                templ, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(np.nan_to_num(res.max())))
    return best


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def draw_candidate(frame, cand, color, label, text_color=WHITE, thickness=3):
    q = np.round(cand["quad"]).astype(np.int32)
    cv2.polylines(frame, [q], True, color, thickness, cv2.LINE_AA)
    x, y = int(q[:, 0].min()), int(q[:, 1].min())
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
    y0 = max(th + 10, y - 6)
    cv2.rectangle(frame, (x, y0 - th - 8), (x + tw + 8, y0 + 4), color, -1)
    cv2.putText(frame, label, (x + 4, y0 - 3), FONT, 0.6, text_color, 2, cv2.LINE_AA)


def draw_banner(frame, text, color):
    cv2.putText(frame, text, (20, 40), FONT, 0.9, BLACK, 5, cv2.LINE_AA)
    cv2.putText(frame, text, (20, 40), FONT, 0.9, color, 2, cv2.LINE_AA)


def _roi_from_quad(frame, quad, padding=ROI_PADDING_PX):
    x, y, w, h = cv2.boundingRect(np.asarray(quad, np.float32))
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1, y1 = min(fw, x + w + padding), min(fh, y + h + padding)
    return frame[y0:y1, x0:x1].copy()


# ----------------------------------------------------------------------------
# Phase 2 - capture the reference
# ----------------------------------------------------------------------------
def capture_reference_pattern(cap):
    """Highlight QR(s), confirm the largest one 3x, return its pattern or None."""
    print(f"[INFO] Capture phase: show the target QR (timeout {CAPTURE_TIMEOUT_S:.0f} s, "
          f"'q' aborts). Keep other QR codes out of view.")
    start = time.monotonic()
    confirmations = 0
    last_confirm_t = 0.0
    prev = None  # (center, size) of the target in the previous frame

    while True:
        elapsed = time.monotonic() - start
        if elapsed > CAPTURE_TIMEOUT_S:
            print(f"[ERROR] QR pattern not confirmed within {CAPTURE_TIMEOUT_S:.0f} "
                  f"seconds. Check distance, lighting and focus, then retry.")
            return None

        frame = grab_frame(cap)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cands = detect_qr_candidates(gray)
        target = max(cands, key=lambda c: c["size"]) if cands else None

        view = frame.copy()
        for c in cands:
            if c is not target:
                draw_candidate(view, c, YELLOW, "QR-like", BLACK, 2)

        if target is None:
            confirmations, prev = 0, None
            status, color = "Searching for QR...", RED
        else:
            stable = (prev is not None
                      and np.linalg.norm(target["center"] - prev[0]) <= STABILITY_TOL_PX
                      and 0.8 <= target["size"] / prev[1] <= 1.25)
            prev = (target["center"], target["size"])
            now = time.monotonic()
            if not stable:
                confirmations, last_confirm_t = 0, now
                status, color = "QR seen, stabilising...", YELLOW
            else:
                if now - last_confirm_t >= STABILITY_DELAY_S:
                    confirmations += 1
                    last_confirm_t = now
                    print(f"Confirmation: {confirmations}/{REQUIRED_CONFIRMATIONS}")
                status, color = f"Confirmation: {confirmations}/{REQUIRED_CONFIRMATIONS}", GREEN
            draw_candidate(view, target, GREEN, f"TARGET ({target['method']})", BLACK)

            if confirmations >= REQUIRED_CONFIRMATIONS:
                patch = rectify(gray, target["quad"])
                roi = _roi_from_quad(frame, target["quad"])
                print(f"[INFO] Reference captured (ROI {roi.shape[1]}x{roi.shape[0]} px, "
                      f"found by {target['method']}).")
                return {"roi": roi, "patch": patch}

        draw_banner(view, f"{status}  ({max(0.0, CAPTURE_TIMEOUT_S - elapsed):4.1f}s)", color)
        cv2.imshow("QR capture", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Capture aborted by user.")
            return None


# ----------------------------------------------------------------------------
# Phase 3 - user verification
# ----------------------------------------------------------------------------
def verify_with_user(reference):
    """Show captured ROI + the rectified patch the matcher will use."""
    roi = reference["roi"]
    h = 256
    roi_s = cv2.resize(roi, (max(1, int(roi.shape[1] * h / roi.shape[0])), h))
    patch_s = cv2.cvtColor(cv2.resize(reference["patch"], (h, h)), cv2.COLOR_GRAY2BGR)
    combo = np.hstack([roi_s, patch_s])

    answer = {}
    thread = threading.Thread(target=lambda: answer.update(v=input(VERIFY_PROMPT)),
                              daemon=True)
    thread.start()
    while thread.is_alive():
        cv2.imshow("Reference (left: crop, right: matcher view)", combo)
        cv2.waitKey(50)
    return answer.get("v", "").strip().lower() == "y"


# ----------------------------------------------------------------------------
# Phase 4 - live search
# ----------------------------------------------------------------------------
def match_pattern_loop(cap, reference):
    """Return True once the matching QR is verified 3x, False if user quits."""
    ref_prep = _prep(reference["patch"])
    confirmations = 0
    last_confirm_t = 0.0
    print(f"[INFO] Live search started (match threshold {MATCH_SCORE_THRESHOLD:.2f}, "
          f"'q' quits).")

    while True:
        frame = grab_frame(cap)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cands = detect_qr_candidates(gray)

        best = None
        for c in cands:
            c["score"] = pattern_score(ref_prep, rectify(gray, c["quad"]))
            c["match"] = c["score"] >= MATCH_SCORE_THRESHOLD
            if c["match"] and (best is None or c["score"] > best["score"]):
                best = c

        now = time.monotonic()
        if best is None:
            confirmations = 0                      # must be consecutive
        elif confirmations == 0 or now - last_confirm_t >= STABILITY_DELAY_S:
            confirmations += 1
            last_confirm_t = now
            print(f"Match confirmation: {confirmations}/{REQUIRED_CONFIRMATIONS} "
                  f"(score {best['score']:.2f})")

        for c in cands:
            if c is best:
                draw_candidate(frame, c, GREEN, f"MATCH {c['score']:.2f}", BLACK, 4)
            else:
                draw_candidate(frame, c, RED, f"NOT MATCHING QR {c['score']:.2f}", WHITE, 3)

        if best is None:
            draw_banner(frame, f"Searching for matching QR...  (QRs in view: {len(cands)})", RED)
        else:
            draw_banner(frame, f"Match verification {confirmations}/{REQUIRED_CONFIRMATIONS}", GREEN)

        if confirmations >= REQUIRED_CONFIRMATIONS:
            draw_banner(frame, "FOUND EXACT MATCH", GREEN)
            cv2.imshow("QR search", frame)
            cv2.waitKey(1500)
            print("\n[SUCCESS] FOUND EXACT MATCH - matching QR verified "
                  f"{REQUIRED_CONFIRMATIONS} times (score {best['score']:.2f}).")
            return True

        cv2.imshow("QR search", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Stopped by user before an exact match was found.")
            return False


# ----------------------------------------------------------------------------
# Phase 5 - main with guaranteed cleanup
# ----------------------------------------------------------------------------
def main():
    cam = None
    exit_code = 0
    try:
        cam = init_camera()

        reference = capture_reference_pattern(cam)
        if reference is None:
            return 1

        if ASK_USER_CONFIRMATION and not verify_with_user(reference):
            print("Exiting without live matching.")
            return 0

        cv2.destroyAllWindows()
        exit_code = 0 if match_pattern_loop(cam, reference) else 2

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        exit_code = 130
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Unexpected failure: {exc}")
        exit_code = 1
    finally:
        if cam is not None:
            for fn in (cam.stop, cam.close):
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    pass
        cv2.destroyAllWindows()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())