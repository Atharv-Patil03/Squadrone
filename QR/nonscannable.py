
#!/usr/bin/env python3

from pathlib import Path
import cv2
import numpy as np
from picamera2 import Picamera2

# ---------- SETTINGS ----------
BASE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = BASE_DIR / "references"

WARP_SIZE = 160
DARK_THRESHOLD = 100
MIN_MARKER_AREA = 2500

# Reject weak matches. Tune after testing real camera images.
MIN_MATCH_SCORE = 65.0

# ------------------------------


def order_corners(points):
    """Return corners in top-left, top-right,
    bottom-right, bottom-left order."""
    points = points.reshape(4, 2).astype(np.float32)

    sums = points.sum(axis=1)
    diffs = points[:, 1] - points[:, 0]

    return np.array([
        points[np.argmin(sums)],
        points[np.argmin(diffs)],
        points[np.argmax(sums)],
        points[np.argmax(diffs)]
    ], dtype=np.float32)


def warp_quad(image, corners):
    """Convert a quadrilateral into a square image."""
    dst = np.array([
        [0, 0],
        [WARP_SIZE - 1, 0],
        [WARP_SIZE - 1, WARP_SIZE - 1],
        [0, WARP_SIZE - 1]
    ], dtype=np.float32)

    transform = cv2.getPerspectiveTransform(
        order_corners(corners), dst
    )

    return cv2.warpPerspective(
        image, transform, (WARP_SIZE, WARP_SIZE)
    )


def find_square_candidates(gray):
    """Find dark, approximately square quadrilaterals."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        blurred, DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in sorted(
        contours, key=cv2.contourArea, reverse=True
    ):
        area = cv2.contourArea(contour)

        if area < MIN_MARKER_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(
            contour, 0.03 * perimeter, True
        )

        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        x, y, w, h = cv2.boundingRect(approx)

        if h == 0:
            continue

        ratio = w / h

        # Allow some perspective distortion.
        if 0.70 <= ratio <= 1.30:
            candidates.append((area, approx, (x, y, w, h)))

    return candidates


def normalize_marker(image):
    """Binarize a rectified marker for comparison."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
        if image.ndim == 3 else image

    gray = cv2.resize(gray, (WARP_SIZE, WARP_SIZE))

    # Otsu threshold adapts to image brightness.
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary


def load_references():
    """Build a template bank from the 10 reference images."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted(
        p for p in REFERENCE_DIR.iterdir()
        if p.suffix.lower() in extensions
    )

    if not files:
        raise RuntimeError(
            f"No reference images found in {REFERENCE_DIR}"
        )

    templates = []

    for path in files:
        image = cv2.imread(str(path))

        if image is None:
            print(f"Skipping unreadable image: {path.name}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = find_square_candidates(gray)

        if not candidates:
            print(f"Could not find square in: {path.name}")
            continue

        _, corners, _ = candidates[0]
        warped = warp_quad(image, corners)
        binary = normalize_marker(warped)

        # Compare all four possible rotations.
        rotations = [
            np.rot90(binary, k).copy() for k in range(4)
        ]

        templates.append({
            "name": path.stem,
            "rotations": rotations
        })

        print(f"Loaded reference: {path.name}")

    if not templates:
        raise RuntimeError(
            "No valid reference markers could be loaded."
        )

    print(f"\nLoaded {len(templates)} reference markers.")
    return templates


def match_marker(warped, templates):
    """Compare a detected marker against every template."""
    observed = normalize_marker(warped)
    best_name = None
    best_score = -1.0

    for template in templates:
        for reference in template["rotations"]:
            difference = cv2.absdiff(observed, reference)
            score = 100.0 * (
                1.0 - np.mean(difference) / 255.0
            )

            if score > best_score:
                best_score = score
                best_name = template["name"]

    return best_name, best_score


def main():
    templates = load_references()

    camera = Picamera2()
    camera.configure(
        camera.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
    )

    camera.start()
    print("\nCamera started. Show one marker to the camera.")
    print("Press Q to quit.")

    try:
        while True:
            # Picamera2 RGB frame -> OpenCV BGR frame.
            rgb = camera.capture_array()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            candidates = find_square_candidates(gray)

            if candidates:
                _, corners, (x, y, w, h) = candidates[0]
                warped = warp_quad(frame, corners)

                name, score = match_marker(warped, templates)

                if score >= MIN_MATCH_SCORE:
                    label = f"{name} | {score:.1f}%"
                else:
                    label = f"Unknown | best {score:.1f}%"

                cv2.polylines(
                    frame, [corners], True, (0, 255, 0), 2
                )

                cv2.putText(
                    frame, label, (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if score >= MIN_MATCH_SCORE
                    else (0, 0, 255),
                    2
                )

                cv2.imshow("Rectified marker", warped)

            cv2.imshow("Raspberry Pi AI Camera - Marker Scanner", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
