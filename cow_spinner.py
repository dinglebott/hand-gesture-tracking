import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
COW_PATH = os.path.join(SCRIPT_DIR, "assets", "cow.png")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# A pinch is detected when (thumb_tip to index_tip distance) / (wrist to middle_mcp distance)
# falls below this. Normalising by hand size keeps it stable across distances from the camera.
PINCH_RATIO_THRESHOLD = 0.45

# Pixels of slack around the cow's bounding box where a pinch still counts as "on the cow".
GRAB_MARGIN_PX = 30

MIN_SCALE = 0.1
MAX_SCALE = 4.0
INITIAL_SCALE = 0.5

# Multiplies the camera's native frame dimensions to size the display window. Processing stays
# at the camera's native resolution; only the on-screen window is scaled.
WINDOW_SCALE_FACTOR = 2.0


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def load_cow():
    img = cv2.imread(COW_PATH, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load {COW_PATH}")
    if img.shape[2] == 3:
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
        img = cv2.merge([img[:, :, 0], img[:, :, 1], img[:, :, 2], alpha])
    return img


def overlay_bgra(bg, fg_bgra, cx, cy, scale, rotation):
    """Alpha-composite a scaled & rotated BGRA foreground onto a BGR background, centred at (cx, cy).

    Rotation is in radians, counter-clockwise. The image rotates about its own centre.
    """
    h0, w0 = fg_bgra.shape[:2]
    bg_h, bg_w = bg.shape[:2]
    cos_t = math.cos(rotation)
    sin_t = math.sin(rotation)
    src_cx, src_cy = w0 / 2.0, h0 / 2.0
    # 2x3 affine that scales, rotates, then translates the source centre to (cx, cy).
    M = np.array([
        [scale * cos_t, -scale * sin_t, cx - scale * (cos_t * src_cx - sin_t * src_cy)],
        [scale * sin_t,  scale * cos_t, cy - scale * (sin_t * src_cx + cos_t * src_cy)],
    ], dtype=np.float32)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    warped = cv2.warpAffine(fg_bgra, M, (bg_w, bg_h), flags=interp,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    bg[:] = (alpha * warped[:, :, :3].astype(np.float32) + (1 - alpha) * bg.astype(np.float32)).astype(np.uint8)


def _px(lm, w, h):
    return (lm.x * w, lm.y * h)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hand_pinch(landmarks, w, h):
    """Return (pinch_point_px, is_pinching)."""
    thumb_tip = _px(landmarks[4], w, h)
    index_tip = _px(landmarks[8], w, h)
    wrist = _px(landmarks[0], w, h)
    middle_mcp = _px(landmarks[9], w, h)

    hand_size = max(1.0, _dist(wrist, middle_mcp))
    ratio = _dist(thumb_tip, index_tip) / hand_size
    pinch_point = ((thumb_tip[0] + index_tip[0]) / 2, (thumb_tip[1] + index_tip[1]) / 2)
    return pinch_point, ratio < PINCH_RATIO_THRESHOLD


def point_on_cow(pt, center, scale, rotation, cow_w, cow_h):
    # Transform the point into the cow's local (un-rotated, un-scaled) frame so the hitbox
    # tracks the cow when it's been spun.
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    cos_t = math.cos(-rotation)
    sin_t = math.sin(-rotation)
    lx = cos_t * dx - sin_t * dy
    ly = sin_t * dx + cos_t * dy
    half_w = cow_w * scale / 2 + GRAB_MARGIN_PX
    half_h = cow_h * scale / 2 + GRAB_MARGIN_PX
    return abs(lx) <= half_w and abs(ly) <= half_h


def main():
    ensure_model()
    cow = load_cow()
    cow_h, cow_w = cow.shape[:2]

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    if not ok:
        print("Camera not available")
        return
    fh, fw = frame.shape[:2]

    cv2.namedWindow("Cow Spinner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Cow Spinner", int(fw * WINDOW_SCALE_FACTOR), int(fh * WINDOW_SCALE_FACTOR))

    center = [fw / 2.0, fh / 2.0]
    scale = INITIAL_SCALE
    rotation = 0.0  # radians

    # Interaction state.
    mode = "idle"  # "idle" | "drag" | "scale"
    drag_label = None
    drag_offset = (0.0, 0.0)  # image_center - pinch_point, captured when drag began
    scale_pair = (None, None)
    scale_initial_dist = 1.0
    scale_initial_scale = scale
    scale_initial_mid = (0.0, 0.0)
    scale_initial_center = (0.0, 0.0)
    # Rotation accumulates per-frame deltas, so a full 360° spin wraps cleanly.
    scale_prev_angle = 0.0

    start = time.monotonic()
    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            # Per-frame map: handedness label -> (pinch_point, is_pinching, landmarks).
            hands = {}
            if result.hand_landmarks:
                for i, lms in enumerate(result.hand_landmarks):
                    label = "?"
                    if result.handedness and i < len(result.handedness) and result.handedness[i]:
                        label = result.handedness[i][0].category_name
                    if label in hands:
                        label = f"{label}_{i}"
                    pt, pinching = hand_pinch(lms, fw, fh)
                    hands[label] = (pt, pinching, lms)

            pinchers = {k: v for k, v in hands.items() if v[1]}

            if mode == "idle":
                for label, (pt, _, _) in pinchers.items():
                    if point_on_cow(pt, center, scale, rotation, cow_w, cow_h):
                        mode = "drag"
                        drag_label = label
                        drag_offset = (center[0] - pt[0], center[1] - pt[1])
                        break

            elif mode == "drag":
                if drag_label not in pinchers:
                    mode = "idle"
                    drag_label = None
                else:
                    pt = pinchers[drag_label][0]
                    center = [pt[0] + drag_offset[0], pt[1] + drag_offset[1]]
                    # Promote to scale if a second hand pinches on the cow.
                    for label, (pt2, _, _) in pinchers.items():
                        if label == drag_label:
                            continue
                        if point_on_cow(pt2, center, scale, rotation, cow_w, cow_h):
                            pt1 = pinchers[drag_label][0]
                            mode = "scale"
                            scale_pair = (drag_label, label)
                            scale_initial_dist = max(1.0, _dist(pt1, pt2))
                            scale_initial_scale = scale
                            scale_initial_mid = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)
                            scale_initial_center = (center[0], center[1])
                            scale_prev_angle = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
                            break

            elif mode == "scale":
                a, b = scale_pair
                if a in pinchers and b in pinchers:
                    pt1, pt2 = pinchers[a][0], pinchers[b][0]
                    cur_dist = max(1.0, _dist(pt1, pt2))
                    new_scale = scale_initial_scale * (cur_dist / scale_initial_dist)
                    scale = max(MIN_SCALE, min(MAX_SCALE, new_scale))
                    mid = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)
                    center = [
                        scale_initial_center[0] + (mid[0] - scale_initial_mid[0]),
                        scale_initial_center[1] + (mid[1] - scale_initial_mid[1]),
                    ]
                    # Per-frame delta wrapped to [-pi, pi] so crossing the +/-pi seam doesn't snap.
                    cur_angle = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
                    delta = cur_angle - scale_prev_angle
                    delta = (delta + math.pi) % (2 * math.pi) - math.pi
                    rotation += delta
                    scale_prev_angle = cur_angle
                else:
                    # Drop to drag with whichever hand is still pinching, else idle.
                    remaining = [l for l in (a, b) if l in pinchers]
                    if remaining:
                        mode = "drag"
                        drag_label = remaining[0]
                        pt = pinchers[drag_label][0]
                        drag_offset = (center[0] - pt[0], center[1] - pt[1])
                    else:
                        mode = "idle"
                        drag_label = None
                    scale_pair = (None, None)

            overlay_bgra(frame, cow, center[0], center[1], scale, rotation)

            for label, (pt, pinching, lms) in hands.items():
                color = (0, 255, 0) if pinching else (0, 200, 255)
                for lm in lms:
                    cv2.circle(frame, (int(lm.x * fw), int(lm.y * fh)), 2, color, -1)
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 8, color, 2)

            rot_deg = (math.degrees(rotation) + 180) % 360 - 180
            cv2.putText(frame, f"mode: {mode}  scale: {scale:.2f}  rot: {rot_deg:+.0f}°", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "pinch on cow to drag  |  two-hand pinch to resize/rotate  |  q to quit",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Cow Spinner", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
