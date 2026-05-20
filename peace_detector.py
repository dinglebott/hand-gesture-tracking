import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def is_peace_sign(landmarks) -> bool:
    # Finger is "extended" when its tip is above (smaller y than) its PIP joint.
    # Assumes a roughly upright hand; good enough for a minimal demo.
    tips_pips = [(8, 6), (12, 10), (16, 14), (20, 18)]  # index, middle, ring, pinky
    extended = [landmarks[t].y < landmarks[p].y for t, p in tips_pips]
    index_ext, middle_ext, ring_ext, pinky_ext = extended
    return index_ext and middle_ext and not ring_ext and not pinky_ext


def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)


def main():
    ensure_model()

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    was_peace = False
    start = time.monotonic()

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            is_peace = False
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                draw_landmarks(frame, landmarks)
                is_peace = is_peace_sign(landmarks)

            # Edge-trigger: print once when the gesture starts, not every frame.
            if is_peace and not was_peace:
                print("Peace sign detected!")
            was_peace = is_peace

            cv2.imshow("Hand Gesture Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
