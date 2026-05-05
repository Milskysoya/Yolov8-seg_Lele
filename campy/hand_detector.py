# hand_detector.py
import mediapipe as mp
import math
from config import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, MAX_HANDS

class HandDetector:
    def __init__(self):
        # Kompatibel dengan mediapipe 0.10.x (gunakan mp.python.solutions)
        try:
            # Coba cara lama dulu (mediapipe < 0.10)
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
        except AttributeError:
            # Cara baru untuk mediapipe >= 0.10
            from mediapipe.python.solutions import hands as mp_hands_module
            from mediapipe.python.solutions import drawing_utils as mp_drawing_module
            import types
            self.mp_hands = mp_hands_module
            self.mp_drawing = mp_drawing_module

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

    def detect_hands(self, rgb_frame):
        return self.hands.process(rgb_frame)

    def draw_landmarks(self, frame, hand_landmarks):
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

    def count_fingers(self, hand_landmarks, handedness):
        fingers = []
        wrist = hand_landmarks[0]

        thumb_tip = hand_landmarks[4]
        thumb_ip  = hand_landmarks[3]

        if handedness == "Left":  # Tangan kanan fisik (karena di-flip)
            fingers.append(1 if thumb_tip.x < thumb_ip.x else 0)
        else:
            fingers.append(1 if thumb_tip.x > thumb_ip.x else 0)

        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]

        for tip_idx, pip_idx in zip(tips, pips):
            tip = hand_landmarks[tip_idx]
            pip = hand_landmarks[pip_idx]
            dist_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
            dist_pip = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
            fingers.append(1 if dist_tip > dist_pip else 0)

        return sum(fingers)

    def recognize_gesture(self, hand_landmarks, handedness):
        count = self.count_fingers(hand_landmarks, handedness)
        gestures = {
            0: "Kepalan",
            1: "Satu Jari",
            2: "Dua Jari",
            3: "Tiga Jari",
            4: "Empat Jari",
            5: "Lima Jari (Buka)"
        }
        return gestures.get(count, "Unknown"), count
