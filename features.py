"""Shared code: turns a MediaPipe hand into a 63-number feature vector.

The same function is used for data collection and for live recognition,
so the model always sees exactly the same kind of input.
"""
import numpy as np


def landmarks_to_features(hand_landmarks, handedness_label):
    """21 landmarks (x, y, z) -> 63 numbers, independent of position and size."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                   dtype=np.float32)
    pts -= pts[0]                      # wrist becomes the origin
    if handedness_label == "Left":     # mirror left hand so one model fits both
        pts[:, 0] *= -1
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale < 1e-6:
        scale = 1.0
    pts /= scale                       # hand size no longer matters
    return pts.flatten()


class HandTracker:
    """Small wrapper around MediaPipe Hands (pretrained hand landmark model)."""

    def __init__(self, max_hands=1):
        import mediapipe as mp         # imported here so train.py does not need it
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._drawer = mp.solutions.drawing_utils
        self._connections = mp.solutions.hands.HAND_CONNECTIONS

    def process(self, frame_bgr):
        """Returns (features, hand_landmarks), or (None, None) if no hand."""
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None, None
        hand = result.multi_hand_landmarks[0]
        label = result.multi_handedness[0].classification[0].label
        return landmarks_to_features(hand, label), hand

    def draw(self, frame_bgr, hand):
        self._drawer.draw_landmarks(frame_bgr, hand, self._connections)

    def close(self):
        self.hands.close()
