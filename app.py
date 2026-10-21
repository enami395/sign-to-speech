"""Step 3 - Live sign recognition: webcam -> text -> voice.

Keys:  s = speak the text    c = clear the text    q = quit
"""
import os
import queue
import sys
import threading
from collections import Counter, deque

import cv2
import joblib
import numpy as np

from features import HandTracker

MODEL_FILE = "model.joblib"
CONF_THRESHOLD = 0.70   # ignore predictions below this probability
WINDOW = 12             # look at the last 12 frames...
NEEDED = 9              # ...a sign must win 9 of them to be accepted


class Stabilizer:
    """Accepts a sign only if it is held steadily, and only once per hold.

    To type the same letter twice (e.g. L, L), drop your hand briefly between.
    """

    def __init__(self, window=WINDOW, needed=NEEDED):
        self.history = deque(maxlen=window)
        self.needed = needed
        self.locked = False
        self.last = None

    def update(self, label):
        """Returns (accepted_label_or_None, progress between 0 and 1)."""
        self.history.append(label)
        votes = Counter(l for l in self.history if l is not None)
        top, count = votes.most_common(1)[0] if votes else (None, 0)
        progress = min(count / self.needed, 1.0)
        stable = top if count >= self.needed else None

        if stable is None or stable != self.last:
            self.locked = False
        if stable is not None and not self.locked:
            self.locked = True
            self.last = stable
            return stable, progress
        return None, progress


class Speaker:
    """Text-to-speech in a background thread so the video never freezes."""

    def __init__(self):
        self.q = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            import pyttsx3
        except Exception as exc:  # e.g. pyttsx3 not installed
            print(f"[TTS disabled] {exc}")
            pyttsx3 = None
        while True:
            text = self.q.get()
            if pyttsx3 is None:
                print(f"[would say] {text}")
                continue
            # A reused engine only speaks once on Windows (SAPI5 bug),
            # so build a fresh one for every utterance.
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 160)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception as exc:  # e.g. Linux without espeak
                print(f"[TTS error] {exc}")

    def say(self, text):
        if text.strip():
            self.q.put(text)


def put(frame, text, org, color=(255, 255, 255), scale=0.8, thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


def apply_action(label, text, speaker):
    if label == "SPACE":
        return text + " "
    if label == "DELETE":
        return text[:-1]
    if label == "SPEAK":
        speaker.say(text)
        return text
    if label == "I_LOVE_YOU":
        return text + "I LOVE YOU "
    return text + label


def main():
    if not os.path.exists(MODEL_FILE):
        sys.exit("No model found. Run collect_data.py then train.py first.")
    clf = joblib.load(MODEL_FILE)
    tracker = HandTracker()
    speaker = Speaker()
    stabilizer = Stabilizer()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit("Cannot open the webcam.")

    text = ""
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        feats, hand = tracker.process(frame)

        label, conf = None, 0.0
        if feats is not None:
            tracker.draw(frame, hand)
            probs = clf.predict_proba(feats.reshape(1, -1))[0]
            i = int(np.argmax(probs))
            conf = float(probs[i])
            if conf >= CONF_THRESHOLD:
                label = str(clf.classes_[i])

        accepted, progress = stabilizer.update(label)
        if accepted is not None:
            text = apply_action(accepted, text, speaker)

        # --- overlay ---
        shown = f"{label} ({conf:.0%})" if label else "-"
        put(frame, shown, (10, 35), (0, 255, 255), 1.0, 2)
        cv2.rectangle(frame, (10, 50), (210, 62), (60, 60, 60), -1)
        cv2.rectangle(frame, (10, 50), (10 + int(200 * progress), 62),
                      (0, 255, 0), -1)
        h = frame.shape[0]
        cv2.rectangle(frame, (0, h - 50), (frame.shape[1], h), (0, 0, 0), -1)
        put(frame, text[-35:] + "_", (10, h - 15), (255, 255, 255), 0.9, 2)

        cv2.imshow("Sign language to speech", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            text = ""
        elif key == ord("s"):
            speaker.say(text)

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
