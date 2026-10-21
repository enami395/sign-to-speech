"""Step 1 - Record training samples with your webcam.

Keys:  n = next class     p = previous class     r = start / stop recording
       c = clear the samples of the current class    q = save and quit

While recording, move your hand slightly (closer, farther, tilted a bit)
so the model learns variations. Target: 200 samples per class.
"""
import csv
import os
from collections import Counter

import cv2

from classes import CLASSES, HINTS
from features import HandTracker

DATA_FILE = os.path.join("data", "landmarks.csv")
TARGET = 400  # samples per class


def load_samples():
    rows = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, newline="") as f:
            for row in csv.reader(f):
                if row:
                    rows.append((row[0], [float(v) for v in row[1:]]))
    return rows


def save_samples(rows):
    os.makedirs("data", exist_ok=True)
    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        for label, feats in rows:
            writer.writerow([label] + [round(float(v), 5) for v in feats])
    print(f"Saved {len(rows)} samples -> {DATA_FILE}")


def put(frame, text, y, color=(255, 255, 255), scale=0.6):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def main():
    rows = load_samples()
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit("Cannot open the webcam.")

    idx, recording, frame_id = 0, False, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)          # mirror view, like a selfie
        feats, hand = tracker.process(frame)
        if hand is not None:
            tracker.draw(frame, hand)

        label = CLASSES[idx]
        counts = Counter(l for l, _ in rows)
        frame_id += 1

        if recording and feats is not None and frame_id % 2 == 0:
            rows.append((label, feats.tolist()))
            counts[label] += 1
            if counts[label] >= TARGET:
                recording = False
                save_samples(rows)
                print(f"Class {label} done. Press n for the next class.")

        put(frame, f"Class {idx + 1}/{len(CLASSES)}: {label}   "
                   f"{counts[label]}/{TARGET}", 25, (0, 255, 255), 0.7)
        put(frame, HINTS[label], 50, scale=0.5)
        if recording:
            status = "RECORDING" if hand is not None else "RECORDING - no hand!"
            put(frame, status, 80, (0, 0, 255), 0.7)
        else:
            put(frame, "r=record  n/p=next/prev  c=clear  q=quit", 80,
                (200, 200, 200), 0.5)
        put(frame, f"Total: {len(rows)}/{TARGET * len(CLASSES)}",
            frame.shape[0] - 15, (0, 255, 0), 0.6)

        cv2.imshow("Collect data", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            idx, recording = (idx + 1) % len(CLASSES), False
        elif key == ord("p"):
            idx, recording = (idx - 1) % len(CLASSES), False
        elif key == ord("r"):
            recording = not recording
            if not recording:
                save_samples(rows)
        elif key == ord("c"):
            rows = [r for r in rows if r[0] != label]
            recording = False
            print(f"Cleared class {label}.")

    save_samples(rows)
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
