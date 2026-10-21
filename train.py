"""Step 2 - Train the classifier on the recorded landmarks (takes seconds)."""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_FILE = os.path.join("data", "landmarks.csv")
MODEL_FILE = "model.joblib"


def main():
    if not os.path.exists(DATA_FILE):
        sys.exit("No data found. Run collect_data.py first.")

    df = pd.read_csv(DATA_FILE, header=None)
    y = df[0].values
    X = df.iloc[:, 1:].values.astype(np.float32)

    counts = pd.Series(y).value_counts().sort_index()
    print("Samples per class:\n", counts.to_string(), "\n")
    if counts.min() < 30:
        print("WARNING: some classes have fewer than 30 samples. "
              "Record more for better results.\n")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)

    print(f"Test accuracy: {accuracy_score(y_te, pred):.1%}\n")
    print(classification_report(y_te, pred, zero_division=0))

    labels = sorted(set(y))
    cm = confusion_matrix(y_te, pred, labels=labels)
    confused = [(cm[i, j], labels[i], labels[j])
                for i in range(len(labels)) for j in range(len(labels))
                if i != j and cm[i, j] > 0]
    confused.sort(reverse=True)
    if confused:
        print("Most confused pairs (true -> predicted):")
        for n, true, wrong in confused[:5]:
            print(f"  {true} -> {wrong}: {n}")
        print("If a pair is confused a lot, change one of the two gestures "
              "or record more varied samples.\n")

    print("NOTE: this score is optimistic, because neighbouring video frames "
          "look alike. The real test is the live app.\n")

    # Final model: retrain on ALL the data
    final = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    final.fit(X, y)
    joblib.dump(final, MODEL_FILE)
    print(f"Model saved -> {MODEL_FILE}")


if __name__ == "__main__":
    main()
