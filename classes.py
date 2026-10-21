"""The 15 classes of the project: 11 ASL letters + 4 control gestures.

You record your OWN data, so you can change any gesture below.
Rule of thumb: keep gestures visually different from each other.
"""

CLASSES = [
    "A", "B", "C", "D", "F", "I", "L", "O", "V", "W", "Y",
    "I_LOVE_YOU", "SPACE", "DELETE", "SPEAK",
]

HINTS = {
    "A": "Fist, thumb resting against the side of the index finger",
    "B": "Flat hand, fingers together and pointing up, thumb folded on the palm",
    "C": "Curved hand shaped like the letter C",
    "D": "Index finger up, other fingers touching the thumb",
    "F": "Thumb and index touching (OK sign), other 3 fingers up",
    "I": "Fist with only the pinky up",
    "L": "Thumb and index forming an L",
    "O": "All fingertips touching the thumb, forming an O",
    "V": "Index and middle fingers up, spread (peace sign)",
    "W": "Index, middle and ring fingers up and spread",
    "Y": "Thumb and pinky out, other fingers folded",
    "I_LOVE_YOU": "Thumb, index and pinky out",
    "SPACE": "Open hand, all 5 fingers spread, palm to camera",
    "DELETE": "Fist with the thumb pointing DOWN",
    "SPEAK": "Fist with the thumb pointing UP (reads the text aloud)",
}

# Classes that are actions, not letters
CONTROLS = {"SPACE", "DELETE", "SPEAK", "I_LOVE_YOU"}
