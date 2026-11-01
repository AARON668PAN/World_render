"""MediaPipe Hands 2D landmarks (21 per hand, same joint order as the MANO 21-joint layout) for every video frame.

usage: detect_hands_2d.py <video.mp4> <out.json>
"""

import json
import sys

import cv2
import mediapipe as mp

video, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(video)
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)
frames = []
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    h, w = bgr.shape[:2]
    result = hands.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    found = []
    for lm, hd in zip(result.multi_hand_landmarks or [], result.multi_handedness or []):
        found.append({"label": hd.classification[0].label, "score": hd.classification[0].score,
                      "uv": [[p.x * w, p.y * h] for p in lm.landmark]})
    frames.append(found)
json.dump({"width": w, "height": h, "frames": frames}, open(out, "w"))
counts = [len(f) for f in frames]
print(f"{len(frames)} frames; two hands in {counts.count(2)}, one in {counts.count(1)}, none in {counts.count(0)}")
