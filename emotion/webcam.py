"""Face-localized single-person inference without any dataset dependency."""
import time

import cv2
import numpy as np
import torch
from PIL import Image

from emotion import CLASSES
from emotion.data import make_transform, write_json
from emotion.models import select_device
from emotion.training import load_checkpoint, synchronize


def crop_largest_face(frame, boxes):
    if len(boxes) == 0:
        return None, None
    x, y, width, height = map(int, max(boxes, key=lambda box: int(box[2]) * int(box[3])))
    side = min(max(width, height), frame.shape[0], frame.shape[1])
    left = min(max(0, x + width // 2 - side // 2), frame.shape[1] - side)
    top = min(max(0, y + height // 2 - side // 2), frame.shape[0] - side)
    return frame[top:top + side, left:left + side], (left, top, side, side)


def timing_report(samples):
    if not samples:
        return {"frames": 0}
    result = {"frames": len(samples)}
    for key in samples[0]:
        values = np.array([row[key] for row in samples]) * 1000
        result[key + "_ms"] = {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}
    result["updates_per_second"] = len(samples) / sum(row["total"] for row in samples)
    return result


def webcam(checkpoint, device_name="cpu", camera=0, timing_output=None, max_frames=0):
    device = select_device(device_name)
    model, metadata = load_checkpoint(checkpoint, device)
    transform = make_transform(metadata["preprocessing"])
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("OpenCV Haar cascade could not be loaded.")
    capture = cv2.VideoCapture(camera)
    samples = []
    try:
        if not capture.isOpened():
            raise RuntimeError("Camera could not open; check camera index and macOS camera permission.")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        with torch.inference_mode():
            while True:
                started = time.perf_counter()
                ok, frame = capture.read()
                captured = time.perf_counter()
                if not ok or frame is None:
                    raise RuntimeError("Camera stopped returning frames.")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
                crop, box = crop_largest_face(frame, boxes)
                detected = time.perf_counter()
                preprocessed = inferred = detected
                label = "No face detected"
                if crop is not None:
                    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    tensor = transform(image).unsqueeze(0).to(device)
                    synchronize(device)
                    preprocessed = time.perf_counter()
                    probabilities = model(tensor).softmax(dim=1)[0]
                    score, index = probabilities.max(dim=0)
                    score, index = score.item(), index.item()
                    synchronize(device)
                    inferred = time.perf_counter()
                    label = f"{CLASSES[index]} | model score {score:.0%}"
                    x, y, width, height = box
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 210, 40), 2)
                cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 210, 40), 2)
                cv2.imshow("Emotion Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                finished = time.perf_counter()
                samples.append(dict(capture=captured - started, detection=detected - captured,
                                    preprocessing=preprocessed - detected, model=inferred - preprocessed,
                                    display=finished - inferred, total=finished - started))
                if key in (27, ord("q")) or (max_frames and len(samples) >= max_frames):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if timing_output:
            write_json(timing_output, {"device": str(device), **timing_report(samples)})
