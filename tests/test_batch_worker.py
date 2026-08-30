from __future__ import annotations

import threading
import time

from yolo_world_annotator.app.inference_worker import InferenceWorker


class FakeDetector:
    def __init__(self, worker: InferenceWorker, cancel_after_first: bool = False) -> None:
        self.worker = worker
        self.cancel_after_first = cancel_after_first
        self.calls = 0

    def set_classes(self, classes) -> None:
        self.classes = classes

    def predict(self, image, **kwargs):
        self.calls += 1
        if self.cancel_after_first and self.calls == 1:
            self.worker.request_cancel()
        return []


def _payload(count: int) -> dict:
    return {
        "images": [f"image-{index}.jpg" for index in range(count)],
        "classes": ["person"],
        "confidence": 0.25,
        "iou": 0.45,
        "imgsz": 640,
    }


def test_batch_cancel_stops_after_current_inference(qapp) -> None:
    worker = InferenceWorker()
    detector = FakeDetector(worker, cancel_after_first=True)
    worker.detector = detector
    finished = []
    worker.batch_finished.connect(lambda cancelled, done, total: finished.append((cancelled, done, total)))
    worker.start_batch(_payload(5))
    assert detector.calls == 1
    assert finished == [(True, 1, 5)]


def test_batch_pause_resume_uses_thread_safe_gate(qapp) -> None:
    worker = InferenceWorker()
    detector = FakeDetector(worker)
    worker.detector = detector
    original_predict = detector.predict

    def predict_and_pause(image, **kwargs):
        result = original_predict(image, **kwargs)
        if detector.calls == 1:
            worker.request_pause()
            threading.Timer(0.05, worker.request_resume).start()
        return result

    detector.predict = predict_and_pause
    started = time.perf_counter()
    worker.start_batch(_payload(2))
    assert detector.calls == 2
    assert time.perf_counter() - started >= 0.04
