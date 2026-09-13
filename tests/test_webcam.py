import numpy as np
import pytest
import torch

from emotion.webcam import crop_largest_face, webcam


def test_largest_crop_is_square_and_bounded():
    frame = np.zeros((100, 150, 3), dtype=np.uint8)
    crop, box = crop_largest_face(frame, [(5, 5, 10, 10), (120, 70, 40, 50)])
    assert crop.shape == (50, 50, 3)
    assert box == (100, 50, 50, 50)
    assert crop_largest_face(frame, []) == (None, None)


@pytest.mark.parametrize("opened,read_ok,boxes", [(False, True, []), (True, False, []), (True, True, []), (True, True, [(0, 0, 48, 48)])])
def test_camera_cleanup_and_single_forward(monkeypatch, opened, read_ok, boxes):
    import emotion.webcam as module
    state = {"released": False, "closed": False, "calls": 0}

    class Camera:
        def isOpened(self): return opened
        def set(self, *args): pass
        def read(self): return read_ok, np.zeros((64, 64, 3), dtype=np.uint8)
        def release(self): state["released"] = True

    class Detector:
        def empty(self): return False
        def detectMultiScale(self, *args, **kwargs): return boxes

    class Model:
        def __call__(self, images):
            state["calls"] += 1
            return torch.zeros(1, 6)

    monkeypatch.setattr(module, "load_checkpoint", lambda *args: (Model(), {"preprocessing": {"size": 48, "mean": 0.5, "std": 0.2}}))
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda *_: Camera())
    monkeypatch.setattr(module.cv2, "CascadeClassifier", lambda *_: Detector())
    monkeypatch.setattr(module.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(module.cv2, "waitKey", lambda *args: 27)
    monkeypatch.setattr(module.cv2, "destroyAllWindows", lambda: state.update(closed=True))
    if not opened or not read_ok:
        with pytest.raises(RuntimeError, match="Camera"):
            webcam("unused.pt")
    else:
        webcam("unused.pt")
    assert state["released"] and state["closed"]
    assert state["calls"] == (1 if opened and read_ok and boxes else 0)
