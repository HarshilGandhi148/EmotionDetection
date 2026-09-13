import numpy as np
import pytest
import torch
from PIL import Image

from emotion import CLASSES
from emotion.data import prepare_data


@pytest.fixture(autouse=True)
def few_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def image_roots(tmp_path):
    rng = np.random.default_rng(7)
    for split, count in (("train", 14), ("test", 3)):
        for label in CLASSES:
            directory = tmp_path / split / label
            directory.mkdir(parents=True)
            for i in range(count):
                Image.fromarray(rng.integers(0, 256, (48, 48), dtype=np.uint8)).save(directory / f"{i}.png")
    return tmp_path / "train", tmp_path / "test"


@pytest.fixture
def manifest_file(tmp_path, image_roots):
    target = tmp_path / "manifest.json"
    prepare_data(*image_roots, target)
    return target
