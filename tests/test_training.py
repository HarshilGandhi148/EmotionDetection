import importlib
import json

import pytest
import torch

from emotion.models import make_model
from emotion.training import evaluate, load_checkpoint, train


@pytest.mark.parametrize("architecture", ["baseline", "cnn"])
def test_output_and_gradients(architecture):
    torch.manual_seed(4)
    model = make_model(architecture)
    logits = model(torch.randn(4, 1, 48, 48))
    assert logits.shape == (4, 6)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_tiny_batch_can_overfit():
    torch.manual_seed(12)
    model = make_model("cnn")
    images = torch.zeros(6, 1, 48, 48)
    for i in range(6):
        images[i, :, :, :] = i / 3 - 1
        images[i, :, i * 6:i * 6 + 6, :] = 2
    targets = torch.arange(6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    for _ in range(60):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(images), targets)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        assert (model(images).argmax(1) == targets).all()


def test_training_checkpoint_evaluation(manifest_file, tmp_path):
    checkpoint = train(manifest_file, tmp_path / "run", epochs=1, batch_size=32, device_name="cpu")
    model, metadata = load_checkpoint(checkpoint)
    assert metadata["classes"] == ["anger", "fear", "happiness", "neutral", "sadness", "surprise"]
    assert metadata["epoch"] == 0
    second, _ = load_checkpoint(checkpoint)
    sample = torch.randn(2, 1, 48, 48)
    with torch.no_grad():
        assert torch.equal(model(sample), second(sample))
    report = evaluate(manifest_file, checkpoint, tmp_path / "test.json", device_name="cpu")
    assert 0 <= report["accuracy"] <= 1
    assert (tmp_path / "run/tensorboard").is_dir()
    assert json.loads((tmp_path / "run/epochs/000.json").read_text())["validation"]["confusion_matrix"]


def test_legacy_checkpoint_rejected():
    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        load_checkpoint("saved_model.pth.tar")


def test_import_has_no_dataset_or_camera_side_effects():
    importlib.import_module("emotionRecognition")
    importlib.import_module("customDataset")


def test_resume_matches_uninterrupted_cpu_training(manifest_file, tmp_path, monkeypatch):
    import emotion.training as module
    options = dict(epochs=2, architecture="baseline", batch_size=32, device_name="cpu")
    uninterrupted = train(manifest_file, tmp_path / "full", **options)
    original_save = module.save_checkpoint

    def interrupt_after_first_epoch(path, value):
        original_save(path, value)
        if path.name == "best.pt" and value["epoch"] == 0:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(module, "save_checkpoint", interrupt_after_first_epoch)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train(manifest_file, tmp_path / "resumed", **options)
    monkeypatch.setattr(module, "save_checkpoint", original_save)
    resumed = train(manifest_file, tmp_path / "resumed", resume=tmp_path / "resumed/latest.pt", **options)
    _, full = load_checkpoint(uninterrupted)
    _, continuation = load_checkpoint(resumed)
    assert full["epoch"] == continuation["epoch"]
    for key in full["model_state"]:
        assert torch.equal(full["model_state"][key], continuation["model_state"][key])
