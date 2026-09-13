"""Reproducible training, checkpoint loading, and held-out evaluation."""
import hashlib
import importlib.metadata
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from emotion import ARCHITECTURE_VERSION, CLASSES
from emotion.data import ManifestDataset, load_manifest, write_json
from emotion.metrics import summarize_confusion
from emotion.models import make_model, select_device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def loader(dataset, batch_size, workers, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                      shuffle=shuffle, generator=generator, worker_init_fn=seed_worker)


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate_loader(model, batches, device):
    model.eval()
    matrix = np.zeros((6, 6), dtype=np.int64)
    loss_sum, sample_count = 0.0, 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for images, labels in batches:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        if logits.shape != (len(labels), 6):
            raise ValueError(f"Expected six logits per image; got {tuple(logits.shape)}")
        loss_sum += criterion(logits, labels).item()
        sample_count += len(labels)
        targets, predictions = labels.cpu().numpy(), logits.argmax(1).cpu().numpy()
        matrix += np.bincount(targets * 6 + predictions, minlength=36).reshape(6, 6)
    if not sample_count:
        raise ValueError("Cannot evaluate an empty loader.")
    metrics = summarize_confusion(matrix)
    metrics["loss"] = loss_sum / sample_count
    return metrics


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("architecture_version") != ARCHITECTURE_VERSION
            or checkpoint.get("classes") != CLASSES):
        raise ValueError("Incompatible checkpoint. The legacy saved_model.pth.tar cannot be loaded; train a new versioned checkpoint.")
    model = make_model(checkpoint["architecture"])
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except RuntimeError as error:
        raise ValueError("Checkpoint tensor shapes do not match the declared architecture.") from error
    model.to(device).eval()
    return model, checkpoint


def rng_state(generator):
    state = {"python": random.getstate(), "numpy": np.random.get_state()[1].tolist(),
             "numpy_position": int(np.random.get_state()[2]),
             "numpy_has_gauss": int(np.random.get_state()[3]),
             "numpy_cached_gauss": float(np.random.get_state()[4]),
             "torch": torch.get_rng_state(), "loader": generator.get_state()}
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(("MT19937", np.asarray(state["numpy"], dtype=np.uint32),
                         state["numpy_position"], state["numpy_has_gauss"], state["numpy_cached_gauss"]))
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, value):
    # Replace only the explicitly selected run's checkpoint, atomically.
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def log_metrics(writer, prefix, values, epoch):
    for key, value in values.items():
        if isinstance(value, (float, int)):
            writer.add_scalar(f"{prefix}/{key}", value, epoch)
    for name, metrics in values["per_class"].items():
        for key, value in metrics.items():
            if value is not None:
                writer.add_scalar(f"{prefix}/{name}/{key}", value, epoch)


def train(manifest_path, run_dir, architecture="cnn", weighted=False, seed=42,
          epochs=80, patience=12, batch_size=64, workers=0, device_name="auto", resume=None):
    if epochs < 1 or patience < 1 or batch_size < 1 or workers < 0:
        raise ValueError("Epochs, patience, and batch size must be positive; workers must be nonnegative.")
    manifest = load_manifest(manifest_path)
    fingerprint = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise ValueError(f"Run directory is not empty: {run_dir}; select a new run or use --resume.")
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    device = select_device(device_name)
    config = dict(architecture=architecture, weighted=weighted, seed=seed, epochs=epochs,
                  patience=patience, batch_size=batch_size, workers=workers, device=str(device),
                  learning_rate=0.001, weight_decay=0.0001, manifest_sha256=fingerprint)
    config["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                             "packages": {name: importlib.metadata.version(name) for name in
                                          ("torch", "torchvision", "numpy", "Pillow", "scikit-learn", "opencv-python")}}
    model = make_model(architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    generator = torch.Generator().manual_seed(seed)
    augmented = ManifestDataset(manifest, "train", augment=True)
    train_loader = loader(augmented, batch_size, workers, True, generator)
    train_eval = loader(ManifestDataset(manifest, "train"), batch_size, workers)
    validation = loader(ManifestDataset(manifest, "validation"), batch_size, workers)
    counts = torch.tensor([manifest["counts"]["train"][c] for c in CLASSES], dtype=torch.float32)
    weights = (counts.sum() / (6 * counts)).to(device) if weighted else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    start_epoch, best_f1, best_loss, stale = 0, -1.0, float("inf"), 0
    if resume:
        if Path(resume).resolve() != (run_dir / "latest.pt").resolve():
            raise ValueError("Resume must use latest.pt in the original run directory.")
        model, checkpoint = load_checkpoint(resume, device)
        for key in ("architecture", "weighted", "seed", "epochs", "patience", "batch_size", "workers", "manifest_sha256"):
            if checkpoint["config"][key] != config[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        restore_rng(checkpoint["rng_state"], generator)
        start_epoch = checkpoint["epoch"] + 1
        best_f1, best_loss, stale = checkpoint["best_f1"], checkpoint["best_loss"], checkpoint["stale"]
        if not (run_dir / "best.pt").is_file():
            raise ValueError("Resume in the original run directory containing best.pt.")
    write_json(run_dir / "config.json", config)
    writer = SummaryWriter(str(run_dir / "tensorboard"), purge_step=start_epoch if resume else None)
    try:
        for epoch in range(start_epoch, epochs):
            if stale >= patience:
                break
            started = time.perf_counter()
            model.train()
            running_loss, seen = 0.0, 0
            learning_rate = optimizer.param_groups[0]["lr"]
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss.")
                loss.backward()
                optimizer.step()
                denominator = weights[labels].sum().item() if weighted else len(labels)
                running_loss += loss.item() * denominator
                seen += denominator
            train_metrics = evaluate_loader(model, train_eval, device)
            valid_metrics = evaluate_loader(model, validation, device)
            f1, val_loss = valid_metrics["macro_f1"], valid_metrics["loss"]
            improved = f1 > best_f1 or (f1 == best_f1 and val_loss < best_loss)
            if improved:
                best_f1, best_loss, stale = f1, val_loss, 0
            else:
                stale += 1
            scheduler.step()
            record = {"epoch": epoch, "train": train_metrics, "validation": valid_metrics,
                      "augmented_optimization_loss": running_loss / seen,
                      "learning_rate": learning_rate, "seconds": time.perf_counter() - started}
            write_json(run_dir / "epochs" / f"{epoch:03d}.json", record)
            log_metrics(writer, "train", train_metrics, epoch)
            log_metrics(writer, "validation", valid_metrics, epoch)
            writer.add_scalar("learning_rate", learning_rate, epoch)
            writer.add_scalar("epoch_seconds", record["seconds"], epoch)
            writer.flush()
            checkpoint = {
                "architecture_version": ARCHITECTURE_VERSION, "architecture": architecture,
                "classes": CLASSES, "preprocessing": manifest["preprocessing"], "config": config,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(), "rng_state": rng_state(generator),
                "epoch": epoch, "best_f1": best_f1, "best_loss": best_loss, "stale": stale,
                "validation": valid_metrics,
            }
            save_checkpoint(run_dir / "latest.pt", checkpoint)
            if improved:
                save_checkpoint(run_dir / "best.pt", checkpoint)
            print(f"epoch={epoch + 1}/{epochs} train_loss={train_metrics['loss']:.4f} "
                  f"val_f1={f1:.4f} val_accuracy={valid_metrics['accuracy']:.4f} "
                  f"seconds={record['seconds']:.1f}", flush=True)
    finally:
        writer.close()
    return run_dir / "best.pt"


def evaluate(manifest_path, checkpoint_path, output, split="test", device_name="auto", batch_size=64, workers=0):
    manifest = load_manifest(manifest_path)
    device = select_device(device_name)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    fingerprint = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    if checkpoint["config"]["manifest_sha256"] != fingerprint:
        raise ValueError("Evaluation manifest differs from the training manifest.")
    batches = loader(ManifestDataset(manifest, split), batch_size, workers)
    report = evaluate_loader(model, batches, device)
    report.update({"split": split, "checkpoint": str(checkpoint_path), "classes": CLASSES,
                   "seed": checkpoint["config"]["seed"], "manifest_sha256": fingerprint})
    report["data_audit"] = manifest.get("audit", {})
    write_json(output, report)
    print(json.dumps({k: report[k] for k in ("accuracy", "macro_f1", "macro_sensitivity", "macro_specificity")}, indent=2))
    return report
