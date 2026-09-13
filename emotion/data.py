"""Audited manifests and common image preprocessing for training and webcam."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from emotion import CLASSES

ALIASES = {
    "angry": "anger", "anger": "anger", "fear": "fear", "fearful": "fear",
    "happy": "happiness", "happiness": "happiness", "neutral": "neutral",
    "sad": "sadness", "sadness": "sadness", "surprise": "surprise",
    "surprised": "surprise", "suprise": "surprise",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def label_for(path):
    path = Path(path)
    for candidate in (path.parent.name.lower(), path.stem.lower().split("-")[0]):
        if candidate in ALIASES:
            return CLASSES.index(ALIASES[candidate])
    raise ValueError(f"Unknown emotion label: {path}")


def grayscale(image):
    return ImageOps.exif_transpose(image).convert("L")


def scan(root, partition, excluded_labels):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    records, errors, ignored, excluded = [], [], [], []
    shapes = Counter()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            ignored.append(str(path))
            continue
        tokens = {path.parent.name.lower(), path.stem.lower().split("-")[0]}
        if tokens & set(excluded_labels):
            excluded.append(str(path))
            continue
        try:
            label = label_for(path)
            with Image.open(path) as image:
                image.load()
                shapes[f"{image.width}x{image.height}/{image.mode}"] += 1
                decoded = grayscale(image)
                digest = hashlib.sha256(str(decoded.size).encode() + decoded.tobytes()).hexdigest()
            records.append({"path": str(path), "label": label, "hash": digest, "split": partition})
        except (ValueError, OSError) as error:
            errors.append({"path": str(path), "error": str(error)})
    return records, {"root": str(root), "errors": errors, "ignored_files": ignored,
                     "excluded_files": excluded, "image_shapes": dict(shapes)}


def prepare_data(train_root, test_root, output, seed=42, excluded_labels=(), identities=None, clean_training=False):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Refusing to replace existing manifest: {output}")
    training, train_audit = scan(train_root, "train", excluded_labels)
    testing, test_audit = scan(test_root, "test", excluded_labels)
    audit = {"training": train_audit, "testing": test_audit}
    all_records = training + testing
    by_hash = defaultdict(list)
    for row in all_records:
        by_hash[row["hash"]].append(row)
    audit["duplicate_groups"] = [[r["path"] for r in group] for group in by_hash.values() if len(group) > 1]
    audit["conflicting_labels"] = [h for h, group in by_hash.items() if len({r["label"] for r in group}) > 1]
    audit["test_overlap"] = sorted({r["hash"] for r in training} & {r["hash"] for r in testing})
    audit["test_label_conflicts"] = [h for h, group in by_hash.items()
                                     if len({r["label"] for r in group if r["split"] == "test"}) > 1]
    audit["original_counts"] = {split: dict(Counter(CLASSES[r["label"]] for r in all_records if r["split"] == split))
                                for split in ("train", "test")}
    audit["clean_training"] = clean_training
    blocked_hashes = set(audit["conflicting_labels"]) | set(audit["test_overlap"])
    audit["excluded_training_records"] = [r for r in training if r["hash"] in blocked_hashes] if clean_training else []
    audit_path = output.with_suffix(".audit.json")
    write_json(audit_path, audit)
    if train_audit["errors"] or test_audit["errors"] or (not clean_training and (audit["conflicting_labels"] or audit["test_overlap"])):
        raise ValueError(f"Dataset audit failed; inspect {audit_path}. Resolve corrupt/unknown labels or overlapping data explicitly.")
    if clean_training:
        training = [r for r in training if r["hash"] not in blocked_hashes]
        all_records = training + testing
    for name, records in (("training", training), ("testing", testing)):
        if {r["label"] for r in records} != set(range(len(CLASSES))):
            raise ValueError(f"{name} must contain all six classes; inspect {audit_path}.")

    # Union duplicates and optional known identities before splitting.
    parent = {r["hash"]: r["hash"] for r in all_records}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    identity_map = json.loads(Path(identities).read_text()) if identities else {}
    identity_groups = defaultdict(list)
    for row in all_records:
        identity = identity_map.get(row["path"])
        if identities and identity is None:
            raise ValueError(f"Identity JSON must map every absolute image path to an identity: {row['path']}")
        if identity is not None:
            identity_groups[str(identity)].append(row)
    for group in identity_groups.values():
        if len({r["split"] for r in group}) > 1:
            raise ValueError("Known identity overlaps training and test; resolve the partition before training.")
        for row in group[1:]:
            parent[find(row["hash"])] = find(group[0]["hash"])
    groups = defaultdict(list)
    for row in training:
        groups[find(row["hash"])].append(row)
    if identities:
        from sklearn.model_selection import StratifiedGroupKFold
        splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
        _, valid_indices = next(splitter.split(training, [r["label"] for r in training],
                                               [find(r["hash"]) for r in training]))
        validation_hashes = {training[i]["hash"] for i in valid_indices}
    else:
        keys = sorted(groups)
        try:
            _, valid_keys = train_test_split(keys, test_size=0.15, random_state=seed,
                                            stratify=[groups[k][0]["label"] for k in keys])
        except ValueError as error:
            raise ValueError("Not enough independent samples per class for a stratified 15% validation split.") from error
        validation_hashes = set(valid_keys)
    for row in training:
        row["split"] = "validation" if row["hash"] in validation_hashes else "train"
    counts = {split: dict(Counter(CLASSES[r["label"]] for r in all_records if r["split"] == split))
              for split in ("train", "validation", "test")}
    if any(len(value) != len(CLASSES) for value in counts.values()):
        raise ValueError("Split lacks a class; provide more independent examples or revise identity groups.")
    count, total, squares = 0, 0.0, 0.0
    for row in training:
        if row["split"] != "train":
            continue
        with Image.open(row["path"]) as image:
            pixels = np.asarray(grayscale(image).resize((48, 48), Image.Resampling.BILINEAR), dtype=np.float64) / 255
        count += pixels.size
        total += float(pixels.sum())
        squares += float(np.square(pixels).sum())
    mean = total / count
    std = max((max(squares / count - mean ** 2, 0)) ** 0.5, 1e-6)
    manifest = {"version": 1, "classes": CLASSES, "seed": seed,
                "preprocessing": {"size": 48, "mean": mean, "std": std},
                "counts": counts, "identity_grouped": bool(identities), "records": all_records,
                "audit": {"clean_training": clean_training,
                          "excluded_training_images": len(audit["excluded_training_records"]),
                          "test_label_conflict_hashes": audit["test_label_conflicts"],
                          "test_partition_preserved": True}}
    write_json(output, manifest)
    print(json.dumps({"manifest": str(output), "counts": counts, "preprocessing": manifest["preprocessing"]}, indent=2))
    return manifest


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("version") != 1 or manifest.get("classes") != CLASSES:
        raise ValueError("Unsupported manifest version or class mapping.")
    return manifest


def make_transform(stats, augment=False):
    operations = [transforms.Lambda(grayscale),
                  transforms.Resize((stats["size"], stats["size"]), interpolation=transforms.InterpolationMode.BILINEAR)]
    if augment:
        operations += [transforms.RandomHorizontalFlip(),
                       transforms.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1),
                                               interpolation=transforms.InterpolationMode.BILINEAR),
                       transforms.ColorJitter(brightness=0.15, contrast=0.15)]
    operations += [transforms.ToTensor(), transforms.Normalize([stats["mean"]], [stats["std"]])]
    return transforms.Compose(operations)


class ManifestDataset(Dataset):
    def __init__(self, manifest, split, augment=False):
        self.records = [r for r in manifest["records"] if r["split"] == split]
        if not self.records:
            raise ValueError(f"Empty split: {split}")
        self.transform = make_transform(manifest["preprocessing"], augment)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image)
        return tensor, torch.tensor(row["label"], dtype=torch.long)
