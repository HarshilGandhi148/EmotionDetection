import json
import shutil

import numpy as np
import pytest
import torch
from PIL import Image

from emotion.data import ManifestDataset, label_for, load_manifest, make_transform, prepare_data


def test_label_aliases():
    assert label_for("Training/suprise-1.png") == 5
    assert label_for("Training/surprise-1.png") == 5
    assert label_for("Training/happy/abc.png") == 2
    with pytest.raises(ValueError, match="Unknown emotion"):
        label_for("Training/unknown-1.png")


def test_split_reproducible_and_disjoint(manifest_file, image_roots, tmp_path):
    first = load_manifest(manifest_file)
    second = prepare_data(*image_roots, tmp_path / "second.json")
    assert first == second
    sets = [{r["hash"] for r in first["records"] if r["split"] == split}
            for split in ("train", "validation", "test")]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert all(len(counts) == 6 for counts in first["counts"].values())


def test_duplicates_stay_together(image_roots, tmp_path):
    train, test = image_roots
    shutil.copyfile(train / "anger/0.png", train / "anger/copy.png")
    manifest = prepare_data(train, test, tmp_path / "manifest.json")
    original = next(r for r in manifest["records"] if r["path"] == str(train / "anger/0.png"))
    duplicate = next(r for r in manifest["records"] if r["path"] == str(train / "anger/copy.png"))
    assert original["split"] == duplicate["split"]


def test_leakage_is_rejected(image_roots, tmp_path):
    train, test = image_roots
    shutil.copyfile(train / "anger/0.png", test / "anger/copy.png")
    with pytest.raises(ValueError, match="audit failed"):
        prepare_data(train, test, tmp_path / "manifest.json")
    assert json.loads((tmp_path / "manifest.audit.json").read_text())["test_overlap"]


def test_bad_file_is_reported(image_roots, tmp_path):
    train, test = image_roots
    (train / "anger/bad.png").write_bytes(b"broken")
    with pytest.raises(ValueError, match="audit failed"):
        prepare_data(train, test, tmp_path / "manifest.json")


def test_preprocessing_matches_webcam(manifest_file):
    manifest = load_manifest(manifest_file)
    dataset = ManifestDataset(manifest, "validation")
    with Image.open(dataset.records[0]["path"]) as image:
        rgb = np.asarray(image.convert("RGB"))
    live_tensor = make_transform(manifest["preprocessing"])(Image.fromarray(rgb))
    assert torch.equal(dataset[0][0], live_tensor)
    assert live_tensor.shape == (1, 48, 48)


def test_statistics_use_training_only(manifest_file):
    manifest = load_manifest(manifest_file)
    pixels = []
    for row in manifest["records"]:
        if row["split"] == "train":
            with Image.open(row["path"]) as image:
                pixels.append(np.asarray(image, dtype=np.float64).ravel() / 255)
    values = np.concatenate(pixels)
    assert manifest["preprocessing"]["mean"] == pytest.approx(values.mean())
    assert manifest["preprocessing"]["std"] == pytest.approx(values.std())


def test_identities_are_grouped(image_roots, tmp_path):
    train, test = image_roots
    identities = {str(path): f"{root.name}-{path.stem}" for root in (train, test)
                  for path in root.rglob("*.png")}
    identity_path = tmp_path / "identities.json"
    identity_path.write_text(json.dumps(identities))
    manifest = prepare_data(train, test, tmp_path / "grouped.json", identities=identity_path)
    assignments = {}
    for row in manifest["records"]:
        identity = identities[row["path"]]
        assert identity not in assignments or assignments[identity] == row["split"]
        assignments[identity] = row["split"]


def test_unknown_class_requires_explicit_exclusion(image_roots, tmp_path):
    train, test = image_roots
    (train / "disgust").mkdir()
    Image.fromarray(np.zeros((48, 48), dtype=np.uint8)).save(train / "disgust/a.png")
    with pytest.raises(ValueError, match="audit failed"):
        prepare_data(train, test, tmp_path / "failed.json")
    prepare_data(train, test, tmp_path / "ok.json", excluded_labels=["disgust"])
    audit = json.loads((tmp_path / "ok.audit.json").read_text())
    assert len(audit["training"]["excluded_files"]) == 1


def test_clean_training_preserves_test_and_files(image_roots, tmp_path):
    train, test = image_roots
    shutil.copyfile(train / "anger/0.png", test / "anger/copy.png")
    shutil.copyfile(train / "anger/1.png", train / "fear/conflict.png")
    manifest = prepare_data(train, test, tmp_path / "clean.json", clean_training=True)
    assert len([r for r in manifest["records"] if r["split"] == "test"]) == 19
    assert manifest["audit"]["excluded_training_images"] == 3
    assert (train / "anger/0.png").exists()
    assert (train / "anger/1.png").exists()
    assert (train / "fear/conflict.png").exists()
    training_hashes = {r["hash"] for r in manifest["records"] if r["split"] != "test"}
    testing_hashes = {r["hash"] for r in manifest["records"] if r["split"] == "test"}
    assert not training_hashes & testing_hashes
