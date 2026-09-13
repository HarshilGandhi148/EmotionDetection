"""Download the requested Kaggle FER2013 dataset and verify every image.

Run from the repository: .venv/bin/python scripts/import_fer2013.py
This imports the seven-class source dataset; it does not replace training splits.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
LABELS = {"angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    target = ROOT / "data" / "fer2013"
    if not args.verify_only:
        os.environ.setdefault("KAGGLEHUB_CACHE", str(ROOT / "data" / "kagglehub-cache"))
        import kagglehub
        downloaded = kagglehub.dataset_download("msambare/fer2013", output_dir=str(target))
        target = Path(downloaded).resolve()
        print(f"Downloaded dataset: {target}", flush=True)

    errors, counts, formats = [], {}, Counter()
    old_manifest = ROOT / "data" / "manifest.json"
    previous_hashes = set()
    if old_manifest.exists():
        previous_hashes = {r["hash"] for r in json.loads(old_manifest.read_text())["records"]}
    overlap = 0
    for split in ("train", "test"):
        directory = target / split
        if not directory.is_dir():
            errors.append(f"Missing partition: {directory}")
            continue
        found_labels = {p.name for p in directory.iterdir() if p.is_dir()}
        if found_labels != LABELS:
            errors.append(f"Unexpected classes in {split}: {sorted(found_labels)}")
        counts[split] = {}
        for label in sorted(LABELS):
            files = sorted(p for p in (directory / label).glob("*") if p.is_file())
            counts[split][label] = len(files)
            if not files:
                errors.append(f"No images: {split}/{label}")
            for path in files:
                try:
                    with Image.open(path) as image:
                        image.load()
                        formats[f"{image.width}x{image.height}/{image.mode}"] += 1
                        if image.size != (48, 48):
                            errors.append(f"Unexpected image dimensions: {path}: {image.size}")
                        gray = image.convert("L")
                        digest = hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest()
                        overlap += digest in previous_hashes
                except (OSError, ValueError) as error:
                    errors.append(f"Unreadable image: {path}: {error}")
        print(f"Verified {split}: {sum(counts[split].values())} images", flush=True)
    totals = {split: sum(values.values()) for split, values in counts.items()}
    # Published FER2013 partitions; a changed upstream layout warrants review.
    expected = {"train": 28709, "test": 7178}
    if totals != expected:
        errors.append(f"Unexpected totals: {totals}; expected {expected}")
    report = {"source": "msambare/fer2013", "path": str(target),
              "verified": not errors, "counts": counts, "totals": totals,
              "image_formats": dict(formats), "errors": errors,
              "images_matching_previous_manifest": overlap,
              "note": "FER2013 original labels, not FER+ annotations. Seven source classes include disgust; existing six-class training manifest is unchanged."}
    report_path = ROOT / "data" / "fer2013-import-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Report saved: {report_path}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
