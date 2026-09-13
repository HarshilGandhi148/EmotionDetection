"""Explicit timing checks; no test labels are used for tuning."""
import time

import torch

from emotion.data import ManifestDataset, load_manifest, write_json
from emotion.models import make_model, select_device
from emotion.training import load_checkpoint, loader, synchronize
from emotion.webcam import timing_report


def benchmark(manifest_path, output, checkpoint=None, batch_size=64, batches=30):
    dataset = ManifestDataset(load_manifest(manifest_path), "train", augment=True)
    worker_results = {}
    for workers in (0, 2, 4):
        try:
            times = []
            for _ in range(2):
                started = time.perf_counter()
                count = 0
                for i, (images, _) in enumerate(loader(dataset, batch_size, workers)):
                    count += len(images)
                    if i + 1 >= batches:
                        break
                times.append(count / (time.perf_counter() - started))
            worker_results[str(workers)] = {"images_per_second": sum(times) / len(times)}
        except (RuntimeError, OSError) as error:
            worker_results[str(workers)] = {"error": str(error)}
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    model_results = {}
    for name in devices:
        device = select_device(name)
        model = load_checkpoint(checkpoint, device)[0] if checkpoint else make_model("cnn").to(device).eval()
        image = dataset[0][0].unsqueeze(0).to(device)
        samples = []
        with torch.inference_mode():
            for _ in range(10):
                model(image)
            synchronize(device)
            for _ in range(100):
                started = time.perf_counter()
                model(image)
                synchronize(device)
                samples.append({"total": time.perf_counter() - started})
        model_results[name] = timing_report(samples)
    valid = {k: v for k, v in worker_results.items() if "images_per_second" in v}
    report = {"workers": worker_results, "batch_one_model_only": model_results,
              "recommended_workers": int(max(valid, key=lambda k: valid[k]["images_per_second"])) if valid else 0,
              "recommended_webcam_device": max(model_results, key=lambda k: model_results[k]["updates_per_second"]),
              "note": "Model-only timings exclude camera, Haar detection, transfers, and display. Use webcam --timing-output for end-to-end measurement."}
    write_json(output, report)
    return report
