"""Select on validation only, then explicitly evaluate the frozen selection."""
import json
from pathlib import Path

import numpy as np

from emotion.data import write_json
from emotion.training import evaluate, load_checkpoint, train


def experiments(manifest, output, epochs=80, patience=12, batch_size=64, workers=0, device_name="auto", resume=False):
    output = Path(output)
    if (output / "selection.json").exists():
        raise ValueError("This experiment is already selected; choose a new output directory.")
    results = {}
    for name, architecture, weighted in (("baseline", "baseline", False), ("cnn", "cnn", False), ("cnn_weighted", "cnn", True)):
        results[name] = []
        for seed in (42, 43):
            directory = output / name / str(seed)
            latest = directory / "latest.pt"
            checkpoint = train(manifest, directory, architecture=architecture, weighted=weighted, seed=seed,
                               epochs=epochs, patience=patience, batch_size=batch_size,
                               workers=workers, device_name=device_name,
                               resume=str(latest) if resume and latest.exists() else None)
            _, metadata = load_checkpoint(checkpoint)
            results[name].append({"seed": seed, "checkpoint": str(checkpoint.resolve()),
                                  "validation_macro_f1": metadata["validation"]["macro_f1"],
                                  "validation_loss": metadata["validation"]["loss"]})
        write_json(output / "validation_comparison.json", results)
    chosen = max(results, key=lambda name: (
        np.mean([r["validation_macro_f1"] for r in results[name]]),
        -np.mean([r["validation_loss"] for r in results[name]])))
    selection = {"selected_configuration": chosen, "runs": results,
                 "criterion": "mean validation macro-F1; lower mean validation loss breaks ties",
                 "live_speed_status": "unverified: benchmark webcam before accepting deployment",
                 "seeds": [42, 43]}
    write_json(output / "selection.json", selection)
    return selection


def evaluate_selection(manifest, selection, output, device_name="auto"):
    selection = json.loads(Path(selection).read_text())
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory for the final test report.")
    results = {}

    def summary(values):
        defined = [value for value in values if value is not None]
        return {"mean": float(np.mean(defined)) if defined else None,
                "std": float(np.std(defined)) if defined else None, "seeds": values}

    for name in dict.fromkeys(("baseline", selection["selected_configuration"])):
        reports = [evaluate(manifest, run["checkpoint"], output / f"{name}_{run['seed']}.json",
                            device_name=device_name) for run in selection["runs"][name]]
        result = {metric: summary([r[metric] for r in reports]) for metric in
                  ("accuracy", "macro_f1", "macro_sensitivity", "macro_specificity")}
        result["per_class"] = {label: {
            metric: summary([r["per_class"][label][metric] for r in reports])
            for metric in ("sensitivity", "specificity", "precision", "f1")}
            for label in reports[0]["per_class"]}
        results[name] = result
    write_json(output / "comparison.json", results)
    return results
