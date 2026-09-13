"""Read progress from persisted reports without touching a running trainer."""
import json
from pathlib import Path


def status(run_dir):
    root = Path(run_dir)
    if not root.is_dir():
        raise ValueError(f"Run directory does not exist: {root}")
    configurations = [root / "config.json"] if (root / "config.json").exists() else sorted(root.glob("*/*/config.json"))
    if not configurations:
        print("No runs have started in this directory.")
        return
    for config_path in configurations:
        directory = config_path.parent
        config = json.loads(config_path.read_text())
        records = []
        for path in sorted((directory / "epochs").glob("*.json")):
            try:
                records.append(json.loads(path.read_text()))
            except json.JSONDecodeError:
                continue  # The trainer may currently be writing this epoch.
        name = str(directory.relative_to(root)) if directory != root else root.name
        if not records:
            print(f"{name}: first epoch in progress or not yet saved")
            continue
        latest = records[-1]
        best = max(records, key=lambda r: (r["validation"]["macro_f1"], -r["validation"]["loss"]))
        print(f"{name}: saved epoch {latest['epoch'] + 1}/{config['epochs']}; "
              f"best validation F1={best['validation']['macro_f1']:.4f}, "
              f"accuracy={best['validation']['accuracy']:.4f}; "
              f"last epoch {latest['seconds']:.1f}s")
    selection_path = root / "selection.json"
    if selection_path.exists():
        selected = json.loads(selection_path.read_text())["selected_configuration"]
        print(f"Training comparison finished; selected configuration: {selected}")
