"""Command-line interface; importing it does not initialize training or a camera."""
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a six-emotion CNN from scratch.")
    commands = parser.add_subparsers(dest="command", required=True)
    status_parser = commands.add_parser("status", help="Show progress saved by a training run or experiment suite")
    status_parser.add_argument("--run-dir", required=True)
    prepare = commands.add_parser("prepare-data", help="Audit images and create a fixed split manifest")
    prepare.add_argument("--train-root", required=True)
    prepare.add_argument("--test-root", required=True)
    prepare.add_argument("--output", default="data/manifest.json")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--exclude-label", action="append", default=[], help="Explicit out-of-scope label to exclude and record")
    prepare.add_argument("--identities", help="JSON mapping absolute image paths to subject IDs")
    prepare.add_argument("--clean-training", action="store_true",
                         help="Exclude overlapping/conflicting training images from the manifest, preserving source files and the test partition")
    for name in ("train", "experiments"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", required=True)
        command.add_argument("--run-dir" if name == "train" else "--output", required=True)
        command.add_argument("--epochs", type=int, default=80)
        command.add_argument("--patience", type=int, default=12)
        command.add_argument("--batch-size", type=int, default=64)
        command.add_argument("--workers", type=int, default=0)
        command.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
        if name == "train":
            command.add_argument("--architecture", choices=("baseline", "cnn"), default="cnn")
            command.add_argument("--weighted", action="store_true")
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--resume", help="Original run's latest.pt; keep configuration unchanged")
        else:
            command.add_argument("--resume", action="store_true", help="Continue interrupted runs with their original configuration")
    for name in ("evaluate", "evaluate-selection"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", required=True)
        command.add_argument("--checkpoint" if name == "evaluate" else "--selection", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
        if name == "evaluate":
            command.add_argument("--split", choices=("train", "validation", "test"), default="test")
    live = commands.add_parser("webcam")
    live.add_argument("--checkpoint", required=True)
    live.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    live.add_argument("--camera", type=int, default=0)
    live.add_argument("--timing-output")
    live.add_argument("--max-frames", type=int, default=0)
    benchmark_parser = commands.add_parser("benchmark")
    benchmark_parser.add_argument("--manifest", required=True)
    benchmark_parser.add_argument("--checkpoint", help="Omit for pre-training timing with randomly initialized CNN weights")
    benchmark_parser.add_argument("--output", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if "device" in args:
        args["device_name"] = args.pop("device")
    try:
        if command == "status":
            from emotion.status import status
            status(**args)
        elif command == "prepare-data":
            from emotion.data import prepare_data
            args["excluded_labels"] = args.pop("exclude_label")
            prepare_data(**args)
        elif command == "train":
            from emotion.training import train
            args["manifest_path"] = args.pop("manifest")
            train(**args)
        elif command == "evaluate":
            from emotion.training import evaluate
            args["manifest_path"] = args.pop("manifest")
            args["checkpoint_path"] = args.pop("checkpoint")
            evaluate(**args)
        elif command == "webcam":
            from emotion.webcam import webcam
            webcam(**args)
        elif command == "benchmark":
            from emotion.benchmark import benchmark
            args["manifest_path"] = args.pop("manifest")
            benchmark(**args)
        elif command == "experiments":
            from emotion.experiments import experiments
            experiments(**args)
        elif command == "evaluate-selection":
            from emotion.experiments import evaluate_selection
            evaluate_selection(**args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
