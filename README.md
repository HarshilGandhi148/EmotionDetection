# Emotion Detection

A PyTorch facial-expression classifier trained **from scratch**, with a corrected
baseline, a compact CNN, reproducible splits, per-emotion metrics, and live face
cropping. The classes are anger, fear, happiness, neutral, sadness, and surprise.
No pretrained convolutional network is used. OpenCV's pretrained non-neural Haar
cascade is used only to locate the face.

## Install

Python 3.11 or newer is required. On Apple Silicon, use a native ARM Python.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

All commands also work as `python -m emotion ...` or
`python emotionRecognition.py ...`. Importing modules does not open a camera or
load datasets. Training chooses MPS on supported Macs, then CUDA, then CPU.
Webcam inference defaults to CPU; benchmark CPU/MPS for your machine.
`requirements.lock.txt` records the verified Mac/Python 3.13 dependency versions;
use `python -m pip install -r requirements.lock.txt -e .` to reproduce that environment.

## Prepare the data

Original source: [Facial Recognition Dataset on Kaggle](https://www.kaggle.com/datasets/apollo2506/facial-recognition-dataset/data).
Download and extract it locally. Both class directories (`train/happy/a.png`) and
the old filenames (`Training/happy-1.png`) are supported. Paths are explicit;
substitute the actual extracted training and testing directories below.

```sh
emotion prepare-data --train-root Training --test-root Testing --output data/manifest.json
```

The audit checks labels, readable images, image shapes, decoded-image duplicates,
and test overlap. It writes `data/manifest.audit.json` even when the audit fails.
Unknown labels, conflicting duplicate labels, and train/test overlap stop
preparation. They must be investigated; files are never silently deleted.
After reviewing the audit, `--clean-training` excludes from the working manifest
all training images that overlap the test partition or belong to a conflicting-label
duplicate group. Source files and every original test image are preserved. Exclusions
are listed in the audit. Any conflicting labels within the preserved test partition
are also reported in the manifest and final evaluation as a test-data limitation.
Non-image files are listed as ignored. An additional class can be explicitly
excluded with `--exclude-label disgust`; exclusions are recorded in the audit.

Validation is approximately 15% of training, stratified by emotion at the
duplicate-group level. The supplied test partition is preserved. Optional
`--identities identities.json` accepts a mapping of **every absolute image path**
to a subject ID and groups subjects before splitting (one of seven stratified
group folds). Without identity metadata, subject independence cannot be assured.
Exact decoded duplicates are checked; near-duplicates and label quality still
require visual review. An existing manifest is never overwritten.

The manifest stores fixed splits, counts, absolute paths, and training-only pixel
mean/std. All inputs share grayscale conversion, bilinear 48×48 resizing, and
normalization. Only training adds flips, affine transforms, and brightness/contrast
variation. Keep source images unchanged after preparation; checkpoints record the
manifest digest. Relocating the dataset requires a new manifest and new run.

## Train and compare

```sh
emotion benchmark --manifest data/manifest.json --output runs/preflight-benchmark.json
emotion train --manifest data/manifest.json --run-dir runs/cnn-42 --device mps
```

The preflight benchmark does not train a model. Add `--workers N` to training
using the measured worker recommendation; the portable default is zero.

Defaults: batch size 64, AdamW at 0.001, weight decay 0.0001, cosine learning-rate
decay, up to 80 epochs, and early stopping after 12 epochs without improvement.
Model selection uses validation macro-F1, with lower validation loss breaking ties.
`--architecture baseline` uses the original convolution widths with its corrected
six-class head. The default CNN uses two Conv/BatchNorm/ReLU layers per block,
widths 32/64/128, max pooling, global average pooling, dropout 0.3, and six logits.
`--weighted` uses inverse-frequency class weights computed from training only.

Training writes `best.pt`, `latest.pt`, `config.json`, per-epoch JSON reports, and
TensorBoard events. Checkpoints contain class order, preprocessing, architecture
version, optimizer/scheduler, and RNG states. Resume an interrupted run in the
same directory, with the original options and original total epoch budget:

```sh
emotion train --manifest data/manifest.json --run-dir runs/cnn-42 --device mps --resume runs/cnn-42/latest.pt
tensorboard --logdir runs
```

For the planned six-run experiment (three configurations × seeds 42 and 43):

```sh
emotion experiments --manifest data/manifest.json --output runs/comparison --device mps
emotion status --run-dir runs/comparison
```

This trains baseline, CNN, and weighted CNN, then freezes the validation-based
choice in `selection.json`. It never uses test scores for tuning. Add `--resume`
to the same experiment command to continue an interrupted suite with its original
settings. Runtime depends
on dataset size; no accuracy or duration is guaranteed. CPU/MPS kernels may not
be bitwise identical across hardware/library versions.

## Metrics and final evaluation

```sh
emotion evaluate --manifest data/manifest.json --checkpoint runs/cnn-42/best.pt --output runs/cnn-42/test.json
emotion evaluate-selection --manifest data/manifest.json --selection runs/comparison/selection.json --output runs/final-test
```

Use the first command for a single frozen model, or the second for the corrected
baseline versus selected configuration across both seeds. Freeze all choices
before final testing. The comparison reports means, population standard deviations,
and individual seed scores; two seeds are not a confidence interval.

Every evaluation reports accuracy, macro-F1, balanced accuracy (macro recall),
macro sensitivity/specificity, and per-class support, precision, F1, TP/TN/FP/FN,
sensitivity, and specificity. Confusion-matrix rows are true classes; columns are
predictions, in the fixed class order above. Each emotion is evaluated versus
the other five:

- Sensitivity/recall = `TP / (TP + FN)`.
- Specificity = `TN / (TN + FP)`.
- Undefined ratios are JSON `null`, emit a warning, and are excluded from macro
  averages. The data preparation step requires all classes in every split.

Epoch training/validation metrics use deterministic preprocessing and unweighted
cross-entropy for comparable losses, even for weighted training. The separate
`augmented_optimization_loss` describes the training objective on augmented batches.
Specificity can appear high in multiclass problems; always inspect sensitivity,
macro-F1, and the full confusion matrix alongside it.

## Benchmark and webcam

```sh
emotion benchmark --manifest data/manifest.json --checkpoint runs/cnn-42/best.pt --output runs/benchmark.json
emotion webcam --checkpoint runs/cnn-42/best.pt --device cpu --timing-output runs/webcam-cpu.json
emotion webcam --checkpoint runs/cnn-42/best.pt --device mps --timing-output runs/webcam-mps.json
```

The benchmark compares loader workers 0/2/4 and warmed-up batch-one CPU/MPS model
timings. Use its recommended worker count for subsequent training runs. Model-only
speed does **not** establish webcam performance; the live report includes capture,
Haar detection, preprocessing, model, and display p50/p95 latency and update rate.
The acceptance target is 15 updates/second with one face at 640×480. Verify this
before deployment; selection based on validation is provisional until speed passes.

The largest detected face is cropped and classified once per frame. No detection
shows “No face detected.” The displayed softmax percentage is a model score, not
a calibrated probability of someone's internal emotion. Press Escape or Q to exit.
Use `--camera 1` for a different camera or `--max-frames 300` for a bounded check.
Allow camera access in macOS when prompted. The camera is released on errors and
exit. Video is not recorded. Labeled webcam data is needed to quantify webcam
accuracy separately from dataset test accuracy.

## Repository changes and limitations

The original code returned 128 outputs, skipped its dropout/final layer, evaluated
random weights before loading, and classified whole frames twice. These paths have
been replaced. `saved_model.pth.tar` is preserved as a legacy artifact but cannot
load into the new architecture; loading it produces an explicit compatibility error.
No improved accuracy should be claimed until real-data training and final evaluation
have completed. Tests cover metrics, preprocessing, leakage, gradients, tiny-batch
learning, checkpoint loading, and mocked camera cleanup; mocks do not validate a
physical camera or Haar detection quality.
