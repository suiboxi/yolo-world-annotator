# YOLO-World Annotator

A local-first desktop application for reviewing images, generating YOLO labels with open-vocabulary YOLO-World models, and correcting boxes by hand.

The application automatically uses CUDA when PyTorch reports a working NVIDIA GPU and falls back to CPU otherwise. Images and labels stay on your machine; no cloud API is required.

[简体中文说明](docs/README.zh-CN.md)

## Highlights

- Automatic `CUDA → CPU` device selection, with explicit `cpu` and `cuda:N` overrides.
- Open-vocabulary prompts can differ from the final dataset class names.
- In-place, atomic YOLO label writes after inference and manual edits.
- Batch annotation that can preserve existing non-empty labels.
- Direct box creation, movement, resizing, class changes, and deletion.
- Unicode image paths and JPG, PNG, BMP, and WebP input.
- Source installation on Windows and Linux; a Windows PyInstaller build is included.

## Device behavior

| Request | Behavior |
| --- | --- |
| `auto` | Use `cuda:0` when CUDA is available; otherwise use CPU. |
| `cpu` | Force CPU and FP32 inference. |
| `cuda` | Require `cuda:0`; fail clearly if CUDA is unavailable. |
| `cuda:N` | Require the selected CUDA device index. |

CPU mode is compatible but can be much slower than CUDA, especially with large weights or large image sizes. CUDA uses FP16; CPU uses FP32.

## Requirements

- Python 3.10, 3.11, or 3.12
- Windows 10/11 or a Linux desktop supported by PySide6
- Enough memory for the selected YOLO-World model
- Optional: an NVIDIA GPU with a working CUDA-enabled PyTorch installation

macOS can run the source application in CPU mode, but it is not part of the automated CI matrix and no macOS bundle is currently produced.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The default PyPI dependency path is CPU-capable. For CUDA, install the PyTorch build matching your driver from the [official PyTorch installer](https://pytorch.org/get-started/locally/) before installing this project; a compatible existing CUDA build is preserved.

Optional SigLIP/VLM dependencies:

```bash
python -m pip install -e ".[verification]"
```

## Run

```bash
python -m yolo_world_annotator
```

Choose a device explicitly when needed:

```bash
python -m yolo_world_annotator --device cpu
python -m yolo_world_annotator --device cuda:1
```

On Windows, `启动标注器.bat` runs the installed package with automatic device selection.

## Models and offline use

The preset list contains YOLOv8 World V1/V2 S, M, L, and X weights. Missing official weights are downloaded by Ultralytics on first use. You can also select an existing local `.pt` YOLO-World weight.

Set a custom writable model directory with:

```powershell
$env:YOLO_WORLD_WEIGHTS_DIR = "D:\models\yolo-world"
```

In a source checkout, an existing `models/weights` directory is reused. Frozen Windows builds use `models/weights` next to the executable. Installed packages otherwise use the platform user-data directory. For offline use, put the weight file in that directory before starting inference.

Model weights are deliberately excluded from Git and binary builds. Review the license attached to every weight you download or distribute.

## Typical workflow

1. Open a folder containing images.
2. Select a preset or local YOLO-World weight.
3. Enter final dataset classes and matching English model prompts, one per line.
4. Annotate the current image or the entire folder.
5. Review boxes and correct them directly on the canvas.

The prompt and final class may differ. For example, a visual prompt such as `strawberry` may be mapped to a final class named `raspberry` when that produces better proposals for a particular dataset.

## Dataset layout

The default workflow writes labels next to images:

```text
dataset/
├── sample_001.jpg
├── sample_001.txt
├── sample_002.png
├── sample_002.txt
├── classes.txt
├── project.json
└── annotations.json
```

Each `.txt` uses standard normalized YOLO rows:

```text
class_id x_center y_center width height
```

`classes.txt` preserves class order, `project.json` stores settings, and `annotations.json` retains confidence and review metadata. Writes are atomic and no `labels/` directory is created by the default window.

## Development

```bash
python -m pip install -e ".[dev,build]"
python -m pytest -q
python -m ruff check .
python -m build
```

The test suite does not download model weights. Public CI uses CPU PyTorch on Windows and Ubuntu. Real CPU/CUDA inference is a release smoke test when local weights and hardware are available.

The production package follows a `src/` layout:

```text
src/yolo_world_annotator/
├── app/        # PySide6 windows, canvas, and workers
├── core/       # annotation and dataset domain logic
├── inference/  # tiling, crops, and merge helpers
├── models/     # detector and verifier adapters
└── utils/      # device, paths, logging, and configuration
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

## Windows executable

Install build dependencies and run:

```powershell
python -m pip install -e ".[build]"
python build_exe.py
```

The artifact is written to `dist/YOLOWorldAnnotator/`. Keep the whole directory together. Weights are not bundled; add them under `models/weights` beside the executable or let the application download a preset on first use.

## Privacy and security

Inference and label editing are local. The program only makes network requests when a selected model or optional verifier must be downloaded. Security issues should be reported privately as described in [SECURITY.md](SECURITY.md).

## License

This project is licensed under [GNU AGPL-3.0-or-later](LICENSE).

Ultralytics states that its open-source software and trained YOLO models use AGPL-3.0 by default. If you cannot comply with those terms—for example, for a proprietary distribution—obtain appropriate licensing and legal advice. This repository does not grant rights to third-party model weights beyond their own licenses.
