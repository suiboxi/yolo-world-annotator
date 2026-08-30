# YOLO-World Annotator 中文说明

这是一个本地优先的 YOLO-World 桌面自动标注工具。有可用 NVIDIA CUDA 时默认使用 `cuda:0`，没有 CUDA 时自动改用 CPU；也可以通过 `--device cpu` 或 `--device cuda:N` 显式指定。

## 安装

要求 Python 3.10–3.12。建议使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m yolo_world_annotator
```

需要 CUDA 时，请先按 [PyTorch 官方安装器](https://pytorch.org/get-started/locally/)安装和驱动匹配的 CUDA 版 PyTorch，再安装本项目。普通 PyPI 安装可直接使用 CPU。

可选 SigLIP/VLM 依赖：

```powershell
python -m pip install -e ".[verification]"
```

## 使用

1. 打开包含 JPG/PNG/BMP/WebP 的图片目录。
2. 选择官方预设或本机 YOLO-World `.pt` 权重。
3. 分别填写最终标签类别与英文模型提示词，两区逐行对应。
4. 自动标注当前图片或整个目录。
5. 在画布上拖动、缩放、新建、改类或删除标注框。

程序默认把标准 YOLO `.txt` 与图片放在一起，并保存 `classes.txt`、`project.json` 和 `annotations.json`。每次推理或人工修改都会原子写入，批量任务中途取消不会丢失已完成结果。

## 模型目录

官方权重缺失时会由 Ultralytics 首次下载。源码仓库会复用 `models/weights`；Windows 冻结包使用 EXE 旁的 `models/weights`；常规安装使用系统用户数据目录。可用环境变量覆盖：

```powershell
$env:YOLO_WORLD_WEIGHTS_DIR = "D:\models\yolo-world"
```

模型权重、日志、用户数据和构建产物都不会提交到 Git。

## 开发

```powershell
python -m pip install -e ".[dev,build]"
python -m pytest -q
python -m ruff check .
python -m build
```

项目采用 AGPL-3.0-or-later。Ultralytics 软件和默认模型可能带来额外的 AGPL 或商业许可义务，分发前请阅读官方许可说明和具体权重的条款。
