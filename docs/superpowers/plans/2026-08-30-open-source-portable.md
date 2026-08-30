# YOLO-World Annotator 通用开源版实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 CUDA-only 本地程序改造成自动优先 GPU、无 CUDA 自动使用 CPU，并具备标准 GitHub 开源工程结构的桌面应用。

**Architecture:** 生产代码迁移到 `src/yolo_world_annotator` 命名空间包。集中式设备解析层为模型和 GUI 提供同一契约，当前默认 `AnnotatorWindow` 渐进接入；高级推理链保留但不扩大范围。

**Tech Stack:** Python 3.10–3.12、PyTorch、Ultralytics YOLO-World、PySide6、OpenCV、pytest、Ruff、setuptools、PyInstaller、GitHub Actions。

**Spec:** `docs/superpowers/specs/2026-08-30-open-source-portable-design.md`

## Global Constraints

- 默认设备请求必须是 `auto`：CUDA 可用则 `cuda:0`，否则 `cpu`。
- 显式 `cuda`/`cuda:N` 不可用时必须报错，不得静默降级。
- CPU 必须使用 FP32；CUDA 才允许半精度。
- 保留原地 YOLO `.txt`、`classes.txt`、`project.json`、`annotations.json` 格式及批量保存语义。
- 生产代码统一从 `yolo_world_annotator.*` 导入。
- Git 不包含模型权重、日志、用户数据、构建目录或二进制发布物。
- 项目许可证为 `AGPL-3.0-or-later`。

---

### Task 1: 集中设备解析契约

**Files:**
- Create: `src/yolo_world_annotator/utils/device.py`
- Create: `tests/test_device.py`

**Interfaces:**
- Produces: `DeviceSelectionError(ValueError)`；`DeviceInfo(requested, torch_device, description, use_half)`；`resolve_device(requested: str | None = None) -> DeviceInfo`。

- [ ] **Step 1: 写失败测试**

测试使用 monkeypatch 控制 `torch.cuda.is_available/device_count/get_device_name/get_device_properties`，覆盖 `auto→cuda:0`、`auto→cpu`、显式 `cpu`、合法 `cuda:1`、CUDA 不可用、越界索引和非法请求。手工期望值分别固定为 `"cuda:0"`、`"cpu"` 与布尔 `use_half`，不复用实现函数生成期望。

- [ ] **Step 2: 运行并确认 RED**

Run: `python -m pytest tests/test_device.py -q`

Expected: collection fails with `ModuleNotFoundError: yolo_world_annotator` 或缺少 `resolve_device`。

- [ ] **Step 3: 写最小实现**

使用 `@dataclass(frozen=True, slots=True)` 定义结果；规范化空值和大小写；只接受 `auto/cpu/cuda/cuda:<non-negative int>`。CPU 描述为 `CPU / float32`；CUDA 描述包含名称、索引、显存 GiB 和 `float16`。

- [ ] **Step 4: 运行并确认 GREEN**

Run: `python -m pytest tests/test_device.py -q`

Expected: all device tests pass。

### Task 2: YOLO detector 接入 CPU/GPU

**Files:**
- Modify: `src/yolo_world_annotator/models/yolo_world.py`
- Replace: `tests/test_gpu_only.py` → `tests/test_yolo_world_device.py`

**Interfaces:**
- Consumes: `resolve_device(requested)`。
- Produces: `YOLOWorldDetector(model_path: Path, device: str | None = "auto")`；属性 `device_info`、`device`、`device_description`。

- [ ] **Step 1: 写失败测试**

为存在的临时权重文件注入 fake `YOLOWorld`，验证 CPU 构造成功，`predict()` 调用包含 `device="cpu"`、`half=False`；CUDA fake 验证 `device="cuda:0"`、`half=True`。增加显式 CUDA 不可用时抛 `DeviceSelectionError` 的测试。

- [ ] **Step 2: 运行并确认 RED**

Run: `python -m pytest tests/test_yolo_world_device.py -q`

Expected: 现有构造器拒绝 CPU，或不接受 `device` 参数。

- [ ] **Step 3: 写最小实现**

删除 GPU-only guard 和 `quantize=16`；构造时解析设备；Ultralytics predict 使用解析后的设备与 `half`。仅在 CUDA OOM 时清缓存并给出降低尺寸/换小模型的错误；CPU RuntimeError 原样传播。

- [ ] **Step 4: 运行并确认 GREEN**

Run: `python -m pytest tests/test_yolo_world_device.py -q`

Expected: all detector device tests pass。

### Task 3: 默认 GUI 接入设备选择

**Files:**
- Modify: `src/yolo_world_annotator/app/annotator_window.py`
- Modify: `tests/test_annotator_window.py`
- Modify: `tests/test_batch_stability.py`

**Interfaces:**
- Consumes: `resolve_device()`、`YOLOWorldDetector(..., device=...)`。
- Produces: inference job 字段 `device: str`；GUI `device_combo`；状态文本不硬编码 GPU。

- [ ] **Step 1: 写失败测试**

在 `torch.cuda.is_available=False` 下创建窗口，断言两个自动标注按钮保持可用、设备标签包含 `CPU`。验证 `_settings()` 或发出的 job 中 `device` 等于组合框 data。更新批处理 fake detector 使其接受 `device` 并提供 CPU 描述。

- [ ] **Step 2: 运行并确认 RED**

Run: `python -m pytest tests/test_annotator_window.py tests/test_batch_stability.py -q`

Expected: 当前窗口禁用 CPU 按钮，且 job 不含设备字段。

- [ ] **Step 3: 写最小实现**

新增自动/CPU/CUDA 0 三项；设备变更时刷新描述。`_start_inference` 不再拒绝 CPU，只在 `resolve_device` 对显式请求报错时显示错误。engine 将设备变化纳入模型重载条件。所有进度和关闭提示使用“推理任务”或 detector 实际描述。

- [ ] **Step 4: 运行并确认 GREEN**

Run: `python -m pytest tests/test_annotator_window.py tests/test_batch_stability.py -q`

Expected: selected GUI tests pass。

### Task 4: 命名空间包与公共 CLI

**Files:**
- Move: `app/`, `core/`, `inference/`, `models/`, `utils/` → `src/yolo_world_annotator/`
- Create: `src/yolo_world_annotator/__init__.py`
- Create: `src/yolo_world_annotator/__main__.py`
- Create: `src/yolo_world_annotator/cli.py`
- Modify: `main.py`
- Create: `pyproject.toml`
- Create: `tests/test_cli.py`
- Modify: all production and test imports.

**Interfaces:**
- Produces: `yolo_world_annotator.__version__ == "0.1.0"`；`cli.main(argv: Sequence[str] | None = None) -> int`；console script `yolo-world-annotator`。

- [ ] **Step 1: 写失败测试**

测试 `main(["--version"])` 输出 `0.1.0` 且返回 0；测试 `parse_args(["--device", "cpu"])` 得到 `cpu`；测试非法设备由 argparse 返回非零。测试不创建 QApplication。

- [ ] **Step 2: 运行并确认 RED**

Run: `python -m pytest tests/test_cli.py -q`

Expected: package/CLI 尚不存在。

- [ ] **Step 3: 迁移和实现**

使用 setuptools `package-dir = {"" = "src"}` 与自动包发现；所有导入改为完整命名。CLI 支持 `--device`、`--version`，将选择值写入 `YOLO_WORLD_DEVICE` 后延迟导入 Qt。根 `main.py` 只调用 `yolo_world_annotator.cli.main()`。

- [ ] **Step 4: 验证包入口**

Run: `python -m pytest tests/test_cli.py -q`

Run: `python -m pip install -e . --no-deps`

Run: `python -m yolo_world_annotator --version`

Expected: tests pass，安装成功，输出 `0.1.0`。

### Task 5: 依赖、启动和 Windows 构建

**Files:**
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`
- Modify: `启动GPU标注器.bat` → rename `启动标注器.bat`
- Modify: `一键生成EXE.bat`
- Modify: `build_exe.py`
- Modify: `YOLOWorldAnnotator.spec`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `python -m yolo_world_annotator`。
- Produces: 无个人绝对路径的启动/构建命令；Windows onedir artifact。

- [ ] **Step 1: 更新声明**

`requirements.txt` 攓为兼容入口 `-e .`；`requirements-dev.txt` 使用 `-e .[dev,build]`。BAT 使用当前 PATH 中的 `python`，失败时提示创建虚拟环境。spec 指向 namespaced entry，Windows DLL 收集仅在 `sys.platform == "win32"` 执行，不强制权重存在。

- [ ] **Step 2: 执行静态构建检查**

Run: `python -m compileall -q src build_exe.py main.py`

Run: `python build_exe.py --check-only`

Expected: exit 0，且不访问个人 Conda 路径或要求已有模型权重。

### Task 6: 测试稳定性与项目质量配置

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_responsive_controls.py`
- Modify: other tests only where namespaced imports require it.
- Modify: `pyproject.toml`
- Create: `.pre-commit-config.yaml`

**Interfaces:**
- Produces: 稳定 session QApplication 生命周期；pytest marker `gpu` 与 `integration`；Ruff configuration。

- [ ] **Step 1: 稳定复现现有 Qt 崩溃**

Run the full suite twice with `QT_QPA_PLATFORM=offscreen` and record whether the access violation only appears after prior GUI tests。用逐步扩大测试集合定位留下窗口/线程的测试，确认对象生命周期根因后再改 fixture 或测试清理。

- [ ] **Step 2: 添加能捕获泄漏的回归断言**

在相关 GUI 测试结束时显式关闭窗口并等待 worker thread，随后处理事件；断言线程 `isRunning()` 为 false。先运行现有实现并确认断言失败或原生崩溃可复现。

- [ ] **Step 3: 修复根因并验证**

确保每个 `AnnotatorWindow` 都调用其 `closeEvent` 完成 thread quit/wait；fixture 只在 session 结束处理残余 top-level widgets，不在 synthetic event 后调用已释放对象。运行全套两次，均须 exit 0。

- [ ] **Step 4: 质量检查**

Run: `python -m ruff check .`

Run: `python -m pytest -q`

Expected: zero lint errors and all tests pass。

### Task 7: GitHub 开源文件与 CI

**Files:**
- Modify: `README.md`
- Create: `LICENSE`
- Create: `CONTRIBUTING.md`
- Create: `CODE_OF_CONDUCT.md`
- Create: `SECURITY.md`
- Create: `CHANGELOG.md`
- Create: `.github/workflows/ci.yml`
- Create: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Create: `.github/ISSUE_TEMPLATE/feature_request.yml`
- Create: `.github/pull_request_template.md`
- Create: `.github/dependabot.yml`

**Interfaces:**
- Produces: GitHub 可直接展示、贡献、报告漏洞和执行 CI 的仓库。

- [ ] **Step 1: 写文档和治理文件**

README 包含功能、截图占位目录说明、CPU/GPU 安装矩阵、源码启动、数据格式、测试、构建、许可和模型下载。不得含本机路径、硬件承诺或手写测试数量。SECURITY 使用私密 GitHub Security Advisory 报告流程。

- [ ] **Step 2: 配置 CI**

Ubuntu/Windows 上的 Python 3.10/3.11/3.12 矩阵安装 CPU 依赖并运行 pytest；单独 Ubuntu 3.12 job 运行 Ruff 与 `python -m build`。使用 pip 缓存，不下载权重，不要求 CUDA。

- [ ] **Step 3: 验证仓库卫生**

Run: `git grep -n -E 'C:\\\\Users\\\\ROG|cuda:0.*不会|CPU fallback is disabled' -- . ':!docs/superpowers/**'`

Expected: no matches。

Run: `git status --short --ignored`

Expected: weights、logs、build、dist 均 ignored，源代码和开源文件 tracked/untracked visible。

### Task 8: 最终集成验证与审查

**Files:**
- Review: complete branch diff against this plan and spec.

**Interfaces:**
- Consumes: all preceding tasks。
- Produces: 可交付的 GitHub 开源候选版本。

- [ ] **Step 1: 全量验证**

Run: `python -m pytest -q`

Run: `python -m ruff check .`

Run: `python -m build`

Run: `python -m yolo_world_annotator --device cpu --version`

Run: `git diff --check`

Expected: all commands exit 0。

- [ ] **Step 2: CPU smoke**

使用已缓存 `yolov8s-worldv2.pt` 和本地测试图片运行一次 `YOLOWorldDetector(..., device="cpu")`，设置单一提示词并断言返回 `list[BoundingBox]`；不把图片或输出提交到 Git。

- [ ] **Step 3: CUDA smoke**

在当前 CUDA 环境对同一最小样本运行 `device="auto"`，断言解析为 `cuda:0` 且推理返回列表。显存不足时用 S 权重与 640 尺寸重试一次，不改生产代码。

- [ ] **Step 4: 独立代码审查**

审查者按 spec、计划、`git diff` 和验证输出检查 Critical/Important 问题。所有 Critical/Important 必须修复并重新运行受影响测试与完整验证。
