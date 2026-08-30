# YOLO-World Annotator 通用开源版设计

## 背景与目标

现有程序是面向单台 Windows + NVIDIA CUDA 机器制作的本地桌面工具。默认入口使用 `AnnotatorWindow`，直接实例化写死 `cuda:0` 的 `YOLOWorldDetector`；仓库同时保留一套未作为默认入口的高级推理窗口。项目还缺少标准 Python 包元数据、CPU 运行能力、可复现依赖、持续集成和开源治理文件。

本次改造的目标是把现有产品整理成可直接发布到 GitHub 的通用桌面项目：在默认 `auto` 模式下优先使用可用 CUDA GPU，没有 CUDA 时自动使用 CPU；允许用户显式选择 `cpu` 或指定 `cuda:N`；保留现有标注、原地 YOLO 标签存储、批处理和人工编辑行为；提供规范的安装、启动、测试、构建和贡献路径。

## 方案选择

评估过三个方案：

1. 只删除 CUDA 拒绝逻辑。改动最少，但仍保留顶层 `core/models/utils` 包名、个人路径、不可复现依赖和不可发布结构，不能满足开源化目标。
2. 迁移为 `src/yolo_world_annotator` 命名空间包，新增集中设备策略并渐进适配当前默认 GUI。该方案能建立稳定公共入口，同时复用现有功能和测试，风险可控。
3. 彻底合并 `AnnotatorWindow` 与 `MainWindow`、重写完整推理服务。长期架构最纯净，但会把 CPU 适配与两套产品链重写绑定在一起，范围和回归风险过大。

采用方案 2。高级 `MainWindow` 相关代码继续保留为内部实验能力，不把它提升为默认入口；默认入口仍是用户已经使用和验证过的 `AnnotatorWindow`。

## 代码结构

```text
src/yolo_world_annotator/
├── __init__.py          # 版本信息
├── __main__.py          # python -m 入口
├── cli.py               # 参数解析、日志和 GUI 启动
├── app/                 # Qt 窗口、画布、worker
├── core/                # 标注领域、数据集、导出、评估
├── inference/           # SAHI、裁剪和检测合并
├── models/              # 模型适配器和生命周期
└── utils/               # 配置、设备、图像和日志
```

测试继续放在 `tests/`。仓库根目录保留一个薄 `main.py` 兼容入口，但所有生产导入使用 `yolo_world_annotator.*` 完整命名。模型权重、日志和构建产物不进入 Git。

## 设备策略

新增不可变 `DeviceInfo` 与 `resolve_device(requested="auto")`：

- `auto`：`torch.cuda.is_available()` 为真时选择 `cuda:0`，否则选择 `cpu`。
- `cpu`：始终选择 CPU，即使存在 GPU。
- `cuda`：等价于 `cuda:0`。
- `cuda:N`：校验 CUDA 可用且索引位于 `[0, device_count)`；不满足时抛出包含修复建议的 `DeviceSelectionError`。
- 其他值：拒绝并列出合法格式。

`DeviceInfo` 提供 Ultralytics 使用的设备字符串、可读描述和 `use_half`。只有 CUDA 使用半精度；CPU 使用 FP32。设备选择来源优先级为 GUI 选择值，其次 CLI `--device` 初始值，最后 `YOLO_WORLD_DEVICE`，默认 `auto`。

设备在创建 detector 时解析并保持到该模型释放。显式请求 CUDA 不可用时不静默改用 CPU，因为这会违背用户的明确要求；只有 `auto` 会自动降级。CUDA 运行中显存不足也不改用 CPU，而是保留当前清晰错误，避免同一批任务中性能和结果语义突然变化。

## 推理和 GUI 数据流

`AnnotatorWindow` 在设置区提供“自动 / CPU / CUDA 0”选择。窗口启动时仅探测并展示设备，不提前加载模型。点击自动标注后，job 携带规范化设备请求；`InferenceEngine` 在模型路径、提示词或设备变化时重新构建 `YOLOWorldDetector`。detector 调用 Ultralytics 时传入 `device=<cpu|cuda:N>` 和 `half=<bool>`。

状态文本使用“设备”或实际设备描述，不再硬编码“GPU”。CPU 模式保持按钮可用，并明确提示推理会较慢。标签写入、批量跳过规则、取消语义和模型自动下载行为保持不变。

## 安装与依赖

项目要求 Python 3.10–3.12。`pyproject.toml` 声明所有直接运行依赖，包括 `torch`、`ultralytics`、`PySide6`、`opencv-python`、`numpy`、`Pillow` 与 `PyYAML`。可选依赖分为：

- `verification`：Transformers、Accelerate、SentencePiece，用于 SigLIP/VLM。
- `test`：pytest、pytest-cov。
- `dev`：测试依赖、Ruff、mypy、pre-commit。
- `build`：PyInstaller。

默认 PyPI 安装提供 CPU 可运行环境。需要 NVIDIA CUDA 的用户先按 PyTorch 官方说明安装与驱动匹配的 CUDA wheel，再安装本项目；满足版本约束时不会被替换。

## 跨平台与打包

源码运行支持 Windows、Linux 和 macOS；CUDA 仍只在 PyTorch 实际报告可用时启用。日志目录改用 `platformdirs` 风格的跨平台逻辑，但不新增依赖：Windows 使用 `LOCALAPPDATA`，macOS 使用 `~/Library/Logs`，其他平台使用 `XDG_STATE_HOME` 或 `~/.local/state`。

PyInstaller 仍只发布 Windows onedir 包，但 spec 中的 Windows DLL 收集必须有平台保护。发布包默认不捆绑大型权重；首次运行按 Ultralytics 行为下载，或由用户选择本地 `.pt`。构建脚本只验证应用结构，不要求本机存在特定个人 Conda 环境。

## 开源治理与许可

项目采用 `AGPL-3.0-or-later`。原因是核心 Ultralytics 软件与其默认模型按官方说明使用 AGPL-3.0；统一采用兼容的强 copyleft 许可能给贡献者和分发者最清晰的默认路径。README 同时说明：需要闭源或专有分发时，使用者必须自行评估并取得相应商业许可；第三方模型权重可能有额外条款。

仓库补齐 `LICENSE`、`CONTRIBUTING.md`、`CODE_OF_CONDUCT.md`、`SECURITY.md`、`CHANGELOG.md`、Issue/PR 模板、Dependabot 和 GitHub Actions。CI 在 Windows 与 Ubuntu、Python 3.10/3.11/3.12 上安装 CPU 依赖并运行测试、Ruff 和包构建；真实 CUDA 推理不作为公共 runner 的合并门槛。

## 测试和验收

设备解析与 detector 参数采用单元测试，使用真实业务对象和受控 fake，仅隔离模型下载/Ultralytics 推理。GUI 测试验证 CPU 模式不会禁用操作按钮，设备切换会进入 job。现有标注格式、数据集、编辑和批处理测试必须继续通过。

验收命令：

```powershell
python -m pytest -q
python -m ruff check .
python -m build
python -m yolo_world_annotator --help
python -m yolo_world_annotator --device cpu --version
```

CPU 真实推理 smoke test 使用已缓存的小型 YOLO-World 权重和一张本地图像；如果权重不可用，自动化测试不联网下载，而由发布检查清单记录手动验证结果。CUDA 真实推理在 NVIDIA 机器上作为附加验证。

## 非目标

- 本次不重写高级 SigLIP/VLM/SAHI 流水线，也不把它设为默认 GUI。
- 不在 Git 中提交模型权重、用户数据、日志或 PyInstaller 产物。
- 不承诺 CPU 与 CUDA 有相同性能，只保证功能可用和结果格式一致。
- 不自动发布到 PyPI 或 GitHub Release；仓库只准备好可审查的发布工作流和构建入口。
