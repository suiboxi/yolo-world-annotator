# YOLO-World 数据集自动标注器

[![版本](https://img.shields.io/badge/版本-v0.0.1-2ea44f)](https://github.com/suiboxi/yolo-world-annotator/tree/v0.0.1)
[![持续集成](https://github.com/suiboxi/yolo-world-annotator/actions/workflows/ci.yml/badge.svg)](https://github.com/suiboxi/yolo-world-annotator/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776ab)](https://www.python.org/)
[![许可证](https://img.shields.io/badge/许可证-AGPL--3.0--or--later-blue)](LICENSE)

一个本地运行的 YOLO-World 桌面数据集自动标注工具。程序可以根据自然语言类别生成 YOLO 检测框，支持人工新建、移动、缩放、改类和删除标注；有可用的 NVIDIA CUDA 显卡时优先使用显卡，没有 CUDA 时自动使用中央处理器。

当前版本：`0.0.1`。这是首个公开的早期版本，建议先复制一小批图片验证模型效果和数据流程，再用于正式数据集。

## 目录

- [开始前必须了解的事项](#开始前必须了解的事项)
- [主要功能](#主要功能)
- [运行环境与硬件建议](#运行环境与硬件建议)
- [安装](#安装)
- [启动程序](#启动程序)
- [设备选择规则](#设备选择规则)
- [模型与权重文件](#模型与权重文件)
- [第一次使用](#第一次使用)
- [类别与提示词](#类别与提示词)
- [基础推理参数](#基础推理参数)
- [人工检查与编辑](#人工检查与编辑)
- [批量自动标注](#批量自动标注)
- [项目目录与标签格式](#项目目录与标签格式)
- [导出标准 YOLO 数据集](#导出标准-yolo-数据集)
- [高级推理功能](#高级推理功能)
- [隐私与安全](#隐私与安全)
- [常见问题](#常见问题)
- [开发、测试与打包](#开发测试与打包)
- [版本、贡献与许可证](#版本贡献与许可证)

## 开始前必须了解的事项

> [!IMPORTANT]
> 默认工作模式会把每张图片对应的 `.txt` 标签直接保存在图片旁边。批量任务中的“覆盖所有标签”会替换已有标注，正式处理前请备份数据集。

> [!WARNING]
> `classes.txt` 的行号就是永久的类别编号。数据集产生标签以后，不要删除、交换或重命名旧类别，否则已有标签的含义会改变。

> [!WARNING]
> PyTorch 模型权重通常使用 Python 序列化机制。只应加载可信来源的 `.pt` 文件，不要运行陌生人提供的权重。

还需要注意：

- 仓库不包含大型模型权重。第一次使用某个模型时通常需要联网下载，下载时间取决于网络环境。
- 自动标注只是初始结果，不等同于人工审核完成。用于训练前应逐图检查误检、漏检和类别错误。
- 中央处理器可以运行基础 YOLO-World 流程，但速度通常明显慢于 CUDA 显卡，批量处理大量高分辨率图片时尤其明显。
- 显式选择 `cuda` 或 `cuda:N` 时，如果 CUDA 不可用，程序会报错，不会悄悄改用中央处理器；选择 `auto` 才会自动回退。
- SigLIP、VLM 和切片推理属于可选高级功能，默认关闭。它们会增加下载量、显存占用和处理时间。
- 本项目目前主要验证 Windows 和 Linux。macOS 可尝试中央处理器模式，但尚未纳入持续集成，也没有提供打包程序。

## 主要功能

- YOLO-World 开放词汇检测，可按自定义类别自动生成 YOLO 标注。
- 自动设备选择：CUDA 可用时使用第 0 张显卡，否则使用中央处理器。
- 支持强制使用 `cpu`、`cuda` 或指定显卡 `cuda:N`。
- 支持单张图片和整个文件夹的自动标注。
- 支持创建、移动、缩放、改类和删除检测框。
- 支持空标签，便于明确记录“已检查但没有目标”的图片。
- 每张图片处理完成后立即保存，取消批量任务不会丢失已完成结果。
- 支持中文路径和中文文件名。
- 支持导出带 `images/train`、`images/val`、`labels/train`、`labels/val` 和 `data.yaml` 的标准 YOLO 数据集。
- 可选 SAHI 风格切片推理、SigLIP2 二次验证和 Qwen3-VL 困难样本复核。
- 提供数据集统计、评估和多流水线基准比较功能。

## 运行环境与硬件建议

| 项目 | 要求或建议 |
| --- | --- |
| Python | `3.10`、`3.11` 或 `3.12`；不支持 `3.13` |
| 操作系统 | Windows 10/11 或常见 Linux 桌面发行版 |
| 显卡 | 可选；推荐支持 CUDA 的 NVIDIA 显卡 |
| 中央处理器 | 可以运行基础功能，建议先用 S 模型和 `640` 输入尺寸 |
| 内存 | 基础流程建议至少 8 GB；启用高级模型时需要更多 |
| 显存 | S 模型压力较低；M/L/X、较大输入尺寸、切片推理和验证模型会逐步增加占用 |
| 磁盘 | 除代码外，还需要为 Python 环境、PyTorch 和模型缓存预留数 GB 空间 |

模型选择建议：

- 第一次使用、中央处理器或显存有限：从 `yolov8s-worldv2.pt` 开始。
- 速度和检测能力平衡：尝试 M 模型。
- 小目标较多且硬件充足：尝试 L/X 模型，或开启切片推理。
- 8 GB 显存设备出现显存不足时：先降低输入尺寸，再减小模型，最后关闭 SigLIP/VLM。

## 安装

### 1. 获取代码

```powershell
git clone https://github.com/suiboxi/yolo-world-annotator.git
cd yolo-world-annotator
```

也可以在 GitHub Desktop 中选择“克隆仓库”，然后进入克隆后的目录。

### 2. 创建独立 Python 环境

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

如果 PowerShell 阻止激活脚本，可以仅对当前用户允许本地脚本：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Linux：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. 安装 PyTorch 和项目

仅使用中央处理器时，建议先安装官方中央处理器版本：

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3"
python -m pip install -e .
```

使用 NVIDIA 显卡时，应先根据显卡驱动和 CUDA 环境，在 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 选择对应命令安装 PyTorch，然后安装本项目：

```powershell
python -m pip install -e .
```

不要仅以电脑安装了 CUDA 工具包作为判断依据；最终应以 `torch.cuda.is_available()` 为准。

### 4. 可选高级依赖

只有需要 SigLIP2 或 VLM 功能时才安装：

```powershell
python -m pip install -e ".[verification]"
```

开发、测试和构建依赖：

```powershell
python -m pip install -e ".[dev,build]"
```

### 5. 检查安装结果

```powershell
yolo-world-annotator --version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available()); print('显卡数量:', torch.cuda.device_count())"
```

版本命令应输出：

```text
yolo-world-annotator 0.0.1
```

## 启动程序

自动选择设备：

```powershell
yolo-world-annotator
```

也可以使用模块入口：

```powershell
python -m yolo_world_annotator
```

在源码目录中还可以双击 `启动标注器.bat`。该脚本使用当前命令行可找到的 Python，因此请先正确安装项目；如果使用虚拟环境，最稳妥的方式仍是在已激活环境的终端中执行启动命令。

强制使用中央处理器：

```powershell
yolo-world-annotator --device cpu
```

强制使用第一张 CUDA 显卡：

```powershell
yolo-world-annotator --device cuda:0
```

使用第二张 CUDA 显卡：

```powershell
yolo-world-annotator --device cuda:1
```

## 设备选择规则

| 设置 | 行为 |
| --- | --- |
| `auto` | 检测到 CUDA 时使用 `cuda:0`，否则使用 `cpu` |
| `cpu` | 始终使用中央处理器和 FP32 |
| `cuda` | 使用默认 CUDA 显卡；CUDA 不可用时直接报错 |
| `cuda:N` | 使用编号为 `N` 的显卡；编号不存在时直接报错 |

CUDA 推理默认使用半精度以降低显存占用；中央处理器使用 FP32。程序界面中的设备选择与命令行规则一致。

也可以通过环境变量设置默认设备。Windows PowerShell 示例：

```powershell
$env:YOLO_WORLD_DEVICE = "cpu"
yolo-world-annotator
```

优先级为：界面中实际选择的设备、命令行 `--device`、环境变量 `YOLO_WORLD_DEVICE`、默认值 `auto`。

## 模型与权重文件

### 内置模型选项

界面提供以下 Ultralytics YOLO-World 权重：

| 系列 | 可选规模 | 说明 |
| --- | --- | --- |
| YOLO-World V2 | S、M、L、X | 默认推荐 V2；默认模型为 `yolov8s-worldv2.pt` |
| YOLO-World V1 | S、M、L、X | 主要用于兼容旧项目 |

模型规模从 S 到 X 逐步增大。更大的模型通常更慢、占用更多内存和显存，不保证对每个数据集都更准确，应以人工抽检结果为准。

### 下载和保存位置

程序会优先使用本地权重；缺少权重时，Ultralytics 通常会在首次加载时联网下载。请保持网络连接并耐心等待，下载过程中不要反复点击“加载模型”。

权重目录按以下顺序决定：

1. 环境变量 `YOLO_WORLD_WEIGHTS_DIR` 指定的目录。
2. Windows 打包版程序旁的 `models/weights`。
3. 源码仓库中已经存在的 `models/weights`。
4. 当前用户的数据目录：Windows 默认为 `%LOCALAPPDATA%\yolo-world-annotator\weights`，Linux 默认为 `~/.local/share/yolo-world-annotator/weights`。

自定义目录示例：

```powershell
$env:YOLO_WORLD_WEIGHTS_DIR = "D:\models\yolo-world"
yolo-world-annotator
```

离线电脑应提前在联网电脑下载可信权重，再复制到上述目录。不要把大型权重直接提交到本仓库。

## 第一次使用

建议严格按以下顺序操作：

1. 复制少量代表性图片到测试目录，并备份原始数据。
2. 启动程序，打开包含图片的文件夹。
3. 选择设备和模型；不确定时保持 `Auto` 与 S 模型。
4. 设置检测类别，确认每行一个类别且顺序正确。
5. 加载模型，等待设备状态显示就绪。
6. 保持默认参数，先对当前图片执行自动标注。
7. 检查检测框，手动修正误检、漏检和类别。
8. 保存当前图片，确认图片旁出现同名 `.txt` 文件。
9. 用几张不同场景的图片重复验证，调整类别描述和置信度。
10. 确认结果可靠后，再启动批量自动标注。

程序只扫描所选目录当前层级中的 `.jpg`、`.jpeg`、`.png`、`.bmp` 和 `.webp` 文件，不递归扫描子目录。图片按文件名稳定排序。

## 类别与提示词

类别编辑框中每行填写一个目标类别，例如：

```text
football player
football
referee
goalkeeper
```

对应关系为：

| 行号 | 类别编号 | 类别名称 |
| --- | --- | --- |
| 第 1 行 | `0` | `football player` |
| 第 2 行 | `1` | `football` |
| 第 3 行 | `2` | `referee` |
| 第 4 行 | `3` | `goalkeeper` |

使用建议：

- 使用清晰、具体、可视觉识别的名词短语，通常比宽泛概念更稳定。
- 先用少量图片测试中文和英文描述的效果；开放词汇模型在不同领域中的最佳措辞可能不同。
- 不要写空行、重复类别或含义几乎相同的多个类别，避免标签难以区分。
- 产生标签后只能在末尾追加新类别。不要改变旧类别的位置和名称。
- `classes.txt` 是类别编号的权威来源，移动数据集时必须连同它一起移动。

## 基础推理参数

| 参数 | 默认值 | 作用 | 调整建议 |
| --- | ---: | --- | --- |
| 置信度 | `0.25` | 过滤低分检测框 | 漏检多时适当降低；误检多时适当提高 |
| 重叠阈值 | `0.45` | 控制重复框抑制 | 相邻目标被错误合并时可提高；重复框多时可降低 |
| 输入尺寸 | `640` | 模型推理时的缩放尺寸 | 小目标多时可提高，但显存和耗时会增加 |
| 审核阈值 | `0.50` | 决定结果是否进入人工复核 | 质量优先时可适当提高 |
| 训练集比例 | `0.80` | 导出时分配到训练集的比例 | 其余样本进入验证集 |

第一次使用建议保持默认值，每次只改变一个参数并抽检相同图片。不要同时修改模型、类别描述和多个阈值，否则难以判断改善来自哪里。

## 人工检查与编辑

- 选中检测框后可拖动整个框。
- 拖动四角控制点可调整大小。
- 可为选中的框切换类别。
- 使用“新建框”按钮后，在图片上拖拽可补充漏检目标。
- 选中框后按 `Delete`，或点击删除按钮，可移除误检。
- 鼠标滚轮可缩放图片。
- 编辑会同步到当前标注状态；离开当前图片前仍建议主动保存。

快捷键：

| 快捷键 | 功能 |
| --- | --- |
| `A` | 上一张图片 |
| `D` | 下一张图片 |
| `Ctrl+S` | 保存当前图片标签 |
| `Delete` | 删除选中的框 |
| `Esc` | 取消正在创建的框 |

自动标注结果必须人工检查，尤其注意画面边缘、小目标、遮挡目标、相似类别和密集场景。

## 批量自动标注

主界面提供两种批量策略：

| 模式 | 行为 | 建议 |
| --- | --- | --- |
| 跳过已有非空 `.txt`，空 `.txt` 重新检测 | 保留已有框；对没有标签文件或空标签的图片重新推理 | 默认推荐，适合继续未完成任务 |
| 重新标注并覆盖所有 `.txt` | 所有图片重新推理并替换现有标签 | 仅在确认要废弃旧结果时使用 |

批量处理的重要规则：

- 每完成一张图片就立即写入标签，不会等整个任务结束后一次性保存。
- 点击取消后，已完成图片的标签会保留，未开始图片不会被改动。
- 空 `.txt` 表示图片已经处理但没有检测到目标。推荐批量模式会重新检测空标签，适合参数调整后再次尝试。
- 运行前应确认类别顺序、模型、设备和参数；任务开始后不要在外部同时修改同一目录的标签。
- 如需完整保留人工结果，请复制数据集或使用版本控制，再进行覆盖模式。
- 批量完成后仍需通过抽检或逐图审核确认质量。

## 项目目录与标签格式

默认工作模式把图片和同名 YOLO 标签放在同一目录，不会自动创建 `images` 或 `labels` 子目录。例如：

```text
my-dataset/
├── image_001.jpg
├── image_001.txt
├── image_002.png
├── image_002.txt
├── classes.txt
├── project.json
├── annotations.json
├── hard_samples.json
└── class_profiles.json
```

文件说明：

| 文件 | 作用 | 是否应保留 |
| --- | --- | --- |
| `图片名.txt` | YOLO 训练标签；与图片同名 | 必须 |
| `classes.txt` | 按行保存类别顺序 | 必须 |
| `project.json` | 模型、类别、阈值和高级设置 | 建议 |
| `annotations.json` | 置信度、来源和审核状态等扩展元数据 | 建议；YOLO `.txt` 仍是权威标签 |
| `hard_samples.json` | 困难样本和复核记录 | 使用高级审核时建议保留 |
| `class_profiles.json` | 类别特征和高级验证配置 | 使用高级审核时建议保留 |

每一行 YOLO 标签固定包含五列：

```text
class_id x_center y_center width height
```

示例：

```text
0 0.512500 0.487500 0.225000 0.350000
```

其中坐标和宽高都归一化到 `0` 至 `1`，保存时保留六位小数。`class_id` 必须是 `classes.txt` 中有效的从零开始编号。空标签文件表示该图片没有目标。

标签、配置和元数据采用临时文件替换的原子写入方式，降低程序中断时产生半截文件的风险；这不能代替外部备份。

## 导出标准 YOLO 数据集

“导出 YOLO Dataset”会把已标注项目复制为训练常用结构：

```text
exported-dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

导出规则：

- 默认训练集比例为 `0.80`，其余进入验证集。
- 导出会复制图片和标签，不移动原始文件。
- 未标注且没有 `.txt` 的图片不会导出；空标签图片可以导出。
- 为避免覆盖，导出目录必须为空，并且不能是当前项目目录。
- 同名但扩展名不同的图片会产生相同标签名，因此程序会拒绝导出并提示冲突。
- 只有一张已标注图片时，会同时用于训练和验证烟雾测试；正式训练应准备独立验证样本。
- `data.yaml` 使用导出目录的绝对路径，移动目录后请检查并更新其中的 `path`。

## 高级推理功能

高级功能默认关闭，基础流程稳定后再逐项启用。

### 切片推理

切片推理会把大图分块检测后合并结果，适合高分辨率图片中的小目标。它会增加推理次数和耗时，重叠比例或合并阈值设置不当还可能产生重复框。建议先保留默认切片参数，在少量图片上比较结果。

### SigLIP2 二次验证

默认模型为 `google/siglip2-base-patch16-224`。它对 YOLO 候选框裁剪区域做图文相似度复核，并保留各阶段分数，默认不启用。

启用前应安装 `verification` 依赖。首次使用会下载额外模型；批量大小、精度和输入尺寸都会影响显存。如果显存不足，程序会尝试降低批量大小，但仍可能需要关闭该功能。

### Qwen3-VL 困难样本复核

默认模型为 `Qwen/Qwen3-VL-8B-Instruct`，采用按需加载。8B 级视觉语言模型体积和资源开销很大，不适合普通中央处理器上的常规批量流程。它只应作为困难样本的辅助判断，不能替代人工审核，也不应把模型自报置信度当成经过校准的概率。

### 推荐启用顺序

1. 先使用基础 YOLO-World 流程建立基准。
2. 小目标漏检明显时尝试切片推理。
3. 类别相似、误检较多时尝试 SigLIP2。
4. 仅对少量冲突或困难样本启用 VLM。
5. 使用数据集统计、评估和基准比较功能记录速度与质量变化。

## 隐私与安全

- 图片、标签和项目配置默认都在本机处理，不会由本项目主动上传到服务器。
- 第一次下载 YOLO、SigLIP 或 VLM 模型时会访问对应模型提供方；离线环境应预先准备模型。
- 项目日志可能包含本地文件路径和错误堆栈。提交问题前请检查并删除用户名、目录结构或其他敏感信息。
- 不要加载不可信 `.pt` 权重，不要运行来源不明的打包程序。
- 公开数据集前，请确认图片版权、肖像权、商业秘密和个人信息处理符合所在地区法律及数据来源许可。
- 安全漏洞请按照 [安全政策](SECURITY.md) 私下报告，不要先在公开议题中披露可利用细节。

## 常见问题

### 程序为什么使用中央处理器，没有使用显卡？

先执行：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.device_count())"
```

如果 `torch.cuda.is_available()` 为 `False`，通常是安装了中央处理器版 PyTorch、显卡驱动不兼容，或当前环境没有可用 NVIDIA 显卡。请按 PyTorch 官方页面重新安装匹配的版本。不要在不同虚拟环境之间混用命令。

### 选择 `cuda:1` 后立即报错怎么办？

编号从 `0` 开始。先查看 `torch.cuda.device_count()`；只有返回值大于 `1` 时才存在 `cuda:1`。需要自动回退时改用 `auto`。

### 中央处理器运行太慢怎么办？

- 使用 `yolov8s-worldv2.pt`。
- 保持或降低 `640` 输入尺寸。
- 关闭切片推理、SigLIP2 和 VLM。
- 先用少量图片调好类别和参数，再启动批量任务。

### 第一次加载模型长时间没有响应怎么办？

通常正在下载权重或初始化推理框架。检查终端输出、网络、磁盘空间和模型目录。不要连续点击加载按钮。离线环境请提前复制权重。

### Linux 启动时报 `libEGL.so.1` 缺失怎么办？

Ubuntu 或 Debian 可安装 Qt 所需运行库：

```bash
sudo apt-get update
sudo apt-get install --yes libegl1
```

远程服务器还需要可用图形桌面或显示转发；本项目是桌面程序，不是无界面网络服务。

### 已有标签为什么发生变化？

最常见原因是选择了“覆盖所有标签”、更改了 `classes.txt` 顺序，或外部工具同时写入同一目录。立即停止批量任务，从备份恢复，并检查 `classes.txt` 和批量模式。

### 为什么生成了空 `.txt`？

空文件表示该图片已处理，但当前参数下没有保留任何检测框。这是有效的 YOLO 负样本标签。降低置信度、修改类别描述或提高输入尺寸后，可以使用推荐批量模式重新检测空标签。

### 到哪里查看启动错误？

严重启动异常会写入 `startup_error.log`。默认日志目录：

| 系统 | 日志目录 |
| --- | --- |
| Windows | `%LOCALAPPDATA%\yolo-world-annotator\logs` |
| Linux | `~/.local/state/yolo-world-annotator/logs`，或 `$XDG_STATE_HOME/yolo-world-annotator/logs` |
| macOS | `~/Library/Logs/yolo-world-annotator/logs` |

提交问题时请附版本号、操作系统、Python 版本、PyTorch 版本、设备选择、复现步骤和脱敏后的错误日志。

## 开发、测试与打包

### 开发环境

```powershell
git clone https://github.com/suiboxi/yolo-world-annotator.git
cd yolo-world-annotator
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,build]"
```

### 质量检查与测试

```powershell
python -m ruff check .
python -m pytest -q
python -m build
python build_exe.py --check-only
```

测试默认使用无界面 Qt 和中央处理器，不要求 CUDA 显卡。GitHub Actions 会在 Windows、Ubuntu 和 Python 3.10、3.11、3.12 的组合上运行测试，并单独执行代码质量和包构建检查。

### 构建 Windows 程序

```powershell
python -m pip install -e ".[build]"
python build_exe.py
```

也可以双击 `一键生成EXE.bat`。输出位于：

```text
dist/YOLOWorldAnnotator/YOLOWorldAnnotator.exe
```

必须保留整个 `YOLOWorldAnnotator` 目录，不能只复制单个 `.exe`。构建过程默认不打包大型模型权重，首次运行仍可能需要下载模型；如需离线分发，请把可信权重放入程序目录旁的 `models/weights`，并同时遵守模型自身许可证。

## 版本、贡献与许可证

项目使用[语义化版本](https://semver.org/lang/zh-CN/)：

- `0.0.1` 表示仍处于早期开发阶段，接口和项目文件格式可能继续调整。
- 发布记录见 [CHANGELOG.md](CHANGELOG.md)。
- Git 标签使用 `v` 前缀，例如 `v0.0.1`。

欢迎提交问题和改进。开始贡献前请阅读 [贡献指南](CONTRIBUTING.md) 和 [行为准则](CODE_OF_CONDUCT.md)。提交代码时应包含必要测试，并确保质量检查全部通过。

本项目采用 [GNU Affero General Public License v3.0 或更高版本](LICENSE)。修改并分发本程序，或通过网络向用户提供修改后的程序功能时，需要遵守 AGPL 的源代码开放义务。PyTorch、Ultralytics、Qt、模型权重及其他依赖有各自的许可证，使用和再分发前请分别确认。
