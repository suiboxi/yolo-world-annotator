# YOLO-World GPU 自动标注器

本地中文桌面标注程序，使用本机 `yolo26` 环境、YOLO-World 权重和 NVIDIA GPU。程序不调用云端 API，也不允许自动回退到 CPU。

## 启动

已经封装好的版本位于 `dist\YOLOWorldAnnotator\YOLOWorldAnnotator.exe`，双击该 EXE 即可启动，不需要再打开命令行。发布时必须保留整个 `YOLOWorldAnnotator` 文件夹，不能只单独复制 EXE；同级的 `_internal` 和 `models` 包含 GPU 运行库与模型权重。

如需重新生成 EXE，双击 `一键生成EXE.bat`。构建完成后会自动检查程序文件、CUDA 依赖、CLIP 词表和默认权重是否齐全。

源码方式仍可双击 `启动GPU标注器.bat`，或运行：

```powershell
& "C:\Users\ROG\anaconda3\envs\yolo26\python.exe" main.py
```

已核验环境：RTX 4060 Laptop GPU / `cuda:0`，PyTorch `2.6.0+cu124`，权重 `models/weights/yolov8s-worldv2.pt`。

## 使用流程

1. 点“打开图片文件夹”，选择 JPG/PNG/BMP/WebP 所在目录。
2. 在“预设型号”中选择 YOLOv8-World V2 或经典版的 S/M/L/X；未下载的官方权重会在首次使用时自动下载，也可以浏览选择已有的 YOLO-World `.pt` 文件。
3. 在右侧分别输入“最终标签类别”和“YOLO-World 模型提示词”，两区每行一一对应。提示词可与最终类别不同，例如最终保存 `raspberry`，模型使用对当前仿真果实更有效的 `strawberry` 搜框。
4. 点“自动标注当前图片”或“自动标注全部图片”。
5. 在中间预览中人工修改；鼠标释放后立即保存。

普通 `yolo11n/s/m/l/x.pt` 是固定类别检测模型，不具备 YOLO-World 的开放词汇提示能力，因此不会列入 World 预设。

## 原地保存

程序不创建 `images` 或 `labels` 文件夹。标注和图片放在一起：

```text
我的图片/
├── sample_001.jpg
├── sample_001.txt
├── sample_002.png
├── sample_002.txt
├── classes.txt
├── project.json
└── annotations.json
```

同名 txt 是标准 YOLO 归一化格式：`class_id x_center y_center width height`。`classes.txt` 保存类别顺序，`project.json` 保存参数，`annotations.json` 保留置信度和编辑元数据；它们都是同目录文件，不是新标注文件夹。

## 编辑方法

- 新建：选择右侧“当前类别”，直接在图片空白处按住左键拖动即可拉出新框，不需要先点新建按钮。
- 移动：在已有框内按住左键拖动。
- 改大小：拖动选中框四角的蓝色控制点。
- 删除：单击已有框，然后点击框旁边弹出的红色“删除此框”按钮。
- 改类：选中框，选新类别，点“将此类别应用到选中框”；双击框也可直接应用当前类别。
- `Esc` 取消正在拖动的新框，`A` / `D` 切换上/下一张。

## 参数

- 置信度阈值：越低检出越多，误检也可能增多；建议 `0.25`。
- 重叠框去重阈值：控制重叠框合并；建议 `0.45`。
- 推理图像尺寸：越大越利于小目标，但更慢且占显存；4060 建议 `640`–`960`。
- 低置信框提示阈值：低于它的框以黄色虚线提醒，不会自动删除。
- 批量模式：默认跳过已有非空 txt 以保护人工结果；空 txt 会自动重新检测，便于修复旧版产生的空标签。

## 验证

已通过 42 项自动化测试（包含直接拖框、弹出按钮删除、205 张图批量稳定性回归）、离屏 GUI 启动，并使用本机 S/X 权重在 `cuda:0` 上完成真实图片推理。EXE 发布包还经过了本机 RTX 4060 Laptop GPU 的界面启动和实际推理验证。

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
& "C:\Users\ROG\anaconda3\envs\yolo26\python.exe" -m pytest -q
```
