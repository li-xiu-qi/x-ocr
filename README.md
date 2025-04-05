# X-OCR: 基于 ONNX 运行时的高效 OCR 系统

X-OCR 是一个高效、灵活的光学字符识别(OCR)系统，基于 ONNX 运行时开发，专为中英文混合文本识别设计。该系统实现了文本检测、方向分类和文本识别三大功能模块，可以快速准确地从图像中提取文字信息。

## 功能特点

- **高效推理**: 使用 ONNX 运行时进行模型推理，性能优越，跨平台兼容性好
- **完整 OCR 流程**: 集成了文本检测(DB)、方向分类和文本识别(CTC)等完整 OCR 组件
- **灵活配置**: 支持自定义模型路径、参数阈值、推理线程等多项配置
- **多格式支持**: 支持多种输入图像格式，包括文件路径、numpy 数组、bytes、PIL 图像
- **结果可视化**: 提供内置的结果可视化功能，方便调试和展示

## 系统要求

- Python 3.6+
- OpenCV (cv2)
- NumPy
- ONNX Runtime
- Pyclipper
- Shapely
- PIL (Pillow)

## 安装

```bash
# 克隆仓库
git clone https://github.com/li-xiu-qi/x-ocr.git
cd x-ocr

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 基本用法

```python
from x_ocr import OnnxOCR

# 初始化OCR系统（使用默认配置）
ocr = OnnxOCR()

# 识别图像文本
image_path = "path/to/your/image.jpg"
result, times = ocr(image_path)

# 打印结果
if result:
    print(f"识别到 {len(result)} 个文本区域:")
    for i, item in enumerate(result):
        if len(item) > 2:  # 完整结果包含框、文本和置信度
            box, text, score = item[0], item[1], item[2]
            print(f"区域 {i+1}: 文本: {text}, 置信度: {score:.4f}")
else:
    print("未识别到文本")

# 可视化结果
vis_img = ocr.visualize(image_path, result)
cv2.imwrite("visualized_result.jpg", vis_img)
```

### 自定义配置

```python

from x_ocr import OnnxOCR

# 自定义配置参数
config = {
    "use_cls": False,  # 不使用方向分类
    "text_score": 0.6,  # 设置文本置信度阈值
    "det": {
        "model_path": "path/to/your/detection_model.onnx",
        "box_thresh": 0.6,
        "unclip_ratio": 1.8
    },
    "rec": {
        "model_path": "path/to/your/recognition_model.onnx",
        "batch_num": 8
    }
}

# 使用自定义配置初始化OCR系统
ocr = OnnxOCR(config)

# 进行OCR识别
result, times = ocr("path/to/your/image.jpg")
```

## 项目结构

```
x-ocr/
├── x_ocr/                   # 主代码包
│   ├── __init__.py          # 包初始化
│   ├── ocr_system.py        # OCR主系统实现
│   ├── detection.py         # 文本检测模块
│   ├── classification.py    # 方向分类模块
│   ├── recognition.py       # 文本识别模块
│   └── utils/               # 工具函数包
│       ├── __init__.py
│       └── image_utils.py   # 图像处理工具
├── models/                  # 预训练模型目录
│   ├── ch_PP-OCRv4_det_infer.onnx    # 文本检测模型
│   ├── ch_ppocr_mobile_v2.0_cls_infer.onnx  # 方向分类模型（可选）
│   └── ch_PP-OCRv4_rec_infer.onnx    # 文本识别模型
├── test.py                  # 使用示例
├── .gitignore               # Git忽略文件
├── LICENSE                  # 许可证文件
└── README.md                # 项目说明
```

## API 参考

### OnnxOCR 类

主要的 OCR 系统类，集成了检测、分类和识别功能。

```python
ocr = OnnxOCR(config=None)
```

#### 参数

- `config` (dict, optional): 配置字典，包含以下可选键：
  - `use_det` (bool): 是否使用文本检测，默认为 True
  - `use_cls` (bool): 是否使用方向分类，默认为 True
  - `use_rec` (bool): 是否使用文本识别，默认为 True
  - `text_score` (float): 文本置信度阈值，默认为 0.5
  - `max_side_len` (int): 图像最大边长，默认为 2000
  - `min_side_len` (int): 图像最小边长，默认为 30
  - `det` (dict): 文本检测配置
  - `cls` (dict): 方向分类配置
  - `rec` (dict): 文本识别配置

#### 方法

##### **call**

```python
result, times = ocr(img_content, use_det=None, use_cls=None, use_rec=None, **kwargs)
```

执行 OCR 推理过程。

- **参数**:

  - `img_content`: 输入图像，可以是路径字符串、numpy 数组、bytes 或 Path 对象
  - `use_det`: 是否使用文本检测
  - `use_cls`: 是否使用方向分类
  - `use_rec`: 是否使用文本识别
  - `**kwargs`: 其他参数，如`text_score`、`box_thresh`等

- **返回**:
  - `result`: OCR 结果列表，每个元素包含框坐标、文本内容和置信度
  - `times`: 各阶段处理时间列表

##### visualize

```python
img = ocr.visualize(img_content, result, font_path=None)
```

可视化 OCR 结果。

- **参数**:

  - `img_content`: 原始图像
  - `result`: OCR 结果
  - `font_path`: 字体文件路径，用于显示中文

- **返回**:
  - `img`: 可视化后的图像

## 性能优化

要获得最佳性能，可以考虑以下几点：

1. 设置合适的线程数(`threads`参数)以充分利用 CPU
2. 调整批处理大小(`batch_num`参数)以平衡内存占用和速度
3. 适当调整图像大小限制(`limit_side_len`参数)
4. 根据实际需求开启或关闭各组件(`use_det`, `use_cls`, `use_rec`)

## 模型下载

系统默认使用的是 PP-OCRv4 模型，您可以从[PaddleOCR 官方仓库](https://github.com/PaddlePaddle/PaddleOCR)下载这些模型，并转换为 ONNX 格式。

## 许可证

X-OCR采用Apache License 2.0开源许可证，这是一个非常灵活且商业友好的许可证。

- 您可以自由使用、修改和分发此代码，包括用于商业目的
- 您修改后的代码无需开源
- 您需要保留原始代码中的版权和许可证声明
- 请查看[LICENSE](LICENSE)文件获取完整的许可证文本

## 贡献

欢迎提交问题和贡献代码，请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支
3. 提交您的变更
4. 提交合并请求

## 联系方式

- 作者：筱可
- GitHub：[https://github.com/li-xiu-qi](https://github.com/li-xiu-qi)
- 项目地址：[https://github.com/li-xiu-qi/x-ocr](https://github.com/li-xiu-qi/x-ocr)
