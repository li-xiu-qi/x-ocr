import cv2
import os
import sys
from x_ocr import OnnxOCR
from pathlib import Path

# 获取模型的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
det_model_path = os.path.join(current_dir, "models", "ch_PP-OCRv4_det_infer.onnx")
rec_model_path = os.path.join(current_dir, "models", "ch_PP-OCRv4_rec_infer.onnx")

# 配置OCR引擎
config = {
    "use_cls": False,  # 禁用方向分类功能，因为没有分类模型
    "print_verbose": True,  # 显示详细日志
    "det": {
        "model_path": det_model_path
    },
    "rec": {
        "model_path": rec_model_path
    }
}

# 初始化OCR引擎
ocr = OnnxOCR(config)

# 使用image.png作为测试图像
image_path = os.path.join(current_dir, "image.png")
result, times = ocr(image_path)

# 打印识别结果
if result:
    print(f"识别到 {len(result)} 个文本区域:")
    for i, item in enumerate(result):
        if len(item) > 2:  # 如果有文本结果
            box, text, score = item[0], item[1], item[2]
            print(f"区域 {i+1}: 文本: {text}, 置信度: {score:.4f}")
        else:
            print(f"区域 {i+1}: {item}")
else:
    print("未识别到文本")

print(f"处理时间: 检测={times[0]:.4f}s, 方向分类={times[1]:.4f}s, 识别={times[2]:.4f}s, 总计={sum(times):.4f}s")

# 可视化结果并保存（自动获取系统字体）
if os.path.exists(image_path) and result:
    # 使用新的可视化并保存方法
    ocr.visualize_to_file(image_path, result, "./visualized_image.jpg")