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

# 获取中文字体路径
def get_system_font():
    """获取系统可用的中文字体"""
    if sys.platform.startswith('win'):
        # Windows系统字体路径
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",   # 宋体
            "C:/Windows/Fonts/simkai.ttf",   # 楷体
            "C:/Windows/Fonts/msyh.ttc"      # 微软雅黑
        ]
    elif sys.platform.startswith('darwin'):
        # macOS系统字体路径
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc"
        ]
    else:
        # Linux系统字体路径
        font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc"
        ]
    
    # 检查字体文件是否存在
    for font_path in font_paths:
        if os.path.exists(font_path):
            return font_path
    
    return None

# 获取系统中文字体
font_path = get_system_font()
if font_path:
    print(f"使用中文字体: {font_path}")
else:
    print("未找到系统中文字体，将使用默认字体")

# 可视化结果（如果有图像路径）
if os.path.exists(image_path):
    # 使用中文字体
    vis_img = ocr.visualize(image_path, result, font_path)
    
    # 保存可视化结果
    cv2.imwrite("./visualized_image.jpg", vis_img)
    print("可视化结果已保存到 ./visualized_image.jpg")