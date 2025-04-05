import os
import sys
import cv2
import numpy as np
from typing import List, Optional, Union
from pathlib import Path

from .image_utils import LoadImage, VisualizeResult


class OCRVisualizer:
    """
    OCR结果可视化工具
    """
    def __init__(self, print_verbose: bool = False):
        """
        初始化可视化器
        
        Args:
            print_verbose: 是否打印详细信息
        """
        self.print_verbose = print_verbose
        self.load_img = LoadImage()
        
    def get_system_font(self) -> Optional[str]:
        """获取系统可用的中文字体"""
        if sys.platform.startswith('win'):
            # Windows系统字体路径
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",  # 黑体
                "C:/Windows/Fonts/simsun.ttc",  # 宋体
                "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
            ]
        elif sys.platform.startswith('darwin'):
            # macOS系统字体路径
            font_paths = [
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/STHeiti Light.ttc",
                "/System/Library/Fonts/Hiragino Sans GB.ttc",
            ]
        else:
            # Linux系统字体路径
            font_paths = [
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/truetype/arphic/uming.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            ]
        
        # 检查字体文件是否存在
        for font_path in font_paths:
            if os.path.exists(font_path):
                return font_path
        
        return None

    def visualize(self, img_content: Union[str, np.ndarray, bytes, Path], 
                  result: List, font_path: Optional[str] = None) -> np.ndarray:
        """
        可视化OCR结果
        
        Args:
            img_content: 输入图像
            result: OCR识别结果
            font_path: 字体路径，如果为None则自动获取系统字体
            
        Returns:
            可视化后的图像
        """
        img = self.load_img(img_content)
        
        # 如果没有结果，直接返回原图
        if result is None or not result:
            return img
        
        # 自动获取系统字体（如果未提供）
        if font_path is None:
            font_path = self.get_system_font()
            if font_path and self.print_verbose:
                print(f"使用系统字体: {font_path}")
            
        vis = VisualizeResult()
        
        # 根据结果类型选择可视化方法
        if len(result[0]) == 1:  # 只有框
            boxes = [item[0] for item in result]
            return vis(img, boxes)
        elif len(result[0]) > 2:  # 有框和文本
            boxes = [item[0] for item in result]
            texts = [item[1] for item in result]
            scores = [item[2] if len(item) > 2 else 1.0 for item in result]
            return vis(img, boxes, texts, scores, font_path)
        
        return img
        
    def visualize_to_file(self, img_content: Union[str, np.ndarray, bytes, Path], 
                         result: List, output_path: str, 
                         font_path: Optional[str] = None) -> None:
        """
        将可视化结果保存到文件
        
        Args:
            img_content: 输入图像
            result: OCR识别结果
            output_path: 输出图像路径
            font_path: 字体路径，如果为None则自动获取系统字体
        """
        # 获取可视化结果
        vis_img = self.visualize(img_content, result, font_path)
        
        # 保存到文件
        cv2.imwrite(output_path, vis_img)
        
        if self.print_verbose:
            print(f"可视化结果已保存到 {output_path}")