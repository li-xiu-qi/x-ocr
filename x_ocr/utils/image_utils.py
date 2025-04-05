# -*- encoding: utf-8 -*-
# 图像处理工具
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


class LoadImage:
    """图像加载类，支持多种输入格式"""
    
    def __init__(self):
        pass
        
    def __call__(self, img: Union[str, np.ndarray, bytes, Path, Image.Image]) -> np.ndarray:
        """
        加载图像
        
        Args:
            img: 输入图像，支持多种格式
            
        Returns:
            加载后的图像数组
        """
        # 检查输入类型
        if not isinstance(img, (str, np.ndarray, bytes, Path, Image.Image)):
            raise ValueError(f"不支持的图像类型: {type(img)}")
            
        # 记录原始类型
        origin_img_type = type(img)
        
        # 加载图像
        img = self._load_img(img)
        
        # 转换为OpenCV格式
        img = self._convert_img(img, origin_img_type)
        
        return img
        
    def _load_img(self, img: Union[str, np.ndarray, bytes, Path, Image.Image]) -> np.ndarray:
        """从不同格式加载图像"""
        if isinstance(img, (str, Path)):
            # 检查文件是否存在
            if not Path(img).exists():
                raise FileNotFoundError(f"图像文件不存在: {img}")
                
            try:
                # 使用PIL加载图像，然后转换为numpy数组
                img = self._img_to_ndarray(Image.open(img))
            except UnidentifiedImageError as e:
                raise ValueError(f"无法识别图像文件: {img}") from e
            return img
            
        if isinstance(img, bytes):
            # 从字节数据加载
            img = self._img_to_ndarray(Image.open(BytesIO(img)))
            return img
            
        if isinstance(img, np.ndarray):
            # 已经是numpy数组
            return img
            
        if isinstance(img, Image.Image):
            # 从PIL图像转换
            return self._img_to_ndarray(img)
            
        raise ValueError(f"不支持的图像类型: {type(img)}")
        
    @staticmethod
    def _img_to_ndarray(img: Image.Image) -> np.ndarray:
        """将PIL图像转换为numpy数组"""
        if img.mode == "1":
            img = img.convert("L")
            return np.array(img)
        return np.array(img)
        
    def _convert_img(self, img: np.ndarray, origin_img_type: type) -> np.ndarray:
        """将图像转换为BGR格式"""
        if img.ndim == 2:
            # 灰度图像转RGB
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        if img.ndim == 3:
            channel = img.shape[2]
            if channel == 1:
                # 单通道转RGB
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
            if channel == 3:
                # 如果来源是PIL或文件，转换RGB为BGR
                if issubclass(origin_img_type, (str, Path, bytes, Image.Image)):
                    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return img
                
            if channel == 4:
                # 处理带Alpha通道的图像
                b, g, r, a = cv2.split(img)
                img_rgb = cv2.merge((b, g, r))
                
                # 创建遮罩
                mask = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
                
                # 合并图像
                img = cv2.bitwise_and(img_rgb, img_rgb, mask=a)
                return img
                
            raise ValueError(f"不支持的图像通道数: {channel}")
            
        raise ValueError(f"不支持的图像维度: {img.ndim}")


class VisualizeResult:
    """OCR结果可视化类"""
    
    def __init__(self, text_score: float = 0.5):
        """
        初始化结果可视化工具
        
        Args:
            text_score: 文本置信度阈值
        """
        self.text_score = text_score
        self.load_img = LoadImage()
        
    def __call__(
        self,
        img_content: Union[str, np.ndarray, bytes, Path, Image.Image],
        dt_boxes: np.ndarray,
        txts: Optional[List[str]] = None,
        scores: Optional[List[float]] = None,
        font_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        可视化OCR结果
        
        Args:
            img_content: 原始图像
            dt_boxes: 检测到的文本框
            txts: 识别的文本内容
            scores: 文本置信度
            font_path: 字体文件路径
            
        Returns:
            可视化结果图像
        """
        if txts is None:
            return self._draw_dt_boxes(img_content, dt_boxes)
        return self._draw_ocr_box_txt(img_content, dt_boxes, txts, scores, font_path)
    
    def _draw_dt_boxes(self, img_content: Union[str, np.ndarray, bytes, Path, Image.Image], 
                     dt_boxes: np.ndarray) -> np.ndarray:
        """绘制检测框"""
        img = self.load_img(img_content)
        
        for idx, box in enumerate(dt_boxes):
            # 随机颜色
            color = self._get_random_color()
            
            # 绘制多边形
            points = np.array(box)
            cv2.polylines(img, np.int32([points]), 1, color=color, thickness=2)
            
            # 添加索引标签
            start_point = (round(points[0][0]), round(points[0][1]))
            cv2.putText(img, f"{idx}", start_point, 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                      
        return img
        
    def _draw_ocr_box_txt(
        self,
        img_content: Union[str, np.ndarray, bytes, Path, Image.Image],
        dt_boxes: np.ndarray,
        txts: List[str],
        scores: Optional[List[float]] = None,
        font_path: Optional[str] = None,
    ) -> np.ndarray:
        """绘制检测框和识别文本"""
        # 检查字体路径
        if font_path is None or not Path(font_path).exists():
            # 使用OpenCV绘制（无法显示中文）
            return self._draw_ocr_box_txt_cv(img_content, dt_boxes, txts, scores)
            
        image = Image.fromarray(cv2.cvtColor(self.load_img(img_content), cv2.COLOR_BGR2RGB))
        h, w = image.height, image.width
        
        # 创建左右两个图像
        img_left = image.copy()
        img_right = Image.new("RGB", (w, h), (255, 255, 255))
        
        # 创建绘图对象
        draw_left = ImageDraw.Draw(img_left)
        draw_right = ImageDraw.Draw(img_right)
        
        # 设置随机种子以保证颜色一致性
        np.random.seed(0)
        
        # 遍历所有文本框
        for idx, (box, txt) in enumerate(zip(dt_boxes, txts)):
            if scores is not None and float(scores[idx]) < self.text_score:
                continue
                
            # 随机颜色
            color = self._get_random_color()
            
            # 绘制多边形
            box_list = np.array(box).reshape(8).tolist()
            draw_left.polygon(box_list, fill=color)
            draw_right.polygon(box_list, outline=color)
            
            # 计算文本框高宽
            box_height = self._get_box_height(box)
            box_width = self._get_box_width(box)
            
            # 根据文本框形状调整文本大小和位置
            if box_height > 2 * box_width:  # 竖排文本
                font_size = max(int(box_width * 0.9), 10)
                font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
                cur_y = box[0][1]
                
                # 逐字绘制
                for c in txt:
                    char_size = self._get_char_size(font, c)
                    draw_right.text((box[0][0] + 3, cur_y), c, 
                                  fill=(0, 0, 0), font=font)
                    cur_y += char_size
            else:  # 横排文本
                font_size = max(int(box_height * 0.8), 10)
                font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
                draw_right.text([box[0][0], box[0][1]], txt, 
                              fill=(0, 0, 0), font=font)
        
        # 混合左侧图像
        img_left = Image.blend(image, img_left, 0.5)
        
        # 组合左右图像
        img_show = Image.new("RGB", (w * 2, h), (255, 255, 255))
        img_show.paste(img_left, (0, 0, w, h))
        img_show.paste(img_right, (w, 0, w * 2, h))
        
        return cv2.cvtColor(np.array(img_show), cv2.COLOR_RGB2BGR)
    
    def _draw_ocr_box_txt_cv(
        self,
        img_content: Union[str, np.ndarray, bytes, Path, Image.Image],
        dt_boxes: Union[np.ndarray, List],
        txts: List[str],
        scores: Optional[List[float]] = None,
    ) -> np.ndarray:
        """使用OpenCV绘制文本和框（当没有字体文件时使用）"""
        img = self.load_img(img_content)
        h, w = img.shape[:2]
        
        # 创建左右两个图像
        img_left = img.copy()
        img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # 设置随机种子以保证颜色一致性
        np.random.seed(0)
        
        # 遍历所有文本框
        for idx, (box, txt) in enumerate(zip(dt_boxes, txts)):
            if scores is not None and float(scores[idx]) < self.text_score:
                continue
                
            # 随机颜色
            color = self._get_random_color()
            
            # 确保box是numpy数组
            box_array = np.array(box, dtype=np.int32)
            
            # 左侧半透明框
            overlay = img.copy()
            cv2.fillPoly(overlay, [box_array], color)
            cv2.addWeighted(overlay, 0.5, img_left, 0.5, 0, img_left)
            
            # 右侧框和文本
            cv2.polylines(img_right, [box_array], True, color, 2)
            
            # 计算文本位置 - 适用于列表或NumPy数组
            box_points = np.array(box)
            min_x = np.min(box_points[:, 0]) if box_points.ndim == 2 else box[0][0]
            min_y = np.min(box_points[:, 1]) if box_points.ndim == 2 else box[0][1]
            
            # 绘制文本（如果是中文，可能显示不正确）
            cv2.putText(img_right, txt, (int(min_x), int(min_y) - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # 组合左右图像
        img_show = np.hstack((img_left, img_right))
        return img_show
    
    @staticmethod
    def _get_random_color() -> Tuple[int, int, int]:
        """生成随机RGB颜色"""
        return (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )
        
    @staticmethod
    def _get_box_height(box: List[List[float]]) -> float:
        """计算文本框高度"""
        return np.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][1])**2)
        
    @staticmethod
    def _get_box_width(box: List[List[float]]) -> float:
        """计算文本框宽度"""
        return np.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)
        
    @staticmethod
    def _get_char_size(font, char_str: str) -> float:
        """获取字符高度"""
        # 兼容不同版本的Pillow
        if hasattr(font, "getsize"):
            get_size_func = getattr(font, "getsize")
            return get_size_func(char_str)[1]
            
        if hasattr(font, "getlength"):
            get_size_func = getattr(font, "getlength")
            return get_size_func(char_str)
            
        # 如果都没有，使用默认值
        return font.size