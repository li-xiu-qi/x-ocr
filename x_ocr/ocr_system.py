# -*- encoding: utf-8 -*-
# OCR系统主类
import os
import time
import copy
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .utils.image_utils import LoadImage, VisualizeResult
from .detection import TextDetector
from .classification import TextClassifier
from .recognition import TextRecognizer


class OnnxOCR:
    """
    基于ONNX运行时的OCR系统，包含文字检测、方向分类和文字识别
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化OCR系统
        
        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        self.root_dir = Path(__file__).resolve().parent
        default_config = {
            "use_det": True,
            "use_cls": True,
            "use_rec": True,
            "text_score": 0.5,
            "max_side_len": 2000,
            "min_side_len": 30,
            "min_height": 30,
            "width_height_ratio": 8,
            "print_verbose": False,
            
            "det": {
                "model_path": str(self.root_dir / "models/ch_PP-OCRv4_det_infer.onnx"),
                "limit_side_len": 736,
                "limit_type": "min",
                "thresh": 0.3,
                "box_thresh": 0.5,
                "max_candidates": 1000,
                "unclip_ratio": 1.6,
                "use_dilation": True,
                "score_mode": "fast",
                "threads": 0
            },
            
            "cls": {
                "model_path": str(self.root_dir / "models/ch_ppocr_mobile_v2.0_cls_infer.onnx"),
                "image_shape": [3, 48, 192],
                "batch_num": 6,
                "thresh": 0.9,
                "label_list": ['0', '180'],
                "threads": 0
            },
            
            "rec": {
                "model_path": str(self.root_dir / "models/ch_PP-OCRv4_rec_infer.onnx"),
                "img_shape": [3, 48, 320],
                "batch_num": 6,
                "threads": 0
            }
        }
        
        self.config = default_config
        if config:
            self._update_config(config)
            
        # 图像处理相关
        self.load_img = LoadImage()
        self.max_side_len = self.config["max_side_len"]
        self.min_side_len = self.config["min_side_len"]
        self.text_score = self.config["text_score"]
        self.min_height = self.config["min_height"]
        self.width_height_ratio = self.config["width_height_ratio"]
        
        # 初始化OCR组件
        self.use_det = self.config["use_det"]
        self.use_cls = self.config["use_cls"]
        self.use_rec = self.config["use_rec"]
        
        if self.use_det:
            self.text_detector = TextDetector(self.config["det"])
            
        if self.use_cls:
            self.text_classifier = TextClassifier(self.config["cls"])
            
        if self.use_rec:
            self.text_recognizer = TextRecognizer(self.config["rec"])
        
    def _update_config(self, config: Dict[str, Any]) -> None:
        """更新配置参数"""
        for k, v in config.items():
            if isinstance(v, dict) and k in self.config and isinstance(self.config[k], dict):
                self.config[k].update(v)
            else:
                self.config[k] = v
    
    def __call__(self, 
                img_content: Union[str, np.ndarray, bytes, Path],
                use_det: Optional[bool] = None,
                use_cls: Optional[bool] = None,
                use_rec: Optional[bool] = None,
                **kwargs) -> Tuple[Optional[List], List[float]]:
        """
        运行OCR系统进行文本检测与识别
        
        Args:
            img_content: 输入图像，可以是路径、ndarray或bytes
            use_det: 是否使用文本检测
            use_cls: 是否使用方向分类
            use_rec: 是否使用文本识别
            
        Returns:
            检测和识别的结果，以及处理时间
        """
        # 设置使用的组件
        use_det = self.use_det if use_det is None else use_det
        use_cls = self.use_cls if use_cls is None else use_cls
        use_rec = self.use_rec if use_rec is None else use_rec
        
        # 处理传入的参数
        if kwargs:
            if "text_score" in kwargs:
                self.text_score = kwargs["text_score"]
            if "box_thresh" in kwargs:
                self.text_detector.box_thresh = kwargs["box_thresh"]
            if "unclip_ratio" in kwargs:
                self.text_detector.unclip_ratio = kwargs["unclip_ratio"]
        
        # 读取并预处理图像
        img = self.load_img(img_content)
        raw_h, raw_w = img.shape[:2]
        
        # 记录图像处理操作
        op_record = {}
        img, ratio_h, ratio_w = self._preprocess_image(img)
        op_record["preprocess"] = {"ratio_h": ratio_h, "ratio_w": ratio_w}
        
        # 初始化变量
        dt_boxes, cls_res, rec_res = None, None, None
        det_elapse, cls_elapse, rec_elapse = 0.0, 0.0, 0.0
        
        # 1. 文本检测
        if use_det:
            img, op_record = self._maybe_add_letterbox(img, op_record)
            dt_boxes, det_elapse = self._text_detect(img)
            if dt_boxes is None:
                return None, [det_elapse]
            
            # 根据检测框裁剪文本区域
            img_list = self._get_crop_img_list(img, dt_boxes)
        else:
            img_list = [img]
        
        # 2. 文本方向分类
        if use_cls:
            img_list, cls_res, cls_elapse = self.text_classifier(img_list)
            
        # 3. 文本识别
        if use_rec:
            rec_res, rec_elapse = self.text_recognizer(img_list)
        
        # 恢复检测框坐标到原图
        if dt_boxes is not None:
            dt_boxes = self._get_origin_box_points(dt_boxes, op_record, raw_h, raw_w)
        
        # 生成最终结果
        ocr_res = self._get_final_result(dt_boxes, cls_res, rec_res, 
                                        det_elapse, cls_elapse, rec_elapse)
        return ocr_res
    
    def _preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """预处理图像，调整大小"""
        h, w = img.shape[:2]
        ratio_h = ratio_w = 1.0
        
        # 处理过大的图像
        if max(h, w) > self.max_side_len:
            if h > w:
                ratio = float(self.max_side_len) / h
            else:
                ratio = float(self.max_side_len) / w
                
            resize_h = int(h * ratio)
            resize_w = int(w * ratio)
            
            resize_h = int(round(resize_h / 32) * 32)
            resize_w = int(round(resize_w / 32) * 32)
            
            img = cv2.resize(img, (resize_w, resize_h))
            ratio_h = h / resize_h
            ratio_w = w / resize_w
            
        # 处理过小的图像
        h, w = img.shape[:2]
        if min(h, w) < self.min_side_len:
            if h < w:
                ratio = float(self.min_side_len) / h
            else:
                ratio = float(self.min_side_len) / w
                
            resize_h = int(h * ratio)
            resize_w = int(w * ratio)
            
            resize_h = int(round(resize_h / 32) * 32)
            resize_w = int(round(resize_w / 32) * 32)
            
            img = cv2.resize(img, (resize_w, resize_h))
            ratio_h = h / resize_h
            ratio_w = w / resize_w
            
        return img, ratio_h, ratio_w
    
    def _maybe_add_letterbox(self, img: np.ndarray, op_record: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """针对特定情况添加letterbox"""
        h, w = img.shape[:2]
        
        # 检查是否需要添加letterbox
        if self.width_height_ratio == -1:
            use_limit_ratio = False
        else:
            use_limit_ratio = w / h > self.width_height_ratio
            
        if h <= self.min_height or use_limit_ratio:
            # 计算需要的填充
            new_h = max(int(w / self.width_height_ratio), self.min_height) * 2
            padding_h = int(abs(new_h - h) / 2)
            
            # 添加填充
            img_padded = np.zeros((h + padding_h * 2, w, 3), dtype=np.uint8)
            img_padded[padding_h:padding_h + h, :, :] = img
            
            op_record["padding"] = {"top": padding_h, "left": 0}
            return img_padded, op_record
        
        op_record["padding"] = {"top": 0, "left": 0}
        return img, op_record
    
    def _text_detect(self, img: np.ndarray) -> Tuple[Optional[List[np.ndarray]], float]:
        """检测文本区域"""
        dt_boxes, det_elapse = self.text_detector(img)
        if dt_boxes is None or len(dt_boxes) < 1:
            return None, det_elapse
        
        dt_boxes = self._sort_boxes(dt_boxes)
        return dt_boxes, det_elapse
        
    def _get_crop_img_list(self, img: np.ndarray, dt_boxes: List[np.ndarray]) -> List[np.ndarray]:
        """根据检测框裁剪文本区域"""
        img_crop_list = []
        for box in dt_boxes:
            tmp_box = copy.deepcopy(box)
            img_crop = self._get_rotate_crop_image(img, tmp_box)
            img_crop_list.append(img_crop)
        return img_crop_list
        
    @staticmethod
    def _get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
        """旋转并裁剪图像区域"""
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )
        pts_std = np.array(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        ).astype(np.float32)
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img
        
    @staticmethod
    def _sort_boxes(dt_boxes: np.ndarray) -> List[np.ndarray]:
        """
        对文本框按照从上到下，从左到右的顺序进行排序
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        # 对相近高度的框进行二次排序
        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if (
                    abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10
                    and _boxes[j + 1][0][0] < _boxes[j][0][0]
                ):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes
        
    def _get_origin_box_points(self, dt_boxes: List[np.ndarray], op_record: Dict[str, Any], 
                              raw_h: int, raw_w: int) -> np.ndarray:
        """将检测框坐标还原到原始图像尺寸"""
        dt_boxes_array = np.array(dt_boxes).astype(np.float32)
        
        # 逆向处理所有操作
        for op in reversed(list(op_record.keys())):
            v = op_record[op]
            if "padding" in op:
                top, left = v.get("top", 0), v.get("left", 0)
                dt_boxes_array[:, :, 0] -= left
                dt_boxes_array[:, :, 1] -= top
            elif "preprocess" in op:
                ratio_h = v.get("ratio_h")
                ratio_w = v.get("ratio_w")
                dt_boxes_array[:, :, 0] *= ratio_w
                dt_boxes_array[:, :, 1] *= ratio_h

        # 确保坐标在图像范围内
        dt_boxes_array = np.clip(dt_boxes_array, 0, [raw_w, raw_h])
        return dt_boxes_array
        
    def _get_final_result(self, dt_boxes: Optional[List[np.ndarray]], 
                         cls_res: Optional[List], 
                         rec_res: Optional[List], 
                         det_elapse: float, 
                         cls_elapse: float,
                         rec_elapse: float) -> Tuple[Optional[List], List[float]]:
        """整合最终的OCR结果"""
        if dt_boxes is None and rec_res is None and cls_res is not None:
            return cls_res, [cls_elapse]

        if dt_boxes is None and rec_res is None:
            return None, None

        if dt_boxes is None and rec_res is not None:
            return [[res[0], res[1]] for res in rec_res], [rec_elapse]

        if dt_boxes is not None and rec_res is None:
            return [box.tolist() for box in dt_boxes], [det_elapse]
            
        # 根据置信度过滤结果
        filtered_boxes = []
        filtered_rec_res = []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result[0], rec_result[1]
            if float(score) >= self.text_score:
                filtered_boxes.append(box)
                filtered_rec_res.append(rec_result)
                
        if not filtered_boxes or not filtered_rec_res:
            return None, None
                
        # 组合最终结果
        result = [[box.tolist(), *res] for box, res in zip(filtered_boxes, filtered_rec_res)]
        times = [det_elapse, cls_elapse, rec_elapse]
            
        return result, times

    def visualize(self, img_content: Union[str, np.ndarray, bytes, Path], 
                 result: List, font_path: Optional[str] = None) -> np.ndarray:
        """可视化OCR结果"""
        img = self.load_img(img_content)
        
        # 如果没有结果，直接返回原图
        if result is None or not result:
            return img
            
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