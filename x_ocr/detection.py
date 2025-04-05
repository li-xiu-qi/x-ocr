# -*- encoding: utf-8 -*-
# 文本检测模块
import time
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper
from shapely.geometry import Polygon


class TextDetector:
    """文本检测器，使用DB(Differentiable Binarization)模型检测文本区域"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化文本检测器
        
        Args:
            config: 配置参数
        """
        self.thresh = config.get("thresh", 0.3)
        self.box_thresh = config.get("box_thresh", 0.5)
        self.max_candidates = config.get("max_candidates", 1000)
        self.unclip_ratio = config.get("unclip_ratio", 1.6)
        self.use_dilation = config.get("use_dilation", True)
        self.score_mode = config.get("score_mode", "fast")
        self.limit_side_len = config.get("limit_side_len", 736)
        self.limit_type = config.get("limit_type", "min")
        
        # 配置ONNX会话
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("Model path must be provided for text detector")
            
        # 设置推理选项
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        sess_options.enable_cpu_mem_arena = False
        
        threads = config.get("threads", 0)
        if threads > 0:
            sess_options.intra_op_num_threads = threads
        
        # 创建ONNX运行时会话
        self.session = ort.InferenceSession(model_path, sess_options)
        
        # 预处理参数
        self.mean = config.get("mean", [0.5, 0.5, 0.5])
        self.std = config.get("std", [0.5, 0.5, 0.5])
        
        # 如果使用膨胀，预先创建kernel
        self.dilation_kernel = None
        if self.use_dilation:
            self.dilation_kernel = np.array([[1, 1], [1, 1]])
            
    def __call__(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        检测文本区域
        
        Args:
            img: 输入图像
            
        Returns:
            检测到的文本框和处理时间
        """
        start_time = time.time()
        
        # 图像预处理
        ori_img_shape = img.shape[:2]  # h, w
        img = self._preprocess(img)
        if img is None:
            return None, 0
            
        # 模型推理
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: img})
        preds = outputs[0]
        
        # 后处理
        boxes, scores = self._postprocess(preds, ori_img_shape)
        boxes = self._filter_boxes(boxes, ori_img_shape)
        
        elapse = time.time() - start_time
        return boxes, elapse
        
    def _preprocess(self, img: np.ndarray) -> Optional[np.ndarray]:
        """图像预处理"""
        h, w = img.shape[:2]
        
        # 调整图像大小
        if self.limit_type == "max":
            if max(h, w) > self.limit_side_len:
                if h > w:
                    ratio = float(self.limit_side_len) / h
                else:
                    ratio = float(self.limit_side_len) / w
            else:
                ratio = 1.0
        else:
            if min(h, w) < self.limit_side_len:
                if h < w:
                    ratio = float(self.limit_side_len) / h
                else:
                    ratio = float(self.limit_side_len) / w
            else:
                ratio = 1.0

        resize_h = int(h * ratio)
        resize_w = int(w * ratio)
        
        # 确保尺寸是32的倍数
        resize_h = int(round(resize_h / 32) * 32)
        resize_w = int(round(resize_w / 32) * 32)
        
        # 如果尺寸太小，返回None
        if resize_h <= 0 or resize_w <= 0:
            return None
            
        # 调整图像大小
        img = cv2.resize(img, (resize_w, resize_h))
        
        # 图像归一化 - 明确指定为float32类型
        img = img.astype(np.float32)
        img = img / 255.0
        img = (img - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
        
        # 调整通道顺序并增加批次维度
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32)  # 确保输出是float32类型
        
        return img
        
    def _postprocess(self, pred: np.ndarray, ori_shape: Tuple[int, int]) -> Tuple[np.ndarray, List[float]]:
        """模型输出后处理"""
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh
        
        mask = segmentation[0]
        if self.dilation_kernel is not None:
            mask = cv2.dilate(mask.astype(np.uint8), self.dilation_kernel)
            
        boxes, scores = self._boxes_from_bitmap(pred[0], mask, ori_shape[1], ori_shape[0])
        return boxes, scores
        
    def _boxes_from_bitmap(self, pred: np.ndarray, bitmap: np.ndarray, 
                         dest_width: int, dest_height: int) -> Tuple[np.ndarray, List[float]]:
        """从二值图中提取文本框"""
        height, width = bitmap.shape
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        
        num_contours = min(len(contours), self.max_candidates)
        
        boxes, scores = [], []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self._get_mini_boxes(contour)
            if sside < 3:
                continue
                
            # 计算文本框得分
            if self.score_mode == "fast":
                score = self._box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self._box_score_slow(pred, contour)
                
            if self.box_thresh > score:
                continue
                
            # 扩展文本框
            box = self._unclip(points, self.unclip_ratio)
            box, sside = self._get_mini_boxes(box)
            if sside < 5:
                continue
                
            # 将框坐标映射回原始图像
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
                
            boxes.append(box.astype(np.int32))
            scores.append(score)
            
        return np.array(boxes, dtype=np.int32), scores
        
    def _get_mini_boxes(self, contour: np.ndarray) -> Tuple[np.ndarray, float]:
        """获取最小矩形框"""
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        
        # 确保框的点顺序为左上、右上、右下、左下
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0

        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
            
        box = np.array([
            points[index_1], points[index_2], 
            points[index_3], points[index_4]
        ])
        return box, min(bounding_box[1])
        
    def _box_score_fast(self, bitmap: np.ndarray, _box: np.ndarray) -> float:
        """快速计算文本框得分"""
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)
        
        # 创建掩码用于计算平均分数
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        
        # 计算区域内的平均分数
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
        
    def _box_score_slow(self, bitmap: np.ndarray, contour: np.ndarray) -> float:
        """使用多边形计算平均分数（慢但更准确）"""
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))
        
        # 计算边界框
        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)
        
        # 创建掩码
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin
        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        
        # 计算区域内的平均分数
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
        
    def _unclip(self, box: np.ndarray, unclip_ratio: float = 1.5) -> np.ndarray:
        """扩展文本框"""
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance)).reshape((-1, 1, 2))
        return expanded
        
    def _filter_boxes(self, dt_boxes: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """过滤和整理文本框"""
        img_height, img_width = image_shape
        dt_boxes_new = []
        for box in dt_boxes:
            # 确保框的点顺序为顺时针方向
            box = self._order_points_clockwise(box)
            # 将框限制在图像边界内
            box = self._clip_det_res(box, img_height, img_width)
            
            # 如果框太小，则跳过
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
                
            dt_boxes_new.append(box)
            
        return np.array(dt_boxes_new)
        
    def _order_points_clockwise(self, pts: np.ndarray) -> np.ndarray:
        """将点按顺时针方向排序"""
        # 基于x坐标排序
        x_sorted = pts[np.argsort(pts[:, 0]), :]
        
        # 获取最左侧和最右侧的点
        left_most = x_sorted[:2, :]
        right_most = x_sorted[2:, :]
        
        # 根据y坐标对左侧点进行排序
        left_most = left_most[np.argsort(left_most[:, 1]), :]
        top_left, bottom_left = left_most
        
        # 根据y坐标对右侧点进行排序
        right_most = right_most[np.argsort(right_most[:, 1]), :]
        top_right, bottom_right = right_most
        
        # 顺时针返回点：左上、右上、右下、左下
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
        
    def _clip_det_res(self, points: np.ndarray, img_height: int, img_width: int) -> np.ndarray:
        """将检测框限制在图像边界内"""
        for i in range(points.shape[0]):
            points[i, 0] = int(min(max(points[i, 0], 0), img_width - 1))
            points[i, 1] = int(min(max(points[i, 1], 0), img_height - 1))
        return points