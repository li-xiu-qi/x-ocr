# -*- encoding: utf-8 -*-
# 文本方向分类模块
import math
import time
from typing import Dict, List, Tuple, Union, Any

import cv2
import numpy as np
import onnxruntime as ort


class TextClassifier:
    """文本方向分类器，用于检测文本方向(0°或180°)并校正"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化文本方向分类器
        
        Args:
            config: 配置参数
        """
        self.image_shape = config.get("image_shape", [3, 48, 192])
        self.batch_num = config.get("batch_num", 6)
        self.thresh = config.get("thresh", 0.9)
        self.label_list = config.get("label_list", ['0', '180'])
        
        # 配置ONNX会话
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("Model path must be provided for text classifier")
            
        # 设置推理选项
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        sess_options.enable_cpu_mem_arena = False
        
        threads = config.get("threads", 0)
        if threads > 0:
            sess_options.intra_op_num_threads = threads
        
        # 创建ONNX运行时会话
        self.session = ort.InferenceSession(model_path, sess_options)
    
    def __call__(
        self, img_list: Union[np.ndarray, List[np.ndarray]]
    ) -> Tuple[List[np.ndarray], List[List[Union[str, float]]], float]:
        """
        分类文本方向并校正
        
        Args:
            img_list: 输入图像列表
            
        Returns:
            校正后的图像列表，分类结果和处理时间
        """
        # 确保输入是列表
        if isinstance(img_list, np.ndarray):
            img_list = [img_list]
            
        # 复制图像列表以避免修改原始数据
        img_list = img_list.copy()
        
        # 计算所有图像的宽高比
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
        
        # 排序可以加速分类过程
        indices = np.argsort(np.array(width_list))
        
        img_num = len(img_list)
        cls_res = [["", 0.0]] * img_num
        elapse = 0
        
        # 分批处理图像
        for beg_img_no in range(0, img_num, self.batch_num):
            end_img_no = min(img_num, beg_img_no + self.batch_num)
            
            norm_img_batch = []
            for idx in range(beg_img_no, end_img_no):
                norm_img = self._resize_norm_img(img_list[indices[idx]])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            
            # 合并批次
            norm_img_batch = np.concatenate(norm_img_batch).astype(np.float32)
            
            # 模型推理
            starttime = time.time()
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: norm_img_batch})
            prob_out = outputs[0]
            
            # 后处理
            cls_result = self._postprocess(prob_out)
            elapse += time.time() - starttime
            
            # 根据分类结果旋转图像
            for rno, (label, score) in enumerate(cls_result):
                cls_res[indices[beg_img_no + rno]] = [label, score]
                if "180" in label and score > self.thresh:
                    img_list[indices[beg_img_no + rno]] = cv2.rotate(
                        img_list[indices[beg_img_no + rno]], cv2.ROTATE_180
                    )
                    
        return img_list, cls_res, elapse
        
    def _resize_norm_img(self, img: np.ndarray) -> np.ndarray:
        """调整图像大小并归一化"""
        img_c, img_h, img_w = self.image_shape
        h, w = img.shape[:2]
        ratio = w / float(h)
        
        # 计算新的宽度，同时保持纵横比
        if math.ceil(img_h * ratio) > img_w:
            resized_w = img_w
        else:
            resized_w = int(math.ceil(img_h * ratio))
        
        # 调整图像大小
        resized_image = cv2.resize(img, (resized_w, img_h))
        resized_image = resized_image.astype("float32")
        
        # 灰度图像和彩色图像处理方式不同
        if img_c == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
            
        # 归一化
        resized_image -= 0.5
        resized_image /= 0.5
        
        # 创建填充图像
        padding_im = np.zeros((img_c, img_h, img_w), dtype=np.float32)
        padding_im[:, :, :resized_w] = resized_image
        return padding_im
        
    def _postprocess(self, preds: np.ndarray) -> List[Tuple[str, float]]:
        """模型输出后处理"""
        # 获取最高概率的类别索引
        pred_idxs = preds.argmax(axis=1)
        
        # 将索引转换为标签和置信度
        decode_out = [
            (self.label_list[idx], preds[i, idx]) for i, idx in enumerate(pred_idxs)
        ]
        
        return decode_out