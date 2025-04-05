# -*- encoding: utf-8 -*-
# 文本识别模块
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2
import numpy as np
import onnxruntime as ort


class TextRecognizer:
    """文本识别器，使用CTC模型识别文本内容"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化文本识别器
        
        Args:
            config: 配置参数
        """
        self.img_shape = config.get("img_shape", [3, 48, 320])
        self.batch_num = config.get("batch_num", 6)
        
        # 配置ONNX会话
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("Model path must be provided for text recognizer")
            
        # 设置推理选项
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        sess_options.enable_cpu_mem_arena = False
        
        threads = config.get("threads", 0)
        if threads > 0:
            sess_options.intra_op_num_threads = threads
        
        # 创建ONNX运行时会话
        self.session = ort.InferenceSession(model_path, sess_options)
        
        # 获取字符集，可能嵌入在模型中或通过外部文件提供
        character_path = config.get("keys_path", None)
        self.character = None
        
        # 尝试从模型元数据中获取字符集
        try:
            metadata = self.session.get_modelmeta().custom_metadata_map
            if 'character' in metadata:
                self.character = metadata['character'].splitlines()
        except:
            pass
            
        # 如果模型中没有字符集，尝试从文件加载
        if self.character is None and character_path:
            with open(character_path, 'rb') as f:
                lines = f.readlines()
                self.character = [line.decode('utf-8').strip('\n').strip('\r\n') for line in lines]
                
        # 如果都没有，使用默认中英文字符集
        if self.character is None:
            # 默认使用中英文字符集，实际应用中应该从外部加载
            self.character = ["blank"] + [chr(i) for i in range(32, 127)] + ["，", "。", "！", "？", "：", "；", """, """, "'", "'"]
            
        # 在字符集中添加空白字符用于CTC解码
        if "blank" not in self.character:
            self.character.insert(0, "blank")
            
        # 添加空格字符
        if " " not in self.character:
            self.character.append(" ")
            
        self.character_dict = {char: i for i, char in enumerate(self.character)}
        
    def __call__(
        self, img_list: Union[np.ndarray, List[np.ndarray]], return_word_box: bool = False
    ) -> Tuple[List[Tuple[str, float]], float]:
        """
        识别文本内容
        
        Args:
            img_list: 输入图像列表
            return_word_box: 是否返回单词级别的边界框
            
        Returns:
            识别结果和处理时间
        """
        # 确保输入是列表
        if isinstance(img_list, np.ndarray):
            img_list = [img_list]
            
        # 计算所有图像的宽高比
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
        
        # 排序可以加速识别过程
        indices = np.argsort(np.array(width_list))
        
        img_num = len(img_list)
        rec_res = [("", 0.0)] * img_num
        elapse = 0
        
        # 分批处理图像
        for beg_img_no in range(0, img_num, self.batch_num):
            end_img_no = min(img_num, beg_img_no + self.batch_num)
            
            # 计算本批次的最大宽高比
            max_wh_ratio = self.img_shape[2] / self.img_shape[1]
            for idx in range(beg_img_no, end_img_no):
                h, w = img_list[indices[idx]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
                
            # 预处理图像
            norm_img_batch = []
            for idx in range(beg_img_no, end_img_no):
                norm_img = self._resize_norm_img(img_list[indices[idx]], max_wh_ratio)
                norm_img_batch.append(norm_img[np.newaxis, :])
                
            norm_img_batch = np.concatenate(norm_img_batch).astype(np.float32)
            
            # 模型推理
            starttime = time.time()
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: norm_img_batch})
            preds = outputs[0]
            
            # CTC解码得到文本
            rec_result = self._ctc_decode(preds)
            elapse += time.time() - starttime
            
            # 整理结果
            for rno, rec in enumerate(rec_result):
                rec_res[indices[beg_img_no + rno]] = rec
                
        return rec_res, elapse
        
    def _resize_norm_img(self, img: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        """调整图像大小并归一化"""
        img_channel, img_height, img_width = self.img_shape
        
        # 确保通道数匹配
        assert img_channel == img.shape[2]
        
        # 根据宽高比调整宽度
        img_width = int(img_height * max_wh_ratio)
        
        h, w = img.shape[:2]
        ratio = w / float(h)
        
        # 计算调整后的宽度，同时保持纵横比
        if math.ceil(img_height * ratio) > img_width:
            resized_w = img_width
        else:
            resized_w = int(math.ceil(img_height * ratio))
            
        # 调整图像大小
        resized_image = cv2.resize(img, (resized_w, img_height))
        resized_image = resized_image.astype(np.float32)  # 明确使用float32类型
        
        # 通道调整和归一化
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0
        resized_image -= np.array([0.5, 0.5, 0.5], dtype=np.float32)[:, np.newaxis, np.newaxis]
        resized_image /= np.array([0.5, 0.5, 0.5], dtype=np.float32)[:, np.newaxis, np.newaxis]
        
        # 创建填充图像
        padding_im = np.zeros((img_channel, img_height, img_width), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im
        
    def _ctc_decode(self, preds: np.ndarray) -> List[Tuple[str, float]]:
        """CTC解码，将预测结果转换为文本"""
        result = []
        
        for i, pred in enumerate(preds):
            # 获取每个时间步的最可能字符
            pred_idx = np.argmax(pred, axis=1)
            
            # 获取每个预测字符的置信度
            pred_prob = np.max(pred, axis=1)
            
            # 合并相同的连续字符
            groups = []
            for idx, prob in zip(pred_idx, pred_prob):
                if len(groups) == 0 or idx != groups[-1][0]:
                    groups.append((idx, [prob]))
                else:
                    groups[-1][1].append(prob)
                    
            # 忽略空白字符 (通常是第0个字符)
            ignored_char = 0  # blank index
            
            # 构建文本
            text = ""
            conf = []
            for idx, probs in groups:
                if idx != ignored_char:
                    text += self.character[idx]
                    conf.append(np.mean(probs))
                    
            # 计算整体置信度
            if len(conf) > 0:
                confidence = np.mean(conf)
            else:
                confidence = 0.0
                
            result.append((text, float(confidence)))
            
        return result