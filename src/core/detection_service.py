# src/core/detection_service.py
# -*- coding: utf-8 -*-
"""
基于YOLOv8的垃圾目标检测系统 - 检测服务模块
"""
import time
import numpy as np
from ultralytics import YOLO
from typing import List

from src.config import settings


class DetectionResult:
    """检测结果数据类"""
    
    def __init__(self, results, elapsed_time: float):
        self.raw_results = results
        self.elapsed_time = elapsed_time
        self.locations = []
        self.classes = []
        self.confidences = []
        self.confidence_strings = []
        self._parse_results()
    
    def _parse_results(self):
        """解析YOLO检测结果"""
        if self.raw_results.boxes is not None and len(self.raw_results.boxes) > 0:
            location_list = self.raw_results.boxes.xyxy.tolist()
            self.locations = [list(map(int, bbox)) for bbox in location_list]
            cls_list = self.raw_results.boxes.cls.tolist()
            self.classes = [int(cls) for cls in cls_list]
            conf_list = self.raw_results.boxes.conf.tolist()
            self.confidences = conf_list
            self.confidence_strings = [f'{conf * 100:.2f} %' for conf in conf_list]
    
    @property
    def count(self) -> int:
        return len(self.classes)
    
    @property
    def has_detections(self) -> bool:
        return self.count > 0
    
    def get_plotted_image(self) -> np.ndarray:
        """获取绘制了检测框的图像"""
        return self.raw_results.plot()
    
    def get_classification_guide(self) -> List[dict]:
        """获取分类指导信息"""
        config = settings.get_current_config()
        guides = []
        for cls_id in self.classes:
            guide = config['classification_guide'].get(cls_id, {})
            ch_name = config['CH_names'][cls_id] if cls_id < len(config['CH_names']) else f'类别{cls_id}'
            guides.append({
                'name': ch_name,
                'category': guide.get('category', '未知'),
                'color': guide.get('color', 'gray'),
                'tip': guide.get('tip', '请查阅分类指南')
            })
        return guides


class DetectionService:
    """目标检测服务类"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or str(settings.CURRENT_MODEL_PATH)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载YOLO模型"""
        try:
            self.model = YOLO(self.model_path, task='detect')
            # 预热模型
            self.model(np.zeros((48, 48, 3), dtype=np.uint8))
            print(f"[INFO] 模型加载成功: {self.model_path}")
        except Exception as e:
            print(f"[ERROR] 模型加载失败: {e}")
            raise
    
    def detect(self, source) -> DetectionResult:
        """执行目标检测"""
        start_time = time.time()
        results = self.model(source)[0]
        elapsed_time = time.time() - start_time
        return DetectionResult(results, elapsed_time)
    
    def detect_video_frame(self, frame) -> DetectionResult:
        """检测视频帧"""
        return self.detect(frame)
    
    def reload_model(self, model_path: str = None):
        """重新加载模型"""
        self.model_path = model_path or str(settings.CURRENT_MODEL_PATH)
        self._load_model()
