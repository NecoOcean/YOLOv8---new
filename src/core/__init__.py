# src/core/__init__.py
"""
核心业务逻辑模块
- detection_service: 目标检测服务
- statistics_manager: 统计管理
- voice_service: 语音播报服务
"""
from .detection_service import DetectionService, DetectionResult
from .statistics_manager import StatisticsManager
from .voice_service import VoiceService
