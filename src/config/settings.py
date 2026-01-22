# src/config/settings.py
# -*- coding: utf-8 -*-
"""
基于YOLOv8的垃圾目标检测系统 - 统一配置管理
当前默认支持4类混合垃圾分类模型（分层抽样版本）
"""
import os
from pathlib import Path

# ============ 路径配置 ============
# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = DATA_DIR / 'models'
DATASETS_DIR = DATA_DIR / 'datasets'

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / 'output'
DETECTIONS_DIR = OUTPUT_DIR / 'detections'
STATISTICS_DIR = OUTPUT_DIR / 'statistics'
EXPORTS_DIR = OUTPUT_DIR / 'exports'

# ============ 模型配置 ============
# 预训练模型
PRETRAINED_MODEL_PATH = MODELS_DIR / 'pretrained' / 'yolov8n.pt'

# 4类训练后模型路径
MODEL_4CLS_PATH = MODELS_DIR / 'trained' / 'best_kitchen_mixed_stratified_20260122.pt'

# 当前配置模式 (默认4类)
CURRENT_MODE = 'cls4'
CURRENT_MODEL_PATH = MODEL_4CLS_PATH

# ============ 配置定义 ============

# 4类配置 (Kitchen Mixed Stratified)
CONFIG_4CLS = {
    'id': 'cls4',
    'NUM_CLASSES': 4,
    'names': {
        0: 'kitchen_waste',
        1: 'recyclable',
        2: 'hazardous',
        3: 'other',
    },
    'CH_names': ['厨余垃圾', '可回收物', '有害垃圾', '其他垃圾'],
    'classification_guide': {
        0: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'},
        1: {'category': '可回收物', 'color': 'blue', 'tip': '请清洗后投入蓝色可回收垃圾桶'},
        2: {'category': '有害垃圾', 'color': 'red', 'tip': '请投入红色有害垃圾桶'},
        3: {'category': '其他垃圾', 'color': 'gray', 'tip': '请投入灰色其他垃圾桶'},
    },
    'model_path': MODEL_4CLS_PATH,
    'data_yaml': DATASETS_DIR / 'kitchen_mixed_stratified' / 'data.yaml',
    'description': '4类混合垃圾分类模型（分层抽样优化版）',
}

# ============ 配置注册表 (支持未来扩展) ============
# 后续添加新配置只需在此定义并注册即可
CONFIG_REGISTRY = {
    'cls4': CONFIG_4CLS,
}

def get_current_config():
    """获取当前激活模式的配置字典"""
    return CONFIG_REGISTRY.get(CURRENT_MODE, CONFIG_4CLS)


def set_mode(mode: str):
    """切换系统配置模式"""
    global CURRENT_MODE, CURRENT_MODEL_PATH
    if mode in CONFIG_REGISTRY:
        CURRENT_MODE = mode
        CURRENT_MODEL_PATH = CONFIG_REGISTRY[mode]['model_path']
    else:
        # 默认回退
        CURRENT_MODE = 'cls4'
        CURRENT_MODEL_PATH = MODEL_4CLS_PATH

# 导出当前配置变量（向后兼容旧代码）
_config = get_current_config()
NUM_CLASSES = _config['NUM_CLASSES']
names = _config['names']
CH_names = _config['CH_names']
classification_guide = _config['classification_guide']
model_path = str(_config['model_path'])
save_path = str(DETECTIONS_DIR)

# ============ 视觉外观配置 ============
CATEGORY_COLORS = {
    '厨余垃圾': '#4CAF50',  # 绿色
    '可回收物': '#2196F3',  # 蓝色
    '有害垃圾': '#F44336',  # 红色
    '其他垃圾': '#9E9E9E'   # 灰色
}
