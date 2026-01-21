# src/config/settings.py
# -*- coding: utf-8 -*-
"""
基于YOLOv8的垃圾目标检测系统 - 统一配置管理
支持5类和40类配置动态切换
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

# 训练后模型
MODEL_5CLS_PATH = MODELS_DIR / 'trained' / 'best_5cls.pt'
MODEL_23CLS_PATH = MODELS_DIR / 'trained' / 'best_23cls.pt'
MODEL_40CLS_PATH = MODELS_DIR / 'trained' / 'best_40cls.pt'
MODEL_4CLS_PATH = MODELS_DIR / 'trained' / 'best_4cls.pt'

# 当前使用的模型（默认23类）
CURRENT_MODEL_PATH = MODEL_23CLS_PATH

# ============ 当前配置模式 ============
# 'cls4', 'cls5', 'cls23' 或 'cls40'
CURRENT_MODE = 'cls23'

# ============ 5类配置 ============
CONFIG_5CLS = {
    'NUM_CLASSES': 5,
    'names': {
        0: 'fruit_peel',
        1: 'tea_leaves',
        2: 'zip_top_can',
        3: 'expired_medicine',
        4: 'other_garbage'
    },
    'CH_names': ['果皮', '茶叶渣', '易拉罐', '过期药品', '其他垃圾'],
    'classification_guide': {
        0: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'},
        1: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'},
        2: {'category': '可回收物', 'color': 'blue', 'tip': '请清洗后投入蓝色可回收垃圾桶'},
        3: {'category': '有害垃圾', 'color': 'red', 'tip': '请投入红色有害垃圾桶'},
        4: {'category': '其他垃圾', 'color': 'gray', 'tip': '请投入灰色其他垃圾桶'}
    },
    'model_path': MODEL_5CLS_PATH,
    'data_yaml': DATASETS_DIR / 'kitchen_garbage' / 'data.yaml'
}

# ============ 4类配置 ============
CONFIG_4CLS = {
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
    'description': '4类混合垃圾分类模型（kitchen_mixed_stratified 分层抽样版本）',
}

# ============ 23类配置 ============
CONFIG_23CLS = {
    'NUM_CLASSES': 23,
    'names': {
        0: 'vegetable', 1: 'fruit_peel', 2: 'fruit_core', 3: 'bone',
        4: 'fish_bone', 5: 'eggshell', 6: 'rice', 7: 'noodle',
        8: 'bread', 9: 'meat', 10: 'fish', 11: 'leftover',
        12: 'plastic_bottle', 13: 'plastic_bag', 14: 'plastic_container',
        15: 'glass_bottle', 16: 'metal_can', 17: 'paper_box',
        18: 'paper', 19: 'aluminum_foil', 20: 'battery',
        21: 'cigarette', 22: 'other_waste'
    },
    'CH_names': [
        '蔬菜', '果皮', '果核', '骨头', '鱼骨', '蛋壳',
        '米饭', '面条', '面包', '肉类', '鱼类', '剩饭剩菜',
        '塑料瓶', '塑料袋', '塑料容器', '玻璃瓶', '金属罐',
        '纸盒', '纸张', '铝箔', '电池', '烟头', '其他垃圾'
    ],
    'classification_guide': {
        # 厨余垃圾 (0-11): 蔬菜、果皮、果核、骨头、鱼骨、蛋壳、米饭、面条、面包、肉类、鱼类、剩饭
        **{i: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'} for i in range(12)},
        # 可回收物 (12-19): 塑料瓶、塑料袋、塑料容器、玻璃瓶、金属罐、纸盒、纸张、铝箔
        **{i: {'category': '可回收物', 'color': 'blue', 'tip': '请清洗后投入蓝色可回收垃圾桶'} for i in range(12, 20)},
        # 有害垃圾 (20): 电池
        20: {'category': '有害垃圾', 'color': 'red', 'tip': '请投入红色有害垃圾桶'},
        # 其他垃圾 (21-22): 烟头、其他垃圾
        21: {'category': '其他垃圾', 'color': 'gray', 'tip': '请投入灰色其他垃圾桶'},
        22: {'category': '其他垃圾', 'color': 'gray', 'tip': '请投入灰色其他垃圾桶'},
    },
    'model_path': MODEL_23CLS_PATH,
    'data_yaml': DATASETS_DIR / 'kitchen_garbage_merged' / 'data.yaml'
}

# ============ 40类配置 ============
CONFIG_40CLS = {
    'NUM_CLASSES': 40,
    'names': {
        0: 'plastic_container_1', 1: 'plastic_container_2', 2: 'plastic_box',
        3: 'plastic_tray', 4: 'plastic_cup', 5: 'plastic_bottle',
        6: 'plastic_bag', 7: 'plastic_wrap',
        8: 'fruit_peel', 9: 'vegetable_waste', 10: 'tea_leaves',
        11: 'food_residue', 12: 'bone', 13: 'eggshell',
        14: 'paper_box', 15: 'carton', 16: 'paper_bag',
        17: 'newspaper', 18: 'tissue',
        19: 'metal_can_1', 20: 'metal_can_2', 21: 'aluminum_foil',
        22: 'tin_can', 23: 'zip_top_can',
        24: 'glass_bottle', 25: 'glass_jar',
        26: 'disposable_chopsticks', 27: 'disposable_tableware', 28: 'cigarette_butt',
        29: 'straw', 30: 'toothpick',
        31: 'broken_ceramic', 32: 'dust_debris', 33: 'worn_fabric',
        34: 'rubber_band', 35: 'pen', 36: 'battery_shell',
        37: 'battery', 38: 'chemical_bottle', 39: 'expired_medicine'
    },
    'CH_names': [
        '塑料容器1', '塑料容器2', '塑料盒', '塑料托盘', '塑料杯', '塑料瓶', '塑料袋', '保鲜膜',
        '果皮', '蔬菜残渣', '茶叶渣', '食物残渣', '骨头', '蛋壳',
        '纸盒', '纸箱', '纸袋', '报纸', '餐巾纸',
        '金属罐1', '金属罐2', '铝箔', '罐头盒', '易拉罐',
        '玻璃瓶', '玻璃罐',
        '一次性筷子', '一次性餐具', '烟头', '吸管', '牙签',
        '碎陶瓷', '灰尘杂物', '破旧织物', '橡皮筋', '笔', '电池外壳',
        '电池', '化学品瓶', '过期药品'
    ],
    'classification_guide': {
        # 可回收物 - 塑料 (0-7)
        **{i: {'category': '可回收物', 'color': 'blue', 'tip': '请清洗后投入蓝色可回收垃圾桶'} for i in range(8)},
        # 厨余垃圾 (8-13)
        **{i: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'} for i in range(8, 14)},
        # 可回收物 - 纸类 (14-17)
        **{i: {'category': '可回收物', 'color': 'blue', 'tip': '请投入蓝色可回收垃圾桶'} for i in range(14, 18)},
        # 其他垃圾 - 污染餐巾纸 (18)
        18: {'category': '其他垃圾', 'color': 'gray', 'tip': '污染的餐巾纸请投入灰色其他垃圾桶'},
        # 可回收物 - 金属 (19-23)
        **{i: {'category': '可回收物', 'color': 'blue', 'tip': '请清洗后投入蓝色可回收垃圾桶'} for i in range(19, 24)},
        # 可回收物 - 玻璃 (24-25)
        **{i: {'category': '可回收物', 'color': 'blue', 'tip': '请清洗后轻放入蓝色可回收垃圾桶'} for i in range(24, 26)},
        # 其他垃圾 - 一次性用品 (26-30)
        **{i: {'category': '其他垃圾', 'color': 'gray', 'tip': '请投入灰色其他垃圾桶'} for i in range(26, 31)},
        # 其他垃圾 - 杂项 (31-36)
        **{i: {'category': '其他垃圾', 'color': 'gray', 'tip': '请投入灰色其他垃圾桶'} for i in range(31, 37)},
        # 有害垃圾 (37-39)
        **{i: {'category': '有害垃圾', 'color': 'red', 'tip': '请投入红色有害垃圾桶'} for i in range(37, 40)},
    },
    'model_path': MODEL_40CLS_PATH,
    'data_yaml': DATASETS_DIR / 'data_40cls.yaml'
}

# ============ 当前配置 ============

def get_current_config():
    """获取当前配置"""
    if CURRENT_MODE == 'cls40':
        return CONFIG_40CLS
    elif CURRENT_MODE == 'cls23':
        return CONFIG_23CLS
    elif CURRENT_MODE == 'cls5':
        return CONFIG_5CLS
    elif CURRENT_MODE == 'cls4':
        return CONFIG_4CLS
    return CONFIG_23CLS


def set_mode(mode: str):
    """设置配置模式: 'cls4', 'cls5', 'cls23' 或 'cls40'"""
    global CURRENT_MODE, CURRENT_MODEL_PATH
    if mode == 'cls40':
        CURRENT_MODE = 'cls40'
        CURRENT_MODEL_PATH = MODEL_40CLS_PATH
    elif mode == 'cls23':
        CURRENT_MODE = 'cls23'
        CURRENT_MODEL_PATH = MODEL_23CLS_PATH
    elif mode == 'cls5':
        CURRENT_MODE = 'cls5'
        CURRENT_MODEL_PATH = MODEL_5CLS_PATH
    elif mode == 'cls4':
        CURRENT_MODE = 'cls4'
        CURRENT_MODEL_PATH = MODEL_4CLS_PATH
    else:
        # 默认回退到23类配置
        CURRENT_MODE = 'cls23'
        CURRENT_MODEL_PATH = MODEL_23CLS_PATH

# 导出当前配置变量（向后兼容）
_config = get_current_config()
NUM_CLASSES = _config['NUM_CLASSES']
names = _config['names']
CH_names = _config['CH_names']
classification_guide = _config['classification_guide']
model_path = str(_config['model_path'])
save_path = str(DETECTIONS_DIR)

# ============ 四大类颜色 ============
CATEGORY_COLORS = {
    '厨余垃圾': '#4CAF50',  # 绿色
    '可回收物': '#2196F3',  # 蓝色
    '有害垃圾': '#F44336',  # 红色
    '其他垃圾': '#9E9E9E'   # 灰色
}
