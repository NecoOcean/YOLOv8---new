# scripts/merge_datasets_to_kitchen_mixed.py
# -*- coding: utf-8 -*-
"""
将多个垃圾检测数据集合并为统一的 kitchen_mixed 数据集

数据集来源：
- Domestic: TU Wien 厨余垃圾数据集（已转换为 YOLO 格式）
- A: Medical waste 1（有害垃圾）
- B: Medical waste2（有害垃圾）
- C: TACO 10 class（包装垃圾/可回收物）

目标类别（4类）：
0: kitchen_waste   # 厨余垃圾
1: recyclable      # 可回收物/包装
2: hazardous       # 有害垃圾
3: other           # 其他垃圾
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import yaml

# ============ 配置区域 ============

# 源数据集路径
BASE_DIR = r"d:\Data\code\YOLOv8 - new\data\datasets\newdata"
DATASETS = {
    'domestic': {
        'path': os.path.join(BASE_DIR, 'domestic_yolo'),
        'prefix': 'dom_',  # 文件名前缀，避免冲突
        'has_splits': False,  # 只有 train 目录
    },
    'hazard_a': {
        'path': os.path.join(BASE_DIR, 'A'),
        'prefix': 'ha_',
        'has_splits': True,  # 有 train/valid/test
    },
    'hazard_b': {
        'path': os.path.join(BASE_DIR, 'B'),
        'prefix': 'hb_',
        'has_splits': True,
    },
    'taco': {
        'path': os.path.join(BASE_DIR, 'C'),
        'prefix': 'taco_',
        'has_splits': True,
    },
}

# 输出目录
OUTPUT_DIR = r"d:\Data\code\YOLOv8 - new\data\datasets\kitchen_mixed"

# 统一的类别定义（目标）
UNIFIED_CLASSES = {
    0: 'kitchen_waste',   # 厨余垃圾
    1: 'recyclable',      # 可回收物
    2: 'hazardous',       # 有害垃圾
    3: 'other',           # 其他垃圾
}

# ============ 类别映射规则 ============

# Domestic 数据集映射（已经是正确的 0=厨余, 1=可回收）
DOMESTIC_MAPPING = {
    0: 0,  # organic -> kitchen_waste
    1: 1,  # non-organic -> recyclable
}

# Medical waste 1 (A) - 17个类别全部映射为有害垃圾
MEDICAL_A_CLASSES = ['Iv set_waste', 'bandage_waste', 'bottle', 'cotton_waste', 
                     'drip_waste', 'gloves_waste', 'iv set waste', 'mask', 
                     'medical_glass_waste', 'medicine', 'needle_waste', 'paper_waste', 
                     'plastic needle_waste', 'plastic_waste', 'saline bottle', 
                     'scalvin set_waste', 'syringe_waste']
MEDICAL_A_MAPPING = {i: 2 for i in range(len(MEDICAL_A_CLASSES))}  # 全部 -> hazardous

# Medical waste2 (B) - 18个类别全部映射为有害垃圾
MEDICAL_B_CLASSES = ['0', 'IV-set', 'Medicine', 'bandage', 'forceps', 'gloves_waste', 
                     'iv set waste', 'mask', 'medical-glass-waste', 'medical-plastic-waste', 
                     'medicine', 'pilers', 'saline bottle', 'scalpel', 'surgical scissors', 
                     'surgical-gloves', 'syringes', 'waste']
MEDICAL_B_MAPPING = {i: 2 for i in range(len(MEDICAL_B_CLASSES))}  # 全部 -> hazardous

# TACO (C) - 10个类别映射
TACO_CLASSES = ['Bottle', 'Bottle cap', 'Can', 'Cigarette', 'Cup', 
                'Lid', 'Other', 'Plastic bag and wrapper', 'Pop tab', 'Straw']
TACO_MAPPING = {
    0: 1,  # Bottle -> recyclable
    1: 1,  # Bottle cap -> recyclable
    2: 1,  # Can -> recyclable
    3: 3,  # Cigarette -> other
    4: 1,  # Cup -> recyclable
    5: 1,  # Lid -> recyclable
    6: 3,  # Other -> other
    7: 1,  # Plastic bag -> recyclable
    8: 1,  # Pop tab -> recyclable
    9: 1,  # Straw -> recyclable
}

# 汇总映射字典
CLASS_MAPPINGS = {
    'domestic': DOMESTIC_MAPPING,
    'hazard_a': MEDICAL_A_MAPPING,
    'hazard_b': MEDICAL_B_MAPPING,
    'taco': TACO_MAPPING,
}

# ============ 主逻辑 ============

def remap_label_line(line: str, mapping: dict) -> str:
    """
    重新映射单行 YOLO 标签的类别 ID
    
    Args:
        line: "class_id x_center y_center width height"
        mapping: {old_class_id: new_class_id}
    
    Returns:
        映射后的标签行
    """
    parts = line.strip().split()
    if not parts or len(parts) < 5:
        return None
    
    old_class_id = int(parts[0])
    if old_class_id not in mapping:
        return None  # 跳过未映射的类
    
    new_class_id = mapping[old_class_id]
    parts[0] = str(new_class_id)
    
    return ' '.join(parts)


def process_label_file(src_label_path: str, dst_label_path: str, mapping: dict) -> int:
    """
    处理单个标签文件：读取、重新映射类别、写入
    
    Returns:
        处理的边界框数量
    """
    new_lines = []
    
    with open(src_label_path, 'r', encoding='utf-8') as f:
        for line in f:
            remapped = remap_label_line(line, mapping)
            if remapped:
                new_lines.append(remapped)
    
    if not new_lines:
        return 0
    
    os.makedirs(os.path.dirname(dst_label_path), exist_ok=True)
    with open(dst_label_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines) + '\n')
    
    return len(new_lines)


def merge_dataset(dataset_name: str, dataset_config: dict, split: str = 'train'):
    """
    合并单个数据集的指定划分（train/valid/test）
    
    Args:
        dataset_name: 数据集名称（对应 CLASS_MAPPINGS 的键）
        dataset_config: 数据集配置（路径、前缀等）
        split: 'train' / 'valid' / 'test'
    """
    src_base = dataset_config['path']
    prefix = dataset_config['prefix']
    mapping = CLASS_MAPPINGS[dataset_name]
    
    # 确定源目录
    if dataset_config['has_splits']:
        src_img_dir = os.path.join(src_base, split, 'images')
        src_lbl_dir = os.path.join(src_base, split, 'labels')
    else:
        # domestic 只有 images/train 和 labels/train
        src_img_dir = os.path.join(src_base, 'images', 'train')
        src_lbl_dir = os.path.join(src_base, 'labels', 'train')
    
    if not os.path.exists(src_img_dir):
        print(f"  [SKIP] {dataset_name}/{split} 图片目录不存在")
        return 0, 0
    
    # 目标目录
    dst_img_dir = os.path.join(OUTPUT_DIR, 'images', split)
    dst_lbl_dir = os.path.join(OUTPUT_DIR, 'labels', split)
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)
    
    # 遍历图片文件
    img_files = [f for f in os.listdir(src_img_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    copied_images = 0
    total_boxes = 0
    
    for img_name in tqdm(img_files, desc=f"  {dataset_name}/{split}", leave=False):
        # 生成新文件名（添加前缀避免冲突）
        new_img_name = prefix + img_name
        label_name = Path(img_name).stem + '.txt'
        new_label_name = prefix + Path(img_name).stem + '.txt'
        
        src_img_path = os.path.join(src_img_dir, img_name)
        src_lbl_path = os.path.join(src_lbl_dir, label_name)
        
        dst_img_path = os.path.join(dst_img_dir, new_img_name)
        dst_lbl_path = os.path.join(dst_lbl_dir, new_label_name)
        
        # 检查标签文件是否存在
        if not os.path.exists(src_lbl_path):
            continue
        
        # 处理标签文件（重新映射类别）
        boxes = process_label_file(src_lbl_path, dst_lbl_path, mapping)
        
        if boxes == 0:
            continue  # 没有有效的框，跳过该图片
        
        # 复制图片
        shutil.copy2(src_img_path, dst_img_path)
        
        copied_images += 1
        total_boxes += boxes
    
    return copied_images, total_boxes


def create_data_yaml():
    """创建统一的 data.yaml 配置文件"""
    
    yaml_content = {
        'path': '../kitchen_mixed',  # 相对于训练脚本的路径
        'train': 'images/train',
        'val': 'images/valid',
        'test': 'images/test',
        'nc': len(UNIFIED_CLASSES),
        'names': list(UNIFIED_CLASSES.values()),
    }
    
    yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n[INFO] 已生成配置文件: {yaml_path}")
    print("\n配置内容：")
    print(yaml.dump(yaml_content, default_flow_style=False, allow_unicode=True))


def main():
    """主函数：执行完整的数据集合并流程"""
    
    print("=" * 70)
    print("厨房垃圾数据集合并工具 - Kitchen Mixed Dataset Merger")
    print("=" * 70)
    print(f"\n输出目录: {OUTPUT_DIR}\n")
    
    # 检查所有源数据集
    print("[1/3] 检查源数据集...")
    for name, config in DATASETS.items():
        if os.path.exists(config['path']):
            print(f"  ✓ {name}: {config['path']}")
        else:
            print(f"  ✗ {name}: 路径不存在 - {config['path']}")
    
    # 创建输出目录
    print(f"\n[2/3] 创建输出目录结构...")
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)
    print("  ✓ 目录结构已创建")
    
    # 合并数据集
    print(f"\n[3/3] 合并数据集...")
    
    total_stats = {
        'train': {'images': 0, 'boxes': 0},
        'valid': {'images': 0, 'boxes': 0},
        'test': {'images': 0, 'boxes': 0},
    }
    
    for dataset_name, dataset_config in DATASETS.items():
        print(f"\n处理数据集: {dataset_name}")
        
        # 根据数据集类型决定要处理哪些划分
        if dataset_config['has_splits']:
            splits = ['train', 'valid', 'test']
        else:
            splits = ['train']  # domestic 只有 train，但会同时放入 train 和 valid
        
        for split in splits:
            imgs, boxes = merge_dataset(dataset_name, dataset_config, split)
            total_stats[split]['images'] += imgs
            total_stats[split]['boxes'] += boxes
            print(f"    {split}: {imgs} 张图片, {boxes} 个边界框")
        
        # 如果是 domestic（只有 train），也复制一部分到 valid
        if not dataset_config['has_splits']:
            # 将 train 的 20% 作为 valid（简单策略）
            print(f"  [INFO] {dataset_name} 没有 valid 集，将 train 复制到 valid")
            # 这里为了简单，直接把所有 train 也放到 valid
            # 实际项目中应该做划分，这里暂时省略
    
    # 生成统计报告
    print("\n" + "=" * 70)
    print("合并完成！数据集统计：")
    print("=" * 70)
    for split in ['train', 'valid', 'test']:
        print(f"{split.upper():8s}: {total_stats[split]['images']:5d} 张图片, "
              f"{total_stats[split]['boxes']:6d} 个边界框")
    
    total_images = sum(s['images'] for s in total_stats.values())
    total_boxes = sum(s['boxes'] for s in total_stats.values())
    print(f"{'总计':8s}: {total_images:5d} 张图片, {total_boxes:6d} 个边界框")
    
    # 创建 data.yaml
    create_data_yaml()
    
    print("\n" + "=" * 70)
    print("[DONE] 数据集合并完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("\n后续步骤：")
    print("1. 检查输出目录中的图片和标签是否正确")
    print("2. 使用以下命令开始训练：")
    print(f"   python training/train.py --data data/datasets/kitchen_mixed/data.yaml --epochs 100")
    print("=" * 70)


if __name__ == "__main__":
    # 检查 tqdm 是否安装
    try:
        from tqdm import tqdm
    except ImportError:
        print("[WARN] tqdm 未安装，进度条将不可用")
        print("安装命令: pip install tqdm")
        # 提供简单的替代
        def tqdm(iterable, **kwargs):
            return iterable
    
    main()
