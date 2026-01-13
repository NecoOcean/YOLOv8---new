# prepare_dataset.py
# -*- coding: utf-8 -*-
"""
数据集整合与准备脚本
基于现有数据集创建厨房垃圾分类训练数据
"""
import os
import shutil
from pathlib import Path
from collections import defaultdict

# 现有数据集中实际存在的类别及其映射
# 原class_id -> (新class_id, 中文名, 英文名, 垃圾分类)
EXISTING_CLASSES = {
    8: (0, '果皮', 'fruit_peel', '厨余垃圾'),
    10: (1, '茶叶渣', 'tea_leaves', '厨余垃圾'),
    23: (2, '易拉罐', 'zip_top_can', '可回收物'),
    39: (3, '过期药品', 'expired_medicine', '有害垃圾'),
}

# 可选：将其他类别映射到"其他垃圾"
INCLUDE_OTHER = True
OTHER_CLASS_ID = 4
OTHER_CLASS_NAME = ('其他垃圾', 'other_garbage', '其他垃圾')

def analyze_labels(labels_dir):
    """分析标注文件中的类别分布"""
    class_counts = defaultdict(int)
    file_counts = defaultdict(int)
    
    for label_file in Path(labels_dir).glob('*.txt'):
        classes_in_file = set()
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    classes_in_file.add(class_id)
        
        for cls in classes_in_file:
            file_counts[cls] += 1
    
    return class_counts, file_counts

def convert_and_copy(src_labels, src_images, dst_labels, dst_images):
    """转换标注并复制文件"""
    dst_labels.mkdir(parents=True, exist_ok=True)
    dst_images.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total_files': 0,
        'converted_files': 0,
        'total_objects': 0,
        'class_counts': defaultdict(int)
    }
    
    for label_file in src_labels.glob('*.txt'):
        stats['total_files'] += 1
        converted_lines = []
        
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                old_class_id = int(parts[0])
                new_class_id = None
                
                # 检查是否是已知类别
                if old_class_id in EXISTING_CLASSES:
                    new_class_id = EXISTING_CLASSES[old_class_id][0]
                elif INCLUDE_OTHER:
                    new_class_id = OTHER_CLASS_ID
                
                if new_class_id is not None:
                    parts[0] = str(new_class_id)
                    converted_lines.append(' '.join(parts))
                    stats['class_counts'][new_class_id] += 1
                    stats['total_objects'] += 1
        
        # 只保存有有效标注的文件
        if converted_lines:
            # 保存标注
            with open(dst_labels / label_file.name, 'w', encoding='utf-8') as f:
                f.write('\n'.join(converted_lines) + '\n')
            
            # 复制对应图片
            img_name = label_file.stem
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                img_file = src_images / (img_name + ext)
                if img_file.exists():
                    shutil.copy2(img_file, dst_images / img_file.name)
                    break
            
            stats['converted_files'] += 1
    
    return stats

def generate_config_file(output_path):
    """生成新的配置文件"""
    # 构建类别配置
    names = {}
    ch_names = []
    classification_guide = {}
    
    for old_id, (new_id, ch_name, en_name, category) in EXISTING_CLASSES.items():
        names[new_id] = en_name
        ch_names.append(ch_name)
        
        color_map = {
            '厨余垃圾': 'green',
            '可回收物': 'blue', 
            '有害垃圾': 'red',
            '其他垃圾': 'gray'
        }
        tip_map = {
            '厨余垃圾': '请投入绿色厨余垃圾桶',
            '可回收物': '请清洗后投入蓝色可回收垃圾桶',
            '有害垃圾': '请投入红色有害垃圾桶',
            '其他垃圾': '请投入灰色其他垃圾桶'
        }
        classification_guide[new_id] = {
            'category': category,
            'color': color_map[category],
            'tip': tip_map[category]
        }
    
    if INCLUDE_OTHER:
        names[OTHER_CLASS_ID] = OTHER_CLASS_NAME[1]
        ch_names.append(OTHER_CLASS_NAME[0])
        classification_guide[OTHER_CLASS_ID] = {
            'category': '其他垃圾',
            'color': 'gray',
            'tip': '请投入灰色其他垃圾桶'
        }
    
    num_classes = len(names)
    
    config_content = f'''# Config.py
# -*- coding: utf-8 -*-
"""
基于YOLOv8的垃圾目标检测算法 - 配置文件
基于现有数据集整理的{num_classes}类别配置
"""

# 图片及视频检测结果保存路径
save_path = 'save_data'

# 使用的模型路径
model_path = 'models/best.pt'

# 类别数量
NUM_CLASSES = {num_classes}

# 类别配置（基于现有数据集）
names = {repr(names)}

# 中文类别名称
CH_names = {repr(ch_names)}

# 垃圾分类指导映射
classification_guide = {repr(classification_guide)}
'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    return num_classes

def generate_data_yaml(output_path, num_classes):
    """生成data.yaml配置文件"""
    names_list = []
    for old_id, (new_id, ch_name, en_name, _) in sorted(EXISTING_CLASSES.items(), key=lambda x: x[1][0]):
        names_list.append(f"  {new_id}: {en_name}  # {ch_name}")
    
    if INCLUDE_OTHER:
        names_list.append(f"  {OTHER_CLASS_ID}: {OTHER_CLASS_NAME[1]}  # {OTHER_CLASS_NAME[0]}")
    
    yaml_content = f'''# 厨房垃圾分类数据集配置
# 基于现有数据集整理

path: datasets/kitchen_garbage
train: images/train
val: images/val

nc: {num_classes}

names:
{chr(10).join(names_list)}
'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

def main():
    print("=" * 60)
    print("厨房垃圾分类数据集整合工具")
    print("=" * 60)
    
    # 源数据路径
    src_train_labels = Path('datasets/labels/train')
    src_train_images = Path('datasets/images/train')
    src_val_labels = Path('datasets/labels/val')
    src_val_images = Path('datasets/images/val')
    
    # 目标数据路径
    dst_base = Path('datasets/kitchen_garbage')
    dst_train_labels = dst_base / 'labels' / 'train'
    dst_train_images = dst_base / 'images' / 'train'
    dst_val_labels = dst_base / 'labels' / 'val'
    dst_val_images = dst_base / 'images' / 'val'
    
    # 1. 分析源数据
    print("\n[1/4] 分析源数据集...")
    if src_train_labels.exists():
        class_counts, file_counts = analyze_labels(src_train_labels)
        print(f"训练集类别分布（标注数 / 文件数）:")
        for cls_id in sorted(class_counts.keys()):
            mapping = EXISTING_CLASSES.get(cls_id, (OTHER_CLASS_ID, '其他', 'other', '其他'))
            status = "[OK]" if cls_id in EXISTING_CLASSES else "->other"
            print(f"  class {cls_id}: {class_counts[cls_id]:>5} / {file_counts[cls_id]:>5} {status}")
    
    # 2. 转换训练集
    print("\n[2/4] 转换训练集...")
    if src_train_labels.exists() and src_train_images.exists():
        train_stats = convert_and_copy(
            src_train_labels, src_train_images,
            dst_train_labels, dst_train_images
        )
        print(f"  文件: {train_stats['converted_files']}/{train_stats['total_files']}")
        print(f"  标注: {train_stats['total_objects']}")
        print(f"  类别分布: {dict(train_stats['class_counts'])}")
    
    # 3. 转换验证集
    print("\n[3/4] 转换验证集...")
    if src_val_labels.exists() and src_val_images.exists():
        val_stats = convert_and_copy(
            src_val_labels, src_val_images,
            dst_val_labels, dst_val_images
        )
        print(f"  文件: {val_stats['converted_files']}/{val_stats['total_files']}")
        print(f"  标注: {val_stats['total_objects']}")
    
    # 4. 生成配置文件
    print("\n[4/4] 生成配置文件...")
    num_classes = generate_config_file('Config_kitchen.py')
    generate_data_yaml(dst_base / 'data.yaml', num_classes)
    print(f"  Config_kitchen.py - {num_classes}个类别")
    print(f"  {dst_base}/data.yaml")
    
    print("\n" + "=" * 60)
    print("数据集整合完成!")
    print("=" * 60)
    print(f"\n新数据集位置: {dst_base}")
    print(f"类别数量: {num_classes}")
    print("\n类别映射:")
    for old_id, (new_id, ch_name, en_name, category) in EXISTING_CLASSES.items():
        print(f"  {new_id}: {ch_name} ({en_name}) - {category}")
    if INCLUDE_OTHER:
        print(f"  {OTHER_CLASS_ID}: 其他垃圾 (other_garbage) - 其他垃圾")
    
    print("\n下一步操作:")
    print("1. cp Config_kitchen.py Config.py  # 使用新配置")
    print("2. 修改train.py中的data参数为 'datasets/kitchen_garbage/data.yaml'")
    print("3. python train.py  # 开始训练")

if __name__ == '__main__':
    main()
