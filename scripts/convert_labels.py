# convert_labels.py
# -*- coding: utf-8 -*-
"""
标注文件class_id转换脚本
将现有数据集的class_id映射到新的厨房垃圾分类配置
"""
import os
import shutil
from pathlib import Path

# 旧class_id -> 新class_id 映射表
# 基于现有数据集分析结果
OLD_TO_NEW_MAPPING = {
    # 现有数据集中已识别的类别
    0: None,      # 通用垃圾 (fimg/img) - 跳过或映射到其他垃圾
    8: 2,         # Fruitpeels 果皮 -> 新配置 fruit_peel (2)
    10: 10,       # Tealeaves 茶叶 -> 新配置 tea_leaves (10)
    22: None,     # Oldclothes 旧衣服 -> 不在厨房垃圾分类中，跳过
    23: 19,       # Zip-topcan 易拉罐 -> 新配置 zip_top_can (19)
    39: 25,       # Medications 药品 -> 新配置 expired_medicine (25)
}

# 是否保留未映射的类别（映射到其他垃圾）
KEEP_UNMAPPED = True
UNMAPPED_CLASS_ID = 28  # 其他垃圾

def convert_label_file(input_path, output_path):
    """转换单个标注文件"""
    converted_lines = []
    skipped_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) < 5:
            continue
        
        old_class_id = int(parts[0])
        
        # 查找映射
        if old_class_id in OLD_TO_NEW_MAPPING:
            new_class_id = OLD_TO_NEW_MAPPING[old_class_id]
        elif KEEP_UNMAPPED:
            new_class_id = UNMAPPED_CLASS_ID
        else:
            new_class_id = None
        
        if new_class_id is not None:
            parts[0] = str(new_class_id)
            converted_lines.append(' '.join(parts))
        else:
            skipped_count += 1
    
    # 写入转换后的文件
    if converted_lines:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(converted_lines) + '\n')
        return True, len(converted_lines), skipped_count
    return False, 0, skipped_count

def analyze_dataset(labels_dir):
    """分析数据集中的class_id分布"""
    class_counts = {}
    
    for label_file in Path(labels_dir).glob('*.txt'):
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    return class_counts

def main():
    # 数据集路径
    train_labels = Path('datasets/labels/train')
    val_labels = Path('datasets/labels/val')
    
    # 备份目录
    backup_dir = Path('datasets/labels_backup')
    
    # 输出目录（转换后的标注）
    output_train = Path('datasets/labels_converted/train')
    output_val = Path('datasets/labels_converted/val')
    
    print("=" * 60)
    print("标注文件class_id转换工具")
    print("=" * 60)
    
    # 1. 分析现有数据集
    print("\n[1/4] 分析现有数据集...")
    if train_labels.exists():
        train_stats = analyze_dataset(train_labels)
        print(f"训练集class_id分布:")
        for cid, count in sorted(train_stats.items()):
            mapping = OLD_TO_NEW_MAPPING.get(cid, UNMAPPED_CLASS_ID if KEEP_UNMAPPED else '跳过')
            print(f"  class_id {cid}: {count} 个标注 -> 新class_id: {mapping}")
    
    # 2. 创建备份
    print("\n[2/4] 创建备份...")
    if not backup_dir.exists():
        if train_labels.exists():
            shutil.copytree(train_labels, backup_dir / 'train')
        if val_labels.exists():
            shutil.copytree(val_labels, backup_dir / 'val')
        print(f"备份已创建: {backup_dir}")
    else:
        print(f"备份已存在: {backup_dir}")
    
    # 3. 转换标注文件
    print("\n[3/4] 转换标注文件...")
    
    for src_dir, dst_dir in [(train_labels, output_train), (val_labels, output_val)]:
        if not src_dir.exists():
            continue
        
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        total_files = 0
        converted_files = 0
        total_objects = 0
        skipped_objects = 0
        
        for label_file in src_dir.glob('*.txt'):
            total_files += 1
            success, obj_count, skip_count = convert_label_file(
                label_file, dst_dir / label_file.name
            )
            if success:
                converted_files += 1
                total_objects += obj_count
            skipped_objects += skip_count
        
        print(f"\n{src_dir.name}:")
        print(f"  总文件数: {total_files}")
        print(f"  转换成功: {converted_files}")
        print(f"  总标注数: {total_objects}")
        print(f"  跳过标注: {skipped_objects}")
    
    # 4. 输出结果
    print("\n[4/4] 转换完成!")
    print(f"\n转换后的标注文件位于: datasets/labels_converted/")
    print("\n下一步操作:")
    print("1. 检查转换结果是否正确")
    print("2. 如果正确，将 labels_converted 重命名为 labels")
    print("3. 更新 data.yaml 中的 nc 为 29")
    print("4. 运行训练: python train.py")

if __name__ == '__main__':
    main()
