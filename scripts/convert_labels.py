# convert_labels.py
# -*- coding: utf-8 -*-
"""
标注文件class_id转换脚本
将40类数据集的class_id映射到29类厨房垃圾分类配置

40类配置 (data_40cls.yaml) -> 29类配置 (data_29cls.yaml)
"""
import os
import shutil
from pathlib import Path

# ============================================================
# 40类 -> 29类 完整映射表
# ============================================================
# 29类目标配置：
#   0: vegetable_leaves (菜叶)      13: plastic_bag (食品袋)
#   1: vegetable_roots (菜根)       14: plastic_wrap (保鲜膜)
#   2: fruit_peel (果皮)            15: plastic_container (塑料盒)
#   3: fruit_core (果核)            16: paper_box (纸盒)
#   4: bone (骨头)                  17: wrapping_paper (包装纸)
#   5: meat_skin (肉皮)             18: tissue (餐巾纸)
#   6: offal (内脏)                 19: zip_top_can (易拉罐)
#   7: rice (米饭)                  20: tin_can (罐头盒)
#   8: noodles (面条)               21: aluminum_foil (铝箔)
#   9: bread_crumbs (面包屑)        22: seasoning_bottle (调料瓶)
#  10: tea_leaves (茶叶渣)          23: wine_bottle (酒瓶)
#  11: coffee_grounds (咖啡渣)      24: expired_seasoning (过期调料)
#  12: eggshell (蛋壳)              25: expired_medicine (过期药品)
#                                   26: cleaner_container (清洁剂容器)
#                                   27: battery (电池)
#                                   28: other_garbage (其他垃圾)
# ============================================================

OLD_TO_NEW_MAPPING = {
    # ===== 可回收物 - 塑料制品 (40类: 0-7) =====
    0: 15,     # plastic_container_1 -> plastic_container
    1: 15,     # plastic_container_2 -> plastic_container
    2: 15,     # plastic_box -> plastic_container
    3: 15,     # plastic_tray -> plastic_container
    4: 15,     # plastic_cup -> plastic_container
    5: 15,     # plastic_bottle -> plastic_container
    6: 13,     # plastic_bag -> plastic_bag
    7: 14,     # plastic_wrap -> plastic_wrap
    
    # ===== 厨余垃圾 - 食物类 (40类: 8-13) =====
    8: 2,      # fruit_peel -> fruit_peel
    9: 0,      # vegetable_waste -> vegetable_leaves
    10: 10,    # tea_leaves -> tea_leaves
    11: 28,    # food_residue -> other_garbage (无直接对应)
    12: 4,     # bone -> bone
    13: 12,    # eggshell -> eggshell
    
    # ===== 可回收物 - 纸类 (40类: 14-18) =====
    14: 16,    # paper_box -> paper_box
    15: 16,    # carton -> paper_box
    16: 17,    # paper_bag -> wrapping_paper
    17: 17,    # newspaper -> wrapping_paper
    18: 18,    # tissue -> tissue
    
    # ===== 可回收物 - 金属 (40类: 19-23) =====
    19: 20,    # metal_can_1 -> tin_can
    20: 20,    # metal_can_2 -> tin_can
    21: 21,    # aluminum_foil -> aluminum_foil
    22: 20,    # tin_can -> tin_can
    23: 19,    # zip_top_can -> zip_top_can
    
    # ===== 可回收物 - 玻璃 (40类: 24-25) =====
    24: 23,    # glass_bottle -> wine_bottle
    25: 22,    # glass_jar -> seasoning_bottle
    
    # ===== 其他垃圾 - 一次性用品 (40类: 26-30) =====
    26: 28,    # disposable_chopsticks -> other_garbage
    27: 28,    # disposable_tableware -> other_garbage
    28: 28,    # cigarette_butt -> other_garbage
    29: 28,    # straw -> other_garbage
    30: 28,    # toothpick -> other_garbage
    
    # ===== 其他垃圾 - 杂项 (40类: 31-36) =====
    31: 28,    # broken_ceramic -> other_garbage
    32: 28,    # dust_debris -> other_garbage
    33: 28,    # worn_fabric -> other_garbage
    34: 28,    # rubber_band -> other_garbage
    35: 28,    # pen -> other_garbage
    36: 28,    # battery_shell -> other_garbage
    
    # ===== 有害垃圾 (40类: 37-39) =====
    37: 27,    # battery -> battery
    38: 26,    # chemical_bottle -> cleaner_container
    39: 25,    # expired_medicine -> expired_medicine
}

# 是否保留未映射的类别（映射到其他垃圾）
KEEP_UNMAPPED = True
UNMAPPED_CLASS_ID = 28  # 其他垃圾

# 40类名称（用于显示）
CLASS_NAMES_40 = {
    0: "plastic_container_1", 1: "plastic_container_2", 2: "plastic_box",
    3: "plastic_tray", 4: "plastic_cup", 5: "plastic_bottle",
    6: "plastic_bag", 7: "plastic_wrap", 8: "fruit_peel",
    9: "vegetable_waste", 10: "tea_leaves", 11: "food_residue",
    12: "bone", 13: "eggshell", 14: "paper_box",
    15: "carton", 16: "paper_bag", 17: "newspaper",
    18: "tissue", 19: "metal_can_1", 20: "metal_can_2",
    21: "aluminum_foil", 22: "tin_can", 23: "zip_top_can",
    24: "glass_bottle", 25: "glass_jar", 26: "disposable_chopsticks",
    27: "disposable_tableware", 28: "cigarette_butt", 29: "straw",
    30: "toothpick", 31: "broken_ceramic", 32: "dust_debris",
    33: "worn_fabric", 34: "rubber_band", 35: "pen",
    36: "battery_shell", 37: "battery", 38: "chemical_bottle",
    39: "expired_medicine"
}

# 29类名称（用于显示）
CLASS_NAMES_29 = {
    0: "vegetable_leaves", 1: "vegetable_roots", 2: "fruit_peel",
    3: "fruit_core", 4: "bone", 5: "meat_skin",
    6: "offal", 7: "rice", 8: "noodles",
    9: "bread_crumbs", 10: "tea_leaves", 11: "coffee_grounds",
    12: "eggshell", 13: "plastic_bag", 14: "plastic_wrap",
    15: "plastic_container", 16: "paper_box", 17: "wrapping_paper",
    18: "tissue", 19: "zip_top_can", 20: "tin_can",
    21: "aluminum_foil", 22: "seasoning_bottle", 23: "wine_bottle",
    24: "expired_seasoning", 25: "expired_medicine", 26: "cleaner_container",
    27: "battery", 28: "other_garbage"
}

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
    # 项目根目录
    project_root = Path(__file__).parent.parent
    
    # 数据集路径（使用原始40类标注）
    train_labels = project_root / 'data' / 'datasets' / 'labels' / 'train'
    val_labels = project_root / 'data' / 'datasets' / 'labels' / 'val'
    
    # 备份目录
    backup_dir = project_root / 'data' / 'datasets' / 'labels_backup'
    
    # 输出目录（转换后的29类标注）
    output_train = project_root / 'data' / 'datasets' / 'labels_converted' / 'train'
    output_val = project_root / 'data' / 'datasets' / 'labels_converted' / 'val'
    
    print("=" * 60)
    print("40类 -> 29类 标注文件转换工具")
    print("=" * 60)
    print(f"\n输入目录: {train_labels.parent}")
    print(f"输出目录: {output_train.parent}")
    
    # 显示映射规则摘要
    print("\n【映射规则摘要】")
    mapping_summary = {}
    for old_id, new_id in OLD_TO_NEW_MAPPING.items():
        if new_id not in mapping_summary:
            mapping_summary[new_id] = []
        mapping_summary[new_id].append(old_id)
    
    for new_id in sorted(mapping_summary.keys()):
        old_ids = mapping_summary[new_id]
        new_name = CLASS_NAMES_29.get(new_id, 'unknown')
        old_names = [CLASS_NAMES_40.get(oid, 'unknown') for oid in old_ids]
        if len(old_ids) <= 3:
            print(f"  {new_id:2d} ({new_name:20s}) <- 40类: {old_ids}")
        else:
            print(f"  {new_id:2d} ({new_name:20s}) <- 40类: {old_ids[:3]}... ({len(old_ids)}个)")
    
    # 1. 分析现有数据集
    print("\n[1/4] 分析原始40类数据集...")
    if train_labels.exists():
        train_stats = analyze_dataset(train_labels)
        print(f"\n训练集 class_id 分布 ({len(train_stats)} 个类别):")
        for cid, count in sorted(train_stats.items()):
            old_name = CLASS_NAMES_40.get(cid, 'unknown')
            new_id = OLD_TO_NEW_MAPPING.get(cid, UNMAPPED_CLASS_ID)
            new_name = CLASS_NAMES_29.get(new_id, 'unknown')
            print(f"  {cid:2d} ({old_name:22s}): {count:6d} -> {new_id:2d} ({new_name})")
    else:
        print(f"错误：训练集标注目录不存在: {train_labels}")
        return
    
    # 2. 创建备份
    print("\n[2/4] 创建备份...")
    if not backup_dir.exists():
        if train_labels.exists():
            shutil.copytree(train_labels, backup_dir / 'train')
            print(f"  已备份训练集: {backup_dir / 'train'}")
        if val_labels.exists():
            shutil.copytree(val_labels, backup_dir / 'val')
            print(f"  已备份验证集: {backup_dir / 'val'}")
    else:
        print(f"  备份已存在: {backup_dir}")
    
    # 3. 转换标注文件
    print("\n[3/4] 转换标注文件 (40类 -> 29类)...")
    
    new_class_distribution = {}
    
    for src_dir, dst_dir in [(train_labels, output_train), (val_labels, output_val)]:
        if not src_dir.exists():
            print(f"  跳过不存在的目录: {src_dir}")
            continue
        
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        total_files = 0
        converted_files = 0
        total_objects = 0
        skipped_objects = 0
        
        label_files = list(src_dir.glob('*.txt'))
        print(f"\n  处理 {src_dir.name} ({len(label_files)} 个文件)...")
        
        for i, label_file in enumerate(label_files):
            total_files += 1
            success, obj_count, skip_count = convert_label_file(
                label_file, dst_dir / label_file.name
            )
            if success:
                converted_files += 1
                total_objects += obj_count
            skipped_objects += skip_count
            
            if (i + 1) % 5000 == 0:
                print(f"    已处理 {i + 1}/{len(label_files)} 个文件...")
        
        print(f"\n  {src_dir.name} 转换结果:")
        print(f"    总文件数: {total_files}")
        print(f"    转换成功: {converted_files}")
        print(f"    总标注数: {total_objects}")
        print(f"    跳过标注: {skipped_objects}")
    
    # 4. 验证转换结果
    print("\n[4/4] 验证转换结果...")
    if output_train.exists():
        converted_stats = analyze_dataset(output_train)
        print(f"\n转换后的类别分布 ({len(converted_stats)} 个类别):")
        for cid, count in sorted(converted_stats.items()):
            name = CLASS_NAMES_29.get(cid, 'unknown')
            print(f"  {cid:2d} ({name:20s}): {count:6d}")
    
    print("\n" + "=" * 60)
    print("转换完成!")
    print("=" * 60)
    print(f"\n转换后的标注文件位于: {output_train.parent}")
    print("\n下一步操作:")
    print("1. 运行验证脚本检查结果:")
    print("   python scripts/validate_labels_29cls.py")
    print("2. 开始训练29类模型:")
    print("   python training/train_29cls.py")

if __name__ == '__main__':
    main()
