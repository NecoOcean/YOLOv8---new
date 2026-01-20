# validate_labels_40cls.py
# -*- coding: utf-8 -*-
"""
标注文件验证脚本 - 检查 class_id 与 data_40cls.yaml 是否匹配
"""
import os
import yaml
from pathlib import Path
from collections import defaultdict

# 40类配置中的有效 class_id 范围
VALID_CLASS_IDS = set(range(40))  # 0-39

# 从 data_40cls.yaml 加载类别定义
def load_class_names():
    """加载40类配置中的类别名称"""
    yaml_path = Path(__file__).parent.parent / 'data' / 'datasets' / 'data_40cls.yaml'
    if yaml_path.exists():
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data.get('names', {})
    return {}


def analyze_label_file(file_path):
    """分析单个标注文件，返回 class_id 统计"""
    class_counts = defaultdict(int)
    invalid_lines = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                invalid_lines.append((line_num, "格式错误: 字段不足5个", line))
                continue
            
            try:
                class_id = int(parts[0])
                class_counts[class_id] += 1
                
                # 验证坐标范围 (0-1)
                x, y, w, h = map(float, parts[1:5])
                if not all(0 <= v <= 1 for v in [x, y, w, h]):
                    invalid_lines.append((line_num, "坐标超出归一化范围[0,1]", line))
                    
            except ValueError as e:
                invalid_lines.append((line_num, f"解析错误: {e}", line))
    
    return dict(class_counts), invalid_lines


def validate_dataset(labels_dir):
    """验证整个标注目录"""
    labels_path = Path(labels_dir)
    if not labels_path.exists():
        print(f"❌ 目录不存在: {labels_path}")
        return None
    
    # 统计
    total_files = 0
    total_annotations = 0
    global_class_counts = defaultdict(int)
    files_with_invalid_ids = []
    all_invalid_lines = []
    
    # 遍历所有标注文件
    txt_files = list(labels_path.glob('*.txt'))
    if not txt_files:
        print(f"⚠️ 目录为空或没有 .txt 文件: {labels_path}")
        return None
    
    for txt_file in txt_files:
        total_files += 1
        class_counts, invalid_lines = analyze_label_file(txt_file)
        
        # 汇总统计
        for cid, count in class_counts.items():
            global_class_counts[cid] += count
            total_annotations += count
            
            # 检查是否有无效的 class_id
            if cid not in VALID_CLASS_IDS:
                files_with_invalid_ids.append((txt_file.name, cid, count))
        
        # 收集无效行
        for line_info in invalid_lines:
            all_invalid_lines.append((txt_file.name, *line_info))
    
    return {
        'total_files': total_files,
        'total_annotations': total_annotations,
        'class_distribution': dict(global_class_counts),
        'invalid_class_ids': files_with_invalid_ids,
        'invalid_lines': all_invalid_lines
    }


def print_report(results, class_names):
    """打印验证报告"""
    if results is None:
        return
    
    print("\n" + "=" * 70)
    print("📊 标注文件验证报告 - data_40cls.yaml 配置")
    print("=" * 70)
    
    # 基本统计
    print(f"\n📁 总文件数: {results['total_files']}")
    print(f"📝 总标注数: {results['total_annotations']}")
    
    # class_id 分布
    print("\n" + "-" * 70)
    print("📈 Class ID 分布:")
    print("-" * 70)
    
    sorted_classes = sorted(results['class_distribution'].items())
    valid_count = 0
    invalid_count = 0
    
    for cid, count in sorted_classes:
        if cid in VALID_CLASS_IDS:
            name = class_names.get(cid, "未定义")
            status = "✅"
            valid_count += count
        else:
            name = "⚠️ 超出范围"
            status = "❌"
            invalid_count += count
        
        print(f"  {status} Class {cid:2d}: {count:6d} 个标注  ({name})")
    
    # 验证结果
    print("\n" + "-" * 70)
    print("🔍 验证结果:")
    print("-" * 70)
    
    if not results['invalid_class_ids'] and not results['invalid_lines']:
        print("✅ 所有标注均符合 data_40cls.yaml 配置要求！")
        print(f"   有效标注: {valid_count}")
    else:
        if results['invalid_class_ids']:
            print(f"❌ 发现 {len(results['invalid_class_ids'])} 个文件包含无效的 class_id:")
            for fname, cid, count in results['invalid_class_ids'][:10]:  # 只显示前10个
                print(f"   - {fname}: class_id={cid} ({count}个标注)")
            if len(results['invalid_class_ids']) > 10:
                print(f"   ... 还有 {len(results['invalid_class_ids']) - 10} 个文件")
        
        if results['invalid_lines']:
            print(f"\n❌ 发现 {len(results['invalid_lines'])} 行格式错误:")
            for fname, line_num, reason, content in results['invalid_lines'][:5]:
                print(f"   - {fname} 第{line_num}行: {reason}")
            if len(results['invalid_lines']) > 5:
                print(f"   ... 还有 {len(results['invalid_lines']) - 5} 行")
    
    # 覆盖率统计
    found_classes = set(results['class_distribution'].keys()) & VALID_CLASS_IDS
    coverage = len(found_classes) / 40 * 100
    
    print(f"\n📊 类别覆盖率: {len(found_classes)}/40 ({coverage:.1f}%)")
    
    missing_classes = VALID_CLASS_IDS - found_classes
    if missing_classes:
        print(f"⚠️ 未出现的类别 ({len(missing_classes)}个):")
        for cid in sorted(missing_classes):
            name = class_names.get(cid, "未定义")
            print(f"   - Class {cid}: {name}")


def main():
    print("=" * 70)
    print("🔍 标注文件验证工具 - 检查与 data_40cls.yaml 的一致性")
    print("=" * 70)
    
    # 加载类别名称
    class_names = load_class_names()
    if not class_names:
        print("⚠️ 无法加载 data_40cls.yaml，将仅使用 class_id")
    
    # 数据集路径
    base_dir = Path(__file__).parent.parent / 'data' / 'datasets'
    
    # 检查 labels 目录
    for split in ['train', 'val']:
        labels_dir = base_dir / 'labels' / split
        print(f"\n\n{'#' * 70}")
        print(f"# 验证 {split} 集: {labels_dir}")
        print('#' * 70)
        
        results = validate_dataset(labels_dir)
        print_report(results, class_names)
    
    # 也检查 labels_converted 目录（如果存在）
    converted_dir = base_dir / 'labels_converted'
    if converted_dir.exists():
        for split in ['train', 'val']:
            labels_dir = converted_dir / split
            if labels_dir.exists():
                print(f"\n\n{'#' * 70}")
                print(f"# 验证转换后的 {split} 集: {labels_dir}")
                print('#' * 70)
                
                results = validate_dataset(labels_dir)
                print_report(results, class_names)
    
    print("\n\n" + "=" * 70)
    print("验证完成！")
    print("=" * 70)
    print("\n下一步建议:")
    print("1. 如果存在无效的 class_id，需要修正标注或更新映射脚本")
    print("2. 如果类别覆盖率低，考虑补充数据或合并类别")
    print("3. 确认标注质量后，可以开始训练: python training/train_40cls.py")


if __name__ == '__main__':
    main()
