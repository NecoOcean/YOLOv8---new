# validate_labels_29cls.py
# coding:utf-8
"""
29类数据标注验证脚本

功能：
1. 验证标注格式（YOLO格式）
2. 检查 class_id 范围（0-28）
3. 验证坐标合法性（归一化值0-1）
4. 检查图像-标注对应关系
5. 统计类别分布
6. 生成验证报告
7. 自动修复简单错误
"""
from pathlib import Path
from collections import defaultdict
import os

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 29类类别定义
CLASS_NAMES_29 = {
    # 厨余垃圾（湿垃圾）- 蔬菜类
    0: "vegetable_leaves",    # 菜叶
    1: "vegetable_roots",     # 菜根
    2: "fruit_peel",          # 果皮
    3: "fruit_core",          # 果核
    # 厨余垃圾（湿垃圾）- 肉类
    4: "bone",                # 骨头
    5: "meat_skin",           # 肉皮
    6: "offal",               # 内脏
    # 厨余垃圾（湿垃圾）- 主食类
    7: "rice",                # 米饭
    8: "noodles",             # 面条
    9: "bread_crumbs",        # 面包屑
    # 厨余垃圾（湿垃圾）- 其他
    10: "tea_leaves",         # 茶叶渣
    11: "coffee_grounds",     # 咖啡渣
    12: "eggshell",           # 蛋壳
    # 可回收垃圾 - 塑料
    13: "plastic_bag",        # 食品袋
    14: "plastic_wrap",       # 保鲜膜
    15: "plastic_container",  # 塑料盒
    # 可回收垃圾 - 纸类
    16: "paper_box",          # 纸盒
    17: "wrapping_paper",     # 包装纸
    18: "tissue",             # 餐巾纸
    # 可回收垃圾 - 金属
    19: "zip_top_can",        # 易拉罐
    20: "tin_can",            # 罐头盒
    21: "aluminum_foil",      # 铝箔
    # 可回收垃圾 - 玻璃
    22: "seasoning_bottle",   # 调料瓶
    23: "wine_bottle",        # 酒瓶
    # 有害垃圾
    24: "expired_seasoning",  # 过期调料
    25: "expired_medicine",   # 过期药品
    26: "cleaner_container",  # 清洁剂容器
    27: "battery",            # 电池
    # 其他垃圾
    28: "other_garbage",      # 其他垃圾
}

# 有效 class_id 范围
VALID_CLASS_IDS = set(range(29))  # 0-28


class ValidationError:
    """验证错误信息类"""
    def __init__(self, file_path, line_num, error_type, message, original_line=None):
        self.file_path = file_path
        self.line_num = line_num
        self.error_type = error_type
        self.message = message
        self.original_line = original_line
        self.can_fix = False
        self.fixed_line = None


class LabelValidator:
    """29类标注验证器"""
    
    def __init__(self, labels_dir, images_dir=None):
        self.labels_dir = Path(labels_dir)
        self.images_dir = Path(images_dir) if images_dir else None
        self.errors = []
        self.warnings = []
        self.stats = {
            'total_files': 0,
            'valid_files': 0,
            'total_annotations': 0,
            'valid_annotations': 0,
            'class_distribution': defaultdict(int),
            'error_types': defaultdict(int),
        }
    
    def validate_line(self, line, file_path, line_num):
        """验证单行标注"""
        line = line.strip()
        if not line:
            return None  # 空行跳过
        
        parts = line.split()
        
        # 检查字段数量
        if len(parts) < 5:
            error = ValidationError(
                file_path, line_num, 'FORMAT_ERROR',
                f'字段数量不足: 期望5个, 实际{len(parts)}个',
                line
            )
            self.errors.append(error)
            self.stats['error_types']['FORMAT_ERROR'] += 1
            return None
        
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError as e:
            error = ValidationError(
                file_path, line_num, 'PARSE_ERROR',
                f'数值解析失败: {e}',
                line
            )
            self.errors.append(error)
            self.stats['error_types']['PARSE_ERROR'] += 1
            return None
        
        # 检查 class_id 范围
        if class_id not in VALID_CLASS_IDS:
            error = ValidationError(
                file_path, line_num, 'CLASS_ID_ERROR',
                f'无效的class_id: {class_id} (有效范围: 0-28)',
                line
            )
            self.errors.append(error)
            self.stats['error_types']['CLASS_ID_ERROR'] += 1
            return None
        
        # 检查坐标范围
        coords_valid = True
        fixed_coords = [x_center, y_center, width, height]
        
        for i, (name, val) in enumerate([('x_center', x_center), ('y_center', y_center), 
                                          ('width', width), ('height', height)]):
            if val < 0 or val > 1:
                coords_valid = False
                # 尝试修复：裁剪到有效范围
                fixed_val = max(0, min(1, val))
                fixed_coords[i] = fixed_val
                
                error = ValidationError(
                    file_path, line_num, 'COORD_ERROR',
                    f'{name}超出范围: {val} (应为0-1)',
                    line
                )
                error.can_fix = True
                error.fixed_line = f"{class_id} {fixed_coords[0]:.6f} {fixed_coords[1]:.6f} {fixed_coords[2]:.6f} {fixed_coords[3]:.6f}"
                self.errors.append(error)
                self.stats['error_types']['COORD_ERROR'] += 1
        
        if coords_valid:
            self.stats['valid_annotations'] += 1
            self.stats['class_distribution'][class_id] += 1
        
        self.stats['total_annotations'] += 1
        return class_id if coords_valid else None
    
    def validate_file(self, file_path):
        """验证单个标注文件"""
        self.stats['total_files'] += 1
        file_valid = True
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            error = ValidationError(
                file_path, 0, 'FILE_ERROR',
                f'文件读取失败: {e}'
            )
            self.errors.append(error)
            self.stats['error_types']['FILE_ERROR'] += 1
            return False
        
        for line_num, line in enumerate(lines, 1):
            result = self.validate_line(line, file_path, line_num)
            if result is None and line.strip():
                file_valid = False
        
        if file_valid:
            self.stats['valid_files'] += 1
        
        return file_valid
    
    def check_image_correspondence(self):
        """检查图像-标注对应关系"""
        if not self.images_dir or not self.images_dir.exists():
            return
        
        # 获取所有图像和标注文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        image_files = set()
        for ext in image_extensions:
            for img in self.images_dir.glob(f'*{ext}'):
                image_files.add(img.stem)
            for img in self.images_dir.glob(f'*{ext.upper()}'):
                image_files.add(img.stem)
        
        label_files = set(f.stem for f in self.labels_dir.glob('*.txt'))
        
        # 检查缺失的标注
        missing_labels = image_files - label_files
        if missing_labels:
            for name in list(missing_labels)[:10]:  # 只显示前10个
                self.warnings.append(f"图像 {name} 缺少对应的标注文件")
            if len(missing_labels) > 10:
                self.warnings.append(f"... 还有 {len(missing_labels) - 10} 个图像缺少标注")
        
        # 检查缺失的图像
        missing_images = label_files - image_files
        if missing_images:
            for name in list(missing_images)[:10]:
                self.warnings.append(f"标注 {name}.txt 缺少对应的图像文件")
            if len(missing_images) > 10:
                self.warnings.append(f"... 还有 {len(missing_images) - 10} 个标注缺少图像")
    
    def validate_all(self):
        """验证所有标注文件"""
        if not self.labels_dir.exists():
            print(f"错误：标注目录不存在: {self.labels_dir}")
            return False
        
        label_files = list(self.labels_dir.glob('*.txt'))
        if not label_files:
            print(f"警告：标注目录为空: {self.labels_dir}")
            return False
        
        print(f"开始验证 {len(label_files)} 个标注文件...")
        
        for i, file_path in enumerate(label_files):
            self.validate_file(file_path)
            if (i + 1) % 1000 == 0:
                print(f"  已处理 {i + 1}/{len(label_files)} 个文件...")
        
        # 检查图像对应关系
        self.check_image_correspondence()
        
        return len(self.errors) == 0
    
    def print_report(self):
        """打印验证报告"""
        print("\n" + "=" * 60)
        print("29类数据标注验证报告")
        print("=" * 60)
        
        print(f"\n【文件统计】")
        print(f"  总文件数: {self.stats['total_files']}")
        print(f"  有效文件: {self.stats['valid_files']}")
        print(f"  错误文件: {self.stats['total_files'] - self.stats['valid_files']}")
        
        print(f"\n【标注统计】")
        print(f"  总标注数: {self.stats['total_annotations']}")
        print(f"  有效标注: {self.stats['valid_annotations']}")
        print(f"  错误标注: {self.stats['total_annotations'] - self.stats['valid_annotations']}")
        
        print(f"\n【类别分布】")
        if self.stats['class_distribution']:
            for class_id in sorted(self.stats['class_distribution'].keys()):
                count = self.stats['class_distribution'][class_id]
                name = CLASS_NAMES_29.get(class_id, 'unknown')
                print(f"  class_id {class_id:2d} ({name:20s}): {count:6d} 个")
            
            # 检查缺失的类别
            found_classes = set(self.stats['class_distribution'].keys())
            missing_classes = VALID_CLASS_IDS - found_classes
            if missing_classes:
                print(f"\n⚠️ 缺失的类别 ({len(missing_classes)} 个):")
                for class_id in sorted(missing_classes):
                    print(f"    class_id {class_id}: {CLASS_NAMES_29.get(class_id, 'unknown')}")
        else:
            print("  无有效标注数据")
        
        if self.stats['error_types']:
            print(f"\n【错误类型统计】")
            for error_type, count in self.stats['error_types'].items():
                print(f"  {error_type}: {count} 个")
        
        if self.warnings:
            print(f"\n【警告信息】({len(self.warnings)} 条)")
            for warning in self.warnings[:20]:
                print(f"  ⚠️ {warning}")
            if len(self.warnings) > 20:
                print(f"  ... 还有 {len(self.warnings) - 20} 条警告")
        
        if self.errors:
            print(f"\n【错误详情】(显示前20条)")
            for error in self.errors[:20]:
                print(f"  ❌ {error.file_path.name}:{error.line_num} - {error.error_type}")
                print(f"     {error.message}")
                if error.original_line:
                    print(f"     原始内容: {error.original_line}")
                if error.can_fix:
                    print(f"     ✅ 可自动修复")
        
        # 总结
        print("\n" + "=" * 60)
        if len(self.errors) == 0:
            print("✅ 验证通过！所有标注文件格式正确。")
        else:
            fixable = sum(1 for e in self.errors if e.can_fix)
            print(f"❌ 验证失败！发现 {len(self.errors)} 个错误")
            if fixable > 0:
                print(f"   其中 {fixable} 个可自动修复")
        print("=" * 60)
    
    def auto_fix(self):
        """自动修复可修复的错误"""
        fixable_errors = [e for e in self.errors if e.can_fix]
        if not fixable_errors:
            print("没有可自动修复的错误")
            return 0
        
        # 按文件分组
        files_to_fix = defaultdict(list)
        for error in fixable_errors:
            files_to_fix[error.file_path].append(error)
        
        fixed_count = 0
        for file_path, errors in files_to_fix.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # 按行号创建修复映射
                line_fixes = {e.line_num: e.fixed_line for e in errors}
                
                # 应用修复
                new_lines = []
                for i, line in enumerate(lines, 1):
                    if i in line_fixes:
                        new_lines.append(line_fixes[i] + '\n')
                        fixed_count += 1
                    else:
                        new_lines.append(line)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                
            except Exception as e:
                print(f"修复文件 {file_path} 时出错: {e}")
        
        print(f"已修复 {fixed_count} 个错误")
        return fixed_count
    
    def export_report(self, output_path):
        """导出详细报告到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("29类数据标注验证报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"标注目录: {self.labels_dir}\n")
            f.write(f"总文件数: {self.stats['total_files']}\n")
            f.write(f"有效文件: {self.stats['valid_files']}\n")
            f.write(f"总标注数: {self.stats['total_annotations']}\n")
            f.write(f"有效标注: {self.stats['valid_annotations']}\n\n")
            
            f.write("类别分布:\n")
            for class_id in sorted(self.stats['class_distribution'].keys()):
                count = self.stats['class_distribution'][class_id]
                name = CLASS_NAMES_29.get(class_id, 'unknown')
                f.write(f"  {class_id:2d}: {name:20s} - {count:6d}\n")
            
            if self.errors:
                f.write(f"\n错误列表 ({len(self.errors)} 条):\n")
                for error in self.errors:
                    f.write(f"  {error.file_path.name}:{error.line_num} - {error.error_type}\n")
                    f.write(f"    {error.message}\n")
        
        print(f"报告已导出到: {output_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("29类数据标注验证工具")
    print("=" * 60)
    
    # 标注目录路径
    labels_converted_train = PROJECT_ROOT / 'data' / 'datasets' / 'labels_converted' / 'train'
    labels_converted_val = PROJECT_ROOT / 'data' / 'datasets' / 'labels_converted' / 'val'
    images_train = PROJECT_ROOT / 'data' / 'datasets' / 'images' / 'train'
    images_val = PROJECT_ROOT / 'data' / 'datasets' / 'images' / 'val'
    
    # 验证训练集
    print("\n【验证训练集标注】")
    if labels_converted_train.exists():
        validator_train = LabelValidator(labels_converted_train, images_train)
        validator_train.validate_all()
        validator_train.print_report()
        train_errors = validator_train.errors
        train_stats = validator_train.stats
    else:
        print(f"训练集标注目录不存在: {labels_converted_train}")
        train_errors = []
        train_stats = None
    
    # 验证验证集
    print("\n【验证验证集标注】")
    if labels_converted_val.exists():
        validator_val = LabelValidator(labels_converted_val, images_val)
        validator_val.validate_all()
        validator_val.print_report()
        val_errors = validator_val.errors
    else:
        print(f"验证集标注目录不存在: {labels_converted_val}")
        val_errors = []
    
    # 交互式操作
    all_errors = train_errors + val_errors
    if all_errors:
        print("\n可用操作:")
        print("  1. 导出详细报告")
        print("  2. 自动修复可修复的错误")
        print("  3. 退出")
        
        try:
            choice = input("\n请选择操作 (1/2/3): ").strip()
            
            if choice == '1':
                report_path = PROJECT_ROOT / 'validation_report_29cls.txt'
                if train_stats:
                    validator_train.export_report(report_path)
            elif choice == '2':
                if labels_converted_train.exists():
                    validator_train.auto_fix()
                if labels_converted_val.exists():
                    validator_val.auto_fix()
            else:
                print("退出")
        except KeyboardInterrupt:
            print("\n操作已取消")
    else:
        print("\n✅ 所有标注文件验证通过！")


if __name__ == '__main__':
    main()
