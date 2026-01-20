"""
将 tany0699/garbage265 分类数据集转换为 YOLO 目标检测格式

数据集来源: https://www.modelscope.cn/datasets/tany0699/garbage265
- 147,674 张图片，265 个类别
- 4大类：厨余垃圾、可回收垃圾、其他垃圾、有害垃圾

数据集原结构（分类格式）:
    images/train/{class_id}/image.jpg
    images/val/{class_id}/image.jpg
    garbage265/classname.txt  # 类名文件

转换后结构（YOLO检测格式）:
    images/train/image.jpg
    labels/train/image.txt
    data.yaml  # 带中文类名
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm


def load_classnames(classname_file: str) -> dict:
    """
    加载中文类名文件
    
    Args:
        classname_file: classname.txt 路径
    
    Returns:
        dict: {class_id: class_name}
    """
    classnames = {}
    with open(classname_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            name = line.strip()
            if name:
                classnames[idx] = name
    return classnames


def convert_classification_to_detection(
    src_dir: str,
    dest_dir: str = None,
    class_mapping: dict = None,
    full_image_bbox: bool = True,
    classname_file: str = None
):
    """
    将分类数据集转换为YOLO目标检测格式
    
    Args:
        src_dir: 源数据集目录 (包含 images/train/{class_id}/... 结构)
        dest_dir: 目标目录，默认为 src_dir + "_yolo"
        class_mapping: 类别映射字典 {原始class_id: 新class_id}，None表示保持原样
        full_image_bbox: 是否使用全图边界框 (0.5 0.5 1.0 1.0)
        classname_file: 类名文件路径 (可选，用于日志输出)
    
    Returns:
        Path: 转换后的数据集目录路径
    """
    src_path = Path(src_dir)
    dest_path = Path(dest_dir) if dest_dir else src_path.parent / f"{src_path.name}_yolo"
    
    print(f"源目录: {src_path}")
    print(f"目标目录: {dest_path}")
    
    # 创建目标目录结构
    for split in ['train', 'val']:
        (dest_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dest_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    stats = {'train': {'images': 0, 'classes': set()}, 'val': {'images': 0, 'classes': set()}}
    
    # 处理 train 和 val 分割
    for split in ['train', 'val']:
        split_src = src_path / 'images' / split
        if not split_src.exists():
            print(f"警告: {split_src} 不存在，跳过")
            continue
        
        # 获取所有类别文件夹
        class_dirs = [d for d in split_src.iterdir() if d.is_dir()]
        print(f"\n处理 {split} 集，共 {len(class_dirs)} 个类别")
        
        for class_dir in tqdm(class_dirs, desc=f"转换 {split}"):
            try:
                original_class_id = int(class_dir.name)
            except ValueError:
                print(f"警告: 无法解析类别目录名 '{class_dir.name}'，跳过")
                continue
            
            # 应用类别映射
            if class_mapping:
                if original_class_id not in class_mapping:
                    continue
                class_id = class_mapping[original_class_id]
            else:
                class_id = original_class_id
            
            stats[split]['classes'].add(class_id)
            
            # 处理该类别下的所有图片
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    continue
                
                # 生成唯一文件名（添加类别前缀避免冲突）
                new_filename = f"c{class_id:03d}_{img_file.name}"
                
                # 复制图片
                dest_img = dest_path / 'images' / split / new_filename
                shutil.copy2(img_file, dest_img)
                
                # 生成标签文件
                label_filename = dest_img.stem + '.txt'
                dest_label = dest_path / 'labels' / split / label_filename
                
                if full_image_bbox:
                    # 全图边界框: class_id x_center y_center width height
                    bbox = f"{class_id} 0.5 0.5 1.0 1.0"
                else:
                    # 仅类别标签
                    bbox = f"{class_id} 0.5 0.5 0.8 0.8"  # 稍小的边界框
                
                with open(dest_label, 'w') as f:
                    f.write(bbox + '\n')
                
                stats[split]['images'] += 1
    
    # 打印统计信息
    print("\n" + "="*50)
    print("转换完成！")
    print("="*50)
    for split in ['train', 'val']:
        if stats[split]['images'] > 0:
            print(f"{split}: {stats[split]['images']} 张图片, {len(stats[split]['classes'])} 个类别")
            print(f"  类别范围: {min(stats[split]['classes'])} - {max(stats[split]['classes'])}")
    
    return dest_path


def create_yaml_config(dataset_path: str, nc: int, names: dict = None):
    """
    创建数据集 YAML 配置文件
    
    Args:
        dataset_path: 数据集目录
        nc: 类别数量
        names: 类别名称字典 {id: name}
    """
    yaml_path = Path(dataset_path) / 'data.yaml'
    
    if names is None:
        names = {i: f"class_{i}" for i in range(nc)}
    
    content = f"""# tany0699 垃圾分类数据集 (转换后)
# 原始来源: https://www.modelscope.cn/datasets/tany0699/garbage265

path: {dataset_path}
train: images/train
val: images/val

nc: {nc}

names:
"""
    for idx, name in sorted(names.items()):
        content += f"  {idx}: {name}\n"
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"配置文件已创建: {yaml_path}")
    return yaml_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='转换 tany0699 分类数据集为 YOLO 检测格式')
    parser.add_argument('--src', type=str, 
                        default='data/datasets/tany0699',
                        help='源数据集目录')
    parser.add_argument('--dest', type=str, default=None,
                        help='目标目录，默认为 src_yolo')
    parser.add_argument('--classname', type=str, 
                        default='data/datasets/tany0699/garbage265/classname.txt',
                        help='类名文件路径')
    
    args = parser.parse_args()
    
    # 加载中文类名
    classname_path = Path(args.classname)
    if classname_path.exists():
        print(f"加载类名文件: {classname_path}")
        classnames = load_classnames(str(classname_path))
        print(f"已加载 {len(classnames)} 个类名")
    else:
        print(f"警告: 类名文件不存在 {classname_path}，使用默认类名")
        classnames = None
    
    # 执行转换
    dest_path = convert_classification_to_detection(
        args.src, args.dest, classname_file=str(classname_path)
    )
    
    # 创建配置文件（使用中文类名）
    nc = len(classnames) if classnames else 265
    create_yaml_config(str(dest_path), nc, classnames)
    
    print(f"\n下一步:")
    print(f"1. 检查转换结果: {dest_path}")
    print(f"2. 使用配置文件训练: python training/train.py --data {dest_path}/data.yaml")
