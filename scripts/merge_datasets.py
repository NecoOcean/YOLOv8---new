"""
数据集合并脚本
合并 TACO + Food Waste Detection 数据集

功能：
1. 统一类别映射（创建项目专用的类别体系）
2. 合并图片和标签文件
3. 生成统一的 data.yaml 配置文件
4. 自动划分训练集和验证集
"""

import os
import shutil
import yaml
import random
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


# =============================================================================
# 项目目标类别定义（厨房垃圾分类）
# =============================================================================
TARGET_CLASSES = {
    # 厨余垃圾（湿垃圾）
    0: "vegetable",        # 蔬菜类
    1: "fruit_peel",       # 果皮
    2: "fruit_core",       # 果核
    3: "bone",             # 骨头
    4: "fish_bone",        # 鱼骨
    5: "eggshell",         # 蛋壳
    6: "rice",             # 米饭/剩饭
    7: "noodle",           # 面条
    8: "bread",            # 面包
    9: "meat",             # 肉类
    10: "fish",            # 鱼类
    11: "leftover",        # 其他剩菜
    
    # 可回收物
    12: "plastic_bottle",  # 塑料瓶
    13: "plastic_bag",     # 塑料袋
    14: "plastic_container", # 塑料容器
    15: "glass_bottle",    # 玻璃瓶
    16: "metal_can",       # 金属罐（易拉罐）
    17: "paper_box",       # 纸盒
    18: "paper",           # 纸张
    19: "aluminum_foil",   # 铝箔
    
    # 有害垃圾
    20: "battery",         # 电池
    21: "cigarette",       # 烟头（有害）
    
    # 其他垃圾
    22: "other_waste",     # 其他垃圾
}


# =============================================================================
# Food Waste Detection 类别映射
# =============================================================================
FOOD_WASTE_MAPPING = {
    "Apple": 11,           # leftover
    "Apple-core": 2,       # fruit_core
    "Apple-peel": 1,       # fruit_peel
    "Banana": 11,          # leftover
    "Bone": 3,             # bone
    "Bone-fish": 4,        # fish_bone
    "Bread": 8,            # bread
    "Bun": 8,              # bread
    "Chicken-skin": 9,     # meat
    "Congee": 6,           # rice
    "Cucumber": 0,         # vegetable
    "Drink": 22,           # other_waste
    "Egg-hard": 11,        # leftover
    "Egg-scramble": 11,    # leftover
    "Egg-shell": 5,        # eggshell
    "Egg-steam": 11,       # leftover
    "Egg-yolk": 11,        # leftover
    "Fish": 10,            # fish
    "Meat": 9,             # meat
    "Mushroom": 0,         # vegetable
    "Mussel": 10,          # fish (seafood)
    "Mussel-shell": 22,    # other_waste
    "Noodle": 7,           # noodle
    "Orange": 11,          # leftover
    "Orange-peel": 1,      # fruit_peel
    "Other-waste": 22,     # other_waste
    "Pear": 11,            # leftover
    "Pear-core": 2,        # fruit_core
    "Pear-peel": 1,        # fruit_peel
    "Potato": 0,           # vegetable
    "Rice": 6,             # rice
    "Tomato": 0,           # vegetable
    "Vegetable": 0,        # vegetable
}


# =============================================================================
# TACO 类别映射
# =============================================================================
TACO_MAPPING = {
    # 塑料类
    "Clear plastic bottle": 12,      # plastic_bottle
    "Plastic bottle cap": 14,        # plastic_container
    "Other plastic bottle": 12,      # plastic_bottle
    "Plastic film": 13,              # plastic_bag
    "Six pack rings": 13,            # plastic_bag
    "Garbage bag": 13,               # plastic_bag
    "Single-use carrier bag": 13,    # plastic_bag
    "Polypropylene bag": 13,         # plastic_bag
    "Crisp packet": 13,              # plastic_bag
    "Spread tub": 14,                # plastic_container
    "Tupperware": 14,                # plastic_container
    "Disposable plastic cup": 14,    # plastic_container
    "Foam cup": 14,                  # plastic_container
    "Other plastic cup": 14,         # plastic_container
    "Plastic lid": 14,               # plastic_container
    "Plastic straw": 14,             # plastic_container
    "Disposable food container": 14, # plastic_container
    "Foam food container": 14,       # plastic_container
    "Other plastic container": 14,   # plastic_container
    "Plastic utensils": 14,          # plastic_container
    "Plastic glove": 13,             # plastic_bag
    "Other plastic wrapper": 13,     # plastic_bag
    "Other plastic": 14,             # plastic_container
    
    # 玻璃类
    "Glass bottle": 15,              # glass_bottle
    "Glass jar": 15,                 # glass_bottle
    "Glass cup": 15,                 # glass_bottle
    "Broken glass": 15,              # glass_bottle
    "Other glass": 15,               # glass_bottle
    
    # 金属类
    "Aluminium foil": 19,            # aluminum_foil
    "Aluminium blister pack": 19,    # aluminum_foil
    "Drink can": 16,                 # metal_can
    "Food can": 16,                  # metal_can
    "Aerosol": 16,                   # metal_can
    "Metal bottle cap": 16,          # metal_can
    "Scrap metal": 16,               # metal_can
    "Pop tab": 16,                   # metal_can
    "Metal lid": 16,                 # metal_can
    "Other metal": 16,               # metal_can
    
    # 纸类
    "Corrugated carton": 17,         # paper_box
    "Egg carton": 17,                # paper_box
    "Drink carton": 17,              # paper_box
    "Toilet tube": 17,               # paper_box
    "Other carton": 17,              # paper_box
    "Paper": 18,                     # paper
    "Paper bag": 18,                 # paper
    "Wrapping paper": 18,            # paper
    "Newspaper": 18,                 # paper
    "Tissues": 18,                   # paper
    "Magazine paper": 18,            # paper
    "Paper cup": 17,                 # paper_box
    "Paper straw": 18,               # paper
    "Normal paper": 18,              # paper
    
    # 有害/其他
    "Battery": 20,                   # battery
    "Cigarette": 21,                 # cigarette
    "Unlabeled litter": 22,          # other_waste
    "Rope & strings": 22,            # other_waste
    "Shoe": 22,                      # other_waste
    "Squeezable tube": 22,           # other_waste
    "Styrofoam piece": 22,           # other_waste
}


# =============================================================================
# tany0699 类别映射（中文类名）
# =============================================================================
TANY0699_MAPPING = {
    # 厨余垃圾 - 蔬菜类
    "厨余垃圾-菜根菜叶": 0,          # vegetable
    "厨余垃圾-蔬菜": 0,              # vegetable
    "厨余垃圾-萝卜": 0,              # vegetable
    "厨余垃圾-蘑菇": 0,              # vegetable
    "厨余垃圾-番茄": 0,              # vegetable
    "厨余垃圾-辣椒": 0,              # vegetable
    "厨余垃圾-地瓜": 0,              # vegetable
    
    # 厨余垃圾 - 果皮果核
    "厨余垃圾-果皮": 1,              # fruit_peel
    "厨余垃圾-果壳": 2,              # fruit_core
    "厨余垃圾-苹果": 11,             # leftover
    "厨余垃圾-橙子": 11,             # leftover
    "厨余垃圾-菠萝": 11,             # leftover
    "厨余垃圾-草莓": 11,             # leftover
    "厨余垃圾-核桃": 2,              # fruit_core
    "厨余垃圾-瓜子": 2,              # fruit_core
    
    # 厨余垃圾 - 骨头肉类
    "厨余垃圾-骨头": 3,              # bone
    "厨余垃圾-鱼骨": 4,              # fish_bone
    "厨余垃圾-肉类": 9,              # meat
    "厨余垃圾-鸡翅": 9,              # meat
    "厨余垃圾-火腿": 9,              # meat
    
    # 厨余垃圾 - 蛋类
    "厨余垃圾-蛋": 5,                # eggshell
    "厨余垃圾-蛋挞": 11,             # leftover
    "厨余垃圾-蛋糕": 11,             # leftover
    
    # 厨余垃圾 - 主食
    "厨余垃圾-残渣剩饭": 6,          # rice
    "厨余垃圾-粉条": 7,              # noodle
    "厨余垃圾-面包": 8,              # bread
    "厨余垃圾-饼干": 8,              # bread
    
    # 厨余垃圾 - 其他
    "厨余垃圾-茶叶": 11,             # leftover
    "厨余垃圾-咖啡": 11,             # leftover
    "厨余垃圾-豆腐": 11,             # leftover
    "厨余垃圾-巧克力": 11,           # leftover
    
    # 可回收物 - 塑料
    "可回收物-塑料瓶": 12,           # plastic_bottle
    "可回收物-塑料袋": 13,           # plastic_bag
    "可回收物-塑料盒": 14,           # plastic_container
    "可回收物-保鲜膜内芯": 14,       # plastic_container
    
    # 可回收物 - 玻璃
    "可回收物-玻璃瓶": 15,           # glass_bottle
    "可回收物-玻璃器皿": 15,         # glass_bottle
    "可回收物-玻璃制品": 15,         # glass_bottle
    
    # 可回收物 - 金属
    "可回收物-易拉罐": 16,           # metal_can
    "可回收物-罐头盒": 16,           # metal_can
    "可回收物-不锈钢制品": 16,       # metal_can
    
    # 可回收物 - 纸类
    "可回收物-纸箱": 17,             # paper_box
    "可回收物-纸盒": 17,             # paper_box
    "可回收物-报纸": 18,             # paper
    "可回收物-书籍": 18,             # paper
    
    # 有害垃圾
    "有害垃圾-电池": 20,             # battery
    "有害垃圾-纽扣电池": 20,         # battery
    "有害垃圾-药片": 22,             # other_waste
    "有害垃圾-药瓶": 22,             # other_waste
    "有害垃圾-灯": 22,               # other_waste
    
    # 其他垃圾
    "其他垃圾-餐巾纸": 22,           # other_waste
    "其他垃圾-烟蒂": 21,             # cigarette
    "其他垃圾-一次性杯子": 22,       # other_waste
    "其他垃圾-牙签": 22,             # other_waste
    "其他垃圾-竹筷": 22,             # other_waste
}


def load_yaml_classes(yaml_path: Path) -> Dict[int, str]:
    """加载数据集的 data.yaml 获取类别名称"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    names = config.get('names', {})
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    return names


def create_class_mapping(source_classes: Dict[int, str], 
                         mapping_dict: Dict[str, int]) -> Dict[int, int]:
    """
    创建源类别ID到目标类别ID的映射
    
    Args:
        source_classes: 源数据集的类别 {id: name}
        mapping_dict: 类别名称到目标ID的映射 {name: target_id}
    
    Returns:
        Dict[int, int]: {source_id: target_id}
    """
    result = {}
    for src_id, src_name in source_classes.items():
        # 尝试精确匹配
        if src_name in mapping_dict:
            result[src_id] = mapping_dict[src_name]
        else:
            # 尝试模糊匹配
            for map_name, target_id in mapping_dict.items():
                if map_name.lower() in src_name.lower() or src_name.lower() in map_name.lower():
                    result[src_id] = target_id
                    break
            else:
                # 无法匹配，归类为其他垃圾
                result[src_id] = 22  # other_waste
                print(f"警告: 类别 '{src_name}' 无法匹配，归类为 other_waste")
    
    return result


def convert_labels(src_label_path: Path, 
                   dst_label_path: Path,
                   class_mapping: Dict[int, int]) -> bool:
    """
    转换标签文件的类别ID
    
    Args:
        src_label_path: 源标签文件路径
        dst_label_path: 目标标签文件路径
        class_mapping: 类别映射 {src_id: dst_id}
    
    Returns:
        bool: 是否成功转换
    """
    try:
        with open(src_label_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                src_class = int(parts[0])
                if src_class in class_mapping:
                    parts[0] = str(class_mapping[src_class])
                    new_lines.append(' '.join(parts) + '\n')
        
        if new_lines:
            with open(dst_label_path, 'w') as f:
                f.writelines(new_lines)
            return True
        return False
    except Exception as e:
        print(f"转换标签失败 {src_label_path}: {e}")
        return False


def merge_datasets(
    dataset_configs: List[Dict],
    output_dir: str,
    train_ratio: float = 0.85
):
    """
    合并多个数据集
    
    Args:
        dataset_configs: 数据集配置列表
            [{
                'path': 数据集路径,
                'mapping': 类别映射字典,
                'prefix': 文件名前缀
            }]
        output_dir: 输出目录
        train_ratio: 训练集比例
    """
    output_path = Path(output_dir)
    
    # 创建输出目录
    (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    all_samples = []  # [(img_path, label_path, class_mapping, prefix)]
    
    # 收集所有样本
    for config in dataset_configs:
        dataset_path = Path(config['path'])
        mapping_dict = config['mapping']
        prefix = config.get('prefix', '')
        
        print(f"\n处理数据集: {dataset_path}")
        
        # 加载类别定义
        yaml_path = dataset_path / 'data.yaml'
        if yaml_path.exists():
            source_classes = load_yaml_classes(yaml_path)
            class_mapping = create_class_mapping(source_classes, mapping_dict)
            print(f"  类别数: {len(source_classes)}")
        else:
            print(f"  警告: data.yaml 不存在，跳过")
            continue
        
        # 查找图片和标签
        for split in ['train', 'valid', 'val', 'test']:
            img_dir = dataset_path / 'train' / 'images' if split == 'train' else None
            if img_dir is None or not img_dir.exists():
                img_dir = dataset_path / split / 'images'
            if not img_dir.exists():
                img_dir = dataset_path / 'images' / split
            if not img_dir.exists():
                continue
            
            label_dir = img_dir.parent / 'labels' if (img_dir.parent / 'labels').exists() else None
            if label_dir is None:
                label_dir = dataset_path / split / 'labels'
            if not label_dir.exists():
                label_dir = dataset_path / 'labels' / split
            
            if not label_dir.exists():
                print(f"  警告: 标签目录不存在 {label_dir}")
                continue
            
            for img_file in img_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    label_file = label_dir / (img_file.stem + '.txt')
                    if label_file.exists():
                        all_samples.append((img_file, label_file, class_mapping, prefix))
    
    print(f"\n总样本数: {len(all_samples)}")
    
    # 随机打乱并划分
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print(f"训练集: {len(train_samples)}")
    print(f"验证集: {len(val_samples)}")
    
    # 复制文件
    def copy_samples(samples, split):
        for img_path, label_path, class_mapping, prefix in tqdm(samples, desc=f"复制 {split}"):
            # 生成唯一文件名
            new_name = f"{prefix}_{img_path.stem}" if prefix else img_path.stem
            
            # 复制图片
            dst_img = output_path / 'images' / split / (new_name + img_path.suffix)
            shutil.copy2(img_path, dst_img)
            
            # 转换并复制标签
            dst_label = output_path / 'labels' / split / (new_name + '.txt')
            convert_labels(label_path, dst_label, class_mapping)
    
    copy_samples(train_samples, 'train')
    copy_samples(val_samples, 'val')
    
    # 生成 data.yaml
    yaml_content = f"""# 厨房垃圾分类混合数据集
# 来源: TACO + Food Waste Detection
# 生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

path: {output_path.absolute()}
train: images/train
val: images/val

nc: {len(TARGET_CLASSES)}

names:
"""
    for idx, name in sorted(TARGET_CLASSES.items()):
        yaml_content += f"  {idx}: {name}\n"
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"\n合并完成！")
    print(f"输出目录: {output_path}")
    print(f"配置文件: {yaml_path}")
    
    return output_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='合并垃圾分类数据集')
    parser.add_argument('--taco', type=str, 
                        default='data/datasets/TACO',
                        help='TACO 数据集路径')
    parser.add_argument('--food-waste', type=str,
                        default='data/datasets/Food-Waste-Detection',
                        help='Food Waste Detection 数据集路径')
    parser.add_argument('--tany0699', type=str,
                        default='',
                        help='tany0699_yolo 数据集路径（可选）')
    parser.add_argument('--output', type=str,
                        default='data/datasets/kitchen_garbage_merged',
                        help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.85,
                        help='训练集比例')
    
    args = parser.parse_args()
    
    # 数据集配置
    datasets = [
        {
            'path': args.taco,
            'mapping': TACO_MAPPING,
            'prefix': 'taco'
        },
        {
            'path': args.food_waste,
            'mapping': FOOD_WASTE_MAPPING,
            'prefix': 'food'
        }
    ]
    
    # 添加 tany0699（如果指定）
    if args.tany0699 and Path(args.tany0699).exists():
        datasets.append({
            'path': args.tany0699,
            'mapping': TANY0699_MAPPING,
            'prefix': 'tany'
        })
    
    # 执行合并
    merge_datasets(datasets, args.output, args.train_ratio)
    
    print(f"\n下一步:")
    print(f"python training/train.py --data {args.output}/data.yaml")
