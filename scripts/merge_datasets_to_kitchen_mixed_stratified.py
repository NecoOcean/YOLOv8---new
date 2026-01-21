# scripts/merge_datasets_to_kitchen_mixed_stratified.py
# -*- coding: utf-8 -*-
"""
从 newdata 下 4 个子数据集构建新的 kitchen_mixed 分层数据集：
- domestic_yolo  (TU Wien 厨余 -> 已是 YOLO, 2 类: 0=organic,1=non-organic)
- A              (medical waste 1 -> 全部映射为 hazardous)
- B              (medical waste 2 -> 全部映射为 hazardous)
- C              (TACO 10 class -> recyclable + other)

通过全局多标签分层抽样，重新划分 train/val/test：
- 保证 4 个类别 (0,1,2,3) 都在各个 split 中有代表性样本
- 采用 7 : 1.5 : 1.5 的划分比例
- 确保每个 split 中每类 box 数量不少于一定阈值（在总量允许的前提下）
"""

import os
import shutil
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List
import random

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

# ================= 配置区域 =================

# 源数据集根目录
BASE_DIR = r"d:\Data\code\YOLOv8 - new\data\datasets\newdata"

# 输出目录（使用新的目录，避免与旧结果混淆）
OUTPUT_DIR = r"d:\Data\code\YOLOv8 - new\data\datasets\kitchen_mixed_stratified"

# 统一 4 类定义
UNIFIED_CLASSES: Dict[int, str] = {
    0: "kitchen_waste",  # 厨余垃圾
    1: "recyclable",     # 可回收物
    2: "hazardous",      # 有害垃圾
    3: "other",          # 其他垃圾
}

# 各子数据集配置
DATASETS: Dict[str, Dict] = {
    "domestic": {
        # 注意：domestic_yolo 在 newdata 下也有 images/labels 目录
        "img_dir": os.path.join(BASE_DIR, "domestic_yolo", "images", "train"),
        "lbl_dir": os.path.join(BASE_DIR, "domestic_yolo", "labels", "train"),
        "prefix": "dom_",
    },
    "hazard_a": {
        "path": os.path.join(BASE_DIR, "A"),
        "prefix": "ha_",
        "has_splits": True,
    },
    "hazard_b": {
        "path": os.path.join(BASE_DIR, "B"),
        "prefix": "hb_",
        "has_splits": True,
    },
    "taco": {
        "path": os.path.join(BASE_DIR, "C"),
        "prefix": "taco_",
        "has_splits": True,
    },
}

# 各子数据集到统一 4 类的映射
DOMESTIC_MAPPING: Dict[int, int] = {
    0: 0,  # organic -> kitchen_waste
    1: 1,  # non-organic -> recyclable
}

MEDICAL_A_MAPPING: Dict[int, int] = {i: 2 for i in range(17)}  # 全部 -> hazardous
MEDICAL_B_MAPPING: Dict[int, int] = {i: 2 for i in range(18)}  # 全部 -> hazardous

TACO_MAPPING: Dict[int, int] = {
    0: 1,  # Bottle -> recyclable
    1: 1,  # Bottle cap -> recyclable
    2: 1,  # Can -> recyclable
    3: 3,  # Cigarette -> other
    4: 1,  # Cup -> recyclable
    5: 1,  # Lid -> recyclable
    6: 3,  # Other -> other
    7: 1,  # Plastic bag and wrapper -> recyclable
    8: 1,  # Pop tab -> recyclable
    9: 1,  # Straw -> recyclable
}

CLASS_MAPPINGS: Dict[str, Dict[int, int]] = {
    "domestic": DOMESTIC_MAPPING,
    "hazard_a": MEDICAL_A_MAPPING,
    "hazard_b": MEDICAL_B_MAPPING,
    "taco": TACO_MAPPING,
}

# 划分比例：train : val : test = 7 : 1.5 : 1.5
SPLIT_RATIOS: Dict[str, float] = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}

# 每个 split 每个类别的期望最少 box 数（在总量允许时尽量满足）
MIN_BOXES_PER_CLASS_PER_SPLIT = 100

# 随机种子
RANDOM_SEED = 42


# ================= 数据结构 =================

@dataclass
class ImageInfo:
    dataset: str
    src_img_path: str
    dst_img_name: str
    remapped_lines: List[str]
    class_counts: Counter
    total_boxes: int


# ================= 工具函数 =================

def find_image_for_label(img_dir: str, stem: str) -> str:
    """根据标签 stem 在 img_dir 中查找图片文件。"""
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    for ext in exts:
        img_path = os.path.join(img_dir, stem + ext)
        if os.path.exists(img_path):
            return img_path
    return None


def remap_label_to_memory(label_path: str, mapping: Dict[int, int]) -> (List[str], Counter):
    """将单个标签文件重映射到 4 类，返回行列表和类计数。"""
    new_lines: List[str] = []
    cls_counter: Counter = Counter()

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                old_id = int(parts[0])
            except ValueError:
                continue
            if old_id not in mapping:
                continue
            new_id = mapping[old_id]
            parts[0] = str(new_id)
            new_line = " ".join(parts)
            new_lines.append(new_line)
            cls_counter[new_id] += 1

    return new_lines, cls_counter


# ================= 数据收集 =================

def collect_domestic_images() -> List[ImageInfo]:
    infos: List[ImageInfo] = []
    cfg = DATASETS["domestic"]
    img_dir = cfg["img_dir"]
    lbl_dir = cfg["lbl_dir"]
    mapping = CLASS_MAPPINGS["domestic"]

    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"[WARN] domestic 路径不存在: img_dir={img_dir}, lbl_dir={lbl_dir}")
        return infos

    label_files = [f for f in os.listdir(lbl_dir) if f.lower().endswith(".txt")]
    print(f"[INFO] domestic: 找到 {len(label_files)} 个标签文件")

    for lbl_name in tqdm(label_files, desc="  domestic", leave=False):
        stem = Path(lbl_name).stem
        img_path = find_image_for_label(img_dir, stem)
        if img_path is None:
            continue
        label_path = os.path.join(lbl_dir, lbl_name)
        remapped_lines, cls_counter = remap_label_to_memory(label_path, mapping)
        if not remapped_lines:
            continue

        dst_img_name = cfg["prefix"] + Path(img_path).name
        infos.append(
            ImageInfo(
                dataset="domestic",
                src_img_path=img_path,
                dst_img_name=dst_img_name,
                remapped_lines=remapped_lines,
                class_counts=cls_counter,
                total_boxes=sum(cls_counter.values()),
            )
        )

    return infos


def collect_split_images_from_path(dataset_name: str) -> List[ImageInfo]:
    """从有 train/valid/test 划分的数据集 (A/B/C) 收集样本。"""
    infos: List[ImageInfo] = []
    cfg = DATASETS[dataset_name]
    base = cfg["path"]
    prefix = cfg["prefix"]
    mapping = CLASS_MAPPINGS[dataset_name]

    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(base, split, "images")
        lbl_dir = os.path.join(base, split, "labels")
        if not os.path.exists(lbl_dir):
            print(f"[WARN] {dataset_name}/{split} 标签目录不存在: {lbl_dir}")
            continue
        label_files = [f for f in os.listdir(lbl_dir) if f.lower().endswith(".txt")]
        print(f"[INFO] {dataset_name}/{split}: {len(label_files)} 个标签文件")

        for lbl_name in tqdm(label_files, desc=f"  {dataset_name}/{split}", leave=False):
            stem = Path(lbl_name).stem
            img_path = find_image_for_label(img_dir, stem)
            if img_path is None:
                continue
            label_path = os.path.join(lbl_dir, lbl_name)
            remapped_lines, cls_counter = remap_label_to_memory(label_path, mapping)
            if not remapped_lines:
                continue

            dst_img_name = prefix + Path(img_path).name
            infos.append(
                ImageInfo(
                    dataset=dataset_name,
                    src_img_path=img_path,
                    dst_img_name=dst_img_name,
                    remapped_lines=remapped_lines,
                    class_counts=cls_counter,
                    total_boxes=sum(cls_counter.values()),
                )
            )

    return infos


def collect_all_images() -> List[ImageInfo]:
    """从 4 个子数据集收集所有样本，并统一到 4 类。"""
    all_infos: List[ImageInfo] = []

    # domestic
    print("[STEP] 收集 domestic_yolo 样本 (class 0/1)...")
    all_infos.extend(collect_domestic_images())

    # A/B/C
    for name in ["hazard_a", "hazard_b", "taco"]:
        print(f"[STEP] 收集 {name} 样本...")
        all_infos.extend(collect_split_images_from_path(name))

    # 全局统计
    total_cls = Counter()
    for info in all_infos:
        total_cls.update(info.class_counts)

    print("\n[INFO] 全局类别盒数统计：")
    for cid in sorted(total_cls.keys()):
        print(f"  class {cid} ({UNIFIED_CLASSES.get(cid, 'unknown')}): {total_cls[cid]} boxes")

    if total_cls.get(0, 0) == 0:
        print("[ERROR] 没有收集到任何 class 0 (kitchen_waste) 样本，请检查 domestic_yolo 数据路径和标签！")

    print(f"\n[INFO] 共收集到 {len(all_infos)} 张图片")
    return all_infos


# ================= 划分逻辑 =================

def simple_random_split(images: List[ImageInfo]) -> Dict[str, List[ImageInfo]]:
    """按照 SPLIT_RATIOS 对图像做简单随机划分。"""
    random.seed(RANDOM_SEED)
    images = images[:]
    random.shuffle(images)

    n = len(images)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    n_test = n - n_train - n_val

    assignments: Dict[str, List[ImageInfo]] = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    print("\n[INFO] 初始随机划分：")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {len(assignments[split])} images")

    return assignments


def compute_class_boxes_per_split(assignments: Dict[str, List[ImageInfo]]) -> Dict[str, Counter]:
    stats: Dict[str, Counter] = {s: Counter() for s in assignments.keys()}
    for split, infos in assignments.items():
        for info in infos:
            stats[split].update(info.class_counts)
    return stats


def ensure_min_boxes_per_class(
    assignments: Dict[str, List[ImageInfo]],
    min_boxes: int = MIN_BOXES_PER_CLASS_PER_SPLIT,
) -> Dict[str, List[ImageInfo]]:
    """
    在可能的前提下，保证每个 split 中每个类别至少有 min_boxes 个 box：
    - 先计算全局每类盒数，如果总量不足，则降低该类的阈值（均匀分配到各 split）。
    - 通过在 split 间搬运包含该类的图片来增加盒数。
    """
    splits = list(assignments.keys())
    global_cls = Counter()
    for split, infos in assignments.items():
        for info in infos:
            global_cls.update(info.class_counts)

    # 按全局情况为每个类计算实际可行的最小阈值
    per_class_min: Dict[int, int] = {}
    for cid, total in global_cls.items():
        if total == 0:
            per_class_min[cid] = 0
            continue
        # 理论上每个 split 至少能分到的平均数量
        avg_per_split = total // len(splits)
        target_min = min(min_boxes, avg_per_split)
        # 至少 1 个 box
        per_class_min[cid] = max(1, target_min) if avg_per_split > 0 else 0

    print("\n[INFO] 每类每个 split 的最小 box 要求：")
    for cid, v in sorted(per_class_min.items()):
        print(f"  class {cid} ({UNIFIED_CLASSES.get(cid, 'unknown')}): min {v} boxes/ split")

    # 当前各 split 盒数
    current = compute_class_boxes_per_split(assignments)

    # 对每个类、每个 split 检查是否低于要求
    for cid, required_min in per_class_min.items():
        if required_min == 0:
            continue
        for split in splits:
            if current[split][cid] >= required_min:
                continue

            needed = required_min - current[split][cid]
            if needed <= 0:
                continue

            # 从其他 split 搬运包含该类的图片
            for other in splits:
                if other == split:
                    continue
                # other 也必须有足够富余
                # 为简单起见，只要 other 当前盒数 > required_min 就允许搬运
                if current[other][cid] <= required_min:
                    continue

                moved_infos: List[ImageInfo] = []
                for info in assignments[other]:
                    if info.class_counts.get(cid, 0) > 0:
                        moved_infos.append(info)
                        current[other].subtract(info.class_counts)
                        current[split].update(info.class_counts)
                        needed -= info.class_counts[cid]
                        if needed <= 0:
                            break

                # 真正移动
                for info in moved_infos:
                    assignments[other].remove(info)
                    assignments[split].append(info)

                if needed <= 0:
                    break

            print(
                f"[INFO] after balancing: split={split}, class {cid} has {current[split][cid]} boxes (required >= {required_min})"
            )

    return assignments


# ================= 输出与验证 =================

def write_dataset(assignments: Dict[str, List[ImageInfo]]) -> None:
    """将划分结果写入 OUTPUT_DIR 下的 images/labels 结构。"""
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(OUTPUT_DIR, "images", split)
        lbl_dir = os.path.join(OUTPUT_DIR, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for info in tqdm(assignments[split], desc=f"  写入 {split}", leave=False):
            dst_img_path = os.path.join(img_dir, info.dst_img_name)
            dst_lbl_name = Path(info.dst_img_name).stem + ".txt"
            dst_lbl_path = os.path.join(lbl_dir, dst_lbl_name)

            shutil.copy2(info.src_img_path, dst_img_path)
            with open(dst_lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(info.remapped_lines) + "\n")


def summarize_dataset(assignments: Dict[str, List[ImageInfo]]) -> None:
    print("\n" + "=" * 70)
    print("合并后数据集统计：")
    print("=" * 70)

    for split in ["train", "val", "test"]:
        infos = assignments[split]
        img_count = len(infos)
        cls_counter = Counter()
        box_count = 0
        for info in infos:
            cls_counter.update(info.class_counts)
            box_count += info.total_boxes

        print(f"{split.upper():8s}: {img_count:5d} 张图片, {box_count:6d} 个边界框")
        for cid in sorted(cls_counter.keys()):
            print(f"    class {cid} ({UNIFIED_CLASSES.get(cid, 'unknown')}): {cls_counter[cid]} boxes")


def create_data_yaml() -> None:
    import yaml

    yaml_content = {
        "path": ".",  # 相对于 data.yaml 所在目录
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(UNIFIED_CLASSES),
        "names": {int(k): v for k, v in UNIFIED_CLASSES.items()},
    }

    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    print(f"\n[INFO] 已生成 data.yaml: {yaml_path}")


# ================= 主入口 =================

def main() -> None:
    print("=" * 70)
    print("Kitchen Mixed Dataset Builder - Stratified Version")
    print("=" * 70)
    print(f"源数据根目录: {BASE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}\n")

    # 1. 收集数据
    all_images = collect_all_images()
    if not all_images:
        print("[ERROR] 未收集到任何样本，脚本终止。")
        return

    # 2. 随机划分
    assignments = simple_random_split(all_images)

    # 3. 保证每个 split 每类 box 数不低于阈值（在总量允许的前提下）
    assignments = ensure_min_boxes_per_class(assignments, min_boxes=MIN_BOXES_PER_CLASS_PER_SPLIT)

    # 4. 写入数据集
    write_dataset(assignments)

    # 5. 打印统计
    summarize_dataset(assignments)

    # 6. 生成 data.yaml
    create_data_yaml()

    print("\n" + "=" * 70)
    print("[DONE] 分层抽样 + 数据集合并完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("可以使用该 data.yaml 进行 YOLOv8 训练。")
    print("=" * 70)


if __name__ == "__main__":
    main()
