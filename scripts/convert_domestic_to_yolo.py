# scripts/convert_domestic_to_yolo.py
# -*- coding: utf-8 -*-
"""
将 TU Wien Domestic Organic Waste 数据集的分割 mask 转换为 YOLO 检测框格式

使用方法：
1. 下载数据集并解压到 datasets/domestic_raw/
2. 确认 images/ 和 masks/ 目录结构
3. 运行此脚本：python scripts/convert_domestic_to_yolo.py
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil

# ============ 配置区域 ============
# 源数据目录 - TU Wien 数据集解压后的位置
SRC_BASE_DIR = r"d:\d\D\TUW_2024-SmartTrashCan_Organic_Waste_Dataset"
SRC_IMG_DIR = os.path.join(SRC_BASE_DIR, "images")
SRC_MASK_DIR = os.path.join(SRC_BASE_DIR, "masks")

# 输出目录（YOLO格式）
OUT_BASE_DIR = r"data/datasets/domestic_yolo"
OUT_IMG_DIR = os.path.join(OUT_BASE_DIR, "images/train")
OUT_LABEL_DIR = os.path.join(OUT_BASE_DIR, "labels/train")

# RGB 颜色到类别ID的映射（根据 labelmap.csv）
# 注意：OpenCV 读取图片是 BGR 格式
COLOR_TO_CLASS = {
    (112, 224, 131): 0,  # organic (BGR) -> kitchen_waste (厨余垃圾)
    (94, 53, 255): 1,    # non-organic (BGR) -> recyclable (可回收物)
    # background (0,0,0) 不需要标注
}

# 是否使用灰度 mask（如果 mask 是灰度图）
USE_GRAYSCALE_MASK = False

# 灰度像素值到类别ID的映射（备用，如果是灰度 mask）
PIXEL_TO_CLASS = {
    1: 0,  # organic -> kitchen_waste
    2: 1,  # non-organic -> recyclable
}

# 最小边界框尺寸（像素），过滤噪声
MIN_BOX_SIZE = 10

# ============ 主逻辑 ============

def find_mask_path(img_name: str, mask_dir: str) -> str:
    """
    根据图片名找到对应的 mask 文件
    尝试多种常见的命名模式
    """
    base_name = Path(img_name).stem
    
    # 常见的 mask 命名模式
    patterns = [
        f"{base_name}_mask.png",
        f"{base_name}.png",
        f"{base_name}_mask.jpg",
        f"{base_name}_seg.png",
        f"{base_name}_label.png",
        img_name,  # 同名
    ]
    
    for pattern in patterns:
        mask_path = os.path.join(mask_dir, pattern)
        if os.path.exists(mask_path):
            return mask_path
    
    return None


def mask_to_yolo_boxes(mask: np.ndarray, img_width: int, img_height: int) -> list:
    """
    将分割 mask 转换为 YOLO 格式的边界框
    支持 RGB 彩色 mask 和灰度 mask
    
    返回: [(class_id, x_center, y_center, width, height), ...]
    """
    yolo_boxes = []
    
    if USE_GRAYSCALE_MASK or len(mask.shape) == 2:
        # 灰度 mask 模式
        for pixel_value, class_id in PIXEL_TO_CLASS.items():
            binary_mask = (mask == pixel_value).astype(np.uint8)
            if binary_mask.sum() == 0:
                continue
            yolo_boxes.extend(_extract_boxes_from_mask(binary_mask, class_id, img_width, img_height))
    else:
        # RGB/BGR 彩色 mask 模式
        for color_bgr, class_id in COLOR_TO_CLASS.items():
            # 创建该颜色的二值掩码
            lower = np.array(color_bgr, dtype=np.uint8)
            upper = np.array(color_bgr, dtype=np.uint8)
            binary_mask = cv2.inRange(mask, lower, upper)
            
            if binary_mask.sum() == 0:
                continue
            yolo_boxes.extend(_extract_boxes_from_mask(binary_mask, class_id, img_width, img_height))
    
    return yolo_boxes


def _extract_boxes_from_mask(binary_mask: np.ndarray, class_id: int, 
                              img_width: int, img_height: int) -> list:
    """从二值掩码中提取边界框"""
    boxes = []
    
    # 查找轮廓
    contours, _ = cv2.findContours(
        binary_mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    for contour in contours:
        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 过滤太小的框（噪声）
        if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
            continue
        
        # 转换为 YOLO 格式（归一化到 0-1）
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        norm_w = w / img_width
        norm_h = h / img_height
        
        boxes.append((class_id, x_center, y_center, norm_w, norm_h))
    
    return boxes


def convert_dataset():
    """转换整个数据集 - 支持 TU Wien 的子文件夹结构"""
    
    # 创建输出目录
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_LABEL_DIR, exist_ok=True)
    
    # 检查源目录
    if not os.path.exists(SRC_IMG_DIR):
        print(f"[ERROR] 图片目录不存在: {SRC_IMG_DIR}")
        print("请确保数据集已下载并解压到正确位置")
        return
    
    if not os.path.exists(SRC_MASK_DIR):
        print(f"[ERROR] Mask 目录不存在: {SRC_MASK_DIR}")
        print("请确保数据集已下载并解压到正确位置")
        return
    
    # TU Wien 数据集结构：images/{organic,non-organic,background}/
    # 收集所有子文件夹中的图片
    image_files = []
    subfolders = ['organic', 'non-organic', 'background']
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(SRC_IMG_DIR, subfolder)
        if os.path.exists(subfolder_path):
            for f in os.listdir(subfolder_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append((subfolder, f))
    
    # 如果没有子文件夹，尝试直接读取
    if not image_files:
        for f in os.listdir(SRC_IMG_DIR):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(('', f))
    
    if not image_files:
        print(f"[ERROR] 在 {SRC_IMG_DIR} 中没有找到图片文件")
        return
    
    print(f"[INFO] 找到 {len(image_files)} 张图片")
    print(f"[INFO] 颜色映射: {COLOR_TO_CLASS}")
    
    converted = 0
    skipped = 0
    total_boxes = 0
    
    for i, (subfolder, img_name) in enumerate(image_files, 1):
        # 构建完整路径
        if subfolder:
            img_path = os.path.join(SRC_IMG_DIR, subfolder, img_name)
            mask_subfolder = os.path.join(SRC_MASK_DIR, subfolder)
        else:
            img_path = os.path.join(SRC_IMG_DIR, img_name)
            mask_subfolder = SRC_MASK_DIR
        
        # 查找对应的 mask
        mask_path = find_mask_path(img_name, mask_subfolder)
        
        if not mask_path:
            print(f"[WARN] 未找到 mask: {img_name}")
            skipped += 1
            continue
        
        # 读取图片和 mask
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] 无法读取图片: {img_path}")
            skipped += 1
            continue
        
        # 根据配置决定读取方式
        if USE_GRAYSCALE_MASK:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)  # BGR 格式
        
        if mask is None:
            print(f"[WARN] 无法读取 mask: {mask_path}")
            skipped += 1
            continue
        
        h, w = img.shape[:2]
        
        # 转换为 YOLO 格式
        yolo_boxes = mask_to_yolo_boxes(mask, w, h)
        
        if not yolo_boxes:
            # 没有检测到任何物体
            skipped += 1
            continue
        
        # 复制图片到输出目录
        out_img_path = os.path.join(OUT_IMG_DIR, img_name)
        shutil.copy2(img_path, out_img_path)
        
        # 写入 YOLO 标签文件
        label_name = Path(img_name).stem + ".txt"
        out_label_path = os.path.join(OUT_LABEL_DIR, label_name)
        
        with open(out_label_path, 'w', encoding='utf-8') as f:
            for box in yolo_boxes:
                class_id, x_c, y_c, bw, bh = box
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
        
        converted += 1
        total_boxes += len(yolo_boxes)
        
        # 进度显示
        if i % 50 == 0 or i == len(image_files):
            print(f"[PROGRESS] {i}/{len(image_files)} ({converted} 转换, {skipped} 跳过)")
    
    print("\n" + "=" * 50)
    print("[DONE] 转换完成!")
    print(f"  - 成功转换: {converted} 张图片")
    print(f"  - 跳过: {skipped} 张")
    print(f"  - 生成边界框: {total_boxes} 个")
    print(f"  - 输出目录: {OUT_IMG_DIR}")
    print(f"  - 标签目录: {OUT_LABEL_DIR}")
    print("=" * 50)


def analyze_mask_values():
    """分析 mask 中的唯一像素值（帮助确定类别映射）"""
    
    if not os.path.exists(SRC_MASK_DIR):
        print(f"[ERROR] Mask 目录不存在: {SRC_MASK_DIR}")
        return
    
    mask_files = [f for f in os.listdir(SRC_MASK_DIR) 
                  if f.lower().endswith(('.png', '.jpg', '.bmp'))]
    
    if not mask_files:
        print("[ERROR] 没有找到 mask 文件")
        return
    
    print(f"[INFO] 分析 {len(mask_files)} 个 mask 文件...")
    
    all_values = set()
    
    for fname in mask_files[:50]:  # 只分析前50个
        mask_path = os.path.join(SRC_MASK_DIR, fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            unique_vals = np.unique(mask)
            all_values.update(unique_vals.tolist())
    
    print(f"\n[INFO] Mask 中发现的唯一像素值: {sorted(all_values)}")
    print("\n请根据数据集文档确定每个值对应的类别，然后修改 PIXEL_TO_CLASS 映射")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        # 分析模式：查看 mask 中的像素值
        analyze_mask_values()
    else:
        # 转换模式
        convert_dataset()
