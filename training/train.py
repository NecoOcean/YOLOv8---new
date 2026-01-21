# training/train.py
# -*- coding: utf-8 -*-
"""
基于YOLOv8的垃圾目标检测系统 - 统一训练入口
支持 5类、23类、40类 模型训练（默认23类）
"""
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径（使用 resolve() 确保绝对路径）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 训练配置
TRAIN_CONFIGS = {
    'cls4': {
        'data_yaml': 'data/datasets/kitchen_mixed/data.yaml',
        'base_model': 'yolov8n.pt',
        'name_prefix': 'kitchen_mixed_4cls',
        'description': '4类混合垃圾分类（厨余/可回收/有害/其他）',
    },
    'cls5': {
        'data_yaml': 'data/datasets/kitchen_garbage/data.yaml',
        'base_model': 'yolov8n.pt',
        'name_prefix': 'kitchen_garbage_5cls',
        'description': '5类简化版垃圾分类',
    },
    'cls23': {
        'data_yaml': 'data/datasets/kitchen_garbage_merged/data.yaml',
        'base_model': 'yolov8n.pt',
        'name_prefix': 'kitchen_garbage_23cls',
        'description': '23类厨房垃圾分类（标准版）',
    },
    'cls40': {
        'data_yaml': 'data/datasets/data_40cls.yaml',
        'base_model': 'yolov8s.pt',
        'name_prefix': 'garbage_40cls',
        'description': '40类精细化垃圾分类',
    },
    'mixed': {
        'data_yaml': 'data/datasets/kitchen_mixed/data.yaml',
        'base_model': 'yolov8s.pt',
        'name_prefix': 'kitchen_mixed',
        'description': '混合数据集训练（TU Wien + 医疗废物 + TACO）',
    },
}


def train_model(mode: str = 'cls23', epochs: int = 100, batch: int = 16, 
                device: str = '0', resume: bool = False):
    """
    训练模型
    
    Args:
        mode: 训练模式 'cls5', 'cls23' 或 'cls40'（默认cls23）
        epochs: 训练轮次
        batch: 批次大小
        device: 设备 '0' GPU或 'cpu'
        resume: 是否继续训练
    """
    
    # 获取配置
    if mode not in TRAIN_CONFIGS:
        print(f"错误: 未知模式 '{mode}'，可用模式: {list(TRAIN_CONFIGS.keys())}")
        return
    
    config = TRAIN_CONFIGS[mode]
    data_yaml = str(PROJECT_ROOT / config['data_yaml'])
    base_model = config['base_model']
    
    # 生成时间戳命名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f"{config['name_prefix']}_{timestamp}"
    
    print(f"\n{'='*50}")
    print(f"训练模式: {mode.upper()}")
    print(f"数据集配置: {data_yaml}")
    print(f"基础模型: {base_model}")
    print(f"训练轮次: {epochs}")
    print(f"批次大小: {batch}")
    print(f"设备: {device}")
    print(f"{'='*50}\n")
    
    # 加载模型
    model = YOLO(base_model)
    
    # 训练配置 (优化后的数据增强策略)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=device,
        resume=resume,
        project=str((PROJECT_ROOT / 'training' / 'runs').resolve()),
        name=project_name,
        
        # ========== 优化的学习率策略 (基于lr_finder结果) ==========
        optimizer='AdamW',      # 比 Adam 更稳定
        lr0=0.0005,             # 初始学习率 (基于lr_finder推荐，折中值)
        lrf=0.01,               # 最终学习率比例
        cos_lr=True,            # 余弦退火
        warmup_epochs=5,        # 预热轮次
        warmup_momentum=0.8,    # 预热动量
        
        # ========== 训练控制 ==========
        patience=50,            # 早停耐心值 (增加)
        save=True,
        save_period=50,         # 每50轮保存
        
        # ========== 强化数据增强策略 ==========
        # 颜色增强
        hsv_h=0.015,            # 色调变化
        hsv_s=0.7,              # 饱和度变化
        hsv_v=0.4,              # 亮度变化
        
        # 几何变换 (增强)
        degrees=15.0,           # 旋转角度 (10→15)
        translate=0.2,          # 平移比例 (0.1→0.2)
        scale=0.9,              # 缩放范围 (0.5→0.9)
        shear=5.0,              # 剪切角度 (新增)
        perspective=0.001,      # 透视变换 (新增)
        
        # 翻转
        flipud=0.5,             # 垂直翻转 (新增)
        fliplr=0.5,             # 水平翻转
        
        # 高级增强 (新增)
        mosaic=1.0,             # Mosaic增强
        mixup=0.3,              # MixUp增强 (新增)
        copy_paste=0.3,         # Copy-Paste (新增，针对小目标)
        erasing=0.4,            # 随机擦除 (新增)
        
        # 其他
        workers=8,              # 数据加载线程
        verbose=True,
        plots=True,
    )
    
    # 验证（不保存预测结果到额外目录）
    print("\n" + "="*50)
    print("模型验证结果:")
    print("="*50)
    metrics = model.val(
        save=False,           # 不保存预测图片
        save_txt=False,       # 不保存预测标签
        save_json=False,      # 不保存 JSON
        project=str((PROJECT_ROOT / 'training' / 'runs').resolve()),  # 验证结果也放在 training/runs
        name=f"{project_name}_val",
        exist_ok=True,
    )
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    # 输出路径
    output_dir = PROJECT_ROOT / 'training' / 'runs' / project_name
    print(f"\n训练完成！")
    print(f"最佳模型保存在: {output_dir / 'weights' / 'best.pt'}")
    
    # 自动复制模型到 models 目录
    model_dst_name = {
        'cls4': 'best_4cls.pt',
        'cls5': 'best_5cls.pt',
        'cls23': 'best_23cls.pt',
        'cls40': 'best_40cls.pt',
        'mixed': 'best_mixed.pt',
    }.get(mode, 'best.pt')
    
    best_model_src = output_dir / 'weights' / 'best.pt'
    best_model_dst = PROJECT_ROOT / 'data' / 'models' / 'trained' / model_dst_name
    
    if best_model_src.exists():
        import shutil
        best_model_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_model_src, best_model_dst)
        print(f"已自动复制模型到: {best_model_dst}")
    else:
        print(f"\n请执行以下命令复制模型:")
        print(f"copy \"{best_model_src}\" \"{best_model_dst}\"")


def main():
    parser = argparse.ArgumentParser(description='垃圾检测模型训练 - 统一入口')
    parser.add_argument('--mode', type=str, default='cls23', 
                        choices=['cls4', 'cls5', 'cls23', 'cls40', 'mixed'],
                        help='训练模式: cls4(4类混合), cls5(5类), cls23(23类,默认), cls40(40类), mixed(混合数据集)')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次 (默认100)')
    parser.add_argument('--batch', type=int, default=64, help='批次大小 (默认64)')
    parser.add_argument('--device', type=str, default='0', help='设备: 0/1/cpu')
    parser.add_argument('--resume', action='store_true', help='继续训练')
    
    args = parser.parse_args()
    
    # 显示可用配置
    print("\n" + "=" * 60)
    print("垃圾检测模型训练 - 统一入口")
    print("=" * 60)
    print("可用训练模式:")
    for mode_key, mode_config in TRAIN_CONFIGS.items():
        marker = " (默认)" if mode_key == 'cls23' else ""
        print(f"  --mode {mode_key}: {mode_config['description']}{marker}")
    print("=" * 60)
    
    train_model(
        mode=args.mode,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        resume=args.resume
    )


if __name__ == '__main__':
    main()
