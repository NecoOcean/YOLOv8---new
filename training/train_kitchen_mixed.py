# training/train_kitchen_mixed.py
# -*- coding: utf-8 -*-
"""
厨房混合垃圾数据集专用训练脚本
支持 kitchen_mixed 数据集（4类：厨余/可回收/有害/其他）
针对 autoDL 云服务器优化

使用方法:
    # 基础训练
    python training/train_kitchen_mixed.py
    
    # 自定义参数
    python training/train_kitchen_mixed.py --epochs 200 --batch 32 --model yolov8s.pt
    
    # 多GPU训练
    python training/train_kitchen_mixed.py --device 0,1 --batch 64
    
    # 使用预训练模型继续训练
    python training/train_kitchen_mixed.py --resume --weights path/to/last.pt
"""
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import shutil
import yaml

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ========== 数据集配置 ==========
DATASET_CONFIG = {
    'name': 'kitchen_mixed_stratified',
    'data_yaml': 'data/datasets/kitchen_mixed_stratified/data.yaml',
    'classes': ['kitchen_waste', 'recyclable', 'hazardous', 'other'],
    'nc': 4,
    'description': '混合厨房垃圾数据集（TU Wien + 医疗废物 + TACO，分层抽样版本）',
}

# ========== 模型配置 ==========
MODEL_CONFIGS = {
    'yolov8n': {
        'weights': 'yolov8n.pt',
        'description': 'YOLOv8 Nano - 速度快，适合边缘设备',
        'recommended_batch': 64,
    },
    'yolov8s': {
        'weights': 'yolov8s.pt', 
        'description': 'YOLOv8 Small - 平衡型，推荐使用',
        'recommended_batch': 32,
    },
    'yolov8m': {
        'weights': 'yolov8m.pt',
        'description': 'YOLOv8 Medium - 更高精度',
        'recommended_batch': 16,
    },
    'yolov8l': {
        'weights': 'yolov8l.pt',
        'description': 'YOLOv8 Large - 高精度',
        'recommended_batch': 8,
    },
    'yolov8x': {
        'weights': 'yolov8x.pt',
        'description': 'YOLOv8 XLarge - 最高精度',
        'recommended_batch': 4,
    },
}

# ========== 数据增强策略 ==========
AUGMENTATION_PRESETS = {
    'light': {
        'hsv_h': 0.01,
        'hsv_s': 0.5,
        'hsv_v': 0.3,
        'degrees': 5.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.5,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'erasing': 0.0,
    },
    'standard': {
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.15,
        'scale': 0.7,
        'shear': 3.0,
        'perspective': 0.0005,
        'flipud': 0.3,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.2,
        'copy_paste': 0.2,
        'erasing': 0.3,
    },
    'aggressive': {
        'hsv_h': 0.02,
        'hsv_s': 0.8,
        'hsv_v': 0.5,
        'degrees': 20.0,
        'translate': 0.25,
        'scale': 0.9,
        'shear': 10.0,
        'perspective': 0.001,
        'flipud': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.4,
        'copy_paste': 0.4,
        'erasing': 0.5,
    },
}


def get_autodl_optimizations():
    """
    获取 autoDL 云服务器优化参数
    """
    optimizations = {
        'workers': 8,  # autoDL 通常有较好的CPU配置
        'cache': 'ram',  # 利用大内存缓存数据
        'amp': True,  # 启用混合精度训练
        'close_mosaic': 10,  # 最后10轮关闭mosaic增强
    }
    
    # 检测是否在 autoDL 环境
    if os.path.exists('/root/autodl-tmp') or os.path.exists('/root/autodl-fs'):
        print("检测到 autoDL 环境，启用云服务器优化...")
        optimizations['project'] = '/root/autodl-tmp/training_runs'
    else:
        optimizations['project'] = str((PROJECT_ROOT / 'training' / 'runs').resolve())
    
    return optimizations


def train_kitchen_mixed(
    model_type: str = 'yolov8s',
    epochs: int = 150,
    batch: int = None,
    imgsz: int = 640,
    device: str = '0',
    resume: bool = False,
    weights: str = None,
    augment: str = 'standard',
    lr0: float = 0.001,
    lrf: float = 0.01,
    optimizer: str = 'AdamW',
    patience: int = 50,
    save_period: int = 20,
    name: str = None,
    exist_ok: bool = False,
):
    """
    训练 kitchen_mixed 数据集
    
    Args:
        model_type: 模型类型 (yolov8n/s/m/l/x)
        epochs: 训练轮次
        batch: 批次大小 (None则自动设置)
        imgsz: 输入图像尺寸
        device: 设备 (0/1/0,1/cpu)
        resume: 是否从上次中断处继续训练
        weights: 自定义权重路径
        augment: 数据增强策略 (light/standard/aggressive)
        lr0: 初始学习率
        lrf: 最终学习率比例
        optimizer: 优化器 (SGD/Adam/AdamW)
        patience: 早停耐心值
        save_period: 保存周期
        name: 实验名称 (None则自动生成)
        exist_ok: 是否允许覆盖已存在的实验
    """
    
    # 获取模型配置
    if model_type not in MODEL_CONFIGS:
        print(f"错误: 未知模型类型 '{model_type}'")
        print(f"可用模型: {list(MODEL_CONFIGS.keys())}")
        return None
    
    model_config = MODEL_CONFIGS[model_type]
    
    # 设置批次大小
    if batch is None:
        batch = model_config['recommended_batch']
    
    # 获取增强配置
    if augment not in AUGMENTATION_PRESETS:
        print(f"警告: 未知增强策略 '{augment}'，使用 'standard'")
        augment = 'standard'
    aug_config = AUGMENTATION_PRESETS[augment]
    
    # 获取 autoDL 优化参数
    autodl_opts = get_autodl_optimizations()
    
    # 数据集路径
    data_yaml = str(PROJECT_ROOT / DATASET_CONFIG['data_yaml'])

    # 校验 data.yaml 中的类别定义是否与配置一致
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        yaml_nc = yaml_data.get("nc")
        yaml_names = yaml_data.get("names")
        if yaml_nc != DATASET_CONFIG['nc']:
            print(f"[WARN] data.yaml nc={yaml_nc} 与配置 nc={DATASET_CONFIG['nc']} 不一致")
        loaded_names = None
        if isinstance(yaml_names, dict):
            # 按 key 排序，提取名称
            loaded_names = [yaml_names[i] for i in sorted(yaml_names.keys())]
        else:
            loaded_names = yaml_names
        if loaded_names and list(loaded_names) != DATASET_CONFIG['classes']:
            print(f"[WARN] data.yaml names={loaded_names} 与配置 classes={DATASET_CONFIG['classes']} 不一致")
    except Exception as e:
        print(f"[WARN] 无法读取或解析 data.yaml: {e}")
    
    # 生成实验名称
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"kitchen_mixed_{model_type}_{timestamp}"
    
    # 加载模型
    if weights and Path(weights).exists():
        print(f"加载自定义权重: {weights}")
        model = YOLO(weights)
    else:
        print(f"加载预训练模型: {model_config['weights']}")
        model = YOLO(model_config['weights'])
    
    # 打印训练配置
    print("\n" + "=" * 60)
    print("Kitchen Mixed 数据集训练配置")
    print("=" * 60)
    print(f"数据集: {DATASET_CONFIG['name']}")
    print(f"类别数: {DATASET_CONFIG['nc']} ({', '.join(DATASET_CONFIG['classes'])})")
    print(f"模型: {model_type} - {model_config['description']}")
    print(f"训练轮次: {epochs}")
    print(f"批次大小: {batch}")
    print(f"图像尺寸: {imgsz}")
    print(f"设备: {device}")
    print(f"增强策略: {augment}")
    print(f"学习率: {lr0} -> {lr0 * lrf}")
    print(f"优化器: {optimizer}")
    print(f"早停耐心值: {patience}")
    print(f"项目路径: {autodl_opts['project']}")
    print(f"实验名称: {name}")
    print("=" * 60 + "\n")
    
    # 训练参数
    train_args = {
        # 基础参数
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'resume': resume,
        'project': autodl_opts['project'],
        'name': name,
        'exist_ok': exist_ok,
        
        # 学习率策略
        'optimizer': optimizer,
        'lr0': lr0,
        'lrf': lrf,
        'cos_lr': True,
        'warmup_epochs': 5,
        'warmup_momentum': 0.8,
        
        # 训练控制
        'patience': patience,
        'save': True,
        'save_period': save_period,
        
        # 数据增强
        **aug_config,
        
        # autoDL 优化
        'workers': autodl_opts['workers'],
        'amp': autodl_opts['amp'],
        'close_mosaic': autodl_opts.get('close_mosaic', 10),
        
        # 其他
        'verbose': True,
        'plots': True,
    }
    
    # 如果支持缓存到RAM
    if autodl_opts.get('cache') == 'ram':
        train_args['cache'] = 'ram'
    
    # 开始训练
    results = model.train(**train_args)
    
    # 验证
    print("\n" + "=" * 60)
    print("模型验证结果:")
    print("=" * 60)
    metrics = model.val(
        save=False,
        save_txt=False,
        save_json=False,
        project=autodl_opts['project'],
        name=f"{name}_val",
        exist_ok=True,
    )
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    # 输出路径
    output_dir = Path(autodl_opts['project']) / name
    print(f"\n训练完成！")
    print(f"最佳模型: {output_dir / 'weights' / 'best.pt'}")
    print(f"最新模型: {output_dir / 'weights' / 'last.pt'}")
    
    # 自动复制到 data/models/trained，按数据集命名规范
    best_model_src = output_dir / 'weights' / 'best.pt'
    trained_dir = PROJECT_ROOT / 'data' / 'models' / 'trained'
    trained_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y%m%d")
    model_prefix = "best_kitchen_mixed_stratified"
    suffix = today
    candidate = trained_dir / f"{model_prefix}_{suffix}.pt"
    version = 1
    while candidate.exists():
        suffix = f"{today}_v{version}"
        candidate = trained_dir / f"{model_prefix}_{suffix}.pt"
        version += 1

    if best_model_src.exists():
        shutil.copy(best_model_src, candidate)
        print(f"已复制最佳模型到: {candidate}")

        # 兼容旧路径: 更新 best_mixed.pt 别名（供现有配置/UI 使用）
        alias_path = trained_dir / 'best_mixed.pt'
        shutil.copy(candidate, alias_path)
        print(f"已更新当前使用模型别名: {alias_path}")

        # 保存当前 data.yaml 快照到同一目录
        try:
            yaml_target = trained_dir / f"data_kitchen_mixed_stratified_{suffix}.yaml"
            shutil.copy(data_yaml, yaml_target)
            print(f"已复制数据配置到: {yaml_target}")
        except Exception as e:
            print(f"[WARN] 复制 data.yaml 失败: {e}")
    
    return results, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Kitchen Mixed 数据集训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础训练
  python training/train_kitchen_mixed.py
  
  # 使用大模型
  python training/train_kitchen_mixed.py --model yolov8m --epochs 200
  
  # 多GPU训练
  python training/train_kitchen_mixed.py --device 0,1 --batch 64
  
  # 自定义学习率
  python training/train_kitchen_mixed.py --lr0 0.0005 --optimizer AdamW
  
  # 继续训练
  python training/train_kitchen_mixed.py --resume --weights path/to/last.pt
        """
    )
    
    # 模型参数
    parser.add_argument('--model', type=str, default='yolov8s',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='模型类型 (默认: yolov8s)')
    parser.add_argument('--weights', type=str, default=None,
                        help='自定义权重路径')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, 
                        help='训练轮次 (默认: 100)')
    parser.add_argument('--batch', type=int, default=None, 
                        help='批次大小 (默认: 自动)')
    parser.add_argument('--imgsz', type=int, default=640, 
                        help='输入图像尺寸 (默认: 640)')
    parser.add_argument('--device', type=str, default='0', 
                        help='设备: 0/1/0,1/cpu (默认: 0)')
    parser.add_argument('--resume', action='store_true', 
                        help='继续训练')
    
    # 学习率参数
    parser.add_argument('--lr0', type=float, default=0.001, 
                        help='初始学习率 (默认: 0.001)')
    parser.add_argument('--lrf', type=float, default=0.01, 
                        help='最终学习率比例 (默认: 0.01)')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['SGD', 'Adam', 'AdamW'],
                        help='优化器 (默认: AdamW)')
    
    # 增强和训练控制
    parser.add_argument('--augment', type=str, default='standard',
                        choices=list(AUGMENTATION_PRESETS.keys()),
                        help='数据增强策略 (默认: standard)')
    parser.add_argument('--patience', type=int, default=50, 
                        help='早停耐心值 (默认: 50)')
    parser.add_argument('--save-period', type=int, default=20, 
                        help='保存周期 (默认: 20)')
    
    # 实验参数
    parser.add_argument('--name', type=str, default=None, 
                        help='实验名称 (默认: 自动生成)')
    parser.add_argument('--exist-ok', action='store_true', 
                        help='允许覆盖已存在的实验')
    
    args = parser.parse_args()
    
    # 显示配置信息
    print("\n" + "=" * 60)
    print("Kitchen Mixed 数据集训练脚本")
    print("=" * 60)
    print("\n可用模型:")
    for name, config in MODEL_CONFIGS.items():
        print(f"  {name}: {config['description']}")
    print("\n可用增强策略:")
    for name in AUGMENTATION_PRESETS.keys():
        print(f"  {name}")
    print("=" * 60)
    
    # 开始训练
    train_kitchen_mixed(
        model_type=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        resume=args.resume,
        weights=args.weights,
        augment=args.augment,
        lr0=args.lr0,
        lrf=args.lrf,
        optimizer=args.optimizer,
        patience=args.patience,
        save_period=args.save_period,
        name=args.name,
        exist_ok=args.exist_ok,
    )


if __name__ == '__main__':
    main()
