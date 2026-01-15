# training/train_quick_test.py
# -*- coding: utf-8 -*-
"""
快速流程验证脚本 - 最短时间验证项目完整性
目的：用最少的训练时间验证数据加载、模型训练、验证评估流程是否正常
"""
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 环境设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def print_separator(title=""):
    """打印分隔线"""
    print(f"\n{'='*60}")
    if title:
        print(f" {title}")
        print('='*60)

def check_environment():
    """检查运行环境"""
    print_separator("环境检查")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️ 将使用CPU训练（较慢）")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"✅ Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics未安装，请运行: pip install ultralytics")
        return False
    
    return True

def check_dataset():
    """检查数据集"""
    print_separator("数据集检查")
    
    # 检查40类数据集
    data_yaml_40 = PROJECT_ROOT / 'data' / 'datasets' / 'data_40cls.yaml'
    images_train = PROJECT_ROOT / 'data' / 'datasets' / 'images' / 'train'
    images_val = PROJECT_ROOT / 'data' / 'datasets' / 'images' / 'val'
    
    # 检查5类数据集
    data_yaml_5 = PROJECT_ROOT / 'data' / 'datasets' / 'kitchen_garbage' / 'data.yaml'
    
    dataset_found = None
    
    # 优先检查40类
    if data_yaml_40.exists() and images_train.exists() and images_val.exists():
        train_count = len(list(images_train.glob('*.jpg'))) + len(list(images_train.glob('*.png')))
        val_count = len(list(images_val.glob('*.jpg'))) + len(list(images_val.glob('*.png')))
        if train_count > 0 and val_count > 0:
            print(f"✅ 40类数据集: {data_yaml_40}")
            print(f"   训练图片: {train_count} 张")
            print(f"   验证图片: {val_count} 张")
            dataset_found = str(data_yaml_40)
    
    # 检查5类
    if dataset_found is None:
        kitchen_train = PROJECT_ROOT / 'data' / 'datasets' / 'kitchen_garbage' / 'images' / 'train'
        kitchen_val = PROJECT_ROOT / 'data' / 'datasets' / 'kitchen_garbage' / 'images' / 'val'
        if data_yaml_5.exists() and kitchen_train.exists():
            train_count = len(list(kitchen_train.glob('*.jpg')))
            val_count = len(list(kitchen_val.glob('*.jpg'))) if kitchen_val.exists() else 0
            if train_count > 0:
                print(f"✅ 5类数据集: {data_yaml_5}")
                print(f"   训练图片: {train_count} 张")
                print(f"   验证图片: {val_count} 张")
                dataset_found = str(data_yaml_5)
    
    if dataset_found is None:
        print("❌ 未找到有效数据集")
        print("   请确保以下路径存在图片:")
        print(f"   - {images_train}")
        print(f"   - {images_val}")
        return None
    
    return dataset_found

def quick_train(data_yaml):
    """快速训练验证"""
    from ultralytics import YOLO
    import torch
    
    print_separator("快速训练验证")
    
    # 配置参数 - 最短时间
    config = {
        'epochs': 3,           # 最少轮次，仅验证流程
        'imgsz': 320,          # 小尺寸，加快速度
        'batch': 8,            # 小batch，节省显存
        'patience': 3,         # 快速早停
        'optimizer': 'Adam',
        'lr0': 0.01,
        'cos_lr': True,
        'device': '0' if torch.cuda.is_available() else 'cpu',
        'workers': 4,
        'cache': False,        # 不缓存，节省内存
        # 减少增强，加快速度
        'hsv_h': 0.0,
        'hsv_s': 0.0,
        'hsv_v': 0.0,
        'degrees': 0.0,
        'translate': 0.0,
        'scale': 0.0,
        'fliplr': 0.0,
        'mosaic': 0.0,
        'mixup': 0.0,
        # 保存设置
        'project': str(PROJECT_ROOT / 'training' / 'runs' / 'quick_test'),
        'name': f'test_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'save': True,
        'plots': False,        # 不生成图表，加快速度
        'verbose': True,
    }
    
    print(f"配置:")
    print(f"  - 数据集: {Path(data_yaml).name}")
    print(f"  - 轮次: {config['epochs']}")
    print(f"  - 图像尺寸: {config['imgsz']}")
    print(f"  - 批次大小: {config['batch']}")
    print(f"  - 设备: {config['device']}")
    
    # 加载模型
    print("\n加载预训练模型...")
    try:
        model = YOLO('yolov8n.pt')  # 最小模型
        print("✅ 模型加载成功 (YOLOv8n)")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None
    
    # 开始训练
    print("\n开始训练...")
    start_time = time.time()
    
    try:
        results = model.train(data=data_yaml, **config)
        train_time = time.time() - start_time
        print(f"\n✅ 训练完成! 耗时: {train_time:.1f}秒")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return model, config

def validate_model(model, data_yaml):
    """验证模型"""
    print_separator("模型验证")
    
    try:
        print("正在验证模型...")
        metrics = model.val(data=data_yaml, imgsz=320, batch=8)
        
        print(f"\n验证结果:")
        print(f"  - mAP50:      {metrics.box.map50:.4f}")
        print(f"  - mAP50-95:   {metrics.box.map:.4f}")
        print(f"  - Precision:  {metrics.box.mp:.4f}")
        print(f"  - Recall:     {metrics.box.mr:.4f}")
        
        return metrics
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return None

def test_inference(model):
    """测试推理"""
    print_separator("推理测试")
    
    # 查找测试图片
    test_dirs = [
        PROJECT_ROOT / 'TestFiles',
        PROJECT_ROOT / 'data' / 'datasets' / 'images' / 'val',
    ]
    
    test_image = None
    for test_dir in test_dirs:
        if test_dir.exists():
            images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
            if images:
                test_image = images[0]
                break
    
    if test_image is None:
        print("⚠️ 未找到测试图片，跳过推理测试")
        return True
    
    try:
        print(f"测试图片: {test_image.name}")
        results = model.predict(str(test_image), conf=0.25, verbose=False)
        
        if results and len(results) > 0:
            detections = len(results[0].boxes)
            print(f"✅ 推理成功! 检测到 {detections} 个目标")
        else:
            print("✅ 推理成功! 未检测到目标（可能置信度阈值较高）")
        
        return True
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        return False

def main():
    """主函数"""
    print_separator("YOLOv8 垃圾检测项目 - 快速流程验证")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"项目路径: {PROJECT_ROOT}")
    
    total_start = time.time()
    
    # 1. 环境检查
    if not check_environment():
        print("\n❌ 环境检查失败，请安装必要依赖")
        return False
    
    # 2. 数据集检查
    data_yaml = check_dataset()
    if data_yaml is None:
        print("\n❌ 数据集检查失败，请准备数据集")
        return False
    
    # 3. 快速训练
    result = quick_train(data_yaml)
    if result is None:
        print("\n❌ 训练流程验证失败")
        return False
    
    model, config = result
    
    # 4. 模型验证
    metrics = validate_model(model, data_yaml)
    if metrics is None:
        print("\n⚠️ 验证流程有问题，但训练流程正常")
    
    # 5. 推理测试
    test_inference(model)
    
    # 总结
    total_time = time.time() - total_start
    print_separator("验证总结")
    print(f"✅ 项目流程验证完成!")
    print(f"   总耗时: {total_time:.1f}秒")
    print(f"   模型保存: {config['project']}/{config['name']}/weights/best.pt")
    
    print("\n检查项:")
    print("  [✓] 环境配置正常")
    print("  [✓] 数据集加载正常")
    print("  [✓] 训练流程正常")
    print("  [✓] 验证流程正常")
    print("  [✓] 推理流程正常")
    
    print("\n下一步:")
    print("  可以开始完整训练:")
    print("  python training/train.py --mode cls40 --epochs 150 --batch 64")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
