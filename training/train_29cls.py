# train_29cls.py
# coding:utf-8
"""
基于YOLOv8的垃圾目标检测算法 - 29类厨房垃圾分类模型训练脚本

注意：使用此脚本前，需要确保标注数据已转换为29类格式。
如果使用 labels_converted/ 目录，当前只有5个类别有数据。
建议先运行 scripts/convert_labels.py 扩展映射规则，或使用 train_40cls.py 训练完整40类模型。
"""
from ultralytics import YOLO
from pathlib import Path
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

if __name__ == '__main__':
    # 数据集配置路径
    data_yaml = str(PROJECT_ROOT / 'data' / 'datasets' / 'data_29cls.yaml')
    
    print("=" * 60)
    print("29类厨房垃圾分类模型训练")
    print("=" * 60)
    print(f"数据集配置: {data_yaml}")
    
    # 检查标注目录
    labels_dir = PROJECT_ROOT / 'data' / 'datasets' / 'labels_converted' / 'train'
    if labels_dir.exists():
        # 分析当前标注的class_id分布
        class_ids = set()
        for txt_file in labels_dir.glob('*.txt'):
            with open(txt_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_ids.add(class_id)
        
        print(f"\n⚠️ 当前 labels_converted/ 目录中实际使用的 class_id: {sorted(class_ids)}")
        print(f"   共 {len(class_ids)} 个类别有标注数据")
        
        if len(class_ids) < 29:
            print("\n警告：标注数据不完整！")
            print("建议操作：")
            print("  1. 扩展 scripts/convert_labels.py 的映射规则")
            print("  2. 或使用 train_40cls.py 训练完整40类模型")
            print("  3. 或修改 data_29cls.yaml 使用原始 labels/ 目录")
            
            user_input = input("\n是否继续训练？(y/n): ")
            if user_input.lower() != 'y':
                print("训练已取消")
                exit(0)
    
    # 加载预训练模型
    print("\n加载预训练模型...")
    model = YOLO('yolov8s.pt')  # 使用小型模型以获得更好的精度
    
    # 训练配置 - 29类厨房垃圾分类
    print("\n开始训练...")
    results = model.train(
        data=data_yaml,                       # 29类数据集配置
        epochs=120,                           # 训练轮次
        imgsz=640,                            # 输入图像尺寸
        batch=16,                             # 批次大小（根据显存调整）
        cos_lr=True,                          # 余弦学习率调度
        optimizer='Adam',                     # 优化器
        device='0',                           # GPU设备，无GPU使用'cpu'
        patience=25,                          # 早停耐心值
        save=True,                            # 保存模型
        project='runs/detect',                # 输出目录
        name='garbage_29cls',                 # 实验名称
        
        # 数据增强参数
        hsv_h=0.015,                          # 色调增强
        hsv_s=0.7,                            # 饱和度增强
        hsv_v=0.4,                            # 明度增强
        degrees=10.0,                         # 旋转角度
        translate=0.1,                        # 平移比例
        scale=0.5,                            # 缩放比例
        fliplr=0.5,                           # 左右翻转概率
        mosaic=1.0,                           # 马赛克增强
        
        # 性能优化参数
        workers=8,                            # 数据加载线程数
        cache=True,                           # 缓存图像以加速训练
    )
    
    # 验证模型
    print("\n" + "=" * 60)
    print("模型验证结果 (29类厨房垃圾分类):")
    print("=" * 60)
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"\n最佳模型保存在: runs/detect/garbage_29cls/weights/best.pt")
    print("\n下一步操作:")
    print("1. 将 best.pt 复制到 data/models/trained/ 目录并重命名为 best_29cls.pt")
    print("2. 修改 src/config/settings.py 添加29类配置")
    print("3. 运行 python main.py 启动应用")
