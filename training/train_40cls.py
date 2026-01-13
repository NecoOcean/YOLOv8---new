# train_40cls.py
# coding:utf-8
"""
基于YOLOv8的垃圾目标检测算法 - 40类精细化模型训练脚本
"""
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    # 加载预训练模型
    # yolov8n: 最轻量，速度快（推荐用于快速验证）
    # yolov8s: 小型，平衡速度和精度（推荐用于实际部署）
    # yolov8m: 中型，精度更高（推荐用于追求更高精度）
    model = YOLO('yolov8s.pt')  # 使用小型模型以获得更好的精度
    
    # 训练配置 - 40类精细化垃圾分类
    results = model.train(
        data='datasets/data_40cls.yaml',     # 40类数据集配置
        epochs=150,                           # 训练轮次（增加以提高精度）
        imgsz=640,                            # 输入图像尺寸
        batch=16,                             # 批次大小（根据显存调整，8GB显存建议8-16）
        cos_lr=True,                          # 余弦学习率调度
        optimizer='Adam',                     # 优化器
        device='0',                           # GPU设备，无GPU使用'cpu'
        patience=30,                          # 早停耐心值（增加以避免过早停止）
        save=True,                            # 保存模型
        project='runs/detect',                # 输出目录
        name='garbage_40cls',                 # 实验名称 - 40类精细化分类
        
        # 数据增强参数（针对垃圾识别优化）
        hsv_h=0.015,                          # 色调增强
        hsv_s=0.7,                            # 饱和度增强
        hsv_v=0.4,                            # 明度增强
        degrees=15.0,                         # 旋转角度（增加以适应不同摆放）
        translate=0.1,                        # 平移比例
        scale=0.5,                            # 缩放比例
        fliplr=0.5,                           # 左右翻转概率
        flipud=0.1,                           # 上下翻转概率（垃圾可能倒置）
        mosaic=1.0,                           # 马赛克增强
        mixup=0.1,                            # MixUp增强
        
        # 性能优化参数
        workers=8,                            # 数据加载线程数
        cache=True,                           # 缓存图像以加速训练
    )
    
    # 验证模型
    print("\n" + "="*60)
    print("模型验证结果 (40类精细化分类):")
    print("="*60)
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    # 按类别输出精度（可选）
    if hasattr(metrics.box, 'ap_class_index'):
        print("\n各类别AP50:")
        for i, ap in enumerate(metrics.box.ap50):
            if ap > 0:
                print(f"  Class {i}: {ap:.4f}")
    
    # 导出模型（可选）
    # model.export(format='onnx')
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"\n最佳模型保存在: runs/detect/garbage_40cls/weights/best.pt")
    print("\n下一步操作:")
    print("1. 将 best.pt 复制到 models/ 目录并重命名为 best_40cls.pt")
    print("2. 将 Config_40cls.py 复制为 Config.py（或修改 main.py 导入）")
    print("3. 运行 python main.py 启动应用")
