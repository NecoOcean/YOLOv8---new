# train.py
# coding:utf-8
"""
基于YOLOv8的垃圾目标检测算法 - 模型训练脚本
"""
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    # 加载预训练模型
    # yolov8n: 最轻量，速度快
    # yolov8s: 小型，平衡速度和精度
    # yolov8m: 中型，精度更高
    model = YOLO('yolov8n.pt')
    
    # 训练配置 - 厨房垃圾分类（5个类别）
    results = model.train(
        data='datasets/kitchen_garbage/data.yaml',  # 整合后的数据集配置
        epochs=100,                              # 训练轮次
        imgsz=640,                               # 输入图像尺寸
        batch=16,                                # 批次大小（根据显存调整，8GB显存建议8-16）
        cos_lr=True,                             # 余弦学习率调度
        optimizer='Adam',                        # 优化器
        device='0',                              # GPU设备，无GPU使用'cpu'
        patience=20,                             # 早停耐心值
        save=True,                               # 保存模型
        project='runs/detect',                   # 输出目录
        name='kitchen_garbage_5cls',             # 实验名称 - 厨房垃圾分类(5类)
        # 数据增强参数
        hsv_h=0.015,                             # 色调增强
        hsv_s=0.7,                               # 饱和度增强
        hsv_v=0.4,                               # 明度增强
        degrees=10.0,                            # 旋转角度
        translate=0.1,                           # 平移比例
        scale=0.5,                               # 缩放比例
        fliplr=0.5,                              # 左右翻转概率
        mosaic=1.0,                              # 马赛克增强
    )
    
    # 验证模型
    print("\n" + "="*50)
    print("模型验证结果:")
    print("="*50)
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    # 导出模型（可选）
    # model.export(format='onnx')
    
    print("\n训练完成！")
    print(f"最佳模型保存在: runs/detect/kitchen_garbage_5cls/weights/best.pt")
    print("请将best.pt复制到models目录下使用")
