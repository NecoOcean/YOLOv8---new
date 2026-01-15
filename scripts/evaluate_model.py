#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 模型评估脚本
====================
功能：
1. 定量评估 - mAP、Precision、Recall 等指标
2. 类别分析 - 各类别 AP 排名
3. 推理速度测试 - FPS、延迟
4. 实际场景测试 - 检测率、置信度分布
5. 生成可视化报告 - 混淆矩阵、PR曲线等
6. 输出 JSON 评估报告

使用方法：
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --model path/to/model.pt --data path/to/data.yaml
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 PATH
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_environment():
    """检查运行环境"""
    print("\n" + "=" * 60)
    print("1. 环境检查")
    print("=" * 60)
    
    env_info = {}
    
    # Python 版本
    import platform
    env_info['python_version'] = platform.python_version()
    print(f"Python 版本: {env_info['python_version']}")
    
    # PyTorch 版本
    try:
        import torch
        env_info['torch_version'] = torch.__version__
        env_info['cuda_available'] = torch.cuda.is_available()
        if env_info['cuda_available']:
            env_info['gpu_name'] = torch.cuda.get_device_name(0)
            env_info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        print(f"PyTorch 版本: {env_info['torch_version']}")
        print(f"CUDA 可用: {env_info['cuda_available']}")
        if env_info['cuda_available']:
            print(f"GPU: {env_info['gpu_name']} ({env_info['gpu_memory']})")
    except ImportError:
        print("❌ PyTorch 未安装")
        return None
    
    # ultralytics 版本
    try:
        import ultralytics
        env_info['ultralytics_version'] = ultralytics.__version__
        print(f"Ultralytics 版本: {env_info['ultralytics_version']}")
    except ImportError:
        print("❌ ultralytics 未安装")
        return None
    
    print("✅ 环境检查通过")
    return env_info


def load_model(model_path):
    """加载模型"""
    print("\n" + "=" * 60)
    print("2. 加载模型")
    print("=" * 60)
    
    from ultralytics import YOLO
    
    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    model = YOLO(model_path)
    
    # 模型信息
    model_info = {
        'path': str(model_path),
        'file_size': f"{Path(model_path).stat().st_size / 1024**2:.2f}MB",
    }
    
    print(f"模型路径: {model_info['path']}")
    print(f"模型大小: {model_info['file_size']}")
    print("✅ 模型加载成功")
    
    return model, model_info


def quantitative_evaluation(model, data_yaml):
    """定量评估"""
    print("\n" + "=" * 60)
    print("3. 定量评估 (mAP, Precision, Recall)")
    print("=" * 60)
    
    # 运行验证
    metrics = model.val(data=data_yaml, plots=True, save_json=True, verbose=False)
    
    eval_results = {
        'mAP50': round(metrics.box.map50, 4),
        'mAP50-95': round(metrics.box.map, 4),
        'precision': round(metrics.box.mp, 4),
        'recall': round(metrics.box.mr, 4),
    }
    
    # 计算 F1 分数
    if eval_results['precision'] + eval_results['recall'] > 0:
        eval_results['f1_score'] = round(
            2 * eval_results['precision'] * eval_results['recall'] / 
            (eval_results['precision'] + eval_results['recall']), 4
        )
    else:
        eval_results['f1_score'] = 0.0
    
    print(f"\n{'指标':<15} {'值':<10} {'标准':<10} {'状态'}")
    print("-" * 50)
    
    standards = {
        'mAP50': 0.6,
        'mAP50-95': 0.4,
        'precision': 0.7,
        'recall': 0.7,
        'f1_score': 0.65,
    }
    
    for key, value in eval_results.items():
        standard = standards.get(key, 0)
        status = "✅ 达标" if value >= standard else "⚠️ 需提升"
        print(f"{key:<15} {value:<10.4f} {standard:<10} {status}")
    
    return eval_results, metrics


def class_analysis(metrics):
    """类别性能分析"""
    print("\n" + "=" * 60)
    print("4. 类别性能分析")
    print("=" * 60)
    
    class_names = list(metrics.names.values())
    ap50_list = metrics.box.ap50.tolist()
    
    # 按 AP50 排序
    results = sorted(zip(class_names, ap50_list), key=lambda x: x[1], reverse=True)
    
    class_results = {
        'class_count': len(class_names),
        'strong_classes': [],  # AP50 >= 0.5
        'medium_classes': [],  # 0.3 <= AP50 < 0.5
        'weak_classes': [],    # AP50 < 0.3
        'all_classes': {},
    }
    
    for name, ap in results:
        class_results['all_classes'][name] = round(ap, 4)
        if ap >= 0.5:
            class_results['strong_classes'].append(name)
        elif ap >= 0.3:
            class_results['medium_classes'].append(name)
        else:
            class_results['weak_classes'].append(name)
    
    print(f"\n总类别数: {class_results['class_count']}")
    print(f"强类别 (AP50 >= 0.5): {len(class_results['strong_classes'])}")
    print(f"中等类别 (0.3-0.5): {len(class_results['medium_classes'])}")
    print(f"弱类别 (AP50 < 0.3): {len(class_results['weak_classes'])}")
    
    print("\n表现最好的类别 (Top 10):")
    for name, ap in results[:10]:
        print(f"  {name:<30} AP50: {ap:.4f}")
    
    print("\n需要改进的类别 (Bottom 10):")
    for name, ap in results[-10:]:
        status = "🔴" if ap < 0.2 else "🟡"
        print(f"  {status} {name:<30} AP50: {ap:.4f}")
    
    return class_results


def speed_test(model, test_image_dir, num_tests=100):
    """推理速度测试"""
    print("\n" + "=" * 60)
    print("5. 推理速度测试")
    print("=" * 60)
    
    import torch
    
    # 获取测试图片
    test_images = list(Path(test_image_dir).glob("*.jpg"))[:num_tests]
    
    if not test_images:
        print(f"⚠️ 未找到测试图片: {test_image_dir}")
        return None
    
    # 预热
    print("预热模型...")
    for _ in range(5):
        _ = model.predict(str(test_images[0]), verbose=False)
    
    # 测试
    print(f"测试推理速度 ({len(test_images)} 张图片)...")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    
    for img_path in test_images:
        results = model.predict(str(img_path), verbose=False)
        if results and hasattr(results[0], 'speed'):
            speed = results[0].speed
            preprocess_times.append(speed.get('preprocess', 0))
            inference_times.append(speed.get('inference', 0))
            postprocess_times.append(speed.get('postprocess', 0))
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    
    speed_results = {
        'total_images': len(test_images),
        'total_time': round(total_time, 2),
        'fps': round(len(test_images) / total_time, 2),
        'avg_latency_ms': round(total_time / len(test_images) * 1000, 2),
        'preprocess_ms': round(sum(preprocess_times) / len(preprocess_times), 2) if preprocess_times else 0,
        'inference_ms': round(sum(inference_times) / len(inference_times), 2) if inference_times else 0,
        'postprocess_ms': round(sum(postprocess_times) / len(postprocess_times), 2) if postprocess_times else 0,
    }
    
    print(f"\n{'指标':<20} {'值':<15} {'标准':<10} {'状态'}")
    print("-" * 55)
    print(f"{'测试图片数':<20} {speed_results['total_images']}")
    print(f"{'总耗时':<20} {speed_results['total_time']:.2f}s")
    print(f"{'FPS':<20} {speed_results['fps']:<15.2f} {'> 30':<10} {'✅' if speed_results['fps'] > 30 else '⚠️'}")
    print(f"{'平均延迟':<20} {speed_results['avg_latency_ms']:.2f}ms{'':<8} {'< 50ms':<10} {'✅' if speed_results['avg_latency_ms'] < 50 else '⚠️'}")
    print(f"{'预处理':<20} {speed_results['preprocess_ms']:.2f}ms")
    print(f"{'推理':<20} {speed_results['inference_ms']:.2f}ms")
    print(f"{'后处理':<20} {speed_results['postprocess_ms']:.2f}ms")
    
    return speed_results


def real_world_test(model, test_dir, conf_threshold=0.25):
    """实际场景测试"""
    print("\n" + "=" * 60)
    print("6. 实际场景测试")
    print("=" * 60)
    
    test_images = list(Path(test_dir).glob("*.jpg")) + list(Path(test_dir).glob("*.png"))
    
    if not test_images:
        print(f"⚠️ 未找到测试图片: {test_dir}")
        return None
    
    # 限制测试图片数量以避免内存问题
    max_test_images = min(500, len(test_images))
    test_images = test_images[:max_test_images]
    
    print(f"测试图片数: {len(test_images)} (最多500张)")
    print(f"置信度阈值: {conf_threshold}")
    
    # 使用流式处理避免内存溢出，逐张处理
    results = []
    for i, img_path in enumerate(test_images):
        if (i + 1) % 100 == 0:
            print(f"  处理进度: {i+1}/{len(test_images)}")
        result = model.predict(str(img_path), conf=conf_threshold, verbose=False)
        if result:
            results.extend(result)
    
    # 统计
    total_images = len(results)
    images_with_detections = 0
    class_counts = {}
    confidence_list = []
    detection_count = 0
    
    for r in results:
        if len(r.boxes) > 0:
            images_with_detections += 1
            for box in r.boxes:
                cls_name = r.names[int(box.cls)]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                confidence_list.append(box.conf.item())
                detection_count += 1
    
    real_world_results = {
        'total_images': total_images,
        'images_with_detections': images_with_detections,
        'detection_rate': round(images_with_detections / total_images, 4) if total_images > 0 else 0,
        'total_detections': detection_count,
        'avg_confidence': round(sum(confidence_list) / len(confidence_list), 4) if confidence_list else 0,
        'min_confidence': round(min(confidence_list), 4) if confidence_list else 0,
        'max_confidence': round(max(confidence_list), 4) if confidence_list else 0,
        'avg_detections_per_image': round(detection_count / total_images, 2) if total_images > 0 else 0,
        'class_distribution': dict(sorted(class_counts.items(), key=lambda x: -x[1])),
    }
    
    print(f"\n{'指标':<25} {'值'}")
    print("-" * 40)
    print(f"{'有检测结果的图片':<25} {images_with_detections}/{total_images} ({real_world_results['detection_rate']*100:.1f}%)")
    print(f"{'总检测数':<25} {detection_count}")
    print(f"{'平均每张图片检测数':<25} {real_world_results['avg_detections_per_image']}")
    print(f"{'平均置信度':<25} {real_world_results['avg_confidence']:.4f}")
    print(f"{'置信度范围':<25} {real_world_results['min_confidence']:.4f} - {real_world_results['max_confidence']:.4f}")
    
    print("\n类别检测分布 (Top 15):")
    for i, (cls, count) in enumerate(list(real_world_results['class_distribution'].items())[:15]):
        print(f"  {i+1:2}. {cls:<25} {count}")
    
    return real_world_results


def generate_report(env_info, model_info, eval_results, class_results, speed_results, real_world_results, output_path):
    """生成评估报告"""
    print("\n" + "=" * 60)
    print("7. 生成评估报告")
    print("=" * 60)
    
    report = {
        'report_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'environment': env_info,
        'model': model_info,
        'quantitative_evaluation': eval_results,
        'class_analysis': class_results,
        'speed_test': speed_results,
        'real_world_test': real_world_results,
    }
    
    # 总体评估
    overall_score = 0
    max_score = 0
    
    if eval_results:
        # mAP50 权重 30%
        max_score += 30
        overall_score += min(30, eval_results['mAP50'] / 0.6 * 30)
        
        # F1 权重 20%
        max_score += 20
        overall_score += min(20, eval_results['f1_score'] / 0.65 * 20)
    
    if speed_results:
        # FPS 权重 25%
        max_score += 25
        overall_score += min(25, speed_results['fps'] / 30 * 25)
    
    if real_world_results:
        # 检测率 权重 25%
        max_score += 25
        overall_score += min(25, real_world_results['detection_rate'] / 0.8 * 25)
    
    report['overall_score'] = round(overall_score / max_score * 100, 1) if max_score > 0 else 0
    
    # 评级
    if report['overall_score'] >= 90:
        report['grade'] = 'A'
        report['recommendation'] = '模型表现优秀，可以投入生产使用'
    elif report['overall_score'] >= 75:
        report['grade'] = 'B'
        report['recommendation'] = '模型表现良好，建议进一步优化弱类别'
    elif report['overall_score'] >= 60:
        report['grade'] = 'C'
        report['recommendation'] = '模型表现一般，需要增加训练轮次或数据量'
    else:
        report['grade'] = 'D'
        report['recommendation'] = '模型需要重新训练，建议检查数据质量和训练配置'
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"报告已保存: {output_path}")
    
    print("\n" + "=" * 60)
    print("评估总结")
    print("=" * 60)
    print(f"综合评分: {report['overall_score']}分")
    print(f"评级: {report['grade']}")
    print(f"建议: {report['recommendation']}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 模型评估脚本')
    parser.add_argument('--model', type=str, default='data/models/trained/best_40cls.pt',
                        help='模型路径')
    parser.add_argument('--data', type=str, default='data/datasets/data_40cls.yaml',
                        help='数据集配置文件')
    parser.add_argument('--test-dir', type=str, default='data/datasets/images/val',
                        help='测试图片目录')
    parser.add_argument('--output', type=str, default='evaluation_report.json',
                        help='输出报告路径')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    
    args = parser.parse_args()
    
    # 切换到项目根目录
    os.chdir(PROJECT_ROOT)
    
    print("\n" + "=" * 60)
    print("YOLOv8 模型评估")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"数据集: {args.data}")
    print(f"测试目录: {args.test_dir}")
    
    # 1. 环境检查
    env_info = check_environment()
    if env_info is None:
        return
    
    # 2. 加载模型
    result = load_model(args.model)
    if result is None:
        return
    model, model_info = result
    
    # 3. 定量评估
    eval_results, metrics = quantitative_evaluation(model, args.data)
    
    # 4. 类别分析
    class_results = class_analysis(metrics)
    
    # 5. 速度测试
    speed_results = speed_test(model, args.test_dir)
    
    # 6. 实际场景测试
    real_world_results = real_world_test(model, args.test_dir, args.conf)
    
    # 7. 生成报告
    report = generate_report(
        env_info, model_info, eval_results, class_results,
        speed_results, real_world_results, args.output
    )
    
    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
