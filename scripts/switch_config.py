# switch_config.py
# -*- coding: utf-8 -*-
"""
配置切换脚本 - 在5类、23类和40类配置之间快速切换
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings


def update_settings_file(mode: str):
    """更新 settings.py 中的默认模式"""
    settings_path = PROJECT_ROOT / 'src' / 'config' / 'settings.py'
    
    with open(settings_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换 CURRENT_MODE
    import re
    content = re.sub(
        r"CURRENT_MODE = '[^']+'",
        f"CURRENT_MODE = '{mode}'",
        content
    )
    
    with open(settings_path, 'w', encoding='utf-8') as f:
        f.write(content)


def switch_to_4cls():
    """切换到4类配置"""
    update_settings_file('cls4')
    print("✅ 已切换到 4类配置")
    print("   类别: 厨余垃圾, 可回收物, 有害垃圾, 其他垃圾")
    print("   模型: data/models/trained/best_4cls.pt")


def switch_to_5cls():
    """切换到5类配置"""
    update_settings_file('cls5')
    print("✅ 已切换到 5类配置")
    print("   类别: 果皮, 茶叶渣, 易拉罐, 过期药品, 其他垃圾")
    print("   模型: data/models/trained/best_5cls.pt")


def switch_to_23cls():
    """切换到23类配置"""
    update_settings_file('cls23')
    print("✅ 已切换到 23类配置")
    print("   类别: 蔬菜, 果皮, 果核, 骨头, 鱼骨, 蛋壳, 米饭, 面条, 面包, 肉类, 鱼类, 剩饭剩菜,")
    print("         塑料瓶, 塑料袋, 塑料容器, 玻璃瓶, 金属罐, 纸盒, 纸张, 铝箔, 电池, 烟头, 其他垃圾")
    print("   模型: data/models/trained/best_23cls.pt")
    print("\n⚠️  请确保已训练23类模型并放置在 data/models/trained/best_23cls.pt")


def switch_to_40cls():
    """切换到40类配置"""
    update_settings_file('cls40')
    print("✅ 已切换到 40类配置")
    print("   类别: 40种精细化垃圾分类")
    print("   模型: data/models/trained/best_40cls.pt")
    print("\n⚠️  请确保已训练40类模型并放置在 data/models/trained/best_40cls.pt")


def show_current():
    """显示当前配置"""
    mode = settings.CURRENT_MODE
    config = settings.get_current_config()
    
    print(f"📌 当前配置模式: {mode}")
    print(f"   类别数量: {config['NUM_CLASSES']}")
    print(f"   模型路径: {config['model_path']}")
    
    # 检查模型是否存在
    if Path(config['model_path']).exists():
        print("   模型状态: ✅ 已存在")
    else:
        print("   模型状态: ❌ 未找到，请先训练模型")


def main():
    print("=" * 50)
    print("垃圾分类配置切换工具")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        show_current()
        print("\n使用方法:")
        print("  python scripts/switch_config.py 4    # 切换到4类配置")
        print("  python scripts/switch_config.py 5    # 切换到5类配置")
        print("  python scripts/switch_config.py 23   # 切换到23类配置")
        print("  python scripts/switch_config.py 40   # 切换到40类配置")
        print("  python scripts/switch_config.py show # 显示当前配置")
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == '4' or cmd == 'cls4':
        switch_to_4cls()
    elif cmd == '5' or cmd == 'cls5':
        switch_to_5cls()
    elif cmd == '23' or cmd == 'cls23':
        switch_to_23cls()
    elif cmd == '40' or cmd == 'cls40':
        switch_to_40cls()
    elif cmd == 'show':
        show_current()
    else:
        print(f"❌ 未知命令: {cmd}")
        print("可用命令: 4, 5, 23, 40, show")


if __name__ == '__main__':
    main()
