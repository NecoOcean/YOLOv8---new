# switch_config.py
# -*- coding: utf-8 -*-
"""
配置切换脚本 - 在5类和40类配置之间快速切换
"""
import shutil
import os
import sys

def switch_to_5cls():
    """切换到5类配置"""
    if os.path.exists('Config_kitchen.py'):
        shutil.copy('Config_kitchen.py', 'Config.py')
        print("✅ 已切换到 5类配置 (Config_kitchen.py)")
        print("   类别: 果皮, 茶叶渣, 易拉罐, 过期药品, 其他垃圾")
        print("   模型: models/best.pt")
    else:
        print("❌ 未找到 Config_kitchen.py")

def switch_to_40cls():
    """切换到40类配置"""
    if os.path.exists('Config_40cls.py'):
        shutil.copy('Config_40cls.py', 'Config.py')
        print("✅ 已切换到 40类配置 (Config_40cls.py)")
        print("   类别: 40种精细化垃圾分类")
        print("   模型: models/best_40cls.pt")
        print("\n⚠️  请确保已训练40类模型并放置在 models/best_40cls.pt")
    else:
        print("❌ 未找到 Config_40cls.py")

def show_current():
    """显示当前配置"""
    if os.path.exists('Config.py'):
        with open('Config.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'NUM_CLASSES = 5' in content:
                print("📌 当前配置: 5类 (简化版)")
            elif 'NUM_CLASSES = 40' in content:
                print("📌 当前配置: 40类 (精细化版)")
            else:
                print("📌 当前配置: 未知")
    else:
        print("❌ Config.py 不存在")

def main():
    print("=" * 50)
    print("垃圾分类配置切换工具")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        show_current()
        print("\n使用方法:")
        print("  python switch_config.py 5    # 切换到5类配置")
        print("  python switch_config.py 40   # 切换到40类配置")
        print("  python switch_config.py show # 显示当前配置")
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == '5' or cmd == '5cls':
        switch_to_5cls()
    elif cmd == '40' or cmd == '40cls':
        switch_to_40cls()
    elif cmd == 'show':
        show_current()
    else:
        print(f"❌ 未知命令: {cmd}")
        print("可用命令: 5, 40, show")

if __name__ == '__main__':
    main()
