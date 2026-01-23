"""
配置管理工具 - 查看当前配置及维护 settings.py
"""
import sys
import os
import re
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings

def update_settings_mode(mode: str):
    """更新 settings.py 中的 CURRENT_MODE"""
    settings_path = PROJECT_ROOT / 'src' / 'config' / 'settings.py'
    
    if not settings_path.exists():
        print(f"❌ 错误: 未找到配置文件 {settings_path}")
        return

    with open(settings_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查模式是否存在于注册表
    if mode not in settings.CONFIG_REGISTRY:
        print(f"⚠️  警告: 模式 '{mode}' 不在 settings.CONFIG_REGISTRY 中。")
        print(f"目前支持的模式: {list(settings.CONFIG_REGISTRY.keys())}")
        return

    # 替换 CURRENT_MODE = 'xxx'
    new_content = re.sub(
        r"CURRENT_MODE = ['\"][^'\"]+['\"]",
        f"CURRENT_MODE = '{mode}'",
        content
    )
    
    if new_content == content:
        print(f"ℹ️  配置已是 {mode}，无需修改。")
    else:
        with open(settings_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✅ 已将 CURRENT_MODE 更新为: {mode}")

def show_status():
    """显示当前系统配置状态"""
    print("\n" + "="*40)
    print("      垃圾分类系统配置状态")
    print("="*40)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"当前模式:   {settings.CURRENT_MODE}")
    
    config = settings.get_current_config()
    print(f"类别数量:   {config['NUM_CLASSES']}")
    print(f"中文类别:   {' / '.join(config['CH_names'])}")
    print(f"模型路径:   {config['model_path']}")
    
    # 检查模型物理存在
    model_path = Path(config['model_path'])
    if model_path.exists():
        print(f"模型状态:   ✅ 已就绪 ({model_path.name})")
    else:
        print(f"模型状态:   ❌ 缺失 (请检查 data/models/trained/)")
    
    print("-" * 40)
    print(f"可用模式:   {', '.join(settings.CONFIG_REGISTRY.keys())}")
    print("="*40 + "\n")

def main():
    if len(sys.argv) < 2:
        show_status()
        print("用法:")
        print("  python scripts/switch_config.py show          # 查看当前状态")
        print("  python scripts/switch_config.py <mode_id>    # 切换模式 (如 cls4)")
        return

    cmd = sys.argv[1].lower()
    
    if cmd == 'show':
        show_status()
    elif cmd in ['4', 'cls4']:
        update_settings_mode('cls4')
    else:
        # 尝试直接作为 mode_id 切换
        update_settings_mode(cmd)

if __name__ == '__main__':
    main()
