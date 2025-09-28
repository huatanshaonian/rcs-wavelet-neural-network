"""
Unicode字符支持修复工具 ✨
解决Windows GBK编码限制，让可爱的Unicode字符正常显示！
"""

import sys
import os

def fix_unicode_output():
    """修复Unicode输出，支持可爱的表情符号 🎉"""

    if sys.platform.startswith('win'):
        try:
            import codecs
            import io

            # 方法1: 重新配置stdout和stderr为UTF-8
            if hasattr(sys.stdout, 'buffer'):
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer,
                    encoding='utf-8',
                    errors='replace'
                )
            if hasattr(sys.stderr, 'buffer'):
                sys.stderr = io.TextIOWrapper(
                    sys.stderr.buffer,
                    encoding='utf-8',
                    errors='replace'
                )

            print("✨ Unicode输出修复成功!")
            return True

        except Exception as e:
            print(f"Unicode修复失败: {e}")
            return False
    else:
        print("✅ 非Windows系统，Unicode支持正常")
        return True

def test_unicode_output():
    """测试Unicode字符输出效果"""

    test_chars = [
        "✅ 成功标记",
        "❌ 错误标记",
        "⚠️ 警告标记",
        "🎉 庆祝表情",
        "🚀 火箭表情",
        "📊 图表表情",
        "🎯 目标表情",
        "💾 保存表情",
        "📂 文件夹表情",
        "🔄 循环表情",
        "📋 剪贴板表情",
        "⚡ 闪电表情"
    ]

    print("\n=== Unicode字符测试 ===")
    for char in test_chars:
        print(char)
    print("========================")

if __name__ == "__main__":
    # 修复Unicode输出
    success = fix_unicode_output()

    if success:
        # 测试Unicode字符
        test_unicode_output()

        print("\n🎊 现在可以使用可爱的Unicode字符了!")
        print("在其他脚本开头导入这个模块即可:")
        print("from unicode_fix import fix_unicode_output")
        print("fix_unicode_output()")
    else:
        print("\n无法修复Unicode输出，可能需要设置环境变量")
        print("建议在cmd中运行: set PYTHONIOENCODING=utf-8")