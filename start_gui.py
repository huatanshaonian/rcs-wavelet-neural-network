"""
RCS小波神经网络GUI启动器

修复了tkinter环境问题的启动脚本
"""

import os
import sys

def setup_tkinter_environment():
    """设置tkinter环境"""
    print("正在配置tkinter环境...")

    # 清理可能冲突的环境变量
    env_vars_to_clear = ['TCL_LIBRARY', 'TK_LIBRARY', 'TCLLIBPATH']
    for var in env_vars_to_clear:
        if var in os.environ:
            del os.environ[var]
            print(f"清理环境变量: {var}")

    # 设置正确的Tcl/Tk路径
    anaconda_base = r"G:\anaconda\envs\RCS_OP1"
    tcl_path = os.path.join(anaconda_base, "tcl", "tcl8.6")
    tk_path = os.path.join(anaconda_base, "tcl", "tk8.6")

    if os.path.exists(tcl_path):
        os.environ['TCL_LIBRARY'] = tcl_path
        print(f"设置TCL_LIBRARY: {tcl_path}")

    if os.path.exists(tk_path):
        os.environ['TK_LIBRARY'] = tk_path
        print(f"设置TK_LIBRARY: {tk_path}")

def start_rcs_gui():
    """启动RCS GUI"""
    try:
        # 设置环境
        setup_tkinter_environment()

        # 导入GUI模块
        from gui import RCSWaveletGUI
        import tkinter as tk

        print("正在启动RCS小波神经网络GUI...")

        # 创建主窗口
        root = tk.Tk()
        app = RCSWaveletGUI(root)

        print("GUI界面已启动！")
        root.mainloop()

    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有依赖已安装: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"GUI启动失败: {e}")
        print("\n可以尝试以下解决方案:")
        print("1. 使用命令行模式: python main.py --mode train")
        print("2. 运行功能测试: python quick_test.py")
        print("3. 检查tkinter安装: python -c 'import tkinter; print(\"tkinter可用\")'")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("RCS小波神经网络预测系统 GUI启动器")
    print("版本: 1.0")
    print("=" * 60)

    success = start_rcs_gui()

    if success:
        print("GUI已正常退出")
    else:
        print("GUI启动失败，请检查错误信息")
        sys.exit(1)