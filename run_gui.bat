@echo off
echo ======================================
echo RCS小波神经网络预测系统 GUI启动器
echo ======================================
echo.

echo 正在检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请确保Python已正确安装并添加到PATH
    pause
    exit /b 1
)

echo 正在启动GUI界面 (已修复tkinter问题)...
python start_gui.py

if errorlevel 1 (
    echo.
    echo GUI启动失败，尝试备用方案...
    echo 正在启动命令行功能测试...
    python quick_test.py
    echo.
    echo 其他可用命令:
    echo   python main.py --mode train    # 训练模型
    echo   python main.py --mode predict  # 预测RCS
    pause
)