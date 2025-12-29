@echo off
echo Clearing all Python cache files...
del /s /q *.pyc 2>nul
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
echo Cache cleared!
echo.
echo Please restart the GUI:
echo   python main.py --mode gui
pause
