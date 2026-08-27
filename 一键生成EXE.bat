@echo off
chcp 65001 >nul
set "BUILD_PYTHON=C:\Users\ROG\anaconda3\envs\yolo26\python.exe"
"%BUILD_PYTHON%" "%~dp0build_exe.py"
if errorlevel 1 (
    echo.
    echo EXE build failed. Check the error above.
    pause
    exit /b 1
)
echo.
echo EXE build completed.
pause
