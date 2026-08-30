@echo off
chcp 65001 >nul
setlocal EnableExtensions

pushd "%~dp0"
if errorlevel 1 goto folder_error
python -m yolo_world_annotator --device auto
set "YOLO_EXIT_CODE=%ERRORLEVEL%"
popd
if not "%YOLO_EXIT_CODE%"=="0" goto app_error
exit /b 0

:folder_error
echo ERROR: Cannot open the application folder.
pause
exit /b 2

:app_error
echo ERROR: Application exited with code %YOLO_EXIT_CODE%.
echo Activate the virtual environment and run: pip install -e .
pause
exit /b %YOLO_EXIT_CODE%
