@echo off
setlocal EnableExtensions
set "YOLO_PYTHON=C:\Users\ROG\anaconda3\envs\yolo26\python.exe"

if not exist "%YOLO_PYTHON%" goto python_missing

"%YOLO_PYTHON%" -c "import sys,torch; print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE'); sys.exit(0 if torch.cuda.is_available() else 2)"
if errorlevel 1 goto gpu_missing

pushd "%~dp0"
if errorlevel 1 goto folder_error
"%YOLO_PYTHON%" main.py
set "YOLO_EXIT_CODE=%ERRORLEVEL%"
popd
if not "%YOLO_EXIT_CODE%"=="0" goto app_error
exit /b 0

:python_missing
echo ERROR: yolo26 Python was not found:
echo %YOLO_PYTHON%
pause
exit /b 1

:gpu_missing
echo ERROR: CUDA GPU check failed. CPU fallback is disabled.
pause
exit /b 2

:folder_error
echo ERROR: Cannot open the application folder.
pause
exit /b 3

:app_error
echo ERROR: Application exited with code %YOLO_EXIT_CODE%.
echo Check startup_error.log in the application folder.
pause
exit /b %YOLO_EXIT_CODE%
