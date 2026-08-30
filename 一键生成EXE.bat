@echo off
chcp 65001 >nul
pushd "%~dp0"
python build_exe.py
set "BUILD_EXIT_CODE=%ERRORLEVEL%"
popd
if not "%BUILD_EXIT_CODE%"=="0" (
    echo.
    echo EXE build failed. Check the error above.
    pause
    exit /b %BUILD_EXIT_CODE%
)
echo.
echo EXE build completed.
pause
