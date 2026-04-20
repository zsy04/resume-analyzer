@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   智能简历分析系统 - 启动中...
echo ========================================
echo.

cd /d "%~dp0"

if not exist "venv" (
    echo [INFO] 未检测到虚拟环境，正在使用系统Python...
    python -m streamlit run app.py --server.port 8502
) else (
    echo [INFO] 检测到虚拟环境，正在激活...
    call venv\Scripts\activate.bat
    python -m streamlit run app.py --server.port 8502
)

pause
