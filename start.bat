@echo off
echo ========================================
echo   VisionFM 眼科图像分割系统
echo   启动脚本
echo ========================================
echo.

REM 检查虚拟环境是否存在
if not exist "backend\venv\Scripts\activate.bat" (
    echo [错误] 后端虚拟环境未找到
    echo 请先运行: cd backend && python -m venv venv
    pause
    exit /b 1
)

REM 检查前端依赖是否安装
if not exist "frontend\node_modules\" (
    echo [错误] 前端依赖未安装
    echo 请先运行: cd frontend && npm install
    pause
    exit /b 1
)

echo [1/3] 启动后端服务...
start "VisionFM Backend" cmd /k "cd backend && venv\Scripts\activate && python main.py"

REM 等待后端启动
echo 等待后端服务启动...
timeout /t 5 /nobreak > nul

echo [2/3] 启动前端服务...
start "VisionFM Frontend" cmd /k "cd frontend && npm run dev"

echo [3/3] 服务启动完成！
echo.
echo ========================================
echo   后端 API: http://localhost:8000
echo   API 文档: http://localhost:8000/docs
echo   前端界面: http://localhost:5173
echo ========================================
echo.
echo 按任意键关闭此窗口（服务将继续运行）...
pause > nul
