#!/bin/bash

echo "========================================"
echo "  VisionFM 眼科图像分割系统"
echo "  启动脚本"
echo "========================================"
echo ""

# 检查虚拟环境是否存在
if [ ! -d "backend/venv" ]; then
    echo "[错误] 后端虚拟环境未找到"
    echo "请先运行: cd backend && python -m venv venv"
    exit 1
fi

# 检查前端依赖是否安装
if [ ! -d "frontend/node_modules" ]; then
    echo "[错误] 前端依赖未安装"
    echo "请先运行: cd frontend && npm install"
    exit 1
fi

echo "[1/3] 启动后端服务..."
cd backend && source venv/bin/activate && python main.py &
BACKEND_PID=$!

# 等待后端启动
echo "等待后端服务启动..."
sleep 5

cd ..

echo "[2/3] 启动前端服务..."
cd frontend && npm run dev &
FRONTEND_PID=$!

cd ..

echo "[3/3] 服务启动完成！"
echo ""
echo "========================================"
echo "  后端 API: http://localhost:8000"
echo "  API 文档: http://localhost:8000/docs"
echo "  前端界面: http://localhost:5173"
echo "========================================"
echo ""
echo "按 Ctrl+C 停止所有服务"

# 捕获退出信号
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT TERM

# 等待进程
wait
