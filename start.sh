#!/bin/bash

echo "========================================"
echo "  VisionFM 眼科图像分割系统"
echo "  启动脚本"
echo "========================================"
echo ""

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 检查后端依赖
if ! python -c "import fastapi" 2>/dev/null; then
    echo "[警告] 未检测到 fastapi，尝试安装后端依赖..."
    pip install fastapi uvicorn python-multipart -q
fi

# 检查前端依赖是否安装
if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
    echo "[错误] 前端依赖未安装"
    echo "请先运行: cd frontend && npm install"
    exit 1
fi

echo "[1/3] 启动后端服务..."
cd "$SCRIPT_DIR/backend"
python main.py &
BACKEND_PID=$!

# 等待后端启动
echo "等待后端服务启动..."
sleep 5

# 检查后端是否成功启动
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "[错误] 后端服务启动失败"
    exit 1
fi

echo "[2/3] 启动前端服务..."
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

# 等待前端启动
sleep 3

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
trap "echo ''; echo '正在停止服务...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

# 等待进程
wait
