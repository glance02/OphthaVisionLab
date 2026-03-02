#!/bin/bash

echo "========================================"
echo "  VisionFM 眼科图像分割系统"
echo "  启动脚本"
echo "========================================"
echo ""

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 检查虚拟环境/Conda环境是否存在
# 优先级: backend/venv > conda环境 > 系统Python
if [ -d "$SCRIPT_DIR/backend/venv" ]; then
    PYTHON_CMD="$SCRIPT_DIR/backend/venv/bin/python"
    echo "[信息] 使用虚拟环境: backend/venv"
elif conda env list | grep -q "^vfm "; then
    PYTHON_CMD="/home/seborid/miniconda3/envs/vfm/bin/python"
    echo "[信息] 使用 Conda 环境: vfm"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "[错误] 未找到 Python 解释器"
    exit 1
fi

# 检查前端依赖是否安装
if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
    echo "[错误] 前端依赖未安装"
    echo "请先运行: cd frontend && npm install"
    exit 1
fi

echo "[1/3] 启动后端服务..."
cd "$SCRIPT_DIR/backend"
$PYTHON_CMD main.py &
BACKEND_PID=$!

# 等待后端启动
echo "等待后端服务启动..."
sleep 5

# 检查后端是否成功启动 (检查端口是否可访问)
MAX_RETRIES=10
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/docs &> /dev/null; then
        echo "[成功] 后端服务已启动"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "等待后端响应... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "[错误] 后端服务启动失败或无响应"
    kill $BACKEND_PID 2>/dev/null
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
