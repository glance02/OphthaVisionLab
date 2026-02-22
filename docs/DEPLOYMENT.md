# VisionFM Web 部署文档

## 项目概述

VisionFM 眼科图像分割系统已成功部署到本地环境，包括：
- **后端**: FastAPI + PyTorch，提供图像分割 API
- **前端**: Vue 3 + Element Plus，提供用户界面

## 当前状态

### ✅ 已完成

1. **后端环境**
   - Python 虚拟环境: `backend/venv/`
   - 依赖已安装: FastAPI, Uvicorn, PyTorch, OpenCV, 等
   - API 服务正常运行: http://localhost:8000
   - API 文档: http://localhost:8000/docs

2. **前端环境**
   - Vue 3 项目初始化完成
   - Element Plus 组件库已安装
   - 开发服务器正常运行: http://localhost:5173

3. **核心功能**
   - 图片上传组件（支持拖拽）
   - 分割结果展示（原图 + 掩码 + 叠加）
   - 透明度调节
   - 结果下载功能

### ⚠️ 待完成

1. **模型权重文件** (关键)
   - `backend/pretrain_weights/VFM_Fundus_weights.pth` - 预训练编码器权重
   - `backend/checkpoints/checkpoint_108_linear.pth` - 分割头权重

   > 这些文件是模型运行所必需的，需要从原项目或其他位置复制到这些目录。

## 目录结构

```
visonFM_/
├── backend/                 # 后端服务
│   ├── venv/               # Python 虚拟环境
│   ├── models/             # 模型定义
│   │   ├── __init__.py
│   │   ├── vision_transformer.py
│   │   └── unetr_head.py
│   ├── pretrain_weights/   # ⚠️ 需要添加权重文件
│   ├── checkpoints/        # ⚠️ 需要添加权重文件
│   ├── main.py             # FastAPI 主入口
│   ├── model_service.py    # 模型服务
│   ├── utils.py            # 工具函数
│   └── requirements.txt    # Python 依赖
│
├── frontend/               # 前端应用
│   ├── src/
│   │   ├── components/
│   │   │   ├── ImageUpload.vue
│   │   │   └── ResultDisplay.vue
│   │   ├── App.vue
│   │   └── main.ts
│   ├── .env                # API 配置
│   └── package.json        # Node 依赖
│
├── start.bat               # Windows 启动脚本
├── start.sh                # Linux/Mac 启动脚本
└── DEPLOYMENT.md           # 本文档
```

## 启动服务

### 方式 1: 使用启动脚本（推荐）

**Windows:**
```cmd
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

### 方式 2: 手动启动

**启动后端:**
```bash
cd backend
venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/Mac
python main.py
```

**启动前端:**
```bash
cd frontend
npm run dev
```

## 访问地址

| 服务 | 地址 |
|------|------|
| 前端界面 | http://localhost:5173 |
| 后端 API | http://localhost:8000 |
| API 文档 | http://localhost:8000/docs |
| 健康检查 | http://localhost:8000/health |

## 添加模型权重文件

### 步骤 1: 获取权重文件

从原项目或其他位置获取以下文件：
- `VFM_Fundus_weights.pth` (预训练编码器)
- `checkpoint_108_linear.pth` (分割头权重)

### 步骤 2: 放置到正确位置

```bash
# 复制预训练权重
cp VFM_Fundus_weights.pth backend/pretrain_weights/

# 复制分割头权重
cp checkpoint_108_linear.pth backend/checkpoints/
```

### 步骤 3: 重启后端服务

停止并重新启动后端服务，模型会自动加载。

## 使用说明

1. 打开浏览器访问 http://localhost:5173
2. 点击或拖拽上传眼底图像（JPG/PNG 格式）
3. 等待处理完成
4. 查看分割结果：
   - 原图
   - 分割结果（原图 + 掩码叠加）
   - 纯掩码
5. 调整透明度滑块查看效果
6. 下载结果图片

## API 端点

### POST /api/segment

图像分割接口

**请求:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (图像文件)

**响应:**
```json
{
  "success": true,
  "message": "分割成功",
  "data": {
    "originalImage": "data:image/png;base64,...",
    "maskImage": "data:image/png;base64,...",
    "threshold": 0.5,
    "shape": "512x512"
  }
}
```

### GET /health

健康检查接口

**响应:**
```json
{
  "status": "healthy",
  "device": "cpu",
  "cuda_available": false,
  "gpu_count": 0
}
```

## 技术栈

### 后端
- FastAPI 0.129.0
- PyTorch 2.10.0
- OpenCV 4.13.0
- Uvicorn 0.41.0

### 前端
- Vue 3.5.13
- Element Plus
- Vite 7.3.1
- Axios
- TypeScript

## 故障排查

### 后端无法启动

1. 检查虚拟环境是否存在
   ```bash
   cd backend
   python -m venv venv
   ```

2. 重新安装依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 检查端口 8000 是否被占用

### 前端无法启动

1. 检查 Node.js 版本（需要 18+）
   ```bash
   node --version
   ```

2. 重新安装依赖
   ```bash
   cd frontend
   rm -rf node_modules
   npm install
   ```

3. 检查端口 5173 是否被占用

### 模型加载失败

1. 确认权重文件存在
   ```bash
   ls backend/pretrain_weights/
   ls backend/checkpoints/
   ```

2. 检查文件大小是否正确
3. 查看后端日志获取详细错误信息

## 下一步

1. **添加模型权重文件** - 使系统完全可用
2. **测试真实图像** - 使用眼底图像验证分割效果
3. **性能优化** - 根据需要优化推理速度
4. **UI 美化** - 根据需求调整界面样式
5. **生产部署** - 考虑 Docker 化部署

## 联系方式

如有问题，请查阅项目文档或联系开发团队。
