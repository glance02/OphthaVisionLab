# VisionFM 分割模型后端

基于 FastAPI 的眼科图像分割后端服务，支持眼底血管分割。

## 目录结构

```
backend/
├── main.py                 # FastAPI 主入口
├── model_service.py        # 模型服务封装
├── utils.py               # 工具函数
├── models/                # 模型定义
│   ├── __init__.py
│   ├── vision_transformer.py
│   └── unetr_head.py
├── pretrain_weights/     # 预训练权重（需手动添加）
│   └── VFM_Fundus_weights.pth
├── checkpoints/           # 训练权重（需手动添加）
│   ├── seg/
│   │   └── checkpoint_108_linear.pth   # 分割模型
│   └── single_cls/
│       └── checkpoint_teacher_linear.pth  # 二分类模型
└── requirements.txt        # Python 依赖
```

## 安装步骤

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 准备模型权重文件

需要从原项目复制以下文件：

**预训练权重**（必需）:
```bash
# 从原项目根目录复制
cp ../pretrain_weights/VFM_Fundus_weights.pth pretrain_weights/
```

**分割模型权重**（必需）:
```bash
# 从原项目根目录复制
cp ../results/single_seg_debug/checkpoint_108_linear.pth checkpoints/seg/
```

**二分类模型权重**（可选）:
```bash
# 从原项目根目录复制
cp ../myProcessResults/single_cls_260106_Binary_finetune/checkpoint_teacher_linear.pth checkpoints/single_cls/
```

如果文件不存在，启动时会显示警告但不报错。

### 3. 启动服务

```bash
python main.py
```

服务将在 `http://localhost:8000` 启动

## API 接口

### 1. 健康检查

```http
GET /health
```

返回 GPU/CPU 状态信息

### 2. 图像分割

```http
POST /api/segment
Content-Type: multipart/form-data
```

参数:
- `file`: 图像文件（JPG/PNG）
- `threshold`: 分割阈值，默认 0.5（可选）

响应:
```json
{
  "success": true,
  "message": "分割成功",
  "data": {
    "originalImage": "data:image/jpeg;base64,...",
    "maskImage": "data:image/png;base64,...",
    "threshold": 0.5,
    "shape": "512x512"
  }
}
```

### 3. 模型信息

```http
GET /model/info
```

返回模型架构和参数信息

## API 文档

启动后访问: http://localhost:8000/docs

## 测试

使用 curl 测试:

```bash
# 健康检查
curl http://localhost:8000/health

# 图像分割
curl -X POST "http://localhost:8000/api/segment" \
  -F "file=@test_image.jpg" \
  -F "threshold=0.5"
```

## 注意事项

1. **首次启动会较慢**: 模型加载需要 10-30 秒
2. **GPU 可选**: 没有会自动使用 CPU
3. **内存占用**: 模型加载后约占用 2-3GB 显存/内存
4. **图片格式**: 仅支持 JPG/PNG，最大 10MB

## 常见问题

**Q: 启动时提示找不到权重文件？**

A: 将以下文件复制到对应目录：
- `VFM_Fundus_weights.pth` → `backend/pretrain_weights/`
- `checkpoint_108_linear.pth` → `backend/checkpoints/seg/`
- `checkpoint_teacher_linear.pth` → `backend/checkpoints/single_cls/`

**Q: 推理速度很慢？**

A: 检查是否使用了 GPU：
```bash
curl http://localhost:8000/health
```

返回 `cuda_available: true` 表示使用 GPU。

**Q: 前端 CORS 错误？**

A: 已配置 CORS 允许所有来源。生产环境建议修改 `main.py` 中的 `allow_origins` 为具体前端域名。
