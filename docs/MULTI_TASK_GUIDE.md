# VisionFM 多任务系统部署文档

## 系统概述

VisionFM 多任务系统现已支持**三种主要任务类型**：

| 任务类型 | 端点 | 应用场景 |
|---------|------|---------|
| **眼底血管分割** | `/api/segment` | 视网膜血管分割、疾病诊断辅助 |
| **二分类** | `/api/classify/binary` | 糖尿病视网膜病变（DR）筛查 |
| **多分类** | `/api/classify/multiclass` | 疾病分级、多类别识别 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                     前端 (Vue 3)                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  任务选择器 (TaskSelector)                        │  │
│  │  - 分割、二分类、多分类                            │  │
│  │  - 高级设置（checkpoint、阈值等）                  │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │  图片上传 (ImageUpload)                           │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌──────────────────┬────────────────┬──────────────┐  │
│  │ 分割结果展示     │ 二分类结果     │ 多分类结果   │  │
│  │ ResultDisplay   │ Classification │ Classification│  │
│  │                 │ Result         │ Result       │  │
│  └──────────────────┴────────────────┴──────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │ HTTP
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   后端 (FastAPI)                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │        推理服务 (InferenceService)                │  │
│  │  - 模型工厂 (ModelFactory)                        │  │
│  │  - 模型缓存管理                                   │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌──────────────────┬────────────────┬──────────────┐  │
│  │ SegModel        │ ClsModel       │ ClsModel     │  │
│  │ 分割            │ 二分类         │ 多分类       │  │
│  └──────────────────┴────────────────┴──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 后端 API 端点

### 1. 分割 API

**端点**: `POST /api/segment`

**参数**:
| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | File | ✅ | - | 图像文件 (JPG/PNG) |
| `checkpoint` | str | ❌ | checkpoints/checkpoint_108_linear.pth | 模型权重路径 |
| `threshold` | float | ❌ | 0.5 | 分割阈值 (0-1) |
| `input_size` | int | ❌ | 512 | 输入图像尺寸 |

**响应**:
```json
{
  "success": true,
  "task": "segmentation",
  "data": {
    "originalImage": "data:image/png;base64,...",
    "maskImage": "data:image/png;base64,...",
    "threshold": 0.5,
    "shape": "512x512"
  }
}
```

---

### 2. 二分类 API

**端点**: `POST /api/classify/binary`

**参数**:
| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | File | ✅ | - | 图像文件 |
| `checkpoint` | str | ✅ | - | **必须提供**训练好的模型路径 |
| `input_size` | int | ❌ | 224 | 输入图像尺寸 |
| `n_last_blocks` | int | ❌ | 4 | 使用的最后层数 |
| `avgpool` | int | ❌ | 0 | 池化模式 (0=CLS, 1=patch avg) |

**响应**:
```json
{
  "success": true,
  "task": "binary_classification",
  "data": {
    "predicted_class": 1,
    "probability": 0.85,
    "confidence": 0.85,
    "logit": 1.75
  }
}
```

---

### 3. 多分类 API

**端点**: `POST /api/classify/multiclass`

**参数**:
| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | File | ✅ | - | 图像文件 |
| `checkpoint` | str | ✅ | - | **必须提供**训练好的模型路径 |
| `num_labels` | int | ✅ | - | 类别数量 |
| `input_size` | int | ❌ | 224 | 输入图像尺寸 |
| `n_last_blocks` | int | ❌ | 4 | 使用的最后层数 |
| `avgpool` | int | ❌ | 0 | 池化模式 |

**响应**:
```json
{
  "success": true,
  "task": "multiclass_classification",
  "data": {
    "predicted_class": 2,
    "confidence": 0.78,
    "probabilities": [0.05, 0.10, 0.78, 0.07],
    "num_classes": 4
  }
}
```

---

## 前端使用指南

### 1. 任务选择

前端顶部有三个任务按钮：
- **分割** - 眼底血管分割（有默认checkpoint）
- **二分类** - 需要提供checkpoint
- **多分类** - 需要提供checkpoint和类别数

### 2. 高级设置

点击"高级设置"可配置：
- Checkpoint 路径
- 阈值（分割任务）
- 类别数量（多分类）
- 输入尺寸等参数

### 3. 上传和预测

1. 选择任务类型
2. 配置参数（如需要）
3. 拖拽或点击上传图像
4. 等待处理完成
5. 查看结果

---

## 模型权重文件

### 预训练权重（必需）

**文件**: `backend/pretrain_weights/VFM_Fundus_weights.pth`

**下载**: [Google Drive](https://drive.google.com/file/d/13uWm0a02dCWyARUcrCdHZIcEgRfBmVA4/view?usp=sharing)

### 分割权重

**文件**: `backend/checkpoints/checkpoint_108_linear.pth`

**获取方式**: 需要训练或从原始项目获取

### 分类权重

**二分类/多分类**: 需要根据具体任务训练

**训练脚本**:
- 二分类: 使用 `evaluation/train_cls_decoder_finetune.py`
- 多分类: 使用 `finetune_visionfm_for_multiclass_classification.py`

---

## 启动服务

### 启动后端

```bash
cd backend
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac
python main.py
```

访问 http://localhost:8000/docs 查看 API 文档

### 启动前端

```bash
cd frontend
npm run dev
```

访问 http://localhost:5173 使用 Web 界面

---

## 项目结构

```
visonFM_/
├── backend/                     # 后端服务
│   ├── main.py                  # FastAPI 主入口（多任务支持）
│   ├── inference_service.py     # 通用推理服务
│   ├── model_factory.py         # 模型工厂
│   ├── model_service.py         # 旧版分割服务（保留）
│   ├── models/                  # 模型定义
│   ├── pretrain_weights/        # 预训练权重
│   └── checkpoints/             # 任务权重
│
├── frontend/                    # 前端应用
│   ├── src/
│   │   ├── components/
│   │   │   ├── TaskSelector.vue        # 任务选择器（新）
│   │   │   ├── ClassificationResult.vue # 分类结果（新）
│   │   │   ├── ImageUpload.vue         # 图片上传
│   │   │   └── ResultDisplay.vue       # 分割结果
│   │   └── App.vue              # 主应用（已更新）
│   └── .env                     # API 配置
│
├── tools/                       # 命令行工具
│   ├── seg_pth_use.py           # 分割推理
│   ├── inference_binary.py      # 二分类推理
│   └── inference_visionfm_...   # 多分类推理
│
└── MULTI_TASK_GUIDE.md          # 本文档
```

---

## 使用示例

### Web 界面使用

1. **分割任务**
   - 选择"分割"
   - 上传眼底图像
   - 调整透明度查看血管分割结果
   - 下载结果

2. **二分类任务**
   - 选择"二分类"
   - 点击"高级设置"
   - 提供训练好的 checkpoint 路径
   - 上传图像
   - 查看分类结果和置信度

3. **多分类任务**
   - 选择"多分类"
   - 点击"高级设置"
   - 提供 checkpoint 和类别数量
   - 上传图像
   - 查看各类别概率分布

### API 调用示例

**分割**:
```bash
curl -X POST "http://localhost:8000/api/segment" \
  -F "file=@image.jpg" \
  -F "threshold=0.5"
```

**二分类**:
```bash
curl -X POST "http://localhost:8000/api/classify/binary" \
  -F "file=@image.jpg" \
  -F "checkpoint=path/to/model.pth"
```

**多分类**:
```bash
curl -X POST "http://localhost:8000/api/classify/multiclass" \
  -F "file=@image.jpg" \
  -F "checkpoint=path/to/model.pth" \
  -F "num_labels=5"
```

---

## 技术栈

### 后端
- **FastAPI 2.0** - Web 框架
- **PyTorch 2.10** - 深度学习
- **Vision Transformer** - 模型架构
- **UNETR** - 分割头
- **CLS Head** - 分类头

### 前端
- **Vue 3.5** - 前端框架
- **Element Plus** - UI 组件库
- **TypeScript** - 类型支持
- **Vite 7.3** - 构建工具
- **Axios** - HTTP 客户端

---

## 故障排查

### 后端启动失败

1. 检查虚拟环境是否激活
2. 确认所有依赖已安装: `pip install -r requirements.txt`
3. 检查端口 8000 是否被占用

### 前端无法连接后端

1. 确认后端服务正在运行: `curl http://localhost:8000/health`
2. 检查前端 `.env` 文件中的 API 地址
3. 查看浏览器控制台错误信息

### 模型加载失败

1. 确认预训练权重文件存在
2. 检查 checkpoint 路径是否正确
3. 查看后端日志获取详细错误

---

## 下一步

1. **添加模型权重** - 获取或训练所需的模型权重文件
2. **测试真实图像** - 使用真实眼底图像验证功能
3. **性能优化** - 根据需要优化推理速度
4. **部署上线** - Docker 化部署到生产环境

---

## 参考资源

- [VisionFM 论文 (NEJM AI)](https://ai.nejm.org/doi/full/10.1056/AIoa2300221)
- [GitHub 仓库](https://github.com/ABILab-CUHK/VisionFM)
- [Fine-tuning 教程](./Fine-tuning/README.md)

---

*最后更新: 2026-02-21*
*版本: 2.0.0 - 多任务支持*
