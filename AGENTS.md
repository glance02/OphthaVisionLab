# VisionFM - 智能体开发指南

> 本文档为在 VisionFM 项目上工作的 AI 编码智能体提供必要的信息。
> 
> VisionFM: 面向通用眼科人工智能的视觉基础模型
> 论文: [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIoa2300221)

---

## 1. 项目概述

### VisionFM 是什么？

VisionFM 是一个**多模态多任务视觉基础模型**，基于来自超过 50 万名受试者的 340 万张眼科图像进行预训练。它使通用眼科人工智能能够：

- 处理 **8 种成像模态**：眼底彩照、OCT、FFA、裂隙灯、B 超、MRI、外眼像、UBM
- 执行 **多种任务**：疾病识别、疾病进展预测、分割、标志点检测、生物标志物预测

### 架构

```
VisionFM 采用 Encoder-Decoder 架构:

┌─────────────────────────────────────────────────────────────┐
│                      VisionFM 架构                           │
├─────────────────────────────────────────────────────────────┤
│  Encoder (ViT-Base)                                          │
│  ├── 基于 Vision Transformer                                 │
│  ├── 预训练权重: pretrain_weights/VFM_*.pth                  │
│  └── 8种模态各自独立的编码器                                  │
│                                                              │
│  Decoder (任务特定)                                          │
│  ├── Segmentation: UNETR Head                                │
│  ├── Classification: ClsHead (MLP)                          │
│  └── 其他任务: 专用解码器                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 技术栈

### 核心技术

| 组件 | 技术 | 版本 | 用途 |
|------|------|------|------|
| **后端** | Python | 3.8+ | 主要语言 |
| **深度学习** | PyTorch | 1.11.0+cu113 | 模型训练/推理 |
| **Web 框架** | FastAPI | 0.104.0+ | API 服务 |
| **前端** | Vue 3 | 3.5.28+ | Web UI |
| **构建工具** | Vite | 7.3.1 | 前端构建 |
| **UI 库** | Element Plus | 2.13.2+ | UI 组件 |

### 关键依赖

```
# requirements.txt 关键依赖
torch==1.11.0+cu113           # PyTorch GPU 版本
torchvision==0.12.0+cu113     # 视觉工具库
einops==0.6.1                 # Tensor 操作
monai==1.2.0                  # 医学影像 AI
opencv-python==4.7.0.72       # 图像处理
pillow==9.5.0                 # 图像处理
numpy==1.21.6                 # 数值计算
scikit-learn==1.2.2           # 机器学习工具
wandb==0.15.4                 # 实验追踪
```

---

## 3. 项目结构

```
VisionFM/
│
├── 📁 根目录 (核心文件)
│   ├── main_pretrain.py              # 预训练主脚本
│   ├── finetune_visionfm_for_multiclass_classification.py  # 多分类微调
│   ├── utils.py                      # 通用工具函数
│   ├── loader.py                     # 预训练数据加载器
│   ├── requirements.txt              # Python 依赖
│   └── train*.sh                     # 训练启动脚本
│
├── 📁 models/                        # 模型定义
│   ├── __init__.py                   # 导出 vision_transformer
│   ├── vision_transformer.py         # ViT 编码器 (核心)
│   ├── head.py                       # 分类头/分割头定义
│   ├── unetr_head.py                 # UNETR 分割解码器
│   └── head_improved.py              # 改进版头
│
├── 📁 evaluation/                    # 训练脚本 (下游任务)
│   ├── train_seg_decoder.py          # 分割解码器训练
│   ├── train_cls_decoder.py          # 单模态分类训练
│   ├── train_cls_multi_decoder.py    # 多模态分类训练
│   ├── train_cls_decoder_finetune.py # 微调分类器
│   ├── train_forecasting_decoder.py  # 疾病预测训练
│   ├── train_landmark_decoder.py     # 标志点检测训练
│   ├── train_metric_reg_multi_decoder.py  # 生物标志物预测
│   ├── extract_features.py           # 特征提取
│   ├── dataset.py                    # 数据集定义
│   ├── transforms.py                 # 数据增强
│   ├── evaluation_funcs.py           # 评估函数
│   └── random_data.py                # 生成随机测试数据
│
├── 📁 tools/                         # 推理工具
│   ├── seg_pth_use.py                # 分割推理脚本
│   ├── inference_binary.py           # 二分类推理
│   └── inference_visionfm_for_multiclass_classification.py  # 多分类推理
│
├── 📁 backend/                       # Web 服务后端
│   ├── main.py                       # FastAPI 主入口
│   ├── inference_service.py          # 通用推理服务
│   ├── model_factory.py              # 模型工厂
│   ├── model_service.py              # 模型服务封装
│   ├── utils.py                      # 后端工具函数
│   ├── requirements.txt              # 后端依赖
│   ├── pretrain_weights/             # 预训练权重目录
│   ├── checkpoints/                  # 任务权重目录
│   │   ├── seg/                     # 分割模型权重
│   │   └── single_cls/              # 二分类模型权重
│
├── 📁 frontend/                      # Web 前端 (Vue 3)
│   ├── src/
│   │   ├── components/               # Vue 组件
│   │   │   ├── TaskSelector.vue      # 任务选择器
│   │   │   ├── ImageUpload.vue       # 图片上传
│   │   │   ├── ResultDisplay.vue     # 分割结果显示
│   │   │   └── ClassificationResult.vue  # 分类结果显示
│   │   ├── App.vue                   # 主应用
│   │   ├── main.ts                   # 入口
│   │   └── router/                   # 路由
│   ├── package.json                  # Node 依赖
│   └── .env                          # API 配置
│
├── 📁 Fine-tuning/                   # 微调教程
│   ├── README.md                     # 英文教程
│   └── README_zh.md                  # 中文教程
│
├── 📁 docs/                          # 项目文档
│   ├── ai_analysis_plan.md
│   └── web_deployment_plan.md
│
├── 📁 pretrain_weights/              # 预训练权重存储
├── 📁 results/                       # 训练结果存储
├── 📁 dataset/                       # 数据集目录 (用户创建)
│
├── 📄 DEPLOYMENT.md                  # 部署文档
├── 📄 MODEL_GUIDE.md                 # 模型使用指南 (中文)
├── 📄 MULTI_TASK_GUIDE.md            # 多任务部署文档
├── 📄 PROJECT_EXPLAINED.md           # 项目详解 (中文)
└── 📄 README.md / README_zh.md       # 项目说明
```

---

## 4. 构建和运行命令

### 环境设置

```bash
# 1. 创建 Conda 环境
conda create -n vfm python=3.8
conda activate vfm

# 2. 安装依赖
pip install -r requirements.txt

# 后端额外依赖 (如果需要 Web 服务)
pip install -r backend/requirements.txt
```

### 预训练

```bash
# 预训练 Fundus 编码器 (4 GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nnodes 1 --node_rank 0 --nproc_per_node=4 \
    --master_addr=127.0.0.1 --master_port=29500 \
    main_pretrain.py \
    --data_path ./dataset/pretrain \
    --modality Fundus \
    --epochs 400 \
    --batch_size_per_gpu 32 \
    --output_dir ./results \
    --name Pretrain_Fundus
```

### 微调

```bash
# 多分类微调 (单 GPU)
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=20030 \
    finetune_visionfm_for_multiclass_classification.py \
    --pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
    --data_path ./data/PAPILA \
    --num_labels 3 \
    --output_dir ./results/PAPILA_FT
```

### 训练解码器

```bash
# 分割解码器训练
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=29509 \
    evaluation/train_seg_decoder.py \
    --pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
    --data_path ./dataset/seg_data \
    --num_labels 1 \
    --output_dir ./results/seg

# 分类解码器训练
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=29501 \
    evaluation/train_cls_decoder.py \
    --pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
    --data_path ./dataset/cls_data \
    --num_labels 5 \
    --output_dir ./results/cls
```

### Web 服务

```bash
# 方式 1: 使用启动脚本 (推荐)
# Windows
start.bat

# Linux/Mac
./start.sh

# 方式 2: 手动启动
# 后端
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
python main.py

# 前端 (新终端)
cd frontend
npm install  # 首次运行
npm run dev
```

### 推理

```bash
# 分割推理
python tools/seg_pth_use.py \
    --checkpoint checkpoints/seg/checkpoint_108_linear.pth \
    --pretrained-weights pretrain_weights/VFM_Fundus_weights.pth \
    --image path/to/image.jpg \
    --output seg_result.png

# 二分类推理
python tools/inference_binary.py \
    --checkpoint path/to/checkpoint.pth \
    --image path/to/image.jpg \
    --finetune
```
---

## 6. 测试策略

### 数据验证

```bash
# 生成随机测试数据
python evaluation/random_data.py --task pretrain --dst_dir ./test_data/pretrain
python evaluation/random_data.py --task segmentation --dst_dir ./test_data/seg
python evaluation/random_data.py --task single_cls --dst_dir ./test_data/cls
```

### 模型测试

```bash
# 测试模型加载
cd backend
python -c "from model_factory import ModelFactory; mf = ModelFactory(); print('OK')"

# API 健康检查
curl http://localhost:8000/health
```

### 单元测试 (前端)

```bash
cd frontend
npm run test:unit
```

---

## 7. 关键配置文件

### 无根级 pyproject.toml/package.json

本项目没有统一的 `pyproject.toml` 或根级 `package.json`，配置分散如下:

| 文件 | 用途 |
|------|------|
| `requirements.txt` | Python 训练依赖 |
| `backend/requirements.txt` | 后端服务依赖 |
| `frontend/package.json` | 前端 Node 依赖 |

### 重要路径

```python
# 预训练权重路径 (必须下载)
PRETRAIN_WEIGHTS = {
    'Fundus': 'pretrain_weights/VFM_Fundus_weights.pth',
    'OCT': 'pretrain_weights/VFM_OCT_weights.pth',
    'FFA': 'pretrain_weights/VFM_FFA_weights.pth',
    'External': 'pretrain_weights/VFM_External_weights.pth',
    # ... 其他模态
}

# 解码器权重路径 (训练生成)
CHECKPOINT_DIR = 'results/'
```

---

## 8. 部署信息

### Web 服务架构

```
┌─────────────────────────────────────────────────────────────┐
│                     前端 (Vue 3)                             │
│  - 端口: 5173                                               │
│  - 构建: npm run build                                      │
│  - 开发: npm run dev                                        │
└─────────────────────────────────────────────────────────────┘
                            │ HTTP
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  后端 (FastAPI)                              │
│  - 端口: 8000                                               │
│  - API 文档: /docs                                          │
│  - 健康检查: /health                                        │
│                                                             │
│  端点:                                                      │
│  - POST /api/segment          # 分割任务                    │
│  - POST /api/classify/binary  # 二分类                      │
│  - POST /api/classify/multiclass  # 多分类                  │
│  - GET  /model/info           # 模型信息                    │
│  - GET  /tasks                # 任务列表                    │
└─────────────────────────────────────────────────────────────┘
```

### 部署所需文件

部署 Web 服务前必须准备:

1. **预训练权重** (必须下载):
   - `backend/pretrain_weights/VFM_Fundus_weights.pth`
   - 下载链接: [Google Drive](https://drive.google.com/file/d/13uWm0a02dCWyARUcrCdHZIcEgRfBmVA4/view)

2. **任务权重** (训练或获取):
   - 分割: `backend/checkpoints/seg/checkpoint_108_linear.pth`
   - 二分类: `backend/checkpoints/single_cls/checkpoint_teacher_linear.pth`
   - 多分类: 根据具体任务训练

---

## 9. 安全考虑

### 文件上传

- 后端限制文件大小: **10MB**
- 仅接受图片格式: `image/*`
- 文件类型验证在 `backend/main.py`

### CORS 配置

```python
# backend/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 许可

本项目采用 **研究教育许可**:
- ✅ 允许: 研究、教育用途
- ❌ 禁止: 商业用途
- 详见 `LICENSE` 文件

---

## 10. 常见问题与解决方案

### 问题 1: CUDA 内存不足

```bash
# 解决方案: 减小 batch_size
--batch_size_per_gpu 8  # 默认可能是 32/64
```

### 问题 2: 模块未找到

```bash
# 确保在项目根目录运行
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 问题 3: 前端无法连接后端

```bash
# 检查 .env 文件
cat frontend/.env
# 应为: VITE_API_BASE_URL=http://localhost:8000
```

### 问题 4: 模型加载失败

```bash
# 检查权重文件存在性
ls -lh backend/pretrain_weights/
ls -lh backend/checkpoints/seg/
ls -lh backend/checkpoints/single_cls/
```

---

## 11. 开发工作流

### 添加新任务

1. **创建解码器** (如需要): `models/your_head.py`
2. **创建训练脚本**: `evaluation/train_your_decoder.py`
3. **创建推理脚本**: `tools/inference_your_task.py`
4. **添加到后端**:
   - 更新 `backend/inference_service.py`
   - 添加端点到 `backend/main.py`
5. **前端集成**:
   - 添加任务到 `frontend/src/components/TaskSelector.vue`
   - 创建结果展示组件

---

## 12. 参考资源

- **论文**: [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIoa2300221)
- **GitHub**: https://github.com/ABILab-CUHK/VisionFM
- **中文文档**: `PROJECT_EXPLAINED.md`, `MODEL_GUIDE.md`, `MULTI_TASK_GUIDE.md`
- **英文文档**: `README.md`, `Fine-tuning/README.md`

---

*本文档为 AI 编码智能体维护。人类贡献者请参阅 README.md*

*最后更新: 2026-02-22*