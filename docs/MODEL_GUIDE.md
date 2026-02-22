# VisionFM 模型使用指南

本文档介绍 VisionFM 项目中训练完成的模型及其使用方法。

## 目录

- [项目概述](#项目概述)
- [已训练模型](#已训练模型)
  - [二分类模型](#二分类模型)
  - [分割模型](#分割模型)
- [文件依赖关系](#文件依赖关系)
- [使用指南](#使用指南)
  - [二分类模型使用](#二分类模型使用)
  - [分割模型使用](#分割模型使用)
- [快速开始](#快速开始)

---

## 项目概述

VisionFM 是一个眼科 AI 基础模型项目，支持多种眼科成像模态和任务：

- **支持的模态**: 眼底摄影 (Fundus)、光学相干断层扫描 (OCT)、荧光素眼底血管造影 (FFA) 等 8 种模态
- **支持的任务**: 分类、分割、预测、标志点检测等多种任务
- **架构**: 基于 Vision Transformer (ViT) 的编码器-解码器结构

---

## 已训练模型

### 二分类模型

#### 模型信息

| 属性 | 值 |
|------|-----|
| **模型文件** | `myProcessResults/single_cls_260106_Binary_finetune/checkpoint_teacher_linear.pth` |
| **预训练权重** | `pretrain_weights/VFM_Fundus_weights.pth` |
| **任务类型** | 二分类 (Binary Classification) |
| **输入尺寸** | 224 × 224 |
| **性能** | F1 分数 ≈ 80% |
| **用途** | 眼底图像二分类任务（如：健康 vs 疾病） |

#### 模型架构

```
输入图像 (224×224×3)
    ↓
ViT Encoder (预训练 + 微调)
    ↓
特征提取 (最后 4 层 CLS tokens 拼接)
    ↓
分类头 (ClsHead: 3层 MLP)
    ↓
输出 (二分类 logit → sigmoid 概率)
```

#### 核心文件

- **模型定义**: [models/vision_transformer.py](models/vision_transformer.py) - ViT 编码器
- **分类头**: [models/head.py](models/head.py:L245-L298) - `ClsHead` 类
- **训练脚本**: [evaluation/train_cls_decoder_finetune.py](evaluation/train_cls_decoder_finetune.py)
- **推理脚本**: [inference_binary.py](inference_binary.py)

---

### 分割模型

#### 模型信息

| 属性 | 值 |
|------|-----|
| **模型文件** | `results/single_seg_debug/checkpoint_108_linear.pth` |
| **预训练权重** | `pretrain_weights/VFM_Fundus_weights.pth` |
| **任务类型** | 图像分割 (Segmentation) |
| **输入尺寸** | 512 × 512 |
| **性能** | Dice 分数 ≈ 78.17% |
| **训练轮次** | 第 108 轮检查点 |
| **用途** | 医学图像分割（如：视网膜血管分割） |

#### 模型架构

```
输入图像 (512×512×3)
    ↓
ViT Encoder (多尺度特征提取)
    ↓
提取 4 层特征 (第 3, 5, 7, 11 层 patch tokens)
    ↓
UNETR 分割头 (Unetr_Head)
    ↓
上采样至原始尺寸
    ↓
输出分割掩码 (H×W×num_classes)
```

#### 核心文件

- **模型定义**: [models/vision_transformer.py](models/vision_transformer.py) - ViT 编码器
- **分割头**: [models/unetr_head.py](models/unetr_head.py) - `Unetr_Head` 类
- **替代分割头**: [models/head.py](models/head.py:L198-L226) - `linSeg` 类
- **训练脚本**: [evaluation/train_seg_decoder.py](evaluation/train_seg_decoder.py)
- **推理脚本**: [tools/seg_pth_use.py](tools/seg_pth_use.py)

---

## 文件依赖关系

```
VisionFM 项目结构
│
├── 预训练权重/
│   ├── VFM_Fundus_weights.pth    # 眼底摄影预训练权重
│   └── VFM_OCT_weights.pth       # OCT 预训练权重
│
├── 模型文件 (models/)
│   ├── vision_transformer.py     # ViT 编码器（核心）
│   ├── head.py                   # 分类头 ClsHead、线性分割头 linSeg
│   ├── unetr_head.py            # UNETR 分割头
│   └── __init__.py
│
├── 已训练模型/
│   ├── myProcessResults/
│   │   └── single_cls_260106_Binary_finetune/
│   │       └── checkpoint_teacher_linear.pth    # 二分类模型
│   └── results/
│       └── single_seg_debug/
│           └── checkpoint_108_linear.pth        # 分割模型
│
├── 推理脚本/
│   ├── inference_binary.py       # 二分类推理
│   ├── tools/
│   │   └── seg_pth_use.py        # 分割模型使用
│   └── model_service.py          # 模型服务接口
│
└── 工具文件/
    ├── utils.py                  # 通用工具函数
    ├── loader.py                 # 数据加载器
    └── evaluation/
        ├── dataset.py            # 数据集定义
        └── transforms.py         # 数据增强
```

**依赖关系图**:

```
┌─────────────────────────────────────────────────────────────┐
│                        推理流程                               │
└─────────────────────────────────────────────────────────────┘

二分类任务:
  预训练权重 ──┐
               ├──→ ViT Encoder ──→ ClsHead ──→ 分类结果
  微调权重 ────┘

分割任务:
  预训练权重 ──┐
               ├──→ ViT Encoder ──→ Unetr_Head ──→ 分割掩码
  分割权重 ────┘
```

---

## 使用指南

### 二分类模型使用

#### 1. 所需文件清单

| 文件 | 路径 | 用途 |
|------|------|------|
| 预训练权重 | `pretrain_weights/VFM_Fundus_weights.pth` | ViT 编码器基础权重 |
| 训练模型 | `myProcessResults/single_cls_260106_Binary_finetune/checkpoint_teacher_linear.pth` | 微调后的分类器权重 |
| 推理脚本 | `inference_binary.py` | 执行推理 |
| 工具函数 | `utils.py` | 预处理、加载权重等 |
| 模型定义 | `models/vision_transformer.py`, `models/head.py` | 模型架构 |

#### 2. 环境依赖

```bash
pip install torch torchvision pillow numpy
```

#### 3. 使用方法

**单张图像预测**:

```bash
python tools/inference_binary.py \
    --checkpoint myProcessResults/single_cls_260106_Binary_finetune/checkpoint_teacher_linear.pth \
    --pretrained_weights pretrain_weights/VFM_Fundus_weights.pth \
    --image path/to/your/image.jpg \
    --finetune
```

**批量预测**:

```bash
python tools/inference_binary.py \
    --checkpoint myProcessResults/single_cls_260106_Binary_finetune/checkpoint_teacher_linear.pth \
    --pretrained_weights pretrain_weights/VFM_Fundus_weights.pth \
    --image_dir path/to/image/folder \
    --output_file predictions.txt \
    --finetune
```

#### 4. 参数说明

| 参数 | 说明 | 必填 |
|------|------|------|
| `--checkpoint` | 训练好的模型 checkpoint 路径 | ✅ |
| `--pretrained_weights` | 预训练权重路径（默认：./pretrain_weights/VFM_Fundus_weights.pth） | ❌ |
| `--image` | 单张图像路径 | ❌* |
| `--image_dir` | 图像文件夹路径（批量预测） | ❌* |
| `--output_file` | 输出文件路径（默认：predictions.txt） | ❌ |
| `--finetune` | 是否使用微调模式（加载 backbone 权重） | ❌ |

*注：`--image` 和 `--image_dir` 至少需要提供一个

#### 5. 输出格式

**单张图像输出**:
```
Logit: 2.3456
Probability: 0.9123
Predicted Class: 1 (Positive)
```

**批量预测输出文件** (predictions.txt):
```
image1.jpg    2.3456    0.9123    1
image2.jpg    -1.2345   0.2234    0
...
```

---

### 分割模型使用

#### 1. 所需文件清单

| 文件 | 路径 | 用途 |
|------|------|------|
| 预训练权重 | `pretrain_weights/VFM_Fundus_weights.pth` | ViT 编码器基础权重 |
| 训练模型 | `results/single_seg_debug/checkpoint_108_linear.pth` | 训练的分割头权重 |
| 推理脚本 | `tools/seg_pth_use.py` | 执行推理 |
| 工具函数 | `utils.py` | 预处理、加载权重等 |
| 模型定义 | `models/vision_transformer.py`, `models/unetr_head.py` | 模型架构 |

#### 2. 环境依赖

```bash
pip install torch torchvision pillow numpy tqdm
```

#### 3. 使用方法

**单张图像分割**:

```bash
python tools/seg_pth_use.py \
    --checkpoint results/single_seg_debug/checkpoint_108_linear.pth \
    --pretrained-weights pretrain_weights/VFM_Fundus_weights.pth \
    --image path/to/your/image.jpg \
    --output seg_mask.png
```

**批量图像分割**:

```bash
python tools/seg_pth_use.py \
    --checkpoint results/single_seg_debug/checkpoint_108_linear.pth \
    --pretrained-weights pretrain_weights/VFM_Fundus_weights.pth \
    --image-dir path/to/image/folder \
    --output-dir ./seg_masks
```

#### 4. 参数说明

| 参数 | 说明 | 必填 |
|------|------|------|
| `--checkpoint` | 训练好的分割模型 checkpoint 路径 | ✅ |
| `--pretrained-weights` | 预训练权重路径（默认：pretrain_weights/VFM_Fundus_weights.pth） | ❌ |
| `--image` | 单张图像路径 | ❌* |
| `--image-dir` | 图像文件夹路径（批量预测） | ❌* |
| `--output` | 单张图像的输出路径（默认：图像同目录下的 seg_mask.png） | ❌ |
| `--output-dir` | 批量处理时的输出目录（默认：输入目录下的 seg_masks 文件夹） | ❌ |
| `--arch` | 模型架构（默认：vit_base） | ❌ |
| `--patch-size` | Patch 大小（默认：16） | ❌ |
| `--input-size` | 模型输入尺寸（默认：512） | ❌ |
| `--num-labels` | 分割类别数（默认：1） | ❌ |

*注：`--image` 和 `--image-dir` 至少需要提供一个

#### 5. 输出格式

**单张图像**: 输出黑白二值掩码图（分割区域白色255，背景黑色0）

**批量处理**: 每张图像对应一个 `{原文件名}_mask.png` 掩码文件

---

## 附录

### 模型性能参考

| 模型 | 任务 | 数据集 | 指标 | 数值 |
|------|------|--------|------|------|
| 二分类 | Binary Classification | dataset260106 | F1 Score | ~80% |
| 分割 | Vessel Segmentation | DRIVE | Dice | 0.7817 |

### 常见问题

**Q: 如何使用不同的预训练权重？**

A: 修改推理命令中的 `--pretrained_weights` 参数：
```bash
--pretrained_weights pretrain_weights/VFM_OCT_weights.pth  # 用于 OCT 图像
```

**Q: 如何调整分类阈值？**

A: 在 [inference_binary.py:130](inference_binary.py#L130) 中修改阈值：
```python
pred_class = 1 if prob > 0.5 else 0  # 将 0.5 改为其他值
```

**Q: 分割模型支持多类别吗？**

A: 支持，通过修改 `--num-labels` 参数指定类别数。

---

*文档生成时间: 2026-01-29*
*项目版本: 基于 commit 0232587*
