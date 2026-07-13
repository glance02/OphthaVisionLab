# VisionFM API 文档

> 眼科人工智能多任务服务 - 版本 2.1.0

## 基础信息

- **Base URL**: `http://localhost:8000`
- **API文档**: `http://localhost:8000/docs`
- **服务启动**: `python backend/main.py`

---

## 目录

1. [基础接口](#基础接口)
2. [分割任务](#分割任务)
3. [微动脉瘤分割](#微动脉瘤分割)
4. [二分类任务](#二分类任务)
5. [多分类任务](#多分类任务)
6. [AI智能分析](#ai智能分析)
7. [其他接口](#其他接口)

---

## 基础接口

### 1. 根路径 - API 信息

**GET** `/`

返回API的基本信息和所有支持的端点。

**响应示例:**
```json
{
  "name": "VisionFM Multi-Task API",
  "version": "2.1.0",
  "description": "眼科人工智能通用基础模型服务",
  "tasks": {
    "segmentation": "眼底血管分割",
    "idrid_ma": "IDRiD微动脉瘤分割",
    "binary_classification": "二分类（如糖尿病视网膜病变检测）",
    "multiclass_classification": "多分类（如疾病分级）",
    "ai_analysis": "AI 智能分析（VisionFM + 多模态大模型）"
  }
}
```

---

### 2. 健康检查

**GET** `/health`

检查服务状态和GPU信息。

**响应示例:**
```json
{
  "status": "healthy",
  "device": "cuda:0",
  "cuda_available": true,
  "gpu_count": 1,
  "gpu_name": "NVIDIA RTX 3080",
  "loaded_models": ["seg_model", "cls_model"]
}
```

---

## 分割任务

### 3. 眼底血管分割

**POST** `/api/segment`

对眼底图像进行血管分割。

**请求参数:**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| file | File | 是 | - | 图像文件 (JPG/PNG) |
| checkpoint | string | 否 | `checkpoints/seg/checkpoint_108_linear.pth` | 分割模型checkpoint路径 |
| threshold | float | 否 | 0.5 | 分割阈值 (0-1) |
| input_size | int | 否 | 512 | 输入图像尺寸 |

**响应示例:**
```json
{
  "success": true,
  "task": "segmentation",
  "data": {
    "originalImage": "data:image/jpeg;base64,...",
    "maskImage": "data:image/png;base64,...",
    "threshold": 0.5,
    "shape": "512x512"
  }
}
```

---

## 微动脉瘤分割

### 4. IDRiD 微动脉瘤分割

**POST** `/api/idrid/ma`

对眼底图像进行微动脉瘤分割（专门针对IDRiD数据集）。

**请求参数:**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| file | File | 是 | - | 图像文件 (JPG/PNG) |
| checkpoint | string | 否 | `checkpoints/idrid_ma/net.pt7` | 模型checkpoint路径 |
| threshold | float | 否 | 0.5 | 分割阈值 (0-1) |
| patch_size | int | 否 | 96 | Patch 尺寸 |
| stride | int | 否 | 50 | 步长 |

**响应示例:**
```json
{
  "success": true,
  "task": "idrid_microaneurysm_segmentation",
  "data": {
    "originalImage": "data:image/jpeg;base64,...",
    "maskImage": "data:image/png;base64,...",
    "num_lesions": 5,
    "lesion_area": 1234,
    "lesion_ratio": "0.15%",
    "threshold": 0.5,
    "shape": "512x512"
  }
}
```

---

## 二分类任务

### 5. 二分类

**POST** `/api/classify/binary`

二分类任务，如糖尿病视网膜病变检测。

**请求参数:**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| file | File | 是 | - | 图像文件 (JPG/PNG) |
| checkpoint | string | 是 | - | 二分类模型checkpoint路径 |
| input_size | int | 否 | 224 | 输入图像尺寸 |
| n_last_blocks | int | 否 | 4 | 使用的最后层数 |
| avgpool | int | 否 | 0 | 池化模式: 0=CLS, 1=patch avg |

**响应示例:**
```json
{
  "success": true,
  "task": "binary_classification",
  "data": {
    "predicted_class": 1,
    "probability": 0.87,
    "confidence": 0.87
  }
}
```

---

## 多分类任务

### 6. 多分类

**POST** `/api/classify/multiclass`

多分类任务，如疾病分级。

**请求参数:**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| file | File | 是 | - | 图像文件 (JPG/PNG) |
| checkpoint | string | 是 | - | 多分类模型checkpoint路径 |
| num_labels | int | 是 | - | 类别数量 |
| input_size | int | 否 | 224 | 输入图像尺寸 |
| n_last_blocks | int | 否 | 4 | 使用的最后层数 |
| avgpool | int | 否 | 0 | 池化模式 |

**响应示例:**
```json
{
  "success": true,
  "task": "multiclass_classification",
  "data": {
    "predicted_class": 2,
    "confidence": 0.92,
    "probabilities": [0.02, 0.06, 0.92, 0.0]
  }
}
```

---

## AI智能分析

### 7. AI 智能分析

**POST** `/api/ai/analyze`

整合 VisionFM 本地推理 + 百炼多模态大模型，生成综合分析报告。

**请求参数:**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| file | File | 是 | - | 眼底图像文件 (JPG/PNG) |
| run_segmentation | bool | 否 | true | 是否运行分割模型 |
| run_classification | bool | 否 | true | 是否运行分类模型 |
| seg_checkpoint | string | 否 | `checkpoints/seg/checkpoint_108_linear.pth` | 分割模型路径 |
| cls_checkpoint | string | 否 | `checkpoints/single_cls/checkpoint_teacher_linear.pth` | 分类模型路径 |
| seg_threshold | float | 否 | 0.5 | 分割阈值 |
| temperature | float | 否 | 0.7 | AI 分析创意度 (0-1) |

**响应示例:**
```json
{
  "success": true,
  "data": {
    "model_results": {
      "segmentation": {...},
      "classification": {...}
    },
    "ai_analysis": "根据图像分析，...",
    "metadata": {
      "tokens_used": 1500,
      "processing_time": "2.5s"
    },
    "images": {
      "original": "data:image/jpeg;base64,...",
      "mask": "data:image/png;base64,..."
    }
  }
}
```

---

### 8. AI 分析状态

**GET** `/api/ai/status`

检查 AI 分析服务状态。

**响应示例:**
```json
{
  "enabled": true,
  "api_key_configured": true,
  "model": "qwen-vl-plus",
  "cache_size": 5
}
```

---

### 9. 测试 API 连接

**POST** `/api/ai/test-connection`

测试百炼 API 连通性。

**响应示例:**
```json
{
  "success": true,
  "message": "API 连接正常",
  "model": "qwen-vl-plus"
}
```

---

### 10. 清空缓存

**POST** `/api/ai/clear-cache`

清空分析缓存。

**响应示例:**
```json
{
  "success": true,
  "message": "缓存已清空"
}
```

---

## 其他接口

### 11. 模型信息

**GET** `/model/info`

获取支持的模型信息。

---

### 12. 任务列表

**GET** `/tasks`

列出所有支持的任务类型。

---

## 错误响应

所有接口在出错时返回标准 HTTP 错误：

```json
{
  "detail": "错误描述信息"
}
```

常见状态码：
- `400` - 请求参数错误
- `500` - 服务器内部错误
- `503` - 服务未配置（如缺少 API Key）

---

## 技术栈

- **框架**: FastAPI
- **模型**: VisionFM (Vision Transformer)
- **AI服务**: 阿里云百炼 (DashScope)
- **运行环境**: PyTorch + CUDA