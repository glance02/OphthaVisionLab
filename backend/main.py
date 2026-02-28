"""
VisionFM 多任务 FastAPI 后端服务
支持二分类、多分类、分割等多种任务
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64
import io
from pathlib import Path
from typing import Optional

from inference_service import inference_service
from routers.ai_analysis import router as ai_router

app = FastAPI(
    title="VisionFM Multi-Task API",
    description="眼科人工智能多任务服务 - 支持分类、分割、AI 智能分析",
    version="2.1.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 注册 AI 分析路由
app.include_router(ai_router)

# ============================================================================
# 基础接口
# ============================================================================

@app.get("/")
async def root():
    """根路径 - API 信息"""
    return {
        "name": "VisionFM Multi-Task API",
        "version": "2.1.0",
        "description": "眼科人工智能通用基础模型服务",
        "tasks": {
            "segmentation": "眼底血管分割",
            "binary_classification": "二分类（如糖尿病视网膜病变检测）",
            "multiclass_classification": "多分类（如疾病分级）",
            "ai_analysis": "AI 智能分析（VisionFM + 多模态大模型）"
        },
        "endpoints": {
            "health": "/health",
            "segmentation": "/api/segment",
            "binary_classify": "/api/classify/binary",
            "multiclass_classify": "/api/classify/multiclass",
            "ai_analyze": "/api/ai/analyze",
            "ai_status": "/api/ai/status",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """健康检查接口"""
    import torch
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0

    device_info = {
        "status": "healthy",
        "device": str(inference_service.device),
        "cuda_available": gpu_available,
        "gpu_count": gpu_count,
        "loaded_models": list(inference_service.models.keys())
    }

    if gpu_available:
        device_info["gpu_name"] = torch.cuda.get_device_name(0)

    return device_info


# ============================================================================
# 分割任务
# ============================================================================

@app.post("/api/segment")
async def segment_image(
    file: UploadFile = File(..., description="图像文件（JPG/PNG）"),
    checkpoint: str = Form("checkpoints/seg/checkpoint_108_linear.pth", description="分割模型checkpoint路径"),
    threshold: float = Form(0.5, description="分割阈值（0-1）"),
    input_size: int = Form(512, description="输入图像尺寸")
):
    """
    眼底图像血管分割

    参数:
    - file: 图像文件
    - checkpoint: 模型checkpoint路径
    - threshold: 分割阈值，默认0.5
    - input_size: 输入尺寸，默认512

    返回:
    - originalImage: 原图base64
    - maskImage: 掩码base64
    """
    # 验证文件
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="仅支持图片文件")

    content = await file.read()

    MAX_SIZE = 10 * 1024 * 1024
    if len(content) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="文件过大，最大10MB")

    try:
        result = inference_service.predict_segmentation(
            image_bytes=content,
            checkpoint_path=checkpoint,
            input_size=input_size,
            threshold=threshold
        )

        # 转换为base64
        original_b64 = base64.b64encode(content).decode('utf-8')
        mask_b64 = base64.b64encode(result['mask']).decode('utf-8')

        return {
            "success": True,
            "task": "segmentation",
            "data": {
                "originalImage": f"data:{file.content_type};base64,{original_b64}",
                "maskImage": f"data:image/png;base64,{mask_b64}",
                "threshold": threshold,
                "shape": f"{result['shape'][0]}x{result['shape'][1]}"
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"分割失败: {str(e)}")


# ============================================================================
# 二分类任务
# ============================================================================

@app.post("/api/classify/binary")
async def classify_binary(
    file: UploadFile = File(..., description="图像文件（JPG/PNG）"),
    checkpoint: str = Form(..., description="二分类模型checkpoint路径"),
    input_size: int = Form(224, description="输入图像尺寸"),
    n_last_blocks: int = Form(4, description="使用的最后层数"),
    avgpool: int = Form(0, description="池化模式: 0=CLS, 1=patch avg")
):
    """
    二分类任务（如：糖尿病视网膜病变检测）

    参数:
    - file: 图像文件
    - checkpoint: 模型checkpoint路径
    - input_size: 输入尺寸，默认224
    - n_last_blocks: 使用的最后层数，默认4
    - avgpool: 池化模式，默认0（CLS token）

    返回:
    - predicted_class: 预测类别（0或1）
    - probability: 阳性概率
    - confidence: 置信度
    """
    # 验证文件
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="仅支持图片文件")

    content = await file.read()

    MAX_SIZE = 10 * 1024 * 1024
    if len(content) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="文件过大，最大10MB")

    try:
        result = inference_service.predict_binary(
            image_bytes=content,
            checkpoint_path=checkpoint,
            input_size=input_size,
            n_last_blocks=n_last_blocks,
            avgpool=avgpool
        )

        return {
            "success": True,
            "task": "binary_classification",
            "data": result
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"分类失败: {str(e)}")


# ============================================================================
# 多分类任务
# ============================================================================

@app.post("/api/classify/multiclass")
async def classify_multiclass(
    file: UploadFile = File(..., description="图像文件（JPG/PNG）"),
    checkpoint: str = Form(..., description="多分类模型checkpoint路径"),
    num_labels: int = Form(..., description="类别数量"),
    input_size: int = Form(224, description="输入图像尺寸"),
    n_last_blocks: int = Form(4, description="使用的最后层数"),
    avgpool: int = Form(0, description="池化模式")
):
    """
    多分类任务（如：疾病分级、多类别识别）

    参数:
    - file: 图像文件
    - checkpoint: 模型checkpoint路径
    - num_labels: 类别数量
    - input_size: 输入尺寸，默认224
    - n_last_blocks: 使用的最后层数
    - avgpool: 池化模式

    返回:
    - predicted_class: 预测类别
    - confidence: 置信度
    - probabilities: 所有类别的概率分布
    """
    # 验证文件
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="仅支持图片文件")

    content = await file.read()

    MAX_SIZE = 10 * 1024 * 1024
    if len(content) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="文件过大，最大10MB")

    try:
        result = inference_service.predict_multiclass(
            image_bytes=content,
            checkpoint_path=checkpoint,
            num_labels=num_labels,
            input_size=input_size,
            n_last_blocks=n_last_blocks,
            avgpool=avgpool
        )

        return {
            "success": True,
            "task": "multiclass_classification",
            "data": result
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"分类失败: {str(e)}")


# ============================================================================
# 模型信息
# ============================================================================

@app.get("/model/info")
async def model_info():
    """支持的模型信息"""
    return {
        "modelName": "VisionFM",
        "architecture": "Vision Transformer (ViT-Base)",
        "supportedTasks": {
            "segmentation": {
                "description": "眼底血管分割",
                "inputSize": [512, 512],
                "outputSize": [512, 512],
                "checkpoint": "checkpoints/seg/checkpoint_108_linear.pth",
                "pretrainedWeights": "pretrain_weights/VFM_Fundus_weights.pth"
            },
            "binary_classification": {
                "description": "二分类（如DR检测）",
                "inputSize": [224, 224],
                "checkpoint": "需要提供训练好的checkpoint",
                "pretrainedWeights": "pretrain_weights/VFM_Fundus_weights.pth"
            },
            "multiclass_classification": {
                "description": "多分类（如疾病分级）",
                "inputSize": [224, 224],
                "checkpoint": "需要提供训练好的checkpoint",
                "pretrainedWeights": "pretrain_weights/VFM_Fundus_weights.pth"
            }
        },
        "supportedFormats": ["jpg", "jpeg", "png"],
        "modality": "Fundus Photography"
    }


@app.get("/tasks")
async def list_tasks():
    """列出所有支持的任务类型"""
    return {
        "tasks": [
            {
                "id": "segmentation",
                "name": "眼底血管分割",
                "endpoint": "/api/segment",
                "method": "POST",
                "required_params": ["file"],
                "optional_params": ["checkpoint", "threshold", "input_size"]
            },
            {
                "id": "binary_classification",
                "name": "二分类",
                "endpoint": "/api/classify/binary",
                "method": "POST",
                "required_params": ["file", "checkpoint"],
                "optional_params": ["input_size", "n_last_blocks", "avgpool"]
            },
            {
                "id": "multiclass_classification",
                "name": "多分类",
                "endpoint": "/api/classify/multiclass",
                "method": "POST",
                "required_params": ["file", "checkpoint", "num_labels"],
                "optional_params": ["input_size", "n_last_blocks", "avgpool"]
            },
            {
                "id": "ai_analysis",
                "name": "AI 智能分析",
                "endpoint": "/api/ai/analyze",
                "method": "POST",
                "required_params": ["file"],
                "optional_params": [
                    "run_segmentation", "run_classification",
                    "seg_checkpoint", "cls_checkpoint",
                    "seg_threshold", "temperature"
                ]
            }
        ]
    }


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    from config import config as app_config
    app_config.print_config()

    print("=" * 60)
    print("     VisionFM 多任务 API 服务 v2.1")
    print("=" * 60)
    print(f"     服务地址: http://0.0.0.0:8000")
    print(f"     API 文档: http://0.0.0.0:8000/docs")
    print(f"     支持任务: 分割、二分类、多分类、AI 智能分析")
    print("=" * 60)
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
