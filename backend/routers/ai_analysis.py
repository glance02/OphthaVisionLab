"""
AI 智能分析路由
整合 VisionFM 本地推理 + 百炼多模态大模型，生成综合分析报告
"""
import base64
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from config import config
from inference_service import inference_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai", tags=["AI Analysis"])

# ============================================================================
# 延迟初始化：仅在首次请求时创建分析服务实例
# ============================================================================
_analysis_service = None


def _get_analysis_service():
    """懒加载分析服务（避免启动时 API Key 未设置导致崩溃）"""
    global _analysis_service
    if _analysis_service is None:
        from services.bailian_client import BaiLianClient
        from services.analysis_service import AnalysisService

        api_key = config.DASHSCOPE_API_KEY
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="AI 分析服务未配置：请设置 DASHSCOPE_API_KEY 环境变量"
            )
        client = BaiLianClient(api_key=api_key, model=config.DASHSCOPE_MODEL)
        _analysis_service = AnalysisService(bailian_client=client)
        logger.info("AI 分析服务已初始化")
    return _analysis_service


# ============================================================================
# API 端点
# ============================================================================

@router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(..., description="眼底图像文件（JPG/PNG）"),
    run_segmentation: bool = Form(True, description="是否运行分割模型"),
    run_classification: bool = Form(True, description="是否运行分类模型"),
    seg_checkpoint: str = Form(
        "checkpoints/seg/checkpoint_108_linear.pth",
        description="分割模型 checkpoint 路径"
    ),
    cls_checkpoint: str = Form(
        "checkpoints/single_cls/checkpoint_teacher_linear.pth",
        description="分类模型 checkpoint 路径"
    ),
    seg_threshold: float = Form(0.5, description="分割阈值"),
    temperature: float = Form(0.7, description="AI 分析创意度 (0-1)"),
):
    """
    AI 智能分析 - 上传眼底图像，获取综合分析报告

    流程：
    1. VisionFM 本地模型推理（分割 + 分类）
    2. 提取结构化特征
    3. 生成分割叠加图
    4. 调用百炼多模态大模型分析
    5. 返回综合报告

    参数:
    - file: 眼底图像
    - run_segmentation: 是否运行分割（默认 True）
    - run_classification: 是否运行分类（默认 True）
    - seg_checkpoint: 分割模型路径
    - cls_checkpoint: 分类模型路径
    - seg_threshold: 分割阈值
    - temperature: AI 创意度

    返回:
    - model_results: VisionFM 结构化检测数据
    - ai_analysis: AI 生成的自然语言分析
    - metadata: 调用元信息（tokens、耗时等）
    """
    # --- 验证文件 ---
    # 先读取文件内容
    content = await file.read()

    # 检查文件头识别图片类型（支持扩展名与实际内容不符的情况）
    if len(content) < 12:
        raise HTTPException(status_code=400, detail="文件过小，不是有效的图片")

    # 常见图片格式魔数
    image_magic = {
        b'\xff\xd8\xff': 'image/jpeg',      # JPEG
        b'\x89PNG\r\n\x1a\n': 'image/png',   # PNG
        b'GIF87a': 'image/gif',             # GIF87a
        b'GIF89a': 'image/gif',             # GIF89a
        b'RIFF': 'image/webp',              # WebP (需要进一步检查)
        b'BM': 'image/bmp',                 # BMP
    }

    detected_type = None
    for magic, img_type in image_magic.items():
        if content.startswith(magic):
            detected_type = img_type
            break

    if not detected_type:
        raise HTTPException(status_code=400, detail="仅支持图片文件（JPEG/PNG/GIF/BMP/WebP）")
    max_size = config.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大，最大 {config.MAX_FILE_SIZE_MB}MB"
        )

    # --- 获取分析服务 ---
    service = _get_analysis_service()

    # --- VisionFM 本地推理 ---
    seg_result = None
    cls_result = None

    try:
        if run_segmentation:
            logger.info("运行分割模型...")
            seg_result = inference_service.predict_segmentation(
                image_bytes=content,
                checkpoint_path=seg_checkpoint,
                threshold=seg_threshold,
            )
    except Exception as e:
        logger.warning(f"分割模型推理失败（跳过）: {e}")

    try:
        if run_classification:
            logger.info("运行分类模型...")
            cls_result = inference_service.predict_binary(
                image_bytes=content,
                checkpoint_path=cls_checkpoint,
            )
    except Exception as e:
        logger.warning(f"分类模型推理失败（跳过）: {e}")

    # --- 调用 AI 分析 ---
    try:
        result = await service.analyze_image(
            image_bytes=content,
            seg_result=seg_result,
            cls_result=cls_result,
            temperature=temperature,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI 分析失败: {str(e)}")

    # --- 构建响应 ---
    response_data = {
        "success": True,
        "data": result,
    }

    # 附加原图和掩码的 base64（供前端展示）
    original_b64 = base64.b64encode(content).decode('utf-8')
    response_data["data"]["images"] = {
        "original": f"data:{detected_type};base64,{original_b64}",
    }
    if seg_result and 'mask' in seg_result:
        mask_b64 = base64.b64encode(seg_result['mask']).decode('utf-8')
        response_data["data"]["images"]["mask"] = f"data:image/png;base64,{mask_b64}"

    return response_data


@router.get("/status")
async def ai_status():
    """
    检查 AI 分析服务状态
    """
    api_key_set = bool(config.DASHSCOPE_API_KEY)
    model_name = config.DASHSCOPE_MODEL

    status = {
        "enabled": config.AI_ANALYSIS_ENABLED,
        "api_key_configured": api_key_set,
        "model": model_name,
    }

    # 如果服务已初始化，返回缓存信息
    if _analysis_service is not None:
        status["cache_size"] = len(_analysis_service._cache)

    return status


@router.post("/test-connection")
async def test_ai_connection():
    """
    测试百炼 API 连通性
    """
    try:
        service = _get_analysis_service()
        ok = service.bailian.test_connection()
        return {
            "success": ok,
            "message": "API 连接正常" if ok else "API 连接失败",
            "model": config.DASHSCOPE_MODEL,
        }
    except HTTPException:
        raise
    except Exception as e:
        return {
            "success": False,
            "message": f"连接测试异常: {str(e)}",
        }


@router.post("/clear-cache")
async def clear_cache():
    """
    清空分析缓存
    """
    if _analysis_service is not None:
        _analysis_service.clear_cache()
        return {"success": True, "message": "缓存已清空"}
    return {"success": True, "message": "服务未初始化，无需清空"}
