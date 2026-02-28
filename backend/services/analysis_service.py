"""
AI 分析整合服务
整合 VisionFM 本地推理结果 + 阿里云百炼多模态分析，生成综合报告
"""
import base64
import hashlib
import io
import logging
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .bailian_client import BaiLianClient

logger = logging.getLogger(__name__)

# ============================================================================
# Prompt 模板
# ============================================================================

FUNDUS_ANALYSIS_PROMPT = """你是一位专业的眼科影像分析助手，具有丰富的眼底疾病诊断经验。

【任务】
请分析这张眼底照片，并结合 VisionFM 模型的检测结果给出专业意见。

【模型检测结果】
{model_results_text}

【输出要求】
请按以下结构输出分析结果：

1. **影像所见描述**
   - 描述图像中可见的主要结构（视盘、血管等）
   - 描述分割模型识别出的血管分布特征

2. **异常发现分析**
   - 结合模型预测概率分析疾病风险
   - 说明模型检测到的形态学特征（如血管密度、弯曲度等）

3. **风险评估**
   - 根据概率给出风险等级（低/中/高）
   - 说明模型置信度

4. **进一步检查建议**
   - 建议的后续检查项目
   - 建议的随访时间

请用专业但易懂的语言回答，供医生参考。

【免责声明】
请在分析最后附上：本分析由 AI 辅助生成，仅供医生参考，不作为最终诊断依据。"""


class AnalysisService:
    """AI 分析整合服务"""

    def __init__(self, bailian_client: BaiLianClient):
        """
        初始化分析服务

        Args:
            bailian_client: 百炼 API 客户端实例
        """
        self.bailian = bailian_client
        # 简易内存缓存: md5 -> result
        self._cache: dict = {}
        self._cache_max_size = 100
        logger.info("AI 分析服务初始化完成")

    # ============================================================================
    # 核心分析方法
    # ============================================================================

    async def analyze_image(
        self,
        image_bytes: bytes,
        seg_result: Optional[dict] = None,
        cls_result: Optional[dict] = None,
        temperature: float = 0.7,
        use_cache: bool = True,
    ) -> dict:
        """
        综合分析眼底图像

        流程:
        1. 提取结构化特征（从本地模型推理结果）
        2. 生成分割叠加图（如果有分割结果）
        3. 准备带有结构化数据的 Prompt
        4. 调用百炼多模态 API
        5. 返回整合后的分析报告

        Args:
            image_bytes: 原始图像字节
            seg_result: 分割推理结果 (来自 inference_service.predict_segmentation)
            cls_result: 分类推理结果 (来自 inference_service.predict_binary)
            temperature: API 创意度
            use_cache: 是否使用缓存

        Returns:
            完整的分析报告 dict
        """
        start_time = time.time()

        # 缓存检查
        image_hash = hashlib.md5(image_bytes).hexdigest()
        if use_cache and image_hash in self._cache:
            logger.info(f"命中缓存: {image_hash[:8]}...")
            cached = self._cache[image_hash].copy()
            cached["from_cache"] = True
            return cached

        # 1. 提取结构化特征
        features = self._extract_features(seg_result, cls_result)

        # 2. 图片 Base64 编码 + 压缩
        image_b64 = self._encode_and_compress_image(image_bytes, max_size=1024)

        # 3. 生成分割叠加图
        overlay_b64 = None
        if seg_result and 'mask' in seg_result:
            overlay_b64 = self._create_overlay_image(image_bytes, seg_result['mask'])

        # 4. 构建 Prompt
        prompt = self._build_prompt(features)

        # 5. 调用百炼 API
        api_result = self.bailian.analyze_fundus_image(
            image_base64=image_b64,
            prompt=prompt,
            overlay_base64=overlay_b64,
            temperature=temperature,
        )

        total_time_ms = int((time.time() - start_time) * 1000)

        # 6. 组装最终结果
        result = {
            "model_results": features,
            "ai_analysis": {
                "content": api_result["content"],
                "model_used": api_result["model"],
            },
            "metadata": {
                "image_hash": image_hash,
                "timestamp": datetime.now().isoformat(),
                "tokens_used": api_result["total_tokens"],
                "input_tokens": api_result["input_tokens"],
                "output_tokens": api_result["output_tokens"],
                "api_response_time_ms": api_result["response_time_ms"],
                "total_time_ms": total_time_ms,
            },
            "from_cache": False,
        }

        # 写入缓存
        if use_cache:
            self._put_cache(image_hash, result)

        logger.info(
            f"分析完成: hash={image_hash[:8]}, "
            f"total={total_time_ms}ms, tokens={api_result['total_tokens']}"
        )
        return result

    # ============================================================================
    # 特征提取
    # ============================================================================

    def _extract_features(
        self,
        seg_result: Optional[dict],
        cls_result: Optional[dict],
    ) -> dict:
        """
        从 VisionFM 推理结果中提取结构化特征

        Args:
            seg_result: 分割结果 dict (含 mask bytes, shape 等)
            cls_result: 分类结果 dict (含 probability, predicted_class 等)

        Returns:
            结构化特征字典
        """
        features: dict = {}

        # --- 分类特征 ---
        if cls_result:
            probability = cls_result.get('probability', 0)
            predicted_class = cls_result.get('predicted_class', 0)
            features["classification"] = {
                "probability": round(probability, 4),
                "predicted_class": predicted_class,
                "class_label": "疾病" if predicted_class == 1 else "正常",
                "confidence": round(cls_result.get('confidence', 0), 4),
            }

        # --- 分割特征 ---
        if seg_result and 'mask' in seg_result:
            seg_features = self._compute_segmentation_features(seg_result['mask'])
            features["segmentation"] = seg_features

        # --- 模型元信息 ---
        features["model_info"] = {
            "name": "VisionFM",
            "modality": "Fundus",
            "encoder": "ViT-Base",
        }

        return features

    def _compute_segmentation_features(self, mask_bytes: bytes) -> dict:
        """
        从分割掩码中计算形态学特征

        Args:
            mask_bytes: PNG 格式的掩码字节

        Returns:
            分割特征字典
        """
        try:
            mask_img = Image.open(io.BytesIO(mask_bytes)).convert('L')
            mask_np = np.array(mask_img)

            total_pixels = mask_np.size
            vessel_pixels = int(np.sum(mask_np > 127))
            vessel_area_ratio = round(vessel_pixels / total_pixels, 4) if total_pixels > 0 else 0

            # 血管密度：使用分块统计
            h, w = mask_np.shape
            block_size = max(h // 4, 1)
            densities = []
            for i in range(0, h, block_size):
                for j in range(0, w, block_size):
                    block = mask_np[i:i + block_size, j:j + block_size]
                    if block.size > 0:
                        densities.append(float(np.mean(block > 127)))
            vessel_density = round(float(np.mean(densities)), 4) if densities else 0

            # 血管弯曲度指数（简化近似：使用轮廓周长 / 面积比）
            binary_mask = (mask_np > 127).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours and vessel_pixels > 0:
                total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
                tortuosity_index = round(total_perimeter / (vessel_pixels ** 0.5), 4)
            else:
                tortuosity_index = 0.0

            return {
                "vessel_area_ratio": vessel_area_ratio,
                "vessel_density": vessel_density,
                "tortuosity_index": tortuosity_index,
                "vessel_pixels": vessel_pixels,
                "total_pixels": total_pixels,
            }
        except Exception as e:
            logger.warning(f"分割特征计算失败: {e}")
            return {
                "vessel_area_ratio": 0,
                "vessel_density": 0,
                "tortuosity_index": 0,
                "error": str(e),
            }

    # ============================================================================
    # 图像处理
    # ============================================================================

    def _encode_and_compress_image(
        self, image_bytes: bytes, max_size: int = 1024, quality: int = 85
    ) -> str:
        """
        对图像进行压缩和 Base64 编码

        Args:
            image_bytes: 原始图像字节
            max_size: 最大边长（像素），超过则缩放
            quality: JPEG 压缩质量 (1-100)

        Returns:
            Base64 编码字符串（不含前缀）
        """
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # 缩放
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # 编码为 JPEG
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def _create_overlay_image(
        self, image_bytes: bytes, mask_bytes: bytes, alpha: float = 0.4
    ) -> Optional[str]:
        """
        将分割掩码叠加到原始图像上，生成可视化叠加图

        Args:
            image_bytes: 原始图像字节
            mask_bytes: 掩码图像字节 (PNG)
            alpha: 叠加透明度

        Returns:
            叠加图的 Base64 字符串，失败返回 None
        """
        try:
            # 读取原图和掩码
            orig = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            mask = Image.open(io.BytesIO(mask_bytes)).convert('L')

            # 将掩码调整为与原图相同大小
            mask = mask.resize(orig.size, Image.NEAREST)

            orig_np = np.array(orig)
            mask_np = np.array(mask)

            # 创建彩色叠加（红色通道标记血管）
            overlay = orig_np.copy()
            vessel_mask = mask_np > 127
            overlay[vessel_mask, 0] = np.clip(
                overlay[vessel_mask, 0].astype(np.float32) * (1 - alpha) + 255 * alpha, 0, 255
            ).astype(np.uint8)
            overlay[vessel_mask, 1] = (
                overlay[vessel_mask, 1].astype(np.float32) * (1 - alpha)
            ).astype(np.uint8)
            overlay[vessel_mask, 2] = (
                overlay[vessel_mask, 2].astype(np.float32) * (1 - alpha)
            ).astype(np.uint8)

            # 编码为 PNG -> Base64
            overlay_img = Image.fromarray(overlay)
            # 压缩到合理大小
            w, h = overlay_img.size
            max_size = 1024
            if max(w, h) > max_size:
                scale = max_size / max(w, h)
                overlay_img = overlay_img.resize(
                    (int(w * scale), int(h * scale)), Image.LANCZOS
                )

            buf = io.BytesIO()
            overlay_img.save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode('utf-8')

        except Exception as e:
            logger.warning(f"叠加图生成失败: {e}")
            return None

    # ============================================================================
    # Prompt 构建
    # ============================================================================

    def _build_prompt(self, features: dict) -> str:
        """
        根据结构化特征生成分析 Prompt

        Args:
            features: 提取的结构化特征字典

        Returns:
            完整的分析提示词
        """
        model_results_lines = []

        # 分类结果
        cls = features.get("classification")
        if cls:
            prob_pct = round(cls["probability"] * 100, 1)
            model_results_lines.append(f"- 疾病概率：{prob_pct}%")
            model_results_lines.append(f"- 预测类别：{cls['class_label']}")
            model_results_lines.append(f"- 模型置信度：{round(cls['confidence'] * 100, 1)}%")

        # 分割结果
        seg = features.get("segmentation")
        if seg:
            model_results_lines.append(f"- 血管面积比：{seg.get('vessel_area_ratio', 'N/A')}")
            model_results_lines.append(f"- 血管密度：{seg.get('vessel_density', 'N/A')}")
            model_results_lines.append(f"- 血管弯曲度指数：{seg.get('tortuosity_index', 'N/A')}")

        if not model_results_lines:
            model_results_lines.append("- 暂无本地模型检测结果（仅使用图像进行分析）")

        model_results_text = "\n".join(model_results_lines)

        return FUNDUS_ANALYSIS_PROMPT.format(model_results_text=model_results_text)

    # ============================================================================
    # 缓存管理
    # ============================================================================

    def _put_cache(self, key: str, value: dict):
        """写入缓存，超过上限时清除最早的条目"""
        if len(self._cache) >= self._cache_max_size:
            # 移除最早的一半
            keys_to_remove = list(self._cache.keys())[: self._cache_max_size // 2]
            for k in keys_to_remove:
                del self._cache[k]
        self._cache[key] = value

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("分析缓存已清空")
