"""
阿里云百炼 API 客户端
封装 DashScope MultiModalConversation API，用于眼底图像分析
"""
import time
import logging
from typing import Optional

import dashscope
from dashscope import MultiModalConversation

logger = logging.getLogger(__name__)


class BaiLianClient:
    """阿里云百炼多模态 API 客户端"""

    def __init__(self, api_key: str, model: str = "qwen-vl-max"):
        """
        初始化百炼客户端

        Args:
            api_key: 阿里云百炼 API Key
            model: 使用的模型名称，默认 qwen-vl-max
        """
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY 未设置，请配置环境变量")
        dashscope.api_key = api_key
        self.model = model
        logger.info(f"百炼客户端初始化完成，使用模型: {self.model}")

    def analyze_fundus_image(
        self,
        image_base64: str,
        prompt: str,
        overlay_base64: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict:
        """
        分析眼底图像

        Args:
            image_base64: 原始图像的 Base64 编码（不含 data:...;base64, 前缀）
            prompt: 分析提示词
            overlay_base64: 可选的分割叠加图 Base64
            temperature: 创意度（0-1）
            max_tokens: 最大输出 token 数

        Returns:
            dict: {
                "content": AI 生成的分析文本,
                "input_tokens": 输入 token 数,
                "output_tokens": 输出 token 数,
                "total_tokens": 总 token 数,
                "model": 使用的模型,
                "response_time_ms": 响应时间（毫秒）
            }
        """
        # 构建 content 列表：包含图片和文本
        content = []

        # 添加原始图像（根据输入判断格式，默认尝试 jpeg）
        image_format = "jpeg"
        # 检查 base64 开头是否可以推断格式（如果有额外参数的话）
        # 默认使用 jpeg，因为大多数眼底图像是 jpeg 格式
        content.append({
            "image": f"data:image/{image_format};base64,{image_base64}"
        })

        # 如果有分割叠加图，也添加进去
        if overlay_base64:
            content.append({
                "image": f"data:image/png;base64,{overlay_base64}"
            })

        # 添加提示文本
        content.append({"text": prompt})

        messages = [{"role": "user", "content": content}]

        start_time = time.time()
        try:
            response = MultiModalConversation.call(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            elapsed_ms = int((time.time() - start_time) * 1000)

            if response.status_code != 200:
                error_msg = getattr(response, 'message', '未知错误')
                logger.error(f"百炼 API 调用失败: {error_msg} (status={response.status_code})")
                raise RuntimeError(f"百炼 API 调用失败: {error_msg}")

            # 提取结果
            ai_content = response.output.choices[0].message.content
            # content 可能是列表或字符串
            if isinstance(ai_content, list):
                # 提取所有文本部分
                text_parts = [
                    item.get("text", "") for item in ai_content if isinstance(item, dict) and "text" in item
                ]
                ai_text = "\n".join(text_parts) if text_parts else str(ai_content)
            else:
                ai_text = str(ai_content)

            # 提取 token 使用信息
            usage = response.usage
            input_tokens = getattr(usage, 'input_tokens', 0)
            output_tokens = getattr(usage, 'output_tokens', 0)
            total_tokens = getattr(usage, 'total_tokens', input_tokens + output_tokens)

            result = {
                "content": ai_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "model": self.model,
                "response_time_ms": elapsed_ms,
            }

            logger.info(
                f"百炼 API 调用成功: model={self.model}, "
                f"tokens={total_tokens}, time={elapsed_ms}ms"
            )
            return result

        except RuntimeError:
            raise
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"百炼 API 调用异常: {e} (time={elapsed_ms}ms)")
            raise RuntimeError(f"百炼 API 调用异常: {str(e)}")

    def test_connection(self) -> bool:
        """
        测试 API 连通性（使用纯文本请求，不消耗图像 token）

        Returns:
            bool: 连通性测试是否成功
        """
        try:
            messages = [
                {
                    "role": "user",
                    "content": [{"text": "你好，请简单介绍一下你自己。"}],
                }
            ]
            response = MultiModalConversation.call(
                model=self.model,
                messages=messages,
                temperature=0.7,
            )
            logger.info(f"API 响应状态: {response.status_code}")
            logger.info(f"API 响应内容: {response}")
            return response.status_code == 200
        except Exception as e:
            import traceback
            logger.error(f"百炼连通性测试失败: {e}")
            logger.error(f"详细错误: {traceback.format_exc()}")
            return False
