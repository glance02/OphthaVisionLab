# VisionFM Backend Configuration

import os
from typing import Optional

from dotenv import load_dotenv

# 自动加载 .env 文件（先尝试当前目录，再尝试父目录）
load_dotenv()                         # backend/.env (如果存在)
load_dotenv(dotenv_path='../.env')     # 项目根目录 .env


class Config:
    """应用配置管理"""

    # === 阿里云百炼 API 配置 ===
    DASHSCOPE_API_KEY: Optional[str] = os.getenv('DASHSCOPE_API_KEY')
    DASHSCOPE_MODEL: str = os.getenv('DASHSCOPE_MODEL', 'qwen-vl-chat-v1')  # 默认使用基础模型

    # === VisionFM 模型配置 ===
    # 预训练权重路径
    PRETRAIN_WEIGHTS_DIR: str = "pretrain_weights"
    CHECKPOINTS_DIR: str = "checkpoints"

    # 模型文件映射
    MODEL_WEIGHTS = {
        'Fundus': 'VFM_Fundus_weights.pth',
        'OCT': 'VFM_OCT_weights.pth',
        'FFA': 'VFM_FFA_weights.pth',
        'External': 'VFM_External_weights.pth',
    }

    # === Web 服务配置 ===
    HOST: str = os.getenv('HOST', '0.0.0.0')
    PORT: int = int(os.getenv('PORT', '8000'))
    DEBUG: bool = os.getenv('DEBUG', 'false').lower() == 'true'

    # === AI 分析配置 ===
    AI_ANALYSIS_ENABLED: bool = True
    AI_MODEL_TEMPERATURE: float = float(os.getenv('AI_MODEL_TEMPERATURE', '0.7'))
    AI_MAX_TOKENS: int = int(os.getenv('AI_MAX_TOKENS', '2048'))

    # === 安全配置 ===
    MAX_FILE_SIZE_MB: int = 10  # 最大上传文件大小（MB）
    ALLOWED_EXTENSIONS: set = {'.jpg', '.jpeg', '.png', '.bmp'}

    @classmethod
    def validate_config(cls) -> list[str]:
        """验证配置，返回错误信息列表"""
        errors = []

        # 检查 API Key
        if not cls.DASHSCOPE_API_KEY:
            errors.append("DASHSCOPE_API_KEY 环境变量未设置")

        # 检查模型权重文件
        for modality, weight_file in cls.MODEL_WEIGHTS.items():
            weight_path = os.path.join(cls.PRETRAIN_WEIGHTS_DIR, weight_file)
            if not os.path.exists(weight_path):
                errors.append(f"模型权重文件不存在: {weight_path}")

        return errors

    @classmethod
    def print_config(cls):
        """打印当前配置（隐藏敏感信息）"""
        print("=== VisionFM Backend Configuration ===")
        print(f"Host: {cls.HOST}:{cls.PORT}")
        print(f"Debug: {cls.DEBUG}")
        print(f"AI Analysis: {'Enabled' if cls.AI_ANALYSIS_ENABLED else 'Disabled'}")
        print(f"DashScope Model: {cls.DASHSCOPE_MODEL}")
        print(f"API Key: {'***' + cls.DASHSCOPE_API_KEY[-4:] if cls.DASHSCOPE_API_KEY else 'Not set'}")
        print(f"Max File Size: {cls.MAX_FILE_SIZE_MB}MB")
        print("=" * 40)

# 全局配置实例
config = Config()