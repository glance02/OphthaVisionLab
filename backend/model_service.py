"""
VisionFM 分割模型服务
封装 Vision Transformer + UNETR Head，提供图像分割推理接口
"""
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
import sys
from pathlib import Path
import io
import cv2

# 添加项目路径
proj_root = Path(__file__).parent
sys.path.insert(0, str(proj_root))

import models
import utils
from models.unetr_head import Unetr_Head


class SegModel(nn.Module):
    """分割模型封装 - Encoder + Head"""
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        # 获取 encoder 中间特征（与训练时一致的处理）
        n = len(self.encoder.blocks)
        inter = self.encoder.get_intermediate_layers(x, n)

        # 选择层索引（与训练时选取一致）
        if n == 12:
            selected = [3, 5, 7, 11]
        elif n == 24:
            selected = [5, 11, 17, 23]
        else:
            # 若未知，取最后四层
            selected = list(range(max(0, n - 4), n))

        # 提取 patch token (去掉 CLS token)
        feats = [inter[idx][:, 1:] for idx in selected]

        # UNETR head 需要原始图像尺寸
        out = self.head(feats, x)
        return out


class SegmentationService:
    """分割模型服务单例"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def load_model(self):
        """加载模型（启动时调用一次）"""
        if self.model is not None:
            return

        print(f"使用设备: {self.device}")

        # 1. 创建 Encoder
        encoder = models.vit_base(
            img_size=[512],
            patch_size=16,
            num_classes=0
        )

        # 2. 加载预训练权重
        pretrained_path = Path(__file__).parent / "pretrain_weights" / "VFM_Fundus_weights.pth"
        if pretrained_path.exists():
            utils.load_pretrained_weights(encoder, str(pretrained_path), 'teacher', 'vit_base', 16)
            print("✓ 预训练权重加载成功")
        else:
            print(f"⚠ 预训练权重文件不存在: {pretrained_path}")
            print("  请将 VFM_Fundus_weights.pth 放在 backend/pretrain_weights/ 目录下")

        # 3. 创建分割头
        embed_dim = getattr(encoder, "embed_dim", 768)
        head = Unetr_Head(embed_dim=embed_dim, num_classes=1, img_dim=512)

        # 4. 加载分割头权重
        checkpoint_path = Path(__file__).parent / "checkpoints" / "checkpoint_108_linear.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu')

            # 去掉 'module.' 前缀并提取 head 权重
            state_dict = checkpoint['state_dict']
            head_state = {}
            for k, v in state_dict.items():
                new_key = k[7:] if k.startswith('module.') else k
                if new_key.startswith('head.'):
                    head_state[new_key[5:]] = v

            head.load_state_dict(head_state, strict=False)
            best_dice = checkpoint.get('best_dice', 'N/A')
            print(f"✓ 分割头权重加载成功 (Dice: {best_dice})")
        else:
            print(f"⚠ checkpoint 文件不存在: {checkpoint_path}")
            print("  请将 checkpoint_108_linear.pth 放在 backend/checkpoints/ 目录下")

        # 5. 组合模型
        self.model = SegModel(encoder, head).to(self.device).eval()
        print("✓ 模型加载完成")

    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """预处理图像"""
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(img)

        # CLAHE 增强（如果图像过暗）
        if img_array.mean() / 255.0 < 0.15:
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l_channel, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)

            lab_clahe = cv2.merge([l_channel, a, b])
            img_array = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
            img = Image.fromarray(img_array)

        mean, std = utils.get_stats('Fundus')
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return transform(img).unsqueeze(0)

    def predict(self, image_bytes: bytes) -> np.ndarray:
        """执行分割预测"""
        if self.model is None:
            self.load_model()

        # 预处理
        input_tensor = self.preprocess(image_bytes)

        # 推理
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
            prob = torch.sigmoid(output)
            prob_np = prob.squeeze().cpu().numpy()

        return prob_np

    def predict_to_mask(self, image_bytes: bytes, threshold: float = 0.5) -> bytes:
        """预测并返回掩码图像 bytes"""
        prob_np = self.predict(image_bytes)

        # 生成掩码图像
        mask = (prob_np > threshold).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask)
        mask_bytes = io.BytesIO()
        mask_img.save(mask_bytes, format='PNG')
        return mask_bytes.getvalue()


# 全局服务实例
seg_service = SegmentationService()
