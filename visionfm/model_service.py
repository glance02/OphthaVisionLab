"""
模型服务封装
整合DR分类模型和分割模型的推理接口
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

import utils
import models.vision_transformer as vit
from models.head import ClsHead
from models.unetr_head import Unetr_Head


class SegModelWrapper(nn.Module):
    """分割模型包装器，封装encoder和head"""
    def __init__(self, encoder, head, arch):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.arch = arch

    def forward(self, x):
        # 获取encoder中间特征
        n = len(self.encoder.blocks) if hasattr(self.encoder, "blocks") else 12
        inter = self.encoder.get_intermediate_layers(x, n)

        # 根据模型架构选择特征层索引
        if 'base' in self.arch or 'small' in self.arch:  # 12层
            selected_indices = [3, 5, 7, 11]
        elif 'large' in self.arch:  # 24层
            selected_indices = [5, 11, 17, 23]
        else:  # 未知时，取最后四层
            selected_indices = list(range(max(0, n - 4), n))

        # 提取patch token (去掉CLS token)
        feats = [inter[idx][:, 1:] for idx in selected_indices]

        # UNETR head需要原始图像尺寸
        out = self.head(feats, x)
        return out


class ModelService:
    """统一的模型服务接口"""

    def __init__(self,
                 dr_checkpoint_path=None,
                 seg_checkpoint_path=None,
                 pretrained_weights_path=None,
                 device='cuda'):
        """
        初始化模型服务

        Args:
            dr_checkpoint_path: DR分类模型checkpoint路径
            seg_checkpoint_path: 分割模型checkpoint路径
            pretrained_weights_path: 预训练权重路径
            device: 运行设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 默认路径
        if dr_checkpoint_path is None:
            dr_checkpoint_path = './myProcessResults/single_cls_260106_Binary_finetune/checkpoint_teacher_linear.pth'
        if seg_checkpoint_path is None:
            seg_checkpoint_path = './results/single_seg_debug/checkpoint_108_linear.pth'
        if pretrained_weights_path is None:
            pretrained_weights_path = './pretrain_weights/VFM_Fundus_weights.pth'

        # 加载DR分类模型
        self.dr_model, self.dr_classifier = self._load_dr_model(
            dr_checkpoint_path, pretrained_weights_path
        )

        # 加载分割模型
        self.seg_model = self._load_seg_model(
            seg_checkpoint_path, pretrained_weights_path
        )

        # 图像预处理
        mean, std = utils.get_stats('Fundus')
        self.dr_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.seg_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        print("模型服务初始化完成！")

    def _load_dr_model(self, checkpoint_path, pretrained_weights):
        """加载DR分类模型"""
        print("正在加载DR分类模型...")

        # 加载ViT backbone
        model = vit.__dict__['vit_base'](
            img_size=[224],
            patch_size=16,
            num_classes=0,
            use_mean_pooling=False,
        )
        utils.load_pretrained_weights(model, pretrained_weights, "teacher", "vit_base", 16)
        model = model.to(self.device)

        # 加载分类头
        embed_dim = model.embed_dim
        linear_classifier = ClsHead(
            embed_dim=embed_dim * 4,  # n_last_blocks=4
            num_classes=1,
        )
        linear_classifier = linear_classifier.to(self.device)

        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 加载分类器权重
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # 移除 'module.' 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            linear_classifier.load_state_dict(new_state_dict)
            print("  ✓ 分类器权重加载成功")

        # 加载fine-tuning backbone权重
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("  ✓ Fine-tuning backbone权重加载成功")
        else:
            print("  ✓ 使用预训练backbone（冻结模式）")

        model.eval()
        linear_classifier.eval()

        return model, linear_classifier

    def _load_seg_model(self, checkpoint_path, pretrained_weights):
        """加载分割模型"""
        print("正在加载分割模型...")

        # 加载encoder
        encoder = vit.__dict__['vit_base'](
            img_size=[512],
            patch_size=16,
            num_classes=0
        ).to(self.device)
        encoder.eval()

        # 加载预训练权重
        utils.load_pretrained_weights(encoder, pretrained_weights, 'teacher', 'vit_base', 16)
        print("  ✓ Encoder预训练权重加载成功")

        # 构建head
        embed_dim = encoder.embed_dim
        head = Unetr_Head(embed_dim=embed_dim, num_classes=1, img_dim=512)

        # 加载checkpoint权重
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt["state_dict"]

        # 去掉 "module." 前缀
        sd_fixed = {}
        for k, v in sd.items():
            nk = k[len("module."):] if k.startswith("module.") else k
            sd_fixed[nk] = v

        # 映射权重
        mapped = {}
        ckeys = list(sd_fixed.keys())
        hkeys = list(head.state_dict().keys())
        for hk in hkeys:
            for ck in ckeys:
                if ck.endswith(hk):
                    mapped[hk] = sd_fixed[ck]
                    break

        print(f"  ✓ 分割head权重加载成功 ({len(mapped)}/{len(hkeys)})")
        head.load_state_dict(mapped, strict=True)
        head = head.to(self.device)

        # 组合完整模型
        model = SegModelWrapper(encoder, head, 'vit_base').to(self.device)
        model.eval()

        return model

    def predict_dr(self, image):
        """
        DR二分类预测

        Args:
            image: PIL Image对象

        Returns:
            dict: {
                'class': 预测类别 (0=健康, 1=DR阳性),
                'probability': 概率值 (0-1),
                'confidence': 置信度 ('高'/'中'/'低'),
                'logit': 原始logit值
            }
        """
        # 预处理图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')

        image_tensor = self.dr_transform(image).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            # 提取特征（获取最后4层）
            intermediate_output = self.dr_model.get_intermediate_layers(image_tensor, n=4)

            # 拼接多层的CLS tokens
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)

            # 通过分类头
            logit = self.dr_classifier(output).squeeze()
            prob = torch.sigmoid(logit).item()
            pred_class = 1 if prob > 0.5 else 0

        # 计算置信度
        if prob > 0.8 or prob < 0.2:
            confidence = "高"
        elif prob > 0.65 or prob < 0.35:
            confidence = "中"
        else:
            confidence = "低"

        return {
            'class': int(pred_class),
            'probability': float(prob),
            'probability_percent': f"{prob * 100:.1f}%",
            'confidence': confidence,
            'logit': float(logit),
            'class_name': 'DR阳性' if pred_class == 1 else '健康'
        }

    def predict_seg(self, image):
        """
        病变区域分割预测

        Args:
            image: PIL Image对象

        Returns:
            dict: {
                'mask': 分割掩码 (PIL Image),
                'mask_array': 分割掩码numpy数组,
                'lesion_area': 病变面积 (像素数),
                'lesion_ratio': 病变占比
            }
        """
        # 预处理图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')

        original_size = image.size
        image_tensor = self.seg_transform(image).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            output = self.seg_model(image_tensor)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()

        # 后处理
        mask = (prob > 0.5).astype(np.uint8) * 255

        # 计算病变面积和占比
        lesion_area = np.sum(mask > 0)
        total_area = mask.shape[0] * mask.shape[1]
        lesion_ratio = lesion_area / total_area * 100

        # 转换为PIL Image
        mask_image = Image.fromarray(mask).resize(original_size, Image.NEAREST)

        return {
            'mask': mask_image,
            'mask_array': mask,
            'lesion_area': int(lesion_area),
            'lesion_ratio': float(lesion_ratio),
            'lesion_ratio_str': f"{lesion_ratio:.2f}%"
        }

    def predict_combined(self, image):
        """
        综合预测（DR分类 + 分割）

        Args:
            image: PIL Image对象或图像路径

        Returns:
            dict: 包含DR预测和分割结果的完整字典
        """
        dr_result = self.predict_dr(image)
        seg_result = self.predict_seg(image)

        return {
            'dr': dr_result,
            'segmentation': seg_result
        }


# 测试代码
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='测试模型服务')
    parser.add_argument('--image', type=str, help='测试图像路径')
    parser.add_argument('--device', type=str, default='cuda', help='运行设备')
    args = parser.parse_args()

    # 初始化模型服务
    service = ModelService(device=args.device)

    if args.image:
        # 测试单张图像
        print(f"\n{'='*50}")
        print(f"测试图像: {args.image}")
        print(f"{'='*50}")

        result = service.predict_combined(args.image)

        print("\n【DR分类结果】")
        print(f"  类别: {result['dr']['class_name']}")
        print(f"  概率: {result['dr']['probability_percent']}")
        print(f"  置信度: {result['dr']['confidence']}")

        print("\n【分割结果】")
        print(f"  病变面积: {result['segmentation']['lesion_area']} 像素")
        print(f"  病变占比: {result['segmentation']['lesion_ratio_str']}")
    else:
        print("请使用 --image 参数指定测试图像")
