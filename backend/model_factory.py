"""
VisionFM 模型工厂
支持多种任务类型：二分类、多分类、分割
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

import models
import utils
from models.unetr_head import Unetr_Head
from models.head import ClsHead


class SegModel(nn.Module):
    """分割模型封装 - Encoder + UNETR Head"""
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        n = len(self.encoder.blocks)
        inter = self.encoder.get_intermediate_layers(x, n)

        # 选择层索引
        if n == 12:
            selected = [3, 5, 7, 11]
        elif n == 24:
            selected = [5, 11, 17, 23]
        else:
            selected = list(range(max(0, n - 4), n))

        # 提取 patch token
        feats = [inter[idx][:, 1:] for idx in selected]
        out = self.head(feats, x)
        return out


class ClsModel(nn.Module):
    """分类模型封装 - Encoder + Classifier Head"""
    def __init__(self, encoder, head, n_last_blocks=4, avgpool=0):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.n_last_blocks = n_last_blocks
        self.avgpool = avgpool

    def forward(self, x):
        # 获取中间层特征
        inter = self.encoder.get_intermediate_layers(x, n=self.n_last_blocks)

        # 特征提取策略
        if self.avgpool == 0:
            # 只使用 CLS token
            output = torch.cat([x[:, 0] for x in inter], dim=-1)
        elif self.avgpool == 1:
            # 只使用 patch tokens 平均
            output = torch.mean(inter[-1][:, 1:], dim=1)
        else:
            # CLS + patch tokens 平均
            output = [x[:, 0] for x in inter] + [torch.mean(inter[-1][:, 1:], dim=1)]
            output = torch.cat(output, dim=-1)

        # 通过分类头
        return self.head(output)


class ModelFactory:
    """模型工厂 - 根据任务类型创建不同的模型"""

    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pretrain_dir = Path(__file__).parent / "pretrain_weights"
        self.checkpoint_dir = Path(__file__).parent / "checkpoints"

    def create_encoder(self, arch='vit_base', patch_size=16, input_size=512):
        """创建编码器"""
        encoder = models.__dict__[arch](
            img_size=[input_size],
            patch_size=patch_size,
            num_classes=0,
            use_mean_pooling=False,
        )
        return encoder

    def load_pretrained_encoder(self, encoder, modality='Fundus'):
        """加载预训练权重"""
        pretrained_path = self.pretrain_dir / f"VFM_{modality}_weights.pth"

        if pretrained_path.exists():
            utils.load_pretrained_weights(
                encoder,
                str(pretrained_path),
                'teacher',
                'vit_base' if 'vit_base' in str(type(encoder)) else 'vit_base',
                16
            )
            print(f"✓ 预训练权重加载成功: {pretrained_path.name}")
            return True
        else:
            print(f"⚠ 预训练权重不存在: {pretrained_path}")
            return False

    def create_segmentation_model(self, checkpoint_path, pretrained_weights=True,
                                  arch='vit_base', patch_size=16, input_size=512, num_classes=1):
        """创建分割模型"""
        # 创建编码器
        encoder = self.create_encoder(arch, patch_size, input_size)

        # 加载预训练权重
        if pretrained_weights:
            self.load_pretrained_encoder(encoder, 'Fundus')

        # 创建分割头
        embed_dim = getattr(encoder, "embed_dim", 768)
        head = Unetr_Head(embed_dim=embed_dim, num_classes=num_classes, img_dim=input_size)

        # 加载 checkpoint
        if checkpoint_path and Path(checkpoint_path).exists():
            # 兼容不同 PyTorch 版本
            torch_version = tuple(map(int, torch.__version__.split('+')[0].split('.')[:2]))
            if torch_version >= (1, 12):
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['state_dict']

            # 去掉 'module.' 前缀并提取 head 权重
            # checkpoint中的键格式: 'module.d1.deconv.weight' -> 去掉 'module.' -> 'd1.deconv.weight'
            head_state = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    # 去掉 'module.' 前缀
                    new_key = k[7:]
                    head_state[new_key] = v
                elif not k.startswith('encoder'):  # 只加载head相关的权重
                    head_state[k] = v

            # 检查是否有权重被提取
            if len(head_state) > 0:
                missing, unexpected = head.load_state_dict(head_state, strict=False)
                best_dice = checkpoint.get('best_dice', 'N/A')
                print(f"✓ 分割头权重加载成功 (Dice: {best_dice}, 加载参数: {len(head_state)})")
                if missing:
                    print(f"  缺失参数 (使用默认值): {len(missing)}")
                if unexpected:
                    print(f"  未匹配参数: {len(unexpected)}")
            else:
                print(f"⚠ 警告: 未能从checkpoint提取任何head权重!")

        # 组合模型
        model = SegModel(encoder, head).to(self.device).eval()
        return model

    def create_binary_classifier(self, checkpoint_path, pretrained_weights=True,
                                  arch='vit_base', patch_size=16, input_size=224,
                                  n_last_blocks=4, avgpool=0, finetune=True):
        """创建二分类模型"""
        # 创建编码器
        encoder = self.create_encoder(arch, patch_size, input_size)

        # 加载预训练权重
        if pretrained_weights:
            self.load_pretrained_encoder(encoder, 'Fundus')

        # 创建分类头
        embed_dim = getattr(encoder, "embed_dim", 768)
        head = ClsHead(embed_dim=embed_dim * n_last_blocks, num_classes=1, layers=3)

        # 加载 checkpoint
        if checkpoint_path and Path(checkpoint_path).exists():
            # 兼容不同 PyTorch 版本
            torch_version = tuple(map(int, torch.__version__.split('+')[0].split('.')[:2]))
            if torch_version >= (1, 12):
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 加载分类头权重
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k[7:] if k.startswith('module.') else k
                    new_state_dict[new_key] = v
                head.load_state_dict(new_state_dict)
                print(f"✓ 分类头权重加载成功")

            # 如果是 fine-tuning 模型，加载 backbone
            if finetune and "model_state_dict" in checkpoint:
                encoder.load_state_dict(checkpoint["model_state_dict"])
                print(f"✓ Fine-tuning backbone 权重加载成功")

        # 组合模型
        model = ClsModel(encoder, head, n_last_blocks, avgpool).to(self.device).eval()
        return model

    def create_multiclass_classifier(self, checkpoint_path, num_labels, pretrained_weights=True,
                                      arch='vit_base', patch_size=16, input_size=224,
                                      n_last_blocks=4, avgpool=0, finetune=True):
        """创建多分类模型"""
        # 创建编码器
        encoder = self.create_encoder(arch, patch_size, input_size)

        # 加载预训练权重
        if pretrained_weights:
            self.load_pretrained_encoder(encoder, 'Fundus')

        # 创建分类头
        embed_dim = getattr(encoder, "embed_dim", 768)
        head = ClsHead(embed_dim=embed_dim * n_last_blocks, num_classes=num_labels, layers=3)

        # 加载 checkpoint
        if checkpoint_path and Path(checkpoint_path).exists():
            # 兼容不同 PyTorch 版本
            torch_version = tuple(map(int, torch.__version__.split('+')[0].split('.')[:2]))
            if torch_version >= (1, 12):
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 加载分类器权重
            if "classifier_state_dict" in checkpoint:
                head.load_state_dict(checkpoint["classifier_state_dict"], strict=False)
                print(f"✓ 多分类头权重加载成功")
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k[7:] if k.startswith('module.') else k
                    new_state_dict[new_key] = v
                head.load_state_dict(new_state_dict, strict=False)
                print(f"✓ 分类头权重加载成功")

            # 如果是 fine-tuning 模型
            if finetune and "model_state_dict" in checkpoint:
                encoder.load_state_dict(checkpoint["model_state_dict"])
                print(f"✓ Fine-tuning backbone 权重加载成功")

        # 组合模型
        model = ClsModel(encoder, head, n_last_blocks, avgpool).to(self.device).eval()
        return model
