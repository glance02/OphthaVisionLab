# VisionFM Backend Models - 分类头
import torch
import torch.nn as nn


class ClsHead(nn.Module):
    """
    分类头 - 用于二分类/多分类任务
    Wu, Jianfang, et al. "Vision Transformer‐based recognition of diabetic retinopathy grade." Medical Physics 48.12 (2021): 7850-7863.
    """
    def __init__(self, embed_dim, num_classes, layers=3):
        super(ClsHead, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.layers = layers

        if self.layers == 3:
            channels = [self.embed_dim, self.embed_dim//2, self.embed_dim//4, self.num_classes]
            self.classifier = nn.Sequential(
                nn.Linear(channels[0], channels[1]),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(channels[1], channels[2]),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(channels[2], channels[3])
            )
        elif self.layers == 2:
            channels = [self.embed_dim, self.embed_dim//4, self.num_classes]
            self.classifier = nn.Sequential(
                nn.Linear(channels[0], channels[1]),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(channels[1], channels[2])
            )
        elif self.layers == 1:
            channels = [self.embed_dim, self.num_classes]
            self.classifier = nn.Sequential(
                nn.Linear(channels[0], channels[1]),
            )

        from utils import trunc_normal_
        self.channel_ln = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.init_weights()

    def init_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias.data, 0.0)
                nn.init.normal_(m.weight.data, mean=0.0, std=0.01)

    def forward(self, x):
        # x shape: [B, C] or [B, C, 1, 1]
        if len(x.shape) == 4:
            x = x.squeeze(-1).squeeze(-1)
        x = self.channel_ln(x)
        return self.classifier(x)
