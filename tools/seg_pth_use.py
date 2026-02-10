"""
分割模型推理脚本
支持单张图像或批量图像的分割预测
"""
import sys
from pathlib import Path
import argparse
from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm

# --- 项目路径设置 ---
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

import models
import utils
from models.unetr_head import Unetr_Head


# --- 辅助模型定义 ---
class SegModel(nn.Module):
    """一个将 encoder 和 head 封装在一起的完整分割模型"""
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        # 获取 encoder 中间特征（与训练时一致的处理）
        n = len(self.encoder.blocks) if hasattr(self.encoder, "blocks") else 12
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

        # UNETR head 可能需要原始图像尺寸
        out = self.head(feats, x)
        return out


# --- 核心功能函数 ---
def load_decoder_weights(head, ckpt_path):
    """加载 Checkpoint 权重到 head 模型中"""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 提取 state_dict
    state_dict = ckpt["state_dict"]

    # 去掉 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith('module.') else k
        new_state_dict[new_key] = v

    # 只保留 head 中存在的键（确保完全匹配）
    head_state = {k: v for k, v in new_state_dict.items() if k in head.state_dict()}

    # 加载权重（严格模式）
    missing, unexpected = head.load_state_dict(head_state, strict=True)
    if missing or unexpected:
        print(f"  注意: missing={len(missing)}, unexpected={len(unexpected)}")

    best_dice = ckpt.get('best_dice', 'N/A')
    print(f"✓ 分割头权重加载成功 (Dice: {best_dice})")
    return head


def preprocess_image(image_path, input_size, mean, std):
    """预处理单张图像 - 支持 CLAHE 增强以适应不同亮度的眼底图像"""
    import cv2

    # 读取图像
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    # 检查图像亮度，如果过暗则使用 CLAHE 增强
    img_mean = img_array.mean() / 255.0
    if img_mean < 0.15:
        # 转换到 LAB 色彩空间，只对 L 通道做 CLAHE
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)

        # 应用 CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel_clahe = clahe.apply(l_channel)

        # 合并通道并转回 RGB
        lab_clahe = cv2.merge([l_channel_clahe, a, b])
        img_array = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        img = Image.fromarray(img_array)

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transform(img).unsqueeze(0)


def predict(model, image_tensor, device):
    """执行分割预测"""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        out = model(image_tensor)
        # 二分类分割用 sigmoid，多分类用 softmax
        if out.shape[1] == 1:
            prob = torch.sigmoid(out)
        else:
            prob = torch.softmax(out, dim=1)[:, 1:, ...]
        prob_np = prob.squeeze().cpu().numpy()
    return prob_np


def save_mask(prob_np, output_path, threshold=0.5):
    """保存分割掩码（黑白二值图）"""
    # 调整尺寸到原始图像大小（如果需要）
    mask = (prob_np > threshold).astype(np.uint8) * 255
    Image.fromarray(mask).save(output_path)


def process_single_image(args):
    """处理单张图像"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # --- 1. 准备 Encoder（先不移到 device）---
    encoder = models.__dict__[args.arch](
        img_size=[args.input_size],
        patch_size=args.patch_size,
        num_classes=0
    ).eval()  # 先不 .to(device)

    utils.load_pretrained_weights(encoder, args.pretrained_weights, 'teacher', args.arch, args.patch_size)
    print(f"✓ Encoder 预训练权重加载成功")

    mean, std = utils.get_stats('Fundus')

    # --- 2. 加载分割头 ---
    embed_dim = getattr(encoder, "embed_dim", None)
    head = Unetr_Head(embed_dim=embed_dim, num_classes=args.num_labels, img_dim=args.input_size)
    head = load_decoder_weights(head, args.checkpoint)

    # --- 3. 组合模型（然后整体移到 device）---
    model = SegModel(encoder, head).to(device).eval()  # 与 notebook 一致：只传 2 个参数

    # --- 4. 加载并预处理图像 ---
    print(f"\n{'='*50}")
    print(f"预测图像: {args.image}")
    print(f"{'='*50}")

    img_tensor = preprocess_image(args.image, args.input_size, mean, std)

    # --- 5. 推理 ---
    prob_np = predict(model, img_tensor, device)

    # --- 6. 保存结果 ---
    output_path = Path(args.output) if args.output else Path(args.image).parent / "seg_mask.png"
    save_mask(prob_np, output_path)

    print(f"✓ 分割掩码已保存至: {output_path}")


def process_image_directory(args):
    """批量处理图像目录"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # --- 1. 准备 Encoder（先不移到 device）---
    encoder = models.__dict__[args.arch](
        img_size=[args.input_size],
        patch_size=args.patch_size,
        num_classes=0
    ).eval()  # 先不 .to(device)

    utils.load_pretrained_weights(encoder, args.pretrained_weights, 'teacher', args.arch, args.patch_size)
    print(f"✓ Encoder 预训练权重加载成功")

    mean, std = utils.get_stats('Fundus')

    # --- 2. 加载分割头 ---
    embed_dim = getattr(encoder, "embed_dim", None)
    head = Unetr_Head(embed_dim=embed_dim, num_classes=args.num_labels, img_dim=args.input_size)
    head = load_decoder_weights(head, args.checkpoint)

    # --- 3. 组合模型（然后整体移到 device）---
    model = SegModel(encoder, head).to(device).eval()  # 与 notebook 一致

    # --- 4. 批量处理图像 ---
    image_dir = Path(args.image_dir)
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpeg'))

    if not image_files:
        print(f"错误: 在 '{image_dir}' 中没有找到图像文件。")
        return

    print(f"\n找到 {len(image_files)} 张图像，开始处理...")

    output_dir = Path(args.output_dir) if args.output_dir else image_dir / "seg_masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(image_files, desc="处理图像"):
        img_tensor = preprocess_image(str(img_path), args.input_size, mean, std)
        prob_np = predict(model, img_tensor, device)

        # 输出文件名: 原文件名_mask.png
        output_filename = f"{img_path.stem}_mask.png"
        save_mask(prob_np, output_dir / output_filename)

    print(f"\n✓ 批量处理完成！结果已保存至: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='分割模型推理脚本')

    # --- 必需参数 ---
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='训练好的分割模型 checkpoint 路径')

    # --- 输入参数（二选一）---
    parser.add_argument('--image', type=str,
                        help='单张图像路径')
    parser.add_argument('--image-dir', type=str,
                        help='图像文件夹路径（批量预测）')

    # --- 输出参数 ---
    parser.add_argument('--output', type=str,
                        help='单张图像的输出路径（默认: 图像同目录下的 seg_mask.png）')
    parser.add_argument('--output-dir', type=str,
                        help='批量处理时的输出目录（默认: 输入目录下的 seg_masks 文件夹）')

    # --- 预训练权重 ---
    parser.add_argument('--pretrained-weights', type=str,
                        default='pretrain_weights/VFM_Fundus_weights.pth',
                        help='Encoder 预训练权重路径')

    # --- 模型参数（需与训练时保持一致）---
    parser.add_argument('--arch', type=str, default='vit_base',
                        help='模型架构 (e.g., vit_base)')
    parser.add_argument('--patch-size', type=int, default=16,
                        help='Patch 大小')
    parser.add_argument('--input-size', type=int, default=512,
                        help='模型输入尺寸')
    parser.add_argument('--num-labels', type=int, default=1,
                        help='分割类别数')

    args = parser.parse_args()

    # 验证输入参数
    if not args.image and not args.image_dir:
        print("错误: 请指定 --image 或 --image-dir 参数")
        return
    if args.image and args.image_dir:
        print("错误: --image 和 --image-dir 不能同时指定")
        return

    # 执行推理
    if args.image:
        process_single_image(args)
    else:
        process_image_directory(args)


if __name__ == "__main__":
    main()
