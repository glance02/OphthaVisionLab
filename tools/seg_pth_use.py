"""
批量对分割模型的 Checkpoint 文件进行推理，并保存可视化结果。
"""
import sys
import re
from pathlib import Path
import argparse
from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm

# --- 项目路径设置 ---
# 确保脚本可以找到 models, utils 等模块
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

import models
import utils
from models.unetr_head import Unetr_Head

# --- 辅助模型定义 ---
class SegModel(nn.Module):
    """一个将 encoder 和 head 封装在一起的完整分割模型"""
    def __init__(self, encoder, head, arch):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.arch = arch

    def forward(self, x):
        # 获取 encoder 中间特征（与训练时一致的处理）
        n = len(self.encoder.blocks) if hasattr(self.encoder, "blocks") else 12
        inter = self.encoder.get_intermediate_layers(x, n)
        
        # 根据模型架构选择特征层索引
        if 'base' in self.arch or 'small' in self.arch: # 12层
            selected_indices = [3, 5, 7, 11]
        elif 'large' in self.arch: # 24层
            selected_indices = [5, 11, 17, 23]
        else: # 未知时，取最后四层
            selected_indices = list(range(max(0, n - 4), n))
            
        # 提取 patch token (去掉 CLS token)
        feats = [inter[idx][:, 1:] for idx in selected_indices]
        
        # UNETR head 可能需要原始图像尺寸
        out = self.head(feats, x)
        return out

# --- 核心功能函数 ---
def load_decoder_weights(head, ckpt_path):
    """加载 Checkpoint 权重到 head 模型中，保留你原有的复杂映射逻辑"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # 1. 提取 state_dict
    sd = ckpt["state_dict"]

    # 2. 去掉 "module." 前缀
    sd_fixed = {}
    for k, v in sd.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        sd_fixed[nk] = v
    
    # 映射：尝试用 ckpt 中的键后缀匹配 head 的键
    mapped = {}
    ckeys = list(sd.keys())
    hkeys = list(head.state_dict().keys())
    for hk in hkeys:
        for ck in ckeys:
            if ck.endswith(hk):
                mapped[hk] = sd[ck]
                break

    print("mapped keys count:", len(mapped), " / ", len(hkeys))
    
    # # 4. 加载权重
    # missing_keys, unexpected_keys = head.load_state_dict(mapped, strict=False)
    # if missing_keys or unexpected_keys:
    #     print(f"  加载 {Path(ckpt_path).name}:")
    #     if missing_keys:
    #         print(f"    - 缺失的键: {len(missing_keys)}")
    #     if unexpected_keys:
    #         print(f"    - 未预期的键: {len(unexpected_keys)}")
    head.load_state_dict(mapped, strict=True)
    return head

def parse_checkpoint_num(path: Path) -> str:
    """从 pth 文件名中提取数字，如 checkpoint_96_linear.pth -> 96"""
    match = re.search(r'(\d+)', path.stem)
    return match.group(1) if match else path.stem

def main(args):
    """主执行函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # --- 1. 准备 Encoder 和图像预处理 ---
    # Encoder 只需要加载一次
    encoder = models.__dict__[args.arch](
        img_size=[args.input_size],
        patch_size=args.patch_size,
        num_classes=0
    ).to(device).eval()
    
    # 如果有预训练权重，加载它
    utils.load_pretrained_weights(encoder, args.pretrained_weights, 'teacher', args.arch, args.patch_size)
    print("成功加载 Encoder 预训练权重。")

    # 图像预处理
    mean, std = utils.get_stats('Fundus') # 假设是眼底图
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    # 加载并预处理输入图像
    img = Image.open(args.image_path).convert('RGB')
    inp = transform(img).unsqueeze(0).to(device)

    # --- 2. 批量处理 Checkpoint ---
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pth_files = sorted(list(checkpoint_dir.glob("*.pth")))
    if not pth_files:
        print(f"错误: 在 '{checkpoint_dir}' 中没有找到 .pth 文件。")
        return

    print(f"找到 {len(pth_files)} 个 checkpoint 文件，开始批量推理...")
    
    for pth_path in tqdm(pth_files, desc="处理 Checkpoints"):
        # a. 构建 Head
        embed_dim = getattr(encoder, "embed_dim", None)
        head = Unetr_Head(embed_dim=embed_dim, num_classes=args.num_labels, img_dim=args.input_size)
        
        # b. 加载 Head 的权重
        head = load_decoder_weights(head, pth_path)
        
        # c. 组合成完整模型
        model = SegModel(encoder, head, args.arch).to(device).eval()

        # d. 推理
        with torch.no_grad():
            out = model(inp)
            prob = torch.sigmoid(out) if out.shape[1] == 1 else torch.softmax(out, dim=1)[:, 1:, ...]
            prob_np = prob.squeeze().cpu().numpy()

        # e. 后处理并保存
        mask = (prob_np > 0.5).astype(np.uint8) * 255
        
        num = parse_checkpoint_num(pth_path)
        output_filename = f"seg_{num}.png"
        Image.fromarray(mask).save(output_dir / output_filename)

    print(f"\n处理完成！结果已保存至 '{output_dir}'。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量运行分割模型 Checkpoint 进行推理。")
    
    # --- 路径参数 ---
    parser.add_argument('--checkpoint-dir', type=str,default="/root/autodl-tmp/VisionFM/results/single_seg_debug", 
                        help='存放所有 .pth checkpoint 文件的文件夹路径。')
    parser.add_argument('--output-dir', type=str,  default="/root/autodl-tmp/VisionFM/results/seg_pth_outputs",
                        help='存放生成的分割结果图片的文件夹路径。')
    parser.add_argument('--image-path', type=str, 
                        default="/root/autodl-tmp/VisionFM/dataset/ProcessedDatasets/SingleModalSeg/VesselSegmentation/DRIVE/test/images/01.png",
                        help='用于推理的单张输入图片路径。')
    parser.add_argument('--pretrained-weights', type=str, default="/root/autodl-tmp/VisionFM/pretrain_weights/VFM_Fundus_weights.pth",
                        help='(可选) Encoder 的预训练权重路径。')

    # --- 模型参数 (需与训练时保持一致) ---
    parser.add_argument('--arch', type=str, default='vit_base', help='模型架构 (e.g., vit_base)。')
    parser.add_argument('--patch-size', type=int, default=16, help='Patch 大小。')
    parser.add_argument('--input-size', type=int, default=512, help='模型输入尺寸。')
    parser.add_argument('--num-labels', type=int, default=1, help='分割类别数。')
    
    args = parser.parse_args()
    
    main(args)