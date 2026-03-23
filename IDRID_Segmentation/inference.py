"""
单张Fundus图像推理脚本
输入: 一张Fundus图像 (RGB)
输出: 微动脉瘤分割掩码
"""

import numpy as np
import cv2
import torch
import os
import sys
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path for imports
sys.path.insert(0, './')

from networks import MSRNet
from lib.pre_processing import my_PreProc
from lib.extract_patches import extract_ordered_overlap, recompone_overlap, paint_border_overlap
from lib.help_functions import load_hdf5, rgb2gray

# ============ 配置 ============
# 图像块参数
patch_height = 96
patch_width = 96
stride_height = 50
stride_width = 50

# 模型路径
check_path = "net.pt7"

# 设备选择
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


def load_image(image_path):
    """加载RGB图像"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    # BGR转RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def preprocess_image(img):
    """预处理图像，转换为模型输入格式"""
    # 转换为float32并归一化到0-1
    img = img.astype(np.float32) / 255.0

    # 转换为4D数组 (1, 3, H, W)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    # 应用预处理 (与训练/测试相同)
    # 注意: my_PreProc会进行灰度转换、归一化、CLAHE、gamma校正
    img = my_PreProc(img)

    return img


def pad_image(img, patch_height, patch_width, stride_height, stride_width):
    """填充图像边界，使其与patch提取兼容"""
    img_padded = paint_border_overlap(img, patch_height, patch_width, stride_height, stride_width)
    return img_padded


def extract_patches(img, patch_height, patch_width, stride_height, stride_width):
    """提取图像块"""
    patches = extract_ordered_overlap(img, patch_height, patch_width, stride_height, stride_width)
    return patches


def load_model(checkpoint_path):
    """加载模型权重"""
    net = MSRNet(out_channels=2)

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 处理DataParallel保存的权重(带有"module."前缀)
    state_dict = checkpoint['net']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 去掉"module."前缀
        else:
            new_state_dict[k] = v

    net.load_state_dict(new_state_dict)

    net.eval()
    net = net.to(device)

    return net


def predict_patches(net, patches, batch_size=128, device='cuda'):
    """批量预测图像块"""
    class TestDataset(Dataset):
        def __init__(self, patches_imgs):
            self.imgs = patches_imgs

        def __len__(self):
            return self.imgs.shape[0]

        def __getitem__(self, idx):
            return torch.from_numpy(self.imgs[idx, ...]).float()

    test_set = TestDataset(patches)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    preds = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = net(inputs)

            # Softmax并取第1通道(微动脉瘤概率)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            outputs = outputs.permute(0, 2, 3, 1)
            shape = list(outputs.shape)
            outputs = outputs[..., 1]  # 取微动脉瘤通道
            outputs = outputs.view(-1, 1, shape[1], shape[2])
            outputs = outputs.data.cpu().numpy()
            preds.append(outputs)

    predictions = np.concatenate(preds, axis=0)
    return predictions


def reconstruct_image(pred_patches, img_height, img_width, patch_height, patch_width, stride_height, stride_width):
    """重建完整图像"""
    pred_imgs = recompone_overlap(pred_patches, img_height, img_width, stride_height, stride_width)
    return pred_imgs


def apply_threshold(pred_img, threshold=0.5):
    """应用阈值分割"""
    pred_binary = pred_img.copy()
    pred_binary[pred_binary >= threshold] = 1
    pred_binary[pred_binary < threshold] = 0
    return pred_binary


def save_result(pred_mask, output_path):
    """保存分割结果"""
    # 确保是uint8格式 (0或255)
    if pred_mask.ndim == 4:
        pred_mask = pred_mask[0, 0]  # 取出第一张图像的第一个通道
    elif pred_mask.ndim == 3:
        pred_mask = pred_mask[0]

    # 转换为0-255的uint8图像
    pred_mask = (pred_mask * 255).astype(np.uint8)

    cv2.imwrite(output_path, pred_mask)
    print(f"结果已保存至: {output_path}")


def inference(image_path, output_path, threshold=0.5):
    """
    推理主函数

    Args:
        image_path: 输入图像路径
        output_path: 输出掩码路径
        threshold: 分割阈值 (默认0.5)
    """
    print(f"\n{'='*50}")
    print(f"开始推理: {image_path}")
    print(f"{'='*50}\n")

    # 1. 加载图像
    print("[1/6] 加载图像...")
    img_rgb = load_image(image_path)
    orig_height, orig_width = img_rgb.shape[:2]
    print(f"    图像尺寸: {orig_width} x {orig_height}")

    # 2. 预处理
    print("[2/6] 预处理图像...")
    img_preproc = preprocess_image(img_rgb)
    print(f"    预处理后形状: {img_preproc.shape}")

    # 3. 填充边界并提取patches
    print("[3/6] 提取图像块...")
    img_padded = pad_image(img_preproc, patch_height, patch_width, stride_height, stride_width)
    padded_height, padded_width = img_padded.shape[2], img_padded.shape[3]
    print(f"    填充后尺寸: {padded_width} x {padded_height}")

    patches = extract_patches(img_padded, patch_height, patch_width, stride_height, stride_width)
    print(f"    提取的patches数量: {patches.shape[0]}")

    # 4. 加载模型
    print("[4/6] 加载模型...")
    checkpoint_path = './checkpoint/' + check_path
    net = load_model(checkpoint_path)
    print(f"    模型加载成功: {checkpoint_path}")

    # 5. 模型推理
    print("[5/6] 模型推理中...")
    predictions = predict_patches(net, patches, device=device)
    print(f"    预测完成，输出形状: {predictions.shape}")

    # 6. 重建图像
    print("[6/6] 重建图像...")
    pred_img = reconstruct_image(predictions, padded_height, padded_width,
                                   patch_height, patch_width, stride_height, stride_width)
    print(f"    重建后形状: {pred_img.shape}")

    # 裁剪回原始尺寸
    pred_img = pred_img[:, :, 0:orig_height, 0:orig_width]
    print(f"    裁剪后形状: {pred_img.shape}")

    # 应用阈值
    pred_binary = apply_threshold(pred_img, threshold)

    # 保存结果
    save_result(pred_binary, output_path)

    # 同时保存概率图(可选)
    prob_path = output_path.replace('.png', '_prob.png')
    prob_img = pred_img[0, 0]
    prob_img = (prob_img * 255).astype(np.uint8)
    cv2.imwrite(prob_path, prob_img)
    print(f"    概率图已保存至: {prob_path}")

    print(f"\n{'='*50}")
    print("推理完成!")
    print(f"{'='*50}\n")

    return pred_img


def main():
    """主函数 - 命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='单张Fundus图像微动脉瘤分割')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入图像路径')
    parser.add_argument('--output', '-o', type=str, default='output_mask.png',
                        help='输出掩码路径 (默认: output_mask.png)')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='分割阈值 (默认: 0.5)')

    args = parser.parse_args()

    # 运行推理
    inference(args.input, args.output, args.threshold)


if __name__ == '__main__':
    main()
