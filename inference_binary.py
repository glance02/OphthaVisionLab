"""
二分类模型推理脚本
支持单张图像或批量图像的预测
"""
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from PIL import Image
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils
import models.vision_transformer as vit
from models.head import ClsHead
from torchvision import transforms


def load_model(checkpoint_path, pretrained_weights, num_labels=1, finetune=True, n_last_blocks=4):
    """
    加载训练好的模型

    Args:
        checkpoint_path: 训练好的checkpoint路径
        pretrained_weights: 预训练权重路径
        num_labels: 分类数量（二分类为1）
        finetune: 是否为fine-tuning模型
        n_last_blocks: 使用的最后几层块数量（默认4）
    """
    # 加载ViT backbone
    model = vit.__dict__['vit_base'](
        img_size=[224],
        patch_size=16,
        num_classes=0,
        use_mean_pooling=False,
    )
    utils.load_pretrained_weights(model, pretrained_weights, "teacher", "vit_base", 16)
    model = model.cuda()

    # 加载分类头（注意embed_dim要乘以n_last_blocks）
    embed_dim = model.embed_dim
    linear_classifier = ClsHead(
        embed_dim=embed_dim * n_last_blocks,
        num_classes=num_labels,
    )
    linear_classifier = linear_classifier.cuda()

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
        print(f"✓ 分类器权重加载成功")

    # 如果是fine-tuning模型，加载backbone权重
    if finetune and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ Fine-tuning backbone权重加载成功")
    else:
        print(f"✓ 使用预训练backbone（冻结模式）")

    model = model.cuda()
    model.eval()
    linear_classifier.eval()

    return model, linear_classifier


def preprocess_image(image_path, input_size=224):
    """预处理图像"""
    mean, std = utils.get_stats('Fundus')

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]
    return image_tensor


def predict(model, linear_classifier, image_tensor, finetune=True, n_last_blocks=4, avg_pool=0):
    """
    对图像进行预测

    Args:
        model: ViT backbone模型
        linear_classifier: 分类器
        image_tensor: 输入图像张量
        finetune: 是否为fine-tuning模式
        n_last_blocks: 使用的最后几层块数量
        avg_pool: patch token平均池化模式（0=只用CLS, 1=只用patch tokens）

    Returns:
        logit: 原始输出logit
        prob: 转换为概率（0-1之间）
        pred_class: 预测类别（0或1）
    """
    image_tensor = image_tensor.cuda()

    with torch.no_grad():
        # 提取特征（获取最后n层）
        if finetune:
            intermediate_output = model.get_intermediate_layers(image_tensor, n=n_last_blocks)
        else:
            with torch.no_grad():
                intermediate_output = model.get_intermediate_layers(image_tensor, n=n_last_blocks)

        # 拼接多层的CLS tokens
        if avg_pool == 0:
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)  # [1, embed_dim*n]
        else:
            output = torch.mean(intermediate_output[-1][:, 1:], dim=1)  # [1, embed_dim]

        # 通过分类头
        logit = linear_classifier(output).squeeze()  # 标量
        prob = torch.sigmoid(logit).item()  # 转换为概率
        pred_class = 1 if prob > 0.5 else 0

    return logit, prob, pred_class


def main():
    parser = argparse.ArgumentParser(description='二分类模型推理')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='训练好的checkpoint路径')
    parser.add_argument('--pretrained_weights', type=str,
                        default='./pretrain_weights/VFM_Fundus_weights.pth',
                        help='预训练权重路径')
    parser.add_argument('--image', type=str,
                        help='单张图像路径')
    parser.add_argument('--image_dir', type=str,
                        help='图像目录路径（批量预测）')
    parser.add_argument('--output_file', type=str, default='predictions.txt',
                        help='预测结果输出文件')
    parser.add_argument('--finetune', action='store_true',
                        help='是否使用fine-tuning模型')
    args = parser.parse_args()

    # 加载模型
    print(f"正在加载模型: {args.checkpoint}")
    model, linear_classifier = load_model(
        args.checkpoint,
        args.pretrained_weights,
        num_labels=1,
        finetune=args.finetune
    )

    # 单张图像预测
    if args.image:
        print(f"\n{'='*50}")
        print(f"预测图像: {args.image}")
        print(f"{'='*50}")

        image_tensor = preprocess_image(args.image)
        logit, prob, pred_class = predict(model, linear_classifier, image_tensor, args.finetune)

        print(f"Logit: {logit:.4f}")
        print(f"Probability: {prob:.4f}")
        print(f"Predicted Class: {pred_class} ({'Positive' if pred_class == 1 else 'Healthy'})")

    # 批量预测
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))

        print(f"\n找到 {len(image_files)} 张图像")
        print(f"正在预测...\n")

        results = []
        for img_path in image_files:
            image_tensor = preprocess_image(str(img_path))
            logit, prob, pred_class = predict(model, linear_classifier, image_tensor, args.finetune)

            result = {
                'image': str(img_path),
                'logit': logit,
                'probability': prob,
                'prediction': pred_class
            }
            results.append(result)

            print(f"{img_path.name:30s} | Prob: {prob:.4f} | Class: {pred_class}")

        # 保存结果
        with open(args.output_file, 'w') as f:
            for r in results:
                f.write(f"{r['image']}\t{r['logit']:.4f}\t{r['probability']:.4f}\t{r['prediction']}\n")
        print(f"\n结果已保存到: {args.output_file}")

    else:
        print("请指定 --image 或 --image_dir 参数")


if __name__ == '__main__':
    main()
