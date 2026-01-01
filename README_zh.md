# VisionFM：用于通用眼科人工智能的视觉基础模型

VisionFM 官方实现 - 一个 multimodal multitask vision foundation model，使用来自超过 50 万受试者的 340 万张眼科图像进行预训练，以实现通用眼科人工智能。VisionFM 能够处理八种常见的眼科成像模式，包括眼底摄影、光学相干断层扫描 (OCT)、眼底荧光血管造影 (FFA)、裂隙灯、B 型超声、外眼成像、MRI 和超声生物显微镜 (UBM)，并可应用于解决各种眼科 AI 任务，如眼部疾病识别、疾病进展预测、疾病表型和解剖标志的分割和检测，以及全身生物标志物和疾病预测。VisionFM 的功能可以通过新成像模式的自监督预训练和新临床任务的监督微调进一步扩展，有可能解决各种全球眼科疾病和不同的临床挑战。

## 最新消息

- [2024/11] :tada: 恭喜！VisionFM 已在 [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIoa2300221) 上发表。
- [2024/05] 微调代码已发布，同时发布了在八个公共多类疾病识别数据集上的微调权重

## 引用
如果您认为这个仓库有用，请考虑引用这篇论文：
```text
@article{qiu2024development,
  title={Development and validation of a multimodal multitask vision foundation model for generalist ophthalmic artificial intelligence},
  author={Qiu, Jianing and Wu, Jian and Wei, Hao and Shi, Peilun and Zhang, Minqing and Sun, Yunyun and Li, Lin and Liu, Hanruo and Liu, Hongyi and Hou, Simeng and others},
  journal={NEJM AI},
  volume={1},
  number={12},
  pages={AIoa2300221},
  year={2024},
  publisher={Massachusetts Medical Society}
}
```

## 0. 安装环境

使用 conda 命令创建环境：
```shell
conda create -n vfm python=3.8
conda activate vfm
```

安装依赖：
```shell
git clone https://github.com/ABILab-CUHK/VisionFM.git
cd VisionFM
pip install -r requirements.txt
```
## 1. 微调
如果您想利用我们的权重在您的数据上进行微调，请参考此 [说明](./Fine-tuning/README.md)。

## 2. 预训练
在此步骤中，您可以在自己的数据上预训练自己的 VisionFM 编码器。请按照以下说明开始预训练。

### 2.1. 准备预训练数据集

在我们的研究中，我们使用了 `8` 种模式：`Fundus, OCT, External Eye, UBM, B-Ultrasound, MRI, Silt Lamp, and FFA`。
对于每种模式，例如 Fundus，其数据路径应为 `/xxx/Fundus/`，其中包含所有 Fundus 图像
具有相同或不同的后缀：

```
.
├── /dst_dir/Fundus/
│   ├── 1.jpg
│   ├── 2.png
│   └── ....
├── /dst_dir/OCT/
│   ├── 1.png
│   ├── 2.jpg
│   └── ...
└── ...
 
```

如果您手头没有眼底照片，可以运行以下命令生成随机图像：
```shell
cd evaluation
python random_data.py --task pretrain --dst_dir ../dataset/pretrain_random
```

### 2.2. 预训练 VisionFM 编码器

1. 在 `Fundus` 模式上训练 `vit-base`：

```shell
# 运行以下命令来训练 Fundus 编码器
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29500 main_pretrain.py \
--local-rank=0 \
--data_path ./dataset/pretrain_random \
--modality Fundus \ 
--norm_last_layer true \
--epochs 400 \
--batch_size_per_gpu 12 \
--shared_head true \
--out_dim 8192 \
--output_dir ./results \
--global_crops_scale 0.32 1.0 \
--pred_ratio 0 0.3 \
--pred_ratio_var 0 0.2 \
--name Train_Random_Fundus \
--load_pretrain > train_fundus.log 2>&1 &
# or 
bash train_vitb_fundus.sh # 包含相同的命令

# 注意：默认的 batch size 是 128，batch_size=12 仅用于调试。
```

通过更改模式，可以训练不同的 VisionFM 编码器。

## 3. 为下游任务训练解码器

### 3.1. 下载 VisionFM 的预训练权重
请根据您想要进行研究的模式下载相应的模型权重。

| 模式   | Google Drive                                                                                      |
|------------|---------------------------------------------------------------------------------------------------|
| Fundus     | [Download](https://drive.google.com/file/d/13uWm0a02dCWyARUcrCdHZIcEgRfBmVA4/view?usp=sharing) |
| OCT        | [Download](https://drive.google.com/file/d/1o6E-ine2QLx2pxap-c77u-SU0FjxwypA/view?usp=sharing) |
| FFA        | [Download](https://drive.google.com/file/d/128izBUNV00Ojb9w9Dq3GhBvhWqzU-mla/view?usp=sharing) |
| Ultrasound | [Download](https://drive.google.com/file/d/1IlD0snowxdEVvxmiIBZGR0D9uOcrCT2D/view?usp=sharing) |
| External Eye  | [Download](https://drive.google.com/file/d/16zGHTD4ZcGAYW382kKHBw3TU6D1OtvTD/view?usp=sharing) |
| Silt Lamp   | [Download](https://drive.google.com/file/d/1pemWDkGoZYlqLQ6ooFINktyk8xnv9wY_/view?usp=sharing) |
| MRI        | [Download](https://drive.google.com/file/d/1fcfylnOWhfnZHBAKT9pQPufyS5ZYCXu0/view?usp=sharing) |
| UBM        | [Download](https://drive.google.com/file/d/1q2fVOgFBnWNu1BsXaza1A-OIcCiifNUQ/view?usp=sharing) |


### 3.2.训练分类解码器 [多模态]

`多模态` 意味着解码器是在不同模态上同时训练的。考虑到不同编码器的存在（每种模态都有自己的编码器），
我们采用两阶段流水线：`从不同模态的 VisionFM 编码器中预提取特征` 和 `使用聚合的图像特征训练解码器`：

对于第一步，我们需要根据模态提取图像特征。例如，我们可以通过 Fundus 和 OCT 编码器分别提取 Fundus 和 OCT 特征。
然后，对于第二步，我们可以开始使用从这两种模态中提取的组合特征来训练解码器。

#### 3.2.1. 准备数据集
请将您的数据集组织成以下目录结构（我们称这种目录结构样式为 `vfm`）：
```
.
├── /XXX/FundusClassification/
│   ├── dataset_A/
│   │   ├── training/
│   │   │   ├── 1.png
│   │   │   └── ...
│   │   ├── test/
│   │   │   ├── 2.png
│   │   │   └── ...
│   │   ├── training_labels.txt
│   │   └── test_labels.txt
│   ├── dataset_B/
│   │   └── ....
│   └── ....
├── /XXX/OCTClassification/
│   ├── dataset_A/
│   │   ├── training/
│   │   │   ├── 1.png
│   │   │   └── ...
│   │   ├── test/
│   │   │   ├── 2.png
│   │   │   └── ...
│   │   ├── training_labels.txt
│   │   └── test_labels.txt
│   ├── dataset_B/
│   │   └── ....
│   └── ....
...
```

相应的 `training_labels.txt` 和 `test_labels.txt` 组织如下作为示例：
```
# 类别列表: ['Healthy', 'DR-1', 'DR-2', 'DR-3', 'DR-4', 'DR', 'Glaucoma', 'AMD', 'Cataract', 'Hypertensive Retinopathy', 'Retinal Vein Occlusion', 'Myopia', 'Retinal Detachment']
# 在 training_labels.txt 中
# 注意: 建议使用多标签样式（非 one-hot）来标记，以方便识别标记有多种疾病的图像，同时简化分级和非分级数据的使用。
training/1.jpg;0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
training/2.jpg;0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0
```

您可以下载我们预处理的公共 [数据集](https://drive.google.com/file/d/1QShoYrkhZetF41vmFuf6ds3I1W05YONk/view?usp=drive_link)（包含 IDRiD 和 OCTID 数据集）来开始训练。
解压下载的数据集后，请使用以下结构组织数据集：
```text
./dataset/ProcessedDatasets/MultiModalCls/IDRiD
./dataset/ProcessedDatasets/MultiModalCls/OCTID
```

或者您可以使用以下命令生成随机数据集：
```shell
python evaluation/random_data.py --task multi_cls --dst_dir ./dataset/multi_cls_random
```

#### 3.2.2. 特征提取
以下命令使用预训练的 VisionFM 编码器提取图像特征：

```shell
#cd evaluation
#分别通过 Fundus 和 OCT 编码器提取 Fundus 和 OCT 特征
#CUDA_VISIBLE_DEVICES=0 nohup python evaluation/extract_features.py \
# 提取 Fundus 特征
CUDA_VISIBLE_DEVICES=1,2 nohup python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node=2 --master_port=29503 evaluation/extract_features.py \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--batch_size_per_gpu 768 \
--data_path ./dataset/multi_cls_random/FundusClassificationMulti/ \
--modality Fundus \
--dst_root ./dataset/multi_cls_random/FunClsFeat/ > extract_feats.log 2>&1 &

# 提取 OCT 特征
CUDA_VISIBLE_DEVICES=1,2 nohup python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node=2 --master_port=29503 evaluation/extract_features.py \
--pretrained_weights ./pretrain_weights/VFM_OCT_weights.pth \
--batch_size_per_gpu 768 \
--data_path ./dataset/multi_cls_random/OCTClassificationMulti/ \
--modality OCT \
--dst_root ./dataset/multi_cls_random/OCTClsFeat/ > extract_feats.log 2>&1 &

# 对于提供的预处理数据集，您应该设置以下参数：
--data_path ./dataset/ProcessedDatasets/MultiModalCls/FundusClassificationMulti/
--dst_root ./dataset/ProcessedDatasets/MultiModalCls/FunClsFeat/
# 或者
--data_path ./dataset/ProcessedDatasets/MultiModalCls/OCTClassificationMulti/
--dst_root ./dataset/ProcessedDatasets/MultiModalCls/OCTClsFeat/
```

#### 3.2.3. 基于提取的多模态特征训练解码器
然后，通过以下命令训练分类解码器：
```shell
#cd evaluation
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 evaluation/train_cls_multi_decoder.py \
--name train_debug \ 
--output_dir ./results \
--datasets FunClsFeat OCTClsFeat \
--data_path ./dataset/multi_cls_random/ \
--batch_size_per_gpu 8192 > train_cls_multi.log 2>&1 &
```

### 3.3. 训练分类解码器 [单模态]

此任务主要关注单模态，例如基于 Fundus 的 DR 分级任务。

#### 3.3.1. 准备数据集
请将您的数据集组织成以下目录结构（我们称这种目录结构样式为 `vfm`）：
```
.
├── /XXX/FundusClassification/ # 所有数据集应为同一任务，具有相同的类别定义
│   ├── dataset_A/
│   │   ├── training/
│   │   │   ├── 1.png
│   │   │   └── ...
│   │   ├── test/
│   │   │   ├── 2.png
│   │   │   └── ...
│   │   ├── training_labels.txt
│   │   └── test_labels.txt
│   ├── dataset_B/
│   │   └── ....
│   └── ....
├── /XXX/OCTClassification/
│   ├── dataset_A/
│   │   ├── training/
│   │   │   ├── 1.png
│   │   │   └── ...
│   │   ├── test/
│   │   │   ├── 2.png
│   │   │   └── ...
│   │   ├── training_labels.txt
│   │   └── test_labels.txt
│   ├── dataset_B/
│   │   └── ....
│   └── ....
...
```

`training_labels.txt` 和 `test_labels.txt` 包含图像路径及其对应的标签：
```text
# 在 training_labels.txt 中
training/1.jpg;1
training/2.jpg;2
```

您可以下载我们预处理的 [数据集](https://drive.google.com/file/d/1QShoYrkhZetF41vmFuf6ds3I1W05YONk/view?usp=drive_link)（包含处理后的 IDRiD 和 OCTID 以开始训练。
解压下载的数据集后，您应该使用以下结构组织数据集：
```text
./dataset/ProcessedDatasets/SingleModalCls/FundusClassification/IDRiD 
./dataset/ProcessedDatasets/SingleModalCls/OCTClassification/OCTID
```

或者您可以使用以下命令生成随机数据集：
```shell
python evaluation/random_data.py --task single_cls --dst_dir ./dataset/single_cls_random # 用于 DR 分级任务
```

除了提到的目录结构（称为 vfm），您还可以使用以下目录结构（ImageNet 格式）：
```test
├── 数据文件夹
    ├──train
        ├──class_a
        ├──class_b
        ├──class_c
    ├──val
        ├──class_a
        ├──class_b
        ├──class_c
    ├──test
        ├──class_a
        ├──class_b
        ├──class_c
```
如果您的数据集以此结构组织，请设置 `--dataset_format ImageNet`。


#### 3.3.2. 训练解码器
然后，通过以下命令为分类任务训练解码器：
```shell
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 evaluation/train_cls_decoder.py \
--name single_cls_debug \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--output_dir ./results \
--data_path ./dataset/single_cls_random/FundusClassification/ \
--num_labels 5 \
--batch_size_per_gpu 32 > train_single_cls.log 2>&1 &

#注意：对于二分类任务，请设置 --num_labels 1

# 对于处理后的数据集：基于 Fundus 的 DR 分级
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 evaluation/train_cls_decoder.py \
--name single_cls_debug \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--output_dir ./results \ 
--data_path ./dataset/ProcessedDatasets/SingleModalCls/FundusClassification  \
--num_labels 5 \
--batch_size_per_gpu 32 > train_single_cls.log 2>&1 &

# 对于 ImageNet 格式的数据集：基于 Fundus 的 DR 分级
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 evaluation/train_cls_decoder.py \
--name single_cls_debug \
--dataset_format ImageNet \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--output_dir ./results \
--data_path ./dataset/xxx/  \
--num_labels 5 \
--batch_size_per_gpu 32 > train_single_cls.log 2>&1 &

# 对于处理后的数据集：基于 OCT 的二分类（Health, DR）
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 evaluation/train_cls_decoder.py \
--name single_cls_debug \
--pretrained_weights ./pretrain_weights/VFM_OCT_weights.pth \
--output_dir ./results \
--data_path ./dataset/ProcessedDatasets/SingleModalCls/OCTClassification  \
--num_labels 1 \
--modality OCT \
--batch_size_per_gpu 32 > train_single_cls.log 2>&1 &

# 在解码器训练完成后，您可以使用以下命令评估训练好的解码器
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 evaluation/train_cls_decoder.py \
--name single_cls_debug \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--output_dir ./results \
--data_path ./dataset/ProcessedDatasets/SingleModalCls/FundusClassification  \
--num_labels 5 \
--load_from ./results/single_cls_debug/checkpoint_teacher_linear.pth\
--test \
--batch_size_per_gpu 32 > train_single_cls.log 2>&1 &


# 您也可以通过添加两个额外参数来加载 RETFound 权重：
--arch vit_large \
--checkpoint_key model \
```

### 3.4.训练分割解码器 [单模态]
在分割任务中，我们为不同的任务和模态训练不同的解码器。

#### 3.4.1. 准备数据集
请将您的数据集组织成以下目录结构（我们称这种目录结构样式为 `vfm`）：
```
├── /dst_dir/VesselSegmentation/ # 所有数据集应为同一任务，例如，眼底血管分割
│   ├── dataset_A/
│   │   ├── training/
│   │   │   ├── images/
│   │   │   │   ├── 1.png
│   │   │   │   └── ...
│   │   │   └── labels/
│   │   │       ├── 1.png
│   │   │       └── ...
│   │   └── test/
│   │       └── ...
│   ├── dataset_B/
│   │   └── ...
│   └── ...
└── ...
labels 目录中图像的像素值范围应为 [0, C-1]，其中 C 是类别数。
```
您可以下载我们预处理的公共 [数据集](https://drive.google.com/file/d/1QShoYrkhZetF41vmFuf6ds3I1W05YONk/view?usp=drive_link)（包含 DRIVE 数据集用于血管分割）来开始训练。
解压下载的数据集后，请将数据集组织成以下结构：
```text
./dataset/ProcessedDatasets/SingleModalSeg/VesselSegmentation/DRIVE 
```
或者您可以使用以下命令生成随机数据集：
```shell
python evaluation/random_data.py --task segmentation --dst_dir ./dataset/seg_random
```

#### 3.4.2. 训练解码器
然后，通过以下命令为分割任务训练解码器：
```shell
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29509 evaluation/train_seg_decoder.py \
--name single_seg_debug \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--input_size 512 \
--modality Fundus \
--num_labels 5 \
--output_dir ./results \
--data_path ./dataset/seg_random/VesselSegmentation/ \
--batch_size_per_gpu 5 > train_seg.log 2>&1 &

# 对于提供的预处理数据集，您应该设置以下参数：
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29509 evaluation/train_seg_decoder.py \
--name single_seg_debug \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--input_size 512 \
--modality Fundus \
--num_labels 1 \
--output_dir ./results \
--data_path ./dataset/ProcessedDatasets/SingleModalSeg/VesselSegmentation/ \
--batch_size_per_gpu 5 > train_seg.log 2>&1 &
```


### 3.5. 训练地标检测解码器 [单模态]
在此任务中，我们训练一个解码器来检测 UBM 图像上的前房角 (ACA) 地标。

#### 3.5.1. 准备数据集
请将数据集组织成与分割任务相同的目录结构（标签的后缀应为 .npy）。

或者您可以使用以下命令生成随机数据集：
```shell
python evaluation/random_data.py --task landmark --dst_dir ./dataset/landmark_random
```

#### 3.5.2. 训练解码器
然后，使用以下命令为地标检测任务训练解码器：
```shell
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29508 evaluation/train_landmark_decoder.py \
--name train_landmark \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--output_dir ./results \
--data_path ./dataset/landmark_random/LandmarkDetection \
--batch_size_per_gpu 32  > train_landmark.log 2>&1 &
```

### 3.6. 训练生物标志物预测解码器 [多模态]
在我们的实验中，我们在 Fundus 和 External 图像上训练解码器以预测生物标志物。

#### 3.6.1. 准备数据集
请将数据集组织成以下目录结构（我们称这种目录结构样式为 `vfm`），这与分类任务相同：
```
.
├── /XXX/FundusRegression/
│   ├── dataset_A/
│   │   ├── training/
│   │   │   ├── 1.jpg
│   │   │   └── ...
│   │   ├── test/
│   │   │   ├── 2.jpg
│   │   │   └── ...
│   │   ├── training_labels.txt
│   │   └── test_labels.txt
│   ├── dataset_B/
│   │   └── ....
│   └── ....
├── /XXX/ExternalRegression/
│   ├── dataset_A/
│   │   ├── training/
│   │   │   ├── 1.jpg
│   │   │   └── ...
│   │   ├── test/
│   │   │   ├── 2.jpg
│   │   │   └── ...
│   │   ├── training_labels.txt
│   │   └── test_labels.txt
│   ├── dataset_B/
│   │   └── ....
│   └── ....
...
```
相应的 `training_labels.txt` 和 `test_labels.txt` 组织如下（38 个值）：
```
# 在 training_labels.txt 中
training/1.jpg;38.8,2.5,37.0,11.4,8.9,0.05,0.4,0.13,1.1,46.6,157.0,3.87,31.3,32.8,337.0,..., 3.4,4.13,0.93,2.62,3.17
```
或者您可以使用以下命令生成随机数据集：
```shell
python evaluation/random_data.py --task metric_reg --dst_dir ./dataset/metric_random
```

#### 3.6.2. 提取特征
首先，使用以下命令提取图像特征：
```shell
# 提取 Fundus 图像的特征
CUDA_VISIBLE_DEVICES=0,1 nohup python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=29503 evaluation/extract_features.py \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--batch_size_per_gpu 768 \
--data_path ./dataset/metric_random/FundusRegression \
--modality Fundus \
--dst_root ./dataset/metric_random/FunRegFeat/ > extract_feats.log 2>&1

#提取 External 图像的特征
CUDA_VISIBLE_DEVICES=0,1 nohup python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=29503 evaluation/extract_features.py \
--pretrained_weights ./pretrain_weights/VFM_External_weights.pth \
--batch_size_per_gpu 768 \
--data_path ./dataset/metric_random/ExternalRegression \
--modality External \
--dst_root ./dataset/metric_random/ExternalRegFeat/ > extract_feats.log 2>&1
```

#### 3.6.3. 使用提取的特征训练解码器
然后，使用以下命令训练生物标志物预测解码器：
```shell
CUDA_VISIBLE_DEVICES=0,1 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29504 evaluation/train_metric_reg_multi_decoder.py  \
--name train_metric_reg_multi \
--output_dir ./results \
--datasets FunRegFeat ExternalRegFeat \
--data_path ./dataset/metric_random/ \
--batch_size_per_gpu 4096 > train_metric_reg_multi.log 2>&1 &

```


### 3.7. 训练青光眼进展预测解码器 [单模态]

#### 3.7.1 准备数据
请将数据组织成以下结构：
```

├── /dataset/glaucoma_forecasting/
│   ├── training/
│   │   ├── 1.jpg
│   │   └── ...
│   ├── test/
│   │   ├── 1.jpg
│   │   └── ...
│   ├── training_labels.txt
│   └── test_labels.txt

```
相应的 `training_labels.txt` 和 `test_labels.txt` 组织如下 (path/to/image, label, time interval)：
```
# 在 training_labels.txt 中
./dataset/glaucoma_forecasting/training/1.jpg, 0, 309
# 在 test_labels.txt 中
./dataset/glaucoma_forecasting/test/1.jpg, 0, 690
```


#### 3.7.2 训练解码器
青光眼预测解码器可以使用以下命令进行训练：

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch evaluation/train_forecasting_decoder.py  --data_path ./dataset/glaucoma_forecasting/ --pretrained_weights /path/to/checkpoint.pth --n_last_blocks 4 --avgpool_patchtokens 1 --input_size 224 --checkpoint_key teacher --output_dir ./results/glaucoma_forecasting --num_labels 2 --lr 0.001 --batch_size_per_gpu 128 --epochs 100
```


## 4. VisionFM 私有评估数据和合成图像

可以访问合成图像和我们私有评估数据的一个子集。请下载 [数据请求和协议表](resource/visionfm_dataset_agreement_form.pdf)，签名后发送至 visionfm-datasets@googlegroups.com

## 许可证
本项目根据仅允许用于研究和教育目的的许可证发布。该模型的商业使用是不允许的。使用该模型时，请确保遵守此许可证的条款。有关更多信息，请参阅本仓库中包含的 LICENSE 文件。