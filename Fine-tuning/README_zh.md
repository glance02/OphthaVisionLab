# 微调 VisionFM 教程

本教程介绍如何对 VisionFM 进行多类别疾病识别的微调。

## 数据集下载

我们以 RETFound 提供的数据集为例。请从[官方 github 页面](https://github.com/rmaphoh/RETFound_MAE/blob/main/BENCHMARK.md)下载。解压后建议保持原有的数据结构。

## 在 PAPILA 数据集上微调 VisionFM

### 使用训练集微调并用验证集选择最佳 checkpoint

执行以下命令将在 PAPILA 数据集上微调 VisionFM。请将 `--pretrained_weights` 替换为你本地下载的 VisionFM fundus 预训练权重路径（可从[这里](https://drive.google.com/file/d/13uWm0a02dCWyARUcrCdHZIcEgRfBmVA4/view?usp=sharing)下载），`--data_path` 替换为 PAPILA 数据集的本地路径。

```shell
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=20030 finetune_visionfm_for_multiclass_classification.py --pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth --arch vit_base --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/PAPILA_FT_VisionFM_Val --data_path ./data/PAPILA --task PAPILA_FT_VisionFM_Val --modality Fundus --num_workers 4 --batch_size_per_gpu 128 --num_labels 3 --extra 10 > PAPILA_FT_VisionFM_Val.log 2>&1 &
```

代码会自动训练 100 个 epoch。训练完成后，验证集上的 AUROC 和 AUPR 结果会保存在 `PAPILA_FT_VisionFM_Val.log` 文件中：

```shell
cat PAPILA_FT_VisionFM_Val.log
```

最佳微调权重（`checkpoint_best_finetune.pth`）会保存在你指定的输出目录下。

### 在测试集上评估

以下命令会加载微调后的权重并在测试集上评估。请将 `--pretrained_weights` 替换为你微调后权重的路径。

```shell
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=20030 inference_visionfm_for_multiclass_classification.py --pretrained_weights ./results/PAPILA_FT_VisionFM_Val/checkpoint_best_finetune.pth --arch vit_base --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/PAPILA_FT_VisionFM_test --data_path ./data/PAPILA --task PAPILA_FT_VisionFM_test --modality Fundus --num_workers 4 --batch_size_per_gpu 128 --num_labels 3 --extra 10 > PAPILA_FT_VisionFM_test.log 2>&1 &
```

测试集上的 AUROC 和 AUPR 结果会保存在 `PAPILA_FT_VisionFM_test.log` 文件中：

```shell
cat PAPILA_FT_VisionFM_test.log
```

### 测试官方微调权重

我们也提供了微调好的权重，便于复现。请从[这里](https://drive.google.com/file/d/1eI77YhiWbgnxOR35pmqaGD70rsJawz3d/view?usp=sharing)下载，并将 `--pretrained_weights` 替换为下载的 `checkpoint_papila.pth` 路径。

```shell
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=20030 inference_visionfm_for_multiclass_classification.py --pretrained_weights path/to/checkpoint_papila.pth --arch vit_base --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/PAPILA_FT_VisionFM_test --data_path ./data/PAPILA --task PAPILA_FT_VisionFM_test --modality Fundus --num_workers 4 --batch_size_per_gpu 128 --num_labels 3 --extra 10 > PAPILA_FT_VisionFM_test.log 2>&1 &
```

## 其余多分类数据集的微调

其余 7 个数据集的微调方法完全相同。请确保数据集路径和输出目录正确，并为每个数据集指定不同的日志文件名。注意 `--num_labels` 需根据数据集调整。OCTID 数据集请将 `--modality` 改为 `OCT`，并从[这里](https://drive.google.com/file/d/1o6E-ine2QLx2pxap-c77u-SU0FjxwypA/view?usp=sharing)下载 OCT 预训练权重。IDRiD、MESSIDOR2、Kaggle APTOS-2019 建议微调 5 个 epoch 即可，其他数据集建议 100 个 epoch。

你也可以直接用下表命令微调和测试 PAPILA 及其余 7 个数据集（注意权重下载链接在表格最右侧，部分屏幕需左右滑动查看）：

| 数据集   | 微调命令             | 测试命令 | 官方微调权重 |
|------------|-------------------------|---------------|-------------------|
| IDRiD     | ```CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=20030 finetune_visionfm_for_multiclass_classification.py --pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth --arch vit_base --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/IDRiD_data_FT_VisionFM_Val --data_path ./data/IDRiD_data --task IDRiD_data_FT_VisionFM_Val --modality Fundus --num_workers 4 --batch_size_per_gpu 128 --num_labels 5 --epoch 5 --extra 10 > IDRiD_data_FT_VisionFM_Val.log 2>&1 &``` | ```CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=20030 inference_visionfm_for_multiclass_classification.py --pretrained_weights ./results/IDRiD_data_FT_VisionFM_Val/checkpoint_best_finetune.pth --arch vit_base --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/IDRiD_data_FT_VisionFM_test --data_path ./data/IDRiD_data --task IDRiD_data_FT_VisionFM_test --modality Fundus --num_workers 4 --batch_size_per_gpu 128 --num_labels 5 --extra 10 > IDRiD_data_FT_VisionFM_test.log 2>&1 &``` | [权重](https://drive.google.com/file/d/1_svIZnnnVJAJSBltutzGA_qE-tWEnLig/view?usp=sharing)  |
| MESSIDOR2 | ```CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=20030 finetune_visionfm_for_multiclass_classification.py --pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth --arch vit_base --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/MESSIDOR2_FT_VisionFM_Val --data_path ./data/MESSIDOR2 --task MESSIDOR2_FT_VisionFM_Val --modality Fundus --num_workers 4 --batch_size_per_gpu 128 --num_labels 5 --epoch 5 --extra 10 > MESSIDOR2_FT_VisionFM_Val.log 2>&1 &``` | ```CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=20030 inference_visionfm_for_multiclass_classification.py --pretrained_weights ./results/MESSIDOR2_FT_VisionFM_Val/checkpoint_best_finetune.pth --arch vit_base --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/MESSIDOR2_FT_VisionFM_test --data_path ./data/MESSIDOR2 --task MESSIDOR2_FT_VisionFM_test --modality Fundus --num_workers 4 --batch_size_per_gpu 128 --num_labels 5 --extra 10 > MESSIDOR2_FT_VisionFM_test.log 2>&1 &``` | [权重](https://drive.google.com/file/d/1wo7MW5dSLLcbhmOhGvB8J1rZ-uC4f7jY/view?usp=sharing)  |
| APTOS-2019| ```CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=20030 finetune_visionfm_for_multiclass_classification.py --pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth --arch vit_base --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/APTOS2019_FT_VisionFM_Val --data_path ./data/APTOS2019 --task APTOS2019_FT_VisionFM_Val --modality Fundus --num_workers 4 --batch_size_per_gpu 128 --num_labels 5 --epoch 5 --extra 10 > APTOS2019_FT_VisionFM_Val.log 2>&1 &``` | ```CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=20030 inference_visionfm_for_multiclass_classification.py --pretrained_weights ./results/APTOS2019_FT_VisionFM_Val/checkpoint_best_finetune.pth --arch vit_base --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/APTOS2019_FT_VisionFM_test --data_path ./data/APTOS2019 --task APTOS2019_FT_VisionFM_test --modality Fundus --num_workers 4 --batch_size_per_gpu 128 --num_labels 5 --extra 10 > APTOS2019_FT_VisionFM_test.log 2>&1 &``` | [权重](https://drive.google.com/file/d/1jRDJyFNaw_NoXZtROU2KTiJI458sk3S8/view?usp=sharing)   |
| PAPILA    | ```CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=20030 finetune_visionfm_for_multiclass_classification.py --pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth --arch vit_base --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/PAPILA_FT_VisionFM_Val --data_path ./data/PAPILA --task PAPILA_FT_VisionFM_Val --modality Fundus --num_workers 4 --batch_size_per_gpu 128 --num_labels 3 --extra 10 > PAPILA_FT_VisionFM_Val.log 2>&1 &``` | ```CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=20030 inference_visionfm_for_multiclass_classification.py --pretrained_weights ./results/PAPILA_FT_VisionFM_Val/checkpoint_best_finetune.pth --arch vit_base --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/PAPILA_FT_VisionFM_test --data_path ./data/PAPILA --task PAPILA_FT_VisionFM_test --modality Fundus --num_workers 4 --batch_size_per_gpu 128 --num_labels 3 --extra 10 > PAPILA_FT_VisionFM_test.log 2>&1 &``` | [权重](https://drive.google.com/file/d/1eI77YhiWbgnxOR35pmqaGD70rsJawz3d/view?usp=sharing)  |
| Glaucoma Fundus | ```CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=20030 finetune_visionfm_for_multiclass_classification.py --pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth --arch vit_base --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/Glaucoma_fundus_FT_VisionFM_Val --data_path ./data/Glaucoma_fundus --task Glaucoma_fundus_FT_VisionFM_Val --modality Fundus --num_workers 4 --batch_size_per_gpu 128 --num_labels 3 --extra 10 > Glaucoma_fundus_FT_VisionFM_Val.log 2>&1 &``` | ```CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=20030 inference_visionfm_for_multiclass_classification.py --pretrained_weights ./results/Glaucoma_fundus_FT_VisionFM_Val/checkpoint_best_finetune.pth --arch vit_base --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/Glaucoma_fundus_FT_VisionFM_test --data_path ./data/Glaucoma_fundus --task Glaucoma_fundus_FT_VisionFM_test --modality Fundus --num_workers 4 --batch_size_per_gpu 128 --num_labels 3 --extra 10 > Glaucoma_fundus_FT_VisionFM_test.log 2>&1 &``` | [权重](https://drive.google.com/file/d/1RYEI2cZF4mnJ9os9KZYi4-jwpgrBy20g/view?usp=sharing)  |
| JSIEC  | ```CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=20030 finetune_visionfm_for_multiclass_classification.py --pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth --arch vit_base --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/JSIEC_FT_VisionFM_Val --data_path ./data/JSIEC --task JSIEC_FT_VisionFM_Val --modality Fundus --num_workers 4 --batch_size_per_gpu 128 --num_labels 39 --extra 10 > JSIEC_FT_VisionFM_Val.log 2>&1 &``` | ```CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=20030 inference_visionfm_for_multiclass_classification.py --pretrained_weights ./results/JSIEC_FT_VisionFM_Val/checkpoint_best_finetune.pth --arch vit_base --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/JSIEC_FT_VisionFM_test --data_path ./data/JSIEC --task JSIEC_FT_VisionFM_test --modality Fundus --num_workers 4 --batch_size_per_gpu 128 --num_labels 39 --extra 10 > JSIEC_FT_VisionFM_test.log 2>&1 &```  | [权重](https://drive.google.com/file/d/1I2Fy7a22BRcBahql6ML3EdCR7bLLeb4u/view?usp=sharing)  |
| Retina   | ```CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=20030 finetune_visionfm_for_multiclass_classification.py --pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth --arch vit_base --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/Retina_FT_VisionFM_Val --data_path ./data/Retina --task Retina_FT_VisionFM_Val --modality Fundus --num_workers 4 --batch_size_per_gpu 128 --num_labels 4 --extra 10 > Retina_FT_VisionFM_Val.log 2>&1 &``` | ```CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=20030 inference_visionfm_for_multiclass_classification.py --pretrained_weights ./results/Retina_FT_VisionFM_Val/checkpoint_best_finetune.pth --arch vit_base --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/Retina_FT_VisionFM_test --data_path ./data/Retina --task Retina_FT_VisionFM_test --modality Fundus --num_workers 4 --batch_size_per_gpu 128 --num_labels 4 --extra 10 > Retina_FT_VisionFM_test.log 2>&1 &``` | [权重](https://drive.google.com/file/d/1oSwjhD6hwASGPNam_x7X0Wx2V9nahw6U/view?usp=sharing)  |
| OCTID        | ```CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=20030 finetune_visionfm_for_multiclass_classification.py --pretrained_weights ./pretrain_weights/VFM_OCT_weights.pth --arch vit_base --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/OCTID_FT_VisionFM_Val --data_path ./data/OCTID --task OCTID_FT_VisionFM_Val --modality OCT --num_workers 4 --batch_size_per_gpu 128 --num_labels 5 --extra 10 > OCTID_FT_VisionFM_Val.log 2>&1 &``` | ```CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=20030 inference_visionfm_for_multiclass_classification.py --pretrained_weights ./results/OCTID_FT_VisionFM_Val/checkpoint_best_finetune.pth --arch vit_base --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/OCTID_FT_VisionFM_test --data_path ./data/OCTID --task OCTID_FT_VisionFM_test --modality OCT --num_workers 4 --batch_size_per_gpu 128 --num_labels 5 --extra 10 > OCTID_FT_VisionFM_test.log 2>&1 &``` | [权重](https://drive.google.com/file/d/16SZjt_DBWemDgJid_9uTCSUIYhYGHZI8/view?usp=sharing)  |
