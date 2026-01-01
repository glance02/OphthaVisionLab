# ===== 方案4: 更多层特征 + avgpool + 原学习率 + 早停 =====
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29503 evaluation/train_cls_decoder.py \
--name single_cls_260101 \
--n_last_blocks 4 \
--avgpool_patchtokens 1 \
--pretrained_weights ./pretrained_weights/VFM_Fundus_weights.pth \
--output_dir ./myProcessResults \
--data_path ./dataset/my_process/dataset260101/ \
--num_labels 5 \
--batch_size_per_gpu 64 \
--num_workers 8 \
--val_freq 1 \
--epochs 10 \
--lr 0.001 \
--seed 7 > train_single_cls.log 2>&1 &