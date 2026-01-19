CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29503 evaluation/train_cls_decoder_finetune.py \
--name IDRiD_Binary_finetune \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--output_dir ./IDRiDResults \
--data_path ./FundusClassification \
--num_labels 1 \
--batch_size_per_gpu 16 \
--weight_decay 0.1 \
--early_stop_patience 25 \
--finetune \
--backbone_lr_ratio 0.03 \
--epochs 100 \
> train_idrid.log 2>&1 &
