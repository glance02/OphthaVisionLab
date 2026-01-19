CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29502 evaluation/train_cls_decoder_finetune.py \
--name single_cls_260106_Binary_finetune \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--output_dir ./myProcessResults \
--data_path ./dataset/my_process/dataset260106  \
--num_labels 1 \
--batch_size_per_gpu 16 \
--weight_decay 0.05 \
--early_stop_patience 8 \
--finetune \
--backbone_lr_ratio 0.05 \
> train_finetune.log 2>&1 &
