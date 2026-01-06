CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 evaluation/train_cls_decoder.py \
--name single_cls_260106_Binary \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--output_dir ./myProcessResults \
--data_path ./dataset/my_process/dataset260106  \
--num_labels 1 \
--batch_size_per_gpu 64 \
> train_single_cls.log 2>&1 &