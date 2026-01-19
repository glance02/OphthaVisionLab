# 测试dataset260106模型
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29502 evaluation/train_cls_decoder_finetune.py \
--name test_260106_model \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--output_dir ./myProcessResults \
--data_path ./dataset/my_process/dataset260106 \
--num_labels 1 \
--batch_size_per_gpu 16 \
--finetune \
--test \
--load_from ./myProcessResults/single_cls_260106_Binary_finetune/checkpoint_teacher_linear.pth
