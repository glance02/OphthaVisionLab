CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 evaluation/train_cls_decoder.py \
--name single_cls \
--pretrained_weights ./backend/pretrain_weights/VFM_Fundus_weights.pth \
--output_dir ./backend/checkpoints \
--data_path dataset/SingleModalCls/myprocess \
--num_labels 1 \
--load_from ./backend/checkpoints/single_cls/checkpoint_teacher_linear.pth \
--test \
--batch_size_per_gpu 32 > single_cls_test.log 2>&1 &