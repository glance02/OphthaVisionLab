python tools/seg_pth_use.py \
    --checkpoint results/single_seg_debug/checkpoint_108_linear.pth \
    --pretrained-weights pretrain_weights/VFM_Fundus_weights.pth \
    --image /root/autodl-tmp/VisionFM/dataset/ProcessedDatasets/SingleModalCls/FundusClassification/IDRiD/test/IDRiD_002.jpg \
    --output 1.png