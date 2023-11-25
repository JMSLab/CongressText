#!/bin/bash

cd ../tools

# python convert_prima_to_coco.py \
#     --prima_datapath ../data/prima \
#     --anno_savepath ../data/prima/annotations.json 

# MASK
# python train_net.py \
#     --dataset_name          ocr1 \
#     --json_annotation_train ../../test-train/train.json \
#     --image_path_train      ../.. \
#     --json_annotation_val   ../../test-train/test.json \
#     --image_path_val        ../.. \
#     --config-file           ../configs/prima/mask_rcnn_R_50_FPN_3x.yaml \
#     OUTPUT_DIR  ../outputs/mask_rcnn_R_50_FPN_3x/ \
#     SOLVER.IMS_PER_BATCH 2 
    
python train_net.py \
    --dataset_name          ocr1 \
    --json_annotation_train ../../../datastore/scrape/cr-label/test-train/train.json \
    --image_path_train      ../../../datastore/scrape/cr-label  \
    --json_annotation_val   ../../../datastore/scrape/cr-label/test-train/test.json \
    --image_path_val        ../../../datastore/scrape/cr-label  \
    --config-file           ../configs/prima/fast_rcnn_R_50_FPN_3x.yaml \
    --resume                \
    OUTPUT_DIR  ../../../datastore/layout-model-training/outputs/fast_rcnn_R_50_FPN_3x_batch3_manual/ \
    SOLVER.IMS_PER_BATCH 2 
