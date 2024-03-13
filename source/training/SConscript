Import('*')

# clean JSON for training
target = ['#output/scrape/cr-label/sconscript.log',
          '#output/scrape/cr-label/coco_format.json']
source = ['#source/training/update_coco_and_paths.py',
          '#output/scrape/cr-label/output.manual.batch3.json'  # annotations data, sensitive to batch number
          ]
env.Python(target, source)

# split data into training and test set
target = ['#output/scrape/cr-label/test-train/sconscript.log',
          '#output/scrape/cr-label/test-train/train.json',
          '#output/scrape/cr-label/test-train/test.json']
source = ['#source/training/utils/cocosplit.py']
batch3_files = Glob('#output/scrape/cr-label/batch3/*')  # raw data, sensitive to batch number
source.extend(batch3_files)
arguments = '--split-ratio 0.85  --annotation-path #output/scrape/cr-label/coco_format.json  --train #output/scrape/cr-label/test-train/train.json --test #output/scrape/cr-label/test-train/test.json'          
command = f'python {source[0]} {arguments}'          
env.Command(target, source, command)

# train model
target = Glob('#output/layout-model-training/outputs/fast_rcnn_R_50_FPN_3x_batch3_manual/*') # model, sensitive to batch number
source = ['#source/training/tools/train_net.py',
          '#output/scrape/cr-label/test-train/train.json',
          '#output/scrape/cr-label/test-train/test.json']
arguments = "--dataset_name          ocr1 \
    --json_annotation_train ./datastore/scrape/cr-label/test-train/train.json \
    --image_path_train      ./datastore/scrape/cr-label  \
    --json_annotation_val   ./datastore/scrape/cr-label/test-train/test.json \
    --image_path_val        ./datastore/scrape/cr-label  \
    --config-file           ./source/training/configs/prima/fast_rcnn_R_50_FPN_3x.yaml \
    --resume                \
    OUTPUT_DIR  ./datastore/layout-model-training/outputs/fast_rcnn_R_50_FPN_3x_batch3_manual/ \
    SOLVER.IMS_PER_BATCH 2" 
command = f'python {source[0]} {arguments}'          
env.Command(target, source, command)

# visualize model
target = Glob('#output/scrape/cr-label/test/batch3_manual/*') # inference output, sensitive to batch number
source = ['#source/training/visualize_model_output.py',
          '#output/layout-model-training/outputs/fast_rcnn_R_50_FPN_3x_batch3_manual/config.yaml',
          '#output/layout-model-training/outputs/fast_rcnn_R_50_FPN_3x_batch3_manual/model_final.pth',]
command = f'bash {source[0]}'          
env.Python(target, source)












