import layoutparser as lp
import cv2
import os
import pdb
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

import json
import re

from pycocotools.coco import COCO
import layoutparser as lp
import random
import cv2
import subprocess

# task
# "from_aws" get data from AWS Sagemaker and convert to COCO
# "from_coco" get data from local labeling and convert
task = "from_coco"


os.chdir('datastore/scrape/cr-label')
batch = 3

if task == "from_aws":
    ### GET DATA ###
    manifest = 'output.manifest.batch' + str(batch)
    if not os.path.isfile(manifest):
        OUTPUT_MANIFEST = (
    	     "s3://congress-text-ocr/bound-labeled/cr-bound-labeling-batch" + str(batch) +  "/manifests/output/output.manifest"  # Replace with the S3 URI for your output manifest.
    	)
        cmd = f"aws s3 cp {OUTPUT_MANIFEST} 'output.manifest.batch{batch}'"
        subprocess.run(cmd, shell=True, check=True)


    #### CHECK IF YOU WANT TO DOWNLOAD FILES AGAIN #####

    # Function to turn sagemaker manifest into COCO format
    def sagemaker_to_coco(sagemaker_annotations):
        coco_format = {
            "info": {
                "description": "Converted from SageMaker format",
                "version": "1.0",
                "year": 2023,
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        annotation_id = 0
        category_map = {}  # Dictionary to map category names to their original IDs
        
        for image_id, item in enumerate(sagemaker_annotations):
            # Extracting image info
            image_info = {
                "file_name": item["source-ref"],
                "id": image_id,
                "width": item["cr-bound-labeling-batch"+str(batch)]["image_size"][0]["width"],
                "height": item["cr-bound-labeling-batch"+str(batch)]["image_size"][0]["height"]
            }
            coco_format["images"].append(image_info)
            
            # Extracting annotations
            for anno in item["cr-bound-labeling-batch"+str(batch)]["annotations"]:
                annotation_info = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": anno["class_id"],
                    "bbox": [anno["left"], anno["top"], anno["width"], anno["height"]],
                    "iscrowd": 0,
                    "area": anno["width"] * anno["height"],
                }
                coco_format["annotations"].append(annotation_info)
                annotation_id += 1
            
            # Updating the category_map
            class_map = item["cr-bound-labeling-batch"+str(batch)+"-metadata"]["class-map"]
            for class_id, class_name in class_map.items():
                if class_name not in category_map:
                    category_map[class_name] = int(class_id)
        
        # Extracting categories using the category_map
        for class_name, class_id in category_map.items():
            category_info = {
                "id": class_id,
                "name": class_name
            }
            coco_format["categories"].append(category_info)
        
        return coco_format


    with open(manifest, "r") as f:
        sagemaker_annotations = [json.loads(line.strip()) for line in f.readlines()]

    coco_data = sagemaker_to_coco(sagemaker_annotations)
    coco_data['info']

    # Directory where you want to save the downloaded images
    SAVE_DIR = 'batch'+str(batch)

    # Ensure the directory exists
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)   
        
    # Iterate over the images in coco_data and download them
    for image in coco_data['images']:
        s3_url = image['file_name']
        local_path = os.path.join(SAVE_DIR, s3_url.split('/')[-1])
        
        if not os.path.isfile(local_path):
    	    # Use subprocess to execute the AWS CLI command
    	    cmd = f'aws s3 cp {s3_url} {local_path}'
    	    subprocess.run(cmd, shell=True, check=True)
    	    
        # Update the file_name in coco_data with the local path
        image['file_name'] = local_path

    # Save the coco_data to a file
    with open('coco_format.json', 'w') as f:
        json.dump(coco_data, f)

    ### get training data
    coco = COCO('coco_format.json')


    def load_coco_annotations(annotations, coco=None):
        """
        Args:
            annotations (List):
                a list of coco annotaions for the current image
            coco (`optional`, defaults to `False`):
                COCO annotation object instance. If set, this function will
                convert the loaded annotation category ids to category names
                set in COCO.categories
        """
        layout = lp.Layout()

        for ele in annotations:

            x, y, w, h = ele['bbox']

            layout.append(
                lp.TextBlock(
                    block = lp.Rectangle(x, y, w+x, h+y),
                    type  = ele['category_id'] if coco is None else coco.cats[ele['category_id']]['name'],
                    id = ele['id']
                )
            )

        return layout

    COCO_IMG_PATH = SAVE_DIR


    # !python layout-model-training/utils/cocosplit.py   --split-ratio 0.85  --annotation-path coco_format.json  --train test-train/train.json --test test-train/test.json

    # os.chdir('../../../layout-model-training/scripts')
    # !bash train_cr.sh
elif task == "from_coco":
    old_coco_path = 'output.manual.batch' + str(batch)

    with open(old_coco_path, "r") as f:
        annotations = json.loads(f)

    # Function to update file paths
    def update_file_paths(data):
        for image in data["images"]:
            # Split the path and rejoin from "batch3"
            parts = image["file_name"].split('/')
            batch_index = parts.index('batch'+str(batch))  # Find index of batch
            image["file_name"] = '/'.join(parts[batch_index:])  # Join path from 'batch #' onwards

        return data

    coco_data = update_file_paths(annotations)

    # Save the coco_data to a file
    with open('coco_format.json', 'w') as f:
        json.dump(coco_data, f)    





