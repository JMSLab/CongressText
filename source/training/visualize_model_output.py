import cv2
import layoutparser as lp
import os
import pdb

model = lp.Detectron2LayoutModel(
    config_path = "datastore/training/outputs/fast_rcnn_R_50_FPN_3x_batch3_manual/config.yaml",
    model_path = "datastore/training/outputs/fast_rcnn_R_50_FPN_3x_batch3_manual/model_final.pth",
    extra_config = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8] # <-- Only output high accuracy preds
)

image_dir = 'datastore/label/batch3'
output_dir = 'datastore/training/cr-label/test/batch3_manual'

# Iterate through each file in the directory
for filename in os.listdir(image_dir):
    # Check if the file has a .png extension
    if filename.endswith('.png'):
        # Full path to the .png file
        file_path = os.path.join(image_dir, filename)
        image = cv2.imread(file_path) 
        layout = model.detect(image)
        image_with_boxes = lp.draw_box(image, layout)

        # Save the image with boxes
        new_file = os.path.join(output_dir, filename)
        image_with_boxes.save(new_file)

        



