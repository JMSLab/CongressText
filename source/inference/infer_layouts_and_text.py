import cv2
import layoutparser as lp
import os
import pdb

model = lp.Detectron2LayoutModel(
    config_path = "datastore/training/outputs/fast_rcnn_R_50_FPN_3x_batch3_manual/config.yaml",
    model_path = "datastore/training/outputs/fast_rcnn_R_50_FPN_3x_batch3_manual/model_final.pth",
    extra_config = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8] # <-- Only output high accuracy preds
)

image_dir = 'datastore/scrape/cr-bound'
output_dir = 'datastore/inference/cr-bound/layouts'

ocr_agent = lp.TesseractAgent(languages='eng') 

# Iterate through each file in the directory
for filename in os.listdir(image_dir):
    # Check if the file has a .png extension
    if filename.endswith('.png'):
        # Full path to the .png file
        file_path = os.path.join(image_dir, filename)
        image = cv2.imread(file_path) 
        layout = model.detect(image)
        
        # type guide:
        # 0 is title
        # 1 is section
        # 2 is skip
        # 3 is speaker
        # 4 is speech

        # TODO: blank json SET UP

        ## if any skip, skip
        if not any(item.type == 2 for item in layout):
            height, width = image.shape[:2]

            ## handle titles
            titles = lp.Layout([b for b in layout if b.type==0])  
            ## section headings
            sections = lp.Layout([b for b in layout if b.type==1])  
            ## speech
            speeches = lp.Layout([b for b in layout if b.type==4]) 
            ## speakers
            speakers = lp.Layout([b for b in layout if b.type==3]) 

            # columns depends on year
            # after (inclusive) 1941 volume 87 77th Congress, 3 columns
            # before, 2 columns
            year = 0
            match = re.search(r'\d{4}', filepath)
            if match:
                year = match.group()
            else:
                print("No four-digit sequence found.") 

            title_blocks = []
            section_blocks = []
            speech_blocks = []
            speaker_blocks = []

            ### TODO: rare exceptions (multiple titles, halfway down page, need to split?)
            ## split vertically by title block
            for title in titles: 

                # check if last

                # if not, set lower bound for y at next title y

                ## code for 3 columns
                if year >= 1941:

                ## code for 2 columns
                else: 
                    # split alongside middle
                    left_interval = lp.Interval(0, width/2*1.05, axis='x').put_on_canvas(image)

                    # need to repeat this for sections/speeches/speakers
                    left_blocks = text_blocks.filter_by(left_interval, center=True)
                    left_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)
                    # The b.coordinates[1] corresponds to the y coordinate of the region
                    # sort based on that can simulate the top-to-bottom reading order 
                    right_blocks = lp.Layout([b for b in text_blocks if b not in left_blocks])
                    right_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)

                    # And finally combine the two lists and add the index
                    text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])                

                    ## speakers
                    # check for largest intersection
                    # apply until next section or speaker


                for block in text_blocks:
                    segment_image = (block
                                       .pad(left=5, right=5, top=5, bottom=5)
                                       .crop_image(image))
                        # add padding in each image segment can help
                        # improve robustness 
                        
                    text = ocr_agent.detect(segment_image)
                    block.set(text=text, inplace=True)


            # extract to JSON
            


        ## code for skip
        else:
            



        image_with_boxes = lp.draw_box(image, layout)

        # Save the image with boxes
        new_file = os.path.join(output_dir, filename)
        image_with_boxes.save(new_file)

        



