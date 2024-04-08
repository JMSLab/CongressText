import cv2
import layoutparser as lp
import os
import pandas as pd
import pdb
import re
import numpy as np
from pdf2image import convert_from_path


model = lp.Detectron2LayoutModel(
    config_path = "datastore/training/outputs/fast_rcnn_R_50_FPN_3x_batch3_manual/config.yaml",
    model_path = "datastore/training/outputs/fast_rcnn_R_50_FPN_3x_batch3_manual/model_final.pth",
    extra_config = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8] # <-- Only output high accuracy preds
)

image_dir = 'datastore/scrape/cr-bound'
output_dir = 'datastore/inference/cr-bound/layouts'
inference_dir = 'datastore/inference'

ocr_agent = lp.TesseractAgent(languages='eng') 



def files_to_dprogress_df(image_dir)
    """Take all files in dir and return a df containing relevant ones"""
    all_docs = os.listdir(image_dir)

    # only keep files that fit the format (avoid duplicates)
    pattern = r"GPO-CRECB-\d{4}-pt\d{1,2}\.pdf"
    all_docs = [doc for doc in all_docs if re.match(pattern, doc)]

    # extract year and part number (sort chronologically)
    year_part_extracted = [(doc, int(doc.split('-')[2]), int(doc.split('-')[3].replace('pt', '').replace('.pdf', ''))) for doc in filtered_docs]
    # sort
    sorted_docs = sorted(year_part_extracted, key=lambda x: (x[1], x[2]))

    # create df
    docs_df = pd.DataFrame({
        "titles": [doc[0] for doc in sorted_docs],
        "complete": [0] * len(sorted_docs),
        "section_id": [0] * len(sorted_docs),
        "speech_id": [0] * len(sorted_docs),
        "paragraph_id": [0] * len(sorted_docs)
    })
    return docs_df


def pdf_to_cv2_images(pdf_path):
    """Convert a PDF file to a list of images for cv2"""
    pil_images = convert_from_path(pdf_path)
    # convert to array, from RGB to BGR for OpenCV
    cv2_images = [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image in pil_images]
    return cv2_images




# progress, at the document level
docsdf_path = inference_dir + "/docs.csv"
sectionsdf_path = inference_dir + "/sections.csv"
speechesdf_path = inference_dir + "/speeches.csv"

if os.path.isfile(docsdf_path):
    docs_df = pd.read_csv(docsdf_path)
    sections_df = pd.read_csv(sectionsdf_path)
    speeches_df = pd.read_csv(speechesdf_path)

# construct CSV for progress
else:
    docs_df = files_to_dprogress_df(image_dir)
    docs_df.to_csv(docsdf_path, index=False)

    ## TODO: instatiate sections_df, speeches_df
    ## TODO: handle speeches_text at the doc level (file size mgmt)





# iterate through incomplete files
for index, row in docs_df.iterrows():
    if row['complete'] == 0:
        
        file_path = os.path.join(image_dir, row['titles'])
        print(file_path)

        # convert PDF pages to PNG
        all_images = pdf_to_cv2_images(file_path)
        print("  All pages loaded to PNG")

        ## TODO: tqdm per page

        for image in all_images:

            # Full path to the .png file
            
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

                ### TODO: split into functions for testing

                ### TODO: rare exceptions (multiple titles, halfway down page, need to split?)

                ### TODO: initialize block information

                ## split vertically by title block
                for title in titles: 

                    # check if last

                    # if not, set lower bound for y at next title y

                    ## code for 3 columns
                    if year >= 1941:

                    ## code for 2 columns
                    else: 

                        # first, keep everything above lower bound
                        # remove from original list



                        # split alongside middle
                        left_interval = lp.Interval(0, width/2*1.05, axis='x').put_on_canvas(image)

                        # TODO: need to also combine in sections/speeches/speakers
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

                        # TODO: remove speaker if text_block
                        ### check LP documentation

                        segment_image = (block
                                           .pad(left=5, right=5, top=5, bottom=5)
                                           .crop_image(image))
                            # add padding in each image segment can help
                            # improve robustness 
                            
                        text = ocr_agent.detect(segment_image)
                        block.set(text=text, inplace=True)


                # extract to JSON

                # TODO: for later, sections infer from column # and y-pos


            ## code for skip
            else:
                



            image_with_boxes = lp.draw_box(image, layout)

            # Save the image with boxes
            new_file = os.path.join(output_dir, filename)
            image_with_boxes.save(new_file)

        # update progress doc
        ### TODO: savedata
        docs_df.loc[index, 'complete'] = 1
        ## TODO: record max section_id, speech_id, paragraph_id for next row

        docs_df.to_csv(dprogress_path, index=False)        

        



