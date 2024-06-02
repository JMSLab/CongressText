import cv2
import layoutparser as lp
import os
import pandas as pd
import pdb
import re
import numpy as np
from pdf2image import convert_from_path
from tqdm import tqdm
import pytesseract
from efficient_ocr import EffOCR

# SaveData
import sys
sys.path.append('source/lib')
from SaveData import SaveData

# initialize tesseract
home_dir = os.getenv('HOME')
pytesseract.pytesseract.tesseract_cmd = home_dir + '/tesseract/tesseract'
os.environ["TESSDATA_PREFIX"] = home_dir + "/tesseract/tessdata"


model = lp.Detectron2LayoutModel(
    config_path = "datastore/training/outputs/fast_rcnn_R_50_FPN_3x_batch3_manual/config.yaml",
    model_path = "datastore/training/outputs/fast_rcnn_R_50_FPN_3x_batch3_manual/model_final.pth",
    extra_config = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8] # <-- Only output high accuracy preds
)

image_dir = 'datastore/scrape/cr-bound'
output_dir = 'output/inference'
inference_dir = 'datastore/inference'

ocr_agent = lp.TesseractAgent(languages='eng') 



def files_to_dprogress_df(image_dir):
    """
    Take all files in dir and return a df containing relevant ones
    Also initializes dfs for sections, speeches, and speakers
    """
    all_docs = os.listdir(image_dir)

    # only keep files that fit the format (avoid duplicates)
    # get optional version number
    pattern = r"GPO-CRECB-(\d{4})-pt(\d{1,2})(-v\d+)?\.pdf"

    extracted_docs = []
    for doc in all_docs:
        match = re.match(pattern, doc)
        if match:
            year, part, version = match.groups()
            version_number = int(version.replace('-v', '')) if version else 100
            extracted_docs.append((doc, int(year), int(part), version_number))

    # sort by year, part, and version number
    extracted_docs.sort(key=lambda x: (x[1], x[2], -x[3]))

    # get final set of docs, unique ID is year + part
    # first take no version number
    # otherwise, take largest version number 
    final_docs = {}
    for doc, year, part, version_number in extracted_docs:
        key = (year, part)
        # if a new entry or a higher version number
        if key not in final_docs or final_docs[key][3] > version_number:
            final_docs[key] = (doc, year, part, version_number)

    sorted_docs = [info[0] for info in final_docs.values()]

    # create df
    docs_df = pd.DataFrame({
        "title": [doc for doc in sorted_docs],
        "complete": [0] * len(sorted_docs),
        "section_id": [0] * len(sorted_docs),
        "speech_id": [0] * len(sorted_docs),
        "paragraph_id": [0] * len(sorted_docs)
    })


    sections_df = pd.DataFrame(columns = ['year','part','part_page','date',
                                            'volume_page','section_name','section_id'])

    speeches_df = pd.DataFrame(columns = ['section_id','speech_order','speaker_name',
                                            'speaker_id','speech_id'])

    speakers_df = pd.DataFrame(columns = ['speaker_id','speaker_name'])

    return docs_df, sections_df, speeches_df, speakers_df


def new_paragraph_df():
    """Construct df for paragraph level text. One per document."""
    paragraphs_df = pd.DataFrame(columns = ['speech_id','paragraph_text',
                                            'paragraph_order','paragraph_id'])
    return paragraphs_df


def pdf_to_cv2_images(pdf_path):
    """Convert a PDF file to a list of images for cv2"""
    pil_images = convert_from_path(pdf_path)
    # convert to array, from RGB to BGR for OpenCV
    cv2_images = [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image in pil_images]
    return cv2_images

def doc_to_yearpart(filename):
    """Get year and part number from filename  """
    year = 0
    match = re.search(r'\d{4}', filename)
    if match:
        year = int(match.group())
    else:
        print("No four-digit sequence found.") 


    part = 0
    match = re.search(r'pt(\d+)', filename)
    if match:
        part = int(match.group(1))
    else:
        print("No part found found.") 

    return year, part


def init_effocr():
    """Construct EffOCR engine"""
    # TODO: potentially train model

    # English
    effocr = EffOCR(
      config={
          'Recognizer': {
              'char': {
                  'model_backend': 'onnx',
                  'model_dir': './models',
                  'hf_repo_id': 'dell-research-harvard/effocr_en/char_recognizer',
              },
              'word': {
                  'model_backend': 'onnx',
                  'model_dir': './models',
                  'hf_repo_id': 'dell-research-harvard/effocr_en/word_recognizer',
              },
          },
          'Localizer': {
              'model_dir': './models',
              'hf_repo_id': 'dell-research-harvard/effocr_en',
              'model_backend': 'onnx'
          },
          'Line': {
              'model_dir': './models',
              'hf_repo_id': 'dell-research-harvard/effocr_en',
              'model_backend': 'onnx',
          },
      }
    )
    return effocr

def infer_img2txt(engine_type,engine,image):
    """Goes from image to text using an OCR engine"""

    text = None

    if engine_type == "tesseract":
        text = engine.detect(image)
    elif engine_type == "effocr":
        text = engine.infer(image)

    return text


effocr = init_effocr()


# progress, at the document level
docsdf_path = inference_dir + "/docs.csv"
sectionsdf_path = inference_dir + "/sections.csv"
speechesdf_path = inference_dir + "/speeches.csv"
speakersdf_path = inference_dir + "/speakers.csv"
docsdf_log = output_dir + "/docs.log"
sectionsdf_log = output_dir + "/sections.log"
speechesdf_log = output_dir + "/speeches.log"
speakersdf_log = output_dir + "/speakers.log"


docs_df = None
sections_df = None
speeches_df = None
speakers_df = None

# if exists, load data
if os.path.isfile(docsdf_path):
    docs_df = pd.read_csv(docsdf_path)
    sections_df = pd.read_csv(sectionsdf_path)
    speeches_df = pd.read_csv(speechesdf_path)
    speakers_df = pd.read_csv(speakersdf_path)
# else, construct CSV for progress
else:
    docs_df, sections_df, speeches_df, speakers_df = files_to_dprogress_df(image_dir)
    docs_df.to_csv(docsdf_path, index=False)
    sections_df.to_csv(sectionsdf_path, index=False)
    speeches_df.to_csv(speechesdf_path, index=False)
    speakers_df.to_csv(speakersdf_path, index=False)

    ## TODO: handle speeches_text at the doc level (file size mgmt)
    ## TODO: disambiguate speakers


speaker_vals = speakers_df['speaker_name'].values

# iterate through incomplete files
for index, row in docs_df.iterrows():
    if row['complete'] == 0:
        
        file_path = os.path.join(image_dir, row['title'])
        print(file_path)
        year, part = doc_to_yearpart(row['title'])

        # convert PDF pages to PNG
        all_images = pdf_to_cv2_images(file_path)
        print("--- All pages loaded to PNG")

        # set up for new document
        paragraphs_df = new_paragraph_df()
        paragraphsdf_path = image_dir + f"/paragraphs_{year}_pt{part}.csv"
        paragraphsdf_log = output_dir + f"/paragraphs_{year}_pt{part}.log"

        section_id = row['section_id'] 
        speech_id = row['speech_id'] 
        paragraph_id = row['paragraph_id'] 
        speaker_id = 0    # default is null

        sections_new = []
        paragraphs_new = []
        speakers_new = []
        speeches_new = []
        

        for page_id, image in enumerate(tqdm(all_images)):
 
            layout = model.detect(image)
            
            # type guide:
            # 0 is title
            # 1 is section
            # 2 is skip
            # 3 is speaker
            # 4 is speech


            ## if any skip, skip
            if len(layout) > 0 and not any(item.type == 2 for item in layout):
                # pdb.set_trace()
                height, width = image.shape[:2]

                ## set IDs
                layout = lp.Layout([b.set(id=idx) for idx, b in enumerate(layout)])

                ## handle titles
                titles = lp.Layout([b for b in layout if b.type==0])  
                titles.sort(key = lambda b:b.coordinates[1], inplace=True)
                ## section headings
                sections = lp.Layout([b for b in layout if b.type==1])  
                ## speech
                speeches = lp.Layout([b for b in layout if b.type==4]) 
                ## speakers
                speakers = lp.Layout([b for b in layout if b.type==3]) 

                
                # get page data from first title
                part_page = page_id + 1
                ### TODO: extract this correctly; 
                ### format changes over time (need to alternate page numbers for date/year earlier)
                ### maybe exclude middle section
                date = titles[0] if titles else ""


                ### TODO: split into functions for testing

                


                ## split vertically by title block
                for title_id, title in enumerate(titles if titles else [""]): 

                    ### cleaning at title level

                    # get relevant blocks within the title
                    upper_y = height
                    # if not last title, set upper bound for y at next title y
                    if len(titles) > 0 and title != titles[-1]:
                        # 5 px grace
                        upper_y = titles[title_id + 1].coordinates[1] + 5

                    # first, keep everything below upper bound
                    section_Tblocks = [b for b in sections if b.coordinates[1] <= upper_y]
                    speech_Tblocks = [b for b in speeches if b.coordinates[1] <= upper_y]
                    speaker_Tblocks = [b for b in speakers if b.coordinates[1] <= upper_y]

                    # remove from original list
                    sections = [b for b in sections if b.coordinates[1] > upper_y]
                    speeches = [b for b in speeches if b.coordinates[1] > upper_y]
                    speakers = [b for b in speakers if b.coordinates[1] > upper_y]

                    # combine all blocks    
                    all_Tblocks = lp.Layout(section_Tblocks + speech_Tblocks + speaker_Tblocks)
                    # pdb.set_trace()

                    ### order blocks by column and position on page
                    # columns depends on year
                    # after (inclusive) 1941 volume 87 77th Congress, 3 columns
                    # before, 2 columns
                    
                    # code for 3 columns
                    if year >= 1941:
                        pass

                        # TODO: test where on page the column splits are
                        ## maybe also use actual block data to inform (if sufficient # of blocks)

                    # code for 2 columns
                    else: 

                        # split alongside middle
                        left_interval = lp.Interval(0, width/2*1.05, axis='x').put_on_canvas(image)

                        left_blocks = all_Tblocks.filter_by(left_interval, center=True)
                        left_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)
                        # The b.coordinates[1] corresponds to the y coordinate of the region
                        # sort based on that can simulate the top-to-bottom reading order 
                        right_blocks = lp.Layout([b for b in all_Tblocks if b not in left_blocks])
                        right_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)


                        # And finally combine the two lists and add the index
                        # all_Tblocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])
                        all_Tblocks = left_blocks + right_blocks

                    
                    ### order blocks correctly
                    # primary concern: handling speakers that appear after speeches they are associated with
                    ordered_blocks = []
                    # set of IDs
                    inserted_speakers = set()

                    for block in all_Tblocks:
                        # always insert section blocks when encountered
                        if block in section_Tblocks:
                            ordered_blocks.append(block)
                        # for speeches, look ahead for speaker blocks with a high intersection
                        elif block in speech_Tblocks:
                            best_speaker = None
                            max_intersection_area = 0
                            
                            for speaker_block in speaker_Tblocks:
                                if speaker_block.id not in inserted_speakers:
                                    intersection = block.intersect(speaker_block)
                                    intersection_area = intersection.area
                                    
                                    # check for large intersection (and max of all seen)
                                    if intersection_area > 0.7 * speaker_block.area and intersection_area > max_intersection_area:
                                        best_speaker = speaker_block
                                        max_intersection_area = intersection_area

                            # insert speaker before the speech block
                            if best_speaker:
                                ordered_blocks.append(best_speaker)
                                inserted_speakers.add(best_speaker.id)  # Mark this speaker as inserted

                            # insert the speech block
                            ordered_blocks.append(block)

                        # handle speaker blocks only if they haven't been inserted yet
                        elif block in speaker_Tblocks:
                            if block.id not in inserted_speakers:
                                ordered_blocks.append(block)
                                inserted_speakers.add(block.id)


                    ### extract info from blocks
                    for block in ordered_blocks:

                        # get data for sections
                        ### TODO: update date/volume_page! see above
                        if block in section_Tblocks:

                            # crop w/ padding in each segment for robustness
                            segment_image = (block
                                           .pad(left=5, right=5, top=5, bottom=5)
                                           .crop_image(image))
                            # standard ocr extraction
                            # section_name = infer_img2txt("tesseract",ocr_agent,segment_image)
                            section_name = infer_img2txt("effocr",effocr,segment_image)

                            new_row = {'year': year, 'part': part, 'part_page': part_page, 
                                       'date': date, 'volume_page': "", 'section_name': section_name,
                                       'section_id': section_id}
                            sections_new.append(new_row)

                            section_id += 1
                            speaker_id = 0    # reset speaker ID to null
                            speech_order = 1  # always first speech afterwards

                        elif block in speech_Tblocks:

                            # crop w/ padding in each segment for robustness
                            segment_image = (block
                                           .pad(left=5, right=5, top=5, bottom=5)
                                           .crop_image(image))
                            # standard ocr extraction
                            paragraph_text = infer_img2txt("effocr",effocr,segment_image)


                            new_row = {'speech_id': speech_id,'paragraph_text': paragraph_text,
                                            'paragraph_order': paragraph_order,'paragraph_id': paragraph_id}
                            paragraphs_new.append(new_row)

                            paragraph_id += 1
                            paragraph_order += 1


                        elif block in speaker_Tblocks:

                            # crop w/ padding in each segment for robustness
                            segment_image = (block
                                           .pad(left=5, right=5, top=5, bottom=5)
                                           .crop_image(image))
                            # standard ocr extraction
                            speaker_name = infer_img2txt("effocr",effocr,segment_image)

                            # get speaker_id, or update df if none
                            if speaker_name in speaker_vals:
                                try:
                                    speaker_id = speakers_df[speakers_df['speaker_name'] == speaker_name]['speaker_id'].iloc[0]
                                except IndexError:
                                    speaker_id = next((item['speaker_id'] for item in speakers_new if item['speaker_name'] == speaker_name), None)
                            else:
                                speaker_id = max(speaker_id,len(speaker_vals)) + 1

                                new_row = {'speaker_id': speaker_id, 'speaker_name': speaker_name}
                                speakers_new.append(new_row) 

                                speaker_vals = np.append(speaker_vals,speaker_name)  # speakers_df['speaker_name'].values


                            # add new speech data
                            new_row = {'section_id': section_id,'speech_order': speech_order,'speaker_name': speaker_name,
                                            'speaker_id': speaker_id,'speech_id': speech_id}
                            speeches_new.append(new_row)

                            speech_id += 1
                            speech_order += 1
                            paragraph_order = 1

                        else: 
                            print("Block not categorized")


                # extract to JSON?


            ## code for skip
            else:
                # print("SKIP page")
                pass
                

        # update DFs by concat
        sections_new_df = pd.DataFrame(sections_new)
        paragraphs_new_df = pd.DataFrame(paragraphs_new)
        speakers_new_df = pd.DataFrame(speakers_new)
        speeches_new_df = pd.DataFrame(speeches_new)

        sections_df = pd.concat([sections_df,sections_new_df],ignore_index=True)
        paragraphs_df = pd.concat([paragraphs_df,paragraphs_new_df],ignore_index=True)
        speakers_df = pd.concat([speakers_df,speakers_new_df],ignore_index=True)
        speeches_df = pd.concat([speeches_df,speeches_new_df],ignore_index=True)
        
                
        # update progress doc
        docs_df.loc[index, 'complete'] = 1

        ## record max section_id, speech_id, paragraph_id for next row
        docs_df.loc[index+1, 'section_id'] = section_id
        docs_df.loc[index+1, 'speech_id'] = speech_id
        docs_df.loc[index+1, 'paragraph_id'] = paragraph_id

        # save data
        SaveData(docs_df,['title'],docsdf_path,docsdf_log)
        SaveData(sections_df,['section_id'],sectionsdf_path,sectionsdf_log)
        SaveData(speeches_df,['speech_id'],speechesdf_path,speechesdf_log)
        SaveData(speakers_df,['speaker_id'],speakersdf_path,speakersdf_log)
        SaveData(paragraphs_df,['paragraph_id'],paragraphsdf_path,paragraphsdf_log)
        # docs_df.to_csv(dprogress_path, index=False)        
        # sections_df.to_csv(sectionsdf_path, index=False)
        # speeches_df.to_csv(speechesdf_path, index=False)
        # speakers_df.to_csv(speakersdf_path, index=False)
        # paragraphs_df.to_csv(paragraphsdf_path, index=False)

        # para_path = str(year)+'_pt'+str(part)+'_paragraph.csv'
        # file_path = os.path.join(image_dir, para_path)

        pdb.set_trace()
        



