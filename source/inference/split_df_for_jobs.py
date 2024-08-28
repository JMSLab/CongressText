import pandas as pd
import numpy as np
import os
import re

inference_dir = 'datastore/inference'
image_dir = 'datastore/scrape/cr-bound'

docsdf_path = inference_dir + "/docs.csv"
sectionsdf_path = inference_dir + "/sections.csv"
speechesdf_path = inference_dir + "/speeches.csv"
speakersdf_path = inference_dir + "/speakers.csv"


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

# construct CSV for progress if needed
if not os.path.isfile(docsdf_path):
    docs_df, sections_df, speeches_df, speakers_df = files_to_dprogress_df(image_dir)
    docs_df.to_csv(docsdf_path, index=False)
    sections_df.to_csv(sectionsdf_path, index=False)
    speeches_df.to_csv(speechesdf_path, index=False)
    speakers_df.to_csv(speakersdf_path, index=False)




# Load the document list
df = pd.read_csv(f'{inference_dir}/docs.csv')

# Filter out already processed documents
files_to_process = df[df['complete'] == 0]

# Divide the list into chunks
num_chunks = 10  # Adjust based on the number of available CPUs/nodes
chunks = np.array_split(files_to_process, num_chunks)

# Save each chunk to a separate CSV file
for i, chunk in enumerate(chunks):
    chunk.to_csv(f'{inference_dir}/chunk_{i}.csv', index=False)
