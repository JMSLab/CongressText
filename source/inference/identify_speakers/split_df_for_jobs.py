import pandas as pd
import numpy as np
import os
import re

inference_dir = 'datastore/inference'
image_dir = 'datastore/scrape/cr-bound'

docsdf_path = inference_dir + "/docs.csv"

# construct CSV for progress if needed
if not os.path.isfile(docsdf_path):
	raise RuntimeError("Master CSV of files to process does not exist. Run inference/split_df_for_jobs.py")


# Load the document list
df = pd.read_csv(f'{inference_dir}/docs.csv')

# Only documents where first stage inference has completed (speakers collected)
files_to_process = df[df['complete'] == 1]

# Divide the list into chunks
num_chunks = 100  # Adjust based on the number of available CPUs/nodes
chunks = np.array_split(files_to_process, num_chunks)

# Save each chunk to a separate CSV file
for i, chunk in enumerate(chunks):
    chunk.to_csv(f'{inference_dir}/chunk_speaker_{i}.csv', index=False)
