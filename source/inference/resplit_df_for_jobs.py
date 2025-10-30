import os
import re
import glob
import numpy as np
import pandas as pd

inference_dir = "datastore/inference"
num_splits = 10   # produces outputs indexed 0..9 for each original chunk

# Find and sort chunk files numerically
chunk_paths = glob.glob(os.path.join(inference_dir, "chunk_*.csv"))

def chunk_key(path):
    m = re.search(r"chunk_(\d+)\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else float("inf")

chunk_paths.sort(key=chunk_key)

if not chunk_paths:
    print("No chunk_*.csv files found in", inference_dir)

for path in chunk_paths:
    m = re.search(r"chunk_(\d+)\.csv$", os.path.basename(path))
    if not m:
        print(f"Skipping unrecognized file name: {path}")
        continue

    chunk_id = m.group(1)
    print(f"Processing {os.path.basename(path)} (chunk id {chunk_id})")

    # Load and coerce 'complete' to numeric (handles '0'/'1' as strings, NaNs -> 0)
    df = pd.read_csv(path)
    if 'complete' not in df.columns:
        print(f"  Skipping: no 'complete' column in {path}")
        continue
    df['complete'] = pd.to_numeric(df['complete'], errors='coerce').fillna(0).astype(int)

    # Filter uncompleted
    uncompleted = df[df['complete'] == 0]

    # Split into equal-ish parts (always writes num_splits files, even if some are empty)
    splits = np.array_split(uncompleted, num_splits)

    for i, subdf in enumerate(splits):
        out_path = os.path.join(inference_dir, f"chunk_new_{chunk_id}_{i}.csv")
        subdf.to_csv(out_path, index=False)
    print(f"  Wrote {num_splits} files: chunk_new_{chunk_id}_0..{num_splits-1}.csv "
          f"(uncompleted rows: {len(uncompleted)})")
