#!/bin/bash

num_chunks=10   # how number of original chunk_* files 
num_splits=10   # the 0..9 sub-splits per chunk (after re-split)

for i in $(seq 0 $((num_chunks - 1))); do
  for j in $(seq 0 $((num_splits - 1))); do
    sbatch source/inference/historical/job_template.sh "$i" "$j"
  done
done
