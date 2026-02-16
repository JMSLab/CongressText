#!/bin/bash                                                                                                                          

num_chunks=100   # how number of original chunk_* files                                                                              

for i in $(seq 0 $((num_chunks - 1))); do
    sbatch source/inference/identify_speakers/job_template_daily.sh "$i"
done
