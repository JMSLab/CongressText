#!/bin/bash

num_chunks=100   # 10  # Adjust based on the number of chunks to run

for i in $(seq 0 $((num_chunks - 1)))
do
    sbatch source/inference/identify_speakers/job_template.sh $i
done
