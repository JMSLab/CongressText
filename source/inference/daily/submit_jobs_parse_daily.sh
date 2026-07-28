#!/bin/bash

## must have requirements.txt from congressional-record repo 
## https://github.com/unitedstates/congressional-record/blob/ec0850412acffd0fc2dc6ed79fbd376e4699439d/requirements.txt

## over years
for i in $(seq 1994 2025); do
    sbatch job_template.sh "$i"
done

