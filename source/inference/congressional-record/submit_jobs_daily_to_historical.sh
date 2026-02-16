#!/bin/bash                                                                                                                           
## must be run from the congressional-record repo root with venv congressDL activated                                                

## over years                                                                                                                         

for i in $(seq 1994 2025); do
    sbatch job_template_daily_to_historical.sh "$i"
done

