#!/bin/bash                                                                                                                           
## must be run from the congressional-record repo root with venv congressDL activated                                                
## requires parse_local_cr.py to have parsed all relevant years

## over years                                                                                                                         

for i in $(seq 1994 2025); do
    sbatch job_template_daily_to_historical.sh "$i"
done

