#!/bin/bash
#SBATCH --job-name=congress_text_inference  # Adjust for your scheduler
#SBATCH --output=datastore/inference/output_%j.txt
#SBATCH --error=datastore/inference/error_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00  # maximum time
#SBATCH --account=jshapiro_lab
#SBATCH --partition=shared
#SBATCH --mem=100000


module load python  
source activate pDL

python source/inference/infer_layouts_and_text.py chunk_${1}.csv