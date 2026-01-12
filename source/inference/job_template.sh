#!/bin/bash
#SBATCH --job-name=congress_text_inference  # Adjust for your scheduler
#SBATCH --output=datastore/inference/output_%j.txt
#SBATCH --error=datastore/inference/error_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=500:00:00  # check periodically, about 3 weeks 
#SBATCH --mem=100000
#SBATCH --mail-type=END,FAIL,REQUEUE,TIME_LIMIT
## if using SLURM, customize the following:
#SBATCH --account=jshapiro_lab
#SBATCH --partition=jshapiro
#SBATCH --mail-user=andrewkao@g.harvard.edu

CONDA_ENV_NAME="${CONDA_ENV_NAME:-pDL}"

module purge
module load python 
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate
conda activate "${CONDA_ENV_NAME}"

# Usage: sbatch job_template.sh <arg1>
python source/inference/infer_layouts_and_text.py datastore/inference/chunk_${1}.csv
