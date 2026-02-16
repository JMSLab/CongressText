#!/bin/bash
#SBATCH --job-name=congress_text_inference  # Adjust for your scheduler
#SBATCH --output=datastore/inference/output_%j.txt
#SBATCH --error=datastore/inference/error_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=5:00:00  
#SBATCH --account=jshapiro_lab
#SBATCH --partition=jshapiro
#SBATCH --mem=100000
#SBATCH --mail-type=END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH --mail-user=andrewkao@g.harvard.edu                                                                                           

module purge
export PYTHONPATH=/n/home12/andrewkao/.conda/envs/pDL/lib/python3.10/site-packages:$PYTHONPATH
module load python ##/3.10.12-fasrc01
## module load Mambaforge/23.3.1-fasrc01

conda deactivate
conda activate pDL

python source/inference/identify_speakers/identify_speakers.py datastore/inference/daily_harmonized/chunk_speaker_${1}.csv datastore/inference/daily_harmonized





