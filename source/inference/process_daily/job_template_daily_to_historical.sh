#!/bin/bash
#SBATCH --job-name=congress_text_inference  # Adjust for your scheduler
#SBATCH --output=../../../datastore/inference/output_%j.txt
#SBATCH --error=../../../datastore/inference/error_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=50:00:00  # check periodically, about 2 days
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=PARTITION_NAME
#SBATCH --mem=1000
#SBATCH --mail-type=END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH --mail-user=USER@DOMAIN.EDU


module load python ##/3.10.12-fasrc01
## module load Mambaforge/23.3.1-fasrc01
conda deactivate
conda activate congressDL

python daily_json_to_historical_csv.py --start ${1}-01-01 --end ${1}-12-31  --daily-root  ../../../datastore/inference/daily --out-root ../../../datastore/inference/daily_harmonized 

