#!/bin/bash
#SBATCH --job-name=congress_text_inference  # Adjust for your scheduler
#SBATCH --output=datastore/inference/output_%j.txt
#SBATCH --error=datastore/inference/error_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=500:00:00  # check periodically, about 3 weeks 
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=PARTITION_NAME
#SBATCH --mem=100000
#SBATCH --mail-type=END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH --mail-user=USER@DOMAIN.EDU

module purge
export PYTHONPATH=/n/home12/andrewkao/.conda/envs/pDL/lib/python3.10/site-packages:$PYTHONPATH
module load python ##/3.10.12-fasrc01  
## module load Mambaforge/23.3.1-fasrc01   
conda deactivate
conda activate pDL

set -euo pipefail

MODE="${1:-}"
CHUNK_INDEX="${2:-}"

if [[ -z "${MODE}" || -z "${CHUNK_INDEX}" ]]; then
  echo "Usage: sbatch job_template.sh <daily|historical> <chunk_index>"
  exit 1
fi

INPUT_PATH=""
OUTPUT_DIR=""
case "${MODE}" in
  daily)
    INPUT_PATH="datastore/inference/daily_harmonized/chunk_speaker_${CHUNK_INDEX}.csv"
    OUTPUT_DIR="datastore/inference/daily_harmonized"
    ;;
  historical)
    INPUT_PATH="datastore/inference/chunk_speaker_${CHUNK_INDEX}.csv"
    OUTPUT_DIR="datastore/inference"
    ;;
  *)
    echo "Invalid mode '${MODE}'. Expected 'daily' or 'historical'."
    exit 2
    ;;
esac

python source/inference/speaker_disambiguation/identify_speakers.py "${INPUT_PATH}" "${OUTPUT_DIR}"

