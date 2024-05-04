#!/bin/bash
# Submit the job array with one job per subject and number of bands
#SBATCH --output=/scratch/obialas/logs/seg_%A_%a.log
#SBATCH --array=1-20
#SBATCH -t 0-48:00:00
#SBATCH -c 4 --mem-per-cpu=6G
#SBATCH --partition=standard
#SBATCH --mail-type=END
#SBATCH --mail-user=ole.bialas@posteo.de

# set up the environment
source /software/anaconda/2019.10/bin/activate hivemind
cd /scratch/obialas/stationarity

# Extract parameters from the json file
json_file="code/effect_of_segmentation_parameters.json"
sub_list=$(jq -r '.subjects | @csv' "$json_file")

# Convert the CSV string to an array
IFS=',' read -ra sub_array <<< "$sub_list"

# remove quotation marks
sub=$(echo "$sub" | sed 's/^"\(.*\)"$/\1/')

echo "running $sub"

# create throw-away clone in temporay directory
cd /local_scratch/$SLURM_JOB_ID
datalad clone /scratch/obialas/stationarity ds
cd ds
mkdir -p results/fit
git-annex dead here # dont use git annex in the clone
git checkout -b "job-$SLURM_ARRAY_TASK_ID" # new unique branch

datalad run --input "data/$sub" --input "results/spectrogram" --output "results/fit/*" "python code/effect_of_segmentation.py $sub"
datalad push --to origin

