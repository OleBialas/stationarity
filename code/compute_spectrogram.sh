#!/bin/bash
# Submit the job array with one job per subject and number of bands
#SBATCH --output=/scratch/obialas/logs/compute_spectrograms.log
#SBATCH -t 0-48:00:00
#SBATCH -c 8 --mem-per-cpu=4G
#SBATCH --partition=debug
#SBATCH --mail-type=END
#SBATCH --mail-user=ole.bialas@posteo.de

# set up the environment
source /software/anaconda/2019.10/bin/activate hivemind
cd /scratch/obialas/stationarity

datalad run --input "data/stimuli/*" --output "results/spectrogram" "python code/compute_spectrogram.py"

