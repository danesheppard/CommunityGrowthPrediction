#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --account=an-tr043
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=TrainingResults-%j.txt
#SBATCH --job-name=CommunityGrowthPrediction


# Load python
module load python/3.11.5

# Create the virtual environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
# Install the required packages
pip install --no-index --upgrade pip
pip install --no-index numpy dask pandas matplotlib joblib scikit-learn

# Preprocess the census data for each province
python censusProcessing.py
# Assemble the processed data into a single file
python dataAssembly.py
# Train the model and run the analysis
python popGrowthModelTraining.py
