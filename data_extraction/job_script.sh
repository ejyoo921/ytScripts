#!/bin/bash
#SBATCH --account=co2snow
#SBATCH --time=04:00:00
###SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --qos=high         # uncomment if needed
#SBATCH --ntasks-per-node=104
#SBATCH --mail-user=eyoo@nrel.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=xMagVel
#SBATCH --output=job.%j.out  # %j will be replaced with the job ID
#SBATCH --error=job.%j.err

# Simulation will end and save a checkpoint at max_wall_time (in hours)
# Set it to just below your sbatch requested time
module load python
srun -n 104 python extract_slices.py --ifile ./ex_slices.toml  

