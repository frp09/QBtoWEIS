#!/bin/bash
#SBATCH --job-name=LWG_floateropt_Barra           		# Job name
#SBATCH --nodes=1		   		            # Run all processes on a single node	
#SBATCH --ntasks-per-node=1 			    # Number of processes to launch per node		
#SBATCH --cpus-per-task=48           		# Number of CPU cores per task
#SBATCH --partition=qrease         		    # Name of the partition (marte24cpu/marte32cpu)
#SBATCH --output=QBWeis_output.log    		# Standard output and error log

pwd; hostname; date

echo "Run conda" #display run info


source /usr/local/gridware/miniconda3/bin/activate  
conda activate qbweis-env

srun python weis_driver_oc3.py

date