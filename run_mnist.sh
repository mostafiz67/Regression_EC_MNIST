#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --mem=60000M  # memory
#SBATCH --cpus-per-task=32
#SBATCH --output=Regression-EC-MNIST-%j.out  # %N for node name, %j for jobID
#SBATCH --time=00-14:00     # time (DD-HH:MM)
#SBATCH --mail-user=x2020fpt@stfx.ca # used to send emailS
#SBATCH --mail-type=ALL

module load python/3.8
SOURCEDIR=/home/x2020fpt/projects/def-jlevman/x2020fpt/

source /home/x2020fpt/projects/def-jlevman/x2020fpt/Regression-EC-MNIST/.venv/bin/activate && echo "$(date +"%T"):  Activated python virtualenv"

echo "$(date +"%T"):  start running model!"
python3 /home/x2020fpt/projects/def-jlevman/x2020fpt/Regression-EC-MNIST/run.py --choices=Train-Test
echo "$(date +"%T"):  Finished running!"


echo "$(date +"%T"):  start Comparing model prediction!"
python3 /home/x2020fpt/projects/def-jlevman/x2020fpt/Regression-EC-MNIST/run.py --choices=Compare
echo "$(date +"%T"):  Finished comparing!"
