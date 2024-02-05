#!/bin/bash

# ALWAYS specify CPU and RAM resources needed, as well as walltime
#SBATCH --partition=teaching_gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2-00:00:00

# job parameters
#SBATCH --job-name=python-venv
#SBATCH --account=masters

echo Running on $(hostname)
VENV=/opt/local/data/jsin0002/venv2
if [ -d $VENV ]; then
   echo Virtual environment found, activating
   source $VENV/bin/activate
else
   echo Virtual environment not found, creating and activating
   mkdir -p $VENV
   python3 -m venv --system-site-packages --prompt venv2 $VENV
   source $VENV/bin/activate
   pip install -r req1.txt
fi

python3 /opt/users/jsin0002/CICIDS2017_Transformer_DoS_GoldenEye.py
