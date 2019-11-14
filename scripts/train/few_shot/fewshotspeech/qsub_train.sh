#!/bin/sh
#
# ======= PBS OPTIONS ======= (user input required)
#
### Specify queue to run
#PBS -q titan
### Set the job name
#PBS -N FKS
### Specify the # of cpus for your job.
#PBS -l nodes=1:ppn=1:gpus=1,mem=15GB
#PBS -l walltime=24:00:00
### pass the full environment
#PBS -V
#
# ===== END PBS OPTIONS =====

### run job

source activate pyenv

cd $PBS_O_WORKDIR
./train.sh "$way" "$shot" "$flag" "$id" "$alpha" "$margin"

conda deactivate
