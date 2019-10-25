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
#PBS -l walltime=479:59:59
### pass the full environment
#PBS -V
#
# ===== END PBS OPTIONS =====

### run job

source activate pyenv

cd $PBS_O_WORKDIR
./train.sh "$shot" "$way" "$flag" "$id"

conda deactivate