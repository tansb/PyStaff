#!/bin/bash -f

#SBATCH --job-name=PyStaff_red_eyebrow
#SBATCH --ntasks-per-node=40
#SBATCH --nodes=1
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=END,FAIL --mail-user=tbarone@swin.edu.au

module load gcc/14.2.0
module load poetry/2.1.2
module load openmpi/5.0.7

export FNAME=REB_jwst
export TARGET='red_eyebrow'
export NAME_TAG=
export RUN_TAG=$FNAME$NAME_TAG
export NSTEPS=20
export NWALKERS=200

mpirun -np ${SLURM_NTASKS} poetry run python tbar_example.py \
--filename data/${FNAME}.dat \
--target ${TARGET} \
--nsteps ${NSTEPS} \
--nwalkers ${NWALKERS}