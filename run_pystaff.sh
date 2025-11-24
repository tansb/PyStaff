#!/bin/bash -f

#SBATCH --job-name=PyStaff_red_eyebrow
#SBATCH --ntasks-per-node=10
#SBATCH --nodes=1
#SBATCH --time=80:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=END,FAIL --mail-user=tbarone@swin.edu.au

module load gcc/14.2.0
module load poetry/2.1.2
# module load openmpi/4.1.6

export ALF_HOME=/fred/oz041/tbarone/softwares/alf_rosetta_stones/

cd ${ALF_HOME}src

export FNAME=RS_jwst_noisy
export NAME_TAG=_1PL_1age_80k
export NEW_FNAME=$FNAME$NAME_TAG

# remove the directory incase it already exits
rm -rf ../results/${NEW_FNAME}
# make it again so the files can be overwritten.
mkdir ../results/${NEW_FNAME}

# copy the alf and alf vars files to results dir with the name tag
scp alf.f90 ../results/${NEW_FNAME}/alf${NAME_TAG}.txt
scp alf_vars.f90 ../results/${NEW_FNAME}/alf_vars${NAME_TAG}.txt

# copy the input data file both to the results directory
# and into the indata directory with the new fname
scp ../indata/${FNAME}.dat ../indata/${NEW_FNAME}.dat
scp ../indata/${FNAME}.dat ../results/${NEW_FNAME}/${NEW_FNAME}.dat

make clean && make && mpirun -np $SLURM_NTASKS ../bin/alf.exe $NEW_FNAME

# now remove the copied new in data file.
rm ../indata/${NEW_FNAME}.dat

# copy the bestspec, sum and mcmc files to results dir with the name tag
mv ../results/${NEW_FNAME}.bestspec ../results/${NEW_FNAME}/${NEW_FNAME}.bestspec
mv ../results/${NEW_FNAME}.sum ../results/${NEW_FNAME}/${NEW_FNAME}.sum
mv ../results/${NEW_FNAME}.mcmc ../results/${NEW_FNAME}/${NEW_FNAME}.mcmc

#mv slurm-${SLURM_JOB_ID}.out ../results/${NEW_FNAME}/

