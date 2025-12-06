#!/usr/bin/python

import os
import sys
import subprocess
import tempfile


def makejob(venv, configpath, nruns):
    return f"""#!/bin/bash
#SBATCH --job-name=holotrack
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-{nruns}

current_dir=`pwd`
export PATH=$PATH:~/.local/bin

echo "Session " ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}

echo "Running on " $(hostname)
echo "Copying the source directory and data"
rsync -r --exclude logs --exclude logslurms --exclude configs --exclude ./data/ --exclude .env ./ $TMPDIR/
echo "Training"
{venv}/bin/python3 -m src.main {configpath}

if [[ $? != 0 ]]; then
    exit -1
fi
"""

def submit_job(job):
    with open("job.sbatch", "w") as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")

# Ensure the log directory exists
os.system("mkdir -p logslurms")

if len(sys.argv) not in [3, 4]:
    print(f"Usage : {sys.argv[0]} config.yaml <nruns|1>")
    sys.exit(-1)

venv = sys.argv[1]
configpath = sys.argv[2]
if len(sys.argv) == 3:
    nruns = 1
else:
    nruns = int(sys.argv[3])

# Copy the config in a temporary config file
os.system("mkdir -p configs")
tmp_configfilepath = tempfile.mkstemp(dir="./configs", suffix="-config.yml")[1]
os.system(f"cp {configpath} {tmp_configfilepath}")

# Launch the batch jobs
submit_job(makejob(venv, tmp_configfilepath, nruns))
