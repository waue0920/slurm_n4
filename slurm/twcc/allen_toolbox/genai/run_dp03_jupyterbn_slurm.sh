#!/bin/bash

# Submit the job and capture the output
output=$(sbatch dp03_jupyternb.slurm)

# Extract the job ID from the output
job_id=$(echo $output | awk '{print $4}')

# Tail the Slurm output file
tail -f log/waueai_$job_id.out
