#!/bin/bash

# User whose jobs to cancel
USER="waue0920"
# Job name to target
JOB_NAME="interact"

# Get a list of job IDs for the given user and job name
job_ids=$(squeue --noheader -u $USER --format="%.18i %.18P %.18j %.18u %.18T %.18M %.18n" | awk -v user="$USER" -v name="$JOB_NAME" '$4 == user && $3 == name {print $1}')

# Check if there are any jobs to cancel
if [ -z "$job_ids" ]; then
  echo "No jobs found for user '$USER' with name '$JOB_NAME'."
  exit 0
fi

# Cancel each job ID
for job_id in $job_ids; do
  echo "Cancelling job ID: $job_id"
  scancel $job_id
  if [ $? -eq 0 ]; then
    echo "Successfully cancelled job ID: $job_id"
  else
    echo "Failed to cancel job ID: $job_id" >&2
  fi
done

echo "All '$JOB_NAME' jobs for user '$USER' have been processed."
