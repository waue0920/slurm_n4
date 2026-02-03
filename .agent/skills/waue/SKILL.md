---
name: waue
description: Creates a new Slurm job script in workspace/ by copying the workspace/env.sb template and assigning a unique timestamped filename. This skill helps quickly generate a base Slurm script for further modification, preserving essential parameters like -A and -p.
---

# Waue

## Create Slurm Script from Template

This skill provides a workflow for generating a new Slurm job script based on the `workspace/env.sb` template. It ensures that common Slurm parameters are preserved, allowing you to quickly get a working base for your specific job.

### Workflow:

1.  **Read Template:** Read the content of the `workspace/env.sb` file.
2.  **Generate Filename:** Create a new filename in the `workspace/` directory using a timestamp for uniqueness (e.g., `workspace/sbatch_YYYYMMDD_HHMM.sb`).
3.  **Write New Script:** Write the content of the template to the newly generated file.
4.  **Confirm Creation:** Inform the user about the path to the newly created Slurm script.

