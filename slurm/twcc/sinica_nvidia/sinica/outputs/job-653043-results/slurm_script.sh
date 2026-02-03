#!/bin/bash

#SBATCH --account GOV113054               ### project number, Example MST109178
#SBATCH --job-name tinyllama              ### Job name, Exmaple jupyterlab, --job-name=_nemotest_
#SBATCH --partition gp1d                  ### Partition Name, Example ngs1gpu, --partition==gp1d
#SBATCH --nodes=1                         ### Nodes, Default 1, node number
#SBATCH --ntasks-per-node=1               ### Tasks, Default 1, per node tasks
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4                 ### Cores assigned to each task, Example 4, --cpus-per-task=4
#SBATCH --mem=90GB                         ### Job memory request
##########SBATCH --gres=gpu:1                      ### GPU number, Example gpu:1



echo "\$SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "\$SLURM_SUBMIT_DIR = ${SLURM_SUBMIT_DIR}"
echo "\$0 = ${0}"
echo "\$1 = ${1}"

output_dir="outputs/job-${SLURM_JOB_ID}-results/"
mkdir -p ${output_dir}
current_script=${0}
cp ${current_script} ${output_dir}/$(basename $current_script).sh
echo "cp ${current_script} ${output_dir}/$(basename $current_script).sh"

# cp ${current_script} /work/u1160696/singularity_workspace/slurm-logs/${SLURM_JOB_ID}-${current_script}

# image_path="${HOME}/singularity_images/sandbox_nemo_24.05-1.sif"
image_path="/work/waue0920/open_access/ngc2407-nemo-20240901.sif"
srun --mpi=pmix singularity exec \
--nv \
-B ${HOME}/models:/models \
-B ${HOME}/data:/data \
-H ${HOME}:${HOME} \
${image_path} bash -c \
"
pwd
ls -l
nvidia-smi
echo \${0}
echo ${0}
python -c 'import torch; print(\"torch.cuda.device_count() = {}\".format(torch.cuda.device_count()))'
bash run-multi-gpu-pt.sh
"

# printenv
