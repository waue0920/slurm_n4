#!/bin/bash
bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
cd $bin;

sbatch check_env/check_gpu_without_sh.sb

sleep 10

cat check_gpu.log

if [[ -f check_gpu.log ]]; then
    mv check_gpu.log check_env/
fi


