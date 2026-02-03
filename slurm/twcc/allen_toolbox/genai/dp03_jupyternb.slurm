#!/work/u00cjz00/binary/bash5.0/bin/bash
#SBATCH -A GOV109134                                                    ### project number, Example MST109178
#SBATCH -J _t2demo_							### Job name, Exmaple jupyterlab
#SBATCH -p gp4d                                                         ### Partition Name, Example ngs1gpu
#SBATCH --nodes=1                                                       ### Nodes, Default 1, node number
#SBATCH --ntasks-per-node=1                                             ### Tasks, Default 1, per node tasks
#SBATCH -c 4                                                            ### Cores assigned to each task, Example 4
#SBATCH --gres=gpu:1                                                    ### GPU number, Example gpu:1
#SBATCH --time=0-1:00:00                                                ### Runnung time, days-hours:minutes:seconds or hours:minutes:seconds
#SBATCH -o log/waueai_%j.out                                                 ### Log folder, Here %j is job ID
#SBATCH -e log/waueai_%j.err                                                 ### Log folder, Here %j is job ID


## 帳號
iam=$(whoami)

# NODE
noed_hostname=$(hostname -s)
#noed_port=$(python3 /work/u00cjz00/binary/availablePort.py)
noed_port=$(python3 availablePort.py)
node_ip=$(cat /etc/hosts |grep "$(hostname -a)" | awk '{print $1}')

minicondaPath=/work/u00cjz00/share/miniconda3_py38
source ${minicondaPath}/bin/activate


# Jupyter Notebook folder 
#notebook_dir=${HOME}/labs
#mkdir -p ${notebook_dir}
notebook_dir=${HOME}

# LOG
TIMESTAMP=$(date "+%Y%m%d%H%M%S")
logfile=${notebook_dir}/conda_notebook_${TIMESTAMP}.log

## Activate Jupyter Notebook
jupyter-lab --notebook-dir=${notebook_dir} --no-browser \
--ip=0.0.0.0 --port=${noed_port} \
--NotebookApp.base_url /jupyter/${node_ip}/${noed_port} --NotebookApp.allow_remote_access true &

pid=$!
echo "Wait for ....."
sleep 30

## notebook URL
#url01=https://node01.biobank.org.tw/jupyter/${node_ip}/$(cat ${HOME}/.local/share/jupyter/runtime/jpserver-${pid}-open.html |grep ${node_ip}  |grep href | awk -F "/jupyter/${node_ip}/" '{print $2}' |awk -F '">' '{print $1}')
url01=http://103.124.72.7:${noed_port}/jupyter/${node_ip}/$(cat ${HOME}/.local/share/jupyter/runtime/jpserver-${pid}-open.html |grep ${node_ip}  |grep href | awk -F "/jupyter/${node_ip}/" '{print $2}' |awk -F '">' '{print $1}')
url02=http://localhost:${noed_port}/jupyter/${node_ip}/$(cat ${HOME}/.local/share/jupyter/runtime/jpserver-${pid}-open.html |grep ${node_ip}  |grep href | awk -F "/jupyter/${node_ip}/" '{print $2}' |awk -F '">' '{print $1}')


# message
echo "# JOB ID: ${pid}"
echo "# LOG FILE: ${logfile}"
echo "# PASSWD FILE: ${HOME}/.local/share/jupyter/runtime/jpserver-${pid}-open.html"
echo ""
echo "****************************** 請輸入下方指令 ******************************"
echo "# STEP1: Execute cmd in your client below "
echo "方法一, 互動連線, 但容易斷線"
echo ssh -NfL 0.0.0.0:${noed_port}:${noed_hostname}:${noed_port} ${iam}@ln01.twcc.ai
#echo "or"
#echo "方法二, 背景執行連線, 請若要斷線, 請記得刪除你本地端ssh連線"
#echo  ssh -o StrictHostKeyChecking=no -o TCPKeepAlive=yes -o ServerAliveCountMax=20 -o ServerAliveInterval=15 -NfL ${noed_port}:${noed_hostname}:${noed_port} ${iam}@ln01.twcc.ai
#echo ""
echo "****************************** 請打開以下連結 ******************************"
echo "# STEP2: Open url below "
echo "${url01}"
echo ""
echo ""
echo "*****************************************************************************"



## Watting
wait $pid0


