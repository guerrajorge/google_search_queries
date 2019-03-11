# qsub command
# qsub -l GPU=48 -e gsq_lr_error.txt -o gsq_lr_out.txt -q gpu.q github/gsq/scripts/lr_job.sh

echo 'running jobs.sh'
echo ""

export PATH=/home/guerramarj/packages/anaconda3/bin:$PATH

echo 'Loading modules'
module load cuda90
module load cudnn

echo 'Activate virtual environment'
source activate deeplearning

echo 'Moving to GSQ directory'
cd /home/guerramarj/github/gsq/

echo 'Running NO-SHOW'
export KERAS_BACKEND='tensorflow'
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# export CUDA_VISIBLE_DEVICES="0,1,2,3"

python main_4.py -d final_master.csv -m lr -l debug

echo "Finished running gsq"
exit 0
