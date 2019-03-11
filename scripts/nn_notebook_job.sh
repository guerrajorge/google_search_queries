# qsub command
# qsub -l GPU=8 -e nn_basic_error.txt -o nn_basic_out.txt -q gpu.q github/ns/scripts/nn_job.sh

echo 'running jobs.sh'
echo ""

export PATH=/home/guerramarj/packages/anaconda3/bin:$PATH

echo 'Loading modules'
module load cuda90
module load cudnn

echo 'Activate virtual environment'
source activate deeplearning

echo 'Moving to NO-SHOW directory'
cd /home/guerramarj/github/gsq/

echo 'Running NO-SHOW'
export KERAS_BACKEND='tensorflow'
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# export CUDA_VISIBLE_DEVICES="0,1,2,3"

jupyter lab --no-browser --port=8889
