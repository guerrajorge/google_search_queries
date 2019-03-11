# qsub command
# qsub -l GPU=8 -e gsq_nn_error.txt -o gsq_nn_out.txt -q gpu.q github/gsq/scripts/nn_job.sh

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

echo 'Running Google Search Queries'
export KERAS_BACKEND='tensorflow'
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# export CUDA_VISIBLE_DEVICES="0,1,2,3"

# python main.py -d final_master.csv -m nn -l debug
python main.py -d final_master.csv -m nn -p tfidf -l debug

echo "Finished running gsq"
exit 0
