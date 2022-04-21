#!/bin/bash                         #-- what is the language of this shell
#                                   #-- Any line that starts with #$ is an instruction to SGE
#$ -q gpu.q                         # GPU queue
#$ -pe smp 1                        # number of GPUs
#$ -l compute_cap=61,gpu_mem=11000M
#$ -S /bin/bash                     #-- the shell for the job
#$ -o /wynton/protected/home/ichs/dlituiev/logs                        #-- output directory (fill in)
#$ -cwd                            #-- tell the job that it should start in your working directory
#$ -M d.lituiev@gmail.com
#$ -m ae
#$ -r y                            #-- tell the system that if a job crashes, it should be restarted
#$ -j y                            #-- tell the system that the STDERR and STDOUT should be joined
#$ -l mem_free=16G                 #-- submits on nodes with enough free memory (required)
#$ -l scratch=1G                   #-- SGE resources (home and scratch disks)
#$ -l h_rt=48:00:00                #-- runtime limit (see above; this requests 24 hours)
#$ -t 1

##########################################################################
export CUDA_VISIBLE_DEVICES=$SGE_GPU
##########################################################################
if [ ${#SGE_TASK_ID}  -le 0 ]
then
    SGE_TASK_ID=0
fi


#sample=$(awk 'FNR == i {print}' i=${SGE_TASK_ID} ${file_names})
echo "---"

date
hostname
#echo "${SGE_TASK_ID} : $sample"

nvidia-smi


echo "=============================================="
echo "                    Running"
echo "=============================================="


##########################################################################
qstat -j $JOB_ID

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torchnlp

TASK="BC5CDR-chem"
#DATADIR="data/tasks/BC5CDR-chem.model=roberta-large.maxlen=512"
TASK=I2B22012NER
DATADIR="data/tasks/I2B22012NER.model=roberta-large.maxlen=512"
MODEL=models/roberta-large
MODEL=checkpoints/I2B22012NER/checkpoint-16000
MODEL_TYPE=roberta
CHECKPOINTS="checkpoints/${TASK}-v2"

python -m biolm.run_sequence_labelling \
    --data_dir ${DATADIR} \
    --model_type ${MODEL_TYPE} \
    --labels ${DATADIR}/labels.txt \
    --model_name_or_path ${MODEL} \
    --output_dir $CHECKPOINTS \
    --learning_rate 1e-5 \
    --max_seq_length  512 \
    --num_train_epochs 40 \
    --per_gpu_train_batch_size 4 \
    --save_steps 500 \
    --logging_steps 10 \
    --seed 10 \
    --gradient_accumulation_steps 2 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --eval_all_checkpoints \
    --evaluate_during_training

echo "=============================================="
echo "                    DONE"
echo "=============================================="

