#!/bin/bash


#SBATCH --output=fairseq_train_fusion_%J.out
#SBATCH --error=fairseq_train_fusion_%J.err
#SBATCH --job-name=fairseq_train_fusion

#SBATCH --nodes=1
#SBATCH --time=112:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=8192
#SBATCH --mail-type=BEGIN,END,FAIL   
#SBATCH --mail-user=danielzgsilva@knights.ucf.edu 

echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

echo 'Beginning script'

# Load the TensorFlow module
module load cuda/cuda-10.2

source activate ~/my-envs/MOT

# List the modules that are loaded
module list

# Have Nvidia tell us the GPU/CPU mapping so we know
nvidia-smi topo -m

echo

fairseq-train data-bin/writingPrompts --save-dir ./fusion_checkpoints -a fconv_self_att_wp --lr 0.25 --optimizer nag --clip-norm 0.1 --max-tokens 1500 --lr-scheduler reduce_lr_on_plateau --decoder-attention True --encoder-attention False --criterion label_smoothed_cross_entropy --weight-decay .0000001 --label-smoothing 0 --source-lang wp_source --target-lang wp_target --gated-attention True --self-attention True --project-input True --skip-invalid-size-inputs-valid-test --pretrained True --pretrained-checkpoint ./baseline_checkpoints/checkpoint_best.pt --no-epoch-checkpoints

echo

echo "Ending script..."
date

