#!/bin/bash


#SBATCH --output=fairseq_eval_%J.out
#SBATCH --error=fairseq_eval_%J.err
#SBATCH --job-name=fairseq_eval

#SBATCH --nodes=1
#SBATCH --time=112:00:00
#SBATCH --gres=gpu:1
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
export TEXT=./writingPrompts

fairseq-eval-lm data-bin/writingPrompts --path ./checkpoints/checkpoint_best.pt --batch-size 32 --max-tokens 1500 --criterion label_smoothed_cross_entropy --skip-invalid-size-inputs-valid-test
echo

echo "Ending script..."
date

