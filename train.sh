#!/bin/bash
#SBATCH --job-name=dfcgan_train
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --mail-type=END
#SBATCH --mail-user=mehmet.a.bagc@fau.de
#SBATCH --cpus-per-task=4

mkdir -p logs

module load cuda/11.8.0
module load cudnn/9.2.0.82-11

conda activate dfcg

cd ~/DFCG

python trainDFcycleGAN.py \
    --train_path ./data/train/flk_img ./data/train/org_img \
    --eval_path  ./data/eval/flk_img  ./data/eval/org_img  \
    --save_path  ./weights \
    --load_path  ./weights \
    --Model_arch samescale \
    --max_epoch  100 \
    --batch_size 4 \
    --test_step  999
