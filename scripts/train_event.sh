
# #!/bin/bash -eux

# #SBATCH --job-name=segmentation

# #SBATCH --output=/lustre/scratch/client/vinai/users/vietlq4/out/slurm_%A.out

# #SBATCH --error=/lustre/scratch/client/vinai/users/vietlq4/out/slurm_%A.err

# #SBATCH --partition=applied

# #SBATCH --nodes=1

# #SBATCH --ntasks=8

# #SBATCH --gpus-per-node=8

# #SBATCH --mem-per-gpu=64G

# #SBATCH --cpus-per-gpu=32


# # export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export NCCL_SOCKET_IFNAME=^docker0,lo


# srun --container-image="harbor.vinai-systems.com#research/lane_pytorch:1.8-cuda11-pytorch-lightning" \
#      --container-mounts=/lustre/scratch/client/lane_detection/:/workspace/lane_detection,/lustre/scratch/client/vinai/users/vietlq4/Event/:/workspace/Event \
#      --container-workdir /workspace/Event \
python train.py \
--name event_cnnlstm \
--num_classes 23 \
--backbone resnet101 \
--num_threads 16 \
--seed 2021 \
--train_root ~/datasets/CUFED/images \
--train_list filenames/train_single.txt \
--val_list filenames/val.txt \
--batch_size 4 \
--save_dir checkpoints \
--max_epoch 80 \
--optimizer adam \
--lr 1e-4 \
--weight_decay 1e-4 \
--lr_policy cosine \
--lr_milestones 30 50 70 90 100 110 \
--lr_gamma 0.5 \
--patience 20 \
--loss ce \
--gpus 0,1 \
--accelerator ddp