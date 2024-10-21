#!/bin/bash

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh
#!/bin/bash

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

MODEL="resnet18"
LINEAR_BATCH_SIZE=1024
EPOCHS=50
DATASET="cifar100"

COMMAND="python src/supcl/main_linear.py --model $MODEL --disable_progress --learning_rate 5 --batch_size $LINEAR_BATCH_SIZE --epochs $EPOCHS --dataset $DATASET"

# Append any extra arguments to the command
EXTRA_ARGS="$@"
COMMAND="$COMMAND $EXTRA_ARGS"

CUDA_VISIBLE_DEVICES=0 $COMMAND --method SigCLBase_BS --ckpt ./save/SupCon/cifar100_models/SigCLBase_BS_cifar100_resnet18_lr_0.5_decay_0.0001_bsz_3072_temp_0.1_trial_0_cosine_warm/last.pth &
CUDA_VISIBLE_DEVICES=1 $COMMAND --method SigCLBase_2BS --ckpt ./save/SupCon/cifar100_models/SigCLBase_2BS_cifar100_resnet18_lr_0.5_decay_0.0001_bsz_3072_temp_0.1_trial_0_cosine_warm/last.pth &
CUDA_VISIBLE_DEVICES=2 $COMMAND --method SigCL --ckpt ./save/SupCon/cifar100_models/SigCL_cifar100_resnet18_lr_0.5_decay_0.0001_bsz_3072_temp_0.1_trial_0_cosine_warm/last.pth &
CUDA_VISIBLE_DEVICES=3 $COMMAND --ckpt ./save/SupCon/cifar100_models/SupCon_cifar100_resnet18_lr_0.5_decay_0.0001_bsz_3072_temp_0.1_trial_0_cosine_warm/last.pth