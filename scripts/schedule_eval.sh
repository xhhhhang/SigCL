#!/bin/bash

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh
#!/bin/bash

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

MODEL="resnet18"
BATCH_SIZE=512

COMMAND="python src/supcl/main_linear.py --model $MODEL --disable_progress --learning_rate 5 --batch_size $BATCH_SIZE"

# Append any extra arguments to the command
EXTRA_ARGS="$@"
COMMAND="$COMMAND $EXTRA_ARGS"

CUDA_VISIBLE_DEVICES=0 $COMMAND --method SigCLPN --ckpt ./save/SupCon/cifar10_models/SigCLPN_cifar10_resnet18_lr_0.5_decay_0.0001_bsz_3072_temp_0.1_trial_0_cosine_warm/last.pth &
CUDA_VISIBLE_DEVICES=1 $COMMAND --method SigCL --ckpt ./save/SupCon/cifar10_models/SigCL_cifar10_resnet18_lr_0.5_decay_0.0001_bsz_3072_temp_0.1_trial_0_cosine_warm/last.pth &
CUDA_VISIBLE_DEVICES=2 $COMMAND --method SigCLBase --ckpt ./save/SupCon/cifar10_models/SigCLBase_cifar10_resnet18_lr_0.5_decay_0.0001_bsz_3072_temp_0.1_trial_0_cosine_warm/last.pth &
CUDA_VISIBLE_DEVICES=3 $COMMAND --ckpt ./save/SupCon/cifar10_models/SupCon_cifar10_resnet18_lr_0.5_decay_0.0001_bsz_3072_temp_0.1_trial_0_cosine_warm/last.pth
