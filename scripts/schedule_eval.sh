#!/bin/bash

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh
#!/bin/bash

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

MODEL="resnet18"
LINEAR_BATCH_SIZE=1024
EPOCHS=25
DATASET="cifar100"

COMMAND="python src/supcl/main_linear.py --model $MODEL --disable_progress --learning_rate 5 --batch_size $LINEAR_BATCH_SIZE --epochs $EPOCHS --dataset $DATASET"

# Append any extra arguments to the command
EXTRA_ARGS="$@"
COMMAND="$COMMAND $EXTRA_ARGS"

# Function to kill all child processes
cleanup() {
    echo "Killing all child processes..."
    pkill -P $$
    exit 1
}

# Set up trap to call cleanup function if the script receives SIGINT or SIGTERM
trap cleanup SIGINT SIGTERM


CUDA_VISIBLE_DEVICES=0 $COMMAND --method SigCLBase_10 --ckpt "/renhangx/SigCL/save/SupCon/cifar100_models/SigCLBase_10_cifar100_resnet18_lr_1.0_decay_0.0001_bsz_3076_temp_0.1_trial_0_cosine_base_10.0_warm/last.pth" & PID1=$!
CUDA_VISIBLE_DEVICES=1 $COMMAND --method SigCLBase_100 --ckpt "/renhangx/SigCL/save/SupCon/cifar100_models/SigCLBase_100_cifar100_resnet18_lr_1.0_decay_0.0001_bsz_3076_temp_0.1_trial_0_cosine_base_100.0_warm/last.pth" & PID2=$!
# CUDA_VISIBLE_DEVICES=0 $COMMAND --method SigCLBase_10 --ckpt "/renhangx/SigCL/save/SupCon/cifar100_models/SigCLBase_10_cifar100_resnet18_lr_2.0_decay_0.0001_bsz_3076_temp_0.1_trial_0_cosine_base_10.0_warm/last.pth" & PID1=$!
# CUDA_VISIBLE_DEVICES=1 $COMMAND --method SigCLBase_100 --ckpt "/renhangx/SigCL/save/SupCon/cifar100_models/SigCLBase_100_cifar100_resnet18_lr_2.0_decay_0.0001_bsz_3076_temp_0.1_trial_0_cosine_base_100.0_warm/last.pth" & PID2=$!
# CUDA_VISIBLE_DEVICES=2 $COMMAND --method SigCLBase_1000 --ckpt "/renhangx/SigCL/save/SupCon/cifar100_models/SigCLBase_1000_cifar100_resnet18_lr_2.0_decay_0.0001_bsz_3076_temp_0.1_trial_0_cosine_base_1000.0_warm/last.pth" & PID3=$!
# CUDA_VISIBLE_DEVICES=3 $COMMAND --method SigCLBase_3076 --ckpt "/renhangx/SigCL/save/SupCon/cifar100_models/SigCLBase_3076_cifar100_resnet18_lr_2.0_decay_0.0001_bsz_3076_temp_0.1_trial_0_cosine_base_3076.0_warm/last.pth" & PID4=$!
# CUDA_VISIBLE_DEVICES=3 $COMMAND --ckpt ./save/SupCon/cifar100_models/SupCon_cifar100_resnet18_lr_0.5_decay_0.0001_bsz_3072_temp_0.1_trial_0_cosine_warm/last.pth



# Wait for all background processes to finish
wait $PID1 $PID2 $PID3 $PID4

echo "All commands have finished executing."
