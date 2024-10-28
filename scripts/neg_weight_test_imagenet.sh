#!/bin/bash

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

MODEL="resnet50"
DATASET="imagenet"
LEARNING_RATE=2
LOGIT_LEARNING_RATE=0.2
BATCH_SIZE=2048
PRINT_FREQ=100
EPOCHS=350
LINEAR_EPOCHS=50

COMMAND="python src/supcl/main_supcon.py --init_logit_bias -10 --linear_epochs $LINEAR_EPOCHS --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --logit_learning_rate $LOGIT_LEARNING_RATE --temp 0.1 --cosine --warm --model $MODEL --print_freq $PRINT_FREQ --epochs $EPOCHS --dataset $DATASET"

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

# Run commands in background and save their PIDs
CUDA_VISIBLE_DEVICES=0 $COMMAND --method SigCLBase --neg_weight 250 & PID0=$!
CUDA_VISIBLE_DEVICES=1 $COMMAND --method SigCLBase --neg_weight 500 & PID1=$!
CUDA_VISIBLE_DEVICES=2 $COMMAND --method SigCLBase --neg_weight 1000 & PID2=$!
CUDA_VISIBLE_DEVICES=3 $COMMAND --method SigCLBase --neg_weight 2000 & PID3=$!
CUDA_VISIBLE_DEVICES=4 $COMMAND --method SigCLBase --neg_weight 4000 & PID4=$!
CUDA_VISIBLE_DEVICES=5 $COMMAND --method SigCLBase --neg_weight 8000 & PID5=$!
CUDA_VISIBLE_DEVICES=6 $COMMAND --method SigCLBase --neg_weight 16000 & PID6=$!
CUDA_VISIBLE_DEVICES=7 $COMMAND --method SigCLBase --neg_weight 32000 & PID7=$!

# Wait for all background processes to finish
wait $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7


echo "All commands have finished executing."

# used command
# bash scripts/neg_weight_test.sh --log_tensorboard --log_wandb --print_freq 17 || true;
# bash scripts/neg_weight_test.sh --log_tensorboard --log_wandb --print_freq 17 --dataset cifar10 || true;

# bash scripts/neg_weight_test.sh --log_tensorboard --log_wandb --print_freq 17 --learning_rate 5 || true;
# bash scripts/neg_weight_test.sh --log_tensorboard --log_wandb --print_freq 17 --dataset cifar10 --learning_rate 5 || true;
