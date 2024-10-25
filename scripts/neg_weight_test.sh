#!/bin/bash

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

MODEL="resnet18"
DATASET="cifar100"
LEARNING_RATE=2
LOGIT_LEARNING_RATE=0.2
BATCH_SIZE=3076
PRINT_FREQ=17
EPOCHS=700

COMMAND="python src/supcl/main_supcon.py --linear_epochs 50 --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --logit_learning_rate $LOGIT_LEARNING_RATE --temp 0.1 --cosine --warm --model $MODEL --print_freq $PRINT_FREQ --epochs $EPOCHS --dataset $DATASET"

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
# CUDA_VISIBLE_DEVICES=0 $COMMAND --method SigCLBase --neg_weight 1 & PID1=$!
CUDA_VISIBLE_DEVICES=0 $COMMAND --method SigCLBase_10 --neg_weight 10 --init_logit_bias -10 & PID2=$!
CUDA_VISIBLE_DEVICES=1 $COMMAND --method SigCLBase_100 --neg_weight 100 --init_logit_bias -10 & PID3=$!
# CUDA_VISIBLE_DEVICES=1 $COMMAND & PID3=$!
CUDA_VISIBLE_DEVICES=2 $COMMAND --method SigCLBase_1000 --neg_weight 1000 --init_logit_bias -10 & PID4=$!
CUDA_VISIBLE_DEVICES=3 $COMMAND --method SigCLBase_3076 --neg_weight 3076 --init_logit_bias -10 & PID5=$!
# CUDA_VISIBLE_DEVICES=5 python src/supcl/main_ce.py --batch_size $BATCH_SIZE --learning_rate 0.8 --cosine --warm --print_freq $PRINT_FREQ --epochs $EPOCHS --model $MODEL & PID6=$!

# $COMMAND --method SigCLBase_1000 --neg_weight 1000 --init_logit_bias -10 --batch_size 768 & PID4=$!
# Wait for all background processes to finish
wait $PID2 $PID3 $PID4 $PID5

echo "All commands have finished executing."
