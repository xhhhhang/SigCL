#!/bin/bash

MODEL="resnet50"
DATASET="imagenet"
LEARNING_RATE=2
LOGIT_LEARNING_RATE=0.2
BATCH_SIZE=1536
PRINT_FREQ=100
EPOCHS=350
LINEAR_EPOCHS=50

COMMAND="python src/supcl/main_supcon.py --method SigCLBase --init_logit_bias -10 --linear_epochs $LINEAR_EPOCHS --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --logit_learning_rate $LOGIT_LEARNING_RATE --temp 0.1 --cosine --warm --model $MODEL --print_freq $PRINT_FREQ --epochs $EPOCHS --dataset $DATASET"

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

# Machine 1
CUDA_VISIBLE_DEVICES=0,1,2,3 $COMMAND --method SigCLBase --neg_weight 250 & PID0=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 $COMMAND --method SigCLBase --neg_weight 1000 & PID1=$!

wait $PID0 $PID1

echo "All commands have finished executing."