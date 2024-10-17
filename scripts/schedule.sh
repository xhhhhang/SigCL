#!/bin/bash


# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

MODEL="resnet18"
LEARNING_RATE=0.5
BATCH_SIZE=3072
PRINT_FREQ=8
EPOCHS=500

COMMAND="python src/supcl/main_supcon.py --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --temp 0.1 --cosine --warm --model $MODEL --print_freq $PRINT_FREQ --epochs $EPOCHS"

# Append any extra arguments to the command
EXTRA_ARGS="$@"
COMMAND="$COMMAND $EXTRA_ARGS"

# Run commands in background and save their PIDs
CUDA_VISIBLE_DEVICES=0 $COMMAND --method SigCLBase_1 & PID1=$!
CUDA_VISIBLE_DEVICES=1 $COMMAND --method SigCLBase_BS --init_logit_bias -10 --neg_weight $BATCH_SIZE & PID2=$!
CUDA_VISIBLE_DEVICES=2 $COMMAND --method SigCLBase_2BS --init_logit_bias -10 --neg_weight $((2*BATCH_SIZE)) & PID3=$!
CUDA_VISIBLE_DEVICES=3 $COMMAND --method SigCL --max_neg_weight $BATCH_SIZE & PID4=$!
CUDA_VISIBLE_DEVICES=4 $COMMAND & PID5=$!

CUDA_VISIBLE_DEVICES=5 python main_ce.py --batch_size $BATCH_SIZE --learning_rate 0.8 --cosine --warm --print_freq $PRINT_FREQ --epochs $EPOCHS --model $MODEL & PID6=$!

# Wait for all background processes to finish
wait $PID1 $PID2 $PID3 $PID4 $PID5 $PID6

echo "All commands have finished executing."