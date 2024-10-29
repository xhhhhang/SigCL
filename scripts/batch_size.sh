
#!/bin/bash

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

MODEL="resnet18"
DATASET="cifar100"
LEARNING_RATE=2
NEG_WEIGHT=3076
LOGIT_LEARNING_RATE=0.2
PRINT_FREQ=17
EPOCHS=700
LINEAR_EPOCHS=30
TRIAL="Oct27"

COMMAND_SUP="python src/supcl/main_supcon.py --neg_weight $NEG_WEIGHT --linear_epochs $LINEAR_EPOCHS --learning_rate $LEARNING_RATE --temp 0.1 --cosine --warm --model $MODEL --print_freq $PRINT_FREQ --epochs $EPOCHS --dataset $DATASET --trial $TRIAL"
COMMAND="$COMMAND_SUP --logit_learning_rate $LOGIT_LEARNING_RATE "

# echo $COMMAND
# exit
# python src/supcl/main_supcon.py --neg_weight 3076 --linear_epochs 30 --learning_rate 2 --temp 0.1 --cosine --warm --model resnet18 --print_freq 17 --epochs 700 --dataset cifar100 --trial Oct27 --logit_learning_rate 0.2

# Append any extra arguments to the command
EXTRA_ARGS="$@"
COMMAND="$COMMAND $EXTRA_ARGS"
COMMAND_SUP="$COMMAND_SUP $EXTRA_ARGS"

# Function to kill all child processes
cleanup() {
    echo "Killing all child processes..."
    pkill -P $$
    exit 1
}

# Set up trap to call cleanup function if the script receives SIGINT or SIGTERM
trap cleanup SIGINT SIGTERM

CUDA_VISIBLE_DEVICES=0 $COMMAND --method SigCLBase --init_logit_bias -10 --batch_size 512 & PID0=$!
CUDA_VISIBLE_DEVICES=1 $COMMAND --method SigCLBase --init_logit_bias -10 --batch_size 1024 & PID1=$!
CUDA_VISIBLE_DEVICES=2 $COMMAND --method SigCLBase --init_logit_bias -10 --batch_size 2048 & PID2=$!
CUDA_VISIBLE_DEVICES=3 $COMMAND --method SigCLBase --init_logit_bias -10 --batch_size 6144 & PID3=$!

wait $PID0 $PID1 $PID2 $PID3

CUDA_VISIBLE_DEVICES=0 $COMMAND --method SigCLBaseAvg --init_logit_bias -10 --batch_size 512 & PID4=$!
CUDA_VISIBLE_DEVICES=1 $COMMAND --method SigCLBaseAvg --init_logit_bias -10 --batch_size 1024 & PID5=$!
CUDA_VISIBLE_DEVICES=2 $COMMAND --method SigCLBaseAvg --init_logit_bias -10 --batch_size 2048 & PID6=$!
CUDA_VISIBLE_DEVICES=3 $COMMAND --method SigCLBaseAvg --init_logit_bias -10 --batch_size 6144 & PID7=$!

wait $PID4 $PID5 $PID6 $PID7

CUDA_VISIBLE_DEVICES=0 $COMMAND_SUP  --batch_size 512 & PID8=$!
CUDA_VISIBLE_DEVICES=1 $COMMAND_SUP  --batch_size 1024 & PID9=$!
CUDA_VISIBLE_DEVICES=2 $COMMAND_SUP  --batch_size 3072 & PID10=$!
# CUDA_VISIBLE_DEVICES=3 $COMMAND_SUP  --init_logit_bias -10 --batch_size 6144 & PID11=$!

wait $PID8 $PID9 $PID10 # $PID11

echo "All commands have finished executing."

# used command
# bash scripts/exp_gamma_test.sh --log_tensorboard --log_wandb --print_freq 17 --dataset cifar10 || true;
# bash scripts/exp_gamma_test.sh --log_tensorboard --log_wandb --print_freq 17 || true;

# bash scripts/exp_gamma_test.sh --log_tensorboard --log_wandb --print_freq 17 --dataset cifar10 --learning_rate 5 || true;
# bash scripts/exp_gamma_test.sh --log_tensorboard --log_wandb --print_freq 17 --learning_rate 5 || true;

# CUDA_VISIBLE_DEVICES=0,1 python src/supcl/main_supcon.py --neg_weight 3076 --linear_epochs 30 --learning_rate 2 --temp 0.1 --cosine --warm --model resnet18 --print_freq 17 --epochs 700 --dataset cifar100 --trial Oct27 --logit_learning_rate 0.2 --method SigCLBase --init_logit_bias -10 --batch_size 512 --cosine_logit --log_wandb