#!/bin/bash

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh
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

CUDA_VISIBLE_DEVICES=0 $COMMAND --method SigCLPN --init_logit_bias -10 &
CUDA_VISIBLE_DEVICES=1 $COMMAND --method SigCL --max_neg_weight $BATCH_SIZE &
CUDA_VISIBLE_DEVICES=2 $COMMAND --method SigCLBase &
CUDA_VISIBLE_DEVICES=3 $COMMAND

CUDA_VISIBLE_DEVICES=3 python main_ce.py --batch_size $BATCH_SIZE --learning_rate 0.8 --cosine --warm --print_freq $PRINT_FREQ --epochs $EPOCHS --model $MODEL
