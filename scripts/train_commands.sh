#!/bin/bash

python src/supcl/main_supcon.py --batch_size 1024 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine \
  --warm

python src/supcl/main_supcon.py --batch_size 3072 --model resnet18 --epochs 100 --print_freq 1

python src/supcl/main_supcon.py --batch_size 3072 --model resnet18 --epochs 100 --print_freq 1 --warm --learning_rate 0.5 --cosine --method SigCLBase_BS --init_logit_bias -10 --neg_weight 3072 --bidir

python src/supcl/main_supcon.py --batch_size 512 --model resnet50 --dataset cifar100 --epochs 1000 --warm --learning_rate 0.5 --cosine --method SigCLBase_BS --init_logit_bias -10 --neg_weight 512 --bidir --compile

CUDA_VISIBLE_DEVICES=0 python src/supcl/main_linear.py --model resnet50 --disable_progress --learning_rate 5 --batch_size 1024 --epochs 50 --dataset cifar100 \
  --method SigCLBase_BS --ckpt /renhangx/SigCL/save/SupCon/cifar100_models/SigCLBase_BS_cifar100_resnet50_lr_0.1_decay_0.0001_bsz_578_temp_0.07_trial_0_cosine_base578.0_warm/last.pth

# CUDA_VISIBLE_DEVICES=0,1 python src/supcl/main_supcon.py --batch_size 1536 --model resnet18 --dataset cifar100 --epochs 700 --print_freq 17 --warm --learning_rate 2 --cosine --method SigCLBaseAvg --init_logit_bias -10 --neg_weight 3076 --bidir --linear_epochs 50 --num_workers=9 --logit_learning_rate 0.2 --log_wandb
CUDA_VISIBLE_DEVICES=0,1,2 python src/supcl/main_supcon.py --batch_size 1024 --model resnet18 --dataset cifar100 --epochs 700 --print_freq 17 --warm --learning_rate 2 --cosine --method SigCLBaseAvgV2 --init_logit_bias -10 --neg_weight 3076 --bidir --linear_epochs 30 --num_workers=12 --logit_learning_rate 0.2 --log_wandb

CUDA_VISIBLE_DEVICES=1,3 python src/supcl/main_supcon.py --batch_size 1536 --model resnet18 --dataset cifar100 --epochs 700 --print_freq 17 --warm --learning_rate 2 --cosine --method FocalAverage --init_logit_bias -10 --neg_weight 3076 --bidir --linear_epochs 700 --num_workers=9 --logit_learning_rate 0.2 --log_wandb
CUDA_VISIBLE_DEVICES=1,3 python src/supcl/main_supcon.py --batch_size 1536 --model resnet18 --dataset cifar100 --epochs 700 --print_freq 17 --warm --learning_rate 2 --cosine --method FocalAverageNorm --normalize_focal --init_logit_bias -10 --neg_weight 3076 --bidir --linear_epochs 700 --num_workers 9 --logit_learning_rate 0.2 --log_wandb
