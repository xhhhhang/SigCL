#!/bin/bash

python src/supcl/main_supcon.py --batch_size 1024 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine \
  --warm

python src/supcl/main_supcon.py --batch_size 3072 --model resnet18 --epochs 100 --print_freq 1

python src/supcl/main_supcon.py --batch_size 3072 --model resnet18 --epochs 100 --print_freq 1 --warm --learning_rate 0.5 --cosine --method SigCLBase_BS --init_logit_bias -10 --neg_weight 3072 --bidir

python src/supcl/main_supcon.py --batch_size 512 --model resnet50 --dataset cifar100 --epochs 1000 --warm --learning_rate 0.5 --cosine --method SigCLBase_BS --init_logit_bias -10 --neg_weight 512 --bidir --compile
