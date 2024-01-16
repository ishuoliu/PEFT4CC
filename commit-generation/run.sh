#!/bin/bash

lang=$1

python run.py \
   --train_data_file ../datasets/MCMD/javascript/contextual_medits/train.jsonl .. \
   --eval_data_file ../datasets/MCMD/javascript/contextual_medits/valid.jsonl .. \
   --test_data_file ../datasets/MCMD/javascript/contextual_medits/test.jsonl .. \
   --output_dir ../results/mcmd/javascript/codet5/single/checkpoints \
   --pretrained_model codet5 \
   --gradient_accumulation_steps 2 \
   --batch_size 16 \
   --learning_rate 5e-4 \
   --epochs 10 \
   --do_train


python run_lora.py \
   --train_data_file ../datasets/MCMD/javascript/contextual_medits/train.jsonl .. \
   --eval_data_file ../datasets/MCMD/javascript/contextual_medits/valid.jsonl .. \
   --test_data_file ../datasets/MCMD/javascript/contextual_medits/test.jsonl .. \
   --output_dir ../results/mcmd_lora/javascript/codet5/single/checkpoints \
   --pretrained_model codet5 \
   --gradient_accumulation_steps 2 \
   --batch_size 16 \
   --learning_rate 5e-4 \
   --epochs 10 \
   --do_train\
   --use_lora


python run_adapter.py \
    --train_data_file ../datasets/MCMD/javascript/contextual_medits/train.jsonl .. \
    --eval_data_file ../datasets/MCMD/javascript/contextual_medits/valid.jsonl .. \
    --test_data_file ../datasets/MCMD/javascript/contextual_medits/test.jsonl .. \
    --output_dir ../results/mcmd_adapter/javascript/codet5/single/checkpoints \
    --pretrained_model codet5 \
    --gradient_accumulation_steps 2 \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --epochs 10 \
    --do_train


