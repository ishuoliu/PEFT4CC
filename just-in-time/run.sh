#!/bin/bash


python run.py \
   --train_data_file ../datasets/jitfine/changes_train.pkl ../datasets/jitfine/features_train.pkl \
   --eval_data_file ../datasets/jitfine/changes_valid.pkl ../datasets/jitfine/features_valid.pkl \
   --test_data_file ../datasets/jitfine/changes_test.pkl ../datasets/jitfine/features_test.pkl \
   --output_dir ../results/jitfine/plbart/single/checkpoints \
   --pretrained_model plbart \
   --epochs 10 \
   --do_train \


python run_lora.py \
   --train_data_file ../datasets/jitfine/changes_train.pkl ../datasets/jitfine/features_train.pkl \
   --eval_data_file ../datasets/jitfine/changes_valid.pkl ../datasets/jitfine/features_valid.pkl \
   --test_data_file ../datasets/jitfine/changes_test.pkl ../datasets/jitfine/features_test.pkl \
   --output_dir ../results/jitfine_lora/codet5/single/checkpoints \
   --pretrained_model codet5 \
   --learning_rate 1e-4 \
   --epochs 10 \
   --do_train \
   --use_lora


python run_adapter.py \
   --train_data_file ../datasets/jitfine/changes_train.pkl ../datasets/jitfine/features_train.pkl \
   --eval_data_file ../datasets/jitfine/changes_valid.pkl ../datasets/jitfine/features_valid.pkl \
   --test_data_file ../datasets/jitfine/changes_test.pkl ../datasets/jitfine/features_test.pkl \
   --output_dir ../results/jitfine_adapter/codebert/single/checkpoints \
   --pretrained_model codebert \
   --learning_rate 2e-5 \
   --epochs 10 \
   --do_train \
