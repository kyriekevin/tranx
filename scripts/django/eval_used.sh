#!/bin/bash

test_file="data/django/test.bin"
traverse_first="deep"  # breadth OR deep
model_name="model.django.seed0.bidecoder.rescore" # in a depth-first traversal. 

model_name2="model.django.seed0.dropout0.4.share.encoder.multitask.rl2"

python exp.py \
	--cuda \
    --mode result_eval \
    --traverse_first ${traverse_first} \
    --load_model saved_models/django/${model_name} \
    --beam_size 15 \
    --test_file ${test_file} \
    --save_decode_to decodes/django/${model_name}.test.decode \
    --decode_max_time_step 100
