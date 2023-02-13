#!/bin/bash

test_file="data/django/test.bin"
traverse_first="deep"  # breadth OR deep
model_name="model.sup.django.pretrain.seed0" # in a depth-first traversal. 
python exp.py \
	--cuda \
    --mode test \
    --traverse_first ${traverse_first} \
    --load_model saved_models/django/${model_name}.bin \
    --beam_size 15 \
    --test_file ${test_file} \
    --save_decode_to decodes/django/${model_name}.test.decode \
    --decode_max_time_step 100
