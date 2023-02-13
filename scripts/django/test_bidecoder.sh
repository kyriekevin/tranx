#!/bin/bash

test_file="data/django/data_split"
traverse_first="deep"  # breadth OR deep
model_name="model.django.seed0.bidecoder.rescore.r2l" # in a depth-first traversal. 
python exp_bidecoder_rescore.py \
	--cuda \
    --mode test \
    --load_model saved_models/django/${model_name} \
    --beam_size 15 \
    --is_dir \
    --test_file ${test_file} \
    --save_decode_to decodes/django/${model_name}.test.decode \
    --decode_max_time_step 100
