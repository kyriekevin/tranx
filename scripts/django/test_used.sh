#!/bin/bash

test_file="data/django/data_split"
traverse_first="breadth"  # breadth OR deep
model_name="model.django.seed0.dropout0.4.share.encoder.mutual.pretrain.lambda0.0.teacher" # in a depth-first traversal. 
python exp.py \
	--cuda \
    --mode test \
    --is_dir \
    --traverse_first ${traverse_first} \
    --load_model saved_models/django/${model_name}.bin \
    --beam_size 15 \
    --test_file ${test_file} \
    --save_decode_to decodes/django/${model_name}.test.decode \
    --decode_max_time_step 100


#    --right_left \