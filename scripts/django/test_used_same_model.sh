#!/bin/bash

test_file="data/django/test.bin"
traverse_first="layer"  # breadth OR deep
model_name="model.django.seed2020.pretain.lambda0.5.dropout0.4.no.share.layer" # in a depth-first traversal. 
python exp_distillation_share_encoder_same_model.py  \
	--cuda \
    --mode test \
    --traverse_first ${traverse_first} \
    --load_model saved_models/django/${model_name}\
    --beam_size 15 \
    --test_file ${test_file} \
    --save_decode_to decodes/django/${model_name}.test.decode \
    --decode_max_time_step 100
