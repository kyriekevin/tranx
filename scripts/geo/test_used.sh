#!/bin/bash
test_file="data/geo/data_split"
model_name="model.geo.seed0.dropout0.5.src_dropout0.5.mutual.encoder.lambda0.25.r2l.teacher" # in a depth-first traversal. 
traverse_first="deep" # breadth OR deep

python exp.py \
    --mode test \
    --traverse_first ${traverse_first} \
    --right_left \
    --load_model saved_models/geo/${model_name}.bin \
    --is_dir \
    --beam_size 5 \
    --test_file ${test_file} \
    --save_decode_to decodes/geo/${model_name}.test.decode \
    --decode_max_time_step 110


