#!/bin/bash
test_file="data/geo/test.bin"
model_name="model.geo.seed2020.dropout0.5.src_dropout0.5.no.share.lambda0.25.layer" # in a depth-first traversal. 
traverse_first="layer" # breadth OR deep

python exp_distillation_share_encoder_same_model.py  \
    --mode test \
    --traverse_first ${traverse_first} \
    --load_model saved_models/geo/${model_name} \
    --beam_size 5 \
    --test_file ${test_file} \
    --save_decode_to decodes/geo/${model_name}.test.decode \
    --decode_max_time_step 110


