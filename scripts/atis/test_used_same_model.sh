#!/bin/bash
test_file="data/atis/test.bin"
model_name="model.atis.seed2020.mutual.no.share.dropout0.3.src_dropout0.4.mutual.lambda0.5.layer" # in a breadth-first manner. 
traverse_first="layer" # breadth OR deep
python exp_distillation_share_encoder_same_model.py \
    --mode test \
    --traverse_first ${traverse_first} \
    --load_model saved_models/atis/${model_name} \
    --beam_size 5 \
    --test_file ${test_file} \
    --save_decode_to decodes/atis/${model_name}.test.decode \
    --decode_max_time_step 110



