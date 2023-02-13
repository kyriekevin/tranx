#!/bin/bash
test_file="data/atis/data_split"
model_name="model.atis.seed0.mutual.share.encoder.dropout0.3.src_dropout0.4.mutual.lambda0.5.teacher" # in a breadth-first manner. 
traverse_first="de" # breadth OR deep
python exp.py \
    --mode test \
    --is_dir \
    --traverse_first ${traverse_first} \
    --load_model saved_models/atis/${model_name}.bin \
    --beam_size 5 \
    --test_file ${test_file} \
    --save_decode_to decodes/atis/${model_name}.test.decode \
    --decode_max_time_step 110



