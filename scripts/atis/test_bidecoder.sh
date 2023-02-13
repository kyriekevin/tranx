#!/bin/bash
test_file="data/atis/data_split"
model_name="model.atis.seed2.bidecoder.rescore.r2l" # in a breadth-first manner. 

python exp_bidecoder_rescore.py \
    --mode test \
    --is_dir \
    --load_model saved_models/atis/${model_name} \
    --beam_size 5 \
    --test_file ${test_file} \
    --save_decode_to decodes/atis/${model_name}.test.decode \
    --decode_max_time_step 110



