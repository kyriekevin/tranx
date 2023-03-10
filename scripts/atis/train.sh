#!/bin/bash
set -e

seed=${1:-0}
vocab="vocab.freq2.bin"
train_file="train_teacher.bin"
dev_file="dev.bin"
test_file="test.bin"
traverse_first="layer"
dropout=0.3
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=32
type_embed_size=32
lr_decay=0.5
beam_size=5
lstm='lstm'
ls=0.1
max_epoch=80
model_name=model.atis.baseline.seed${seed}.${traverse_first}

echo "**** Writing results to logs/atis/${model_name}.log ****"
mkdir -p logs/atis
echo commit hash: `git rev-parse HEAD` > logs/atis/${model_name}.log

python -u exp.py \
    --cuda \
    --seed ${seed} \
    --mode train \
    --batch_size 10 \
    --asdl_file asdl/lang/lambda_dcs/lambda_asdl.txt \
    --transition_system lambda_dcs \
    --traverse_first ${traverse_first} \
    --train_file data/atis/${train_file} \
    --dev_file data/atis/${dev_file} \
    --test_file data/atis/${test_file} \
    --vocab data/atis/${vocab} \
    --lstm ${lstm} \
    --primitive_token_label_smoothing ${ls} \
    --no_parent_field_type_embed \
    --no_parent_production_embed \
    --hidden_size ${hidden_size} \
    --att_vec_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience 5 \
    --max_num_trial 5 \
    --glorot_init \
    --no_copy \
    --max_epoch ${max_epoch} \
    --lr_decay ${lr_decay} \
    --vaildate_begin_epoch 8\
    --beam_size ${beam_size} \
    --decode_max_time_step 110 \
    --log_every 50 \
    --save_to saved_models/atis/${model_name} 2>&1 | tee -a logs/atis/${model_name}.log

. scripts/atis/test.sh saved_models/atis/${model_name}.bin 2>&1 | tee -a logs/atis/${model_name}.log
