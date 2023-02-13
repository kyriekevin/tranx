#!/bin/bash
set -e

seed=${1:-0}
vocab="data/django/vocab.freq15.bin"
train_file="data/django/train_r2l.bin"
dev_file="data/django/dev.bin"
test_file="data/django/test.bin"
dropout=0.3
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=64
type_embed_size=64
ptrnet_hidden_dim=32
lr=0.001
lr_decay=0.5
beam_size=15
lstm='lstm'  # lstm
traverse_first="deep"
model_name=baseline.seed${seed}.r2l.new


echo "**** Writing results to logs/django/${model_name}.log ****"
mkdir -p logs/django
echo commit hash: `git rev-parse HEAD` > logs/django/${model_name}.log

python exp.py \
    --cuda \
    --seed ${seed} \
    --mode train \
    --batch_size 10 \
    --right_left \
    --vaildate_begin_epoch 10 \
    --asdl_file asdl/lang/py/py_asdl.txt \
    --transition_system python2 \
    --evaluator django_evaluator \
    --train_file ${train_file} \
    --dev_file ${dev_file} \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --traverse_first ${traverse_first} \
    --no_parent_field_type_embed \
    --no_parent_production_embed \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience 5 \
    --max_num_trial 5 \
    --glorot_init \
    --lr ${lr} \
    --lr_decay ${lr_decay} \
    --beam_size ${beam_size} \
    --log_every 50 \
    --save_to saved_models/django/${model_name} 2>&1 | tee -a logs/django/${model_name}.log

. scripts/django/test.sh saved_models/django/${model_name}.bin 2>&1 | tee -a logs/django/${model_name}.log
