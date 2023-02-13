#!/bin/bash
set -e

seed=${1:-0}
vocab="data/django/vocab.freq15.bin"
train_file="data/django/train_teacher.bin"
dev_file="data/django/dev.bin"
test_file="data/django/test.bin"
traverse_first="layer"
parser="default_parser"
dropout=0.3
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=64
type_embed_size=64
ptrnet_hidden_dim=32
max_epoch=40
lr=0.001
lr_decay=0.9
beam_size=15
lstm='lstm'  # lstm
lr_decay_after_epoch=8
model_name=model.django.seed${seed}.breadth.first

echo "**** Writing results to logs/django/${model_name}.log ****"
mkdir -p logs/django
echo commit hash: `git rev-parse HEAD` > logs/django/${model_name}.log

python -u exp.py \
    --cuda \
    --seed ${seed} \
    --mode train \
    --batch_size 10 \
    --asdl_file asdl/lang/py/py_asdl.txt \
    --transition_system python2 \
    --evaluator django_evaluator \
    --dev_file ${dev_file} \
    --traverse_first ${traverse_first} \
    --train_file ${train_file} \
    --test_file ${test_file} \
    --parser ${parser} \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --no_parent_field_type_embed \
    --no_parent_production_embed \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience 5 \
    --max_epoch ${max_epoch} \
    --lr ${lr} \
    --lr_decay ${lr_decay} \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --glorot_init \
    --beam_size ${beam_size} \
    --decay_lr_every_epoch \
    --log_every 50 \
    --vaildate_begin_epoch 18\
    --save_to saved_models/django/${model_name} 2>&1 | tee -a logs/django/${model_name}.log

. scripts/django/test.sh saved_models/django/${model_name}.bin 2>&1 | tee -a logs/django/${model_name}.log