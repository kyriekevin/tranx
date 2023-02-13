#!/bin/bash
set -e

seed=${1:-0}
vocab="data/django/vocab.freq15.bin"
train_file="None"
dev_file="data/django/dev.bin"
evaluator_student="default_evaluator"
evaluator_teacher="default_evaluator"
train_student_file="data/django/train_teacher.bin"
train_teacher_file="data/django/train_student.bin"
parser_student="default_student_parser"
parser="default_student_parser"
parser_teacher="default_teacher_parser"
traverse_first="layer"
dropout=0.4
uniform_init=0.001
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=64
type_embed_size=64
ptrnet_hidden_dim=32
lr_decay_after_epoch=8
lr=0.001
lr_decay=0.90
beam_size=15
max_epoch=50
func="KL"
lstm='lstm'  # lstm
lambda=0.5
src_dropout=0.0
pretrain_epoch=4
model_name=model.django.seed${seed}.pretain.lambda${lambda}.dropout${dropout}.no.share.${traverse_first}.one.way

echo "**** Writing results to logs/django/${model_name}.log ****"
mkdir -p logs/django
echo commit hash: `git rev-parse HEAD` > logs/django/${model_name}.log

python -u exp_distillation_share_encoder_one_way.py \
    --cuda \
    --seed ${seed} \
    --mode train \
    --lambda_num ${lambda} \
    --batch_size 10 \
    --func ${func} \
    --pretrain_teacher saved_models/django/baseline.seed${seed}.deep.bin \
    --traverse_first ${traverse_first} \
    --vaildate_begin_epoch 15\
    --src_dropout ${src_dropout} \
    --pretrain_epoch ${pretrain_epoch} \
    --asdl_file asdl/lang/py/py_asdl.txt \
    --transition_system python2 \
    --evaluator django_evaluator \
    --train_student_file ${train_student_file} \
    --train_teacher_file ${train_teacher_file} \
    --train_file ${train_file} \
    --dev_file ${dev_file}\
    --parser_student ${parser_student} \
    --parser_teacher ${parser_teacher} \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --max_epoch ${max_epoch} \
    --no_parent_field_type_embed \
    --no_parent_production_embed \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience 20 \
    --max_num_trial 5 \
    --glorot_init \
    --lr ${lr} \
    --lr_decay ${lr_decay} \
    --decay_lr_every_epoch \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --beam_size ${beam_size} \
    --log_every 50 \
    --save_to saved_models/django/${model_name} 2>&1 | tee -a logs/django/${model_name}.log

