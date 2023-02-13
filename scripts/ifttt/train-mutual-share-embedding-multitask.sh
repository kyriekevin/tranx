#!/bin/bash
set -e

seed=${1:-0}
vocab="data/ifttt/vocab.freq2.bin"
train_student_file="data/ifttt/train_student.bin"
train_teacher_file="data/ifttt/train_teacher.bin"
dev_file="data/ifttt/dev.bin"
test_file="data/ifttt/test.bin"
parser_student="default_student_parser"
parser="default_parser"
parser_teacher="default_teacher_parser"
dropout=0.5
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=64
type_embed_size=64
ptrnet_hidden_dim=32
lr=0.001
ls=0.1
lr_decay=0.9
lr_decay_after_epoch=6
beam_size=15
lstm='lstm'  
src_dropout=0.3
max_epoch=50
lambda_num=0
model_name=model.ifttt.seed${seed}.mutual.share.embedding.dropout${dropout}.src_dropout${src_dropout}.multitask

echo "**** Writing results to logs/ifttt/${model_name}.log ****"
mkdir -p logs/ifttt
echo commit hash: `git rev-parse HEAD` > logs/ifttt/${model_name}.log

python -u exp_distillation_share_embedding.py \
    --cuda \
    --seed ${seed} \
    --mode train \
    --batch_size 60 \
    --lambda_num ${lambda_num} \
    --evaluator ifttt_evaluator \
    --evaluator_student ifttt_evaluator \
    --evaluator_teacher ifttt_evaluator \
    --asdl_file asdl/lang/ifttt/ifttt_asdl.txt \
    --transition_system ifttt \
    --src_dropout ${src_dropout} \
    --dev_file ${dev_file} \
    --test_file ${test_file} \
    --train_student_file ${train_student_file} \
    --train_teacher_file ${train_teacher_file} \
    --parser ${parser} \
    --parser_student ${parser_student} \
    --parser_teacher ${parser_teacher} \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --primitive_token_label_smoothing ${ls} \
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
    --vaildate_begin_epoch 14 \
     --max_epoch ${max_epoch} \
    --glorot_init \
    --lr ${lr} \
    --lr_decay ${lr_decay} \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --decay_lr_every_epoch \
    --beam_size ${beam_size} \
    --log_every 50 \
    --save_to saved_models/ifttt/${model_name} 2>&1 | tee -a logs/ifttt/${model_name}.log
