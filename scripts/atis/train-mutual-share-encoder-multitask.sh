#!/bin/bash
set -e

seed=${1:-0}
vocab="data/atis/vocab.freq2.bin"
train_student_file="data/atis/train_student.bin"
train_teacher_file="data/atis/train_teacher.bin"
train_file="data/atis/train.bin"
dev_file="data/atis/dev.bin"
#test_file="data/atis/test.bin"
parser_student="default_student_parser"
parser="default_parser"
parser_teacher="default_teacher_parser"
dropout=0.3
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=32
type_embed_size=32
lr_decay=0.5
beam_size=5
src_dropout=0.4
lstm='lstm'
ls=0.1
lambda_num=0
max_epoch=10
model_name=model.atis.seed${seed}.mutual.share.encoder.dropout${dropout}.src_dropout${src_dropout}.multitask.pretrain

echo "**** Writing results to logs/atis/${model_name}.log ****"
mkdir -p logs/atis
echo commit hash: `git rev-parse HEAD` > logs/atis/${model_name}.log

python -u exp_distillation_share_encoder.py \
    --cuda \
    --seed ${seed} \
    --mode train \
    --batch_size 10 \
    --src_dropout ${src_dropout} \
    --lambda_num ${lambda_num} \
    --asdl_file asdl/lang/lambda_dcs/lambda_asdl.txt \
    --transition_system lambda_dcs \
    --dev_file ${dev_file} \
    --train_file ${train_file} \
    --vaildate_begin_epoch 10\
    --train_student_file ${train_student_file} \
    --train_teacher_file ${train_teacher_file} \
    --parser ${parser} \
    --parser_student ${parser_student} \
    --parser_teacher ${parser_teacher} \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --src_dropout ${src_dropout} \
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
    --max_epoch ${max_epoch} \
    --glorot_init \
    --lr_decay ${lr_decay} \
    --beam_size ${beam_size} \
    --decode_max_time_step 110 \
    --log_every 50 \
    --save_to saved_models/atis/${model_name} 2>&1 | tee -a logs/atis/${model_name}.log
