#!/bin/bash
set -e

seed=${1:-0}
vocab="data/ifttt/vocab.freq2.bin"
train_file="data/ifttt/train_teacher.bin"
dev_file="data/ifttt/dev.bin"
test_file="data/ifttt/test.bin"
traverse_first="layer"
dropout=0.5
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=64
type_embed_size=64
ptrnet_hidden_dim=32
lr=0.001
ls=0.1
lr_decay=0.5
lr_decay_after_epoch=5
beam_size=15
lstm='lstm'  # lstm
model_name=model.baseline.ifttt.seed${seed}.breadth
echo "**** Writing results to logs/ifttt/${model_name}.log ****"
mkdir -p logs/ifttt
echo commit hash: `git rev-parse HEAD` > logs/ifttt/${model_name}.log

python -u exp.py \
    --cuda \
    --seed ${seed} \
    --mode train \
    --batch_size 60 \
    --traverse_first  ${traverse_first} \
    --evaluator ifttt_evaluator \
    --asdl_file asdl/lang/ifttt/ifttt_asdl.txt \
    --transition_system ifttt \
    --dev_file ${dev_file} \
    --test_file ${test_file} \
    --train_file ${train_file} \
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
    --max_num_trial 3 \
    --vaildate_begin_epoch 6 \
    --glorot_init \
    --lr ${lr} \
    --lr_decay ${lr_decay} \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --decay_lr_every_epoch \
    --beam_size ${beam_size} \
    --log_every 50 \
    --save_to saved_models/ifttt/${model_name} 2>&1 | tee -a logs/ifttt/${model_name}.log

. scripts/ifttt/test.sh saved_models/ifttt/${model_name}.bin 2>&1 | tee -a logs/ifttt/${model_name}.log

