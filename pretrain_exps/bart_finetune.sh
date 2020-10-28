#!/bin/bash
MODEL_PATH=""
CONFIG_PATH=""
TOKENIZER_NAME=""
DATA_DIR=""
OUTPUT_DIR=""
LOGGING_DIR=""
RUN_NAME=""
export MAX_LEN=128
export BS=4

python bart_finetune_trainer.py \
--model_name_or_path $MODEL_PATH \
--config_name $CONFIG_PATH \
--tokenizer_name $TOKENIZER_NAME \
--output_dir $OUTPUT_DIR \
--overwrite_output_dir \
--data_dir $DATA_DIR \
--embedding_extension \
--task "semparse" \
--max_source_length "" \
--max_target_length "" \
--eval_beams 5 \
--do_train \
--do_eval \
--do_predict \
--evaluate_during_training \
--evaluation_strategy EPOCH \
--per_device_train_batch_size 10 \
--per_device_train_batch_size 20 \
--gradient_accumulation_steps 4 \
--eval_accumulation_steps 2 \
--learning_rate 1e-4 \
--weight_decay 0.01 \
--adam_beta1 0.99 \
--adam_beta2 0.999 \
--adam_epsilon 1e-08 \
--max_grad_norm 0.1 \
--max_steps 20000 \
--warmup_steps 500 \
--logging_dir $LOGGING_DIR \
--logging_steps 100 \
--save_steps 1000 \
--save_total_limit 2 \
--seed 1 \
--local_rank 0 \
--dataloader_drop_last \
--dataloader_num_workers 16 \
--run_name ${RUN_NAME} \
--load_best_model_at_end \
--metric_for_best_model "loss" \
--label_smoothing 0.1 \
--sortish_sampler \
--predict_with_generate \
--dropout 0.1 \
--attention_dropout 0.1 \
--lr_scheduler "polynomial"

#
# python -m torch.distributed.launch --nproc_per_node=8  run_distributed_eval.py \
#     --model_name sshleifer/distilbart-large-xsum-12-3  \
#     --save_dir xsum_generations \
#     --data_dir xsum \
#     --fp16  # you can pass generate kwargs like num_beams here, just like run_eval.py
