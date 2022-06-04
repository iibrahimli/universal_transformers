python3 train_wmt14.py \
    --batch_size 3 --d_model 16 \
    --d_feedforward 32 \
    --wandb_project ut_scratch \
    --max_seq_len 8 \
    --max_time_step 8 \
    --tr_log_interval 1 \
    --val_size 2 \
    --val_interval 5 \
    --lr 0.001
