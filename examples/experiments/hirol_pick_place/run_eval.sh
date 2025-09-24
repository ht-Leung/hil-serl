#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=hirol_pick_place \
    --checkpoint_path=first_run \
    --actor \
    --eval_checkpoint_step=25000 \
    --eval_n_trajs=20 \
    --save_video