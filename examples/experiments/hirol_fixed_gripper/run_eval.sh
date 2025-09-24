#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=hirol_fixed_gripper \
    --checkpoint_path=first_run \
    --actor \
    --eval_checkpoint_step=70000 \
    --eval_n_trajs=200 \
    --save_video