export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=fr3_reach \
    --checkpoint_path=first_run \
    --demo_path=/home/hanyu/code/hil-serl/examples/demo_data/fr3_reach_20_demos_2025-08-14_10-54-30.pkl \
    --learner \