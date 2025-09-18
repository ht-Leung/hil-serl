export XLA_PYTHON_CLIENT_PREALLOCATE=false && \

export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=hirol_pick_place \
    --checkpoint_path=first_run \
    --demo_path=/home/hanyu/code/hil-serl/examples/experiments/hirol_pick_place/demo_data/hirol_pick_place_20_demos_2025-09-15_18-21-08.pkl \
    --learner \