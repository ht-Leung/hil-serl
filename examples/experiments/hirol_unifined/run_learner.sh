export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
t XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=hirol_unifined \
    --checkpoint_path=first_run \
    --demo_path=//home/hanyu/code/hil-serl/examples/experiments/hirol_unifined/demo_data/hirol_unifined_20_demos_2025-08-26_19-27-01.pkl \
    --learner \