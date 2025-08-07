#!/bin/bash

# FR3 Reach Task - Learner Script
# This script runs the learner for policy training

export PYTHONPATH=$PYTHONPATH:/home/hanyu/code/hil-serl
export PYTHONPATH=$PYTHONPATH:/home/hanyu/code/HIROLRobotPlatform

# Use more GPU memory for learner
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3

# Run learner with FR3 reach configuration
python -m serl_launcher.serl_launcher \
    --config=experiments.fr3_reach.config:TrainConfig \
    --mode=learner \
    --ip=127.0.0.1 \
    --debug