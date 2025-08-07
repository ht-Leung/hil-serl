#!/bin/bash

# FR3 Reach Task - Actor Script
# This script runs the actor for data collection

export PYTHONPATH=$PYTHONPATH:/home/hanyu/code/hil-serl
export PYTHONPATH=$PYTHONPATH:/home/hanyu/code/HIROLRobotPlatform

# Disable JAX GPU memory preallocation for actor
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.1

# Run actor with FR3 reach configuration
python -m serl_launcher.serl_launcher \
    --config=experiments.fr3_reach.config:TrainConfig \
    --mode=actor \
    --ip=127.0.0.1 \
    --actor.num_actors=1 \
    --debug