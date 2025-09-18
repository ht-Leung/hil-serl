#!/bin/bash

# Script to retrain classifier with human feedback data

echo "================================"
echo "Classifier Retraining with Human Feedback"
echo "================================"

# Default values
NUM_EPOCHS=150
BATCH_SIZE=256
FEEDBACK_WEIGHT=2.0
INCREMENTAL=False

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --weight)
            FEEDBACK_WEIGHT="$2"
            shift 2
            ;;
        --incremental)
            INCREMENTAL=True
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  - Epochs: $NUM_EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Feedback weight: $FEEDBACK_WEIGHT"
echo "  - Incremental training: $INCREMENTAL"
echo ""

# Run the retraining script
python retrain_classifier.py \
    --exp_name=hirol_online_classifier_fixed_gripper \
    --num_epochs=$NUM_EPOCHS \
    --batch_size=$BATCH_SIZE \
    --feedback_weight=$FEEDBACK_WEIGHT \
    --incremental=$INCREMENTAL

echo ""
echo "Retraining complete!"
echo "New classifier checkpoint saved."