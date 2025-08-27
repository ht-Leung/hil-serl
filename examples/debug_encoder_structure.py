"""Debug the structure of encoders to see why wrist_1 doesn't have pretrained_encoder"""
import sys
import os
sys.path.append('/home/hanyu/code/hil-serl/examples')

import jax
import jax.numpy as jnp
from serl_launcher.networks.reward_classifier import create_classifier
from experiments.mappings import CONFIG_MAPPING
import pickle as pkl

# Get config
config = CONFIG_MAPPING["hirol_unifined"]()

# Create fake sample data with correct structure
fake_sample = {
    'wrist_1': jnp.zeros((256, 256, 3)),
    'front': jnp.zeros((256, 256, 3)),
    'state': jnp.zeros(10)  # Add state if needed
}

print("=" * 60)
print("Testing Encoder Structure")
print("=" * 60)

print(f"\nClassifier keys: {config.classifier_keys}")

rng = jax.random.PRNGKey(0)

# Create classifier
print("\nCreating classifier...")
classifier = create_classifier(
    rng,
    fake_sample,
    config.classifier_keys
)

print("\n--- Analyzing encoder structure ---")

# Check the structure
encoder_def = classifier.params["encoder_def"]

for image_key in config.classifier_keys:
    encoder_name = f"encoder_{image_key}"
    print(f"\n{encoder_name}:")
    
    if encoder_name in encoder_def:
        # Check top-level keys
        top_keys = list(encoder_def[encoder_name].keys())
        print(f"  Top-level keys: {top_keys}")
        
        # Check for pretrained_encoder
        if "pretrained_encoder" in encoder_def[encoder_name]:
            pretrained_keys = list(encoder_def[encoder_name]["pretrained_encoder"].keys())
            print(f"  ✓ Has pretrained_encoder with {len(pretrained_keys)} keys: {pretrained_keys}")
        else:
            print(f"  ✗ NO pretrained_encoder found!")
            
            # Let's see what it does have
            for key in top_keys:
                if isinstance(encoder_def[encoder_name][key], dict):
                    sub_keys = list(encoder_def[encoder_name][key].keys())
                    print(f"    {key}: {sub_keys[:3]}..." if len(sub_keys) > 3 else f"    {key}: {sub_keys}")
    else:
        print(f"  ✗ Encoder not found in params!")

print("\n" + "=" * 60)