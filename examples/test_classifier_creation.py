"""Test script to debug classifier creation and parameter loading"""
import sys
import os
sys.path.append('/home/hanyu/code/hil-serl/examples')

import jax
import pickle as pkl
from experiments.mappings import CONFIG_MAPPING
from serl_launcher.networks.reward_classifier import create_classifier

# Suppress JAX warnings
import warnings
warnings.filterwarnings("ignore")

# Get config
config = CONFIG_MAPPING["hirol_unifined"]()
env = config.get_environment(fake_env=True, save_video=False, classifier=False)

print("=" * 60)
print("Testing Classifier Creation")
print("=" * 60)

# Get a sample observation
sample_obs = env.observation_space.sample()

# Create classifier with debugging
print(f"\nClassifier keys being used: {config.classifier_keys}")
print(f"Number of views: {len(config.classifier_keys)}")

rng = jax.random.PRNGKey(0)

# Manually create the classifier to debug
print("\n--- Creating classifier ---")
classifier = create_classifier(
    rng, 
    {"observations": sample_obs},
    config.classifier_keys
)

print("\n--- Checking classifier structure ---")

# Check what encoders were created
encoder_params = classifier.params["encoder_def"]
print(f"\nEncoders created:")
for key in encoder_params:
    print(f"  - {key}")
    if "pretrained_encoder" in encoder_params[key]:
        pretrained_keys = encoder_params[key]["pretrained_encoder"].keys()
        print(f"    Pretrained layers: {list(pretrained_keys)[:3]}... (showing first 3)")

print("\n--- Checking which encoders get pretrained weights ---")

# Load the pretrained weights
file_path = os.path.expanduser("~/.serl/resnet10_params.pkl")
if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        resnet_params = pkl.load(f)
    
    print(f"\nPretrained ResNet-10 keys available: {list(resnet_params.keys())}")
    
    # Check replacement logic
    for image_key in config.classifier_keys:
        encoder_key = f"encoder_{image_key}"
        print(f"\nChecking {encoder_key}:")
        
        if encoder_key in encoder_params:
            if "pretrained_encoder" in encoder_params[encoder_key]:
                replaced_keys = []
                for k in encoder_params[encoder_key]["pretrained_encoder"]:
                    if k in resnet_params:
                        replaced_keys.append(k)
                
                if replaced_keys:
                    print(f"  ✓ Would replace: {replaced_keys}")
                else:
                    print(f"  ✗ No matching keys to replace!")
            else:
                print(f"  ✗ No pretrained_encoder found!")
        else:
            print(f"  ✗ Encoder not found in params!")
else:
    print("Pretrained weights file not found!")

print("\n" + "=" * 60)