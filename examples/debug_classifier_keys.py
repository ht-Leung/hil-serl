"""Debug script to check which camera views are used in classifier"""
import sys
import os
sys.path.append('/home/hanyu/code/hil-serl/examples')

from experiments.mappings import CONFIG_MAPPING

# Get the config
config = CONFIG_MAPPING["hirol_unifined"]()

print("=" * 50)
print("Configuration Analysis:")
print("=" * 50)

# Check what's in the config
print(f"\n1. image_keys (all camera views): {config.image_keys}")
print(f"2. classifier_keys (views used for reward classifier): {config.classifier_keys}")

# Check the difference
unused_views = set(config.image_keys) - set(config.classifier_keys)
if unused_views:
    print(f"\n⚠️ Views NOT used in classifier: {unused_views}")

print("\n" + "=" * 50)
print("Expected Encoder Names:")
print("=" * 50)
for key in config.classifier_keys:
    print(f"  - encoder_{key}")

print("\n" + "=" * 50)
print("Checking data availability:")
print("=" * 50)

# Check if classifier data exists
classifier_data_path = config.classifier_data_path
print(f"Classifier data path: {classifier_data_path}")

if os.path.exists(classifier_data_path):
    import glob
    success_files = glob.glob(os.path.join(classifier_data_path, "*success*.pkl"))
    failure_files = glob.glob(os.path.join(classifier_data_path, "*failure*.pkl"))
    
    print(f"  - Found {len(success_files)} success files")
    print(f"  - Found {len(failure_files)} failure files")
    
    # Sample check - load one file to see what camera views are in it
    if success_files:
        import pickle
        sample_data = pickle.load(open(success_files[0], 'rb'))
        if sample_data and len(sample_data) > 0:
            obs_keys = sample_data[0]['observations'].keys()
            camera_keys = [k for k in obs_keys if k in config.image_keys]
            print(f"\nCamera views in data: {camera_keys}")
else:
    print(f"  ❌ Data path does not exist!")