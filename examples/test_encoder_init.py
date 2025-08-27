"""Minimal test to reproduce the encoder initialization issue"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from serl_launcher.vision.resnet_v1 import resnetv1_configs, PreTrainedResNetEncoder
from serl_launcher.common.encoding import EncodingWrapper

# Create two image keys like in the real scenario
image_keys = ['wrist_1', 'front']

# Create fake input
fake_input = {
    'wrist_1': jnp.zeros((256, 256, 3)),
    'front': jnp.zeros((256, 256, 3))
}

# Create encoders exactly like in create_classifier
print("Creating encoders...")
pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
    pre_pooling=True,
    name="pretrained_encoder",
)

encoders = {}
for idx, image_key in enumerate(image_keys):
    print(f"Creating encoder for {image_key} (index {idx})")
    encoders[image_key] = PreTrainedResNetEncoder(
        pooling_method="spatial_learned_embeddings",
        num_spatial_blocks=8,
        bottleneck_dim=256,
        pretrained_encoder=pretrained_encoder,
        name=f"encoder_{image_key}",
    )

encoder_def = EncodingWrapper(
    encoder=encoders,
    use_proprio=False,
    enable_stacking=True,
    image_keys=image_keys,
)

# Initialize
print("\nInitializing...")
key = jax.random.PRNGKey(0)
params = encoder_def.init(key, fake_input)["params"]

print("\n" + "=" * 60)
print("Checking initialized parameters:")
print("=" * 60)

print(f"Top-level param keys: {list(params.keys())}")

# The params are under encoder_{image_key} keys
for image_key in image_keys:
    encoder_name = f"encoder_{image_key}"
    if encoder_name in params:
        print(f"\n{encoder_name}:")
        keys = list(params[encoder_name].keys())
        print(f"  Keys: {keys}")
        has_pretrained = "pretrained_encoder" in params[encoder_name]
        print(f"  Has pretrained_encoder: {has_pretrained}")
        
        # If no pretrained_encoder, check why
        if not has_pretrained:
            print(f"  ⚠️  Missing pretrained_encoder!")
    else:
        print(f"\n{encoder_name}: NOT FOUND in params")

print("\n" + "=" * 60)
print("Testing direct initialization (without wrapper):")
print("=" * 60)

# Test direct initialization of a single encoder
for image_key in image_keys:
    print(f"\nDirect test for {image_key}:")
    single_encoder = PreTrainedResNetEncoder(
        pooling_method="spatial_learned_embeddings",
        num_spatial_blocks=8,
        bottleneck_dim=256,
        pretrained_encoder=pretrained_encoder,
        name=f"test_encoder_{image_key}",
    )
    
    single_params = single_encoder.init(key, jnp.zeros((256, 256, 3)))["params"]
    print(f"  Keys: {list(single_params.keys())}")
    has_pretrained = "pretrained_encoder" in single_params
    print(f"  Has pretrained_encoder: {has_pretrained}")