"""Debug RGB+Depth Classifier creation issue"""
import sys
import jax
import jax.numpy as jnp
import numpy as np
import traceback

sys.path.insert(0, '/home/hanyu/code/hil-serl')
sys.path.insert(0, '/home/hanyu/code/hil-serl/examples')
sys.path.insert(0, '/home/hanyu/code/Seg-NN')

from serl_launcher.networks.reward_classifier_rgbd import create_rgbd_classifier
from experiments.hirol_fixed_gripper.config import TrainConfig

def debug_classifier():
    config = TrainConfig()
    config.use_depth = True

    # Create minimal test data
    batch_size = 1  # Start with single sample
    height, width = 128, 128

    sample_obs = {}
    for camera_key in config.image_keys:
        # RGBD data
        rgbd_data = np.random.rand(batch_size, height, width, 4).astype(np.float32)
        sample_obs[camera_key] = rgbd_data
        print(f"Camera '{camera_key}' shape: {rgbd_data.shape}")

    # Add state
    sample_obs['state'] = np.random.rand(batch_size, 15).astype(np.float32)

    key = jax.random.PRNGKey(42)

    print("\nAttempting to create RGB+Depth classifier...")
    print(f"Depth config: {config.depth_encoder_kwargs}")
    print(f"Camera params: {config.camera_params}")

    try:
        # Enable JAX's detailed debugging
        import os
        os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

        classifier = create_rgbd_classifier(
            key=key,
            sample=sample_obs,
            image_keys=config.image_keys,
            n_way=2,
            depth_encoder_kwargs=config.depth_encoder_kwargs,
            camera_params=config.camera_params,
        )
        print("✓ Classifier created successfully!")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()

        # Try to narrow down the issue
        print("\n" + "="*60)
        print("Debugging individual components...")

        # Test RGB encoder
        print("\n1. Testing RGB encoder creation...")
        try:
            from serl_launcher.vision.resnet_v1 import resnetv1_configs, PreTrainedResNetEncoder
            from serl_launcher.common.encoding import EncodingWrapper

            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
                name="pretrained_encoder",
            )
            rgb_encoders = {
                image_key: PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                )
                for image_key in config.image_keys
            }
            rgb_encoder_def = EncodingWrapper(
                encoder=rgb_encoders,
                use_proprio=False,
                enable_stacking=True,
                image_keys=config.image_keys,
            )
            print("✓ RGB encoder created")
        except Exception as e:
            print(f"✗ RGB encoder failed: {e}")

        # Test Depth encoder
        print("\n2. Testing Depth encoder creation...")
        try:
            from serl_launcher.vision.segnn_encoder import create_segnn_encoder

            # Try creating single SegNN encoder
            encoder = create_segnn_encoder(
                camera_params=config.camera_params.get('side', {}),
                **config.depth_encoder_kwargs
            )

            # Test init
            dummy_depth = jnp.ones((height, width))
            dummy_rgb = jnp.ones((height, width, 3))
            init_key = jax.random.PRNGKey(0)

            print("   Initializing SegNN encoder...")
            params = encoder.init(init_key, dummy_depth, dummy_rgb, train=False)
            print("   ✓ SegNN encoder initialized")

        except Exception as e:
            print(f"   ✗ SegNN encoder failed: {e}")
            traceback.print_exc()

        return False

if __name__ == "__main__":
    success = debug_classifier()
    if success:
        print("\n✅ Debug successful!")
    else:
        print("\n❌ Debug revealed issues!")