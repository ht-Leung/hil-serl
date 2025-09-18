"""Training script for RGB+Depth Late-fusion Reward Classifier"""
import glob
import os
import pickle as pkl
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
import numpy as np
import optax
from tqdm import tqdm
from absl import app, flags

from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier_rgbd import (
    create_rgbd_classifier,
    create_rgbd_classifier_from_rgb_checkpoint
)

from experiments.mappings import CONFIG_MAPPING


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 150, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_bool("use_depth", True, "Use depth information in classifier.")
flags.DEFINE_string("rgb_checkpoint", None, "Optional: path to RGB-only classifier checkpoint for transfer learning.")
flags.DEFINE_bool("freeze_rgb", False, "Freeze RGB encoder weights during training.")


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    # Enable depth in environment if requested
    if FLAGS.use_depth and hasattr(config, 'use_depth'):
        config.use_depth = True

    # Create environment - observation space is now handled by proper wrapper
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)

    print(f"Environment observation space:")
    for key in config.classifier_keys:
        if hasattr(env.observation_space, 'spaces') and key in env.observation_space.spaces:
            print(f"  {key}: {env.observation_space.spaces[key].shape}")

    # Check if we have RGBD data by examining first data file
    print(f"Checking data format...")
    classifier_data_dir = config.classifier_data_path if hasattr(config, 'classifier_data_path') else os.path.join(os.getcwd(), "classifier_data")
    success_paths = glob.glob(os.path.join(classifier_data_dir, "*success*.pkl"))

    if success_paths:
        sample_data = pkl.load(open(success_paths[0], "rb"))
        if sample_data:
            sample_obs = sample_data[0]['observations']
            print(f"Data format:")
            for key in config.classifier_keys:
                if key in sample_obs:
                    print(f"  {key}: {sample_obs[key].shape}")

            # Check if we need to fix observation space mismatch
            needs_obs_fix = False
            mismatches = []
            for key in config.classifier_keys:
                if key in sample_obs:
                    data_shape = sample_obs[key].shape
                    if hasattr(env.observation_space, 'spaces') and key in env.observation_space.spaces:
                        env_shape = env.observation_space.spaces[key].shape
                        if data_shape != env_shape:
                            mismatches.append((key, data_shape, env_shape))
                            needs_obs_fix = True

            if needs_obs_fix:
                print("  📝 Fixing observation space for RGBD data compatibility...")
                # Create compatible observation space that matches the actual data
                from gymnasium.spaces import Box, Dict as DictSpace
                fixed_spaces = {}

                # Copy existing spaces and fix mismatched ones
                for key, space in env.observation_space.spaces.items():
                    if key in config.classifier_keys:
                        # Check if this key has mismatched data
                        data_key_match = next((m for m in mismatches if m[0] == key), None)
                        if data_key_match:
                            data_shape = data_key_match[1]
                            fixed_spaces[key] = Box(low=0, high=255, shape=data_shape, dtype=np.uint8)
                            print(f"    Fixed {key}: {data_key_match[2]} -> {data_shape}")
                        else:
                            fixed_spaces[key] = space
                    else:
                        fixed_spaces[key] = space

                # Update environment observation space
                env.observation_space = DictSpace(fixed_spaces)
                print(f"  ✓ Observation space fixed for RGBD compatibility")
            else:
                print("  ✓ Observation space compatible, no fixes needed")
    else:
        print("  ⚠️  No training data found!")

    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)

    # Create buffer for positive transitions
    pos_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=20000,
        include_label=True,
    )

    # Use config-specific classifier data path
    classifier_data_dir = config.classifier_data_path if hasattr(config, 'classifier_data_path') else os.path.join(os.getcwd(), "classifier_data")
    success_paths = glob.glob(os.path.join(classifier_data_dir, "*success*.pkl"))

    print(f"Loading {len(success_paths)} success data files...")
    for path in success_paths:
        success_data = pkl.load(open(path, "rb"))
        for trans in success_data:
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = 1
            trans['actions'] = env.action_space.sample()
            pos_buffer.insert(trans)

    pos_iterator = pos_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )

    # Create buffer for negative transitions
    neg_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=50000,
        include_label=True,
    )
    failure_paths = glob.glob(os.path.join(classifier_data_dir, "*failure*.pkl"))

    print(f"Loading {len(failure_paths)} failure data files...")
    for path in failure_paths:
        failure_data = pkl.load(open(path, "rb"))
        for trans in failure_data:
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = 0
            trans['actions'] = env.action_space.sample()
            neg_buffer.insert(trans)

    neg_iterator = neg_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )

    print(f"Failed buffer size: {len(neg_buffer)}")
    print(f"Success buffer size: {len(pos_buffer)}")
    print(f"Use depth: {FLAGS.use_depth}")

    # Check if data contains depth information
    sample_obs = next(pos_iterator)["observations"]
    has_depth = False
    for key in config.classifier_keys:
        if key in sample_obs:
            data = sample_obs[key]
            # Check if RGBD format (4 channels) or separate depth keys
            if len(data.shape) >= 3 and data.shape[-1] == 4:
                has_depth = True
                print(f"Detected RGBD data in camera '{key}': shape={data.shape}")
                break
        # Check for separate depth keys
        depth_key = f"depth_{key}"
        if depth_key in sample_obs:
            has_depth = True
            print(f"Detected separate depth data '{depth_key}': shape={sample_obs[depth_key].shape}")
            break

    if FLAGS.use_depth and not has_depth:
        print("WARNING: use_depth=True but no depth data detected in observations!")
        print("Falling back to RGB-only mode.")
        FLAGS.use_depth = False

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)

    rng, key = jax.random.split(rng)

    # Create classifier based on configuration
    if FLAGS.use_depth:
        # Get depth encoder config and camera params from config
        depth_encoder_kwargs = getattr(config, 'depth_encoder_kwargs', None)
        camera_params = getattr(config, 'camera_params', None)

        if FLAGS.rgb_checkpoint:
            # Initialize from RGB checkpoint for transfer learning
            print(f"Initializing RGB+Depth classifier from RGB checkpoint: {FLAGS.rgb_checkpoint}")
            classifier = create_rgbd_classifier_from_rgb_checkpoint(
                key,
                sample["observations"],
                config.classifier_keys,
                FLAGS.rgb_checkpoint,
                n_way=2,
                use_depth=FLAGS.use_depth,
                depth_encoder_kwargs=depth_encoder_kwargs,
                camera_params=camera_params,
            )
        else:
            # Create new RGB+Depth classifier from scratch
            print("Creating new RGB+Depth classifier from scratch")
            classifier = create_rgbd_classifier(
                key,
                sample["observations"],
                config.classifier_keys,
                n_way=2,
                use_depth=FLAGS.use_depth,
                depth_encoder_kwargs=depth_encoder_kwargs,
                camera_params=camera_params,
            )
    else:
        # Fall back to RGB-only classifier
        print("Creating RGB-only classifier")
        from serl_launcher.networks.reward_classifier import create_classifier
        classifier = create_classifier(
            key,
            sample["observations"],
            config.classifier_keys,
            n_way=2,
        )

    def data_augmentation_fn(rng, observations):
        """Apply data augmentation to observations"""
        for pixel_key in config.classifier_keys:
            if pixel_key in observations:
                # Apply random crop augmentation to image data
                observations = observations.copy(
                    add_or_replace={
                        pixel_key: batched_random_crop(
                            observations[pixel_key], rng, padding=4, num_batch_dims=2
                        )
                    }
                )
        return observations

    @jax.jit
    def train_step(state, batch, key):
        """Single training step"""
        def loss_fn(params):
            # Handle both depth and non-depth classifiers
            if FLAGS.use_depth:
                logits = state.apply_fn(
                    {"params": params},
                    batch["observations"],
                    rngs={"dropout": key, "pointcloud_sampling": key},
                    train=True
                )
            else:
                logits = state.apply_fn(
                    {"params": params},
                    batch["observations"],
                    rngs={"dropout": key},
                    train=True
                )
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)

        # Optionally freeze RGB encoder gradients
        if FLAGS.freeze_rgb and FLAGS.use_depth:
            # Zero out gradients for RGB encoder
            for image_key in config.classifier_keys:
                encoder_path = ["encoder", "rgb_encoder", image_key]
                if all(k in grads for k in encoder_path[:-1]):
                    target_grads = grads
                    for k in encoder_path[:-1]:
                        target_grads = target_grads[k]
                    if encoder_path[-1] in target_grads:
                        target_grads[encoder_path[-1]] = jax.tree_map(
                            jnp.zeros_like, target_grads[encoder_path[-1]]
                        )

        # Compute accuracy
        if FLAGS.use_depth:
            logits = state.apply_fn(
                {"params": state.params},
                batch["observations"],
                train=False,
                rngs={"dropout": key, "pointcloud_sampling": key}
            )
        else:
            logits = state.apply_fn(
                {"params": state.params},
                batch["observations"],
                train=False,
                rngs={"dropout": key}
            )
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])

        return state.apply_gradients(grads=grads), loss, train_accuracy

    @jax.jit
    def eval_step(state, batch, key):
        """Evaluation step without gradient computation"""
        if FLAGS.use_depth:
            logits = state.apply_fn(
                {"params": state.params},
                batch["observations"],
                train=False,
                rngs={"pointcloud_sampling": key}
            )
        else:
            logits = state.apply_fn(
                {"params": state.params},
                batch["observations"],
                train=False
            )
        loss = optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()
        accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])
        return loss, accuracy

    # Training loop
    print(f"\nStarting training for {FLAGS.num_epochs} epochs...")
    print(f"Batch size: {FLAGS.batch_size}")
    if FLAGS.freeze_rgb:
        print("RGB encoder weights are frozen")

    best_accuracy = 0.0
    checkpoint_dir = config.classifier_ckpt_path if hasattr(config, 'classifier_ckpt_path') else "./classifier_ckpt"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in tqdm(range(FLAGS.num_epochs)):
        # Sample equal number of positive and negative examples
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)

        # Merge and create labels
        batch = concat_batches(pos_sample, neg_sample, axis=0)

        # Apply data augmentation
        rng, key = jax.random.split(rng)
        obs = data_augmentation_fn(key, batch["observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "labels": batch["labels"][..., None],
            }
        )

        # Training step
        rng, key = jax.random.split(rng)
        classifier, loss, train_accuracy = train_step(classifier, batch, key)

        # Periodic evaluation and checkpointing
        if epoch % 10 == 0:
            # Evaluate on validation batch (without augmentation)
            val_pos_sample = next(pos_iterator)
            val_neg_sample = next(neg_iterator)
            val_batch = concat_batches(val_pos_sample, val_neg_sample, axis=0)
            val_batch = val_batch.copy(
                add_or_replace={
                    "labels": val_batch["labels"][..., None],
                }
            )
            rng, eval_key = jax.random.split(rng)
            val_loss, val_accuracy = eval_step(classifier, val_batch, eval_key)

            print(f"Epoch {epoch}: train_loss={loss:.4f}, train_acc={train_accuracy:.3f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.3f}")

            # Save checkpoint if best accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                classifier_type = "rgbd" if FLAGS.use_depth else "rgb"
                checkpoints.save_checkpoint(
                    checkpoint_dir,
                    classifier,
                    epoch,
                    prefix=f"best_{classifier_type}_classifier_{FLAGS.exp_name}_",
                    overwrite=True,
                )
                print(f"Saved best checkpoint with accuracy {best_accuracy:.3f}")

    # Save final checkpoint
    classifier_type = "rgbd" if FLAGS.use_depth else "rgb"
    checkpoints.save_checkpoint(
        checkpoint_dir,
        classifier,
        FLAGS.num_epochs,
        prefix=f"final_{classifier_type}_classifier_{FLAGS.exp_name}_",
        overwrite=True,
    )
    print(f"\nTraining complete! Best accuracy: {best_accuracy:.3f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    app.run(main)