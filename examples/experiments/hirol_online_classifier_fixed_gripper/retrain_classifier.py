#!/usr/bin/env python3

"""
Retrain the classifier using human feedback data.

This script combines the original training data with human feedback
to improve classifier accuracy and reduce false positives.
"""

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
from pathlib import Path
from datetime import datetime

from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier

from experiments.mappings import CONFIG_MAPPING


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "hirol_online_classifier_fixed_gripper", 
                    "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 150, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_float("feedback_weight", 2.0, 
                   "Weight multiplier for human feedback samples (higher = trust human more).")
flags.DEFINE_boolean("incremental", False, 
                    "If True, load existing checkpoint and continue training.")


def load_feedback_data(feedback_dir, pos_buffer, neg_buffer, env, weight=1.0):
    """Load human feedback data into appropriate buffers."""
    feedback_files = sorted(glob.glob(os.path.join(feedback_dir, "feedback_*.pkl")))
    
    if not feedback_files:
        print(f"No feedback files found in {feedback_dir}")
        return 0, 0, 0
    
    total_samples = 0
    false_positives = 0
    false_negatives = 0
    
    for feedback_file in feedback_files:
        print(f"Loading {feedback_file}")
        with open(feedback_file, 'rb') as f:
            feedback_data = pkl.load(f)
        
        for sample in feedback_data:
            obs = sample['observation']
            true_label = sample['true_label']
            classifier_pred = sample['classifier_prediction']
            
            # Track statistics
            if classifier_pred == 1 and true_label == 0:
                false_positives += 1
            elif classifier_pred == 0 and true_label == 1:
                false_negatives += 1
            
            # Create transition for appropriate buffer
            trans = {
                'observations': obs,
                'labels': true_label,
                'actions': env.action_space.sample(),  # Dummy action
                'weight': weight  # Higher weight for human feedback
            }
            
            # Insert into appropriate buffer with weighting
            if true_label == 1:
                for _ in range(int(weight)):
                    pos_buffer.insert(trans)
            else:
                for _ in range(int(weight)):
                    neg_buffer.insert(trans)
            
            total_samples += 1
    
    print(f"Loaded {total_samples} feedback samples")
    print(f"  - False positives corrected: {false_positives}")
    print(f"  - False negatives corrected: {false_negatives}")
    
    return total_samples, false_positives, false_negatives


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)

    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    
    # Create buffers
    pos_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=50000,
        include_label=True,
    )
    
    neg_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=50000,
        include_label=True,
    )

    # Load original training data
    classifier_data_dir = config.classifier_data_path if hasattr(config, 'classifier_data_path') else os.path.join(os.getcwd(), "classifier_data")
    
    # Load success data
    success_paths = glob.glob(os.path.join(classifier_data_dir, "*success*.pkl"))
    original_success_count = 0
    for path in success_paths:
        success_data = pkl.load(open(path, "rb"))
        for trans in success_data:
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = 1
            trans['actions'] = env.action_space.sample()
            trans['weight'] = 1.0  # Original data weight
            pos_buffer.insert(trans)
            original_success_count += 1
    
    # Load failure data
    failure_paths = glob.glob(os.path.join(classifier_data_dir, "*failure*.pkl"))
    original_failure_count = 0
    for path in failure_paths:
        failure_data = pkl.load(open(path, "rb"))
        for trans in failure_data:
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = 0
            trans['actions'] = env.action_space.sample()
            trans['weight'] = 1.0
            neg_buffer.insert(trans)
            original_failure_count += 1
    
    print(f"Loaded original data:")
    print(f"  - {original_success_count} success samples")
    print(f"  - {original_failure_count} failure samples")
    
    # Load human feedback data with higher weight
    feedback_dir = str(config.data_path / "feedback_data") if hasattr(config, 'data_path') else "./feedback_data"
    feedback_count, fp_count, fn_count = load_feedback_data(
        feedback_dir, pos_buffer, neg_buffer, env, weight=FLAGS.feedback_weight
    )
    
    print(f"\nTotal buffer sizes:")
    print(f"  Success buffer: {len(pos_buffer)} samples")
    print(f"  Failure buffer: {len(neg_buffer)} samples")
    
    # Create iterators
    pos_iterator = pos_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size // 2},
        device=sharding.replicate(),
    )
    
    neg_iterator = neg_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size // 2},
        device=sharding.replicate(),
    )

    # Create or load classifier
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)

    if FLAGS.incremental and os.path.exists(config.classifier_ckpt_path):
        print(f"Loading existing checkpoint from {config.classifier_ckpt_path}")
        checkpoint_dict = checkpoints.restore_checkpoint(
            config.classifier_ckpt_path, target=None
        )
        classifier = create_classifier(
            key, 
            sample["observations"], 
            config.classifier_keys,
            encoder=config.encoder_type
        )
        classifier = classifier.replace(params=checkpoint_dict["model"])
        start_step = checkpoint_dict.get("step", 0)
    else:
        print("Creating new classifier")
        rng, key = jax.random.split(rng)
        classifier = create_classifier(
            key, 
            sample["observations"], 
            config.classifier_keys,
            encoder=config.encoder_type
        )
        start_step = 0

    # Training loop with weighted loss
    @jax.jit
    def update_classifier(classifier, batch):
        def loss_fn(params):
            logits = classifier.apply(params, batch["observations"])
            labels = batch["labels"]
            weights = batch.get("weight", jnp.ones_like(labels))
            
            # Weighted binary cross-entropy
            loss = optax.sigmoid_binary_cross_entropy(logits, labels)
            weighted_loss = jnp.mean(loss * weights)
            
            # Calculate accuracy
            predictions = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
            accuracy = jnp.mean((predictions == labels).astype(jnp.float32))
            
            return weighted_loss, accuracy

        (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            classifier.params
        )
        updates, new_opt_state = classifier.tx.update(grads, classifier.opt_state)
        new_params = optax.apply_updates(classifier.params, updates)
        
        return classifier.replace(params=new_params, opt_state=new_opt_state), {
            "loss": loss,
            "accuracy": accuracy,
        }

    # Training
    print(f"\nStarting training from step {start_step}")
    
    best_accuracy = 0.0
    step = start_step
    
    for epoch in range(FLAGS.num_epochs):
        epoch_losses = []
        epoch_accuracies = []
        
        num_batches = min(len(pos_buffer), len(neg_buffer)) // (FLAGS.batch_size // 2)
        
        with tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{FLAGS.num_epochs}") as pbar:
            for _ in pbar:
                pos_batch = next(pos_iterator)
                neg_batch = next(neg_iterator)
                batch = concat_batches(pos_batch, neg_batch, axis=0)
                
                # Apply data augmentation
                rng, key = jax.random.split(rng)
                batch["observations"]["images"] = batched_random_crop(
                    batch["observations"]["images"], key, padding=4
                )
                
                classifier, info = update_classifier(classifier, batch)
                
                epoch_losses.append(float(info["loss"]))
                epoch_accuracies.append(float(info["accuracy"]))
                
                pbar.set_postfix({
                    "loss": f"{info['loss']:.4f}",
                    "acc": f"{info['accuracy']:.3%}"
                })
                
                step += 1

        avg_loss = np.mean(epoch_losses)
        avg_accuracy = np.mean(epoch_accuracies)
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.3%}")
        
        # Save checkpoint if improved
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            
            # Save checkpoint
            checkpoint_path = Path(config.classifier_ckpt_path)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            checkpoints.save_checkpoint(
                config.classifier_ckpt_path,
                {"model": classifier.params, "step": step},
                step=step,
                keep=5,
            )
            
            print(f"  ✓ Saved checkpoint (best accuracy: {best_accuracy:.3%})")

    # Final statistics
    print(f"\n" + "="*60)
    print(f"Training Complete!")
    print(f"="*60)
    print(f"Best accuracy: {best_accuracy:.3%}")
    print(f"Checkpoint saved to: {config.classifier_ckpt_path}")
    print(f"Human feedback statistics:")
    print(f"  - Total feedback samples: {feedback_count}")
    print(f"  - False positives corrected: {fp_count}")
    print(f"  - False negatives corrected: {fn_count}")
    if feedback_count > 0:
        print(f"  - Error rate in original classifier: {(fp_count + fn_count)/feedback_count:.1%}")
    print(f"="*60)


if __name__ == "__main__":
    app.run(main)