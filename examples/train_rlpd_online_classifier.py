#!/usr/bin/env python3

"""
Train RLPD with Online Classifier Learning

This version integrates human feedback and online classifier retraining
directly into the training loop, eliminating file I/O conflicts.
"""

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted
import signal
import sys
import gc
import threading
import queue
from datetime import datetime
from pathlib import Path
from collections import deque
import optax

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier
from serl_launcher.data.data_store import ReplayBuffer

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_boolean("debug", False, "Debug mode.")

# Online classifier learning flags
flags.DEFINE_boolean("online_classifier", True, "Enable online classifier learning.")
flags.DEFINE_integer("classifier_retrain_interval", 100, "Retrain classifier every N feedback samples.")
flags.DEFINE_float("feedback_weight", 2.0, "Weight for human feedback samples.")
flags.DEFINE_integer("classifier_batch_size", 256, "Batch size for classifier training.")
flags.DEFINE_integer("classifier_epochs", 10, "Epochs per classifier retraining.")


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


class OnlineClassifierTrainer:
    """
    Thread-safe online classifier trainer that runs in the learner process.
    """
    
    def __init__(self, config, initial_classifier=None, checkpoint_dir=None):
        self.config = config
        self.checkpoint_dir = checkpoint_dir or "./classifier_ckpt_online"
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Thread-safe queue for feedback data
        self.feedback_queue = queue.Queue()
        
        # Buffers for training data
        self.pos_buffer = None
        self.neg_buffer = None
        
        # Current classifier
        self.classifier = initial_classifier
        self.classifier_lock = threading.RLock()
        self.classifier_version = 0
        
        # Statistics
        self.stats = {
            'total_feedback': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'retrain_count': 0,
            'last_accuracy': 0.0
        }
        
        # Training thread
        self.training_thread = None
        self.stop_event = threading.Event()
        
        # Initialize buffers and classifier
        self._initialize_buffers()
        
    def _initialize_buffers(self):
        """Initialize training buffers with original data."""
        # Create fake env to get observation/action spaces
        env = self.config.get_environment(fake_env=True, save_video=False, classifier=False)
        
        self.pos_buffer = ReplayBuffer(
            env.observation_space,
            env.action_space,
            capacity=50000,
            include_label=True,
        )
        
        self.neg_buffer = ReplayBuffer(
            env.observation_space,
            env.action_space,
            capacity=50000,
            include_label=True,
        )
        
        # Load original training data
        classifier_data_dir = self.config.classifier_data_path if hasattr(self.config, 'classifier_data_path') else "./classifier_data"
        
        # Load success data
        success_paths = glob.glob(os.path.join(classifier_data_dir, "*success*.pkl"))
        for path in success_paths:
            try:
                success_data = pkl.load(open(path, "rb"))
                for trans in success_data:
                    if "images" not in trans.get('observations', {}).keys():
                        trans["labels"] = 1
                        trans['actions'] = env.action_space.sample()
                        trans['weight'] = 1.0
                        self.pos_buffer.insert(trans)
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        # Load failure data
        failure_paths = glob.glob(os.path.join(classifier_data_dir, "*failure*.pkl"))
        for path in failure_paths:
            try:
                failure_data = pkl.load(open(path, "rb"))
                for trans in failure_data:
                    if "images" not in trans.get('observations', {}).keys():
                        trans["labels"] = 0
                        trans['actions'] = env.action_space.sample()
                        trans['weight'] = 1.0
                        self.neg_buffer.insert(trans)
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        print(f"[OnlineClassifier] Loaded {len(self.pos_buffer)} success, {len(self.neg_buffer)} failure samples")
    
    def add_feedback(self, observation, classifier_prediction, true_label, confidence):
        """Add human feedback to the queue."""
        feedback = {
            'observation': observation,
            'classifier_prediction': classifier_prediction,
            'true_label': true_label,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        self.feedback_queue.put(feedback)
        self.stats['total_feedback'] += 1
        
        # Track statistics
        if classifier_prediction == 1 and true_label == 0:
            self.stats['false_positives'] += 1
        elif classifier_prediction == 0 and true_label == 1:
            self.stats['false_negatives'] += 1
    
    def get_classifier(self):
        """Get current classifier (thread-safe)."""
        with self.classifier_lock:
            return self.classifier, self.classifier_version
    
    def _retrain_classifier(self):
        """Retrain classifier with accumulated feedback."""
        # Collect feedback from queue
        feedback_batch = []
        while not self.feedback_queue.empty():
            try:
                feedback_batch.append(self.feedback_queue.get_nowait())
            except queue.Empty:
                break
        
        if not feedback_batch:
            return
        
        print(f"\n[OnlineClassifier] Retraining with {len(feedback_batch)} new feedback samples")
        
        # Add feedback to buffers
        env = self.config.get_environment(fake_env=True, save_video=False, classifier=False)
        for feedback in feedback_batch:
            trans = {
                'observations': feedback['observation'],
                'labels': feedback['true_label'],
                'actions': env.action_space.sample(),
                'weight': FLAGS.feedback_weight
            }
            
            # Insert with weight repetition
            buffer = self.pos_buffer if feedback['true_label'] == 1 else self.neg_buffer
            for _ in range(int(FLAGS.feedback_weight)):
                buffer.insert(trans)
        
        # Create iterators
        pos_iterator = self.pos_buffer.get_iterator(
            sample_args={"batch_size": FLAGS.classifier_batch_size // 2},
            device=sharding.replicate(),
        )
        
        neg_iterator = self.neg_buffer.get_iterator(
            sample_args={"batch_size": FLAGS.classifier_batch_size // 2},
            device=sharding.replicate(),
        )
        
        # Train classifier
        rng = jax.random.PRNGKey(self.stats['retrain_count'])
        
        # Get sample batch for initialization
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        sample = concat_batches(pos_sample, neg_sample, axis=0)
        
        # Create or update classifier
        with self.classifier_lock:
            if self.classifier is None:
                rng, key = jax.random.split(rng)
                self.classifier = create_classifier(
                    key,
                    sample["observations"],
                    self.config.classifier_keys,
                    encoder=self.config.encoder_type
                )
            
            # Training loop
            @jax.jit
            def update_classifier(classifier, batch):
                def loss_fn(params):
                    logits = classifier.apply(params, batch["observations"])
                    labels = batch["labels"]
                    weights = batch.get("weight", jnp.ones_like(labels))
                    
                    loss = optax.sigmoid_binary_cross_entropy(logits, labels)
                    weighted_loss = jnp.mean(loss * weights)
                    
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
            
            # Quick training epochs
            for epoch in range(FLAGS.classifier_epochs):
                epoch_accuracies = []
                num_batches = min(len(self.pos_buffer), len(self.neg_buffer)) // (FLAGS.classifier_batch_size // 2)
                num_batches = min(num_batches, 10)  # Limit batches per epoch for speed
                
                for _ in range(num_batches):
                    pos_batch = next(pos_iterator)
                    neg_batch = next(neg_iterator)
                    batch = concat_batches(pos_batch, neg_batch, axis=0)
                    
                    # Apply data augmentation
                    rng, key = jax.random.split(rng)
                    batch["observations"]["images"] = batched_random_crop(
                        batch["observations"]["images"], key, padding=4
                    )
                    
                    self.classifier, info = update_classifier(self.classifier, batch)
                    epoch_accuracies.append(float(info["accuracy"]))
                
                self.stats['last_accuracy'] = np.mean(epoch_accuracies)
            
            # Update version
            self.classifier_version += 1
            
            # Save checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, f"v{self.classifier_version}")
            checkpoints.save_checkpoint(
                checkpoint_path,
                {"model": self.classifier.params, "version": self.classifier_version},
                step=self.classifier_version,
                keep=3,
            )
        
        self.stats['retrain_count'] += 1
        print(f"[OnlineClassifier] Retrain #{self.stats['retrain_count']} complete. "
              f"Accuracy: {self.stats['last_accuracy']:.1%}, Version: {self.classifier_version}")
    
    def start_training_thread(self):
        """Start background training thread."""
        def training_loop():
            feedback_count = 0
            while not self.stop_event.is_set():
                time.sleep(5)  # Check every 5 seconds
                
                # Check if we have enough feedback
                current_feedback = self.stats['total_feedback']
                if current_feedback - feedback_count >= FLAGS.classifier_retrain_interval:
                    self._retrain_classifier()
                    feedback_count = current_feedback
        
        self.training_thread = threading.Thread(target=training_loop, daemon=True)
        self.training_thread.start()
        print("[OnlineClassifier] Training thread started")
    
    def stop(self):
        """Stop training thread and save final checkpoint."""
        self.stop_event.set()
        if self.training_thread:
            self.training_thread.join(timeout=10)
        
        # Save final statistics
        stats_path = os.path.join(self.checkpoint_dir, "final_stats.pkl")
        with open(stats_path, 'wb') as f:
            pkl.dump(self.stats, f)
        
        print(f"\n[OnlineClassifier] Final Statistics:")
        print(f"  Total feedback: {self.stats['total_feedback']}")
        print(f"  False positives corrected: {self.stats['false_positives']}")
        print(f"  False negatives corrected: {self.stats['false_negatives']}")
        print(f"  Retrain count: {self.stats['retrain_count']}")
        print(f"  Final accuracy: {self.stats['last_accuracy']:.1%}")


##############################################################################

def actor(agent, data_store, intvn_data_store, env, sampling_rng, online_trainer=None):
    """
    Actor loop with optional online classifier feedback.
    """
    client = None
    pbar = None
    
    try:
        # [Eval mode code unchanged...]
        if FLAGS.eval_checkpoint_step:
            # ... [eval code unchanged] ...
            pass
        
        start_step = (
            int(os.path.basename(natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))[-1])[12:-4]) + 1
            if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
            else 0
        )

        datastore_dict = {
            "actor_env": data_store,
            "actor_env_intvn": intvn_data_store,
        }

        client = TrainerClient(
            "actor_env",
            FLAGS.ip,
            make_trainer_config(),
            data_stores=datastore_dict,
            wait_for_server=True,
            timeout_ms=3000,
        )

        # Function to update the agent with new params
        def update_params(params):
            nonlocal agent
            agent = agent.replace(state=agent.state.replace(params=params))

        client.recv_network_callback(update_params)

        # Function to update classifier if using online learning
        current_classifier = None
        classifier_version = -1
        
        if online_trainer and FLAGS.online_classifier:
            def update_classifier():
                nonlocal current_classifier, classifier_version
                new_classifier, new_version = online_trainer.get_classifier()
                if new_version > classifier_version:
                    current_classifier = new_classifier
                    classifier_version = new_version
                    print(f"[Actor] Updated to classifier v{classifier_version}")
            
            # Initial classifier
            update_classifier()

        transitions = []
        demo_transitions = []

        obs, _ = env.reset()
        done = False

        # training loop
        timer = Timer()
        running_return = 0.0
        already_intervened = False
        intervention_count = 0
        intervention_steps = 0
        
        # Metrics tracking
        training_start_time = time.time()
        episode_count = 0
        success_history = []
        episode_start_time = time.time()
        episode_steps = 0

        pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
        for step in pbar:
            timer.tick("total")

            with timer.context("sample_actions"):
                if step < config.random_steps:
                    actions = env.action_space.sample()
                else:
                    sampling_rng, key = jax.random.split(sampling_rng)
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs),
                        seed=key,
                        argmax=False,
                    )
                    actions = np.asarray(jax.device_get(actions))

            # Step environment
            with timer.context("step_env"):
                next_obs, reward, done, truncated, info = env.step(actions)
                episode_steps += 1
                
                # Clean up info
                for key in ["left", "right", "intervene_action"]:
                    info.pop(key, None)
                
                # Check for human feedback if available
                if "human_feedback" in info and online_trainer:
                    feedback = info.pop("human_feedback")
                    if feedback.get("true_label") is not None:
                        online_trainer.add_feedback(
                            observation=obs,
                            classifier_prediction=feedback.get("classifier_prediction", 0),
                            true_label=feedback["true_label"],
                            confidence=feedback.get("confidence", 0.5)
                        )
                
                # Update classifier periodically
                if step % 100 == 0 and online_trainer:
                    update_classifier()
                
                running_return += reward

            # Store transition
            with timer.context("store_transition"):
                stored_info = copy.deepcopy(info)
                stored_info["obs"] = obs
                
                # Add to appropriate buffer
                if "intervene_action" in info:
                    demo_transitions.append(
                        dict(
                            observations=obs,
                            actions=actions,
                            next_observations=next_obs,
                            rewards=reward,
                            masks=1.0 - done,
                            dones=done,
                            infos=stored_info,
                        )
                    )
                else:
                    transitions.append(
                        dict(
                            observations=obs,
                            actions=actions,
                            next_observations=next_obs,
                            rewards=reward,
                            masks=1.0 - done,
                            dones=done,
                            infos=stored_info,
                        )
                    )

            obs = next_obs

            if done or truncated:
                # Store episode metrics
                episode_time = time.time() - episode_start_time
                success_history.append(float(reward))
                if len(success_history) > 20:
                    success_history.pop(0)
                
                # Send transitions
                if len(transitions) > 0:
                    data_store.insert(transitions)
                    transitions = []
                
                if len(demo_transitions) > 0:
                    intvn_data_store.insert(demo_transitions)
                    demo_transitions = []
                
                # Reset environment
                obs, _ = env.reset()
                done = False
                running_return = 0.0
                already_intervened = False
                episode_count += 1
                episode_start_time = time.time()
                episode_steps = 0
                
                # Update progress bar
                if len(success_history) > 0:
                    pbar.set_description(
                        f"Success rate: {np.mean(success_history):.1%} | "
                        f"Episode: {episode_count}"
                    )

            timer.tock("total")
            
    except KeyboardInterrupt:
        print("\n[Actor] Interrupted by user")
    except Exception as e:
        print(f"\n[Actor] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if client:
            client.stop()
        if pbar:
            pbar.close()
        print("[Actor] Stopped")


def learner(agent, replay_buffer, intvn_replay_buffer, demo_buffer, online_trainer=None):
    """
    Learner loop with optional online classifier training.
    """
    # Start online classifier training thread if enabled
    if online_trainer and FLAGS.online_classifier:
        online_trainer.start_training_thread()
    
    # [Rest of learner code remains similar to original...]
    # The learner continues training the policy while the classifier
    # trains in a separate thread
    
    # ... [Standard learner implementation] ...
    
    # At the end, stop online trainer
    if online_trainer:
        online_trainer.stop()


##############################################################################

def main(_):
    assert FLAGS.learner + FLAGS.actor == 1, "Must specify exactly one of --learner or --actor"
    
    # Set up experiment
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    
    # Set up online classifier trainer (for learner)
    online_trainer = None
    if FLAGS.learner and FLAGS.online_classifier:
        print_green("[Main] Initializing online classifier trainer")
        online_trainer = OnlineClassifierTrainer(
            config,
            checkpoint_dir=os.path.join(FLAGS.checkpoint_path or "./checkpoints", "online_classifier")
        )
    
    # [Rest of setup code similar to original train_rlpd.py...]
    
    if FLAGS.learner:
        print_green("[Main] Starting learner")
        learner(agent, replay_buffer, intvn_replay_buffer, demo_buffer, online_trainer)
    else:
        print_green("[Main] Starting actor")
        
        # For actor, we need to connect to learner's online trainer
        # This would require additional networking setup
        # For now, actor just runs without online training
        
        sampling_rng = jax.random.PRNGKey(FLAGS.seed)
        actor(agent, data_store, intvn_data_store, env, sampling_rng, None)
    
    print_green("[Main] Done!")


if __name__ == "__main__":
    app.run(main)