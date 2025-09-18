"""
Human-in-the-Loop Feedback Classifier Wrapper

This module provides a wrapper that combines classifier predictions with human feedback
to reduce false positives and improve classifier accuracy through online learning.
"""

import time
import threading
import pickle as pkl
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import numpy as np
import jax
import jax.numpy as jnp
from gymnasium import Env
import gymnasium as gym
from pynput import keyboard
from collections import deque
import queue


class HumanFeedbackClassifierWrapper(gym.Wrapper):
    """
    Wrapper that combines classifier predictions with human feedback.
    
    Features:
    - Real-time human correction of classifier predictions
    - Automatic data collection for classifier retraining  
    - Multiple feedback modes via keyboard shortcuts
    - Confidence-based queries to reduce human burden
    
    Keyboard controls:
    - 's': Confirm success (true positive)
    - 'f': Mark as false positive (classifier wrong, actually failure)
    - 'n': Mark as false negative (classifier missed success)
    - 'c': Skip/continue (no feedback)
    - 'r': Force classifier retraining
    """
    
    def __init__(
        self,
        env: Env,
        classifier_func: Optional[Callable] = None,
        confidence_threshold: float = 0.85,
        query_threshold: float = 0.65,  # Query human when confidence is between query and confidence threshold
        feedback_buffer_size: int = 1000,
        auto_retrain_interval: int = 100,  # Retrain after N feedback samples
        save_feedback_path: Optional[str] = None,
        target_hz: Optional[float] = None
    ):
        super().__init__(env)
        
        # Classifier settings
        self.classifier_func = classifier_func
        self.confidence_threshold = confidence_threshold
        self.query_threshold = query_threshold
        self.target_hz = target_hz
        
        # Feedback collection
        self.feedback_buffer = deque(maxlen=feedback_buffer_size)
        self.feedback_count = 0
        self.auto_retrain_interval = auto_retrain_interval
        self.save_feedback_path = save_feedback_path or "./feedback_data"
        Path(self.save_feedback_path).mkdir(parents=True, exist_ok=True)
        
        # Threading for keyboard input
        self.lock = threading.Lock()
        self.feedback_queue = queue.Queue()
        self.pending_observation = None
        self.classifier_prediction = None
        self.awaiting_feedback = False
        
        # Statistics tracking
        self.stats = {
            'true_positive': 0,
            'false_positive': 0,
            'false_negative': 0,
            'true_negative': 0,
            'total_queries': 0,
            'total_episodes': 0
        }
        
        # Start keyboard listener
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()
        
        print("\n" + "="*60)
        print("Human Feedback Classifier Wrapper Initialized")
        print("="*60)
        print("Keyboard Controls:")
        print("  's' - Confirm success (true positive)")
        print("  'f' - Mark false positive (actually failure)")
        print("  'n' - Mark false negative (missed success)")
        print("  'c' - Skip/continue (no feedback)")
        print("  'r' - Force classifier retraining")
        print("="*60 + "\n")
    
    def _on_press(self, key):
        """Handle keyboard press events"""
        try:
            if hasattr(key, 'char') and key.char in ['s', 'f', 'n', 'c', 'r']:
                self.feedback_queue.put(key.char)
        except AttributeError:
            pass
    
    def _get_classifier_prediction(self, obs):
        """Get classifier prediction with confidence score"""
        if self.classifier_func is None:
            return 0, 0.0  # No classifier, return failure with no confidence
        
        logit = self.classifier_func(obs)
        
        # Handle both scalar and array outputs
        if hasattr(logit, 'shape') and logit.shape:
            logit = logit.squeeze()
            if logit.shape:  # Still has dimensions
                logit = logit[0]
        
        # Convert to probability
        sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
        confidence = float(sigmoid(logit))
        
        # Binary prediction based on threshold
        prediction = int(confidence > self.confidence_threshold)
        
        return prediction, confidence
    
    def _should_query_human(self, confidence):
        """Determine if human feedback should be requested"""
        # Query when confidence is in uncertain range
        return self.query_threshold <= confidence <= self.confidence_threshold
    
    def _collect_feedback(self, obs, classifier_pred, confidence, timeout=0.5):
        """Collect human feedback for a prediction"""
        with self.lock:
            self.pending_observation = obs
            self.classifier_prediction = classifier_pred
            self.awaiting_feedback = True
        
        # Display prediction info
        print(f"\n[Classifier] Prediction: {'SUCCESS' if classifier_pred else 'FAILURE'} "
              f"(confidence: {confidence:.2%})")
        
        if self._should_query_human(confidence):
            print("[Query] Uncertain prediction - please provide feedback!")
            self.stats['total_queries'] += 1
        
        # Wait for feedback with timeout
        try:
            feedback = self.feedback_queue.get(timeout=timeout)
            
            with self.lock:
                self.awaiting_feedback = False
                
                # Process feedback
                true_label = None
                if feedback == 's':  # Confirm success
                    true_label = 1
                    if classifier_pred == 1:
                        self.stats['true_positive'] += 1
                        print("[Feedback] ✓ Correct success prediction")
                    else:
                        self.stats['false_negative'] += 1
                        print("[Feedback] ✗ Missed success (false negative)")
                
                elif feedback == 'f':  # False positive
                    true_label = 0
                    self.stats['false_positive'] += 1
                    print("[Feedback] ✗ False positive - actually failure")
                
                elif feedback == 'n':  # False negative
                    true_label = 1
                    self.stats['false_negative'] += 1
                    print("[Feedback] ✗ False negative - missed success")
                
                elif feedback == 'c':  # Skip
                    print("[Feedback] Skipped")
                    return None
                
                elif feedback == 'r':  # Force retrain
                    print("[Feedback] Retraining requested")
                    self._trigger_retraining()
                    return None
                
                # Store feedback data
                if true_label is not None:
                    feedback_data = {
                        'observation': obs,
                        'classifier_prediction': classifier_pred,
                        'confidence': confidence,
                        'true_label': true_label,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.feedback_buffer.append(feedback_data)
                    self.feedback_count += 1
                    
                    # Auto-retrain check
                    if self.feedback_count % self.auto_retrain_interval == 0:
                        self._trigger_retraining()
                    
                    return true_label
                    
        except queue.Empty:
            with self.lock:
                self.awaiting_feedback = False
            return None
        
        return None
    
    def _trigger_retraining(self):
        """Save feedback data and trigger classifier retraining"""
        if len(self.feedback_buffer) == 0:
            print("[Retrain] No feedback data to save")
            return
        
        # Save feedback data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feedback_file = Path(self.save_feedback_path) / f"feedback_{timestamp}.pkl"
        
        with open(feedback_file, 'wb') as f:
            pkl.dump(list(self.feedback_buffer), f)
        
        print(f"[Retrain] Saved {len(self.feedback_buffer)} feedback samples to {feedback_file}")
        print(f"[Retrain] Stats: TP={self.stats['true_positive']}, "
              f"FP={self.stats['false_positive']}, "
              f"FN={self.stats['false_negative']}, "
              f"TN={self.stats['true_negative']}")
        
        # TODO: Trigger actual retraining process here
        # This could be done via:
        # 1. Calling a retraining script
        # 2. Sending a signal to a separate training process
        # 3. Using an online learning update
        
    def step(self, action):
        if self.target_hz is not None:
            start_time = time.time()
        
        # Execute environment step
        obs, rew, done, truncated, info = self.env.step(action)
        
        # Get classifier prediction
        classifier_pred, confidence = self._get_classifier_prediction(obs)
        
        # Collect human feedback (non-blocking with short timeout)
        human_label = self._collect_feedback(obs, classifier_pred, confidence, timeout=0.1)
        
        # Determine final reward
        if human_label is not None:
            # Use human feedback as ground truth
            rew = human_label
            info['reward_source'] = 'human'
        else:
            # Use classifier prediction
            rew = classifier_pred
            info['reward_source'] = 'classifier'
        
        # Update episode termination
        done = done or rew == 1
        info['succeed'] = bool(rew)
        info['classifier_confidence'] = confidence
        
        # Timing control
        if self.target_hz is not None:
            elapsed_time = time.time() - start_time
            sleep_time = 1.0 / self.target_hz - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        with self.lock:
            self.pending_observation = None
            self.classifier_prediction = None
            self.awaiting_feedback = False
        
        self.stats['total_episodes'] += 1
        
        # Clear feedback queue
        while not self.feedback_queue.empty():
            try:
                self.feedback_queue.get_nowait()
            except queue.Empty:
                break
        
        info['succeed'] = False
        return obs, info
    
    def close(self):
        """Clean up resources"""
        # Save final feedback data
        self._trigger_retraining()
        
        # Stop keyboard listener
        self.listener.stop()
        
        # Print final statistics
        print("\n" + "="*60)
        print("Human Feedback Classifier - Final Statistics")
        print("="*60)
        print(f"Total Episodes: {self.stats['total_episodes']}")
        print(f"Total Queries: {self.stats['total_queries']}")
        print(f"True Positives: {self.stats['true_positive']}")
        print(f"False Positives: {self.stats['false_positive']}")
        print(f"False Negatives: {self.stats['false_negative']}")
        print(f"True Negatives: {self.stats['true_negative']}")
        
        total_feedback = (self.stats['true_positive'] + self.stats['false_positive'] + 
                         self.stats['false_negative'] + self.stats['true_negative'])
        if total_feedback > 0:
            accuracy = (self.stats['true_positive'] + self.stats['true_negative']) / total_feedback
            print(f"Feedback Accuracy: {accuracy:.2%}")
        print("="*60 + "\n")
        
        super().close()


class AdaptiveClassifierWrapper(HumanFeedbackClassifierWrapper):
    """
    Extended version with online learning capabilities.
    
    This wrapper can update the classifier in real-time based on human feedback,
    reducing the need for full retraining cycles.
    """
    
    def __init__(
        self,
        env: Env,
        classifier_func: Optional[Callable] = None,
        classifier_update_func: Optional[Callable] = None,
        update_frequency: int = 10,  # Update classifier every N feedback samples
        **kwargs
    ):
        super().__init__(env, classifier_func, **kwargs)
        
        self.classifier_update_func = classifier_update_func
        self.update_frequency = update_frequency
        self.online_buffer = []
    
    def _collect_feedback(self, obs, classifier_pred, confidence, timeout=0.5):
        """Extended to support online updates"""
        true_label = super()._collect_feedback(obs, classifier_pred, confidence, timeout)
        
        if true_label is not None and self.classifier_update_func is not None:
            # Add to online buffer
            self.online_buffer.append({
                'observation': obs,
                'label': true_label
            })
            
            # Perform online update if buffer is full
            if len(self.online_buffer) >= self.update_frequency:
                print(f"[Online Update] Updating classifier with {len(self.online_buffer)} samples")
                
                # Call update function (implementation depends on classifier type)
                self.classifier_func = self.classifier_update_func(
                    self.classifier_func,
                    self.online_buffer
                )
                
                # Clear buffer
                self.online_buffer = []
        
        return true_label