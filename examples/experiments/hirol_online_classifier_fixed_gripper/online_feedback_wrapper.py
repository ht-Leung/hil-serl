"""
Lightweight Human Feedback Wrapper for Online Learning

This wrapper is designed to work with train_rlpd_online_classifier.py
to provide thread-safe human feedback without file I/O conflicts.
"""

import time
import threading
import numpy as np
import jax
import jax.numpy as jnp
from gymnasium import Env
import gymnasium as gym
from pynput import keyboard
import queue


class OnlineFeedbackWrapper(gym.Wrapper):
    """
    Lightweight wrapper for collecting human feedback during training.
    
    This version is designed to work with train_rlpd_online_classifier.py
    and sends feedback through the info dict rather than saving to files.
    
    Keyboard controls:
    - 's': Confirm success (true positive)
    - 'f': Mark as false positive 
    - 'n': Mark as false negative
    - 'c': Skip/continue
    - 'p': Pause/unpause feedback collection
    """
    
    def __init__(
        self,
        env: Env,
        classifier_func=None,
        confidence_threshold=0.85,
        query_threshold=0.65,
        auto_query=True,
        target_hz=None
    ):
        super().__init__(env)
        
        self.classifier_func = classifier_func
        self.confidence_threshold = confidence_threshold
        self.query_threshold = query_threshold
        self.auto_query = auto_query
        self.target_hz = target_hz
        
        # State management
        self.lock = threading.Lock()
        self.feedback_queue = queue.Queue()
        self.paused = False
        self.episode_feedback_given = False
        
        # Statistics
        self.stats = {
            'queries': 0,
            'responses': 0,
            'skips': 0,
            'episode_count': 0
        }
        
        # Keyboard listener
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()
        
        print("\n" + "="*50)
        print("Online Feedback Wrapper Active")
        print("="*50)
        print("Keys: s=success, f=false_pos, n=false_neg, c=skip, p=pause")
        print("="*50 + "\n")
    
    def _on_press(self, key):
        """Handle keyboard input."""
        try:
            if hasattr(key, 'char'):
                if key.char == 'p':
                    with self.lock:
                        self.paused = not self.paused
                        status = "PAUSED" if self.paused else "ACTIVE"
                        print(f"\n[Feedback] {status}")
                elif key.char in ['s', 'f', 'n', 'c']:
                    self.feedback_queue.put(key.char)
        except AttributeError:
            pass
    
    def _get_classifier_prediction(self, obs):
        """Get classifier prediction with confidence."""
        if self.classifier_func is None:
            return 0, 0.0
        
        try:
            logit = self.classifier_func(obs)
            
            # Handle different output shapes
            if hasattr(logit, 'shape') and logit.shape:
                logit = logit.squeeze()
                if logit.shape:
                    logit = logit[0]
            
            # Convert to probability
            sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
            confidence = float(sigmoid(logit))
            
            # Binary prediction
            prediction = int(confidence > self.confidence_threshold)
            
            return prediction, confidence
        except Exception as e:
            print(f"[Feedback] Classifier error: {e}")
            return 0, 0.0
    
    def _should_query(self, confidence):
        """Determine if human feedback should be requested."""
        if self.paused or not self.auto_query:
            return False
        
        # Query in uncertain range
        in_uncertain_range = self.query_threshold <= confidence <= self.confidence_threshold
        
        # Also query occasionally for high confidence to check calibration
        random_check = np.random.random() < 0.05  # 5% random checks
        
        return in_uncertain_range or random_check
    
    def _collect_feedback(self, classifier_pred, confidence, timeout=0.3):
        """Collect human feedback with minimal blocking."""
        
        # Quick check if we should query
        should_query = self._should_query(confidence)
        
        if should_query and not self.episode_feedback_given:
            print(f"\r[Query] Pred: {'SUCCESS' if classifier_pred else 'FAILURE'} "
                  f"(conf: {confidence:.0%}) - Press s/f/n/c", end="", flush=True)
            self.stats['queries'] += 1
        
        # Try to get feedback (non-blocking)
        feedback_data = None
        try:
            key = self.feedback_queue.get_nowait()
            
            if key == 's':  # Success confirmed
                true_label = 1
                print(f"\r[Feedback] ✓ Success confirmed" + " "*30)
                self.stats['responses'] += 1
                self.episode_feedback_given = True
                
            elif key == 'f':  # False positive
                true_label = 0
                print(f"\r[Feedback] ✗ False positive" + " "*30)
                self.stats['responses'] += 1
                self.episode_feedback_given = True
                
            elif key == 'n':  # False negative
                true_label = 1
                print(f"\r[Feedback] ✗ False negative" + " "*30)
                self.stats['responses'] += 1
                self.episode_feedback_given = True
                
            elif key == 'c':  # Skip
                print(f"\r[Feedback] Skipped" + " "*30)
                self.stats['skips'] += 1
                return None
            else:
                return None
            
            # Create feedback data
            feedback_data = {
                'classifier_prediction': classifier_pred,
                'true_label': true_label,
                'confidence': confidence
            }
            
        except queue.Empty:
            # No feedback available
            if should_query and time.time() % 5 < 0.1:  # Reminder every 5 seconds
                print(f"\r[Waiting] Press s/f/n/c for feedback..." + " "*20, end="", flush=True)
        
        return feedback_data
    
    def step(self, action):
        """Step with optional feedback collection."""
        if self.target_hz is not None:
            start_time = time.time()
        
        # Execute environment step
        obs, rew, done, truncated, info = self.env.step(action)
        
        # Get classifier prediction
        classifier_pred, confidence = self._get_classifier_prediction(obs)
        
        # Try to collect feedback (non-blocking)
        feedback = self._collect_feedback(classifier_pred, confidence)
        
        # Add feedback to info if available
        if feedback is not None:
            info['human_feedback'] = feedback
            # Override reward with human label if provided
            rew = feedback['true_label']
            done = done or (rew == 1)
        else:
            # Use classifier prediction as reward
            rew = classifier_pred
            done = done or (rew == 1)
        
        info['succeed'] = bool(rew)
        info['classifier_confidence'] = confidence
        
        # Timing control
        if self.target_hz is not None:
            elapsed = time.time() - start_time
            sleep_time = 1.0 / self.target_hz - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        """Reset environment and feedback state."""
        obs, info = self.env.reset(**kwargs)
        
        with self.lock:
            self.episode_feedback_given = False
        
        # Clear feedback queue
        while not self.feedback_queue.empty():
            try:
                self.feedback_queue.get_nowait()
            except queue.Empty:
                break
        
        self.stats['episode_count'] += 1
        
        # Print stats periodically
        if self.stats['episode_count'] % 20 == 0:
            response_rate = (self.stats['responses'] / max(1, self.stats['queries'])) * 100
            print(f"\n[Stats] Episodes: {self.stats['episode_count']}, "
                  f"Queries: {self.stats['queries']}, "
                  f"Response rate: {response_rate:.0f}%\n")
        
        info['succeed'] = False
        return obs, info
    
    def close(self):
        """Clean up resources."""
        self.listener.stop()
        
        print("\n" + "="*50)
        print("Online Feedback Summary")
        print("="*50)
        print(f"Total episodes: {self.stats['episode_count']}")
        print(f"Total queries: {self.stats['queries']}")
        print(f"Total responses: {self.stats['responses']}")
        print(f"Total skips: {self.stats['skips']}")
        
        if self.stats['queries'] > 0:
            response_rate = (self.stats['responses'] / self.stats['queries']) * 100
            print(f"Response rate: {response_rate:.1f}%")
        print("="*50 + "\n")
        
        super().close()


class SimplifiedHumanRewardWrapper(gym.Wrapper):
    """
    Simplified version that only uses keyboard input for rewards.
    
    Press 's' at any time during an episode to mark it as successful.
    This is the most lightweight option with no classifier needed.
    """
    
    def __init__(self, env: Env):
        super().__init__(env)
        
        self.reward_flag = False
        self.lock = threading.Lock()
        
        # Keyboard listener
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()
        
        print("\n[SimplifiedReward] Press 's' to mark success\n")
    
    def _on_press(self, key):
        """Handle keyboard input."""
        try:
            if hasattr(key, 'char') and key.char == 's':
                with self.lock:
                    self.reward_flag = True
                    print("\r[Reward] Success marked!" + " "*30)
        except AttributeError:
            pass
    
    def step(self, action):
        """Step with manual reward override."""
        obs, rew, done, truncated, info = self.env.step(action)
        
        # Check for manual reward
        with self.lock:
            if self.reward_flag:
                rew = 1
                done = True
                info['succeed'] = True
                self.reward_flag = False
            else:
                rew = 0
                info['succeed'] = False
        
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        """Reset environment."""
        obs, info = self.env.reset(**kwargs)
        
        with self.lock:
            self.reward_flag = False
        
        info['succeed'] = False
        return obs, info
    
    def close(self):
        """Clean up."""
        self.listener.stop()
        super().close()