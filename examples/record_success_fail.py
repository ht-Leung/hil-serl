import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
from pynput import keyboard
import signal
import sys

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful transistions to collect.")


success_key = False
def on_press(key):
    global success_key
    try:
        if str(key) == 'Key.space':
            success_key = True
    except AttributeError:
        pass

def main(_):
    global success_key
    listener = None
    env = None
    pbar = None
    cleanup_in_progress = False
    
    def signal_handler(signum, frame):
        nonlocal cleanup_in_progress
        if not cleanup_in_progress:
            print("\n\n[INFO] Interrupt received. Cleaning up resources...")
            cleanup_in_progress = True
            raise KeyboardInterrupt
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
        config = CONFIG_MAPPING[FLAGS.exp_name]()
        env = config.get_environment(fake_env=False, save_video=False, classifier=False)

        obs, _ = env.reset()
        successes = []
        failures = []
        success_needed = FLAGS.successes_needed
        pbar = tqdm(total=success_needed)
        
        while len(successes) < success_needed:
            actions = np.zeros(env.action_space.sample().shape) 
            next_obs, rew, done, truncated, info = env.step(actions)
            if "intervene_action" in info:
                actions = info["intervene_action"]

            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                )
            )
            obs = next_obs
            if success_key:
                successes.append(transition)
                pbar.update(1)
                success_key = False
            else:
                failures.append(transition)

            if done or truncated:
                obs, _ = env.reset()

        # Use config-specific classifier data path
        classifier_data_dir = config.classifier_data_path if hasattr(config, 'classifier_data_path') else "./classifier_data"
        if not os.path.exists(classifier_data_dir):
            os.makedirs(classifier_data_dir)
        uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{classifier_data_dir}/{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"
        with open(file_name, "wb") as f:
            pkl.dump(successes, f)
            print(f"saved {success_needed} successful transitions to {file_name}")

        file_name = f"{classifier_data_dir}/{FLAGS.exp_name}_failure_images_{uuid}.pkl"
        with open(file_name, "wb") as f:
            pkl.dump(failures, f)
            print(f"saved {len(failures)} failure transitions to {file_name}")
        
        # Final reset to safe position before cleanup
        print("\nReturning to reset position...")
        env.reset()
        
    finally:
        # Cleanup
        print("\n[INFO] Starting cleanup...")
        
        # Close progress bar
        if pbar is not None:
            try:
                pbar.close()
            except Exception as e:
                print(f"[WARNING] Error closing progress bar: {e}")
        
        # Stop keyboard listener
        if listener is not None:
            try:
                listener.stop()
            except Exception as e:
                print(f"[WARNING] Error stopping listener: {e}")
        
        # Close environment (which will close SpaceMouse and cameras)
        if env is not None:
            try:
                if hasattr(env, 'close'):
                    env.close()
                    print("[INFO] Environment closed successfully")
            except Exception as e:
                print(f"[WARNING] Error closing environment: {e}")
        
        print("[INFO] Cleanup completed successfully.")
        
if __name__ == "__main__":
    app.run(main)
