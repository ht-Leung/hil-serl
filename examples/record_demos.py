import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time
import signal
import sys

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")

def main(_):
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
        assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
        config = CONFIG_MAPPING[FLAGS.exp_name]()
        env = config.get_environment(fake_env=False, save_video=False, classifier=True)
        
        obs, info = env.reset()
        print("Reset done")
        transitions = []
        success_count = 0
        success_needed = FLAGS.successes_needed
        pbar = tqdm(total=success_needed)
        trajectory = []
        returns = 0
        
        while success_count < success_needed:
            actions = np.zeros(env.action_space.sample().shape) 
            next_obs, rew, done, truncated, info = env.step(actions)
            returns += rew
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
                    infos=info,
                )
            )
            trajectory.append(transition)
            
            pbar.set_description(f"Return: {returns}")

            obs = next_obs
            if done:
                if info["succeed"]:
                    for transition in trajectory:
                        transitions.append(copy.deepcopy(transition))
                    success_count += 1
                    pbar.update(1)
                trajectory = []
                returns = 0
                obs, info = env.reset()
                
        # Use config-specific demo data path
        demo_data_dir = config.demo_data_path if hasattr(config, 'demo_data_path') else "./demo_data"
        if not os.path.exists(demo_data_dir):
            os.makedirs(demo_data_dir)
        uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{demo_data_dir}/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
        with open(file_name, "wb") as f:
            pkl.dump(transitions, f)
            print(f"saved {success_needed} demos to {file_name}")
            
        # Final reset before cleanup
        print("\nReturning to reset position...")
        env.reset()
        
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        raise
    finally:
        print("\n[INFO] Starting cleanup...")
        
        # Close progress bar
        if pbar is not None:
            try:
                pbar.close()
            except Exception as e:
                print(f"[WARNING] Error closing progress bar: {e}")
        
        # Close environment
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