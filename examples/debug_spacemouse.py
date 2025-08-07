#!/usr/bin/env python3
"""
Debug SpaceMouse output to understand the actual value ranges
"""

import sys
import numpy as np
import time

sys.path.insert(0, '/home/hanyu/code/hil-serl/serl_hirol_infra')

from hirol_env.spacemouse.spacemouse_expert import SpaceMouseExpert


def debug_spacemouse():
    """Monitor SpaceMouse output values"""
    print("=" * 60)
    print("SpaceMouse Debug Monitor")
    print("Move the SpaceMouse to see raw output values")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    expert = SpaceMouseExpert()
    
    max_vals = np.zeros(6)
    min_vals = np.zeros(6)
    
    try:
        while True:
            action, buttons = expert.get_action()
            
            # Update max/min
            max_vals = np.maximum(max_vals, action[:6])
            min_vals = np.minimum(min_vals, action[:6])
            
            # Only print if there's significant movement
            if np.linalg.norm(action[:6]) > 0.01:
                print(f"\rAction: X:{action[0]:6.2f} Y:{action[1]:6.2f} Z:{action[2]:6.2f} "
                      f"RX:{action[3]:6.2f} RY:{action[4]:6.2f} RZ:{action[5]:6.2f} "
                      f"Buttons: {buttons[:2]}", end="")
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Summary:")
        print(f"Max values: {max_vals}")
        print(f"Min values: {min_vals}")
        print("=" * 60)
    finally:
        expert.close()


if __name__ == "__main__":
    debug_spacemouse()