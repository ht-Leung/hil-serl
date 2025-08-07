import multiprocessing
import numpy as np
import time
from hirol_env.spacemouse import pyspacemouse
from typing import Tuple


class SpaceMouseExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provides
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self):
        pyspacemouse.open()

        # Manager to handle shared state between processes
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6  # Using lists for compatibility
        self.latest_data["buttons"] = [0, 0, 0, 0]
        
        # Add running flag to gracefully stop the process
        self.running = self.manager.Value('b', True)

        # Start a process to continuously read the SpaceMouse state
        self.process = multiprocessing.Process(target=self._read_spacemouse)
        self.process.daemon = True
        self.process.start()

    def _read_spacemouse(self):
        while self.running.value:
            try:
                state = pyspacemouse.read_all()
                action = [0.0] * 6
                buttons = [0, 0, 0, 0]

                if len(state) == 2:
                    action = [
                        -state[0].y, state[0].x, state[0].z,
                        -state[0].roll, -state[0].pitch, -state[0].yaw,
                        -state[1].y, state[1].x, state[1].z,
                        -state[1].roll, -state[1].pitch, -state[1].yaw
                    ]
                    buttons = state[0].buttons + state[1].buttons
                elif len(state) == 1:
                    action = [
                        -state[0].y, state[0].x, state[0].z,
                        -state[0].roll, -state[0].pitch, -state[0].yaw
                    ]
                    buttons = state[0].buttons

                # Update the shared state
                self.latest_data["action"] = action
                self.latest_data["buttons"] = buttons
            except (EOFError, ConnectionResetError, BrokenPipeError):
                # Gracefully handle connection errors during shutdown
                break
            except Exception as e:
                print(f"SpaceMouse read error: {e}")
                time.sleep(0.01)

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        action = self.latest_data["action"]
        buttons = self.latest_data["buttons"]
        return np.array(action), buttons
    
    def close(self):
        """Gracefully close the SpaceMouse connection and stop the reading process"""
        # Signal the process to stop
        self.running.value = False
        
        # Give it a moment to finish gracefully
        self.process.join(timeout=0.5)
        
        # If it's still running, terminate it
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=0.5)
        
        # Force kill if still alive
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        
        # Close SpaceMouse connection
        try:
            pyspacemouse.close()
        except:
            pass
    
    def __del__(self):
        """Ensure cleanup on deletion"""
        try:
            self.close()
        except:
            pass
