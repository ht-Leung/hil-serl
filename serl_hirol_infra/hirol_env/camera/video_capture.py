import queue
import threading
import time
import numpy as np

class VideoCapture:
    def __init__(self, cap, name=None):
        if name is None:
            name = cap.name
        self.name = name
        self.q = queue.Queue()
        self.cap = cap
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True  # Change to daemon thread for better cleanup
        self.enable = True
        self._closed = False
        self.t.start()
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up"""
        if not self._closed:
            self.close()

    def _reader(self):
        while self.enable:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        # print(self.name, self.q.qsize())
        return self.q.get(timeout=5)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.enable = False
        
        # Clear the queue to unblock any waiting reads
        try:
            while not self.q.empty():
                self.q.get_nowait()
        except:
            pass
        
        # Wait for thread with timeout to prevent hanging
        if self.t.is_alive():
            self.t.join(timeout=2.0)
            if self.t.is_alive():
                print(f"Warning: Camera thread {self.name} did not stop cleanly")
        
        # Close the capture device regardless
        try:
            self.cap.close()
        except Exception as e:
            print(f"Error closing capture device {self.name}: {e}")
