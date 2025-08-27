"""
Global camera resource manager to track and release all cameras
"""
import atexit
import weakref
import pyrealsense2 as rs


class CameraResourceManager:
    """Singleton manager to track all camera resources"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._captures = weakref.WeakSet()
        # Register cleanup on process exit
        atexit.register(self.cleanup_all)
    
    def register_capture(self, capture):
        """Register a capture device for tracking"""
        self._captures.add(capture)
    
    def cleanup_all(self):
        """Force cleanup of all registered captures"""
        print("[CameraResourceManager] Cleaning up all camera resources...")
        
        # Clean up all tracked captures
        for capture in list(self._captures):
            try:
                if hasattr(capture, 'close'):
                    capture.close()
            except Exception as e:
                print(f"[CameraResourceManager] Error closing capture: {e}")
        
        # Force reset all RealSense devices
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            
            for dev in devices:
                try:
                    # Hardware reset if available
                    if hasattr(dev, 'hardware_reset'):
                        dev.hardware_reset()
                except:
                    pass
        except Exception as e:
            print(f"[CameraResourceManager] Error resetting devices: {e}")
        
        print("[CameraResourceManager] Cleanup complete")


# Global instance
camera_manager = CameraResourceManager()