import numpy as np
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
import time
from .camera_manager import camera_manager


class RSCapture:
    def get_device_serial_numbers(self):
        devices = rs.context().devices
        return [d.get_info(rs.camera_info.serial_number) for d in devices]
    
    def __del__(self):
        """Destructor to ensure camera resources are released"""
        try:
            self.close()
        except:
            pass

    def __init__(self, name, serial_number, dim=(640, 480), fps=15, depth=False, exposure=None):
        self.name = name
        self.serial_number = serial_number
        self.depth = depth
        self.pipe = None
        self.cfg = None
        self._is_open = False
        
        # Register with camera manager for cleanup
        camera_manager.register_capture(self)
        
        # Check device availability
        assert serial_number in self.get_device_serial_numbers(), f"Device {serial_number} not found"
        
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial_number)
        self.cfg.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.bgr8, fps)
        if self.depth:
            self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)
        
        # Try to start with retry logic for busy devices
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.profile = self.pipe.start(self.cfg)
                self._is_open = True
                break
            except RuntimeError as e:
                if "busy" in str(e).lower() and attempt < max_retries - 1:
                    print(f"Camera {name} busy, retrying in 2 seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(2)
                else:
                    raise
        self.s = self.profile.get_device().query_sensors()[1] # fix index for color sensor
        if exposure is not None:
            self.s.set_option(rs.option.exposure, exposure)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # Store camera intrinsics for point cloud conversion
        color_stream = self.profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        # Get depth scale if depth is enabled
        depth_scale = 0.001  # default value
        if self.depth:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()

        self.intrinsics = {
            'fx': intrinsics.fx,
            'fy': intrinsics.fy,
            'cx': intrinsics.ppx,
            'cy': intrinsics.ppy,
            'width': intrinsics.width,
            'height': intrinsics.height,
            'depth_scale': depth_scale
        }

    def get_intrinsics(self):
        """Get camera intrinsic parameters"""
        return self.intrinsics

    def read(self):
        frames = self.pipe.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if self.depth:
            depth_frame = aligned_frames.get_depth_frame()

        if color_frame.is_video_frame():
            image = np.asarray(color_frame.get_data())
            if self.depth and depth_frame.is_depth_frame():
                # Convert raw depth to meters using depth scale
                raw_depth = np.asarray(depth_frame.get_data())
                depth_meters = raw_depth * self.intrinsics['depth_scale']

                # Store depth in meters for DepthToPointCloud compatibility
                depth = np.expand_dims(depth_meters, axis=2)
                return True, np.concatenate((image, depth), axis=-1)
            else:
                return True, image
        else:
            return False, None

    def close(self):
        """Close the camera and release all resources"""
        if not self._is_open:
            return
        
        self._is_open = False
        
        try:
            # Stop the pipeline first
            if self.pipe:
                self.pipe.stop()
        except Exception as e:
            print(f"Warning: Error stopping pipeline for {self.name}: {e}")
        
        try:
            # Disable all streams
            if self.cfg:
                self.cfg.disable_all_streams()
        except Exception as e:
            print(f"Warning: Error disabling streams for {self.name}: {e}")
        
        # Clear references
        self.pipe = None
        self.cfg = None
        self.profile = None
