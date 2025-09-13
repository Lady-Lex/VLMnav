#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import time
import numpy as np
from datetime import datetime

try:
    import roslibpy
    from roslibpy import Ros, Topic
except ImportError:
    print("Error: roslibpy not found. Please install with: pip install roslibpy")
    exit(1)

from agent import GOATAgent
from my_ros.utils import SixDOF, format_ros_timestamp
from my_ros.tf_manager import TFManager
from my_ros.sensor_manager import SensorManager
from my_ros.control_manager import ControlManager

class VLMNavNode:
    """Main VLM Navigation Node - coordinates all subsystems"""
    
    def __init__(self):
        # ROS connection
        self.ros_host = 'localhost'
        self.ros_port = 9090
        self.ros = Ros(self.ros_host, self.ros_port)
        self.ros.run()

        # Wait for connection
        while not self.ros.is_connected:
            time.sleep(0.05)
        print(f"Connected to ROS at {self.ros_host}:{self.ros_port}")

        # Load configuration
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'ArenaBenchmark.yaml')
        print(f"Loading configuration from {cfg_path}")
        self.agent_cfg = self._load_agent_cfg(cfg_path)

        # Frame names from configuration
        self.base_frame = self.agent_cfg.get("base_frame", "jackal/base_link")
        self.odom_frame = self.agent_cfg.get("odom_frame", "jackal/odom")
        self.camera_frame = self.agent_cfg.get("camera_frame", "jackal/d415_depth_optical_frame")
        
        # Initialize subsystems
        self.tf_manager = TFManager(
            debug_timestamps=bool(self.agent_cfg.get("debug_timestamps", False)),
            max_tf_age=float(self.agent_cfg.get("tf_max_age", 10.0))
        )
        
        self.sensor_manager = SensorManager(self.ros, self.agent_cfg)
        
        self.control_manager = ControlManager(self.ros, self.agent_cfg)

        # Subscribe to TF topics
        self.tf_manager.subscribe_to_tf_topics(self.ros)

        # Subscribe to sensor topics
        self.sensor_manager.subscribe_to_topics()

        # Initialize agent
        self.agent = GOATAgent(self.agent_cfg)
        self.agent.reset()

        # Initialize goal from configuration
        self._initialize_goal_from_config(cfg_path)

        # Wait for TF to be available before entering main loop
        self.tf_manager.wait_for_tf_ready(
            self.odom_frame, 
            self.base_frame, 
            timeout_sec=float(self.agent_cfg.get("tf_wait_timeout", 5.0))
        )

    def _load_agent_cfg(self, cfg_path):
        """Load agent configuration with defaults"""
        defaults = {
            'sensor_cfg': {'fov': 131, 'res_factor': 1, 'width': 640, 'height': 480},
            'map_scale': 100,
            'navigability_mode': 'depth_sensor',
            'project': True,
            'pivot': False,
            'max_action_dist': 1.7,
            'turn_around_cooldown': 3,
            'navigability_height_threshold': 0.2,
            'min_action_dist': 0.5,
            'clip_frac': 0.66,
            'stopping_action_dist': 1.5,
            'default_action': 0.2,
            'spacing_ratio': 360,
            'num_theta': 60,
            'image_edge_threshold': 0.04,
            'explore_bias': 4,
            'context_history': 0,
            'vlm_cfg': {
                'model_cls': 'GeminiVLM',
                'model_kwargs': {'model': 'gemini-1.5-flash-001'}
            }
        }
        try:
            with open(cfg_path, 'r') as f:
                yaml_cfg = yaml.safe_load(f)
            agent_cfg = yaml_cfg.get("agent_cfg", {})
            # Apply default values
            for k, v in defaults.items():
                if k not in agent_cfg:
                    agent_cfg[k] = v
            print(f"Loaded configuration from {cfg_path}")
            return agent_cfg
        except Exception as e:
            print(f"Failed to load configuration from {cfg_path}: {e}")
            print("Using default configuration")
            return defaults

    def _initialize_goal_from_config(self, cfg_path):
        """Initialize goal from configuration file"""
        try:
            with open(cfg_path, 'r') as f:
                yaml_cfg = yaml.safe_load(f)
            goal_cfg = yaml_cfg.get("goal", {})
            if goal_cfg:
                self.sensor_manager.goal = goal_cfg
                print(f"Initial goal set from config: {self.sensor_manager.goal}")
        except Exception as e:
            print(f"Failed to load goal from config: {e}")

    def wait_for_synchronized_data(self, timeout=None):
        """Wait for all sensor data to have similar timestamps and return synchronized data"""
        reference_base_trans, reference_base_quat, reference_tf_ts = self.tf_manager.lookup_pose(self.odom_frame, self.base_frame)
        reference_base_pose = (reference_base_trans, reference_base_quat)
        
        if not self.sensor_manager.sync_timestamps:
            return True, self.sensor_manager.rgb, self.sensor_manager.depth, reference_base_pose
        
        if self.sensor_manager.sync_strategy == "use_latest":
            # Just ensure all data is available, don't wait for sync
            available = self._ensure_data_available()
            return available, self.sensor_manager.rgb, self.sensor_manager.depth, reference_base_pose
        
        if timeout is None:
            timeout = self.sensor_manager.sync_timeout
        
        # FIXED: Choose the latest sensor as reference and freeze its timestamp
        # Wait for other sensors to match this reference timestamp
        reference_rgb_ts = self.sensor_manager.last_rgb_ts
        reference_depth_ts = self.sensor_manager.last_depth_ts
        reference_rgb_data = self.sensor_manager.rgb.copy() if self.sensor_manager.rgb is not None else None
        reference_depth_data = self.sensor_manager.depth.copy() if self.sensor_manager.depth is not None else None
        
        # Check if we have at least one reference timestamp
        if reference_rgb_ts is None and reference_depth_ts is None and reference_tf_ts is None:
            if self.sensor_manager.debug_timestamps:
                print("No reference timestamps available, cannot sync")
            return False, reference_rgb_data, reference_depth_data, reference_base_pose
        
        # Choose the latest timestamp as the reference target
        reference_timestamps = [ts for ts in [reference_rgb_ts, reference_depth_ts, reference_tf_ts] if ts is not None]
        if not reference_timestamps:
            if self.sensor_manager.debug_timestamps:
                print("No valid reference timestamps")
            return False, reference_rgb_data, reference_depth_data, reference_base_pose
        
        target_timestamp = max(reference_timestamps)
        
        # Determine which sensor has the latest timestamp
        if reference_tf_ts == target_timestamp:
            reference_sensor = "TF"
        elif reference_depth_ts == target_timestamp:
            reference_sensor = "Depth"
        elif reference_rgb_ts == target_timestamp:
            reference_sensor = "RGB"
        else:
            reference_sensor = "Unknown"
        
        if self.sensor_manager.debug_timestamps:
            print(f"Sync target timestamp ({reference_sensor}): {target_timestamp:.6f}")
            rgb_str = f"{reference_rgb_ts:.6f}" if reference_rgb_ts is not None else "None"
            depth_str = f"{reference_depth_ts:.6f}" if reference_depth_ts is not None else "None"
            tf_str = f"{reference_tf_ts:.6f}" if reference_tf_ts is not None else "None"
            print(f"Reference timestamps - RGB: {rgb_str}, Depth: {depth_str}, TF: {tf_str}")
            print(f"Waiting for other sensors to match {reference_sensor} timestamp...")
        
        start_time = time.time()
        last_debug_time = 0
        debug_interval = 0.1  # Print debug info every 100ms
        
        # Store the best synchronized data found so far
        # Initialize with reference data (all cases start the same)
        best_rgb_data = reference_rgb_data
        best_depth_data = reference_depth_data
        best_base_pose = reference_base_pose
        
        # Track which sensors are already synchronized (within threshold)
        rgb_synced = False
        depth_synced = False
        tf_synced = False
        
        best_sync_quality = float('inf')
        
        while time.time() - start_time < timeout:
            current_wait_time = time.time() - start_time
            
            # Get current timestamps - only query unsynced sensors
            rgb_ts = reference_rgb_ts if rgb_synced else self.sensor_manager.last_rgb_ts
            depth_ts = reference_depth_ts if depth_synced else self.sensor_manager.last_depth_ts
            
            # Only query TF if not synced yet
            if tf_synced:
                tf_ts = reference_tf_ts
                base_trans, base_quat = reference_base_pose[0], reference_base_pose[1]
            else:
                base_trans, base_quat, tf_ts = self.tf_manager.lookup_pose(self.odom_frame, self.base_frame)
            
            # Check if all required data is available
            if rgb_ts is None or depth_ts is None or tf_ts is None:
                if self.sensor_manager.debug_timestamps and current_wait_time - last_debug_time > debug_interval:
                    missing = []
                    if rgb_ts is None: missing.append("RGB")
                    if depth_ts is None: missing.append("depth")
                    if tf_ts is None: missing.append("TF")
                    print(f"Waiting for data: missing {', '.join(missing)}")
                    last_debug_time = current_wait_time
                time.sleep(0.01)
                continue
            
            # Calculate timestamp differences from the target timestamp
            rgb_diff = abs(rgb_ts - target_timestamp) if rgb_ts is not None else float('inf')
            depth_diff = abs(depth_ts - target_timestamp) if depth_ts is not None else float('inf')
            tf_diff = abs(tf_ts - target_timestamp) if tf_ts is not None else float('inf')
            
            # Check which sensors are within sync threshold
            rgb_within_threshold = rgb_diff <= self.sensor_manager.sync_threshold
            depth_within_threshold = depth_diff <= self.sensor_manager.sync_threshold
            tf_within_threshold = tf_diff <= self.sensor_manager.sync_threshold
            
            # Update sync status - once a sensor is synced, keep it fixed
            if rgb_within_threshold and not rgb_synced:
                rgb_synced = True
                reference_rgb_ts = rgb_ts  # Update reference to the synced timestamp
                best_rgb_data = self.sensor_manager.rgb.copy() if self.sensor_manager.rgb is not None else None
                if self.sensor_manager.debug_timestamps:
                    print(f"RGB synchronized: diff={rgb_diff:.6f}s, threshold={self.sensor_manager.sync_threshold:.6f}s")
            
            if depth_within_threshold and not depth_synced:
                depth_synced = True
                reference_depth_ts = depth_ts  # Update reference to the synced timestamp
                best_depth_data = self.sensor_manager.depth.copy() if self.sensor_manager.depth is not None else None
                if self.sensor_manager.debug_timestamps:
                    print(f"Depth synchronized: diff={depth_diff:.6f}s, threshold={self.sensor_manager.sync_threshold:.6f}s")
            
            if tf_within_threshold and not tf_synced:
                tf_synced = True
                reference_tf_ts = tf_ts  # Update reference to the synced timestamp
                best_base_pose = (base_trans, base_quat)
                if self.sensor_manager.debug_timestamps:
                    print(f"TF synchronized: diff={tf_diff:.6f}s, threshold={self.sensor_manager.sync_threshold:.6f}s")
            
            # Update best sync quality for non-synced sensors only
            current_max_diff = max(rgb_diff, depth_diff, tf_diff)
            
            # Only update data if we have unsynced sensors and overall quality improved
            has_unsynced = not rgb_synced or not depth_synced or not tf_synced
            if has_unsynced and current_max_diff < best_sync_quality:
                best_sync_quality = current_max_diff
                
                # Only update data for sensors that haven't been synced yet
                if not rgb_synced:
                    best_rgb_data = self.sensor_manager.rgb.copy() if self.sensor_manager.rgb is not None else None
                if not depth_synced:
                    best_depth_data = self.sensor_manager.depth.copy() if self.sensor_manager.depth is not None else None
                if not tf_synced:
                    best_base_pose = (base_trans, base_quat)

            # Check if all sensors are synchronized
            if rgb_within_threshold and depth_within_threshold and tf_within_threshold:
                if self.sensor_manager.debug_timestamps:
                    print(f"All sensors synchronized: max_diff={current_max_diff:.6f}s, threshold={self.sensor_manager.sync_threshold:.6f}s")
                    print(f"  RGB: {rgb_ts:.6f} (diff: {rgb_diff:.6f}s)")
                    print(f"  Depth: {depth_ts:.6f} (diff: {depth_diff:.6f}s)")
                    print(f"  TF: {tf_ts:.6f} (diff: {tf_diff:.6f}s)")
                return True, best_rgb_data, best_depth_data, best_base_pose
            
            if self.sensor_manager.debug_timestamps and current_wait_time - last_debug_time > debug_interval:
                print(f"Waiting for sync: max_diff={current_max_diff:.6f}s > threshold={self.sensor_manager.sync_threshold:.6f}s")
                print(f"  RGB: {rgb_ts:.6f} (diff: {rgb_diff:.6f}s) {'[SYNCED]' if rgb_synced else ''}")
                print(f"  Depth: {depth_ts:.6f} (diff: {depth_diff:.6f}s) {'[SYNCED]' if depth_synced else ''}")
                print(f"  TF: {tf_ts:.6f} (diff: {tf_diff:.6f}s) {'[SYNCED]' if tf_synced else ''}")
                last_debug_time = current_wait_time
            
            time.sleep(0.01)
        
        if self.sensor_manager.debug_timestamps:
            print(f"Sync timeout after {timeout:.3f}s, proceeding with best available data")
            if rgb_ts and depth_ts and tf_ts:
                print(f"  Final timestamp differences: RGB={rgb_diff:.6f}s, Depth={depth_diff:.6f}s, TF={tf_diff:.6f}s")
                print(f"  Best sync quality achieved: {best_sync_quality:.6f}s")
                print(f"  Sync status: RGB={'SYNCED' if rgb_synced else 'NOT_SYNCED'}, Depth={'SYNCED' if depth_synced else 'NOT_SYNCED'}, TF={'SYNCED' if tf_synced else 'NOT_SYNCED'}")
        
        # Return the best synchronized data found, even if not within threshold
        return False, best_rgb_data, best_depth_data, best_base_pose

    def _ensure_data_available(self):
        """Ensure all required data is available without waiting for sync"""
        rgb_ts = self.sensor_manager.last_rgb_ts
        depth_ts = self.sensor_manager.last_depth_ts
        base_trans, base_quat, tf_ts = self.tf_manager.lookup_pose(self.odom_frame, self.base_frame)
        
        if rgb_ts is None or depth_ts is None or tf_ts is None:
            if self.sensor_manager.debug_timestamps:
                missing = []
                if rgb_ts is None: missing.append("RGB")
                if depth_ts is None: missing.append("depth")
                if tf_ts is None: missing.append("TF")
                print(f"Missing data: {', '.join(missing)}")
            return False
        
        if self.sensor_manager.debug_timestamps:
            timestamps = [rgb_ts, depth_ts, tf_ts]
            max_diff = max(timestamps) - min(timestamps)
            print(f"Using latest data: max_diff={max_diff:.6f}s")
            print(f"  RGB: {rgb_ts:.6f}")
            print(f"  Depth: {depth_ts:.6f}")
            print(f"  TF: {tf_ts:.6f}")
        
        return True

    def get_message_timestamps(self):
        """Get timestamps of the latest ROS messages in both Unix and ROS formats"""
        # Get sensor timestamps
        sensor_timestamps = self.sensor_manager.get_sensor_timestamps()
        
        # Get TF timestamps
        base_trans, base_quat, base_timestamp = self.tf_manager.lookup_pose(self.odom_frame, self.base_frame)
        
        # Combine sensor and TF timestamps
        timestamps = sensor_timestamps.copy()
        timestamps["tf_base"] = {
            "unix_timestamp": base_timestamp,
            "ros_timestamp": format_ros_timestamp(base_timestamp),
            "frame": f"{self.odom_frame} -> {self.base_frame}"
        }
        
        return timestamps

    def print_timestamp_info(self):
        """Print current timestamp information for debugging"""
        timestamps = self.get_message_timestamps()
        print("=== ROS Message & TF Timestamps ===")
        for msg_type, ts_info in timestamps.items():
            if ts_info["unix_timestamp"] is not None:
                dt = datetime.fromtimestamp(ts_info["unix_timestamp"])
                print(f"{msg_type}: {ts_info['unix_timestamp']:.6f} ({dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]})")
                if ts_info["ros_timestamp"]:
                    print(f"  ROS format: secs={ts_info['ros_timestamp']['secs']}, nsecs={ts_info['ros_timestamp']['nsecs']}")
                if "frame" in ts_info:
                    print(f"  Frame: {ts_info['frame']}")
                
                # Show age of timestamp
                current_time = time.time()
                age = current_time - ts_info["unix_timestamp"]
                print(f"  Age: {age:.3f}s")
            else:
                print(f"{msg_type}: No timestamp available")
        print("=============================")

    def build_obs(self):
        """Build observation required by agent"""
        if self.sensor_manager.rgb is None:
            print("Warning: No RGB image available")
            return None

        # Wait for synchronized data if enabled and get the synchronized data
        if self.sensor_manager.sync_timestamps:
            sync_success, synced_rgb, synced_depth, synced_base_pose = self.wait_for_synchronized_data()
            if not sync_success and self.sensor_manager.debug_timestamps:
                print("Warning: Data synchronization failed, proceeding with best available data")
        else:
            synced_rgb = self.sensor_manager.rgb
            synced_depth = self.sensor_manager.depth
            synced_base_pose = self.tf_manager.lookup_pose(self.odom_frame, self.base_frame)
        
        # Use synchronized data if available, fallback to current data
        rgb_data = synced_rgb if synced_rgb is not None else self.sensor_manager.rgb
        depth_data = synced_depth if synced_depth is not None else self.sensor_manager.depth
        
        if rgb_data is None:
            print("Warning: No RGB image available after sync")
            return None
        
        # Extract synchronized base pose
        base_trans, base_quat = synced_base_pose[0], synced_base_pose[1]
        if base_trans is None or base_quat is None:
            print("Warning: No synchronized base pose available")
            return None
        
        # Get static transform from base to camera (no sync needed - it's static)
        base_to_cam_trans, base_to_cam_quat, _ = self.tf_manager.lookup_pose(self.base_frame, self.camera_frame)
        if base_to_cam_trans is None or base_to_cam_quat is None:
            raise RuntimeError(f"Cannot find static TF: {self.base_frame} -> {self.camera_frame}")
        
        # Transform camera pose to odom frame: odom_cam = odom_base * base_cam
        # This combines the synchronized base pose with the static base->camera transform
        cam_trans, cam_quat = self.tf_manager.compose_transforms(
            base_trans, base_quat, base_to_cam_trans, base_to_cam_quat
        )
        
        # Create agent state
        class MockAgentState:
            def __init__(self, base_position, base_rotation, cam_position, cam_rotation):
                self.position = base_position
                self.rotation = base_rotation
                self.sensor_states = {
                    'color_sensor': SixDOF(cam_position, cam_rotation),
                    'depth_sensor': SixDOF(cam_position, cam_rotation)
                }

        agent_state = MockAgentState(base_trans, base_quat, cam_trans, cam_quat)

        obs = {
            "agent_state": agent_state,
            "color_sensor": rgb_data,
            "depth_sensor": depth_data if depth_data is not None else np.zeros((self.sensor_manager.height, self.sensor_manager.width), dtype=np.float32),
            "timestamps": self.get_message_timestamps()
        }

        # Goal handling
        if hasattr(self.sensor_manager, 'goal') and self.sensor_manager.goal is not None:
            if self.sensor_manager.goal.get('mode') == 'description' and 'goal_image' not in self.sensor_manager.goal:
                # Use synchronized image as fallback
                self.sensor_manager.goal['goal_image'] = rgb_data
            obs["goal"] = self.sensor_manager.goal
        else:
            obs["goal"] = {"mode": "image", "name": "object", "goal_image": rgb_data}

        return obs

    def step(self):
        """Execute one step of the navigation loop"""
        try:
            obs = self.build_obs()
            if obs is None:
                # No observation or TF not ready
                return

            # Debug timestamp information if enabled
            if self.sensor_manager.debug_timestamps:
                self.print_timestamp_info()

            # Get action from agent
            agent_action, metadata = self.agent.step(obs)
            
            # Execute control action
            self.control_manager.simple_control(
                agent_action, 
                self.tf_manager, 
                self.odom_frame, 
                self.base_frame, 
                mode="open_loop"
            )
            
        except Exception as e:
            print(f"step error: {e}")
            import traceback
            traceback.print_exc()

    def spin(self):
        """Main execution loop"""
        rate_hz = float(self.agent_cfg.get("rate_hz", 10.0))
        dt = 1.0 / rate_hz if rate_hz > 0 else 0.1
        while True:
            self.step()
            time.sleep(dt)

    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'ros') and self.ros.is_connected:
                self.ros.terminate()
        except Exception as e:
            print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    try:
        node = VLMNavNode()
        node.spin()
    except Exception as e:
        print(f"Failed to start VLMNavNode: {e}")
        exit(1)
