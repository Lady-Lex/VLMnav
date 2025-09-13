#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import base64
import json
import time
from datetime import datetime

from .utils import extract_timestamp_from_header, format_ros_timestamp

try:
    from roslibpy import Topic
except ImportError:
    print("Error: roslibpy not found. Please install with: pip install roslibpy")
    Topic = None

class SensorManager:
    """Manages sensor data subscriptions and time synchronization"""
    
    def __init__(self, ros, agent_cfg):
        self.ros = ros
        self.agent_cfg = agent_cfg
        
        # Topic configuration
        self.rgb_topic = self.agent_cfg.get("rgb_topic", "/camera/color/image_raw/compressed")
        self.depth_topic = self.agent_cfg.get("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.goal_img_topic = self.agent_cfg.get("goal_img_topic", "/goal_image")
        self.goal_meta_topic = self.agent_cfg.get("goal_meta_topic", "/goal_meta")
        
        # Camera intrinsics
        self.width = int(self.agent_cfg.get("sensor_cfg", {}).get("width", 640))
        self.height = int(self.agent_cfg.get("sensor_cfg", {}).get("height", 480))
        
        # Debug options
        self.debug_timestamps = bool(self.agent_cfg.get("debug_timestamps", False))
        
        # Time synchronization options
        self.sync_timestamps = bool(self.agent_cfg.get("sync_timestamps", True))
        self.sync_timeout = float(self.agent_cfg.get("sync_timeout", 0.5))  # seconds
        self.sync_threshold = float(self.agent_cfg.get("sync_threshold", 0.1))  # seconds
        self.sync_strategy = self.agent_cfg.get("sync_strategy", "wait_for_sync")  # "wait_for_sync" or "use_latest"
        
        # Runtime cache
        self.rgb = None
        self.depth = None
        self.last_goal_image = None
        self.last_goal_image_ts = None
        self.last_goal_meta_ts = None
        
        # Timestamp tracking
        self.rgb_sub_counter = 0
        self.last_rgb_ts = 0.0
        self.depth_sub_counter = 0
        self.last_depth_ts = 0.0
        
        # Goal management
        self.goal = {"mode": "image", "name": "object", "goal_image": None}
    
    def subscribe_to_topics(self):
        """Subscribe to all sensor topics"""
        # RGB camera
        self.rgb_sub = Topic(self.ros, self.rgb_topic, 'sensor_msgs/CompressedImage')
        self.rgb_sub.subscribe(self.rgb_cb)
        
        # Depth camera
        self.depth_sub = Topic(self.ros, self.depth_topic, 'sensor_msgs/Image')
        self.depth_sub.subscribe(self.depth_cb)
        
        # Goal image
        self.goal_img_sub = Topic(self.ros, self.goal_img_topic, 'sensor_msgs/CompressedImage')
        self.goal_img_sub.subscribe(self.goal_img_cb)
        
        # Goal metadata
        self.goal_meta_sub = Topic(self.ros, self.goal_meta_topic, 'std_msgs/String')
        self.goal_meta_sub.subscribe(self.goal_meta_cb)
    
    def rgb_cb(self, msg):
        """RGB camera callback"""
        try:
            data = msg.get('data', None)
            if data is None:
                return
            
            # Extract timestamp from ROS message header
            msg_timestamp = extract_timestamp_from_header(msg.get('header', {}))
            
            # Fallback to current time if no header timestamp
            if msg_timestamp is None:
                msg_timestamp = datetime.now().timestamp()
                if self.debug_timestamps:
                    print("Warning: No timestamp in RGB message, using current time")
            
            np_arr = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
            rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.rgb = rgb
            self.rgb_sub_counter += 1
            self.last_rgb_ts = msg_timestamp
            
        except Exception as e:
            print(f"Bad RGB: {e}")

    def depth_cb(self, msg):
        """Depth camera callback"""
        try:
            # Extract timestamp from ROS message header
            msg_timestamp = extract_timestamp_from_header(msg.get('header', {}))
            
            # Fallback to current time if no header timestamp
            if msg_timestamp is None:
                msg_timestamp = datetime.now().timestamp()
                if self.debug_timestamps:
                    print("Warning: No timestamp in depth message, using current time")
            
            encoding = msg.get('encoding', '32FC1')
            img_data = msg.get('data', b'')
            if isinstance(img_data, str):
                try:
                    img_data = base64.b64decode(img_data)
                except Exception:
                    img_data = img_data.encode('latin-1', errors='ignore')

            height_actual = int(msg.get('height', self.height))
            width_actual = int(msg.get('width', self.width))

            if '32FC1' in encoding or '32FC' in encoding:
                np_arr = np.frombuffer(img_data, dtype=np.float32)
                img = np_arr.reshape((height_actual, width_actual))
                img = np.nan_to_num(img)
                self.depth = img
            elif '16UC1' in encoding:
                np_arr = np.frombuffer(img_data, dtype=np.uint16)
                img = np_arr.reshape((height_actual, width_actual))
                img = img.astype(np.float32) / 1000.0
                img = np.nan_to_num(img)
                self.depth = img
            elif '8UC1' in encoding:
                np_arr = np.frombuffer(img_data, dtype=np.uint8)
                img = np_arr.reshape((height_actual, width_actual)).astype(np.float32)
                self.depth = img
            else:
                np_arr = np.frombuffer(img_data, dtype=np.uint8)
                data_size = np_arr.size
                total_pixels = height_actual * width_actual
                if total_pixels <= 0:
                    print("Invalid image shape in depth_cb; using zeros")
                    self.depth = np.zeros((self.height, self.width), dtype=np.float32)
                    return
                if data_size % total_pixels == 0:
                    channels = data_size // total_pixels
                    if channels == 1:
                        img = np_arr.reshape((height_actual, width_actual)).astype(np.float32)
                        self.depth = img
                    elif channels == 3:
                        img = np_arr.reshape((height_actual, width_actual, 3))
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
                        self.depth = img
                    elif channels == 4:
                        img = np_arr.reshape((height_actual, width_actual, 4))
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY).astype(np.float32)
                        self.depth = img
                    else:
                        print(f"Unexpected channels: {channels}, fallback as single channel")
                        img = np_arr.reshape((height_actual, width_actual)).astype(np.float32)
                        self.depth = img
                else:
                    print("Cannot determine channel count; fallback reshape")
                    try:
                        img = np_arr.reshape((height_actual, width_actual)).astype(np.float32)
                        self.depth = img
                    except ValueError:
                        print("Fallback reshape failed; using zeros")
                        self.depth = np.zeros((height_actual, width_actual), dtype=np.float32)
            
            self.depth_sub_counter += 1
            self.last_depth_ts = msg_timestamp
            
        except Exception as e:
            print(f"Bad depth: {e}")

    def goal_img_cb(self, msg):
        """Goal image callback"""
        try:
            # Extract timestamp from ROS message header
            msg_timestamp = extract_timestamp_from_header(msg.get('header', {}))
            
            # Fallback to current time if no header timestamp
            if msg_timestamp is None:
                msg_timestamp = datetime.now().timestamp()
                if self.debug_timestamps:
                    print("Warning: No timestamp in goal image message, using current time")
            
            data = msg.get('data', None)
            if data is None:
                return
            np_arr = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
            rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.last_goal_image = rgb
            self.last_goal_image_ts = msg_timestamp
        except Exception as e:
            print(f"Bad goal image: {e}")

    def goal_meta_cb(self, msg):
        """Goal metadata callback"""
        try:
            # Extract timestamp from ROS message header
            msg_timestamp = extract_timestamp_from_header(msg.get('header', {}))
            
            # Fallback to current time if no header timestamp
            if msg_timestamp is None:
                msg_timestamp = datetime.now().timestamp()
                if self.debug_timestamps:
                    print("Warning: No timestamp in goal meta message, using current time")
            
            meta = json.loads(msg.get('data', '{}'))
            if meta.get("mode") == "image" and self.last_goal_image is not None:
                self.goal = {"mode": "image", "name": meta.get("name", "object"), "goal_image": self.last_goal_image}
                self.last_goal_meta_ts = msg_timestamp
                print(f"New image goal set: {self.goal.get('name')}")
            elif meta.get("mode") == "description":
                # Allow description goals; use current RGB as fallback in build_obs if no image
                self.goal = {"mode": "description", "name": meta.get("name", "object")}
                self.last_goal_meta_ts = msg_timestamp
                print(f"New description goal set: {self.goal.get('name')}")
        except Exception as e:
            print(f"Bad goal meta JSON: {e}")

    def get_current_data(self):
        """Get current sensor data with timestamps"""
        return {
            'rgb': self.rgb,
            'depth': self.depth,
            'rgb_ts': self.last_rgb_ts,
            'depth_ts': self.last_depth_ts
        }
    
    def has_data(self):
        """Check if basic sensor data is available"""
        return self.rgb is not None

    def get_sensor_timestamps(self):
        """Get timestamps of sensor messages only"""
        return {
            "rgb": {
                "unix_timestamp": self.last_rgb_ts,
                "ros_timestamp": format_ros_timestamp(self.last_rgb_ts)
            },
            "depth": {
                "unix_timestamp": self.last_depth_ts,
                "ros_timestamp": format_ros_timestamp(self.last_depth_ts)
            }
        }
