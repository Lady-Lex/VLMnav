#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
import time

from .utils import clamp, wrap_to_pi, as_float, quat_to_array, quat_to_rotation_matrix, quat_to_yaw

try:
    from roslibpy import Topic
except ImportError:
    print("Error: roslibpy not found. Please install with: pip install roslibpy")
    Topic = None

class ControlManager:
    """Manages robot motion control and action execution"""
    
    def __init__(self, ros, agent_cfg):
        self.ros = ros
        
        # Control parameters
        self.v_max = float(agent_cfg.get("v_max", 1.0))
        self.w_max = float(agent_cfg.get("w_max", 1.0))
        self.T = float(agent_cfg.get("default_T", 1.0))
        self.two_stage = bool(agent_cfg.get("two_stage", True))
        
        # Publishing
        self.cmd_pub = Topic(ros, '/cmd_vel', 'geometry_msgs/Twist')
    
    def extract_cmd_from_action(self, agent_action):
        """
        Convert PolarAction to velocity command (v, w)
        
        Args:
            agent_action: PolarAction object containing r(distance) and theta(angle)
            
        Returns:
            tuple: (v, w) linear and angular velocity
        """
        # Extract polar coordinate parameters
        r = getattr(agent_action, 'r', 0.0)
        theta = getattr(agent_action, 'theta', 0.0)
        
        # Ensure parameters are floats
        r = as_float(r, 0.0)
        theta = wrap_to_pi(as_float(theta, 0.0))
        
        # Convert polar coordinates to velocity command
        # r: distance(m), theta: angle(rad), T: time(s)
        T = self.T  # Use default time
        
        # Calculate velocity
        v = r / T if T > 0 else 0.0  # Linear velocity = distance/time
        w = theta * 2 / T if T > 0 else 0.0  # Angular velocity = angle/time
        
        # Limit velocity range
        if abs(v) < 1e-3: v = 0.0
        if abs(w) < 1e-3: w = 0.0
        v = clamp(v, -self.v_max, self.v_max)
        w = clamp(w, -self.w_max, self.w_max)
        
        return v, w

    def simple_control(self, agent_action, tf_manager, odom_frame, base_frame, mode="open_loop"):
        """
        Two-stage pose closed-loop control: first move to target position, then rotate to specified direction
        
        Args:
            agent_action: PolarAction object containing r(distance) and theta(angle)
            tf_manager: TFManager instance for pose lookup
            odom_frame: odometry frame name
            base_frame: base frame name
            mode: control mode ("open_loop" or "closed_loop")
        """
        
        # Extract polar coordinate parameters from agent_action
        r = getattr(agent_action, 'r', 0.0)
        theta = getattr(agent_action, 'theta', 0.0)
        
        # Ensure parameters are floats
        r = as_float(r, 0.0)
        theta = wrap_to_pi(as_float(theta, 0.0))
        
        # Set default execution time
        duration = self.T
        
        # 1. Get starting pose (start_static_frame)
        start_trans, start_quat, start_timestamp = tf_manager.lookup_pose(odom_frame, base_frame)
        if mode == "open_loop" or (start_trans is None or start_quat is None):
            print("Using open-loop control.")
            v, w = self.extract_cmd_from_action(agent_action)
            self._open_loop_control(v, w, duration)
            return
        
        print("Using closed-loop control.")
        raise NotImplementedError("Closed-loop control is not completely implemented.")
        
        # The rest of the closed-loop implementation would go here...
        # (keeping the original implementation for future use)
    
    def _open_loop_control(self, v, w, duration):
        """Open-loop control fallback"""
        start_time = time.time()
        publish_rate = 20
        sleep_time = 1.0 / publish_rate
        
        print(f"Open-loop fallback: v={v:.3f}, w={w:.3f} for {duration:.1f}s")
        
        while time.time() - start_time < duration:
            twist = {
                'linear':  {'x': float(v), 'y': 0.0, 'z': 0.0},
                'angular': {'x': 0.0, 'y': 0.0, 'z': float(w)}
            }
            self.cmd_pub.publish(twist)
            time.sleep(sleep_time)
        
        stop_twist = {
            'linear':  {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
        }
        self.cmd_pub.publish(stop_twist)
    
    def stop_robot(self):
        """Stop the robot immediately"""
        stop_twist = {
            'linear':  {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
        }
        self.cmd_pub.publish(stop_twist)
    
    def publish_velocity(self, v, w):
        """Publish velocity command directly"""
        twist = {
            'linear':  {'x': float(v), 'y': 0.0, 'z': 0.0},
            'angular': {'x': 0.0, 'y': 0.0, 'z': float(w)}
        }
        self.cmd_pub.publish(twist)
