#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import base64
import math
from datetime import datetime

quaternion_imported = False
try:
    import quaternion  # pip install numpy-quaternion
    quaternion_imported = True
except Exception:
    pass

# ---------------- Simple Data Structures ----------------
class SixDOF(object):
    def __init__(self, position, rotation):
        self.position = position
        self.rotation = rotation

class AgentState(object):
    def __init__(self, position, rotation, sensor_states):
        self.position = position  # np.array(3,)
        self.rotation = rotation  # np.quaternion or [x,y,z,w]
        self.sensor_states = sensor_states  # { 'color_sensor': SixDOF(...), ... }

# ---------------- Utility Functions ----------------
def to_quat(x, y, z, w):
    if quaternion_imported is not None:
        return np.quaternion(w, x, y, z)  # Note the order
    return np.array([x, y, z, w], dtype=np.float64)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def format_ros_timestamp(timestamp):
    """Convert timestamp to ROS standard format (secs, nsecs)"""
    if timestamp is None:
        return None
    secs = int(timestamp)
    nsecs = int((timestamp - secs) * 1e9)
    return {"secs": secs, "nsecs": nsecs}

def extract_timestamp_from_header(header):
    """Extract timestamp from ROS message header"""
    if not header or 'stamp' not in header:
        return None
    
    stamp = header['stamp']
    if 'secs' in stamp and 'nsecs' in stamp:
        return stamp['secs'] + stamp['nsecs'] * 1e-9
    elif 'sec' in stamp and 'nanosec' in stamp:
        return stamp['sec'] + stamp['nanosec'] * 1e-9
    
    return None

# ---------------- Image Conversion Tools ----------------
def image_msg_to_cv2(msg):
    """Convert ROS Image message to OpenCV format (rosbridge JSON)"""
    try:
        if hasattr(msg, 'get'):
            data = msg.get('data', None)
            encoding = msg.get('encoding', 'rgb8')
            height = int(msg.get('height', 0))
            width  = int(msg.get('width', 0))
        else:
            return None
        if data is None or height <= 0 or width <= 0:
            return None
        np_arr = np.frombuffer(data, dtype=np.uint8)
        if 'rgb8' in encoding or 'bgr8' in encoding:
            img = np_arr.reshape((height, width, 3))
            if 'rgb8' in encoding:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        elif 'rgba8' in encoding or 'bgra8' in encoding:
            img = np_arr.reshape((height, width, 4))
            if 'rgba8' in encoding:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
        elif 'mono8' in encoding or '8UC1' in encoding:
            img = np_arr.reshape((height, width))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img
        else:
            return None
    except Exception:
        return None

def mat_to_quat(R):
    """Convert rotation matrix to quaternion"""
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    return x, y, z, w

def wrap_to_pi(ang):
    """Wrap angle to [-pi, pi]"""
    return (float(ang) + math.pi) % (2 * math.pi) - math.pi

def as_float(x, default):
    """Convert to float with default fallback"""
    try:
        return float(x)
    except Exception:
        return float(default)

def quat_to_array(quat):
    """
    Convert quaternion to array format [w, x, y, z]
    
    Args:
        quat: quaternion, can be np.quaternion or [x, y, z, w] format array
        
    Returns:
        tuple: (w, x, y, z) quaternion components
    """
    # If it's numpy.quaternion type
    if isinstance(quat, quaternion.quaternion):
        return (quat.w, quat.x, quat.y, quat.z)
    
    # If it's list / np.ndarray
    elif isinstance(quat, (list, tuple, np.ndarray)):
        # Assume input is [x, y, z, w], need to swap order
        x, y, z, w = tuple(quat)
        return (w, x, y, z)
    
    else:
        raise TypeError("quat must be np.quaternion or [x, y, z, w] format")

def quat_to_rotation_matrix(quat):
    """Convert quaternion to rotation matrix"""
    w, x, y, z = quat_to_array(quat)
    
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])

def quat_to_yaw(quat):
    """Convert quaternion to yaw angle"""
    w, x, y, z = quat_to_array(quat)
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

def rotate_quaternion(quat, angle):
    """Rotate quaternion around Z-axis"""
    w1, x1, y1, z1 = quat_to_array(quat)
    
    # Create quaternion for rotation around Z-axis
    half_angle = angle / 2
    w2 = math.cos(half_angle)
    x2 = 0
    y2 = 0
    z2 = math.sin(half_angle)
    
    # Quaternion multiplication
    result = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
    
    return result
