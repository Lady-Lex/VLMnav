#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import threading
from datetime import datetime
from collections import deque

from .utils import extract_timestamp_from_header, to_quat, mat_to_quat

try:
    from roslibpy import Topic
except ImportError:
    print("Error: roslibpy not found. Please install with: pip install roslibpy")
    Topic = None

class TFManager:
    """Manages TF (Transform) data and provides pose lookup functionality"""
    
    def __init__(self, debug_timestamps=False, max_tf_age=10.0):
        self.debug_timestamps = debug_timestamps
        self.max_tf_age = max_tf_age
        
        # TF graph storage
        self.tf_graph_dynamic = {}
        self.tf_graph_static = {}
        
        # Cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleaner_loop, daemon=True)
        self.cleanup_thread.start()
    
    def subscribe_to_tf_topics(self, ros):
        """Subscribe to TF topics"""
        self.tf_sub = Topic(ros, '/tf', 'tf2_msgs/TFMessage')
        self.tf_sub.subscribe(lambda msg: self.tf_callback(msg, is_static=False))
        self.tf_static_sub = Topic(ros, '/tf_static', 'tf2_msgs/TFMessage')
        self.tf_static_sub.subscribe(lambda msg: self.tf_callback(msg, is_static=True))
    
    def _make_T(self, trans_dict, rot_dict):
        """Create transformation matrix from translation and rotation dictionaries"""
        tx, ty, tz = float(trans_dict.get('x', 0.0)), float(trans_dict.get('y', 0.0)), float(trans_dict.get('z', 0.0))
        x, y, z, w = float(rot_dict.get('x', 0.0)), float(rot_dict.get('y', 0.0)), float(rot_dict.get('z', 0.0)), float(rot_dict.get('w', 1.0))
        n = (x*x + y*y + z*z + w*w) ** 0.5
        if n == 0.0:
            x=y=z=0.0; w=1.0
        else:
            x/=n; y/=n; z/=n; w/=n
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
            [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w),   1-2*(x*x+y*y)]
        ], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3,:3] = R
        T[:3, 3] = [tx, ty, tz]
        return T

    def _invert_T(self, T):
        """Invert transformation matrix"""
        R = T[:3,:3]
        t = T[:3, 3]
        Tinv = np.eye(4, dtype=np.float64)
        Tinv[:3,:3] = R.T
        Tinv[:3, 3] = -R.T @ t
        return Tinv

    def _matmul_T(self, A, B):
        """Multiply transformation matrices"""
        return A @ B

    def _store_edge(self, graph, parent, child, T, timestamp):
        """Store transform edge in graph"""
        graph.setdefault(parent, {})[child] = {'T': T, 'timestamp': timestamp}
        graph.setdefault(child, {})[parent] = {'T': self._invert_T(T), 'timestamp': timestamp}

    def tf_callback(self, msg, is_static=False):
        """Handle incoming TF messages"""
        try:
            transforms = msg.get('transforms', [])
            if not transforms:
                return
            graph = self.tf_graph_static if is_static else self.tf_graph_dynamic
            for tfs in transforms:
                frame_id = tfs.get('header', {}).get('frame_id', '').lstrip('/')
                child_id = tfs.get('child_frame_id', '').lstrip('/')
                if not frame_id or not child_id:
                    continue
                
                # Extract timestamp from TF message header
                tf_timestamp = extract_timestamp_from_header(tfs.get('header', {}))
                
                # Fallback to current time if no header timestamp
                if tf_timestamp is None:
                    tf_timestamp = time.time()
                    if self.debug_timestamps:
                        print(f"Warning: No timestamp in TF message {frame_id}->{child_id}, using current time")
                
                tr = tfs.get('transform', {}).get('translation', {})
                rq = tfs.get('transform', {}).get('rotation', {})
                T = self._make_T(tr, rq)
                self._store_edge(graph, frame_id, child_id, T, tf_timestamp)
        except Exception as e:
            print(f"TF callback error: {e}")

    def _merged_graph(self):
        """Get merged graph of static and dynamic transforms"""
        merged = {}
        for a, nbrs in self.tf_graph_static.items():
            merged[a] = {b: info.copy() for b, info in nbrs.items()}
        for a, nbrs in self.tf_graph_dynamic.items():
            merged.setdefault(a, {})
            for b, info in nbrs.items():
                merged[a][b] = info
        return merged

    def _find_T_path(self, target, source):
        """Find transform path from source to target frame"""
        if target == source:
            return np.eye(4, dtype=np.float64), time.time()
        graph = self._merged_graph()
        if target not in graph:
            return None, None
        q = deque([target])
        acc_T = {target: np.eye(4, dtype=np.float64)}
        acc_timestamp = {target: time.time()}  # Use current time for identity transform
        visited = set()
        while q:
            cur = q.popleft()
            if cur == source:
                return acc_T[cur], acc_timestamp[cur]
            visited.add(cur)
            for nxt, info in graph.get(cur, {}).items():
                if nxt in visited:
                    continue
                T_cur_nxt = info['T']
                T_acc = self._matmul_T(acc_T[cur], T_cur_nxt)
                # Use the timestamp of the oldest transform in the chain (most conservative)
                # This represents the time when the entire transform chain was valid
                timestamp_cur_nxt = info.get('timestamp', time.time())
                timestamp_acc = min(acc_timestamp[cur], timestamp_cur_nxt)
                if nxt not in acc_T:
                    acc_T[nxt] = T_acc
                    acc_timestamp[nxt] = timestamp_acc
                    q.append(nxt)
        return None, None

    def lookup_pose(self, target_frame, source_frame):
        """
        Returns: (trans[3], quat[x,y,z,w], timestamp), meaning "pose of source in target frame"
        timestamp is the most recent timestamp from the transform chain
        """
        try:
            target = str(target_frame).lstrip('/')
            source = str(source_frame).lstrip('/')
            T, timestamp = self._find_T_path(target, source)
            if T is None:
                if self.debug_timestamps:
                    print(f"[TF] Path does not exist or not yet received: {target} -> ... -> {source}")
                return None, None, None
            R = T[:3,:3]; t = T[:3,3]
            qx, qy, qz, qw = mat_to_quat(R)
            return t.astype(np.float32), to_quat(qx, qy, qz, qw), timestamp
        except Exception as e:
            if self.debug_timestamps:
                print(f"TF lookup failed: {e}")
            return None, None, None

    def cleanup_old_transforms(self, max_age=5.0):
        """Remove old transforms from dynamic graph"""
        now_t = time.time()
        to_del = []
        for a, nbrs in self.tf_graph_dynamic.items():
            for b, info in list(nbrs.items()):
                if now_t - info.get('timestamp', now_t) > max_age:
                    to_del.append((a, b))
        for a, b in to_del:
            try:
                del self.tf_graph_dynamic[a][b]
                if not self.tf_graph_dynamic[a]:
                    del self.tf_graph_dynamic[a]
            except KeyError:
                pass

    def _cleaner_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                self.cleanup_old_transforms(self.max_tf_age)
            except Exception:
                pass
            time.sleep(1.0)

    def print_tf_buffer_status(self):
        """Print current TF buffer status for debugging"""
        mg = self._merged_graph()
        edges = sum(len(nbrs) for nbrs in mg.values())
        print(f"TF graph (merged) nodes={len(mg)} edges={edges}")
        for parent, nbrs in mg.items():
            for child, info in nbrs.items():
                timestamp = info.get('timestamp', None)
                if timestamp:
                    dt = datetime.fromtimestamp(timestamp)
                    age = time.time() - timestamp
                    print(f"  {parent} -> {child}: {timestamp:.6f} ({dt.strftime('%H:%M:%S.%f')[:-3]}, age: {age:.3f}s)")
                else:
                    print(f"  {parent} -> {child}: no timestamp")

    def compose_transforms(self, trans1, quat1, trans2, quat2):
        """
        Compose two transforms: result = transform1 * transform2
        Args:
            trans1, quat1: First transform (translation and quaternion)
            trans2, quat2: Second transform (translation and quaternion)
        Returns:
            (result_trans, result_quat): Composed transform
        """
        # Convert quaternions to rotation matrices
        def quat_to_matrix(quat):
            # Handle both np.quaternion objects and [x,y,z,w] arrays
            if hasattr(quat, 'x'):  # np.quaternion object
                x, y, z, w = quat.x, quat.y, quat.z, quat.w
            else:  # [x,y,z,w] array
                x, y, z, w = quat
            n = (x*x + y*y + z*z + w*w) ** 0.5
            if n == 0.0:
                x=y=z=0.0; w=1.0
            else:
                x/=n; y/=n; z/=n; w/=n
            R = np.array([
                [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
                [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w),   1-2*(x*x+y*y)]
            ], dtype=np.float64)
            return R
        
        # Create transformation matrices
        T1 = np.eye(4, dtype=np.float64)
        T1[:3,:3] = quat_to_matrix(quat1)
        T1[:3, 3] = trans1
        
        T2 = np.eye(4, dtype=np.float64)
        T2[:3,:3] = quat_to_matrix(quat2)
        T2[:3, 3] = trans2
        
        # Compose transforms: T_result = T1 * T2
        T_result = T1 @ T2
        
        # Extract result translation and rotation
        result_trans = T_result[:3, 3].astype(np.float32)
        result_quat = to_quat(*mat_to_quat(T_result[:3,:3]))
        
        return result_trans, result_quat

    def wait_for_tf_ready(self, odom_frame, base_frame, timeout_sec=5.0):
        """Wait for odom->base_link to be queryable before entering main loop"""
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            base_trans, base_quat, base_timestamp = self.lookup_pose(odom_frame, base_frame)
            if base_trans is not None and base_quat is not None:
                print("TF ready: odom -> base_link")
                return True
            time.sleep(0.1)
        print("Warning: TF not ready after timeout; will continue and retry in loop.")
        return False
