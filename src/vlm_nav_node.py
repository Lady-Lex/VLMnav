#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import json
import time
import threading
import numpy as np
import cv2
import base64
import math

try:
    import quaternion  # pip install numpy-quaternion
except Exception:
    quaternion = None

try:
    import roslibpy
    from roslibpy import Ros, Topic, Service
except ImportError:
    print("Error: roslibpy not found. Please install with: pip install roslibpy")
    exit(1)

from agent import GOATAgent

# ---------------- PolarAction 兜底 ----------------
try:
    from simWrapper import PolarAction
    print("Imported PolarAction from simWrapper")
except Exception:
    print("Warning: simWrapper.PolarAction not found, defining locally")
    from collections import namedtuple
    PolarAction = namedtuple("PolarAction", ["v", "w", "T"])  # 速度极坐标兜底

# ---------------- 简易数据结构 ----------------
class SixDOF(object):
    def __init__(self, position, rotation):
        self.position = position
        self.rotation = rotation

class AgentState(object):
    def __init__(self, position, rotation, sensor_states):
        self.position = position  # np.array(3,)
        self.rotation = rotation  # np.quaternion 或 [x,y,z,w]
        self.sensor_states = sensor_states  # { 'color_sensor': SixDOF(...), ... }

# ---------------- 小工具 ----------------
def to_quat(x, y, z, w):
    if quaternion is not None:
        return np.quaternion(w, x, y, z)  # 注意顺序
    return np.array([x, y, z, w], dtype=np.float64)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ---------------- 图像转换工具 ----------------
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

# ---------------- 主节点 ----------------
class VLMNavNode(object):
    def __init__(self):
        # ROS 连接
        self.ros_host = 'localhost'
        self.ros_port = 9090
        self.ros = Ros(self.ros_host, self.ros_port)
        self.ros.run()

        # 等待连接
        while not self.ros.is_connected:
            time.sleep(0.05)
        print(f"Connected to ROS at {self.ros_host}:{self.ros_port}")

        # 载入配置
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'agent_cfg.yaml')
        self.agent_cfg = self._load_agent_cfg(cfg_path)

        # 话题配置
        self.rgb_topic = self.agent_cfg.get("rgb_topic", "/camera/color/image_raw/compressed")
        self.depth_topic = self.agent_cfg.get("depth_topic", "/camera/depth/image_raw")
        self.goal_img_topic = self.agent_cfg.get("goal_img_topic", "/goal_image")
        self.goal_meta_topic = self.agent_cfg.get("goal_meta_topic", "/goal_meta")

        # 相机内参（可由话题覆盖）
        self.width = int(self.agent_cfg.get("sensor_cfg", {}).get("width", 640))
        self.height = int(self.agent_cfg.get("sensor_cfg", {}).get("height", 480))
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.fov_deg = float(self.agent_cfg.get("sensor_cfg", {}).get("fov", 60.0))
        self.res_factor = float(self.agent_cfg.get("sensor_cfg", {}).get("res_factor", 1.0))
        self.camera_frame = None
        # 帧名可被配置覆盖
        self.base_frame = self.agent_cfg.get("base_frame", "jackal/base_link")
        self.odom_frame = self.agent_cfg.get("odom_frame", "jackal/odom")

        # 控制参数
        self.v_max = float(self.agent_cfg.get("v_max", 2.0))
        self.w_max = float(self.agent_cfg.get("w_max", 1.0))
        self.T = float(self.agent_cfg.get("default_T", 0.6))
        self.two_stage = bool(self.agent_cfg.get("two_stage", True))

        # 运行缓存
        self.rgb = None
        self.depth = None
        self.last_goal_image = None

        # TF 图
        self.tf_graph_dynamic = {}
        self.tf_graph_static = {}

        self.tf_sub = Topic(self.ros, '/tf', 'tf2_msgs/TFMessage')
        self.tf_sub.subscribe(lambda msg: self.tf_callback(msg, is_static=False))
        self.tf_static_sub = Topic(self.ros, '/tf_static', 'tf2_msgs/TFMessage')
        self.tf_static_sub.subscribe(lambda msg: self.tf_callback(msg, is_static=True))

        # 订阅
        self.rgb_sub = Topic(self.ros, self.rgb_topic, 'sensor_msgs/CompressedImage')
        self.rgb_sub.subscribe(self.rgb_cb)
        self.depth_sub = Topic(self.ros, self.depth_topic, 'sensor_msgs/Image')
        self.depth_sub.subscribe(self.depth_cb)
        self.goal_img_sub = Topic(self.ros, self.goal_img_topic, 'sensor_msgs/CompressedImage')
        self.goal_img_sub.subscribe(self.goal_img_cb)
        self.goal_meta_sub = Topic(self.ros, self.goal_meta_topic, 'std_msgs/String')
        self.goal_meta_sub.subscribe(self.goal_meta_cb)

        # 发布
        self.cmd_pub = Topic(self.ros, '/cmd_vel', 'geometry_msgs/Twist')

        # Agent
        self.agent = GOATAgent(self.agent_cfg)
        self.agent.reset()

        # PolarAction.default 设为数值 T，避免后续参与计算时报错
        try:
            if not hasattr(PolarAction, 'default') or PolarAction.default is None:
                PolarAction.default = PolarAction(0.0, 0.0, 0.5)
        except Exception:
            pass

        # 从配置初始化 goal
        self.goal = {"mode": "image", "name": "object", "goal_image": None}
        try:
            with open(cfg_path, 'r') as f:
                yaml_cfg = yaml.safe_load(f)
            goal_cfg = yaml_cfg.get("goal", {})
            if goal_cfg:
                self.goal = goal_cfg
                print(f"Initial goal set from config: {self.goal}")
        except Exception as e:
            print(f"Failed to load goal from config: {e}")

        # 清理线程
        self.max_tf_age = float(self.agent_cfg.get("tf_max_age", 10.0))
        self.cleanup_thread = threading.Thread(target=self._cleaner_loop, daemon=True)
        self.cleanup_thread.start()

        # 在进入主循环前，等待 TF 可用
        self._wait_for_tf_ready(timeout_sec=float(self.agent_cfg.get("tf_wait_timeout", 5.0)))

    # ---------- 配置 ----------
    def _load_agent_cfg(self, cfg_path):
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
            # 应用默认值
            for k, v in defaults.items():
                if k not in agent_cfg:
                    agent_cfg[k] = v
            print(f"Loaded configuration from {cfg_path}")
            return agent_cfg
        except Exception as e:
            print(f"Failed to load configuration from {cfg_path}: {e}")
            print("Using default configuration")
            return defaults

    # ---------- 后台清理 ----------
    def _cleaner_loop(self):
        while True:
            try:
                self.cleanup_old_transforms(self.max_tf_age)
            except Exception:
                pass
            time.sleep(1.0)

    # ---------- 传感器回调 ----------
    def rgb_cb(self, msg):
        try:
            data = msg.get('data', None)
            if data is None:
                return
            np_arr = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
            rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.rgb = rgb
            if self.camera_frame is None:
                fid = msg.get('header', {}).get('frame_id', '')
                if fid:
                    self.camera_frame = fid.lstrip('/')
        except Exception as e:
            print(f"Bad RGB: {e}")

    def depth_cb(self, msg):
        try:
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
        except Exception as e:
            print(f"Bad depth: {e}")

    def goal_img_cb(self, msg):
        try:
            data = msg.get('data', None)
            if data is None:
                return
            np_arr = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
            rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.last_goal_image = rgb
        except Exception as e:
            print(f"Bad goal image: {e}")

    def goal_meta_cb(self, msg):
        try:
            meta = json.loads(msg.get('data', '{}'))
            if meta.get("mode") == "image" and self.last_goal_image is not None:
                self.goal = {"mode": "image", "name": meta.get("name", "object"), "goal_image": self.last_goal_image}
                print(f"New image goal set: {self.goal.get('name')}")
                if hasattr(self, 'agent') and hasattr(self.agent, "reset_goal"):
                    self.agent.reset_goal()
            elif meta.get("mode") == "description":
                # 允许描述目标；若无图像则在 build_obs 时用当前 RGB 兜底
                self.goal = {"mode": "description", "name": meta.get("name", "object")}
                print(f"New description goal set: {self.goal.get('name')}")
        except Exception as e:
            print(f"Bad goal meta JSON: {e}")

    # ---------- TF / 位姿 ----------
    def _make_T(self, trans_dict, rot_dict):
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
        R = T[:3,:3]
        t = T[:3, 3]
        Tinv = np.eye(4, dtype=np.float64)
        Tinv[:3,:3] = R.T
        Tinv[:3, 3] = -R.T @ t
        return Tinv

    def _matmul_T(self, A, B):
        return A @ B

    def _store_edge(self, graph, parent, child, T, timestamp):
        graph.setdefault(parent, {})[child] = {'T': T, 'timestamp': timestamp}
        graph.setdefault(child, {})[parent] = {'T': self._invert_T(T), 'timestamp': timestamp}

    def tf_callback(self, msg, is_static=False):
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
                tr = tfs.get('transform', {}).get('translation', {})
                rq = tfs.get('transform', {}).get('rotation', {})
                T = self._make_T(tr, rq)
                now_ts = time.time()
                self._store_edge(graph, frame_id, child_id, T, now_ts)
        except Exception as e:
            print(f"TF callback error: {e}")

    def _merged_graph(self):
        merged = {}
        for a, nbrs in self.tf_graph_static.items():
            merged[a] = {b: info.copy() for b, info in nbrs.items()}
        for a, nbrs in self.tf_graph_dynamic.items():
            merged.setdefault(a, {})
            for b, info in nbrs.items():
                merged[a][b] = info
        return merged

    def _find_T_path(self, target, source):
        if target == source:
            return np.eye(4, dtype=np.float64)
        graph = self._merged_graph()
        if target not in graph:
            return None
        from collections import deque
        q = deque([target])
        acc_T = {target: np.eye(4, dtype=np.float64)}
        visited = set()
        while q:
            cur = q.popleft()
            if cur == source:
                return acc_T[cur]
            visited.add(cur)
            for nxt, info in graph.get(cur, {}).items():
                if nxt in visited:
                    continue
                T_cur_nxt = info['T']
                T_acc = self._matmul_T(acc_T[cur], T_cur_nxt)
                if nxt not in acc_T:
                    acc_T[nxt] = T_acc
                    q.append(nxt)
        return None

    def _mat_to_quat(self, R):
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

    def lookup_pose(self, target_frame, source_frame):
        """
        返回: (trans[3], quat[x,y,z,w])，语义为“source 在 target 下的位姿”
        """
        try:
            target = str(target_frame).lstrip('/')
            source = str(source_frame).lstrip('/')
            T = self._find_T_path(target, source)
            if T is None:
                print(f"[TF] 路径不存在或尚未收到: {target} -> ... -> {source}")
                return None, None
            R = T[:3,:3]; t = T[:3,3]
            qx, qy, qz, qw = self._mat_to_quat(R)
            return t.astype(np.float32), to_quat(qx, qy, qz, qw)
        except Exception as e:
            print(f"TF lookup failed: {e}")
            return None, None

    def cleanup_old_transforms(self, max_age=5.0):
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

    def print_tf_buffer_status(self):
        mg = self._merged_graph()
        edges = sum(len(nbrs) for nbrs in mg.values())
        print(f"TF graph (merged) nodes={len(mg)} edges={edges}")
        for parent, nbrs in mg.items():
            for child in nbrs.keys():
                print(f"  {parent} -> {child}")

    def _wait_for_tf_ready(self, timeout_sec=5.0):
        """进入主循环前等待 odom->base_link 可查询，避免一开机就报路径不存在。"""
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            trans, quat = self.lookup_pose(self.odom_frame, self.base_frame)
            if trans is not None and quat is not None:
                print("TF ready: odom -> base_link")
                return
            time.sleep(0.1)
        print("Warning: TF not ready after timeout; will continue and retry in loop.")

    # ---------- 观测/控制 ----------
    def _wrap_to_pi(self, ang):
        return (float(ang) + math.pi) % (2 * math.pi) - math.pi

    def _as_float(self, x, default):
        try:
            return float(x)
        except Exception:
            return float(default)

    def _extract_cmd_from_action(self, agent_action):
        """
        兼容三类返回：
          1) dict: {v,w,T} 或 {r,theta,T}
          2) 极坐标: 属性 r, theta[, T]
          3) 速度坐标: 属性 v, w[, T]
        返回: (v, w, T)
        """
        # 1) 字典
        if isinstance(agent_action, dict):
            if "v" in agent_action or "w" in agent_action:
                v = self._as_float(agent_action.get("v", 0.0), 0.0)
                w = self._as_float(agent_action.get("w", 0.0), 0.0)
                T = self._as_float(agent_action.get("T", self.T), self.T)
                return v, w, max(1e-3, T)
            if "r" in agent_action or "theta" in agent_action:
                r = self._as_float(agent_action.get("r", 0.0), 0.0)
                theta = self._wrap_to_pi(self._as_float(agent_action.get("theta", 0.0), 0.0))
                T = self._as_float(agent_action.get("T", self.T), self.T)
                T = max(1e-3, T)
                return r / T, theta / T, T

        # 2) 极坐标对象
        r = getattr(agent_action, "r", None)
        theta = getattr(agent_action, "theta", None)
        if r is not None or theta is not None:
            r = self._as_float(r if r is not None else 0.0, 0.0)
            theta = self._wrap_to_pi(self._as_float(theta if theta is not None else 0.0, 0.0))
            T = self._as_float(getattr(agent_action, "T", self.T), self.T)
            T = max(1e-3, T)
            return r / T, theta / T, T

        # 3) 速度坐标对象
        v = getattr(agent_action, "v", None)
        w = getattr(agent_action, "w", None)
        if v is not None or w is not None:
            v = self._as_float(v if v is not None else 0.0, 0.0)
            w = self._as_float(w if w is not None else 0.0, 0.0)
            T = self._as_float(getattr(agent_action, "T", self.T), self.T)
            return v, w, max(1e-3, T)

        # 全部失败
        return 0.0, 0.0, self.T

    def build_obs(self):
        """构建 agent 需要的观测"""
        if self.rgb is None:
            print("Warning: No RGB image available")
            return None

        trans, quat = self.lookup_pose(self.odom_frame, self.base_frame)
        if trans is None or quat is None:
            # TF 仍不可用；暂不前进
            return None

        class MockAgentState:
            def __init__(self, position, rotation):
                self.position = position
                self.rotation = rotation
                self.sensor_states = {
                    'color_sensor': SixDOF(position, rotation),
                    'depth_sensor': SixDOF(position, rotation)
                }

        agent_state = MockAgentState(trans, quat)

        obs = {
            "agent_state": agent_state,
            "color_sensor": self.rgb,
            "depth_sensor": self.depth if self.depth is not None else np.zeros((self.height, self.width), dtype=np.float32)
        }

        # 目标
        if hasattr(self, 'goal') and self.goal is not None:
            if self.goal.get('mode') == 'description' and 'goal_image' not in self.goal:
                # 用当前画面兜底
                self.goal['goal_image'] = self.rgb
            obs["goal"] = self.goal
        else:
            obs["goal"] = {"mode": "image", "name": "object", "goal_image": self.rgb}

        return obs

    def step(self):
        try:
            obs = self.build_obs()
            if obs is None:
                # 没有观测或 TF 未就绪
                return

            agent_action, metadata = self.agent.step(obs)

            v, w, T = self._extract_cmd_from_action(agent_action)

            # 小死区 + 限幅
            if abs(v) < 1e-3: v = 0.0
            if abs(w) < 1e-3: w = 0.0
            v = clamp(v, -self.v_max, self.v_max)
            w = clamp(w, -self.w_max, self.w_max)

            twist = {
                'linear':  {'x': float(v), 'y': 0.0, 'z': 0.0},
                'angular': {'x': 0.0, 'y': 0.0, 'z': float(w)}
            }
            self.cmd_pub.publish(twist)
        except Exception as e:
            print(f"step error: {e}")
            import traceback
            traceback.print_exc()

    def spin(self):
        rate_hz = float(self.agent_cfg.get("rate_hz", 10.0))
        dt = 1.0 / rate_hz if rate_hz > 0 else 0.1
        while True:
            self.step()
            time.sleep(dt)

    def cleanup(self):
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
