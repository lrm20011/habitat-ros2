#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2020-2021 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020-2021 Sotiris Papatheodorou
# SPDX-License-Identifier: BSD-3-Clause

import copy
import math
import os
import threading
import quaternion

import habitat_sim as hs
import numpy as np
import cv2
import time

from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
import rclpy
from rclpy.node import Node
import rclpy.time
import rclpy.duration
import tf2_ros
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped, Transform, TransformStamped, Twist
from sensor_msgs.msg import CameraInfo, Image
from nav_msgs.msg import Odometry
from typing import Any, Dict, List, Tuple, Union
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

image_qos = QoSProfile(
    # Best Effort is fast, but we'll stabilize it with the settings below
    reliability=ReliabilityPolicy.BEST_EFFORT, 
    # Keep only the single most recent frame to prevent backlog lag
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    # Volatile ensures old data isn't saved for late-joiners
    durability=DurabilityPolicy.VOLATILE
)

Config = Dict[str, Any]
Observation = hs.sensor.Observation
Publishers = Dict[str, rclpy.publisher.Publisher] 
Sim = hs.Simulator

def split_pose(T: np.array) -> Tuple[np.array, quaternion.quaternion]:
    """Split a pose in a 4x4 matrix into a position vector and an orientation
    quaternion."""
    return T[0:3, 3], quaternion.from_rotation_matrix(T[0:3, 0:3]).normalized()

def combine_pose(t: np.array, q: quaternion.quaternion) -> np.array:
    """Combine a position vector and an orientation quaternion into a 4x4 pose
    matrix."""
    T = np.identity(4)
    T[0:3, 3] = t
    T[0:3, 0:3] = quaternion.as_rotation_matrix(q.normalized())
    return T

def msg_to_pose(msg: Pose) -> np.array:
    """Convert a ROS Pose message to a 4x4 pose matrix."""
    t = [msg.position.x, msg.position.y, msg.position.z]
    q = quaternion.quaternion(msg.orientation.w, msg.orientation.x,
            msg.orientation.y, msg.orientation.z).normalized()
    return combine_pose(t, q)

def msg_to_transform(msg: Transform) -> np.array:
    """Convert a ROS Transform message to a 4x4 transform matrix."""
    t = [msg.translation.x, msg.translation.y, msg.translation.z]
    q = quaternion.quaternion(msg.rotation.w, msg.rotation.x,
            msg.rotation.y, msg.rotation.z).normalized()
    return combine_pose(t, q)

def transform_to_msg(node: Node, T_TF: np.array, from_frame: str, to_frame: str) -> TransformStamped:
    msg = TransformStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = from_frame
    msg.child_frame_id = to_frame
    msg.transform.translation.x = T_TF[0,3]
    msg.transform.translation.y = T_TF[1,3]
    msg.transform.translation.z = T_TF[2,3]
    q_TF = quaternion.from_rotation_matrix(T_TF[0:3, 0:3]).normalized()
    msg.transform.rotation.x = q_TF.x
    msg.transform.rotation.y = q_TF.y
    msg.transform.rotation.z = q_TF.z
    msg.transform.rotation.w = q_TF.w
    return msg

def list_to_pose(node: Node, l: List) -> Union[np.array, None]:
    """Convert a list to a pose represented by a 4x4 homogeneous matrix. The
    list may have a varying number of elements:
    - 3 (translation: x, y, z)
    - 4 (orientation quaternion: qx, qy, qz, qw)
    - 7 (translation, orientation quaternion)
    - 16 (4x4 homogeneous matrix in row-major from)"""
    n = len(l)
    if n == 3:
        T = np.identity(4)
        T[0:3,3] = np.array(l).T
    elif n == 4:
        q = quaternion.quaternion(l[3], l[0], l[1], l[2]).normalized()
        T = np.identity(4)
        T[0:3,0:3] = quaternion.as_rotation_matrix(q)
    elif n == 7:
        q = quaternion.quaternion(l[6], l[3], l[4], l[5]).normalized()
        T = np.identity(4)
        T[0:3,3] = np.array(l[0:3]).T
        T[0:3,0:3] = quaternion.as_rotation_matrix(q)
    elif n == 16:
        T = np.array(l)
        T = T.reshape((4, 4))
        node.get_logger().warn(str(T))
    else:
        T = None
    return T



def hfov_to_f(hfov: float, width: int) -> float:
    """Convert horizontal field of view in degrees to focal length in pixels.
    https://github.com/facebookresearch/habitat-sim/issues/402"""
    return 1.0 / (2.0 / float(width) * math.tan(math.radians(hfov) / 2.0))

def f_to_hfov(f: float, width: int) -> float:
    """Convert focal length in pixels to horizontal field of view in degrees.
    https://github.com/facebookresearch/habitat-sim/issues/402"""
    return math.degrees(2.0 * math.atan(float(width) / (2.0 * f)))



def find_tf(node: Node, tf_buffer: tf2_ros.Buffer, from_frame: str, to_frame: str) -> Union[np.array, None]:
    try:
        now = rclpy.time.Time()
        timeout = rclpy.duration.Duration(seconds=0.01)
        trans = tf_buffer.lookup_transform(from_frame, to_frame, now, timeout)
        return msg_to_transform(trans.transform)
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        node.get_logger().fatal('Could not find transform from frame "' + from_frame
                + '" to frame "' + to_frame + '"')
        raise



def remove_invalid_objects(objects: List[hs.scene.SemanticObject]) -> List[hs.scene.SemanticObject]:
    return [x for x in objects if x is not None and x.category is not None]

def get_instance_id(o: hs.scene.SemanticObject) -> int:
    """Return the instance ID of the object."""
    s = o.id.strip("_")
    if "_" in s:
        return [int(x) for x in s.split("_")][2]
    else:
        return int(s)


def quaternion_to_euler_angle(w, x, y, z):
	ysqr = y * y

	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + ysqr)
	X = math.atan2(t0, t1)

	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	Y = math.asin(t2)

	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (ysqr + z * z)
	Z = math.atan2(t3, t4)

	return X, Y, Z

class HabitatROS2Node(Node): 
   
    class_colors = np.array([
        [0xff, 0xff, 0xff],
        [0xae, 0xc7, 0xe8],
        [0x70, 0x80, 0x90],
        [0x98, 0xdf, 0x8a],
        [0xc5, 0xb0, 0xd5],
        [0xff, 0x7f, 0x0e],
        [0xd6, 0x27, 0x28],
        [0x1f, 0x77, 0xb4],
        [0xbc, 0xbd, 0x22],
        [0xff, 0x98, 0x96],
        [0x2c, 0xa0, 0x2c],
        [0xe3, 0x77, 0xc2],
        [0xde, 0x9e, 0xd6],
        [0x94, 0x67, 0xbd],
        [0x8c, 0xa2, 0x52],
        [0x84, 0x3c, 0x39],
        [0x9e, 0xda, 0xe5],
        [0x9c, 0x9e, 0xde],
        [0xe7, 0x96, 0x9c],
        [0x63, 0x79, 0x39],
        [0x8c, 0x56, 0x4b],
        [0xdb, 0xdb, 0x8d],
        [0xd6, 0x61, 0x6b],
        [0xce, 0xdb, 0x9c],
        [0xe7, 0xba, 0x52],
        [0x39, 0x3b, 0x79],
        [0xa5, 0x51, 0x94],
        [0xad, 0x49, 0x4a],
        [0xb5, 0xcf, 0x6b],
        [0x52, 0x54, 0xa3],
        [0xbd, 0x9e, 0x39],
        [0xc4, 0x9c, 0x94],
        [0xf7, 0xb6, 0xd2],
        [0x6b, 0x6e, 0xcf],
        [0xff, 0xbb, 0x78],
        [0xc7, 0xc7, 0xc7],
        [0x8c, 0x6d, 0x31],
        [0xe7, 0xcb, 0x94],
        [0xce, 0x6d, 0xbd],
        [0x17, 0xbe, 0xcf],
        [0x7f, 0x7f, 0x7f]
    ])
    # 底层c++的libgdal 和 libtiff有冲突
    _bridge = CvBridge()

    # _rgb_topic_name = "~rgb/"
    # _depth_topic_name = "~depth/"""/home/vlak/habitat_ros/src/00853-5cdEh9F2hJL/5cdEh9F2hJL.glb"
    # _sem_class_topic_name = "~semantic_class/"
    # _sem_instance_topic_name = "~semantic_instance/"
    # _habitat_pose_topic_name = "~pose"
    # _habitat_odom_topic_name = "~odom"

    _rgb_topic_name = "/camera/color/"
    _depth_topic_name = "/camera/depth/"
    _sem_class_topic_name = "~/semantic_class/"
    _sem_instance_topic_name = "~/semantic_instance/"
    _habitat_pose_topic_name = "~/pose"
    _habitat_odom_topic_name = "~/odom"
    _external_pose_topic_name = "/cmd_pose"

    # Subscribed topic names
    _external_pose_topic_name = "/cmd_pose"

    # Transforms between the internal habitat frame I (y-up) and the exported
    # habitat frame H (z-up)
    _T_HI = np.identity(4)
    # _T_HI[0:3, 0:3] = quaternion.as_rotation_matrix(hs.utils.common.quat_from_two_vectors(
    #         hs.geo.GRAVITY, np.array([0.0, 0.0, -1.0])))
    _T_HI[0:3, 0:3] = np.array([
        [1.0, 0.0,  0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0,  0.0]
    ])

    _T_IH = np.linalg.inv(_T_HI)

    # Transforms between the habitat camera frame C (-z-forward, y-up) and the
    # ROS body frame B (x-forward, z-up)
    _T_CB = np.array([(0.0, -1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)])
    _T_BC = np.linalg.inv(_T_CB)

    # Transforms between the TUM camera frame Ctum (z-forward, x-right) and the
    # ROS body frame B (x-forward, z-up)
    _T_BCtum = np.array([(0.0, 0.0, 1.0, 0.0), (-1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)])

    # The default node options
    _default_config = {
            "skip_image": 1,
            "compress_image":False,
            "width": 640,
            "height": 480,
            "near_plane": 0.1,
            "far_plane": 10.0,
            "f": 525.0,
            "fps": 30,
            "enable_semantics": False,
            "depth_noise": False,
            "allowed_classes": [],
            "scene_file": "",
            "initial_T_HB": [],
            "pose_frame_id": "world",
            "pose_frame_at_initial_T_HB": False,
            "visualize_semantics": False,
            "recording_dir": ""}




    def __init__(self):
        super().__init__('habitat_node')
        self.get_logger().info('Habitat node starting up...')
        

        self.config = self._read_node_config()
        self.sim = self._init_habitat(self.config)
        self.pub = self._init_publishers(self.config)


        self.img_count=0
        self.skip_image = self.config["skip_image"]

        self.x_prev = None
        self.y_prev = None
        self.z_prev = None
        self.yaw_prev = None
        self.stamp_prev = None
        self.counter = 0
        self.last_vel_receive_t = self.get_clock().now()
        self.T_HB_mutex = threading.Lock()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


        self.pose_sub = self.create_subscription(
            PoseStamped, 
            self._external_pose_topic_name, 
            self._pose_callback, 
            1)
        self.vel_sub = self.create_subscription(
            Twist, 
            "/cmd_vel", 
            self._vel_callback, 
            1)
        self.get_logger().info("Habitat node ready")


        if self.config["fps"] > 0:
            timer_period = 1.0 / self.config["fps"]
            self.timer = self.create_timer(timer_period, self._timer_callback)
        else:
            self.get_logger().warn("FPS is 0 or negative, rendering loop will not start automatically.")

    def _timer_callback(self):
        #t = time.time()
        observation = self._move_and_render(self.sim, self.config)
        #t1 = time.time()
        self._publish_observation(observation, self.pub, self.config)
        # if self.config["recording_dir"]:
        #     self._record_observation(observation, self.config["recording_dir"])
        # t2=time.time()
        # print(t1-t,t2-t1)

    def _read_node_config(self) -> Config:
        config = copy.deepcopy(self._default_config)
        
        # declare and then get
        for name, default_val in config.items():
            self.declare_parameter("habitat." + name, default_val)
            config[name] = self.get_parameter("habitat." + name).value

        config["scene_file"] = os.path.expanduser(config["scene_file"])
        if not os.path.isabs(config["scene_file"]):
            try:
                package_path = get_package_share_directory("habitat_ros2") + "/"
                config["scene_file"] = package_path + config["scene_file"]
            except Exception as e:
                self.get_logger().error(f"Failed to find package 'habitat_ros2': {e}")
                
        if not config["scene_file"]:
            self.get_logger().fatal("No scene file supplied")
            raise RuntimeError("No scene file supplied")
        elif not os.path.isfile(config["scene_file"]):
            self.get_logger().fatal("Scene file " + config["scene_file"] + " does not exist")
            raise RuntimeError("Scene file does not exist")
            

        T = list_to_pose(self, config["initial_T_HB"])
        if T is None and config["initial_T_HB"]:
            self.get_logger().error("Invalid initial T_HB. Expected list of 3, 4, 7 or 16 elements")
        config["initial_T_HB"] = T
        
        if config["recording_dir"]:
            config["recording_dir"] = os.path.expanduser(config["recording_dir"])
            

        self.get_logger().info("Habitat node parameters:")
        for name, val in config.items():
            self.get_logger().info(f"  {name: <20}: {val}")
            
        return config
    
    def _generateOdom(self, pose):
        (roll, pitch, yaw) = quaternion_to_euler_angle(
            pose.pose.orientation.w, pose.pose.orientation.x , 
            pose.pose.orientation.y, pose.pose.orientation.z)

        x = pose.pose.position.x
        y = pose.pose.position.y
        z = pose.pose.position.z

        stamp = self.get_clock().now()
        odom = Odometry()
        
        if self.counter > 0:
            dt = (stamp - self.stamp_prev).nanoseconds / 1e9
            
            if dt > 0:
                vel_x_world = (x - self.x_prev) / dt
                vel_y_world = (y - self.y_prev) / dt

                twist_x = math.cos(yaw) * vel_x_world + math.sin(yaw) * vel_y_world
                twist_y = math.cos(yaw) * vel_y_world - math.sin(yaw) * vel_x_world

                odom.header.frame_id = self.config["pose_frame_id"]
                odom.child_frame_id = 'base_link'

                odom.header.stamp = stamp.to_msg()

                odom.pose.pose.position.x = x
                odom.pose.pose.position.y = y
                odom.pose.pose.position.z = z
                odom.pose.pose.orientation = pose.pose.orientation

                odom.twist.twist.linear.x = vel_x_world
                odom.twist.twist.linear.y = vel_y_world 
                odom.twist.twist.linear.z = (z - self.z_prev) / dt

                odom.twist.twist.angular.x = 0.0
                odom.twist.twist.angular.y = 0.0
                odom.twist.twist.angular.z = (yaw - self.yaw_prev) / dt
        else:
            self.counter += 1

        self.x_prev = x
        self.y_prev = y
        self.z_prev = z
        self.yaw_prev = yaw
        self.stamp_prev = stamp

        return odom

    def _init_habitat(self, config: Config) -> Sim:
        """Initialize the Habitat simulator, create the sensors and load the
        scene file."""
        backend_config = hs.SimulatorConfiguration()
        backend_config.scene_id = (config["scene_file"])
        agent_config = hs.AgentConfiguration()
        agent_config.sensor_specifications = [self._rgb_sensor_config(config),
                self._depth_sensor_config(config), self._semantic_sensor_config(config)]
        agent_config.height = 0.0
        agent_config.radius = 0.0
        sim = Sim(hs.Configuration(backend_config, [agent_config]))
        # Get the intrinsic camera parameters
        hfov = float(agent_config.sensor_specifications[0].hfov)
        f = hfov_to_f(hfov, config["width"])
        cx = config["width"] / 2.0 - 0.5
        cy = config["height"] / 2.0 - 0.5
        config["K"] = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64)
        config["P"] = np.array([[f, 0.0, cx, 0.0], [0.0, f, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0]],
                dtype=np.float64)
        self.class_id_to_name = self._class_id_to_name_map(sim.semantic_scene.categories)
        # Setup the instance/class conversion map
        if config["enable_semantics"]:
            config["instance_to_class"] = self._instance_to_class_map(
                    remove_invalid_objects(sim.semantic_scene.objects), self.class_id_to_name)
            if config["instance_to_class"].size == 0:
                self.get_logger().warn("The scene contains no semantics")
        # Get or set the initial agent pose
        agent = sim.get_agent(0)
        if config["initial_T_HB"] is None:
            t_IC = agent.get_state().position
            q_IC = agent.get_state().rotation
            T_IC = combine_pose(t_IC, q_IC)
            self.T_HB = self._T_IC_to_T_HB(T_IC)
        else:
            self.T_HB = config["initial_T_HB"]
            t_IC, q_IC = split_pose(self._T_HB_to_T_IC(self.T_HB))
            agent_state = hs.agent.AgentState(t_IC, q_IC)
            agent.set_state(agent_state)
        t_HB, q_HB = split_pose(self.T_HB)
        # Initialize the current pose timestamp to zero.
        self.T_HB_stamp = self.get_clock().now().to_msg()
        self.T_HB_received = False
        self.get_logger().info("Habitat initial t_HB (x,y,z):   {}, {}, {}".format(t_HB[0], t_HB[1], t_HB[2]))
        self.get_logger().info("Habitat initial q_HB (x,y,z,w): {}, {}, {}, {}".format(q_HB.x, q_HB.y, q_HB.z, q_HB.w))
        return sim
    
    def _cv2_to_imgmsg(self, cv_image: np.ndarray, encoding: str) -> Image:
        msg = Image()
        msg.height = cv_image.shape[0]
        msg.width = cv_image.shape[1]
        msg.encoding = encoding
        
        if len(cv_image.shape) == 3:
            msg.step = cv_image.shape[1] * cv_image.shape[2] * cv_image.itemsize
        else:
            msg.step = cv_image.shape[1] * cv_image.itemsize
            
        msg.data = cv_image.tobytes()
        return msg
    
    def _rgb_sensor_config(self, config: Config) -> hs.CameraSensorSpec:
        """Return the configuration for a Habitat color sensor."""
        rgb_sensor_spec = hs.CameraSensorSpec()
        rgb_sensor_spec.uuid = "rgb"
        rgb_sensor_spec.sensor_type = hs.SensorType.COLOR
        rgb_sensor_spec.sensor_subtype = hs.SensorSubType.PINHOLE
        rgb_sensor_spec.resolution = [config["height"], config["width"]]
        rgb_sensor_spec.near = 0.00001
        rgb_sensor_spec.far = 1000
        rgb_sensor_spec.hfov = f_to_hfov(config["f"], config["width"])
        # rgb_sensor_spec.position = np.zeros((3, 1))
        # rgb_sensor_spec.orientation = np.zeros((3, 1))
        rgb_sensor_spec.position = np.zeros(3)
        rgb_sensor_spec.orientation = np.zeros(3)
        return rgb_sensor_spec
    
    def _depth_sensor_config(self, config: Config) -> hs.CameraSensorSpec:
        """Return the configuration for a Habitat depth sensor."""
        depth_sensor_spec = hs.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth"
        depth_sensor_spec.sensor_type = hs.SensorType.DEPTH
        depth_sensor_spec.sensor_subtype = hs.SensorSubType.PINHOLE
        depth_sensor_spec.resolution = [config["height"], config["width"]]
        depth_sensor_spec.near = config["near_plane"]
        depth_sensor_spec.far = config["far_plane"]
        depth_sensor_spec.hfov = f_to_hfov(config["f"], config["width"])
        # depth_sensor_spec.position = np.zeros((3, 1))
        # depth_sensor_spec.orientation = np.zeros((3, 1))
        depth_sensor_spec.position = np.zeros(3)
        depth_sensor_spec.orientation = np.zeros(3)
        if config["depth_noise"]:
            depth_sensor_spec.noise_model = "RedwoodDepthNoiseModel"
        return depth_sensor_spec
    
    def _semantic_sensor_config(self, config: Config) -> hs.CameraSensorSpec:
        """Return the configuration for a Habitat semantic sensor."""
        semantic_sensor_spec = hs.CameraSensorSpec()
        semantic_sensor_spec.uuid = "semantic"
        semantic_sensor_spec.sensor_type = hs.SensorType.SEMANTIC
        semantic_sensor_spec.sensor_subtype = hs.SensorSubType.PINHOLE
        semantic_sensor_spec.resolution = [config["height"], config["width"]]
        semantic_sensor_spec.near = 0.00001
        semantic_sensor_spec.far = 1000
        semantic_sensor_spec.hfov = f_to_hfov(config["f"], config["width"])
        # semantic_sensor_spec.position = np.zeros((3, 1))
        # semantic_sensor_spec.orientation = np.zeros((3, 1))
        semantic_sensor_spec.position = np.zeros(3)
        semantic_sensor_spec.orientation = np.zeros(3)
        return semantic_sensor_spec
    
    def _class_id_to_name_map(self, categories: List) -> Dict[int, str]:
        """Generate a dictionary from class IDs to class names."""
        return {x.index(): x.name() for x in categories if x is not None}

    def _instance_to_class_map(self, objects: List[hs.scene.SemanticObject], classes: Dict[int, str]) -> np.ndarray:
        """Given the objects in the scene, create an array that maps instance
        IDs to class IDs."""
        # Default is -1 so that an empty array is created in the following line
        # if there are no objects.
        max_instance_id = max([get_instance_id(x) for x in objects], default=-1)
        mapping = np.zeros(max_instance_id + 1, dtype=np.uint8)
        for object in objects:
            instance_id = get_instance_id(object)
            mapping[instance_id] = object.category.index()
            if mapping[instance_id] not in classes.keys():
                self.get_logger().warn('Invalid object class ID/name {}/"{}", replacing with 0/"{}"'.format(
                    mapping[instance_id], object.category.name(), classes[0]))
                mapping[instance_id] = 0
        return mapping
    
    def _init_publishers(self, config: Config) -> Publishers:
        image_queue_size = 1
        pub = {}
        # Pose publisher       
        pub["camera_pose"] = self.create_publisher(PoseStamped, "/camera_pose", 1)
        self.br = TransformBroadcaster(self)
        pub["pose"] = self.create_publisher(PoseStamped, self._habitat_pose_topic_name, 1)
        pub["odom"] = self.create_publisher(Odometry, self._habitat_odom_topic_name, 1)
        # Image publishers

        if config["compress_image"]:
            pub["rgb"] = self.create_publisher(CompressedImage,  self._rgb_topic_name +'image_raw/compressed', image_qos)
            pub["depth"] = self.create_publisher(CompressedImage, self._depth_topic_name+'image_raw/compressed', image_qos)
        else:
            pub["rgb"] = self.create_publisher(Image, self._rgb_topic_name + "image_raw", 1)
            pub["depth"] = self.create_publisher(Image, self._depth_topic_name + "image_raw", 1)
        
        if config["enable_semantics"] and config["instance_to_class"].size > 0:
            pub["sem_class"] = self.create_publisher(Image, self._sem_class_topic_name + "image_raw", image_queue_size)
            pub["sem_instance"] = self.create_publisher(Image, self._sem_instance_topic_name + "image_raw", image_queue_size)
            if config["visualize_semantics"]:
                pub["sem_class_render"] = self.create_publisher(Image, self._sem_class_topic_name + "image_color", image_queue_size)
                pub["sem_instance_render"] = self.create_publisher(Image, self._sem_instance_topic_name + "image_color", image_queue_size)
                

        latch_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        
        image_topics = [self._rgb_topic_name, self._depth_topic_name]
        if config["enable_semantics"] and config["instance_to_class"].size > 0:
            image_topics += [self._sem_class_topic_name, self._sem_instance_topic_name]
                    
        pub["rgb_camera_info"] = self.create_publisher(CameraInfo, self._rgb_topic_name + "camera_info", latch_qos)
        pub["depth_camera_info"] = self.create_publisher(CameraInfo, self._depth_topic_name + "camera_info", latch_qos)

            
        return pub

    def _pose_callback(self, pose: PoseStamped) -> None:
        """Callback for receiving external pose messages. It updates the agent
        pose."""
        # Find the transform from the pose frame F to the habitat frame H
       # T_HE = find_tf(self.tf_buffer, "habitat", pose.header.frame_id)
        # Transform the pose
        T_EB = msg_to_pose(pose.pose)
        T_HB =  T_EB
        # Update the pose
        self.T_HB_mutex.acquire()
        self.T_HB = T_HB
        self.T_HB_stamp = pose.header.stamp
        self.T_HB_received = True
        self.T_HB_mutex.release()

    def _vel_callback(self, twist: Twist) -> None:

        def twist_to_transform(T_current, twist_msg, dt):

            v = np.array([twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z], dtype=float)
            w = np.array([twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z], dtype=float)
            w_norm = np.linalg.norm(w)
            if w_norm < 1e-9:
                R_inc = np.eye(3)
            else:
                k = w / w_norm
                theta = w_norm * dt
                K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                R_inc = (np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K))
            t_inc = v * dt
            R_current = T_current[:3, :3]
            R_new = R_current @ R_inc
            t_new = T_current[:3, 3] + R_current @ t_inc
            T_new = np.eye(4)
            T_new[:3, :3] = R_new
            T_new[:3, 3] = t_new
            return T_new
        
        cur_t = self.get_clock().now()

        dt = (cur_t - self.last_vel_receive_t).nanoseconds / 1e9
        
        if dt < 0.2:
            cur_T_HB = np.copy(self.T_HB)
            new_T = twist_to_transform(cur_T_HB, twist, dt)
            
            self.T_HB_mutex.acquire()
            self.T_HB = new_T

            self.T_HB_stamp = self.get_clock().now().to_msg()
            self.T_HB_received = True
            self.T_HB_mutex.release()
        
        self.last_vel_receive_t = cur_t

    def _filter_sem_classes(self, observation: Observation) -> None:
        """Remove object detections whose classes are not in the allowed class
        list. Their class and instance IDs are set to 0."""
        # Generate a per-pixel boolean matrix
        allowed = np.vectorize(lambda x: x in self.config["allowed_classes"])
        allowed_pixels = allowed(observation["sem_classes"])
        # Set all False pixels to 0 on the class and instance images
        class_zeros = np.zeros(observation["sem_classes"].shape, dtype=observation["sem_classes"].dtype)
        instance_zeros = np.zeros(observation["sem_instances"].shape, dtype=observation["sem_instances"].dtype)
        observation["sem_classes"] = np.where(allowed_pixels, observation["sem_classes"], class_zeros)
        observation["sem_instances"] = np.where(allowed_pixels, observation["sem_instances"], instance_zeros)

    def _pose_to_msg(self, observation: Observation) -> PoseStamped:
        """Convert the agent pose from the observation to a ROS PoseStamped
        message."""
        # T_PH = find_tf(self.tf_buffer, self.config["pose_frame_id"], "habitat")
        # t_PB, q_PB = split_pose(T_PH @ observation["T_HB"])
        t_PB, q_PB = split_pose( observation["T_HB"])
        p = PoseStamped()
        p.header.frame_id = self.config["pose_frame_id"]
        p.header.stamp = observation["timestamp"]
        p.pose.position.x = t_PB[0]
        p.pose.position.y = t_PB[1]
        p.pose.position.z = t_PB[2]
        p.pose.orientation.x = q_PB.x
        p.pose.orientation.y = q_PB.y
        p.pose.orientation.z = q_PB.z
        p.pose.orientation.w = q_PB.w
        return p

    def _pose_to_camera(self, observation: Observation):
        """Convert the agent pose from the observation to a ROS PoseStamped
        message."""
       
        t_PB, q_PB = split_pose( observation["T_HB"] @ np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]) )
        p = PoseStamped()
        p.header.frame_id = self.config["pose_frame_id"]
        p.header.stamp = observation["timestamp"]
        p.pose.position.x = t_PB[0]
        p.pose.position.y = t_PB[1]
        p.pose.position.z = t_PB[2]
        p.pose.orientation.x = q_PB.x
        p.pose.orientation.y = q_PB.y
        p.pose.orientation.z = q_PB.z
        p.pose.orientation.w = q_PB.w
        self.pub["camera_pose"].publish(p)

        msg = TransformStamped()
        msg.header = p.header
        msg.child_frame_id = "fixed_camera"

        msg.transform.translation.x = t_PB[0]
        msg.transform.translation.y = t_PB[1]
        msg.transform.translation.z = t_PB[2]

        msg.transform.rotation.x = q_PB.x
        msg.transform.rotation.y = q_PB.y
        msg.transform.rotation.z = q_PB.z
        msg.transform.rotation.w = q_PB.w

        self.br.sendTransform(msg)

    # def _rgb_to_msg(self, observation: Observation) -> Image:
    #     """Convert the RGB image from the observation to a ROS Image message."""
    #     if(self.config["compress_image"]) :
            
    #     else
    #         msg = self._bridge.cv2_to_imgmsg(observation["rgb"], "rgb8")
    #     #msg = self._cv2_to_imgmsg(observation["rgb"], "rgb8")
    #     msg.header.stamp = observation["timestamp"]
    #     return msg

    # def _depth_to_msg(self, observation: Observation) -> Image:
    #     """Convert the depth image from the observation to a ROS Image
    #     message."""
    #     msg = self._bridge.cv2_to_imgmsg(observation["depth"], "32FC1")
    #     #msg = self._cv2_to_imgmsg(observation["depth"], "32FC1")
    #     msg.header.stamp = observation["timestamp"]
    #     return msg

    def _rgb_to_msg(self, observation: Observation) -> Image | CompressedImage:
        if self.config["compress_image"]:
            msg = CompressedImage()
            msg.header.stamp = observation["timestamp"]
            msg.format = "jpeg"
            # Convert RGB to BGR for cv2 encoding
            bgr = cv2.cvtColor(observation["rgb"], cv2.COLOR_RGB2BGR)
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, self.config.get("jpeg_quality", 85)]
            _, compressed = cv2.imencode('.jpg', bgr, encode_param)
            msg.data = compressed.tobytes()
            return msg

        msg = self._bridge.cv2_to_imgmsg(observation["rgb"], "rgb8")
        msg.header.stamp = observation["timestamp"]
        return msg

    def _depth_to_msg(self, observation: Observation) -> Image | CompressedImage:
        depth_mm = (observation["depth"] * 1000).astype(np.uint16)
        if self.config["compress_image"]:
            msg = CompressedImage()
            msg.header.stamp = observation["timestamp"]
            msg.format = "16UC1; png compressed" #"png"  # PNG is lossless, better for depth data
            # Normalize depth to 16-bit for PNG compression
            depth = observation["depth"]
            depth_16bit = (depth * 1000).astype(np.uint16)  # convert meters to mm
            _, compressed = cv2.imencode('.png', depth_16bit,[cv2.IMWRITE_PNG_COMPRESSION, 1])
            msg.data = compressed.tobytes()
            return msg
        # msg = self._bridge.cv2_to_imgmsg(observation["depth"], "32FC1")

  
        msg = self._bridge.cv2_to_imgmsg(depth_mm, encoding="16UC1")
        msg.header.stamp = observation["timestamp"]
        return msg

    def _sem_instances_to_msg(self, observation: Observation) -> Image:
        """Convert the instance ID image from the observation to a ROS Image
        message."""
        # Habitat-Sim produces 16-bit per-pixel instance ID images.
        # msg = self._bridge.cv2_to_imgmsg(observation["sem_instances"].astype(np.uint16), "16UC1")
        msg = self._cv2_to_imgmsg(observation["sem_instances"].astype(np.uint16), "16UC1")
        msg.header.stamp = observation["timestamp"]
        return msg

    def _sem_classes_to_msg(self, observation: Observation) -> Image:
        """Convert the class ID image from the observation to a ROS Image
        message."""
        # Habitat-Sim produces 8-bit per-pixel class ID images.
        # msg = self._bridge.cv2_to_imgmsg(observation["sem_classes"].astype(np.uint8), "8UC1")
        msg = self._cv2_to_imgmsg(observation["sem_classes"].astype(np.uint8), "8UC1")
        msg.header.stamp = observation["timestamp"]
        return msg

    def _render_sem_instances_to_msg(self, observation: Observation) -> Image:
        """Visualize an instance ID image to a ROS Image message with
        per-instance colours."""
        color_img = self.class_colors[observation["sem_instances"] % len(self.class_colors)]
        color_img = color_img / 2 + observation["rgb"] / 2
        # msg = self._bridge.cv2_to_imgmsg(color_img.astype(np.uint8), "rgb8")
        msg = self._cv2_to_imgmsg(color_img.astype(np.uint8), "rgb8")
        msg.header.stamp = observation["timestamp"]
        return msg

    def _render_sem_classes_to_msg(self, observation: Observation) -> Image:
        """Visualize a class ID image to a ROS Image message with per-class
        colours."""
        color_img = self.class_colors[observation["sem_classes"] % len(self.class_colors)]
        color_img = color_img / 2 + observation["rgb"] / 2
        # msg = self._bridge.cv2_to_imgmsg(color_img.astype(np.uint8), "rgb8")
        msg = self._cv2_to_imgmsg(color_img.astype(np.uint8), "rgb8")
        msg.header.stamp = observation["timestamp"]
        return msg

    def _camera_intrinsics_to_msg(self, config: Config) -> CameraInfo:
        """Return a ROS message containing the Habitat-Sim camera intrinsic
        parameters."""
        # TODO Set parameters in the message header?
        # http://docs.ros.org/electric/api/sensor_msgs/html/msg/CameraInfo.html
        msg = CameraInfo()
        msg.width = config["width"]
        msg.height = config["height"]
        msg.k = config["K"].flatten().tolist()
        msg.p = config["P"].flatten().tolist()
        return msg

    def _T_IC_to_T_HB(self, T_IC: np.array) -> np.array:
        """Convert T_IC to T_HB."""
        return self._T_HI @ T_IC @ self._T_CB

    def _T_HB_to_T_IC(self, T_HB: np.array) -> np.array:
        """Convert T_HB to T_IC."""
        return self._T_IH @ T_HB @ self._T_BC

    def _move_and_render(self, sim: Sim, config: Config) -> Observation:
        self.T_HB_mutex.acquire()
        T_HB = np.copy(self.T_HB)
        stamp = copy.deepcopy(self.T_HB_stamp)
        T_HB_received = self.T_HB_received
        self.T_HB_received = False
        self.T_HB_mutex.release()
        
        t_IC, q_IC = split_pose(self._T_HB_to_T_IC(T_HB))
        agent_state = hs.agent.AgentState(t_IC, q_IC)
        self.sim.get_agent(0).set_state(agent_state)
        
        observation = sim.get_sensor_observations()
        
        if T_HB_received:
            observation["timestamp"] = stamp
        else:
            observation["timestamp"] = self.get_clock().now().to_msg()
            
        observation["rgb"] = observation["rgb"][..., 0:3]
        if config["enable_semantics"] and config["instance_to_class"].size > 0:
            observation["sem_instances"] = np.clip(observation["semantic"].astype(np.uint16), 0, 65535)
            del observation["semantic"]
            observation["sem_classes"] = np.array(
                    [config["instance_to_class"][x] for x in observation["sem_instances"]],
                    dtype=np.uint8)
                    
        t_IC = sim.get_agent(0).get_state().position
        q_IC = sim.get_agent(0).get_state().rotation
        T_IC = combine_pose(t_IC, q_IC)
        observation["T_HB"] = self._T_IC_to_T_HB(T_IC)
        return observation

    def _publish_observation(self, obs: Observation, pub: Publishers, config: Config) -> None:
        """Publish the sensor observations and ground truth pose."""
        pos_msg=self._pose_to_msg(obs)
        pub["pose"].publish(pos_msg)
        pub["odom"].publish(self._generateOdom(pos_msg))
        self._pose_to_camera(obs)

        self.img_count += 1
        if self.img_count % self.skip_image == 0:
            t=time.time()
    # Prepare messages
            r = self._rgb_to_msg(obs)
            d = self._depth_to_msg(obs) # Use the 16UC1 raw logic here
            
            t1 = time.time()
            
            # Publish messages
            pub["rgb"].publish(r)
            pub["depth"].publish(d)
            
            t2 = time.time()
            print(f"img pub | Setup: {t1-t:.4f}s | Pub: {t2-t1:.4f}s | Frame: {self.img_count}")
        # pub["rgb"].publish(self._bridge.cv2_to_imgmsg(obs["rgb"], "rgb8"))
        # pub["depth"].publish(self._bridge.cv2_to_imgmsg(obs["depth"], "32FC1"))
            msg=self._camera_intrinsics_to_msg(config)
            pub["depth_camera_info"].publish(msg)
            pub["rgb_camera_info"].publish(msg)

            if config["enable_semantics"] and config["instance_to_class"].size > 0:
                if config["allowed_classes"]:
                    self._filter_sem_classes(obs)
                pub["sem_class"].publish(self._sem_classes_to_msg(obs))
                pub["sem_instance"].publish(self._sem_instances_to_msg(obs))
                # Publish semantics visualisations
                if config["visualize_semantics"]:
                    pub["sem_class_render"].publish(self._render_sem_classes_to_msg(obs))
                    pub["sem_instance_render"].publish(self._render_sem_instances_to_msg(obs))

    def _record_observation(self, obs: Observation, recording_dir: str) -> None:
        os.makedirs(recording_dir, exist_ok=True)
        os.makedirs(recording_dir + "/depth", exist_ok=True)
        os.makedirs(recording_dir + "/rgb", exist_ok=True)
        
        sec = obs["timestamp"].sec
        nanosec = obs["timestamp"].nanosec
        stamp_sec = sec + nanosec / 1e9
        stamp_str = "{:.7f}".format(stamp_sec)
        
        groundtruth_txt = recording_dir + "/groundtruth.txt"
        if not os.path.isfile(groundtruth_txt):
            with open(groundtruth_txt, "w") as f:
                f.write("# ground truth trajectory\n")
                f.write("# timestamp tx ty tz qx qy qz qw\n")
        with open(groundtruth_txt, "a") as f:
            t_PC, q_PC = split_pose( obs["T_HB"] @ self._T_BCtum)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                stamp_str, t_PC[0], t_PC[1], t_PC[2],
                q_PC.x, q_PC.y, q_PC.z, q_PC.w))
                
        for t in ["depth", "rgb"]:
            type_txt = "".join([recording_dir, "/", t, ".txt"])
            image_png = "".join([t, "/", stamp_str, ".png"])
            if not os.path.isfile(type_txt):
                with open(type_txt, "w") as f:
                    f.write("# {} images\n".format(t))
                    f.write("# timestamp filename\n")
            with open(type_txt, "a") as f:
                f.write("{} {}\n".format(stamp_str, image_png))
                
        depth_png = "".join([recording_dir, "/depth/", stamp_str, ".png"])
        depth_constrained = obs["depth"].astype(np.float32)
        depth_constrained[depth_constrained < 0] = 0
        depth_constrained[depth_constrained >= (2**16 - 1) / 5000] = 0
        cv2.imwrite(depth_png, (5000 * depth_constrained).astype(np.uint16))
        
        rgb_png = "".join([recording_dir, "/rgb/", stamp_str, ".png"])
        cv2.imwrite(rgb_png, cv2.cvtColor(obs["rgb"], cv2.COLOR_BGR2RGB))


def main(args=None):
    rclpy.init(args=args)
    node = None  
    try:
        node = HabitatROS2Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Node crashed with error: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()