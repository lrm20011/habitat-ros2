#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2020-2022 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020-2022 Sotiris Papatheodorou
# SPDX-License-Identifier: BSD-3-Clause

import curses
import math
import time
import numpy as np
import quaternion

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from typing import Tuple



_node_name = "teleop"

_pose_input_topic = "/habitat_node/pose"

_output_topic = "/cmd_pose"


class Movement:
    # All movement happens with respect to the body frame.
    _x_step = 0.25 # metres, x-axis movement step
    _y_step = 0.25 # metres, y-axis movement step
    _z_step = 0.25 # metres, z-axis movement step
    _roll_step = 5 # degrees, roll rotation step
    _pitch_step = 5 # degrees, pitch rotation step
    _yaw_step = 5 # degrees, yaw rotation step

    def __init__(self, x: int=0, y: int=0, z: int=0,
            roll: int=0, pitch: int=0, yaw: int=0) -> None:
        self._x = x
        self._y = y
        self._z = z
        self._roll = roll
        self._pitch = pitch
        self._yaw = yaw

    def x(self) -> float:
        return self._x * Movement._x_step

    def y(self) -> float:
        return self._y * Movement._y_step

    def z(self) -> float:
        return self._z * Movement._z_step

    def roll(self) -> float:
        return math.radians(self._roll * Movement._roll_step)

    def pitch(self) -> float:
        return math.radians(self._pitch * Movement._pitch_step)

    def yaw(self) -> float:
        return math.radians(self._yaw * Movement._yaw_step)


def wait_for_key(window) -> Tuple[Movement, bool]:
    m = Movement()
    quit_flag = False
    while True:
        key = window.getkey()
        if key == "Q":
            quit_flag = True
            break
        elif key == "w": m = Movement(x=1); break
        elif key == "s": m = Movement(x=-1); break
        elif key == "a": m = Movement(y=1); break
        elif key == "d": m = Movement(y=-1); break
        elif key == " ": m = Movement(z=1); break
        elif key == "c": m = Movement(z=-1); break
        elif key == "q": m = Movement(yaw=1); break
        elif key == "e": m = Movement(yaw=-1); break
        elif key == "f": m = Movement(pitch=1); break
        elif key == "r": m = Movement(pitch=-1); break
        elif key == "x": m = Movement(roll=1); break
        elif key == "z": m = Movement(roll=-1); break
        time.sleep(0.05)
    return m, quit_flag

def print_help(window) -> None:
    window.addstr(0,  0, "Position:")
    window.addstr(2,  0, "Orientation (w,x,y,z):")
    window.addstr(5,  0, "w/s       forwards/backwards")
    window.addstr(6,  0, "a/d       left/right")
    window.addstr(7,  0, "space/c   up/down")
    window.addstr(8,  0, "q/e       yaw left/right")
    window.addstr(9,  0, "r/f       pitch up/down")
    window.addstr(10, 0, "z/x       roll CCW/CW")
    window.addstr(11, 0, "Q         quit")

def print_pose_stamped(p: PoseStamped, window) -> None:
    position = [p.pose.position.x, p.pose.position.y, p.pose.position.z]
    orientation = [p.pose.orientation.w, p.pose.orientation.x,
            p.pose.orientation.y, p.pose.orientation.z]
    window.move(1, 0)
    window.clrtoeol()
    window.addstr(1, 0, "  " + " ".join(["{: 8.3f}".format(x) for x in position]))
    window.move(3, 0)
    window.clrtoeol()
    window.addstr(3, 0, "  " + " ".join(["{: 8.3f}".format(x) for x in orientation]))


class TeleopNode(Node):
    def __init__(self):
        super().__init__(_node_name)
      
        self.declare_parameter("publish_path", False)
        self.publish_path = self.get_parameter("publish_path").value

        output_topic_type = Path if self.publish_path else PoseStamped
        self.pub = self.create_publisher(output_topic_type, _output_topic, 1)

    def init_pose(self, pose_topic):
        pose_msg = None
        def callback(msg):
            nonlocal pose_msg
            pose_msg = msg

        sub = self.create_subscription(PoseStamped, pose_topic, callback, 1)
        while rclpy.ok() and pose_msg is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.destroy_subscription(sub)
        return pose_msg

    def update_pose(self, p, m):
        new_p = PoseStamped()
        new_p.header.stamp = self.get_clock().now().to_msg()
        new_p.header.frame_id = p.header.frame_id

        T_HB = np.identity(4)
        T_HB[0, 3] = p.pose.position.x
        T_HB[1, 3] = p.pose.position.y
        T_HB[2, 3] = p.pose.position.z

        q_current = quaternion.quaternion(
            p.pose.orientation.w, p.pose.orientation.x,
            p.pose.orientation.y, p.pose.orientation.z
        )
        T_HB[0:3, 0:3] = quaternion.as_rotation_matrix(q_current)

        T_BBnew = np.identity(4)
        T_BBnew[0, 3] += m.x()
        T_BBnew[1, 3] += m.y()
        T_BBnew[2, 3] += m.z()

        q_yaw = quaternion.quaternion(math.cos(m.yaw() / 2), 0, 0, math.sin(m.yaw() / 2))
        q_pitch = quaternion.quaternion(math.cos(m.pitch() / 2), 0, math.sin(m.pitch() / 2), 0)
        q_roll = quaternion.quaternion(math.cos(m.roll() / 2), math.sin(m.roll() / 2), 0, 0)
        q_new = q_yaw * q_pitch * q_roll

        T_BBnew[0:3, 0:3] = quaternion.as_rotation_matrix(q_new)
        T_HBnew = T_HB @ T_BBnew
        q = quaternion.from_rotation_matrix(T_HBnew[0:3, 0:3])

        new_p.pose.position.x = T_HBnew[0, 3]
        new_p.pose.position.y = T_HBnew[1, 3]
        new_p.pose.position.z = T_HBnew[2, 3]
        new_p.pose.orientation.x = q.x
        new_p.pose.orientation.y = q.y
        new_p.pose.orientation.z = q.z
        new_p.pose.orientation.w = q.w
        return new_p

    def pose_to_path(self, pose, new_pose):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = pose.header.frame_id
        path.poses.append(pose)
        path.poses.append(new_pose)
        return path


def main_loop(window, node):
    curses.noecho()
    window.clear()
    window.addstr(1, 0, f"Waiting for initial pose on topic {_pose_input_topic}...")
    window.refresh()

    pose = node.init_pose(_pose_input_topic)
    quit_flag = False
    
    window.clear()
    print_help(window)
    last_t = time.time()

    while rclpy.ok() and not quit_flag:
        print_pose_stamped(pose, window)
        movement, quit_flag = wait_for_key(window)
        
        if quit_flag:
            break

        cur_t = time.time()
      
        if cur_t - last_t > 1:
            synced_pose = node.init_pose(_pose_input_topic)
            if synced_pose:
                pose = synced_pose
            last_t = cur_t

        new_pose = node.update_pose(pose, movement)

        if node.publish_path:
            node.pub.publish(node.pose_to_path(pose, new_pose))
        else:
            node.pub.publish(new_pose)

        pose = new_pose
        rclpy.spin_once(node, timeout_sec=0)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    
    try:
        curses.wrapper(main_loop, node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()