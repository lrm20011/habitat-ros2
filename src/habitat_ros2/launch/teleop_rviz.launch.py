import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('habitat_ros2')
    habitat_config = os.path.join(pkg_dir, 'config', 'habitat.yaml')

    node_env = os.environ.copy()
    node_env['MAGNUM_LOG'] = 'quiet'
    node_env['GLOG_minloglevel'] = '1'

    teleop_arg = DeclareLaunchArgument('teleop', default_value='true')
    publish_path_arg = DeclareLaunchArgument('publish_path', default_value='false')


    habitat_node = Node(
        package='habitat_ros2',
        executable='habitat_node_ros2.py',
        name='habitat_node',
        output='screen',
        parameters=[habitat_config],
        env=node_env  
    )

    mesh_node = Node(
        package='habitat_ros2',
        executable='mesh_publisher',
        name='mesh_publisher',
        output='screen',
        parameters=[{
            'mesh_file': 'file:///home/iot/habitat_ros2/00853-5cdEh9F2hJL/5cdEh9F2hJL.glb'
        }]
    )

    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'habitat']
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )

    teleop_node = Node(
        package='habitat_ros2',
        executable='teleop.py',
        name='teleop',
        output='screen',
        parameters=[{'publish_path': LaunchConfiguration('publish_path')}],
        prefix='xterm -e',
        remappings=[('/teleop/pose', '/habitat_node/pose')]
    )

    return LaunchDescription([
        teleop_arg,
        publish_path_arg,
        static_tf_node,
        habitat_node,
        mesh_node,
        rviz_node,
        teleop_node
    ])