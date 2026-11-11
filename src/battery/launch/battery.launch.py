from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='battery',
            executable='battery_node',
            name='battery_node',
            output='screen',
            parameters=[
                {
                    'publish_topic': '/battery_status',
                    'publish_hz': 10.0,
                    'i2c_bus': 3,
                    'ina_addr': 0x42,
                    'r_shunt_ohm': 0.1,
                    'max_current_a': 2.0,
                    'min_voltage': 6.4,
                    'max_voltage': 8.4,
                    'use_load_voltage': True,
                    'debug_log': False,
                },
            ],
        ),
    ])
