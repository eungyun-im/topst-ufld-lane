from pathlib import Path

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch import LaunchDescription
from launch_ros.actions import Node


def get_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return str(Path('/home/topst/D-Racer/src/config/vehicle_config.yaml'))


def get_auto_control_launch_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'control_pkg' / 'launch' / 'control.launch'
        if candidate.exists():
            return str(candidate)
    return str(Path('/home/topst/D-Racer-Kit/src/control_pkg/launch/control.launch'))


def generate_launch_description():
    vehicle_config_path = get_vehicle_config_path()
    auto_control_launch_path = get_auto_control_launch_path()

    return LaunchDescription([
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(auto_control_launch_path),
        ),
        Node(
            package='control',
            executable='control_node',
            name='control_node',
            output='screen',
            parameters=[
                {
                    'use_joystick_control': True,
                    'vehicle_config_file': vehicle_config_path,
                    'command_hz': 50.0,
                },
            ],
        ),
        Node(
            package='joystick',
            executable='joystick_node',
            name='joystick_node',
            output='screen',
            parameters=[
                {
                    'calibration_mode': True,
                    'use_joystick_control': True,
                    'throttle_scale': 0.3,
                    'throttle_min_output': 0.15,
                    'vehicle_config_file': vehicle_config_path,
                },
            ],
        ),
    ])
