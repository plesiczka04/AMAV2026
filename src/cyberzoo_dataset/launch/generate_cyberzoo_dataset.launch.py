from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # PAPARAZZI_HOME: fallback to AMAV2026 path if not set
    paparazzi_home = os.environ.get(
        "PAPARAZZI_HOME", "/home/plesiczka/AMAV/AMAV2026")

    current_gazebo_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
    
    gazebo_model_path = (
        f"{paparazzi_home}/conf/simulator/gazebo/models:"
        f"{paparazzi_home}/sw/ext/tudelft_gazebo_models/models:"
        f"{get_package_share_directory('cyberzoo_dataset')}/models:"
        f"{current_gazebo_model_path}"
    )

    world_file = os.path.join(
        paparazzi_home,
        "sw/ext/tudelft_gazebo_models/world/cyberzoo_orange_poles.world"
    )

    # FIX: Explicitly call gzserver with all three mandatory ROS 2 plugins
    gzserver_cmd = [
        'gzserver',
        '--verbose',
        '-s', 'libgazebo_ros_init.so',
        '-s', 'libgazebo_ros_factory.so',
        world_file
    ]

    # FIX: Explicitly call gzclient
    gzclient_cmd = ['gzclient']

    return LaunchDescription([
        SetEnvironmentVariable(
            name="PAPARAZZI_HOME",
            value=paparazzi_home,
        ),
        SetEnvironmentVariable(
            name="GAZEBO_MODEL_PATH",
            value=gazebo_model_path,
        ),
        
        # Launch Gazebo Server
        ExecuteProcess(
            cmd=gzserver_cmd,
            output='screen'
        ),
        
        # Launch Gazebo Client (GUI)
        ExecuteProcess(
            cmd=gzclient_cmd,
            output='screen'
        ),
        
        Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            name="spawn_virtual_drone",
            arguments=[
                "-entity", "virtual_drone_cam",
                "-file",
                os.path.join(
                    get_package_share_directory("cyberzoo_dataset"),
                    "models/virtual_drone_cam/model.sdf",
                ),
            ],
            output="screen",
        ),
        Node(
            package="cyberzoo_dataset",
            executable="generate_cyberzoo_dataset",
            name="cyberzoo_dataset_generator",
            output="screen",
            parameters=[{
                "num_episodes": 500,
                "area_size": 4.0,
                "altitude": 1.4,
                "min_obstacles": 5,
                "max_obstacles": 12,
                "camera_topic": "/front_camera/image_raw", # FIX: Point to the actual Gazebo topic
                "output_dir": os.path.join(
                    os.environ.get("HOME", "/tmp"), "cyberzoo_dataset")
            }],
        ),
    ])