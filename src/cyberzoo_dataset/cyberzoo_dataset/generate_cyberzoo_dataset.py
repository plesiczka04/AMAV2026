#!/usr/bin/env python3
import os
import math
import random
import csv
import time
import threading

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, SetEntityState
from gazebo_msgs.msg import EntityState
from cv_bridge import CvBridge
import cv2
import numpy as np



def get_random_quaternion():
    # Random Yaw between 0 and 360 degrees (converted to radians)
    yaw = random.uniform(0, 2 * math.pi)
    
    # Simple Euler to Quaternion conversion (assuming Roll and Pitch are 0)
    # Formula: qz = sin(yaw/2), qw = cos(yaw/2)
    qx = 0.0
    qy = 0.0
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    
    return qx, qy, qz, qw

def discretize_yaw(yaw, num_standard_bins=21, standard_max=1.05):
    """
    Maps continuous yaw to 23 discrete bins (21 standard + 2 extreme).
    Class 0: Extreme Right (-90 deg / -1.57 rad)
    Class 1: Standard Max Right (-60 deg / -1.05 rad)
    Class 11: Dead Center (0.0 rad)
    Class 21: Standard Max Left (+60 deg / +1.05 rad)
    Class 22: Extreme Left (+90 deg / +1.57 rad)
    """
    # 1. Extreme Hard Left (+90 deg)
    if yaw > standard_max + 0.1:
        return num_standard_bins + 1  # Index 22
        
    # 2. Extreme Hard Right (-90 deg)
    if yaw < -standard_max - 0.1:
        return 0  # Index 0
        
    # 3. Standard FOV Steering (Bins 1 to 21)
    yaw = max(min(yaw, standard_max), -standard_max)
    
    # Scale from [-1.05, 1.05] into [0, 20], then shift by +1 so index 0 is saved for Extreme Right
    bin_idx = int(((yaw + standard_max) / (2 * standard_max)) * (num_standard_bins - 1))
    
    return bin_idx + 1
class CyberzooDatasetNode(Node):

  def __init__(self):
    super().__init__("cyberzoo_dataset_generator")



    # Parameters
    self.poses_per_scenario = 250
    self.current_pose_count = 0
    self.current_obstacles = []
    self.declare_parameter("flight_boundary", 4.25)
    self.flight_boundary = self.get_parameter("flight_boundary").get_parameter_value().double_value
    self.declare_parameter("num_episodes",100)
    self.declare_parameter("area_size", 5.0)
    self.declare_parameter("altitude", 1.2)
    self.declare_parameter("min_obstacles", 9)
    self.declare_parameter("max_obstacles", 15)
    self.declare_parameter("obstacle_radius", 0.35)
    self.declare_parameter("obstacle_min_spacing", 1.5)
    self.declare_parameter("drone_min_clearance", 0.3) 
    self.declare_parameter("drone_model_name", "virtual_drone_cam")
    self.declare_parameter("camera_topic", "/front_camera/image_raw")
    default_output = os.path.join(os.getenv("HOME", "/tmp"),
                                  "cyberzoo_dataset")
    self.declare_parameter("output_dir", default_output)

    self.num_episodes = self.get_parameter(
        "num_episodes").get_parameter_value().integer_value
    self.area_size = self.get_parameter(
        "area_size").get_parameter_value().double_value
    self.altitude = self.get_parameter(
        "altitude").get_parameter_value().double_value
        
    # FIX: Ensure min is always strictly less than or equal to max
    min_obs_param = self.get_parameter("min_obstacles").get_parameter_value().integer_value
    max_obs_param = self.get_parameter("max_obstacles").get_parameter_value().integer_value
    self.min_obstacles = min(min_obs_param, max_obs_param)
    self.max_obstacles = max(min_obs_param, max_obs_param)

    self.obstacle_radius = self.get_parameter(
        "obstacle_radius").get_parameter_value().double_value
    self.obstacle_min_spacing = self.get_parameter(
        "obstacle_min_spacing").get_parameter_value().double_value
    self.drone_min_clearance = self.get_parameter(
        "drone_min_clearance").get_parameter_value().double_value
    self.drone_model_name = self.get_parameter(
        "drone_model_name").get_parameter_value().string_value
    self.camera_topic = self.get_parameter(
        "camera_topic").get_parameter_value().string_value
    self.output_dir = self.get_parameter(
        "output_dir").get_parameter_value().string_value

    os.makedirs(self.output_dir, exist_ok=True)
    self.labels_path = os.path.join(self.output_dir, "labels.csv")

    self.get_logger().info(f"Output directory: {self.output_dir}")



    # --- SCENARIO MIX (Adjust these to tune difficulty) ---
    self.gate_range = (1, 4)        # Min, Max gates per scenario
    self.pole_range = (2, 5)        # Min, Max orange poles
    self.plant_range = (3, 9)       # Min, Max plants
    self.wall_count = 1             # Number of black panels

    # --- CAMERA/HEIGHT RANDOMIZATION ---
    self.alt_range = (0.70, 1.25)     # Randomized height (meters)
    self.pitch_range = (-0.1, 0.1)  # Randomized camera tilt (radians)
    self.roll_range = (-0.1, 0.1)

# --- TRACKING STATE ---
    # 1. FIND THE INDEX FIRST
    self.current_folder_idx = self._find_next_folder_idx()
    self.get_logger().info(f"Resuming from folder index: {self.current_folder_idx}")

    # 2. NOW SET THE DIMENSIONS
    self.num_dataset_folders = self.current_folder_idx + 50
    self.images_per_folder = 500
    self.images_per_scenario = 250
    
    # 3. INITIALIZE COUNTERS
    self.current_img_in_folder = 0
    self.current_img_in_scenario = 0
    self.total_images_saved = 0

    # Gazebo services
    self.spawn_cli = self.create_client(SpawnEntity, "/spawn_entity")
    self.delete_cli = self.create_client(DeleteEntity, "/delete_entity")
    self.set_state_cli = self.create_client(SetEntityState, "/set_entity_state")

    for cli, name in [
        (self.spawn_cli, "/spawn_entity"),
        (self.delete_cli, "/delete_entity"),
        (self.set_state_cli, "/set_entity_state"),
    ]:
      if not cli.wait_for_service(timeout_sec=10.0):
        self.get_logger().error(
            f"Service {name} not available. Make sure gazebo_ros is running with factory/state plugins."
        )

    # Camera subscriber
    self.bridge = CvBridge()
    self.last_image = None
    self.create_subscription(Image,
                             self.camera_topic,
                             self.image_cb,
                             qos_profile=10)

    # Load obstacle models from Paparazzi's TU Delft Gazebo models
    self.obstacle_sdfs = self._load_obstacle_sdfs()
    # Fallback simple cylinder if we can't load any of the real models
    self.cylinder_sdf = self._make_cylinder_sdf(self.obstacle_radius,
                                                height=2.5)




    # NOW start the background thread
    self.episode_idx = 0
    self.running = True
    self.generation_thread = threading.Thread(target=self._run_episodes)
    self.generation_thread.start()

  def image_cb(self, msg: Image):
    self.last_image = msg
  def _find_next_folder_idx(self):
    """Scans the output directory to find the next available dataset index."""
    if not os.path.exists(self.output_dir):
        return 0
    
    # Look for directories named 'dataset_XX'
    existing_folders = [d for d in os.listdir(self.output_dir) 
                        if os.path.isdir(os.path.join(self.output_dir, d)) 
                        and d.startswith("dataset_")]
    
    if not existing_folders:
        return 0

    indices = []
    for f in existing_folders:
        try:
            # Splits 'dataset_41' -> ['dataset', '41'] and grabs the number
            idx = int(f.split('_')[1])
            indices.append(idx)
        except (IndexError, ValueError):
            continue

    return max(indices) + 1 if indices else 0
  

  
  def _get_current_paths(self):
    """Returns the current subfolder path and its specific labels.csv."""
    folder_name = f"dataset_{self.current_folder_idx:02d}"
    path = os.path.join(self.output_dir, folder_name)
    os.makedirs(path, exist_ok=True)
    
    csv_path = os.path.join(path, "labels.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["img_name", "yaw_rad", "yaw_bin"])
            
    return path, csv_path
  
  def _make_arrow_sdf(self):
    return """
<sdf version="1.6">
  <model name="target_indicator">
    <static>true</static>
    <link name="link">
      <visual name="shaft">
        <pose>0.5 0 0 0 0 0</pose>
        <geometry><box><size>1.0 0.05 0.05</size></box></geometry>
        <material><ambient>0 0 0 0</ambient><diffuse>0 0 0 0</diffuse></material>
      </visual>
      <visual name="head">
        <pose>1.0 0 0 0 0 0</pose>
        <geometry><sphere><radius>0.1</radius></sphere></geometry>
        <material><ambient>0 0 0 0</ambient><diffuse>0 0 0 0</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>
"""

  def _make_safety_cylinder_sdf(self, radius, height=2.5):
    return f"""
<sdf version="1.6">
  <model name="safety_pillar">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{height}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0 0 0 0</ambient>
          <diffuse>0 0 0 0</diffuse>
          <specular>0 0 0 0</specular>
          <emissive>0 0 0 0</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
  def _make_cylinder_sdf(self, radius, height):
    return f"""
<sdf version="1.6">
  <model name="rand_cylinder">
    <static>true</static>
    <link name="link">
    <gravity>0</gravity>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{height}</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{height}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>1 0.5 0 1</ambient>
          <diffuse>1 0.5 0 1</diffuse>
        </material>
      </visual>
    </link> 
  </model>
</sdf>
"""

  def _random_xy(self, is_drone=True):
      # If it's the drone, pull the boundary in by 0.5 meters
      # If it's an obstacle, use the full area size
      buffer = 0.5 if is_drone else 0.0
      
      limit = (self.flight_boundary - buffer) if is_drone else self.area_size
      
      return (random.uniform(-limit, limit),
              random.uniform(-limit, limit))


  def _make_boundaries_sdf(self):
    fb = self.flight_boundary
    az = self.area_size
    thickness = 0.05  # 5 cm thick lines
    
    return f"""
<sdf version="1.6">
  <model name="cyberzoo_boundaries">
    <static>true</static>
    <link name="link">
      
      <visual name="fb_top">
        <pose>0 {fb} 0.001 0 0 0</pose>
        <geometry><box><size>{fb*2} {thickness} 0.002</size></box></geometry>
        <material><ambient>0 1 0 1</ambient><diffuse>0 1 0 1</diffuse></material>
      </visual>
      <visual name="fb_bottom">
        <pose>0 {-fb} 0.001 0 0 0</pose>
        <geometry><box><size>{fb*2} {thickness} 0.002</size></box></geometry>
        <material><ambient>0 1 0 1</ambient><diffuse>0 1 0 1</diffuse></material>
      </visual>
      <visual name="fb_left">
        <pose>{-fb} 0 0.001 0 0 0</pose>
        <geometry><box><size>{thickness} {fb*2} 0.002</size></box></geometry>
        <material><ambient>0 1 0 1</ambient><diffuse>0 1 0 1</diffuse></material>
      </visual>
      <visual name="fb_right">
        <pose>{fb} 0 0.001 0 0 0</pose>
        <geometry><box><size>{thickness} {fb*2} 0.002</size></box></geometry>
        <material><ambient>0 1 0 1</ambient><diffuse>0 1 0 1</diffuse></material>
      </visual>

      <visual name="az_top">
        <pose>0 {az} 0.001 0 0 0</pose>
        <geometry><box><size>{az*2} {thickness} 0.002</size></box></geometry>
        <material><ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse></material>
      </visual>
      <visual name="az_bottom">
        <pose>0 {-az} 0.001 0 0 0</pose>
        <geometry><box><size>{az*2} {thickness} 0.002</size></box></geometry>
        <material><ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse></material>
      </visual>
      <visual name="az_left">
        <pose>{-az} 0 0.001 0 0 0</pose>
        <geometry><box><size>{thickness} {az*2} 0.002</size></box></geometry>
        <material><ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse></material>
      </visual>
      <visual name="az_right">
        <pose>{az} 0 0.001 0 0 0</pose>
        <geometry><box><size>{thickness} {az*2} 0.002</size></box></geometry>
        <material><ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse></material>
      </visual>

    </link>
  </model>
</sdf>
"""
  def _load_obstacle_sdfs(self):
    """
    Load SDF XML strings for existing TU Delft models:
    orange_pole, Tree_1, Tree_2, traffic_mat, textured_panel_3m1,
    dronerace_gate, etc.
    """
    sdfs = {}
    pap_home = os.getenv("PAPARAZZI_HOME")
    if not pap_home:
      self.get_logger().warn("PAPARAZZI_HOME not set, using cylinder obstacles.")
      return sdfs

    base = os.path.join(pap_home, "sw/ext/tudelft_gazebo_models/models")
    candidates = [
        ("orange_pole", os.path.join("orange_pole", "orange_pole.sdf")),
        ("plant_1", os.path.join("plant_1", "plant_1.sdf")),
        ("plant_2", os.path.join("plant_2", "plant_2.sdf")),
        ("plant_3", os.path.join("plant_3", "plant_3.sdf")),
        #("Tree_1", os.path.join("Tree_1", "Tree_1.sdf")),
        #("Tree_2", os.path.join("Tree_2", "Tree_2.sdf")),
        ("traffic_mat", os.path.join("traffic_mat", "traffic_mat.sdf")),
        ("airport_mat", os.path.join("airport_mat", "airport_mat.sdf")),
        ("racing_gate", os.path.join("racing_gate", "model.sdf")),
        ("blackpanel", os.path.join("blackpanel", "model.sdf")),
        #("textured_panel_3m",
        # os.path.join("textured_panel_3m1", "textured_panel_3m.sdf")),
        #("dronerace_gate",
        # os.path.join("dronerace_gate", "dronerace_gate.sdf")),
    ]

    for key, rel in candidates:
      path = os.path.join(base, rel)
      try:
        with open(path, "r") as f:
          sdfs[key] = f.read()
      except OSError:
        # Just skip missing models; we'll fall back to cylinder if none found
        continue

    # Also allow loading a custom STL-based gate (mav_gate_final) via env var.
    # Set MAV_GATE_FINAL_STL=/full/path/to/mav_gate_final.stl before launching.
    mav_gate_stl = os.getenv("MAV_GATE_FINAL_STL")
    if mav_gate_stl and os.path.isfile(mav_gate_stl):
      # Wrap the STL mesh in a minimal SDF model so Gazebo can spawn it.
      gate_sdf = f"""
<sdf version="1.6">
  <model name="mav_gate_final">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <mesh>
            <uri>file://{mav_gate_stl}</uri>
          </mesh>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <mesh>
            <uri>file://{mav_gate_stl}</uri>
          </mesh>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>
"""
      sdfs["mav_gate_final"] = gate_sdf
    elif mav_gate_stl:
      self.get_logger().warn(
          f"MAV_GATE_FINAL_STL is set but file not found: {mav_gate_stl}")

    if not sdfs:
      self.get_logger().warn(
          "No TU Delft obstacle SDFs could be loaded; using simple cylinders.")
    else:
      self.get_logger().info(
          "Loaded obstacle models: " + ", ".join(sorted(sdfs.keys())))
    return sdfs

  # Helper method to safely wait for services without locking the executor
  def _wait_for_future(self, future, timeout_sec):
    start = time.time()
    while rclpy.ok() and not future.done():
      if time.time() - start > timeout_sec:
        break
      time.sleep(0.01)
    return future.done()

  def _sample_obstacles(self, drone_xy, count):
      obstacles = []
      attempts = 0
      while len(obstacles) < count and attempts < count * 50:
        attempts += 1
        x, y = self._random_xy()
        if math.hypot(x - drone_xy[0], y - drone_xy[1]) < self.drone_min_clearance:
          continue
        good = True
        for ox, oy in obstacles:
          if math.hypot(x - ox, y - oy) < self.obstacle_min_spacing:
            good = False
            break
        if good:
          obstacles.append((x, y))
      return obstacles

  def _generate_scenario_objects(self, drone_xy, num_poles, num_plants, num_gates):
    """Creates a list of items to spawn with coords, yaws, and tags."""
    spawn_pool = []
    
    # 1. Build the list of what we want
    if 'orange_pole' in self.obstacle_sdfs:
        spawn_pool.extend([('orange_pole', 'pole')] * num_poles)
    
    plant_keys = [k for k in self.obstacle_sdfs.keys() if 'plant' in k.lower()]
    for _ in range(num_plants):
        spawn_pool.append((random.choice(plant_keys), 'pole'))
        
    mat_keys = [k for k in self.obstacle_sdfs.keys() if 'mat' in k.lower()]
    if mat_keys:
        spawn_pool.append((random.choice(mat_keys), 'mat'))
    
    spawn_pool.extend([('racing_gate', 'gate')] * num_gates)
    
    if 'blackpanel' in self.obstacle_sdfs:
        spawn_pool.append(('blackpanel', 'wall'))

      
    # Shuffle to randomize placement order
    random.shuffle(spawn_pool)

    # 2. Assign coordinates and yaws
    obstacles = []
    attempts = 0
    for key, obj_type in spawn_pool:
      placed = False
      while not placed and attempts < 1000:
        attempts += 1
        x, y = self._random_xy(is_drone=False)
        
        # Don't spawn on the drone
        if math.hypot(x - drone_xy[0], y - drone_xy[1]) < (self.obstacle_radius + self.drone_min_clearance):
          continue
          
        # Don't spawn on other objects
        good = True
        for ox, oy, _, _, _ in obstacles:
          if math.hypot(x - ox, y - oy) < self.obstacle_min_spacing:
            good = False
            break
            
        if good:
          yaw = random.uniform(-math.pi, math.pi)
          obstacles.append((x, y, yaw, key, obj_type))
          placed = True
          
    return obstacles

  def _spawn_obstacles(self, obstacles):
    ghost_xml = self._make_safety_cylinder_sdf(self.obstacle_radius)

    for i, (x, y, obs_yaw, key, obj_type) in enumerate(obstacles):
      # --- 1. Spawn the Real Visual Model ---
      name = f"rand_obstacle_{i}"
      req = SpawnEntity.Request()
      req.name = name
      req.xml = self.obstacle_sdfs[key]
      req.initial_pose = Pose()
      req.initial_pose.position = Point(x=float(x), y=float(y), z=0.01)
      req.initial_pose.orientation.z = math.sin(obs_yaw / 2.0)
      req.initial_pose.orientation.w = math.cos(obs_yaw / 2.0)
      
      # SYNC CALL: Wait for Gazebo to finish this one
      future = self.spawn_cli.call_async(req)
      self._wait_for_future(future, 1.0)

      # --- 2. Spawn the Mathematical "Ghost" Cylinders ---
      if obj_type == "mat": continue 

# Update inside _spawn_obstacles
      ghost_parts = []
      if obj_type == "gate":
          offset = 1.0 
          # Swapped sin/cos to rotate leg placement by 90 degrees
          ghost_parts = [
              (x + math.cos(obs_yaw) * offset, y + math.sin(obs_yaw) * offset),
              (x - math.cos(obs_yaw) * offset, y - math.sin(obs_yaw) * offset)
          ]
      elif obj_type == "wall":
          offset = 0.30 
          # Swapped sin/cos here as well for the blackpanel
          ghost_parts = [
              (x, y), 
              (x + math.cos(obs_yaw) * offset, y + math.sin(obs_yaw) * offset),
              (x - math.cos(obs_yaw) * offset, y - math.sin(obs_yaw) * offset)
          ]
      else:
          ghost_parts = [(x, y)]

      for j, (px, py) in enumerate(ghost_parts):
          g_req = SpawnEntity.Request()
          g_req.name = f"safety_ghost_{i}_{j}"
          g_req.xml = ghost_xml
          g_req.initial_pose = Pose()
          g_req.initial_pose.position = Point(x=float(px), y=float(py), z=1.25)
          
          # SYNC CALL: Wait for Gazebo to finish this one
          g_future = self.spawn_cli.call_async(g_req)
          self._wait_for_future(g_future, 1.0)


  def _delete_obstacles(self):
    self.get_logger().info("Exorcising ghosts and cleaning the Zoo...")
    
    # We loop through a high range (50) to catch any leftovers from previous crashes
    for i in range(50):
        # 1. Delete the real obstacle
        req = DeleteEntity.Request(name=f"rand_obstacle_{i}")
        future = self.delete_cli.call_async(req)
        # We don't need a long wait, just enough to let the service process
        self._wait_for_future(future, 0.15)
        time.sleep(0.1)  # Small delay to help Gazebo catch up
        
        # 2. Delete all possible ghost parts (up to 4 per obstacle)
        for j in range(4):
            g_req = DeleteEntity.Request(name=f"safety_ghost_{i}_{j}")
            g_future = self.delete_cli.call_async(g_req)
            self._wait_for_future(g_future, 0.15)
            time.sleep(0.1)

    # Give Gazebo a full second to clear its internal physics cache
    time.sleep(1.0)

  def _set_drone_pose(self, x, y, z, roll, pitch, yaw):
    """Sets the drone pose with full 3D orientation randomization."""
    # Euler to Quaternion conversion (Order: Roll -> Pitch -> Yaw)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    state = EntityState()
    state.name = self.drone_model_name
    state.pose.position = Point(x=float(x), y=float(y), z=float(z))
    state.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

    req = SetEntityState.Request()
    req.state = state
    
    # We use a non-blocking call but wait internally to ensure Gazebo moves the camera
    future = self.set_state_cli.call_async(req)
    self._wait_for_future(future, 1.0)

  def _capture_image(self, timeout=5.0):
    """
    Wait for a *new* image after we moved the camera.
    """
    # Remember the previous message pointer so we can detect a fresh frame.
    old_msg = self.last_image
    start = time.time()
    while rclpy.ok() and (time.time() - start) < timeout:
      if self.last_image is not None and self.last_image is not old_msg:
        break
      time.sleep(0.01)

    if self.last_image is None or self.last_image is old_msg:
      return None

    try:
      return self.bridge.imgmsg_to_cv2(self.last_image,
                                       desired_encoding="bgr8")
    except Exception as e:
      self.get_logger().error(f"cv_bridge error: {e}")
      return None

  def _normalize_angle(self, a):
    while a > math.pi:
      a -= 2 * math.pi
    while a < -math.pi:
      a += 2 * math.pi
    return a

  def _normalize_interval(self, a, b):
    a = self._normalize_angle(a)
    b = self._normalize_angle(b)
    if a <= b:
      return [(a, b)]
    else:
      return [(-math.pi, b), (a, math.pi)]

  def _angle_in_interval(self, x, a, b):
    x = self._normalize_angle(x)
    return a <= x <= b


  def _compute_yaw_label(self, drone_xy, drone_yaw, obstacles):
    blocked = []
    r = self.obstacle_radius + 0.15 # 15cm safety margin

    # ---------------------------------------------------------
    # 1. OBSTACLE AVOIDANCE (Standard objects < 3.0m)
    # ---------------------------------------------------------
    for (ox, oy, obs_yaw, key, obj_type) in obstacles:
      if obj_type == "mat": continue
      
      parts = []
      if obj_type == "gate": 
          offset = 0.90
          parts = [
              (ox + math.cos(obs_yaw) * offset, oy + math.sin(obs_yaw) * offset),
              (ox - math.cos(obs_yaw) * offset, oy - math.sin(obs_yaw) * offset)
          ]
      elif obj_type == "wall":
          offset = 0.4
          parts = [
              (ox, oy), 
              (ox + math.cos(obs_yaw) * offset, oy + math.sin(obs_yaw) * offset),
              (ox - math.cos(obs_yaw) * offset, oy - math.sin(obs_yaw) * offset)
          ]
      else:
          parts = [(ox, oy)]

      for px, py in parts:
          dx = px - drone_xy[0]
          dy = py - drone_xy[1]

          dx_b = math.cos(-drone_yaw) * dx - math.sin(-drone_yaw) * dy
          dy_b = math.sin(-drone_yaw) * dx + math.cos(-drone_yaw) * dy

          if dx_b <= 0 or math.hypot(dx_b, dy_b) > 2.5:
            continue

          dist = math.hypot(dx_b, dy_b)
          theta = math.atan2(dy_b, dx_b)
          alpha = math.pi / 2.0 if dist <= r else math.asin(r / dist)
          blocked.append((self._normalize_angle(theta - alpha), self._normalize_angle(theta + alpha)))

    # ---------------------------------------------------------
    # 2. VIRTUAL BOUNDARY WALLS & 1.25m PONG DEFLECTION
    # ---------------------------------------------------------
    fb = self.flight_boundary
    panic_yaw = None
    min_panic_dist = float('inf')
    
    walls = [
        (fb - drone_xy[0], 0.0),             
        (drone_xy[0] - (-fb), math.pi),      
        (fb - drone_xy[1], math.pi / 2.0),   
        (drone_xy[1] - (-fb), -math.pi / 2.0)
    ]

    for dist_to_wall, wall_angle_world in walls:
        rel_wall_angle = self._normalize_angle(wall_angle_world - drone_yaw)
        
        # PANIC CHECK: Is the wall in front of us (< 60 deg) AND closer than 1.25m?
        if abs(rel_wall_angle) < (math.pi / 3.0) and dist_to_wall < 1.25:
            # In case of corners, only panic based on the closest wall
            if dist_to_wall < min_panic_dist:
                min_panic_dist = dist_to_wall
                # THE FIX: If wall is slightly left (> 0), turn hard RIGHT (-1.57).
                # If wall is slightly right or dead center (<= 0), turn hard LEFT (1.57).
                panic_yaw = -1.57 if rel_wall_angle > 0 else 1.57

        # Standard wall repulsion logic (< 1.5m)
        if dist_to_wall < 1.5: 
            if abs(rel_wall_angle) < (math.pi / 2.0):
                safe_dist = max(0.1, dist_to_wall)
                block_width = (math.pi / 2.0) * (1.0 - (safe_dist / 1.5))
                blocked.append((self._normalize_angle(rel_wall_angle - block_width), 
                                self._normalize_angle(rel_wall_angle + block_width)))

    # ---------------------------------------------------------
    # 3. INTERVAL MERGING
    # ---------------------------------------------------------
    intervals = []
    for s, e in blocked:
        if s <= e: intervals.append((s, e))
        else: intervals.extend([(s, math.pi), (-math.pi, e)])
    
    intervals.sort(key=lambda x: x[0])
    
    merged = []
    if intervals:
        curr_s, curr_e = intervals[0]
        for s, e in intervals[1:]:
          if s <= curr_e: 
            curr_e = max(curr_e, e)
          else:
            merged.append((curr_s, curr_e))
            curr_s, curr_e = s, e
        merged.append((curr_s, curr_e))

    def is_angle_blocked(angle):
        norm_angle = self._normalize_angle(angle)
        for s, e in merged:
            if s <= norm_angle <= e:
                return True
        return False

    max_steering = 1.05       # 60 degrees (Normal FOV)

    # ---------------------------------------------------------
    # 4. DECISION HIERARCHY
    # ---------------------------------------------------------
    
    # GOAL A: 1.25m WALL PANIC! (Overrides absolutely everything)
    if panic_yaw is not None:
        return panic_yaw

    # GOAL B: Target Gate
    desired_heading = None
    for (ox, oy, obs_yaw, key, obj_type) in obstacles:
        if obj_type == "gate":
            dx = ox - drone_xy[0]
            dy = oy - drone_xy[1]
            dist_to_gate = math.hypot(dx, dy)
            angle_to_gate = self._normalize_angle(math.atan2(dy, dx) - drone_yaw)
            
            if 1.0 < dist_to_gate < 5.0 and abs(angle_to_gate) < (math.pi / 6.0):
                if not is_angle_blocked(angle_to_gate):
                    desired_heading = angle_to_gate
                    break

    if desired_heading is not None:
        return max(min(desired_heading, max_steering), -max_steering)

    # GOAL C: Fly Straight
    if not is_angle_blocked(0.0):
        return 0.0

    # GOAL D: Standard Dodge (Within 60-deg FOV)
    candidates = []
    push_margin = 0.15 
    for s, e in merged:
        c1 = self._normalize_angle(s - push_margin)
        c2 = self._normalize_angle(e + push_margin)
        if not is_angle_blocked(c1) and abs(c1) <= max_steering: candidates.append(c1)
        if not is_angle_blocked(c2) and abs(c2) <= max_steering: candidates.append(c2)

    if candidates:
        candidates.sort(key=lambda x: abs(x))
        return candidates[0]

    # Extreme edge cases (If normal FOV is entirely blocked but it's not a wall)
    if not is_angle_blocked(max_steering):
        return max_steering
    if not is_angle_blocked(-max_steering):
        return -max_steering

    return 0.0 # Absolute fallback
  def _run_episodes(self):
    time.sleep(2.0)

    self._delete_obstacles()
    
# --- MASTER CLEAN (Only happens once at startup) ---
    self.get_logger().info("Performing initial cleanup...")
    self.delete_cli.call_async(DeleteEntity.Request(name="target_indicator"))
    self.delete_cli.call_async(DeleteEntity.Request(name="cyberzoo_visual_boundaries"))
    time.sleep(1.0)


        # 1. Spawn Boundaries
    self.get_logger().info("Spawning visual boundaries...")
    # ... (your existing boundary spawn code)

    # 2. Spawn Target Indicator (The Arrow)
    self.get_logger().info("Spawning target_indicator...")
    req = SpawnEntity.Request()
    req.name = "target_indicator"
    req.xml = self._make_arrow_sdf()
    req.initial_pose = Pose()
    req.initial_pose.position = Point(x=0.0, y=0.0, z=-5.0) # Hide it
    req.reference_frame = "world"
    
    future = self.spawn_cli.call_async(req)
    # CRITICAL: Wait for the future to complete before starting steps
    self._wait_for_future(future, 5.0) 
    
    self.get_logger().info("Target indicator ready. Starting episodes...")

    while rclpy.ok() and self.running and self.current_folder_idx < self.num_dataset_folders:
        self._step()
        time.sleep(0.01)


  def _draw_top_down_map(self, drone_xy, drone_yaw, obstacles, yaw_label, episode):
      img_size = 1000
      # Create a view limit that is larger than the area_size to provide padding
      view_limit = self.area_size + 2.0 
      
      canvas = np.full((img_size, img_size, 3), 255, dtype=np.uint8)
      # Pixels per meter based on the new, larger view
      ppm = img_size / (2.0 * view_limit) 
      
      def world_to_pix(x, y):
        # Center the map using view_limit instead of area_size
        px = int((x + view_limit) * ppm)
        py = int((view_limit - y) * ppm)
        return px, py

      # --- Draw The "Physical World" Boundary (Black) ---
      # This represents the absolute limit of your Gazebo world
      w_tl = world_to_pix(-self.area_size, self.area_size)
      w_br = world_to_pix(self.area_size, -self.area_size)
      cv2.rectangle(canvas, w_tl, w_br, (200, 200, 200), 2) # Light gray world box

      # --- Draw Safe Zone (Green) ---
      limit = self.flight_boundary
      tl = world_to_pix(-limit, limit)
      br = world_to_pix(limit, -limit)
      cv2.rectangle(canvas, tl, br, (0, 255, 0), 2)

      # --- Draw Obstacles ---
      obs_radius_px = int(self.obstacle_radius * ppm)
      for ox, oy, obs_yaw, key, obj_type in obstacles:
        px, py = world_to_pix(ox, oy)
        
        if obj_type == "mat":
            # Mats are flat; let's draw them as dark gray squares
            rect_size = int(0.5 * ppm)
            cv2.rectangle(canvas, (px-rect_size, py-rect_size), 
                                 (px+rect_size, py+rect_size), (100, 100, 100), -1)
        elif obj_type == "gate":
            # Swapping sin/cos here as well to match your 90-degree gate fix!
            gx1 = px - int(math.cos(obs_yaw) * 0.8 * ppm)
            gy1 = py - int(math.sin(obs_yaw) * 0.8 * ppm)
            gx2 = px + int(math.cos(obs_yaw) * 0.8 * ppm)
            gy2 = py + int(math.sin(obs_yaw) * 0.8 * ppm)
            cv2.line(canvas, (gx1, gy1), (gx2, gy2), (0, 165, 255), 6) # Thick Orange
        else:
            # Poles/Plants
            cv2.circle(canvas, (px, py), obs_radius_px, (150, 150, 255), -1)
            cv2.circle(canvas, (px, py), 3, (0, 0, 255), -1)

      # --- Draw Drone (Red Dot) ---
      dx, dy = drone_xy
      dpx, dpy = world_to_pix(dx, dy)
      cv2.circle(canvas, (dpx, dpy), 8, (0, 0, 255), -1)
      
      # --- Draw Current Heading (Blue Line) ---
      # Reduced to 0.8m so it doesn't clutter the view
      hpx, hpy = world_to_pix(dx + 0.8 * math.cos(drone_yaw), dy + 0.8 * math.sin(drone_yaw))
      cv2.line(canvas, (dpx, dpy), (hpx, hpy), (255, 0, 0), 2)
      
      # --- Draw Calculated Decision (Bright Green Arrow) ---
      # This represents the YAW_LABEL
      target_x = dx + 1.5 * math.cos(drone_yaw + yaw_label)
      target_y = dy + 1.5 * math.sin(drone_yaw + yaw_label)
      opx, opy = world_to_pix(target_x, target_y)
      cv2.arrowedLine(canvas, (dpx, dpy), (opx, opy), (0, 255, 0), 4, tipLength=0.3)

      # Add some text for context
      cv2.putText(canvas, f"Episode: {episode} | Yaw Label: {yaw_label:.2f}", 
                  (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

      cv2.imwrite(os.path.join(self.output_dir, f"episode_{episode:05d}_map.png"), canvas)
  def _pre_spawn_obstacle_pool(self):
      model_keys = list(self.obstacle_sdfs.keys())
      for i in range(self.max_obstacles):
        name = f"rand_obstacle_{i}"
        req = SpawnEntity.Request()
        req.name = name

        # Choose the XML string
        xml_content = ""
        if model_keys:
          key = random.choice(model_keys)
          xml_content = self.obstacle_sdfs[key]
        else:
          xml_content = self.cylinder_sdf

        # Check if the XML is empty before sending!
        if not xml_content or len(xml_content.strip()) == 0:
            self.get_logger().error(f"ABORTING spawn for {name}: XML string is EMPTY. Check your SDF files.")
            continue

        req.xml = xml_content
        req.robot_namespace = ""
        req.initial_pose = Pose()
        req.initial_pose.position = Point(x=0.0, y=0.0, z=-10.0) 
        req.reference_frame = "world"

        self.get_logger().info(f"Requesting spawn for {name}...")
        future = self.spawn_cli.call_async(req)
        self._wait_for_future(future, 5.0) 
  def _teleport_obstacles(self, obstacles):
    for i in range(self.max_obstacles):
      name = f"rand_obstacle_{i}"
      state = EntityState()
      state.name = name
      state.pose = Pose()
      
      if i < len(obstacles):
        x, y = obstacles[i]
        z = 0.01 if self.obstacle_sdfs else self.obstacle_radius
        state.pose.position = Point(x=float(x), y=float(y), z=float(z))
        
        # RANDOM YAW FOR TELEPORT
        qx, qy, qz, qw = get_random_quaternion()
        state.pose.orientation.x = qx
        state.pose.orientation.y = qy
        state.pose.orientation.z = qz
        state.pose.orientation.w = qw
      else:
        # Hide unused obstacles
        state.pose.position = Point(x=0.0, y=0.0, z=-10.0)
        state.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

      req = SetEntityState.Request()
      req.state = state
      future = self.set_state_cli.call_async(req)
      self._wait_for_future(future, 0.5)
  def _step(self):
    if not self.running or self.current_folder_idx >= self.num_dataset_folders:
        return

    # --- A. FOLDER & SCENARIO MANAGEMENT ---
    folder_path, csv_path = self._get_current_paths()

    # Refresh objects every 250 images
    if self.current_img_in_scenario == 0:
        self.get_logger().info(f"Scenario Refresh: Folder {self.current_folder_idx}, Scenario Start.")
        self._delete_obstacles()
        
        # Adjust the mix here
        num_poles = random.randint(*self.pole_range)
        num_plants = random.randint(*self.plant_range)
        num_gates = random.randint(*self.gate_range)
        
        self.current_obstacles = self._generate_scenario_objects((999.0, 999.0), num_poles, num_plants, num_gates)
        self._spawn_obstacles(self.current_obstacles)
        time.sleep(1.2) # Give Gazebo physics time to settle

    # --- B. POSITION & ORIENTATION RANDOMIZATION ---
    valid_pose = False
    attempts = 0
    dx, dy = 0.0, 0.0
    
    while not valid_pose and attempts < 100:
        attempts += 1
        dx, dy = self._random_xy(is_drone=True)
        conflict = False
        for ox, oy, _, _, _ in self.current_obstacles:
            if math.hypot(dx - ox, dy - oy) < (self.obstacle_radius + self.drone_min_clearance):
                conflict = True
                break
        if not conflict:
            valid_pose = True

    if not valid_pose: return

    # Randomize ALL axes for robustness
    yaw = random.uniform(-math.pi, math.pi)
    pitch = random.uniform(*self.pitch_range)
    roll = random.uniform(*self.roll_range)
    alt = random.uniform(*self.alt_range)

    self._set_drone_pose(dx, dy, alt, roll, pitch, yaw)
    time.sleep(0.4) # Wait for camera buffer

    # --- C. CAPTURE & BALANCING ---
    img = self._capture_image()
    if img is None: return

    yaw_label = self._compute_yaw_label((dx, dy), yaw, self.current_obstacles)
    
    # DATA BALANCING: If flying straight, skip 85% of the time.
    # This prevents the "Straight Ahead Bias" during training.
    if abs(yaw_label) < 0.05 and random.random() < 0.85:
        return

    # --- D. SAVE TO SUBFOLDER ---
    img_name = f"img_{self.current_img_in_folder:04d}.png"
    full_img_path = os.path.join(folder_path, img_name)
    cv2.imwrite(full_img_path, img)

    # Calculate bin for the 23-class architecture
    yaw_bin = discretize_yaw(yaw_label)

    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([img_name, yaw_label, yaw_bin])

    # --- E. VISUALIZATION & COUNTERS ---
    #self._draw_top_down_map((dx, dy), yaw, self.current_obstacles, yaw_label, self.total_images_saved)
    
    self.current_img_in_folder += 1
    self.current_img_in_scenario += 1
    self.total_images_saved += 1

    # Check for Scenario switch (Scenario 1 is 0-249, Scenario 2 is 250-499)
    if self.current_img_in_scenario >= self.images_per_scenario:
        self.current_img_in_scenario = 0

    # Check for Folder switch
    if self.current_img_in_folder >= self.images_per_folder:
        self.get_logger().info(f"Folder {self.current_folder_idx} Complete. Moving to next.")
        self.current_folder_idx += 1
        self.current_img_in_folder = 0


def main(args=None):
  rclpy.init(args=args)
  node = CyberzooDatasetNode()
  
  # MultiThreadedExecutor allows the background thread to 
  # receive service responses from Gazebo without blocking.
  executor = rclpy.executors.MultiThreadedExecutor()
  executor.add_node(node)
  
  try:
    executor.spin()
  except KeyboardInterrupt:
    pass
  finally:
    node.running = False
    node.generation_thread.join(timeout=1.0)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
  main()