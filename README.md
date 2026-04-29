
***

## 📦 Installation & Building

### Prerequisites
* **ROS 2** (Humble) installed and sourced.
* **Habitat-Sim** installed in your active Python environment (often done via Conda).

```bash
conda create -n habitat python=3.10
conda activate habitat
conda install habitat-sim withbullet -c conda-forge -c aihabitat-nightly
```

**1. Create a ROS 2 Workspace**
If you don't already have a workspace set up, create one and navigate into its `src` directory:
```bash
mkdir -p ~/habitat_ws
cd ~/habitat_ws
```

**2. Clone the Repository**
Clone this package into your workspace's `src` folder:
```bash
git clone https://github.com/lrm20011/habitat-ros2.git
colcon build --symlink-install
```


**3. Launch**
Finally, source the built workspace so ROS 2 can find your new executable and launch files. You will need to run this command in every new terminal you open:
```bash
source ./install/local_setup.bash
ros2 launch habitat_ros2 teleop_rviz.launch.py
```
It supports to run the simulator in computer A, and deploy algorithms in computer B and link them with ethernet. To do that, you only need to set habitat.compress_image = true, in computer B transfer compressed image to raw image and 
```bash
export export ROS_LOCALHOST_ONLY=0
```
in both computers

## ⚙️ Configuration & Parameters

You can modify parameters in config/habitat.yaml

### Node: `habitat_node`

| Parameter | Type | Default Value | Description |
| :--- | :---: | :--- | :--- |
| **`habitat.width`** | *Integer* | `800` | The width (in pixels) of the rendered sensor images (RGB, Depth, Semantic). |
| **`habitat.height`** | *Integer* | `480` | The height (in pixels) of the rendered sensor images. |
| **`habitat.f`** | *Float* | `525.0` | The focal length of the simulated camera, which dictates the Field of View (FOV). |
| **`habitat.fps`** | *Integer* | `40` | The target frames-per-second for the simulation loop. |
| **`habitat.compress_image`** | *Boolean* | `true` | If enabled, publishes compressed images to save ROS network bandwidth. Only use it when you transfer msg between two computers|
| **`habitat.skip_image`** | *Integer* | `2` | Skips publishing images every *N* frames. |
| **`habitat.enable_semantics`** | *Boolean* | `false` | Enables the semantic segmentation sensor. *Note: The scene file must contain semantic annotations for this to work.* |
| **`habitat.depth_noise`** | *Boolean* | `false` | Injects realistic noise (typically a Redwood noise model) into the depth sensor output to simulate real-world camera inaccuracies. |
| **`habitat.visualize_semantics`** | *Boolean* | `false` | If true, applies a color palette to the semantic sensor output so it can be easily interpreted by humans in RViz. |
| **`habitat.scene_file`** | *String* | `"/home/iot/habitat_ros2/00853-5cdEh9F2hJL/5cdEh9F2hJL.glb"` | The absolute path to the 3D environment file you want to load (e.g., `.glb`, `.gltf`, or `.ply`). |
| **`habitat.pose_frame_id`** | *String* | `"world"` | The name of the root/global TF coordinate frame in ROS 2. |
| **`habitat.pose_frame_at_initial_T_HB`**| *Boolean* | `false` | If true, sets the origin of the `pose_frame_id` (e.g., "world") to exactly match the agent's initial spawn location. If false, it uses the scene's default origin. |
| **`habitat.recording_dir`** | *String* | `""` | Directory to save recorded simulation data. Leave empty (`""`) to disable recording. |
| **`allowed_classes`** | *List* | `[]` | An optional array of semantic class IDs to filter. If left empty, all available classes are published. |
| **`initial_T_HB`** | *List* | `[]` | A 4x4 transformation matrix (or position/quaternion array depending on implementation) defining the agent's starting pose. Leave empty to spawn at the scene's default navigable location. |
