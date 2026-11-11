# ufld_line_pkg

ROS 2 Humble lane node intended to replace `line_pkg` without changing the
control interface.

## Interfaces

```text
subscribe:
  /camera/image/compressed      sensor_msgs/msg/CompressedImage
  /mission_state                std_msgs/msg/Int32

publish:
  /line_offset                  std_msgs/msg/Float32
  /yellow_detected              std_msgs/msg/Bool
  /ufld_line/debug_image/compressed
```

`/line_offset` is kept compatible with the existing `control_pkg`.

## Model Files

Put converted TNN files here:

```text
models/ufld_320x160.tnnproto
models/ufld_320x160.tnnmodel
```

The default launch assumes a 320x160 camera/model input.

## Build Without TNN

This only verifies the ROS package shape. The node will start but cannot infer.

```bash
cd /home/topst/D-Racer-Kit
source /opt/ros/humble/setup.bash
colcon build --packages-select ufld_line_pkg
```

## Build With TNN OpenCL

After building TNN on the D3-G board:

```bash
cd /home/topst/D-Racer-Kit
source /opt/ros/humble/setup.bash
colcon build --packages-select ufld_line_pkg --cmake-args \
  -DUFLD_USE_TNN=ON \
  -DTNN_ROOT=/home/topst/TNN \
  -DTNN_LIB_DIR=/home/topst/TNN/benchmark/benchmark_armlinux/build
```

Adjust `TNN_LIB_DIR` to the directory that contains `libTNN.so`.

## Run

```bash
source /opt/ros/humble/setup.bash
source /home/topst/D-Racer-Kit/install/setup.bash
ros2 launch ufld_line_pkg ufld_line.launch
```

This is meant to replace:

```bash
ros2 launch line_pkg line.launch
```
