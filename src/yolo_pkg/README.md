# yolo_pkg for TOPST D3-G

This package contains only the runtime components required for:

- YOLO26n traffic-light detection with ONNX Runtime CPUExecutionProvider
- green/red Boolean ROS 2 topics
- OpenCV ArUco detection (`DICT_6X6_50`, target ID 3 by default)
- D3-G main-CPU affinity (`taskset -c 2,3`)

The model is fixed at `1x3x320x320`; output is `1x300x6`. Classes are
`0=green_light`, `1=red_light`.

The bundled Ultralytics model declares the AGPL-3.0 license in its ONNX
metadata. Confirm that this is compatible with the intended distribution.

## CPU core allocation

The D3-G main CPU has four cores. Linux CPU numbers are zero-based:

| Team notation | Linux CPU |
|---|---:|
| Main core 1 | CPU 0 |
| Main core 2 | CPU 1 |
| Main core 3 | CPU 2 |
| Main core 4 | CPU 3 |

`yolo.launch` runs YOLO with:

```text
taskset -c 2,3
```

Therefore YOLO is restricted to main CPU cores 3 and 4. ONNX Runtime uses two
intra-op threads and one inter-op thread. The sub CPU is not explicitly used
by this package.

The other nodes in the current `src` do not have explicit CPU affinity
settings. Linux may therefore schedule camera, control, mission, ROS/DDS and
UFLD host-side work on any CPU allowed by the system. `taskset -c 2,3` limits
YOLO to CPUs 2 and 3, but does not prevent other processes from also using
those CPUs; it is not exclusive core reservation.

UFLD neural-network inference remains on the PowerVR GPU through TNN OpenCL.
UFLD still uses CPU time for JPEG decoding, resize, GPU transfers,
postprocessing, BEV and ROS publication. Its `cpu_threads` parameter is only
applied to TNN CPU inference when `use_opencl=false`.

Check the actual affinity on the board:

```bash
taskset -pc "$(pgrep -f yolo_node_ort.py | head -1)"
taskset -pc "$(pgrep -f ufld_line_node | head -1)"
taskset -pc "$(pgrep -f control_node | head -1)"
ps -eLo pid,tid,psr,comm,args
```

## Important merge rule

Do not blindly replace a newer `src/yolo_pkg`. Back it up and compare its
topics, mission logic, QoS, class IDs, ArUco dictionary/ID and launch
parameters. The Python node in this package should be treated as the inference
implementation to merge into the newer tree.

## Runtime dependencies

ROS 2 Humble with `rclpy`, `sensor_msgs`, `std_msgs`, NumPy, OpenCV including
the `aruco` module, and ONNX Runtime for AArch64 are required.

Check before building:

```bash
python3 -c "import cv2, numpy, onnxruntime; print(cv2.__version__, onnxruntime.__version__); print(hasattr(cv2, 'aruco'))"
```

## Build and run

Copy this directory to the target workspace as `src/yolo_pkg`. If a package
with that name already exists, merge it first instead of keeping two copies.

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select yolo_pkg
source install/setup.bash
ros2 launch yolo_pkg yolo.launch
```

Check:

```bash
ros2 topic hz /camera/image/compressed
ros2 topic echo /traffic_light_green
ros2 topic echo /traffic_light_red
ros2 topic echo /aruco_flag
pgrep -af yolo_node_ort.py
taskset -pc "$(pgrep -f yolo_node_ort.py | head -1)"
```

The affinity should report CPUs `2,3`, corresponding to the confirmed main
CPU cores 3 and 4. The node configures ONNX Runtime for two intra-op threads
and one inter-op thread. UFLD should remain on TNN OpenCL GPU. The current UFLD
`cpu_threads` parameter does not create CPU inference workers while
`use_opencl=true`, although JPEG decode, resize and postprocessing still use
CPU.

Leave both YOLO and UFLD debug-image publication disabled during the initial
latency test. Measure UFLD latency, `/line_offset` rate, control jitter,
per-core load, temperature and throttling while YOLO is running.

## Model integrity

```text
224351853f2d8947f871f7f394670f2a899e8142c76dfa41d38899bfafc25a75  models/best.onnx
```
