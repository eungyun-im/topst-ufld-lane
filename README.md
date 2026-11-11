<div align="center">

# topst-ufld-lane

**A deep-learning lane detector (UFLD) running on the TOPST D3-G,<br>a Korean automotive SoC board with no NPU, on a small racing car.**

![ROS 2](https://img.shields.io/badge/ROS_2_Humble-22314E?style=flat-square&logo=ros&logoColor=white)
![C++](https://img.shields.io/badge/C++-00599C?style=flat-square&logo=cplusplus&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TNN](https://img.shields.io/badge/TNN-OpenCL-555555?style=flat-square)
![TOPST D3-G](https://img.shields.io/badge/TOPST_D3--G-Telechips_TCC8050-555555?style=flat-square)

SEA:ME Hackathon 2026 (Volkswagen Foundation) · Gold Prize · Team KUUVe

[Problem](#problem) · [Deployment path](#deployment-path) · [Verification](#verification) · [Packages](#packages) · [Run](#running) · [Limitations](#limitations)

</div>

---

## Problem

The car is a TOPST D-Racer. Its computer is a TOPST D3-G: a Telechips TCC8050 SoC with a PowerVR GPU and **no NPU**. The lane detector has to feed a 20 Hz steering controller, while a traffic-light detector shares the same four CPU cores.

A stock UFLD model at full resolution was estimated at about 3 to 6 FPS on this board. That is too slow for the controller, so the model, the input and the runtime were cut down until the lane output kept up with the 20 Hz loop.

## Deployment path

```text
PyTorch ──export──▶ ONNX (fixed 1x3x160x320)
        ──inspect─▶ check output names / shapes, patch launch params
        ──onnx2tnn─▶ .tnnproto / .tnnmodel
        ──TNN OpenCL─▶ PowerVR GPU on the D3-G
```

| Change | Why |
|---|---|
| MobileNetV2 backbone instead of ResNet | Fewer operations on a GPU without tensor cores |
| Input 160 × 320 (from full resolution) | Inference cost scales with pixels |
| Fixed input shape, optimized TNN graph | No dynamic shapes in the OpenCL runtime |
| Inference on the GPU through TNN OpenCL | Leaves the CPU for JPEG decode, BEV and ROS |
| YOLO pinned to CPU 2 and 3 (`taskset`) | Keeps the traffic-light model from starving the lane node |

The lane model is trained on synthetic frames rendered from the known track texture, so labels are generated automatically ([`training/`](src/ufld_line_pkg/training)).

## Verification

The detector feeds steering directly, so a wrong offset is worse than no offset. The checks below are about catching a wrong output before it reaches the controller.

| Check | How | What it catches |
|---|---|---|
| Model and decoder contract | [`inspect_ufld_onnx.py`](src/ufld_line_pkg/tools/inspect_ufld_onnx.py) reads the ONNX output names and shapes and patches the launch parameters | A shape mismatch that makes the decoder return a plausible but wrong offset, with no error |
| Decoder parity | [`validate_decode.py`](src/ufld_line_pkg/training/validate_decode.py) runs a Python port of `ufld_decoder.cpp` on rendered frames before conversion | A model or decoder that is wrong before it ever reaches the board |
| Build without the runtime | `tnn_ufld_runner_stub.cpp` lets the package build and start without TNN | Interface and launch errors found on a PC |
| Fallback | `line_pkg` falls back to the HSV sliding window if TNN fails to initialize | A missing model or driver stopping the car from starting |
| Output plausibility | Fewer than 4 lane points or a short vertical span marks the frame invalid; the offset may move at most 30 px per frame | Single-frame jumps and half-detected lanes |
| Controller reaction | `control_pkg` treats the lane as lost after 3 invalid frames or 0.30 s without data; an invalid command holds steering and sets throttle to 0 | Acting on stale or missing lane data |
| Runtime metrics | `/ufld_debug` publishes inference time, success ratio and step clamps | Slowdowns and quality drops during a run |
| Edge cases | [`edge_case_capture.py`](tools/edge_case_capture.py) saves raw, overlay and BEV frames whenever `/line_valid` drops | Failure frames to inspect and retrain on |

## Packages

Packages written for the hackathon:

| Package | Role |
|---|---|
| [`ufld_line_pkg`](src/ufld_line_pkg) | UFLD lane node (TNN OpenCL), decoder, ONNX inspection, synthetic training |
| [`line_pkg`](src/line_pkg) | Earlier lane node: UFLD with HSV sliding-window fallback |
| [`control_pkg`](src/control_pkg) | Lane-following PID with curvature gain schedule, slew limit, lost-lane handling |
| [`mission_planner_pkg`](src/mission_planner_pkg) | Traffic light, ArUco stop, warning zone and lane-colour missions |
| [`yolo_pkg`](src/yolo_pkg) | Traffic-light detection (ONNX Runtime, CPU) and ArUco |
| [`stopline_pkg`](src/stopline_pkg) | Stop-line detection |
| [`tools/`](tools) | Live viewer, HSV / BEV tuner, edge-case capture |

`camera`, `control`, `joystick`, `monitor`, `battery`, `opencv`, `topst_utils` and the message packages come from the [TOPST D-Racer Kit](https://github.com/topst-development/D-Racer-Kit). Its original README and guides are in [`docs/`](docs).

## Running

On the D3-G, after building TNN with OpenCL:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select ufld_line_pkg --cmake-args \
  -DUFLD_USE_TNN=ON -DTNN_ROOT=$HOME/TNN \
  -DTNN_LIB_DIR=$HOME/TNN/benchmark/benchmark_armlinux/build
source install/setup.bash
ros2 launch ufld_line_pkg ufld_line.launch
```

On a PC, without TNN (build and interface check only):

```bash
colcon build --packages-select ufld_line_pkg
```

## Limitations

- Trained on rendered frames only. Camera height, pitch and field of view in the renderer are assumptions, so there is a sim-to-real gap.
- `training/model.py` builds a ResNet-18 variant; the deployed model file is the MobileNetV2 variant.
- The offset is ill-defined on sharp curves where one lane edge leaves the frame.
- Apart from the kit's lint tests, the lane and control nodes have no automated tests.
- Some launch files contain paths from the team's board (`/home/topst/...`).

## License and authorship

- Based on the TOPST D-Racer Kit, Apache License 2.0 ([`LICENSE`](LICENSE)).
- The YOLO model in `yolo_pkg/models/best.onnx` was trained with Ultralytics and its metadata declares AGPL-3.0.
- Team KUUVe, Kookmin University. AI tools (Claude Code) were used for implementation and analysis.
