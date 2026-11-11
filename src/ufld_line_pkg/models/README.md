# Model Files

Place converted TNN model files here:

```text
ufld_320x160.tnnproto
ufld_320x160.tnnmodel
```

## Conversion flow

```text
PyTorch .pth ── export ──▶ ONNX (fixed 1x3x160x320)
             ── inspect ──▶ verify output names/shapes, patch launch
             ── onnx2tnn ─▶ .tnnproto / .tnnmodel
```

### 1. Export to ONNX (on the machine with the UFLD-v2 repo + weights)

Export with a **fixed** input shape `1x3x160x320` (NCHW) and named outputs.
UFLD-v2's `forward()` returns a dict; make sure the four heads are exported as
`loc_row, loc_col, exist_row, exist_col` (or note whatever names come out —
step 2 handles renames).

### 2. Inspect the ONNX and sync the launch file  (do this BEFORE onnx2tnn)

The decoder assumes UFLD-v2 layout:

```text
loc_row   : [1, num_grid_row, num_row, num_lanes]
loc_col   : [1, num_grid_col, num_col, num_lanes]
exist_row : [1, 2,            num_row, num_lanes]
exist_col : [1, 2,            num_col, num_lanes]
```

A mismatch between these shapes/names and the launch params makes the decoder
emit **silently wrong** offsets. Use the helper to check and auto-patch:

```bash
# just inspect (add --run to also do an onnxruntime forward pass):
python3 ../tools/inspect_ufld_onnx.py ufld_320x160.onnx --run

# inspect AND rewrite the launch params in place:
python3 ../tools/inspect_ufld_onnx.py ufld_320x160.onnx --patch ../launch/ufld_line.launch
```

It prints the inferred `input_name`, `input_width/height`, `num_grid_row/col`,
`num_row/col`, `num_lanes`, and the four output tensor names, then (with
`--patch`) writes them into `ufld_line.launch`. Requires `onnx` (and
`onnxruntime` for `--run`).

### 3. Convert ONNX -> TNN

```bash
# from the TNN repo:
python3 tools/onnx2tnn/onnx-converter/onnx2tnn.py ufld_320x160.onnx -optimize -v v3.0
# produces ufld_320x160.tnnproto / ufld_320x160.tnnmodel -> copy them here
```

### 4. Build + run

```bash
colcon build --packages-select ufld_line_pkg --cmake-args \
  -DUFLD_USE_TNN=ON -DTNN_ROOT=/home/topst/TNN \
  -DTNN_LIB_DIR=/home/topst/TNN/benchmark/benchmark_armlinux/build
ros2 launch ufld_line_pkg ufld_line.launch
```

If the converter renames outputs again, re-run step 2 against the `.tnnproto`
tensor names, or override the `*_name` params in the launch file.
