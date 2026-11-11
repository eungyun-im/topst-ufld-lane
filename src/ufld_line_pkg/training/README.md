# UFLD training (synthetic, from the piracer_sim track)

Trains the lane model that `ufld_line_pkg` runs, **without Gazebo**. First-person
camera views + pixel-exact lane labels are rendered by projecting the known
top-down track texture (`piracer_sim/.../track.png`) through a pinhole camera at
random on-road poses. Colours in the texture are unambiguous, so labels are
generated automatically (no hand-labelling).

```
track_datagen.py   # render (image, UFLD labels) pairs from the track texture
model.py           # compact ResNet18 + UFLD-v2 head (outputs match the C++ decoder)
train.py           # on-the-fly training loop (GPU)
validate_decode.py # render -> model -> python port of ufld_decoder.cpp -> offset
export_onnx.py     # -> models/ufld_320x160.onnx (names/shapes match ufld_line.launch)
```

## Pipeline

```bash
# 0. deps: torch, torchvision, opencv-python, numpy, onnx (GPU strongly recommended)

# 1. sanity-check the synthetic data + labels
VIZ_OUT=/tmp/labels.png python3 track_datagen.py      # writes an 8-tile overlay

# 2. train (~10 min on an RTX-class GPU; renders data on the fly)
python3 train.py --steps 6000 --batch 32 --out runs/ufld_320x160.pth

# 3. check predictions + offset end-to-end (python decoder mirrors the C++ one)
python3 validate_decode.py --ckpt runs/ufld_320x160.pth --out viz.png

# 4. export ONNX (fixed 1x3x160x320, output names loc_row/loc_col/exist_row/exist_col)
python3 export_onnx.py --ckpt runs/ufld_320x160.pth --out ../models/ufld_320x160.onnx

# 5. confirm the ONNX matches the launch params
python3 ../tools/inspect_ufld_onnx.py ../models/ufld_320x160.onnx --run

# 6. on the board: ONNX -> TNN (see ../models/README.md), then build with -DUFLD_USE_TNN=ON
```

## Contract with the runtime

The model outputs exactly what `ufld_decoder.cpp` reads and `ufld_line.launch`
declares:

| tensor | shape | meaning |
|--------|-------|---------|
| `loc_row`   | `[1,100,56,4]` | ego edges -> lanes **1,2** (row-anchored) |
| `exist_row` | `[1,2,56,4]`   | present/absent per (row, lane) |
| `loc_col`   | `[1,100,41,4]` | col-anchored lanes 0,3 (unused here) |
| `exist_col` | `[1,2,41,4]`   | forced absent in this synthetic set |

Preprocessing (resize to 320x160, RGB, ImageNet mean/std, NCHW) is identical in
`train.py`, `validate_decode.py`, and `tnn_ufld_runner.cpp`.

## Caveats (read before racing)

- **Sim-to-real gap.** This trains on a *rendered* view of the track texture,
  not real camera frames. Camera height/pitch/FOV in `TrackRenderer` are
  assumptions — tune them to the real mount, or fine-tune on real frames.
- **Offset ill-defined on hard curves** where one edge leaves the frame; the
  node's offset filter (`offset_filter_alpha`) smooths this. Validate
  `/line_offset` range against the old `line_pkg` and tune control gain.
- Two ego edges are assigned to row-anchor lanes 1,2; col lanes stay empty. If
  you later train a model with different heads, re-run the inspect tool
  (step 5) to resync the launch params.
