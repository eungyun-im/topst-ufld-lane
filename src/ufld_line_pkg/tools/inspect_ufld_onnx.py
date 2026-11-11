#!/usr/bin/env python3
"""Inspect a UFLD ONNX file and emit matching ufld_line.launch parameters.

The #1 silent failure in this pipeline is a mismatch between the ONNX output
tensor names / head shapes and the params baked into ufld_line.launch. The C++
decoder assumes UFLD-v2 layout:

    loc_row   : [1, num_grid_row, num_row, num_lanes]
    loc_col   : [1, num_grid_col, num_col, num_lanes]
    exist_row : [1, 2,            num_row, num_lanes]
    exist_col : [1, 2,            num_col, num_lanes]

This tool reads the ONNX graph, verifies that layout, infers the head-shape
params from the real tensor dims, and prints ready-to-paste <param> lines
(and a --patch mode that rewrites the launch file in place).

Runs anywhere onnx (and optionally onnxruntime) is installed. No board needed.

Usage:
    python3 inspect_ufld_onnx.py ufld_320x160.onnx
    python3 inspect_ufld_onnx.py ufld_320x160.onnx --patch ../launch/ufld_line.launch
"""
import argparse
import re
import sys

import onnx


def dim_list(value_info):
    dims = []
    for d in value_info.type.tensor_type.shape.dim:
        dims.append(d.dim_value if (d.dim_value and d.dim_value > 0) else None)
    return dims


def classify(outputs):
    """Sort the 4 UFLD outputs into loc_row/loc_col/exist_row/exist_col.

    Prefer the tensor's own name; fall back to shape heuristics (the two
    'exist' tensors have a dim == 2; between the two loc/exist pairs the one
    with more classification cells is 'row' in a wide image, but we rely on
    names first because that is what the launch file must match)."""
    buckets = {}
    for name, dims in outputs.items():
        low = name.lower()
        if "exist" in low and "row" in low:
            buckets["exist_row"] = (name, dims)
        elif "exist" in low and "col" in low:
            buckets["exist_col"] = (name, dims)
        elif ("loc" in low or "cls" in low) and "row" in low:
            buckets["loc_row"] = (name, dims)
        elif ("loc" in low or "cls" in low) and "col" in low:
            buckets["loc_col"] = (name, dims)
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", help="path to UFLD .onnx")
    ap.add_argument("--patch", metavar="LAUNCH", default=None,
                    help="rewrite the given ufld_line.launch in place")
    ap.add_argument("--run", action="store_true",
                    help="run one forward pass with onnxruntime to confirm shapes")
    args = ap.parse_args()

    model = onnx.load(args.onnx)
    try:
        onnx.checker.check_model(model)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] onnx.checker: {e}")

    graph = model.graph
    inputs = {i.name: dim_list(i) for i in graph.input}
    outputs = {o.name: dim_list(o) for o in graph.output}

    print("=== INPUTS ===")
    for n, d in inputs.items():
        print(f"  {n}: {d}")
    print("=== OUTPUTS ===")
    for n, d in outputs.items():
        print(f"  {n}: {d}")

    if len(outputs) != 4:
        print(f"\n[error] expected 4 outputs (loc_row/loc_col/exist_row/exist_col), "
              f"got {len(outputs)}. The decoder needs all four.", file=sys.stderr)

    b = classify(outputs)
    missing = [k for k in ("loc_row", "loc_col", "exist_row", "exist_col") if k not in b]
    if missing:
        print(f"\n[error] could not identify by name: {missing}", file=sys.stderr)
        print("        Rename the export heads, or hand-map names in the launch file.",
              file=sys.stderr)
        # still try to emit what we can
    # input (initializers can crowd graph.input; take the first non-initializer)
    init_names = {i.name for i in graph.initializer}
    real_inputs = [(n, d) for n, d in inputs.items() if n not in init_names]
    if real_inputs:
        in_name, in_dims = real_inputs[0]
        in_w = in_dims[3] if len(in_dims) == 4 and in_dims[3] else 320
        in_h = in_dims[2] if len(in_dims) == 4 and in_dims[2] else 160
    else:
        print("\n[warn] no graph input found; assuming name='input', keeping 320x160.")
        in_name, in_w, in_h = "input", 320, 160

    params = {"input_name": in_name, "input_width": in_w, "input_height": in_h}

    def shape_of(key):
        return b[key][1] if key in b else None

    lr, lc = shape_of("loc_row"), shape_of("loc_col")
    er, ec = shape_of("exist_row"), shape_of("exist_col")

    warnings = []
    if lr and len(lr) == 4:
        params["num_grid_row"] = lr[1]
        params["num_row"] = lr[2]
        params["num_lanes"] = lr[3]
    if lc and len(lc) == 4:
        params["num_grid_col"] = lc[1]
        params["num_col"] = lc[2]
    if er and len(er) == 4 and er[1] != 2:
        warnings.append(f"exist_row dim1={er[1]} (expected 2)")
    if ec and len(ec) == 4 and ec[1] != 2:
        warnings.append(f"exist_col dim1={ec[1]} (expected 2)")
    if lr and er and lr[2:] != er[2:]:
        warnings.append(f"loc_row/exist_row (num_row,num_lanes) mismatch: {lr[2:]} vs {er[2:]}")
    if lc and ec and lc[2:] != ec[2:]:
        warnings.append(f"loc_col/exist_col (num_col,num_lanes) mismatch: {lc[2:]} vs {ec[2:]}")

    for k in ("loc_row", "loc_col", "exist_row", "exist_col"):
        if k in b:
            params[f"{k}_name"] = b[k][0]

    print("\n=== INFERRED launch params ===")
    for k, v in params.items():
        print(f"  {k} = {v}")
    if warnings:
        print("\n=== WARNINGS ===")
        for w in warnings:
            print(f"  ! {w}")

    if args.run:
        try:
            import numpy as np
            import onnxruntime as ort
            sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
            x = np.zeros((1, 3, in_h, in_w), dtype=np.float32)
            outs = sess.run(None, {in_name: x})
            print("\n=== onnxruntime forward OK ===")
            for o, arr in zip(sess.get_outputs(), outs):
                print(f"  {o.name}: {list(arr.shape)}")
        except Exception as e:  # noqa: BLE001
            print(f"\n[warn] onnxruntime run failed: {e}")

    if args.patch:
        patch_launch(args.patch, params)
    else:
        print("\n(Use --patch ../launch/ufld_line.launch to write these into the launch file.)")


def patch_launch(path, params):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    changed = []
    for name, value in params.items():
        # match: <param name="X" ... value="..."/>
        pat = re.compile(rf'(<param\s+name="{re.escape(name)}"[^>]*\bvalue=")[^"]*(")')
        if pat.search(text):
            text = pat.sub(rf'\g<1>{value}\g<2>', text)
            changed.append(name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n[patched] {path}")
    print(f"  updated: {', '.join(changed) if changed else '(none matched)'}")
    not_found = [k for k in params if k not in changed]
    if not_found:
        print(f"  not present in launch (add manually if needed): {', '.join(not_found)}")


if __name__ == "__main__":
    main()
