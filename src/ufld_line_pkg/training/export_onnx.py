"""Export a trained UfldNet checkpoint to ONNX for the onnx2tnn step.

    python3 export_onnx.py --ckpt runs/ufld_320x160.pth --out ../models/ufld_320x160.onnx

Exports with fixed input 1x3x160x320 and output names loc_row/loc_col/
exist_row/exist_col — exactly what ufld_line.launch and the C++ decoder expect.
After exporting, run tools/inspect_ufld_onnx.py to confirm names/shapes.
"""
import argparse
import os
import torch
from model import UfldNet
from track_datagen import UfldHead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ufld_320x160.pth")
    ap.add_argument("--out", default="../models/ufld_320x160.onnx")
    ap.add_argument("--opset", type=int, default=11)
    args = ap.parse_args()

    head = UfldHead()
    net = UfldNet(head.num_lanes, head.num_row, head.num_col,
                  head.num_grid_row, head.num_grid_col, pretrained=False)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model"])
    net.eval()

    dummy = torch.zeros(1, 3, head.H, head.W)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(
        net, dummy, args.out,
        input_names=["input"],
        output_names=["loc_row", "loc_col", "exist_row", "exist_col"],
        opset_version=args.opset, dynamo=False)
    print("exported", args.out)


if __name__ == "__main__":
    main()
