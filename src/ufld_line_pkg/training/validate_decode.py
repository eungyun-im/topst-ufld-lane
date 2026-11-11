"""End-to-end check: render -> model -> (python port of ufld_decoder.cpp) -> offset.

    python3 validate_decode.py --ckpt runs/ufld_320x160.pth --n 8 --out viz.png

Mirrors the C++ decoder's row-anchor path (lanes 1,2) so we can confirm the
trained model produces sensible lane points and a sane /line_offset BEFORE
converting to TNN and testing on the board.
"""
import argparse
import os
import cv2
import numpy as np
import torch

from track_datagen import TrackRenderer, UfldHead, make_sample
from model import UfldNet, IMAGENET_MEAN, IMAGENET_STD

MEAN = np.array(IMAGENET_MEAN, np.float32)
STD = np.array(IMAGENET_STD, np.float32)


def preprocess(bgr):
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None]


def softmax_weighted(vec, center, local=1):
    lo, hi = max(0, center - local), min(len(vec) - 1, center + local)
    seg = vec[lo:hi + 1]
    seg = np.exp(seg - seg.max())
    idx = np.arange(lo, hi + 1)
    return float((seg * idx).sum() / seg.sum()) + 0.5


def decode(loc_row, exist_row, head, W, H, thr=0.5):
    """Port of the lanes {1,2} row-anchor path from ufld_decoder.cpp."""
    GR = head.num_grid_row
    lanes = {}
    bottom_xs = []
    for lane in (1, 2):
        present = exist_row[1, :, lane] > exist_row[0, :, lane]
        if present.sum() <= head.num_row * thr:
            continue
        pts = []
        for k in range(head.num_row):
            if not present[k]:
                continue
            col = loc_row[:, k, lane]
            g = softmax_weighted(col, int(col.argmax()), head.__dict__.get("local_width", 1))
            x = g / (GR - 1) * W
            y = head.row_anchors[k] * H
            pts.append((x, y))
            if y >= H * 0.70:
                bottom_xs.append(x)
        if pts:
            lanes[lane] = pts
    offset, center, valid = 0.0, None, False
    if len(bottom_xs) >= 2:
        bottom_xs.sort()
        m = len(bottom_xs) // 2
        center = (bottom_xs[m - 1] + bottom_xs[m]) / 2
        offset = center - W / 2
        valid = True
    return lanes, offset, center, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ufld_320x160.pth")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="viz_decode.png")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    head = UfldHead()
    head.__dict__["local_width"] = 1
    net = UfldNet(head.num_lanes, head.num_row, head.num_col,
                  head.num_grid_row, head.num_grid_col, pretrained=False).to(dev)
    net.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
    net.eval()
    rend = TrackRenderer(seed=777)

    tiles, errs = [], []
    while len(tiles) < args.n:
        s = make_sample(rend, head)
        if s is None:
            continue
        img = s["image"]
        with torch.no_grad():
            lr, lc, er, ec = net(preprocess(img).to(dev))
        lr = lr[0].cpu().numpy(); er = er[0].cpu().numpy()
        lanes, offset, center, valid = decode(lr, er, head, img.shape[1], img.shape[0])

        # ground-truth offset from labels for comparison
        gt_xs = []
        for k in range(head.num_row):
            y = head.row_anchors[k] * img.shape[0]
            if y < img.shape[0] * 0.70:
                continue
            for lane in (1, 2):
                if s["exist_row"][k, lane]:
                    gt_xs.append(s["loc_row"][k, lane] / (head.num_grid_row - 1) * img.shape[1])
        gt_off = (min(gt_xs) + max(gt_xs)) / 2 - img.shape[1] / 2 if len(gt_xs) >= 2 else None

        vis = img.copy()
        for lane, col in ((1, (0, 0, 255)), (2, (0, 255, 0))):
            for (x, y) in lanes.get(lane, []):
                cv2.circle(vis, (int(x), int(y)), 2, col, -1)
        cv2.line(vis, (img.shape[1] // 2, 0), (img.shape[1] // 2, img.shape[0]), (255, 255, 0), 1)
        if valid:
            cv2.circle(vis, (int(center), int(img.shape[0] * 0.85)), 4, (0, 165, 255), -1)
        txt = "off=%.0f" % offset + (" gt=%.0f" % gt_off if gt_off is not None else "")
        cv2.putText(vis, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if valid and gt_off is not None:
            errs.append(abs(offset - gt_off))
        tiles.append(vis)

    grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    cv2.imwrite(args.out, cv2.resize(grid, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    if errs:
        print("offset MAE vs GT: %.2f px over %d valid frames" % (np.mean(errs), len(errs)))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
