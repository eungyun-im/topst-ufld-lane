"""Train the compact UFLD model on synthetic track data (GPU).

    python3 train.py --steps 5000 --batch 32 --out runs/ufld_320x160.pth

Data is rendered on the fly from the piracer_sim track texture (track_datagen),
so no dataset files are needed. Loss:
  - loc_row  : cross-entropy over grid cells, only where a lane point exists
  - exist_row: cross-entropy (present / absent) for every (row, lane)
  - exist_col: cross-entropy forcing lanes to 'absent' (this synthetic set uses
               only the two row-anchored ego edges; col lanes stay empty)
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from track_datagen import TrackRenderer, UfldHead, make_sample
from model import UfldNet, IMAGENET_MEAN, IMAGENET_STD

MEAN = np.array(IMAGENET_MEAN, np.float32)
STD = np.array(IMAGENET_STD, np.float32)


def preprocess(bgr):
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


class TrackDataset(Dataset):
    def __init__(self, length, head, seed=0):
        self.length = length
        self.head = head
        self.seed = seed
        self._rend = None

    def _renderer(self):
        if self._rend is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info else 0
            self._rend = TrackRenderer(seed=self.seed * 1000 + wid)
        return self._rend

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rend = self._renderer()
        for _ in range(20):
            s = make_sample(rend, self.head)
            if s is not None:
                break
        if s is None:
            s = dict(image=np.zeros((self.head.H, self.head.W, 3), np.uint8),
                     loc_row=np.zeros((self.head.num_row, self.head.num_lanes), np.float32),
                     exist_row=np.zeros((self.head.num_row, self.head.num_lanes), np.float32))
        img = preprocess(s["image"])
        loc = torch.from_numpy(s["loc_row"]).long().clamp(0, self.head.num_grid_row - 1)
        ex = torch.from_numpy(s["exist_row"]).long()
        return img, loc, ex


def compute_loss(out, loc_t, ex_t, ce):
    loc_row, loc_col, exist_row, exist_col = out
    B, GR, NR, NL = loc_row.shape
    # exist_row CE over all (row, lane)
    er = exist_row.permute(0, 2, 3, 1).reshape(-1, 2)
    loss_exist = ce(er, ex_t.reshape(-1))
    # loc_row CE only where a point exists
    lr = loc_row.permute(0, 2, 3, 1).reshape(-1, GR)
    mask = ex_t.reshape(-1) > 0
    if mask.any():
        loss_loc = ce(lr[mask], loc_t.reshape(-1)[mask])
    else:
        loss_loc = torch.zeros((), device=loc_row.device)
    # exist_col forced absent (class 0)
    ec = exist_col.permute(0, 2, 3, 1).reshape(-1, 2)
    loss_col = ce(ec, torch.zeros(ec.shape[0], dtype=torch.long, device=ec.device))
    return loss_loc + loss_exist + 0.2 * loss_col, (loss_loc.item(), loss_exist.item(), loss_col.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/ufld_320x160.pth")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    head = UfldHead()
    ds = TrackDataset(args.steps * args.batch, head, seed=1)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)

    net = UfldNet(head.num_lanes, head.num_row, head.num_col,
                  head.num_grid_row, head.num_grid_col,
                  pretrained=not args.no_pretrained).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    ce = nn.CrossEntropyLoss()

    net.train()
    run = 0.0
    for step, (img, loc, ex) in enumerate(dl):
        img = img.to(dev, non_blocking=True)
        loc = loc.to(dev, non_blocking=True)
        ex = ex.to(dev, non_blocking=True)
        out = net(img)
        loss, parts = compute_loss(out, loc, ex, ce)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        run += loss.item()
        if (step + 1) % 50 == 0:
            print("step %5d/%d  loss %.4f  (loc %.3f exist %.3f col %.3f)  lr %.2e"
                  % (step + 1, args.steps, run / 50, *parts, sched.get_last_lr()[0]), flush=True)
            run = 0.0
        if step + 1 >= args.steps:
            break

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"model": net.state_dict(),
                "head": vars(head) if hasattr(head, "__dict__") else {}}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
