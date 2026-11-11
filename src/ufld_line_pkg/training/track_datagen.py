"""Synthetic UFLD dataset generated from the piracer_sim top-down track texture.

Gazebo is not required. We take the known top-down track image (track.png),
extract the road / white-edge / yellow-line masks (colours are unambiguous in
the texture, so labels are essentially free), then render first-person camera
views by projecting the ground plane through a pinhole camera at random on-road
poses. The SAME projection is applied to the lane masks, so every rendered
320x160 frame comes with pixel-exact lane labels.

Labels follow the UFLD-v2 layout the C++ decoder expects:
    loc_row  : [num_grid_row, num_row, num_lanes]   (ego edges -> lanes 1,2)
    exist_row: [num_row, num_lanes]
    loc_col  : [num_grid_col, num_col, num_lanes]   (unused here -> lanes 0,3 absent)
    exist_col: [num_col, num_lanes]

The two road edges (left/right white boundaries) are assigned to row-anchor
lanes 1 and 2, matching how the decoder reads ego lanes.
"""
import os
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEX = os.path.normpath(os.path.join(
    HERE, "..", "..", "piracer_sim",
    "models", "race_track", "materials", "textures", "track.png"))


class UfldHead:
    """Head geometry; defaults match ufld_line.launch / ufld_types.hpp."""
    def __init__(self, input_w=320, input_h=160, num_lanes=4,
                 num_row=56, num_col=41, num_grid_row=100, num_grid_col=100,
                 row_anchor_min=0.40, row_anchor_max=1.00,
                 col_anchor_min=0.00, col_anchor_max=1.00):
        self.W, self.H = input_w, input_h
        self.num_lanes = num_lanes
        self.num_row, self.num_col = num_row, num_col
        self.num_grid_row, self.num_grid_col = num_grid_row, num_grid_col
        self.row_anchors = np.linspace(row_anchor_min, row_anchor_max, num_row)
        self.col_anchors = np.linspace(col_anchor_min, col_anchor_max, num_col)


class TrackRenderer:
    def __init__(self, tex_path=DEFAULT_TEX, canvas_width_m=6.9,
                 cam_height_m=0.10, pitch_deg=18.0, fov_h=1.20, seed=0):
        self.im = cv2.imread(tex_path)
        if self.im is None:
            raise FileNotFoundError(tex_path)
        self.h, self.w = self.im.shape[:2]
        hsv = cv2.cvtColor(self.im, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(self.im, cv2.COLOR_BGR2GRAY)
        yellow = cv2.inRange(hsv, (15, 60, 110), (45, 255, 255))
        road = cv2.inRange(gray, 0, 25)
        # Drivable corridor = black asphalt + yellow centre lines merged in, so
        # the yellow paint does not carve internal boundaries out of the road.
        # The corridor's OUTER boundary is then exactly the left/right track
        # edges (the white lines). This is TEXT-FREE by construction: the baked
        # dimension annotations are not part of the corridor region, unlike a
        # colour-threshold of white which would pick the text up.
        corridor = cv2.bitwise_or(road, yellow)
        corridor = cv2.morphologyEx(corridor, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        corridor = cv2.morphologyEx(corridor, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        self.road = corridor
        grad = cv2.morphologyEx(corridor, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        self.edge = cv2.dilate(grad, np.ones((5, 5), np.uint8))   # ~paint width
        self.dist = cv2.distanceTransform(self.road, cv2.DIST_L2, 5)
        self.mpp = canvas_width_m / self.w
        self.cam_h_px = cam_height_m / self.mpp
        self.pitch = np.deg2rad(pitch_deg)

        self.IMW, self.IMH = 320, 160
        fx = (self.IMW / 2) / np.tan(fov_h / 2)
        self.fx = self.fy = fx
        self.cx, self.cy = self.IMW / 2, self.IMH / 2
        uu, vv = np.meshgrid(np.arange(self.IMW), np.arange(self.IMH))
        self.rx = (uu - self.cx) / self.fx
        self.ry = (vv - self.cy) / self.fy
        self.rz = np.ones_like(self.rx)
        self.rng = np.random.default_rng(seed)
        self.cand = np.argwhere(self.dist > self.dist.max() * 0.45)

    def road_yaw(self, px, py, win=60):
        x0, x1 = max(0, px - win), min(self.w, px + win)
        y0, y1 = max(0, py - win), min(self.h, py + win)
        ys, xs = np.where(self.road[y0:y1, x0:x1] > 0)
        if len(xs) < 20:
            return None
        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], 1)
        pts -= pts.mean(0)
        w, v = np.linalg.eigh(pts.T @ pts)
        d = v[:, np.argmax(w)]
        return np.arctan2(d[1], d[0])

    def render(self, px, py, yaw):
        f = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        o = f * np.cos(self.pitch) + np.array([0, 0, -1.0]) * np.sin(self.pitch)
        o /= np.linalg.norm(o)
        xcam = np.array([f[1], -f[0], 0.0]); xcam /= np.linalg.norm(xcam)
        ycam = np.cross(o, xcam)
        rw = self.rx[..., None] * xcam + self.ry[..., None] * ycam + self.rz[..., None] * o
        C = np.array([px, py, self.cam_h_px])
        rzw = rw[..., 2]
        t = np.where(rzw < -1e-6, -C[2] / rzw, np.nan)
        gx = (C[0] + t * rw[..., 0]).astype(np.float32)
        gy = (C[1] + t * rw[..., 1]).astype(np.float32)
        valid = np.isfinite(gx) & np.isfinite(gy)
        mx = np.where(valid, gx, -1); my = np.where(valid, gy, -1)
        img = cv2.remap(self.im, mx, my, cv2.INTER_LINEAR, borderValue=(45, 45, 45))
        me = cv2.remap(self.edge, mx, my, cv2.INTER_NEAREST, borderValue=0)
        mroad = cv2.remap(self.road, mx, my, cv2.INTER_NEAREST, borderValue=0)
        img[~valid] = (60, 60, 60)
        return img, me, mroad

    def sample_pose(self):
        py, px = self.cand[self.rng.integers(0, len(self.cand))]
        yaw = self.road_yaw(int(px), int(py))
        if yaw is None:
            return None
        if self.rng.random() < 0.5:
            yaw += np.pi
        yaw += self.rng.normal(0, np.deg2rad(8))          # heading jitter
        return int(px), int(py), yaw


def _clusters(xs, gap=8):
    """Split sorted x indices into contiguous clusters; return centroids."""
    if len(xs) == 0:
        return []
    xs = np.sort(xs)
    groups, cur = [], [xs[0]]
    for x in xs[1:]:
        if x - cur[-1] <= gap:
            cur.append(x)
        else:
            groups.append(cur); cur = [x]
    groups.append(cur)
    return [float(np.mean(g)) for g in groups]


def extract_row_labels(mw, road_view, head: UfldHead):
    """From a rendered white-edge mask, produce left/right lane x per row anchor.

    Returns loc (num_row, num_lanes) grid-cell floats and exist (num_row,
    num_lanes) in {0,1}. Only lanes 1 (left) and 2 (right) are populated."""
    H, W = mw.shape
    GR = head.num_grid_row
    loc = np.zeros((head.num_row, head.num_lanes), np.float32)
    exist = np.zeros((head.num_row, head.num_lanes), np.float32)
    for k, ay in enumerate(head.row_anchors):
        y = int(round(ay * (H - 1)))
        xs = np.where(mw[y] > 0)[0]
        cents = _clusters(xs)
        if not cents:
            continue
        rxs = np.where(road_view[y] > 0)[0]
        road_c = float(np.mean(rxs)) if len(rxs) else W / 2.0
        left = [c for c in cents if c < road_c]
        right = [c for c in cents if c >= road_c]
        if left:
            xL = max(left)                      # innermost left edge
            loc[k, 1] = xL / (W - 1) * (GR - 1)
            exist[k, 1] = 1.0
        if right:
            xR = min(right)                     # innermost right edge
            loc[k, 2] = xR / (W - 1) * (GR - 1)
            exist[k, 2] = 1.0
    return loc, exist


def make_sample(rend: TrackRenderer, head: UfldHead):
    """Render one (image, labels) pair. Returns dict or None."""
    pose = rend.sample_pose()
    if pose is None:
        return None
    px, py, yaw = pose
    img, me, mroad = rend.render(px, py, yaw)
    # Keep only "driving forward" views: the car must be on the road, i.e. the
    # bottom-centre of the frame is drivable. Rejects sideways / off-road poses.
    H, W = mroad.shape
    bc = mroad[int(H * 0.9):, int(W * 0.35):int(W * 0.65)]
    if cv2.countNonZero(bc) < bc.size * 0.5:
        return None
    # mroad (projected drivable region) disambiguates which side each edge is on.
    loc_row, exist_row = extract_row_labels(me, mroad, head)
    if exist_row.sum() < 6:                     # too few lane points -> skip
        return None
    return dict(image=img, loc_row=loc_row, exist_row=exist_row)


if __name__ == "__main__":
    # Visual sanity check: render a grid and overlay decoded row-anchor labels.
    SP = os.environ.get("VIZ_OUT", "/tmp/track_labels_viz.png")
    head = UfldHead()
    rend = TrackRenderer(seed=1)
    tiles = []
    while len(tiles) < 8:
        s = make_sample(rend, head)
        if s is None:
            continue
        img = s["image"].copy()
        H, W = img.shape[:2]
        for k, ay in enumerate(head.row_anchors):
            y = int(round(ay * (H - 1)))
            for lane, col in ((1, (0, 0, 255)), (2, (0, 255, 0))):
                if s["exist_row"][k, lane]:
                    x = int(s["loc_row"][k, lane] / (head.num_grid_row - 1) * (W - 1))
                    cv2.circle(img, (x, y), 2, col, -1)
        tiles.append(img)
    grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, 8, 2)])
    cv2.imwrite(SP, cv2.resize(grid, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    print("wrote", SP)
