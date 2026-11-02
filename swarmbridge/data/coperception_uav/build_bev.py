"""Build per-UAV BEV inputs and labels from the CoPerception-UAV town3 sample.

Each (sample, UAV) gets its own ground grid: SIZE x SIZE metres at RES, centred on the
ground point under the BOTTOM camera's principal ray, axes along the image right/down
directions projected onto the ground. Every grid cell is inverse-warped into the BOTTOM
cameras of all 5 UAVs (same sample) with the verified geometry.

Saved per (sample, UAV) in <out>/<split>/<sample_token>_<uav>.npz:
  rgb  (G,G,3) uint8 : this UAV's BOTTOM image on its own grid (0 outside its footprint)
  obs  (G,G)   uint8 : classes seen by THIS UAV only            {0 other, 1 road, 2 vehicle, 255 unseen}
  gt   (G,G)   uint8 : vehicles from the 3D boxes + road/other from the union of the 5 UAVs'
                       views, all restricted to cells some UAV sees (255 elsewhere)
  seen (5,G,G) bool  : which UAV's camera covers each cell
  origin (2,), axes (2,2) float : world xy of cell (0,0) centre and unit world vectors of
                       +col / +row; world = origin + RES * (col * axes[0] + row * axes[1])

Classes come from seg colours verified by eye (AirSim object colours are reused across
object types, so only colours that were consistently road are used; vehicles are taken from
the 3D boxes, which project onto vehicle pixels for 84-99.8% of boxes per UAV).

    python -m swarmbridge.data.coperception_uav.build_bev data/coperception_uav/town3 data/coperception_uav/bev_town3
"""
import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scipy import ndimage

from .geometry import CX, CY, H, W, apparent_xy, ground_to_pixel, pixel_to_ground, quat_to_R

ROAD = [(235, 124, 133), (139, 242, 254), (215, 252, 228), (155, 154, 125), (197, 177, 129)]
VEHICLE = (95, 90, 229)
OTHER, ROAD_C, VEH_C, UNSEEN = 0, 1, 2, 255


def seg_to_class(seg):
    cls = np.zeros(seg.shape[:2], np.uint8)
    for c in ROAD:
        cls[(seg == c).all(-1)] = ROAD_C
    cls[(seg == VEHICLE).all(-1)] = VEH_C
    return cls


def grid_frame(cam, size, res):
    """origin (world xy of cell (0,0) centre) and axes (+col, +row unit vectors)."""
    R = quat_to_R(cam["rotation"])
    e_u = R[:2, 1] / np.linalg.norm(R[:2, 1])  # image right on the ground
    e_v = R[:2, 2] - (R[:2, 2] @ e_u) * e_u     # image down, made orthogonal
    e_v /= np.linalg.norm(e_v)
    cx, cy = pixel_to_ground(np.array(CX), np.array(CY), cam)
    G = int(round(size / res))
    origin = np.array([cx, cy]) - res * (G - 1) / 2 * (e_u + e_v)
    return origin, np.stack([e_u, e_v]), G


def cell_centres(origin, axes, G, res):
    r, c = np.mgrid[0:G, 0:G]
    return origin + res * (c[..., None] * axes[0] + r[..., None] * axes[1])  # (G,G,2)


def sample_camera(xy, cam, img):
    """Nearest-pixel lookup of img at ground points xy -> (values, inside mask)."""
    u, v, front = ground_to_pixel(xy[..., 0], xy[..., 1], cam)
    inside = front & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    ui = np.clip(np.round(u), 0, W - 1).astype(int)
    vi = np.clip(np.round(v), 0, H - 1).astype(int)
    return img[vi, ui], inside


def box_corners(a):
    w, l = a["size"][0], a["size"][1]
    q = a["rotation"]
    th = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    Rb = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return np.array(a["translation"][:2]) + (Rb @ np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).T / 2).T


def box_visible(a, cams_j, veh_near):
    """-> (inside some camera's image, seen by some camera: its centre at its true height
    lands on vehicle pixels there)."""
    x, y, z = a["translation"]
    inside = False
    for cam, near in zip(cams_j, veh_near):
        u, v, front = ground_to_pixel(np.array(x), np.array(y), cam, z)
        if front and 0 <= u < W and 0 <= v < H:
            inside = True
            if near[int(v), int(u)]:
                return True, True
    return inside, False


def draw_boxes(anns, origin, axes, G, res, cam):
    """Rasterize box footprints onto the grid at their APPARENT position on z = 0 as seen from
    `cam` (elevated roads), so they align with that camera's reprojected image."""
    canvas = Image.new("L", (G, G), 0)
    draw = ImageDraw.Draw(canvas)
    M = np.stack(axes)  # world offset -> (col, row) / res
    for a in anns:
        corners = apparent_xy(box_corners(a), np.full(4, a["translation"][2]), cam)
        cr = (corners - origin) @ M.T / res
        if (cr < -10).all(0).any() or (cr > G + 10).all(0).any():
            continue
        draw.polygon([tuple(p) for p in cr], fill=1)
    return np.array(canvas, bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--size", type=float, default=96.0)
    ap.add_argument("--res", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0, help="only the first N samples (debug)")
    a = ap.parse_args()

    root = Path(a.root) / "airsim_camera_seg"
    meta = lambda n: json.load(open(root / "v1.0-mini" / f"{n}.json"))
    cams = {c["token"]: c for c in meta("calibrated_sensor")}
    scene_name = {s["token"]: s["name"] for s in meta("scene")}
    samples = meta("sample")
    anns = defaultdict(list)
    for x in meta("sample_annotation"):
        anns[x["sample_token"]].append(x)
    bottoms = defaultdict(dict)
    for s in meta("sample_data"):
        if "CAMERA_BOTTOM" in s["filename"]:
            bottoms[s["sample_token"]][int(s["filename"].split("/")[1][-1])] = s

    split_of = {}
    tree = ast.parse((Path(a.root).parent / "raw" / "split.py").read_text())
    for node in tree.body:  # split.py is three list literals; never exec downloaded code
        if isinstance(node, ast.Assign) and node.targets[0].id.endswith("_split"):
            for sc in ast.literal_eval(node.value):
                split_of[sc] = node.targets[0].id[: -len("_split")]

    stats = defaultdict(float)
    todo = samples[: a.limit] if a.limit else samples
    for k, smp in enumerate(todo):
        split = split_of.get(scene_name[smp["scene_token"]])
        if split is None:
            continue
        sds = bottoms[smp["token"]]
        seg = {j: np.array(Image.open(root / sd["filename"].replace(".png", "_seg.png")).convert("RGB")) for j, sd in sds.items()}
        cls = {j: seg_to_class(s) for j, s in seg.items()}
        cams_j = [cams[sds[j]["calibrated_sensor_token"]] for j in sorted(sds)]
        near = [ndimage.binary_dilation(cls[j] == VEH_C, iterations=4) for j in sorted(sds)]
        vis = [box_visible(b, cams_j, near) for b in anns[smp["token"]]]
        seen_boxes = [b for b, (_, v) in zip(anns[smp["token"]], vis) if v]
        hidden_boxes = [b for b, (_, v) in zip(anns[smp["token"]], vis) if not v]
        stats["boxes_in_image"] += sum(i for i, _ in vis)
        stats["boxes_hidden_in_image"] += sum(i and not v for i, v in vis)
        out_dir = Path(a.out) / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, sd in sds.items():
            cam = cams[sd["calibrated_sensor_token"]]
            origin, axes, G = grid_frame(cam, a.size, a.res)
            xy = cell_centres(origin, axes, G, a.res)
            seen = np.zeros((5, G, G), bool)
            union = np.full((G, G), UNSEEN, np.uint8)
            obs = None
            for j, sdj in sds.items():
                vals, ins = sample_camera(xy, cams[sdj["calibrated_sensor_token"]], cls[j])
                seen[j] = ins
                fill = ins & ((union == UNSEEN) | (vals > union) & (vals != UNSEEN))
                union[fill] = vals[fill]
                if j == i:
                    obs = np.where(ins, vals, UNSEEN).astype(np.uint8)
            rgb_img = np.array(Image.open(root / sd["filename"]).convert("RGB"))
            rgb, ins = sample_camera(xy, cam, rgb_img)
            rgb = np.where(ins[..., None], rgb, 0).astype(np.uint8)

            gt = union.copy()
            box = draw_boxes(seen_boxes, origin, axes, G, a.res, cam)
            hidden = draw_boxes(hidden_boxes, origin, axes, G, a.res, cam)
            seg_veh = gt == VEH_C
            stats["seg_veh"] += seg_veh.sum()
            stats["seg_veh_in_box"] += (seg_veh & box).sum()
            gt[seg_veh] = OTHER  # seg vehicles -> rely on boxes for the vehicle class
            # z* is an oracle over the swarm's observations, not world truth: a vehicle is
            # labelled only where some UAV sees the cell AND some camera actually sees the
            # vehicle; vehicles hidden from every camera (e.g. under a bridge) are ignored.
            in_view = seen.any(0)
            gt[hidden & in_view] = UNSEEN
            gt[box & in_view] = VEH_C

            np.savez_compressed(out_dir / f"{smp['token']}_{i}.npz", rgb=rgb, obs=obs, gt=gt,
                                seen=seen, origin=origin, axes=axes)
            stats["n"] += 1
            stats["ego_cov"] += seen[i].mean()
            stats["swarm_cov"] += seen.any(0).mean()
            stats["veh"] += (gt == VEH_C).mean()
            stats["road"] += (gt == ROAD_C).mean()
        if k % 20 == 0:
            print(f"  {k + 1}/{len(todo)} samples", flush=True)

    n = stats["n"]
    print(f"done: {int(n)} (sample, UAV) grids | ego coverage {stats['ego_cov'] / n:.1%} | "
          f"swarm coverage {stats['swarm_cov'] / n:.1%} | road {stats['road'] / n:.1%} | vehicle {stats['veh'] / n:.2%}")
    print(f"seg-vehicle cells inside a box: {stats['seg_veh_in_box'] / max(1, stats['seg_veh']):.1%} "
          "(low -> vehicles without boxes, e.g. static parked cars)")
    print(f"boxes inside some image: {int(stats['boxes_in_image'])}, of which hidden from every camera "
          f"(e.g. under a bridge; ignored in gt): {stats['boxes_hidden_in_image'] / max(1, stats['boxes_in_image']):.1%}")


if __name__ == "__main__":
    main()
