#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize depth, segmentation, and optical flow (forward/backward) saved as .npy.

Usage:
  python scripts/sim_data/visualize_npy.py --root /path/to/output [--recursive]

Behavior:
- Searches recursively (or in-place) for files named:
  depth_*.npy, segmentation_*.npy, forward_flow_*.npy, backward_flow_*.npy
- Writes visualizations into the same folder:
  depth_vis_*.jpg, segmentation_vis_*.jpg, forward_flow_vis_*.jpg, backward_flow_vis_*.jpg

Notes:
- Depth: grayscale with robust min/max per folder (ignores 0, inf, nan),
  using percentiles for clipping to stabilize contrast.
- Segmentation: deterministic color per ID (hash-based), background=0 => dark.
- Flow: HSV color wheel (angle->hue, magnitude->value), magnitude clipped by p95.
"""

import argparse
import os
import re
import math
import numpy as np
from glob import glob
try:
    import matplotlib.cm as _mpl_cm  # for depth colormap
except Exception:  # optional
    _mpl_cm = None

try:  # Prefer lightweight imageio; fall back to skimage if present
    import imageio.v3 as iio
    def _write_image(path, arr):
        iio.imwrite(path, arr)
except Exception:
    try:
        from skimage import io as skio  # type: ignore
        def _write_image(path, arr):
            skio.imsave(path, arr)
    except Exception:
        raise RuntimeError("No suitable image writing library (imageio or skimage) available.")


DEPTH_RE = re.compile(r"depth_(\d{5})\.npy$")
SEGM_RE = re.compile(r"segmentation_(\d{5})\.npy$")
FFLW_RE = re.compile(r"forward_flow_(\d{5})\.npy$")
BFLW_RE = re.compile(r"backward_flow_(\d{5})\.npy$")


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def robust_minmax_depth(values: np.ndarray, pmin: float = 2.0, pmax: float = 98.0) -> tuple[float, float]:
    vals = values[np.isfinite(values) & (values > 0)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(vals, pmin))
    hi = float(np.percentile(vals, pmax))
    if not np.isfinite(lo):
        lo = float(np.nanmin(vals))
    if not np.isfinite(hi):
        hi = float(np.nanmax(vals))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def depth_to_rgb(depth: np.ndarray, lo: float, hi: float, sqrt: bool = True,
                 colormap: str = "viridis_r") -> np.ndarray:
    """Map depth to an RGB colormap (similar to MOVi notebook).

    - Accepts (H,W) or (H,W,1) float depths.
    - Normalizes into [0,1] using provided lo/hi.
    - Optional sqrt compression emphasizes near depths.
    - Uses matplotlib colormap if available; otherwise falls back to grayscale.
    """
    d = np.squeeze(depth).astype(np.float32)
    mask = np.isfinite(d) & (d > 0)
    out = np.zeros_like(d, dtype=np.float32)
    if mask.any():
        norm = (np.clip(d[mask], lo, hi) - lo) / max(1e-9, (hi - lo))
        if sqrt:
            norm = np.sqrt(norm)
        out[mask] = norm
    if _mpl_cm is not None:
        try:
            cmap = getattr(_mpl_cm, colormap)
        except AttributeError:
            cmap = _mpl_cm.get_cmap("viridis")
        # cmap expects indices 0..N-1; supply float in [0,1]
        rgb = cmap(out)[:, :, :3]  # (H,W,3)
        rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        return rgb
    # Fallback grayscale replication if matplotlib unavailable
    gray = (out * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def hash_color_for_id(i: int) -> tuple[int, int, int]:
    # Simple, deterministic color hash
    x = (i * 2654435761) & 0xFFFFFFFF
    r = (x & 0xFF)
    g = ((x >> 8) & 0xFF)
    b = ((x >> 16) & 0xFF)
    # Keep background (0) dark
    if i == 0:
        return (0, 0, 0)
    # Avoid too-dark colors
    if r + g + b < 60:
        r = (r + 64) & 0xFF
        g = (g + 64) & 0xFF
        b = (b + 64) & 0xFF
    return int(r), int(g), int(b)


def colorize_segmentation(seg: np.ndarray) -> np.ndarray:
    """Map segmentation array to RGB.
    Accepts shapes: (H,W), (H,W,1), or (H,W,C>1) where multi-channel vectors are treated as composite IDs.
    """
    if seg.ndim == 3:
        if seg.shape[2] == 1:  # (H,W,1) -> squeeze
            seg_ids = np.squeeze(seg, axis=2)
        else:  # combine multi-channel into unique integer IDs
            flat = seg.reshape(-1, seg.shape[2])
            try:
                view = flat.view(np.dtype((np.void, flat.dtype.itemsize * flat.shape[1])))
                _, inverse = np.unique(view, return_inverse=True)
                seg_ids = inverse.reshape(seg.shape[0], seg.shape[1])
            except Exception:
                seg_ids = seg[..., 0]  # fallback: first channel
    else:
        seg_ids = seg
    seg_ids = seg_ids.astype(np.int64)
    h, w = seg_ids.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in np.unique(seg_ids):
        r, g, b = hash_color_for_id(int(i))
        mask = (seg_ids == i)
        rgb[mask] = (r, g, b)
    return rgb


def flow_to_hsv_rgb(flow: np.ndarray, p_clip: float = 95.0) -> np.ndarray:
    """Convert flow (H, W, 2) to an HSV-based RGB visualization.
    Hue = angle, Value = magnitude (clipped), Saturation = 1.
    """
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("flow must have shape (H, W, 2)")
    u = flow[..., 0].astype(np.float32)
    v = flow[..., 1].astype(np.float32)
    mag = np.sqrt(u*u + v*v)
    ang = np.arctan2(v, u)  # (-pi, pi]
    hue = (ang + math.pi) / (2 * math.pi)  # [0,1)
    # Robust magnitude scaling
    mclip = float(np.percentile(mag, p_clip)) if np.isfinite(mag).any() else 1.0
    if not np.isfinite(mclip) or mclip <= 0:
        mclip = 1.0
    val = np.clip(mag / mclip, 0.0, 1.0)
    sat = np.ones_like(val, dtype=np.float32)
    # HSV -> RGB
    rgb = hsv_to_rgb_np(hue, sat, val)
    return (rgb * 255.0).astype(np.uint8)


def hsv_to_rgb_np(h, s, v):
    # Vectorized HSV->RGB (h in [0,1])
    i = np.floor(h * 6).astype(np.int32)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i_mod = np.mod(i, 6)

    r = np.choose(i_mod, [v, q, p, p, t, v])
    g = np.choose(i_mod, [t, v, v, q, p, p])
    b = np.choose(i_mod, [p, p, t, v, v, q])
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    return rgb


def scan_frames(root: str, recursive: bool):
    pattern = "**/*.npy" if recursive else "*.npy"
    files = glob(os.path.join(root, pattern), recursive=recursive)
    depth = {}
    segm = {}
    fflow = {}
    bflow = {}
    for f in files:
        nm = os.path.basename(f)
        d = DEPTH_RE.search(nm)
        if d:
            depth[d.group(1)] = f
            continue
        s = SEGM_RE.search(nm)
        if s:
            segm[s.group(1)] = f
            continue
        ff = FFLW_RE.search(nm)
        if ff:
            fflow[ff.group(1)] = f
            continue
        bf = BFLW_RE.search(nm)
        if bf:
            bflow[bf.group(1)] = f
            continue
    return depth, segm, fflow, bflow


def process_depth_group(files: dict[str, str]):
    """Visualize depth frames with a perceptual colormap.

    Strategy:
      1. Gather robust per-frame min/max (percentile-based) but aggregate with median
         to stabilize across outlier frames.
      2. For each frame, map depth to RGB using sqrt + viridis_r (MOVi style).
      3. Fallback to grayscale if matplotlib is not installed.
    """
    mins, maxs = [], []
    ordered = sorted(files.items())
    for _, path in ordered:
        try:
            arr = np.load(path)
            lo, hi = robust_minmax_depth(arr)
            mins.append(lo)
            maxs.append(hi)
        except Exception:
            continue
    if not mins:
        return
    # Aggregate range (median reduces impact of a single very close or far frame)
    lo_global = float(np.median(np.array(mins)))
    hi_global = float(np.median(np.array(maxs)))
    if hi_global <= lo_global:
        hi_global = lo_global + 1.0
    for idx, path in ordered:
        try:
            arr = np.load(path)
            rgb = depth_to_rgb(arr, lo_global, hi_global, sqrt=True, colormap="viridis_r")
            out = os.path.join(os.path.dirname(path), f"depth_vis_{idx}.jpg")
            _write_image(out, rgb)
        except Exception as e:
            print(f"[depth] failed {path}: {e}")


def process_segm_group(files: dict[str, str]):
    for idx, path in sorted(files.items()):
        try:
            seg = np.load(path)
            rgb = colorize_segmentation(seg)
            out = os.path.join(os.path.dirname(path), f"segmentation_vis_{idx}.jpg")
            _write_image(out, rgb)
        except Exception as e:
            print(f"[segm] failed {path}: {e}")


def process_flow_group(files: dict[str, str], kind: str):
    for idx, path in sorted(files.items()):
        try:
            flow = np.load(path)
            vis = flow_to_hsv_rgb(flow)
            out = os.path.join(os.path.dirname(path), f"{kind}_vis_{idx}.jpg")
            _write_image(out, vis)
        except Exception as e:
            print(f"[{kind}] failed {path}: {e}")


def main():
    ap = argparse.ArgumentParser(description="Visualize depth/segmentation/flow npy files")
    ap.add_argument("--root", type=str, required=True, help="Root directory (view folder or run output root)")
    ap.add_argument("--recursive", action="store_true", help="Search recursively for npy files")
    ap.add_argument("--only", type=str, default="all", choices=["all","depth","segm","flow","fwd","bwd"],
                    help="Which to visualize")
    args = ap.parse_args()

    depth, segm, fflow, bflow = scan_frames(args.root, args.recursive)

    if args.only in ("all", "depth"):
        process_depth_group(depth)
    if args.only in ("all", "segm"):
        process_segm_group(segm)
    if args.only in ("all", "flow", "fwd"):
        process_flow_group(fflow, "forward_flow")
    if args.only in ("all", "flow", "bwd"):
        process_flow_group(bflow, "backward_flow")


if __name__ == "__main__":
    main()
