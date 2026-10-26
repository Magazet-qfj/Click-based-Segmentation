"""Synthetic benchmark for P0: complex/hair-like edges with *exact* GT masks.

Real benchmarks (GrabCut / Berkeley / DAVIS) aren't reachable on this machine and
hand-annotating the photos needs vision we don't have here, so P0 uses synthetic
images whose ground-truth boundary is known *exactly*. Each case stresses a
different kind of complex edge that a binary mask + naive blur handles poorly:

  - starburst : a disk with thin radiating strands (hair-like filaments)
  - wavy blob : a blob whose radius is sinusoidally perturbed (many fine bumps)
  - comb      : a bar with parallel thin teeth (comb / fringe)

The generated images + GT are saved under ``bench/`` so they can be inspected,
then the same automatic click policy as ``eval_click.py`` is replayed to report
NoC@85/90 (mask-interior convergence) and Boundary IoU / F1 (edge precision) —
the two numbers P2/P3 must move.

Usage:
    python bench_synth.py [--checkpoint weights/sam_vit_b_01ec64.pth] [--max-clicks 20]
"""

import argparse
import os
import time

import cv2
import numpy as np

from segmenter import SamSegmenter
from eval_click import evaluate_one, _noc, _boundary_metrics


# --------------------------------------------------------------- generators
def _texture(gt, fg, bg, seed=0):
    """Paint ``gt`` (0/1) as ``fg`` over ``bg`` with a gradient + mild noise so
    SAM has some texture to latch onto (flat colours make it hallucinate)."""
    size = gt.shape[0]
    rng = np.random.default_rng(seed)
    fg = np.asarray(fg, dtype=np.float32)
    bg = np.asarray(bg, dtype=np.float32)
    xx = np.arange(size, dtype=np.float32)[None, :]
    grad = (xx / size)[..., None] * 0.35  # horizontal brightness ramp
    img = bg[None, None, :] * (0.85 + grad)
    m = gt[..., None].astype(np.float32)
    img = img * (1.0 - m) + fg[None, None, :] * m
    img += rng.standard_normal((size, size, 1)) * 12.0
    return np.clip(img, 0, 255).astype(np.uint8)


def _starburst(size=512, max_frac=0.45):
    """Central disk + thin radiating strands (hair-like); strands reach
    ``max_frac`` of the image half-width."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = size / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    theta = np.arctan2(y - cy, x - cx)
    gt = (r < size * 0.16).astype(np.uint8)
    for k in range(18):
        ang = 2.0 * np.pi * k / 18.0
        dtheta = np.abs(((theta - ang + np.pi) % (2 * np.pi)) - np.pi)
        strand = (r > size * 0.16) & (r < size * max_frac) & (dtheta < 0.02)
        gt |= strand.astype(np.uint8)
    return gt


def _wavy_blob(size=512):
    """Blob whose boundary is sinusoidally perturbed (many fine bumps)."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = size / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    theta = np.arctan2(y - cy, x - cx)
    boundary = size * 0.24 * (1.0 + 0.18 * np.sin(9 * theta) + 0.06 * np.cos(17 * theta))
    return (r < boundary).astype(np.uint8)


def _comb(size=512):
    """A bar with parallel thin teeth (fringe)."""
    gt = np.zeros((size, size), np.uint8)
    gt[size // 3:size // 2, size // 5:4 * size // 5] = 1
    n = 14
    span = 4 * size // 5 - size // 5
    for i in range(n):
        x0 = size // 5 + i * span // n
        gt[size // 3:2 * size // 3, x0:x0 + 3] = 1
    return gt


CASES = [
    ("starburst", _starburst(), (210, 120, 70), (70, 120, 210)),
    ("wavy_blob", _wavy_blob(), (60, 180, 120), (180, 90, 60)),
    ("comb", _comb(), (230, 200, 60), (60, 80, 160)),
    # Large variant: the whole image is downscaled to 1024 for the coarse mask,
    # so zoom-in has real resolution headroom to recover on the thin strands.
    ("starburst_2k", _starburst(2048), (210, 120, 70), (70, 120, 210)),
    # Small object in a large image — the case zoom-in is actually for: the
    # strands are tiny in the 2048 image, so the coarse 1024 encode blurs them.
    ("starburst_small", _starburst(2048, max_frac=0.22), (210, 120, 70), (70, 120, 210)),
]


def main():
    ap = argparse.ArgumentParser(description="Synthetic P0 baseline (NoC + boundary metrics).")
    ap.add_argument('--checkpoint', default='', help='SAM checkpoint (default weights/sam_vit_b_01ec64.pth).')
    ap.add_argument('--max-clicks', type=int, default=20, help='Click budget per case.')
    ap.add_argument('--max-side', type=int, default=1024, help='Working-image longest side.')
    ap.add_argument('--edge-radius', type=int, default=2, help='Band half-width for boundary IoU/F1.')
    ap.add_argument('--refine', action='store_true', help='One-shot zoom-in refinement after the click loop.')
    ap.add_argument('--out', default='bench', help='Directory to save generated images + GT.')
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if not args.checkpoint:
        args.checkpoint = os.path.join(here, 'weights', 'sam_vit_b_01ec64.pth')

    img_dir = os.path.join(args.out, 'images')
    mask_dir = os.path.join(args.out, 'masks')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    seg = SamSegmenter(checkpoint=args.checkpoint, max_side=args.max_side, zoom_in=args.refine)

    rows = []
    for name, gt, fg, bg in CASES:
        image = _texture(gt, fg, bg)
        cv2.imwrite(os.path.join(img_dir, f"{name}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(mask_dir, f"{name}_gt.png"), gt * 255)

        t0 = time.time()
        seg.set_image(image)
        ious = evaluate_one(seg, gt, max_clicks=args.max_clicks)
        noc85 = _noc(ious, 0.85)
        noc90 = _noc(ious, 0.90)
        biou, bf1 = _boundary_metrics(seg.result_mask, gt, radius=args.edge_radius)
        rbiou = rbf1 = float('nan')
        if args.refine:
            seg.refine_mask()
            rbiou, rbf1 = _boundary_metrics(seg.result_mask, gt, radius=args.edge_radius)
        dt = time.time() - t0
        rows.append((name, len(ious), ious[-1], noc85, noc90, biou, bf1, rbiou, rbf1, dt))
        extra = f"  -> refined BIoU={rbiou:.3f} F1={rbf1:.3f}" if args.refine else ""
        print(f"[{name}] clicks={len(ious)}  finalIoU={ious[-1]:.3f}  "
              f"NoC85={noc85}  NoC90={noc90}  BIoU={biou:.3f}  F1={bf1:.3f}{extra}  ({dt:.1f}s)")

    print(f"\n=== synthetic baseline (refine={args.refine}) ===")
    hdr = f"{'case':<11} {'clicks':>6} {'IoU':>6} {'NoC85':>6} {'NoC90':>6} {'BIoU':>7} {'F1':>7}"
    if args.refine:
        hdr += f" {'rBIoU':>7} {'rF1':>7}"
    print(hdr)
    for name, n, iou, n85, n90, biou, bf1, rbiou, rbf1, _ in rows:
        line = f"{name:<11} {n:>6} {iou:>6.3f} {n85:>6} {n90:>6} {biou:>7.3f} {bf1:>7.3f}"
        if args.refine:
            line += f" {rbiou:>7.3f} {rbf1:>7.3f}"
        print(line)
    print(f"\nGenerated {len(CASES)} cases under '{args.out}/'.")


if __name__ == '__main__':
    main()
