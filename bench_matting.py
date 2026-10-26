"""P3 evaluation: closed-form matting vs guided filter vs binary alpha.

SAM outputs a *binary* mask; the question P3 answers is how well each alpha
source recovers the true fractional alpha at a semi-transparent (feathered)
boundary. We isolate the matting step by feeding every method the *correct*
binary mask (``alpha_gt >= 0.5``), so the only difference is the soft-alpha
reconstruction — not SAM's own mask error.

Usage:
    python bench_matting.py [--size 512] [--ramp 6]
"""

import argparse
import time

import cv2
import numpy as np

from matting import estimate_alpha
from segmenter import SamSegmenter  # reuses its guided filter


def _feathered_edge(size=512, ramp=6):
    """A disk with a linear alpha ramp at its edge (semi-transparent ring)."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.sqrt((x - size / 2) ** 2 + (y - size / 2) ** 2)
    return np.clip((size * 0.32 - r) / ramp, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description="P3 matting vs guided vs binary alpha.")
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--ramp', type=float, default=6.0, help='Alpha ramp width in px.')
    ap.add_argument('--trimap-radius', type=int, default=10)
    args = ap.parse_args()

    alpha_gt = _feathered_edge(args.size, args.ramp)
    fg = np.array([215, 60, 55], np.float32)
    bg = np.array([45, 200, 120], np.float32)
    image = (alpha_gt[..., None] * fg + (1 - alpha_gt[..., None]) * bg).astype(np.uint8)
    binary = (alpha_gt >= 0.5).astype(np.uint8)

    ring = (alpha_gt > 0.02) & (alpha_gt < 0.98)   # only the semi-transparent band

    def mae(a):
        return float(np.abs(a[ring] - alpha_gt[ring]).mean())

    guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    a_gf = SamSegmenter._guided_filter(guide, binary.astype(np.float32), 4, 1e-4)

    t0 = time.time()
    a_m = estimate_alpha(image, binary, trimap_radius=args.trimap_radius,
                         max_side=args.size)
    dt = time.time() - t0

    print(f"semi-transparent band (ramp {args.ramp}px, trimap radius {args.trimap_radius}):")
    print(f"  binary        alpha MAE = {mae(binary.astype(np.float32)):.4f}")
    print(f"  guided filter alpha MAE = {mae(a_gf):.4f}")
    print(f"  closed-form   alpha MAE = {mae(a_m):.4f}   (solve {dt:.1f}s)")


if __name__ == '__main__':
    main()
