"""Headless evaluation: measure "number of clicks to reach an IoU threshold" (NoC).

Replays an automatic click policy against (image, ground-truth mask) pairs and
reports the IoU-vs-clicks curve plus NoC@80/85/90. The policy is the classic
interactive-segmentation click sampler: at each step place a click at the
centroid of the largest error region (false negative -> foreground click,
false positive -> background click), then re-run the segmenter.

This is the quantitative baseline for "fewer clicks, finer masks": run it
before and after each engine change to see the click budget drop.

Usage:
    python eval_click.py --image a.png --gt a_gt.png
    python eval_click.py --image a.png --gt a_gt.png --init-box   # first prompt is a GT bbox
    python eval_click.py --image a.png --gt a_gt.png --max-side 640
    python eval_click.py --image a.png --gt a_gt.png --edge-radius 3

Ground-truth masks are 0/1 (anything >127 counts as foreground). For real
numbers, supply masks from a benchmark (GrabCut / Berkeley / DAVIS).

Besides NoC (how fast the mask interior converges), the final mask also reports
Boundary IoU and F1 over a thin band around the two contours — edge quality,
i.e. how precisely the cut-out edge tracks the object outline. NoC alone cannot
capture that: two masks can reach 0.95 IoU yet have very different boundary
fidelity.
"""

import argparse
import os
import time

import cv2
import numpy as np

from segmenter import SamSegmenter


# -------------------------------------------------------------------- io
def _read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_gt(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"cannot read gt: {path}")
    return (img > 127).astype(np.uint8)


# ---------------------------------------------------------------- metrics
def _iou(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def _boundary_metrics(pred, gt, radius=2):
    """Edge quality of the final mask: Boundary IoU and boundary F1 over a
    ``radius``-wide band around the two mask contours (how closely the cut-out
    edge tracks the ground-truth edge, independent of the mask interior)."""
    pred_b = pred.astype(np.uint8)
    gt_b = gt.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    pred_bound = (cv2.dilate(pred_b, k) - cv2.erode(pred_b, k)) > 0
    gt_bound = (cv2.dilate(gt_b, k) - cv2.erode(gt_b, k)) > 0
    inter = int(np.logical_and(pred_bound, gt_bound).sum())
    pred_sum = int(pred_bound.sum())
    gt_sum = int(gt_bound.sum())
    union = int(np.logical_or(pred_bound, gt_bound).sum())

    biou = inter / union if union else 0.0
    precision = inter / pred_sum if pred_sum else 0.0
    recall = inter / gt_sum if gt_sum else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return biou, f1


def _largest_error_region(pred, gt):
    """Centroid of the largest error region, or None if the mask matches GT.

    Returns (cx, cy, is_positive). False negatives (missed foreground) get a
    positive click; false positives (extra background) get a negative click.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    best = None  # (area, cx, cy, is_positive)
    for region, is_pos in [(gt & ~pred, True), (pred & ~gt, False)]:
        n, _, stats, cents = cv2.connectedComponentsWithStats(
            region.astype(np.uint8), connectivity=8
        )
        for i in range(1, n):  # skip the background label (index 0)
            area = int(stats[i, cv2.CC_STAT_AREA])
            if best is None or area > best[0]:
                best = (area, float(cents[i][0]), float(cents[i][1]), is_pos)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _noc(iou_list, threshold):
    """1-indexed click count at which IoU first reaches ``threshold``."""
    for i, v in enumerate(iou_list):
        if v >= threshold:
            return i + 1
    return len(iou_list)


# -------------------------------------------------------------- evaluation
def evaluate_one(segmenter, gt, max_clicks=20, init_box=False):
    """Return the IoU list after each interaction (box counts as interaction 0)."""
    segmenter.reset()
    ious = []

    if init_box:
        ys, xs = np.where(gt > 0)
        if len(xs) == 0:
            return [0.0]
        segmenter.set_box(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        segmenter.run()
        ious.append(_iou(segmenter.result_mask, gt))

    while len(ious) < max_clicks:
        pred = segmenter.result_mask
        if pred is None:
            pred = np.zeros(gt.shape, dtype=np.uint8)
        err = _largest_error_region(pred, gt)
        if err is None:
            break
        cx, cy, is_pos = err
        segmenter.add_click(int(cx), int(cy), is_pos)
        segmenter.run()
        ious.append(_iou(segmenter.result_mask, gt))
        if ious[-1] >= 0.95:
            break
    return ious


def main():
    ap = argparse.ArgumentParser(description="NoC / IoU-vs-clicks evaluation for the SAM app.")
    ap.add_argument('--image', required=True, help='Path to an input image.')
    ap.add_argument('--gt', required=True, help='Path to the 0/1 ground-truth mask.')
    ap.add_argument('--checkpoint', default='', help='SAM checkpoint (default: weights/sam_vit_b_01ec64.pth).')
    ap.add_argument('--max-clicks', type=int, default=20, help='Click budget per image.')
    ap.add_argument('--max-side', type=int, default=1024, help='Working-image longest side.')
    ap.add_argument('--init-box', action='store_true', help='Start from a GT-bbox prompt instead of empty.')
    ap.add_argument('--edge-radius', type=int, default=2, help='Band half-width (px) for boundary IoU/F1.')
    args = ap.parse_args()

    if not args.checkpoint:
        here = os.path.dirname(os.path.abspath(__file__))
        args.checkpoint = os.path.join(here, 'weights', 'sam_vit_b_01ec64.pth')

    image = _read_image(args.image)
    gt = _read_gt(args.gt)

    seg = SamSegmenter(checkpoint=args.checkpoint, max_side=args.max_side)
    t0 = time.time()
    seg.set_image(image)
    print(f"encode took {time.time() - t0:.1f}s  (image {image.shape[1]}x{image.shape[0]})")

    ious = evaluate_one(seg, gt, max_clicks=args.max_clicks, init_box=args.init_box)

    print("\nIoU vs interactions:")
    for i, v in enumerate(ious):
        print(f"  {'box' if (args.init_box and i == 0) else 'click ' + str(i - (1 if args.init_box else 0)):>8}  IoU={v:.4f}")

    print("\nNoC (interactions to reach threshold):")
    for th in (80, 85, 90):
        print(f"  @{th}%: {_noc(ious, th / 100.0)}")

    pred = seg.result_mask
    if pred is not None:
        biou, bf1 = _boundary_metrics(pred, gt, radius=args.edge_radius)
        print(f"\nEdge quality (band {args.edge_radius}px):  BoundaryIoU={biou:.4f}  F1={bf1:.4f}")


if __name__ == '__main__':
    main()
