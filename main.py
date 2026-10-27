"""Entry point for the click-based interactive segmentation app.

Usage:
    python main.py                 # start with an empty window, load an image via the menu
    python main.py --image foo.jpg # start with foo.jpg already loaded
    python main.py --checkpoint path/to/sam_vit_b_01ec64.pth   # custom weights path
"""

import argparse
import logging
import os
import sys
import tkinter as tk

from app import InteractiveDemoApp
from segmenter import SamSegmenter


def _setup_logging():
    """Route INFO-level logs to the console so the user can watch what the app
    is doing (image load, each click, each decode) without a debugger."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if root.handlers:
        return  # already configured (e.g. re-imported)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)-7s %(message)s', datefmt='%H:%M:%S'))
    root.addHandler(handler)


SAM_CHECKPOINTS = {
    'vit_b': 'sam_vit_b_01ec64.pth',
    'vit_l': 'sam_vit_l_0b3195.pth',
}


def default_checkpoint(model_type):
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, 'weights', SAM_CHECKPOINTS[model_type])


def parse_args():
    parser = argparse.ArgumentParser(description="Click-based interactive segmentation (SAM).")
    parser.add_argument('--image', type=str, default='',
                        help='Optional path to an image to load at startup.')
    parser.add_argument('--model-type', type=str, default='vit_b', choices=('vit_b', 'vit_l'),
                        help='SAM encoder size: vit_b (fast, ~375MB, default) or vit_l '
                             '(heavier, ~1.25GB, ~2x encode; helps some thin-structure cases but not uniformly finer).')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to the SAM checkpoint (default: weights/<name> per --model-type).')
    parser.add_argument('--cleanup', type=int, default=9,
                        help='Mask cleanup hole-fill kernel size (odd). Larger fills bigger grid holes, '
                             'smaller preserves finer detail. Default 9. The checkerboard median is '
                             'always 3px (a bigger median erases small round foreground details).')
    parser.add_argument('--max-side', type=int, default=1024,
                        help='Cap the working image at this longest side before encoding '
                             '(SAM native = 1024; lower = less memory but coarser masks).')
    parser.add_argument('--refine', type=str, default='off', choices=('off', 'grabcut'),
                        help='Boundary refinement: off (default) or grabcut (full-res colour '
                             "refinement that recovers detail SAM's 256x256 decoder blurs).")
    return parser.parse_args()


def main():
    _setup_logging()
    args = parse_args()
    checkpoint = args.checkpoint or default_checkpoint(args.model_type)
    root = tk.Tk()
    segmenter = SamSegmenter(checkpoint=checkpoint, model_type=args.model_type,
                             cleanup_kernel=args.cleanup, max_side=args.max_side,
                             refine_mode=args.refine)
    app = InteractiveDemoApp(root, segmenter, initial_image=args.image)
    app.pack(fill='both', expand=True)
    root.mainloop()


if __name__ == '__main__':
    main()
