"""Entry point for the click-based interactive segmentation app.

Usage:
    python main.py                 # start with an empty window, load an image via the menu
    python main.py --image foo.jpg # start with foo.jpg already loaded
    python main.py --checkpoint path/to/sam_vit_b_01ec64.pth   # custom weights path
"""

import argparse
import os
import tkinter as tk

from app import InteractiveDemoApp
from segmenter import SamSegmenter


def default_checkpoint():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, 'weights', 'sam_vit_b_01ec64.pth')


def parse_args():
    parser = argparse.ArgumentParser(description="Click-based interactive segmentation (SAM).")
    parser.add_argument('--image', type=str, default='',
                        help='Optional path to an image to load at startup.')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to the SAM ViT-B checkpoint (default: weights/sam_vit_b_01ec64.pth).')
    parser.add_argument('--cleanup', type=int, default=9,
                        help='Mask cleanup kernel size (odd; median/close). Larger fills bigger grid holes, '
                             'smaller preserves finer detail. Default 9.')
    parser.add_argument('--max-side', type=int, default=1024,
                        help='Cap the working image at this longest side before encoding '
                             '(SAM native = 1024; lower = less memory but coarser masks).')
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = args.checkpoint or default_checkpoint()
    root = tk.Tk()
    segmenter = SamSegmenter(checkpoint=checkpoint, cleanup_kernel=args.cleanup,
                             max_side=args.max_side)
    app = InteractiveDemoApp(root, segmenter, initial_image=args.image)
    app.pack(fill='both', expand=True)
    root.mainloop()


if __name__ == '__main__':
    main()
