"""Entry point for the click-based interactive segmentation app.

Usage:
    python main.py                 # start with an empty window, load an image via the menu
    python main.py --image foo.jpg # start with foo.jpg already loaded
"""

import argparse
import tkinter as tk

from app import InteractiveDemoApp
from segmenter import GrabCutSegmenter


def parse_args():
    parser = argparse.ArgumentParser(description="Click-based interactive segmentation (GrabCut).")
    parser.add_argument('--image', type=str, default='',
                        help='Optional path to an image to load at startup.')
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    segmenter = GrabCutSegmenter()
    app = InteractiveDemoApp(root, segmenter, initial_image=args.image)
    app.pack(fill='both', expand=True)
    root.mainloop()


if __name__ == '__main__':
    main()
