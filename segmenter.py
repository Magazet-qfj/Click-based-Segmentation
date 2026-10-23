"""GrabCut-based interactive segmentation engine.

Converts a set of user clicks (positive = foreground, negative = background)
into a binary segmentation mask using OpenCV's GrabCut algorithm.

We use the *scribble* flavour of GrabCut: the whole image starts out as
"probable background", and each click acts as a hard foreground/background
seed. GrabCut grows the foreground from those seeds. This needs no bounding
box and runs in tens of milliseconds on CPU, so it suits live click-driven
editing (the more fg/bg clicks the user adds, the more precise the result).
"""

import cv2
import numpy as np


class GrabCutSegmenter:
    """Click-driven foreground/background separation via ``cv2.grabCut``."""

    # GrabCut mask constants (same values as cv2.GC_*).
    GC_BGD = 0
    GC_FGD = 1
    GC_PR_BGD = 2
    GC_PR_FGD = 3

    def __init__(self, iter_count=5, max_side=800, seed_radius=5):
        self.iter_count = iter_count
        self.max_side = max_side      # downscale the working image to this longest side
        self.seed_radius = seed_radius

        self.image = None             # original RGB image (H, W, 3), uint8
        self._work_image = None       # downscaled BGR image used by grabCut
        self._scale = 1.0             # original_size -> work_size scale factor
        self._pos_clicks = []         # (x, y) in ORIGINAL image coordinates
        self._neg_clicks = []
        self._mask = None             # binary mask at original resolution

    # ------------------------------------------------------------------ image
    def set_image(self, image):
        """Set the RGB image and clear all segmentation state."""
        self.image = image
        h, w = image.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        self._scale = scale
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if scale < 1.0:
            self._work_image = cv2.resize(
                bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        else:
            self._work_image = bgr
        self._pos_clicks = []
        self._neg_clicks = []
        self._mask = None

    # ----------------------------------------------------------------- clicks
    def add_click(self, x, y, is_positive):
        if is_positive:
            self._pos_clicks.append((x, y))
        else:
            self._neg_clicks.append((x, y))

    def undo_click(self):
        # Undo the most recent click (negative clicks are undone first).
        if self._neg_clicks:
            self._neg_clicks.pop()
        elif self._pos_clicks:
            self._pos_clicks.pop()

    def reset(self):
        self._pos_clicks = []
        self._neg_clicks = []
        self._mask = None

    # ------------------------------------------------------------ segmentation
    def run(self):
        """(Re)compute the binary mask from the current clicks and return it."""
        if self.image is None or not self._pos_clicks:
            self._mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            return self._mask

        h, w = self._work_image.shape[:2]

        # Scribble initialization: everything is probable background; the clicks
        # are painted on top as definite foreground / background seeds.
        mask = np.full((h, w), np.uint8(self.GC_PR_BGD), dtype=np.uint8)
        for cx, cy in self._pos_clicks:
            self._paint_seed(mask, cx, cy, self.GC_FGD)
        for cx, cy in self._neg_clicks:
            self._paint_seed(mask, cx, cy, self.GC_BGD)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(self._work_image, mask, None, bgd_model, fgd_model,
                    self.iter_count, cv2.GC_INIT_WITH_MASK)

        seg = np.where((mask == self.GC_FGD) | (mask == self.GC_PR_FGD), 1, 0).astype(np.uint8)
        self._mask = self._upscale(seg)
        return self._mask

    # ----------------------------------------------------------------- helpers
    def _paint_seed(self, mask, x, y, value):
        h, w = mask.shape[:2]
        cx, cy = int(round(x * self._scale)), int(round(y * self._scale))
        r = self.seed_radius
        mask[max(0, cy - r):min(h, cy + r + 1),
             max(0, cx - r):min(w, cx + r + 1)] = value

    def _upscale(self, seg):
        if self._scale == 1.0:
            return seg
        h, w = self.image.shape[:2]
        return cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)

    # -------------------------------------------------------------- properties
    @property
    def result_mask(self):
        return self._mask

    @property
    def has_clicks(self):
        return bool(self._pos_clicks or self._neg_clicks)

    @property
    def clicks(self):
        return self._pos_clicks, self._neg_clicks
