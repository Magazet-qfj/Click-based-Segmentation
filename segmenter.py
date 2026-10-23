"""SAM-based interactive segmentation engine (replaces GrabCut).

Converts a set of user clicks (positive = foreground, negative = background)
into a binary segmentation mask using Meta's Segment Anything Model (ViT-B).

Why SAM instead of GrabCut: GrabCut's data term is colour-only, so it had no
notion of spatial connectivity (it would pull in any same-coloured region
anywhere in the image) and it re-fitted its global colour model on every click
(so a background click could flip previously-selected regions). SAM instead
encodes the whole image once with a ViT and, for each set of clicks, runs a tiny
prompt-driven mask decoder — giving a connected, click-stable mask in
milliseconds after the one-time image encode.

The heavy image encoder runs once per image inside ``set_image``; every
subsequent ``run`` only re-runs the cheap mask decoder, so interactive editing
stays responsive even on CPU. To keep that first encode fast on a CPU-only
machine we downscale the working copy to ``max_side`` before encoding, then
upsample the resulting mask back to full resolution (the same trick the
GrabCut engine used).

``result_mask`` stays binary (for the overlay / saving a hard mask) while
``result_alpha`` carries the soft 0..255 feathered matte used for cutting out.
A median filter + morphological close first removes the checkerboard and tiny
holes SAM's mask decoder sometimes leaves in smooth regions, so the cut-out has
no grid-like gaps.
"""

import cv2
import numpy as np


class SamSegmenter:
    """Click-driven foreground/background segmentation via Segment Anything."""

    def __init__(self, checkpoint, max_side=640, device='cpu', cleanup_kernel=9):
        self.checkpoint = checkpoint   # path to sam_vit_b_01ec64.pth
        self.max_side = max_side       # downscale the working image to this longest side
        self.device = device
        self.cleanup_kernel = cleanup_kernel  # median/close kernel for artifact cleanup

        self.image = None             # original RGB image (H, W, 3), uint8
        self._predictor = None        # SamPredictor, built lazily on first set_image
        self._work_image = None       # downscaled RGB image passed to set_image
        self._scale = 1.0             # original_size -> work_size scale factor
        self._pos_clicks = []         # (x, y) in ORIGINAL image coordinates
        self._neg_clicks = []
        self._soft_mask = None        # soft (0..1 float) mask at original resolution

    # ------------------------------------------------------------------ model
    def _load_model(self):
        """Build the SAM model + predictor once, on first use."""
        if self._predictor is not None:
            return
        print("[SamSegmenter] loading SAM ViT-B ...")
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry['vit_b'](checkpoint=self.checkpoint)
        sam.to(self.device)
        sam.eval()
        self._predictor = SamPredictor(sam)
        print("[SamSegmenter] model ready.")

    # ------------------------------------------------------------------ image
    def set_image(self, image):
        """Set the RGB image, encode it once, and clear all segmentation state."""
        self.image = image
        self._load_model()

        h, w = image.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        self._scale = scale
        if scale < 1.0:
            self._work_image = cv2.resize(
                image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        else:
            self._work_image = image

        self._predictor.set_image(self._work_image)  # the slow step (image encode)
        self._pos_clicks = []
        self._neg_clicks = []
        self._soft_mask = None

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
        self._soft_mask = None

    # ------------------------------------------------------------ segmentation
    def run(self):
        """(Re)compute the mask from the current clicks, return the binary one."""
        if self.image is None:
            self._soft_mask = None
            return None
        if not self._pos_clicks:
            self._soft_mask = np.zeros(self.image.shape[:2], dtype=np.float32)
            return self.result_mask

        # Build point prompts from every click (foreground label=1, background=0),
        # scaled into the working image's coordinate space.
        coords, labels = [], []
        for x, y in self._pos_clicks:
            coords.append([x * self._scale, y * self._scale])
            labels.append(1)
        for x, y in self._neg_clicks:
            coords.append([x * self._scale, y * self._scale])
            labels.append(0)

        point_coords = np.asarray(coords, dtype=np.float32)   # (N, 2)
        point_labels = np.asarray(labels, dtype=np.int64)     # (N,)

        masks, scores, logits = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )
        seg = masks[0].astype(np.uint8)  # 0/1 at working resolution

        # Clean up checkerboard / holes at the working resolution first (cheap,
        # and where the artifacts are smallest), then upscale and feather.
        seg = self._cleanup_mask(seg)

        if self._scale < 1.0:
            h, w = self.image.shape[:2]
            seg = cv2.resize(seg.astype(np.float32), (w, h),
                             interpolation=cv2.INTER_LINEAR)
        else:
            seg = seg.astype(np.float32)

        self._soft_mask = self._feather(seg)
        return self.result_mask

    # ----------------------------------------------------------------- helpers
    def _cleanup_mask(self, mask):
        """Remove SAM's small mask artifacts at working resolution.

        The mask decoder's transposed-convolution upsampling can leave a faint
        checkerboard in smooth, same-coloured regions, plus occasional tiny
        holes. A median filter kills the alternating checkerboard / salt-and-
        pepper noise, then a morphological close fills remaining small holes and
        bridges thin gaps. Both run at the small working resolution, where these
        artifacts are only a few pixels wide and cheap to remove.

        The kernel size is ``self.cleanup_kernel`` (odd); raise it if a coarser
        grid remains, lower it to preserve very fine structures.
        """
        m = mask.astype(np.uint8)
        k = self.cleanup_kernel
        if k % 2 == 0:
            k += 1
        m = cv2.medianBlur(m, k)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    def _feather(self, soft):
        """Soften the mask edge into a gradual alpha ramp (anti-aliasing)."""
        h, w = soft.shape[:2]
        radius = max(3, int(round(min(h, w) / 300)))
        if radius % 2 == 0:
            radius += 1
        soft = cv2.GaussianBlur(soft, (radius, radius), 0)
        return np.clip(soft, 0.0, 1.0)

    # -------------------------------------------------------------- properties
    @property
    def result_mask(self):
        """Binary mask (0/1), for the overlay and for saving a hard mask."""
        if self._soft_mask is None:
            return None
        return (self._soft_mask >= 0.5).astype(np.uint8)

    @property
    def result_alpha(self):
        """Soft alpha (0..255) with feathered edges, for cutting out the object."""
        if self._soft_mask is None:
            return None
        return (self._soft_mask * 255.0 + 0.5).astype(np.uint8)

    @property
    def has_clicks(self):
        return bool(self._pos_clicks or self._neg_clicks)

    @property
    def clicks(self):
        return self._pos_clicks, self._neg_clicks
