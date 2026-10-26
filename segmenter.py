"""SAM-based interactive segmentation engine (replaces GrabCut).

Converts a set of user clicks (positive = foreground, negative = background)
and/or a bounding box into a binary segmentation mask using Meta's Segment
Anything Model (ViT-B).

Why SAM instead of GrabCut: GrabCut's data term is colour-only, so it had no
notion of spatial connectivity (it would pull in any same-coloured region
anywhere in the image) and it re-fitted its global colour model on every click
(so a background click could flip previously-selected regions). SAM instead
encodes the whole image once with a ViT and, for each set of clicks, runs a tiny
prompt-driven mask decoder — giving a connected, click-stable mask in
milliseconds after the one-time image encode.

The heavy image encoder runs once per image inside ``set_image``; every
subsequent ``run`` only re-runs the cheap mask decoder, so interactive editing
stays responsive even on CPU. SAM's image encoder always runs at its native
1024px resolution no matter the input size, so we feed it near-native detail by
capping the working copy at ``max_side`` (default 1024, SAM's own resolution)
rather than the old 640 cap, which needlessly blurred fine edges. The mask is
then upsampled back to full resolution.

``result_mask`` stays binary (for the overlay / saving a hard mask) while
``result_alpha`` carries the soft 0..255 feathered matte used for cutting out.
A median filter + morphological close first removes the checkerboard and tiny
holes SAM's mask decoder sometimes leaves in smooth regions, so the cut-out has
no grid-like gaps. The binary mask is then converted to a soft alpha by an
edge-aware guided filter (``edge_mode='guided'``), which keeps the alpha ramp
hugging image edges — hair strands stay thin rather than blurring — instead of
a uniform Gaussian blur.
"""

import os

import cv2
import numpy as np


class SamSegmenter:
    """Click-driven foreground/background segmentation via Segment Anything."""

    def __init__(self, checkpoint, max_side=1024, device='cpu', cleanup_kernel=9,
                 edge_mode='guided', edge_radius=4, edge_eps=1e-4,
                 zoom_in=False, zoom_margin=0.4):
        self.checkpoint = checkpoint   # path to sam_vit_b_01ec64.pth
        self.max_side = max_side       # cap the working image at this longest side (SAM native = 1024)
        self.device = device
        self.cleanup_kernel = cleanup_kernel  # median/close kernel for artifact cleanup
        self.edge_mode = edge_mode     # 'guided' (edge-aware) or 'gaussian' (uniform blur)
        self.edge_radius = edge_radius # guided-filter window radius at full resolution
        self.edge_eps = edge_eps       # guided-filter regularizer (for 0..1 guide/src)
        self.zoom_in = zoom_in         # local native-res refinement of the object region
        self.zoom_margin = zoom_margin # extra padding around the object bbox (fraction)

        self.image = None             # original RGB image (H, W, 3), uint8
        self._predictor = None        # SamPredictor for the whole image (built lazily)
        self._sam_model = None        # shared SAM model (the crop predictor reuses it)
        self._crop_predictor = None   # second predictor on a tight crop (built lazily)
        self._work_image = None       # downscaled RGB image passed to set_image
        self._scale = 1.0             # original_size -> work_size scale factor
        self._pos_clicks = []         # (x, y) in ORIGINAL image coordinates
        self._neg_clicks = []
        self._box = None              # (x1, y1, x2, y2) in ORIGINAL image coordinates
        self._soft_mask = None        # soft (0..1 float) mask at original resolution

    # ------------------------------------------------------------------ model
    def _load_model(self):
        """Build the SAM model + predictor once, on first use."""
        if self._predictor is not None:
            return
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(
                f"SAM 权重文件不存在：{self.checkpoint}\n"
                f"请下载 sam_vit_b_01ec64.pth 放到 weights/ 目录（见 README）。")
        print("[SamSegmenter] loading SAM ViT-B ...")
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry['vit_b'](checkpoint=self.checkpoint)
        sam.to(self.device)
        sam.eval()
        self._sam_model = sam
        self._predictor = SamPredictor(sam)
        print("[SamSegmenter] model ready.")

    # ------------------------------------------------------------------ image
    def set_image(self, image):
        """Set the RGB image, encode it once, and clear all segmentation state."""
        self._load_model()  # raises FileNotFoundError if the checkpoint is missing
        self.image = image

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
        self._box = None
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
        self._box = None
        self._soft_mask = None

    # ------------------------------------------------------------------- box
    def set_box(self, x1, y1, x2, y2):
        """Set a bounding-box prompt (XYXY, in ORIGINAL image coordinates)."""
        self._box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def clear_box(self):
        self._box = None

    # ------------------------------------------------------------ segmentation
    def run(self):
        """(Re)compute the mask from the current clicks/box, return the binary one."""
        if self.image is None:
            self._soft_mask = None
            return None
        if not self._pos_clicks and self._box is None:
            self._soft_mask = np.zeros(self.image.shape[:2], dtype=np.float32)
            return self.result_mask

        self._soft_mask = self._feather(self._run_coarse())
        return self.result_mask

    def _build_prompts(self, offset_x, offset_y, scale):
        """Point/box prompts in a target image's coordinates; ``offset`` + ``scale``
        map original-image coordinates into that image."""
        coords, labels = [], []
        for x, y in self._pos_clicks:
            coords.append([(x - offset_x) * scale, (y - offset_y) * scale])
            labels.append(1)
        for x, y in self._neg_clicks:
            coords.append([(x - offset_x) * scale, (y - offset_y) * scale])
            labels.append(0)
        point_coords = np.asarray(coords, dtype=np.float32) if coords else None   # (N, 2)
        point_labels = np.asarray(labels, dtype=np.int64) if labels else None     # (N,)
        box = None
        if self._box is not None:
            x1, y1, x2, y2 = self._box
            box = np.asarray([(x1 - offset_x) * scale, (y1 - offset_y) * scale,
                              (x2 - offset_x) * scale, (y2 - offset_y) * scale], dtype=np.float32)
        return point_coords, point_labels, box

    def _decode(self, predictor, offset_x, offset_y, scale, mask_input=None):
        """Run the mask decoder and keep the best-scoring candidate (0/1 uint8).

        ``mask_input`` (optional) is a 256x256 float logits map fed back as a
        mask prompt, so a crop re-decode *refines* the coarse mask instead of
        re-inferring a fresh (and over-segmented) mask with less context.
        """
        pc, pl, box = self._build_prompts(offset_x, offset_y, scale)
        masks, scores, _ = predictor.predict(
            point_coords=pc, point_labels=pl, box=box, mask_input=mask_input,
            multimask_output=True)
        return masks[np.argmax(scores)].astype(np.uint8)

    def _upsample(self, seg, out_h, out_w):
        seg = seg.astype(np.float32)
        if seg.shape[0] != out_h or seg.shape[1] != out_w:
            seg = cv2.resize(seg, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return seg

    def _resize_work(self, img, scale):
        if scale >= 1.0:
            return img
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def _run_coarse(self):
        """Decode the whole working image, clean up, and upsample to full res."""
        seg = self._decode(self._predictor, 0.0, 0.0, self._scale)
        seg = self._cleanup_mask(seg)
        return self._upsample(seg, *self.image.shape[:2])

    def refine_mask(self):
        """One-shot zoom-in refinement (for export, not the live loop).

        Re-decodes a tight crop around the current object (mask bbox ∪ all click
        points ∪ box, plus margin) at finer-than-whole-image resolution, so small
        objects in large photos get sharper edges. Including the click points in
        the crop keeps every click inside it even where the coarse mask is thin.
        Costs one extra CPU encode, so apply it once the user has finished."""
        if not self.zoom_in or self.image is None:
            return self.result_mask
        if not self._pos_clicks and self._box is None:
            return self.result_mask

        h, w = self.image.shape[:2]
        coarse = self._decode(self._predictor, 0.0, 0.0, self._scale)
        coarse = self._cleanup_mask(coarse)

        ys, xs = np.where(coarse > 0)
        x0 = y0 = 1e18
        x1 = y1 = -1e18
        if len(xs) >= 4:
            x0, x1 = float(xs.min() / self._scale), float(xs.max() / self._scale)
            y0, y1 = float(ys.min() / self._scale), float(ys.max() / self._scale)
        for x, y in self._pos_clicks + self._neg_clicks:
            x0, x1 = min(x0, x), max(x1, x)
            y0, y1 = min(y0, y), max(y1, y)
        if self._box is not None:
            bx1, by1, bx2, by2 = self._box
            x0, x1 = min(x0, bx1), max(x1, bx2)
            y0, y1 = min(y0, by1), max(y1, by2)
        if x1 < 0:
            return self.result_mask

        mw = int((x1 - x0) * self.zoom_margin)
        mh = int((y1 - y0) * self.zoom_margin)
        cx0 = max(0, int(x0) - mw - 2)
        cy0 = max(0, int(y0) - mh - 2)
        cx1 = min(w, int(x1) + mw + 3)
        cy1 = min(h, int(y1) + mh + 3)

        crop_scale = min(1.0, self.max_side / max(cy1 - cy0, cx1 - cx0))
        if crop_scale <= self._scale:
            return self.result_mask  # no resolution headroom

        crop_img = self.image[cy0:cy1, cx0:cx1]
        if self._crop_predictor is None:
            from segment_anything import SamPredictor
            self._crop_predictor = SamPredictor(self._sam_model)
        self._crop_predictor.set_image(self._resize_work(crop_img, crop_scale))

        # Feed the coarse mask back as a mask prompt (in the crop's frame) so the
        # re-decode refines the coarse boundary instead of growing a fatter mask.
        coarse_full = self._upsample(coarse, h, w)
        mask_input = self._mask_to_logits(coarse_full[cy0:cy1, cx0:cx1])

        seg = self._decode(self._crop_predictor, cx0, cy0, crop_scale, mask_input=mask_input)
        seg = self._cleanup_mask(seg)
        seg = self._upsample(seg, cy1 - cy0, cx1 - cx0)

        full = self._upsample(coarse, h, w)
        full[cy0:cy1, cx0:cx1] = seg
        self._soft_mask = self._feather(full)
        return self.result_mask

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _mask_to_logits(mask):
        """Convert a 0/1 float mask (any size) to the 256x256 logits SAM's mask
        decoder expects as a mask prompt."""
        low = cv2.resize(np.clip(mask, 0.0, 1.0).astype(np.float32), (256, 256),
                         interpolation=cv2.INTER_LINEAR)
        low = np.clip(low, 0.01, 0.99)
        return (np.log(low / (1.0 - low)))[None, ...].astype(np.float32)

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
        """Turn the binary mask into a soft alpha (0..1).

        'guided' runs an edge-aware guided filter so the alpha ramp follows the
        *image* edges (hair strands stay thin and sharp instead of blurring);
        'gaussian' is the old uniform Gaussian blur.
        """
        if self.edge_mode == 'guided':
            return self._guided_feather(soft)
        return self._gaussian_feather(soft)

    def _gaussian_feather(self, soft):
        h, w = soft.shape[:2]
        radius = max(3, int(round(min(h, w) / 300)))
        if radius % 2 == 0:
            radius += 1
        soft = cv2.GaussianBlur(soft, (radius, radius), 0)
        return np.clip(soft, 0.0, 1.0)

    def _guided_feather(self, soft):
        guide = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        alpha = self._guided_filter(guide, soft, self.edge_radius, self.edge_eps)
        return np.clip(alpha, 0.0, 1.0)

    @staticmethod
    def _guided_filter(guide, src, radius, eps):
        """Edge-preserving guided filter (He et al. 2013), O(N) box-filter form.

        ``guide`` is a single-channel float image in [0,1]; ``src`` is the float
        image to smooth. The output follows the structure of ``guide``, so the
        alpha ramp hugs image edges instead of blurring across them.
        """
        I = guide.astype(np.float32)
        p = src.astype(np.float32)
        win = (radius, radius)
        mean_I = cv2.boxFilter(I, -1, win)
        mean_p = cv2.boxFilter(p, -1, win)
        mean_Ip = cv2.boxFilter(I * p, -1, win)
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = cv2.boxFilter(I * I, -1, win)
        var_I = mean_II - mean_I * mean_I
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        mean_a = cv2.boxFilter(a, -1, win)
        mean_b = cv2.boxFilter(b, -1, win)
        return mean_a * I + mean_b

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

    @property
    def box(self):
        return self._box
