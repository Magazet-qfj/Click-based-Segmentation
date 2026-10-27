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

import logging
import os
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)


class SamSegmenter:
    """Click-driven foreground/background segmentation via Segment Anything."""

    def __init__(self, checkpoint, model_type='vit_b', max_side=1024, device='cpu',
                 cleanup_kernel=9, edge_mode='guided', edge_radius=4, edge_eps=1e-4,
                 refine_mode='off', grabcut_iters=3, grabcut_max_side=1536,
                 grabcut_band=6):
        self.checkpoint = checkpoint   # path to the SAM checkpoint (vit_b / vit_l)
        self.model_type = model_type   # 'vit_b' or 'vit_l' (sam_model_registry key)
        self.max_side = max_side       # cap the working image at this longest side (SAM native = 1024)
        self.device = device
        self.cleanup_kernel = cleanup_kernel  # close kernel for hole-filling (median fixed at 3)
        self.edge_mode = edge_mode     # 'guided' (edge-aware) or 'gaussian' (uniform blur)
        self.edge_radius = edge_radius # guided-filter window radius at full resolution
        self.edge_eps = edge_eps       # guided-filter regularizer (for 0..1 guide/src)
        self.refine_mode = refine_mode       # 'off' (default) or 'grabcut' full-res boundary refinement
        self.grabcut_iters = grabcut_iters   # GrabCut EM iterations (colour-model refinement)
        self.grabcut_max_side = grabcut_max_side  # cap the GrabCut working resolution (speed)
        self.grabcut_band = grabcut_band      # probable-band half-width (px) around the contour

        self.image = None             # original RGB image (H, W, 3), uint8
        self._predictor = None        # SamPredictor for the whole image (built lazily)
        self._work_image = None       # downscaled RGB image passed to set_image
        self._scale = 1.0             # original_size -> work_size scale factor
        self._pos_clicks = []         # (x, y) in ORIGINAL image coordinates
        self._neg_clicks = []
        self._box = None              # (x1, y1, x2, y2) in ORIGINAL image coordinates
        self._soft_mask = None        # soft (0..1 float) mask at original resolution
        self._matte = None            # cached true alpha matte (0..1 float) at full res
        self.matte_radius = 25        # trimap unknown-band half-width, in FULL-res px (scaled down to the solve res)
        self.matte_max_side = 512     # resolution cap for the closed-form matting solve

    # ------------------------------------------------------------------ model
    def _load_model(self):
        """Build the SAM model + predictor once, on first use."""
        if self._predictor is not None:
            return
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(
                f"SAM 权重文件不存在：{self.checkpoint}\n"
                f"请下载 {os.path.basename(self.checkpoint)} 放到 weights/ 目录（见 README）。")
        log.info("加载 SAM 模型 %s（%s）...", self.model_type.upper(), os.path.basename(self.checkpoint))
        t0 = time.time()
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(self.device)
        sam.eval()
        self._predictor = SamPredictor(sam)
        log.info("SAM 模型就绪（%.1fs）", time.time() - t0)

    # ------------------------------------------------------------------ image
    def set_image(self, image):
        """Set the RGB image, encode it once, and clear all segmentation state."""
        self._load_model()  # raises FileNotFoundError if the checkpoint is missing
        self.image = image

        h, w = image.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        self._scale = scale
        log.info("设置图像 %dx%d（最长边上限 %d）", w, h, self.max_side)
        if scale < 1.0:
            self._work_image = cv2.resize(
                image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
            log.info("降采样到工作图 %dx%d（scale=%.3f）",
                     self._work_image.shape[1], self._work_image.shape[0], scale)
        else:
            self._work_image = image

        t0 = time.time()
        self._predictor.set_image(self._work_image)  # the slow step (image encode)
        log.info("图像编码完成：%.1fs", time.time() - t0)
        self._pos_clicks = []
        self._neg_clicks = []
        self._box = None
        self._soft_mask = None

    def reencode(self):
        """Re-encode the current image after ``max_side`` changed.

        ``set_image`` clears the clicks/box, but they live in original-image
        coordinates, so they survive a resolution change unchanged — restore them.
        """
        if self.image is None:
            return
        pos = list(self._pos_clicks)
        neg = list(self._neg_clicks)
        box = self._box
        self.set_image(self.image)
        self._pos_clicks = pos
        self._neg_clicks = neg
        self._box = box

    # ----------------------------------------------------------------- clicks
    def add_click(self, x, y, is_positive):
        log.info("添加%s点 (%d, %d)", "前景" if is_positive else "背景", int(x), int(y))
        if is_positive:
            self._pos_clicks.append((x, y))
        else:
            self._neg_clicks.append((x, y))

    def undo_click(self):
        # Undo the most recent click (negative clicks are undone first).
        log.info("撤销上一次点击")
        if self._neg_clicks:
            self._neg_clicks.pop()
        elif self._pos_clicks:
            self._pos_clicks.pop()

    def remove_click(self, index, is_positive):
        """Remove a specific click by its index in the foreground/background list."""
        lst = self._pos_clicks if is_positive else self._neg_clicks
        if 0 <= index < len(lst):
            x, y = lst.pop(index)
            log.info("删除%s点 (%d, %d)", "前景" if is_positive else "背景", int(x), int(y))

    def remove_clicks_in_box(self, x1, y1, x2, y2):
        """Remove every foreground/background click whose point falls inside the
        XYXY box (original-image coordinates). Returns the number removed."""
        x1, y1, x2, y2 = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        removed = 0
        for lst in (self._pos_clicks, self._neg_clicks):
            keep = [(cx, cy) for (cx, cy) in lst
                    if not (x1 <= cx <= x2 and y1 <= cy <= y2)]
            removed += len(lst) - len(keep)
            lst[:] = keep
        if removed:
            log.info("框选批量删除 %d 个标记点", removed)
        return removed

    def reset(self):
        log.info("清空所有点击")
        self._pos_clicks = []
        self._neg_clicks = []
        self._box = None
        self._soft_mask = None
        self._matte = None

    # ------------------------------------------------------------------- box
    def set_box(self, x1, y1, x2, y2):
        """Set a bounding-box prompt (XYXY, in ORIGINAL image coordinates)."""
        x1, y1, x2, y2 = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        log.info("设置框提示 (%d,%d)-(%d,%d)", int(x1), int(y1), int(x2), int(y2))
        self._box = (x1, y1, x2, y2)

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

        t0 = time.time()
        soft = self._run_coarse()
        if self.refine_mode == 'grabcut':
            soft = self.refine_boundary((soft >= 0.5).astype(np.uint8)).astype(np.float32)
        self._soft_mask = self._feather(soft)
        self._matte = None
        cov = float(self._soft_mask.mean()) * 100.0
        log.info("分割完成: %.0fms | 前景 %.1f%% | %d 前景点 / %d 背景点%s",
                 (time.time() - t0) * 1000.0, cov,
                 len(self._pos_clicks), len(self._neg_clicks),
                 " + 框" if self._box is not None else "")
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

    def refine_boundary(self, binary):
        """Refine a full-resolution binary mask's contour with GrabCut.

        SAM's mask decoder outputs a fixed 256×256 mask that is then upsampled to
        the image, so the contour is coarse on large images (hair strands / thin
        teeth blur). GrabCut re-fits the boundary from full-resolution colour,
        using the coarse mask only to seed a probable band around its contour, so
        it recovers detail SAM's decoder never had while staying connected and
        click-stable.

        ``binary`` is 0/1 uint8 at full image resolution. The image + mask are
        downscaled to ``grabcut_max_side`` to bound the solve time, refined there,
        and upsampled back. Returns a 0/1 uint8 mask (unchanged if the object is
        smaller than the band or fills the frame — nothing to refine).
        """
        img = self.image
        h, w = binary.shape[:2]
        cap = self.grabcut_max_side
        scale = min(1.0, cap / max(h, w))
        if scale < 1.0:
            tw, th = int(w * scale), int(h * scale)
            img_w = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
            mask_w = (cv2.resize(binary.astype(np.float32), (tw, th),
                                 interpolation=cv2.INTER_LINEAR) >= 0.5).astype(np.uint8)
        else:
            img_w, mask_w = img, binary

        band = self.grabcut_band
        trimap = np.full(mask_w.shape, cv2.GC_PR_BGD, np.uint8)
        trimap[mask_w > 0] = cv2.GC_PR_FGD
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
        trimap[cv2.erode(mask_w, k) > 0] = cv2.GC_FGD
        trimap[cv2.erode((mask_w == 0).astype(np.uint8), k) > 0] = cv2.GC_BGD
        if not (trimap == cv2.GC_FGD).any() or not (trimap == cv2.GC_BGD).any():
            return binary  # object too small / fills frame; nothing to refine

        bgm = np.zeros((1, 65), np.float64)
        fgm = np.zeros((1, 65), np.float64)
        cv2.grabCut(img_w, trimap, None, bgm, fgm, self.grabcut_iters, cv2.GC_INIT_WITH_MASK)
        refined_w = ((trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD)).astype(np.uint8)

        if scale < 1.0:
            return (cv2.resize(refined_w.astype(np.float32), (w, h),
                               interpolation=cv2.INTER_LINEAR) >= 0.5).astype(np.uint8)
        return refined_w

    # ----------------------------------------------------------------- helpers
    def _cleanup_mask(self, mask):
        """Remove SAM's small mask artifacts at working resolution.

        The mask decoder's transposed-convolution upsampling can leave a faint
        checkerboard in smooth, same-coloured regions, plus occasional tiny
        holes. A *small* median filter (3px) kills the alternating checkerboard /
        salt-and-pepper noise, then a morphological close fills remaining small
        holes and bridges thin gaps.

        The median is fixed at 3px on purpose: the checkerboard is a 1-2px
        high-frequency pattern, and a larger median (e.g. 9px) also erases
        *genuine* small round foreground details (spots, buttons, highlights),
        turning them into unexpected background holes. The close kernel is
        ``self.cleanup_kernel`` (odd); raise it to fill bigger holes, lower it to
        preserve very fine structures.
        """
        m = mask.astype(np.uint8)
        m = cv2.medianBlur(m, 3)
        k = self.cleanup_kernel
        if k % 2 == 0:
            k += 1
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

    def matte(self):
        """True alpha matte via closed-form matting (Levin et al. 2007).

        Unlike the guided filter (which feathers a hard edge), this solves for
        semi-transparent pixels at the boundary, so wispy hair blends with the
        background instead of snapping to 0/1. Cached until the next click. Costs
        a sparse linear solve (seconds), so call it at export, not per click.

        ``matte_radius`` is the trimap's unknown-band half-width in *full-res*
        pixels. SAM's coarse boundary swallows the light gaps between dark hair
        strands; matting can recover those gaps only if the unknown band is wide
        enough to span the whole fringe. That fringe depth is ~constant in
        full-res pixels regardless of image size, so the radius is scaled down by
        ``scale`` before the solve (which runs at ``matte_max_side``), keeping the
        recovery resolution-independent.
        """
        if self._matte is None:
            if self.image is None or self._soft_mask is None:
                return None
            from matting import estimate_alpha
            h, w = self.image.shape[:2]
            scale = min(1.0, self.matte_max_side / max(h, w))
            radius_solve = max(1, int(round(self.matte_radius * scale)))
            t0 = time.time()
            self._matte = estimate_alpha(
                self.image, self.result_mask,
                trimap_radius=radius_solve, max_side=self.matte_max_side)
            log.info("闭式 alpha matting 求解完成（trimap 半径 %dpx，%.1fs）",
                     radius_solve, time.time() - t0)
        return self._matte

    @property
    def result_matte(self):
        """Soft alpha (0..255) from closed-form matting, for hair-level cutout."""
        m = self.matte()
        if m is None:
            return None
        return (m * 255.0 + 0.5).astype(np.uint8)

    @property
    def has_clicks(self):
        return bool(self._pos_clicks or self._neg_clicks)

    @property
    def clicks(self):
        return self._pos_clicks, self._neg_clicks

    @property
    def box(self):
        return self._box
