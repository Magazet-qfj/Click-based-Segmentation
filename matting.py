"""Closed-form alpha matting (Levin, Lischinski & Weiss 2007) — pure numpy/scipy.

SAM outputs a *binary* mask; semi-transparent edges (wispy hair) need a real
alpha matte whose boundary pixels take fractional 0..1 values. This module turns
a binary mask into such a matte via the classic closed-form algorithm:

1. Erode/dilate the binary mask into a trimap — definite foreground (1),
   definite background (0), and an "unknown" band between them (0.5).
2. Build the matting Laplacian ``L`` from the color image under the local
   color-line model (each 3x3 window's colors are assumed to lie on a line in
   RGB space), so nearby pixels of similar color get coupled.
3. Solve ``L_UU alpha_U = -L_UF m_F`` for the unknown pixels, holding the known
   (trimap) pixels fixed — a sparse linear solve via ``scipy``.

The result is a full-resolution alpha in [0, 1] with genuine semi-transparency
along the boundary, unlike the guided filter which only feathers a hard edge.

No new dependencies: numpy + scipy (already in the venv) + cv2.
"""

import numpy as np
import cv2
from scipy import sparse
from scipy.sparse.linalg import spsolve


def build_trimap(mask, radius=10):
    """Erode/dilate a 0/1 uint8 mask into a trimap (1 = fg, 0 = bg, 0.5 = unknown).

    ``radius`` is the half-width of the unknown band in pixels (at working res).
    """
    mask = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    fg = cv2.erode(mask, k)
    bg = cv2.dilate(mask, k)
    trimap = np.full(mask.shape, 0.5, dtype=np.float64)
    trimap[fg > 0] = 1.0
    trimap[bg == 0] = 0.0
    return trimap


def _matting_laplacian(img):
    """Levin et al. matting Laplacian for an (h, w, 3) float image in [0, 1].

    Returns a sparse CSR ``(n, n)`` matrix with n = h*w. Each 3x3 window
    contributes a 9x9 block; the per-window mean and regularized covariance are
    computed vectorized via ``cv2.boxFilter``, and the blocks are built with a
    strip-wise ``einsum`` (vectorized over all windows in a horizontal strip)
    rather than a per-pixel Python loop.
    """
    img = img.astype(np.float64)
    h, w = img.shape[:2]
    n = h * w
    eps = 1e-7

    # Per-window (3x3) mean and 3x3 covariance, vectorized with box filters.
    mu = cv2.boxFilter(img, -1, (3, 3))                      # (h, w, 3)
    cov = np.empty((h, w, 3, 3), dtype=np.float64)
    for a in range(3):
        for b in range(a, 3):
            e = cv2.boxFilter(img[..., a] * img[..., b], -1, (3, 3))
            c = e - mu[..., a] * mu[..., b]
            cov[..., a, b] = c
            cov[..., b, a] = c
    # Regularized inverse of (cov + eps*I) at every pixel.
    inv = np.linalg.inv(cov + eps * np.eye(3)[None, None, :, :])  # (h, w, 3, 3)

    img_flat = img.reshape(-1, 3)
    dy = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
    dx = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
    identity = np.eye(9)

    rows, cols, vals = [], [], []
    strip = 64
    for y0 in range(1, h - 1, strip):
        y1 = min(y0 + strip, h - 1)
        cc_y = np.arange(y0, y1)                 # window-center rows
        cc_x = np.arange(1, w - 1)               # window-center cols
        CY, CX = np.meshgrid(cc_y, cc_x, indexing='ij')       # (sy, sx)
        wi = CY[..., None] * w + CX[..., None] + dy * w + dx   # (sy, sx, 9) flat idx
        Xc = img_flat[wi] - mu[CY, CX, None, :]               # (sy, sx, 9, 3)
        G = np.einsum('...ia,...ab,...jb->...ij', Xc, inv[CY, CX], Xc)  # (sy, sx, 9, 9)
        block = identity - (1.0 + G) / 9.0                    # (sy, sx, 9, 9)
        rows.append(np.broadcast_to(wi[..., :, None], block.shape).ravel())
        cols.append(np.broadcast_to(wi[..., None, :], block.shape).ravel())
        vals.append(block.ravel())

    L = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n))
    return L


def estimate_alpha(image, mask, trimap_radius=10, max_side=512):
    """Closed-form alpha matte from an RGB image and a binary mask.

    Runs the solve at ``max_side`` resolution (the sparse system is otherwise too
    large for a one-shot CPU call), then upsamples the alpha back to full res.
    Returns a float ``(H, W)`` alpha in [0, 1].
    """
    h, w = mask.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img_s = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        mask_s = cv2.resize(mask.astype(np.uint8), (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_NEAREST)
    else:
        img_s, mask_s = image, mask.astype(np.uint8)

    trimap = build_trimap(mask_s, radius=trimap_radius)
    unknown = np.abs(trimap - 0.5) < 1e-9

    if unknown.sum() == 0:
        # No unknown band (mask already trivial / all known) — just return the mask.
        return mask.astype(np.float64)

    img_f = img_s.astype(np.float64) / 255.0
    L = _matting_laplacian(img_f)

    m = trimap.ravel()
    U = np.where(unknown.ravel())[0]
    F = np.where(~unknown.ravel())[0]

    L_UU = L[U][:, U].tocsc()
    L_UF = L[U][:, F].tocsr()
    m_F = m[F]

    # Tiny ridge keeps the solve stable if a window is colour-degenerate.
    ridge = 1e-6
    A = L_UU + ridge * sparse.eye(len(U), format='csc')
    alpha_U = spsolve(A, -L_UF @ m_F)

    alpha = m.copy()
    alpha[U] = np.clip(alpha_U, 0.0, 1.0)
    alpha = alpha.reshape(mask_s.shape[:2])

    if scale < 1.0:
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(alpha, 0.0, 1.0)
