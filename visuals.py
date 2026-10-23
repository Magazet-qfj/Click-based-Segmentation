"""Visualization helpers: blend the segmentation mask overlay and draw clicks."""

import cv2
import numpy as np


def draw_with_blend_and_clicks(image, mask=None, alpha=0.5, clicks=None, radius=3):
    """Return a copy of ``image`` with the mask blended in and clicks drawn.

    ``image`` and ``mask`` are RGB uint8 arrays; ``clicks`` is a
    ``(positive_clicks, negative_clicks)`` tuple of (x, y) pixel coordinates.
    """
    vis = image.copy()

    if mask is not None and mask.max() > 0:
        overlay = np.zeros_like(image)
        overlay[mask > 0] = (0, 255, 0)  # green = foreground
        vis = cv2.addWeighted(vis, 1.0 - alpha, overlay, alpha, 0.0)

    if clicks is not None:
        pos_clicks, neg_clicks = clicks
        for x, y in pos_clicks:
            cv2.circle(vis, (int(x), int(y)), radius, (0, 220, 0), -1)
        for x, y in neg_clicks:
            cv2.circle(vis, (int(x), int(y)), radius, (220, 0, 0), -1)

    return vis
