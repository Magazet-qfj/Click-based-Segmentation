"""Glue layer between the UI and the segmentation engine.

The controller owns the image + segmenter state and re-runs segmentation on
every mutation (add/undo/reset click), then asks the UI to redraw.
"""

import logging

from visuals import draw_with_blend_and_clicks

log = logging.getLogger(__name__)


class InteractiveController:
    def __init__(self, segmenter, update_image_callback):
        self.segmenter = segmenter
        self.update_image_callback = update_image_callback
        self.image = None

    # ------------------------------------------------------------ image state
    def set_image(self, image):
        self.segmenter.set_image(image)  # may raise; only update self.image on success
        self.image = image
        self.update_image_callback(reset_canvas=True)

    # ------------------------------------------------------------ interaction
    def add_click(self, x, y, is_positive):
        self.segmenter.add_click(x, y, is_positive)
        self.segmenter.run()
        self.update_image_callback()

    def undo_click(self):
        self.segmenter.undo_click()
        self.segmenter.run()
        self.update_image_callback()

    def reset(self):
        self.segmenter.reset()
        self.update_image_callback()

    # ------------------------------------------------------------------- box
    def set_box(self, x1, y1, x2, y2):
        self.segmenter.set_box(x1, y1, x2, y2)
        self.segmenter.run()
        self.update_image_callback()

    def clear_box(self):
        self.segmenter.clear_box()
        self.segmenter.run()
        self.update_image_callback()

    def set_edge_mode(self, mode):
        self.segmenter.edge_mode = mode
        log.info("边缘羽化方式改为 %s", mode)

    def set_refine_mode(self, mode):
        self.segmenter.refine_mode = mode
        log.info("边界精修改为 %s", mode)

    def set_cleanup_kernel(self, k):
        self.segmenter.cleanup_kernel = k
        log.info("掩码清理核改为 %d", k)

    def resegment(self):
        self.segmenter.run()
        self.update_image_callback()

    # ---------------------------------------------------------------- queries
    def click_counts(self):
        pos, neg = self.segmenter.clicks
        return len(pos), len(neg)

    @property
    def result_mask(self):
        return self.segmenter.result_mask

    @property
    def result_alpha(self):
        return self.segmenter.result_alpha

    def matte(self):
        return self.segmenter.matte()

    @property
    def result_matte(self):
        return self.segmenter.result_matte

    def get_visualization(self, alpha_blend=0.5, click_radius=3):
        if self.image is None:
            return None
        return draw_with_blend_and_clicks(
            self.image,
            mask=self.result_mask,
            alpha=alpha_blend,
            clicks=self.segmenter.clicks,
            radius=click_radius,
        )
