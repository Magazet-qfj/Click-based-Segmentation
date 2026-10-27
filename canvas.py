"""A zoomable, pannable Tk canvas that reports clicks in image pixel coordinates.

The left mouse button follows the active tool (set via ``set_tool``):
    - 'foreground' brush  -> left click adds a foreground point
    - 'background' brush  -> left click adds a background point
    - 'box' tool          -> left drag draws a bounding box
    - 'delete' tool       -> left click deletes the marker under the cursor
Universal (tool-independent) gestures:
    - quick right click   -> negative (background) point
    - middle-button drag  -> pan
    - mouse wheel         -> zoom (keeps the point under the cursor fixed)
"""

import tkinter as tk
from PIL import Image, ImageTk


class CanvasImage:
    def __init__(self, canvas):
        self.canvas = canvas
        self._original = None        # PIL.Image at full resolution
        self._display = None         # PIL.Image at current zoom
        self._tkimage = None         # keep a ref so Tk doesn't GC the PhotoImage
        self.scale = 1.0
        self._offset = (0, 0)        # top-left of the image in canvas coords
        self._click_callback = None

        self._pan_start = None        # middle-drag pan anchor (canvas coords)
        self._right_start = None      # right-press anchor, for move detection
        self._right_moved = False
        self._right_press_time = 0.0

        self._box_callback = None
        self._box_start = None       # image coords of a left-press
        self._box_anchor = None      # canvas coords of a left-press
        self._box_rect_id = None     # canvas id of the in-progress rubber band
        self._box_moved = False

        self._delete_callback = None
        self.tool = 'foreground'     # active tool: foreground | background | box | delete

        self.canvas.bind('<ButtonPress-1>', self._on_left_press)
        self.canvas.bind('<B1-Motion>', self._on_left_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_left_release)
        self.canvas.bind('<ButtonPress-2>', self._on_middle_press)
        self.canvas.bind('<B2-Motion>', self._on_middle_drag)
        self.canvas.bind('<ButtonRelease-2>', self._on_middle_release)
        self.canvas.bind('<ButtonPress-3>', self._on_right_press)
        self.canvas.bind('<B3-Motion>', self._on_right_drag)
        self.canvas.bind('<ButtonRelease-3>', self._on_right_release)
        self.canvas.bind('<MouseWheel>', self._on_wheel)

    def register_click_callback(self, callback):
        """callback(is_positive, x, y) receives image-space pixel coordinates."""
        self._click_callback = callback

    def register_box_callback(self, callback):
        """callback(x1, y1, x2, y2) receives image-space XYXY box coordinates."""
        self._box_callback = callback

    def register_delete_callback(self, callback):
        """callback(x, y) receives image-space coords of a delete-mode click."""
        self._delete_callback = callback

    def set_tool(self, tool):
        """Switch the active tool and swap the cursor for visual feedback."""
        self.tool = tool
        self.canvas.config(cursor='X_cursor' if tool == 'delete' else 'crosshair')

    # -------------------------------------------------------------- rendering
    def set_image(self, image, fit=True):
        """Display a new image, optionally fitting it to the window."""
        self._original = Image.fromarray(image)
        if fit:
            self._fit_to_window()
        self._render()

    def reload_image(self, image):
        """Swap the image content, keeping the current zoom and pan."""
        self._original = Image.fromarray(image)
        self._render()

    def _fit_to_window(self):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            cw, ch = 800, 600
        iw, ih = self._original.size
        self.scale = min(cw / iw, ch / ih, 1.0)
        self._offset = (0, 0)

    def _render(self):
        if self._original is None:
            return
        iw, ih = self._original.size
        dw, dh = max(1, int(iw * self.scale)), max(1, int(ih * self.scale))
        interp = Image.NEAREST if self.scale > 2.0 else Image.BILINEAR
        self._display = self._original.resize((dw, dh), interp)
        self._tkimage = ImageTk.PhotoImage(self._display)
        self.canvas.delete('all')
        self.canvas.create_image(self._offset[0], self._offset[1],
                                 anchor='nw', image=self._tkimage)

    # ------------------------------------------------------------ coordinates
    def _to_image_coords(self, event):
        if self._original is None:
            return None
        x = (event.x - self._offset[0]) / self.scale
        y = (event.y - self._offset[1]) / self.scale
        iw, ih = self._original.size
        if 0 <= x < iw and 0 <= y < ih:
            return int(x), int(y)
        return None

    # ----------------------------------------------------------- mouse events
    def _on_left_press(self, event):
        if self._click_callback is None and self._box_callback is None:
            return
        coords = self._to_image_coords(event)
        if coords is None:
            return
        self._box_start = coords
        self._box_anchor = (event.x, event.y)
        self._box_moved = False
        self._box_rect_id = None

    def _on_left_drag(self, event):
        if self._box_anchor is None:
            return
        dx = event.x - self._box_anchor[0]
        dy = event.y - self._box_anchor[1]
        if abs(dx) > 3 or abs(dy) > 3:
            self._box_moved = True
        if self.tool != 'box':
            return  # only the box tool draws a rubber band
        if not self._box_moved:
            return
        x0, y0 = self._box_anchor
        if self._box_rect_id is not None:
            self.canvas.delete(self._box_rect_id)
        self._box_rect_id = self.canvas.create_rectangle(
            x0, y0, event.x, event.y, outline='#ff3333', width=2)

    def _on_left_release(self, event):
        if self._box_anchor is None:
            return
        if self._box_rect_id is not None:
            self.canvas.delete(self._box_rect_id)
            self._box_rect_id = None
        start = self._box_start
        self._box_anchor = None
        self._box_start = None

        tool = self.tool
        if tool == 'box':
            # Box tool: only a drag commits a box; a plain click does nothing.
            if self._box_moved:
                end = self._to_image_coords(event)
                if end is not None and self._box_callback is not None:
                    x1, y1 = start
                    x2, y2 = end
                    if abs(x2 - x1) >= 3 and abs(y2 - y1) >= 3:
                        self._box_callback(min(x1, x2), min(y1, y2),
                                           max(x1, x2), max(y1, y2))
            return
        if tool == 'delete':
            # Delete tool: a plain click removes the marker under the cursor
            # (the app hit-tests and ignores the click if nothing is there).
            if not self._box_moved and self._delete_callback is not None and start is not None:
                self._delete_callback(start[0], start[1])
            return
        # Foreground/background brush: a plain click adds a point of that type.
        if not self._box_moved and self._click_callback is not None and start is not None:
            self._click_callback(is_positive=(tool == 'foreground'), x=start[0], y=start[1])

    def _on_middle_press(self, event):
        self._pan_start = (event.x, event.y)

    def _on_middle_drag(self, event):
        if self._pan_start is None:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self._offset = (self._offset[0] + dx, self._offset[1] + dy)
        self._pan_start = (event.x, event.y)
        self.canvas.move('all', dx, dy)

    def _on_middle_release(self, event):
        self._pan_start = None

    def _on_right_press(self, event):
        self._right_start = (event.x, event.y)
        self._right_moved = False
        self._right_press_time = event.time / 1000.0

    def _on_right_drag(self, event):
        # Right-drag no longer pans; only track movement so a dragged right-click
        # does not accidentally register as a negative point.
        if self._right_start is None:
            return
        if abs(event.x - self._right_start[0]) > 3 or abs(event.y - self._right_start[1]) > 3:
            self._right_moved = True

    def _on_right_release(self, event):
        dt = event.time / 1000.0 - self._right_press_time
        if not self._right_moved and dt < 0.4 and self._click_callback is not None:
            coords = self._to_image_coords(event)
            if coords:
                self._click_callback(is_positive=False, x=coords[0], y=coords[1])
        self._right_start = None

    def _on_wheel(self, event):
        if self._original is None:
            return
        factor = 1.2 if event.delta > 0 else 1 / 1.2
        new_scale = max(0.05, min(self.scale * factor, 20.0))
        if abs(new_scale - self.scale) < 1e-9:
            return
        # Keep the point under the cursor fixed while zooming.
        cx = (event.x - self._offset[0]) / self.scale
        cy = (event.y - self._offset[1]) / self.scale
        self.scale = new_scale
        self._offset = (event.x - cx * new_scale, event.y - cy * new_scale)
        self._render()
