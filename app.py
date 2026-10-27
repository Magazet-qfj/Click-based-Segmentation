"""Tkinter application window for click-based interactive segmentation.

The window guides the user step by step: load an image, click the foreground
target, optionally click background to exclude regions, then save the result.
"""

import logging
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np

from canvas import CanvasImage
from controller import InteractiveController

log = logging.getLogger(__name__)


def _imread_unicode(path):
    """cv2.imread cannot open paths with non-ASCII (e.g. Chinese) characters on
    Windows. Read the bytes ourselves and let cv2 decode them instead."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _imwrite_unicode(path, image):
    """cv2.imwrite has the same non-ASCII path bug; encode to bytes and write."""
    ext = os.path.splitext(path)[1] or '.png'
    ok, buf = cv2.imencode(ext, image)
    if ok:
        buf.tofile(path)
    return ok


class InteractiveDemoApp(ttk.Frame):
    def __init__(self, master, segmenter, initial_image=''):
        super().__init__(master)
        self.master = master
        master.title(f"点击式交互分割 — SAM {segmenter.model_type}")
        master.geometry("1120x720")

        self.controller = InteractiveController(segmenter, update_image_callback=self._update_image)

        self.alpha_blend = tk.DoubleVar(value=0.5)
        self.click_radius = tk.IntVar(value=3)
        self.edge_mode = tk.StringVar(value=segmenter.edge_mode)
        self.cleanup_kernel = tk.IntVar(value=segmenter.cleanup_kernel)
        self.matte_var = tk.BooleanVar(value=False)
        self.refine_var = tk.BooleanVar(value=(segmenter.refine_mode == 'grabcut'))

        self._build_menu()
        self._build_layout()
        self.image_on_canvas.register_click_callback(self._click_callback)
        self.image_on_canvas.register_box_callback(self._box_callback)
        self._bind_shortcuts()

        if initial_image:
            self._load_image_from_path(initial_image)

    # -------------------------------------------------------------- UI layout
    def _build_menu(self):
        bar = tk.Menu(self.master)
        file_menu = tk.Menu(bar, tearoff=0)
        file_menu.add_command(label="加载图片", command=self._load_image_callback)
        file_menu.add_separator()
        file_menu.add_command(label="保存掩码", command=self._save_mask_callback)
        file_menu.add_command(label="保存前景抠图", command=self._save_foreground_callback)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.master.quit)
        bar.add_cascade(label="文件", menu=file_menu)

        edit_menu = tk.Menu(bar, tearoff=0)
        edit_menu.add_command(label="撤销上次点击", command=self._undo_click)
        edit_menu.add_command(label="清空所有点击", command=self._reset)
        edit_menu.add_command(label="重新分割", command=self._resegment)
        bar.add_cascade(label="编辑", menu=edit_menu)
        self.master.config(menu=bar)

    def _build_layout(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Left: image canvas.
        canvas_frame = ttk.LabelFrame(self, text="图像")
        canvas_frame.grid(row=0, column=0, sticky='nsew', padx=6, pady=6)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_frame, highlightthickness=0, cursor='crosshair',
                                width=820, height=680)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.image_on_canvas = CanvasImage(self.canvas)

        # Right: scrollable control panel (follows window height; wheel scrolls it).
        panel = ttk.Frame(self)
        panel.grid(row=0, column=1, sticky='nsew', padx=6, pady=6)
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)

        self.ctrl_canvas = tk.Canvas(panel, highlightthickness=0, width=300)
        self.ctrl_scrollbar = ttk.Scrollbar(panel, orient='vertical',
                                            command=self.ctrl_canvas.yview)
        self.ctrl_canvas.configure(yscrollcommand=self.ctrl_scrollbar.set)
        self.ctrl_canvas.grid(row=0, column=0, sticky='nsew')
        self.ctrl_scrollbar.grid(row=0, column=1, sticky='ns')

        ctrl = ttk.Frame(self.ctrl_canvas)
        self._ctrl_window = self.ctrl_canvas.create_window((0, 0), window=ctrl, anchor='nw')
        ctrl.columnconfigure(0, weight=1)
        ctrl.bind('<Configure>', self._on_ctrl_configure)
        self.ctrl_canvas.bind('<Configure>', self._on_ctrl_canvas_configure)

        self.instruction = tk.Label(
            ctrl, text="请先加载一张图片", wraplength=270, justify='left',
            anchor='w', bg='#fff3cd', fg='#333',
            font=('Microsoft YaHei', 10), padx=10, pady=10)
        self.instruction.grid(row=0, column=0, sticky='ew', pady=(0, 8))

        self.status = tk.Label(ctrl, text="前景 0 点 · 背景 0 点", anchor='w',
                               font=('Microsoft YaHei', 9), fg='#666')
        self.status.grid(row=1, column=0, sticky='ew', pady=(0, 8))

        clicks_frame = ttk.LabelFrame(ctrl, text="点击管理")
        clicks_frame.grid(row=2, column=0, sticky='ew', pady=4)
        clicks_frame.columnconfigure(0, weight=1)
        ttk.Button(clicks_frame, text="撤销上次点击", command=self._undo_click).grid(
            row=0, column=0, sticky='ew', padx=6, pady=3)
        ttk.Button(clicks_frame, text="清空所有点击", command=self._reset).grid(
            row=1, column=0, sticky='ew', padx=6, pady=3)
        ttk.Button(clicks_frame, text="重新分割", command=self._resegment).grid(
            row=2, column=0, sticky='ew', padx=6, pady=3)

        export_frame = ttk.LabelFrame(ctrl, text="导出")
        export_frame.grid(row=3, column=0, sticky='ew', pady=4)
        export_frame.columnconfigure(0, weight=1)
        ttk.Button(export_frame, text="保存掩码...", command=self._save_mask_callback).grid(
            row=0, column=0, sticky='ew', padx=6, pady=3)
        ttk.Button(export_frame, text="保存前景抠图...", command=self._save_foreground_callback).grid(
            row=1, column=0, sticky='ew', padx=6, pady=3)
        ttk.Checkbutton(export_frame, text="启用 alpha matting",
                        variable=self.matte_var).grid(
            row=2, column=0, sticky='w', padx=6, pady=(2, 3))

        alpha_frame = ttk.LabelFrame(ctrl, text="掩码透明度")
        alpha_frame.grid(row=4, column=0, sticky='ew', pady=4)
        ttk.Scale(alpha_frame, from_=0.0, to=1.0, variable=self.alpha_blend,
                  command=lambda v: self._update_image()).pack(fill='x', padx=6, pady=3)

        radius_frame = ttk.LabelFrame(ctrl, text="点击点大小")
        radius_frame.grid(row=5, column=0, sticky='ew', pady=4)
        ttk.Scale(radius_frame, from_=1, to=8, variable=self.click_radius,
                  command=lambda v: self._update_image()).pack(fill='x', padx=6, pady=3)

        edge_frame = ttk.LabelFrame(ctrl, text="边缘羽化方式")
        edge_frame.grid(row=6, column=0, sticky='ew', pady=4)
        edge_frame.columnconfigure(0, weight=1)
        self.edge_combo = ttk.Combobox(
            edge_frame, textvariable=self.edge_mode,
            values=('guided', 'gaussian'), state='readonly')
        self.edge_combo.grid(row=0, column=0, sticky='ew', padx=6, pady=3)
        self.edge_combo.bind('<<ComboboxSelected>>', self._on_edge_mode_changed)
        tk.Label(edge_frame, text="guided = 边缘感知 · gaussian = 均匀模糊",
                 justify='left', anchor='w', font=('Microsoft YaHei', 8), fg='#888').grid(
            row=1, column=0, sticky='ew', padx=6, pady=(0, 3))
        ttk.Checkbutton(edge_frame, text="GrabCut 全分辨率边界精修（更细，慢约几秒）",
                        variable=self.refine_var, command=self._on_refine_changed).grid(
            row=2, column=0, sticky='w', padx=6, pady=(2, 3))

        cleanup_frame = ttk.LabelFrame(ctrl, text="掩码清理强度")
        cleanup_frame.grid(row=7, column=0, sticky='ew', pady=4)
        self.cleanup_scale = ttk.Scale(
            cleanup_frame, from_=1, to=15, variable=self.cleanup_kernel,
            command=self._set_cleanup)
        self.cleanup_scale.pack(fill='x', padx=6, pady=3)
        self.cleanup_scale.bind('<ButtonRelease-1>', self._on_cleanup_release)
        tk.Label(cleanup_frame, text="越小越保留细丝，越大越去除伪影",
                 justify='left', anchor='w', font=('Microsoft YaHei', 8), fg='#888').pack(
            fill='x', padx=6, pady=(0, 3))

        help_text = ("操作说明：\n"
                     "左键点击 = 标记前景，左键拖拽 = 框选目标\n"
                     "右键点击 = 标记背景\n"
                     "中键拖动 = 平移，滚轮 = 缩放\n"
                     "Ctrl + O 打开, Ctrl + S 保存掩码\n"
                     "Ctrl + Shift + S 保存抠图, Ctrl + Z 撤销, Esc 清空")
        tk.Label(ctrl, text=help_text, justify='left', anchor='w',
                 font=('Microsoft YaHei', 9), fg='#888').grid(
            row=8, column=0, sticky='ew', pady=(8, 0))

        # Bottom status bar: progress / ready — separate from the step guidance.
        self.statusbar = tk.Label(self, text="就绪", anchor='w', relief='sunken',
                                  font=('Microsoft YaHei', 9), padx=6)
        self.statusbar.grid(row=1, column=0, columnspan=2, sticky='ew')

        # Wheel over the panel scrolls it (recursive bind covers every child).
        self._bind_panel_wheel(ctrl)
        self._bind_panel_wheel(self.ctrl_canvas)
        self._bind_panel_wheel(self.ctrl_scrollbar)

    # ------------------------------------------------------------- shortcuts
    def _bind_shortcuts(self):
        self.master.bind('<Control-o>', lambda e: self._load_image_callback())
        self.master.bind('<Control-s>', lambda e: self._save_mask_callback())
        self.master.bind('<Control-S>', lambda e: self._save_foreground_callback())
        self.master.bind('<Control-z>', lambda e: self._undo_click())
        self.master.bind('<Escape>', lambda e: self._reset())

    def _set_busy(self, text):
        self.statusbar.config(text=text)
        self.master.update_idletasks()

    def _set_idle(self):
        self.statusbar.config(text="就绪")

    # --------------------------------------------------------- panel scroll
    def _on_ctrl_configure(self, event):
        self.ctrl_canvas.configure(scrollregion=self.ctrl_canvas.bbox('all'))

    def _on_ctrl_canvas_configure(self, event):
        self.ctrl_canvas.itemconfigure(self._ctrl_window, width=event.width)

    def _bind_panel_wheel(self, widget):
        widget.bind('<MouseWheel>', self._on_ctrl_wheel)
        for child in widget.winfo_children():
            self._bind_panel_wheel(child)

    def _on_ctrl_wheel(self, event):
        self.ctrl_canvas.yview_scroll(-event.delta // 120, 'units')

    # ------------------------------------------------------------- callbacks
    def _load_image_callback(self):
        filename = filedialog.askopenfilename(
            parent=self.master,
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("所有文件", "*.*")],
            title="选择图片")
        if filename:
            self._load_image_from_path(filename)

    def _load_image_from_path(self, path):
        log.info("加载图片：%s", path)
        image = _imread_unicode(path)
        if image is None:
            log.error("无法读取图片：%s", path)
            messagebox.showerror("错误", f"无法读取图片：{path}")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        log.info("图片解码完成：%dx%d", image.shape[1], image.shape[0])

        # CPU 上图像编码要几秒，先给提示再阻塞，避免误以为卡死。
        self._set_busy("正在编码图像...")
        try:
            self.controller.set_image(image)
        except FileNotFoundError as e:
            self._set_idle()
            self._update_instruction()
            messagebox.showerror("缺少 SAM 权重", str(e))
            return
        except Exception as e:
            self._set_idle()
            self._update_instruction()
            messagebox.showerror("加载失败", f"图像编码失败：{e}")
            return
        self._set_idle()

    def _save_mask_callback(self):
        mask = self.controller.result_mask
        if self.controller.image is None or mask is None:
            messagebox.showwarning("提示", "还没有分割结果，请先加载图片并点击分割。")
            return
        filename = filedialog.asksaveasfilename(
            parent=self.master, initialfile='mask.png', defaultextension='.png',
            filetypes=[("PNG", "*.png"), ("BMP", "*.bmp"), ("所有文件", "*.*")])
        if filename:
            log.info("保存掩码到 %s", filename)
            _imwrite_unicode(filename, mask * 255)

    def _save_foreground_callback(self):
        if self.matte_var.get():
            # Matting is a sparse solve (seconds); give a hint before blocking.
            self._set_busy("正在计算 alpha matting...")
            alpha = self.controller.result_matte
            self._set_idle()
        else:
            alpha = self.controller.result_alpha
        image = self.controller.image
        if image is None or alpha is None:
            messagebox.showwarning("提示", "还没有分割结果，请先加载图片并点击分割。")
            return
        filename = filedialog.asksaveasfilename(
            parent=self.master, initialfile='foreground.png', defaultextension='.png',
            filetypes=[("PNG", "*.png"), ("所有文件", "*.*")])
        if filename:
            log.info("保存前景抠图到 %s", filename)
            rgba = np.dstack([image, alpha])
            _imwrite_unicode(filename, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

    def _click_callback(self, is_positive, x, y):
        if self.controller.image is None:
            messagebox.showwarning("提示", "请先加载一张图片。")
            return
        self.controller.add_click(x, y, is_positive)

    def _box_callback(self, x1, y1, x2, y2):
        if self.controller.image is None:
            messagebox.showwarning("提示", "请先加载一张图片。")
            return
        self.controller.set_box(x1, y1, x2, y2)

    def _undo_click(self):
        self.controller.undo_click()

    def _reset(self):
        self.controller.reset()

    def _resegment(self):
        self.controller.resegment()

    def _on_edge_mode_changed(self, event=None):
        self.controller.set_edge_mode(self.edge_mode.get())
        self.controller.resegment()

    def _on_refine_changed(self):
        self.controller.set_refine_mode('grabcut' if self.refine_var.get() else 'off')
        self.controller.resegment()

    def _set_cleanup(self, value):
        k = int(round(float(value)))
        if k % 2 == 0:
            k += 1
        self.cleanup_kernel.set(k)
        self.controller.set_cleanup_kernel(k)

    def _on_cleanup_release(self, event=None):
        self.controller.resegment()

    # --------------------------------------------------------------- redraw
    def _update_image(self, reset_canvas=False):
        image = self.controller.get_visualization(self.alpha_blend.get(), self.click_radius.get())
        if image is not None:
            if reset_canvas:
                self.image_on_canvas.set_image(image, fit=True)
            else:
                self.image_on_canvas.reload_image(image)
        self._update_instruction()

    def _update_instruction(self):
        if self.controller.image is None:
            self.instruction.config(text="请先加载一张图片", bg='#fff3cd')
            self.status.config(text="前景 0 点 · 背景 0 点")
            return

        npos, nneg = self.controller.click_counts()
        self.status.config(text=f"前景 {npos} 点 · 背景 {nneg} 点")

        if npos == 0:
            self.instruction.config(
                text="请在目标上左键点击 1~3 个点，或左键拖拽框出目标", bg='#fff3cd')
        elif nneg == 0:
            self.instruction.config(
                text="已标记前景。可右键点击背景排除误选，或继续左键补充前景", bg='#d9ead3')
        else:
            self.instruction.config(
                text="分割已就绪。继续点击可微调，或从「文件」菜单保存结果", bg='#d9ead3')
