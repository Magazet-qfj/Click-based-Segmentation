"""Modal settings dialog for the SAM interactive-segmentation app.

Groups the app's tunable parameters — engine constants, edge/matte options, and
the quick display controls — into one window reachable from the 设置 menu.
"""

import tkinter as tk
from tkinter import ttk


class SettingsDialog(tk.Toplevel):
    """Edit the app's parameters and hand the result back as a dict.

    ``initial`` is the current values dict. After the dialog closes,
    ``self.result`` is the edited dict, or ``None`` if the user cancelled.
    """

    def __init__(self, master, initial):
        super().__init__(master)
        self.title("参数设置")
        self.resizable(False, False)
        self.transient(master)
        self.result = None
        self._initial = initial
        self._vars = {}
        self._free_floats = set()

        try:
            self.grab_set()          # make it modal
        except tk.TclError:
            pass                     # (e.g. headless tests) still fine via wait_window

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self._build(initial)
        self._center_over(master)

    # ---------------------------------------------------------------- layout
    def _build(self, initial):
        body = ttk.Frame(self, padding=12)
        body.pack(fill='both', expand=True)
        body.columnconfigure(1, weight=1)

        row = 0
        row = self._group(body, "显示", row)
        self._spin(body, row, "掩码透明度 (0~1)", 'alpha_blend', initial, 0.0, 1.0, 0.05); row += 1
        self._spin(body, row, "点击点大小", 'click_radius', initial, 1, 8); row += 1

        row = self._group(body, "掩码清理", row)
        self._spin(body, row, "清理核大小（奇数）", 'cleanup_kernel', initial, 1, 15); row += 1

        row = self._group(body, "边缘羽化", row)
        self._choice(body, row, "羽化方式", 'edge_mode', initial, ('guided', 'gaussian')); row += 1
        self._spin(body, row, "guided 窗口半径", 'edge_radius', initial, 1, 16); row += 1
        self._float_entry(body, row, "guided 正则项 eps", 'edge_eps', initial); row += 1
        self._bool(body, row, "GrabCut 全分辨率边界精修", 'refine', initial); row += 1
        self._spin(body, row, "GrabCut 迭代次数", 'grabcut_iters', initial, 1, 20); row += 1
        self._spin(body, row, "GrabCut 分辨率上限", 'grabcut_max_side', initial, 256, 4096, 64); row += 1
        self._spin(body, row, "GrabCut 概率带半宽", 'grabcut_band', initial, 1, 32); row += 1

        row = self._group(body, "导出 / matting", row)
        self._bool(body, row, "导出时启用 alpha matting", 'matte', initial); row += 1
        self._spin(body, row, "trimap 未知带半宽 (px)", 'matte_radius', initial, 1, 128); row += 1
        self._spin(body, row, "matting 求解分辨率上限", 'matte_max_side', initial, 128, 2048, 64); row += 1

        row = self._group(body, "引擎", row)
        self._readonly(body, row, "模型", initial['model_type']); row += 1
        self._spin(body, row, "工作图最长边（改后重新编码）", 'max_side', initial, 256, 4096, 64); row += 1

        btns = ttk.Frame(body)
        btns.grid(row=row, column=0, columnspan=2, sticky='e', pady=(12, 0))
        ttk.Button(btns, text="取消", command=self._on_cancel).pack(side='right', padx=(6, 0))
        ttk.Button(btns, text="确定", command=self._on_ok).pack(side='right')

    def _group(self, parent, title, row):
        ttk.Label(parent, text=title, font=('Microsoft YaHei', 9, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky='w', pady=(8, 2))
        return row + 1

    def _row(self, parent, row, label, widget):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=(0, 10), pady=2)
        widget.grid(row=row, column=1, sticky='ew', pady=2)

    def _spin(self, parent, row, label, key, initial, lo, hi, inc=None):
        val = initial[key]
        var = (tk.DoubleVar(value=float(val)) if isinstance(val, float)
               else tk.IntVar(value=int(val)))
        self._vars[key] = var
        w = ttk.Spinbox(parent, from_=lo, to=hi, increment=inc if inc is not None else 1,
                        textvariable=var, width=12)
        self._row(parent, row, label, w)

    def _float_entry(self, parent, row, label, key, initial):
        var = tk.StringVar(value=str(initial[key]))
        self._vars[key] = var
        self._free_floats.add(key)
        w = ttk.Entry(parent, textvariable=var, width=14)
        self._row(parent, row, label, w)

    def _bool(self, parent, row, label, key, initial):
        var = tk.BooleanVar(value=bool(initial[key]))
        self._vars[key] = var
        ttk.Checkbutton(parent, text=label, variable=var).grid(
            row=row, column=0, columnspan=2, sticky='w', pady=2)

    def _choice(self, parent, row, label, key, initial, values):
        var = tk.StringVar(value=initial[key])
        self._vars[key] = var
        w = ttk.Combobox(parent, textvariable=var, values=values, state='readonly', width=12)
        self._row(parent, row, label, w)

    def _readonly(self, parent, row, label, value):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=(0, 10), pady=2)
        ttk.Label(parent, text=value).grid(row=row, column=1, sticky='w', pady=2)

    def _center_over(self, master):
        self.update_idletasks()
        w = self.winfo_width()
        h = self.winfo_height()
        x = master.winfo_rootx() + (master.winfo_width() - w) // 2
        y = master.winfo_rooty() + (master.winfo_height() - h) // 2
        self.geometry(f"+{max(0, x)}+{max(0, y)}")

    # --------------------------------------------------------------- results
    def _on_ok(self):
        res = {}
        for key, var in self._vars.items():
            val = var.get()
            if key in self._free_floats:
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = self._initial[key]
            res[key] = val
        self.result = res
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()
