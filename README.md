# 点击式交互分割 (GrabCut)

一个点击驱动的交互式图像分割桌面程序。向用户展示一个窗口，引导用户点击要分割的目标区域，
用 OpenCV 的 **GrabCut** 算法实时把前景从背景中分离出来。

本项目的交互方式与 [SimpleClick](../SimpleClick) 的 demo 一致（左键前景 / 右键背景），
但引擎换成了无需训练、CPU 上毫秒级响应的经典 GrabCut 算法，因此可以在这台无 GPU 的机器上
直接运行。SimpleClick 在这里只作为 UI 与交互范式的参考。

## 运行

**双击运行**：直接双击 `Project\run.bat` 即可启动（自动定位 Python 环境并运行 `main.py`，
出错时会停住显示提示）。

命令行方式：

```bash
cd Project
python main.py                 # 启动空窗口，通过「文件 → 加载图片」载入图像
python main.py --image 图.jpg   # 启动时直接载入某张图
```

> `run.bat` 优先使用 `SimpleClick\simpleclick_env` 里的 Python；找不到时退回系统 `python`。
> 若不想要启动时的黑色控制台窗口，可把 `run.bat` 里的 `python.exe` 改成 `pythonw.exe`。

依赖只有 `opencv-python`、`numpy`、`Pillow`（Tkinter 是 Python 标准库）：

```bash
pip install -r requirements.txt
```

## 操作方式

| 操作 | 效果 |
|------|------|
| 左键点击 | 标记**前景**（要分割的目标） |
| 右键点击 | 标记**背景**（排除区域） |
| 右键拖动 | 平移图像 |
| 滚轮 | 缩放图像 |

引导流程（窗口右侧会有逐步提示）：

1. 加载图片；
2. 在目标上左键点 1~3 个点；
3. （可选）在背景上右键点几个点，排除误选区域；
4. 掩码会随每次点击实时更新；
5. 满意后从「文件」菜单保存**掩码**（二值图）或**前景抠图**（透明背景 PNG）。

## 代码结构

| 文件 | 职责 |
|------|------|
| `main.py` | 入口，创建 Tk 窗口与 `GrabCutSegmenter` |
| `app.py` | Tkinter 界面：菜单、画布、引导提示、控制面板 |
| `canvas.py` | 可缩放/平移的画布，把鼠标事件换算成图像像素坐标 |
| `controller.py` | UI 与引擎之间的胶水层，每次点击后重跑分割并触发重绘 |
| `segmenter.py` | 分割引擎：把点击转成 GrabCut 的种子掩码，输出二值前景掩码 |
| `visuals.py` | 可视化：掩码叠加 + 点击点绘制 |

## 引擎原理（segmenter.py）

采用 GrabCut 的 **scribble（涂抹）式** 用法，无需手动框选：

1. 整幅图初始化为"可能背景"（`GC_PR_BGD`）；
2. 把用户的点击画成硬种子：前景点击 → 确定前景（`GC_FGD`），背景点击 → 确定背景（`GC_BGD`）；
3. 调用 `cv2.grabCut(..., GC_INIT_WITH_MASK)` 迭代，从前景种子向外扩张出目标；
4. 输出"确定前景 ∪ 可能前景"作为二值掩码。

点击越多（前景 + 背景配合），GrabCut 的颜色模型越准，分割越精细。为兼顾大图与实时响应，
计算会在把最长边缩到 800 像素的工作图上进行，最后把掩码放大回原分辨率。
