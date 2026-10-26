# 点击式交互分割 (SAM)

一个点击驱动的交互式图像分割桌面程序。向用户展示一个窗口，引导用户点击要分割的目标区域，
用 Meta 的 **Segment Anything Model（SAM，ViT-B）** 实时把前景从背景中分离出来。

本项目的交互方式与 [SimpleClick](../SimpleClick) 的 demo 一致（左键前景 / 右键背景），
但引擎换成了预训练的 SAM —— 无需自己训练、CPU 上可跑。相比初版用的经典 GrabCut，SAM 的
分割结果具有空间连通性、不会因新增点击而删掉已选区域，精度也更高。SimpleClick 在这里只作为
UI 与交互范式的参考。

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

依赖为 `opencv-python`、`numpy`、`Pillow`、`segment-anything`（Tkinter 是 Python 标准库；
`torch`/`torchvision` 由 `SimpleClick\simpleclick_env` 提供）：

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

右侧控制面板还可调「边缘羽化方式」（`guided` 边缘感知 / `gaussian` 均匀模糊）与「掩码清理强度」
（越小越保留发丝等细丝，越大越去除伪影）。

## 代码结构

| 文件 | 职责 |
|------|------|
| `main.py` | 入口，创建 Tk 窗口与 `SamSegmenter` |
| `app.py` | Tkinter 界面：菜单、画布、引导提示、控制面板 |
| `canvas.py` | 可缩放/平移的画布，把鼠标事件换算成图像像素坐标 |
| `controller.py` | UI 与引擎之间的胶水层，每次点击后重跑分割并触发重绘 |
| `segmenter.py` | 分割引擎：把点击作 SAM 的点提示，输出二值前景掩码 |
| `visuals.py` | 可视化：掩码叠加 + 点击点绘制 |
| `eval_click.py` | 无 GUI 评估：NoC / IoU-随-点击数 + 边界 IoU/F1 边缘质量 |

## 引擎原理（segmenter.py）

采用 **SAM 点提示** 的交互范式：

1. 加载图像后，`SamPredictor.set_image` 用 ViT 把整幅图编码成特征（每张图只做一次，是慢步骤）；
2. 每次点击把"前景/背景"点作为提示，喂给轻量掩码解码器（毫秒级）；
3. 前景点 label=1、背景点 label=0，一次调用输出一张二值掩码；
4. 结果放大回原分辨率，并用边缘感知的 guided filter 羽化（`edge_mode='guided'`，可切回 `'gaussian'`），得到 `result_mask`（二值）与 `result_alpha`（软边抠图）。

点击越多（前景 + 背景配合），SAM 的定位越准。为兼顾大图与 CPU 实时响应，图像编码会在把最长边
缩到 1024 像素的工作图上进行，最后把掩码放大回原分辨率（`max_side` 可调）。SAM 掩码天然具备空间
连通性，且不会因新增点击而整体重算、删掉已选区域——这解决了初版 GrabCut 的两个主要问题。

> 首次运行需先下载权重：`Project\weights\sam_vit_b_01ec64.pth`（约 375MB，来自
> `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`）。

## 评估（headless）

`eval_click.py` 不需要 GUI，按「最大误差区域质心」策略自动点击，跑出 IoU-随-点击数曲线与
NoC@80/85/90，并在收敛后报告**边界 IoU / F1**（`--edge-radius` 控制带宽，默认 2px），衡量
抠图边缘与真值轮廓的贴合度——NoC 只能反映掩码内部收敛快慢，反映不了边缘精度。

```bash
python eval_click.py --image a.png --gt a_gt.png             # 空提示起步
python eval_click.py --image a.png --gt a_gt.png --init-box   # 从真值 bbox 起步
```
