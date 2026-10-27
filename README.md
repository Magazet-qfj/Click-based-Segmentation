# 点击式交互分割 (SAM)

一个点击驱动的交互式图像分割桌面程序。向用户展示一个窗口，引导用户点击要分割的目标区域，
用 Meta 的 **Segment Anything Model（SAM，ViT-B / ViT-L）** 实时把前景从背景中分离出来。

交互方式延续 [SimpleClick](../SimpleClick) 的点选式交互（点选前景、排除背景），并加上
左侧工具栏切换画笔；引擎换成了预训练的 SAM —— 无需自己训练、CPU 上可跑。相比初版用的经典
GrabCut，SAM 的分割结果具有空间连通性、不会因新增点击而删掉已选区域，精度也更高。
SimpleClick 在这里只作为 UI 与交互范式的参考。

## 运行

**双击运行**：直接双击 `Project\run.bat` 即可启动（自动定位 Python 环境并运行 `main.py`，
出错时会停住显示提示）。

命令行方式：

```bash
cd Project
python main.py                 # 启动空窗口，通过「文件 → 加载图片」载入图像
python main.py --image 图.jpg   # 启动时直接载入某张图
python main.py --model-type vit_l   # 换 ViT-L 编码器（需先下 1.25GB 权重；更慢，仅个别细结构场景更细）
```

> `run.bat` 优先使用 `SimpleClick\simpleclick_env` 里的 Python；找不到时退回系统 `python`。
> 若不想要启动时的黑色控制台窗口，可把 `run.bat` 里的 `python.exe` 改成 `pythonw.exe`。
> 启动后黑色控制台会**实时打印调试信息**（加载图片、每次前景/背景点击、图像编码耗时、分割耗时与前景占比、
> 导出等），方便观察程序每一步做了什么。

依赖为 `opencv-python`、`numpy`、`Pillow`、`segment-anything`（Tkinter 是 Python 标准库；
`torch`/`torchvision` 由 `SimpleClick\simpleclick_env` 提供）：

```bash
pip install -r requirements.txt
```

## 操作方式

| 操作 | 效果 |
|------|------|
| 左侧工具栏切换工具 | 前景画笔 / 背景画笔 / 框选目标 / 删除标记 |
| 前景画笔 | 左键添加前景点，右键删除前景点 |
| 背景画笔 | 左键添加背景点，右键删除背景点 |
| 框选目标 | 左键拖拽框出目标范围 |
| 删除标记 | 左键点删标记点（前景或背景），左键拖拽框选批量删除 |
| 中键拖动 | 平移图像 |
| 滚轮 | 缩放图像 |

| 快捷键 | 动作 |
|--------|------|
| `Ctrl+O` | 加载图片 |
| `Ctrl+S` | 保存掩码 |
| `Ctrl+Shift+S` | 保存前景抠图 |
| `Ctrl+Z` | 撤销上一次点击 |
| `Esc` | 清空所有点击 |

引导流程（窗口右侧会有逐步提示）：

1. 加载图片；
2. 在目标上左键点 1~3 个点；
3. （可选）切到「背景画笔」，在背景上左键点几个点，排除误选区域；
4. 掩码会随每次点击实时更新；
5. 满意后从「文件」菜单保存**掩码**（二值图）或**前景抠图**（透明背景 PNG）。

「点击管理」里可撤销上次点击、清空所有点击、重新分割。删除标记点改用左侧工具栏的
「删除标记」工具——左键点击画布上的某个标记点即可**单独删除**该点，左键拖拽则框选**批量删除**
框内的全部标记点（绿=前景，红=背景）；光标会变成 ✕，提示条变红提醒当前处于删除模式，
便于纠正误点而不必重来。

右侧控制面板还可调「边缘羽化方式」（`guided` 边缘感知 / `gaussian` 均匀模糊）、「掩码清理强度」
（越大填越多网格空洞；中值固定 3px，不再误抹小圆斑点）、「GrabCut 全分辨率边界精修」（可选，个别图更细、
慢约几秒，实测对 guided filter 边际增益中性）。导出时可勾选「抠图用真 alpha matting」——用闭式
matting（发丝级，半透明发丝会与背景自然融合）替代 guided filter 羽化，导出前会多花几秒求解；
其 trimap 未知带半径按全分辨率像素自适应缩放（默认 25px），足以覆盖一段发丝边缘、把发丝间的浅色背景
还原出来。

菜单栏「设置 → 参数设置...」会弹出汇总窗口，可一次性调整上述全部参数，以及右侧面板没有暴露的
模型常量——工作图最长边 `max_side`、guided filter 的窗口半径 `edge_radius` / 正则项 `edge_eps`、
GrabCut 的迭代次数 / 分辨率上限 / 概率带半宽、matting 的 trimap 半径 / 求解分辨率上限。其中
「工作图最长边」改动后会**重新编码整图**（几秒，已有点击/框按原图坐标保留），其余改动即时重分割
生效。

## 代码结构

| 文件 | 职责 |
|------|------|
| `main.py` | 入口，创建 Tk 窗口与 `SamSegmenter`（`--model-type` 选 vit_b/vit_l，`--refine grabcut`） |
| `app.py` | Tkinter 界面：菜单、左侧工具栏、画布、引导提示、控制面板 |
| `canvas.py` | 可缩放/平移的画布，把鼠标事件换算成图像像素坐标并按当前工具分发（画笔左右键加点/删点、框选、删除点/框选批量删） |
| `controller.py` | UI 与引擎之间的胶水层，每次点击后重跑分割并触发重绘 |
| `settings.py` | 「设置 → 参数设置」对话框：汇总引擎常量 + 右侧面板选项，一次修改并应用 |
| `segmenter.py` | 分割引擎：把点击作 SAM 的点提示，输出二值前景掩码 |
| `visuals.py` | 可视化：掩码叠加 + 点击点绘制 |
| `eval_click.py` | 无 GUI 评估：NoC / IoU-随-点击数 + 边界 IoU/F1 边缘质量 |
| `bench_synth.py` | 合成基线：真值边界精确已知的样张（starburst/comb/…）跑 NoC + 边界 IoU/F1；`--model-type`、`--refine grabcut` |
| `matting.py` | 闭式 alpha matting（Levin et al. 2007，纯 numpy/scipy）：二值 mask → trimap → 真 alpha |
| `bench_matting.py` | P3 评估：二值 / guided / 闭式 matting 对半透明边缘的 alpha 还原 MAE |

## 引擎原理（segmenter.py）

采用 **SAM 点提示** 的交互范式：

1. 加载图像后，`SamPredictor.set_image` 用 ViT 把整幅图编码成特征（每张图只做一次，是慢步骤）；
2. 每次点击把"前景/背景"点作为提示，喂给轻量掩码解码器（毫秒级）；
3. 前景点 label=1、背景点 label=0，一次调用输出一张二值掩码；
4. 结果放大回原分辨率，并用边缘感知的 guided filter 羽化（`edge_mode='guided'`，可切回 `'gaussian'`），得到 `result_mask`（二值）与 `result_alpha`（软边抠图）。

点击越多（前景 + 背景配合），SAM 的定位越准。为兼顾大图与 CPU 实时响应，图像编码会在把最长边
缩到 1024 像素的工作图上进行，最后把掩码放大回原分辨率（`max_side` 可调）。SAM 掩码天然具备空间
连通性，且不会因新增点击而整体重算、删掉已选区域——这解决了初版 GrabCut 的两个主要问题。

> 首次运行需先下载权重：`Project\weights\sam_vit_b_01ec64.pth`（约 375MB）。若要换用
> ViT-L，另下 `sam_vit_l_0b3195.pth`（约 1.25GB），都用
> `https://dl.fbaipublicfiles.com/segment_anything/` 下的对应文件名。ViT-L 首帧编码约慢一倍，
> 且实测边界精度**非全面提升**（梳齿/小目标发丝更细，大图放射发丝反而变差），故默认仍用 ViT-B。

## 评估（headless）

`eval_click.py` 不需要 GUI，按「最大误差区域质心」策略自动点击，跑出 IoU-随-点击数曲线与
NoC@80/85/90，并在收敛后报告**边界 IoU / F1**（`--edge-radius` 控制带宽，默认 2px），衡量
抠图边缘与真值轮廓的贴合度——NoC 只能反映掩码内部收敛快慢，反映不了边缘精度。

```bash
python eval_click.py --image a.png --gt a_gt.png             # 空提示起步
python eval_click.py --image a.png --gt a_gt.png --init-box   # 从真值 bbox 起步
```

`bench_synth.py` 生成真值边界精确已知的合成发丝/梳齿样张并跑同一批指标（基准基线）：

```bash
python bench_synth.py                 # vit_b：NoC@85/90 + 边界 IoU/F1，样张存到 bench/
python bench_synth.py --model-type vit_l   # 换成 ViT-L 编码器对比
python bench_synth.py --refine grabcut      # 顺带测 GrabCut 全分辨率边界精修
```

`bench_matting.py` 在半透明边缘（真 alpha 精确已知）上对比三种 alpha 来源，量化发丝级抠图：

```bash
python bench_matting.py               # 二值 / guided / 闭式 matting 的边界带 alpha MAE
```
