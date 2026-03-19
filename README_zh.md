# FaceFuck

用脸写 Brainfuck。是的，用脸。

FaceFuck 是一个基于 **PyQt5** 的小型“IDE”，用 **DeepFace** + **OpenCV** 把你的 **面部表情** 和 **头部位置** 映射成 Brainfuck 指令。解释器在**独立进程**里运行，所以就算你写出无限循环，最多是 Brainfuck 自己尴尬，UI 不会跟着陪葬。

如果你是来提高生产力的：建议先喝口水冷静一下。\
如果你是来整活的：欢迎加入面部肌肉训练营。

## 你会得到什么

- 摄像头实时识别 → 表情/位置 → 输入 Brainfuck（手可以休假）
- 抗抖动：平滑 + 滞回确认，减少“刚笑一下就变生气”的跳变
- 多脸场景更稳定：尽量锁定同一张脸，不随机切换目标
- 头部位置输入 `[` / `]`，上/下用于 RUN/STOP 控制
- 视频叠加九宫格参考线，瞄准更直观
- 纸带网格实时显示（内存单元），报错信息清晰

## 环境要求

- 系统：Windows 优先（DirectShow 兼容更好）。Linux/macOS 理论上可用，可能需要小改动。
- Python：建议 3.10+
- 摄像头权限
- 依赖（见 `requirements.txt`）：
  - `opencv-python`
  - `deepface`
  - `pyqt5`

DeepFace 首次运行可能会下载模型（除非你指定本地缓存目录）。

## 安装

### 方案 A：venv（推荐）

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

### 方案 B：conda

```powershell
conda create -n facefuck python=3.10 -y
conda activate facefuck
pip install -r requirements.txt
```

### 配置 DeepFace 模型缓存（推荐）

为了让模型下载/缓存放在项目目录（或你指定的任意目录），设置：

```powershell
$env:DEEPFACE_HOME = (Resolve-Path .).Path
```

`main.py` 默认会把 `DEEPFACE_HOME` 指向项目目录。

## 运行

```powershell
python main.py
```

## 使用方式（零键盘流派）

1. 把脸放进摄像头画面里（建议不要只放刘海，模型对眼镜、口罩的适配效果不佳）。
2. **脸处在中间区域（vertical=middle）** 才允许输入代码。
3. 保持某个触发状态约 **1.5 秒**，系统才会确认输入一个符号。
4. 脸 **上移** 并保持约 **0.8 秒** → **RUN**。
5. 脸 **下移** 并保持约 **0.8 秒** → **STOP**。

界面会显示：

- 人脸框 + `(position / emotion)` 标注
- 单一 HOLD 进度条（up/down 控制时会变天蓝色）
- 纸带网格作为主要运行可视化

## 当前映射表

### 位置（优先级最高）

- **left** → `[`
- **right** → `]`

### 表情（仅当 position 为 center 时生效）

- **happy** → `+`（当前单元 +1）
- **sad** → `-`（当前单元 -1）
- **angry** → `<`（指针左移）
- **surprise** → `>`（指针右移）
- **fear** → `,`（输入 1 字节，可能会等待）
- **disgust** → `.`（输出 1 字节）
- **neutral** → `BACKSPACE`（删除最后一个符号）

## 示例图片（左侧参考栏）

左侧 “Sample” 列会从以下位置读取示例图：

- `icon/<key>.jpg|png`

支持的 key：

- `left`, `right`, `up`, `down`, `center`（或 `middle`）
- `happy`, `sad`, `angry`, `surprise`, `fear`, `disgust`, `neutral`

没有图片就留空，不影响运行。

## 参数调优（如果你的脸太强了）

如果输入太抖或太慢，可以在 `src/qt_ui.py` 调这些参数：

- `analysis_interval`（DeepFace 分析频率）
- `hold_duration`（输入符号所需保持时间）
- `vertical_hold_duration`（RUN/STOP 保持时间）
- `StateFilter` 的窗口与确认帧数（抗抖更强但更“慢热”）

## 常见问题

- “Cannot open camera / 无法打开摄像头”：
  - 关闭占用摄像头的软件（微信/浏览器/会议软件等）
  - 检查系统摄像头权限
  - 尝试修改 `CameraWorker(camera_index=...)`
- 首次运行很慢：
  - DeepFace 在下载模型；设置 `DEEPFACE_HOME` 可复用缓存
- TensorFlow/oneDNN 警告：
  - 一般无害，只是吵。如果你想安静一点：
    - `TF_ENABLE_ONEDNN_OPTS=0`（可选）

## 目录结构（简版）

- `main.py` — 启动 PyQt5 UI
- `src/qt_ui.py` — UI、摄像头线程、与 BF 进程通信
- `src/recognizer.py` — DeepFace 分析 + 位置/上下判断
- `src/filter.py` — 平滑与抗抖滤波
- `src/state_machine.py` — hold-to-trigger 状态机与映射
- `src/bf_worker_process.py` — Brainfuck 独立进程 + 纸带更新

## 隐私说明

本项目在本地使用摄像头，不会上传任何内容。\
只有你、你的表情、以及一门专门用来折磨人类的语言。
