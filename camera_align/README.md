# 让仿真头部相机和真机一样

学生是用仿真渲染的头部相机画面训的。那个仿真相机（`head_cam_calib_training.json`，从宇树官方 G1 Dex3 数据标定出来）
和我们这台 G1 头上的 RealSense D435i 差很多，所以要把仿真相机改成真机的参数，重新渲染示范、重训学生。

## 1. 真机测出来的相机（2026-11-18）

`head_cam_d435i.json`，格式和 `head_cam_calib_training.json` 一样（`position_torso` + `euler_zyx_rad` +
`focal_px` + `image_size`），另外多了 `fx fy cx cy`、畸变、MuJoCo 和 Isaac Sim 能直接填的数。读旧文件的代码换个路径就能用。

| | 训练用的相机 | 真机 D435i | 怎么来的 |
|---|---|---|---|
| 分辨率 | 640x480 | 640x480 | 相机服务实际发出的分辨率 |
| fx / fy（像素） | 327.3 / 327.3 | 608.49 / 607.97 | 相机自己报的内参 |
| cx / cy（像素） | 320 / 240 | 317.81 / 245.49 | 同上 |
| 畸变 | 无 | 无（系数全 0） | 同上 |
| 水平 x 垂直视角 | 88.7° x 72.5° | 55.5° x 43.1° | 由内参算 |
| 相对躯干向下倾斜 | 53.5° | 63.0° | D435i 加速度计 + 骨盆 IMU + 腰部三个关节，同时读 |
| 相对躯干左右歪 | +0.1° | -3.7° | 同上 |
| 相对躯干左右转（yaw） | 1.1° | 1.1°（**没测**，抄的训练值） | 重力测不出 yaw |
| 在躯干上的位置 | (4.1, 2.9, 41.6) cm | 同左（**没测**，抄的训练值） | 用第 4 步的叠图核对 |

两次测量（机载电脑重启前后）向下倾斜分别是 63.1° 和 63.0°。

**头部角度测完之后不要再动**，动了就重测。

## 2. 训练机器上要改的

1. 仿真里头部相机的位姿和内参，改成读 `camera_align/head_cam_d435i.json`：
   - 位姿：挂在 `torso_link` 上，`R.from_euler("zyx", euler_zyx_rad)`，相机系 x 前、y 左、z 上（和
     `tools/head_cam_calib/calibrate_head_cam.py`、`sim_bridge/ik_camera_track.py` 一样）。
   - 内参：640x480，`fx fy cx cy` 照填。只能填一个焦距的渲染器用 `focal_px`；MuJoCo 用 `mujoco_fovy_deg`；
     Isaac Sim 水平孔径 20.955 mm 时焦距填 `isaac_focal_length_mm_at_aperture_20.955`。
2. 先只渲染一张图核对（第 4 步），对齐了再重渲染全部示范。
3. 重训学生。学生输入是 168x224，和 640x480 都是 4:3，缩放方式不用改。
4. 新学生替换 `student/` 后，在真机这台电脑上跑 `./run_selftest.sh`。

## 3. 文件

| 文件 | 在哪跑 | 做什么 |
|---|---|---|
| `head_cam_d435i.json` | | 真机相机，仿真读这个 |
| `head_cam_calib_training.json` | | 原来训练用的相机（main 分支 `sim_bridge/assets/head_cam_calib.json` 的拷贝） |
| `measure_head_cam.py` | 接机器人的电脑，带 `unitree_sdk2py` 的环境 | 测相机倾斜和内参，写 `head_cam_d435i.json` |
| `realsense_probe.py` | 机器人上（`measure_head_cam.py` 通过 ssh 自动传过去跑） | 读 D435i 内参和加速度计 |
| `grab_frame.py` | 接机器人的电脑，本仓库 `.venv` | 存几张头部相机画面，和学生看到的完全一样 |
| `overlay_check.py` | 哪都行（只要 numpy + opencv） | 把 README 摆放位置画到真机图上；或把仿真图和真机图叠在一起 |
| `reference/` | | 测量时拍的真机画面和叠图 |

## 4. 怎么核对仿真和真机对齐了

1. 真机：机器人站成起始姿势，按 README 第 3 节摆盘子和橘子（盘子中心前 35 cm、偏左 4 cm；橘子前 35 cm、偏右 13 cm，
   原点是骨盆正下方），拍一张：
   ```bash
   GR00T_WBC=<GR00T-WholeBodyControl 目录> .venv/bin/python camera_align/grab_frame.py --n 1
   ```
2. 仿真：同样的起始姿势、同样的摆放，用新相机渲染一张 640x480 的图。
3. 叠图：
   ```bash
   python camera_align/overlay_check.py --real camera_align/reference/real_xxx.jpg --cam camera_align/head_cam_d435i.json --sim sim.png
   ```
   看 `check/real_vs_sim.png`：仿真的盘子边缘、桌子边应该压在真机的上面，差几个像素以内算对齐。
   只想检查 json 本身，不带 `--sim`，看 `check/real_layout.png` 里品红圈是否套住真盘子。

对不齐时：上下错开先查倾斜角和相机高度，左右错开查 yaw，大小不一样查盘子是不是摆远了/近了。

## 5. 重新测量

机器人不动（吊着、扶着或站着都行，姿势不影响结果），相机服务开不开都行：
```bash
conda activate groot_wbc   # 任何装了 unitree_sdk2py 的环境
python camera_align/measure_head_cam.py --iface <接机器人的网卡，如 enp11s0> --ssh g1
```
最后一行的加速度计抖动超过 0.3 m/s² 说明测的时候机器人在动，重测。

## 6. 已知的不确定

- yaw 和相机位置没测，抄的训练值。真机头部安装位置如果和宇树数据里的不一样，叠图会显示出来。
- 2026-11-18 的叠图里，真盘子比按 README 位置算的落点偏下、偏右、略大。更像是盘子摆得比 35 cm 近，还没用尺子确认。
- 腰部关节读数直接参与计算。腰如果是锁住的版本，读数不可信，要改成只用骨盆 IMU。
