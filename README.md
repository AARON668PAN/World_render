# G1 橘子任务上真机包

Unitree G1 + Dex3 三指手，官方 GEAR-SONIC 控制器（**默认版**）。学生看头部相机画面和关节角，每秒 50 次给控制器发
"64 维 token + 两只手各 7 个手指角"，消息格式和 NVIDIA 自己的 VLA 客户端逐字节相同。**不给指令机器人不动。**

两句指令（只认这两句）：

- `put the orange on the plate`
- `put the orange back on the table`

## 1. 谁跑什么

| 电脑 | 跑什么 |
|---|---|
| 机器人上的电脑 | 官方 `gear_sonic_deploy`（默认版 SONIC）和官方相机服务。本仓库不改它、也不带它的权重 |
| 一台带 NVIDIA 显卡的电脑，和机器人同一网段 | 本仓库：学生 + 客户端 |

学生必须有显卡：CPU 上一次规划约 320 ms，客户端最多容忍 240 ms。

## 2. 学校电脑上第一次安装

```bash
git clone -b g1-orange-plate-deploy --single-branch git@github.com:AARON668PAN/World_render.git g1_orange_plate_deploy
```
```bash
cd g1_orange_plate_deploy && ./setup_env.sh
```
```bash
./assemble_weights.sh
```
```bash
./run_selftest.sh
```

- `setup_env.sh`：在 `.venv` 里装 Python 3.12 的环境（有 `uv` 用 `uv`，没有用 `python3.12 -m venv`）。
- `assemble_weights.sh`：学生权重 627 MB，GitHub 单个文件上限 100 MB，所以切成 7 段放在 `student/` 里；这一步把它们拼回 `student/model.safetensors` 并核对校验值。
- `run_selftest.sh`：不需要机器人。它检查三件事：权重每一个数都和文件一致；固定输入下算出的数和训练机上一样；
  假机器人整套收发（没指令时 0 条动作消息，做的时候稳定 50 Hz）。最后一行是 `ALL_SELFTESTS_DONE` 才算过。
- 加载学生不联网：文字编码器和图像网络的权重都在 `student/model.safetensors` 里，分词器的小文件在 `assets/`。

## 3. 上机前必须确认

| 事 | 怎么确认 |
|---|---|
| 控制器是**默认版** SONIC | 启动命令里不能有 `--cp policy/sonic_v1_1/...`。v1.1 的 token 含义不同，C++ 不报错，机器人会乱动。默认版两个文件的 SHA-256：`model_encoder.onnx` 以 `013ab0287236aa27` 开头，`model_decoder.onnx` 以 `c7241a123eaa36b5` 开头（仿真用的权重已和它逐位比过，一致） |
| 手指电机顺序 | `ROBOT_IP=<机器人 IP> ./run_client.sh --hand-check`，用手掰一根手指，看哪一项在变；食指、中指反了就加 `--hand-order middle_first` 或 `index_first` |
| 头部相机 | 学生是按 640x480、横向约 89 度、俯视约 53 度的画面训的。视角差很多就要重渲染重训 |
| 场景摆放 | 见下 |

摆放（以机器人骨盆中心正下方为原点，前为 x，左为 y）：

- 桌面高 76 cm，近边在身前 12 cm，桌子左右居中。
- 盘子（灰色，直径约 18 cm）中心：前 35 cm，偏左 4 cm。
- 橘子（直径约 7.8 cm）起点：前 35 cm，偏右 13 cm。放回桌面的落点：前 40 cm，偏右 16 cm。
- 背景尽量和拍示范视频时一样（白桌、素墙）。

## 4. 上机步骤

机器人吊着或有人扶，第一次一定这样。

1. 机器人上（官方部署终端），默认版：
   ```bash
   cd gear_sonic_deploy && ./deploy.sh --input-type zmq_manager --zmq-host <跑学生那台电脑的 IP> real
   ```
   回车确认后机器人到默认站姿等待。**这个终端里之后只碰 `O`（急停，机器人会瘫软）**。别的键会被转发：回车会切到预置
   参考动作，`T` 会播放它（里面有舞蹈）。
2. 机器人上起官方相机服务：
   ```bash
   python -m gear_sonic.camera.composed_camera --ego-view-camera oak --ego-view-device-id <MXID> --port 5555
   ```
3. 学生电脑上，按这个顺序各跑一次（`GR00T_WBC` 指向官方 GR00T-WholeBodyControl 的代码目录，读官方相机流要用；
   用普通 USB 相机就设 `CAMERA=cv:0`）：
   ```bash
   ROBOT_IP=192.168.123.164 GR00T_WBC=<官方代码目录> ./run_client.sh --hand-check
   ```
   ```bash
   ROBOT_IP=192.168.123.164 GR00T_WBC=<官方代码目录> ./run_client.sh --dry-run
   ```
   ```bash
   ROBOT_IP=192.168.123.164 GR00T_WBC=<官方代码目录> ./run_client.sh --no-student
   ```
   ```bash
   ROBOT_IP=192.168.123.164 GR00T_WBC=<官方代码目录> ./run_client.sh
   ```
   - `--hand-check`：只读，核对手指顺序。
   - `--dry-run`：什么都不发，看相机、状态、学生延迟是否正常。
   - `--no-student`：只走到起始姿势再回来，打印真机和仿真起始姿势差多少。
   - 不带开关：正式做。
4. 在客户端里打字：
   - `start`：让控制器自己站住（官方的 PLANNER 模式）。此时学生没接上机器人。
   - 两句指令之一：机器人才动。
   - `stop`：中断，回到起始姿势再交还控制器。
   - `status` / `quit`；`estop` 等于官方急停。

一条指令的过程：切到 token 模式 → 3 秒走到示范的起始姿势 → 停 1 秒 → 核对姿势 → 学生最多做 11.3 秒 → 1.5 秒回起始
姿势 → 交还控制器站着。做完指令就清空，下一次要重新打字。每一局的记录存在 `logs/`。

## 5. 保护措施

客户端这边（C++ 那边对 token 不做任何检查）：

- 指令只认配置里的两句话。空的、别的话、正在做时再给，一律不动。
- 没先 `start`、收不到机器人状态或相机画面时，指令被拒绝。
- 学生开始前核对手臂和腰的角度，离示范起始姿势超过 0.35 rad 就不放学生（`--start-tolerance`）。
- 每条消息：有非数字就中止；token 限在 ±0.6；手指目标限在关节范围内，每拍最多变 0.12 rad。
- 相机超过 0.5 秒或状态超过 0.2 秒没更新，中止并返回。
- 学生 0.5 秒没给出新计划，中止并返回。
- 客户端正常退出或报错退出时，把控制器切回它自己的站立模式。

管不到的：

- 客户端进程被杀或断网时，C++ 会一直保持最后一个 token，没有超时。只能靠人按 `O`。
- 机器人会不会碰到桌子、会不会失去平衡，客户端不判断。

## 6. 已知风险

1. **起始姿势只有一个。** 真机只用"视频原位置"的起始姿势，橘子摆偏了行不行看仿真考试第 4 项。
2. **真机手指夹得比仿真松。** C++ 把手指目标限制在实际角度 ±0.25 rad、刚度 1.5，最大约 0.375 N·m；示范里拇指中节
   用到 0.51、中指根 0.40。仿真考试第 5 项按这个限制考。
3. **SONIC 的仿真到真机差距没测过。** 默认版 SONIC 跟手腕很差，老师是靠挪参考动作来补的；真机响应要是不一样，起始
   姿势就对不上，`--no-student` 那一步能量出来。
4. **相机没核对。** 见第 3 节。
5. 推理延迟：仿真考试第 3 项按 100 ms 考；真机上客户端会打印每次的实际延迟。

## 7. 仿真成绩（2026-11-18）

学生 `ditrn_op_v8tok`：从零训的 `dit_resnet`，30,000 步，302 条示范（102 个摆放位置，v9 渲染的头部相机画面），
输出 64 维 token + 14 个手指角。

| 环节 | 是谁的成绩 | 结果 |
|---|---|---|
| 老师验收（102 个位置 × 32） | 老师 | 3203/3264 = 98.1% |
| 标签重放（老师的 token + 手指按学生的节奏原样放） | 标签，学生的上限 | 同物理 192/192，换物理 191/192 |
| 考试 1：换装仿真（12 个位置 × 8） | 学生 | 39/96 = 41%（桌→盘 21/48，盘→桌 18/48） |
| 考试 2：回放训练用的渲染画面 | 学生 | 进行中 |
| 考试 3：加 100 ms 延迟 | 学生 | 待考 |
| 考试 4：按真机方式从待命姿势起步 | 学生 | 待考 |
| 考试 5：手指力矩限到真手水平 | 学生 | 待考 |

对照：上一版学生（旧格式，多输出 14 个手臂修正量，不能直接上官方控制器）考试 1 是 66/96 = 69%，考试 2 是 83/96 = 86%。

## 8. 文件

| 路径 | 内容 |
|---|---|
| `client/g1_client.py` | 机器人客户端：指令开关、起始流程、保护、收发 |
| `client/gate.py` | "不给指令不动"的状态机 |
| `client/wire.py` | 官方消息格式（和官方打包函数逐字节一致） |
| `client/student_runtime.py` | 学生推理，和仿真考试用的是同一个文件 |
| `client/runtime_server.py` | 学生进程；`client/offline_assets.py` 让它不联网加载并核对权重 |
| `client/selftest.py`、`protocol_test.py`、`fake_robot.py`、`check_protocol.py`、`stub_server.py` | 自检 |
| `lerobot_policy_dit_resnet/` | 学生的模型代码（LeRobot 插件 `dit_resnet`） |
| `config/config_orange_plate.json` | 两句指令各自的起始 token、起始姿势、走过去的 token 序列、时长 |
| `student/` | 学生的配置、归一化参数、自检参考值；权重切成 7 段 `model.safetensors.part00~06`，用 `assemble_weights.sh` 拼回 |
| `assets/` | CLIP 分词器和配置的小文件（MIT 许可，无权重） |
