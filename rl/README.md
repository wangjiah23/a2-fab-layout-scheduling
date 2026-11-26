# 全厂原槽位布局优化 · PPO

从空布局开始，按固定顺序将 1,068 台设备放入同尺寸的原始槽位。48 个 AMHS 设施固定。策略在 907 个决策时点给合法槽位打分，其余 161 次放置由同组规则确定。中间奖励为零，终局由完整运输 LP 计算降幅 `1 − 当前布局费用 / 原厂布局费用`。原厂费用约为 5,803,123,608.50。[建模与实验口径](docs/model.md)

## 安装

以下命令均在仓库的 `rl/` 目录执行。全厂实例、工艺路线数组和 LP 初值已随仓库提供。需要 uv 和支持 C++17 的 `c++` 编译器，路径定价内核在首次运行时编译。[数据来源与文件说明](data/README.md)

| 环境 | 安装命令 |
| --- | --- |
| Apple Silicon，CPU | `uv sync --locked --python 3.14` |
| Linux x86_64，CUDA 12.1，Python 3.10 或 3.11 | `uv sync --locked --python 3.10 --extra cuda` |

GPU 定价使用 CuPy 的 NVRTC 内核，运行环境需要可用的 NVIDIA 驱动和 CUDA 12.x 运行库。

## 在工作台上运行

在本机仓库的 `rl/console/` 目录运行以下命令，浏览器打开 <http://127.0.0.1:7860>。在“全厂基准”选择“强化学习（RL）”，点击“远程训练与推理”。控制台也可通过 <http://127.0.0.1:7860/rl/> 直接打开。

```bash
uv run --locked --python 3.11 python app.py
```

1. 填写服务器地址、SSH 端口和用户名。远端部署目录留空时，工作台自动打开该账号最近一次任务所在的目录，并选中该任务。首次使用或旧目录已删除时使用 `~/.local/share/a2-fab-rl`。也可手填目录，例如 `~/coworker/a2-fab-rl`。连接后页面显示实际绝对路径。可使用本机 SSH 密钥、agent 或密码。首次连接时将页面读取的主机指纹与服务器管理员提供的值核对。主机已记录在本机 `~/.ssh/known_hosts` 时可以留空指纹。
2. 点击“部署 / 校验环境”。远端需要 Linux x86_64、Python 3.8 或更新版本、Git、curl 和 C++17 编译器。使用 GPU 时还需要兼容 CUDA 12.1 的 NVIDIA 驱动。控制台优先从本机 Git 仓库经 SSH 传送已发布的 RL 提交 `b0d852a` 及全厂数据，缺少该提交时由服务器从 GitHub 下载。远端安装 uv 0.10.2、准备 Python 3.11，再按该提交的 `rl/uv.lock` 下载依赖并校验 CUDA。页面显示每个阶段的日志，再次部署时复用相同目录下的代码和依赖。
3. 选择一张 GPU 或 CPU。策略下拉框列出部署目录中 `models/*.pt` 的权重、本目录训练出的检查点，以及随仓库提供的演示策略。演示策略在全厂基准上采样两个布局、训练一轮。将需要复现的训练权重存入 `models/`，即可按文件名选择。点击“启动推理”，终局 LP 校验通过后可打开布局对照或下载 `fab-result-v1` JSON。在线推理使用固定的 10 产品全厂场景。
4. 在训练区选择轮数。“每轮完整布局数”“并行 LP 数”“显存预算”“内存预算”默认均为 `0`。界面显示预计配置，点击“启动训练”时按所选卡的空闲显存、主存预算、CPU 亲和性、cgroup v2 配额与当前 CPU 空闲情况重新计算。布局批量和并行 LP 数可以分别手填。全厂 PPO 会在启动时试跑反向传播，进一步确定梯度微批量。

在有 24 个空闲 CPU 核、24 GiB RTX 4090 和充足主存的实测机器上，自动配置为每轮 64 个完整布局、22 个并行 LP。每个布局包含一次全厂放置和一次终局 LP。运行面板显示实际采用的参数、任务显存和 RSS，以及上一完整轮的布局吞吐量。吞吐量按已通过 LP 校验的布局数除以采样、终局 LP 与 PPO 更新的合计时间计算。每轮保存检查点和采样平均成本降幅。显存和主存预算以 GiB 填写，任务进程组超出采样预算时由工作台回收，瞬时峰值可能高于预算。

同机完成 128 个布局的端到端用时，旧默认配置（每轮 8 个布局、2 个 LP、16 轮）为 184.0 秒，Gradio 自动配置（每轮 64 个布局、22 个 LP、2 轮）为 60.6 秒。两次都包含梯度微批量探测和检查点保存。每轮的布局数也决定策略多久更新一次，比较收敛表现时需要单独评估策略。

在选定的部署目录下，`releases/<提交号>/rl/` 保存代码，里面的 `.venv/` 是训练和推理共用的环境。工作台从该环境加载 CUDA 运行库和 NVRTC，部署时会在有可用 GPU 的服务器上实际执行一个 CuPy 内核。`jobs/<任务编号>/` 保存日志、任务状态和 `output/` 结果。uv、托管 Python 和下载缓存分别位于 `bin/`、`python/` 和 `cache/`。已连接其他目录时，点“查看最近任务”可在当前 SSH 连接中切换回来。训练曲线从首个完整轮次开始显示，每轮增加一个采样平均成本降幅的数据点。

同一远程账号同时运行一个工作台任务，各部署目录通过 `~/.local/state/a2-fab-rl/active.lock` 协调。“停止选中任务”会回收该任务的进程组。本机应用重启后，可重新连接并选择已完成的推理生成新的布局链接，或将之前下载的 JSON 导入工作台。工作台监听本机回环地址。

## 从空布局运行一轮

以下 CPU 命令在完整工厂上采样两个布局，分别求解终局 LP，再执行一轮 PPO 更新。

```bash
uv run --locked python -m smat_rl.train --run runs/cpu-check \
  --device cpu --lp-backend path-cpu --cycles 1 --envs-per-rank 2 \
  --lp-workers 2 --decision-samples 16 --gradient-batch 4 --eval-every 0 --completion-prior
```

## 四卡训练与评估

```bash
bash scripts/train_4gpu.sh runs/e3-routing
bash scripts/train_4gpu.sh runs/e3-local --actor-features local
```

默认种子为 42。每组运行 120 轮，每轮每卡采 64 个完整布局，全局 256 个 LP 标签、16 次 Adam 更新。对照组沿用相同 critic、组合数先验及训练预算。恢复训练时重复原训练命令并追加 `--resume`。每个 rank 从自己的 `last_rankN.pt` 续训，`final.pt` 用于独立评估。

以下命令使用单卡评估一个检查点，输出一个确定性布局，并以固定的独立采样种子评估另外 32 个布局。CPU 评估可使用 `--device cpu --lp-backend path-cpu`。

```bash
uv run --locked --extra cuda python -m smat_rl.train \
  --evaluate-checkpoint runs/e3-routing/final.pt --run runs/eval-routing \
  --device cuda --lp-backend path-cuda --diagnostic-samples 32
```

训练目录内的 `config.json` 记录实例指纹和超参数，`metrics.csv` 是训练采样统计，`update_cycle*.json` 记录 PPO 更新。评估目录内的 `evaluations.csv` 分别记录确定性结果、采样均值和采样最优费用，`evaluation_cycle*.json` 保留各布局的结果与 LP 证书。两个策略的收益以相同评估样本数和种子比较。

## 代码与测试

| 位置 | 职责 |
| --- | --- |
| `smat_rl/env.py`、`placement.py` | 物理环境、尺寸兼容、合法动作和自动填充 |
| `smat_rl/model.py`、`candidate_transport.py` | CNN、GCN、候选槽打分、状态价值和有向端口距离 |
| `smat_rl/rollout.py`、`ppo.py` | 完整布局采样、优势估计与 PPO 更新 |
| `smat_rl/train.py`、`cli.py`、`runtime.py` | 单机和多卡训练、评估、命令行与检查点 |
| `smat_rl/lp/` | 运输 LP、CPU/CUDA 路径定价与终局证书 |

```bash
uv run --locked python -m unittest discover -s tests -v
```
