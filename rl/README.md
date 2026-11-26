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

1. 填写服务器地址、SSH 端口和用户名。可使用本机 SSH 密钥、agent 或密码。首次连接先读取主机指纹并与服务器管理员核对。密码仅用于本次连接。
2. 点击“部署 / 校验环境”。远端需要 Linux x86_64、Python 3.8 或更新版本、Git、curl 和 C++17 编译器，可访问 GitHub、Python 包源及 uv 下载服务。使用 GPU 时还需要兼容 CUDA 12.1 的 NVIDIA 驱动。页面按阶段显示上传、下载、依赖安装和 CUDA 校验日志。工作台在账号目录安装 uv 0.10.2，固定拉取已发布的 RL 提交 `b0d852a`，按该提交的 `rl/uv.lock` 安装。再次部署复用已下载的代码和依赖。
3. 选择一张 GPU 或 CPU。推理可直接选随仓库提供的演示策略，也可选本账号训练出的检查点。该演示策略仅在全厂基准采样两个布局、训练一轮，用于演示接口。点击“启动推理”，终局 LP 校验通过后可打开布局对照或下载 `fab-result-v1` JSON。在线推理使用固定的 10 产品全厂场景。
4. 在训练区设置轮数、每轮完整布局数和并行 LP 数。页面显示当前轮次、采样与终局 LP 阶段、策略更新阶段。每轮结束保存检查点并显示采样平均成本降幅。显存和主存都可指定 GiB 预算，填 0 按启动时的可用量计算。控制台显示整卡利用率、任务进程组的显存和 RSS。显存受框架分配器与监测共同控制，RSS 和总进程显存按间隔采样，瞬时峰值可能超过设定值。

同一远程账号的工作台每次运行一个任务。关闭标签页、退出本机应用或 SSH 断线后可重新连接查看任务。“停止选中任务”回收该任务的进程组。任务日志、检查点和推理原始结果保存在服务器的 `~/.local/share/a2-fab-rl/jobs/<任务编号>/`。本机关闭应用后，旧的结果链接过期，重新连接并选择对应推理即可再生成。工作台监听本机回环地址。

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
