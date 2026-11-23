# A2 组｜数据驱动的半导体制造系统布局与物料搬运调度优化

本仓库用于 A2 组课程设计，汇总半导体工厂设备布局与物料搬运调度的模型、代码和实验。项目包括线性规划（LP）＋启发式与强化学习（RL）两条技术路线。

## 项目结构

| 目录 | 内容 |
| --- | --- |
| [`lp-heuristic/`](lp-heuristic/) | LP 模型、启发式算法、配置与运行入口 |
| [`rl/`](rl/) | 强化学习环境、策略、训练与评估代码 |
| [`data/`](data/) | 两个模块共用的数据说明与可公开的小型算例 |
| [`docs/`](docs/) | 问题定义、实验对比与项目文档 |

## 组员提交

获得仓库写权限后，从 `main` 创建分支。以 LP＋启发式模块为例，先克隆仓库，然后将文件放进 `lp-heuristic/`。

```bash
git clone https://github.com/wangjiah23/a2-fab-layout-scheduling.git
cd a2-fab-layout-scheduling
git switch -c lp-heuristic/implementation
# 将代码、依赖和运行说明放入 lp-heuristic/ 后提交。
git add lp-heuristic/
git commit -m "Add LP and heuristic implementation"
git push -u origin HEAD
```

RL 模块同样从 `main` 建立分支，例如 `rl/environment`，提交 `rl/` 目录下的文件。推送后向 `main` 发起 Pull Request，在描述中写明更新内容、运行命令和验证结果。
