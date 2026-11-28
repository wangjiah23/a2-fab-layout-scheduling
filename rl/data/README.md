# 全厂 E3 输入

槽位几何、设备组和运输轨道来自 SMAT2022 的 LVHM E 工作簿。工艺路线与生产需求来自 SMT2020 LVHM 的 `part.txt`、`order.txt` 与十个 `route_*.txt`。使用的上游版本为 [Course-Design-SMT b4f06b0](https://github.com/Stelle7th/Course-Design-SMT/tree/b4f06b0f73fe69f09bb0b2d264d645bb8f6c8769)。

| 文件 | 用途 |
| --- | --- |
| `original_slot_instance/instance.json`、`arrays.npz` | 1,068 台机器的原始槽位、端口有向距离、设备顺序、需求图和网络输入。训练入口从真实宽高推导 E3 尺寸类。 |
| `original_slot_instance/reference_flow.npz` | 原厂布局的参考运输流。按原始有向距离重计费用约为 5,803,123,608.50。 |
| `original_slot_instance/operations.npz` | 10 个产品、3,866 道工序的组号、单批加工时间、产品边界和需求，另保存 per_piece / per_batch 工时是否随每批片数缩放的标记。每道工序的候选设备由组号匹配得到。 |
| `full_slots/flow_matrix.npz`、`flow_vectors.npz` | 完整弧 LP 的约束矩阵、边界和机器弧索引。路径主问题两次求解仍未通过认证时，使用这组输入重算。 |
| `full_slots/seed_flow.npz` | 初始可行路径的分解使用已存 LP 原始解的非零列编号和流量。每个新布局都重新计算运输成本。 |

用上游 E1 解析器重新生成工序数组时，在 RL 项目目录运行以下命令，并将最后一个路径替换为上游仓库的本地检出位置。

```bash
uv run --extra data python scripts/prepare_operations.py --source-repo /path/to/Course-Design-SMT
```

发布实例按组号、加工时间、产品边界、需求顺序连接小端数组字节后的 SHA-256 为 `2cc04a8b3bcd8edefccadf217f76009c64747d7b7e43dd419716ec6bce66f401`。加载器校验该摘要，并保留历史 PPO 检查点使用的实例指纹。
