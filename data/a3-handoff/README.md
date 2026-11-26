# A2 全厂布局与 A3 排产验证输入

本次交付使用 SMT2020 和 SMAT2022 的 LVHM_E 场景（dataset 4）。布局取自 A2 已保存的 E3（LP＋启发式）运行，覆盖 1,068 台生产机台，其中 1,049 台换到尺寸兼容的既有槽位。机台 ID 和设备组沿用 A3 基准，新机台接入目标槽位的四个装卸端口。48 个固定 AMHS 设施、轨道和车辆沿用基准工作簿。 [A2 E3 原始布局和汇总](https://github.com/wangjiah23/a2-fab-layout-scheduling/tree/a008b2b4ef7d37743f678d0afe6aa535e0e97dce/lp-heuristic/output/e3_global)

| 文件 | 用途 |
| --- | --- |
| [SMAT2022_LVHM_E_A2_E3.xlsx](SMAT2022_LVHM_E_A2_E3.xlsx) | A3 可直接读取的候选布局工作簿。`Equipment` 给出新 Bay 和坐标，`Port.EQP_NAME` 把每个槽位的端口关联到当前机台。端口 ID 标识物理槽位。 |
| [layout.csv](layout.csv) | 以 `equipment_name` 为主键的机台表。`original_slot` 和 `new_slot` 沿用 E3 的零基槽位索引，`slot_equipment_name` 为新槽位上的原机台 ID。坐标和尺寸单位为毫米，端口沿轨偏移单位为米。 |
| [machine_distance_m.csv.gz](machine_distance_m.csv.gz) | Gzip 压缩的 UTF-8 CSV。行是出发机台，列是到达机台，值为米。行列使用相同的 1,068 个机台 ID，按 ID 排序。 |

## 在 A3 仿真里使用

1. 把工作簿放到 A3 仓库的 `03-数据/A2-E3/`。
2. 复制本次实验使用的 `delay` 或 `exact` 配置，将副本的 `network.workbook` 改为 `03-数据/A2-E3/SMAT2022_LVHM_E_A2_E3.xlsx`，坐标比例使用 `0.001`。
3. 用相同排产规则、投片计划、随机种子和车辆设置，分别运行原配置和候选配置。两次运行使用各自的输出目录。

例如把候选配置保存为 `configs/a2_e3_comparison.json` 后，在 A3 仓库根目录运行以下命令。 [A3 仿真入口及配置说明](https://github.com/Jamlet-T/a3-fab-scheduling-simulation/blob/538d06a5146ab2583974f4dfaedf07a15d72e98c/README.md)

```powershell
.venv\Scripts\python.exe -m fabcosim full --config configs/a2_e3_comparison.json --output runs/a2-e3-comparison
```

## 距离口径

矩阵使用 A3 的有向轨道和米制长度，曲轨以两端点之间的弦长近似。跨机台距离从出发机台的出口端口算到到达机台的入口端口，计入端口在轨道内的偏移。A3 按端口 ID 排序取首个为入口、末个为出口。同一机台的内部交接由仿真处理，矩阵对角线为 `0`。两个方向的路径长度可能不同。 [A3 建网](https://github.com/Jamlet-T/a3-fab-scheduling-simulation/blob/538d06a5146ab2583974f4dfaedf07a15d72e98c/src/fabcosim/data/smat_network.py) · [A3 机台和端口映射](https://github.com/Jamlet-T/a3-fab-scheduling-simulation/blob/538d06a5146ab2583974f4dfaedf07a15d72e98c/src/fabcosim/data/full_route.py)

A3 的 `full` 命令从工作簿建网并自行寻路。若在排产算法里直接查矩阵，可用 `pandas.read_csv("machine_distance_m.csv.gz", index_col="equipment_name")`。例如 `DIE0101` 到 `DIE0102` 为 `80.922930 m`，反向为 `71.756263 m`。

## 核对结果

用 A3 原版代码加载了工作簿，解析到 1,068 台生产机台和 48 个固定设施。核对了全部生产机台的入口和出口，并用 A3 的寻路函数复算了 222 对距离，覆盖同轨顺行和绕行，最大差异为 CSV 六位小数舍入所致的 `0.0000005 m`。A2 和 A3 的原始 SMAT2022 工作簿 SHA-256 均为 `8e74e6db2272ad7dbeeb26de2adb1cd69e27a7848fe7baa0524803fd317156c0`。 [A3 基准来源](https://github.com/Jamlet-T/a3-fab-scheduling-simulation/blob/538d06a5146ab2583974f4dfaedf07a15d72e98c/03-%E6%95%B0%E6%8D%AE/SMAT2022-%E6%9D%A5%E6%BA%90%E8%AE%B0%E5%BD%95.md)

用 A3 的 `performance_delay_a1_1d.json` 配置替换布局工作簿后，模拟跑完了 24 小时窗口，完成 67 趟运输和 54 道加工工序。该运行用于确认数据能接进仿真。 [A3 短跑配置](https://github.com/Jamlet-T/a3-fab-scheduling-simulation/blob/538d06a5146ab2583974f4dfaedf07a15d72e98c/configs/performance_delay_a1_1d.json)
