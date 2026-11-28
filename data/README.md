# 工作台数据接口

学校服务器预装 SMAT2022 LVHM_E 的 1,068 台生产设备、48 个固定 AMHS、物理槽位、设备端口和有向轨道。用户打开工作台，上传生产输入，按需导入起始布局，点击“开始求解”。[示例文件](../sample_app/周报演示应用/导入示例/) · [工厂数据来源与版本](../rl/data/README.md)

| 用户输入 | 字段和单位 | 如何参与计算 |
| --- | --- | --- |
| 生产场景 XLSX / CSV | Excel 的“产品需求”表，或 CSV 的 `part,route,demand_lots`。需求单位为 lot。Excel 也支持“新增路线”表。 | 选择内置路线或新路线，重算工艺流量、设备负荷、策略输入和 LP。 |
| SMT2020 `order.txt` | UTF-8 / GBK、制表符分隔，使用组员定义的 13 列表头。 | 沿用 E1 聚合规则，各行计划需求为 0.25 × 10,080 × LOTSPERRPT / REPEAT，同一 PART 累加。RUNITS 填 min。PIECES 改变按片和按批计时的工序用时。已知产品由内置 part 表关联路线。同一产品的片数须一致。优先级、交期与具体到达时刻留给生产仿真使用。 |
| 新工艺路线 | `route,step,step_type,stnfam,time_min_per_lot`，step 为路线内唯一正整数。process 对应本厂设备组，delay 记录等待时间。 | 按序号排序，运输 LP 连接相邻加工工序，delay 保留在场景中。新路线的时间直接按该订单 lot 给出。 |
| 起始布局 `layout.csv` 或 JSON | 每台设备一行，`equipment_name,x,y` 必填。可附 `tool_group,bay,width,height,original_slot,new_slot`；new_slot 为从 0 开始的目标机位。x、y 是 SMAT 原始坐标系下的设备中心。 | 使用全厂已有机位，按坐标、唯一占位和 E3 宽高兼容性校验；与 PPO 候选在相同订单下分别计算成本。孤立坐标或不完整设备表也能作为“外部结果”单独绘图。 |
| 场景 JSON 或 `fab-job-v1` | `scenario` 包含 `factory_id,scope,horizon_min,effective_time_fraction,demands,new_routes` 及可选的 `initial_layout`。工作台导出或服务的 `GET /api/input-schema` 提供准确格式。 | 可调整计划期、有效产能系数、每批片数和起始机位。超产能、未知设备组、重复设备或缺失物理机位会返回对应输入错误。 |

订单文件表头如下，列之间为制表符。原始示例来自组员使用的 SMT2020 LVHM 数据。[下载示例](../sample_app/周报演示应用/导入示例/05_订单.txt) · [启发式桌面应用的输入与输出定义](https://github.com/wangjiah23/a2-fab-layout-scheduling/blob/03458d947e71ce017aa4fc548573b7cbe2a6aad8/lp-heuristic/app/docs_text.py)

```text
LOT	PART	PRIOR	PIECES	START	RDIST	REPEAT	RUNITS	RPT#	LOTSPERRPT	DUE	ORDER	HOTLOT
```

完整请求发送到 `POST /api/jobs`，返回任务编号。通过 `GET /api/jobs/{id}` 查询进度，完成后下载同一路径下的 `result.json` 或 `simulation-input.zip`。下面的请求沿用工艺路线 `r_1`，将计划需求设为 6 lot。

```json
{"schema":"fab-job-v1","method":"rl","scenario":{"factory_id":"SMAT2022-LVHM_E","scope":"full","horizon_min":10080,"effective_time_fraction":0.85,"demands":[{"part":"part_1","route":"r_1","demand_lots":6}],"new_routes":[]}}
```

## 结果和仿真边界

`fab-result-v1` 的 `scenario` 保存提交的需求、路线和起始布局。 `layout` 是推荐方案，`metrics` 给出同场景输入成本、PPO 候选成本和推荐成本。 `replay.slots` 是策略实际执行的物理放置顺序，`automatic_steps` 是从 0 开始的规则放置步号。输入方案更优或成本差在求解精度以内时保留输入，候选及其回放另由 `policy_layout` 对应。模型给出一个可行候选，LP 证书保证给定摆放下的运输分配精度。

仿真输入 ZIP 按组员的命名输出 `layout.csv`、`transport_flow.csv`、`device_utilization.csv`、`from_to_bay.csv` 和 `summary.json`，另含完整结果 JSON。逐弧流量乘以相应有向端口距离的总和会与推荐布局成本复核。物流和利用率来自本次静态规划；逐车行驶、排队、拥堵、交期及周期时间需要离散事件仿真器、车辆参数和调度规则。

若更改机位坐标、设备清单、端口连接或轨道拓扑，需要先重建机位、端口可达距离、兼容类和网络输入，并验证或重新训练策略。单独导入一份新坐标表可显示该表的几何布局，当前在线求解支持既有工厂上的重新摆放。

工作台由 HTML / CSS 和原生 JavaScript 实现，通过 HTML5 Canvas 2D 按真实设备中心、宽高和轨道坐标绘图。新 PPO 回放逐步绘制实际动作；历史第 95 轮展示使用归档抽帧。组员的 LP 桌面应用使用 Tkinter 和 Matplotlib。[工作台绘图实现](../sample_app/周报演示应用/源码/map.js) · [组员桌面绘图实现](https://github.com/wangjiah23/a2-fab-layout-scheduling/blob/03458d947e71ce017aa4fc548573b7cbe2a6aad8/lp-heuristic/app/visualization.py)
