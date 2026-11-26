# -*- coding: utf-8 -*-
"""接口说明文本：用户接口 + 开发者接口。"""

USER_DOC = """\
═══════════════════════════════════════════════════
一、用户接口
═══════════════════════════════════════════════════

【1. 上传文档格式要求】

◆ 订单文件（推理阶段上传）
  - 文件类型：.txt（UTF-8 或 GBK 编码均可），制表符分隔（Tab）。
  - 首行为字段名，其后每行一条订单/批次记录。
  - 必需字段及其含义：

      LOT        订单/批次标识（如 Lot_1）
      PART       产品标识（如 part_1，须与设备工艺路线一致）
      PRIOR      优先级（整数）
      PIECES     每 lot 片数（如 25）
      START      订单开始时间（格式 MM/DD/YY HH:MM:SS）
      RDIST      到达间隔分布类型（如 constant）
      REPEAT     重复到达间隔数值（分钟）
      RUNITS     REPEAT 的时间单位（min）
      RPT#       重复次数
      LOTSPERRPT 每次重复产生的 lot 数量
      DUE        交付时间
      ORDER      订单标识（如 O_Lot_1）
      HOTLOT     是否为 hot lot（yes/no）

  - 注意：设备信息（厂房、设备几何、AMHS 轨道）固定采用本项目内置数据，
    用户无需上传设备信息，只需上传订单文件即可。

◆ 结果文件（训练完成后下载/加载）
  结果保存在 result/<运行编号>/ 目录，包含：

      layout.csv                设备布局：每台设备的新坐标与槽位
      transport_flow.csv        设备间物流：源设备→目标设备的搬运量
      device_utilization.csv    设备使用频率：每台设备的机时负荷与利用率
      from_to_bay.csv           Bay 间物流：Bay 对之间的搬运量
      rails.csv                 走廊通道：每条轨道的几何与物流强度
      summary.json              汇总：参考/优化成本、降幅、移动设备数等

【2. 各结果文件字段含义】

  layout.csv：
      equipment_name 设备实例名    tool_group 工艺组
      bay            所属 Bay       x / y      设备中心坐标
      width / height 设备尺寸       original_slot / new_slot 原始/新槽位编号

  transport_flow.csv：
      source_equipment 源设备     target_equipment 目标设备
      flow_lots        搬运量(lot) path_distance    轨道路径距离
      source_bay / target_bay 源/目标 Bay

  device_utilization.csv：
      equipment_name 设备名       available_time_min 可用机时(分钟)
      load_min       实际负荷(分钟) utilization 利用率(0~1)

  from_to_bay.csv：
      source_bay 源 Bay    target_bay 目标 Bay    flow_lots 搬运量(lot)

  rails.csv：
      rail_name 轨道边名    x1 / y1 起点坐标    x2 / y2 终点坐标
      flow      该轨道上的物流强度(lot)

  summary.json：
      reference_cost 参考成本   optimized_cost 优化成本
      cost_reduction_pct 降幅(%)  moved 移动设备数

【3. 界面操作流程】
  ① 「推理」页：点击"上传订单文件"选择订单 → 点击"开始训练"，
     实时查看进度与日志 → 完成后结果自动保存到 result/ 目录。
  ② 「加载」页：选择结果目录 → 点击"加载" → 勾选需要展示的图层
     （设备布局 / 设备间物流关系 / 通道物流频率热力图）。
     设备布局可切换着色依据（按 Bay / 按设备组 / 按设备使用频率），
     其中"按设备使用频率"以利用率为颜色在布局上直接呈现设备热力图；
     通道物流频率热力图将物流映射到轨道走廊上、采用规则网格呈现。
"""

DEVELOPER_DOC = """\
═══════════════════════════════════════════════════
二、开发者接口
═══════════════════════════════════════════════════

【1. 目录结构】

  app/
    main.py           主窗口与三个 Tab（推理 / 加载 / 接口说明），程序入口
    inference.py      推理（训练）模块：数据准备、E1+E3 调度、进度回调
    visualization.py  结果可视化模块：布局图 / 物流关系 / 两类网格热力图
    rails.py          走廊数据模块：从 SMAT2022 轨道与物流生成 rails.csv
    docs_text.py      本说明文本
  scripts/            现有算法脚本（E0~E3），推理阶段通过 subprocess 调用
  data/               内置数据（SMT2020 生产数据 + SMAT2022 设备信息，设备信息固定）
  result/             训练结果输出目录（每个运行一个子目录）

【2. 核心函数接口】

  rails.py
    generate_rails(smat_file, selected_ports_csv, flow_csv, out_csv) -> int
        读取 SMAT2022 轨道端点坐标与设备端口到轨道的映射，将设备间物流
        映射到轨道走廊，输出 rails.csv（rail_name, x1, y1, x2, y2, flow）。

  inference.py
    prepare_data_root(order_file) -> Path   复制内置数据并用上传订单覆盖 order.txt。
    run_inference(order_file, progress_cb, done_cb, **params)
        后台线程执行 E1→E3 训练；完成后生成 rails.csv 并回调。
    run_stage(argv, on_line) -> int        subprocess 运行脚本阶段，逐行回调输出。

  visualization.py
    load_result(result_dir) -> dict
        读取 result 目录各 CSV（含 rails.csv），返回布局/物流/利用率/轨道数据。
    ResultCanvas(master)                  matplotlib 嵌入 Tk 的画布，含图层勾选
                                          与"按 Bay / 按设备组"着色选项。
    draw_layout(ax, data, by)            设备布局散点（by 可取 "bay"/"group"/"utilization"，
                                          其中 "utilization" 以利用率为颜色呈现设备热力图）。
    draw_flows(ax, data)                 设备间物流连线（流量加权）。
    draw_channel_heatmap(ax, data)       通道物流频率热力图（轨道几何 + 规则网格）。

【3. 功能拓展指引】

  新增一种可视化：
    1) 在 visualization.py 中添加 draw_xxx(ax, data) 函数；
    2) 在 ResultCanvas 中新增勾选变量与对应的重绘分支；
    3) 在 docs_text.py 的用户说明中补充该图层说明。

  修改训练流程（例如接入新算法）：
    在 inference.py 的 run_inference 中，按 E1 → E3 的顺序追加或替换
    run_stage 调用；新增阶段需遵循"读 data-root、写 output-dir"约定。

  接入新的输入数据：
    在 prepare_data_root 中追加文件复制逻辑；若为全新的数据维度，
    需同步扩展 scripts/ 下对应解析函数（e0_data_audit.load_smt 等）。

  调整热力图网格分辨率：
    修改 visualization.py 中的 GRID_SIZE 常量。

【4. 依赖】
  Python ≥ 3.10；numpy、scipy、openpyxl、highspy（可选）、matplotlib、tkinter。
  推理阶段依赖 scripts/ 目录（E1/E3），可视化阶段依赖 matplotlib + tkinter。
"""
