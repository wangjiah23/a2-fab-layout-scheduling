# 半导体工厂布局优化 · 桌面应用

将 E1（参考流量 LP）+ E3（SA + LNS + warm-start LP 启发式）训练流程与结果可视化封装为一个
**桌面 GUI 应用**（tkinter + matplotlib，非 Web/HTML）。

## 运行方式

```powershell
# 项目根目录下执行
python app/main.py
```

依赖：`numpy scipy openpyxl matplotlib`（`highspy` 可选，未安装时 E3 自动回退冷求解）。
GUI 依赖 Python 标准库 `tkinter`（Windows 官方安装包自带）。

## 功能

| Tab | 功能 |
|---|---|
| ① 推理（训练） | 上传订单文件 → 运行 E1→E3 训练 → 实时显示进度与日志 → 结果自动保存到 `result/<运行编号>/` |
| ② 加载（展示） | 选择结果目录，勾选展示：设备布局（可按 Bay / 设备组 / 设备使用频率着色，其中"设备使用频率"以利用率为颜色在布局上呈现设备热力图）/ 设备间物流关系 / 通道物流频率热力图（映射到轨道走廊、采用规则网格） |
| ③ 接口说明 | 用户接口（上传/下载文档格式与字段含义）+ 开发者接口（目录结构、函数接口、拓展指引） |

## 目录

```
app/                桌面应用源码（main / inference / visualization / rails / docs_text）
scripts/            算法脚本（E0~E3），推理阶段通过 subprocess 调用
data/               内置数据（SMT2020 生产数据 + SMAT2022 设备信息，设备信息固定）
result/             训练结果输出目录（每个运行一个子目录，含 sample 示例）
```

## 结果目录结构

```
result/<运行编号>/
    layout.csv                设备布局（坐标、Bay、槽位）
    transport_flow.csv        设备间物流（源→目标搬运量）
    device_utilization.csv    设备使用频率（机时负荷与利用率）
    from_to_bay.csv           Bay 间物流
    rails.csv                 走廊通道（轨道几何与物流强度）
    summary.json              汇总（成本、降幅、移动设备数、求解统计）
```
