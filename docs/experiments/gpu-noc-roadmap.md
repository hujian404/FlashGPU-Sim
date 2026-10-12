# GPU NoC 瓶颈分析与优化 Roadmap

## 当前进展

- Phase 0：已修复 driver API 同步语义，避免 kernel 尚未完成时进程退出；已完成
  VectorAdd、GEMM 基线复验，FlashAttention 正在按修复后的同步路径重跑。
- Phase 0：已增加 `scripts/experiments/collect_baseline_metrics.py`，统一保存 commit、
  config/trace hash、返回码、资源开销、功能成功标志和最终模拟统计。
- Phase 1：采集器已接入现有 local/global NoC 的注入率、冲突、buffer utilization、
  service、width-limited、downstream-full 及热点端口统计。下一步补充 packet/queueing
  latency 分位数和按请求类型拆分的流量矩阵。

基线数据只在返回码为 0、功能成功标志成立且包含 `gpu_tot_sim_cycle` 等最终统计时
标记为完整。周期采样点不能替代最终统计。

## 1. 目标与范围

目标是在 RTX 5090 配置上建立可重复、可归因的 GPU NoC 分析流程，分别回答：

1. GPC/cluster 内部的请求聚合、仲裁和响应分发何时成为瓶颈；
2. GPC 与 LLC/L2 slice 之间的全局互连何时成为瓶颈；
3. 拥塞究竟来自网络本身，还是 L2/DRAM 背压、地址分布或 kernel 并行度；
4. 哪类设计改动能在面积、能耗和公平性约束下提供稳定收益。

代码映射以当前模型为准：

- GPC/cluster 内部互连：`src/gpgpu-sim/local_interconnect.{h,cc}`；
- 全局 GPC↔LLC 互连入口：`src/gpgpu-sim/icnt_wrapper.{h,cc}`；
- 全局网络模型：`src/intersim2/`；
- L2、内存和背压：`src/gpgpu-sim/l2cache.*`、`gpu-cache.*`、`dram*`；
- 配置旋钮：`configs/SM120_RTX5090/gpgpusim.config`。

## 2. 分析原则

- **先可观测，再优化**：缺少逐端口吞吐、等待时间和阻塞原因时不做结构结论。
- **端到端分解**：记录 source queue、local NoC、global NoC、L2 queue、DRAM queue
  各段等待时间，避免把下游背压误判为网络带宽不足。
- **反事实确认**：只有当理想网络或带宽/缓冲扩容显著降低周期数，才判定 NoC 为
  性能瓶颈。
- **固定 trace 单因素扫描**：每次只改变一组参数，至少重复验证 deterministic 输出。
- **性能之外有约束**：同时报告最大/平均队列、P95/P99 延迟、公平性和流量放大。

## 3. Phase 0：基线冻结与自动化（第 1 周）

交付物：

- 固定代码提交、配置 checksum、工具链版本、trace hash 和运行命令；
- `run_manifest.json`：workload、shape、seed、commit、config、返回码、墙钟时间；
- 日志解析器输出统一 CSV/JSON；
- 最新基线重跑 VectorAdd、GEMM、FlashAttention。

验收标准：相同 trace/config 重跑的模拟周期和关键计数完全一致；失败任务不会进入
汇总表；每条结果能反向定位到原始日志。

## 4. Phase 1：NoC 可观测性补齐（第 1–2 周）

### 4.1 每个网络层级必须采集

- injected/ejected packets、bytes/sectors、每周期吞吐；
- input/output queue 平均值、峰值、P50/P95/P99；
- packet 端到端 latency 与各阶段 queueing latency；
- arbitration conflict、width-limited、downstream-full、credit-stall 周期；
- source/destination 热点矩阵，按 GPC、L2 slice、请求/响应、读/写/TMA 分类；
- active cycles、空闲周期和 Jain fairness index。

`local_interconnect.cc` 已有 packets、conflicts、buffer utilization 和 service stats，
应先复用并补齐分位数及逐端口输出。全局网络侧统一字段名和时间单位，确保两个层级
可以在同一时间线上比较。

### 4.2 瓶颈判据

满足以下证据链才标记为 NoC-bound：

1. 网络队列或延迟持续上升，且存在明显 width/arbitration stall；
2. 上游计算单元因注入/响应阻塞而停顿；
3. 下游 L2/DRAM 没有先达到服务上限，或已通过理想下游实验排除；
4. 将目标网络带宽设为 2×/4× 或切换 ideal network 后，总周期显著下降。

建议以 `speedup >= 5%` 作为“值得继续分析”的初筛阈值，同时保留连续响应曲线，
不把阈值本身当作架构结论。

## 5. Phase 2：工作负载矩阵（第 2–4 周）

### 5.1 定向微基准

| 类别 | 访问模式 | 目的 |
| --- | --- | --- |
| streaming copy/read/write | 连续、可合并 | 测峰值注入和双向带宽 |
| random gather/scatter | 随机 L2 slice | 测全局网络与 MSHR 压力 |
| partition camping | 集中到少数 slice | 构造热点与公平性问题 |
| stride/transpose | 改变地址映射 | 分离 coalescing、hash 和网络影响 |
| TMA copy/multicast | 单播、多播、不同 tile | 测请求聚合、响应和流量放大 |
| many-to-one / one-to-many | 人工 GPC↔slice 映射 | 测仲裁与 ejection 瓶颈 |

每类扫描 working set、CTA 数、并发 SM/GPC 数、读写比、packet size 和目标 slice
数量，形成从欠载到饱和的 offered-load 曲线。

### 5.2 代表性模型负载

优先选择能够覆盖不同通信形态且 trace 规模可控的 kernel：

1. **GEMM**：方阵与 tall-skinny/wide-short，扫描 M/N/K、tile 和 split-K；计算密集
   作为对照，低复用变体用于提高 LLC 流量。
2. **FlashAttention**：prefill 与 decode，扫描 batch、head、sequence length、head
   dimension、causal；覆盖 TMA、共享内存复用和响应流量。
3. **MoE dispatch/combine**：不同 expert 数和 token skew；构造稀疏、突发、热点式
   gather/scatter，是全局互连公平性和热点路由的重点负载。
4. **Embedding/SpMV**：低算术强度、随机访问，作为 LLC/DRAM/NoC 解耦用压力负载。
5. **卷积或 stencil**：规则邻域访问，验证优化是否只对随机流量有效。

首轮最小集合建议为：VectorAdd、random gather、partition camping、TMA multicast、
三种 GEMM shape、FlashAttention prefill/decode、两种 MoE skew。先用缩小 shape 建立
趋势，再对关键点运行完整模型规模。

## 6. Phase 3：敏感性与归因实验（第 4–6 周）

对 local 和 global 两层分别执行：

- 链路/端口 service width：0.5×、1×、2×、4×、ideal；
- input/output/ejection buffer：0.5×、1×、2×、4×；
- arbitration：现有策略、round-robin、age/QoS-aware；
- VOQ on/off、multi-grant on/off、subnet 数与 flit size；
- GPC response ingress/dispatch width；
- 地址映射/hash 与 L2 slice 数；
- L2 service width、MSHR 和 DRAM bandwidth 的交叉控制组。

使用二维实验区分耦合：`NoC bandwidth × L2 bandwidth`、`NoC buffer × MSHR`、
`routing/arbitration × traffic skew`。若只扩 NoC 无收益，而同时扩 L2 才有收益，应
归类为下游主导或耦合瓶颈，而不是单纯 NoC-bound。

每个点报告：总周期、speedup、IPC、NoC throughput、平均/P99 latency、队列峰值、
stall breakdown、L2 miss/pending hit、DRAM utilization、仿真墙钟时间。

## 7. Phase 4：设计优化候选（第 6–9 周）

### 7.1 GPC 内部互连

- per-destination VOQ 与 age-aware arbitration，缓解 HOL blocking；
- request/response 独立配额或动态 service budget；
- 小包优先与 starvation guard；
- TMA multicast 在 GPC 内复制/合并，减少重复 response；
- 自适应 ingress/dispatch 配额，根据 L2 返回压力动态调节。

### 7.2 GPC↔LLC 全局互连

- hotspot-aware routing 或 slice-aware traffic steering；
- request/reply 独立虚拟网络和动态带宽分配；
- source throttling，避免少数 GPC 抢占并放大下游拥塞；
- 读、写回、TMA 的 QoS/age-aware arbitration；
- packet coalescing/compression 或 multicast replication placement；
- 分层/分区网络参数调整，减少长距离与集中式仲裁压力。

优化顺序采用“低风险策略 → 缓冲/端口 → 拓扑”三级。每个候选先用微基准证明机制，
再在模型负载验证泛化；避免只针对单一 kernel 调参。

## 8. Phase 5：评估与收敛（第 9–10 周）

### 8.1 主要评价指标

- 几何平均 speedup 及每 workload 最差退化；
- P95/P99 packet latency、队列峰值和饱和点；
- Jain fairness、热点 destination 的 tail latency；
- bytes/flits per useful instruction、额外 buffer/link 成本；
- 若功耗模型可用：NoC energy、energy-delay product。

### 8.2 推荐验收门槛

- 目标 NoC-bound 集合几何平均 speedup ≥ 10%；
- 非 NoC-bound 对照集合退化 ≤ 2%；
- P99 latency 或最大队列至少一项改善 ≥ 20%；
- 不引入 starvation/deadlock，所有功能验证通过；
- 收益在至少三类通信模式和两类模型 workload 上成立。

## 9. 最终交付结构

```text
docs/experiments/
  flash-RTX5090.md
  gpu-noc-roadmap.md
experiments/noc/
  manifests/
  configs/
  scripts/
  raw/          # 建议外部归档，不提交超大日志
  processed/
  plots/
```

最终报告应包含基线、瓶颈证据链、敏感性曲线、设计实现、跨 workload 结果、负面结果
和模型限制。任何性能表都必须标注 commit、config hash、trace hash 与是否为完整运行。
