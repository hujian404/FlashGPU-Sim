# FlashGPU-Sim RTX 5090 实验复现报告

## 摘要

本实验在 RTX 5090 上复现 FlashGPU-Sim 的构建、CUDA VectorAdd、Triton GEMM 与
Triton FlashAttention 工作流。三项 workload 使用同一代码基线和
`SM120_RTX5090` 配置串行运行，并记录功能结果、模拟周期、指令数、IPC、occupancy、
L2 统计、TMA 事务数和仿真墙钟时间。

本文既记录本次结果，也给出从环境准备到结果核验的最小完整步骤，供后续实验直接
复用。GPU NoC 后续研究计划见 [GPU NoC 瓶颈分析与优化 Roadmap](gpu-noc-roadmap.md)。

## 1. 实验目标

1. 验证最新合并基线能够构建并运行 RTX 5090 配置；
2. 验证 CUDA 与 Triton trace 两条工作流的功能正确性；
3. 建立后续 GPU NoC 分析可复用的基线结果与操作流程；
4. 记录版本、配置和运行产物，避免不同基线的数据混用。

## 2. 实验对象与控制变量

### 2.1 代码与配置

| 项目 | 值 |
| --- | --- |
| 实验分支 | `experiment/flash-RTX5090` |
| 报告整理时提交 | `26bf87477d4b6600aa074f231a4568ca793adbf4` |
| 模拟器代码提交 | `d5ee614da5fa1a094d27af6602636af4f3a6e318` |
| 同步的上游提交 | `flash@291fbe172332eb481556632a79eb9cfc93cf46bf` |
| 配置 | `configs/SM120_RTX5090/gpgpusim.config` |
| 配置 SHA-256 | `4131156c15217c9d3e6bf796462ddab93a27ed34cd5ecca85570b50e051cb2f3` |
| 并行设置 | `OMP_NUM_THREADS=1` |

报告提交只修改文档，因此实际构建的模拟器 banner 显示其父提交 `d5ee614d`。三个
workload 使用同一份该模拟器动态库和同一配置文件。

### 2.2 主机与工具链

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090，32607 MiB |
| NVIDIA driver | 580.82.07 |
| OS kernel | Linux 6.8.0-138-generic |
| CUDA Toolkit | 12.8.2，`nvcc` 12.8.93 |
| CUDA host compiler | Ubuntu GCC/G++ 12.3.0 |
| 模拟器构建目录 | `build/gcc-13.4.0/cuda-12080/release` |

路径约定：

```bash
WORK=.../work
REPO="$WORK/FlashGPU-Sim"
MAMBA_ROOT="$WORK/mamba-root"
CUDA_ROOT="$MAMBA_ROOT/envs/flashgpu-env"
SIM_LIB="$REPO/lib/gcc-13.4.0/cuda-12080/release"
RUNTIME12="$WORK/flashgpu-runtime12-current"
```

### 2.3 Workload

| Workload | 输入 |
| --- | --- |
| VectorAdd | 2,000,000 FP32 elements |
| Triton GEMM | M=2560, N=64, K=2560 |
| Triton FlashAttention | B=32, H=32, S=512, D=64，causal |

GEMM 使用仓库附带的离线 trace；FlashAttention trace 由 RTX 5090 上的 PyTorch
2.9.0+cu128、Triton 3.5.0 和项目内 `TritonTrace` 捕获。

| Artifact | SHA-256 |
| --- | --- |
| GEMM PTX | `eef1d121b4051bcf3230d1335fa488fa012139b97fd17c84213489d905a9a440` |
| GEMM CUBIN | `ac7f13a78275b255122d1abba2926b3d65bb31b7f9b8a44e3ddf616270a43be9` |
| FlashAttention PTX | `02d797b0155195d488091e0704078d89dd371d101ca3da4c81d486b8a538d1c0` |
| FlashAttention CUBIN | `01028e8eca7fe217773b6148795dcef38f5cabe3c61efc6394267d7475cfd37f` |

## 3. 实验方法

### 3.1 环境准备

```bash
"$MAMBA_ROOT/bin/micromamba" create -y -r "$MAMBA_ROOT" -n flashgpu-env \
  -c conda-forge -c nvidia \
  cuda-toolkit=12.8 flex bison zlib libglu libgl-devel \
  xorg-makedepend patchelf
```

若 `$CUDA_ROOT/include` 下缺少架构相关 CUDA 头文件入口，可补充链接：

```bash
for h in "$CUDA_ROOT"/targets/x86_64-linux/include/*.h; do
  ln -sf "../targets/x86_64-linux/include/$(basename "$h")" \
    "$CUDA_ROOT/include/$(basename "$h")"
done
ln -sfn ../targets/x86_64-linux/include/crt "$CUDA_ROOT/include/crt"
ln -sfn ../targets/x86_64-linux/include/cuda "$CUDA_ROOT/include/cuda"
```

### 3.2 构建模拟器

```bash
cd "$REPO"
"$MAMBA_ROOT/bin/micromamba" run -r "$MAMBA_ROOT" -n flashgpu-env \
  env CUDA_INSTALL_PATH="$CUDA_ROOT" \
  bash -c 'source setup_environment && make -j16'
```

构建完成后确认目标库存在：

```bash
test -f "$SIM_LIB/libcudart.so"
```

### 3.3 准备 CUDA 12 runtime 名称

Triton harness 使用 `-cudart none` 后显式链接模拟器库。FlashGPU-Sim 会根据应用的
`DT_NEEDED` 项识别 CUDA major version，因此仅有无版本 SONAME 的 `libcudart.so`
不足以完成检测。应为当前构建创建带 `libcudart.so.12` SONAME 的副本：

```bash
mkdir -p "$RUNTIME12"
cp "$SIM_LIB/libcudart.so" "$RUNTIME12/libcudart.so.12"
"$CUDA_ROOT/bin/patchelf" \
  --set-soname libcudart.so.12 "$RUNTIME12/libcudart.so.12"
ln -sfn libcudart.so.12 "$RUNTIME12/libcudart.so"
```

### 3.4 VectorAdd

编译时指定系统 G++，避免 Conda sysroot 与 `nvcc` 组合产生 `_Float32` 类型错误：

```bash
cd "$REPO/tutorials/vectorAdd"
make clean
make CUDA_INSTALL_PATH="$CUDA_ROOT" \
  NVCC_FLAGS='-O2 -std=c++17 -ccbin /usr/bin/g++'

mkdir -p run
cp vectorAdd run/vectorAdd-latest
cp -a "$REPO/configs/SM120_RTX5090/." run/
```

运行时从仓库根目录加载环境，使 SASS-guided PTX reorder 能找到
`GPGPUSIM_ROOT` 和 CUDA 工具：

```bash
cd "$REPO"
CUDA_INSTALL_PATH="$CUDA_ROOT" bash -c '
  source setup_environment
  cd tutorials/vectorAdd/run
  OMP_NUM_THREADS=1 ./vectorAdd-latest 2>&1 | tee vectorAdd-latest.log
'
```

### 3.5 Triton GEMM

准备 trace 和配置：

```bash
cd "$REPO/tutorials/triton-gemm"
mkdir -p run/tracking/data run/tracking/launchers
cp -a trace/data/. run/tracking/data/
cp -a trace/launchers/. run/tracking/launchers/
cp -a "$REPO/configs/SM120_RTX5090/." run/tracking/launchers/

cd run/tracking/launchers
make -f kernel_tma_gemm_launch1_Makefile \
  kernel_tma_gemm_launch1_kernel.fatbin
```

必须在模拟器更新后重新链接 harness，不能复用指向旧模拟器 runtime 的二进制：

```bash
"$CUDA_ROOT/bin/nvcc" -ccbin /usr/bin/g++ \
  -arch=sm_120a -cudart none \
  kernel_tma_gemm_launch1_harness.cu \
  -L"$RUNTIME12" -L"$CUDA_ROOT/lib" \
  -lcudart -lcuda -lz -lgomp \
  -Xlinker=-rpath -Xlinker="$RUNTIME12" \
  -Xlinker=-rpath -Xlinker="$CUDA_ROOT/lib" \
  -o kernel_tma_gemm_launch1-latest

cd "$REPO"
CUDA_INSTALL_PATH="$CUDA_ROOT" bash -c '
  source setup_environment
  cd tutorials/triton-gemm/run/tracking/launchers
  OMP_NUM_THREADS=1 ./kernel_tma_gemm_launch1-latest \
    2>&1 | tee gemm-latest.log
'
```

### 3.6 Triton FlashAttention

首次实验需要先捕获 trace：

```bash
"$MAMBA_ROOT/bin/micromamba" run -r "$MAMBA_ROOT" -n flashgpu-env \
  python -m pip install torch==2.9.0 triton==3.5.0
"$MAMBA_ROOT/bin/micromamba" run -r "$MAMBA_ROOT" -n flashgpu-env \
  python -m pip install -e "$REPO/tools"

VIRTUAL_ENV="$CUDA_ROOT" CUDA_INSTALL_PATH="$CUDA_ROOT" \
  PATH="$CUDA_ROOT/bin:$PATH" \
  bash "$REPO/tutorials/triton-flash-attention/capture.sh"
```

将最新配置复制到 launcher 目录，生成 fatbin 并重新链接：

```bash
cd "$REPO/tutorials/triton-flash-attention/run/tracking/launchers"
cp -a "$REPO/configs/SM120_RTX5090/." .
make -f flash_attention_kernel_launch1_Makefile \
  flash_attention_kernel_launch1_kernel.fatbin

"$CUDA_ROOT/bin/nvcc" -ccbin /usr/bin/g++ \
  -arch=sm_120a -cudart none \
  flash_attention_kernel_launch1_harness.cu \
  -L"$RUNTIME12" -L"$CUDA_ROOT/lib" \
  -lcudart -lcuda -lz -lgomp \
  -Xlinker=-rpath -Xlinker="$RUNTIME12" \
  -Xlinker=-rpath -Xlinker="$CUDA_ROOT/lib" \
  -o flash_attention_kernel_launch1-latest

cd "$REPO"
CUDA_INSTALL_PATH="$CUDA_ROOT" bash -c '
  source setup_environment
  cd tutorials/triton-flash-attention/run/tracking/launchers
  OMP_NUM_THREADS=1 ./flash_attention_kernel_launch1-latest \
    2>&1 | tee flashattention-latest.log
'
```

## 4. 新基线实验结果

### 4.1 汇总

| Workload | 功能结果 | 模拟周期 | 指令数 | IPC | Occupancy | L2 miss rate | 墙钟时间 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VectorAdd | `Test PASSED` | 31,874 | 44,001,408 | 1,380.4796 | 92.4220% | 99.91% | 257.33 s |
| Triton GEMM | 163,840 elements 全匹配 | 73,038 | 21,826,800 | 298.8417 | 8.3333% | 51.84% | 161.36 s |
| Triton FlashAttention | kernel 完整执行、正常退出 | 733,712 | 3,154,432,640 | 4,299.2788 | 16.3385% | 36.36% | 9,718.78 s |

### 4.2 VectorAdd

```text
gpu_tot_sim_cycle = 31874
gpu_tot_sim_insn = 44001408
gpu_tot_ipc = 1380.4796
gpu_tot_issued_cta = 7813
gpu_tot_occupancy = 92.4220%
L2_total_cache_accesses = 750680
L2_total_cache_misses = 750004
L2_total_cache_miss_rate = 0.9991
Test PASSED
```

最大常驻内存约 5,233,684 KiB。结果与同一新基线的前一次复验完全一致，说明固定
trace/config 下模拟周期和主要计数可重复。

### 4.3 Triton GEMM

```text
gpu_tot_sim_cycle = 73038
gpu_tot_sim_insn = 21826800
gpu_tot_ipc = 298.8417
gpu_tot_issued_cta = 40
gpu_tot_occupancy = 8.3333%
gpgpu_n_tma_tx_completed = 3240
gpgpu_n_tma_read_tx_completed = 3200
gpgpu_n_tma_write_tx_completed = 40
L2_total_cache_accesses = 829600
L2_total_cache_misses = 430084
L2_total_cache_miss_rate = 0.5184
Validation PASSED for arg[2]: all 163840 elements match
```

墙钟时间 161.36 s，最大常驻内存约 4,299,724 KiB。相比早期基线记录的 78,989
cycles，新基线减少 5,951 cycles（约 7.53%）；该变化来自合并后模型的综合更新，
不能在没有单因素实验的情况下归因于某个单独组件。

### 4.4 Triton FlashAttention

修复 `cuCtxSynchronize()` 后重新构建 runtime 和 harness，并从头运行完整 workload。
本次返回码为 0，8192 个 CTA 全部完成，日志包含最终统计、
`Kernel execution completed successfully` 和 `GPGPU-Sim: *** exit detected ***`：

```text
gpu_tot_sim_cycle = 733712
gpu_tot_sim_insn = 3154432640
gpu_tot_ipc = 4299.2788
gpu_tot_issued_cta = 8192
gpu_tot_occupancy = 16.3385%
gpgpu_n_tma_tx_completed = 90112
gpgpu_n_tma_read_tx_completed = 81920
gpgpu_n_tma_write_tx_completed = 8192
L2_total_cache_accesses = 23070032
L2_total_cache_misses = 8388616
L2_total_cache_miss_rate = 0.3636
Kernel execution completed successfully
GPGPU-Sim: *** exit detected ***
```

墙钟时间 9,718.78 s（约 2 小时 41 分 59 秒），最大常驻内存约 6,197,904 KiB。
相较早期记录的 794,076 cycles，新基线减少 60,364 cycles（约 7.60%）。采集器确认
配置和 PTX/CUBIN hash 与本报告记录一致，且日志中不再出现未实现同步警告。

该问题的根因不是统计打印函数本身，而是 driver API 的 `cuCtxSynchronize()` 原先为
直接返回成功的空实现。harness 在模拟线程尚未完成时退出，旧日志最后一次采样仅完成
8187/8192 个 CTA，因此无法产生最终汇总。现在该 API 调用模拟器上下文的
`synchronize()`，只有所有 CTA 和事务完成后才返回。

## 5. 结果讨论

### 5.1 功能正确性

VectorAdd 以 `Test PASSED` 为成功条件；GEMM 必须报告所有输出元素匹配；
FlashAttention 当前 harness 的成功条件是 kernel 完整执行且模拟器正常退出。后者
没有像 GEMM 一样逐元素比较输出，因此“运行成功”和“数值验证完成”应明确区分。

### 5.2 性能指标解释

- `gpu_tot_ipc` 是全 GPU 汇总值，不能直接与单 SM IPC 比较；
- 高 L2 miss rate 不能单独证明 NoC 是瓶颈；还需结合网络注入率、队列、冲突、
  width-limited/downstream-full 周期和理想网络反事实实验；
- GEMM 与 FlashAttention 使用 TMA，普通 global load/store 计数可能为 0，应同时查看
  `gpgpu_n_tma_*` 与 L2 流量；
- `OMP_NUM_THREADS=1` 控制仿真主机资源使用，影响墙钟时间，不应改变模拟周期。

### 5.3 模型边界

`SM120_RTX5090` 是有效性能模型，不是物理芯片拓扑的逐项公开还原。某个 workload
上的接近可能来自多个组件误差的抵消，因此后续 NoC 结论必须依赖多 workload、
单因素扫描和反事实对照。

## 6. 可复现性检查清单

每次新实验至少保存：

- Git commit 与工作区是否干净；
- `gpgpusim.config` 的 SHA-256；
- trace/PTX/CUBIN 的 hash；
- host compiler、CUDA、driver 版本；
- 完整运行命令、环境变量与返回码；
- stdout/stderr 原始日志；
- 开始/结束时间、墙钟时间和最大常驻内存；
- 从日志提取的 CSV/JSON 汇总。

建议每次运行创建独立目录，并使用如下结构：

```text
experiment-run/
  manifest.json
  config.sha256
  vectoradd.log
  gemm.log
  flashattention.log
  metrics.csv
```

只有返回码为 0 且功能成功标志存在的运行才能进入结果表。

## 7. 常见问题

| 现象 | 原因 | 处理方法 |
| --- | --- | --- |
| `_Float32 is undefined` | Conda sysroot 与 nvcc host 编译组合不兼容 | 使用 `-ccbin /usr/bin/g++` |
| `Cannot detect the app's CUDA version` | harness 依赖无版本 SONAME | 创建 `libcudart.so.12` runtime 副本并重新链接 |
| `Unknown Option` | 旧 harness 加载了旧模拟器库或旧配置 | 复制最新配置、重新链接，并用 `ldd` 核对 |
| guide generation 缺少环境变量 | 未加载项目环境 | 从仓库根目录 `source setup_environment` |
| `No PTX sections found` | PTX 架构或提取路径不匹配 | 同时生成 `sm_120a`/`compute_120a` 并核对配置 |
| driver harness 正常退出但无最终汇总 | `cuCtxSynchronize()` 为空实现，进程早于模拟线程退出 | 更新到含同步修复的提交，重建 runtime、重新链接 harness 并完整重跑 |

旧 launcher 可能带有指向历史 runtime 的 RPATH。每次模拟器更新后都应重新链接，并
执行以下检查：

```bash
ldd ./kernel_tma_gemm_launch1-latest | grep libcudart
readelf -d ./kernel_tma_gemm_launch1-latest | grep NEEDED
```

预期加载本次生成的 runtime，且 `DT_NEEDED` 包含 `libcudart.so.12`。

## 8. 结论与后续工作

本实验完成了三个 workload 的新基线运行，并修复了 driver harness 的同步语义。
VectorAdd 与 GEMM 的功能验证通过；FlashAttention 的 8192 个 CTA 全部完成并输出
完整最终统计。这些结果可作为 GPU NoC 研究的初始对照组。

下一步继续补齐 local/global NoC 的逐端口排队延迟分位数和按请求类型拆分的流量
矩阵，再使用定向微基准及 GEMM、FlashAttention、MoE 等模型负载开展瓶颈归因。
