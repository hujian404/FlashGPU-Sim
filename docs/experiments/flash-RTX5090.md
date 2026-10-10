# FlashGPU-Sim RTX 5090 复现实验记录

## 1. 当前结论

本分支已在 RTX 5090 主机上以普通用户权限完成 FlashGPU-Sim 构建，并在
`SM120_RTX5090` 配置下完成三个层次的验证：CUDA VectorAdd、Triton GEMM 和
Triton FlashAttention。2026-10-10 同步上游 `flash` 后，模拟器已重新构建，
VectorAdd 已在新代码和新配置上重新验证通过。GEMM 与 FlashAttention 的完整结果
来自同步前的基线，保留为可复现实验基准，后续应在新基线上重跑后再做跨版本比较。

| 工作负载 | 输入规模 | 功能结果 | 模拟周期 | 指令数 | IPC | L2 miss rate | 基线状态 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| VectorAdd | 2,000,000 elements | `Test PASSED` | 31,874 | 44,001,408 | 1,380.4796 | 99.91% | 最新合并基线复验 |
| Triton GEMM | M=2560, N=64, K=2560 | 163,840 elements 全匹配 | 78,989 | 21,632,240 | 273.8640 | 51.85% | 旧基线完整运行 |
| Triton FlashAttention | B=32, H=32, S=512, D=64, causal | kernel 成功完成 | 794,076 | 3,085,066,240 | 3,885.1021 | 36.36% | 旧基线完整运行 |

最新 VectorAdd 相比旧基线的 30,133 cycles 增加到 31,874 cycles。上游同步包含
RTX 5090 L2、内存传输、执行时序和互连服务模型的大量更新，因此该差异不能直接
归因于某一个部件；后续必须用固定 trace、配置差分和单因素敏感性实验定位。

## 2. 版本与复现状态

- 实验分支：`experiment/flash-RTX5090`
- 上游仓库：`https://github.com/FlashGPU-Sim/FlashGPU-Sim.git`
- 上游分支及同步提交：`flash@291fbe172332eb481556632a79eb9cfc93cf46bf`
- 上游合并提交：`d5ee614da5fa1a094d27af6602636af4f3a6e318`
- 初始复现基线：`d20ce45e3399f3a1b5b0787bdbf7c2b544a69ff9`
- 用户级 CUDA Toolkit：12.8.2（`nvcc` 12.8.93）
- 模拟器构建目录：`build/gcc-13.4.0/cuda-12080/release`
- CUDA 示例 host compiler：Ubuntu GCC/G++ 12.3.0
- 目标配置：`configs/SM120_RTX5090`

结果分为两类，避免把不同代码版本的数据混用：

1. **最新基线已复验**：构建和 VectorAdd。
2. **旧基线已完成、待重跑**：Triton GEMM 和 FlashAttention。旧数据可用于验证
   工作流是否完整，但不应与最新配置的结果组成性能结论。

## 3. 环境准备

所有工具安装在 `/data/home/jian.hu/work`，不需要 sudo，也不替换系统 CUDA。

```bash
WORK=/data/home/jian.hu/work
REPO="$WORK/FlashGPU-Sim"
MAMBA_ROOT="$WORK/mamba-root"
CUDA_ROOT="$MAMBA_ROOT/envs/flashgpu-env"

"$MAMBA_ROOT/bin/micromamba" create -y -r "$MAMBA_ROOT" -n flashgpu-env \
  -c conda-forge -c nvidia \
  cuda-toolkit=12.8 flex bison zlib libglu libgl-devel \
  xorg-makedepend patchelf
```

Conda CUDA 的架构相关头文件位于 `targets/x86_64-linux/include`，项目构建系统还会
访问 `$CUDA_INSTALL_PATH/include`。若环境中缺少对应入口，可在用户环境内补链接：

```bash
for h in "$CUDA_ROOT"/targets/x86_64-linux/include/*.h; do
  ln -sf "../targets/x86_64-linux/include/$(basename "$h")" \
    "$CUDA_ROOT/include/$(basename "$h")"
done
ln -sfn ../targets/x86_64-linux/include/crt "$CUDA_ROOT/include/crt"
ln -sfn ../targets/x86_64-linux/include/cuda "$CUDA_ROOT/include/cuda"
```

## 4. 构建最新基线

```bash
cd "$REPO"
"$MAMBA_ROOT/bin/micromamba" run -r "$MAMBA_ROOT" -n flashgpu-env \
  env CUDA_INSTALL_PATH="$CUDA_ROOT" \
  bash -c 'source setup_environment && make -j16'
```

2026-10-10 在合并提交 `d5ee614d` 上返回码为 0。当前上游构建会生成
`libcudart.so.12` 链接，因此旧实验中复制动态库并用 `patchelf` 修改 SONAME 的步骤
通常不再需要。

本分支仍保留 CUDA 6+ PTX 初始化保护：旧的 `pruneSectionList()`/`mergeSections()`
只在 `CUDART_VERSION < 6000` 时执行。否则 CUDA 12.8 新提取路径完成后再次处理空的
旧列表会导致初始化中止。

## 5. 最新基线 VectorAdd 复验

### 5.1 编译

Conda sysroot 与 `nvcc` 组合可能对 `_Float32` 等类型报错。示例程序使用系统
G++ 12.3 作为 host compiler 可稳定编译：

```bash
cd "$REPO/tutorials/vectorAdd"
make clean
make CUDA_INSTALL_PATH="$CUDA_ROOT" \
  NVCC_FLAGS='-O2 -std=c++17 -ccbin /usr/bin/g++'

mkdir -p run
cp vectorAdd run/vectorAdd-latest
cp -a "$REPO/configs/SM120_RTX5090/." run/
```

### 5.2 运行

新版 SASS-guided PTX reorder 会读取 `GPGPUSIM_ROOT` 和
`PTXAS_CUDA_INSTALL_PATH`/`CUDA_INSTALL_PATH`。必须通过 `setup_environment`
启动，不能只设置 `LD_LIBRARY_PATH`：

```bash
cd "$REPO"
CUDA_INSTALL_PATH="$CUDA_ROOT" bash -c '
  source setup_environment
  cd tutorials/vectorAdd/run
  OMP_NUM_THREADS=1 ./vectorAdd-latest 2>&1 | tee vectorAdd-latest.log
'
```

关键输出：

```text
gpu_sim_insn = 44001408
gpu_tot_sim_cycle = 31874
gpu_tot_ipc = 1380.4796
gpu_tot_issued_cta = 7813
gpu_tot_occupancy = 92.4220%
L2_total_cache_accesses = 750680
L2_total_cache_misses = 750004
L2_total_cache_miss_rate = 0.9991
Test PASSED
GPGPU-Sim: *** exit detected ***
```

## 6. Triton GEMM（旧基线完整结果）

仓库带有离线 trace。将 trace、launcher 和 RTX 5090 配置放到同一运行目录后打包
fatbin，并链接模拟器 runtime。以下命令模板应在最新基线重跑时使用新构建目录；
具体 launcher 名称以 trace 目录实际内容为准。

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

旧基线关键结果：

```text
gpu_tot_sim_cycle = 78989
gpu_tot_sim_insn = 21632240
gpu_tot_ipc = 273.8640
gpu_tot_issued_cta = 40
gpu_tot_occupancy = 8.3333%
gpgpu_n_tma_tx_completed = 3240
L2_total_cache_accesses = 829440
L2_total_cache_misses = 430080
L2_total_cache_miss_rate = 0.5185
Validation PASSED for arg[2]: all 163840 elements match
```

## 7. Triton FlashAttention（旧基线完整结果）

FlashAttention 不附带预制 trace。本次使用 PyTorch 2.9.0+cu128、Triton 3.5.0
和项目内 `TritonTrace` 在 RTX 5090 上完成真实捕获，再进行离线回放。

```bash
"$MAMBA_ROOT/bin/micromamba" run -r "$MAMBA_ROOT" -n flashgpu-env \
  python -m pip install torch==2.9.0 triton==3.5.0
"$MAMBA_ROOT/bin/micromamba" run -r "$MAMBA_ROOT" -n flashgpu-env \
  python -m pip install -e "$REPO/tools"

VIRTUAL_ENV="$CUDA_ROOT" CUDA_INSTALL_PATH="$CUDA_ROOT" \
  PATH="$CUDA_ROOT/bin:$PATH" \
  bash "$REPO/tutorials/triton-flash-attention/capture.sh"
```

捕获记录确认：1 个 kernel 编译、1 次 launch，且生成 CUBIN、PTX、metadata、harness、
ptxinfo 和 Makefile。旧基线离线模拟耗时约 2 小时 33 分钟，约 86 simulated
cycles/s。关键输出：

```text
gpu_tot_sim_cycle = 794076
gpu_tot_sim_insn = 3085066240
gpu_tot_ipc = 3885.1021
gpu_tot_issued_cta = 8192
gpu_tot_occupancy = 15.7341%
gpgpu_n_tma_tx_completed = 90112
gpgpu_n_tma_read_tx_completed = 81920
gpgpu_n_tma_write_tx_completed = 8192
L2_total_cache_accesses = 23068672
L2_total_cache_misses = 8388608
L2_total_cache_miss_rate = 0.3636
Kernel execution completed successfully
```

## 8. 结果解读边界

- `gpu_tot_ipc` 是全 GPU 汇总值，不能直接与单 SM IPC 比较。
- 高 L2 miss rate 不等价于 NoC 是瓶颈；必须同时观察注入率、队列占用、冲突、
  width-limited/downstream-full 周期以及反事实带宽扫描。
- GEMM 与 FlashAttention 使用 TMA，普通 global load/store 计数可能为 0；应结合
  `gpgpu_n_tma_*` 和 L2 流量判断。
- `OMP_NUM_THREADS=1` 用于控制共享服务器 CPU 占用，它影响仿真墙钟时间，不应改变
  模拟周期结果。
- 当前配置是有效模型，不是 RTX 5090 物理拓扑的逐项公开还原；单个 workload 的拟合
  可能包含多个部件误差的抵消。

## 9. 已知问题与排障

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `No PTX sections found` | binary/配置/提取路径不一致，或旧初始化路径误处理 | 同时生成 `sm_120a` 和 `compute_120a`，保留本分支初始化保护 |
| `_Float32 is undefined` | Conda sysroot 与 nvcc host 编译组合不兼容 | 示例用 `-ccbin /usr/bin/g++` |
| `guide generation requires GPGPUSIM_ROOT...` | 未加载新版 guide 环境 | 从仓库根目录 `source setup_environment` 后运行 |
| 运行加载真实 CUDA runtime | 动态库搜索顺序错误 | 通过 `setup_environment` 启动，并用 `ldd` 核对 |
| 长任务 SSH 断开 | 详细仿真墙钟时间很长 | 使用 tmux/nohup，保存命令、配置、stdout/stderr 和返回码 |

早期文件 `vectoradd-final.log` 实际停在 PTX 提取错误；旧基线成功证据位于
`vectoradd-patched.log`。本文不再把失败日志当作成功结果来源。

## 10. 下一步

GPU NoC 的工作负载选择、指标、实验矩阵、优化候选和阶段验收标准见
[GPU NoC 瓶颈分析与优化 Roadmap](gpu-noc-roadmap.md)。在进入设计优化前，应先在
最新提交上重跑 GEMM 与 FlashAttention，并把原始结果整理为可机读 CSV/JSON。
