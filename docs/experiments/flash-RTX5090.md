# FlashGPU-Sim `flash` 分支 RTX 5090 复现实验

## 结论

已在 RTX 5090 机器上以无 sudo、完全用户目录内的环境完成 FlashGPU-Sim 构建，并在 `SM120_RTX5090` 配置下跑通 CUDA VectorAdd、Triton GEMM 与 Triton FlashAttention 的离线捕获和详细性能模拟。

| 工作负载 | 配置 | 功能验证 | 模拟周期 |
| --- | --- | --- | ---: |
| VectorAdd（2,000,000 elements） | `SM120_RTX5090` | `Test PASSED` | 30,133 |
| Triton GEMM（M=2560, N=64, K=2560） | `SM120_RTX5090` | `Validation PASSED` | 78,989 |
| Triton FlashAttention（B=32, H=32, S=512, D=64，causal） | `SM120_RTX5090` | `Kernel execution completed successfully` | 794,076 |

VectorAdd 与 Triton GEMM 的周期数均与仓库 README 提供的参考值一致；FlashAttention 使用本次在 RTX 5090 上实际捕获的 Triton trace，完成了端到端离线回放。

## 代码与环境

- 基线分支：`flash`
- 基线提交：`d20ce45e3399f3a1b5b0787bdbf7c2b544a69ff9`
- 用户级 CUDA Toolkit：12.8.2（`nvcc` 12.8.93）
- 编译器：GCC/G++ 14.4.0（Conda-forge）
- Flex 2.6.4、Bison 3.8.2、`makedepend` 1.0.10
- 目标硬件配置：`SM120_RTX5090`

系统只提供 NVIDIA 驱动；所有构建工具和依赖由 Micromamba 安装到工作目录，不使用 sudo，也不修改系统 CUDA。

## 用户级环境准备

以下命令假设 Micromamba 已解压到工作目录的 `mamba-root/bin/`：

```bash
./mamba-root/bin/micromamba create -y -r ./mamba-root -n flashgpu-env \
  -c conda-forge -c nvidia \
  cuda-toolkit=12.8 flex bison zlib libglu libgl-devel xorg-makedepend patchelf
```

Conda CUDA 将架构相关头文件放在 `targets/x86_64-linux/include/`；而项目的 Makefile 使用 `$CUDA_INSTALL_PATH/include`。构建前在该用户级 Toolkit 中建立兼容链接：

```bash
CUDA_ROOT="$PWD/mamba-root/envs/flashgpu-env"
for h in "$CUDA_ROOT"/targets/x86_64-linux/include/*.h; do
  ln -sf "../targets/x86_64-linux/include/$(basename "$h")" "$CUDA_ROOT/include/$(basename "$h")"
done
ln -sfn ../targets/x86_64-linux/include/crt "$CUDA_ROOT/include/crt"
ln -sfn ../targets/x86_64-linux/include/cuda "$CUDA_ROOT/include/cuda"
```

## CUDA 12.8 PTX 初始化修复

CUDA 6+ 使用独立 PTX 文件选择逻辑，但 `cuobjdumpInit_internal()` 在完成新路径提取后仍调用旧 `pruneSectionList()`；旧列表为空会导致运行直接中止。修复方式是将旧 prune/merge 逻辑限制为 `CUDART_VERSION < 6000`：

```cpp
#if (CUDART_VERSION < 6000)
  const char *pre_load = getenv("CUOBJDUMP_SIM_FILE");
  if (pre_load == NULL || strlen(pre_load) == 0) {
    cuobjdumpSectionList = pruneSectionList(context);
    cuobjdumpSectionList = mergeSections();
  }
#endif
```

然后构建模拟器：

```bash
CUDA_ROOT="$PWD/mamba-root/envs/flashgpu-env"
./mamba-root/bin/micromamba run -r ./mamba-root -n flashgpu-env \
  env CUDA_INSTALL_PATH="$CUDA_ROOT" bash -c 'source setup_environment && make -j8'
```

## 运行时 ABI 兼容与 VectorAdd

Conda CUDA 12.8 默认使 `nvcc` 链接版本化的 `libcudart.so.12`，而模拟器库默认 SONAME 为无版本的 `libcudart.so`。为让应用链接模拟器且保留 CUDA major version 自动检测，对模拟器库创建用户级副本并将 SONAME 设为 `libcudart.so.12`：

```bash
SIM_LIB="$PWD/FlashGPU-Sim/lib/gcc-14.4.0/cuda-12080/release"
RUNTIME_DIR="$PWD/flashgpu-runtime12"
mkdir -p "$RUNTIME_DIR"
cp "$SIM_LIB/libcudart.so" "$RUNTIME_DIR/libcudart.so.12"
./mamba-root/bin/micromamba run -r ./mamba-root -n flashgpu-env \
  patchelf --set-soname libcudart.so.12 "$RUNTIME_DIR/libcudart.so.12"
ln -sfn libcudart.so.12 "$RUNTIME_DIR/libcudart.so"
```

以 `sm_120a` 编译示例并显式链接该运行时：

```bash
cd FlashGPU-Sim/tutorials/vectorAdd
nvcc -O2 -std=c++17 -cudart none \
  -gencode arch=compute_120a,code=sm_120a \
  -gencode arch=compute_120a,code=compute_120a \
  vectorAdd.cu -L"$RUNTIME_DIR" -lcudart \
  -Xlinker=-rpath -Xlinker="$RUNTIME_DIR" -o run/vectorAdd
cp -a ../../configs/SM120_RTX5090/. run/
(cd run && OMP_NUM_THREADS=1 ./vectorAdd)
```

实际输出：

```text
gpu_tot_sim_cycle = 30133
Test PASSED
GPGPU-Sim: *** exit detected ***
```

## Triton GEMM 离线重放

仓库已包含离线捕获的 Triton GEMM trace，因此不需要重新执行 GPU capture。
将 trace、launcher 与 RTX 5090 配置置于同一工作目录；先用仓库 Makefile 打包 fatbin，
再以版本化模拟器 runtime 编译 harness：

```bash
cd FlashGPU-Sim/tutorials/triton-gemm
mkdir -p run/tracking/data run/tracking/launchers
cp -a trace/data/. run/tracking/data/
cp -a trace/launchers/. run/tracking/launchers/
cp -a ../../configs/SM120_RTX5090/. run/tracking/launchers/
cd run/tracking/launchers
make -f kernel_tma_gemm_launch1_Makefile kernel_tma_gemm_launch1_kernel.fatbin
nvcc -arch=sm_120a -cudart none kernel_tma_gemm_launch1_harness.cu \
  -L"$RUNTIME_DIR" -lcudart -lcuda \
  -Xlinker=-rpath -Xlinker="$RUNTIME_DIR" -o kernel_tma_gemm_launch1
OMP_NUM_THREADS=1 ./kernel_tma_gemm_launch1
```

实际输出：

```text
gpu_tot_sim_cycle = 78989
Validation PASSED for arg[2]: all 163840 elements match
GPGPU-Sim: *** exit detected ***
```

## Triton FlashAttention：实际 GPU 捕获与离线回放

FlashAttention 教程不提供预制 trace，因此先在 GPU 上使用 PyTorch 2.9.0+cu128、Triton 3.5.0 和项目内的 `TritonTrace` 完成离线编译、参数捕获与 launcher 生成。用户级 Python 环境可按需安装：

```bash
./mamba-root/bin/micromamba run -r ./mamba-root -n flashgpu-env \
  python -m pip install torch==2.9.0 triton==3.5.0
./mamba-root/bin/micromamba run -r ./mamba-root -n flashgpu-env \
  python -m pip install -e FlashGPU-Sim/tools
```

捕获命令（默认形状为 `B=32, H=32, S=512, D=64`，因果注意力）：

```bash
CUDA_ROOT="$PWD/mamba-root/envs/flashgpu-env"
VIRTUAL_ENV="$CUDA_ROOT" CUDA_INSTALL_PATH="$CUDA_ROOT" \
  PATH="$CUDA_ROOT/bin:$PATH" \
  bash FlashGPU-Sim/tutorials/triton-flash-attention/capture.sh
```

捕获后使用教程生成的 Makefile 打包 fatbin，并与上文的版本化模拟器 runtime 重新链接 harness。运行时必须同时设置 `CUDA_INSTALL_PATH` 与 `PATH`，使模拟器能调用 Toolkit 中的 `cuobjdump` 抽取 PTX：

```bash
cd FlashGPU-Sim/tutorials/triton-flash-attention/run/tracking/launchers
cp -a ../../../../configs/SM120_RTX5090/. .
make -f flash_attention_kernel_launch1_Makefile \
  flash_attention_kernel_launch1_kernel.fatbin
nvcc -arch=sm_120a -cudart none \
  flash_attention_kernel_launch1_harness.cu \
  -L"$RUNTIME_DIR" -lcudart -lcuda \
  -Xlinker=-rpath -Xlinker="$RUNTIME_DIR" \
  -o flash_attention_kernel_launch1
OMP_NUM_THREADS=1 CUDA_INSTALL_PATH="$CUDA_ROOT" PATH="$CUDA_ROOT/bin:$PATH" \
  LD_LIBRARY_PATH="$RUNTIME_DIR" ./flash_attention_kernel_launch1
```

实际输出：

```text
gpu_tot_sim_cycle = 794076
Kernel execution completed successfully
GPGPU-Sim: *** exit detected ***
```

本次详细模拟耗时约 2 小时 33 分钟，平均约 86 simulated cycles/s；因而建议将全规模作业以脱离终端会话的方式运行并保存日志。

## 备注

`OMP_NUM_THREADS=1` 用于降低共享机器上的 CPU 占用；可在资源允许时提高以缩短周期级仿真耗时。
