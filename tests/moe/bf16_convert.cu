// Run on hardware and under FlashGPU-Sim. Check RN-even and lane order exactly;
// NaN payloads are implementation-defined, so check their classification.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK(call) do { if ((call) != cudaSuccess) { std::fprintf(stderr, "CUDA failed at %d\n", __LINE__); std::exit(1); } } while (0)

__global__ void convert(const uint32_t *in, uint32_t *packed, uint16_t *scalar,
                        uint32_t *back, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float a = __uint_as_float(in[i]), b = __uint_as_float(in[(i + 1) % n]);
  unsigned p;
  unsigned short s;
  float f;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(p) : "f"(a), "f"(b));
  asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(s) : "f"(a));
  asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f) : "h"(s));
  packed[i] = p;
  scalar[i] = s;
  back[i] = __float_as_uint(f);
}

static unsigned reference(unsigned bits) {
  if ((bits & 0x7f800000u) == 0x7f800000u)
    return (bits >> 16) | ((bits & 0x7fffffu) ? 0x40u : 0u);
  return (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
}
static bool equal(unsigned a, unsigned b) {
  bool nan_a = (a & 0x7f80u) == 0x7f80u && (a & 0x7fu);
  bool nan_b = (b & 0x7f80u) == 0x7f80u && (b & 0x7fu);
  return a == b || (nan_a && nan_b);
}
int main() {
  uint32_t input[] = {0, 0x80000000u, 0x3f800000u, 0xbf800000u,
    0x3f808000u, 0x3f818000u, 0xbf808000u, 0xbf818000u,
    0x00008000u, 0x00008001u, 0x80008001u, 0x007fffffu,
    0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u,
    0x7fc12345u, 0xff812345u};
  constexpr int n = sizeof(input) / sizeof(input[0]);
  uint32_t *in, *p, *f;
  uint16_t *s;
  CHECK(cudaMalloc(&in, sizeof(input))); CHECK(cudaMalloc(&p, n * 4));
  CHECK(cudaMalloc(&s, n * 2)); CHECK(cudaMalloc(&f, n * 4));
  CHECK(cudaMemcpy(in, input, sizeof(input), cudaMemcpyHostToDevice));
  convert<<<1, 32>>>(in, p, s, f, n);
  CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
  uint32_t hp[n], hf[n]; uint16_t hs[n];
  CHECK(cudaMemcpy(hp, p, sizeof(hp), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hs, s, sizeof(hs), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hf, f, sizeof(hf), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    unsigned expected = reference(input[i]);
    if (!equal(hs[i], expected) || !equal(hp[i] >> 16, expected) ||
        !equal(hp[i] & 0xffffu, reference(input[(i + 1) % n])) ||
        !equal(hf[i] >> 16, expected) || (hf[i] & 0xffffu)) {
      std::fprintf(stderr, "FAIL index=%d scalar=%04x packed=%08x back=%08x\n", i, hs[i], hp[i], hf[i]);
      return 1;
    }
  }
  CHECK(cudaFree(in)); CHECK(cudaFree(p)); CHECK(cudaFree(s)); CHECK(cudaFree(f));
  std::puts("BF16 conversion PASSED");
}
