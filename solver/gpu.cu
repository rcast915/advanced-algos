// CUDA evaluator: one thread block per candidate route, threads stride over the (a, b) position pairs of that route.
// Build: see Makefile. Requires -fmad=false so the arithmetic matches the CPU bit for bit.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "gpu.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return false; } } while (0)
static const int BLOCK = 128;
struct Best { double d; int a, b; };
__device__ inline bool better(const Best& x, const Best& y) {          // x strictly better than y: lower delta, then lower a, then lower b (matches the CPU scan order)
  if (x.d != y.d) return x.d < y.d;
  if (x.a != y.a) return x.a < y.a;
  return x.b < y.b;
}
__global__ void k_eval(GpuInstance G, const GpuCand* meta, const int* stops, const int* pp, const double* T, int ireq, double theta, Best* out) {
  __shared__ Best sh[BLOCK];
  const GpuCand m = meta[blockIdx.x];
  int w = m.len - m.k0 + 1;                                            // positions k0 .. len
  Best mine = {1e300, -1, -1};
  for (int idx = threadIdx.x; idx < w * w; idx += BLOCK) {
    int a = m.k0 + idx / w, b = m.k0 + idx % w;
    if (b < a) continue;
    double dl;
    if (gpu_eval_one(G, m, stops, pp, T, ireq, theta, a, b, &dl)) { Best c = {dl, a, b}; if (mine.a < 0 || better(c, mine)) mine = c; }
  }
  sh[threadIdx.x] = mine; __syncthreads();
  for (int s = BLOCK / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) { Best o = sh[threadIdx.x + s]; if (o.a >= 0 && (sh[threadIdx.x].a < 0 || better(o, sh[threadIdx.x]))) sh[threadIdx.x] = o; }
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = sh[0];
}

static GpuInstance dG; static double *dD, *de, *dl, *ddir, *dmr; static int *dpo, *ddn;
static GpuCand* dmeta = nullptr; static int* dstops = nullptr; static int* dpp = nullptr; static double* dT = nullptr; static Best* dout = nullptr;
static size_t cap_c = 0, cap_s = 0;

bool gpu_init(const GpuInstance& h) {
  dG = h; size_t V = h.V, n = h.n;
  CK(cudaMalloc(&dD, V * V * sizeof(double))); CK(cudaMemcpy(dD, h.D, V * V * sizeof(double), cudaMemcpyHostToDevice));
  auto up = [&](double** d, const double* s) { if (cudaMalloc(d, n * sizeof(double)) != cudaSuccess) return false; return cudaMemcpy(*d, s, n * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess; };
  if (!up(&de, h.e) || !up(&dl, h.l) || !up(&ddir, h.direct) || !up(&dmr, h.maxride)) return false;
  CK(cudaMalloc(&dpo, n * sizeof(int))); CK(cudaMemcpy(dpo, h.po, n * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&ddn, n * sizeof(int))); CK(cudaMemcpy(ddn, h.dn, n * sizeof(int), cudaMemcpyHostToDevice));
  dG.D = dD; dG.e = de; dG.l = dl; dG.direct = ddir; dG.maxride = dmr; dG.po = dpo; dG.dn = ddn;
  return true;
}
bool gpu_eval(int nc, const GpuCand* meta, const int* stops, int ns, const int* pp, const double* T, int ireq, double theta, double* od, int* oa, int* ob) {
  if ((size_t)nc > cap_c) { cudaFree(dmeta); cudaFree(dout); cap_c = nc * 2; CK(cudaMalloc(&dmeta, cap_c * sizeof(GpuCand))); CK(cudaMalloc(&dout, cap_c * sizeof(Best))); }
  if ((size_t)ns > cap_s) { cudaFree(dstops); cudaFree(dpp); cudaFree(dT); cap_s = ns * 2; CK(cudaMalloc(&dstops, cap_s * sizeof(int))); CK(cudaMalloc(&dpp, cap_s * sizeof(int))); CK(cudaMalloc(&dT, cap_s * sizeof(double))); }
  CK(cudaMemcpy(dmeta, meta, nc * sizeof(GpuCand), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dstops, stops, ns * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dpp, pp, ns * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dT, T, ns * sizeof(double), cudaMemcpyHostToDevice));
  k_eval<<<nc, BLOCK>>>(dG, dmeta, dstops, dpp, dT, ireq, theta, dout);
  CK(cudaGetLastError());
  std::vector<Best> h(nc);
  CK(cudaMemcpy(h.data(), dout, nc * sizeof(Best), cudaMemcpyDeviceToHost));   // also synchronizes
  for (int c = 0; c < nc; c++) { od[c] = h[c].a >= 0 ? h[c].d : 1e300; oa[c] = h[c].a; ob[c] = h[c].b; }
  return true;
}
void gpu_shutdown() { cudaDeviceReset(); }
