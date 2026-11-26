// CUDA evaluator: one thread block per candidate route.
//   Phase 1: threads stride over pickup positions a and test the pickup insertion in O(1) from the precomputed state at a;
//            feasible positions (and the state after the pickup) go to shared memory.
//   Phase 2: threads stride over (feasible a, b >= a) pairs and replay the route from a.
// Block-level min reduction with ties broken by lowest (a, b), the CPU scan order.
// Two entry points: gpu_eval (routes sent with every call; used by the dynamic mode) and gpu_eval_slots (routes resident on the GPU,
// only changed routes re-sent; used by the static mode).
// Build: see Makefile. Requires -fmad=false so the arithmetic matches the CPU bit for bit.
#include <chrono>
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

// Body shared by both kernels: best insertion of ireq into one route, written to *out by thread 0.
__device__ void eval_route(const GpuInstance& G, const GpuCand& m, const int* r, const int* pr, const double* Tr, const double* Sc, const int* Sl,
                           int ireq, double theta, Best* out, unsigned long long* cnt) {
  __shared__ Best sh[BLOCK];
  __shared__ GpuPick pk[GPU_LMAX + 1];
  __shared__ int alist[GPU_LMAX + 1];
  __shared__ int offs[GPU_LMAX + 2];                                    // offs[j] = first pair index of feasible pickup j
  __shared__ int na;
  if (threadIdx.x == 0) na = 0;
  __syncthreads();
  for (int a = m.k0 + threadIdx.x; a <= m.len; a += BLOCK) {           // phase 1: pickup feasibility per position
    GpuPick q;
    if (gpu_pickup(G, r, Tr, Sc, Sl, ireq, theta, a, &q)) { int j = atomicAdd(&na, 1); pk[j] = q; alist[j] = a; }
  }
  __syncthreads();
  const int w = m.len - m.k0 + 1, nA = na;
  for (int j = threadIdx.x; j < nA; j += BLOCK)                         // phase 1b: last delivery position worth trying (count stored in offs)
    offs[j + 1] = gpu_scan(G, m, r, pr, Tr, pk[j], ireq, theta, alist[j]) - alist[j] + 1;
  __syncthreads();
  if (threadIdx.x == 0) { offs[0] = 0; for (int j = 0; j < nA; j++) offs[j + 1] += offs[j]; }
  __syncthreads();
  const int total = offs[nA];
  Best mine = {1e300, -1, -1};
  for (int idx = threadIdx.x; idx < total; idx += BLOCK) {             // phase 2: packed (a, b) pairs
    int lo = 0, hi = nA - 1;
    while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (offs[mid] <= idx) lo = mid; else hi = mid - 1; }
    int a = alist[lo], b = a + (idx - offs[lo]);
    double dl;
    if (gpu_pair(G, m, r, pr, Tr, pk[lo], ireq, theta, a, b, &dl)) { Best c = {dl, a, b}; if (mine.a < 0 || better(c, mine)) mine = c; }
  }
  if (cnt && threadIdx.x == 0) { atomicAdd(&cnt[0], (unsigned long long)w); atomicAdd(&cnt[1], (unsigned long long)nA); atomicAdd(&cnt[2], (unsigned long long)total); }
  sh[threadIdx.x] = mine; __syncthreads();
  for (int s = BLOCK / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) { Best o = sh[threadIdx.x + s]; if (o.a >= 0 && (sh[threadIdx.x].a < 0 || better(o, sh[threadIdx.x]))) sh[threadIdx.x] = o; }
    __syncthreads();
  }
  if (threadIdx.x == 0) *out = sh[0];
}

__global__ void k_eval(GpuInstance G, const GpuCand* meta, const int* stops, const int* pp, const double* Tt, const double* Sc, const int* Sl,
                       int ireq, double theta, Best* out, unsigned long long* cnt) {
  const GpuCand m = meta[blockIdx.x];
  eval_route(G, m, stops + m.off, pp + m.off, Tt + m.off, Sc + m.qoff, Sl + m.qoff, ireq, theta, out + blockIdx.x, cnt);
}
__global__ void k_scatter(GpuSlot* slots, const int* idx, const GpuSlot* upd) {       // one block per updated slot
  int* dst = reinterpret_cast<int*>(slots + idx[blockIdx.x]); const int* src = reinterpret_cast<const int*>(upd + blockIdx.x);
  for (int w = threadIdx.x; w < (int)(sizeof(GpuSlot) / sizeof(int)); w += blockDim.x) dst[w] = src[w];
}
__global__ void k_eval_slots(GpuInstance G, const GpuSlot* slots, const int* cand, int ireq, double theta, Best* out, unsigned long long* cnt) {
  const GpuSlot& S = slots[cand[blockIdx.x]];
  if (!S.feasible) { if (threadIdx.x == 0) out[blockIdx.x] = Best{1e300, -1, -1}; return; }   // uniform per block: no thread reaches a barrier
  GpuCand m; m.off = 0; m.qoff = 0; m.len = S.len; m.k0 = S.k0; m.base = S.base;
  eval_route(G, m, S.stops, S.pp, S.Tt, S.Sc, S.Sl, ireq, theta, out + blockIdx.x, cnt);
}

static GpuInstance dG; static double *dD, *de, *dl, *ddir, *dmr; static int *dpo, *ddn;
static GpuCand* dmeta = nullptr; static int* dstops = nullptr; static int* dpp = nullptr; static double* dTt = nullptr; static Best* dout = nullptr;
static double* dSc = nullptr; static int* dSl = nullptr;
static size_t cap_c = 0, cap_s = 0, cap_q = 0;
static GpuSlot* dslots = nullptr; static GpuSlot* dupd = nullptr; static int* didx = nullptr; static size_t cap_u = 0, cap_i = 0;
static Best* hout = nullptr; static int* hidx = nullptr; static size_t cap_ho = 0, cap_hi = 0;      // pinned host staging
static bool g_prof_on = false; static GpuProf g_gp; static unsigned long long* dcnt = nullptr;
static inline double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void gpu_set_prof(bool on) {
  g_prof_on = on;
  if (on && !dcnt && cudaMalloc(&dcnt, 3 * sizeof(unsigned long long)) == cudaSuccess) cudaMemset(dcnt, 0, 3 * sizeof(unsigned long long));
}
GpuProf gpu_get_prof() {
  GpuProf p = g_gp; unsigned long long c[3] = {0, 0, 0};
  if (dcnt && cudaMemcpy(c, dcnt, sizeof(c), cudaMemcpyDeviceToHost) == cudaSuccess) { p.positions = (long long)c[0]; p.feas_a = (long long)c[1]; p.pairs = (long long)c[2]; }
  return p;
}

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
bool gpu_eval(int nc, const GpuCand* meta, const int* stops, int ns, const int* pp, const double* Tt, const double* Sc, const int* Sl, int nq,
              int ireq, double theta, double* od, int* oa, int* ob) {
  double t0 = g_prof_on ? now_s() : 0.0, t1;
  if ((size_t)nc > cap_c) { cudaFree(dmeta); cudaFree(dout); cap_c = nc * 2; CK(cudaMalloc(&dmeta, cap_c * sizeof(GpuCand))); CK(cudaMalloc(&dout, cap_c * sizeof(Best))); }
  if ((size_t)ns > cap_s) { cudaFree(dstops); cudaFree(dpp); cudaFree(dTt); cap_s = ns * 2; CK(cudaMalloc(&dstops, cap_s * sizeof(int))); CK(cudaMalloc(&dpp, cap_s * sizeof(int))); CK(cudaMalloc(&dTt, cap_s * sizeof(double))); }
  if ((size_t)nq > cap_q) { cudaFree(dSc); cudaFree(dSl); cap_q = nq * 2; CK(cudaMalloc(&dSc, cap_q * sizeof(double))); CK(cudaMalloc(&dSl, cap_q * sizeof(int))); }
  if (g_prof_on) { t1 = now_s(); g_gp.realloc += t1 - t0; t0 = t1; }
  CK(cudaMemcpy(dmeta, meta, nc * sizeof(GpuCand), cudaMemcpyHostToDevice));
  if (ns > 0) {
    CK(cudaMemcpy(dstops, stops, ns * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpp, pp, ns * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dTt, Tt, ns * sizeof(double), cudaMemcpyHostToDevice));
  }
  CK(cudaMemcpy(dSc, Sc, nq * sizeof(double), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSl, Sl, nq * sizeof(int), cudaMemcpyHostToDevice));
  if (g_prof_on) { t1 = now_s(); g_gp.h2d += t1 - t0; t0 = t1; g_gp.bytes_h2d += (long long)nc * sizeof(GpuCand) + (long long)ns * (2 * sizeof(int) + sizeof(double)) + (long long)nq * (sizeof(double) + sizeof(int)); }
  k_eval<<<nc, BLOCK>>>(dG, dmeta, dstops, dpp, dTt, dSc, dSl, ireq, theta, dout, g_prof_on ? dcnt : nullptr);
  CK(cudaGetLastError());
  if (g_prof_on) { CK(cudaDeviceSynchronize()); t1 = now_s(); g_gp.kernel += t1 - t0; t0 = t1; }
  std::vector<Best> h(nc);
  CK(cudaMemcpy(h.data(), dout, nc * sizeof(Best), cudaMemcpyDeviceToHost));   // also synchronizes
  if (g_prof_on) g_gp.d2h += now_s() - t0;
  for (int c = 0; c < nc; c++) { od[c] = h[c].a >= 0 ? h[c].d : 1e300; oa[c] = h[c].a; ob[c] = h[c].b; }
  return true;
}

bool gpu_slots_init(int nslots) {
  CK(cudaMalloc(&dslots, (size_t)nslots * sizeof(GpuSlot)));
  CK(cudaMemset(dslots, 0, (size_t)nslots * sizeof(GpuSlot)));     // len 0, feasible 0 until first upload
  return true;
}
bool gpu_eval_slots(int nc, const int* cand, int nupd, const int* upd_idx, const GpuSlot* upd, int ireq, double theta, double* od, int* oa, int* ob) {
  double t0 = g_prof_on ? now_s() : 0.0, t1;
  size_t ni = (size_t)nc + nupd;
  if ((size_t)nc > cap_c) { cudaFree(dmeta); cudaFree(dout); cap_c = nc * 2; CK(cudaMalloc(&dmeta, cap_c * sizeof(GpuCand))); CK(cudaMalloc(&dout, cap_c * sizeof(Best))); }
  if (ni > cap_i) { cudaFree(didx); cap_i = ni * 2; CK(cudaMalloc(&didx, cap_i * sizeof(int))); }
  if ((size_t)nupd > cap_u) { cudaFree(dupd); cap_u = nupd * 2 + 4; CK(cudaMalloc(&dupd, cap_u * sizeof(GpuSlot))); }
  if (ni > cap_hi) { cudaFreeHost(hidx); cap_hi = ni * 2; CK(cudaMallocHost(&hidx, cap_hi * sizeof(int))); }
  if ((size_t)nc > cap_ho) { cudaFreeHost(hout); cap_ho = nc * 2; CK(cudaMallocHost(&hout, cap_ho * sizeof(Best))); }
  if (g_prof_on) { t1 = now_s(); g_gp.realloc += t1 - t0; t0 = t1; }
  for (int c = 0; c < nc; c++) hidx[c] = cand[c];                     // one index buffer: candidate slots, then updated slots
  for (int j = 0; j < nupd; j++) hidx[nc + j] = upd_idx[j];
  CK(cudaMemcpy(didx, hidx, ni * sizeof(int), cudaMemcpyHostToDevice));
  if (nupd > 0) CK(cudaMemcpy(dupd, upd, (size_t)nupd * sizeof(GpuSlot), cudaMemcpyHostToDevice));
  if (g_prof_on) { t1 = now_s(); g_gp.h2d += t1 - t0; t0 = t1; g_gp.bytes_h2d += (long long)(ni * sizeof(int) + (size_t)nupd * sizeof(GpuSlot)); }
  if (nupd > 0) k_scatter<<<nupd, 256>>>(dslots, didx + nc, dupd);   // same stream: finishes before the evaluation starts
  k_eval_slots<<<nc, BLOCK>>>(dG, dslots, didx, ireq, theta, dout, g_prof_on ? dcnt : nullptr);
  CK(cudaGetLastError());
  if (g_prof_on) { CK(cudaDeviceSynchronize()); t1 = now_s(); g_gp.kernel += t1 - t0; t0 = t1; }
  CK(cudaMemcpy(hout, dout, nc * sizeof(Best), cudaMemcpyDeviceToHost));   // also synchronizes
  if (g_prof_on) g_gp.d2h += now_s() - t0;
  for (int c = 0; c < nc; c++) { od[c] = hout[c].a >= 0 ? hout[c].d : 1e300; oa[c] = hout[c].a; ob[c] = hout[c].b; }
  return true;
}
void gpu_shutdown() { cudaDeviceReset(); }
