// Interface between solver.cpp (host) and the GPU candidate evaluator. Plain structs only, so g++ and nvcc can both include it.
#pragma once
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
#include <cmath>
#define GPU_LMAX 128                       // max stops per route handled on the GPU; longer routes fall back to the CPU

struct GpuInstance {                       // pointers are device pointers in the CUDA build, host pointers in the CPU stub
  int V, n, Q, depot; double s, H, alpha, beta, gamma, delta, wmax;
  const double *D, *e, *l, *direct, *maxride; const int *po, *dn;
};
struct GpuCand {                           // one vehicle route to evaluate
  int off, len, k0; double base, t; int prev, load, at_depot;   // off: start in flat stops/T/pp; (t,prev,load,at_depot) = state after frozen prefix
};

// Evaluate inserting request ireq with pickup at position a and delivery at position b (a <= b, positions refer to the original route).
// Same arithmetic, in the same order, as step()/finish() in solver.cpp, so results are bit-identical when FMA contraction is off.
// Returns false if infeasible; otherwise *delta = new suffix cost - base.
HD inline bool gpu_eval_one(const GpuInstance& G, const GpuCand& m, const int* stops, const int* pp, const double* T, int ireq, double theta, int a, int b, double* delta) {
  const double EPS = 1e-6;
  double tt[GPU_LMAX];
  const int* r = stops + m.off; const int* pr = pp + m.off; const double* Tr = T + m.off;
  double t = m.t, cost = 0.0, ti = 0.0; int prev = m.prev, load = m.load; bool atd = m.at_depot != 0;
  int total = m.len - m.k0 + 2;
  for (int q = 0; q < total; q++) {
    int p = m.k0 + q, kind, oi = -1, v;
    if (p < a) { kind = 0; oi = p; }
    else if (p == a) kind = 1;
    else if (p <= b) { kind = 0; oi = p - 1; }
    else if (p == b + 1) kind = 2;
    else { kind = 0; oi = p - 2; }
    int req, x; bool isp; double ew, lw;
    if (kind == 0) { v = r[oi]; isp = v < G.n; req = isp ? v : v - G.n; }
    else { isp = (kind == 1); req = ireq; }
    x = isp ? G.po[req] : G.dn[req];
    ew = isp ? G.e[req] : G.e[req] + G.s + G.direct[req];
    lw = isp ? G.l[req] : G.l[req] + G.s + G.maxride[req];
    double dep = atd ? 0.0 : t + G.s;
    if (dep < theta) dep = theta;
    double d = G.D[(size_t)prev * G.V + x];
    cost += G.alpha * d;
    double tn = dep + d; if (ew > tn) tn = ew;
    if (tn > lw + EPS) return false;
    if (isp) {
      if (++load > G.Q) return false;
      double w = tn - G.e[req];
      double ex = w - G.wmax; if (ex < 0.0) ex = 0.0;
      cost += G.beta * w + G.delta * ex;
      if (kind == 1) ti = tn;
    } else {
      --load;
      double pk;
      if (kind == 2) pk = ti; else { int pi = pr[oi]; pk = (pi < m.k0) ? Tr[pi] : tt[pi - m.k0]; }
      double rt = tn - (pk + G.s);
      if (rt > G.maxride[req] + EPS) return false;
      double ex = rt - G.direct[req]; if (ex < 0.0) ex = 0.0;
      cost += G.gamma * ex;
    }
    if (kind == 0) tt[oi - m.k0] = tn;     // time of original stop at original index oi (>= k0 here)
    t = tn; prev = x; atd = false;
  }
  if (!atd) {
    double back = G.D[(size_t)prev * G.V + G.depot];
    if (t + G.s + back > G.H + EPS) return false;
    cost += G.alpha * back;
  }
  *delta = cost - m.base;
  return true;
}

// Host-side API (implemented in gpu.cu for CUDA, gpu_stub.cpp for the CPU emulation used for testing without a GPU).
bool gpu_init(const GpuInstance& hostInst);                                    // uploads the instance once
bool gpu_eval(int ncand, const GpuCand* meta, const int* stops, int nstops, const int* pp, const double* T, int ireq, double theta,
              double* out_delta, int* out_a, int* out_b);                      // best (min delta, then lowest a, then lowest b) per candidate route
void gpu_shutdown();
