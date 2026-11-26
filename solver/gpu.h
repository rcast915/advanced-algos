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
// One vehicle route to evaluate. Per-stop arrays (stops, pp, Tt) start at off and have len entries; per-position arrays (Sc, Sl) start
// at qoff and have len + 1 entries. Built by the host in solver.cpp (route_states):
//   Tt[p]  time of stop p in the current schedule (stored time for the frozen prefix p < k0, recomputed at theta from k0 on)
//   Sl[p]  load after the first p stops;  Sc[p]  cost of stops k0 .. p-1 (0 for p <= k0), summed in the same order as the CPU
//   pp[p]  for a delivery stop, the position of its pickup in this route (-1 for pickups)
struct GpuCand { int off, qoff, len, k0; double base; };
struct GpuPick { double t, cost; int load; };   // state right after inserting the new pickup at position a

// Same arithmetic, in the same order, as step()/finish() in solver.cpp, so results are bit-identical when FMA contraction is off.

// Phase 1: insert the pickup of request ireq at position a (before original stop a), starting from the state after the first a stops.
HD inline bool gpu_pickup(const GpuInstance& G, const int* r, const double* Tt, const double* Sc, const int* Sl, int ireq, double theta, int a, GpuPick* out) {
  const double EPS = 1e-6;
  double t, cost; int prev, load; bool atd;
  if (a == 0) { t = 0.0; prev = G.depot; load = 0; cost = 0.0; atd = true; }
  else { int v = r[a - 1]; t = Tt[a - 1]; prev = v < G.n ? G.po[v] : G.dn[v - G.n]; load = Sl[a]; cost = Sc[a]; atd = false; }
  double dep = atd ? 0.0 : t + G.s;
  if (dep < theta) dep = theta;
  int x = G.po[ireq];
  double d = G.D[(size_t)prev * G.V + x];
  cost += G.alpha * d;
  double tn = dep + d; if (G.e[ireq] > tn) tn = G.e[ireq];
  if (tn > G.l[ireq] + EPS) return false;
  if (++load > G.Q) return false;
  double w = tn - G.e[ireq];
  double ex = w - G.wmax; if (ex < 0.0) ex = 0.0;
  cost += G.beta * w + G.delta * ex;
  out->t = tn; out->cost = cost; out->load = load;
  return true;
}

// Replay original stop p (a <= p < len) after the new pickup at a. tt[q - a] holds the new times of stops a .. p-1 and receives stop p's.
HD inline bool gpu_step_orig(const GpuInstance& G, const int* r, const int* pr, const double* Tt, double* tt, int a, int p, double theta,
                             double& t, double& cost, int& load, int& prev) {
  const double EPS = 1e-6;
  int v = r[p]; bool isp = v < G.n; int req = isp ? v : v - G.n;
  int x = isp ? G.po[req] : G.dn[req];
  double ew = isp ? G.e[req] : G.e[req] + G.s + G.direct[req];
  double lw = isp ? G.l[req] : G.l[req] + G.s + G.maxride[req];
  double dep = t + G.s; if (dep < theta) dep = theta;
  double d = G.D[(size_t)prev * G.V + x];
  cost += G.alpha * d;
  double tn = dep + d; if (ew > tn) tn = ew;
  if (tn > lw + EPS) return false;
  if (isp) {
    if (++load > G.Q) return false;
    double w = tn - G.e[req];
    double ex = w - G.wmax; if (ex < 0.0) ex = 0.0;
    cost += G.beta * w + G.delta * ex;
  } else {
    --load;
    int pi = pr[p]; double pkt = (pi < a) ? Tt[pi] : tt[pi - a];
    double rt = tn - (pkt + G.s);
    if (rt > G.maxride[req] + EPS) return false;
    double ex = rt - G.direct[req]; if (ex < 0.0) ex = 0.0;
    cost += G.gamma * ex;
  }
  tt[p - a] = tn; t = tn; prev = x;
  return true;
}

// Phase 1b: with the new passenger aboard (pickup at a, delivery not yet made), replay the original stops until one fails.
// Returns the first failing position f (len if none). Every delivery position b > f replays exactly this failing step before the
// delivery, so it is infeasible: only b in [a, f] need evaluating. Exact, not a heuristic.
HD inline int gpu_scan(const GpuInstance& G, const GpuCand& m, const int* r, const int* pr, const double* Tt, const GpuPick& pk, int ireq, double theta, int a) {
  double tt[GPU_LMAX];
  double t = pk.t, cost = pk.cost; int load = pk.load, prev = G.po[ireq];
  for (int p = a; p < m.len; p++) if (!gpu_step_orig(G, r, pr, Tt, tt, a, p, theta, t, cost, load, prev)) return p;
  return m.len;
}

// Phase 2: given the pickup inserted at a, insert the delivery at position b >= a (before original stop b) and replay the rest of the route.
// Returns false if infeasible; otherwise *delta = new suffix cost - base.
HD inline bool gpu_pair(const GpuInstance& G, const GpuCand& m, const int* r, const int* pr, const double* Tt, const GpuPick& pk, int ireq, double theta, int a, int b, double* delta) {
  const double EPS = 1e-6;
  double tt[GPU_LMAX];                     // new times of original stops a .. len-1, indexed p - a
  double t = pk.t, cost = pk.cost; int load = pk.load, prev = G.po[ireq];
  for (int p = a; p <= m.len; p++) {
    if (p == b) {                          // the new delivery goes right before original stop b
      int x = G.dn[ireq];
      double ew = G.e[ireq] + G.s + G.direct[ireq], lw = G.l[ireq] + G.s + G.maxride[ireq];
      double dep = t + G.s; if (dep < theta) dep = theta;
      double d = G.D[(size_t)prev * G.V + x];
      cost += G.alpha * d;
      double tn = dep + d; if (ew > tn) tn = ew;
      if (tn > lw + EPS) return false;
      --load;
      double rt = tn - (pk.t + G.s);
      if (rt > G.maxride[ireq] + EPS) return false;
      double ex = rt - G.direct[ireq]; if (ex < 0.0) ex = 0.0;
      cost += G.gamma * ex;
      t = tn; prev = x;
    }
    if (p == m.len) break;
    if (!gpu_step_orig(G, r, pr, Tt, tt, a, p, theta, t, cost, load, prev)) return false;
  }
  double back = G.D[(size_t)prev * G.V + G.depot];
  if (t + G.s + back > G.H + EPS) return false;
  cost += G.alpha * back;
  *delta = cost - m.base;
  return true;
}

// Host-side API (implemented in gpu.cu for CUDA, gpu_stub.cpp for the CPU emulation used for testing without a GPU).
bool gpu_init(const GpuInstance& hostInst);                                    // uploads the instance once
bool gpu_eval(int ncand, const GpuCand* meta, const int* stops, int nstops, const int* pp, const double* Tt,
              const double* Sc, const int* Sl, int nq, int ireq, double theta,
              double* out_delta, int* out_a, int* out_b);                      // best (min delta, then lowest a, then lowest b) per candidate route
void gpu_shutdown();

// Resident routes (static mode). Each vehicle's route lives on the GPU in a fixed slot (slot k = vehicle k, one extra slot for the
// local-search route with a request removed); the host re-sends only the routes that changed since the last call.
// Fields as in GpuCand / the per-position arrays above, at fixed capacity. feasible = 0 marks a route infeasible at the current theta.
struct GpuSlot {
  int len, k0, feasible, pad_; double base;
  int stops[GPU_LMAX]; int pp[GPU_LMAX]; double Tt[GPU_LMAX]; double Sc[GPU_LMAX + 1]; int Sl[GPU_LMAX + 1];
};
bool gpu_slots_init(int nslots);
// Writes upd[j] into slot upd_idx[j] (j < nupd), then evaluates request ireq against the routes in cand_slots[0 .. ncand).
bool gpu_eval_slots(int ncand, const int* cand_slots, int nupd, const int* upd_idx, const GpuSlot* upd, int ireq, double theta,
                    double* out_delta, int* out_a, int* out_b);

// Profiling counters, summed over all gpu_eval calls. Collected only after gpu_set_prof(true); the kernel is then followed by a
// cudaDeviceSynchronize so its time is separated from the copies. Results are unaffected.
//   positions: pickup positions tried, feas_a: of those, feasible, pairs: (a, b) pairs evaluated in phase 2.
struct GpuProf { double h2d = 0, kernel = 0, d2h = 0, realloc = 0; long long bytes_h2d = 0, positions = 0, feas_a = 0, pairs = 0; };
void gpu_set_prof(bool on);
GpuProf gpu_get_prof();
