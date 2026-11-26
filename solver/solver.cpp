// Dial-a-Ride heuristics (static insertion + relocate local search, online insertion with commitments).
// Build: g++ -O2 -fopenmp -std=c++17 solver.cpp -o solver
// Run:   ./solver instance.txt MODE [threads] [ls_passes]    MODE = static | static_nols | dynamic
// Stop ids: pickup of request i = i, delivery = n+i. Depot is a graph node. Distances come from a precomputed all-pairs matrix.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <sys/resource.h>
#include <vector>
#include <string>
#ifdef USE_CUDA
#include "gpu.h"
#endif
using namespace std;
// Optional profiling (DARP_PROF=1): wall-time split of the candidate search, one JSON line on stderr at exit. Off by default; never changes results.
static inline double now_s() { return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count(); }
struct Prof { bool on = false; double search = 0, prep = 0, dev = 0; long long calls = 0, dev_calls = 0, cands = 0, stops = 0, fallback = 0, upd = 0, bat_calls = 0, bat_entries = 0, bat_used = 0; };
static Prof g_prof;
// Resident memory from /proc (VmRSS now, VmHWM = peak of this process only; ru_maxrss would inherit the parent's peak).
static long proc_kb(const char* key) {
  ifstream f("/proc/self/status"); string line; size_t L = string(key).size();
  while (getline(f, line)) if (line.compare(0, L, key) == 0) return atol(line.c_str() + L + 1);
  return -1;
}

static const double EPS = 1e-6;
struct Inst {
  int V, n, m, Q, depot;
  double rho, s, H, alpha, beta, gamma, delta, wmax;
  vector<double> D;
  vector<int> po, dn;
  vector<double> e, l, rel, direct, maxride;
};
static inline double dd(const Inst& I, int a, int b) { return I.D[(size_t)a * I.V + b]; }
static inline int loc(const Inst& I, int v) { return v < I.n ? I.po[v] : I.dn[v - I.n]; }
static inline double win_e(const Inst& I, int v) { return v < I.n ? I.e[v] : I.e[v - I.n] + I.s + I.direct[v - I.n]; }
static inline double win_l(const Inst& I, int v) { return v < I.n ? I.l[v] : I.l[v - I.n] + I.s + I.maxride[v - I.n]; }

struct alignas(64) Scratch { vector<double> pick; long long steps = 0, evals = 0; };  // 64-byte aligned: no false sharing between threads
struct State { double t = 0; int prev = 0; int load = 0; double cost = 0; bool at_depot = true; };

// One forward step: serve stop v as early as allowed; departures are floored at theta (time "now" for online use).
static inline bool step(const Inst& I, State& s, int v, double theta, Scratch& S) {
  S.steps++;
  double dep = s.at_depot ? 0.0 : s.t + I.s;
  if (dep < theta) dep = theta;
  int x = loc(I, v);
  double d = dd(I, s.prev, x);
  s.cost += I.alpha * d;
  double t = max(dep + d, win_e(I, v));
  if (t > win_l(I, v) + EPS) return false;
  if (v < I.n) {
    if (++s.load > I.Q) return false;
    S.pick[v] = t;
    double w = t - I.e[v];
    s.cost += I.beta * w + I.delta * max(0.0, w - I.wmax);
  } else {
    --s.load;
    int i = v - I.n;
    double rt = t - (S.pick[i] + I.s);
    if (rt > I.maxride[i] + EPS) return false;
    s.cost += I.gamma * max(0.0, rt - I.direct[i]);
  }
  s.t = t; s.prev = x; s.at_depot = false;
  return true;
}
static inline bool finish(const Inst& I, State& s) {  // return to depot by H
  if (s.at_depot) return true;
  double back = dd(I, s.prev, I.depot);
  if (s.t + I.s + back > I.H + EPS) return false;
  s.cost += I.alpha * back;
  return true;
}
// State after the frozen prefix r[0..k0) using stored times T (fixes onboard pickup times).
static State prefix_state(const Inst& I, const vector<int>& r, const vector<double>& T, int k0, Scratch& S) {
  State s; s.prev = I.depot;
  for (int i = 0; i < k0; i++) {
    int v = r[i]; s.load += v < I.n ? 1 : -1;
    if (v < I.n) S.pick[v] = T[i];
  }
  if (k0 > 0) { s.t = T[k0 - 1]; s.prev = loc(I, r[k0 - 1]); s.at_depot = false; }
  return s;
}
// Cost of the suffix r[k0..] (including return leg); false if infeasible. Optionally writes times.
static bool suffix_cost(const Inst& I, const vector<int>& r, const vector<double>& T, int k0, double theta, Scratch& S, double& cost, vector<double>* tout) {
  S.evals++;
  State s = prefix_state(I, r, T, k0, S);
  if (tout) tout->resize(r.size());
  if (tout) for (int i = 0; i < k0; i++) (*tout)[i] = T[i];
  for (size_t i = k0; i < r.size(); i++) {
    if (!step(I, s, r[i], theta, S)) return false;
    if (tout) (*tout)[i] = s.t;
  }
  if (!finish(I, s)) return false;
  cost = s.cost; return true;
}
static const vector<double> NOT = {};
struct Cand { double delta = 1e300; int a = -1, b = -1; };

// Cheapest feasible insertion of request i into route r at pickup position a >= k0.
static Cand best_insert(const Inst& I, const vector<int>& r, const vector<double>& T, int k0, double theta, int i, double base, Scratch& S) {
  Cand best; int len = (int)r.size(), n = I.n;
  State sa = prefix_state(I, r, T, k0, S);
  for (int a = k0; a <= len; a++) {
    if (a > k0) { if (!step(I, sa, r[a - 1], theta, S)) break; }
    State sp = sa;
    if (!step(I, sp, i, theta, S)) continue;
    State sb = sp;
    for (int b = a; b <= len; b++) {
      if (b > a) { if (!step(I, sb, r[b - 1], theta, S)) break; if (sb.load > I.Q) break; }
      State sc = sb; S.evals++;
      bool ok = step(I, sc, n + i, theta, S);
      for (int x = b; ok && x < len; x++) ok = step(I, sc, r[x], theta, S);
      if (ok && finish(I, sc)) { double dl = sc.cost - base; if (dl < best.delta) { best.delta = dl; best.a = a; best.b = b; } }
    }
  }
  return best;
}
static vector<int> insert_pair(const vector<int>& r, int i, int n, int a, int b) {
  vector<int> o; o.reserve(r.size() + 2);
  for (int x = 0; x < a; x++) o.push_back(r[x]);
  o.push_back(i);
  for (int x = a; x < b; x++) o.push_back(r[x]);
  o.push_back(n + i);
  for (size_t x = b; x < r.size(); x++) o.push_back(r[x]);
  return o;
}

struct Sol {
  vector<vector<int>> R; vector<vector<double>> T; vector<char> served;
  vector<int> veh;                                                // veh[i] = vehicle whose route holds request i (kept up to date by apply)
};

static bool read_inst(const char* path, Inst& I) {
  ifstream f(path); if (!f) return false;
  f >> I.V >> I.n >> I.m >> I.Q >> I.depot >> I.rho >> I.s >> I.H >> I.alpha >> I.beta >> I.gamma >> I.delta >> I.wmax;
  I.D.resize((size_t)I.V * I.V);
  for (auto& x : I.D) f >> x;
  I.po.resize(I.n); I.dn.resize(I.n); I.e.resize(I.n); I.l.resize(I.n); I.rel.resize(I.n); I.direct.resize(I.n); I.maxride.resize(I.n);
  for (int i = 0; i < I.n; i++) {
    f >> I.po[i] >> I.dn[i] >> I.e[i] >> I.l[i] >> I.rel[i];
    I.direct[i] = dd(I, I.po[i], I.dn[i]); I.maxride[i] = I.rho * I.direct[i];
  }
  return true;
}

#ifdef USE_CUDA
static bool g_gpu = false; static int g_gpu_min = 32;
static bool g_resident = false; static vector<char> g_dirty;     // resident routes (static mode): g_dirty[k] = route k changed since last upload
static int g_batch = 128;                                         // max local-search requests per GPU call (DARP_GPU_BATCH; 1 = one at a time).
                                                                  // The batch size adapts between min(4, max) and max; results never depend on it.
#endif
struct Ctx { const Inst& I; int threads; vector<Scratch> S; Ctx(const Inst& i, int t) : I(i), threads(t), S(t) { for (auto& s : S) s.pick.assign(i.n, 0.0); } };

#ifdef USE_CUDA
// Per-position route states for the GPU (layout in gpu.h): appends len entries to Tt and len + 1 to Sc / Sl. Uses the same
// prefix_state()/step()/finish() calls as best_insert, so the values are bit-identical to what the CPU path computes.
// Returns false (caller rolls the arrays back) if the route is infeasible at theta; base = cost of the suffix from k0 incl. return.
static bool route_states(const Inst& I, const vector<int>& r, const vector<double>& T, int k0, double theta, Scratch& S, double& base,
                         vector<double>& Tt, vector<double>& Sc, vector<int>& Sl) {
  S.evals++;
  int len = (int)r.size(), ld = 0;
  State s = prefix_state(I, r, T, k0, S);
  for (int p = 0; p < k0; p++) { Sl.push_back(ld); Sc.push_back(0.0); Tt.push_back(T[p]); ld += r[p] < I.n ? 1 : -1; }
  Sl.push_back(s.load); Sc.push_back(s.cost);
  for (int p = k0; p < len; p++) {
    if (!step(I, s, r[p], theta, S)) return false;
    Tt.push_back(s.t); Sl.push_back(s.load); Sc.push_back(s.cost);
  }
  if (!finish(I, s)) return false;
  base = s.cost; return true;
}
// Fill one resident slot record for route r (static mode: k0 = 0). Caller guarantees r.size() + 2 <= GPU_LMAX.
static void fill_slot(const Inst& I, const vector<int>& r, const vector<double>& T, double theta, Scratch& S, GpuSlot& g) {
  static thread_local vector<double> Tt, Sc; static thread_local vector<int> Sl, where;
  Tt.clear(); Sc.clear(); Sl.clear(); double base = 0;
  int len = (int)r.size();
  g.len = len; g.k0 = 0; g.pad_ = 0;
  g.feasible = route_states(I, r, T, 0, theta, S, base, Tt, Sc, Sl) ? 1 : 0; g.base = base;
  if (!g.feasible) return;
  if ((int)where.size() < I.n) where.assign(I.n, -1);
  for (int p = 0; p < len; p++) {
    int v = r[p]; g.stops[p] = v; g.Tt[p] = Tt[p];
    if (v < I.n) { where[v] = p; g.pp[p] = -1; } else g.pp[p] = where[v - I.n];
  }
  for (int p = 0; p <= len; p++) { g.Sc[p] = Sc[p]; g.Sl[p] = Sl[p]; }
}
#endif
// Find best placement of request i over all vehicles; routes[k] may be overridden via 'over' (index, route) for local search.
struct Placement { double delta = 1e300; int veh = -1, a = -1, b = -1; };
static Placement best_over_vehicles_impl(Ctx& C, const Sol& sol, const vector<int>* frozen, double theta, int i, int override_k, const vector<int>* over_r, const vector<double>* over_T) {
  const Inst& I = C.I; int m = I.m;
  vector<int> cands; bool empty_taken = false;
  for (int k = 0; k < m; k++) {
    const vector<int>& r = (k == override_k) ? *over_r : sol.R[k];
    if (r.empty()) { if (empty_taken) continue; empty_taken = true; }
    cands.push_back(k);
  }
  vector<Cand> res(cands.size());
#ifdef USE_CUDA
  if (g_gpu && g_resident && frozen == nullptr && (int)cands.size() >= g_gpu_min) {   // static mode: routes stay on the GPU, only changed ones are sent
    double tp0 = g_prof.on ? now_s() : 0.0;
    bool too_long = false;
    for (int k : cands) { const vector<int>& r = (k == override_k) ? *over_r : sol.R[k]; if ((int)r.size() + 2 > GPU_LMAX) { too_long = true; break; } }
    if (!too_long) {
      static thread_local vector<int> cslot, creq, uidx; static thread_local vector<GpuSlot> upd;
      cslot.clear(); uidx.clear(); upd.clear(); creq.assign(cands.size(), i);
      for (int k : cands) {
        if (k == override_k) { upd.emplace_back(); uidx.push_back(m); fill_slot(I, *over_r, *over_T, theta, C.S[0], upd.back()); cslot.push_back(m); }
        else {
          if (g_dirty[k]) { upd.emplace_back(); uidx.push_back(k); fill_slot(I, sol.R[k], sol.T[k], theta, C.S[0], upd.back()); g_dirty[k] = 0; }
          cslot.push_back(k);
        }
      }
      if (g_prof.on) g_prof.prep += now_s() - tp0;
      vector<double> od(cands.size()); vector<int> oa(cands.size()), ob(cands.size());
      double td0 = g_prof.on ? now_s() : 0.0;
      if (!gpu_eval_slots((int)cslot.size(), cslot.data(), creq.data(), (int)upd.size(), uidx.data(), upd.data(), theta, od.data(), oa.data(), ob.data())) { fprintf(stderr, "gpu_eval_slots failed\n"); exit(2); }
      if (g_prof.on) { g_prof.dev += now_s() - td0; g_prof.dev_calls++; g_prof.cands += (long long)cslot.size(); g_prof.upd += (long long)upd.size(); }
      Placement p;
      for (size_t c = 0; c < cands.size(); c++) if (oa[c] >= 0 && od[c] < p.delta) { p.delta = od[c]; p.veh = cands[c]; p.a = oa[c]; p.b = ob[c]; }
      return p;
    }
    g_prof.fallback++;                                              // a route was too long for the GPU: this call runs on the CPU
  } else if (g_gpu && (int)cands.size() >= g_gpu_min) {                  // offload: every (vehicle, pickup position, delivery position) candidate is evaluated on the GPU
    static thread_local vector<int> where;
    double tp0 = g_prof.on ? now_s() : 0.0;
    vector<GpuCand> meta; vector<int> stops, pp, cmap, Sl; vector<double> Tf, Sc; bool too_long = false; Scratch& S0 = C.S[0];
    if ((int)where.size() < I.n) where.assign(I.n, -1);
    for (size_t c = 0; c < cands.size() && !too_long; c++) {
      int k = cands[c]; const vector<int>& r = (k == override_k) ? *over_r : sol.R[k]; const vector<double>& T = (k == override_k) ? *over_T : sol.T[k];
      int k0 = frozen ? (*frozen)[k] : 0, len = (int)r.size();
      if (len + 2 > GPU_LMAX) { too_long = true; break; }
      size_t s0 = Tf.size(), q0 = Sc.size(); double base = 0;
      if (!route_states(I, r, T, k0, theta, S0, base, Tf, Sc, Sl)) { Tf.resize(s0); Sc.resize(q0); Sl.resize(q0); res[c].delta = 1e300; continue; }
      GpuCand g; g.off = (int)stops.size(); g.qoff = (int)q0; g.len = len; g.k0 = k0; g.base = base;
      for (int p = 0; p < len; p++) {
        int v = r[p]; stops.push_back(v);
        if (v < I.n) { where[v] = p; pp.push_back(-1); } else pp.push_back(where[v - I.n]);
      }
      meta.push_back(g); cmap.push_back((int)c);
    }
    if (g_prof.on) g_prof.prep += now_s() - tp0;
    if (!too_long) {
      vector<double> od(meta.size()); vector<int> oa(meta.size()), ob(meta.size());
      double td0 = g_prof.on ? now_s() : 0.0;
      if (!meta.empty() && !gpu_eval((int)meta.size(), meta.data(), stops.data(), (int)stops.size(), pp.data(), Tf.data(), Sc.data(), Sl.data(), (int)Sc.size(),
                                     i, theta, od.data(), oa.data(), ob.data())) { fprintf(stderr, "gpu_eval failed\n"); exit(2); }
      if (g_prof.on) { g_prof.dev += now_s() - td0; g_prof.dev_calls++; g_prof.cands += (long long)meta.size(); g_prof.stops += (long long)stops.size(); }
      for (size_t j = 0; j < meta.size(); j++) { res[cmap[j]].delta = od[j]; res[cmap[j]].a = oa[j]; res[cmap[j]].b = ob[j]; }
      Placement p;
      for (size_t c = 0; c < cands.size(); c++) if (res[c].a >= 0 && res[c].delta < p.delta) { p.delta = res[c].delta; p.veh = cands[c]; p.a = res[c].a; p.b = res[c].b; }
      return p;
    }
    g_prof.fallback++;                                              // a route was too long for the GPU: this call runs on the CPU
    for (auto& x : res) x = Cand();
  }
#endif
#pragma omp parallel for schedule(dynamic) num_threads(C.threads)
  for (int c = 0; c < (int)cands.size(); c++) {
    int k = cands[c]; Scratch& S = C.S[omp_get_thread_num()];
    const vector<int>& r = (k == override_k) ? *over_r : sol.R[k];
    const vector<double>& T = (k == override_k) ? *over_T : sol.T[k];
    int k0 = frozen ? (*frozen)[k] : 0;
    double base = 0;
    if (!suffix_cost(I, r, T, k0, theta, S, base, nullptr)) { res[c].delta = 1e300; continue; }
    res[c] = best_insert(I, r, T, k0, theta, i, base, S);
  }
  Placement p;
  for (size_t c = 0; c < cands.size(); c++) if (res[c].a >= 0 && res[c].delta < p.delta) { p.delta = res[c].delta; p.veh = cands[c]; p.a = res[c].a; p.b = res[c].b; }
  return p;
}
static Placement best_over_vehicles(Ctx& C, const Sol& sol, const vector<int>* frozen, double theta, int i, int override_k, const vector<int>* over_r, const vector<double>* over_T) {
  if (!g_prof.on) return best_over_vehicles_impl(C, sol, frozen, theta, i, override_k, over_r, over_T);
  double t = now_s(); Placement p = best_over_vehicles_impl(C, sol, frozen, theta, i, override_k, over_r, over_T);
  g_prof.search += now_s() - t; g_prof.calls++; return p;
}
static void apply(Ctx& C, Sol& sol, int k, const vector<int>& nr, int k0, double theta) {
  vector<double> T; double cost;
  suffix_cost(C.I, nr, sol.T[k], k0, theta, C.S[0], cost, &T);
  sol.R[k] = nr; sol.T[k] = T;
  for (int v : nr) if (v < C.I.n) sol.veh[v] = k;
#ifdef USE_CUDA
  if (!g_dirty.empty()) g_dirty[k] = 1;
#endif
}
static double route_cost(const Inst& I, const vector<int>& r, Scratch& S, bool& ok) {
  vector<double> none; double c = 0; ok = suffix_cost(I, r, none, 0, 0.0, S, c, nullptr); return c;
}

// One local-search step for request i, split into "prepare" (what to evaluate) and "decide/apply" so the sequential loop and the
// batched GPU loop run exactly the same logic.
struct LsEntry { int i = -1; bool repair = false; int ks = -1; vector<int> rem; vector<double> Trem; double saving = 0; };
static bool ls_prepare(Ctx& C, const Sol& sol, int i, LsEntry& e) {   // false: nothing to evaluate for i (removal leaves an infeasible route)
  const Inst& I = C.I; int n = I.n;
  e.i = i; e.repair = !sol.served[i]; e.ks = -1; e.rem.clear(); e.Trem.clear(); e.saving = 0;
  if (e.repair) return true;                                  // repair: try to serve a rejected request
  int ks = sol.veh[i];
  e.ks = ks;
  for (int v : sol.R[ks]) if (v != i && v != n + i) e.rem.push_back(v);
  bool ok1, ok2; double c_old = route_cost(I, sol.R[ks], C.S[0], ok1), c_rem = route_cost(I, e.rem, C.S[0], ok2);
  if (!ok2) return false;                                     // removal leaves an infeasible route: treated as infinite cost
  e.saving = c_old - c_rem;
  double dummy; suffix_cost(I, e.rem, {}, 0, 0.0, C.S[0], dummy, &e.Trem);
  return true;
}
static Placement ls_eval(Ctx& C, const Sol& sol, const LsEntry& e) {
  return e.repair ? best_over_vehicles(C, sol, nullptr, 0.0, e.i, -1, nullptr, nullptr) : best_over_vehicles(C, sol, nullptr, 0.0, e.i, e.ks, &e.rem, &e.Trem);
}
static bool ls_apply(Ctx& C, Sol& sol, const LsEntry& e, const Placement& p) {   // true if the solution changed
  int n = C.I.n, i = e.i;
  if (e.repair) {
    if (p.veh < 0) return false;
    apply(C, sol, p.veh, insert_pair(sol.R[p.veh], i, n, p.a, p.b), 0, 0.0); sol.served[i] = 1; return true;
  }
  if (!(p.veh >= 0 && p.delta < e.saving - 1e-7)) return false;
  const vector<int>& base = (p.veh == e.ks) ? e.rem : sol.R[p.veh];
  vector<int> nr = insert_pair(base, i, n, p.a, p.b);
  if (p.veh != e.ks) apply(C, sol, e.ks, e.rem, 0, 0.0);
  apply(C, sol, p.veh, nr, 0, 0.0);
  return true;
}

#ifdef USE_CUDA
// Evaluate every entry of a local-search batch against the CURRENT solution in one GPU call. Entry b uses its own override slot m + b
// for its route with request i removed. Returns false (nothing evaluated, no state changed) if the batch cannot go to the GPU;
// the caller then evaluates the entries one at a time. Same candidate lists and same selection rule as best_over_vehicles_impl.
static bool ls_eval_batch(Ctx& C, const Sol& sol, const vector<LsEntry>& B, vector<Placement>& P) {
  const Inst& I = C.I; int m = I.m;
  double t0 = g_prof.on ? now_s() : 0.0;
  for (int k = 0; k < m; k++) if ((int)sol.R[k].size() + 2 > GPU_LMAX) return false;
  static thread_local vector<int> bslot, breq, bent, bveh, uidx; static thread_local vector<GpuSlot> upd;
  bslot.clear(); breq.clear(); bent.clear(); bveh.clear(); uidx.clear(); upd.clear();
  for (size_t b = 0; b < B.size(); b++) {                     // candidate list per entry, exactly as best_over_vehicles_impl builds it
    const LsEntry& e = B[b]; int ov = e.repair ? -1 : e.ks; bool empty_taken = false; size_t first = bslot.size();
    for (int k = 0; k < m; k++) {
      const vector<int>& r = (k == ov) ? e.rem : sol.R[k];
      if (r.empty()) { if (empty_taken) continue; empty_taken = true; }
      bslot.push_back(k == ov ? m + (int)b : k); breq.push_back(e.i); bent.push_back((int)b); bveh.push_back(k);
    }
    if ((int)(bslot.size() - first) < g_gpu_min) return false;
  }
  for (int k = 0; k < m; k++) if (g_dirty[k]) { upd.emplace_back(); uidx.push_back(k); fill_slot(I, sol.R[k], sol.T[k], 0.0, C.S[0], upd.back()); g_dirty[k] = 0; }
  for (size_t b = 0; b < B.size(); b++) if (!B[b].repair) { upd.emplace_back(); uidx.push_back(m + (int)b); fill_slot(I, B[b].rem, B[b].Trem, 0.0, C.S[0], upd.back()); }
  if (g_prof.on) g_prof.prep += now_s() - t0;
  size_t nb = bslot.size(); vector<double> od(nb); vector<int> oa(nb), ob(nb);
  double td0 = g_prof.on ? now_s() : 0.0;
  if (!gpu_eval_slots((int)nb, bslot.data(), breq.data(), (int)upd.size(), uidx.data(), upd.data(), 0.0, od.data(), oa.data(), ob.data())) { fprintf(stderr, "gpu_eval_slots failed\n"); exit(2); }
  P.assign(B.size(), Placement());
  for (size_t c = 0; c < nb; c++) { Placement& p = P[bent[c]]; if (oa[c] >= 0 && od[c] < p.delta) { p.delta = od[c]; p.veh = bveh[c]; p.a = oa[c]; p.b = ob[c]; } }
  if (g_prof.on) { double t1 = now_s(); g_prof.dev += t1 - td0; g_prof.search += t1 - t0; g_prof.calls += (long long)B.size(); g_prof.dev_calls++;
                   g_prof.cands += (long long)nb; g_prof.upd += (long long)upd.size(); g_prof.bat_calls++; g_prof.bat_entries += (long long)B.size(); }
  return true;
}
#endif

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: solver inst MODE [threads] [ls_passes]\n"); return 1; }
  Inst I; if (!read_inst(argv[1], I)) { fprintf(stderr, "cannot read instance\n"); return 1; }
  string mode = argv[2]; int threads = argc > 3 ? atoi(argv[3]) : 1; int max_passes = argc > 4 ? atoi(argv[4]) : 30;
  long rss_load = proc_kb("VmRSS");
  Ctx C(I, threads); int n = I.n, m = I.m;
  g_prof.on = getenv("DARP_PROF") && atoi(getenv("DARP_PROF"));
#ifdef USE_CUDA
  if (getenv("DARP_GPU") && atoi(getenv("DARP_GPU"))) {            // DARP_GPU=1 enables the GPU path; DARP_GPU_MIN = fewest candidate routes worth a GPU call (default 32)
    GpuInstance G{I.V, I.n, I.Q, I.depot, I.s, I.H, I.alpha, I.beta, I.gamma, I.delta, I.wmax, I.D.data(), I.e.data(), I.l.data(), I.direct.data(), I.maxride.data(), I.po.data(), I.dn.data()};
    if (!gpu_init(G)) { fprintf(stderr, "gpu_init failed\n"); return 2; }
    g_gpu = true; if (getenv("DARP_GPU_MIN")) g_gpu_min = atoi(getenv("DARP_GPU_MIN"));
    gpu_set_prof(g_prof.on);
    // DARP_GPU_RESIDENT=0 turns off resident routes (every call then re-sends all candidate routes, as the dynamic mode always does).
    g_resident = !(getenv("DARP_GPU_RESIDENT") && atoi(getenv("DARP_GPU_RESIDENT")) == 0);
    if (getenv("DARP_GPU_BATCH")) g_batch = max(1, atoi(getenv("DARP_GPU_BATCH")));
    if (g_resident) {
      if (!gpu_slots_init(m + max(1, g_batch))) { fprintf(stderr, "gpu_slots_init failed\n"); return 2; }
      g_dirty.assign(m, 1);
    }
  }
#endif
  Sol sol; sol.R.assign(m, {}); sol.T.assign(m, {}); sol.served.assign(n, 0); sol.veh.assign(n, -1);
  vector<int> order(n); iota(order.begin(), order.end(), 0);
  long long ls_moves = 0, passes = 0, violations = 0; vector<double> decision_theta(n, -1);
  auto t0 = chrono::steady_clock::now();
  if (mode == "static" || mode == "static_nols") {
    stable_sort(order.begin(), order.end(), [&](int a, int b) { return I.e[a] < I.e[b]; });
    for (int i : order) {
      Placement p = best_over_vehicles(C, sol, nullptr, 0.0, i, -1, nullptr, nullptr);
      if (p.veh >= 0) { apply(C, sol, p.veh, insert_pair(sol.R[p.veh], i, n, p.a, p.b), 0, 0.0); sol.served[i] = 1; }
    }
    if (mode == "static") {
      bool changed = true;
      while (changed && passes < max_passes) {
        changed = false; passes++;
        bool batched = false;
#ifdef USE_CUDA
        batched = g_gpu && g_resident && g_batch > 1;
        if (batched) {
          // Speculative batches: evaluate the next g_batch requests against the current solution in one GPU call, then decide them in
          // order. Every entry before the first accepted move saw exactly the solution the sequential loop would have shown it; at the
          // first accepted move the rest of the batch is discarded and the next batch starts right after it. Same results as below.
          vector<LsEntry> B; vector<Placement> P; int i = 0;
          const int kmin = min(4, g_batch); int K = min(8, g_batch);       // halve after an accepted move, double after a batch with none
          while (i < n) {
            B.clear(); int j = i;
            while (j < n && (int)B.size() < K) { B.emplace_back(); if (!ls_prepare(C, sol, j, B.back())) B.pop_back(); j++; }
            bool on_gpu = !B.empty() && ls_eval_batch(C, sol, B, P);
            int next = j; bool accepted = false;
            for (size_t b = 0; b < B.size(); b++) {
              Placement p = on_gpu ? P[b] : ls_eval(C, sol, B[b]);
              if (g_prof.on && on_gpu) g_prof.bat_used++;
              if (ls_apply(C, sol, B[b], p)) { changed = true; ls_moves++; next = B[b].i + 1; accepted = true; break; }
            }
            K = accepted ? max(kmin, K / 2) : min(g_batch, K * 2);
            i = next;
          }
        }
#endif
        if (!batched) {
          LsEntry e;
          for (int i = 0; i < n; i++) {
            if (!ls_prepare(C, sol, i, e)) continue;
            if (ls_apply(C, sol, e, ls_eval(C, sol, e))) { changed = true; ls_moves++; }
          }
        }
      }
    }
  } else if (mode == "dynamic") {
    stable_sort(order.begin(), order.end(), [&](int a, int b) { return I.rel[a] != I.rel[b] ? I.rel[a] < I.rel[b] : I.e[a] < I.e[b]; });
    for (int i : order) {
      double theta = I.rel[i]; decision_theta[i] = theta;
      vector<int> frozen(m, 0);
      for (int k = 0; k < m; k++) {
        int f = 0, len = (int)sol.R[k].size();
        while (f < len && sol.T[k][f] <= theta + 1e-12) f++;
        if (f < len) f++;                        // the next stop is already pending: one commitment per vehicle
        frozen[k] = f;
      }
      Placement p = best_over_vehicles(C, sol, &frozen, theta, i, -1, nullptr, nullptr);
      if (p.veh >= 0) {
        int k = p.veh; vector<int> nr = insert_pair(sol.R[k], i, n, p.a, p.b);
        vector<int> pre(sol.R[k].begin(), sol.R[k].begin() + frozen[k]); vector<double> preT(sol.T[k].begin(), sol.T[k].begin() + frozen[k]);
        apply(C, sol, k, nr, frozen[k], theta); sol.served[i] = 1;
        for (int x = 0; x < frozen[k]; x++) if (sol.R[k][x] != pre[x] || fabs(sol.T[k][x] - preT[x]) > 1e-9) violations++;
      }
    }
  } else { fprintf(stderr, "bad mode\n"); return 1; }
  double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
  long long steps = 0, evals = 0; for (auto& s : C.S) { steps += s.steps; evals += s.evals; }
  long rss_peak = proc_kb("VmHWM");
  if (g_prof.on) {
    double h2d = 0, kern = 0, d2h = 0, realloc_s = 0; long long bytes = 0, positions = 0, feas_a = 0, pairs = 0;
#ifdef USE_CUDA
    if (g_gpu) { GpuProf gp = gpu_get_prof(); h2d = gp.h2d; kern = gp.kernel; d2h = gp.d2h; realloc_s = gp.realloc; bytes = gp.bytes_h2d; positions = gp.positions; feas_a = gp.feas_a; pairs = gp.pairs; }
#endif
    fprintf(stderr, "{\"prof\":{\"runtime\":%.6f,\"search\":%.6f,\"calls\":%lld,\"prep\":%.6f,\"dev\":%.6f,\"dev_calls\":%lld,\"cands\":%lld,\"stops\":%lld,\"fallback\":%lld,\"upd\":%lld,\"bat_calls\":%lld,\"bat_entries\":%lld,\"bat_used\":%lld,"
                    "\"h2d\":%.6f,\"kernel\":%.6f,\"d2h\":%.6f,\"realloc\":%.6f,\"bytes_h2d\":%lld,\"positions\":%lld,\"feas_a\":%lld,\"pairs\":%lld,\"steps\":%lld,\"passes\":%lld,\"m\":%d}}\n",
            secs, g_prof.search, g_prof.calls, g_prof.prep, g_prof.dev, g_prof.dev_calls, g_prof.cands, g_prof.stops, g_prof.fallback, g_prof.upd, g_prof.bat_calls, g_prof.bat_entries, g_prof.bat_used,
            h2d, kern, d2h, realloc_s, bytes, positions, feas_a, pairs, steps, passes, m);
  }
  printf("{\"mode\":\"%s\",\"threads\":%d,\"runtime\":%.6f,\"steps\":%lld,\"evals\":%lld,\"maxrss_kb\":%ld,\"rss_load_kb\":%ld,\"ls_moves\":%lld,\"passes\":%lld,\"commit_violations\":%lld,\"served\":[", mode.c_str(), threads, secs, steps, evals, rss_peak, rss_load, ls_moves, passes, violations);
  for (int i = 0; i < n; i++) printf("%s%d", i ? "," : "", (int)sol.served[i]);
  printf("],\"routes\":[");
  for (int k = 0; k < m; k++) { printf("%s[", k ? "," : ""); for (size_t j = 0; j < sol.R[k].size(); j++) printf("%s%d", j ? "," : "", sol.R[k][j]); printf("]"); }
  printf("],\"times\":[");
  for (int k = 0; k < m; k++) { printf("%s[", k ? "," : ""); for (size_t j = 0; j < sol.T[k].size(); j++) printf("%s%.9g", j ? "," : "", sol.T[k][j]); printf("]"); }
  printf("]}\n");
  return 0;
}
