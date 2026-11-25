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
using namespace std;
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

struct Ctx { const Inst& I; int threads; vector<Scratch> S; Ctx(const Inst& i, int t) : I(i), threads(t), S(t) { for (auto& s : S) s.pick.assign(i.n, 0.0); } };

// Find best placement of request i over all vehicles; routes[k] may be overridden via 'over' (index, route) for local search.
struct Placement { double delta = 1e300; int veh = -1, a = -1, b = -1; };
static Placement best_over_vehicles(Ctx& C, const Sol& sol, const vector<int>* frozen, double theta, int i, int override_k, const vector<int>* over_r, const vector<double>* over_T) {
  const Inst& I = C.I; int m = I.m;
  vector<int> cands; bool empty_taken = false;
  for (int k = 0; k < m; k++) {
    const vector<int>& r = (k == override_k) ? *over_r : sol.R[k];
    if (r.empty()) { if (empty_taken) continue; empty_taken = true; }
    cands.push_back(k);
  }
  vector<Cand> res(cands.size());
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
static void apply(Ctx& C, Sol& sol, int k, const vector<int>& nr, int k0, double theta) {
  vector<double> T; double cost;
  suffix_cost(C.I, nr, sol.T[k], k0, theta, C.S[0], cost, &T);
  sol.R[k] = nr; sol.T[k] = T;
}
static double route_cost(const Inst& I, const vector<int>& r, Scratch& S, bool& ok) {
  vector<double> none; double c = 0; ok = suffix_cost(I, r, none, 0, 0.0, S, c, nullptr); return c;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: solver inst MODE [threads] [ls_passes]\n"); return 1; }
  Inst I; if (!read_inst(argv[1], I)) { fprintf(stderr, "cannot read instance\n"); return 1; }
  string mode = argv[2]; int threads = argc > 3 ? atoi(argv[3]) : 1; int max_passes = argc > 4 ? atoi(argv[4]) : 30;
  long rss_load = proc_kb("VmRSS");
  Ctx C(I, threads); int n = I.n, m = I.m;
  Sol sol; sol.R.assign(m, {}); sol.T.assign(m, {}); sol.served.assign(n, 0);
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
        for (int i = 0; i < n; i++) {
          if (!sol.served[i]) {  // repair: try to serve a rejected request
            Placement p = best_over_vehicles(C, sol, nullptr, 0.0, i, -1, nullptr, nullptr);
            if (p.veh >= 0) { apply(C, sol, p.veh, insert_pair(sol.R[p.veh], i, n, p.a, p.b), 0, 0.0); sol.served[i] = 1; changed = true; ls_moves++; }
            continue;
          }
          int ks = -1; for (int k = 0; k < m && ks < 0; k++) for (int v : sol.R[k]) if (v == i) { ks = k; break; }
          vector<int> rem; for (int v : sol.R[ks]) if (v != i && v != n + i) rem.push_back(v);
          bool ok1, ok2; double c_old = route_cost(I, sol.R[ks], C.S[0], ok1), c_rem = route_cost(I, rem, C.S[0], ok2);
          if (!ok2) continue;                       // removal leaves an infeasible route: treated as infinite cost
          double saving = c_old - c_rem;
          vector<double> Trem; double dummy; suffix_cost(I, rem, {}, 0, 0.0, C.S[0], dummy, &Trem);
          Placement p = best_over_vehicles(C, sol, nullptr, 0.0, i, ks, &rem, &Trem);
          if (p.veh >= 0 && p.delta < saving - 1e-7) {
            const vector<int>& base = (p.veh == ks) ? rem : sol.R[p.veh];
            vector<int> nr = insert_pair(base, i, n, p.a, p.b);
            if (p.veh != ks) apply(C, sol, ks, rem, 0, 0.0);
            apply(C, sol, p.veh, nr, 0, 0.0); changed = true; ls_moves++;
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
  printf("{\"mode\":\"%s\",\"threads\":%d,\"runtime\":%.6f,\"steps\":%lld,\"evals\":%lld,\"maxrss_kb\":%ld,\"rss_load_kb\":%ld,\"ls_moves\":%lld,\"passes\":%lld,\"commit_violations\":%lld,\"served\":[", mode.c_str(), threads, secs, steps, evals, rss_peak, rss_load, ls_moves, passes, violations);
  for (int i = 0; i < n; i++) printf("%s%d", i ? "," : "", (int)sol.served[i]);
  printf("],\"routes\":[");
  for (int k = 0; k < m; k++) { printf("%s[", k ? "," : ""); for (size_t j = 0; j < sol.R[k].size(); j++) printf("%s%d", j ? "," : "", sol.R[k][j]); printf("]"); }
  printf("],\"times\":[");
  for (int k = 0; k < m; k++) { printf("%s[", k ? "," : ""); for (size_t j = 0; j < sol.T[k].size(); j++) printf("%s%.9g", j ? "," : "", sol.T[k][j]); printf("]"); }
  printf("]}\n");
  return 0;
}
