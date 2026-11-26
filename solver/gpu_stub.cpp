// CPU emulation of gpu.cu that runs the SAME gpu_pickup() / gpu_pair() serially. Lets the host-side marshalling and the evaluation
// logic be tested without a GPU.
#include <vector>
#include "gpu.h"
static GpuInstance g;
bool gpu_init(const GpuInstance& h) { g = h; return true; }
bool gpu_eval(int nc, const GpuCand* meta, const int* stops, int, const int* pp, const double* Tt, const double* Sc, const int* Sl, int,
              int ireq, double theta, double* od, int* oa, int* ob) {
  for (int c = 0; c < nc; c++) {
    const GpuCand& m = meta[c]; double bd = 1e300; int ba = -1, bb = -1;
    const int* r = stops + m.off; const int* pr = pp + m.off; const double* Tr = Tt + m.off;
    for (int a = m.k0; a <= m.len; a++) {
      GpuPick q; if (!gpu_pickup(g, r, Tr, Sc + m.qoff, Sl + m.qoff, ireq, theta, a, &q)) continue;
      int f = gpu_scan(g, m, r, pr, Tr, q, ireq, theta, a);
      for (int b = a; b <= f; b++) {
        double dl; if (gpu_pair(g, m, r, pr, Tr, q, ireq, theta, a, b, &dl) && (ba < 0 || dl < bd)) { bd = dl; ba = a; bb = b; }
      }
    }
    od[c] = ba >= 0 ? bd : 1e300; oa[c] = ba; ob[c] = bb;
  }
  return true;
}
void gpu_shutdown() {}

static std::vector<GpuSlot> hs;
bool gpu_slots_init(int nslots) { hs.assign(nslots, GpuSlot()); for (auto& s : hs) { s.len = 0; s.k0 = 0; s.feasible = 0; s.base = 0; } return true; }
bool gpu_eval_slots(int nc, const int* cand, int nupd, const int* upd_idx, const GpuSlot* upd, int ireq, double theta, double* od, int* oa, int* ob) {
  for (int j = 0; j < nupd; j++) hs[upd_idx[j]] = upd[j];
  for (int c = 0; c < nc; c++) {
    const GpuSlot& S = hs[cand[c]]; double bd = 1e300; int ba = -1, bb = -1;
    if (S.feasible) {
      GpuCand m; m.off = 0; m.qoff = 0; m.len = S.len; m.k0 = S.k0; m.base = S.base;
      for (int a = m.k0; a <= m.len; a++) {
        GpuPick q; if (!gpu_pickup(g, S.stops, S.Tt, S.Sc, S.Sl, ireq, theta, a, &q)) continue;
        int f = gpu_scan(g, m, S.stops, S.pp, S.Tt, q, ireq, theta, a);
        for (int b = a; b <= f; b++) {
          double dl; if (gpu_pair(g, m, S.stops, S.pp, S.Tt, q, ireq, theta, a, b, &dl) && (ba < 0 || dl < bd)) { bd = dl; ba = a; bb = b; }
        }
      }
    }
    od[c] = ba >= 0 ? bd : 1e300; oa[c] = ba; ob[c] = bb;
  }
  return true;
}
void gpu_set_prof(bool) {}
GpuProf gpu_get_prof() { return GpuProf(); }
