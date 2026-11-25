// CPU emulation of gpu.cu that runs the SAME gpu_eval_one() serially. Lets the host-side marshalling and the evaluation logic be tested without a GPU.
#include <vector>
#include "gpu.h"
static GpuInstance g;
bool gpu_init(const GpuInstance& h) { g = h; return true; }
bool gpu_eval(int nc, const GpuCand* meta, const int* stops, int, const int* pp, const double* T, int ireq, double theta, double* od, int* oa, int* ob) {
  for (int c = 0; c < nc; c++) {
    const GpuCand& m = meta[c]; double bd = 1e300; int ba = -1, bb = -1;
    for (int a = m.k0; a <= m.len; a++) for (int b = a; b <= m.len; b++) {
      double dl; if (gpu_eval_one(g, m, stops, pp, T, ireq, theta, a, b, &dl) && (ba < 0 || dl < bd)) { bd = dl; ba = a; bb = b; }
    }
    od[c] = ba >= 0 ? bd : 1e300; oa[c] = ba; ob[c] = bb;
  }
  return true;
}
void gpu_shutdown() {}
