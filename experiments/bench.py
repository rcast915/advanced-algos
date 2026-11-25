"""Timing on the H100 node: python3 experiments/bench.py   (needs bin/solver_cpu and bin/solver_gpu from `make cpu gpu`)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen, subprocess, tempfile, time, json
from paths import BIN
def t(binary, I, env, threads):
    p = tempfile.mktemp(suffix='.txt'); gen.write_instance(I, p); e = dict(os.environ); e.update(env); t0 = time.time()
    out = json.loads(subprocess.run([binary, p, 'static', str(threads), '30'], capture_output=True, text=True, check=True, env=e).stdout); os.remove(p); return time.time() - t0, out
print('n, cpu_1thread_s, cpu_2thread_s, gpu_s (wall time incl. startup and CUDA init), speedup_vs_1thread')
for n in (500, 2000, 5000, 10000):
    I = gen.make_instance('F1', n, 1)
    c1, _ = t(str(BIN / 'solver_cpu'), I, {}, 1); c2, _ = t(str(BIN / 'solver_cpu'), I, {}, 2); g, _ = t(str(BIN / 'solver_gpu'), I, {'DARP_GPU': '1'}, 1)
    print(f'{n}, {c1:.2f}, {c2:.2f}, {g:.2f}, {c1 / g:.2f}x', flush=True)
