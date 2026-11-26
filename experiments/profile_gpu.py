"""Where does the GPU path spend its time?  python3 experiments/profile_gpu.py [n ...]   (default n = 2000 5000)

Runs bin/solver_cpu and bin/solver_gpu with DARP_PROF=1 on the F1 instance (seed 1, static, 1 CPU thread) and prints the time split.
Needs `make cpu gpu`. Profiling adds a cudaDeviceSynchronize after each kernel, so GPU totals are slightly above an unprofiled run.
"""
import os, sys, json, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen
from paths import BIN


def run(binary, path, env):
    e = dict(os.environ, DARP_PROF='1', **env)
    r = subprocess.run([str(BIN / binary), path, 'static', '1', '30'], capture_output=True, text=True, check=True, env=e)
    sol = json.loads(r.stdout)
    prof = json.loads([l for l in r.stderr.splitlines() if l.startswith('{"prof"')][-1])['prof']
    return sol, prof


ns = [int(x) for x in sys.argv[1:]] or [2000, 5000]
for n in ns:
    I = gen.make_instance('F1', n, 1)
    d = tempfile.mkdtemp(prefix='darp_prof_'); p = os.path.join(d, 'inst.txt'); gen.write_instance(I, p)
    sc, pc = run('solver_cpu', p, {})
    sg, pg = run('solver_gpu', p, {'DARP_GPU': '1'})
    os.remove(p); os.rmdir(d)
    same = sc['routes'] == sg['routes'] and sc['served'] == sg['served']
    calls = max(pg['calls'], 1); dc = max(pg['dev_calls'], 1)
    rest_g = pg['runtime'] - pg['search']; host_other = pg['search'] - pg['prep'] - pg['dev']
    print(f'\n=== n = {n}  (m = {pg["m"]} vehicles, local-search passes = {pg["passes"]}, identical routes CPU vs GPU: {same})')
    print(f'CPU  total {pc["runtime"]:8.2f} s | candidate search {pc["search"]:8.2f} s over {pc["calls"]} calls = {1e3 * pc["search"] / max(pc["calls"], 1):.3f} ms/call | rest {pc["runtime"] - pc["search"]:.2f} s')
    print(f'GPU  total {pg["runtime"]:8.2f} s | candidate search {pg["search"]:8.2f} s over {pg["calls"]} calls = {1e3 * pg["search"] / calls:.3f} ms/call | rest {rest_g:.2f} s')
    sent = f'{pg["upd"] / dc:.2f} changed routes sent (resident)' if pg.get('upd') else f'all routes sent, {pg["stops"] / dc:.0f} stops'
    print(f'  per GPU call ({pg["dev_calls"]} calls, {pg["fallback"]} fell back to CPU; avg {pg["cands"] / dc:.0f} candidate routes, {sent}, {pg["bytes_h2d"] / dc / 1024:.1f} KiB uploaded):')
    for k, lab in (('prep', 'host prep (suffix_cost + marshalling)'), ('realloc', 'device realloc'), ('h2d', 'host->device copies'),
                   ('kernel', 'kernel (launch + run)'), ('d2h', 'device->host copy')):
        print(f'    {lab:40s} {pg[k]:8.2f} s  {100 * pg[k] / max(pg["search"], 1e-12):5.1f} %  {1e6 * pg[k] / dc:8.1f} us/call')
    print(f'    {"other host work in search":40s} {host_other:8.2f} s  {100 * host_other / max(pg["search"], 1e-12):5.1f} %')
    if pg['positions']:
        print(f'  kernel work: {pg["positions"]:,} pickup positions tried, {pg["feas_a"]:,} feasible ({100 * pg["feas_a"] / pg["positions"]:.1f} %), '
              f'{pg["pairs"]:,} (a,b) pairs evaluated in phase 2')
