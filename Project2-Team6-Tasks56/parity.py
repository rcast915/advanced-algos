"""Parity + timing: compares solver binaries on generated instances. Usage: python3 parity.py CPU_BIN OTHER_BIN [env KEY=VAL ...] ; set N list via PARITY_N."""
import sys, os, json, subprocess, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen, validate
def run(binary, I, mode, env, threads=1):
    p = tempfile.mktemp(suffix='.txt'); gen.write_instance(I, p); e = dict(os.environ); e.update(env)
    t0 = time.time(); out = subprocess.run([binary, p, mode, str(threads), '30'], capture_output=True, text=True, check=True, env=e).stdout; dt = time.time() - t0
    os.remove(p); return json.loads(out), dt
a_bin, b_bin = sys.argv[1], sys.argv[2]; env = dict(kv.split('=') for kv in sys.argv[3:])
ns = [int(x) for x in os.environ.get('PARITY_N', '100,500').split(',')]; bad = 0
for fam in ('F1', 'F2', 'F5', 'F6'):
    for n in ns:
        I = gen.make_instance(fam, n, 1)
        for mode in ('static', 'dynamic'):
            sa, ta = run(a_bin, I, mode, {}); sb, tb = run(b_bin, I, mode, env)
            ma, mb = validate.metrics(I, sa), validate.metrics(I, sb); ok, why = validate.validate(I, sb, mode)
            same = sa['R'] == sb['R'] if 'R' in sa else (ma['served_pct'] == mb['served_pct'] and abs(ma['Z'] - mb['Z']) < 1e-9)
            bad += (not same) or (not ok)
            print(f'{fam} n={n:5d} {mode:8s} served {ma["served_pct"]:.1f}/{mb["served_pct"]:.1f}  Z {ma["Z"]:.6f}/{mb["Z"]:.6f}  valid={ok}  identical={same}  time {ta:.2f}s -> {tb:.2f}s')
print('PARITY OK' if bad == 0 else f'PARITY FAILED ({bad})')
