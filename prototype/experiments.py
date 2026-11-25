import os, sys, csv, time, platform, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'task3')   # override with DARP_RESULTS
OUT = os.environ.get('DARP_RESULTS', OUT); os.makedirs(OUT, exist_ok=True)
import darp, exact
import scipy, numpy
SMALL = {4: 30, 6: 30, 8: 20, 10: 10}; TL = 120
LARGE = [10, 20, 40, 60, 80, 100]; LSEEDS = 10
def small():
    rows = []
    for n, reps in SMALL.items():
        for seed in range(1, reps+1):
            I = darp.make_instance(n, seed)
            t = time.time(); c = darp.solve_heuristic(I, ls=False); tc = time.time()-t
            t = time.time(); h = darp.solve_heuristic(I, ls=True); th = time.time()-t
            e = exact.solve_exact(I, TL)
            ev = darp.validate(I, e['routes'], e['T'])[0] if e['Z'] is not None else None
            hv = darp.validate(I, h['routes'], h['T'])[0] if h else None
            rows.append(dict(n=n, seed=seed, K=I.K, Z_construct=c and c['Z'], Z_heur=h and h['Z'], Z_exact=e['Z'], exact_bound=e['bound'],
                             exact_status=e['status'], t_construct=tc, t_heur=th, t_exact=e['runtime'], heur_valid=hv, exact_valid=ev))
            print(rows[-1], flush=True)
    w = csv.DictWriter(open(os.path.join(OUT, 'results_small.csv'), 'w', newline=''), fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
def large():
    rows = []
    for n in LARGE:
        for seed in range(1, LSEEDS+1):
            I = darp.make_instance(n, 1000+seed)
            t = time.time(); c = darp.solve_heuristic(I, ls=False); tc = time.time()-t
            t = time.time(); h = darp.solve_heuristic(I, ls=True); th = time.time()-t
            rows.append(dict(n=n, seed=1000+seed, K=I.K, feasible=h is not None, Z_construct=c and c['Z'], Z_heur=h and h['Z'],
                             ls_iters=h and h['ls_iters'], t_construct=tc, t_heur=th, valid=h and darp.validate(I, h['routes'], h['T'])[0]))
            print(rows[-1], flush=True)
    w = csv.DictWriter(open(os.path.join(OUT, 'results_large.csv'), 'w', newline=''), fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
if __name__ == '__main__':
    open(os.path.join(OUT, 'env.txt'), 'w').write(f"python {platform.python_version()}, numpy {numpy.__version__}, scipy {scipy.__version__}, {platform.platform()}\n")
    {'small': small, 'large': large}[sys.argv[1]]()
