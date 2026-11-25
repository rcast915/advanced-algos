"""Run all Task 5-6 experiments. Usage: python run_experiments.py main|scaling|sweeps|graphs|threads"""
import sys, csv, time, statistics, numpy as np
sys.path.insert(0, '/home/claude/darp2')
import gen, validate

SEEDS = {20: [1, 2, 3, 4, 5], 100: [1, 2, 3, 4, 5], 500: [1, 2, 3]}

def row_for(I, mode, threads=1, reps=1, passes=30, extra=None):
    times = []; sol = None
    for _ in range(reps):
        sol, ok, bad = validate.run(I, mode, threads=threads, passes=passes); times.append(sol['runtime'])
    sol['runtime'] = statistics.median(times)
    mt = validate.metrics(I, sol); st = gen.instance_stats(I)
    n = I['n']; sv = [i for i in range(n) if sol['served'][i]]
    direct = I['D'][I['po'], I['do']]
    mt['ride_over_direct'] = (mt['ride_mean'] / float(direct[sv].mean())) if sv else 0.0
    r = dict(family=I['fam'], graph=I['graph'], seed=I['seed'], mode=mode, threads=threads, valid=ok, n_violations=len(bad), **st, **mt)
    if extra: r.update(extra)
    return r

def write(rows, path):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def main():
    rows = []
    for fam in gen.FAMILIES:
        for n, seeds in SEEDS.items():
            for sd in seeds:
                I = gen.make_instance(fam, n, sd)
                for mode in ('static', 'static_nols', 'dynamic'):
                    rows.append(row_for(I, mode)); print(fam, n, sd, mode, round(rows[-1]['served_pct'], 1), rows[-1]['valid'], flush=True)
    write(rows, 'results_main.csv')

def scaling():
    rows = []
    for n in (20, 50, 100, 200, 500, 1000, 2000, 5000):
        for sd in (1, 2, 3):
            I = gen.make_instance('F1', n, sd)
            for mode in ('static', 'static_nols', 'dynamic'):
                rows.append(row_for(I, mode, reps=3 if n <= 1000 else 1)); print('scaling', n, sd, mode, round(rows[-1]['runtime'], 3), flush=True)
    write(rows, 'results_scaling.csv')

def sweeps():
    rows = []
    for n in (100, 500):
        seeds = [1, 2, 3, 4, 5] if n == 100 else [1, 2, 3]
        for ratio in (5, 8, 10, 15, 20, 30, 40, 60):
            for W in (2, 4, 8, 15, 30, 60):
                for sd in seeds:
                    I = gen.make_instance('F1', n, sd, ratio=ratio, W=W)
                    for mode in ('static', 'dynamic'):
                        rows.append(row_for(I, mode, extra=dict(sweep_ratio=ratio, sweep_W=W)))
                print('sweep', n, ratio, flush=True)
    write(rows, 'results_sweeps.csv')

def graphs():
    rows = []
    for g in ('grid', 'biggrid', 'ring', 'clusters', 'geo'):
        for n, seeds in ((100, [1, 2, 3, 4, 5]), (500, [1, 2, 3])):
            for sd in seeds:
                I = gen.make_instance('F1', n, sd, graph=g)
                for mode in ('static', 'dynamic'):
                    rows.append(row_for(I, mode)); print('graph', g, n, sd, mode, round(rows[-1]['served_pct'], 1), flush=True)
    write(rows, 'results_graphs.csv')

def threads():
    rows = []
    for n in (500, 2000, 5000):
        I = gen.make_instance('F1', n, 1)
        for th in (1, 2):
            rows.append(row_for(I, 'static', threads=th, reps=3 if n < 5000 else 1)); print('threads', n, th, round(rows[-1]['runtime'], 3), flush=True)
    write(rows, 'results_threads.csv')

if __name__ == '__main__':
    globals()[sys.argv[1]]()
