"""Independent validator and metrics. Works from the instance dict (numpy) and the solver's JSON output only."""
import json, subprocess, numpy as np, os, tempfile
import gen
from paths import SOLVER
EPS = 1e-6

def validate(I, sol, mode):
    n, m, D = I['n'], I['m'], I['D']; po, do = I['po'], I['do']; s = I['s']
    bad = []; served = sol['served']; R = sol['routes']; T = sol['times']
    seen = [v for r in R for v in r]
    exp = sorted([i for i in range(n) if served[i]] + [n+i for i in range(n) if served[i]])
    if sorted(seen) != exp: bad.append('C1 served stops exactly once / rejected never')
    veh = {v: k for k, r in enumerate(R) for v in r}; pos = {v: j for r in R for j, v in enumerate(r)}
    loc = lambda v: po[v] if v < n else do[v-n]
    tm = {}
    for k, r in enumerate(R):
        for j, v in enumerate(r): tm[v] = T[k][j]
    for i in range(n):
        if served[i]:
            if veh[i] != veh[n+i]: bad.append(f'C2 req {i}')
            elif pos[i] > pos[n+i]: bad.append(f'C3 req {i}')
    for k, r in enumerate(R):
        load = 0; prev = I['depot']; tprev = None
        for j, v in enumerate(r):
            dep = 0.0 if j == 0 else tprev + s
            if tm[v] < dep + D[prev, loc(v)] - EPS: bad.append(f'C4 veh {k} stop {v}')
            if v < n: e_, l_ = I['e'][v], I['l'][v]
            else:
                i = v-n; dr = D[po[i], do[i]]; e_, l_ = I['e'][i]+s+dr, I['l'][i]+s+I['rho']*dr
            if not (e_-EPS <= tm[v] <= l_+EPS): bad.append(f'C5 veh {k} stop {v}')
            load += 1 if v < n else -1
            if load > I['Q'] or load < 0: bad.append(f'C8 veh {k}')
            prev = loc(v); tprev = tm[v]
        if r and tprev + s + D[prev, I['depot']] > I['H'] + EPS: bad.append(f'C6 veh {k}')
    for i in range(n):
        if served[i] and tm[n+i] - (tm[i] + s) > I['rho']*D[po[i], do[i]] + EPS: bad.append(f'C7 req {i}')
    if mode == 'dynamic':   # online validity: nothing before release, past stops stay before newly inserted stops, no commitment violations
        if sol['commit_violations'] != 0: bad.append('O commit violations (solver invariant)')
        for i in range(n):
            if not served[i]: continue
            if tm[i] < I['rel'][i] - EPS: bad.append(f'O served before release {i}')
            k = veh[i]
            for u in R[k]:
                if u not in (i, n+i) and tm[u] <= I['rel'][i] + 1e-12 and pos[u] > pos[i]: bad.append(f'O past stop {u} after new req {i}')
    return (not bad), bad

def metrics(I, sol):
    n, D, po, do, s = I['n'], I['D'], I['po'], I['do'], I['s']; R, T = sol['routes'], sol['times']
    loc = lambda v: po[v] if v < n else do[v-n]
    tt = 0.0; tm = {}
    for k, r in enumerate(R):
        prev = I['depot']
        for j, v in enumerate(r): tt += D[prev, loc(v)]; prev = loc(v); tm[v] = T[k][j]
        if r: tt += D[prev, I['depot']]
    sv = [i for i in range(n) if sol['served'][i]]
    wait = [tm[i] - I['e'][i] for i in sv]; ride = [tm[n+i] - (tm[i] + s) for i in sv]
    exr = [max(0, ride[j] - D[po[i], do[i]]) for j, i in enumerate(sv)]
    Z = I['alpha']*tt + I['beta']*sum(wait) + I['gamma']*sum(exr) + I['delta']*sum(max(0, w - I['wmax']) for w in wait)
    return dict(served_pct=100.0*len(sv)/n, served=len(sv), travel_min=tt, Z=Z, wait_mean=float(np.mean(wait)) if wait else 0, wait_max=max(wait) if wait else 0,
                ride_mean=float(np.mean(ride)) if ride else 0, ride_max=max(ride) if ride else 0, vehicles_used=sum(1 for r in R if r),
                runtime=sol['runtime'], steps=sol['steps'], evals=sol['evals'], maxrss_kb=sol['maxrss_kb'], rss_load_kb=sol['rss_load_kb'], ls_moves=sol['ls_moves'], passes=sol['passes'])

def run(I, mode, threads=1, passes=30, tag=None):
    d = tempfile.mkdtemp(prefix='darp_'); p = os.path.join(d, 'inst.txt'); gen.write_instance(I, p)
    out = subprocess.run([str(SOLVER), p, mode, str(threads), str(passes)], capture_output=True, text=True, check=True).stdout
    sol = json.loads(out); ok, bad = validate(I, sol, mode)
    os.remove(p); os.rmdir(d)
    return sol, ok, bad
