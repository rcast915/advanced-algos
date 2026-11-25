"""Exact 3-index MIP for the static DARP (same objective/constraints as darp.py), solved with HiGHS via scipy.milp."""
import numpy as np, math, time
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
import darp
from darp import S, Q, ALPHA, BETA, GAMMA, DELTA, WMAX

def solve_exact(I, time_limit=60):
    n, K, H = I.n, I.K, I.H; N = 2*n
    nodes = list(range(N)) + [N, N+1]       # N = depot start o, N+1 = depot end f
    o, f = N, N+1
    def d(u, v):
        uu = -1 if u in (o, f) else u; vv = -1 if v in (o, f) else v
        return darp.dist(I, uu, vv)
    arcs = []   # (k,u,v)
    for k in range(K):
        for u in nodes:
            for v in nodes:
                if u == v or u == f or v == o: continue
                if u == o and v == f: arcs.append((k, u, v)); continue
                if u == o and v >= n and v < N: continue          # first stop must be a pickup
                if v == f and u < n: continue                      # last stop must be a delivery
                if u < N and v < N and u >= n and v == u - n: continue
                if u < n and v < N and v == u + n: pass
                arcs.append((k, u, v))
    ix = {a: j for j, a in enumerate(arcs)}
    nx = len(arcs)
    # variable layout: x | T[0..N-1] | Tf[k] | L[0..N-1] | w[i] | ex[i] | lw[i]
    oT = nx; oTf = oT + N; oL = oTf + K; oW = oL + N; oE = oW + n; oLW = oE + n; nv = oLW + n
    c = np.zeros(nv); lb = np.zeros(nv); ub = np.ones(nv); integ = np.zeros(nv)
    for (k, u, v), j in ix.items(): c[j] = ALPHA * d(u, v); integ[j] = 1
    for v in range(N): lb[oT+v], ub[oT+v] = I.e[v], I.l[v]
    for k in range(K): lb[oTf+k], ub[oTf+k] = 0, H
    for v in range(N): lb[oL+v], ub[oL+v] = (1 if v < n else 0), (Q if v < n else Q-1)
    for i in range(n): c[oW+i] = BETA; c[oE+i] = GAMMA; c[oLW+i] = DELTA; ub[oW+i] = ub[oE+i] = ub[oLW+i] = H
    rows = []; lo = []; hi = []
    A = lil_matrix((60000, nv)); r = 0
    def add(coefs, l_, h_):
        nonlocal r
        for j, a in coefs: A[r, j] += a
        lo.append(l_); hi.append(h_); r += 1
    out = {u: [] for u in nodes}; inn = {u: [] for u in nodes}
    for a, j in ix.items(): out[a[1]].append((a[0], j, a)); inn[a[2]].append((a[0], j, a))
    for v in range(N): add([(j, 1) for _, j, _ in out[v]], 1, 1)                      # C1 visit once
    for k in range(K):
        add([(ix[a], 1) for _, _, a in [(0, 0, a) for a in arcs if a[0] == k and a[1] == o]], 1, 1)  # one start
        add([(ix[a], 1) for a in arcs if a[0] == k and a[2] == f], 1, 1)
        for v in range(N):                                                            # flow
            add([(ix[a], 1) for a in arcs if a[0] == k and a[1] == v] + [(ix[a], -1) for a in arcs if a[0] == k and a[2] == v], 0, 0)
    for i in range(n):                                                                # C2 same vehicle
        for k in range(K):
            add([(ix[a], 1) for a in arcs if a[0] == k and a[1] == i] + [(ix[a], -1) for a in arcs if a[0] == k and a[1] == n+i], 0, 0)
    Mt = H + S + 60
    for (k, u, v), j in ix.items():
        if u == o or v == f:
            if v == f and u != o:                                                      # return time
                add([(oTf+k, 1), (oT+u, -1), (j, -Mt)], S + d(u, f) - Mt, np.inf)
            elif u == o and v != f:
                add([(oT+v, 1), (j, -Mt)], d(o, v) - Mt, np.inf)
            continue
        add([(oT+v, 1), (oT+u, -1), (j, -Mt)], S + d(u, v) - Mt, np.inf)              # C4 timing
        add([(oL+v, 1), (oL+u, -1), (j, -(Q+1))], I.q[v] - (Q+1), np.inf)             # C8 load
    for i in range(n):
        add([(oT+n+i, 1), (oT+i, -1)], S + I.direct[i], np.inf)                        # C3 precedence+direct
        add([(oT+n+i, 1), (oT+i, -1)], -np.inf, S + I.maxride[i] + S - S + 0)         # C7: T(D)-T(P)-S <= maxride
        add([(oW+i, 1), (oT+i, -1)], -I.e[i], np.inf)                                  # wait = T-e
        add([(oW+i, 1), (oT+i, -1)], -I.e[i], -I.e[i])                                 # w == T-e
        add([(oE+i, 1), (oT+n+i, -1), (oT+i, 1)], -S - I.direct[i], np.inf)           # excess >= rt-direct
        add([(oLW+i, 1), (oW+i, -1)], -WMAX, np.inf)                                   # lw >= w-Wmax
    cons = LinearConstraint(A.tocsr()[:r], lo, hi)
    t0 = time.time()
    res = milp(c, constraints=cons, integrality=integ, bounds=Bounds(lb, ub),
               options=dict(time_limit=time_limit, disp=False, mip_rel_gap=1e-6))
    rt = time.time() - t0
    out = dict(status=res.status, runtime=rt, bound=getattr(res, 'mip_dual_bound', None), Z=None)
    if res.x is not None:
        x = res.x; routes = []
        for k in range(K):
            nxt = {a[1]: a[2] for a in arcs if a[0] == k and x[ix[a]] > .5}
            cur = nxt.get(o); rr = []
            while cur is not None and cur != f: rr.append(cur); cur = nxt.get(cur)
            routes.append(rr)
        T = {v: float(x[oT+v]) for v in range(N)}
        out.update(routes=routes, T=T, Z=float(res.fun))
    return out
