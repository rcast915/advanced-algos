"""Static DARP: instance generator, independent validator, insertion + relocate heuristic.
Node ids: pickup i -> i, delivery i -> n+i (i = 0..n-1); depot o is implicit (id -1)."""
import math, random
from dataclasses import dataclass, field

ALPHA, BETA, GAMMA, DELTA = 1.0, 0.5, 0.5, 1.0   # objective weights (Z = a*TT + b*wait + g*excess + d*longwait)
WMAX, RHO, S, Q, SPEED = 10.0, 2.0, 1.0, 4, 0.5   # long-wait threshold (min), ride factor, service (min), capacity, km/min
EPS = 1e-6

@dataclass
class Inst:
    n: int; K: int; H: float
    xy: list            # coordinates, index 0..2n-1; depot in self.depot
    depot: tuple
    e: list; l: list    # windows per node
    q: list             # load change per node (+1 pickup, -1 delivery)
    direct: list        # per request direct travel time
    maxride: list       # per request max ride time
    seed: int = 0

def tt(a, b):  # travel time in minutes, Euclidean km / speed
    return math.hypot(a[0]-b[0], a[1]-b[1]) / SPEED

def make_instance(n, seed, K=None, H=480.0, width=15.0):
    rng = random.Random(seed)
    K = K or max(2, math.ceil(n / 4))
    depot = (2.5, 2.5)
    xy, e, l, direct, maxride = [None]*(2*n), [0]*(2*n), [0]*(2*n), [], []
    for i in range(n):
        while True:
            p = (rng.uniform(0, 5), rng.uniform(0, 5)); d = (rng.uniform(0, 5), rng.uniform(0, 5))
            if tt(p, d) >= 3.0: break
        ei = rng.uniform(20, H - 100); dr = tt(p, d); mr = RHO * dr
        xy[i], xy[n+i] = p, d
        e[i], l[i] = ei, ei + width
        e[n+i], l[n+i] = ei + S + dr, ei + width + S + mr
        direct.append(dr); maxride.append(mr)
    q = [1]*n + [-1]*n
    return Inst(n, K, H, xy, depot, e, l, q, direct, maxride, seed)

def dist(I, u, v):
    a = I.depot if u < 0 else I.xy[u]; b = I.depot if v < 0 else I.xy[v]
    return tt(a, b)

def objective(I, routes, T):
    """Z from routes and a time dict T[node]; also returns parts."""
    TT = w = ex = lw = 0.0
    for r in routes:
        prev = -1
        for v in r + [-1]:
            TT += dist(I, prev, v); prev = v
    for i in range(I.n):
        wi = T[i] - I.e[i]; rt = T[I.n+i] - (T[i] + S)
        w += wi; ex += max(0.0, rt - I.direct[i]); lw += max(0.0, wi - WMAX)
    return ALPHA*TT + BETA*w + GAMMA*ex + DELTA*lw, dict(TT=TT, wait=w, excess=ex, longwait=lw)

def validate(I, routes, T):
    """Independent check of C1-C8. Returns (ok, list_of_violations)."""
    bad = []; n = I.n
    seen = [v for r in routes for v in r]
    if len(routes) > I.K: bad.append("C1 too many vehicles")
    if sorted(seen) != list(range(2*n)): bad.append("C1 each stop exactly once")
    veh = {v: k for k, r in enumerate(routes) for v in r}
    pos = {v: j for r in routes for j, v in enumerate(r)}
    for i in range(n):
        if i in veh and n+i in veh:
            if veh[i] != veh[n+i]: bad.append(f"C2 request {i} split")
            elif pos[i] > pos[n+i]: bad.append(f"C3 precedence {i}")
    for k, r in enumerate(routes):
        load = 0; prev = -1; t_prev = 0.0
        for v in r:
            earliest = (t_prev + (S if prev >= 0 else 0)) + dist(I, prev, v)
            if T[v] < earliest - EPS: bad.append(f"C4 timing at {v}")
            if not (I.e[v]-EPS <= T[v] <= I.l[v]+EPS): bad.append(f"C5 window at {v}")
            load += I.q[v]
            if load > Q or load < 0: bad.append(f"C8 capacity at {v}")
            prev, t_prev = v, T[v]
        if r and t_prev + S + dist(I, prev, -1) > I.H + EPS: bad.append(f"C6 return veh {k}")
    for i in range(n):
        if (T[n+i] - (T[i] + S)) > I.maxride[i] + EPS: bad.append(f"C7 ride time {i}")
    return (not bad), bad

# ---------- earliest-start forward schedule of one route (O(L)) ----------
def route_eval(I, r):
    """Returns (feasible, cost, times list) with every stop served as early as allowed."""
    n = I.n; t = 0.0; prev = -1; load = 0; cost = 0.0; times = []; tp = {}
    for v in r:
        arr = (t + (S if prev >= 0 else 0)) + dist(I, prev, v)
        cost += ALPHA * dist(I, prev, v)
        t = max(arr, I.e[v])
        if t > I.l[v] + EPS: return False, math.inf, None
        load += I.q[v]
        if load > Q: return False, math.inf, None
        if v < n:
            tp[v] = t; wi = t - I.e[v]; cost += BETA*wi + DELTA*max(0.0, wi - WMAX)
        else:
            rt = t - (tp[v-n] + S)
            if rt > I.maxride[v-n] + EPS: return False, math.inf, None
            cost += GAMMA * max(0.0, rt - I.direct[v-n])
        times.append(t); prev = v
    if r:
        if t + S + dist(I, prev, -1) > I.H + EPS: return False, math.inf, None
        cost += ALPHA * dist(I, prev, -1)
    return True, cost, times

def best_insertion(I, r, i):
    """Cheapest feasible way to insert request i into route r: returns (delta, new_route) or None."""
    n = I.n; _, base, _ = route_eval(I, r); best = None
    for a in range(len(r)+1):
        for b in range(a, len(r)+1):
            cand = r[:a] + [i] + r[a:b] + [n+i] + r[b:]
            ok, c, _ = route_eval(I, cand)
            if ok and (best is None or c - base < best[0]): best = (c - base, cand)
    return best

def construct(I, order="earliest"):
    reqs = sorted(range(I.n), key=lambda i: I.e[i])
    routes = [[] for _ in range(I.K)]
    for i in reqs:
        best = None
        for k, r in enumerate(routes):
            res = best_insertion(I, r, i)
            if res and (best is None or res[0] < best[0]): best = (res[0], k, res[1])
        if best is None: return None          # heuristic fails (incomplete)
        routes[best[1]] = best[2]
    return routes

def total_cost(I, routes):
    return sum(route_eval(I, r)[1] for r in routes)

def local_search(I, routes, max_iter=1000):
    """Relocate: remove a request, reinsert at best position of any route; accept only strict improvements."""
    n = I.n; it = 0; improved = True
    while improved and it < max_iter:
        improved = False
        for i in range(n):
            ks = next(k for k, r in enumerate(routes) if i in r)
            rem = [v for v in routes[ks] if v not in (i, n+i)]
            gain_base = route_eval(I, routes[ks])[1] - route_eval(I, rem)[1]   # saved by removing i
            best = None
            for k, r in enumerate(routes):
                base_r = rem if k == ks else r
                res = best_insertion(I, base_r, i)
                if res and (best is None or res[0] < best[0]): best = (res[0], k, res[1])
            if best and best[0] < gain_base - 1e-7:
                routes[ks] = rem; routes[best[1]] = best[2]; improved = True; it += 1
    return routes, it

def schedule_of(I, routes):
    T = {}
    for r in routes:
        ok, _, times = route_eval(I, r)
        for v, t in zip(r, times): T[v] = t
    return T

def solve_heuristic(I, ls=True):
    routes = construct(I)
    if routes is None: return None
    z0 = total_cost(I, routes); iters = 0
    if ls: routes, iters = local_search(I, routes)
    routes = [r for r in routes]
    T = schedule_of(I, routes)
    Z, parts = objective(I, routes, T)
    return dict(routes=routes, T=T, Z=Z, parts=parts, Z_construct=z0, ls_iters=iters)
