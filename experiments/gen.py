"""Instance generation for Tasks 5-6: road-network graphs, 6 families, structure statistics, instance files."""
import math, numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components

H = 480.0; S = 1.0; Q = 4
WEIGHTS = dict(alpha=1.0, beta=0.5, gamma=0.5, delta=1.0, wmax=10.0)

# ---------------- graphs: undirected, edge weight = travel minutes ----------------
def grid_graph(r=12, c=12, t=2.0):
    idx = lambda i, j: i*c + j; E = []
    for i in range(r):
        for j in range(c):
            if j+1 < c: E.append((idx(i, j), idx(i, j+1), t))
            if i+1 < r: E.append((idx(i, j), idx(i+1, j), t))
    return r*c, E

def ring_graph(nv=100, t=1.2, chords=5, ct=8.0, rng=None):
    E = [(i, (i+1) % nv, t) for i in range(nv)]
    for _ in range(chords):
        a = int(rng.integers(nv)); b = (a + nv//2 + int(rng.integers(-8, 9))) % nv
        E.append((a, b, ct))
    return nv, E

def cluster_graph(k=5, r=5, c=5, t=1.5, link=10.0):
    E = []; off = 0; centers = []
    for _ in range(k):
        nv, ge = grid_graph(r, c, t); E += [(a+off, b+off, w) for a, b, w in ge]
        centers.append(off + (r//2)*c + c//2); off += nv
    hub = off
    for ctr in centers: E.append((hub, ctr, link))
    return off+1, E

def geo_graph(nv=150, k=4, speed=0.5, rng=None):
    pts = rng.uniform(0, 12, size=(nv, 2)); E = set()
    for i in range(nv):
        d = np.hypot(*(pts - pts[i]).T); for_k = np.argsort(d)[1:k+1]
        for j in for_k: E.add((min(i, int(j)), max(i, int(j)), float(d[j]/speed)))
    return nv, list(E)

def build(nv, E):
    rows = [a for a, b, w in E] + [b for a, b, w in E]; cols = [b for a, b, w in E] + [a for a, b, w in E]
    w = [w for a, b, w in E]*2
    A = csr_matrix((w, (rows, cols)), shape=(nv, nv))
    ncomp, _ = connected_components(A)
    # keep the graph connected by linking components (rare for geo graphs)
    if ncomp > 1:
        lab = connected_components(A)[1]
        for c in range(1, ncomp):
            a = int(np.where(lab == 0)[0][0]); b = int(np.where(lab == c)[0][0]); E.append((a, b, 10.0))
        return build(nv, E)
    D = shortest_path(A, method='D', directed=False)
    Hop = shortest_path(csr_matrix((np.ones(len(w)), (rows, cols)), shape=(nv, nv)), method='D', directed=False, unweighted=True)
    return A, D, Hop, E

def graph_stats(nv, E, D, Hop):
    return dict(V=nv, E=len(E), diameter_min=float(D.max()), diameter_hops=int(Hop.max()), avg_degree=2*len(E)/nv,
                mean_dist_min=float(D[np.triu_indices(nv, 1)].mean()))

# ---------------- families ----------------
FAMILIES = {
 'F1': dict(name='Baseline grid', graph='grid', ratio=10, W=15, rho=2.0, lead=30, kind='uniform'),
 'F2': dict(name='Tight windows', graph='grid', ratio=10, W=4, rho=2.0, lead=30, kind='uniform'),
 'F3': dict(name='Scarce fleet', graph='grid', ratio=25, W=15, rho=2.0, lead=30, kind='uniform'),
 'F4': dict(name='Sparse ring network', graph='ring', ratio=10, W=15, rho=2.0, lead=30, kind='uniform'),
 'F5': dict(name='Adversary A: delay-to-pool pairs', graph='biggrid', ratio=8, W=15, rho=1.1, lead=30, kind='pairs'),
 'F6': dict(name='Adversary B: committed-vehicle trap', graph='grid', ratio=12, W=15, rho=2.0, lead=30, kind='decoy'),
}
GRAPHS = {}
def get_graph(kind, seed=0):
    if kind in GRAPHS: return GRAPHS[kind]
    rng = np.random.default_rng(12345)
    nv, E = {'grid': lambda: grid_graph(), 'biggrid': lambda: grid_graph(30, 30, 2.0), 'ring': lambda: ring_graph(rng=rng), 'clusters': lambda: cluster_graph(), 'geo': lambda: geo_graph(rng=rng)}[kind]()
    A, D, Hop, E = build(nv, E); GRAPHS[kind] = (nv, E, D, Hop); return GRAPHS[kind]

def make_instance(fam, n, seed, graph=None, ratio=None, W=None, rho=None):
    f = dict(FAMILIES[fam]); 
    if graph: f['graph'] = graph
    if ratio: f['ratio'] = ratio
    if W is not None: f['W'] = W
    if rho: f['rho'] = rho
    nv, E, D, Hop = get_graph(f['graph'])
    rng = np.random.default_rng(abs(hash((fam, n, seed, f['graph'], f['ratio'], f['W']))) % (2**32) if False else (seed*7919 + n*31 + sum(map(ord, fam+f['graph']))*101 + int(f['ratio']*10) + int(f['W']*3)))
    depot = int(np.argmin(D.max(axis=1))); m = max(1, math.ceil(n / f['ratio']))
    po = np.zeros(n, int); do = np.zeros(n, int); e = np.zeros(n); Wv = np.full(n, float(f['W'])); Lv = np.full(n, float(f['lead']))
    def far(o, lo, hi):
        c = np.where((D[o] >= lo) & (D[o] <= hi))[0]; return int(rng.choice(c)) if len(c) else int(np.argmax(D[o]))
    if f['kind'] == 'uniform':
        for i in range(n):
            po[i] = rng.integers(nv); do[i] = far(po[i], 3.0, 1e9); e[i] = rng.uniform(10, 360)
    elif f['kind'] == 'decoy':
        # m decoys (known at t=0, long trips, window opens at ~140, 120 min wide), 4m clustered burst requests (e in [150,156], 8 min wide,
        # revealed 1 min ahead), the rest ordinary background requests. Decoys occupy every vehicle when the burst arrives.
        nd = m; nb = min(4*m, max(0, n - nd)); dec = []; bur = []
        hub_o = int(rng.integers(nv)); hub_d = far(hub_o, 30.0, 40.0)
        near_o = np.where(D[hub_o] <= 3.0)[0]; near_d = np.where(D[hub_d] <= 3.0)[0]
        for i in range(n):
            if i < nd:
                po[i] = rng.integers(nv); do[i] = far(po[i], 35.0, 1e9); e[i] = 140 + rng.uniform(0, 5); Wv[i] = 120.0; Lv[i] = e[i]
            elif i < nd + nb:
                po[i] = int(rng.choice(near_o)); do[i] = int(rng.choice(near_d)); e[i] = 150 + rng.uniform(0, 6); Wv[i] = 8.0; Lv[i] = 1.0
            else:
                po[i] = rng.integers(nv); do[i] = far(po[i], 3.0, 1e9); e[i] = rng.uniform(10, 360)
    elif f['kind'] == 'pairs':
        cent = [40, 140, 240, 340]; assert n % 2 == 0
        for p in range(n//2):
            a, b = 2*p, 2*p+1
            po[a] = rng.integers(nv); do[a] = far(po[a], 28.0, 40.0)
            po[b] = po[a]; do[b] = do[a]          # partner has the same origin and destination: only the intended pair can pool
            ea = cent[p % 4] + rng.uniform(0, 6); e[a] = ea; e[b] = ea + f['W'] + 2.0
    l = e + Wv; rel = np.maximum(0.0, e - Lv)
    return dict(fam=fam, n=n, m=m, seed=seed, graph=f['graph'], nv=nv, nE=len(E), D=D, Hop=Hop, depot=depot, po=po, do=do, e=e, l=l,
                rel=rel, rho=f['rho'], W=float(Wv.mean()), Wmin=float(Wv.min()), Wmax_=float(Wv.max()), lead=float((e-rel).mean()), W_cfg=f['W'], kind=f['kind'], ratio=f['ratio'], Q=Q, H=H, s=S, **WEIGHTS)

def instance_stats(I):
    nv, E, D, Hop = get_graph(I['graph']); st = graph_stats(nv, E, D, Hop)
    direct = D[I['po'], I['do']]
    bins = np.histogram(I['e'], bins=np.arange(0, I['H']+10, 10))[0]
    st.update(n=I['n'], m=I['m'], Q=I['Q'], W=I['W'], Wmin=I['Wmin'], Wmax=I['Wmax_'], rho=I['rho'], lead=I['lead'], density_mean=I['n']/I['H'], density_peak10=float(bins.max())/10,
              mean_direct=float(direct.mean()), work_bound_vehicles=float(((direct + I['s']).sum()) / I['H']))
    return st

def write_instance(I, path):
    with open(path, 'w') as f:
        f.write(f"{I['nv']} {I['n']} {I['m']} {I['Q']} {I['depot']} {I['rho']} {I['s']} {I['H']} {I['alpha']} {I['beta']} {I['gamma']} {I['delta']} {I['wmax']}\n")
        for row in I['D']: f.write(' '.join(repr(float(x)) for x in row) + '\n')
        for i in range(I['n']): f.write(f"{int(I['po'][i])} {int(I['do'][i])} {float(I['e'][i])!r} {float(I['l'][i])!r} {float(I['rel'][i])!r}\n")
