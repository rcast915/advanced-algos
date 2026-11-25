import pandas as pd, numpy as np, json, time
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import sys; sys.path.insert(0, '/home/claude/darp2')
BLUE, ORANGE, AQUA, INK, GREY = '#257de0', '#f47230', '#00b894', '#191919', '#9c9c9c'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#666666',
                     'axes.labelcolor': '#191919', 'xtick.color': '#444444', 'ytick.color': '#444444', 'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.6, 'axes.axisbelow': True})
out = {}
M = pd.read_csv('results_main.csv'); S = pd.read_csv('results_sweeps.csv'); G = pd.read_csv('results_graphs.csv'); SC = pd.read_csv('results_scaling.csv'); TH = pd.read_csv('results_threads.csv')

# ---- capacity model: requests served per vehicle at saturation (served*n/m) ----
S['per_veh'] = S.served * 1.0 / S.m
sat = S[(S['mode'] == 'static') & (S.served_pct < 90)]
kap = sat.groupby(['n', 'sweep_W']).per_veh.mean().unstack('n').round(1); out['kappa'] = kap
kap_dyn = S[(S['mode'] == 'dynamic') & (S.served_pct < 90)].groupby(['n', 'sweep_W']).per_veh.mean().unstack('n').round(1); out['kappa_dyn'] = kap_dyn

# ---- Fig 1: served % vs fleet size ratio, W=15 ----
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, n in zip(axs, (100, 500)):
    d = S[(S.n == n) & (S.sweep_W == 15)]
    for mode, c, lab in (('static', BLUE, 'Static (insertion + local search)'), ('dynamic', ORANGE, 'Dynamic (online insertion)')):
        g = d[d['mode'] == mode].groupby('sweep_ratio').agg(y=('served_pct', 'mean'), lo=('served_pct', 'min'), hi=('served_pct', 'max'), m=('m', 'mean'))
        ax.plot(g.index, g.y, '-o', color=c, lw=1.6, ms=4, label=lab, mfc=c, mec='white', mew=0.8)
        ax.fill_between(g.index, g.lo, g.hi, color=c, alpha=0.12, lw=0)
    k = float(kap.loc[15, n]); xs = np.linspace(5, 60, 100)
    ax.plot(xs, np.minimum(100, 100 * k / xs), ls='--', color=GREY, lw=1.2, label=f'Capacity model, kappa = {k:g} requests/vehicle')
    ax.set_title(f'n = {n} requests', fontsize=9.5, loc='left'); ax.set_xlabel('requests per vehicle, n/m'); ax.set_ylim(0, 105); ax.set_xlim(4, 62)
axs[0].set_ylabel('requests served (%)'); axs[0].legend(fontsize=7.2, frameon=False, loc='lower left')
fig.tight_layout(); fig.savefig('figs/fig1_fleet.png', dpi=200); plt.close(fig)

# ---- Fig 2: heatmap ratio x W (static, n=100 and 500) ----
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, n in zip(axs, (100, 500)):
    t = S[(S.n == n) & (S['mode'] == 'static')].pivot_table(index='sweep_ratio', columns='sweep_W', values='served_pct', aggfunc='mean')
    im = ax.imshow(t.values, cmap='Blues', vmin=0, vmax=100, aspect='auto', origin='lower')
    ax.set_xticks(range(len(t.columns))); ax.set_xticklabels(t.columns); ax.set_yticks(range(len(t.index))); ax.set_yticklabels(t.index)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]): ax.text(j, i, f'{t.values[i, j]:.0f}', ha='center', va='center', fontsize=7, color='white' if t.values[i, j] > 60 else INK)
    ax.set_title(f'n = {n}', fontsize=9.5, loc='left'); ax.set_xlabel('time-window width W (min)'); ax.set_ylabel('requests per vehicle, n/m'); ax.grid(False)
fig.tight_layout(); fig.savefig('figs/fig2_heatmap.png', dpi=200); plt.close(fig)

# ---- Fig 3: window sweep at ratio 20 and 30 ----
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
for ax, n in zip(axs, (100, 500)):
    for ratio, c, ls in ((20, BLUE, '-'), (30, ORANGE, '-'), (15, AQUA, '-')):
        g = S[(S.n == n) & (S['mode'] == 'static') & (S.sweep_ratio == ratio)].groupby('sweep_W').served_pct.mean()
        ax.plot(g.index, g.values, '-o', color=c, lw=1.6, ms=4, mec='white', mew=0.8, label=f'n/m = {ratio}')
    ax.set_xscale('log'); ax.set_xticks([2, 4, 8, 15, 30, 60]); ax.set_xticklabels([2, 4, 8, 15, 30, 60]); ax.set_xlabel('time-window width W (min)'); ax.set_title(f'n = {n}', fontsize=9.5, loc='left'); ax.set_ylim(0, 105)
axs[0].set_ylabel('requests served (%), static'); axs[0].legend(fontsize=7.5, frameon=False, loc='lower right')
fig.tight_layout(); fig.savefig('figs/fig3_windows.png', dpi=200); plt.close(fig)

# ---- Fig 4: predictions vs actual, per family and size ----
P = pd.read_csv('prediction_vs_actual.csv')
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=True, sharey=True)
for ax, (a, p, ttl) in zip(axs, (('act_s', 'pred_s', 'Static'), ('act_d', 'pred_d', 'Dynamic'))):
    ax.plot([0, 100], [0, 100], color=GREY, lw=1, ls='--')
    ax.scatter(P[p], P[a], s=22, color=BLUE if ttl == 'Static' else ORANGE, edgecolor='white', linewidth=0.6, zorder=3)
    for _, r in P.iterrows():
        if abs(r[p] - r[a]) > 12: ax.annotate(f"{r.f}/{r.n}", (r[p], r[a]), fontsize=6.5, xytext=(3, 3), textcoords='offset points', color='#444444')
    ax.set_xlabel('predicted served (%)'); ax.set_title(ttl, fontsize=9.5, loc='left'); ax.set_xlim(25, 105); ax.set_ylim(25, 105)
axs[0].set_ylabel('measured served (%)'); fig.tight_layout(); fig.savefig('figs/fig4_predictions.png', dpi=200); plt.close(fig)

# ---- Fig 5: runtime and steps vs n (log-log) ----
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
lab = {'static': ('Static (with local search)', BLUE), 'static_nols': ('Static, insertion only', AQUA), 'dynamic': ('Dynamic', ORANGE)}
for mode, (l, c) in lab.items():
    g = SC[SC['mode'] == mode].groupby('n').agg(rt=('runtime', 'median'), st=('steps', 'median'))
    axs[0].plot(g.index, g.rt, '-o', color=c, lw=1.6, ms=4, mec='white', mew=0.8, label=l); axs[1].plot(g.index, g.st, '-o', color=c, lw=1.6, ms=4, mec='white', mew=0.8, label=l)
ns = np.array([100, 5000.]); axs[0].plot(ns, 3e-3 * (ns / 100) ** 2, color=GREY, ls='--', lw=1, label='slope 2 reference'); axs[1].plot(ns, 3.8e5 * (ns / 100) ** 2, color=GREY, ls='--', lw=1)
for ax, yl in zip(axs, ('running time (s, median of 3 seeds)', 'elementary steps (stop evaluations)')):
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('number of requests n  (n/m = 10)'); ax.set_ylabel(yl)
axs[0].legend(fontsize=7, frameon=False, loc='upper left'); fig.tight_layout(); fig.savefig('figs/fig5_runtime.png', dpi=200); plt.close(fig)
fits = {}
for mode in lab:
    g = SC[(SC['mode'] == mode) & (SC.n >= 100)].groupby('n').agg(rt=('runtime', 'median'), st=('steps', 'median'))
    fits[mode] = (np.polyfit(np.log(g.index), np.log(g.rt), 1)[0], np.polyfit(np.log(g.index), np.log(g.st), 1)[0])
out['fits'] = fits

# ---- Fig 6: thread speedup ----
t1 = TH[TH.threads == 1].set_index('n').runtime; t2 = TH[TH.threads == 2].set_index('n').runtime
out['threads'] = pd.DataFrame({'t1': t1, 't2': t2, 'speedup': t1 / t2}).round(3)

# ---- evidence: what drives running time / quality ----
S['L'] = 2 * S.n / S.m
st = S[S['mode'] == 'static']
X = np.column_stack([np.ones(len(st)), np.log(st.n), np.log(st.m)]); b = np.linalg.lstsq(X, np.log(st.steps), rcond=None)[0]
out['steps_fit'] = dict(const=b[0], log_n=b[1], log_m=b[2])
dyn = S[S['mode'] == 'dynamic']; X = np.column_stack([np.ones(len(dyn)), np.log(dyn.n), np.log(dyn.m)]); b2 = np.linalg.lstsq(X, np.log(dyn.steps), rcond=None)[0]; out['steps_fit_dyn'] = dict(const=b2[0], log_n=b2[1], log_m=b2[2])
unsat = st
out['spearman_quality'] = {c: round(spearmanr(st[c], st.served_pct)[0], 2) for c in ('sweep_ratio', 'sweep_W', 'n')}
g5 = G[(G['mode'] == 'static')]
out['graph_corr'] = {c: dict(served=round(spearmanr(g5[c], g5.served_pct)[0], 2), steps=round(spearmanr(g5[c], g5.steps)[0], 2), ride=round(spearmanr(g5[c], g5.ride_over_direct)[0], 2)) for c in ('V', 'E', 'diameter_min', 'diameter_hops', 'avg_degree', 'mean_dist_min')}
# main-experiment drivers of served %
mm = M[M['mode'] == 'static']
out['main_corr'] = {c: round(spearmanr(mm[c], mm.served_pct)[0], 2) for c in ('n', 'm', 'W', 'rho', 'work_bound_vehicles', 'density_mean', 'density_peak10')}
mm = mm.assign(ratio=mm.n / mm.m, vehicles_per_work=mm.m / mm.work_bound_vehicles); out['main_corr']['n_over_m'] = round(spearmanr(mm.ratio, mm.served_pct)[0], 2); out['main_corr']['m_over_workbound'] = round(spearmanr(mm.vehicles_per_work, mm.served_pct)[0], 2)

# ---- APSP precomputation time vs |V| ----
import gen
from scipy.sparse.csgraph import shortest_path
ap = []
for r_, c_ in ((8, 8), (12, 12), (20, 20), (30, 30), (45, 45), (60, 60)):
    nv, E = gen.grid_graph(r_, c_); t0 = time.time(); gen.build(nv, E); ap.append((nv, len(E), time.time() - t0))
out['apsp'] = pd.DataFrame(ap, columns=['V', 'E', 'seconds']).round(4)
pd.to_pickle(out, 'analysis_out.pkl')
for k, v in out.items(): print('==', k); print(v)
