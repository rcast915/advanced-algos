import os, pandas as pd, numpy as np, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESULTS, FIGS, TABLES; FIGS.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import gen
M = pd.read_csv(RESULTS / 'results_main.csv'); S = pd.read_csv(RESULTS / 'results_sweeps.csv'); G = pd.read_csv(RESULTS / 'results_graphs.csv'); SC = pd.read_csv(RESULTS / 'results_scaling.csv'); TH = pd.read_csv(RESULTS / 'results_threads.csv')
NAMES = {k: v['name'] for k, v in gen.FAMILIES.items()}
def md(df, align=None):
    cols = list(df.columns); out = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in df.iterrows(): out.append('| ' + ' | '.join(str(x) for x in r.values) + ' |')
    return '\n'.join(out) + '\n'
def save(name, df): open(TABLES / f'tbl_{name}.md', 'w').write(md(df))
f1 = lambda x: f'{x:.1f}'
# ---- T2: structure per family and size ----
st = M.groupby(['family', 'n']).agg(V=('V', 'first'), E=('E', 'first'), dm=('diameter_min', 'first'), dh=('diameter_hops', 'first'), deg=('avg_degree', 'first'), m=('m', 'first'), Q=('Q', 'first'),
        W=('W', 'mean'), Wmin=('Wmin', 'min'), Wmax=('Wmax', 'max'), rho=('rho', 'first'), lead=('lead', 'mean'), dens=('density_mean', 'first'), peak=('density_peak10', 'mean')).reset_index()
t = pd.DataFrame({'Family': st.family, 'n': st.n, 'm': st.m, 'Vehicle capacity': st.Q, '|V|': st.V, '|E|': st.E, 'Diameter (min / hops)': [f'{a:.0f} / {b}' for a, b in zip(st.dm, st.dh)], 'Avg. degree': st.deg.map(lambda x: f'{x:.2f}'),
    'Window width (min)': [f'{w:g}' if lo == hi else f'{lo:g}-{hi:g} (mean {w:.0f})' for w, lo, hi in zip(st.W, st.Wmin, st.Wmax)], 'rho': st.rho, 'Req. density, mean / peak (per min)': [f'{a:.2f} / {b:.2f}' for a, b in zip(st.dens, st.peak)]})
save('structure', t)
# per-instance csv
cols = ['family', 'seed', 'graph', 'V', 'E', 'diameter_min', 'diameter_hops', 'avg_degree', 'mean_dist_min', 'n', 'm', 'Q', 'W', 'Wmin', 'Wmax', 'rho', 'lead', 'density_mean', 'density_peak10', 'mean_direct', 'work_bound_vehicles']
M[M['mode'] == 'static'][cols].to_csv(RESULTS / 'instance_stats_per_instance.csv', index=False)
# ---- T3/T4: results per mode ----
for mode, nm in (('static', 'static'), ('dynamic', 'dynamic')):
    d = M[M['mode'] == mode].groupby(['family', 'n']).agg(served=('served_pct', 'mean'), travel=('travel_min', 'mean'), wm=('wait_mean', 'mean'), wx=('wait_max', 'max'), rm=('ride_mean', 'mean'), rx=('ride_max', 'max'), rt=('runtime', 'mean'), steps=('steps', 'mean')).reset_index()
    save(f'res_{nm}', pd.DataFrame({'Family': d.family, 'n': d.n, 'Served (%)': d.served.map(f1), 'Travel (min)': d.travel.map(lambda x: f'{x:.0f}'), 'Wait mean / max (min)': [f'{a:.1f} / {b:.1f}' for a, b in zip(d.wm, d.wx)],
        'Ride mean / max (min)': [f'{a:.1f} / {b:.0f}' for a, b in zip(d.rm, d.rx)], 'Time (s)': d.rt.map(lambda x: f'{x:.3f}'), 'Steps': d.steps.map(lambda x: f'{x:,.0f}')}))
# ---- T5: predictions ----
P = pd.read_csv(RESULTS / 'prediction_vs_actual.csv'); txt = open(RESULTS / 'predictions.md').read(); rng = {}
for m_ in re.finditer(r'\| (F\d) \| (\d+) \| (\d+) \((\d+)-(\d+)\) \| (\d+) \((\d+)-(\d+)\) \|', txt):
    f, n, sp, sl, sh, dp, dl, dh = m_.groups(); rng[(f, int(n))] = (sp, f'{sl}-{sh}', dp, f'{dl}-{dh}')
rows = []
for _, r in P.iterrows():
    k = (r.f, int(r.n)); rows.append({'Family': r.f, 'n': int(r.n), 'Static: predicted (range)': f'{rng[k][0]} ({rng[k][1]})', 'Static: measured': f'{r.act_s:.1f}', 'In range': 'yes' if r.in_s else 'no',
        'Dynamic: predicted (range)': f'{rng[k][2]} ({rng[k][3]})', 'Dynamic: measured': f'{r.act_d:.1f}', 'In range ': 'yes' if r.in_d else 'no'})
save('pred', pd.DataFrame(rows))
# ---- T6: kappa ----
sat = S[(S.served_pct < 90)].assign(per_veh=lambda x: x.served / x.m)
kp = sat.groupby(['sweep_W', 'n', 'mode']).per_veh.mean().unstack(['n', 'mode']).round(1)
t = pd.DataFrame({'Window width W (min)': kp.index, 'n=100 static': kp[(100, 'static')].values, 'n=100 dynamic': kp[(100, 'dynamic')].values, 'n=500 static': kp[(500, 'static')].values, 'n=500 dynamic': kp[(500, 'dynamic')].values})
save('kappa', t)
# ---- T7: graphs ----
g = G.groupby(['graph', 'n', 'mode']).agg(V=('V', 'first'), E=('E', 'first'), dm=('diameter_min', 'first'), dh=('diameter_hops', 'first'), deg=('avg_degree', 'first'), md=('mean_dist_min', 'first'), served=('served_pct', 'mean'), rod=('ride_over_direct', 'mean'), steps=('steps', 'mean')).reset_index()
g = g[g.n == 500].pivot_table(index=['graph', 'V', 'E', 'dm', 'dh', 'deg', 'md'], columns='mode', values=['served', 'rod', 'steps']).reset_index()
g.columns = [a if not b else f'{a}_{b}' for a, b in g.columns]
order = {'geo': 0, 'grid': 1, 'ring': 2, 'clusters': 3, 'biggrid': 4}; g = g.sort_values('graph', key=lambda s: s.map(order))
save('graphs', pd.DataFrame({'Graph': g.graph, '|V|': g.V, '|E|': g.E, 'Diameter (min / hops)': [f'{a:.0f} / {b}' for a, b in zip(g.dm, g.dh)], 'Avg. degree': g.deg.map(lambda x: f'{x:.2f}'), 'Mean trip (min)': g.md.map(lambda x: f'{x:.1f}'),
    'Served static (%)': g.served_static.map(f1), 'Served dynamic (%)': g.served_dynamic.map(f1), 'Ride / direct (static)': g.rod_static.map(lambda x: f'{x:.2f}'), 'Steps (static)': g.steps_static.map(lambda x: f'{x:,.0f}')}))
# ---- T8: threads, memory ----
th = TH.pivot_table(index='n', columns='threads', values='runtime').reset_index()
save('threads', pd.DataFrame({'n': th.n, '1 thread (s)': th[1].map(lambda x: f'{x:.3f}'), '2 threads (s)': th[2].map(lambda x: f'{x:.3f}'), 'Speedup': (th[1] / th[2]).map(lambda x: f'{x:.2f}x')}))
mem = M[M['mode'] == 'static'].groupby('family').agg(V=('V', 'first'), peak=('maxrss_kb', 'max'), load=('rss_load_kb', 'max')).reset_index()
sm = SC[SC['mode'] == 'static'].groupby('n').agg(peak=('maxrss_kb', 'max')).reset_index()
save('memory', pd.DataFrame({'n': sm.n, 'Peak resident memory (MB), 144-node grid': (sm.peak / 1024).map(lambda x: f'{x:.1f}')}))
# ---- Fig 7: request density over time ----
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=False)
for ax, fam, c in zip(axs, ('F1', 'F5', 'F6'), ('#257de0', '#f47230', '#00b894')):
    I = gen.make_instance(fam, 100, 1); h = np.histogram(I['e'], bins=np.arange(0, 490, 10))[0] / 10.0
    ax.bar(np.arange(0, 480, 10) + 5, h, width=8.5, color=c); ax.set_title(f'{fam}  {gen.FAMILIES[fam]["name"].split(":")[0]}', fontsize=8.5, loc='left'); ax.set_xlabel('time of window start (min)')
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', color='#e6e6e6', lw=0.6); ax.set_axisbelow(True)
axs[0].set_ylabel('requests per minute (10-min bins)'); fig.tight_layout(); fig.savefig(FIGS / 'fig7_density.png', dpi=200); plt.close(fig)
print('ok'); print(open(TABLES / 'tbl_structure.md').read()[:600])
