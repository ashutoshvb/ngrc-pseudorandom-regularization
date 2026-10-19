import pandas as pd, numpy as np
from scipy.stats import wilcoxon
def holm(ps):
    ps=np.array(ps); o=np.argsort(ps); m=len(ps); adj=np.empty(m); run=0
    for r,i in enumerate(o):
        run=max(run,(m-r)*ps[i]); adj[i]=min(1,run)
    return adj
d=pd.read_csv("results/ablation_rules_folds.csv")
pd.set_option("display.width",200); pd.set_option("display.float_format","{:.4g}".format)
# ---- A: per-market x TL (N>=8), Holm across all (comparator x cell)
comps=["abs_0.1","abs_1","abs_10","rel_trace_c0.001","rel_n_a0.01","abs_0.01","tuned_orig","tuned_wide"]
rows=[]
for c in comps:
    for (ds,tl),g in d.groupby(["dataset","train_len"]):
        if len(g)<8: continue
        rows.append(dict(comp=c,ds=ds,tl=tl,n=len(g),p=wilcoxon(g["rel_lmax_c1e-5"],g[c]).pvalue,
                         med_rel=g["rel_lmax_c1e-5"].median(),med_c=g[c].median(),rel_better=(g["rel_lmax_c1e-5"]<g[c]).mean()))
t=pd.DataFrame(rows); t["holm"]=holm(t.p.values)
print("A: tests",len(t))
print(t.groupby("comp").agg(min_p=("p","min"),min_holm=("holm","min"),n_sig=("holm",lambda s:(s<0.05).sum())))
print(t[t.holm<0.05][["comp","ds","tl","n","p","holm","med_rel","med_c"]])
# catastrophic by market for headline rules
heads=["rel_lmax_c1e-5","abs_0.01","abs_0.1","abs_1","abs_10","rel_trace_c0.001","tuned_orig","tuned_wide","oracle"]
s=pd.read_csv("results/ablation_rules_summary.csv").set_index("rule")
print(s.loc[heads,["mean","median","max","n_cat"]])
print("pooled n=",len(d))
# ---- B: noise, cell-level (dataset x TL, N>=8), fold-mean over reps
n=pd.read_csv("results/ablation_noise_runs.csv")
models=["abs_0_01","abs_0_1","tuned","rel_1e5"]
f=n.groupby(["dataset","train_len","fold","sigma"])[models].mean().reset_index()
rows=[]
for m in models:
  for sg in [0.01,0.03,0.1,0.3]:
    for (ds,tl),g in f.groupby(["dataset","train_len"]):
        if g.fold.nunique()<8: continue
        a=g[g.sigma==sg].set_index("fold")[m]; b=g[g.sigma==0].set_index("fold")[m]
        rows.append(dict(model=m,sigma=sg,ds=ds,tl=tl,p=wilcoxon(a,b).pvalue,chg=(a-b).median(),worse=(a>b).mean()))
tn=pd.DataFrame(rows); tn["holm"]=holm(tn.p.values)
print("B tests",len(tn))
print(tn.groupby(["model","sigma"]).agg(min_p=("p","min"),min_holm=("holm","min"),n_cells=("p","size"),cells_worse=("chg",lambda s:(s>0).sum()),cells_sig_worse=("holm",lambda s:(s<0.05).sum())))
# improvement test: any cell where noise significantly IMPROVES?
print("cells significantly improved:",((tn.holm<0.05)&(tn.chg<0)).sum())
# catastrophic RATES per run
for m in models:
    print(m, {sg: round(100*(g[m]>5).mean(),1) for sg,g in n.groupby("sigma")})
tn.to_csv("results/ablation_noise_cells.csv",index=False); t.to_csv("results/ablation_rules_cells.csv",index=False)
