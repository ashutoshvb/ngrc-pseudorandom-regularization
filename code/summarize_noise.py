import pandas as pd, numpy as np
from scipy.stats import wilcoxon
d=pd.read_csv("results/ablation_noise_runs.csv")
pd.set_option("display.width",220); pd.set_option("display.float_format","{:.4g}".format)
models=["abs_0_01","abs_0_1","tuned","rel_1e5"]
# average noise reps within fold first -> one value per fold per sigma (avoids pseudo-replication)
f=d.groupby(["dataset","train_len","fold","sigma"])[models].mean().reset_index()
cat=d.assign(**{m:(d[m]>5) for m in models}).groupby("sigma")[models].sum()
n=d.groupby("sigma").size()
print("runs per sigma",n.to_dict())
print("MEAN NRMSE by sigma"); print(f.groupby("sigma")[models].mean())
print("MEDIAN NRMSE by sigma"); print(f.groupby("sigma")[models].median())
print("Catastrophic runs (>5) by sigma"); print(cat)
# paired test: sigma>0 vs sigma=0 on fold-level means (162 folds)
base=f[f.sigma==0].set_index(["dataset","train_len","fold"])
rows=[]
for s in sorted(f.sigma.unique()):
    if s==0: continue
    cur=f[f.sigma==s].set_index(["dataset","train_len","fold"])
    for m in models:
        diff=(cur[m]-base[m])
        p=wilcoxon(cur[m],base[m]).pvalue
        rows.append(dict(sigma=s,model=m,mean_change=diff.mean(),median_change=diff.median(),frac_improved=(diff<0).mean(),p=p))
t=pd.DataFrame(rows); print(t.to_string(index=False)); t.to_csv("results/ablation_noise_vs_clean.csv",index=False)
# At each sigma: rel vs abs_0.01 head to head
print("rel_1e5 vs abs_0_01 paired p by sigma:", {s:round(wilcoxon(g.rel_1e5,g.abs_0_01).pvalue,4) for s,g in f.groupby("sigma")})
# restrict to short training length where failures live
print("TL=60 mean by sigma"); print(f[f.train_len==60].groupby("sigma")[models].mean())
print("TL=60 catastrophic folds-avg>5"); print(f[f.train_len==60].assign(**{m:(f[f.train_len==60][m]>5) for m in models}).groupby("sigma")[models].sum())
cat.to_csv("results/ablation_noise_cat.csv"); f.groupby("sigma")[models].agg(["mean","median"]).to_csv("results/ablation_noise_summary.csv")
