import pandas as pd, numpy as np
from scipy.stats import wilcoxon
d = pd.read_csv("results/ablation_rules_folds.csv")
full = pd.read_csv("results/full_both_datasets.csv")
print(full.columns.tolist()[:12])
# 1. reproduction check vs main backtest
m = d.merge(full, on=["dataset","train_len","fold"])
print("n merged", len(m))
for a,b in [("rel_lmax_c1e-5","nrmse_NGRC_pseudorandom_relridge"),("tuned_orig","nrmse_NGRC_pseudorandom")]:
    diff = (m[a]-m[b]).abs()
    print(a, "vs", b, "max abs diff", diff.max(), "n>1e-6:", (diff>1e-6).sum())
# 2. pooled summary
rules = [c for c in d.columns if c not in ("dataset","train_len","fold","lmax","lmin","tr","n")]
rows=[]
for r in rules:
    s=d[r]
    p = wilcoxon(s, d["rel_lmax_c1e-5"]).pvalue if r!="rel_lmax_c1e-5" else np.nan
    rows.append(dict(rule=r, mean=s.mean(), median=s.median(), p95=s.quantile(.95), max=s.max(), n_cat=(s>5).sum(), n_gt2=(s>2).sum(), p_vs_rel=p))
t=pd.DataFrame(rows).sort_values("mean")
pd.set_option("display.width",200); pd.set_option("display.float_format","{:.4g}".format)
print(t.to_string(index=False))
t.to_csv("results/ablation_rules_summary.csv", index=False)
