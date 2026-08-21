from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

# Fixed, known repo layout: this script lives in code/, both its inputs
# (full_*_tl*.csv) and outputs (sig_*.csv) live in the sibling results/
# directory. Resolved relative to the script's own location so it works
# regardless of the caller's current working directory.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

BASELINES = ["persistence", "AR1", "NGRC_fixed", "NGRC_tuned", "ESN", "GARCH", "HAR"]
FOCALS = ["NGRC_pseudorandom", "NGRC_pseudorandom_relridge"]

for dataset in ["sp500", "nasdaq"]:
    df_res = pd.concat(
        [pd.read_csv(RESULTS_DIR / f"full_{dataset}_tl{tl}.csv") for tl in [60, 120, 250, 500, 1000]],
        ignore_index=True,
    )
    for focal in FOCALS:
        sig_rows = []
        for tl, grp in df_res.groupby("train_len"):
            fvals = grp[f"nrmse_{focal}"].values
            for b in BASELINES:
                other = grp[f"nrmse_{b}"].values
                mask = np.isfinite(fvals) & np.isfinite(other)
                n = int(mask.sum())
                d = fvals[mask] - other[mask]
                if n < 8 or np.all(d == 0):
                    p = np.nan
                else:
                    try:
                        _, p = wilcoxon(d)
                    except ValueError:
                        p = np.nan
                sig_rows.append(dict(
                    train_len=tl, compared_to=b, n=n,
                    med_focal=np.median(fvals[mask]) if n else np.nan,
                    med_other=np.median(other[mask]) if n else np.nan,
                    focal_better=bool(np.median(fvals[mask]) < np.median(other[mask])) if n else None,
                    p=p,
                ))
        sig_df = pd.DataFrame(sig_rows)
        testable = sig_df["p"].notna()
        m = testable.sum()
        order = sig_df.loc[testable, "p"].sort_values().index
        holm_p = pd.Series(np.nan, index=sig_df.index)
        running_max = 0.0
        for rank, idx in enumerate(order, start=1):
            adj = (m - rank + 1) * sig_df.loc[idx, "p"]
            running_max = max(running_max, adj)
            holm_p[idx] = min(running_max, 1.0)
        sig_df["holm_p"] = holm_p
        sig_df["sig"] = sig_df["holm_p"] < 0.05
        sig_df.to_csv(RESULTS_DIR / f"sig_{dataset}_{focal}.csv", index=False)
        print(f"=== {dataset} / {focal}: {m} testable comparisons ===")
        print(sig_df.to_string(index=False))
        print()
