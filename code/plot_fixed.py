import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

TRAIN_LENS = [60, 120, 250, 500, 1000]
FILENAMES = [f"full_sp500_tl{tl}.csv" for tl in TRAIN_LENS]

# This script lives in code/; the release layout puts the data in a sibling
# results/ directory. That relative path is fixed and known, so use it
# directly rather than guessing from the current working directory (which
# depends on how the script happens to be launched and caused the bugs
# fixed above). Fall back to the script's own directory only for the case
# where someone has copied the CSVs to sit right next to the script.
script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "results"
if not all((data_dir / f).exists() for f in FILENAMES):
    data_dir = script_dir
if not all((data_dir / f).exists() for f in FILENAMES):
    raise FileNotFoundError(
        f"Could not find {FILENAMES} in {script_dir.parent / 'results'} or {script_dir}.\n"
        "Expected the packaged repo layout: code/plot_fixed.py next to a sibling "
        "results/ directory containing the full_sp500_tl*.csv files."
    )

# Built from the per-training-length outputs of run_full_pipeline.py (dataset="sp500"),
# which is the single canonical source of the corrected non-overlapping-fold results.
df_res = pd.concat(
    [pd.read_csv(data_dir / f"full_sp500_tl{tl}.csv") for tl in TRAIN_LENS],
    ignore_index=True,
)
models = ["persistence", "AR1", "NGRC_fixed", "NGRC_tuned", "NGRC_pseudorandom", "ESN"]

agg = df_res.groupby("train_len").agg(
    **{f"nrmse_{m}_med": (f"nrmse_{m}", "median") for m in models},
    **{f"nrmse_{m}_q25": (f"nrmse_{m}", lambda s: s.quantile(0.25)) for m in models},
    **{f"nrmse_{m}_q75": (f"nrmse_{m}", lambda s: s.quantile(0.75)) for m in models},
    n_folds=("fold", "count"),
).reset_index()

colors = dict(persistence="gray", AR1="C2", NGRC_fixed="C0", NGRC_tuned="C5",
              NGRC_pseudorandom="C6", ESN="C3")
markers = dict(persistence="x", AR1="d", NGRC_fixed="o", NGRC_tuned="P",
               NGRC_pseudorandom="^", ESN="s")

fig, ax = plt.subplots(figsize=(9.5, 5.8))
for i, m in enumerate(models):
    lo = agg[f"nrmse_{m}_med"] - agg[f"nrmse_{m}_q25"]
    hi = agg[f"nrmse_{m}_q75"] - agg[f"nrmse_{m}_med"]
    x = agg.train_len * (1 + 0.015 * (i - len(models) / 2))
    ax.errorbar(x, agg[f"nrmse_{m}_med"], yerr=[lo, hi], marker=markers[m], capsize=2,
                color=colors[m], label=m)
ax.set_xscale("log")
ax.set_ylim(0, 4)
ax.set_xlabel("training length (trading days)")
ax.set_ylabel("NRMSE (20-day-ahead forecast), median + IQR")
ax.set_title("S&P 500 realized volatility: NGRC variants vs ESN vs baselines\n"
              "(non-overlapping folds; N per train_len shown on x-axis)")
for _, row in agg.iterrows():
    ax.annotate(f"N={int(row.n_folds)}", (row.train_len, 3.85), ha="center", fontsize=8, color="dimgray")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
out_path = data_dir / "sp500_ngrc_variants_FIXED.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"wrote {out_path}")
print("median NRMSE by model/train_len (non-overlapping folds):")
print(agg[["train_len","n_folds"] + [f"nrmse_{m}_med" for m in models]].to_string(index=False))
