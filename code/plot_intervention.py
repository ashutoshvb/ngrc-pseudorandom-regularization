"""
Visualizes the two interventions behind Sec. "Mechanistic diagnosis"
(Tables 4-5 in the manuscript): forcing alpha at fixed M, and varying M at
fixed alpha, for the three catastrophic diagnostic folds.

Usage: python plot_intervention.py
Reads results/intervention_force_alpha.csv, results/intervention_vary_M.csv
Writes results/intervention_mechanism.png
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "results"
needed = ["intervention_force_alpha.csv", "intervention_vary_M.csv"]
if not all((data_dir / f).exists() for f in needed):
    data_dir = script_dir
if not all((data_dir / f).exists() for f in needed):
    raise FileNotFoundError(
        f"Could not find {needed} in {script_dir.parent / 'results'} or {script_dir}.\n"
        "Expected the packaged repo layout with a sibling results/ directory."
    )

df_alpha = pd.read_csv(data_dir / "intervention_force_alpha.csv")
df_M = pd.read_csv(data_dir / "intervention_vary_M.csv")

fold_labels = {
    (1000, 3): "TL=1000, f.3",
    (250, 10): "TL=250, f.10",
    (60, 23): "TL=60, f.23",
}
colors = {(1000, 3): "C0", (250, 10): "C1", (60, 23): "C2"}
markers = {(1000, 3): "o", (250, 10): "s", (60, 23): "^"}

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

ax = axes[0]
for key, label in fold_labels.items():
    tl, fold = key
    sub = df_alpha[(df_alpha.train_len == tl) & (df_alpha.fold == fold)].sort_values("alpha")
    ax.plot(sub.alpha, sub.nrmse, marker=markers[key], color=colors[key], label=label)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"forced $\alpha$ (M held fixed at 100)")
ax.set_ylabel("NRMSE (log scale)")
ax.axhline(5, color="gray", linestyle=":", linewidth=1)
ax.legend(fontsize=8)
ax.set_title("(a) forcing regularization strength", fontsize=10)

ax = axes[1]
for key, label in fold_labels.items():
    tl, fold = key
    sub = df_M[(df_M.train_len == tl) & (df_M.fold == fold)].sort_values("M")
    ax.plot(sub.M, sub.nrmse, marker=markers[key], color=colors[key], label=label)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"projection dimension $M$ (fixed $\alpha=0.01$)")
ax.set_ylabel("NRMSE (log scale)")
ax.axhline(5, color="gray", linestyle=":", linewidth=1)
ax.legend(fontsize=8)
ax.set_title("(b) varying projection dimension", fontsize=10)

fig.suptitle("Interventions on the three catastrophic diagnostic folds", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out_path = data_dir / "intervention_mechanism.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"wrote {out_path}")
