import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "results"
if not (data_dir / "sensitivity_c.csv").exists():
    data_dir = script_dir
if not (data_dir / "sensitivity_c.csv").exists():
    raise FileNotFoundError(
        f"Could not find sensitivity_c.csv in {script_dir.parent / 'results'} or {script_dir}.\n"
        "Run sensitivity_c.py first."
    )

df = pd.read_csv(data_dir / "sensitivity_c.csv")
CATASTROPHIC_THRESHOLD = 5.0
CHOSEN_C = 1e-5

summary = df.groupby("c")["nrmse"].agg(
    mean_nrmse="mean", median_nrmse="median", max_nrmse="max", n_folds="count",
).reset_index()
summary["n_catastrophic"] = df.groupby("c")["nrmse"].apply(
    lambda s: (s > CATASTROPHIC_THRESHOLD).sum()
).values
summary["frac_catastrophic"] = summary["n_catastrophic"] / summary["n_folds"]
summary = summary.sort_values("c")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

ax = axes[0]
ax.plot(summary.c, summary.mean_nrmse, marker="o", color="C0", label="mean")
ax.plot(summary.c, summary.median_nrmse, marker="s", color="C2", label="median")
ax.set_xscale("log")
ax.set_xlabel(r"$c$")
ax.set_ylabel("NRMSE across all folds\n(both markets, all train lengths, $N=162$ per $c$)")
ax.axvline(CHOSEN_C, color="gray", linestyle="--", linewidth=1)
ax.annotate("chosen\n$c=10^{-5}$", xy=(CHOSEN_C, ax.get_ylim()[1]), xytext=(8, -4),
            textcoords="offset points", fontsize=7.5, color="dimgray",
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
ax.legend(fontsize=8)
ax.set_title("(a) central tendency", fontsize=10)

ax = axes[1]
ax.plot(summary.c, summary.frac_catastrophic * 100, marker="o", color="C3")
ax.set_xscale("log")
ax.set_xlabel(r"$c$")
ax.set_ylabel(f"folds with NRMSE $>{CATASTROPHIC_THRESHOLD:.0f}$ (\\%)")
ax.axvline(CHOSEN_C, color="gray", linestyle="--", linewidth=1)
ax.annotate("chosen\n$c=10^{-5}$", xy=(CHOSEN_C, ax.get_ylim()[1]), xytext=(8, -4),
            textcoords="offset points", fontsize=7.5, color="dimgray",
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
ax.set_title("(b) catastrophic-failure rate", fontsize=10)

fig.suptitle(r"Sensitivity of the conditioning-relative ridge rule ($\alpha=c\cdot\lambda_{\max}$) to $c$", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out_path = data_dir / "sensitivity_c.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"wrote {out_path}")
print(summary.to_string(index=False))
