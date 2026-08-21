# Transferability of Noise-Robustness in Pseudorandom-Projection NGRC

Code, data, and results supporting the manuscript "Transferability of
Noise-Robustness in Pseudorandom-Projection NGRC: From Synthetic Chaos to
Real-World Financial Volatility" (currently in preparation/under review).
A link to the preprint or published version will be added here once
available.

## Repository layout

```
code/           Python source. Every script resolves its inputs/outputs relative to
                its own file location (code/../data/, code/../results/), so they can
                be run from anywhere as `python /path/to/code/<script>.py ...`.
                The only requirement is that the code/, data/, and results/ folders
                stay siblings of each other, as they are in this release.
  ngrc_lib.py               Core NGRC / pseudorandom-projection NGRC / ESN implementations
  relative_ridge.py         Conditioning-aware relative-ridge NGRC variant (the paper's fix)
  run_full_pipeline.py      Main corrected (non-overlapping-fold) backtest, all 8 models.
                             Usage: python run_full_pipeline.py <sp500|nasdaq> <train_len>
                             Reads data/{dataset}_{realized_vol,logret}.npy,
                             writes results/full_{dataset}_tl{train_len}.csv
  significance_with_har.py  Holm-Bonferroni significance testing, 7-baseline family.
                             Reads all ten results/full_*_tl*.csv, writes results/sig_*.csv
  plot_fixed.py              Regenerates Figure 1 (results/sp500_ngrc_variants_FIXED.png)
                             from results/full_sp500_tl*.csv
  sensitivity_c.py           c-sensitivity check (Sec. "choice of c is not fragile"): reruns
                             the relative-ridge fit for c in {1e-7...1e-2} across all 162
                             non-overlapping folds (both markets, all train lengths). Writes
                             results/sensitivity_c.csv and results/sensitivity_c_summary.csv
  plot_sensitivity_c.py      Regenerates the c-sensitivity figure (results/sensitivity_c.png)
                             from results/sensitivity_c.csv
  plot_intervention.py       Regenerates the intervention figure (results/intervention_mechanism.png)
                             from results/intervention_force_alpha.csv and intervention_vary_M.csv

  repro_folds.py            Legacy/standalone: reproduces the *original* overlapping-fold
                             sampling used before the non-overlapping-fold correction
                             (Sec. 2.3). Loads S&P 500 data directly from the `arch`
                             package (no input files needed), writes
                             results/{realized_vol.npy,folds.pkl}
  rerun_widegrid.py         Legacy/standalone: the widened-ridge-grid sweep behind the
                             "practical fix test" paragraph in Sec. 2.5. Run
                             repro_folds.py first; reads its output from results/,
                             writes results/sp500_ngrc_pseudorandom_widegrid.csv
  verify.py                 Legacy/standalone: spot-checks two of the catastrophic-failure
                             NRMSE values quoted in Sec. 3.3/Table 3 (182.69, 71.25) against
                             a fresh fit. Run repro_folds.py first.
  diagnose_folds.py         Legacy/standalone: regenerates Tables 3-5 in full (diagnostic
                             fold statistics, and the alpha-forcing/M-varying
                             interventions) from source. Run repro_folds.py first; reads
                             its output from results/, writes
                             results/{diagnostic_condition_table,intervention_force_alpha,
                             intervention_vary_M}.csv

data/           Raw input series
  {sp500,nasdaq}_realized_vol.npy   20-day trailing realized volatility
  {sp500,nasdaq}_logret.npy         Daily log returns (used by the GARCH baseline)

results/        Output CSVs and figure referenced by name in the manuscripts
  full_both_datasets.csv                    Combined 8-model, 2-market backtest results
  full_{sp500,nasdaq}_tl{60,120,250,500,1000}.csv   Per-fold results by training length
  sig_{sp500,nasdaq}_NGRC_pseudorandom[_relridge].csv   Holm-adjusted significance tables
  diagnostic_condition_table.csv            Table 3 (catastrophic vs. stable fold diagnostics)
  intervention_{force_alpha,vary_M}.csv     Tables 4-5 and the intervention figure (interventions)
  sp500_ngrc_variants_FIXED.png             Figure 1
  intervention_mechanism.png                Intervention (alpha/M) figure
  sensitivity_c.csv, sensitivity_c_summary.csv   Per-fold and summary results for the c-sensitivity check
  sensitivity_c.png                         c-sensitivity figure
```

## Reproducing the results

Commands below use bash syntax. On Windows, run them in Git Bash or WSL
rather than Command Prompt or PowerShell (or replace the `for ... done`
loops with five individual calls, one per training length).

```bash
pip install -r requirements.txt

# Run the corrected, non-overlapping-fold backtest for every dataset/training length
# (each call reads data/*.npy and writes results/full_<dataset>_tl<n>.csv):
for tl in 60 120 250 500 1000; do python code/run_full_pipeline.py sp500 $tl; done
for tl in 60 120 250 500 1000; do python code/run_full_pipeline.py nasdaq $tl; done

# Combine and test significance (Holm-Bonferroni, 7-baseline family incl. GARCH and HAR):
python code/significance_with_har.py

# Regenerate Figure 1:
python code/plot_fixed.py

# Regenerate the c-sensitivity check and figure:
python code/sensitivity_c.py
python code/plot_sensitivity_c.py

# Regenerate the intervention figure (from the results/intervention_*.csv
# files already provided; these are not regenerated by run_full_pipeline.py):
python code/plot_intervention.py
```

These commands can also be run with a full path to the script, from any
working directory (`python /some/path/code/run_full_pipeline.py sp500 60`).

Concatenating all ten `full_<dataset>_tl<n>.csv` files reproduces
`full_both_datasets.csv`. `significance_with_har.py` reads those files and
reproduces the `sig_*.csv` significance tables and the Holm-adjusted
p-values reported in the manuscripts.

To reproduce the diagnostic/practical-fix-test numbers in Sec. 2.5/3.3
instead (the original overlapping-fold sampling, not the main pipeline
above), run in order: `repro_folds.py`, then `rerun_widegrid.py`,
`verify.py`, and/or `diagnose_folds.py`. `diagnose_folds.py` regenerates
Tables 3-5 in full from source; its output matches the shipped
`results/diagnostic_condition_table.csv`, `intervention_force_alpha.csv`,
and `intervention_vary_M.csv` exactly.

## Citation

If you use this code or data, please cite the manuscript (details to be
added on acceptance).
