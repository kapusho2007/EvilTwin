"""Figure C1: seed stability (fold-mean test F1 and PR-AUC +- fold SD per seed) from cv_sensitivity_<ds>.csv."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as C

ARMS = [("awid3", "XGB"), ("awid3", "AE"), ("awid3", "LSTM"), ("rids", "XGB"), ("rids", "LSTM")]
fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
for ax, metric in zip(axes, ("F1", "PR_AUC")):
    for i, (ds, m) in enumerate(ARMS):
        d = pd.read_csv(C.OUT_DIR / f"cv_sensitivity_{ds}.csv")
        g = d[(d.experiment == "seed") & (d.model == m)].groupby("value")[metric].agg(["mean", "std"])
        x = i + (g.index.astype(int) - 2) * 0.1
        ax.errorbar(x, g["mean"], g["std"], fmt="o", ms=3, capsize=2, color="k",
                    label="one seed: mean ± SD over 5 test folds" if i == 0 else None)
    ax.set_xticks(range(len(ARMS)), [f"{m}\n{ds.upper()}" for ds, m in ARMS], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("model, dataset (5 points = seeds 0-4)", fontsize=8)
    ax.set_ylabel(f"test {metric.replace('_', '-')} (fold mean)", fontsize=8)
    ax.set_title(f"test {metric.replace('_', '-')}, seeds 0-4", fontsize=9)
    ax.tick_params(labelsize=8)
axes[1].legend(*axes[0].get_legend_handles_labels(), fontsize=7, loc="lower left", frameon=False)
fig.tight_layout()
fig.savefig(C.ROOT / "report" / "figures" / "figC1_seeds.png", dpi=200)
