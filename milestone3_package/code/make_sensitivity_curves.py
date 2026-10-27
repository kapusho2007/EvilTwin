"""Figure C2: sensitivity curves (Section 1.5): test PR-AUC (test F1 for the AE quantile q) (mean ± SD over the 5 test folds) and
validation PR-AUC per knob value, one knob at a time, from cv_sensitivity_<ds>.csv."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as C

PANELS = [("XGB", "max_depth", "XGBoost: max depth"), ("XGB", "learning_rate", "XGBoost: learning rate"),
          ("OCSVM", "gamma", "OC-SVM: γ"), ("OCSVM", "nu", "OC-SVM: ν"),
          ("AE", "bottleneck", "AE: bottleneck b"), ("AE", "q", "AE: threshold quantile q"),
          ("LSTM", "units", "LSTM: units (h₁, h₂)")]
STYLE = {"awid3": ("k", "o", "AWID3"), "rids": ("0.55", "s", "RIDS")}

fig, axes = plt.subplots(2, 4, figsize=(10, 4.6))
for ax, (model, knob, title) in zip(axes.flat, PANELS):
    for ds, (c, mk, lab) in STYLE.items():
        f = C.OUT_DIR / (f"cv_sensitivity_lstm_{ds}.csv" if model == "LSTM" else f"cv_sensitivity_{ds}.csv")
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[(d.model == model) & (d.experiment == "param") & (d.knob == knob)]
        if d.empty:
            continue
        if model == "LSTM":
            g = d.set_index("value")[["PR_AUC", "PR_AUC_sd", "val_PR_AUC"]].rename(
                columns={"PR_AUC": "mean", "PR_AUC_sd": "std", "val_PR_AUC": "val"})
        else:
            m = "F1" if knob == "q" else "PR_AUC"          # q moves the threshold only, not the ranking
            g = d.groupby("value", sort=False).agg(mean=(m, "mean"), std=(m, "std"), val=("val_PR_AUC", "mean"))
            if knob == "q":
                g["val"] = float("nan")
        x = range(len(g))
        ax.errorbar(x, g["mean"], g["std"], fmt=mk + "-", ms=3, lw=1, capsize=2, color=c, label=f"{lab} test")
        ax.plot(x, g["val"], mk + "--", ms=3, lw=0.8, color=c, mfc="none", label=f"{lab} validation")
        ax.set_xticks(list(x), [str(v).replace(", ", ",").replace(",0)", ")") for v in g.index], fontsize=6.5, rotation=30)
    ax.set_title(title, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("test F1" if knob == "q" else "PR-AUC", fontsize=7.5)
    ax.tick_params(labelsize=7)
leg = axes.flat[-1]
leg.axis("off")
h, l = axes.flat[0].get_legend_handles_labels()
leg.legend(h, l, fontsize=7.5, loc="center", frameon=False,
           title="mean ± SD over 5 folds\n(AWID3 and RIDS)", title_fontsize=7.5)
fig.tight_layout()
fig.savefig(C.ROOT / "report" / "figures" / "figC2_sensitivity.png", dpi=200)
print("saved figC2_sensitivity.png")
