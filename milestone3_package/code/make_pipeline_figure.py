"""Figure 1.1: the five-stage pipeline as implemented (same order as the code)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import config as C

BOXES = [  # (x, y, w, h, title, body, dataset_aware)
    (0.2, 3.2, 2.4, 1.5, "1. Data Ingestion", "run_ingest.py / ingestion.py\nraw CSV -> unified frame table\nlabels, provenance filters", True),
    (3.0, 3.2, 2.4, 1.5, "2. Preprocessing", "preprocess.clean()\ndedup, range checks,\ndrop frames with no AP", False),
    (5.8, 3.2, 2.6, 1.5, "3. Feature Sel. & Eng.", "build_windows() / build_sequences()\nAP x 5 s window (View A)\n64 frames evenly spaced\nover the window (View B)", False),
    (5.8, 0.4, 2.6, 1.5, "4. Model Training", "Normalizer fit on train only\nXGBoost | LSTM\nOC-SVM | Autoencoder", False),
    (8.8, 0.4, 2.4, 1.5, "5. Evaluation", "P, R, F1, FPR, FNR,\nROC-AUC, PR-AUC\nerror analysis", False),
]


def main():
    fig, ax = plt.subplots(figsize=(11.6, 5.2))
    for x, y, w, h, title, body, aware in BOXES:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                    fc="#e8e8e8" if aware else "white", ec="black", lw=1.2))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center", fontsize=10.5, weight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.2, body, ha="center", va="center", fontsize=8.5)
    arrow = dict(arrowstyle="-|>", lw=1.2, color="black")
    ax.annotate("", (3.0, 3.95), (2.6, 3.95), arrowprops=arrow)
    ax.annotate("", (5.8, 3.95), (5.4, 3.95), arrowprops=arrow)
    ax.plot([7.1, 7.1, 2.8], [3.16, 2.6, 2.6], color="black", lw=1.2)        # features -> split
    ax.annotate("", (2.8, 1.98), (2.8, 2.62), arrowprops=arrow)
    ax.annotate("", (8.8, 1.15), (8.4, 1.15), arrowprops=arrow)
    # episode-blocked purged 5-fold between features and training (fold k = 2 shown)
    ax.add_patch(FancyBboxPatch((0.2, 0.4), 5.2, 1.5, boxstyle="round,pad=0.04", fc="white", ec="black", ls="--"))
    ax.text(2.8, 1.62, "Episode-blocked purged 5-fold (fold k = 2 shown)", ha="center", fontsize=10, weight="bold")
    segs = [("train", 1.55), ("P", 0.22), ("test\n(locked)", 0.85), ("P", 0.22), ("val", 0.85), ("P", 0.22),
            ("train", 0.95)]
    for i, (lab, w) in enumerate(segs):
        x0 = 0.35 + sum(v for _, v in segs[:i])
        ax.add_patch(plt.Rectangle((x0, 0.62), w - 0.04, 0.6, fc="#d0d0d0" if lab == "P" else "white", ec="black"))
        ax.text(x0 + w / 2, 0.92, lab, ha="center", va="center", fontsize=8)
    ax.text(2.8, 0.47, "chunk = 1/5 of attack span + 1/5 of each session's negatives;  P = 5 s purge",
            ha="center", fontsize=7)
    ax.annotate("", (5.8, 1.15), (5.4, 1.15), arrowprops=arrow)
    ax.text(0.2, 4.95, "grey = dataset-aware (only stage 1); white = dataset-agnostic, reads the unified schema only",
            fontsize=8.5, style="italic")
    ax.set_xlim(0, 11.4); ax.set_ylim(0.2, 5.15); ax.axis("off")
    out = C.ROOT / "report" / "figures" / "fig1_1_pipeline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
