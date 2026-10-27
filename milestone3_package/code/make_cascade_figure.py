"""Figure 3.1: the hybrid cascade as implemented (same paths as pipeline/cascade.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import config as C

BOX = dict(boxstyle="round,pad=0.04", ec="black", lw=1.2)
ARROW = dict(arrowstyle="-|>", lw=1.2, color="black")
EXIT = dict(arrowstyle="-|>", lw=1.2, color="black", linestyle="dashed")


def box(ax, x, y, w, h, title, body, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, fc=fc, **BOX))
    ax.text(x + w / 2, y + h - 0.26, title, ha="center", va="center", fontsize=10, weight="bold")
    ax.text(x + w / 2, y + (h - 0.3) / 2, body, ha="center", va="center", fontsize=8.2)


def main():
    fig, ax = plt.subplots(figsize=(11.6, 4.6))
    box(ax, 0.1, 1.9, 1.9, 1.3, "AP x 5 s window", "15 features (View A)\n64 frames (View B)", "#e8e8e8")
    box(ax, 2.4, 1.9, 2.3, 1.3, "Stage 1: gate", "Autoencoder or OC-SVM\n(benign-only fit)\nalert iff s >= t_gate\nt_gate: val. recall >= 0.99")
    box(ax, 5.1, 1.9, 2.3, 1.3, "Stage 2: classifiers", "XGBoost (View A)\nLSTM (View B)\neach at its val. max-F1\nthreshold")
    box(ax, 7.8, 1.9, 1.7, 1.3, "agree?", "a_XGB == a_LSTM\n-> take that label")
    box(ax, 7.6, 0.1, 2.1, 1.2, "Stage 3: arbiter", "local LLM (Llama 3.1 8B)\nor one-feature rule\nJSON verdict + evidence")
    box(ax, 10.0, 1.9, 1.3, 1.3, "ALERT", "analyst queue", "#e8e8e8")
    box(ax, 2.6, 0.1, 1.9, 1.2, "BENIGN exit", "no further compute\n(unrecoverable:\ngate FN are final)", "#e8e8e8")

    ax.annotate("", (2.4, 2.55), (2.0, 2.55), arrowprops=ARROW)
    ax.annotate("", (5.1, 2.55), (4.7, 2.55), arrowprops=ARROW)
    ax.annotate("", (3.55, 1.3), (3.55, 1.9), arrowprops=EXIT)                      # gate benign exit
    ax.text(3.45, 1.45, "s < t_gate", ha="right", fontsize=8)
    ax.annotate("", (7.8, 2.55), (7.4, 2.55), arrowprops=ARROW)
    ax.annotate("", (10.0, 2.55), (9.5, 2.55), arrowprops=ARROW)                    # agree on attack
    ax.text(9.75, 2.66, "1", ha="center", fontsize=8)
    ax.plot([8.3, 8.3, 4.2], [1.9, 1.62, 1.62], color="black", lw=1.2, ls="--")    # agree on benign
    ax.annotate("", (4.2, 1.3), (4.2, 1.62), arrowprops=EXIT)
    ax.text(6.3, 1.68, "agree, 0", ha="center", fontsize=8)
    ax.annotate("", (9.0, 1.3), (9.0, 1.9), arrowprops=ARROW)                       # disagree -> arbiter
    ax.text(9.1, 1.55, "disagree", ha="left", fontsize=8)
    ax.annotate("", (4.5, 0.45), (7.6, 0.45), arrowprops=EXIT)                      # arbiter -> benign
    ax.text(6.0, 0.53, "verdict benign, 0", ha="center", fontsize=8)
    ax.plot([9.7, 10.65], [0.7, 0.7], color="black", lw=1.2)                        # arbiter -> alert
    ax.annotate("", (10.65, 1.9), (10.65, 0.7), arrowprops=ARROW)
    ax.text(10.75, 1.2, "verdict\nattack, 1", ha="left", fontsize=8)
    ax.text(0.1, 3.45, "Every threshold is set on the validation fold; the test fold is scored once. "
                       "Dashed = benign exit.", fontsize=8.5, style="italic")
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 3.7)
    ax.axis("off")
    out = C.ROOT / "report" / "figures" / "fig3_1_cascade.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
