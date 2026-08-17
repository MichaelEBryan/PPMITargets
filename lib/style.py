import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from . import paths

NAVY = "#2a5a8a"
DEEP = "#1d4166"
MID = "#7ea2c0"
PALE = "#c5d7e4"
FILL = "#e8eff4"
GREY = "#8a8a86"
GREY_L = "#d6d6d2"
GREY_XL = "#eeeeea"
INK = "#2b2b2b"
WARM = "#b07d3a"
SAGE = "#5c7a5e"

SEQ = LinearSegmentedColormap.from_list("clin", ["#f7fafc", FILL, PALE, MID, NAVY, DEEP])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#b0b0ac",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8.4, "ytick.labelsize": 8.4,
    "xtick.color": "#b0b0ac", "ytick.color": "#b0b0ac",
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3.4, "ytick.major.size": 3.4,
    "legend.fontsize": 8.2, "legend.frameon": False,
    "legend.handlelength": 1.2, "legend.handletextpad": 0.55,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,
    "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
    "text.color": INK, "pdf.fonttype": 42,
})


def leader(ax, text, xy, xytext, color=NAVY, fontsize=8.8, ha="left"):
    ax.annotate(text, xy=xy, xytext=xytext, color=color, fontsize=fontsize,
                va="center", ha=ha,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.7,
                                shrinkA=1, shrinkB=3))


def panel_label(fig, ax, s, x=0.012, dy=0.010):
    p = ax.get_position()
    fig.text(x, p.y1 + dy, s, fontsize=11.5, fontweight="bold", va="bottom",
             ha="left", color=INK)


def hide_y_spine(ax):
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def save(fig, name, outdir=str(paths.ROOT / "figures")):
    for ext in ("pdf", "png"):
        fig.savefig(f"{outdir}/{name}.{ext}")
    plt.close(fig)
    print(f"  {name}")
