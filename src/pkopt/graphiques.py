import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .donnees import RACINE

FOND = "#fcfcfb"
TEXTE = "#0b0b0b"
TEXTE_2 = "#52514e"
GRILLE = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRIS = "#a3a29b"
BLEUS = LinearSegmentedColormap.from_list(
    "bleus", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

DOSSIER = RACINE / "figures"

plt.rcParams.update({
    "figure.facecolor": FOND, "axes.facecolor": FOND, "savefig.facecolor": FOND,
    "axes.edgecolor": GRILLE, "axes.labelcolor": TEXTE_2, "text.color": TEXTE,
    "xtick.color": TEXTE_2, "ytick.color": TEXTE_2, "axes.grid": True,
    "grid.color": GRILLE, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "font.size": 9.5, "lines.linewidth": 2,
    "legend.frameon": False, "figure.dpi": 110,
})


def enregistrer(fig, nom):
    DOSSIER.mkdir(exist_ok=True)
    fig.savefig(DOSSIER / nom, dpi=150, bbox_inches="tight")
    plt.close(fig)
