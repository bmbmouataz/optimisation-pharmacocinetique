"""Figures et tableau récapitulatif à partir de resultats/benchmark.csv."""
import csv
from collections import defaultdict

import numpy as np

from pkopt import donnees
from pkopt.graphiques import BLEUS, SERIES, TEXTE, TEXTE_2, enregistrer, plt

RES = donnees.RACINE / "resultats"
NIVEAUX = list(donnees.NIVEAUX_BRUIT)
ETIQ = {"sans": "sans bruit", "faible": "5 %", "moyen": "10 %", "fort": "25 %"}

lignes = list(csv.DictReader(open(RES / "benchmark.csv", encoding="utf-8")))
METHODES = list(dict.fromkeys(l["methode"] for l in lignes))
par = defaultdict(list)
for l in lignes:
    par[l["methode"], l["bruit"]].append(l)


def colonne(methode, bruit, cle):
    return np.array([float(l[cle]) for l in par[methode, bruit]])


def figure_reussite():
    taux = np.array([[100 * colonne(m, n, "succes").mean() for n in NIVEAUX] for m in METHODES])
    evals = [np.median(np.concatenate([colonne(m, n, "n_eval") for n in NIVEAUX])) for m in METHODES]

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.imshow(taux, cmap=BLEUS, vmin=0, vmax=100, aspect="auto")
    for i, j in np.ndindex(taux.shape):
        ax.text(j, i, f"{taux[i, j]:.0f} %", ha="center", va="center", fontsize=9,
                color="white" if taux[i, j] > 55 else TEXTE)
    for i, e in enumerate(evals):
        ax.text(len(NIVEAUX) - 0.35, i, f"{e:,.0f}".replace(",", " "), va="center",
                fontsize=9, color=TEXTE_2)
    ax.text(len(NIVEAUX) - 0.35, -0.85, "évaluations\n(médiane)", fontsize=8.5,
            color=TEXTE_2, va="bottom")
    ax.set_xticks(range(len(NIVEAUX)), [ETIQ[n] for n in NIVEAUX])
    ax.set_yticks(range(len(METHODES)), METHODES)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(-0.5, len(NIVEAUX) + 0.6)
    ax.set_title("Taux de réussite : minimum global atteint à 1 % près", pad=34)
    enregistrer(fig, "02_taux_de_reussite.png")


def figure_convergence():
    courbes = dict(np.load(RES / "convergence.npz"))
    choix = ["Gradient pas optimal", "Gradient pas optimal (log θ)", "Recherche aléatoire",
             "Recuit simulé", "DE + L-BFGS-B (retenue)"]
    j_min = min(c.min() for c in courbes.values())

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for nom, couleur in zip(choix, SERIES):
        c = courbes[nom]
        exces = np.maximum(c - j_min, 1e-12)
        n = np.arange(1, len(c) + 1)
        ax.plot(n, exces, color=couleur, label=nom)
    ax.set(xscale="log", yscale="log", xlabel="évaluations du critère",
           ylabel="J(meilleur point) − J*", ylim=(1e-12, None),
           title="Convergence sur un patient synthétique, bruit de 10 %")
    ax.legend(loc="lower left", fontsize=8.5)
    enregistrer(fig, "03_convergence.png")


def figure_erreur_estimation():
    m = "DE + L-BFGS-B (retenue)"
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, p, nom in zip(axes, ("ka", "ke", "V"), ("ka", "ke", "V")):
        donnees_p = [100 * colonne(m, n, f"err_{p}") for n in NIVEAUX[1:]]
        bp = ax.boxplot(donnees_p, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color=TEXTE, lw=1.5),
                        whiskerprops=dict(color=TEXTE_2), capprops=dict(color=TEXTE_2))
        for b in bp["boxes"]:
            b.set(facecolor="#9ec5f4", edgecolor=SERIES[0])
        ax.set_xticks(range(1, 4), [ETIQ[n] for n in NIVEAUX[1:]])
        ax.set_title(nom)
        ax.set_xlabel("bruit")
    axes[0].set_ylabel("erreur relative sur le paramètre (%)")
    fig.suptitle("Erreur d'estimation au minimum global : elle vient du bruit, pas de "
                 "l'optimiseur", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    enregistrer(fig, "04_erreur_estimation.png")


def tableau():
    entete = ("| Méthode | " + " | ".join(f"Réussite {ETIQ[n]}" for n in NIVEAUX)
              + " | Évaluations (médiane) |")
    sep = "|" + "---|" * (len(NIVEAUX) + 2)
    corps = []
    for m in METHODES:
        taux = [f"{100 * colonne(m, n, 'succes').mean():.0f} %" for n in NIVEAUX]
        n_eval = np.median(np.concatenate([colonne(m, n, "n_eval") for n in NIVEAUX]))
        corps.append(f"| {m} | " + " | ".join(taux) + f" | {n_eval:,.0f} |".replace(",", " "))

    m = "DE + L-BFGS-B (retenue)"
    err = ["", "| Bruit | ka | ke | V |", "|---|---|---|---|"]
    for n in NIVEAUX[1:]:
        cases = []
        for p in ("ka", "ke", "V"):
            e = 100 * colonne(m, n, f"err_{p}")
            cases.append(f"{np.median(e):.1f} % (90 % des cas < {np.percentile(e, 90):.0f} %)")
        err.append(f"| {ETIQ[n]} | " + " | ".join(cases) + " |")

    texte = ("# Benchmark\n\nRéussite : part des jeux de données où chaque paramètre est à "
             "1 % près du meilleur point trouvé toutes méthodes confondues.\n\n"
             + "\n".join([entete, sep] + corps)
             + "\n\n## Erreur relative de la méthode retenue par rapport aux vrais paramètres\n"
             + "\n".join(err) + "\n")
    (RES / "benchmark_resume.md").write_text(texte, encoding="utf-8")
    print(texte)


if __name__ == "__main__":
    figure_reussite()
    figure_convergence()
    figure_erreur_estimation()
    tableau()
