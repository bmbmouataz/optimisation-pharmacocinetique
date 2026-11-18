"""Forme du critère : non-convexité (flip-flop) et conditionnement.

U est linéaire en 1/V, donc pour (ka, ke) fixés le meilleur V se calcule
exactement. On trace le critère profilé J(ka, ke) = min_V J(ka, ke, V).
"""
import csv

import numpy as np

from pkopt import donnees, modele, objectif, strategie
from pkopt.graphiques import BLEUS, SERIES, TEXTE, TEXTE_2, enregistrer, plt

SORTIE = donnees.RACINE / "resultats"


def critere_profile(ka, ke, p):
    g = modele.concentration([ka, ke, 1.0], p.t, p.dose)  # U pour V = 1
    inv_v = max((g @ p.y) / (g @ g), 1e-12)
    r = inv_v * g - p.y
    return r @ r


def figure_paysage(p, theta):
    ka = np.geomspace(0.02, 10, 260)
    ke = np.geomspace(0.02, 10, 260)
    Z = np.array([[critere_profile(a, e, p) for a in ka] for e in ke])
    miroir = np.array([theta[1], theta[0], theta[2] * theta[1] / theta[0]])

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    niveaux = np.geomspace(Z.min() * 1.02, np.percentile(Z, 95), 18)
    cs = ax.contourf(ka, ke, Z, levels=niveaux, cmap=BLEUS.reversed(), extend="max")
    ax.contour(ka, ke, Z, levels=niveaux, colors="white", linewidths=0.4, alpha=0.6)
    ax.plot([0.02, 10], [0.02, 10], color=TEXTE_2, lw=1, ls="--")
    ax.text(3.2, 4.6, "ka = ke", color=TEXTE_2, rotation=45, fontsize=8.5)
    etiquettes = [(theta, f"ka > ke : ka = {theta[0]:.2f}, ke = {theta[1]:.3f}, V = {theta[2]:.1f} L", (-150, -38)),
                  (miroir, f"ka < ke : ka = {miroir[0]:.3f}, ke = {miroir[1]:.2f}, V = {miroir[2]:.2f} L", (-20, 22))]
    for th, nom, decalage in etiquettes:
        ax.plot(th[0], th[1], "o", ms=9, mfc="white", mec=SERIES[1], mew=2.2)
        ax.annotate(nom, th[:2], xytext=decalage, textcoords="offset points", fontsize=8.5,
                    color=TEXTE, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"))
    ax.set(xscale="log", yscale="log", xlabel="ka (h⁻¹)", ylabel="ke (h⁻¹)",
           title=f"Critère profilé sur V, {p.nom} (échelle log)")
    ax.grid(False)
    cb = fig.colorbar(cs, ax=ax, shrink=0.85)
    cb.set_label("min_V J(ka, ke, V)")
    cb.set_ticks([])
    enregistrer(fig, "01_paysage_flip_flop.png")
    return miroir


def conditionnement(p, theta):
    _, _, h = objectif.probleme(p)
    _, _, hl = objectif.probleme(p, log=True)
    return np.linalg.cond(h(theta)), np.linalg.cond(hl(np.log(theta)))


if __name__ == "__main__":
    SORTIE.mkdir(exist_ok=True)
    patients = donnees.charger_theophylline()
    p = patients[0]
    theta = strategie.estimer(p).x
    miroir = figure_paysage(p, theta)
    print(f"{p.nom} : θ = {np.round(theta, 4)}, miroir = {np.round(miroir, 4)}, "
          f"J = {objectif.critere(theta, p):.4f} / {objectif.critere(miroir, p):.4f}")

    with open(SORTIE / "conditionnement.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["patient", "cond_hessienne_theta", "cond_hessienne_log_theta"])
        for i, q in enumerate(patients, 1):
            c, cl = conditionnement(q, strategie.estimer(q).x)
            w.writerow([i, f"{c:.3g}", f"{cl:.3g}"])
            print(f"patient {i:2d} : cond(H) = {c:9.3g} en θ, {cl:9.3g} en log θ")
