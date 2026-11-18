"""Estimation sur les 12 patients de l'étude théophylline, avec incertitude.

Sorties : resultats/donnees_reelles.csv, figures 05 et 06.
"""
import csv

import numpy as np
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit

from pkopt import donnees, incertitude, modele, objectif, strategie
from pkopt.graphiques import SERIES, TEXTE, TEXTE_2, enregistrer, plt

SORTIE = donnees.RACINE / "resultats"
B = 1000
POPULATION = np.array([1.5, 0.086, 32.0])  # valeurs de population publiées (NONMEM)


def reference_curve_fit(p):
    """Moindres carrés non linéaires de scipy, sans bornes (Levenberg-Marquardt)."""
    popt, _ = curve_fit(lambda t, ka, ke, V: modele.concentration([ka, ke, V], t, p.dose),
                        p.t, p.y, p0=POPULATION, maxfev=20_000)
    return modele.branche_physiologique(popt)


def r2(theta, p):
    r = objectif.residus(theta, p)
    return 1 - r @ r / np.sum((p.y - p.y.mean()) ** 2)


def figure_ajustements(patients, thetas, tirages, nom="05_ajustements_patients.png",
                       titre="Ajustement sur les 12 patients : mesures, modèle estimé et bande "
                             "de confiance à 95 % (bootstrap)"):
    t = np.linspace(0, 25, 300)
    fig, axes = plt.subplots(3, 4, figsize=(12, 7.8), sharex=True, sharey=True)
    for i, (ax, p, th, tb) in enumerate(zip(axes.flat, patients, thetas, tirages)):
        courbes = np.array([modele.concentration(x, t, p.dose) for x in tb[::5]])
        bas, haut = np.percentile(courbes, [2.5, 97.5], axis=0)
        ax.fill_between(t, bas, haut, color=SERIES[0], alpha=0.18, lw=0)
        ax.plot(t, modele.concentration(th, t, p.dose), color=SERIES[0])
        ax.plot(p.t, p.y, "o", ms=4.5, color=TEXTE, mfc=TEXTE, zorder=3)
        ax.set_title(f"Patient {i + 1}", fontsize=9.5)
        ax.text(0.97, 0.94, f"ka = {th[0]:.2f}\nke = {th[1]:.3f}\nV = {th[2]:.1f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=TEXTE_2)
    for ax in axes[-1]:
        ax.set_xlabel("temps (h)")
    for ax in axes[:, 0]:
        ax.set_ylabel("concentration (mg/L)")
    fig.suptitle(titre, x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    enregistrer(fig, nom)


def figure_identifiabilite(thetas, tirages, choix=(0, 8)):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, i in zip(axes, choix):
        tb = tirages[i]
        ax.scatter(tb[:, 0], tb[:, 2], s=6, color=SERIES[0], alpha=0.35, lw=0)
        ax.plot(*thetas[i][[0, 2]], "o", ms=8, mfc="white", mec=SERIES[1], mew=2)
        ax.set(xscale="log", xlabel="ka (h⁻¹, échelle log)", ylabel="V (L)",
               title=f"Patient {i + 1} : {B} réestimations bootstrap")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(ScalarFormatter())
        ax.tick_params(axis="x", which="minor", labelsize=8)
    axes[1].annotate("borne ka = 20", (20, axes[1].get_ylim()[1]), xytext=(-4, -12),
                     textcoords="offset points", ha="right", fontsize=8.5, color=TEXTE_2)
    fig.tight_layout()
    enregistrer(fig, "06_identifiabilite_ka.png")


if __name__ == "__main__":
    SORTIE.mkdir(exist_ok=True)
    rng = np.random.default_rng(2026)
    patients = donnees.charger_theophylline()
    reels = donnees.charger_theophylline(dose_reelle=True)

    lignes, thetas, tirages = [], [], []
    for i, (p, pr) in enumerate(zip(patients, reels), 1):
        th = strategie.estimer(p).x
        ref = reference_curve_fit(p)
        ic = incertitude.asymptotique(th, p)
        tb = incertitude.bootstrap(th, p, B, rng)
        bas_b, haut_b = np.percentile(tb, [2.5, 97.5], axis=0)
        V_reel = strategie.estimer(pr).x[2]
        thetas.append(th)
        tirages.append(tb)
        lignes.append({
            "patient": i, "ka": th[0], "ke": th[1], "V": th[2],
            "J": objectif.critere(th, p), "R2": r2(th, p),
            "t_max_h": modele.t_max(th), "demi_vie_h": modele.demi_vie(th),
            **{f"{n}_ic_bas": ic["bas"][k] for k, n in enumerate(modele.NOMS)},
            **{f"{n}_ic_haut": ic["haut"][k] for k, n in enumerate(modele.NOMS)},
            **{f"{n}_boot_bas": bas_b[k] for k, n in enumerate(modele.NOMS)},
            **{f"{n}_boot_haut": haut_b[k] for k, n in enumerate(modele.NOMS)},
            "corr_ka_V": ic["correlation"][0, 2], "corr_ka_ke": ic["correlation"][0, 1],
            "ecart_curve_fit": np.max(np.abs(ref / th - 1)),
            "dose_reelle_mg": pr.dose, "V_dose_reelle": V_reel,
        })
        print(f"patient {i:2d} : ka = {th[0]:6.3f} [{ic['bas'][0]:6.3f} ; {ic['haut'][0]:7.3f}]  "
              f"ke = {th[1]:.4f} [{ic['bas'][1]:.4f} ; {ic['haut'][1]:.4f}]  "
              f"V = {th[2]:5.2f} [{ic['bas'][2]:5.2f} ; {ic['haut'][2]:5.2f}]  "
              f"R² = {lignes[-1]['R2']:.3f}  écart curve_fit = {lignes[-1]['ecart_curve_fit']:.1e}")

    with open(SORTIE / "donnees_reelles.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lignes[0]))
        w.writeheader()
        for l in lignes:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in l.items()})

    thetas = np.array(thetas)
    moy = np.exp(np.log(thetas).mean(axis=0))
    print(f"\nmoyenne géométrique : ka = {moy[0]:.2f}, ke = {moy[1]:.4f}, V = {moy[2]:.1f} "
          f"(population publiée : {POPULATION})")
    figure_ajustements(patients, thetas, tirages)
    figure_identifiabilite(thetas, tirages)
