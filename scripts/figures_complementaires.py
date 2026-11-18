"""Figures 08 à 13 : données, trajectoires du gradient, sensibilité au point de départ
et aux hyperparamètres, estimations contre vrais paramètres, résidus.

Sorties : figures 08 à 13, resultats/points_initiaux.csv et resultats/hyperparametres.csv.
"""
import csv
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from pkopt import deterministes as det
from pkopt import donnees, modele, objectif, strategie
from pkopt import stochastiques as sto
from pkopt.graphiques import GRIS, SERIES, TEXTE, TEXTE_2, enregistrer, plt

RES = donnees.RACINE / "resultats"
THETAS = donnees.patients_synthetiques()
X0 = THETAS.mean(axis=0)
NOMS_BRUIT = {"faible": "bruit 5 %", "moyen": "bruit 10 %", "fort": "bruit 25 %"}
BL = strategie.BORNES_LOG


def patient_bruite(i, niveau="moyen", rep=0):
    """Même tirage du bruit que dans le benchmark."""
    graine = [list(donnees.NIVEAUX_BRUIT).index(niveau), rep, i]
    return donnees.bruiter(THETAS[i], donnees.NIVEAUX_BRUIT[niveau], np.random.default_rng(graine))


def ecart(theta, ref):
    return np.max(np.abs(np.asarray(theta) / ref - 1))


def figure_donnees():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    t = np.linspace(0, 25, 300)
    for p in donnees.charger_theophylline():
        axes[0].plot(p.t, p.y, "-o", ms=3, lw=1.1, color=SERIES[0], alpha=0.55)
    axes[0].set_title("12 patients réels (théophylline)")
    for th in THETAS:
        axes[1].plot(t, modele.concentration(th, t), lw=1.4, color=SERIES[0], alpha=0.55)
    axes[1].set_title("12 patients synthétiques, sans bruit")

    ax = axes[2]
    ax.plot(t, modele.concentration(THETAS[0], t), color=TEXTE, lw=1.6, label="modèle exact")
    for (niveau, nom), couleur in zip(NOMS_BRUIT.items(), SERIES):
        p = patient_bruite(0, niveau)
        ax.plot(p.t, p.y, "o-", ms=5, lw=0.9, color=couleur, label=nom)
    ax.set_title("Patient synthétique 1 et ses trois versions bruitées")
    ax.legend(fontsize=8.5)
    for a in axes:
        a.set_xlabel("temps (h)")
    axes[0].set_ylabel("concentration (mg/L)")
    fig.tight_layout()
    enregistrer(fig, "08_donnees.png")


def figure_trajectoires():
    p = patient_bruite(0)
    ref = strategie.estimer(p).x
    f, g, _ = objectif.probleme(p)
    fl, gl, _ = objectif.probleme(p, log=True)
    nat = det.descente_gradient(f, g, X0, max_iter=300).iteres
    log = np.exp(det.descente_gradient(fl, gl, np.log(X0), max_iter=300).iteres)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for k, (ax, nom) in enumerate(zip(axes, ("ka", "ke", "V"))):
        for traj, couleur, etiquette in ((nat, SERIES[0], "gradient en θ"),
                                         (log, SERIES[1], "gradient en log θ")):
            ax.plot(np.arange(1, len(traj) + 1), 100 * (traj[:, k] / ref[k] - 1),
                    color=couleur, lw=1.6, label=etiquette)
        ax.axhline(0, color=TEXTE_2, lw=1)
        ax.set(xscale="log", xlabel="itération", title=nom)
    axes[0].set_ylabel("écart à l'optimum (%)")
    axes[0].legend(fontsize=8.5)
    fig.suptitle("Trajectoire du gradient à pas optimal, patient synthétique 1 (bruit 10 %) : "
                 "en θ, V ne bouge presque pas et ke oscille", x=0.01, ha="left",
                 fontweight="semibold")
    fig.tight_layout()
    enregistrer(fig, "09_trajectoires_gradient.png")


METHODES_LOCALES = {
    "Gradient (θ)": lambda p, x0: _nat(p, x0, lambda f, g, h, x: det.descente_gradient(
        f, g, x, max_iter=300)),
    "Gradient projeté (θ)": lambda p, x0: _nat(p, x0, lambda f, g, h, x: det.descente_gradient(
        f, g, x, bornes=strategie.BORNES, max_iter=300)),
    "Newton (θ)": lambda p, x0: _nat(p, x0, lambda f, g, h, x: det.newton(f, g, h, x)),
    "Gradient (log θ)": lambda p, x0: _log(p, x0, lambda f, g, h, x: det.descente_gradient(
        f, g, x, max_iter=300)),
    "Newton (log θ)": lambda p, x0: _log(p, x0, lambda f, g, h, x: det.newton(f, g, h, x)),
    "L-BFGS-B (log θ)": lambda p, x0: _log(p, x0, lambda f, g, h, x: strategie.lbfgsb(
        f, g, x, BL)),
}


def _nat(p, x0, methode):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return methode(*objectif.probleme(p), x0).x


def _log(p, x0, methode):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.exp(methode(*objectif.probleme(p, log=True), np.log(x0)).x)


def tache_point_initial(args):
    i, x0 = args
    p = patient_bruite(i)
    ref = strategie.estimer(p).x
    miroir = np.array([ref[1], ref[0], ref[2] * ref[1] / ref[0]])
    lignes = []
    for nom, lancer in METHODES_LOCALES.items():
        x = lancer(p, x0)
        if np.all(np.isfinite(x)) and ecart(x, ref) < 0.01:
            issue = "minimum physiologique"
        elif np.all(np.isfinite(x)) and ecart(x, miroir) < 0.01:
            issue = "minimum miroir"
        else:
            issue = "échec"
        lignes.append({"methode": nom, "patient": i + 1, "ka0": x0[0], "ke0": x0[1],
                       "V0": x0[2], "issue": issue})
    return lignes


def figure_points_initiaux(pool):
    rng = np.random.default_rng(7)
    lo, hi = np.log([0.2, 0.02, 10]), np.log([5, 0.3, 80])
    taches = [(i, np.exp(rng.uniform(lo, hi))) for i in range(4) for _ in range(25)]
    lignes = [l for ls in pool.map(tache_point_initial, taches) for l in ls]
    ecrire(RES / "points_initiaux.csv", lignes)

    issues = ["minimum physiologique", "minimum miroir", "échec"]
    couleurs = [SERIES[0], SERIES[1], GRIS]
    noms = list(METHODES_LOCALES)
    parts = {iss: np.array([100 * np.mean([l["issue"] == iss for l in lignes if l["methode"] == m])
                            for m in noms]) for iss in issues}

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    gauche = np.zeros(len(noms))
    for iss, couleur in zip(issues, couleurs):
        ax.barh(noms, parts[iss], left=gauche, color=couleur, height=0.6, label=iss,
                edgecolor="white", linewidth=2)
        for y, (g, v) in enumerate(zip(gauche, parts[iss])):
            if v >= 8:
                ax.text(g + v / 2, y, f"{v:.0f} %", ha="center", va="center", fontsize=8.5,
                        color="white" if couleur != GRIS else TEXTE)
        gauche += parts[iss]
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("part des 100 départs (4 patients × 25 points initiaux tirés au hasard)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    ax.set_title("Où aboutissent les méthodes locales selon le point de départ", pad=28)
    enregistrer(fig, "10_points_initiaux.png")


REGLAGES = (
    [("Recuit simulé : température initiale", "T0 / J(x0)", c) for c in (1e-4, 1e-3, 1e-2, 0.1, 1, 10)]
    + [("Recuit simulé : budget", "évaluations", n) for n in (1_000, 3_000, 10_000, 30_000, 100_000)]
    + [("Metropolis : température", "T / J(x0)", c) for c in (1e-4, 1e-3, 1e-2, 0.1, 1, 10)]
    + [("Recherche aléatoire : budget", "évaluations", n) for n in (1_000, 10_000, 100_000, 1_000_000)]
)


def tache_reglage(args):
    k, i = args
    panneau, _, valeur = REGLAGES[k]
    p = patient_bruite(i)
    ref = strategie.estimer(p).x
    f, _, _ = objectif.probleme(p, log=True)
    x0 = np.log(X0)
    rng = np.random.default_rng([k, i])
    if panneau.startswith("Recuit simulé : temp"):
        res = sto.recuit_simule(f, x0, 0.2, valeur * f(x0), BL, 20_000, rng)
    elif panneau.startswith("Recuit simulé : budget"):
        res = sto.recuit_simule(f, x0, 0.2, 0.1 * f(x0), BL, valeur, rng)
    elif panneau.startswith("Metropolis"):
        res = sto.metropolis(f, x0, 0.05, valeur * f(x0), BL, 20_000, rng)
    else:
        res = sto.recherche_aleatoire(f, BL, valeur, rng)
    return {"reglage": panneau, "valeur": valeur, "patient": i + 1,
            "ecart_optimum": ecart(np.exp(res.x), ref)}


def figure_hyperparametres(pool):
    taches = [(k, i) for k in range(len(REGLAGES)) for i in range(len(THETAS))]
    lignes = list(pool.map(tache_reglage, taches))
    ecrire(RES / "hyperparametres.csv", lignes)

    panneaux = list(dict.fromkeys(r[0] for r in REGLAGES))
    choisis = {"Recuit simulé : température initiale": 0.1, "Recuit simulé : budget": 20_000,
               "Metropolis : température": 0.01, "Recherche aléatoire : budget": 20_000}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for ax, panneau in zip(axes.flat, panneaux):
        valeurs = sorted({l["valeur"] for l in lignes if l["reglage"] == panneau})
        e = np.array([[100 * l["ecart_optimum"] for l in lignes
                       if l["reglage"] == panneau and l["valeur"] == v] for v in valeurs])
        ax.fill_between(valeurs, np.percentile(e, 10, axis=1), np.percentile(e, 90, axis=1),
                        color=SERIES[0], alpha=0.18, lw=0, label="10e – 90e centile")
        ax.plot(valeurs, np.median(e, axis=1), "o-", color=SERIES[0], ms=6, label="médiane")
        ax.axhline(1, color=TEXTE_2, lw=1, ls="--")
        ax.axvline(choisis[panneau], color=SERIES[1], lw=1.2, ls=":")
        ax.set(xscale="log", yscale="log", title=panneau,
               xlabel=next(r[1] for r in REGLAGES if r[0] == panneau))
    axes[0, 0].text(1.1e-4, 1.15, "seuil de réussite (1 %)", fontsize=8, color=TEXTE_2)
    for ax in axes[:, 0]:
        ax.set_ylabel("écart au minimum global (%)")
    axes[0, 0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Sensibilité des méthodes stochastiques à leurs réglages (12 patients, bruit 10 %) ; "
                 "pointillé orange : valeur utilisée dans le benchmark",
                 x=0.01, ha="left", fontweight="semibold", fontsize=10.5)
    fig.tight_layout()
    enregistrer(fig, "11_hyperparametres.png")


def figure_estime_vs_vrai():
    lignes = [l for l in csv.DictReader(open(RES / "benchmark.csv", encoding="utf-8"))
              if l["methode"] == "DE + L-BFGS-B (retenue)" and l["bruit"] != "sans"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for k, (ax, nom, unite) in enumerate(zip(axes, ("ka", "ke", "V"), ("h⁻¹", "h⁻¹", "L"))):
        for (niveau, etiq), couleur in zip(NOMS_BRUIT.items(), SERIES):
            sel = [l for l in lignes if l["bruit"] == niveau]
            vrai = [THETAS[int(l["patient"]) - 1][k] for l in sel]
            ax.plot(vrai, [float(l[nom]) for l in sel], "o", ms=5, color=couleur, alpha=0.75,
                    mec="white", mew=0.6, label=etiq)
        lim = ax.get_xlim()
        ax.plot(lim, lim, color=TEXTE_2, lw=1, ls="--")
        ax.set(xlabel=f"vrai {nom} ({unite})", ylabel=f"{nom} estimé ({unite})", title=nom)
    axes[0].legend(fontsize=8.5)
    fig.suptitle("Paramètres estimés (stratégie retenue) contre vrais paramètres, "
                 "36 jeux de données par niveau de bruit", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    enregistrer(fig, "12_estime_vs_vrai.png")


def figure_residus():
    pred, res, temps = [], [], []
    for p in donnees.charger_theophylline():
        th = strategie.estimer(p).x
        pred.append(modele.concentration(th, p.t))
        res.append(-objectif.residus(th, p))
        temps.append(p.t)
    pred, res, temps = map(np.concatenate, (pred, res, temps))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.plot(pred, res, "o", ms=4.5, color=SERIES[0], alpha=0.7, mec="white", mew=0.5)
    bords = np.quantile(pred, np.linspace(0, 1, 6))
    centres = [(a + b) / 2 for a, b in zip(bords[:-1], bords[1:])]
    rms = [np.sqrt(np.mean(res[(pred >= a) & (pred <= b)] ** 2))
           for a, b in zip(bords[:-1], bords[1:])]
    ax.plot(centres, rms, "s-", color=SERIES[1], ms=6, label="écart quadratique moyen par classe")
    ax.plot(centres, [-r for r in rms], "s-", color=SERIES[1], ms=6)
    ax.axhline(0, color=TEXTE_2, lw=1)
    ax.set(xlabel="concentration prédite (mg/L)", ylabel="mesure − modèle (mg/L)",
           title="Résidus contre concentration prédite")
    ax.legend(fontsize=8.5, loc="upper left")

    ax = axes[1]
    ax.plot(temps, res, "o", ms=4.5, color=SERIES[0], alpha=0.7, mec="white", mew=0.5)
    ax.axhline(0, color=TEXTE_2, lw=1)
    ax.set(xlabel="temps (h)", ylabel="mesure − modèle (mg/L)", title="Résidus contre temps")
    fig.suptitle("Résidus des 12 patients réels (132 mesures)", x=0.01, ha="left",
                 fontweight="semibold")
    fig.tight_layout()
    enregistrer(fig, "13_residus.png")


def ecrire(chemin, lignes):
    with open(chemin, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)


if __name__ == "__main__":
    figure_donnees()
    figure_trajectoires()
    figure_estime_vs_vrai()
    figure_residus()
    with ProcessPoolExecutor() as pool:
        figure_points_initiaux(pool)
        figure_hyperparametres(pool)
