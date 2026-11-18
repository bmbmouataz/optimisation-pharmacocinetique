"""Compare les méthodes d'optimisation sur des patients synthétiques bruités.

Pour chaque jeu de données on distingue deux erreurs :
- l'échec d'optimisation : la méthode n'atteint pas le minimum global de J,
  repéré comme le meilleur point trouvé par l'ensemble des méthodes ;
- l'erreur d'estimation : même au minimum global, θ̂ diffère du vrai θ à cause du bruit.
Le notebook d'origine mélangeait les deux.

Sortie : resultats/benchmark.csv et resultats/convergence.npz.
"""
import csv
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from pkopt import deterministes as det
from pkopt import donnees, objectif, strategie
from pkopt import stochastiques as sto
from pkopt.donnees import RACINE

REPETITIONS = 3
BUDGET = 20_000
SEUIL_SUCCES = 0.01  # chaque paramètre à 1 % près du minimum global
SORTIE = RACINE / "resultats"

THETAS = donnees.patients_synthetiques()
X0 = THETAS.mean(axis=0)  # même point de départ pour tous : le patient moyen


def _log(fonction):
    """Lance une méthode en log θ et ramène le résultat en θ."""
    def lancer(p, rng):
        res = fonction(*objectif.probleme(p, log=True), np.log(X0), rng)
        res.x = np.exp(res.x)
        return res
    return lancer


def _nat(fonction):
    def lancer(p, rng):
        # Sans changement de variable, les itérés peuvent partir très loin
        # (ke ≫ ka) et faire déborder les exponentielles : J vaut alors inf.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return fonction(*objectif.probleme(p), X0, rng)
    return lancer


B, BL = strategie.BORNES, strategie.BORNES_LOG

METHODES = {
    # nom : (famille, fonction)
    "Gradient pas optimal": ("locale", _nat(
        lambda f, g, h, x0, rng: det.descente_gradient(f, g, x0, max_iter=300))),
    "Gradient projeté": ("locale", _nat(
        lambda f, g, h, x0, rng: det.descente_gradient(f, g, x0, bornes=B, max_iter=300))),
    "Gradient pas optimal (log θ)": ("locale", _log(
        lambda f, g, h, x0, rng: det.descente_gradient(f, g, x0, max_iter=300))),
    "Newton": ("locale", _nat(
        lambda f, g, h, x0, rng: det.newton(f, g, h, x0))),
    "Newton projeté (log θ)": ("locale", _log(
        lambda f, g, h, x0, rng: det.newton(f, g, h, x0, bornes=BL))),
    "L-BFGS-B (log θ)": ("locale", _log(
        lambda f, g, h, x0, rng: strategie.lbfgsb(f, g, x0, BL))),
    "Recherche aléatoire": ("globale", _log(
        lambda f, g, h, x0, rng: sto.recherche_aleatoire(f, BL, BUDGET, rng))),
    "Recherche localisée": ("stochastique", _log(
        lambda f, g, h, x0, rng: sto.recherche_locale(f, x0, 0.05, BL, BUDGET, rng))),
    "Metropolis": ("stochastique", _log(
        lambda f, g, h, x0, rng: sto.metropolis(f, x0, 0.05, 0.01 * f(x0), BL, BUDGET, rng))),
    "Recuit simulé": ("stochastique", _log(
        lambda f, g, h, x0, rng: sto.recuit_simule(f, x0, 0.2, 0.1 * f(x0), BL, BUDGET, rng))),
    "Évolution différentielle": ("globale", _log(
        lambda f, g, h, x0, rng: strategie.evolution_differentielle(
            f, BL, int(rng.integers(2**31)))))
}
METHODES["DE + L-BFGS-B (retenue)"] = (
    "globale", lambda p, rng: strategie.estimer(p, graine=int(rng.integers(2**31))))


def jeux_de_donnees():
    for niveau_nom, niveau in donnees.NIVEAUX_BRUIT.items():
        for rep in range(1 if niveau == 0 else REPETITIONS):
            for i, theta in enumerate(THETAS):
                yield niveau_nom, niveau, rep, i


def traiter(tache):
    niveau_nom, niveau, rep, i = tache
    graine = [list(donnees.NIVEAUX_BRUIT).index(niveau_nom), rep, i]
    rng = np.random.default_rng(graine)
    theta = THETAS[i]
    p = donnees.bruiter(theta, niveau, rng)
    lignes, courbes = [], {}
    for nom, (famille, lancer) in METHODES.items():
        debut = time.perf_counter()
        res = lancer(p, np.random.default_rng(graine + [len(lignes)]))
        duree = time.perf_counter() - debut
        x = np.asarray(res.x)
        lignes.append({"methode": nom, "famille": famille, "bruit": niveau_nom, "rep": rep,
                       "patient": i + 1, "ka": x[0], "ke": x[1], "V": x[2],
                       "J": objectif.critere(x, p), "n_eval": res.n_eval, "temps_s": duree,
                       **{f"err_{n}": abs(x[k] / theta[k] - 1)
                          for k, n in enumerate(("ka", "ke", "V"))}})
        if niveau_nom == "moyen" and rep == 0 and i == 0 and res.historique is not None:
            courbes[nom] = res.courbe()
    ref = min(lignes, key=lambda l: l["J"])
    for l in lignes:
        l["J_ref"] = ref["J"]
        ecart = max(abs(l[n] / ref[n] - 1) for n in ("ka", "ke", "V"))
        l["succes"] = int(ecart < SEUIL_SUCCES)
    return lignes, courbes


if __name__ == "__main__":
    SORTIE.mkdir(exist_ok=True)
    taches = list(jeux_de_donnees())
    debut = time.perf_counter()
    lignes, courbes = [], {}
    with ProcessPoolExecutor() as pool:
        for l, c in pool.map(traiter, taches):
            lignes += l
            courbes.update(c)
    with open(SORTIE / "benchmark.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)
    np.savez(SORTIE / "convergence.npz", **courbes)
    print(f"{len(taches)} jeux de données, {len(METHODES)} méthodes, "
          f"{time.perf_counter() - debut:.0f} s")
