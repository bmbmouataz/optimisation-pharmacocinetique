"""Méthodes de scipy et stratégie retenue : évolution différentielle puis L-BFGS-B."""
import numpy as np
from scipy.optimize import differential_evolution, minimize

from . import modele, objectif
from .suivi import Compteur, Resultat

# Bornes larges en (ka, ke, V). ke ≤ 0.5 h⁻¹ écarte la branche « flip-flop »
# (ka et ke échangés) tant que ka > 0.5 h⁻¹, ce qui est le cas pour la théophylline.
BORNES = (np.array([0.05, 0.005, 2.0]), np.array([20.0, 0.5, 200.0]))
BORNES_LOG = (np.log(BORNES[0]), np.log(BORNES[1]))


def lbfgsb(f, g, x0, bornes):
    f = Compteur(f)
    g = Compteur(g)
    res = minimize(f, x0, jac=g, method="L-BFGS-B", bounds=list(zip(*bornes)),
                   options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-10})
    return Resultat(res.x, res.fun, f.n + g.n, np.array(f.valeurs), res.nit)


def evolution_differentielle(f, bornes, graine, maxiter=200, popsize=15):
    f = Compteur(f)
    res = differential_evolution(f, list(zip(*bornes)), seed=graine, maxiter=maxiter,
                                 popsize=popsize, tol=1e-10, mutation=(0.5, 1),
                                 recombination=0.7, polish=False)
    return Resultat(res.x, res.fun, f.n, np.array(f.valeurs), res.nit)


def estimer(p, graine=0, maxiter=200):
    """Stratégie retenue sur un patient : recherche globale, puis affinage local.

    Les deux phases travaillent en log θ. On renvoie θ sur la branche ka > ke.
    """
    f, g, _ = objectif.probleme(p, log=True)
    glob = evolution_differentielle(f, BORNES_LOG, graine, maxiter=maxiter)
    loc = lbfgsb(f, g, glob.x, BORNES_LOG)
    theta = modele.branche_physiologique(np.exp(loc.x))
    hist = np.concatenate([glob.historique, loc.historique])
    return Resultat(theta, objectif.critere(theta, p), glob.n_eval + loc.n_eval, hist,
                    glob.n_iter + loc.n_iter)
