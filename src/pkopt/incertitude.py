"""Incertitude sur les paramètres estimés."""
import numpy as np
from scipy import stats

from . import modele, objectif, strategie
from .objectif import Patient


def asymptotique(theta, p, niveau=0.95):
    """Intervalles de confiance par linéarisation, calculés en log θ.

    cov(log θ) ≈ s² (Jₗᵀ Jₗ)⁻¹ avec Jₗ la jacobienne en log θ et s² = J(θ)/(n - 3).
    Passer par le log donne des intervalles positifs et asymétriques, plus
    réalistes quand un paramètre est mal déterminé.
    """
    n = len(p.t)
    s2 = objectif.critere(theta, p) / (n - 3)
    Jl = modele.jacobienne(theta, p.t, p.dose) * np.asarray(theta)[:, None]
    cov = s2 * np.linalg.inv(Jl @ Jl.T)
    se = np.sqrt(np.diag(cov))
    q = stats.t.ppf((1 + niveau) / 2, n - 3)
    bas, haut = theta * np.exp(-q * se), theta * np.exp(q * se)
    correlation = cov / np.outer(se, se)
    return {"se_log": se, "bas": bas, "haut": haut, "correlation": correlation,
            "conditionnement": np.linalg.cond(Jl @ Jl.T)}


def bootstrap(theta, p, B, rng):
    """Bootstrap des résidus : on rééchantillonne les résidus centrés,
    on reconstruit des mesures et on réestime θ depuis l'estimation initiale."""
    ajuste = modele.concentration(theta, p.t, p.dose)
    r = p.y - ajuste
    r = r - r.mean()
    x0 = np.log(theta)
    tirages = np.empty((B, 3))
    for b in range(B):
        pb = Patient(p.t, ajuste + rng.choice(r, size=r.size), p.dose)
        f, g, _ = objectif.probleme(pb, log=True)
        res = strategie.lbfgsb(f, g, x0, strategie.BORNES_LOG)
        tirages[b] = modele.branche_physiologique(np.exp(res.x))
    return tirages
