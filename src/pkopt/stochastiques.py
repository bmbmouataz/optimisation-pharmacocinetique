"""Méthodes stochastiques. Toutes reçoivent un budget d'évaluations de f
et un générateur aléatoire, pour pouvoir les comparer à coût égal et
reproduire chaque exécution."""
import numpy as np

from .suivi import Compteur, Resultat, projeter


def recherche_aleatoire(f, bornes, n_eval, rng):
    """Tirages uniformes indépendants dans le pavé."""
    f = Compteur(f)
    lo, hi = map(np.asarray, bornes)
    X = rng.uniform(lo, hi, size=(n_eval, len(lo)))
    vals = [f(x) for x in X]
    i = int(np.argmin(vals))
    return Resultat(X[i], vals[i], f.n, np.array(f.valeurs), n_eval)


def recherche_locale(f, x0, rayon, bornes, n_eval, rng):
    """Exploration localisée : on ne garde un déplacement uniforme dans
    [-rayon, rayon] que s'il fait baisser f."""
    f = Compteur(f)
    x = projeter(np.asarray(x0, dtype=float), bornes)
    fx = f(x)
    for _ in range(n_eval - 1):
        y = projeter(x + rng.uniform(-rayon, rayon, size=x.size), bornes)
        fy = f(y)
        if fy <= fx:
            x, fx = y, fy
    return Resultat(x, fx, f.n, np.array(f.valeurs), n_eval)


def _marche(f, x0, sigma, temperatures, bornes, rng, sigmas=None):
    f = Compteur(f)
    x = projeter(np.asarray(x0, dtype=float), bornes)
    fx = f(x)
    meilleur, f_meilleur = x, fx
    n = len(temperatures)
    pas = sigma * np.ones(n)[:, None] if sigmas is None else sigmas[:, None] * sigma
    bruit = rng.normal(size=(n, x.size)) * pas
    u = rng.uniform(size=n)
    for T, eps, ui in zip(temperatures, bruit, u):
        y = projeter(x + eps, bornes)
        fy = f(y)
        if fy <= fx or ui < np.exp(-(fy - fx) / T):
            x, fx = y, fy
            if fx < f_meilleur:
                meilleur, f_meilleur = x, fx
    return Resultat(meilleur, f_meilleur, f.n, np.array(f.valeurs), n)


def metropolis(f, x0, sigma, T, bornes, n_eval, rng):
    """Marche de Metropolis à température fixe ; on renvoie le meilleur point visité."""
    return _marche(f, x0, sigma, np.full(n_eval - 1, T), bornes, rng)


def recuit_simule(f, x0, sigma, T0, bornes, n_eval, rng, facteur_final=1e-6,
                  reduction_pas=100):
    """Recuit simulé à refroidissement géométrique.

    La température passe de T0 à T0·facteur_final sur le budget, et la taille
    des propositions diminue en parallèle d'un facteur reduction_pas : on
    explore au début, on affine à la fin.
    """
    n = n_eval - 1
    temperatures = T0 * facteur_final ** (np.arange(n) / n)
    sigmas = reduction_pas ** (-np.arange(n) / n)
    return _marche(f, x0, sigma, temperatures, bornes, rng, sigmas)
