"""Modèle à un compartiment avec absorption d'ordre 1 et ses dérivées.

    A' = -ka A,                A(0) = dose
    U' = ka/V A - ke U,        U(0) = 0

donne U(t) = dose·ka / (V (ka - ke)) · (e^{-ke t} - e^{-ka t}).

On réécrit cette solution U = (dose/V) · ka · t · e^{-ke t} · φ((ka - ke) t)
avec φ(x) = (1 - e^{-x}) / x. La fonction φ est lisse en 0, ce qui supprime
le cas particulier ka = ke et la perte de précision quand ka est proche de ke.
"""
import numpy as np

DOSE = 300.0  # mg, dose de l'énoncé
NOMS = ("ka", "ke", "V")

_SEUIL = 1e-3


def _phi(x):
    """φ(x) = (1 - e^{-x}) / x et ses deux premières dérivées."""
    x = np.asarray(x, dtype=float)
    petit = np.abs(x) < _SEUIL
    xs = np.where(petit, 1.0, x)  # évite la division par zéro dans la branche exacte
    e = np.exp(-xs)
    n = e * (1 + xs) - 1
    p0 = -np.expm1(-xs) / xs
    p1 = n / xs**2
    p2 = -e / xs - 2 * n / xs**3

    # Développements de Taylor à l'ordre 3 près de 0
    p0 = np.where(petit, 1 - x / 2 + x**2 / 6 - x**3 / 24, p0)
    p1 = np.where(petit, -1 / 2 + x / 3 - x**2 / 8 + x**3 / 30, p1)
    p2 = np.where(petit, 1 / 3 - x / 4 + x**2 / 10 - x**3 / 36, p2)
    return p0, p1, p2


def _termes(theta, t, dose):
    ka, ke, V = theta
    t = np.asarray(t, dtype=float)
    p0, p1, p2 = _phi((ka - ke) * t)
    E = np.exp(-ke * t)
    C = dose / V
    return ka, ke, V, t, E, C, p0, p1, p2


def concentration(theta, t, dose=DOSE):
    """Concentration plasmatique U(t) en mg/L."""
    ka, ke, V, t, E, C, p0, _, _ = _termes(theta, t, dose)
    return C * ka * t * E * p0


def jacobienne(theta, t, dose=DOSE):
    """Dérivées de U par rapport à (ka, ke, V), tableau de forme (3, n)."""
    ka, ke, V, t, E, C, p0, p1, _ = _termes(theta, t, dose)
    u = C * ka * t * E * p0
    d_ka = C * t * E * (p0 + ka * t * p1)
    d_ke = -C * ka * t**2 * E * (p0 + p1)
    d_V = -u / V
    return np.array([d_ka, d_ke, d_V])


def hessienne(theta, t, dose=DOSE):
    """Dérivées secondes de U, tableau de forme (3, 3, n)."""
    ka, ke, V, t, E, C, p0, p1, p2 = _termes(theta, t, dose)
    u = C * ka * t * E * p0
    d_ka = C * t * E * (p0 + ka * t * p1)
    d_ke = -C * ka * t**2 * E * (p0 + p1)

    h = np.empty((3, 3) + t.shape)
    h[0, 0] = C * t**2 * E * (2 * p1 + ka * t * p2)
    h[0, 1] = h[1, 0] = -C * t**2 * E * (p0 + (1 + ka * t) * p1 + ka * t * p2)
    h[1, 1] = C * ka * t**3 * E * (p0 + 2 * p1 + p2)
    h[0, 2] = h[2, 0] = -d_ka / V
    h[1, 2] = h[2, 1] = -d_ke / V
    h[2, 2] = 2 * u / V**2
    return h


def branche_physiologique(theta):
    """Renvoie la solution équivalente qui vérifie ka > ke.

    Le modèle est invariant par (ka, ke, V) -> (ke, ka, V·ke/ka) : les deux jeux
    de paramètres donnent exactement la même courbe (phénomène de flip-flop).
    Pour un médicament absorbé par voie orale, on retient en général ka > ke.
    """
    ka, ke, V = theta
    if ka >= ke:
        return np.array([ka, ke, V], dtype=float)
    return np.array([ke, ka, V * ke / ka])


def t_max(theta):
    """Instant du pic de concentration (h)."""
    ka, ke, _ = theta
    if np.isclose(ka, ke):
        return 1 / ka
    return np.log(ka / ke) / (ka - ke)


def demi_vie(theta):
    """Demi-vie d'élimination (h)."""
    return np.log(2) / theta[1]
