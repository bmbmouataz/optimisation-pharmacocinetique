"""Critère des moindres carrés J(θ) = Σ (U(t_i; θ) - y_i)², son gradient et sa hessienne."""
from dataclasses import dataclass

import numpy as np

from . import modele


@dataclass
class Patient:
    t: np.ndarray
    y: np.ndarray
    dose: float = modele.DOSE
    nom: str = ""

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float)
        self.y = np.asarray(self.y, dtype=float)


def residus(theta, p):
    return modele.concentration(theta, p.t, p.dose) - p.y


def critere(theta, p):
    r = residus(theta, p)
    return float(r @ r)


def gradient(theta, p):
    r = residus(theta, p)
    return 2 * modele.jacobienne(theta, p.t, p.dose) @ r


def hessienne(theta, p):
    r = residus(theta, p)
    J = modele.jacobienne(theta, p.t, p.dose)
    H = modele.hessienne(theta, p.t, p.dose)
    return 2 * (J @ J.T + H @ r)


def hessienne_gauss_newton(theta, p):
    """Approximation 2 JᵀJ, toujours semi-définie positive."""
    J = modele.jacobienne(theta, p.t, p.dose)
    return 2 * J @ J.T


def probleme(p, log=False):
    """Renvoie (f, g, h) comme fonctions de x seul.

    Avec log=True, on optimise en x = log θ. Les trois paramètres ont des
    ordres de grandeur très différents (ke ≈ 0.1, V ≈ 30) : en échelle log,
    le problème est bien mieux conditionné et la positivité est automatique.
    """
    if not log:
        return (lambda x: critere(x, p),
                lambda x: gradient(x, p),
                lambda x: hessienne(x, p))

    def f(x):
        return critere(np.exp(x), p)

    def g(x):
        th = np.exp(x)
        return th * gradient(th, p)

    def h(x):
        th = np.exp(x)
        return np.outer(th, th) * hessienne(th, p) + np.diag(th * gradient(th, p))

    return f, g, h


def gradient_numerique(f, x, eps=1e-6):
    """Différences finies centrées, utilisées dans les tests."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(len(x)):
        h = np.zeros_like(x)
        h[i] = eps * max(1.0, abs(x[i]))
        g[i] = (f(x + h) - f(x - h)) / (2 * h[i])
    return g
