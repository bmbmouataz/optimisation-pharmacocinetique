from dataclasses import dataclass, field

import numpy as np


class Compteur:
    """Enveloppe une fonction et garde la trace de chaque appel."""

    def __init__(self, f):
        self.f = f
        self.valeurs = []

    def __call__(self, x):
        v = self.f(x)
        self.valeurs.append(v)
        return v

    @property
    def n(self):
        return len(self.valeurs)


@dataclass
class Resultat:
    x: np.ndarray
    f: float
    n_eval: int
    historique: np.ndarray = field(repr=False, default=None)
    n_iter: int = 0
    iteres: np.ndarray = field(repr=False, default=None)

    def courbe(self):
        """Meilleure valeur atteinte après chaque évaluation de f."""
        return np.minimum.accumulate(self.historique)


def projeter(x, bornes):
    if bornes is None:
        return x
    return np.clip(x, bornes[0], bornes[1])
