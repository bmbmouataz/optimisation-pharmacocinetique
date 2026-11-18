from pathlib import Path

import numpy as np

from . import modele
from .objectif import Patient

RACINE = Path(__file__).resolve().parents[2]
FICHIER_THEO = RACINE / "data" / "theophylline.csv"
FICHIER_ANNEXE = RACINE / "data" / "synthetiques_annexe.csv"

# Instants de prélèvement nominaux de l'étude
T_REF = np.array([0, 0.25, 0.5, 1, 2, 3.52, 5.07, 7.07, 9.03, 12.05, 24.15])

NIVEAUX_BRUIT = {"sans": 0.0, "faible": 0.05, "moyen": 0.10, "fort": 0.25}


def charger_theophylline(dose_reelle=False):
    """Les 12 patients de l'étude théophylline (Boeckmann, Sheiner et Beal, 1994).

    Par défaut on garde la dose de 300 mg de l'énoncé. Avec dose_reelle=True,
    chaque patient reçoit sa dose effective (mg/kg × poids, environ 320 mg).
    """
    d = np.genfromtxt(FICHIER_THEO, delimiter=",", names=True)
    patients = []
    for k in np.unique(d["patient"]).astype(int):
        lignes = d[d["patient"] == k]
        dose = float(lignes["dose_mg"][0]) if dose_reelle else modele.DOSE
        patients.append(Patient(lignes["temps_h"], lignes["concentration_mg_L"], dose,
                                nom=f"patient {k}"))
    return patients


def charger_annexe():
    """Les 12 patients synthétiques bruités fournis en annexe du sujet (paramètres inconnus)."""
    d = np.genfromtxt(FICHIER_ANNEXE, delimiter=",", names=True)
    return [Patient(d["temps_h"][d["patient"] == k], d["concentration_mg_L"][d["patient"] == k],
                    nom=f"patient {k}")
            for k in np.unique(d["patient"]).astype(int)]


def _normale_tronquee(rng, moy, ecart, lo, hi):
    while True:
        x = rng.normal(moy, ecart)
        if lo <= x <= hi:
            return x


def patients_synthetiques(n=12, graine=42):
    """Tire n jeux de paramètres plausibles pour la théophylline.

    Les lois (normales tronquées) sont calées à la main sur les estimations
    obtenues sur les vrais patients.
    """
    rng = np.random.default_rng(graine)
    return np.array([[_normale_tronquee(rng, 1.2, 0.4, 0.5, 1.8),
                      _normale_tronquee(rng, 0.07, 0.015, 0.04, 0.1),
                      _normale_tronquee(rng, 29, 5, 16, 35)] for _ in range(n)])


def bruiter(theta, niveau, rng, t=T_REF):
    """Mesures simulées avec un bruit gaussien d'écart-type niveau × concentration moyenne."""
    u = modele.concentration(theta, t)
    y = u + rng.normal(0, niveau * u.mean(), size=u.size)
    return Patient(t, np.maximum(y, 0))
