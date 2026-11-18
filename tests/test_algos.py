import numpy as np
import pytest

from pkopt import deterministes as det
from pkopt import donnees, objectif, strategie
from pkopt import stochastiques as sto

VRAI = np.array([1.3, 0.08, 28.0])
X0 = np.log([1.0, 0.07, 25.0])


@pytest.fixture
def probleme():
    p = objectif.Patient(donnees.T_REF, donnees.modele.concentration(VRAI, donnees.T_REF))
    return p, objectif.probleme(p, log=True)


def erreur(x):
    return np.max(np.abs(np.exp(x) / VRAI - 1))


def test_section_doree():
    assert abs(det.section_doree(lambda x: (x - 0.3) ** 2, 0, 1) - 0.3) < 1e-5


def test_gradient_optimal_en_log(probleme):
    _, (f, g, _) = probleme
    res = det.descente_gradient(f, g, X0, max_iter=3000, tol=1e-12)
    assert erreur(res.x) < 1e-3


def test_newton(probleme):
    _, (f, g, h) = probleme
    res = det.newton(f, g, h, X0)
    assert erreur(res.x) < 1e-6


def test_newton_projete_reste_dans_les_bornes(probleme):
    _, (f, g, h) = probleme
    lo, hi = np.log([0.2, 0.02, 16]), np.log([1.2, 0.12, 35])
    res = det.newton(f, g, h, X0, bornes=(lo, hi))
    assert np.all(res.x >= lo) and np.all(res.x <= hi)
    assert np.isclose(np.exp(res.x[0]), 1.2)


def test_recuit_simule(probleme):
    _, (f, _, _) = probleme
    rng = np.random.default_rng(1)
    res = sto.recuit_simule(f, X0, 0.1, f(X0) * 0.1, strategie.BORNES_LOG, 20_000, rng)
    assert erreur(res.x) < 0.02
    assert res.n_eval == 20_000


def test_strategie(probleme):
    p, _ = probleme
    res = strategie.estimer(p)
    assert np.allclose(res.x, VRAI, rtol=1e-6)


def test_compteur_d_evaluations(probleme):
    _, (f, _, _) = probleme
    res = sto.recherche_aleatoire(f, strategie.BORNES_LOG, 500, np.random.default_rng(0))
    assert res.n_eval == 500 and len(res.courbe()) == 500
