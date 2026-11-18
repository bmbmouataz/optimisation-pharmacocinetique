import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pkopt import donnees, modele, objectif

T = donnees.T_REF
CAS = [np.array([1.3, 0.08, 28.0]),
       np.array([0.4, 0.9, 15.0]),       # ka < ke
       np.array([0.1, 0.1 + 1e-7, 30.0]),  # ka ≈ ke
       np.array([0.1, 0.1, 30.0])]       # ka = ke


def derivee_numerique(fonc, theta, i, h=1e-6):
    e = np.zeros(3)
    e[i] = h * abs(theta[i])
    return (fonc(theta + e) - fonc(theta - e)) / (2 * e[i])


def formule_directe(theta, t):
    ka, ke, V = theta
    return modele.DOSE * ka / (V * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def test_concentration_egale_la_formule_classique():
    theta = CAS[0]
    assert np.allclose(modele.concentration(theta, T), formule_directe(theta, T), rtol=1e-12)


def test_limite_ka_egal_ke():
    k, V = 0.1, 30.0
    attendu = modele.DOSE * k / V * T * np.exp(-k * T)
    assert np.allclose(modele.concentration([k, k, V], T), attendu, rtol=1e-12)
    assert np.allclose(modele.concentration([k, k + 1e-9, V], T), attendu, rtol=1e-7)


def test_solution_de_l_edo():
    ka, ke, V = CAS[0]

    def systeme(t, z):
        a, u = z
        return [-ka * a, ka / V * a - ke * u]

    sol = solve_ivp(systeme, (0, T[-1]), [modele.DOSE, 0], t_eval=T, rtol=1e-11, atol=1e-12)
    assert np.allclose(sol.y[1], modele.concentration(CAS[0], T), atol=1e-8)


@pytest.mark.parametrize("theta", CAS)
def test_jacobienne(theta):
    J = modele.jacobienne(theta, T)
    for i in range(3):
        num = derivee_numerique(lambda th: modele.concentration(th, T), theta, i)
        assert np.allclose(J[i], num, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("theta", CAS)
def test_hessienne(theta):
    H = modele.hessienne(theta, T)
    for i in range(3):
        num = derivee_numerique(lambda th: modele.jacobienne(th, T), theta, i)
        assert np.allclose(H[:, i], num, rtol=1e-5, atol=1e-7)


def test_ancienne_derivee_fausse_quand_ka_egal_ke():
    """La branche ka = ke du notebook d'origine donnait t·e^{-kt}·(1 - k²)·dose/V
    pour ∂U/∂ka, au lieu de t·e^{-kt}·(1 - k t/2)·dose/V."""
    k, V = 0.1, 30.0
    ancienne = modele.DOSE / V * T * np.exp(-k * T) * (1 - k**2)
    juste = modele.DOSE / V * T * np.exp(-k * T) * (1 - k * T / 2)
    assert np.allclose(modele.jacobienne([k, k, V], T)[0], juste)
    assert not np.allclose(ancienne, juste)


@pytest.mark.parametrize("log", [False, True])
def test_gradient_et_hessienne_du_critere(log):
    rng = np.random.default_rng(0)
    p = donnees.bruiter(CAS[0], 0.1, rng)
    f, g, h = objectif.probleme(p, log=log)
    x = np.log(CAS[0] * 1.1) if log else CAS[0] * 1.1
    assert np.allclose(g(x), objectif.gradient_numerique(f, x), rtol=1e-5)
    Hnum = np.array([objectif.gradient_numerique(lambda z: g(z)[i], x) for i in range(3)])
    assert np.allclose(h(x), Hnum, rtol=1e-4, atol=1e-6)


def test_flip_flop():
    theta = CAS[0]
    miroir = np.array([theta[1], theta[0], theta[2] * theta[1] / theta[0]])
    assert np.allclose(modele.concentration(theta, T), modele.concentration(miroir, T))
    assert np.allclose(modele.branche_physiologique(miroir), theta)


def test_donnees_theophylline():
    patients = donnees.charger_theophylline()
    assert len(patients) == 12
    assert all(len(p.t) == 11 for p in patients)
    assert np.allclose(patients[0].y[:4], [0.74, 2.84, 6.57, 10.5])
    reels = donnees.charger_theophylline(dose_reelle=True)
    assert all(260 < p.dose < 325 for p in reels)
    assert round(reels[8].dose) == 268

    annexe = donnees.charger_annexe()
    assert len(annexe) == 12
    assert all(np.allclose(p.t, donnees.T_REF) for p in annexe)
