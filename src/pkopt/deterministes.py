"""Méthodes de descente : section dorée, gradient (pas fixe ou optimal), Newton.

Chaque méthode accepte des bornes (min, max) : on projette alors chaque itéré
sur le pavé, ce qui donne les versions « projetées ».
"""
import numpy as np

from .suivi import Compteur, Resultat, projeter

INV_OR = (np.sqrt(5) - 1) / 2


def section_doree(phi, a, b, tol=1e-6):
    """Minimise une fonction unimodale phi sur [a, b]."""
    c = b - INV_OR * (b - a)
    d = a + INV_OR * (b - a)
    fc, fd = phi(c), phi(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - INV_OR * (b - a)
            fc = phi(c)
        else:
            a, c, fc = c, d, fd
            d = a + INV_OR * (b - a)
            fd = phi(d)
    return (a + b) / 2


def descente_gradient(f, g, x0, pas="optimal", bornes=None, max_iter=500,
                      tol=1e-8, longueur_max=1.0, tol_ls=1e-6):
    """Descente de gradient à pas fixe (pas=float) ou à pas optimal.

    Pour le pas optimal, la section dorée cherche ρ dans [0, longueur_max/‖∇f‖] :
    le déplacement est au plus longueur_max, quelle que soit la norme du gradient.
    """
    f = Compteur(f)
    x = projeter(np.asarray(x0, dtype=float), bornes)
    f(x)
    n_grad = 0
    iteres = [x]
    for k in range(max_iter):
        d = -g(x)
        n_grad += 1
        norme = np.linalg.norm(d)
        if norme < tol:
            break
        if pas == "optimal":
            rho = section_doree(lambda r: f(projeter(x + r * d, bornes)),
                                0.0, longueur_max / norme, tol=tol_ls * longueur_max / norme)
        else:
            rho = pas
        x_new = projeter(x + rho * d, bornes)
        if not np.all(np.isfinite(x_new)):
            break
        fini = np.linalg.norm(x_new - x) < tol * (1 + np.linalg.norm(x))
        x = x_new
        iteres.append(x)
        if fini:
            break
    fx = f(x)
    return Resultat(x, fx, f.n + n_grad, np.array(f.valeurs), k + 1, np.array(iteres))


def newton(f, g, h, x0, bornes=None, max_iter=100, tol=1e-10):
    """Newton amorti.

    Quand la hessienne n'est pas définie positive (fréquent loin du minimum),
    la direction de Newton peut monter : on revient alors à -∇f. Un retour
    arrière d'Armijo garantit ensuite que J décroît à chaque itération.
    """
    f = Compteur(f)
    x = projeter(np.asarray(x0, dtype=float), bornes)
    fx = f(x)
    n_der = 0
    iteres = [x]
    for k in range(max_iter):
        gx, H = g(x), h(x)
        n_der += 2
        try:
            d = np.linalg.solve(H, -gx)
        except np.linalg.LinAlgError:
            d = -gx
        if gx @ d >= 0:
            d = -gx / max(1.0, np.linalg.norm(gx))

        rho = 1.0
        while rho > 1e-12:
            x_new = projeter(x + rho * d, bornes)
            f_new = f(x_new)
            if f_new <= fx + 1e-4 * gx @ (x_new - x):
                break
            rho /= 2
        else:
            break

        pas = np.linalg.norm(x_new - x)
        x, fx = x_new, f_new
        iteres.append(x)
        if pas < tol * (1 + np.linalg.norm(x)):
            break
    return Resultat(x, fx, f.n + n_der, np.array(f.valeurs), k + 1, np.array(iteres))
