"""Second cas d'étude : les 12 patients synthétiques bruités fournis en annexe du sujet.

Leurs vrais paramètres ne sont pas communiqués : on ne peut donc juger que la
qualité de l'ajustement et la précision des estimations.
Sorties : resultats/donnees_annexe.csv, figure 07.
"""
import csv

import numpy as np

from donnees_reelles import B, SORTIE, figure_ajustements, r2, reference_curve_fit
from pkopt import donnees, incertitude, modele, objectif, strategie

if __name__ == "__main__":
    SORTIE.mkdir(exist_ok=True)
    rng = np.random.default_rng(2027)
    patients = donnees.charger_annexe()

    lignes, thetas, tirages = [], [], []
    for i, p in enumerate(patients, 1):
        th = strategie.estimer(p).x
        ic = incertitude.asymptotique(th, p)
        thetas.append(th)
        tirages.append(incertitude.bootstrap(th, p, B, rng))
        lignes.append({
            "patient": i, "ka": th[0], "ke": th[1], "V": th[2],
            "J": objectif.critere(th, p), "R2": r2(th, p),
            **{f"{n}_ic_bas": ic["bas"][k] for k, n in enumerate(modele.NOMS)},
            **{f"{n}_ic_haut": ic["haut"][k] for k, n in enumerate(modele.NOMS)},
            "ecart_curve_fit": np.max(np.abs(reference_curve_fit(p) / th - 1)),
        })
        print(f"patient {i:2d} : ka = {th[0]:.3f} [{ic['bas'][0]:.3f} ; {ic['haut'][0]:.3f}]  "
              f"ke = {th[1]:.4f}  V = {th[2]:.2f}  J = {lignes[-1]['J']:.4f}  "
              f"R² = {lignes[-1]['R2']:.3f}")

    with open(SORTIE / "donnees_annexe.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lignes[0]))
        w.writeheader()
        for l in lignes:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in l.items()})

    figure_ajustements(patients, np.array(thetas), tirages, "07_ajustements_annexe.png",
                       "Données synthétiques de l'annexe : mesures, modèle estimé et bande "
                       "de confiance à 95 % (bootstrap)")
