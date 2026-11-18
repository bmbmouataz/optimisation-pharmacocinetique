# Benchmark

Réussite : part des jeux de données où chaque paramètre est à 1 % près du meilleur point trouvé toutes méthodes confondues.

| Méthode | Réussite sans bruit | Réussite 5 % | Réussite 10 % | Réussite 25 % | Évaluations (médiane) |
|---|---|---|---|---|---|
| Gradient pas optimal | 0 % | 25 % | 17 % | 17 % | 9 602 |
| Gradient projeté | 0 % | 19 % | 25 % | 22 % | 9 602 |
| Gradient pas optimal (log θ) | 100 % | 100 % | 100 % | 100 % | 9 602 |
| Newton | 92 % | 100 % | 94 % | 92 % | 24 |
| Newton projeté (log θ) | 100 % | 100 % | 100 % | 100 % | 20 |
| L-BFGS-B (log θ) | 100 % | 100 % | 100 % | 100 % | 39 |
| Recherche aléatoire | 0 % | 0 % | 0 % | 0 % | 20 000 |
| Recherche localisée | 100 % | 100 % | 100 % | 100 % | 20 000 |
| Metropolis | 100 % | 100 % | 94 % | 92 % | 20 000 |
| Recuit simulé | 100 % | 100 % | 94 % | 97 % | 20 000 |
| Évolution différentielle | 100 % | 97 % | 94 % | 94 % | 3 150 |
| DE + L-BFGS-B (retenue) | 100 % | 100 % | 100 % | 100 % | 3 212 |

## Erreur relative de la méthode retenue par rapport aux vrais paramètres

| Bruit | ka | ke | V |
|---|---|---|---|
| 5 % | 5.3 % (90 % des cas < 10 %) | 5.0 % (90 % des cas < 9 %) | 1.5 % (90 % des cas < 5 %) |
| 10 % | 7.5 % (90 % des cas < 27 %) | 6.1 % (90 % des cas < 17 %) | 3.1 % (90 % des cas < 10 %) |
| 25 % | 19.9 % (90 % des cas < 48 %) | 22.1 % (90 % des cas < 49 %) | 11.2 % (90 % des cas < 20 %) |
