"""Exercice 3 : predire(train, test), jugée par son exactitude sur des étiquettes que la fonction ne voit pas.

Les seuils sont fixés d'après les mesures de verif/seance_code_3.py (classe majoritaire, solution bâclée, correction),
qui vérifie aussi, dans les deux environnements, que la correction passe chaque seuil avec une marge.
Usage autonome : python tests_e3.py reference
"""
import sys

import numpy as np
import pandas as pd

from _communs import jeu
from _correction import predire as reference

FONCTION = "predire"
LIMITE_S = 30.0    # un entraînement par test : plus large que les 20 secondes des exercices 1 et 2


class Exactitude:
    """Résultat attendu d'un test : des prédictions 0/1, une par ligne du test, d'exactitude au moins égale au seuil."""

    def __init__(self, y, seuil):
        self.y, self.seuil = np.asarray(y), seuil

    def __repr__(self):
        return f"{len(self.y)} prédictions 0 ou 1, exactitude >= {self.seuil:.2f}"

    def mesurer(self, obtenu):
        """Exactitude des prédictions, ou None si elles n'ont pas le bon format."""
        if isinstance(obtenu, (pd.Series, pd.DataFrame)):
            obtenu = obtenu.to_numpy()
        try:
            p = np.asarray(obtenu, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if p.shape != self.y.shape or not np.isin(p, [0.0, 1.0]).all():
            return None
        return float((p == self.y).mean())


def comparer(obtenu, attendu):
    acc = attendu.mesurer(obtenu)
    return acc is not None and acc >= attendu.seuil


# (nom, graine, lignes d'apprentissage, lignes de test, seuil, options du générateur)
VISIBLES = [
    ("format : 5 lignes de test, 5 prédictions 0 ou 1", 3301, 400, 5, 0.0, {}),
    ("jeu standard", 3303, 1500, 1000, 0.78, {}),
]
CACHES = [
    ("format : une seule ligne de test", 3311, 300, 1, 0.0, {}),
    ("jeu standard, autre tirage", 3313, 1500, 1000, 0.78, {}),
    ("modalité de forfait inconnue au test", 3315, 1500, 1000, 0.78, {"part_pro_test": 0.25}),
    ("nombreuses valeurs manquantes, à l'apprentissage et au test", 3317, 1500, 1000, 0.78, {"p_manquant": 0.15}),
    ("appels en chiffres seulement (colonne d'entiers)", 3319, 1500, 1000, 0.78, {"appels_en_lettres": 0}),
    ("petit jeu d'apprentissage (200 lignes)", 3321, 200, 1000, 0.76, {}),
    ("index non trivial au test", 3323, 1500, 1000, 0.78, {}),
    ("grande taille (20000 lignes d'apprentissage)", 3325, 20000, 5000, 0.78, {}),
]


def _cas(nom, graine, n_train, n_test, seuil, options):
    train, test, y = jeu(graine, n_train, n_test, **options)
    if "index non trivial" in nom:
        test.index = np.random.default_rng(graine).permutation(len(test)) + 500
    return {"nom": nom, "args": (train, test), "attendu": Exactitude(y, seuil)}


def cas_visibles():
    return [_cas(*c) for c in VISIBLES]


def cas_caches():
    return [_cas(*c) for c in CACHES]


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "reference":
    tous = cas_visibles() + cas_caches()
    ok = sum(comparer(reference(*c["args"]), c["attendu"]) for c in tous)
    print(f"{ok}/{len(tous)} ; visibles {len(cas_visibles())}, cachés {len(cas_caches())}")
    sys.exit(0 if ok == len(tous) else 1)
