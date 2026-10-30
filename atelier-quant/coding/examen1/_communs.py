"""Outils communs aux tests de la question de code de l'examen blanc 1 : comparaison, panels aléatoires."""
import math

import numpy as np
import pandas as pd

LIMITE_S = 10.0
COLONNES = ["date", "actif", "prix"]
NAN = float("nan")


def proches(a, b):
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-9


def cadre(lignes, index=None):
    return pd.DataFrame(lignes, columns=COLONNES, index=index)


def panel(rng, n_dates, n_actifs, p_absent=0.1, p_nan=0.1):
    """Lignes (date, actif, prix) dans un ordre aléatoire, avec dates manquantes et prix manquants."""
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="D").strftime("%Y-%m-%d")
    lignes = []
    for a in range(n_actifs):
        prix = 100 * np.exp(np.cumsum(0.02 * rng.standard_normal(n_dates)))
        for i, d in enumerate(dates):
            if rng.random() < p_absent:
                continue
            lignes.append((d, f"A{a:04d}", NAN if rng.random() < p_nan else float(prix[i])))
    return [lignes[i] for i in rng.permutation(len(lignes))]


def panel_egalites(rng, n_dates, n_actifs):
    """Prix entiers (stockés en flottants) de la forme b * m / 10, b multiple de 10, m entier de 8 à 12 :
    le rapport de deux prix successifs vaut m'/m, le même réel pour beaucoup d'actifs, donc le même flottant.
    Les égalités de rendement passé sont alors nombreuses, et la règle de départage par l'actif compte."""
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="D").strftime("%Y-%m-%d")
    lignes = []
    for a in range(n_actifs):
        b = int(rng.choice([10, 20, 50, 100]))
        for d in dates:
            lignes.append((d, f"A{a:04d}", float(b * int(rng.integers(8, 13)) // 10)))
    return [lignes[i] for i in rng.permutation(len(lignes))]
