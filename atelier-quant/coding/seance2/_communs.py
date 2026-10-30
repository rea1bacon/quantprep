"""Outils communs aux tests de la séance de code 2 : comparaisons, générateurs de données aléatoires."""
import math

import numpy as np
import pandas as pd

LIMITE_S = 10.0
TOLERANCE = 1e-9


def proches(a, b):
    a, b = float(a), float(b)
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= TOLERANCE


def tableaux_proches(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= TOLERANCE)))


def series_egales(obtenu, attendu):
    """Mêmes étiquettes d'index (ordre libre, sans doublon), mêmes valeurs à 1e-9 près, NaN aux mêmes places."""
    if not isinstance(obtenu, pd.Series) or len(obtenu) != len(attendu):
        return False
    if obtenu.index.has_duplicates or set(obtenu.index) != set(attendu.index):
        return False
    try:
        return tableaux_proches(obtenu.reindex(attendu.index).to_numpy(dtype=float), attendu.to_numpy(dtype=float))
    except (TypeError, ValueError):
        return False


def cadres_egaux(obtenu, attendu):
    """Mêmes colonnes dans le même ordre, mêmes étiquettes d'index (ordre libre), valeurs numériques à 1e-9 près."""
    if not isinstance(obtenu, pd.DataFrame) or list(obtenu.columns) != list(attendu.columns) or len(obtenu) != len(attendu):
        return False
    if obtenu.index.has_duplicates or set(obtenu.index) != set(attendu.index):
        return False
    try:
        return tableaux_proches(obtenu.reindex(attendu.index).to_numpy(dtype=float), attendu.to_numpy(dtype=float))
    except (TypeError, ValueError):
        return False


def dates_ouvrees(n, debut="2026-01-01"):
    """n jours ouvrés (lundi à vendredi) à partir de debut, en chaînes AAAA-MM-JJ."""
    return list(pd.bdate_range(debut, periods=n).strftime("%Y-%m-%d"))


def melanger(rng, lignes):
    return [lignes[i] for i in rng.permutation(len(lignes))]


def panel(rng, n_dates, n_actifs, p_absent=0.1, p_nan=0.1, avec_signal=True, signaux_entiers=False):
    """Lignes (date, actif, prix, signal) ou (date, actif, prix), dans un ordre aléatoire, avec trous et valeurs manquantes."""
    dates = dates_ouvrees(n_dates)
    lignes = []
    for a in range(n_actifs):
        prix = 100 * np.exp(np.cumsum(0.02 * rng.standard_normal(n_dates)))
        for i, d in enumerate(dates):
            if rng.random() < p_absent:
                continue
            p = float("nan") if rng.random() < p_nan else float(prix[i])
            if not avec_signal:
                lignes.append((d, f"A{a:04d}", p))
                continue
            s = float(rng.integers(-2, 3)) if signaux_entiers else float(rng.standard_normal())
            lignes.append((d, f"A{a:04d}", p, float("nan") if rng.random() < p_nan else s))
    return melanger(rng, lignes)
