"""Outils communs aux tests de la séance de code 1 : comparaison de séries, données aléatoires."""
import math

import numpy as np
import pandas as pd

LIMITE_S = 10.0


def series_egales(obtenu, attendu, meme_ordre=False):
    """Mêmes étiquettes d'index (dans le même ordre si meme_ordre), mêmes valeurs à 1e-9 près, NaN aux mêmes places."""
    if not isinstance(obtenu, pd.Series) or len(obtenu) != len(attendu):
        return False
    if meme_ordre:
        if list(obtenu.index) != list(attendu.index):
            return False
    else:
        if set(obtenu.index) != set(attendu.index) or obtenu.index.has_duplicates:
            return False
        obtenu = obtenu.reindex(attendu.index)
    try:
        a = obtenu.to_numpy(dtype=float)
    except (TypeError, ValueError):
        return False
    b = attendu.to_numpy(dtype=float)
    return bool(np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-9)))


def proches(a, b):
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-9


def panel(rng, n_dates, n_actifs, p_absent=0.1, p_nan=0.1, signaux_entiers=False):
    """Lignes (date, actif, prix, signal) dans un ordre aléatoire, avec trous et valeurs manquantes."""
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="D").strftime("%Y-%m-%d")
    lignes = []
    for a in range(n_actifs):
        prix = 100 * np.exp(np.cumsum(0.02 * rng.standard_normal(n_dates)))
        for i, d in enumerate(dates):
            if rng.random() < p_absent:
                continue
            signal = float(rng.integers(-2, 3)) if signaux_entiers else float(rng.standard_normal())
            lignes.append((d, f"A{a:04d}", float("nan") if rng.random() < p_nan else float(prix[i]),
                           float("nan") if rng.random() < p_nan else signal))
    return [lignes[i] for i in rng.permutation(len(lignes))]
