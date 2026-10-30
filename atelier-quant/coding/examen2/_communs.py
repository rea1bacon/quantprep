"""Outils communs aux tests de la question de code de l'examen blanc 2 : cadres typés, comparaison, données aléatoires."""
import math

import numpy as np
import pandas as pd

LIMITE_S = 10.0
COLONNES_T = ["temps", "actif", "prix", "quantite"]
COLONNES_C = ["temps", "actif", "achat", "vente"]
NAN = float("nan")


def cadre_transactions(lignes, index=None):
    """Types garantis par l'énoncé, même pour un cadre vide : temps et quantite int64, prix float64."""
    df = pd.DataFrame(list(lignes), columns=COLONNES_T, index=index)
    return df.astype({"temps": "int64", "actif": object, "prix": "float64", "quantite": "int64"})


def cadre_cotations(lignes, index=None):
    df = pd.DataFrame(list(lignes), columns=COLONNES_C, index=index)
    return df.astype({"temps": "int64", "actif": object, "achat": "float64", "vente": "float64"})


def lignes_de(df):
    return [tuple(x) for x in df.itertuples(index=False)]


def serie_de(paires):
    """Liste de (actif, valeur) -> Series indexée par actif."""
    return pd.Series([v for _, v in paires], index=[a for a, _ in paires], dtype=float)


def comparer(obtenu, attendu):
    """Series, mêmes étiquettes dans le même ordre (actifs triés), valeurs à 1e-9 près, NaN aux mêmes places."""
    if not isinstance(obtenu, pd.Series) or len(obtenu) != len(attendu):
        return False
    if [str(x) for x in obtenu.index] != [str(x) for x in attendu.index]:
        return False
    try:
        a = obtenu.to_numpy(dtype=float)
    except (TypeError, ValueError):
        return False
    b = attendu.to_numpy(dtype=float)
    return bool(np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-9)))


def proches(a, b):
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-9


def aleatoire(rng, n_transactions, n_cotations, n_actifs, p_invalide=0.1, p_nan=0.05, duree_ms=600_000,
              p_piege=0.15):
    """Transactions et cotations aléatoires, lignes dans le désordre.

    Tous les prix sont des multiples de 0,25. Une part p_piege des transactions est placée exactement
    sur le temps d'une cotation du même actif, ou 5000 ms ou 5001 ms après elle, pour exercer les bornes.
    """
    noms = [f"X{a:03d}" for a in range(n_actifs)]
    base = rng.integers(40, 2000, size=n_actifs) * 4  # prix de référence de chaque actif, en quarts d'unité
    par_actif = max(1, min(n_cotations // n_actifs, duree_ms))
    # Cotations : temps distincts à l'intérieur de chaque actif.
    actif_c = np.repeat(np.arange(n_actifs), par_actif)
    temps_c = np.concatenate([rng.choice(duree_ms, size=par_actif, replace=False) for _ in range(n_actifs)])
    achat_q = base[actif_c] + rng.integers(-8, 9, size=len(actif_c))
    vente_q = achat_q + rng.integers(1, 5, size=len(actif_c))
    invalide = rng.random(len(actif_c)) < p_invalide
    vente_q = np.where(invalide, achat_q - rng.integers(0, 3, size=len(actif_c)), vente_q)  # bloquée ou croisée
    achat, vente = achat_q / 4, vente_q / 4
    incomplete = rng.random(len(actif_c)) < p_nan
    cote_nan = rng.random(len(actif_c)) < 0.5
    achat = np.where(incomplete & cote_nan, np.nan, achat)
    vente = np.where(incomplete & ~cote_nan, np.nan, vente)
    # Transactions : une part p_piege est calée sur le temps d'une cotation du même actif (+0, +5000 ou +5001 ms).
    actif_t = rng.integers(n_actifs, size=n_transactions)
    temps_t = rng.integers(duree_ms + 6000, size=n_transactions)
    piege = rng.random(n_transactions) < p_piege
    rang = rng.integers(par_actif, size=n_transactions)
    decalage = np.array([0, 5000, 5001])[rng.integers(3, size=n_transactions)]
    temps_t = np.where(piege, temps_c[actif_t * par_actif + rang] + decalage, temps_t)
    prix = (base[actif_t] + rng.integers(-10, 12, size=n_transactions)) / 4
    quantite = rng.integers(1, 1000, size=n_transactions)
    transactions = [(int(t), noms[a], float(p), int(q)) for t, a, p, q in zip(temps_t, actif_t, prix, quantite)]
    cotations = [(int(t), noms[a], float(b), float(v)) for t, a, b, v in zip(temps_c, actif_c, achat, vente)]
    ordre_t = rng.permutation(len(transactions))
    ordre_c = rng.permutation(len(cotations))
    return [transactions[i] for i in ordre_t], [cotations[i] for i in ordre_c]
