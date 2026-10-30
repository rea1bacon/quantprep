"""Corrections de la séance de code 1. À n'ouvrir qu'après avoir tenté les exercices.

Chaque fonction existe en deux versions indépendantes : pandas, et sans bibliothèque externe.
Les tests exigent qu'elles soient d'accord.
"""
import math

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ exercice 1
def rendement_futur(df: pd.DataFrame) -> pd.Series:
    # 1. Trier par actif puis par date : la ligne suivante d'un actif devient sa ligne chronologiquement suivante.
    d = df.sort_values(["actif", "date"])
    # 2. Décaler À L'INTÉRIEUR de chaque actif, sur le cadre trié. Décaler le cadre non trié donnerait
    #    la ligne suivante dans l'ordre d'arrivée des données, qui n'a aucun sens chronologique.
    prix_suivant = d.groupby("actif")["prix"].shift(-1)
    r = prix_suivant / d["prix"] - 1
    # 3. Revenir à l'index d'origine : pandas aligne par étiquette d'index, pas par position.
    return r.reindex(df.index)


def rendement_futur_pur(lignes):
    """lignes : liste de (index, date, actif, prix). Renvoie {index: rendement ou nan}."""
    par_actif = {}
    for i, date, actif, prix in lignes:
        par_actif.setdefault(actif, []).append((date, i, prix))
    sortie = {}
    for serie in par_actif.values():
        serie.sort()
        for k, (date, i, prix) in enumerate(serie):
            suivant = serie[k + 1][2] if k + 1 < len(serie) else float("nan")
            sortie[i] = suivant / prix - 1 if not (math.isnan(prix) or math.isnan(suivant)) else float("nan")
    return sortie


# ------------------------------------------------------------------ exercice 2
def rang_transversal(df: pd.DataFrame) -> pd.Series:
    # rank() : ordre croissant, rang moyen pour les égalités, NaN laissés NaN. Calculé date par date.
    return df.groupby("date")["signal"].rank()


def rangs_moyens(valeurs):
    """Rangs moyens (à partir de 1) d'une liste de nombres sans NaN."""
    ordre = sorted(range(len(valeurs)), key=lambda i: valeurs[i])
    rangs = [0.0] * len(valeurs)
    i = 0
    while i < len(ordre):
        j = i
        while j + 1 < len(ordre) and valeurs[ordre[j + 1]] == valeurs[ordre[i]]:
            j += 1
        for k in range(i, j + 1):
            rangs[ordre[k]] = (i + j) / 2 + 1
        i = j + 1
    return rangs


def rang_transversal_pur(lignes):
    """lignes : liste de (index, date, signal). Renvoie {index: rang ou nan}."""
    par_date = {}
    sortie = {}
    for i, date, signal in lignes:
        if math.isnan(signal):
            sortie[i] = float("nan")
        else:
            par_date.setdefault(date, []).append((i, signal))
    for couples in par_date.values():
        for (i, _), r in zip(couples, rangs_moyens([s for _, s in couples])):
            sortie[i] = r
    return sortie


# ------------------------------------------------------------------ exercice 3
def correlation_par_date(df: pd.DataFrame) -> pd.Series:
    def correlation(g):
        g = g.dropna()
        if len(g) < 3 or g["x"].nunique() == 1 or g["y"].nunique() == 1:
            return np.nan
        return np.corrcoef(g["x"], g["y"])[0, 1]

    if df.empty:
        return pd.Series(dtype=float)
    # Sélectionner les colonnes avant apply : fonctionne sans avertissement en pandas 1.4 comme en 2.x.
    return df.groupby("date")[["x", "y"]].apply(correlation).astype(float)


def correlation_pure(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    return sxy / math.sqrt(sxx * syy)


def correlation_par_date_pur(lignes):
    """lignes : liste de (date, x, y). Renvoie une liste de (date, corrélation ou nan) triée par date."""
    par_date = {}
    for date, x, y in lignes:
        par_date.setdefault(date, [])
        if not (math.isnan(x) or math.isnan(y)):
            par_date[date].append((x, y))
    sortie = []
    for date in sorted(par_date):
        couples = par_date[date]
        xs, ys = [c[0] for c in couples], [c[1] for c in couples]
        if len(couples) < 3 or len(set(xs)) == 1 or len(set(ys)) == 1:
            sortie.append((date, float("nan")))
        else:
            sortie.append((date, correlation_pure(xs, ys)))
    return sortie
