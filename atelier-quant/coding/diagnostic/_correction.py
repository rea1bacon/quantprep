"""Correction de la question de code du test de positionnement. À n'ouvrir qu'après l'avoir tentée.

Deux implémentations indépendantes : l'une avec pandas, l'autre sans aucune
bibliothèque externe. Les tests cachés exigent qu'elles soient d'accord.
"""
import math

import numpy as np
import pandas as pd


def ic_moyen(df: pd.DataFrame) -> float:
    """Solution de référence avec pandas."""
    # Tri par actif puis par date : la ligne suivante d'un actif est alors sa ligne chronologiquement suivante.
    # Le format AAAA-MM-JJ fait coïncider l'ordre des chaînes avec l'ordre chronologique.
    d = df.sort_values(["actif", "date"])
    # shift(-1) à l'intérieur de chaque actif : NaN sur la dernière ligne de l'actif, jamais le prix d'un autre actif.
    prix_suivant = d.groupby("actif")["prix"].shift(-1)
    d = d.assign(rendement=prix_suivant / d["prix"] - 1)
    d = d.dropna(subset=["signal", "rendement"])
    ics = []
    for _, g in d.groupby("date"):
        if len(g) < 3:
            continue
        rang_signal = g["signal"].rank()  # méthode par défaut : rang moyen pour les égalités
        rang_rendement = g["rendement"].rank()
        if rang_signal.nunique() == 1 or rang_rendement.nunique() == 1:
            continue
        ics.append(np.corrcoef(rang_signal, rang_rendement)[0, 1])
    return float(np.mean(ics)) if ics else float("nan")


def _rangs_moyens(valeurs):
    ordre = sorted(range(len(valeurs)), key=lambda i: valeurs[i])
    rangs = [0.0] * len(valeurs)
    i = 0
    while i < len(ordre):
        j = i
        while j + 1 < len(ordre) and valeurs[ordre[j + 1]] == valeurs[ordre[i]]:
            j += 1
        for k in range(i, j + 1):
            rangs[ordre[k]] = (i + j) / 2 + 1  # moyenne des rangs i+1, ..., j+1
        i = j + 1
    return rangs


def ic_moyen_sans_bibliotheque(lignes) -> float:
    """Même calcul sans pandas ni numpy. `lignes` : liste de tuples (date, actif, prix, signal), None ou NaN = manquant."""
    manquant = lambda x: x is None or (isinstance(x, float) and math.isnan(x))
    par_actif = {}
    for date, actif, prix, signal in lignes:
        par_actif.setdefault(actif, []).append((date, prix, signal))
    par_date = {}
    for serie in par_actif.values():
        serie.sort()  # les dates sont uniques par actif : le tri ne compare jamais les prix
        for (date, prix, signal), (_, prix_suivant, _) in zip(serie, serie[1:]):
            if manquant(prix) or manquant(prix_suivant) or manquant(signal):
                continue
            par_date.setdefault(date, []).append((signal, prix_suivant / prix - 1))
    ics = []
    for couples in par_date.values():
        if len(couples) < 3:
            continue
        rs = _rangs_moyens([c[0] for c in couples])
        rr = _rangs_moyens([c[1] for c in couples])
        ms, mr = sum(rs) / len(rs), sum(rr) / len(rr)
        var_s = sum((a - ms) ** 2 for a in rs)
        var_r = sum((b - mr) ** 2 for b in rr)
        if var_s == 0 or var_r == 0:
            continue
        ics.append(sum((a - ms) * (b - mr) for a, b in zip(rs, rr)) / math.sqrt(var_s * var_r))
    return sum(ics) / len(ics) if ics else float("nan")
