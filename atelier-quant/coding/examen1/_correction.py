"""Correction de la question de code de l'examen blanc 1. À n'ouvrir qu'après l'avoir tentée.

Deux implémentations indépendantes : l'une avec pandas, l'autre sans aucune bibliothèque externe.
Les tests exigent qu'elles soient d'accord.
"""
import math

import numpy as np
import pandas as pd


def rendement_long_short(df: pd.DataFrame, k: int) -> float:
    """Solution de référence avec pandas (compatible pandas 1.4.2 et 2.x)."""
    # 1. Trier par actif puis par date : dans chaque actif, la ligne précédente (suivante) du cadre trié
    #    est alors la ligne chronologiquement précédente (suivante). Le format AAAA-MM-JJ trie comme les dates.
    d = df.sort_values(["actif", "date"])
    prix = d.groupby("actif")["prix"]
    # 2. Décalages À L'INTÉRIEUR de chaque actif, sur le cadre trié.
    #    Pas de pct_change : il remplace d'abord un prix manquant par le dernier prix connu
    #    (fill_method="pad" par défaut, vérifié en pandas 1.4.2 et 2.3.3), ce que l'énoncé interdit.
    d = d.assign(passe=d["prix"] / prix.shift(1) - 1, futur=prix.shift(-1) / d["prix"] - 1)
    # 3. Lignes exploitables : les deux rendements sont connus.
    d = d.dropna(subset=["passe", "futur"])
    rendements = []
    for _, g in d.groupby("date"):
        if len(g) < 2 * k:
            continue  # date ignorée : pas assez de lignes pour deux jambes disjointes de k actifs
        # 4. Ordre de l'énoncé : rendement passé croissant, puis identifiant d'actif croissant en cas d'égalité.
        #    Les k premières lignes forment la jambe vendeuse, les k dernières la jambe acheteuse.
        g = g.sort_values(["passe", "actif"])
        rendements.append(g["futur"].iloc[-k:].mean() - g["futur"].iloc[:k].mean())
    # 5. Moyenne sur les dates retenues ; NaN si aucune.
    return float(np.mean(rendements)) if rendements else float("nan")


def rendement_long_short_pur(lignes, k):
    """Même calcul sans pandas ni numpy. `lignes` : liste de tuples (date, actif, prix), None ou NaN = manquant."""
    manquant = lambda x: x is None or (isinstance(x, float) and math.isnan(x))
    par_actif = {}
    for date, actif, prix in lignes:
        par_actif.setdefault(actif, []).append((date, prix))
    par_date = {}
    for actif, serie in par_actif.items():
        serie.sort(key=lambda c: c[0])  # une seule ligne par (date, actif) : tri chronologique sans ambiguïté
        for j in range(1, len(serie) - 1):
            (_, p0), (date, p1), (_, p2) = serie[j - 1], serie[j], serie[j + 1]
            if manquant(p0) or manquant(p1) or manquant(p2):
                continue
            par_date.setdefault(date, []).append((p1 / p0 - 1, actif, p2 / p1 - 1))
    rendements = []
    for lignes_date in par_date.values():
        if len(lignes_date) < 2 * k:
            continue
        lignes_date.sort()  # rendement passé croissant, puis actif croissant : (passé, actif) est unique par date
        vendeuse = [c[2] for c in lignes_date[:k]]
        acheteuse = [c[2] for c in lignes_date[-k:]]
        rendements.append(sum(acheteuse) / k - sum(vendeuse) / k)
    return sum(rendements) / len(rendements) if rendements else float("nan")
