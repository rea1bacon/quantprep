"""Outils communs à la séance de code 3 : générateur de données synthétiques sales (abonnés d'un service), comparaisons.

Les données sont inventées. Le générateur n'utilise que numpy (graine fixe) et produit les mêmes cadres dans les deux
environnements (python de la machine et .venv-hackerrank).
"""
import math

import numpy as np
import pandas as pd

LIMITE_S = 20.0
TOLERANCE = 1e-9

MOTS = ["zéro", "un", "deux", "trois", "quatre", "cinq"]          # appels au support, de 0 à 5, écrits en lettres
FORFAITS = ["base", "plus", "premium"]
NUMERIQUES = ["age", "revenu", "appels", "anciennete_mois"]
COLONNES = ["id", "age", "revenu", "appels", "anciennete_mois", "forfait"]
CIBLE = "resilie"

# Modèle qui fabrique la cible : probabilité de résiliation = 1 / (1 + exp(-PENTE * z)).
ORDONNEE = -0.3
EFFET_APPELS = 1.0         # par 1,5 appel au-dessus de 2
EFFET_FORFAIT = {"base": 0.7, "plus": -0.7, "premium": 0.2, "pro": 0.0}   # non monotone : une étiquette, pas une quantité
EFFET_ANCIENNETE = -0.8    # par 20 mois au-dessus de 36
EFFET_REVENU = -0.6        # par 700 euros au-dessus de 2600
EFFET_AGE = 0.3            # par 12 ans au-dessus de 45
PENTE = 2.0


def abonnes(graine, n, p_manquant=0.04, p_sale=0.35, part_pro=0.0, appels_en_lettres=0.4, premier_id=1):
    """Rend (cadre sale aux colonnes COLONNES, tableau des cibles 0/1).

    p_manquant : probabilité, colonne par colonne, qu'une valeur manque (ou soit impossible pour l'ancienneté).
    p_sale : probabilité qu'une valeur présente soit écrite sous une forme sale (unité, séparateur, casse, espaces).
    part_pro : part des abonnés au forfait "pro", modalité absente quand part_pro = 0.
    appels_en_lettres : part des nombres d'appels écrits en lettres ; à 0, la colonne est une colonne d'entiers."""
    rng = np.random.default_rng(graine)
    age = rng.integers(18, 76, n)
    revenu = np.round(2600 + 700 * rng.standard_normal(n)).clip(900, 6000).astype(int)
    appels = rng.integers(0, 6, n)
    anciennete = rng.integers(1, 73, n)
    forfait = np.array(FORFAITS)[rng.integers(0, 3, n)].astype(object)
    forfait[rng.random(n) < part_pro] = "pro"
    z = (ORDONNEE + EFFET_APPELS * (appels - 2) / 1.5 + np.array([EFFET_FORFAIT[f] for f in forfait])
         + EFFET_ANCIENNETE * (anciennete - 36) / 20 + EFFET_REVENU * (revenu - 2600) / 700 + EFFET_AGE * (age - 45) / 12)
    y = (rng.random(n) < 1 / (1 + np.exp(-PENTE * z))).astype(int)
    # Saletés, tirées après la cible : elles ne portent aucune information sur elle.
    manque = rng.random((n, 4)) < p_manquant
    sale = rng.random((n, 5)) < p_sale
    forme = rng.integers(0, 3, (n, 5))
    lignes = []
    for i in range(n):
        a = "inconnu" if manque[i, 0] else (f"{age[i]} ans" if sale[i, 0] else str(age[i]))
        if manque[i, 1]:
            r = math.nan
        elif sale[i, 1]:
            r = [f"{revenu[i]:,}".replace(",", " "), f"{revenu[i]:,}".replace(",", " ") + " €", f"{revenu[i]},00"][forme[i, 1]]
        else:
            r = str(revenu[i])
        if appels_en_lettres == 0:
            c = int(appels[i])
        else:
            c = MOTS[appels[i]] if rng.random() < appels_en_lettres else str(appels[i])
        m = -1.0 if manque[i, 2] else (math.nan if manque[i, 3] else float(anciennete[i]))
        f = str(forfait[i])
        if sale[i, 4]:
            f = [f.capitalize(), f.upper(), " " + f + " "][forme[i, 4]]
        lignes.append((premier_id + i, a, r, c, m, f))
    return pd.DataFrame(lignes, columns=COLONNES), y


def jeu(graine, n_train, n_test, part_pro_test=0.0, **options):
    """Rend (train sale avec la cible, test sale sans la cible, cibles du test)."""
    train, y_train = abonnes(graine, n_train, **options)
    test, y_test = abonnes(graine + 1, n_test, part_pro=part_pro_test, premier_id=100001, **options)
    train[CIBLE] = y_train
    return train, test, y_test


def tableaux_proches(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= TOLERANCE)))


def cadres_numeriques_egaux(obtenu, attendu):
    """Mêmes colonnes dans le même ordre, mêmes étiquettes d'index (ordre libre, sans doublon), valeurs à 1e-9 près."""
    if not isinstance(obtenu, pd.DataFrame) or list(obtenu.columns) != list(attendu.columns) or len(obtenu) != len(attendu):
        return False
    if obtenu.index.has_duplicates or set(obtenu.index) != set(attendu.index):
        return False
    try:
        return tableaux_proches(obtenu.reindex(attendu.index).to_numpy(dtype=float), attendu.to_numpy(dtype=float))
    except (TypeError, ValueError):
        return False


def textes_egaux(obtenu, attendu):
    """Deux séries de chaînes de même index : mêmes chaînes, valeurs manquantes aux mêmes places."""
    o, a = obtenu.reindex(attendu.index), attendu
    if not (o.isna().to_numpy() == a.isna().to_numpy()).all():
        return False
    return bool((o[a.notna()] == a[a.notna()]).all())
