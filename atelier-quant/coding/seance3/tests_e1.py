"""Exercice 1 : nettoyer(df). Usage autonome : python tests_e1.py reference"""
import math
import sys

import numpy as np
import pandas as pd

from _communs import LIMITE_S, NUMERIQUES, abonnes, cadres_numeriques_egaux, textes_egaux  # noqa: F401  (LIMITE_S lu par le noyau)
from _correction import nettoyer as reference
from _correction import nettoyer_pur

FONCTION = "nettoyer"
NAN = float("nan")
COLONNES = ["id", "age", "revenu", "appels", "anciennete_mois", "forfait"]

EXEMPLE = [
    (11, "34 ans", "2 450 €", "trois", 12.0, " Premium "),
    (12, "inconnu", "1980", "0", -1.0, "base"),
    (13, " 51", NAN, "cinq", NAN, "PLUS"),
    (14, "27", "3 100,50", "2", 40.0, "plus"),
]


def cadre(lignes, index=None, colonnes=COLONNES):
    return pd.DataFrame(lignes, columns=colonnes, index=index)


def comparer(obtenu, attendu):
    """Mêmes colonnes dans le même ordre, mêmes étiquettes d'index (ordre libre) ; colonnes numériques à 1e-9 près
    avec NaN aux mêmes places ; forfait : mêmes chaînes, manquants aux mêmes places ; autres colonnes inchangées."""
    if not isinstance(obtenu, pd.DataFrame) or list(obtenu.columns) != list(attendu.columns) or len(obtenu) != len(attendu):
        return False
    if obtenu.index.has_duplicates or set(obtenu.index) != set(attendu.index):
        return False
    # Les quatre colonnes nettoyées doivent être numériques : une colonne de chaînes "40" n'est pas une colonne de nombres.
    if not all(pd.api.types.is_numeric_dtype(obtenu[c]) and not pd.api.types.is_bool_dtype(obtenu[c]) for c in NUMERIQUES):
        return False
    if not cadres_numeriques_egaux(obtenu[NUMERIQUES], attendu[NUMERIQUES]):
        return False
    autres = [c for c in attendu.columns if c not in NUMERIQUES]
    try:
        return all(textes_egaux(obtenu[c], attendu[c]) for c in autres)
    except (TypeError, ValueError):
        return False


def _cas(nom, df):
    avant = df.copy()
    attendu = reference(df)
    assert df.equals(avant), f"la correction modifie son entrée : {nom}"
    pur = pd.DataFrame(nettoyer_pur(df.to_dict("records")), columns=df.columns, index=df.index)
    pur[NUMERIQUES] = pur[NUMERIQUES].astype(float)   # un cadre sans ligne construit depuis une liste vide a des colonnes object
    assert comparer(pur, attendu), f"correction incohérente : {nom}"
    return {"nom": nom, "args": (df,), "attendu": attendu}


def cas_visibles():
    return [
        _cas("exemple de l'énoncé", cadre(EXEMPLE)),
        _cas("une seule ligne, déjà propre", cadre([(1, "40", "2000", "1", 5.0, "base")])),
    ]


def cas_caches():
    grand, _ = abonnes(3101, 20000, p_manquant=0.1, p_sale=0.5)
    sans_lettres, _ = abonnes(3102, 300, appels_en_lettres=0)
    avec_cible, y = abonnes(3103, 200)
    avec_cible["resilie"] = y
    melange, _ = abonnes(3104, 150, p_manquant=0.2, part_pro=0.2)
    melange.index = np.random.default_rng(3105).permutation(150) + 1000
    return [
        _cas("DataFrame vide (colonnes présentes)", cadre([])),
        _cas("index d'étiquettes mélangées", cadre(EXEMPLE, index=[40, 7, 13, 2])),
        _cas("textes illisibles : valeurs manquantes",
             cadre([(1, "", "n/a", "beaucoup", 3.0, "base"), (2, "trente", "€", "", 0.0, "plus"), (3, "34ans", "12 000,5 €", " deux ", -0.5, "PREMIUM ")])),
        _cas("colonnes revenu et forfait entièrement manquantes", cadre([(1, "20", NAN, "un", 1.0, NAN), (2, "30 ans", NAN, "4", 2.0, NAN)])),
        _cas("colonnes age, revenu et appels déjà numériques",
             cadre([(1, 34, 2450.0, 3, 12.0, "base"), (2, 51, 1980.5, 0, -3.0, " Plus"), (3, 27, 3100.0, 5, 0.0, "premium")])),
        _cas("appels en chiffres seulement (colonne d'entiers)", sans_lettres),
        _cas("colonne supplémentaire (la cible) laissée inchangée", avec_cible),
        _cas("aléatoire, nombreuses valeurs manquantes, index non trivial", melange),
        _cas(f"grande taille ({len(grand)} lignes)", grand),
    ]


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "reference":
    tous = cas_visibles() + cas_caches()
    ok = sum(comparer(reference(*c["args"]), c["attendu"]) for c in tous)
    print(f"{ok}/{len(tous)} ; visibles {len(cas_visibles())}, cachés {len(cas_caches())}")
    sys.exit(0 if ok == len(tous) else 1)
