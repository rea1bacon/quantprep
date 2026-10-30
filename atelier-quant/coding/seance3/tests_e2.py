"""Exercice 2 : aligner(train, test, colonnes). Usage autonome : python tests_e2.py reference"""
import sys

import numpy as np
import pandas as pd

from _communs import LIMITE_S, cadres_numeriques_egaux  # noqa: F401  (LIMITE_S lu par le noyau)
from _correction import aligner as reference
from _correction import aligner_pur

FONCTION = "aligner"
NAN = float("nan")

TRAIN = pd.DataFrame({"revenu": [2450.0, 1980.0, 3100.0, 2700.0], "forfait": ["plus", "base", "plus", "premium"]})
TEST = pd.DataFrame({"revenu": [2000.0, 2600.0, 3300.0], "forfait": ["base", "pro", "base"]})


def comparer(obtenu, attendu):
    """Un couple de deux DataFrame, chacun égal à l'attendu : colonnes dans le même ordre, index, valeurs à 1e-9 près."""
    if not isinstance(obtenu, (tuple, list)) or len(obtenu) != 2:
        return False
    return all(cadres_numeriques_egaux(o, a) for o, a in zip(obtenu, attendu))


def _cas(nom, train, test, colonnes):
    avant = (train.copy(), test.copy())
    attendu = reference(train, test, colonnes)
    assert train.equals(avant[0]) and test.equals(avant[1]), f"la correction modifie ses entrées : {nom}"
    pur = aligner_pur(train.to_dict("records"), test.to_dict("records"), colonnes)
    purs = tuple(pd.DataFrame(p, columns=a.columns, index=d.index) for p, a, d in zip(pur, attendu, (train, test)))
    assert [list(p[0]) for p in pur if p] == [list(a.columns) for p, a in zip(pur, attendu) if p], f"ordre des colonnes : {nom}"
    assert comparer(purs, attendu), f"correction incohérente : {nom}"
    assert list(attendu[0].columns) == list(attendu[1].columns)
    return {"nom": nom, "args": (train, test, colonnes), "attendu": attendu}


def aleatoire(rng, n, modalites_forfait, modalites_ligne, p_nan=0.0):
    forfait = np.array(modalites_forfait, dtype=object)[rng.integers(0, len(modalites_forfait), n)]
    forfait[rng.random(n) < p_nan] = NAN
    return pd.DataFrame({"x": rng.standard_normal(n), "forfait": forfait,
                         "ligne": np.array(modalites_ligne)[rng.integers(0, len(modalites_ligne), n)], "y": rng.standard_normal(n)})


def cas_visibles():
    return [
        _cas("exemple de l'énoncé : modalité inconnue au test, modalités absentes du test", TRAIN, TEST, ["forfait"]),
        _cas("mêmes modalités des deux côtés", TRAIN, TRAIN.iloc[[3, 0, 1]], ["forfait"]),
    ]


def cas_caches():
    rng = np.random.default_rng(3201)
    grand = aleatoire(rng, 50000, ["base", "plus", "premium"], [1, 2, 3, 4])
    return [
        _cas("test vide (colonnes présentes)", TRAIN, TEST.iloc[:0], ["forfait"]),
        _cas("une seule ligne de chaque côté, modalités différentes", TRAIN.iloc[:1], TEST.iloc[1:2], ["forfait"]),
        _cas("index d'étiquettes non triviales", TRAIN.set_axis([40, 7, 13, 2], axis=0), TEST.set_axis(["u", "v", "w"], axis=0), ["forfait"]),
        _cas("valeurs manquantes dans la colonne catégorielle, des deux côtés",
             aleatoire(rng, 60, ["base", "plus", "premium"], [1, 2], p_nan=0.3), aleatoire(rng, 40, ["base", "plus", "pro"], [1, 2], p_nan=0.3), ["forfait"]),
        _cas("deux colonnes catégorielles, dont une à modalités entières",
             aleatoire(rng, 80, ["base", "plus"], [1, 2, 4]), aleatoire(rng, 50, ["plus", "premium", "pro"], [2, 3, 4]), ["forfait", "ligne"]),
        _cas("ordre des colonnes catégorielles inversé", aleatoire(rng, 30, ["b", "a", "c"], [3, 1]), aleatoire(rng, 30, ["c", "d"], [1, 2]), ["ligne", "forfait"]),
        _cas("modalité du test absente partout, colonne catégorielle du test entièrement manquante",
             aleatoire(rng, 20, ["base", "plus"], [1, 2]), aleatoire(rng, 10, ["base"], [1, 2], p_nan=1.0), ["forfait"]),
        _cas(f"grande taille ({len(grand)} lignes d'apprentissage)", grand, aleatoire(rng, 20000, ["plus", "pro", "premium"], [0, 2, 4, 5]), ["forfait", "ligne"]),
    ]


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "reference":
    tous = cas_visibles() + cas_caches()
    ok = sum(comparer(reference(*c["args"]), c["attendu"]) for c in tous)
    print(f"{ok}/{len(tous)} ; visibles {len(cas_visibles())}, cachés {len(cas_caches())}")
    sys.exit(0 if ok == len(tous) else 1)
