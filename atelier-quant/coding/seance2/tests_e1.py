"""Exercice 1 : moyenne_glissante(df, fenetre, min_periodes). Usage autonome : python tests_e1.py reference"""
import sys

import numpy as np
import pandas as pd

from _communs import LIMITE_S, panel, series_egales  # noqa: F401  (LIMITE_S lu par le noyau)
from _correction import moyenne_glissante as reference
from _correction import moyenne_glissante_pur

FONCTION = "moyenne_glissante"
COLONNES = ["date", "actif", "rendement"]
NAN = float("nan")

EXEMPLE = [
    ("2026-01-07", "A", 3.0), ("2026-01-05", "B", 10.0), ("2026-01-05", "A", 1.0),
    ("2026-01-08", "A", 5.0), ("2026-01-06", "A", NAN), ("2026-01-06", "B", 20.0),
]


def cadre(lignes, index=None):
    return pd.DataFrame(lignes, columns=COLONNES, index=index)


def comparer(obtenu, attendu):
    return series_egales(obtenu, attendu)


def _cas(nom, df, fenetre, min_periodes):
    attendu = reference(df, fenetre, min_periodes)
    pur = moyenne_glissante_pur([(i, *ligne) for i, ligne in zip(df.index, df.itertuples(index=False))], fenetre, min_periodes)
    assert series_egales(pd.Series(pur, dtype=float), attendu), f"correction incohérente : {nom}"
    return {"nom": nom, "args": (df, fenetre, min_periodes), "attendu": attendu}


def rendements(rng, n_dates, n_actifs, **options):
    """Panel (date, actif, rendement) mélangé : rendements de l'ordre de quelques pour cent, avec trous et NaN."""
    return [(d, a, p / 100 - 1) for d, a, p in panel(rng, n_dates, n_actifs, avec_signal=False, **options)]


def cas_visibles():
    return [
        _cas("exemple de l'énoncé (fenêtre 3, au moins 2 valeurs)", cadre(EXEMPLE), 3, 2),
        _cas("un seul actif, dans l'ordre (fenêtre 2, au moins 2 valeurs)",
             cadre([("2026-01-05", "A", 1.0), ("2026-01-06", "A", 2.0), ("2026-01-07", "A", 4.0), ("2026-01-08", "A", 8.0)]), 2, 2),
    ]


def cas_caches():
    rng = np.random.default_rng(1201)
    cas = [
        _cas("DataFrame vide", cadre([]), 3, 1),
        _cas("une seule ligne", cadre([("2026-01-05", "A", 0.5)]), 3, 1),
        _cas("index d'étiquettes mélangées", cadre(EXEMPLE, index=[40, 7, 13, 2, 99, 5]), 3, 2),
        _cas("index de chaînes, fenêtre 2, au moins 1 valeur", cadre(EXEMPLE, index=list("uvwxyz")), 2, 1),
        _cas("fenêtre 1 : chaque valeur elle-même", cadre(rendements(rng, 15, 4, p_nan=0.3)), 1, 1),
        _cas("aléatoire, nombreuses valeurs manquantes (fenêtre 5, au moins 3)", cadre(rendements(rng, 40, 10, p_absent=0.3, p_nan=0.3)), 5, 3),
        _cas("aléatoire, fenêtre plus longue que l'historique", cadre(rendements(rng, 20, 6)), 50, 1),
    ]
    grand = cadre(rendements(np.random.default_rng(1202), 500, 400, p_absent=0.05, p_nan=0.05))
    cas.append({"nom": f"grande taille ({len(grand)} lignes, fenêtre 20, au moins 10)", "args": (grand, 20, 10), "attendu": reference(grand, 20, 10)})
    return cas


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "reference":
    tous = cas_visibles() + cas_caches()
    ok = sum(comparer(reference(*c["args"]), c["attendu"]) for c in tous)
    print(f"{ok}/{len(tous)} ; visibles {len(cas_visibles())}, cachés {len(cas_caches())}")
    sys.exit(0 if ok == len(tous) else 1)
