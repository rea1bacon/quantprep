"""Exercice 2 : rang_transversal(df). Usage autonome : python tests_e2.py reference"""
import sys

import numpy as np
import pandas as pd

from _communs import LIMITE_S, panel, series_egales  # noqa: F401
from _correction import rang_transversal as reference
from _correction import rang_transversal_pur

FONCTION = "rang_transversal"
COLONNES = ["date", "actif", "signal"]
NAN = float("nan")

EXEMPLE = [
    ("2026-01-05", "A", 3.0), ("2026-01-05", "B", 1.0), ("2026-01-05", "C", 3.0),
    ("2026-01-06", "A", NAN), ("2026-01-06", "B", 2.0),
]


def cadre(lignes, index=None):
    return pd.DataFrame(lignes, columns=COLONNES, index=index)


def comparer(obtenu, attendu):
    return series_egales(obtenu, attendu)


def _cas(nom, df):
    attendu = reference(df)
    pur = rang_transversal_pur([(i, r.date, r.signal) for i, r in zip(df.index, df.itertuples(index=False))])
    assert series_egales(pd.Series(pur, dtype=float), attendu), f"correction incohérente : {nom}"
    return {"nom": nom, "args": (df,), "attendu": attendu}


def cas_visibles():
    return [
        _cas("exemple de l'énoncé", cadre(EXEMPLE)),
        _cas("une seule date sans égalité", cadre([("2026-01-05", "A", 0.5), ("2026-01-05", "B", -1.0), ("2026-01-05", "C", 2.0)])),
    ]


def cas_caches():
    rng = np.random.default_rng(201)
    sans_prix = lambda lignes: [(d, a, s) for d, a, _, s in lignes]
    cas = [
        _cas("DataFrame vide", cadre([])),
        _cas("une seule ligne", cadre([("2026-01-05", "A", 1.0)])),
        _cas("tous égaux sur une date", cadre([("2026-01-05", a, 7.0) for a in "ABCD"])),
        _cas("triple égalité", cadre([("2026-01-05", a, s) for a, s in zip("ABCDE", (2.0, 2.0, 2.0, 1.0, 5.0))])),
        _cas("index d'étiquettes mélangées", cadre(EXEMPLE, index=[40, 7, 13, 2, 99])),
        _cas("aléatoire, signaux entiers, nombreuses égalités", cadre(sans_prix(panel(rng, 30, 12, signaux_entiers=True)))),
        _cas("aléatoire, valeurs manquantes", cadre(sans_prix(panel(rng, 30, 10, p_nan=0.3)))),
    ]
    grand = cadre(sans_prix(panel(np.random.default_rng(202), 500, 400, p_absent=0.05, p_nan=0.05)))
    cas.append({"nom": f"grande taille ({len(grand)} lignes)", "args": (grand,), "attendu": reference(grand)})
    return cas


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "reference":
    tous = cas_visibles() + cas_caches()
    ok = sum(comparer(reference(*c["args"]), c["attendu"]) for c in tous)
    print(f"{ok}/{len(tous)} ; visibles {len(cas_visibles())}, cachés {len(cas_caches())}")
    sys.exit(0 if ok == len(tous) else 1)
