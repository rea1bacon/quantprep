"""Exercice 1 : rendement_futur(df). Usage autonome : python tests_e1.py reference"""
import sys

import numpy as np
import pandas as pd

from _communs import LIMITE_S, panel, series_egales  # noqa: F401  (LIMITE_S lu par le noyau)
from _correction import rendement_futur as reference
from _correction import rendement_futur_pur

FONCTION = "rendement_futur"
COLONNES = ["date", "actif", "prix"]
NAN = float("nan")

EXEMPLE = [
    ("2026-01-06", "A", 110.0), ("2026-01-05", "A", 100.0), ("2026-01-05", "B", 50.0),
    ("2026-01-07", "A", 99.0), ("2026-01-07", "B", 55.0),
]


def cadre(lignes, index=None):
    return pd.DataFrame(lignes, columns=COLONNES, index=index)


def comparer(obtenu, attendu):
    return series_egales(obtenu, attendu)


def _cas(nom, df):
    attendu = reference(df)
    pur = rendement_futur_pur([(i, *ligne) for i, ligne in zip(df.index, df.itertuples(index=False))])
    assert series_egales(pd.Series(pur, dtype=float), attendu), f"correction incohérente : {nom}"
    return {"nom": nom, "args": (df,), "attendu": attendu}


def cas_visibles():
    return [
        _cas("exemple de l'énoncé", cadre(EXEMPLE)),
        _cas("un seul actif, dans l'ordre", cadre([("2026-01-05", "A", 10.0), ("2026-01-06", "A", 12.0), ("2026-01-07", "A", 9.0)])),
    ]


def cas_caches():
    rng = np.random.default_rng(101)
    sans_signal = lambda lignes: [l[:3] for l in lignes]
    cas = [
        _cas("DataFrame vide", cadre([])),
        _cas("une seule ligne", cadre([("2026-01-05", "A", 10.0)])),
        _cas("index d'étiquettes mélangées", cadre(EXEMPLE, index=[40, 7, 13, 2, 99])),
        _cas("index de chaînes", cadre(EXEMPLE, index=list("vwxyz"))),
        _cas("prix manquants", cadre([("2026-01-05", "A", 10.0), ("2026-01-06", "A", NAN), ("2026-01-07", "A", 9.0),
                                      ("2026-01-05", "B", 5.0), ("2026-01-06", "B", 6.0)])),
        _cas("aléatoire, lignes dans le désordre", cadre(sans_signal(panel(rng, 30, 8)))),
        _cas("aléatoire, nombreux trous", cadre(sans_signal(panel(rng, 40, 10, p_absent=0.4, p_nan=0.2)))),
    ]
    grand = cadre(sans_signal(panel(np.random.default_rng(102), 500, 400, p_absent=0.05, p_nan=0.05)))
    cas.append({"nom": f"grande taille ({len(grand)} lignes)", "args": (grand,), "attendu": reference(grand)})
    return cas


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "reference":
    tous = cas_visibles() + cas_caches()
    ok = sum(comparer(reference(*c["args"]), c["attendu"]) for c in tous)
    print(f"{ok}/{len(tous)} ; visibles {len(cas_visibles())}, cachés {len(cas_caches())}")
    sys.exit(0 if ok == len(tous) else 1)
