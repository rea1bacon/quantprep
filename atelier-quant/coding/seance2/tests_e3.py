"""Exercice 3 : normaliser(df, q_bas, q_haut). Usage autonome : python tests_e3.py reference"""
import sys

import numpy as np
import pandas as pd

from _communs import LIMITE_S, cadres_egaux, panel  # noqa: F401
from _correction import normaliser as reference
from _correction import normaliser_pur

FONCTION = "normaliser"
COLONNES = ["date", "actif", "signal"]
NAN = float("nan")

EXEMPLE = [
    ("2026-01-05", "A", 3.0), ("2026-01-06", "A", 7.0), ("2026-01-05", "B", 100.0), ("2026-01-05", "C", 1.0),
    ("2026-01-07", "A", 5.0), ("2026-01-05", "D", 4.0), ("2026-01-05", "E", NAN), ("2026-01-05", "F", 2.0),
    ("2026-01-07", "B", 5.0),
]


def cadre(lignes, index=None):
    return pd.DataFrame(lignes, columns=COLONNES, index=index)


def comparer(obtenu, attendu):
    return cadres_egaux(obtenu, attendu)


def _cas(nom, df, q_bas, q_haut):
    attendu = reference(df, q_bas, q_haut)
    pur = normaliser_pur([(i, r.date, r.signal) for i, r in zip(df.index, df.itertuples(index=False))], q_bas, q_haut)
    assert cadres_egaux(pd.DataFrame.from_dict(pur, orient="index", columns=["w", "z", "rang"], dtype=float)
                        if pur else pd.DataFrame(columns=["w", "z", "rang"], dtype=float), attendu), f"correction incohérente : {nom}"
    return {"nom": nom, "args": (df, q_bas, q_haut), "attendu": attendu}


def sans_prix(lignes):
    return [(d, a, s) for d, a, _, s in lignes]


def cas_visibles():
    return [
        _cas("exemple de l'énoncé (quantiles 0,25 et 0,75)", cadre(EXEMPLE), 0.25, 0.75),
        _cas("exemple de l'énoncé (quantiles 0,1 et 0,9 : interpolation)", cadre(EXEMPLE), 0.1, 0.9),
    ]


def cas_caches():
    rng = np.random.default_rng(3201)
    cas = [
        _cas("DataFrame vide", cadre([]), 0.05, 0.95),
        _cas("une seule ligne", cadre([("2026-01-05", "A", 1.5)]), 0.05, 0.95),
        _cas("valeurs toutes égales à 0,1 (écart-type calculé non nul)", cadre([("2026-01-05", a, 0.1) for a in "ABC"]), 0.1, 0.9),
        _cas("égalités, valeur manquante, index de chaînes",
             cadre([("2026-01-05", "A", 2.0), ("2026-01-05", "B", 2.0), ("2026-01-05", "C", NAN), ("2026-01-05", "D", 9.0),
                    ("2026-01-05", "E", 2.0), ("2026-01-06", "A", -1.0), ("2026-01-06", "B", 3.0)], index=list("pqrstuv")), 0.2, 0.8),
        _cas("quantiles 0 et 1 : aucune winsorisation", cadre(sans_prix(panel(rng, 10, 8))), 0.0, 1.0),
        _cas("aléatoire, signaux entiers, nombreuses égalités", cadre(sans_prix(panel(rng, 30, 12, signaux_entiers=True))), 0.1, 0.9),
    ]
    lignes = sans_prix(panel(rng, 30, 15, p_nan=0.3))
    cas.append(_cas("aléatoire, valeurs manquantes, index décalé", cadre(lignes, index=3 * np.arange(1000, 1000 + len(lignes))), 0.05, 0.95))
    grand = cadre(sans_prix(panel(np.random.default_rng(3202), 500, 400, p_absent=0.05, p_nan=0.05)))
    cas.append({"nom": f"grande taille ({len(grand)} lignes)", "args": (grand, 0.01, 0.99), "attendu": reference(grand, 0.01, 0.99)})
    return cas


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "reference":
    tous = cas_visibles() + cas_caches()
    ok = sum(comparer(reference(*c["args"]), c["attendu"]) for c in tous)
    print(f"{ok}/{len(tous)} ; visibles {len(cas_visibles())}, cachés {len(cas_caches())}")
    sys.exit(0 if ok == len(tous) else 1)
