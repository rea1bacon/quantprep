"""Exercice 4 : backtest(df, nb_groupes). Usage autonome : python tests_e4.py reference"""
import sys

import numpy as np
import pandas as pd

from _communs import LIMITE_S, panel, proches  # noqa: F401
from _correction import backtest as reference
from _correction import backtest_pur

FONCTION = "backtest"
COLONNES = ["date", "actif", "prix", "signal"]
NAN = float("nan")

EXEMPLE = [
    ("2026-01-06", "D", NAN, 1.0), ("2026-01-05", "A", 100.0, 0.9), ("2026-01-07", "C", 19.0, -0.1),
    ("2026-01-05", "B", 50.0, 0.1), ("2026-01-06", "A", 110.0, 0.2), ("2026-01-05", "C", 20.0, -0.5),
    ("2026-01-07", "A", 99.0, NAN), ("2026-01-06", "B", 50.0, 0.2), ("2026-01-05", "D", 10.0, 0.3),
    ("2026-01-07", "B", 55.0, 0.0), ("2026-01-06", "C", 19.0, 0.4), ("2026-01-07", "D", 12.0, 0.5),
]


def cadre(lignes, index=None):
    return pd.DataFrame(lignes, columns=COLONNES, index=index)


def comparer(obtenu, attendu):
    """Un couple (Series, nombre) : index de dates (chaînes ou Timestamp) dans l'ordre chronologique, valeurs à 1e-9 près."""
    if not isinstance(obtenu, (tuple, list)) or len(obtenu) != 2:
        return False
    serie, sharpe = obtenu
    s_att, sh_att = attendu
    if not isinstance(serie, pd.Series) or len(serie) != len(s_att):
        return False
    try:
        if len(s_att) and list(pd.to_datetime(pd.Index(serie.index))) != list(pd.to_datetime(pd.Index(s_att.index))):
            return False
        return all(proches(x, y) for x, y in zip(serie.to_numpy(dtype=float), s_att.to_numpy(dtype=float))) and proches(sharpe, sh_att)
    except (TypeError, ValueError):
        return False


def _cas(nom, df, nb_groupes):
    attendu = reference(df, nb_groupes)
    serie, sharpe = backtest_pur(list(df.itertuples(index=False, name=None)), nb_groupes)
    pur = (pd.Series([v for _, v in serie], index=[d for d, _ in serie], dtype=float), sharpe)
    assert comparer(pur, attendu), f"correction incohérente : {nom}"
    return {"nom": nom, "args": (df, nb_groupes), "attendu": attendu}


SANS_MANQUANT = [
    ("2026-02-02", "A", 10.0, 0.3), ("2026-02-02", "B", 20.0, -0.2), ("2026-02-02", "C", 30.0, 0.8), ("2026-02-02", "D", 40.0, 0.1),
    ("2026-02-03", "A", 11.0, -0.4), ("2026-02-03", "B", 19.0, 0.5), ("2026-02-03", "C", 33.0, 0.2), ("2026-02-03", "D", 38.0, 0.0),
    ("2026-02-04", "A", 12.0, 0.1), ("2026-02-04", "B", 21.0, 0.2), ("2026-02-04", "C", 30.0, 0.3), ("2026-02-04", "D", 40.0, 0.4),
    ("2026-02-05", "A", 12.0, 0.0), ("2026-02-05", "B", 22.0, 0.0), ("2026-02-05", "C", 31.0, 0.0), ("2026-02-05", "D", 41.0, 0.0),
]


def cas_visibles():
    return [
        _cas("exemple de l'énoncé (2 groupes)", cadre(EXEMPLE), 2),
        _cas("quatre dates, quatre actifs, aucune valeur manquante (2 groupes)", cadre(SANS_MANQUANT), 2),
    ]


def cas_caches():
    rng = np.random.default_rng(4201)
    egalites = [(d, a, p, 1.0) for d, (pz, py, px, pw) in [("2026-03-02", (10.0, 10.0, 10.0, 10.0)), ("2026-03-03", (12.0, 11.0, 9.0, 8.0)),
                                                           ("2026-03-04", (12.0, 12.0, 9.0, 9.0))]
                for a, p in [("Z", pz), ("Y", py), ("X", px), ("W", pw)]]
    trous = [
        ("2026-01-05", "A", 10.0, 1.0), ("2026-01-05", "B", 10.0, 2.0), ("2026-01-05", "C", 10.0, 3.0), ("2026-01-05", "D", 10.0, 4.0),
        ("2026-01-06", "A", 12.0, 0.5), ("2026-01-06", "B", 11.0, 0.6), ("2026-01-06", "C", 9.0, 0.7),
        ("2026-01-07", "A", 12.0, 0.1), ("2026-01-07", "B", 12.0, 0.2), ("2026-01-07", "C", 10.0, 0.3), ("2026-01-07", "D", 15.0, 0.4),
    ]
    peu = [
        ("2026-05-04", "A", 10.0, 0.1), ("2026-05-04", "B", 10.0, 0.2), ("2026-05-05", "A", 11.0, 0.3), ("2026-05-05", "B", 10.0, NAN),
        ("2026-05-06", "A", 12.0, 0.2), ("2026-05-06", "B", 9.0, 0.1), ("2026-05-07", "A", 12.0, 0.4), ("2026-05-07", "B", 10.0, 0.5),
        ("2026-05-08", "A", 13.0, 0.0), ("2026-05-08", "B", 11.0, 0.1),
    ]
    cas = [
        _cas("DataFrame vide", cadre([]), 5),
        _cas("une seule date", cadre(SANS_MANQUANT[:4]), 2),
        _cas("égalités de signal à la frontière, lignes dans l'ordre inverse des actifs", cadre(egalites), 2),
        _cas("prix manquant en t+1, actif absent un jour puis de retour", cadre(trous, index=list(range(110, 99, -1))), 2),
        _cas("dates sans position (trop peu d'actifs éligibles)", cadre(peu), 2),
        _cas("plus de groupes que d'actifs : aucune position, Sharpe non défini", cadre(SANS_MANQUANT), 5),
    ]
    lignes = panel(rng, 40, 12, p_absent=0.15, p_nan=0.15)
    cas.append(_cas("aléatoire, valeurs manquantes, index décalé (3 groupes)", cadre(lignes, index=7 * np.arange(len(lignes)) + 5), 3))
    grand = cadre(panel(np.random.default_rng(4202), 500, 400, p_absent=0.05, p_nan=0.05))
    cas.append({"nom": f"grande taille ({len(grand)} lignes, 5 groupes)", "args": (grand, 5), "attendu": reference(grand, 5)})
    return cas


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "reference":
    tous = cas_visibles() + cas_caches()
    ok = sum(comparer(reference(*c["args"]), c["attendu"]) for c in tous)
    print(f"{ok}/{len(tous)} ; visibles {len(cas_visibles())}, cachés {len(cas_caches())}")
    sys.exit(0 if ok == len(tous) else 1)
