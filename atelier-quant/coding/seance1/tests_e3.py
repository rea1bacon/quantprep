"""Exercice 3 : correlation_par_date(df). Usage autonome : python tests_e3.py reference"""
import sys

import numpy as np
import pandas as pd

from _communs import LIMITE_S, proches, series_egales  # noqa: F401
from _correction import correlation_par_date as reference
from _correction import correlation_par_date_pur

FONCTION = "correlation_par_date"
COLONNES = ["date", "x", "y"]
NAN = float("nan")

EXEMPLE = [
    ("2026-01-06", 1.0, 5.0), ("2026-01-05", 1.0, 2.0), ("2026-01-05", 2.0, 4.0), ("2026-01-05", 3.0, 7.0),
    ("2026-01-06", 2.0, 3.0), ("2026-01-06", NAN, 1.0), ("2026-01-07", 4.0, 1.0), ("2026-01-07", 5.0, 1.0),
    ("2026-01-07", 6.0, 1.0),
]


def cadre(lignes):
    return pd.DataFrame(lignes, columns=COLONNES)


def comparer(obtenu, attendu):
    if not isinstance(obtenu, pd.Series):
        return False
    return series_egales(obtenu.rename(index=str), attendu.rename(index=str), meme_ordre=True)


def _cas(nom, df):
    attendu = reference(df)
    pur = correlation_par_date_pur(list(df.itertuples(index=False, name=None)))
    assert [d for d, _ in pur] == list(attendu.index) and all(proches(v, attendu[d]) for d, v in pur), f"correction incohérente : {nom}"
    return {"nom": nom, "args": (df,), "attendu": attendu}


def aleatoire(rng, n_dates, n_par_date, p_nan=0.1):
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="D").strftime("%Y-%m-%d")
    lignes = []
    for d in dates:
        rho = rng.uniform(-1, 1)
        for _ in range(int(rng.integers(1, n_par_date + 1))):
            x = float(rng.standard_normal())
            y = rho * x + float(rng.standard_normal())
            lignes.append((d, NAN if rng.random() < p_nan else x, NAN if rng.random() < p_nan else y))
    return [lignes[i] for i in rng.permutation(len(lignes))]


def cas_visibles():
    return [_cas("exemple de l'énoncé", cadre(EXEMPLE))]


def cas_caches():
    rng = np.random.default_rng(301)
    cas = [
        _cas("DataFrame vide", cadre([])),
        _cas("une seule date, deux lignes seulement", cadre([("2026-01-05", 1.0, 2.0), ("2026-01-05", 2.0, 1.0)])),
        _cas("x constant", cadre([("2026-01-05", 1.0, y) for y in (1.0, 2.0, 3.0)])),
        _cas("corrélation parfaite négative", cadre([("2026-01-05", x, -2 * x) for x in (1.0, 2.0, 3.0, 4.0)])),
        _cas("aléatoire, dates dans le désordre", cadre(aleatoire(rng, 20, 8))),
        _cas("aléatoire, nombreuses valeurs manquantes", cadre(aleatoire(rng, 30, 6, p_nan=0.4))),
    ]
    grand = cadre(aleatoire(np.random.default_rng(302), 2000, 150))
    cas.append({"nom": f"grande taille ({len(grand)} lignes)", "args": (grand,), "attendu": reference(grand)})
    return cas


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "reference":
    tous = cas_visibles() + cas_caches()
    ok = sum(comparer(reference(*c["args"]), c["attendu"]) for c in tous)
    print(f"{ok}/{len(tous)} ; visibles {len(cas_visibles())}, cachés {len(cas_caches())}")
    sys.exit(0 if ok == len(tous) else 1)
