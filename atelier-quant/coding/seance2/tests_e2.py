"""Exercice 2 : mensualiser(prix, signaux). Usage autonome : python tests_e2.py reference"""
import sys

import numpy as np
import pandas as pd

from _communs import LIMITE_S, dates_ouvrees, melanger, panel, proches  # noqa: F401
from _correction import COLONNES_E2, mensualiser_pur
from _correction import mensualiser as reference

FONCTION = "mensualiser"
NAN = float("nan")

PRIX = [
    ("2026-01-30", "A", 11.0), ("2026-03-31", "B", NAN), ("2026-01-29", "A", 10.0),
    ("2026-01-30", "B", 20.0), ("2026-02-27", "A", 12.0), ("2026-03-30", "B", 21.0),
]
SIGNAUX = [
    ("2026-01-30", "A", 0.7), ("2026-03-30", "B", 2.0), ("2026-01-29", "A", 0.5), ("2026-01-31", "B", -1.0),
]


def cadre_prix(lignes, index=None):
    return pd.DataFrame(lignes, columns=["date", "actif", "prix"], index=index)


def cadre_signaux(lignes, index=None):
    return pd.DataFrame(lignes, columns=["date", "actif", "signal"], index=index)


def comparer(obtenu, attendu):
    """Colonnes exactes, lignes dans l'ordre demandé ; l'index n'est pas vérifié ; dates en chaînes ou en Timestamp."""
    if not isinstance(obtenu, pd.DataFrame) or list(obtenu.columns) != COLONNES_E2 or len(obtenu) != len(attendu):
        return False
    if len(attendu) == 0:
        return True
    try:
        o, a = obtenu.reset_index(drop=True), attendu.reset_index(drop=True)
        if list(o["actif"].astype(str)) != list(a["actif"].astype(str)):
            return False
        for c in ["fin_de_mois", "date"]:
            if not pd.to_datetime(o[c]).reset_index(drop=True).equals(pd.to_datetime(a[c]).reset_index(drop=True)):
                return False
        return all(proches(x, y) for c in ["prix", "signal"] for x, y in zip(o[c], a[c]))
    except (TypeError, ValueError):
        return False


def _cas(nom, prix, signaux):
    attendu = reference(prix, signaux)
    pur = mensualiser_pur(list(prix.itertuples(index=False, name=None)), list(signaux.itertuples(index=False, name=None)))
    assert comparer(pd.DataFrame(pur, columns=COLONNES_E2), attendu), f"correction incohérente : {nom}"
    return {"nom": nom, "args": (prix, signaux), "attendu": attendu}


def signaux_aleatoires(rng, n_jours, n_actifs, p=0.2):
    """Signaux publiés à des dates calendaires quelconques, week-ends compris."""
    jours = list(pd.date_range("2025-12-20", periods=n_jours, freq="D").strftime("%Y-%m-%d"))
    lignes = [(j, f"A{a:04d}", float(np.round(rng.standard_normal(), 6))) for a in range(n_actifs) for j in jours if rng.random() < p]
    return melanger(rng, lignes)


def cas_visibles():
    return [
        _cas("exemple de l'énoncé", cadre_prix(PRIX), cadre_signaux(SIGNAUX)),
        _cas("un actif, un mois, signal de la veille",
             cadre_prix([("2026-02-02", "A", 5.0), ("2026-02-03", "A", 6.0)]), cadre_signaux([("2026-02-02", "A", 1.5)])),
    ]


def cas_caches():
    rng = np.random.default_rng(2201)
    cas = [
        _cas("DataFrames vides", cadre_prix([]), cadre_signaux([])),
        _cas("aucun signal", cadre_prix(PRIX), cadre_signaux([])),
        _cas("une seule ligne, signal du jour même seulement", cadre_prix([("2026-01-05", "A", 3.0)]), cadre_signaux([("2026-01-05", "A", 9.0)])),
        _cas("index non triviaux, signal plus récent d'un autre actif",
             cadre_prix([("2026-04-30", "A", 1.0), ("2026-04-29", "B", 2.0)], index=[70, 3]),
             cadre_signaux([("2026-04-10", "A", 0.1), ("2026-04-28", "B", 0.2), ("2026-04-29", "B", 0.3)], index=["s", "t", "u"])),
        _cas("dernier jour du mois sans prix, mois entièrement manquant, mois vide entre deux mois",
             cadre_prix([("2026-01-28", "A", 1.0), ("2026-01-30", "A", NAN), ("2026-02-27", "A", NAN), ("2026-04-01", "A", 2.0)]),
             cadre_signaux([("2026-01-27", "A", -0.5), ("2026-01-29", "A", 0.5), ("2026-03-15", "A", 0.8)])),
        _cas("aléatoire, quelques actifs", cadre_prix(panel(rng, 80, 5, p_absent=0.3, p_nan=0.2, avec_signal=False)),
             cadre_signaux(signaux_aleatoires(rng, 140, 5))),
        _cas("aléatoire, signaux rares", cadre_prix(panel(rng, 120, 8, p_absent=0.1, p_nan=0.1, avec_signal=False)),
             cadre_signaux(signaux_aleatoires(rng, 200, 8, p=0.02))),
    ]
    g = np.random.default_rng(2202)
    prix = cadre_prix(panel(g, 500, 400, p_absent=0.05, p_nan=0.05, avec_signal=False))
    signaux = cadre_signaux(signaux_aleatoires(g, 760, 400, p=0.15))
    cas.append({"nom": f"grande taille ({len(prix)} prix, {len(signaux)} signaux)", "args": (prix, signaux), "attendu": reference(prix, signaux)})
    return cas


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "reference":
    tous = cas_visibles() + cas_caches()
    ok = sum(comparer(reference(*c["args"]), c["attendu"]) for c in tous)
    print(f"{ok}/{len(tous)} ; visibles {len(cas_visibles())}, cachés {len(cas_caches())}")
    sys.exit(0 if ok == len(tous) else 1)
