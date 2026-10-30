"""Tests de la question de code du test de positionnement, utilisés par le notebook de la page.

Interface attendue par outils/noyau.py : FONCTION, LIMITE_S, cas_visibles(), cas_caches(), comparer().
Les valeurs attendues des tests cachés ne sont pas écrites à la main : elles sont calculées par
la correction pandas, et l'accord avec la correction sans bibliothèque est exigé.

Usage autonome : python tests.py reference   (vérifie la correction sur tous les tests)
"""
import math
import sys

import numpy as np
import pandas as pd

from _correction import ic_moyen as reference
from _correction import ic_moyen_sans_bibliotheque as reference_pure

FONCTION = "ic_moyen"
LIMITE_S = 10.0
COLONNES = ["date", "actif", "prix", "signal"]
NAN = float("nan")


def comparer(obtenu, attendu):
    if not isinstance(obtenu, (float, np.floating)):
        return False
    if math.isnan(attendu):
        return math.isnan(obtenu)
    return abs(obtenu - attendu) <= 1e-9


def cadre(lignes):
    return pd.DataFrame(lignes, columns=COLONNES)


EXEMPLE = [
    ("2026-01-05", "A", 100.0, 0.3), ("2026-01-05", "B", 50.0, -1.0), ("2026-01-05", "C", 20.0, 2.0),
    ("2026-01-06", "A", 101.0, 1.0), ("2026-01-06", "B", 49.0, 2.0), ("2026-01-06", "C", 22.0, 3.0),
]


def cas_visibles():
    return [
        {"nom": "exemple de l'énoncé", "args": (cadre(EXEMPLE),), "attendu": 1.0},
        {"nom": "signal inversé, lignes dans le désordre", "args": (cadre([
            ("2026-01-06", "C", 22.0, 3.0), ("2026-01-05", "B", 50.0, 5.0), ("2026-01-06", "A", 101.0, 1.0),
            ("2026-01-05", "C", 20.0, -4.0), ("2026-01-05", "A", 100.0, 0.0), ("2026-01-06", "B", 49.0, 2.0)]),),
         "attendu": -1.0},
        {"nom": "seulement deux actifs : la date est ignorée", "args": (cadre([
            ("2026-01-05", "A", 100.0, 0.3), ("2026-01-05", "B", 50.0, -1.0),
            ("2026-01-06", "A", 101.0, 1.0), ("2026-01-06", "B", 49.0, 2.0)]),),
         "attendu": NAN},
    ]


def aleatoire(rng, n_dates, n_actifs, p_absent=0.1, p_nan=0.1, signaux_entiers=False):
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="D").strftime("%Y-%m-%d")
    lignes = []
    for a in range(n_actifs):
        prix = 100 * np.exp(np.cumsum(0.02 * rng.standard_normal(n_dates)))
        for i, d in enumerate(dates):
            if rng.random() < p_absent:
                continue
            signal = float(rng.integers(-2, 3)) if signaux_entiers else float(rng.standard_normal())
            lignes.append((
                d, f"A{a:04d}",
                NAN if rng.random() < p_nan else float(prix[i]),
                NAN if rng.random() < p_nan else signal,
            ))
    ordre = rng.permutation(len(lignes))
    return [lignes[i] for i in ordre]


def _entrees_cachees():
    rng = np.random.default_rng(7)
    return [
        ("DataFrame vide", []),
        ("une seule ligne", [("2026-01-05", "A", 100.0, 1.0)]),
        ("deux actifs seulement", [
            ("2026-01-05", "A", 100.0, 0.3), ("2026-01-05", "B", 50.0, -1.0),
            ("2026-01-06", "A", 101.0, 1.0), ("2026-01-06", "B", 49.0, 2.0)]),
        ("signal constant sur la seule date exploitable", [
            ("2026-01-05", a, p, 1.0) for a, p in zip("ABC", (100.0, 50.0, 20.0))] + [
            ("2026-01-06", a, p, 0.0) for a, p in zip("ABC", (101.0, 49.0, 22.0))]),
        ("rendements tous égaux sur la seule date exploitable", [
            ("2026-01-05", a, 100.0, s) for a, s in zip("ABC", (1.0, 2.0, 3.0))] + [
            ("2026-01-06", a, 110.0, 0.0) for a in "ABC"]),
        ("égalités dans les signaux : rangs moyens", [
            ("2026-01-05", "A", 100.0, 1.0), ("2026-01-05", "B", 100.0, 1.0),
            ("2026-01-05", "C", 100.0, 2.0), ("2026-01-05", "D", 100.0, 3.0),
            ("2026-01-06", "A", 103.0, 0.0), ("2026-01-06", "B", 101.0, 0.0),
            ("2026-01-06", "C", 102.0, 0.0), ("2026-01-06", "D", 99.0, 0.0)]),
        ("actif absent à une date : rendement sur sa ligne suivante", [
            ("2026-01-05", "A", 100.0, 1.0), ("2026-01-05", "B", 100.0, 2.0), ("2026-01-05", "C", 100.0, 3.0),
            ("2026-01-06", "A", 90.0, 1.0), ("2026-01-06", "B", 105.0, 2.0),
            ("2026-01-07", "A", 95.0, 0.0), ("2026-01-07", "B", 100.0, 0.0), ("2026-01-07", "C", 130.0, 0.0)]),
        ("prix manquant : deux rendements invalidés", [
            ("2026-01-05", a, 100.0, s) for a, s in zip("ABCD", (1.0, 2.0, 3.0, 4.0))] + [
            ("2026-01-06", a, p, s) for a, p, s in zip("ABCD", (101.0, NAN, 99.0, 104.0), (4.0, 3.0, 2.0, 1.0))] + [
            ("2026-01-07", a, 100.0, 0.0) for a in "ABCD"]),
        ("signal manquant", [
            ("2026-01-05", a, 100.0, s) for a, s in zip("ABCD", (1.0, NAN, 3.0, 4.0))] + [
            ("2026-01-06", a, p, 0.0) for a, p in zip("ABCD", (101.0, 150.0, 99.0, 104.0))]),
        ("prix entiers", [
            ("2026-01-05", "A", 100, 3.0), ("2026-01-05", "B", 50, 1.0), ("2026-01-05", "C", 20, 2.0),
            ("2026-01-06", "A", 101, 0.0), ("2026-01-06", "B", 49, 0.0), ("2026-01-06", "C", 22, 0.0)]),
        ("aléatoire, 5 dates, 4 actifs", aleatoire(rng, 5, 4)),
        ("aléatoire, 30 dates, 12 actifs", aleatoire(rng, 30, 12)),
        ("aléatoire, signaux entiers avec nombreuses égalités", aleatoire(rng, 40, 15, signaux_entiers=True)),
        ("aléatoire, moitié des valeurs manquantes", aleatoire(rng, 40, 10, p_absent=0.3, p_nan=0.3)),
        ("aléatoire sans valeur manquante", aleatoire(rng, 25, 8, p_absent=0.0, p_nan=0.0)),
    ]


def cas_caches():
    cas = []
    for nom, lignes in _entrees_cachees():
        attendu = reference(cadre(lignes))
        pur = reference_pure(lignes)
        assert (math.isnan(attendu) and math.isnan(pur)) or abs(attendu - pur) < 1e-9, f"correction incohérente : {nom}"
        cas.append({"nom": nom, "args": (cadre(lignes),), "attendu": attendu})
    grand = cadre(aleatoire(np.random.default_rng(11), 500, 400, p_absent=0.05, p_nan=0.05))
    cas.append({"nom": f"grande taille ({len(grand)} lignes, limite {LIMITE_S:g} s)", "args": (grand,), "attendu": reference(grand)})
    return cas


def ecrire_fichier_exploration(chemin="prix_signaux.csv"):
    """Fichier d'exploration offert dans le répertoire du notebook ; il ne sert à aucun test."""
    cadre(sorted(aleatoire(np.random.default_rng(3), 20, 6))).to_csv(chemin, index=False)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "reference":
    # Valeurs de l'exemple de l'énoncé : rendements futurs au 2026-01-05.
    d = cadre(EXEMPLE).sort_values(["actif", "date"])
    rendements = (d.groupby("actif")["prix"].shift(-1) / d["prix"] - 1).dropna().round(12).tolist()
    assert rendements == [0.01, -0.02, 0.1], rendements
    tous = [("visible", c) for c in cas_visibles()] + [("caché", c) for c in cas_caches()]
    reussis = 0
    for genre, c in tous:
        ok = comparer(reference(*c["args"]), c["attendu"])
        reussis += ok
        print(f"[{'OK' if ok else 'ÉCHEC'}] {genre} : {c['nom']}")
    print(f"{reussis}/{len(tous)} ; visibles {len(cas_visibles())}, cachés {len(tous) - len(cas_visibles())}")
    sys.exit(0 if reussis == len(tous) else 1)
