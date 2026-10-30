"""Tests de la question de code de l'examen blanc 1, utilisés par le notebook de la page.

Interface attendue par outils/noyau.py : FONCTION, LIMITE_S, cas_visibles(), cas_caches(), comparer().
Les valeurs attendues ne sont pas écrites à la main : elles sont calculées par la correction pandas,
et l'accord avec la correction sans bibliothèque est exigé (sauf pour le test de grande taille).

Usage autonome : python tests.py reference   (vérifie la correction sur tous les tests)
"""
import math
import sys

import numpy as np

from _communs import LIMITE_S, NAN, cadre, panel, panel_egalites, proches  # noqa: F401  (LIMITE_S lu par le noyau)
from _correction import rendement_long_short as reference
from _correction import rendement_long_short_pur as reference_pure

FONCTION = "rendement_long_short"

EXEMPLE = [
    ("2026-01-05", "A", 100.0), ("2026-01-05", "B", 50.0), ("2026-01-05", "C", 20.0),
    ("2026-01-06", "A", 110.0), ("2026-01-06", "B", 45.0), ("2026-01-06", "C", 21.0),
    ("2026-01-07", "A", 99.0), ("2026-01-07", "B", 54.0), ("2026-01-07", "C", 21.0),
    ("2026-01-08", "A", 99.0), ("2026-01-08", "B", 54.0), ("2026-01-08", "C", 23.1),
]

EGALITE = [
    ("2026-01-05", "A", 100.0), ("2026-01-05", "B", 50.0), ("2026-01-05", "C", 10.0), ("2026-01-05", "D", 20.0),
    ("2026-01-06", "A", 110.0), ("2026-01-06", "B", 55.0), ("2026-01-06", "C", 9.0), ("2026-01-06", "D", 20.0),
    ("2026-01-07", "A", 110.0), ("2026-01-07", "B", 66.0), ("2026-01-07", "C", 9.0), ("2026-01-07", "D", 21.0),
]


def comparer(obtenu, attendu):
    if isinstance(obtenu, bool) or not isinstance(obtenu, (float, np.floating)):
        return False
    return proches(float(obtenu), attendu)


def _cas(nom, lignes, k, index=None):
    df = cadre(lignes, index=index)
    attendu = reference(df, k)
    pur = reference_pure(lignes, k)
    assert proches(attendu, pur), f"correction incohérente : {nom}"
    return {"nom": nom, "args": (df, k), "attendu": attendu}


def cas_visibles():
    return [
        _cas("exemple de l'énoncé, k = 1", EXEMPLE, 1),
        _cas("égalité de rendement passé départagée par l'actif, k = 1", EGALITE, 1),
        _cas("trop peu de lignes exploitables : toutes les dates ignorées, k = 2", EXEMPLE, 2),
    ]


def _entrees_cachees():
    rng = np.random.default_rng(2026100701)
    frontiere = [
        ("2026-01-05", a, 100.0) for a in "ABCDE"] + [
        ("2026-01-06", a, p) for a, p in zip("ABCDE", (90.0, 90.0, 100.0, 110.0, 120.0))] + [
        ("2026-01-07", a, p) for a, p in zip("ABCDE", (81.0, 99.0, 100.0, 110.0, 120.0))]
    absent = [
        ("2026-01-05", "A", 100.0), ("2026-01-05", "B", 100.0), ("2026-01-05", "C", 100.0),
        ("2026-01-06", "A", 110.0), ("2026-01-06", "B", 95.0),
        ("2026-01-07", "A", 99.0), ("2026-01-07", "B", 95.0), ("2026-01-07", "C", 120.0),
        ("2026-01-08", "A", 99.0), ("2026-01-08", "B", 100.0), ("2026-01-08", "C", 120.0)]
    manquant = [
        ("2026-01-05", a, 100.0) for a in "ABCD"] + [
        ("2026-01-06", a, p) for a, p in zip("ABCD", (NAN, 105.0, 95.0, 100.0))] + [
        ("2026-01-07", a, p) for a, p in zip("ABCD", (120.0, 105.0, 95.0, 110.0))] + [
        ("2026-01-08", a, p) for a, p in zip("ABCD", (120.0, 100.0, 100.0, 110.0))]
    entiers = [(d, a, int(p)) for d, a, p in EXEMPLE[:-1]] + [("2026-01-08", "C", 23)]
    melange = [EXEMPLE[i] for i in (7, 2, 11, 0, 5, 9, 3, 10, 1, 6, 8, 4)]
    return [
        ("DataFrame vide", [], 1, None),
        ("une seule ligne", [("2026-01-05", "A", 100.0)], 1, None),
        ("k trop grand : 4 actifs pour k = 3", EGALITE, 3, None),
        ("exactement 2k lignes exploitables, k = 2", EGALITE, 2, None),
        ("égalité à la frontière de la jambe vendeuse", frontiere, 1, None),
        ("actif absent à une date : rendement depuis sa ligne précédente", absent, 1, None),
        ("prix manquant : trois lignes deviennent inexploitables", manquant, 1, None),
        ("lignes dans le désordre, index de chaînes", melange, 1, [f"l{i}" for i in range(12)]),
        ("prix entiers", entiers, 1, None),
        ("aléatoire, 10 dates, 6 actifs, k = 1", panel(rng, 10, 6), 1, None),
        ("aléatoire, 40 dates, 20 actifs, nombreux trous, k = 3", panel(rng, 40, 20, p_absent=0.2, p_nan=0.1), 3, None),
        ("aléatoire, nombreuses égalités de rendement passé, k = 2", panel_egalites(rng, 30, 15), 2, None),
        ("aléatoire sans valeur manquante, k = 2", panel(rng, 25, 12, p_absent=0.0, p_nan=0.0), 2, None),
    ]


def cas_caches():
    cas = [_cas(nom, lignes, k, index) for nom, lignes, k, index in _entrees_cachees()]
    grand = cadre(panel(np.random.default_rng(2026100702), 250, 300, p_absent=0.05, p_nan=0.05))
    cas.append({"nom": f"grande taille ({len(grand)} lignes, k = 15, limite {LIMITE_S:g} s)", "args": (grand, 15),
                "attendu": reference(grand, 15)})
    return cas


def ecrire_fichier_exploration(chemin="prix_panel.csv"):
    """Fichier d'exploration offert dans le répertoire du notebook ; il ne sert à aucun test."""
    texte = cadre(sorted(panel(np.random.default_rng(2026100703), 20, 8))).to_csv(index=False)
    with open(chemin, "w", encoding="utf-8", newline="\n") as f:  # fins de ligne Unix quel que soit le système
        f.write(texte.replace("\r\n", "\n"))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "reference":
    tous = [("visible", c) for c in cas_visibles()] + [("caché", c) for c in cas_caches()]
    reussis = 0
    for genre, c in tous:
        ok = comparer(reference(*c["args"]), c["attendu"])
        reussis += ok
        valeur = c["attendu"]
        print(f"[{'OK' if ok else 'ÉCHEC'}] {genre} : {c['nom']} (attendu {'NaN' if math.isnan(valeur) else round(valeur, 6)})")
    print(f"{reussis}/{len(tous)} ; visibles {len(cas_visibles())}, cachés {len(tous) - len(cas_visibles())}")
    sys.exit(0 if reussis == len(tous) else 1)
