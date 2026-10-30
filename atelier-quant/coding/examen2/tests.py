"""Tests de la question de code de l'examen blanc 2 (EB2-C1), utilisés par le notebook de la page.

Interface attendue par outils/noyau.py : FONCTION, LIMITE_S, cas_visibles(), cas_caches(), comparer().
Les valeurs attendues ne sont pas écrites à la main : elles sont calculées par la correction pandas, et
l'accord avec la correction sans bibliothèque est exigé sur chaque cas, grande taille comprise.

Usage autonome : python tests.py reference   (vérifie la correction sur tous les tests)
"""
import sys

import numpy as np

from _communs import LIMITE_S, NAN, aleatoire, cadre_cotations, cadre_transactions, comparer, lignes_de, serie_de  # noqa: F401
from _correction import desequilibre as reference
from _correction import desequilibre_sans_bibliotheque as reference_pure

FONCTION = "desequilibre"

# Exemple de l'énoncé.
EXEMPLE_T = [
    (3000, "A", 100.5, 10), (3000, "B", 50.0, 5), (4000, "C", 20.0, 1), (5000, "A", 99.75, 4),
    (6500, "A", 100.25, 7), (8000, "B", 50.25, 6), (9200, "B", 51.0, 2), (9500, "B", 51.5, 3),
]
EXEMPLE_C = [
    (1000, "A", 99.5, 100.5), (2500, "A", 101.0, 100.0), (3000, "B", 50.0, 50.5), (9000, "B", 51.0, 51.5),
]


def _cas(nom, trans, cot, index_t=None, index_c=None):
    t = cadre_transactions(trans, index=index_t)
    c = cadre_cotations(cot, index=index_c)
    attendu = reference(t, c)
    pur = serie_de(reference_pure(lignes_de(t), lignes_de(c)))
    assert comparer(pur, attendu), f"correction incohérente : {nom}"
    return {"nom": nom, "args": (t, c), "attendu": attendu}


def cas_visibles():
    return [
        _cas("exemple de l'énoncé", EXEMPLE_T, EXEMPLE_C),
        _cas("deux actifs, lignes dans le désordre, index quelconque",
             [(12000, "Y", 10.25, 300), (2000, "Z", 7.5, 100), (11000, "Y", 9.75, 200),
              (4000, "Z", 7.25, 50), (15000, "Y", 10.0, 100), (3000, "Z", 7.0, 80)],
             [(10000, "Y", 9.75, 10.25), (1000, "Z", 7.0, 7.5), (14000, "Y", 9.5, 10.0)],
             index_t=[17, 4, 9, 30, 2, 11], index_c=[5, 0, 8]),
        _cas("aucune cotation valide",
             [(2000, "A", 10.0, 5), (2000, "B", 20.0, 5)],
             [(1000, "A", 10.5, 10.0), (1000, "B", NAN, 20.5), (1500, "A", 10.0, 10.0)]),
    ]


def cas_caches():
    rng = np.random.default_rng(29)
    cas = [
        _cas("transactions vides", [], EXEMPLE_C),
        _cas("cotations vides", EXEMPLE_T, []),
        _cas("cotation au même instant seulement", [(5000, "A", 10.5, 10)], [(5000, "A", 10.0, 10.5)]),
        _cas("cotation au même instant et cotation antérieure",
             [(5000, "A", 10.25, 10), (6000, "A", 10.0, 5)],
             [(5000, "A", 10.25, 10.75), (4000, "A", 9.75, 10.25)]),
        _cas("écart de exactement 5000 ms : cotation retenue", [(6000, "A", 10.5, 7)], [(1000, "A", 10.0, 10.5)]),
        _cas("écart de 5001 ms : cotation trop ancienne",
             [(6001, "A", 10.5, 7), (3000, "A", 10.0, 3)], [(1000, "A", 10.0, 10.5)]),
        _cas("dernière cotation bloquée (achat = vente), la précédente est valide",
             [(4000, "A", 10.25, 4)], [(3500, "A", 10.5, 10.5), (1000, "A", 10.0, 10.25)]),
        _cas("cotations incomplètes ignorées",
             [(4000, "A", 10.0, 6), (4500, "B", 20.0, 2)],
             [(3900, "A", NAN, 10.5), (2000, "A", 10.25, 10.5), (4400, "B", 19.5, NAN), (4000, "B", 19.75, 20.25)]),
        _cas("cotation plus récente d'un autre actif",
             [(5000, "A", 10.25, 3), (5000, "B", 10.25, 2)],
             [(4900, "B", 10.5, 11.0), (1000, "A", 9.75, 10.25), (2000, "B", 10.0, 10.25)]),
        _cas("prix égal au milieu",
             [(2000, "A", 10.25, 5), (3000, "A", 10.25, 5), (2500, "B", 5.0, 1), (2600, "B", 5.25, 9)],
             [(1000, "A", 10.0, 10.5), (1000, "B", 4.75, 5.25)]),
        _cas("actif présent seulement dans les cotations",
             [(2000, "B", 5.5, 4)], [(1000, "A", 10.0, 10.5), (1000, "B", 5.0, 5.5)]),
        _cas("index de chaînes, lignes dans le désordre",
             [EXEMPLE_T[i] for i in (5, 2, 7, 0, 3, 6, 1, 4)], [EXEMPLE_C[i] for i in (3, 1, 0, 2)],
             index_t=list("pqrstuvw"), index_c=list("dcba")),
        _cas("aléatoire, 2000 transactions, 8 actifs", *aleatoire(rng, 2000, 2000, 8)),
        _cas("aléatoire, nombreuses cotations invalides ou incomplètes",
             *aleatoire(rng, 3000, 600, 6, p_invalide=0.4, p_nan=0.2)),
    ]
    trans, cot = aleatoire(np.random.default_rng(31), 50_000, 50_000, 40, duree_ms=3_000_000)
    cas.append(_cas(f"grande taille ({len(trans)} transactions, {len(cot)} cotations, limite {LIMITE_S:g} s)", trans, cot))
    return cas


def ecrire_fichiers_exploration(dossier="."):
    """Fichiers d'exploration offerts dans le répertoire du notebook ; ils ne servent à aucun test."""
    trans, cot = aleatoire(np.random.default_rng(3), 400, 400, 4, duree_ms=120_000)
    for nom, cadre in [("transactions", cadre_transactions(sorted(trans))), ("cotations", cadre_cotations(sorted(cot)))]:
        texte = cadre.to_csv(index=False).replace("\r\n", "\n")  # fins de ligne Unix sur toute plateforme
        with open(f"{dossier}/{nom}.csv", "w", encoding="utf-8", newline="\n") as f:
            f.write(texte)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "reference":
    tous = [("visible", c) for c in cas_visibles()] + [("caché", c) for c in cas_caches()]
    reussis = 0
    for genre, c in tous:
        ok = comparer(reference(*c["args"]), c["attendu"])
        reussis += ok
        print(f"[{'OK' if ok else 'ÉCHEC'}] {genre} : {c['nom']}")
    print(f"{reussis}/{len(tous)} ; visibles {len(cas_visibles())}, cachés {len(tous) - len(cas_visibles())}")
    sys.exit(0 if reussis == len(tous) else 1)
