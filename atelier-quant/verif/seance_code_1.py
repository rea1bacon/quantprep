"""Valeurs des exemples de la séance de code 1, et preuve que les tests attrapent les erreurs visées.

Les erreurs visées sont celles fréquentes
(suivi/code/diagnostic-DIAG-C1-20261004-231724.py) :
  1. décalage shift(-1) calculé sur le cadre non trié au lieu du cadre trié par actif et par date ;
  2. rangs par double argsort, qui ne donnent pas le rang moyen en cas d'égalité ;
  3. prix suivant utilisé à la place du rendement.
Sortie : verif/seance_code_1.out.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DOSSIER = Path(__file__).resolve().parent.parent / "coding" / "seance1"
sys.path.insert(0, str(DOSSIER))
import tests_e1  # noqa: E402
import tests_e2  # noqa: E402
import tests_e3  # noqa: E402
from _correction import correlation_par_date, rang_transversal, rendement_futur  # noqa: E402

V = {}

# Exemple de l'exercice 1
r = rendement_futur(tests_e1.cadre(tests_e1.EXEMPLE))
assert np.allclose(r.iloc[:3], [-0.1, 0.1, 0.1]) and r.iloc[3:].isna().all()
assert np.isclose(99 / 110 - 1, -0.1) and np.isclose(110 / 100 - 1, 0.1) and np.isclose(55 / 50 - 1, 0.1)
V["E1_exemple"] = [None if np.isnan(x) else round(float(x), 12) for x in r]

# Exemple de l'exercice 2
r = rang_transversal(tests_e2.cadre(tests_e2.EXEMPLE))
assert r.iloc[[0, 1, 2, 4]].tolist() == [2.5, 1.0, 2.5, 1.0] and np.isnan(r.iloc[3])
V["E2_exemple"] = [None if np.isnan(x) else float(x) for x in r]

# Exemple de l'exercice 3 : corrélation du 2026-01-05, calculée aussi à la main.
r = correlation_par_date(tests_e3.cadre(tests_e3.EXEMPLE))
x, y = np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 7.0])
a_la_main = np.sum((x - x.mean()) * (y - y.mean())) / np.sqrt(np.sum((x - x.mean()) ** 2) * np.sum((y - y.mean()) ** 2))
assert np.isclose(r["2026-01-05"], a_la_main) and np.isnan(r["2026-01-06"]) and np.isnan(r["2026-01-07"])
assert list(r.index) == ["2026-01-05", "2026-01-06", "2026-01-07"]
V["E3_correlation_0105"] = float(a_la_main)
assert np.isclose(x.mean(), 2) and np.isclose(y.mean(), 13 / 3) and np.isclose(np.sum((y - y.mean()) ** 2), 38 / 3)
assert np.isclose(a_la_main, 5 / np.sqrt(2 * 38 / 3))
V["E3_cov_num"] = float(np.sum((x - x.mean()) * (y - y.mean())))
V["E3_sxx"] = float(np.sum((x - x.mean()) ** 2))
V["E3_syy"] = float(np.sum((y - y.mean()) ** 2))


# Les tests cachés attrapent chaque erreur visée.
def decalage_sur_cadre_non_trie(df):
    trie = df.sort_values(["date", "actif"])
    prix_suivant = df.groupby("actif")["prix"].shift(-1)  # erreur : calculé sur df, pas sur trie
    return (prix_suivant / trie["prix"] - 1).reindex(df.index)


def decalage_apres_tri_par_date(df):
    # Variante correcte : trier par date suffit, car groupby conserve l'ordre des lignes dans chaque groupe.
    trie = df.sort_values("date")
    return (trie.groupby("actif")["prix"].shift(-1) / trie["prix"] - 1).reindex(df.index)


def prix_suivant_au_lieu_du_rendement(df):
    trie = df.sort_values(["actif", "date"])
    return trie.groupby("actif")["prix"].shift(-1).reindex(df.index)


def rang_double_argsort(df):
    sortie = pd.Series(np.nan, index=df.index)
    for _, g in df.dropna(subset=["signal"]).groupby("date"):
        sortie[g.index] = np.argsort(np.argsort(g["signal"].to_numpy())) + 1
    return sortie


def score(module, fonction):
    cas = module.cas_caches()
    return sum(module.comparer(fonction(*c["args"]), c["attendu"]) for c in cas), len(cas)


for nom, module, fonction, doit_reussir in [
    ("E1_decalage_non_trie", tests_e1, decalage_sur_cadre_non_trie, False),
    ("E1_tri_par_date_seul", tests_e1, decalage_apres_tri_par_date, True),
    ("E1_prix_suivant", tests_e1, prix_suivant_au_lieu_du_rendement, False),
    ("E2_double_argsort", tests_e2, rang_double_argsort, False),
]:
    reussis, total = score(module, fonction)
    assert (reussis == total) == doit_reussir, (nom, reussis, total)
    V[nom] = f"{reussis}/{total}"
    print(f"{nom:28s} {reussis}/{total} tests cachés")

Path(__file__).with_name("seance_code_1.out.json").write_text(json.dumps(V, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(V, ensure_ascii=False))
print("Toutes les vérifications passent.")
