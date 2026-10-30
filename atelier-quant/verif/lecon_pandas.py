"""Vérifie les affirmations des points de contrôle de la leçon pandas (outils/lecons.py, L-PANDAS-1).

À exécuter dans les deux environnements :
    .venv-hackerrank/Scripts/python verif/lecon_pandas.py   (pandas 1.4.2, comme HackerRank)
    python verif/lecon_pandas.py                            (pandas local)
Sortie : verif/lecon_pandas.<version de pandas>.out.json
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

V = {"pandas": pd.__version__}
df = pd.DataFrame(
    [("2026-01-06", "A", 110.0, 0.3), ("2026-01-05", "A", 100.0, 0.2), ("2026-01-05", "B", 50.0, -1.0),
     ("2026-01-07", "A", 99.0, 0.1), ("2026-01-07", "B", 55.0, 0.3), ("2026-01-06", "B", 52.0, 0.3)],
    columns=["date", "actif", "prix", "signal"])

# Contrôle « alignement » : bonne réponse et explications des options.
d = df.sort_values("prix")
s = d["prix"] - df["prix"]
assert (s == 0).all() and sorted(s.index) == list(range(6))
filtre = df["prix"][df["prix"] > 60]
assert (df["prix"] - filtre).isna().sum() == 6 - len(filtre)  # NaN pour les étiquettes absentes d'une des séries
apres_reset = d["prix"].reset_index(drop=True) - df["prix"]
assert apres_reset.notna().all() and (apres_reset != 0).any()  # pas de NaN, mais des différences fausses
V["alignement"] = "vérifié"

# Contrôle « formes ».
assert df.groupby("date")["signal"].transform("max").index.equals(df.index)
assert len(df.groupby("date")["signal"].max()) == df["date"].nunique()
assert len(df.groupby("date")["signal"].agg(["max", "min"])) == df["date"].nunique()
assert len(df.groupby("date")[["signal", "prix"]].apply(lambda g: g["signal"].max())) == df["date"].nunique()
V["formes"] = "vérifié"

# Contrôle « décalage » : le code du diagnostic donne des valeurs fausses, la version triée des valeurs justes.
trie = df.sort_values(["date", "actif"])
trie["nxt"] = df.groupby("actif")["prix"].shift(-1)
juste = df.sort_values("date").groupby("actif")["prix"].shift(-1)
assert trie["nxt"].loc[0] == 100.0 and juste.loc[0] == 99.0
assert trie.groupby("actif")["prix"].shift(-1).loc[0] == 99.0
V["decalage"] = {"faux_ligne_0": 100.0, "juste_ligne_0": 99.0}

# Section 5 et contrôle « rang ».
r = df.groupby("date")["signal"].rank()
assert r.loc[0] == 1.5 and r.loc[5] == 1.5  # égalité A et B le 2026-01-06
x = pd.Series([2.0, 5.0, 2.0, 2.0])
attendus = {"average": [2, 4, 2, 2], "first": [1, 4, 2, 3], "dense": [1, 2, 1, 1], "min": [1, 4, 1, 1]}
for methode, valeurs in attendus.items():
    assert x.rank(method=methode).tolist() == valeurs, methode
assert pd.Series([3.0, 1.0, 3.0]).rank().tolist() == [2.5, 1.0, 2.5]
assert (np.argsort(np.argsort(np.array([3.0, 1.0, 3.0]))) + 1).tolist() != [2.5, 1.0, 2.5]
V["rangs"] = attendus

# Section 6 : comportement de Series.corr.
a, b = pd.Series([1.0, 2.0, 3.0, np.nan]), pd.Series([3.0, 1.0, 2.0, 5.0])
assert np.isclose(a.corr(b), np.corrcoef([1, 2, 3], [3, 1, 2])[0, 1])  # paires incomplètes ignorées
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    assert np.isnan(pd.Series([1.0, 1.0, 1.0]).corr(pd.Series([1.0, 2.0, 3.0])))  # série constante
t, u = pd.Series([1.0, 1.0, 2.0, 3.0]), pd.Series([0.03, 0.01, 0.02, -0.01])
assert np.isclose(t.corr(u, method="spearman"), np.corrcoef(t.rank(), u.rank())[0, 1])  # rangs moyens
V["corr"] = "vérifié"

# Contrôle « apply » : sans sélection, la fonction reçoit aussi la colonne de regroupement.
colonnes_vues = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    df.groupby("date").apply(lambda g: colonnes_vues.append(list(g.columns)))
V["apply_colonnes_recues"] = colonnes_vues[0]
assert "date" in colonnes_vues[0] or pd.__version__ >= "3"
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        df.groupby("date").apply(lambda g: g.mean().mean())
        V["apply_moyenne_toutes_colonnes"] = "fonctionne" + (" avec avertissement" if w else "")
    except TypeError:
        V["apply_moyenne_toutes_colonnes"] = "échoue (TypeError)"
# Section 7 : le pipeline de la leçon donne exactement le résultat de la correction du diagnostic.
import sys  # noqa: E402

DIAG = Path(__file__).resolve().parent.parent / "coding" / "diagnostic"
sys.path.insert(0, str(DIAG))
from _correction import ic_moyen  # noqa: E402

p = pd.read_csv(DIAG / "prix_signaux.csv")
d = p.sort_values("date")
p["r"] = d.groupby("actif")["prix"].shift(-1) / d["prix"] - 1


def ic(g):
    g = g.dropna()
    if len(g) < 3 or g["signal"].nunique() == 1 or g["r"].nunique() == 1:
        return np.nan
    return g["signal"].corr(g["r"], method="spearman")


moyenne_lecon = p.groupby("date")[["signal", "r"]].apply(ic).mean()
assert abs(moyenne_lecon - ic_moyen(pd.read_csv(DIAG / "prix_signaux.csv"))) < 1e-12
V["ic_moyen_prix_signaux"] = round(float(moyenne_lecon), 12)
print(json.dumps(V, ensure_ascii=False))
Path(__file__).with_name(f"lecon_pandas.{pd.__version__}.out.json").write_text(json.dumps(V, ensure_ascii=False, indent=1), encoding="utf-8")
print("Toutes les vérifications passent.")
