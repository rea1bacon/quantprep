"""Vérifie les affirmations de la leçon pandas 2 (outils/lecons_code_2.py, L-PANDAS-2) sur le comportement de pandas.

À exécuter dans les deux environnements :
    .venv-hackerrank/Scripts/python verif/lecon_pandas_2.py   (pandas 1.4.2, comme HackerRank)
    python verif/lecon_pandas_2.py                            (pandas local)
Sortie : verif/lecon_pandas_2.<version de pandas>.out.json
Les nombres cités dans la leçon viennent de verif/seance_code_2.py ; ce script vérifie les comportements.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

V = {"pandas": pd.__version__}
MAJEURE = int(pd.__version__.split(".")[0])


def avertissements(f):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        resultat = f()
    return resultat, [x.category.__name__ for x in w]


# Section 2 : rolling compte les valeurs non manquantes ; sans min_periods, m = n.
s = pd.Series([1.0, np.nan, 3.0, 4.0])
assert s.rolling(3, min_periods=2).mean().iloc[3] == 3.5 and s.rolling(3, min_periods=1).mean().iloc[1] == 1.0
assert s.rolling(3).mean().equals(s.rolling(3, min_periods=3).mean()) and s.rolling(3).mean().isna().all()
V["rolling_min_periods"] = "compte les valeurs non manquantes ; défaut = taille de la fenêtre"
r = pd.DataFrame({"date": ["2026-01-05", "2026-01-06", "2026-01-07"] * 2, "actif": ["A"] * 3 + ["B"] * 3,
                  "rendement": [1.0, np.nan, 3.0, 10.0, 20.0, 30.0]})
d = r.sort_values(["date", "actif"])
m = d.groupby("actif")["rendement"].rolling(2, min_periods=1).mean()
assert m.index.nlevels == 2 and list(m.index.get_level_values(0).unique()) == ["A", "B"]
try:
    d["glissante"] = m
    V["affectation_directe_groupby_rolling"] = "fonctionne"
except TypeError as e:
    V["affectation_directe_groupby_rolling"] = f"TypeError : {e}"
assert V["affectation_directe_groupby_rolling"].startswith("TypeError")
juste = m.reset_index(level=0, drop=True)
avec_transform = d.groupby("actif")["rendement"].transform(lambda x: x.rolling(2, min_periods=1).mean())
assert juste.reindex(d.index).equals(avec_transform.reindex(d.index))
assert juste.loc[3] == 10.0 and d["rendement"].rolling(2, min_periods=1).mean().loc[3] == 5.5  # ligne de B au 2026-01-05

# Section 3 : resample.
regle = "ME" if MAJEURE >= 2 else "M"
p = pd.Series([10.0, 11.0, np.nan, 12.0], index=pd.to_datetime(["2026-01-29", "2026-01-30", "2026-01-31", "2026-03-02"]))
mensuel, w_m = avertissements(lambda: p.resample("M").last())
assert list(mensuel.index.strftime("%Y-%m-%d")) == ["2026-01-31", "2026-02-28", "2026-03-31"]
assert mensuel.iloc[0] == 11.0 and np.isnan(mensuel.iloc[1]) and mensuel.iloc[2] == 12.0
V["resample_M_avertissements"] = w_m
try:
    p.resample("ME").last()
    V["resample_ME"] = "fonctionne"
except ValueError as e:
    V["resample_ME"] = f"ValueError : {e}"
assert V["resample_ME"].startswith("ValueError") == (pd.__version__ == "1.4.2")
assert (w_m == []) == (pd.__version__ == "1.4.2")
q = pd.DataFrame({"date": pd.to_datetime(["2026-01-29", "2026-01-30", "2026-03-02", "2026-01-30"]),
                  "actif": ["A", "A", "A", "B"], "prix": [10.0, 11.0, 12.0, 20.0]})
par_actif = q.set_index("date").groupby("actif")["prix"].resample(regle).last()
assert len(par_actif) == 4 and np.isnan(par_actif.loc[("A", pd.Timestamp("2026-02-28"))]) and len(par_actif.loc["B"]) == 1
t = pd.DataFrame({"prix": [10.0, 11.0, np.nan], "jour": pd.to_datetime(["2026-01-29", "2026-01-30", "2026-01-31"])},
                 index=pd.to_datetime(["2026-01-29", "2026-01-30", "2026-01-31"]))
sans, avec = t.resample(regle).last(), t.dropna(subset=["prix"]).resample(regle).last()
assert sans["prix"].iloc[0] == 11.0 and sans["jour"].iloc[0] == pd.Timestamp("2026-01-31") and avec["jour"].iloc[0] == pd.Timestamp("2026-01-30")
V["resample"] = "étiquette = dernier jour civil ; mois vides en NaN ; last saute les NaN colonne par colonne"

# Section 4 : merge_asof.
obs = pd.DataFrame({"date": pd.to_datetime(["2026-01-30", "2026-01-30"]), "actif": ["A", "B"]})
sig = pd.DataFrame({"date": pd.to_datetime(["2026-01-28", "2026-01-29", "2026-01-30"]), "actif": ["A", "B", "A"], "signal": [0.5, -1.0, 0.7]})
assert pd.merge_asof(obs, sig, on="date", by="actif")["signal"].tolist() == [0.7, -1.0]
assert pd.merge_asof(obs, sig, on="date", by="actif", allow_exact_matches=False)["signal"].tolist() == [0.5, -1.0]
assert pd.merge_asof(obs, sig, on="date", by="actif", direction="forward")["signal"].iloc[0] == 0.7
assert pd.merge(obs, sig, on=["date", "actif"])["signal"].tolist() == [0.7]
erreurs = {}
for nom, f in [("cles_chaines", lambda: pd.merge_asof(obs.astype({"date": str}), sig.astype({"date": str}), on="date", by="actif")),
               ("gauche_non_triee", lambda: pd.merge_asof(pd.DataFrame({"date": pd.to_datetime(["2026-01-30", "2026-01-29"]), "actif": ["A", "B"]}),
                                                          sig, on="date", by="actif"))]:
    try:
        f()
        erreurs[nom] = "aucune erreur"
    except Exception as e:  # noqa: BLE001
        erreurs[nom] = f"{type(e).__name__} : {e}"
assert erreurs["cles_chaines"].startswith("MergeError") and erreurs["gauche_non_triee"].startswith("ValueError")
V["merge_asof_erreurs"] = erreurs
V["merge_asof_gauche_vide_colonnes"] = list(pd.merge_asof(obs.assign(x=1).iloc[:0], sig, on="date", by="actif").columns)
assert (V["merge_asof_gauche_vide_colonnes"] != ["date", "actif", "x", "signal"]) == (pd.__version__ == "1.4.2")

# Section 5 : quantile linéaire par défaut = définition de la leçon, pour pandas et numpy, sur des tirages aléatoires.
rng = np.random.default_rng(6201)


def quantile_definition(valeurs, q):
    x = sorted(valeurs)
    h = (len(x) - 1) * q
    i = int(np.floor(h))
    f = h - i
    return x[i] if f == 0 or i + 1 >= len(x) else x[i] + f * (x[i + 1] - x[i])


for _ in range(2000):
    n = int(rng.integers(1, 30))
    valeurs = list(np.round(rng.standard_normal(n), int(rng.integers(0, 3))))  # arrondis : égalités fréquentes
    q = float(rng.choice([0.0, 1.0, rng.random(), 0.1, 0.9, 0.05, 0.95]))
    attendu = quantile_definition(valeurs, q)
    assert abs(pd.Series(valeurs).quantile(q) - attendu) <= 1e-12 and abs(np.quantile(valeurs, q) - attendu) <= 1e-12
V["quantile_defaut"] = "linéaire, h = (n - 1) q, 2000 tirages"
x = pd.DataFrame({"date": ["d1"] * 4 + ["d2"] * 3, "signal": [3.0, 1.0, 10.0, 2.0, 5.0, np.nan, -1.0]})
g = x.groupby("date")["signal"]
w = x["signal"].clip(lower=x["date"].map(g.quantile(0.1)), upper=x["date"].map(g.quantile(0.9)))
assert np.isclose(w.iloc[2], 7.9) and np.isclose(w.iloc[1], 1.3) and w.iloc[0] == 3.0 and np.isnan(w.iloc[5])
assert g.rank(pct=True).iloc[[4, 6]].tolist() == [1.0, 0.5]  # d2 : 2 valeurs non manquantes
d1 = w[x["date"] == "d1"]
assert np.isclose(d1.std(), np.std(d1, ddof=1)) and not np.isclose(d1.std(), np.std(d1))
V["std_trois_01"] = float(pd.Series([0.1, 0.1, 0.1]).std())
assert V["std_trois_01"] != 0.0

# Section 6 : pivot trie les colonnes ; shift(-1) donne la date suivante du calendrier ; rank(method="first") départage par colonne.
b = pd.DataFrame({"date": ["d1", "d1", "d2", "d3", "d3"], "actif": ["B", "A", "A", "A", "B"], "prix": [50.0, 100.0, 110.0, 99.0, 55.0]})
prix = b.pivot(index="date", columns="actif", values="prix")
assert list(prix.columns) == ["A", "B"] and list(prix.index) == ["d1", "d2", "d3"]
rdt = prix.shift(-1) / prix - 1
assert np.isnan(rdt.loc["d1", "B"]) and np.isclose(rdt.loc["d1", "A"], 0.1)
assert b.sort_values("date").groupby("actif")["prix"].shift(-1).loc[0] == 55.0  # format long : ligne suivante de B, en d3
rg = pd.DataFrame({"A": [0.2, 1.0], "B": [0.2, np.nan], "C": [0.1, 1.0]}).rank(axis=1, method="first")
assert rg.iloc[0].tolist() == [2.0, 3.0, 1.0] and rg.iloc[1, 0] == 1.0 and np.isnan(rg.iloc[1, 1]) and rg.iloc[1, 2] == 2.0
V["backtest"] = "pivot trié ; shift(-1) sur le calendrier ; égalités départagées par colonne"

print(json.dumps(V, ensure_ascii=False))
Path(__file__).with_name(f"lecon_pandas_2.{pd.__version__}.out.json").write_bytes(json.dumps(V, ensure_ascii=False, indent=1).encode("utf-8"))  # Python 3.9 : pas de newline=
print("Toutes les vérifications passent.")
