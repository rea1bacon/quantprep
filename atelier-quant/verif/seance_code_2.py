"""Valeurs des exemples de la séance de code 2, et preuve que les tests cachés attrapent les erreurs visées.

Pour chaque exercice :
  1. l'exemple de l'énoncé est recalculé par la correction pandas, par la version sans bibliothèque et,
     quand c'est possible, à la main ; les trois doivent concorder ;
  2. les deux versions de la correction concordent aussi sur le grand cas caché ;
  3. des solutions fausses typiques (fuite entre actifs, information du jour même, mauvais départage,
     écart-type à dénominateur n, etc.) échouent à au moins un test caché, et des variantes justes
     écrites autrement les passent tous.
Écrit aussi le fichier d'exploration coding/seance2/panel.csv (panel synthétique, graine fixe).
Sortie : verif/seance_code_2.out.json (et la sortie texte dans verif/seance_code_2.out.txt).
Usage : python verif/seance_code_2.py
"""
import datetime as dt
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DOSSIER = Path(__file__).resolve().parent.parent / "coding" / "seance2"
sys.path.insert(0, str(DOSSIER))
import tests_e1  # noqa: E402
import tests_e2  # noqa: E402
import tests_e3  # noqa: E402
import tests_e4  # noqa: E402
from _correction import (backtest, backtest_pur, mensualiser, moyenne_glissante, moyenne_glissante_pur,  # noqa: E402
                         normaliser, normaliser_pur, quantile_lineaire)

warnings.simplefilter("ignore", FutureWarning)  # les solutions fausses simulées utilisent "M" ; la correction n'en émet pas
V = {}
LIGNES_TEXTE = []


def dire(texte):
    LIGNES_TEXTE.append(texte)
    print(texte)


def arrondi(x, n=12):
    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else round(float(x), n)


# ============================================================ exercice 1 : exemple
r = moyenne_glissante(tests_e1.cadre(tests_e1.EXEMPLE), 3, 2)
a_la_main = [(1.0 + 3.0) / 2, math.nan, math.nan, (3.0 + 5.0) / 2, math.nan, (10.0 + 20.0) / 2]
assert all((math.isnan(x) and math.isnan(y)) or x == y for x, y in zip(r, a_la_main)), list(r)
V["E1_ex_0"], V["E1_ex_3"], V["E1_ex_5"] = float(r.loc[0]), float(r.loc[3]), float(r.loc[5])
V["E1_exemple"] = [arrondi(x) for x in r]
# Fuite sans groupby, sur l'exemple trié par date : la ligne 0 (A, 2026-01-07) recevrait la moyenne de
# A le 2026-01-06 (NaN), B le 2026-01-06 (20) et A le 2026-01-07 (3).
d = tests_e1.cadre(tests_e1.EXEMPLE).sort_values(["date", "actif"])
fuite = d["rendement"].rolling(3, min_periods=2).mean().reindex(range(6))
assert fuite.loc[0] == (20.0 + 3.0) / 2 and fuite.loc[0] != r.loc[0]
V["E1_fuite_ligne_0"] = float(fuite.loc[0])

# ============================================================ exercice 2 : exemple
r = mensualiser(tests_e2.cadre_prix(tests_e2.PRIX), tests_e2.cadre_signaux(tests_e2.SIGNAUX))
attendu = [("A", "2026-01-31", "2026-01-30", 11.0, 0.5), ("A", "2026-02-28", "2026-02-27", 12.0, 0.7),
           ("B", "2026-01-31", "2026-01-30", 20.0, math.nan), ("B", "2026-03-31", "2026-03-30", 21.0, -1.0)]
assert tests_e2.comparer(r, pd.DataFrame(attendu, columns=r.columns)), r
# Jours de la semaine cités dans l'énoncé.
jours = {j: dt.date.fromisoformat(j).strftime("%A") for j in ["2026-01-30", "2026-01-31", "2026-02-27", "2026-03-30", "2026-03-31"]}
assert [dt.date.fromisoformat(j).weekday() for j in jours] == [4, 5, 4, 0, 1], jours  # vendredi, samedi, vendredi, lundi, mardi
V["E2_jours_semaine"] = jours
V["E2_lignes_exemple"] = len(r)

# ============================================================ exercice 3 : exemple
ex3 = tests_e3.cadre(tests_e3.EXEMPLE)
r25 = normaliser(ex3, 0.25, 0.75)
assert r25.loc[[0, 2, 3, 5, 7], "w"].tolist() == [3.0, 4.0, 2.0, 4.0, 2.0]
assert np.allclose(r25.loc[[0, 2, 3, 5, 7], "z"], [0.0, 1.0, -1.0, 1.0, -1.0], atol=1e-12)
assert r25.loc[[0, 2, 3, 5, 7], "rang"].tolist() == [0.6, 1.0, 0.2, 0.8, 0.4]
assert r25.loc[6].isna().all() and r25.loc[1, "w"] == 7.0 and np.isnan(r25.loc[1, "z"]) and r25.loc[1, "rang"] == 1.0
assert r25.loc[[4, 8], "w"].tolist() == [5.0, 5.0] and r25.loc[[4, 8], "z"].isna().all() and r25.loc[[4, 8], "rang"].tolist() == [0.75, 0.75]
V["E3_ex25"] = {c: [arrondi(x) for x in r25[c]] for c in r25.columns}
triees = [1.0, 2.0, 3.0, 4.0, 100.0]
for q, cle in [(0.25, "E3_q25"), (0.75, "E3_q75"), (0.1, "E3_q10"), (0.9, "E3_q90")]:
    h = (len(triees) - 1) * q
    V[cle + "_h"] = round(h, 12)
    V[cle] = round(quantile_lineaire(triees, q), 12)
    assert np.isclose(V[cle], np.quantile(triees, q)) and np.isclose(V[cle], pd.Series(triees).quantile(q))
assert (V["E3_q10"], V["E3_q90"], V["E3_q25"], V["E3_q75"]) == (1.4, 61.6, 2.0, 4.0)
V["E3_q90_f"] = round(V["E3_q90_h"] - math.floor(V["E3_q90_h"]), 12)
assert V["E3_q90_f"] == 0.6
w25 = np.array([3.0, 4.0, 2.0, 4.0, 2.0])  # valeurs winsorisées du 2026-01-05, quantiles 0,25 et 0,75
assert w25.mean() == 3.0 and np.sum((w25 - 3.0) ** 2) == 4.0 and w25.std(ddof=1) == 1.0
r10 = normaliser(ex3, 0.1, 0.9)
w = np.array([3.0, 61.6, 1.4, 4.0, 2.0])
z = (w - w.mean()) / w.std(ddof=1)
assert np.allclose(r10.loc[[0, 2, 3, 5, 7], "w"], w) and np.allclose(r10.loc[[0, 2, 3, 5, 7], "z"], z)
V["E3_ex10_moyenne"] = round(float(w.mean()), 12)
V["E3_ex10_et"] = round(float(w.std(ddof=1)), 12)
for lettre, valeur in zip("ABCDF", z):
    V[f"E3_ex10_z_{lettre}"] = round(float(valeur), 12)
V["E3_ex10_z_B_ddof0"] = round(float((61.6 - w.mean()) / w.std(ddof=0)), 12)
# Le piège des valeurs égales : écart-type calculé de trois fois 0,1.
V["E3_std_trois_01"] = float(pd.Series([0.1, 0.1, 0.1]).std())
assert V["E3_std_trois_01"] != 0.0

# ============================================================ exercice 4 : exemple
serie, sharpe = backtest(tests_e4.cadre(tests_e4.EXEMPLE), 2)
r_a, r_b, r_c, r_d = 110 / 100 - 1, 50 / 50 - 1, 19 / 20 - 1, 0.0
long_0105, court_0105 = (r_d + r_a) / 2, (r_c + r_b) / 2
pf_0105 = long_0105 - court_0105
pf_0106 = (19 / 19 - 1) - (99 / 110 - 1)
moy = (pf_0105 + pf_0106) / 2
et = math.sqrt(((pf_0105 - moy) ** 2 + (pf_0106 - moy) ** 2) / 1)
assert np.allclose(serie.to_numpy(), [pf_0105, pf_0106]) and list(serie.index) == ["2026-01-05", "2026-01-06"]
assert np.isclose(sharpe, math.sqrt(252) * moy / et)
V.update({"E4_r_A": round(r_a, 12), "E4_r_C": round(r_c, 12), "E4_long_0105": round(long_0105, 12), "E4_court_0105": round(court_0105, 12),
          "E4_pf_0105": round(pf_0105, 12), "E4_r_A_0106": round(99 / 110 - 1, 12), "E4_pf_0106": round(pf_0106, 12),
          "E4_moyenne": round(moy, 12), "E4_et": round(et, 12), "E4_sharpe": round(float(sharpe), 12), "E4_racine_252": round(math.sqrt(252), 12)})
# Les deux erreurs de l'exemple, chiffrées : sélection parmi les actifs dont le prix de t+1 est connu,
# et départage de A et B par l'ordre des lignes (B avant A) le 2026-01-06.
V["E4_pf_0105_lookahead"] = round((r_a - r_c), 12)  # D écarté : 3 éligibles, k = 1, long A, court C
V["E4_pf_0106_mauvais_departage"] = round((19 / 19 - 1) - (55 / 50 - 1), 12)  # court B au lieu de A

# ============================================================ leçon L-PANDAS-2 (outils/lecons_code_2.py)
# Section 5 : quantiles des valeurs 3, 1, 10, 2 par la définition linéaire, et la convention concurrente h = n q - 1.
x = sorted([3.0, 1.0, 10.0, 2.0])
for q, cle in [(0.1, "L_q10"), (0.9, "L_q90")]:
    h = (len(x) - 1) * q
    V[cle + "_h"], V[cle + "_f"] = round(h, 12), round(h - math.floor(h), 12)
    V[cle] = round(quantile_lineaire(x, q), 12)
    assert np.isclose(V[cle], np.quantile(x, q)) and np.isclose(V[cle], pd.Series(x).quantile(q))
assert (V["L_q10"], V["L_q90"], V["L_q10_h"], V["L_q90_h"], V["L_q90_f"]) == (1.3, 7.9, 0.3, 2.7, 0.7)
h_bis = len(x) * 0.9 - 1
V["L_q90_bis_h"] = round(h_bis, 12)
V["L_q90_bis"] = round(x[math.floor(h_bis)] + (h_bis - math.floor(h_bis)) * (x[math.floor(h_bis) + 1] - x[math.floor(h_bis)]), 12)
assert V["L_q90_bis"] != V["L_q90"]
assert pd.Series(x).quantile(0.9, interpolation="lower") == 3.0
assert pd.Series(x).quantile(0.9, interpolation="higher") == 10.0 and pd.Series(x).quantile(0.9, interpolation="nearest") == 10.0
# Contrôle « fuite » : rolling(2) sans groupby sur le cadre r de la section 2, trié par date puis actif.
r_lecon = pd.DataFrame({"date": ["2026-01-05", "2026-01-06", "2026-01-07"] * 2, "actif": ["A"] * 3 + ["B"] * 3,
                        "rendement": [1.0, np.nan, 3.0, 10.0, 20.0, 30.0]})
fuite = r_lecon.sort_values(["date", "actif"])["rendement"].rolling(2).mean()
V["L_fuite"] = float(fuite.loc[3])  # ligne de B au 2026-01-05
assert V["L_fuite"] == (1.0 + 10.0) / 2
# Contrôle « resample » : observations les 2026-01-15, 2026-01-20 et 2026-04-02.
from _correction import _regle_mensuelle  # noqa: E402
regle = _regle_mensuelle()  # "M" en pandas 1.4.2, "ME" en 2.3.3
lignes_resample = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2026-01-15", "2026-01-20", "2026-04-02"])).resample(regle).last()
V["L_resample_lignes"] = len(lignes_resample)
assert V["L_resample_lignes"] == 4 and int(lignes_resample.isna().sum()) == 2

# ============================================================ concordance sur les grands cas cachés
for nom, module in [("e1", tests_e1), ("e2", tests_e2), ("e3", tests_e3), ("e4", tests_e4)]:
    c = module.cas_caches()[-1]
    if nom == "e1":
        df, f, m = c["args"]
        pur = moyenne_glissante_pur([(i, *l) for i, l in zip(df.index, df.itertuples(index=False))], f, m)
        ok = tests_e1.comparer(pd.Series(pur, dtype=float), c["attendu"])
    elif nom == "e2":
        from _correction import mensualiser_pur
        p, s = c["args"]
        pur = mensualiser_pur(list(p.itertuples(index=False, name=None)), list(s.itertuples(index=False, name=None)))
        ok = tests_e2.comparer(pd.DataFrame(pur, columns=tests_e2.COLONNES_E2), c["attendu"])
    elif nom == "e3":
        df, qb, qh = c["args"]
        pur = normaliser_pur([(i, x.date, x.signal) for i, x in zip(df.index, df.itertuples(index=False))], qb, qh)
        ok = tests_e3.comparer(pd.DataFrame.from_dict(pur, orient="index", columns=["w", "z", "rang"]), c["attendu"])
    else:
        df, k = c["args"]
        s, sh = backtest_pur(list(df.itertuples(index=False, name=None)), k)
        ok = tests_e4.comparer((pd.Series([v for _, v in s], index=[d for d, _ in s]), sh), c["attendu"])
    assert ok, nom
    V[f"{nom}_grand_cas_pandas_egal_pur"] = True
    dire(f"{nom} : correction pandas = version sans bibliothèque sur « {c['nom']} »")


# ============================================================ solutions fausses et variantes justes
# Exercice 1
def e1_sans_groupby(df, fenetre, min_periodes):
    d = df.sort_values(["actif", "date"])
    return d["rendement"].rolling(fenetre, min_periods=min_periodes).mean().reindex(df.index)


def e1_sans_tri(df, fenetre, min_periodes):
    m = df.groupby("actif")["rendement"].rolling(fenetre, min_periods=min_periodes).mean()
    return m.reset_index(level=0, drop=True).reindex(df.index)


def e1_min_periods_oublie(df, fenetre, min_periodes):
    d = df.sort_values("date")
    return d.groupby("actif")["rendement"].transform(lambda s: s.rolling(fenetre).mean()).reindex(df.index)


def e1_nan_comme_zero(df, fenetre, min_periodes):
    d = df.sort_values("date").fillna({"rendement": 0.0})
    return d.groupby("actif")["rendement"].transform(lambda s: s.rolling(fenetre, min_periods=min_periodes).mean()).reindex(df.index)


def e1_transform_juste(df, fenetre, min_periodes):
    d = df.sort_values(["actif", "date"])
    return d.groupby("actif")["rendement"].transform(lambda s: s.rolling(fenetre, min_periods=min_periodes).mean()).reindex(df.index)


# Exercice 2
def _mensuel(prix, garder_lignes_sans_prix=False, garder_mois_vides=False):
    p = prix if garder_lignes_sans_prix else prix.dropna(subset=["prix"])
    p = p.assign(date=pd.to_datetime(p["date"]))
    p = p.assign(jour=p["date"]).set_index("date").sort_index()
    m = p.groupby("actif")[["jour", "prix"]].resample("M").last()
    if not garder_mois_vides:
        m = m.dropna(subset=["prix"])
    return m.reset_index().rename(columns={"date": "fin_de_mois"})


def _asof(m, signaux, **options):
    s = signaux.assign(jour=pd.to_datetime(signaux["date"]))[["jour", "actif", "signal"]].sort_values("jour")
    s = s.astype({"actif": object, "signal": float})
    m = m.astype({"actif": object})
    r = pd.merge_asof(m.sort_values("jour"), s, on="jour", **options)
    r = r.sort_values(["actif", "fin_de_mois"]).reset_index(drop=True)
    r["date"] = r["jour"]
    return r[tests_e2.COLONNES_E2]


def e2_jour_meme_inclus(prix, signaux):
    if prix.dropna(subset=["prix"]).empty:
        return mensualiser(prix, signaux)
    return _asof(_mensuel(prix), signaux, by="actif")


def e2_sans_by(prix, signaux):
    if prix.dropna(subset=["prix"]).empty:
        return mensualiser(prix, signaux)
    m = _mensuel(prix)
    s = signaux.assign(jour=pd.to_datetime(signaux["date"]))[["jour", "signal"]].sort_values("jour").astype({"signal": float})
    r = pd.merge_asof(m.sort_values("jour"), s, on="jour", allow_exact_matches=False)
    r = r.sort_values(["actif", "fin_de_mois"]).reset_index(drop=True)
    r["date"] = r["jour"]
    return r[tests_e2.COLONNES_E2]


def e2_date_de_la_derniere_ligne(prix, signaux):
    # resample sur toutes les lignes : last() saute les NaN colonne par colonne, si bien que « jour » est la
    # date de la dernière ligne du mois même quand son prix manque.
    if prix.dropna(subset=["prix"]).empty:
        return mensualiser(prix, signaux)
    return _asof(_mensuel(prix, garder_lignes_sans_prix=True), signaux, by="actif", allow_exact_matches=False)


def e2_mois_vides_gardes(prix, signaux):
    if prix.dropna(subset=["prix"]).empty:
        return mensualiser(prix, signaux)
    return _asof(_mensuel(prix, garder_mois_vides=True), signaux, by="actif", allow_exact_matches=False)


def e2_signal_avant_fin_de_mois(prix, signaux):
    if prix.dropna(subset=["prix"]).empty:
        return mensualiser(prix, signaux)
    m = _mensuel(prix)
    s = signaux.assign(fin_de_mois=pd.to_datetime(signaux["date"]))[["fin_de_mois", "actif", "signal"]].sort_values("fin_de_mois")
    s = s.astype({"actif": object, "signal": float})
    r = pd.merge_asof(m.astype({"actif": object}).sort_values("fin_de_mois"), s, on="fin_de_mois", by="actif", allow_exact_matches=False)
    r = r.sort_values(["actif", "fin_de_mois"]).reset_index(drop=True)
    r["date"] = r["jour"]
    return r[tests_e2.COLONNES_E2]


def e2_grouper_juste(prix, signaux):
    """Variante juste sans resample ni merge_asof : groupby sur (actif, mois), puis recherche par actif."""
    p = prix.dropna(subset=["prix"])
    if p.empty:
        return mensualiser(prix, signaux)
    p = p.assign(mois=p["date"].str[:7]).sort_values("date")
    m = p.groupby(["actif", "mois"]).last().reset_index()
    m["fin_de_mois"] = (pd.to_datetime(m["mois"] + "-01") + pd.offsets.MonthEnd(0)).dt.strftime("%Y-%m-%d")
    sig = {a: g.sort_values("date") for a, g in signaux.groupby("actif")}

    def dernier(ligne):
        g = sig.get(ligne["actif"])
        if g is None:
            return float("nan")
        avant = g[g["date"] < ligne["date"]]
        return float(avant["signal"].iloc[-1]) if len(avant) else float("nan")

    m["signal"] = [dernier(x) for _, x in m.iterrows()]
    return m.sort_values(["actif", "fin_de_mois"]).reset_index(drop=True)[tests_e2.COLONNES_E2]


# Exercice 3
def _normaliser(df, q_bas, q_haut, quantiles_globaux=False, ddof=1, z_sur_brut=False, rang_sur_w=False,
                test_std_nul=False, methode="linear", rang_sur_taille=False):
    if df.empty:
        return normaliser(df, q_bas, q_haut)
    g = df.groupby("date")["signal"]
    if quantiles_globaux:
        bas, haut = df["signal"].quantile(q_bas), df["signal"].quantile(q_haut)
    else:
        bas = g.transform(lambda s: s.quantile(q_bas, interpolation=methode))
        haut = g.transform(lambda s: s.quantile(q_haut, interpolation=methode))
    w = df["signal"].clip(lower=bas, upper=haut)
    base = df["signal"] if z_sur_brut else w
    gb = base.groupby(df["date"])
    et = gb.transform(lambda s: s.std(ddof=ddof))
    z = (base - gb.transform("mean")) / et
    z = z.where(et != 0) if test_std_nul else z.where(gb.transform("nunique") > 1)
    if rang_sur_w:
        rang = w.groupby(df["date"]).rank(pct=True)
    elif rang_sur_taille:
        rang = g.rank() / g.transform("size")
    else:
        rang = g.rank(pct=True)
    return pd.DataFrame({"w": w, "z": z, "rang": rang}, index=df.index).astype(float)


def e3_variante_numpy_juste(df, q_bas, q_haut):
    """Variante juste écrite avec numpy, date par date."""
    sortie = pd.DataFrame(np.nan, index=df.index, columns=["w", "z", "rang"])
    for _, g in df.dropna(subset=["signal"]).groupby("date"):
        x = g["signal"].to_numpy()
        w = np.clip(x, np.quantile(x, q_bas), np.quantile(x, q_haut))
        z = (w - w.mean()) / w.std(ddof=1) if len(np.unique(w)) > 1 else np.full(len(w), np.nan)
        sortie.loc[g.index, "w"], sortie.loc[g.index, "z"] = w, z
        sortie.loc[g.index, "rang"] = g["signal"].rank().to_numpy() / len(x)
    return sortie


# Exercice 4
def _backtest(df, nb_groupes, passe=False, selection_future=False, ligne_suivante=False, ordre_lignes=False,
              ddof=1, facteur=math.sqrt(252), sans_dates_vides=False):
    if df.empty:
        return backtest(df, nb_groupes)
    prix = df.pivot(index="date", columns="actif", values="prix").sort_index().sort_index(axis=1)
    signal = df.pivot(index="date", columns="actif", values="signal").reindex(index=prix.index, columns=prix.columns)
    if ligne_suivante:
        d = df.dropna(subset=["prix"]).sort_values("date")
        rl = d.groupby("actif")["prix"].shift(-1) / d["prix"] - 1
        r = d.assign(r=rl).pivot(index="date", columns="actif", values="r").reindex(index=prix.index, columns=prix.columns)
    elif passe:
        r = prix / prix.shift(1) - 1
    else:
        r = prix.shift(-1) / prix - 1
    eligible = prix.notna() & signal.notna()
    if selection_future:
        eligible &= r.notna()
    r = r.fillna(0.0)
    if ordre_lignes:
        # Rang par date dans l'ordre d'arrivée des lignes : les égalités sont départagées par cet ordre.
        longue = df.assign(e=(df["prix"].notna() & df["signal"].notna()))
        longue = longue[longue["e"]]
        rl = longue.groupby("date")["signal"].rank(method="first")
        rang = longue.assign(rang=rl).pivot(index="date", columns="actif", values="rang").reindex(index=prix.index, columns=prix.columns)
    else:
        rang = signal.where(eligible).rank(axis=1, method="first")
    n = eligible.sum(axis=1)
    k = n // nb_groupes
    pf = (r.where(rang.gt(n - k, axis=0)).sum(axis=1) - r.where(rang.le(k, axis=0)).sum(axis=1)) / k.where(k > 0)
    garde = pf.notna().iloc[:-1]
    pf = pf.fillna(0.0).iloc[:-1].astype(float)
    base = pf[garde] if sans_dates_vides else pf
    sh = float("nan") if len(base) < 2 or base.nunique() == 1 else float(facteur * base.mean() / base.std(ddof=ddof))
    return pf, sh


def score(module, fonction):
    cas = module.cas_caches()
    resultats = []
    for c in cas:
        try:
            resultats.append(bool(module.comparer(fonction(*c["args"]), c["attendu"])))
        except Exception:  # noqa: BLE001 : une solution fausse peut lever une exception, comptée comme échec
            resultats.append(False)
    return sum(resultats), len(cas), [c["nom"] for c, ok in zip(cas, resultats) if not ok]


SIMULATIONS = [
    ("E1_sans_groupby", tests_e1, e1_sans_groupby, False),
    ("E1_sans_tri", tests_e1, e1_sans_tri, False),
    ("E1_min_periods_oublie", tests_e1, e1_min_periods_oublie, False),
    ("E1_nan_comme_zero", tests_e1, e1_nan_comme_zero, False),
    ("E1_variante_transform", tests_e1, e1_transform_juste, True),
    ("E2_jour_meme_inclus", tests_e2, e2_jour_meme_inclus, False),
    ("E2_sans_by", tests_e2, e2_sans_by, False),
    ("E2_date_derniere_ligne", tests_e2, e2_date_de_la_derniere_ligne, False),
    ("E2_mois_vides_gardes", tests_e2, e2_mois_vides_gardes, False),
    ("E2_signal_avant_fin_de_mois", tests_e2, e2_signal_avant_fin_de_mois, False),
    ("E2_variante_groupby_mois", tests_e2, e2_grouper_juste, True),
    ("E3_quantiles_globaux", tests_e3, lambda d, a, b: _normaliser(d, a, b, quantiles_globaux=True), False),
    ("E3_ecart_type_ddof0", tests_e3, lambda d, a, b: _normaliser(d, a, b, ddof=0), False),
    ("E3_z_sur_signal_brut", tests_e3, lambda d, a, b: _normaliser(d, a, b, z_sur_brut=True), False),
    ("E3_rang_sur_winsorise", tests_e3, lambda d, a, b: _normaliser(d, a, b, rang_sur_w=True), False),
    ("E3_test_std_nul", tests_e3, lambda d, a, b: _normaliser(d, a, b, test_std_nul=True), False),
    ("E3_quantile_nearest", tests_e3, lambda d, a, b: _normaliser(d, a, b, methode="nearest"), False),
    ("E3_rang_sur_taille_avec_nan", tests_e3, lambda d, a, b: _normaliser(d, a, b, rang_sur_taille=True), False),
    ("E3_variante_transform", tests_e3, lambda d, a, b: _normaliser(d, a, b), True),
    ("E3_variante_numpy", tests_e3, e3_variante_numpy_juste, True),
    ("E4_rendement_passe", tests_e4, lambda d, k: _backtest(d, k, passe=True), False),
    ("E4_selection_avec_prix_futur", tests_e4, lambda d, k: _backtest(d, k, selection_future=True), False),
    ("E4_ligne_suivante_de_l_actif", tests_e4, lambda d, k: _backtest(d, k, ligne_suivante=True), False),
    ("E4_departage_ordre_des_lignes", tests_e4, lambda d, k: _backtest(d, k, ordre_lignes=True), False),
    ("E4_sharpe_ddof0", tests_e4, lambda d, k: _backtest(d, k, ddof=0), False),
    ("E4_sharpe_fois_252", tests_e4, lambda d, k: _backtest(d, k, facteur=252), False),
    ("E4_sharpe_sans_dates_vides", tests_e4, lambda d, k: _backtest(d, k, sans_dates_vides=True), False),
    ("E4_variante_reecrite", tests_e4, lambda d, k: _backtest(d, k), True),
]
for nom, module, fonction, doit_reussir in SIMULATIONS:
    reussis, total, echoues = score(module, fonction)
    assert (reussis == total) == doit_reussir, (nom, reussis, total, echoues)
    V[nom] = f"{reussis}/{total}"
    dire(f"{nom:34s} {reussis}/{total} tests cachés" + (f" ; échoue sur : {', '.join(echoues)}" if echoues else ""))

# ============================================================ fichier d'exploration
rng = np.random.default_rng(5201)
dates = pd.bdate_range("2026-01-01", periods=61).strftime("%Y-%m-%d")
n_actifs = 20
r = 0.02 * rng.standard_normal((61, n_actifs))
prix = 100 * np.exp(np.cumsum(r, axis=0))
# Signal synthétique : bruit plus une petite part du rendement du lendemain (planté, pour que le backtest ait quelque chose à trouver).
futur = np.vstack([np.log(prix[1:] / prix[:-1]), np.zeros((1, n_actifs))])
signal = rng.standard_normal((61, n_actifs)) + 10 * futur
lignes = []
for i, d in enumerate(dates):
    for a in range(n_actifs):
        if rng.random() < 0.03:
            continue
        p = np.nan if rng.random() < 0.03 else round(float(prix[i, a]), 4)
        s = np.nan if rng.random() < 0.05 else round(float(signal[i, a]), 4)
        lignes.append((d, f"A{a:02d}", p, s))
lignes = [lignes[i] for i in rng.permutation(len(lignes))]
panel = pd.DataFrame(lignes, columns=["date", "actif", "prix", "signal"])
texte = panel.to_csv(index=False)  # le fichier est écrit avec des fins de ligne Unix ci-dessous
texte = texte.replace("\r\n", "\n")
(DOSSIER / "panel.csv").write_text(texte, encoding="utf-8", newline="\n")
relu = pd.read_csv(DOSSIER / "panel.csv")
s5, sh5 = backtest(relu, 5)
s_pur, sh_pur = backtest_pur(list(relu.itertuples(index=False, name=None)), 5)
assert np.isclose(sh5, sh_pur) and np.allclose(s5.to_numpy(), [v for _, v in s_pur])
V["panel_lignes"] = len(relu)
V["panel_dates"] = int(relu["date"].nunique())
V["panel_actifs"] = int(relu["actif"].nunique())
V["E4_sharpe_panel"] = round(float(sh5), 10)
V["E4_rendement_moyen_panel"] = round(float(s5.mean()), 12)
dire(f"panel.csv : {len(relu)} lignes ; Sharpe annualisé du backtest à 5 groupes : {sh5:.4f}")

Path(__file__).with_name("seance_code_2.out.json").write_text(json.dumps(V, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
dire(json.dumps(V, ensure_ascii=False))
dire("Toutes les vérifications passent.")
Path(__file__).with_name("seance_code_2.out.txt").write_text("\n".join(LIGNES_TEXTE) + "\n", encoding="utf-8", newline="\n")
