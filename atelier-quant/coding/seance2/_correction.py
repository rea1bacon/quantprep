"""Corrections de la séance de code 2. À n'ouvrir qu'après avoir tenté les exercices.

Chaque fonction existe en deux versions indépendantes : pandas, et sans bibliothèque externe
(bibliothèque standard seulement : math, calendar). Les tests exigent qu'elles soient d'accord.
Testées avec pandas 1.4.2 (environnement des notebooks de la plateforme) et pandas 2.3.3.
"""
import calendar
import math

import numpy as np
import pandas as pd

NAN = float("nan")


def _nan(x):
    return isinstance(x, float) and math.isnan(x)


# ------------------------------------------------------------------ exercice 1
def moyenne_glissante(df: pd.DataFrame, fenetre: int, min_periodes: int) -> pd.Series:
    if df.empty:
        return pd.Series(np.nan, index=df.index, dtype=float)
    # 1. Trier par date : rolling, comme shift, travaille dans l'ordre des lignes.
    #    Trier par date suffit, car groupby conserve l'ordre des lignes à l'intérieur de chaque actif.
    d = df.sort_values("date")
    # 2. La fenêtre glisse À L'INTÉRIEUR de chaque actif : sans groupby, la fenêtre de la première
    #    date d'un actif contiendrait les dernières lignes d'un autre actif.
    #    min_periods compte les valeurs NON MANQUANTES de la fenêtre.
    m = d.groupby("actif")["rendement"].rolling(fenetre, min_periods=min_periodes).mean()
    # 3. groupby(...).rolling(...) renvoie un index à deux niveaux (actif, étiquette d'origine) :
    #    on retire le niveau actif, puis on revient à l'ordre de df.
    return m.reset_index(level=0, drop=True).reindex(df.index)


def moyenne_glissante_pur(lignes, fenetre, min_periodes):
    """lignes : liste de (index, date, actif, rendement). Renvoie {index: moyenne ou nan}."""
    par_actif = {}
    for i, date, actif, r in lignes:
        par_actif.setdefault(actif, []).append((date, i, r))
    sortie = {}
    for serie in par_actif.values():
        serie.sort(key=lambda t: t[0])  # au plus une ligne par (date, actif) : pas d'égalité de dates
        for k in range(len(serie)):
            valeurs = [r for _, _, r in serie[max(0, k - fenetre + 1):k + 1] if not _nan(r)]
            sortie[serie[k][1]] = sum(valeurs) / len(valeurs) if len(valeurs) >= min_periodes else NAN
    return sortie


# ------------------------------------------------------------------ exercice 2
COLONNES_E2 = ["actif", "fin_de_mois", "date", "prix", "signal"]


def _regle_mensuelle():
    # "M" en pandas 1.4.2 ; "ME" dans les versions récentes (2.3.3 testée), où "M" émet un avertissement.
    try:
        pd.Series([1.0], index=pd.to_datetime(["2026-01-05"])).resample("ME").last()
        return "ME"
    except ValueError:
        return "M"


def mensualiser(prix: pd.DataFrame, signaux: pd.DataFrame) -> pd.DataFrame:
    # 1. Ne garder que les prix connus AVANT de rééchantillonner : sinon la « dernière ligne du mois »
    #    pourrait être une ligne sans prix, et sa date ne serait pas celle du prix retenu.
    p = prix.dropna(subset=["prix"])
    if p.empty:
        return pd.DataFrame({c: pd.Series(dtype=float if c in ("prix", "signal") else object) for c in COLONNES_E2})
    p = p.assign(date=pd.to_datetime(p["date"]))
    p = p.assign(jour=p["date"]).set_index("date").sort_index()
    # 2. Dernier prix connu du mois, actif par actif. resample étiquette chaque mois par son dernier
    #    jour civil et crée une ligne vide pour un mois sans observation situé entre deux mois observés :
    #    dropna les retire.
    m = (p.groupby("actif")[["jour", "prix"]].resample(_regle_mensuelle()).last()
         .dropna(subset=["prix"]).reset_index().rename(columns={"date": "fin_de_mois"}))
    # 3. merge_asof : pour chaque ligne de gauche, la dernière ligne de droite du même actif (by) dont la
    #    date est STRICTEMENT antérieure (allow_exact_matches=False). Les deux cadres doivent être triés
    #    par la clé on, qui doit être une date ou un nombre (pas une chaîne).
    s = signaux.assign(jour=pd.to_datetime(signaux["date"]))[["jour", "actif", "signal"]].sort_values("jour")
    s = s.astype({"actif": m["actif"].dtype, "signal": float})
    r = pd.merge_asof(m.sort_values("jour"), s, on="jour", by="actif", allow_exact_matches=False)
    r = r.sort_values(["actif", "fin_de_mois"]).reset_index(drop=True)
    r["fin_de_mois"] = r["fin_de_mois"].dt.strftime("%Y-%m-%d")
    r["date"] = r["jour"].dt.strftime("%Y-%m-%d")
    return r[COLONNES_E2]


def mensualiser_pur(lignes_prix, lignes_signaux):
    """lignes_prix : (date, actif, prix) ; lignes_signaux : (date, actif, signal).
    Renvoie la liste triée des (actif, fin_de_mois, date, prix, signal)."""
    dernier = {}
    for date, actif, p in lignes_prix:
        if _nan(p):
            continue
        cle = (actif, date[:7])
        if cle not in dernier or date > dernier[cle][0]:
            dernier[cle] = (date, p)
    par_actif = {}
    for date, actif, s in lignes_signaux:
        par_actif.setdefault(actif, []).append((date, s))
    for v in par_actif.values():
        v.sort(key=lambda t: t[0])
    sortie = []
    for actif, mois in sorted(dernier):
        date, p = dernier[(actif, mois)]
        annee, numero = int(mois[:4]), int(mois[5:7])
        fin = f"{mois}-{calendar.monthrange(annee, numero)[1]:02d}"
        anterieurs = [s for d, s in par_actif.get(actif, []) if d < date]  # chaînes AAAA-MM-JJ : ordre = chronologie
        sortie.append((actif, fin, date, p, anterieurs[-1] if anterieurs else NAN))
    return sortie


# ------------------------------------------------------------------ exercice 3
def normaliser(df: pd.DataFrame, q_bas: float, q_haut: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"w": [], "z": [], "rang": []}, index=df.index, dtype=float)
    g = df.groupby("date")["signal"]
    # 1. Quantiles de chaque date (interpolation linéaire, la définition de l'énoncé, qui est celle
    #    par défaut de pandas et de numpy), répandus sur les lignes de la date par map.
    bas = df["date"].map(g.quantile(q_bas))
    haut = df["date"].map(g.quantile(q_haut))
    w = df["signal"].clip(lower=bas, upper=haut)
    # 2. z-score du signal winsorisé, date par date ; std de pandas : dénominateur n - 1.
    gw = w.groupby(df["date"])
    z = (w - gw.transform("mean")) / gw.transform("std")
    # Valeurs toutes égales : on renvoie NaN par la règle de l'énoncé, sans tester std == 0, car l'écart-type
    # calculé de valeurs égales n'est pas toujours exactement nul en flottants (0,1 trois fois par exemple).
    # nunique ignore les NaN : il vaut 1 pour une seule valeur non manquante aussi.
    z = z.where(gw.transform("nunique") > 1)
    # 3. Rang du signal BRUT divisé par le nombre de signaux non manquants de la date.
    rang = g.rank(pct=True)
    return pd.DataFrame({"w": w, "z": z, "rang": rang}, index=df.index).astype(float)


def quantile_lineaire(triees, q):
    """Quantile d'une liste triée non vide, par la définition de l'énoncé."""
    h = (len(triees) - 1) * q
    i = math.floor(h)
    f = h - i
    if f == 0 or i + 1 >= len(triees):
        return triees[i]
    return triees[i] + f * (triees[i + 1] - triees[i])


def rangs_moyens(valeurs):
    """Rangs moyens (à partir de 1) d'une liste de nombres sans NaN."""
    ordre = sorted(range(len(valeurs)), key=lambda i: valeurs[i])
    rangs = [0.0] * len(valeurs)
    i = 0
    while i < len(ordre):
        j = i
        while j + 1 < len(ordre) and valeurs[ordre[j + 1]] == valeurs[ordre[i]]:
            j += 1
        for k in range(i, j + 1):
            rangs[ordre[k]] = (i + j) / 2 + 1
        i = j + 1
    return rangs


def normaliser_pur(lignes, q_bas, q_haut):
    """lignes : liste de (index, date, signal). Renvoie {index: (w, z, rang)}."""
    par_date = {}
    sortie = {}
    for i, date, s in lignes:
        if _nan(s):
            sortie[i] = (NAN, NAN, NAN)
        else:
            par_date.setdefault(date, []).append((i, s))
    for couples in par_date.values():
        valeurs = [s for _, s in couples]
        n = len(valeurs)
        triees = sorted(valeurs)
        b, h = quantile_lineaire(triees, q_bas), quantile_lineaire(triees, q_haut)
        w = [min(max(s, b), h) for s in valeurs]
        if len(set(w)) > 1:
            moy = sum(w) / n
            et = math.sqrt(sum((x - moy) ** 2 for x in w) / (n - 1))
            z = [(x - moy) / et for x in w]
        else:
            z = [NAN] * n
        for (i, _), wi, zi, r in zip(couples, w, z, rangs_moyens(valeurs)):
            sortie[i] = (wi, zi, r / n)
    return sortie


# ------------------------------------------------------------------ exercice 4
JOURS_PAR_AN = 252


def sharpe_annualise(rendements):
    """Ratio de Sharpe annualisé d'une Series : racine(252) x moyenne / écart-type (dénominateur n - 1)."""
    if len(rendements) < 2 or rendements.nunique() == 1:
        return NAN
    return float(math.sqrt(JOURS_PAR_AN) * rendements.mean() / rendements.std())


def backtest(df: pd.DataFrame, nb_groupes: int):
    # 1. Format large : une ligne par date du calendrier (toutes les dates de df), une colonne par actif,
    #    triées. Une absence de ligne devient NaN, à la bonne place dans le calendrier.
    prix = df.pivot(index="date", columns="actif", values="prix").sort_index().sort_index(axis=1)
    signal = df.pivot(index="date", columns="actif", values="signal").reindex(index=prix.index, columns=prix.columns)
    # 2. Rendement réalisé de t à t+1, rangé à la ligne t : shift(-1) va chercher la date SUIVANTE du calendrier.
    #    Prix en t+1 inconnu : la position rapporte 0 (convention de l'énoncé).
    r = (prix.shift(-1) / prix - 1).fillna(0.0)
    # 3. Éligibles en t : signal et prix connus en t. Aucune information de t+1 n'intervient ici.
    eligible = prix.notna() & signal.notna()
    # 4. Rangs 1..n_t ; method="first" départage les égalités par position de colonne, donc par actif croissant.
    rang = signal.where(eligible).rank(axis=1, method="first")
    n = eligible.sum(axis=1)
    k = n // nb_groupes
    long_ = rang.gt(n - k, axis=0)
    court = rang.le(k, axis=0)
    # 5. Poids +1/k et -1/k ; k = 0 : aucune position, rendement 0.
    pf = (r.where(long_).sum(axis=1) - r.where(court).sum(axis=1)) / k.where(k > 0)
    pf = pf.fillna(0.0).iloc[:-1].astype(float)  # la dernière date n'a pas de rendement réalisé
    return pf, sharpe_annualise(pf)


def backtest_pur(lignes, nb_groupes):
    """lignes : liste de (date, actif, prix, signal). Renvoie ([(date, rendement)], sharpe)."""
    par_date = {}
    for date, actif, p, s in lignes:
        par_date.setdefault(date, {})[actif] = (p, s)
    dates = sorted(par_date)
    serie = []
    for t, suivante in zip(dates, dates[1:]):
        eligibles = sorted((s, a) for a, (p, s) in par_date[t].items() if not (_nan(p) or _nan(s)))
        k = len(eligibles) // nb_groupes
        if k == 0:
            serie.append((t, 0.0))
            continue

        def rendement(a):
            p1 = par_date[suivante].get(a, (NAN, NAN))[0]
            return 0.0 if _nan(p1) else p1 / par_date[t][a][0] - 1

        courts = [a for _, a in eligibles[:k]]
        longs = [a for _, a in eligibles[-k:]]
        serie.append((t, sum(rendement(a) for a in longs) / k - sum(rendement(a) for a in courts) / k))
    valeurs = [v for _, v in serie]
    if len(valeurs) < 2 or len(set(valeurs)) == 1:
        return serie, NAN
    moy = sum(valeurs) / len(valeurs)
    et = math.sqrt(sum((v - moy) ** 2 for v in valeurs) / (len(valeurs) - 1))
    return serie, math.sqrt(JOURS_PAR_AN) * moy / et
