"""Correction de l'entraînement « utilitaires de desk » : sept problèmes pandas."""
import numpy as np
import pandas as pd


# Problème 1 : rapprocher les confirmations
def reconcile_fills(confirmations):
    df = confirmations[confirmations["status"] == "FILLED"]
    df = df[(df["price"] > 0) & (df["quantity"] > 0)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", kind="mergesort").drop_duplicates(subset="trade_id", keep="first")
    return df.reset_index(drop=True)


# Problème 2 : noter la qualité d'exécution
def _min_max(s):
    if s.min() == s.max():
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def execution_quality_scores(fills):
    moyennes = fills.groupby("asset")[["slippage_bps", "fill_time_ms"]].mean()
    # Les deux signaux vont dans le même sens : plus bas est meilleur, donc 1 - ... sur les deux.
    score = 0.1 + 0.9 * (0.5 * (1 - _min_max(moyennes["slippage_bps"])) + 0.5 * (1 - _min_max(moyennes["fill_time_ms"])))
    return {a: round(float(v), 6) for a, v in score.items()}


# Problème 3 : valeurs de référence par lot
def batch_reference_prices(prices_long, queries, max_staleness_minutes):
    df = prices_long.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["price"].notna() & (df["price"] > 0)]
    resultats = []
    for _, requete in queries.iterrows():
        instant = pd.to_datetime(requete["ts"])
        passe = df[(df["asset"] == requete["asset"]) & (df["timestamp"] <= instant)].sort_values("timestamp")
        if passe.empty:
            resultats.append(None)
            continue
        dernier = passe.iloc[-1]
        age = (instant - dernier["timestamp"]).total_seconds() / 60
        resultats.append(None if age > max_staleness_minutes else float(dernier["price"]))
    return resultats


# Problème 4 : joindre le référentiel
def enrich_fills(fills, instruments):
    dernieres = instruments.assign(as_of=pd.to_datetime(instruments["as_of"]))
    dernieres = dernieres.sort_values("as_of").drop_duplicates(subset="asset", keep="last")
    # validate="m:1" transforme une explosion silencieuse du nombre de lignes en exception.
    sortie = fills.merge(dernieres[["asset", "multiplier", "currency"]], on="asset", how="left", validate="m:1")
    sortie["notional"] = (sortie["quantity"] * sortie["fill_price"] * sortie["multiplier"]).round(6)
    sortie["unmapped"] = sortie["multiplier"].isna()
    return sortie.reset_index(drop=True)


# Problème 5 : agréger le relevé en barres
def bar_summary(tape, freq):
    df = tape.copy()
    df["bar"] = pd.to_datetime(df["timestamp"]).dt.floor(freq)
    df["notionnel"] = df["price"] * df["size"]
    barres = df.groupby(["bar", "asset"], as_index=False).agg(notionnel=("notionnel", "sum"), volume=("size", "sum"))
    barres["vwap"] = (barres["notionnel"] / barres["volume"]).round(6)
    # transform : le total de la barre rendu à chaque ligne de la barre, sans changer la forme du cadre.
    barres["share"] = (barres["volume"] / barres.groupby("bar")["volume"].transform("sum")).round(6)
    barres["volume"] = barres["volume"].round(6)
    barres = barres.sort_values(["bar", "asset"]).reset_index(drop=True)
    barres["prev_vwap"] = barres.groupby("asset")["vwap"].shift(1)   # décalage à l'intérieur de chaque actif
    return barres[["bar", "asset", "vwap", "volume", "share", "prev_vwap"]]


# Problème 6 : résultat réalisé et latent
def track_book_with_realized(filled_requests, prices):
    actifs = [c for c in prices.columns if c != "timestamp"]
    position = {a: 0.0 for a in actifs}
    cout = {a: 0.0 for a in actifs}
    realise = {a: 0.0 for a in actifs}
    for _, ligne in filled_requests.iterrows():
        a, prix = ligne["asset"], ligne["fill_price"]
        signe_qte = -ligne["quantity"] if ligne["side"] == "BUY" else ligne["quantity"]
        p = position[a]
        if p == 0 or (p > 0) == (signe_qte > 0):      # ouverture ou renforcement : le coût moyen change
            nouvelle = p + signe_qte
            cout[a] = (abs(p) * cout[a] + abs(signe_qte) * prix) / abs(nouvelle)
            position[a] = nouvelle
        else:                                         # réduction : le coût moyen ne bouge pas, le gain se réalise
            realise[a] += min(abs(signe_qte), abs(p)) * (prix - cout[a]) * np.sign(p)
            position[a] = p + signe_qte
    dernier = prices.iloc[-1]
    latent = {a: (dernier[a] - cout[a]) * position[a] for a in actifs}
    return {"position": position, "avg_cost": cout, "realized_pnl": {a: round(v, 6) for a, v in realise.items()},
            "unrealized_pnl": {a: round(v, 6) for a, v in latent.items()},
            "total_pnl": round(sum(realise.values()) + sum(latent.values()), 6)}


# Problème 7 : le tableau de bord quotidien
def daily_scorecard(fills):
    df = fills.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.normalize()
    df["notionnel"] = df["quantity"] * df["fill_price"]
    stats = df.groupby("asset").agg(trades=("fill_price", "count"), volume=("quantity", "sum"),
                                    avg_price=("fill_price", "mean"), price_std=("fill_price", "std"))
    stats[["volume", "avg_price", "price_std"]] = stats[["volume", "avg_price", "price_std"]].round(6)
    # pivot_table et non pivot : un même actif a plusieurs exécutions le même jour.
    jour = df.pivot_table(index="date", columns="asset", values="notionnel", aggfunc="sum", fill_value=0.0)
    jour = jour.sort_index().sort_index(axis=1).round(6)
    return {"stats": stats, "daily": jour, "cumulative": jour.cumsum().round(6)}
