"""Correction des six problèmes de l'OA Millennium x AmplifyME (reconstitution de mémoire, non officielle)."""
import numpy as np
import pandas as pd


def _actifs(prix):
    return [c for c in prix.columns if c != "timestamp"]


# Problème 1 : réparer le relevé de prix
def clean_prices(prices):
    df = prices.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Tri stable : parmi des lignes de même horodatage, keep="first" garde bien la première du fichier.
    df = df.sort_values("timestamp", kind="mergesort").drop_duplicates(subset="timestamp", keep="first")
    df = df.reset_index(drop=True)
    actifs = _actifs(df)
    df[actifs] = df[actifs].mask(df[actifs] <= 0).ffill()   # jamais bfill : ce serait regarder le futur
    return df


# Problème 2 : mesurer la liquidité
def _min_max(s):
    if s.min() == s.max():
        return pd.Series(0.5, index=s.index)      # étendue nulle : la normalisation 0/0 est remplacée par 0,5
    return (s - s.min()) / (s.max() - s.min())


def liquidity_scores(prices, requests):
    actifs = _actifs(prices)
    vol = _min_max(pd.Series({a: prices[a].pct_change().std() for a in actifs}))
    nombres = requests["asset"].value_counts() if len(requests) else pd.Series(dtype=float)
    freq = _min_max(pd.Series({a: float(nombres.get(a, 0)) for a in actifs}))
    return {a: round(float(0.1 + 0.9 * (0.5 * (1 - vol[a]) + 0.5 * freq[a])), 6) for a in actifs}


# Problème 3 : valeur de référence à un instant donné
def reference_price(prices, asset, ts, max_staleness_minutes):
    temps = pd.to_datetime(prices["timestamp"])
    instant = pd.to_datetime(ts)
    # Validité jugée sur la seule colonne demandée : un autre actif cassé ne disqualifie pas la ligne.
    garde = (temps <= instant) & prices[asset].notna() & (prices[asset] > 0)
    if not garde.any():
        return {"price": None}
    valides = prices.loc[garde, asset].to_numpy()
    instants = temps[garde]
    dernier = instants.to_numpy().argmax()
    age_minutes = (instant - instants.iloc[dernier]).total_seconds() / 60    # total_seconds, jamais .seconds
    return {"price": None if age_minutes > max_staleness_minutes else float(valides[dernier])}


# Problème 4 : coter une demande
def quote_request(reference, half_spread, side, quantity, size_params, tick_size):
    penalite = size_params["k"] * quantity
    if side == "BUY":   # le client achète : le teneur vend, au-dessus de la référence, arrondi vers le haut
        return float(np.ceil(np.round((reference + half_spread + penalite) / tick_size, 9)) * tick_size)
    return float(np.floor(np.round((reference - half_spread - penalite) / tick_size, 9)) * tick_size)


# Problème 5 : flux de cotations et impact cumulé
def quote_impact_stream(reference, impact_params, liquidity_score, requests_for_asset):
    k, decay, half_spread = impact_params["k"], impact_params["decay"], impact_params["half_spread"]
    impact_cumule, cotes = 0.0, []
    for demande in requests_for_asset:
        signe = 1.0 if demande["side"] == "BUY" else -1.0
        cotes.append(reference + signe * half_spread + impact_cumule)   # l'état AVANT l'impact de cette demande
        impact = k * demande["quantity"] * (1 - liquidity_score)
        impact_cumule = (impact_cumule + signe * impact) * decay        # le déclin porte sur tout l'état
    return cotes


# Problème 6 : tenir le livre
def track_book(filled_requests, prices):
    actifs = _actifs(prices)
    inventaire, cash = {a: 0 for a in actifs}, 0.0
    for _, ligne in filled_requests.iterrows():
        signe = -1 if ligne["side"] == "BUY" else 1      # le teneur prend le côté opposé du client
        inventaire[ligne["asset"]] += signe * ligne["quantity"]
        cash -= signe * ligne["fill_price"] * ligne["quantity"]
    dernier = prices.iloc[-1]
    pnl = cash + sum(inventaire[a] * dernier[a] for a in actifs)
    return {"inventory": inventaire, "cash": round(cash, 6), "pnl": round(pnl, 6)}
