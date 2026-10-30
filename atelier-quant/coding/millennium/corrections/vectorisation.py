"""Correction de l'entraînement « vectorisation numpy » : neuf problèmes sans boucle Python sur les données."""
import numpy as np


# Problème 1 : z-score glissant
def rolling_zscore(prices, window):
    prices = np.asarray(prices, dtype=float)
    sortie = np.full(prices.size, np.nan)
    if window < 1 or window > prices.size:
        return sortie
    vues = np.lib.stride_tricks.sliding_window_view(prices, window)
    moyennes = vues.mean(axis=1)
    ecarts = vues.std(axis=1, ddof=1) if window > 1 else np.zeros(vues.shape[0])
    z = np.zeros_like(moyennes)
    positifs = ecarts > 0
    z[positifs] = (prices[window - 1:][positifs] - moyennes[positifs]) / ecarts[positifs]
    sortie[window - 1:] = z
    return sortie


# Problème 2 : recherche « au plus tard à cet instant » vectorisée
def asof_indices(tape_ts, query_ts, max_staleness_minutes):
    tape_ts, query_ts = np.asarray(tape_ts), np.asarray(query_ts)
    indices = np.searchsorted(tape_ts, query_ts, side="right") - 1   # side="left" exclurait l'égalité exacte
    existe = indices >= 0
    age = (query_ts - tape_ts[np.where(existe, indices, 0)]) / np.timedelta64(1, "m")
    return np.where(existe & (age <= max_staleness_minutes), indices, -1).astype(np.int64)


# Problème 3 : prix moyen pondéré par étiquette
def vwap_by_label(prices, sizes, labels):
    prices, sizes, labels = np.asarray(prices, dtype=float), np.asarray(sizes, dtype=float), np.asarray(labels)
    if labels.size == 0:
        return labels.copy(), np.array([], dtype=float)
    distinctes, codes = np.unique(labels, return_inverse=True)
    codes = codes.ravel()
    notionnel = np.bincount(codes, weights=prices * sizes, minlength=distinctes.size)
    volume = np.bincount(codes, weights=sizes, minlength=distinctes.size)
    vwap = np.full(distinctes.size, np.nan)
    vwap[volume > 0] = notionnel[volume > 0] / volume[volume > 0]
    return distinctes, vwap


# Problème 4 : perte maximale depuis un sommet
def max_drawdown(equity):
    equity = np.asarray(equity, dtype=float)
    if equity.size == 0:
        return {"max_drawdown": 0.0, "peak_idx": -1, "trough_idx": -1}
    sommet = np.maximum.accumulate(equity)
    baisse = equity / sommet - 1.0
    creux = int(np.argmin(baisse))                     # argmin rend la première occurrence
    pic = int(np.argmax(equity[: creux + 1] == sommet[creux]))
    return {"max_drawdown": round(float(baisse[creux]), 6), "peak_idx": pic, "trough_idx": creux}


# Problème 5 : bêtas multifactoriels
def factor_betas(asset_returns, factor_returns):
    y = np.asarray(asset_returns, dtype=float)
    F = np.asarray(factor_returns, dtype=float)
    X = np.column_stack([np.ones(y.shape[0]), F])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    residus = y - X @ coef
    centre = y - y.mean()
    ss_tot = float(centre @ centre)
    r2 = 0.0 if ss_tot == 0 else 1.0 - float(residus @ residus) / ss_tot
    return {"alpha": round(float(coef[0]), 6), "betas": np.round(coef[1:], 6), "r_squared": round(float(r2), 6)}


# Problème 6 : les k meilleurs par ligne
def top_k_indices(scores, k):
    scores = np.asarray(scores, dtype=float)
    k = min(k, scores.shape[1])
    if k <= 0:
        return np.empty((scores.shape[0], 0), dtype=np.int64)
    # NaN classé après tout réel ; le tri stable sur l'opposé départage vers la plus petite colonne.
    rang = np.where(np.isnan(scores), -np.inf, scores)
    return np.argsort(-rang, axis=1, kind="stable")[:, :k].astype(np.int64)


# Problème 7 : winsoriser puis standardiser
def winsorize_zscore(signal, lower_pct, upper_pct):
    signal = np.asarray(signal, dtype=float)
    if np.isnan(signal).all():
        return signal.copy()
    bas, haut = np.nanpercentile(signal, [lower_pct, upper_pct])
    borne = np.clip(signal, bas, haut)
    ecart = np.nanstd(borne, ddof=1) if np.count_nonzero(~np.isnan(borne)) > 1 else 0.0
    if ecart == 0:
        return np.where(np.isnan(borne), np.nan, 0.0)
    return (borne - np.nanmean(borne)) / ecart


# Problème 8 : propagation vers le bas d'un tableau à deux dimensions
def ffill_2d(tape):
    tape = np.asarray(tape, dtype=float)
    if tape.size == 0:
        return tape.copy()
    lignes = np.arange(tape.shape[0])[:, None]
    derniere = np.maximum.accumulate(np.where(~np.isnan(tape), lignes, -1), axis=0)  # dernière ligne valide au-dessus
    recopie = np.take_along_axis(tape, np.clip(derniere, 0, None), axis=0)
    return np.where(derniere >= 0, recopie, np.nan)                                  # NaN de tête conservé


# Problème 9 : valoriser le livre
def book_exposure(positions, prices, fx_rates):
    valeurs = np.asarray(positions, dtype=float) * np.asarray(prices, dtype=float) * np.asarray(fx_rates, dtype=float)
    net = valeurs.sum(axis=1, keepdims=True)             # (n_jours, 1) : se diffuse sur les actifs
    poids = np.zeros_like(valeurs)
    vivant = net.ravel() != 0
    poids[vivant] = valeurs[vivant] / net[vivant]
    return {"value": np.round(net.ravel(), 6), "weights": np.round(poids, 6), "gross": np.round(np.abs(valeurs).sum(axis=1), 6)}
