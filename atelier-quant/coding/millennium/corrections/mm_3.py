"""Correction de la séance MM 3 : bêta, bande de non-intervention, rendements couverts sans regard vers le futur."""
import numpy as np
import pandas as pd

from amplify_quant_trading import ExchangeTrade, HedgerBase


class RollingHedger(HedgerBase):
    def __init__(self, window=60, no_trade_band_pct=0.03):
        self.window = window
        self.no_trade_band_pct = no_trade_band_pct

    def hedge_ratio(self, asset_returns, hedge_returns):
        # Numérateur et dénominateur avec le même nombre de degrés de liberté : np.cov(x, y) rend une matrice en ddof=1,
        # alors que np.var(y) est en ddof=0 par défaut. Les mélanger gonfle le bêta d'un facteur n / (n - 1).
        c = np.cov(np.asarray(asset_returns, dtype=float), np.asarray(hedge_returns, dtype=float))
        return float(c[0, 1] / c[1, 1])

    def rebalance(self, equity_value, current_hedge_notional, target_ratio, hedge_price, timestamp=0):
        if hedge_price <= 0:
            raise ValueError("hedge_price doit être strictement positif")
        cible = -equity_value * target_ratio
        ajustement = cible - current_hedge_notional
        if ajustement == 0 or abs(ajustement) < self.no_trade_band_pct * abs(equity_value):
            return None
        # Ajustement négatif : la couverture doit devenir plus courte, donc on vend.
        return ExchangeTrade(ticker="HEDGE", trade_volume=abs(ajustement) / hedge_price, ref_price=hedge_price,
                             action="sell" if ajustement < 0 else "buy", timestamp=timestamp)


def rendements_couverts(actif, couverture, fenetre):
    """Pour t >= fenetre, le bêta est estimé sur les fenetre rendements strictement antérieurs à t."""
    actif, couverture = np.asarray(actif, dtype=float), np.asarray(couverture, dtype=float)
    sortie = np.full(actif.size, np.nan)
    for t in range(fenetre, actif.size):
        c = np.cov(actif[t - fenetre:t], couverture[t - fenetre:t])
        sortie[t] = actif[t] - c[0, 1] / c[1, 1] * couverture[t]
    return pd.Series(sortie)
