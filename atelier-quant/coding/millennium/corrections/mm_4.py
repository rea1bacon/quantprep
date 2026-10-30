"""Correction de la séance MM 4 : demi-vie d'un écart, et décision d'arbitrage entre un ETF et son panier."""
import numpy as np

from amplify_quant_trading import ArbitrageBotBase


def estimate_half_life(mispricing):
    x = np.asarray(mispricing, dtype=float)
    phi, _constante = np.polyfit(x[:-1], x[1:], 1)
    if phi >= 1:          # aucun retour à la moyenne détecté : la demi-vie n'est pas finie
        return float("inf")
    if phi <= 0:          # aucune persistance : l'écart disparaît dès le pas suivant
        return 0.0
    return float(np.log(2) / -np.log(phi))


class CointegrationArbBot(ArbitrageBotBase):
    def __init__(self, entry_z=2.0, commission_pct=0.001, min_edge_multiple=2.0):
        self.entry_z = entry_z
        self.commission_pct = commission_pct
        self.min_edge_multiple = min_edge_multiple

    def synthetic_price(self, weights, prices):
        return sum(w * prices[ticker] for ticker, w in weights.items())

    def zscore(self, mispricing_history):
        h = np.asarray(mispricing_history, dtype=float)
        ecart_type = h.std()      # écart-type de population, ddof=0, comme le fixe l'énoncé
        return 0.0 if ecart_type == 0 else float((h[-1] - h.mean()) / ecart_type)

    def signal(self, mispricing_pct, z_score):
        if abs(z_score) < self.entry_z:                                             # barre statistique
            return "no_trade"
        if abs(mispricing_pct) < self.min_edge_multiple * (2 * self.commission_pct):  # barre économique
            return "no_trade"
        return "sell_etf_buy_basket" if mispricing_pct > 0 else "buy_etf_sell_basket"
