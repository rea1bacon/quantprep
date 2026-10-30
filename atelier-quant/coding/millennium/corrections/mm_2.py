"""Correction de la séance MM 2 : prix d'indifférence et teneur de marché d'Avellaneda-Stoikov."""
import numpy as np

from amplify_quant_trading import MarketMakerBase, MarketState, Quote

JOURS_PAR_AN = 252


def prix_indifference(s, q, gamma, sigma, tau):
    """Rend (r_achat, r_vente, r) ; sigma en euros par racine d'année, tau en années."""
    penalite = gamma * sigma**2 * tau
    return s - penalite * (2 * q + 1) / 2, s - penalite * (2 * q - 1) / 2, s - q * penalite


class AvellanedaStoikovMarketMaker(MarketMakerBase):
    def __init__(self, gamma=0.1, k=1.5, max_skew_bps=400.0):
        self.gamma = gamma
        self.k = k
        self.max_skew_bps = max_skew_bps

    def quote(self, state: MarketState) -> Quote:
        if state.ref_price <= 0:
            raise ValueError("ref_price doit être strictement positif")
        # Deux conversions d'unités : une fraction de journée en années, une volatilité relative en euros.
        tau = state.time_remaining / JOURS_PAR_AN
        sigma = state.realized_vol * state.ref_price
        terme_risque = self.gamma * sigma**2 * tau

        decalage_max = state.ref_price * self.max_skew_bps / 10_000
        decalage = float(np.clip(state.inventory * terme_risque, -decalage_max, decalage_max))
        r = state.ref_price - decalage

        terme_liquidite = (2 / self.gamma) * np.log(1 + self.gamma / self.k)
        ecart = max(terme_risque + terme_liquidite, state.ref_price * 0.001)  # plancher : 0,1 % du prix

        bid, ask = round(r - ecart / 2, 2), round(r + ecart / 2, 2)
        if bid <= 0:                  # le terme de liquidité ne dépend pas du prix : il peut dépasser un petit prix
            bid = round(r * 0.5, 2)
        if bid >= ask:
            ask = round(bid + max(ecart, 0.01), 2)
        return Quote(bid=bid, ask=ask)
