"""Correction de la séance MM 1 : teneur de marché sensible à la volatilité, et modèle de Glosten-Milgrom."""
from amplify_quant_trading import MarketMakerBase, MarketState, Quote


class VolAwareMarketMaker(MarketMakerBase):
    def __init__(self, base_spread_bps=300.0, vol_sensitivity=800.0, min_spread_bps=100.0, max_spread_bps=900.0):
        self.base_spread_bps = base_spread_bps
        self.vol_sensitivity = vol_sensitivity
        self.min_spread_bps = min_spread_bps
        self.max_spread_bps = max_spread_bps

    def quote(self, state: MarketState) -> Quote:
        if state.ref_price <= 0:
            raise ValueError("ref_price doit être strictement positif")
        # L'écart dépend de l'état seul : aucune mémoire, donc une fonction pure, comme le demande l'énoncé.
        spread_bps = self.base_spread_bps + self.vol_sensitivity * state.realized_vol
        spread_bps = min(max(spread_bps, self.min_spread_bps), self.max_spread_bps)
        # 1 point de base vaut 1/10000 ; la moitié de l'écart de chaque côté du prix de référence.
        demi_ecart = state.ref_price * (spread_bps / 10_000) / 2
        return Quote(bid=round(state.ref_price - demi_ecart, 2), ask=round(state.ref_price + demi_ecart, 2))


def glosten_milgrom(v_bas, v_haut, proba_haut, part_informes):
    """Rend (bid, ask) : les espérances de la valeur sachant une vente, puis sachant un achat."""
    if not v_bas < v_haut or not 0 <= proba_haut <= 1 or not 0 <= part_informes <= 1:
        raise ValueError("paramètres hors de leur domaine")
    p, a = proba_haut, part_informes
    # Probabilité d'un achat sachant la valeur : l'informé achète si et seulement si la valeur est haute,
    # le non-informé achète avec probabilité 1/2.
    achat_si_haut, achat_si_bas = a + (1 - a) / 2, (1 - a) / 2
    vente_si_haut, vente_si_bas = (1 - a) / 2, a + (1 - a) / 2

    def esperance(vraisemblance_haut, vraisemblance_bas):
        numerateur_haut = p * vraisemblance_haut      # formule de Bayes, numérateur pour la valeur haute
        numerateur_bas = (1 - p) * vraisemblance_bas
        total = numerateur_haut + numerateur_bas
        if total == 0:                                # ordre impossible : le prix ne peut être que la valeur certaine
            return v_haut if p == 1 else v_bas
        return (numerateur_haut * v_haut + numerateur_bas * v_bas) / total

    return esperance(vente_si_haut, vente_si_bas), esperance(achat_si_haut, achat_si_bas)
