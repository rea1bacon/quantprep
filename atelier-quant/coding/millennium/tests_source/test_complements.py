"""Tests ajoutés aux suites d'origine, pour ce qu'elles ne contrôlaient pas (formule exacte, ddof, cas dégénérés)
et pour les exercices propres à la plateforme (glosten_milgrom, prix_indifference, rendements_couverts)."""
import numpy as np
import pandas as pd

import _cible as mod
import _mini_pytest as pytest
from amplify_quant_trading import MarketState


def etat(ref_price=100.0, realized_vol=0.3, time_remaining=1.0, inventory=0.0):
    return MarketState(timestamp=0, ref_price=ref_price, realized_vol=realized_vol,
                       time_remaining=time_remaining, inventory=inventory)


# --- Glosten-Milgrom ---------------------------------------------------------------------------
def test_gm_cas_symetrique():
    assert mod.glosten_milgrom(98.0, 102.0, 0.5, 0.25) == pytest.approx((99.5, 100.5))


def test_gm_sans_informes_ecart_nul():
    b, a = mod.glosten_milgrom(98.0, 102.0, 0.7, 0.0)
    assert b == pytest.approx(100.8) and a == pytest.approx(100.8)


def test_gm_tous_informes():
    assert mod.glosten_milgrom(98.0, 102.0, 0.5, 1.0) == pytest.approx((98.0, 102.0))


def test_gm_a_priori_asymetrique():
    # Calcul à la main : p = 0,7, a = 0,5 ; P(achat | haut) = 0,75, P(achat | bas) = 0,25.
    b, a = mod.glosten_milgrom(0.0, 1.0, 0.7, 0.5)
    assert a == pytest.approx(0.525 / 0.6) and b == pytest.approx(0.175 / 0.4)


def test_gm_valeur_certaine():
    assert mod.glosten_milgrom(98.0, 102.0, 1.0, 0.5) == pytest.approx((102.0, 102.0))


def test_gm_parametres_invalides():
    with pytest.raises(ValueError):
        mod.glosten_milgrom(102.0, 98.0, 0.5, 0.5)


# --- Prix d'indifférence et Avellaneda-Stoikov ---------------------------------------------------
def test_indifference_valeurs():
    assert mod.prix_indifference(100.0, 2, 0.5, 2.0, 0.25) == pytest.approx((98.75, 99.25, 99.0))


def test_indifference_stock_nul():
    a, v, r = mod.prix_indifference(100.0, 0, 0.5, 2.0, 0.25)
    assert r == pytest.approx(100.0) and a == pytest.approx(99.75) and v == pytest.approx(100.25)


def test_indifference_ecart_entre_achat_et_vente():
    a, v, _ = mod.prix_indifference(55.0, -7, 0.3, 4.0, 0.5)
    assert v - a == pytest.approx(0.3 * 16.0 * 0.5)


def test_as_formule_exacte():
    # gamma = 0,1 ; k = 1,5 ; sigma = 30 euros ; tau = 1/252 ; q = 5
    risque = 0.1 * 900 / 252
    ecart = risque + 20 * np.log(1 + 0.1 / 1.5)
    r = 100 - 5 * risque
    q = mod.AvellanedaStoikovMarketMaker().quote(etat(inventory=5.0))
    assert q.bid == pytest.approx(round(r - ecart / 2, 2)) and q.ask == pytest.approx(round(r + ecart / 2, 2))


def test_as_decalage_plafonne():
    q = mod.AvellanedaStoikovMarketMaker(max_skew_bps=400.0).quote(etat(inventory=1e6))
    assert q.mid == pytest.approx(96.0, abs=0.02), "le décalage est plafonné à 400 points de base du prix"


def test_as_fin_de_journee_sans_risque():
    risque_nul = mod.AvellanedaStoikovMarketMaker().quote(etat(inventory=50.0, time_remaining=0.0))
    assert risque_nul.mid == pytest.approx(100.0, abs=0.01), "à la clôture, le terme de risque s'annule"


# --- Couverture ----------------------------------------------------------------------------------
def test_beta_egale_pente_des_moindres_carres():
    x = pd.Series([0.01, -0.02, 0.03, 0.00, -0.01, 0.02])
    y = pd.Series([0.02, -0.01, 0.02, 0.01, -0.03, 0.01])
    pente = np.polyfit(x, y, 1)[0]
    assert mod.RollingHedger().hedge_ratio(y, x) == pytest.approx(pente, rel=1e-9), (
        "covariance et variance doivent utiliser le même ddof")


def test_ajustement_nul_sans_ordre():
    assert mod.RollingHedger(no_trade_band_pct=0.0).rebalance(100_000, -70_000, 0.7, 50.0) is None


def test_bande_atteinte_exactement():
    t = mod.RollingHedger(no_trade_band_pct=0.03).rebalance(100_000, -67_000, 0.7, 50.0)
    assert t is not None and t.action == "sell", "seul un ajustement strictement inférieur à la bande est ignoré"


def _rendements(n=200, graine=3):
    rng = np.random.default_rng(graine)
    c = rng.normal(0, 0.01, n)
    return pd.Series(0.8 * c + rng.normal(0, 0.005, n)), pd.Series(c)


def test_couverts_forme():
    a, c = _rendements()
    s = mod.rendements_couverts(a, c, 60)
    assert len(s) == 200 and s.iloc[:60].isna().all() and s.iloc[60:].notna().all()


def test_couverts_premiere_valeur():
    a, c = _rendements()
    s = mod.rendements_couverts(a, c, 60)
    b = np.cov(a.iloc[:60], c.iloc[:60])
    assert s.iloc[60] == pytest.approx(a.iloc[60] - b[0, 1] / b[1, 1] * c.iloc[60], rel=1e-9)


def test_couverts_sans_regard_vers_le_futur():
    a, c = _rendements()
    s = mod.rendements_couverts(a, c, 60)
    a2 = a.copy()
    a2.iloc[100:] = a2.iloc[100:] * 5
    s2 = mod.rendements_couverts(a2, c, 60)
    assert s.iloc[60:100].tolist() == pytest.approx(s2.iloc[60:100].tolist(), rel=1e-12), (
        "une valeur future a modifié un rendement couvert passé")


def test_couverts_reduit_la_variance():
    a, c = _rendements()
    s = mod.rendements_couverts(a, c, 60).dropna()
    assert s.var() < 0.5 * a.iloc[60:].var()


# --- Arbitrage -----------------------------------------------------------------------------------
def test_demi_vie_exacte():
    assert mod.estimate_half_life(pd.Series(0.5 ** np.arange(30))) == pytest.approx(1.0, rel=1e-6)


def test_demi_vie_infinie():
    assert mod.estimate_half_life(pd.Series(1.01 ** np.arange(40))) == float("inf")


def test_demi_vie_nulle():
    assert mod.estimate_half_life(pd.Series([1.0, -1.0] * 15)) == 0.0


def test_zscore_serie_constante():
    assert mod.CointegrationArbBot().zscore(pd.Series([0.01] * 5)) == 0.0


def test_signal_bornes_inclusives():
    bot = mod.CointegrationArbBot(entry_z=2.0, commission_pct=0.001, min_edge_multiple=2.0)
    assert bot.signal(0.004, 2.0) == "sell_etf_buy_basket"


# --- OA : le tri doit être stable ----------------------------------------------------------------
def test_clean_prices_tri_stable_sur_un_grand_releve():
    n = 400
    temps = pd.date_range("2026-01-01", periods=n, freq="min")
    brut = pd.DataFrame({"timestamp": list(temps[::-1]) + list(temps[::-1]), "A": [1.0] * n + [2.0] * n})
    sortie = mod.clean_prices(brut)
    assert len(sortie) == n and (sortie["A"] == 1.0).all(), "la première ligne du fichier doit être conservée"


def test_clean_prices_horodatages_convertis():
    sortie = mod.clean_prices(pd.DataFrame({"timestamp": ["2026-01-01 09:40:00"], "A": [1.0]}))
    assert pd.api.types.is_datetime64_any_dtype(sortie["timestamp"])
