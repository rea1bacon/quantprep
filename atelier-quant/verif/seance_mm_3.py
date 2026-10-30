"""Valeurs de la séance MM 3 : couverture, erreur d'estimation du bêta, efficacité, défaut de ddof."""
import numpy as np

from mm_commun import correction, ecrire

from amplify_quant_trading import generate_stock_etf_pair, hedge_effectiveness

C = correction("mm_3.py")
V, T = {}, []

# Exemple de couverture
E, b, choc = 800_000, 0.72, 0.01
V.update(H_E=E, H_b=b, H_not=round(E * b), H_perte=round(E * b * choc), H_pct=round(b * choc * 100, 2))

# Efficacité théorique et mesurée (hors échantillon, fenêtre de 60 jours)
beta, sm, si = 0.9, 0.18, 0.12
r2 = beta**2 * sm**2 / (beta**2 * sm**2 + si**2)
p = generate_stock_etf_pair(n_days=1500, beta=beta, idio_vol=si, seed=5)
r = p[["etf", "stock"]].pct_change().dropna()
c = C["rendements_couverts"](r["stock"], r["etf"], 60)
gar = c.notna().to_numpy()
eff = hedge_effectiveness(c.to_numpy()[gar], r["stock"].to_numpy()[gar])
V.update(R2_th=round(r2, 4), R2_mes=round(float(eff), 4), R2_b=beta, R2_sm=sm, R2_si=si)
assert abs(eff - r2) < 0.05
T.append(f"efficacité théorique {r2:.4f} ; mesurée hors échantillon {eff:.4f}")

# Erreur d'estimation : var(beta estimé) = si²/(sm²(n-3)), exact pour des rendements normaux ; Monte-Carlo
rng = np.random.default_rng(303)
dt = 1 / 252
for n in (10, 60, 250):
    x = rng.normal(0, sm * np.sqrt(dt), (200_000, n))
    y = beta * x + rng.normal(0, si * np.sqrt(dt), (200_000, n))
    xc = x - x.mean(axis=1, keepdims=True)
    bh = (xc * y).sum(axis=1) / (xc**2).sum(axis=1)
    ex = si / (sm * np.sqrt(n - 3))
    mc = bh.std(ddof=1)
    se = mc / np.sqrt(2 * (bh.size - 1))
    assert abs(mc - ex) < 5 * se + 0.01 * ex, (n, mc, ex)
    V[f"SE_ex_{n}"] = round(float(ex), 4)
    V[f"SE_mc_{n}"] = round(float(mc), 4)
    T.append(f"n = {n} : écart-type du bêta estimé exact {ex:.4f}, Monte-Carlo {mc:.4f}")

# Défaut d'origine : numérateur en ddof=1, dénominateur en ddof=0
x = r["etf"].to_numpy()[:60]
y = r["stock"].to_numpy()[:60]
bon = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
faux = np.cov(y, x)[0, 1] / np.var(x)
assert np.isclose(faux / bon, 60 / 59)
assert np.isclose(C["RollingHedger"]().hedge_ratio(y, x), np.polyfit(x, y, 1)[0])
V.update(DD_rapport=round(60 / 59, 4), DD_bon=round(float(bon), 4), DD_faux=round(float(faux), 4))

# Bande de non-intervention : les trois cas de l'énoncé
h = C["RollingHedger"](no_trade_band_pct=0.03)
assert h.rebalance(100_000, -69_000, 0.70, 50.0) is None
t1 = h.rebalance(100_000, 0.0, 0.70, 50.0)
t2 = h.rebalance(100_000, -95_000, 0.70, 25.0)
V.update(BA_v1=t1.trade_volume, BA_v2=t2.trade_volume)
assert t1.action == "sell" and t2.action == "buy"
ecrire("seance_mm_3", V, T)
