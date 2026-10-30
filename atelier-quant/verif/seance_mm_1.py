"""Valeurs de la séance MM 1 : Glosten-Milgrom (exact et Monte-Carlo), écart et volatilité, moteur de simulation."""
from fractions import Fraction as F

import numpy as np

from mm_commun import correction, ecrire

from amplify_quant_trading import generate_trading_day, simulate_market_making

C = correction("mm_1.py")
V, T = {}, []

# Exemple de Glosten-Milgrom, en fractions exactes
vb, vh, p, a = F(98), F(102), F(1, 2), F(1, 4)
pa_h, pa_b = a + (1 - a) / 2, (1 - a) / 2
ask = (p * pa_h * vh + (1 - p) * pa_b * vb) / (p * pa_h + (1 - p) * pa_b)
pv_h, pv_b = (1 - a) / 2, a + (1 - a) / 2
bid = (p * pv_h * vh + (1 - p) * pv_b * vb) / (p * pv_h + (1 - p) * pv_b)
assert (ask, bid) == (F(201, 2), F(199, 2)) and ask - bid == a * (vh - vb)
assert np.allclose(C["glosten_milgrom"](98.0, 102.0, 0.5, 0.25), (float(bid), float(ask)))
p_inf_achat = p * a / (p * pa_h + (1 - p) * pa_b)
gain_noninforme, perte_informe = ask - 100, ask - vh
assert (1 - p_inf_achat) * gain_noninforme + p_inf_achat * perte_informe == 0
V.update(GM_vb=98, GM_vh=102, GM_a=0.25, GM_ask=float(ask), GM_bid=float(bid), GM_ecart=float(ask - bid), GM_pa_h=float(pa_h), GM_pa_b=float(pa_b),
         GM_p_inf=float(p_inf_achat), GM_p_noninf=float(1 - p_inf_achat), GM_gain=float(gain_noninforme), GM_perte=float(-perte_informe))

# Monte-Carlo : gain moyen du teneur sur les achats, au prix ask exact
rng = np.random.default_rng(101)
n = 2_000_000
haut = rng.random(n) < 0.5
informe = rng.random(n) < 0.25
achat = np.where(informe, haut, rng.random(n) < 0.5)
valeur = np.where(haut, 102.0, 98.0)
e_achat = valeur[achat].mean()
se = valeur[achat].std(ddof=1) / np.sqrt(achat.sum())
assert abs(e_achat - 100.5) < 4 * se
V.update(GM_mc=round(float(e_achat), 4), GM_mc_se=round(float(se), 4))
T.append(f"E[V | achat] exact 100.5 ; Monte-Carlo {e_achat:.4f} ; erreur standard {se:.4f}")

# Cas asymétrique de l'énoncé
b, a2 = C["glosten_milgrom"](0.0, 1.0, 0.7, 0.5)
assert np.isclose(a2, 0.525 / 0.6) and np.isclose(b, 0.175 / 0.4)

# Règle d'écart de l'exercice, jours calmes et de tension
base, sens = 300.0, 800.0
V.update(EC_calme=base + sens * 0.15, EC_tension=base + sens * 0.55, EC_ajout=sens * 0.20)

# Probabilité d'exécution du moteur et demi-écart qui maximise le gain espéré par pas
b0, d = 0.35, 140.0
h_opt = 1 / d
V.update(FD=d, FB=b0, F_hopt_bps=round(h_opt * 1e4, 1), F_ecart_opt=round(2 * h_opt * 1e4, 1), F_p420=round(b0 * np.exp(-d * 0.021), 4), F_popt=round(b0 * np.exp(-1), 4))
h = np.linspace(1e-5, 0.05, 200_001)
assert abs(h[np.argmax(h * b0 * np.exp(-d * h))] - h_opt) < 1e-6

# Moteur : l'écart suit la volatilité estimée, avec retard ; les exécutions ne dépendent pas du mouvement suivant
jour = generate_trading_day(n_ticks=4000, seed=11)
m = C["VolAwareMarketMaker"]()
r = simulate_market_making(m, jour["price"].to_numpy(), dt=jour.attrs["dt"], seed=99, max_inventory=40)
deb, fin = jour.attrs["stress_window"]
tension = np.zeros(len(r), bool)
tension[deb:fin] = True
V.update(B_calme=round(float(r.loc[~tension, "spread_bps"].mean()), 1), B_tension=round(float(r.loc[tension, "spread_bps"].mean()), 1))
seuil = (r.loc[~tension, "spread_bps"].mean() + r.loc[tension, "spread_bps"].mean()) / 2
retard = int(np.argmax(r["spread_bps"].to_numpy()[deb:] > seuil))
V.update(B_retard=retard)

# Corrélation entre le sens net des exécutions et le rendement des 20 pas suivants, sur 30 journées
corr = []
for g in range(30):
    j = generate_trading_day(n_ticks=4000, seed=1000 + g)
    s = simulate_market_making(C["VolAwareMarketMaker"](), j["price"].to_numpy(), dt=j.attrs["dt"], seed=2000 + g, max_inventory=40)
    x = s["inventory"].diff().fillna(0).to_numpy()[:-20]
    y = np.log(j["price"].to_numpy()[20:] / j["price"].to_numpy()[:-20])
    corr.append(np.corrcoef(x, y)[0, 1])
c, se_c = float(np.mean(corr)), float(np.std(corr, ddof=1) / np.sqrt(len(corr)))
assert abs(c) < 3 * se_c + 1e-3
V.update(S_corr=round(c, 4), S_corr_se=round(se_c, 4), S_jours=len(corr))
T.append(f"corrélation exécutions et rendement suivant : {c:.4f} (erreur standard {se_c:.4f})")
T.append(f"écart moyen calme {V['B_calme']} pb ; tension {V['B_tension']} pb ; retard {retard} pas")
ecrire("seance_mm_1", V, T)
