"""Valeurs de la séance MM 2 : équivalent certain, prix d'indifférence, demi-écart optimal, Avellaneda-Stoikov."""
import math

import numpy as np
import sympy as sp

from mm_commun import correction, ecrire

from amplify_quant_trading import MarketMakerBase, Quote, generate_trading_day, simulate_market_making

C = correction("mm_2.py")
V, T = {}, []

# Équivalent certain d'une richesse normale : symbolique et Monte-Carlo
g, m, v, x = sp.symbols("gamma mu v x", positive=True)
densite = sp.exp(-(x - m) ** 2 / (2 * v)) / sp.sqrt(2 * sp.pi * v)
assert sp.simplify(sp.integrate(sp.exp(-g * x) * densite, (x, -sp.oo, sp.oo)).simplify() / sp.exp(-g * m + g**2 * v / 2)) == 1
rng = np.random.default_rng(202)
gam, q, s0, sig, tau = 0.1, 5.0, 100.0, 30.0, 1 / 252
w = q * rng.normal(s0, sig * math.sqrt(tau), 5_000_000)
u = np.exp(-gam * w)
ce_mc = -np.log(u.mean()) / gam
se = (u.std(ddof=1) / np.sqrt(u.size)) / (gam * u.mean())
ce_ex = q * s0 - gam / 2 * q**2 * sig**2 * tau
assert abs(ce_mc - ce_ex) < 4 * se
V.update(CE_ex=round(ce_ex, 4), CE_mc=round(float(ce_mc), 4), CE_se=round(float(se), 4))
T.append(f"équivalent certain exact {ce_ex:.4f} ; Monte-Carlo {ce_mc:.4f} ; erreur standard {se:.4f}")

# Exemple numérique complet
k = 1.5
risque = gam * sig**2 * tau
liquidite = 2 / gam * math.log(1 + gam / k)
r = s0 - q * risque
ecart = risque + liquidite
ra, rv, r2 = C["prix_indifference"](s0, q, gam, sig, tau)
assert abs(r - r2) < 1e-12 and abs(rv - ra - risque) < 1e-12
V.update(EX_sig2tau=round(sig**2 * tau, 4), EX_risque=round(risque, 4), EX_decal=round(q * risque, 4), EX_r=round(r, 4), EX_liq=round(liquidite, 4),
         EX_ecart=round(ecart, 4), EX_bid=round(r - ecart / 2, 2), EX_ask=round(r + ecart / 2, 2), EX_ra=round(ra, 4), EX_rv=round(rv, 4))
V.update(ER_sans252=round(q * gam * sig**2 * 1.0, 1), ER_sans_euros=round(q * gam * 0.3**2 * tau, 7), EX_lim=round(2 / k, 4))

# Demi-écart optimal : maximum numérique contre formule
d = np.linspace(1e-6, 10, 2_000_001)
obj = np.exp(-k * d) * (1 - np.exp(-gam * d))
d_num = float(d[np.argmax(obj)])
d_ex = math.log(1 + gam / k) / gam
assert abs(d_num - d_ex) < 1e-4
V.update(D_ex=round(d_ex, 4), D_num=round(d_num, 4), D_neutre=round(1 / k, 4))
x_, g_, k_ = sp.symbols("delta gamma k", positive=True)
foc = sp.diff(sp.exp(-k_ * x_) * (1 - sp.exp(-g_ * x_)), x_).subs(x_, sp.log(1 + g_ / k_) / g_)
assert sp.simplify(foc) == 0

# Lien entre k et le moteur : fill_decay relatif, k en inverse d'euros
V.update(K_moteur=140 / 100)


class EcartFixe(MarketMakerBase):
    def quote(self, state):
        return Quote(bid=round(state.ref_price * 0.985, 2), ask=round(state.ref_price * 1.015, 2))


rapports = []
for g2 in range(20):
    j = generate_trading_day(n_ticks=5000, seed=3000 + g2)
    arg = dict(dt=j.attrs["dt"], seed=4000 + g2, max_inventory=60)
    a = simulate_market_making(C["AvellanedaStoikovMarketMaker"](), j["price"].to_numpy(), **arg)
    b = simulate_market_making(EcartFixe(), j["price"].to_numpy(), **arg)
    rapports.append(a["inventory"].std() / b["inventory"].std())
V.update(B_rapport=round(float(np.mean(rapports)), 3), B_rapport_se=round(float(np.std(rapports, ddof=1) / np.sqrt(len(rapports))), 3), B_n=len(rapports))
T.append(f"rapport des écarts-types d'inventaire {V['B_rapport']} (erreur standard {V['B_rapport_se']}) sur {len(rapports)} journées")
ecrire("seance_mm_2", V, T)
