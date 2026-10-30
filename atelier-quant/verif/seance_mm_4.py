"""Valeurs de la séance MM 4 : z-score borné, demi-vie, biais de l'estimateur, coûts, stratégie du test."""
import numpy as np
import pandas as pd

from mm_commun import correction, ecrire

from amplify_quant_trading import generate_etf_basket

C = correction("mm_4.py")
V, T = {}, []

# z-score : exemple et borne
z = C["CointegrationArbBot"]().zscore(pd.Series([0.0, 0.0, 0.0, 0.0, 0.02]))
assert np.isclose(z, 2.0)
V.update(Z_ex=float(round(z, 6)), Z_m=0.004, Z_s=0.008, Z_borne_61=round(float(np.sqrt(60)), 4))
rng = np.random.default_rng(404)
pire = max(abs(C["CointegrationArbBot"]().zscore(pd.Series(rng.normal(size=5)))) for _ in range(2000))
assert pire <= 2 + 1e-9

# Demi-vie : OU et discret
theta, eta, dt = 4.0, 0.004, 1 / 252
phi = 1 - theta * dt
hl_ou = np.log(2) / theta * 252
hl_ar = np.log(2) / -np.log(phi)
V.update(HL_theta=theta, HL_phi=round(phi, 5), HL_ou=round(float(hl_ou), 2), HL_ar=round(float(hl_ar), 2))
sd = eta / np.sqrt(2 * theta)
sd_ar = eta * np.sqrt(dt) / np.sqrt(1 - phi**2)
V.update(OU_sd_pb=round(float(sd * 1e4), 2), AR_sd_pb=round(float(sd_ar * 1e4), 2))

# Biais de l'estimateur AR(1) : exact approximatif de Kendall -(1 + 3 phi)/n, Monte-Carlo
rng = np.random.default_rng(405)
for n in (252, 3000):
    k = 20_000 if n == 252 else 2_000
    e = rng.normal(size=(k, n))
    x = np.zeros((k, n))
    x[:, 0] = e[:, 0] / np.sqrt(1 - phi**2)
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    a, b = x[:, :-1], x[:, 1:]
    ac, bc = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
    ph = (ac * bc).sum(axis=1) / (ac**2).sum(axis=1)
    hl = np.where(ph < 1, np.log(2) / -np.log(np.clip(ph, 1e-9, 1 - 1e-12)), np.inf)
    V[f"BI_phi_{n}"] = round(float(ph.mean()), 5)
    V[f"BI_se_{n}"] = round(float(ph.std(ddof=1) / np.sqrt(k)), 5)
    V[f"BI_hl_{n}"] = round(float(np.median(hl)), 1)
    V[f"BI_k_{n}"] = round(float(phi - (1 + 3 * phi) / n), 5)
    T.append(f"n = {n} : phi moyen {ph.mean():.5f} ; demi-vie médiane {np.median(hl):.1f}")

# Coûts
c, m = 0.0002, 4.0
V.update(CO_seuil_pb=round(m * 2 * c * 1e4, 2), CO_ar_pb=round(4 * c * 1e4, 2), CO_c_pb=c * 1e4, CO_m=m, CO_sigmas=round(float(m * 2 * c / sd_ar), 2))

# Stratégie de la suite de tests, sur 3000 jours
b_, _ = generate_etf_basket(n_days=3000, n_components=4, seed=11)
mis = (b_["ETF"] / b_["synthetic_true"] - 1).to_numpy()
hl_est = C["estimate_half_life"](pd.Series(mis))
bot = C["CointegrationArbBot"](entry_z=2.0, commission_pct=c, min_edge_multiple=m)
dans, entree, debut, total, n_tr = False, 0.0, 0, 0.0, 0
for t in range(60, len(mis)):
    z_t = bot.zscore(pd.Series(mis[t - 60:t + 1]))
    if not dans:
        if bot.signal(mis[t], z_t) != "no_trade":
            dans, entree, debut, n_tr = True, abs(mis[t]), t, n_tr + 1
    elif abs(mis[t]) <= entree * 0.3 or t - debut >= 40:
        total += entree - abs(mis[t]) - 4 * c
        dans = False
assert n_tr >= 3 and total > 0
V.update(ST_n=n_tr, ST_total_pb=round(float(total * 1e4), 1), ST_hl=round(float(hl_est), 1), ST_sd_pb=round(float(mis.std() * 1e4), 2))
T.append(f"{n_tr} opérations ; gain cumulé {total * 1e4:.1f} pb ; demi-vie estimée {hl_est:.1f}")
ecrire("seance_mm_4", V, T)
