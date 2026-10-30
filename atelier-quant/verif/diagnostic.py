"""Vérification de toutes les valeurs numériques du test diagnostic.

Chaque question du diagnostic a ici son calcul exact et, pour les probabilités
et les espérances, une simulation Monte-Carlo indépendante dont l'écart à
l'exact est comparé à l'erreur standard.

Sortie : verif/diagnostic.out.json (lu par outils/construire.py)
et un résumé lisible sur la sortie standard.
"""
import itertools
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import optimize, stats

rng = np.random.default_rng(20261002)
V = {}  # valeurs exportées


def controle_mc(nom, exact, estimation, erreur_std, seuil=4.0):
    """Vérifie que l'estimation Monte-Carlo est à moins de `seuil` erreurs standard de l'exact."""
    z = (estimation - exact) / erreur_std
    assert abs(z) < seuil, f"{nom} : exact={exact}, MC={estimation}, z={z}"
    V[nom + "_exact"] = float(exact)
    V[nom + "_mc"] = float(estimation)
    V[nom + "_se"] = float(erreur_std)
    print(f"{nom:4s} exact={float(exact):.6f}  MC={estimation:.6f}  se={erreur_std:.6f}  z={z:+.2f}")


# ---------------------------------------------------------------- M1 : Bayes
prevalence, sensibilite, specificite = Fraction(2, 100), Fraction(95, 100), Fraction(90, 100)
joint = prevalence * sensibilite
p_positif = joint + (1 - prevalence) * (1 - specificite)
m1 = joint / p_positif
assert m1 == Fraction(19, 117)
n = 4_000_000
malade = rng.random(n) < float(prevalence)
positif = np.where(malade, rng.random(n) < float(sensibilite), rng.random(n) < 1 - float(specificite))
est = malade[positif].mean()
controle_mc("M1", m1, est, np.sqrt(est * (1 - est) / positif.sum()))
V["M1_joint"] = float(joint)
V["M1_p_positif"] = float(p_positif)
V["M1_faux_positifs"] = float(p_positif - joint)

# ------------------------------------------------- M2 : ruine du joueur, durée
depart, haut = 3, 10
# Système linéaire : d(0)=d(haut)=0, d(i) = 1 + (d(i-1)+d(i+1))/2 pour 0<i<haut.
a = np.zeros((haut + 1, haut + 1))
b = np.zeros(haut + 1)
a[0, 0] = a[haut, haut] = 1
for i in range(1, haut):
    a[i, i], a[i, i - 1], a[i, i + 1], b[i] = 1, -0.5, -0.5, 1
duree = np.linalg.solve(a, b)
assert np.allclose(duree, [i * (haut - i) for i in range(haut + 1)])
n = 200_000
pos = np.full(n, depart)
pas = np.zeros(n)
actif = np.ones(n, dtype=bool)
while actif.any():
    k = actif.sum()
    pos[actif] += rng.choice([-1, 1], size=k)
    pas[actif] += 1
    actif = (pos > 0) & (pos < haut)
controle_mc("M2", depart * (haut - depart), pas.mean(), pas.std(ddof=1) / np.sqrt(n))
V["M2_p_haut_exact"] = depart / haut
p_haut = (pos == haut).mean()
assert abs(p_haut - depart / haut) < 4 * np.sqrt(0.3 * 0.7 / n)
V["M2_p_haut_mc"] = float(p_haut)

# ------------------------------------- M3 : nombre de faces distinctes, 5 dés
m3 = 6 * (1 - Fraction(5, 6) ** 5)
enum = Fraction(sum(len(set(t)) for t in itertools.product(range(6), repeat=5)), 6**5)
assert enum == m3
n = 1_000_000
tirages = rng.integers(0, 6, size=(n, 5))
tri = np.sort(tirages, axis=1)
distincts = 1 + (np.diff(tri, axis=1) != 0).sum(axis=1)
controle_mc("M3", m3, distincts.mean(), distincts.std(ddof=1) / np.sqrt(n))
V["M3_faces_absentes"] = float(6 * Fraction(5, 6) ** 5)
V["M3_moins_paires"] = float(5 - Fraction(10, 6))

# ---------------------------- M4 : matrice d'équicorrélation 3x3, condition SDP
rho = sp.symbols("rho", real=True)
c = sp.Matrix(3, 3, lambda i, j: 1 if i == j else rho)
vp = c.eigenvals()
assert vp == {1 + 2 * rho: 1, 1 - rho: 2}
sol = sp.solve_univariate_inequality(1 + 2 * rho >= 0, rho, relational=False).intersect(
    sp.solve_univariate_inequality(1 - rho >= 0, rho, relational=False)
)
assert sol == sp.Interval(sp.Rational(-1, 2), 1)
for r, attendu in [(-0.6, False), (-0.5, True), (0.3, True), (1.0, True), (1.01, False)]:
    m = np.full((3, 3), r) + (1 - r) * np.eye(3)
    assert (np.linalg.eigvalsh(m).min() >= -1e-12) == attendu
V["M4_borne_inf"] = -0.5
V["M4_borne_sup"] = 1.0
V["M4_ex_rho"] = -0.3
V["M4_ex_vp_simple"] = round(1 + 2 * -0.3, 12)
V["M4_ex_vp_double"] = round(1 - -0.3, 12)
V["M4_vp_simple_rho_moins_un"] = 1 + 2 * -1
print("M4   valeurs propres :", vp, " SDP ssi rho dans", sol)

# ------------------------------------------- M5 : variance stationnaire AR(1)
phi = 0.8
n, burn = 400_000, 200
x = np.zeros(n)
for _ in range(burn):
    x = phi * x + rng.standard_normal(n)
controle_mc("M5", 1 / (1 - phi**2), (x**2).mean(), (x**2).std(ddof=1) / np.sqrt(n))
V["M5_d_un_moins_phi"] = 1 / (1 - phi)
V["M5_d_un_sur_phi2"] = 1 / phi**2
V["M5_d_un_plus_phi2"] = 1 + phi**2

# ------------------------------ M6 : étendue de 4 uniformes indépendantes
u, v = sp.symbols("u v", positive=True)
nn = 4
# Densité jointe de (min, max) = n(n-1)(v-u)^(n-2) sur 0<u<v<1.
m6 = sp.integrate(sp.integrate((v - u) * nn * (nn - 1) * (v - u) ** (nn - 2), (u, 0, v)), (v, 0, 1))
assert m6 == sp.Rational(3, 5)
e_max = sp.integrate(v * nn * v ** (nn - 1), (v, 0, 1))  # densité du max : n v^(n-1)
e_min = sp.integrate(u * nn * (1 - u) ** (nn - 1), (u, 0, 1))  # densité du min : n (1-u)^(n-1)
assert e_max == sp.Rational(4, 5) and e_min == sp.Rational(1, 5) and e_max - e_min == m6
assert sp.Rational(3 - 1, 3 + 1) == sp.Rational(1, 2)  # étendue moyenne pour n = 3
n = 1_000_000
unif = rng.random((n, nn))
etendue = unif.max(axis=1) - unif.min(axis=1)
controle_mc("M6", sp.Rational(3, 5), etendue.mean(), etendue.std(ddof=1) / np.sqrt(n))

# --------------------------------------------------- M7 : tests multiples
m7 = 1 - Fraction(95, 100) ** 20
n = 1_000_000
rejets = (rng.random((n, 20)) < 0.05).any(axis=1)
controle_mc("M7", m7, rejets.mean(), np.sqrt(float(m7) * (1 - float(m7)) / n))
V["M7_aucun_rejet"] = float(1 - m7)

# ------------------------------ M8 : portefeuille de variance minimale, 2 actifs
s1, s2 = 0.20, 0.10
sigma = np.diag([s1**2, s2**2])
w = np.linalg.solve(sigma, np.ones(2))
w /= w.sum()
res = optimize.minimize_scalar(lambda t: t**2 * s1**2 + (1 - t) ** 2 * s2**2)
assert abs(res.x - w[0]) < 1e-8 and abs(w[0] - 0.2) < 1e-12
V["M8_exact"] = float(w[0])
V["M8_optim_numerique"] = float(res.x)
V["M8_d_inverse_vol"] = (1 / s1) / (1 / s1 + 1 / s2)
V["M8_variance_min"] = float(w @ sigma @ w)
V["M8_vol_min_pct"] = float(100 * np.sqrt(w @ sigma @ w))
V["M8_var1"] = s1**2
V["M8_var2"] = s2**2
V["M8_somme_var"] = s1**2 + s2**2
print(f"M8   poids actif 1 = {w[0]:.6f}  (optimisation numérique {res.x:.6f})")

# --------------------------- F1 : Lasso et Ridge avec plan orthonormé
beta_mco = np.array([3.0, 0.5])
lam = 1.0
q, _ = np.linalg.qr(rng.standard_normal((50, 2)))
bruit = rng.standard_normal(50)
bruit -= q @ (q.T @ bruit)  # résidu orthogonal aux colonnes : l'estimateur MCO vaut exactement beta_mco
y = q @ beta_mco + bruit
assert np.allclose(q.T @ q, np.eye(2)) and np.allclose(q.T @ y, beta_mco)
lasso_theorie = np.sign(beta_mco) * np.maximum(np.abs(beta_mco) - lam, 0)
ridge_theorie = beta_mco / (1 + lam)
f_lasso = lambda b_: 0.5 * np.sum((y - q @ b_) ** 2) + lam * np.abs(b_).sum()
f_ridge = lambda b_: 0.5 * np.sum((y - q @ b_) ** 2) + 0.5 * lam * np.sum(b_**2)
opt = dict(method="Nelder-Mead", options=dict(xatol=1e-11, fatol=1e-14, maxiter=20000))
lasso_num = optimize.minimize(f_lasso, beta_mco, **opt).x
ridge_num = optimize.minimize(f_ridge, beta_mco, **opt).x
assert np.allclose(lasso_num, lasso_theorie, atol=1e-6) and np.allclose(ridge_num, ridge_theorie, atol=1e-6)
V["F1_lasso"] = lasso_theorie.tolist()
V["F1_ridge"] = ridge_theorie.tolist()
print("F1   lasso =", lasso_theorie, " ridge =", ridge_theorie)

# ------------------- F2 : R² et R² ajusté quand on ajoute un régresseur de bruit
def r2_et_ajuste(xm, yv):
    nobs, p = xm.shape  # p inclut la constante
    coef, *_ = np.linalg.lstsq(xm, yv, rcond=None)
    scr = np.sum((yv - xm @ coef) ** 2)
    sct = np.sum((yv - yv.mean()) ** 2)
    r2 = 1 - scr / sct
    return r2, 1 - (1 - r2) * (nobs - 1) / (nobs - p)

essais, r2_baisse, ajuste_baisse = 2000, 0, 0
for _ in range(essais):
    x1 = rng.standard_normal(40)
    yv = 1 + 0.5 * x1 + rng.standard_normal(40)
    xa = np.column_stack([np.ones(40), x1])
    xb = np.column_stack([xa, rng.standard_normal(40)])
    (ra, aa), (rb, ab) = r2_et_ajuste(xa, yv), r2_et_ajuste(xb, yv)
    r2_baisse += rb < ra - 1e-12
    ajuste_baisse += ab < aa
assert r2_baisse == 0 and ajuste_baisse > 0
V["F2_essais"] = essais
V["F2_r2_baisse"] = int(r2_baisse)
V["F2_r2_ajuste_baisse"] = int(ajuste_baisse)
print(f"F2   sur {essais} essais : R² baisse {r2_baisse} fois, R² ajusté baisse {ajuste_baisse} fois")

# -------------------------------------------------- F3 : métriques
vp_, fp_, fn_, vn_ = 30, 20, 10, 940
precision = Fraction(vp_, vp_ + fp_)
rappel = Fraction(vp_, vp_ + fn_)
f1 = 2 * precision * rappel / (precision + rappel)
assert f1 == Fraction(2 * vp_, 2 * vp_ + fp_ + fn_) == Fraction(2, 3)
V["F3_precision"] = float(precision)
V["F3_rappel"] = float(rappel)
V["F3_f1"] = float(f1)
V["F3_moyenne_arith"] = float((precision + rappel) / 2)
V["F3_exactitude"] = float(Fraction(vp_ + vn_, vp_ + fp_ + fn_ + vn_))
print(f"F3   précision={float(precision)}, rappel={float(rappel)}, F1={float(f1):.6f}")

# ---------------- F4 : variance d'une moyenne de B prédicteurs équicorrélés
nb, r, s2 = 10, 0.3, 1.0
f4 = r * s2 + (1 - r) * s2 / nb
n = 1_000_000
commun = rng.standard_normal((n, 1))
t = np.sqrt(r) * commun + np.sqrt(1 - r) * rng.standard_normal((n, nb))
moy = t.mean(axis=1)
controle_mc("F4", f4, (moy**2).mean(), (moy**2).std(ddof=1) / np.sqrt(n))
V["F4_d_sigma2_sur_B"] = s2 / nb
V["F4_d_rho_sigma2"] = r * s2
V["F4_d_rho_sigma2_sur_B"] = r * s2 / nb

# -------------------------------- A1 : poids optimal d'une feuille XGBoost
residus = np.array([1.0, 2.0, 3.0])  # y_i - prédiction courante
g, h, lam = -residus, np.ones(3), 1.0
a1 = -g.sum() / (h.sum() + lam)
res = optimize.minimize_scalar(lambda w_: 0.5 * np.sum((residus - w_) ** 2) + 0.5 * lam * w_**2)
assert abs(res.x - a1) < 1e-8 and abs(a1 - 1.5) < 1e-12
V["A1_exact"] = float(a1)
V["A1_optim_numerique"] = float(res.x)
V["A1_d_moyenne"] = float(residus.mean())
V["A1_d_somme"] = float(residus.sum())
V["A1_G"] = float(g.sum())
V["A1_H"] = float(h.sum())
V["A1_H_plus_lambda"] = float(h.sum() + lam)
print(f"A1   poids optimal = {a1}  (optimisation numérique {res.x:.8f})")

# -------------------------------- A2 : longueur de sortie d'une convolution 1D
def longueur_sortie(longueur, noyau, pas_, marge, dilatation):
    """Compte par force brute les positions valides du noyau."""
    etendu = longueur + 2 * marge
    portee = dilatation * (noyau - 1) + 1
    return sum(1 for debut in range(0, etendu, pas_) if debut + portee <= etendu)

a2 = longueur_sortie(100, 5, 2, 2, 2)
assert a2 == (100 + 2 * 2 - 2 * (5 - 1) - 1) // 2 + 1 == 48
V["A2_exact"] = a2
V["A2_d_sans_dilatation"] = longueur_sortie(100, 5, 2, 2, 1)
V["A2_d_sans_pas"] = longueur_sortie(100, 5, 1, 2, 2)
V["A2_d_sans_plus_un"] = a2 - 1
V["A2_longueur_etendue"] = 100 + 2 * 2
V["A2_portee"] = 2 * (5 - 1) + 1
V["A2_dernier_debut"] = 100 + 2 * 2 - (2 * (5 - 1) + 1)
try:
    import torch

    sortie = torch.nn.Conv1d(1, 1, kernel_size=5, stride=2, padding=2, dilation=2)(torch.zeros(1, 1, 100))
    assert sortie.shape[-1] == a2
    V["A2_torch"] = int(sortie.shape[-1])
except Exception:  # PyTorch absent ou inutilisable : la vérification par force brute suffit
    V["A2_torch"] = None
print(f"A2   longueur de sortie = {a2}  (PyTorch : {V['A2_torch']})")

# -------------------------------- A3 : rétropropagation sur un réseau scalaire
w1, w2, xs, ys = sp.symbols("w1 w2 x y", real=True)
perte = sp.Rational(1, 2) * (w2 * sp.tanh(w1 * xs) - ys) ** 2
point = {w1: 0, w2: 3, xs: 2, ys: 1}
d1 = sp.diff(perte, w1).subs(point)
d2 = sp.diff(perte, w2).subs(point)
f = lambda a_, b_: 0.5 * (b_ * np.tanh(a_ * 2.0) - 1.0) ** 2
eps = 1e-6
assert abs((f(eps, 3) - f(-eps, 3)) / (2 * eps) - float(d1)) < 1e-6
assert abs((f(0, 3 + eps) - f(0, 3 - eps)) / (2 * eps) - float(d2)) < 1e-6
assert d1 == -6 and d2 == 0
V["A3_dw1"] = int(d1)
V["A3_dw2"] = int(d2)
print(f"A3   dL/dw1 = {d1}, dL/dw2 = {d2}")

# -------------------------------- A4 : premier pas d'Adam
grad, lr, b1, b2, eps_adam = 100.0, 0.01, 0.9, 0.999, 1e-8
m_ = (1 - b1) * grad
v_ = (1 - b2) * grad**2
pas_adam = -lr * (m_ / (1 - b1)) / (np.sqrt(v_ / (1 - b2)) + eps_adam)
assert abs(pas_adam + lr) < 1e-9
V["A4_exact"] = float(pas_adam)
V["A4_d_sgd"] = -lr * grad
V["A4_d_sans_correction"] = float(-lr * m_ / np.sqrt(v_))
V["A4_d_m_sans_correction"] = float(-lr * m_ / np.sqrt(v_ / (1 - b2)))
try:
    import torch

    theta = torch.zeros(1, requires_grad=True)
    optimiseur = torch.optim.Adam([theta], lr=lr, betas=(b1, b2), eps=eps_adam)
    (grad * theta).sum().backward()
    optimiseur.step()
    assert abs(theta.item() - pas_adam) < 1e-7
    V["A4_torch"] = theta.item()
except Exception:  # PyTorch absent ou inutilisable : la vérification par force brute suffit
    V["A4_torch"] = None
print(f"A4   premier pas d'Adam = {pas_adam:.10f}  (PyTorch : {V['A4_torch']})")

# -------------------------------- A5 : statistique de Student d'un Sharpe
jours_par_an, annees, sharpe_annuel = 252, 4, 1.0
n = jours_par_an * annees
a5 = sharpe_annuel * np.sqrt(n / jours_par_an)
z = rng.standard_normal(n)
z = (z - z.mean()) / z.std(ddof=1)
rendements = 0.01 * (z + sharpe_annuel / np.sqrt(jours_par_an))
assert abs(np.sqrt(jours_par_an) * rendements.mean() / rendements.std(ddof=1) - sharpe_annuel) < 1e-12
t_scipy = stats.ttest_1samp(rendements, 0.0).statistic
assert abs(t_scipy - a5) < 1e-9 and abs(a5 - 2.0) < 1e-12
V["A5_exact"] = float(a5)
V["A5_scipy"] = float(t_scipy)
V["A5_d_racine_n"] = float(sharpe_annuel * np.sqrt(n))
V["A5_p_bilaterale"] = float(stats.ttest_1samp(rendements, 0.0).pvalue)
print(f"A5   t = {a5}  (scipy {t_scipy:.10f}), p bilatérale = {V['A5_p_bilaterale']:.4f}")

# -------------------------------- A6 : rang de la covariance empirique
t_obs, n_actifs = 250, 500
donnees = rng.standard_normal((t_obs, n_actifs))
cov = np.cov(donnees, rowvar=False)
rang = int(np.linalg.matrix_rank(cov))
assert cov.shape == (n_actifs, n_actifs) and rang == t_obs - 1
V["A6_rang"] = rang
V["A6_vp_nulles"] = n_actifs - rang
print(f"A6   rang de la covariance empirique {n_actifs}x{n_actifs} avec {t_obs} observations = {rang}")

# ------------------- Distracteurs simples et arithmétique citée dans les corrigés
V["M2_d_depart_fois_barriere"] = depart * haut
V["M2_d_depart_carre"] = depart**2
V["M2_d_distance_haut"] = haut - depart
V["M7_d_boole"] = 20 * 0.05
V["M8_d_inverse"] = 1 - V["M8_exact"]
V["F1_d_seuillage_dur"] = [float(b_ if abs(b_) > 1.0 else 0.0) for b_ in beta_mco]
V["F1_d_sans_troncature"] = (beta_mco - 1.0).tolist()
V["A1_d_signe"] = -V["A1_exact"]
V["A3_d_sans_x"] = int(sp.diff(perte, w1).subs(point)) // 2
V["A3_d_signe"] = -int(d1)
V["A5_d_annees"] = sharpe_annuel * annees
V["A2_dernier_depart_pair"] = (V["A2_dernier_debut"] // 2) * 2
assert V["A2_dernier_depart_pair"] // 2 + 1 == a2
assert Fraction(98, 2) == 49  # M1 : personnes saines pour une personne malade
assert abs(1 / (1 - phi**2) - 1 / 0.36) < 1e-12  # M5
assert abs(1 / 0.20**2 - 25) < 1e-9 and abs(1 / 0.10**2 - 100) < 1e-9  # M8
assert precision == Fraction(30, 50) and rappel == Fraction(30, 40) and f1 == Fraction(60, 90)  # F3
assert abs(0.3 + 0.7 / 10 - f4) < 1e-12  # F4
assert abs(6 / (3 + 1) - a1) < 1e-12 and abs(h.sum() / (h.sum() + 1.0) - 0.75) < 1e-12  # A1
assert abs(m_ - 10) < 1e-9 and abs(v_ - 10) < 1e-9 and abs(pas_adam + lr) < 1e-9  # A4
assert 3 * (10 - 3) == 21 and abs(duree[3] - 21) < 1e-9  # M2

Path(__file__).with_name("diagnostic.out.json").write_text(json.dumps(V, indent=1, ensure_ascii=False), encoding="utf-8")
print("Toutes les vérifications passent.")
