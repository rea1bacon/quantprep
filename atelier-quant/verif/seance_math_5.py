"""Vérification de tous les nombres de la séance de mathématiques 5 (leçon L-MATH-5 et questions M5-01 à M5-12).

Thèmes : forme d'une loi (moyenne, médiane, mode, asymétrie, aplatissement), covariance et corrélation, cote Z,
tests d'adéquation (khi-deux, Kolmogorov-Smirnov, Anderson-Darling), densités conjointes, estimation bayésienne
de l'intensité d'une loi de Poisson, bâton coupé par des points uniformes, problème des deux enfants.

Chaque probabilité ou espérance a deux vérifications indépendantes :
  1. un calcul exact (fractions, intégrale ou système symbolique avec sympy, énumération complète) ;
  2. une simulation Monte-Carlo qui suit les règles de l'énoncé, dont l'écart à l'exact doit rester inférieur
     à 4 erreurs standard.
Les énoncés généraux de la leçon sont aussi contrôlés (formules symboliques, identités, invariances), ainsi que
les distracteurs et l'arithmétique citée dans les corrections.
Les niveaux de tests « constatés par simulation » (clés terminées par _sim) n'ont pas de valeur exacte : ils
sont exportés comme résultats d'expérience, et la leçon les présente ainsi.

Graine fixe : la sortie est identique à chaque exécution.
Sorties : verif/seance_math_5.out.json (lu par outils/construire.py) et verif/seance_math_5.out.txt (ce résumé).
"""
import json
import math
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate as sci_int
from scipy import stats

rng = np.random.default_rng(20261009)
V = {}
LIGNES = []


def dire(texte):
    print(texte)
    LIGNES.append(texte)


def fraction(nom, f):
    """Exporte une fraction : valeur décimale, numérateur et dénominateur (forme irréductible)."""
    if isinstance(f, sp.Basic):
        f = sp.Rational(f)
        f = Fraction(int(f.p), int(f.q))
    f = Fraction(f)
    V[nom] = float(f)
    V[nom + "_num"] = f.numerator
    V[nom + "_den"] = f.denominator


def controle_mc(nom, exact, estimation, erreur_std, seuil=4.0):
    """Vérifie que l'estimation Monte-Carlo est à moins de `seuil` erreurs standard de la valeur exacte."""
    z = (float(estimation) - float(exact)) / float(erreur_std)
    assert abs(z) < seuil, f"{nom} : exact={float(exact)}, MC={estimation}, z={z}"
    V[nom + "_mc"] = float(estimation)
    V[nom + "_se"] = float(erreur_std)
    dire(f"{nom:26s} exact={float(exact):.6f}  MC={float(estimation):.6f}  se={float(erreur_std):.6f}  z={z:+.2f}")


def mc_moyenne(nom, exact, echantillon):
    echantillon = np.asarray(echantillon, dtype=float)
    controle_mc(nom, exact, echantillon.mean(), echantillon.std(ddof=1) / np.sqrt(len(echantillon)))


def mc_proba(nom, exact, indicatrices):
    indicatrices = np.asarray(indicatrices, dtype=float)
    p = float(exact)
    controle_mc(nom, exact, indicatrices.mean(), np.sqrt(p * (1 - p) / len(indicatrices)))


def mc_variance(nom, exact, echantillon):
    """Variance empirique ; erreur standard estimée par la méthode delta sur l'échantillon."""
    x = np.asarray(echantillon, dtype=float)
    c = x - x.mean()
    v = (c**2).mean()
    controle_mc(nom, exact, v, np.sqrt(((c**2 - v) ** 2).mean() / len(x)))


def mc_covariance(nom, exact, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    prod = (x - x.mean()) * (y - y.mean())
    controle_mc(nom, exact, prod.mean(), prod.std(ddof=1) / np.sqrt(len(x)))


def R(num, den=1):
    return sp.Rational(num, den)


def forme_discrete(valeurs, probas):
    """Moyenne, variance, moments centrés d'ordre 3 et 4, médiane (plus petite) et mode d'une loi discrète finie."""
    valeurs = [Fraction(v) for v in valeurs]
    probas = [Fraction(p) for p in probas]
    assert sum(probas) == 1 and valeurs == sorted(valeurs)
    mu = sum(v * p for v, p in zip(valeurs, probas))
    c = lambda r: sum((v - mu) ** r * p for v, p in zip(valeurs, probas))
    cumul, mediane = Fraction(0), None
    for v, p in zip(valeurs, probas):
        cumul += p
        if cumul >= Fraction(1, 2) and mediane is None:
            mediane = v
    mode = valeurs[probas.index(max(probas))]
    return mu, c(2), c(3), c(4), mediane, mode


# =====================================================================================
# LEÇON, PARTIE 1 : forme d'une loi
# =====================================================================================
x = sp.symbols("x", positive=True)
xr = sp.symbols("x_r", real=True)


def forme(densite, inf, sup, var):
    """Moyenne, variance, coefficient d'asymétrie, kurtosis d'une loi à densité (calcul symbolique)."""
    m1 = sp.integrate(var * densite, (var, inf, sup))
    c = lambda r: sp.simplify(sp.integrate((var - m1) ** r * densite, (var, inf, sup)))
    v = c(2)
    return sp.simplify(m1), v, sp.simplify(c(3) / v ** R(3, 2)), sp.simplify(c(4) / v**2)


# 1a. Loi exponentielle de paramètre lambda : asymétrie 2, kurtosis 9, médiane ln 2 / lambda.
lam = sp.symbols("lambda", positive=True)
m_e, v_e, sk_e, ku_e = forme(lam * sp.exp(-lam * x), 0, sp.oo, x)
assert m_e == 1 / lam and v_e == 1 / lam**2 and sk_e == 2 and ku_e == 9
assert sp.simplify(sp.integrate(lam * sp.exp(-lam * x), (x, 0, sp.log(2) / lam)) - R(1, 2)) == 0
V["La_exp_skew"] = 2
V["La_exp_kurt"] = 9
V["La_exp_exces"] = 6
V["La_exp_moy"] = 2
V["La_exp_med"] = 2 * math.log(2)
V["La_exp_ecart"] = 2 - 2 * math.log(2)
V["La_exp_sigma"] = 2
assert V["La_exp_ecart"] <= V["La_exp_sigma"]
xe = rng.exponential(2.0, 1_000_000)
mc_proba("La_exp_med_p", 0.5, xe <= V["La_exp_med"])
mc_moyenne("La_exp_c3", 2 * 2**3, (xe - 2) ** 3)  # asymétrie 2 fois sigma^3
mc_moyenne("La_exp_c4", 9 * 2**4, (xe - 2) ** 4)  # kurtosis 9 fois sigma^4

# 1b. Loi log-normale X = exp(a + b Z).
b_s = sp.symbols("b", positive=True)
w_s = sp.exp(b_s**2)
mom = {r: sp.exp(R(r**2, 2) * b_s**2) for r in (1, 2, 3)}  # E[X^r] pour a = 0
for r in (1, 2, 3):  # contrôle de E[exp(r b Z)] = exp(r^2 b^2 / 2) par intégration numérique, b = 0,5
    val, _ = sci_int.quad(lambda z: math.exp(r * 0.5 * z) * math.exp(-z * z / 2) / math.sqrt(2 * math.pi), -40, 40)
    assert abs(val - float(mom[r].subs(b_s, R(1, 2)))) < 1e-10
var_ln = mom[2] - mom[1] ** 2
c3_ln = mom[3] - 3 * mom[1] * mom[2] + 2 * mom[1] ** 3
for b_num in [R(3, 10), R(1, 2), 1, R(17, 10)]:
    ecart = (c3_ln / var_ln ** R(3, 2) - (w_s + 2) * sp.sqrt(w_s - 1)).subs(b_s, b_num)
    assert abs(float(ecart)) < 1e-9
assert sp.simplify(var_ln - w_s * (w_s - 1)) == 0
t_s, a_s = sp.symbols("t a", real=True)
assert sp.solve(sp.diff(-(t_s - a_s) ** 2 / (2 * b_s**2) - t_s, t_s), t_s) == [a_s - b_s**2]  # mode en ln x = a - b^2
b_l = 0.5
w_l = math.exp(b_l**2)
V["La_ln_mode"] = math.exp(-(b_l**2))
V["La_ln_med"] = 1
V["La_ln_moy"] = math.exp(b_l**2 / 2)
V["La_ln_sigma"] = math.sqrt(w_l * (w_l - 1))
V["La_ln_skew"] = (w_l + 2) * math.sqrt(w_l - 1)
assert V["La_ln_mode"] < V["La_ln_med"] < V["La_ln_moy"] and V["La_ln_moy"] - 1 <= V["La_ln_sigma"]
xl = np.exp(b_l * rng.normal(0, 1, 2_000_000))
mc_moyenne("La_ln_moy", V["La_ln_moy"], xl)
mc_proba("La_ln_med_p", 0.5, xl <= 1)
mc_moyenne("La_ln_c3", V["La_ln_skew"] * V["La_ln_sigma"] ** 3, (xl - V["La_ln_moy"]) ** 3)

# 1c. Loi gamma de forme a (paramètre d'échelle 1) : variance a, asymétrie 2/racine(a), excès 6/a.
a_g = sp.symbols("a", positive=True)
mr = lambda r: sp.gammasimp(sp.gamma(a_g + r) / sp.gamma(a_g))
mu_g = mr(1)
var_g = sp.expand(mr(2) - mu_g**2)
c3_g = sp.expand(mr(3) - 3 * mu_g * mr(2) + 2 * mu_g**3)
c4_g = sp.expand(mr(4) - 4 * mu_g * mr(3) + 6 * mu_g**2 * mr(2) - 3 * mu_g**4)
assert mu_g == a_g and var_g == a_g and c3_g == 2 * a_g and sp.expand(c4_g - 3 * a_g**2 - 6 * a_g) == 0
V["La_gam_F1"] = 1 - 2 / math.e  # P(X <= 1) pour a = 2
V["La_gam_F2"] = 1 - 3 / math.e**2  # P(X <= 2)
assert abs(V["La_gam_F1"] - stats.gamma.cdf(1, 2)) < 1e-12 and abs(V["La_gam_F2"] - stats.gamma.cdf(2, 2)) < 1e-12
assert V["La_gam_F1"] < 0.5 < V["La_gam_F2"]
med_g = float(stats.gamma.ppf(0.5, 2))
assert abs(1 - math.exp(-med_g) * (1 + med_g) - 0.5) < 1e-12
V["La_gam_med"] = med_g
V["La_gam_skew"] = 2 / math.sqrt(2)
V["La_gam_exces"] = 6 / 2
V["La_gam_grille_min"], V["La_gam_grille_max"] = 1, 60
for a_num in np.arange(1.0, 60.0001, 0.1):  # mode < médiane < moyenne sur une grille de formes
    assert a_num - 1 < stats.gamma.ppf(0.5, a_num) < a_num
xg = rng.gamma(2.0, 1.0, 1_000_000)
mc_proba("La_gam_med_p", 0.5, xg <= med_g)
mc_moyenne("La_gam_c3", 4, (xg - 2) ** 3)

# 1d. Somme de n variables indépendantes de même loi : asymétrie divisée par racine(n), excès divisé par n.
p_b, n_b = Fraction(3, 10), 10
q_b = 1 - p_b
vals = list(range(n_b + 1))
prob = [math.comb(n_b, k) * p_b**k * q_b ** (n_b - k) for k in vals]
mu_b, v_b, c3_b, c4_b, _, _ = forme_discrete(vals, prob)
_, v_1, c3_1, c4_1, _, _ = forme_discrete([0, 1], [q_b, p_b])
assert c3_b**2 / v_b**3 == (c3_1**2 / v_1**3) / n_b  # carré de l'asymétrie divisé par n
assert c4_b / v_b**2 - 3 == (c4_1 / v_1**2 - 3) / n_b  # excès divisé par n
V["La_somme_n"] = 25
V["La_somme_skew"] = 2 / math.sqrt(25)
V["La_somme_exces"] = 6 / 25
assert abs(V["La_somme_skew"] - float((c3_g / var_g ** R(3, 2)).subs(a_g, 25))) < 1e-12
assert abs(V["La_somme_exces"] - float((c4_g / var_g**2 - 3).subs(a_g, 25))) < 1e-12
s25 = rng.exponential(1.0, (400_000, 25)).sum(axis=1)
mc_moyenne("La_somme_c3", 2 * 25, (s25 - 25) ** 3)

# 1e. Contre-exemples discrets.
# Loi A : asymétrie nulle sans symétrie. Loi B : asymétrie strictement positive, moyenne < médiane = mode.
muA, vA, c3A, c4A, medA, modeA = forme_discrete([-2, 1, 3], [Fraction(4, 10), Fraction(5, 10), Fraction(1, 10)])
assert muA == 0 and c3A == 0 and medA == 1 and modeA == 1
fraction("La_A_var", vA)
muB, vB, c3B, c4B, medB, modeB = forme_discrete([-2, 1, 4], [Fraction(4, 10), Fraction(5, 10), Fraction(1, 10)])
assert medB == 1 and modeB == 1 and muB < medB and c3B > 0
V["La_B_moy"] = float(muB)
V["La_B_var"] = float(vB)
V["La_B_c3"] = float(c3B)
V["La_B_skew"] = float(c3B) / float(vB) ** 1.5
V["La_B_kurt"] = float(c4B / vB**2)
V["La_B_med"] = int(medB)
assert V["La_B_kurt"] < 3 and abs(float(muB) - 1) <= math.sqrt(float(vB))
xb = rng.choice([-2.0, 1.0, 4.0], p=[0.4, 0.5, 0.1], size=1_000_000)
mc_moyenne("La_B_moy", muB, xb)
mc_moyenne("La_B_c3", c3B, (xb - float(muB)) ** 3)
mc_proba("La_B_p_inf", Fraction(4, 10), xb < 1)  # P(X < 1) = 0,4 < 1/2 et P(X <= 1) = 0,9 : médiane 1

# 1f. Contre-exemple continu unimodal : loi de Weibull de forme 3,5, X = Y^(1/k) avec Y exponentielle de paramètre 1.
k_w = 3.5
g_w = lambda r: math.gamma(1 + r / k_w)  # E[X^r] = E[Y^(r/k)] = Gamma(1 + r/k)
for r in (1, 2, 3):
    val, _ = sci_int.quad(lambda y: y ** (r / k_w) * math.exp(-y), 0, np.inf)
    assert abs(val - g_w(r)) < 1e-9
moy_w = g_w(1)
var_w = g_w(2) - moy_w**2
c3_w = g_w(3) - 3 * moy_w * g_w(2) + 2 * moy_w**3
V["La_w_k"] = k_w
V["La_w_moy"] = moy_w
V["La_w_med"] = math.log(2) ** (1 / k_w)
V["La_w_mode"] = ((k_w - 1) / k_w) ** (1 / k_w)
V["La_w_skew"] = c3_w / var_w**1.5
assert V["La_w_skew"] > 0 and V["La_w_moy"] < V["La_w_med"] < V["La_w_mode"]
assert abs(V["La_w_skew"] - float(stats.weibull_min.stats(k_w, moments="s"))) < 1e-9
assert abs(V["La_w_med"] - V["La_w_moy"]) <= math.sqrt(var_w)
xw = rng.exponential(1.0, 4_000_000) ** (1 / k_w)
mc_moyenne("La_w_moy", moy_w, xw)
mc_proba("La_w_med_p", 0.5, xw <= V["La_w_med"])
mc_moyenne("La_w_c3", c3_w, (xw - moy_w) ** 3)

# 1g. Kurtosis de référence.
_, _, sk_u, ku_u = forme(sp.Integer(1), 0, 1, x)
assert sk_u == 0 and ku_u == R(9, 5)
V["La_unif_kurt"] = float(ku_u)
V["La_unif_exces"] = float(ku_u) - 3
phi_s = sp.exp(-xr**2 / 2) / sp.sqrt(2 * sp.pi)
assert sp.integrate(xr**4 * phi_s, (xr, -sp.oo, sp.oo)) == 3 and sp.integrate(xr**3 * phi_s, (xr, -sp.oo, sp.oo)) == 0
V["La_norm_kurt"] = 3
exces_t5 = float(stats.t.stats(5, moments="k"))
m2_t5, _ = sci_int.quad(lambda u: u**2 * stats.t.pdf(u, 5), -np.inf, np.inf)
m4_t5, _ = sci_int.quad(lambda u: u**4 * stats.t.pdf(u, 5), -np.inf, np.inf)
assert abs(m4_t5 / m2_t5**2 - 3 - exces_t5) < 1e-6 and abs(exces_t5 - 6) < 1e-9
V["La_t5_exces"] = 6
_, v_pm, _, c4_pm, _, _ = forme_discrete([-1, 1], [Fraction(1, 2), Fraction(1, 2)])
assert c4_pm / v_pm**2 == 1  # kurtosis minimal, atteint par la loi de +1 ou -1 équiprobables
dire("Partie 1 : formules d'asymétrie et de kurtosis vérifiées (exponentielle, log-normale, gamma, sommes, Weibull).")

# Point de contrôle « forme » : rien de garanti sur l'ordre, seulement |moyenne - médiane| <= écart-type (loi B).
V["Ca_ecart"] = float(abs(muB - medB))
V["Ca_sigma"] = math.sqrt(float(vB))

# =====================================================================================
# LEÇON, PARTIE 2 : covariance, corrélation, cote Z
# =====================================================================================
# 2a. Changement d'unité : rendements en décimal puis en pourcentage.
sx, sy, cxy = Fraction(1, 100), Fraction(2, 100), Fraction(12, 100000)
rho_xy = cxy / (sx * sy)
assert rho_xy == Fraction(6, 10)
V["Lb_rho"] = float(rho_xy)
V["Lb_cov_dec"] = float(cxy)
V["Lb_cov_pct"] = float(100 * 100 * cxy)
V["Lb_sx_pct"], V["Lb_sy_pct"] = float(100 * sx), float(100 * sy)
assert (100 * 100 * cxy) / (100 * sx * 100 * sy) == rho_xy
V["Lb_pente"] = float(cxy / sx**2)
assert cxy / sx**2 == rho_xy * sy / sx == Fraction(12, 10)
V["Lb_var_res"] = float(1 - rho_xy**2)  # part de variance non expliquée par la meilleure droite
cov_m = np.array([[float(sx**2), float(cxy)], [float(cxy), float(sy**2)]])
xy = rng.multivariate_normal([0.0005, 0.001], cov_m, size=400_000)
mc_covariance("Lb_cov_pct", V["Lb_cov_pct"], 100 * xy[:, 0], 100 * xy[:, 1])
r_dec = np.corrcoef(xy[:, 0], xy[:, 1])[0, 1]
r_pct = np.corrcoef(100 * xy[:, 0], 100 * xy[:, 1])[0, 1]
r_neg = np.corrcoef(-3 * xy[:, 0] + 7, 100 * xy[:, 1])[0, 1]
assert abs(r_dec - r_pct) < 1e-12 and abs(r_dec - 0.6) < 0.005 and abs(r_neg + r_dec) < 1e-12
V["Lb_rho_mc"] = float(r_dec)
pente_mc = np.polyfit(xy[:, 0], xy[:, 1], 1)[0]
assert abs(pente_mc - 1.2) < 0.01
# Formules générales : Cov(aX + b, cY + d) = ac Cov(X, Y) ; meilleure droite et variance résiduelle.
vx_s, vy_s, be_s = sp.symbols("v_x v_y beta", positive=True)
c_s = sp.symbols("c", real=True)
res = vy_s - 2 * be_s * c_s + be_s**2 * vx_s  # E[(Y~ - beta X~)^2] pour des variables centrées
b_opt = sp.solve(sp.diff(res, be_s), be_s)[0]
assert b_opt == c_s / vx_s and sp.simplify(res.subs(be_s, b_opt) - vy_s * (1 - c_s**2 / (vx_s * vy_s))) == 0
ech = rng.normal(0, 1, (2, 200_000))
u_, v_ = ech[0], 0.5 * ech[0] + ech[1] ** 2
for a_c, b_c, c_c, d_c in [(2.0, 1.0, 3.0, -4.0), (-1.8, 32.0, 0.001, 0.0), (100.0, 0.0, -100.0, 5.0)]:
    cov0 = np.cov(u_, v_)[0, 1]
    assert abs(np.cov(a_c * u_ + b_c, c_c * v_ + d_c)[0, 1] - a_c * c_c * cov0) < 1e-9 * max(1.0, abs(a_c * c_c))
    assert abs(np.corrcoef(a_c * u_ + b_c, c_c * v_ + d_c)[0, 1] - np.sign(a_c * c_c) * np.corrcoef(u_, v_)[0, 1]) < 1e-12

# 2b. Corrélation nulle sans indépendance : X uniforme sur {-1, 0, 1}, Y = X^2.
loi_x = {-1: Fraction(1, 3), 0: Fraction(1, 3), 1: Fraction(1, 3)}
e_x = sum(v * p for v, p in loi_x.items())
e_y = sum(v**2 * p for v, p in loi_x.items())
e_xy = sum(v**3 * p for v, p in loi_x.items())
assert e_xy - e_x * e_y == 0
fraction("Lb_dep_joint", loi_x[0])  # P(X = 0, Y = 0)
fraction("Lb_dep_prod", loi_x[0] * loi_x[0])  # P(X = 0) P(Y = 0)
fraction("Lb_dep_ey", e_y)
xd = rng.integers(-1, 2, 600_000)
mc_covariance("Lb_dep_cov", 0, xd, xd**2)
mc_proba("Lb_dep_joint_p", Fraction(1, 3), (xd == 0) & (xd**2 == 0))
# Deux variables normales non corrélées et dépendantes : Y = S X, S signe indépendant.
zn = rng.normal(0, 1, 600_000)
sg = rng.choice([-1.0, 1.0], size=600_000)
mc_covariance("Lb_signe_cov", 0, zn, sg * zn)
assert abs(np.corrcoef(zn**2, (sg * zn) ** 2)[0, 1] - 1) < 1e-12

# 2c. Pearson contre Spearman : Z normale centrée réduite et exp(Z).
th_g = sp.symbols("theta_g", real=True)
gen = sp.exp(th_g**2 / 2)  # E[exp(theta Z)] (séance de mathématiques 4) ; sa dérivée en theta est E[Z exp(theta Z)]
e_zez, e_ez, e_e2z = sp.diff(gen, th_g).subs(th_g, 1), gen.subs(th_g, 1), gen.subs(th_g, 2)
dens_n = lambda z: math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
for exact_s, fonction in [(e_zez, lambda z: z * math.exp(z)), (e_ez, math.exp), (e_e2z, lambda z: math.exp(2 * z))]:
    val, _ = sci_int.quad(lambda z: fonction(z) * dens_n(z), -40, 40)
    assert abs(val - float(exact_s)) < 1e-9
V["Lb_pearson_exp"] = 1 / math.sqrt(math.e - 1)
assert abs(V["Lb_pearson_exp"] - float(e_zez / sp.sqrt(e_e2z - e_ez**2))) < 1e-12
zz = rng.normal(0, 1, 2_000_000)
mc_covariance("Lb_exp_cov", math.exp(0.5), zz, np.exp(zz))
mc_variance("Lb_exp_var", math.e**2 - math.e, np.exp(zz))
V["Lb_pearson_exp_mc"] = float(np.corrcoef(zz, np.exp(zz))[0, 1])
assert abs(V["Lb_pearson_exp_mc"] - V["Lb_pearson_exp"]) < 0.02
assert abs(stats.spearmanr(zz[:20_000], np.exp(zz[:20_000]))[0] - 1) < 1e-12  # rangs identiques
V["Lb_spearman_exp"] = 1

# 2d. Cote Z.
V["Lb_z1"] = float(Fraction(14 - 11, 2))
V["Lb_z2"] = float(Fraction(620 - 500, 100))
assert V["Lb_z1"] == 1.5 and V["Lb_z2"] == 1.2
V["Lb_phi_15"] = float(stats.norm.cdf(1.5))
V["Lb_phi_12"] = float(stats.norm.cdf(1.2))
V["Lb_exp_15"] = 1 - math.exp(-2.5)  # exponentielle de moyenne 1 : cote Z de 1,5 pour la valeur 2,5
V["Lb_exp_x"] = 2.5
V["Lb_phi_14"] = float(stats.norm.cdf(1.4))
assert V["Lb_phi_14"] > V["Lb_exp_15"] and V["Lb_phi_15"] > V["Lb_exp_15"]
mc_proba("Lb_phi_15", V["Lb_phi_15"], rng.normal(0, 1, 1_000_000) <= 1.5)
mc_proba("Lb_exp_15", V["Lb_exp_15"], rng.exponential(1.0, 1_000_000) <= 2.5)
mc_proba("Lb_phi_14", V["Lb_phi_14"], rng.normal(0, 1, 1_000_000) <= 1.4)

# Point de contrôle « unites » : distracteurs.
V["Cb_cov_une_fois"] = float(100 * cxy)
V["Cb_rho_cent"] = float(100 * rho_xy)

# =====================================================================================
# LEÇON, PARTIE 3 : tests d'adéquation
# =====================================================================================
# 3a. Khi-deux, deux classes : la statistique est le carré de la binomiale centrée réduite.
o1, n_s, p1 = sp.symbols("o_1 n p_1", positive=True)
c_2 = (o1 - n_s * p1) ** 2 / (n_s * p1) + ((n_s - o1) - n_s * (1 - p1)) ** 2 / (n_s * (1 - p1))
assert sp.simplify(c_2 - (o1 - n_s * p1) ** 2 / (n_s * p1 * (1 - p1))) == 0
# Espérance exacte de la statistique : k - 1, par énumération complète d'une loi multinomiale.
n_m, p_m = 6, [Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)]
esp_c = Fraction(0)
for o in product(range(n_m + 1), repeat=3):
    if sum(o) == n_m:
        pr = Fraction(math.factorial(n_m), math.prod(math.factorial(v) for v in o)) * math.prod(p**v for p, v in zip(p_m, o))
        esp_c += pr * sum((v - n_m * p) ** 2 / (n_m * p) for p, v in zip(p_m, o))
assert esp_c == 2
# Covariance des écarts réduits : I - u u^T avec u = racine(p), projecteur de rang k - 1.
pv = np.array([0.5, 0.2, 0.2, 0.1])
vp = np.linalg.eigvalsh(np.eye(4) - np.outer(np.sqrt(pv), np.sqrt(pv)))
assert np.allclose(sorted(vp), [0, 1, 1, 1])
o_sim = rng.multinomial(50, pv, size=300_000)
red = (o_sim - 50 * pv) / np.sqrt(50 * pv)
assert np.allclose(np.cov(red.T), np.eye(4) - np.outer(np.sqrt(pv), np.sqrt(pv)), atol=0.01)
mc_moyenne("Lc_esp_stat", 3, (red**2).sum(axis=1))

# 3b. Exemple du dé : 120 lancers.
obs_de = [15, 27, 18, 22, 14, 24]
assert sum(obs_de) == 120
c_de = sum(Fraction((o - 20) ** 2, 20) for o in obs_de)
fraction("Lc_de_stat", c_de)
V["Lc_de_ddl"] = 5
V["Lc_de_seuil"] = float(stats.chi2.ppf(0.95, 5))
V["Lc_de_p"] = float(stats.chi2.sf(float(c_de), 5))
assert abs(stats.chisquare(obs_de)[0] - float(c_de)) < 1e-12 and abs(stats.chisquare(obs_de)[1] - V["Lc_de_p"]) < 1e-12
o_sim = rng.multinomial(120, [1 / 6] * 6, size=400_000)
c_sim = ((o_sim - 20) ** 2 / 20).sum(axis=1)
V["Lc_de_p_sim"] = float(np.mean(c_sim >= float(c_de) - 1e-9))
assert abs(V["Lc_de_p_sim"] - V["Lc_de_p"]) < 0.01  # approximation asymptotique : contrôle à 0,01 près
mc_moyenne("Lc_de_esp", 5, c_sim)

# 3c. Exemple de Poisson : 100 intervalles, classes 0, 1, 2, 3, au moins 4.
obs_po = [20, 36, 24, 14, 6]
total_po = 152  # nombre total d'événements (les 6 intervalles de la dernière classe en totalisent 26)
assert sum(obs_po) == 100 and 0 * 20 + 1 * 36 + 2 * 24 + 3 * 14 + 26 == total_po
lam_po = total_po / 100
pr_po = [math.exp(-lam_po) * lam_po**j / math.factorial(j) for j in range(4)]
pr_po.append(1 - sum(pr_po))
att_po = [100 * p for p in pr_po]
assert min(att_po) >= 5
c_po = sum((o - e) ** 2 / e for o, e in zip(obs_po, att_po))
V["Lc_po_lambda"] = lam_po
V["Lc_po_total"] = total_po
V["Lc_po_att"] = att_po
V["Lc_po_stat"] = c_po
V["Lc_po_ddl"] = 3
V["Lc_po_seuil"] = float(stats.chi2.ppf(0.95, 3))
V["Lc_po_p"] = float(stats.chi2.sf(c_po, 3))
assert c_po < V["Lc_po_seuil"]
mc_proba("Lc_po_queue", pr_po[4], rng.poisson(lam_po, 1_000_000) >= 4)
# Niveau constaté du test (intensité 1,5 ; 100 intervalles ; lambda estimé par la moyenne) avec 3 puis 4 degrés de liberté.
k_sim = rng.poisson(1.5, (40_000, 100))
lam_hat = k_sim.mean(axis=1)
compte = np.stack([(k_sim == j).sum(axis=1) for j in range(4)] + [(k_sim >= 4).sum(axis=1)], axis=1)
pr_hat = np.stack([np.exp(-lam_hat) * lam_hat**j / math.factorial(j) for j in range(4)], axis=1)
pr_hat = np.concatenate([pr_hat, 1 - pr_hat.sum(axis=1, keepdims=True)], axis=1)
c_sim = ((compte - 100 * pr_hat) ** 2 / (100 * pr_hat)).sum(axis=1)
V["Lc_po_niveau_ddl3_sim"] = float(np.mean(c_sim >= stats.chi2.ppf(0.95, 3)))
V["Lc_po_niveau_ddl4_sim"] = float(np.mean(c_sim >= stats.chi2.ppf(0.95, 4)))
V["Lc_po_moy_stat_sim"] = float(c_sim.mean())
assert abs(V["Lc_po_niveau_ddl3_sim"] - 0.05) < 0.01 and V["Lc_po_niveau_ddl4_sim"] < 0.035
dire(f"Khi-deux de Poisson, lambda estimé : niveau constaté {V['Lc_po_niveau_ddl3_sim']:.4f} avec 3 ddl, "
     f"{V['Lc_po_niveau_ddl4_sim']:.4f} avec 4 ddl ; moyenne de la statistique {V['Lc_po_moy_stat_sim']:.3f}")


# 3d. Kolmogorov-Smirnov.
def d_ks(u):
    """Statistique de Kolmogorov-Smirnov, ligne par ligne, à partir de u = F0(x)."""
    u = np.sort(np.atleast_2d(u), axis=1)
    n = u.shape[1]
    i = np.arange(1, n + 1)
    return np.maximum((i / n - u).max(axis=1), (u - (i - 1) / n).max(axis=1))


durees = [0.3, 0.9, 1.5, 2.4, 5.1]  # H0 : loi exponentielle de moyenne 2
u_ks = [1 - math.exp(-d / 2) for d in durees]
haut = [(i + 1) / 5 - u for i, u in enumerate(u_ks)]
bas = [u - i / 5 for i, u in enumerate(u_ks)]
d_obs = max(max(haut), max(bas))
assert abs(d_obs - d_ks(u_ks)[0]) < 1e-15
res_ks = stats.kstest(durees, lambda t: 1 - np.exp(-np.asarray(t) / 2))
assert abs(res_ks.statistic - d_obs) < 1e-12
V["Lc_ks_u"] = u_ks
V["Lc_ks_haut"] = haut
V["Lc_ks_bas"] = bas
V["Lc_ks_d"] = d_obs
V["Lc_ks_seuil5"] = float(stats.kstwo.ppf(0.95, 5))
V["Lc_ks_p"] = float(stats.kstwo.sf(d_obs, 5))
assert abs(res_ks.pvalue - V["Lc_ks_p"]) < 1e-9 and d_obs < V["Lc_ks_seuil5"]
assert d_obs == max(bas) and max(bas) > max(haut)  # ici le maximum est atteint juste avant un saut
V["Lc_ks_c95"] = float(stats.kstwobign.ppf(0.95))
serie = 1 - 2 * sum((-1) ** (j - 1) * math.exp(-2 * j**2 * V["Lc_ks_c95"] ** 2) for j in range(1, 60))
assert abs(serie - 0.95) < 1e-9
# Loi de D_n indépendante de la loi continue testée : n = 10, trois lois, même seuil exact.
n_k, nb_k = 10, 200_000
seuil10 = stats.kstwo.ppf(0.95, n_k)
V["Lc_ks_seuil10"] = float(seuil10)
d_norm = d_ks(stats.norm.cdf(rng.normal(3.0, 2.0, (nb_k, n_k)), 3.0, 2.0))
d_expo = d_ks(1 - np.exp(-rng.exponential(2.0, (nb_k, n_k)) / 2.0))
d_unif = d_ks(rng.random((nb_k, n_k)))
mc_proba("Lc_ks_niv_norm", 0.05, d_norm > seuil10)
mc_proba("Lc_ks_niv_expo", 0.05, d_expo > seuil10)
mc_proba("Lc_ks_niv_unif", 0.05, d_unif > seuil10)
# Loi discrète : sup de |F_n - F_0| sur le support d'une loi de Poisson de paramètre 3 ; niveau réel plus petit.
k_d = rng.poisson(3.0, (40_000, n_k))
support = np.arange(0, 40)
f_n = (k_d[:, :, None] <= support).mean(axis=1)
d_disc = np.abs(f_n - stats.poisson.cdf(support, 3.0)).max(axis=1)
V["Lc_ks_niv_discret_sim"] = float(np.mean(d_disc > seuil10))
assert V["Lc_ks_niv_discret_sim"] < 0.02
# Paramètres estimés : niveau réel avec la table de Kolmogorov, et seuil simulé (Lilliefors), n = 50.
n_l, nb_l = 50, 100_000
seuil50 = stats.kstwo.ppf(0.95, n_l)
xs_l = rng.normal(0, 1, (nb_l, n_l))
z_l = (xs_l - xs_l.mean(axis=1, keepdims=True)) / xs_l.std(axis=1, ddof=1, keepdims=True)
d_est = d_ks(stats.norm.cdf(z_l))
V["Lc_lil_niv_table_sim"] = float(np.mean(d_est > seuil50))
V["Lc_lil_seuil_sim"] = float(np.quantile(d_est, 0.95))
V["Lc_ks_seuil50"] = float(seuil50)
assert V["Lc_lil_niv_table_sim"] < 0.002 and V["Lc_lil_seuil_sim"] < 0.7 * seuil50
# Invariance par changement de position et d'échelle : même statistique pour z et pour mu + sigma z.
x_t = 7.0 - 3.0 * 0 + 4.0 * xs_l[:1000]
z_t = (x_t - x_t.mean(axis=1, keepdims=True)) / x_t.std(axis=1, ddof=1, keepdims=True)
assert np.allclose(d_ks(stats.norm.cdf(z_t)), d_est[:1000], atol=1e-12)
dire(f"KS : niveau constaté avec une loi discrète {V['Lc_ks_niv_discret_sim']:.4f} ; avec paramètres estimés et table de "
     f"Kolmogorov {V['Lc_lil_niv_table_sim']:.5f} ; seuil simulé {V['Lc_lil_seuil_sim']:.4f} contre {seuil50:.4f}")

# 3e. Anderson-Darling : formule de calcul contre l'intégrale pondérée, et variance de F_n.
u_ad = np.sort(rng.random(7))
n_ad = len(u_ad)
i_ad = np.arange(1, n_ad + 1)
a2_formule = -n_ad - np.mean((2 * i_ad - 1) * (np.log(u_ad) + np.log(1 - u_ad[::-1])))
bornes = np.concatenate([[0.0], u_ad, [1.0]])
a2_integrale = 0.0
for j in range(n_ad + 1):
    val, _ = sci_int.quad(lambda t, g=j / n_ad: (g - t) ** 2 / (t * (1 - t)), bornes[j], bornes[j + 1])
    a2_integrale += n_ad * val
assert abs(a2_formule - a2_integrale) < 1e-8
u_k = np.sort(np.array(u_ks))
V["Lc_ad_stat"] = float(-5 - np.mean((2 * np.arange(1, 6) - 1) * (np.log(u_k) + np.log(1 - u_k[::-1]))))
fn_x = (rng.random((400_000, 20)) <= 0.9).mean(axis=1)
mc_variance("Lc_ad_var_fn", 0.9 * 0.1 / 20, fn_x)
# Jarque-Bera : formule n/6 (S^2 + (K - 3)^2 / 4) contre scipy.
x_jb = rng.normal(0, 1, 500)
s_jb, k_jb = stats.skew(x_jb), stats.kurtosis(x_jb, fisher=False)
assert abs(500 / 6 * (s_jb**2 + (k_jb - 3) ** 2 / 4) - stats.jarque_bera(x_jb)[0]) < 1e-9
dire("Partie 3 : identités du khi-deux, statistique de KS, formule d'Anderson-Darling et de Jarque-Bera vérifiées.")

# =====================================================================================
# LEÇON, PARTIE 4 : densités conjointes
# =====================================================================================
xs, ys = sp.symbols("x y", positive=True)
# Exemple A : f(x, y) = c x y sur 0 < x < y < 1.
masse = sp.integrate(sp.integrate(xs * ys, (xs, 0, ys)), (ys, 0, 1))
assert 1 / masse == 8
V["Ld_A_c"] = 8
fA = 8 * xs * ys
fA_x = sp.integrate(fA, (ys, xs, 1))
fA_y = sp.integrate(fA, (xs, 0, ys))
assert sp.simplify(fA_x - 4 * xs * (1 - xs**2)) == 0 and fA_y == 4 * ys**3
condA = sp.simplify(fA / fA_x)
assert sp.simplify(condA - 2 * ys / (1 - xs**2)) == 0
pA = sp.integrate(condA.subs(xs, R(1, 2)), (ys, R(3, 4), 1))
eA = sp.integrate(ys * condA.subs(xs, R(1, 2)), (ys, R(1, 2), 1))
pA_marg = sp.integrate(fA_y, (ys, R(3, 4), 1))
assert pA == R(7, 12) and eA == R(7, 9) and pA_marg == R(175, 256)
fraction("Ld_A_p", pA)
fraction("Ld_A_e", eA)
fraction("Ld_A_p_marg", pA_marg)
fraction("Ld_A_fx_demi", fA_x.subs(xs, R(1, 2)))
assert sp.simplify(fA_x * fA_y - fA) != 0  # pas d'indépendance
fraction("Ld_A_prod", (fA_x * fA_y).subs({xs: R(3, 4), ys: R(1, 4)}))  # produit des marginales là où f = 0
n_a = 6_000_000
ya = rng.random(n_a) ** 0.25  # densité 4 y^3
xa = ya * np.sqrt(rng.random(n_a))  # sachant Y = y, densité 2x / y^2 sur ]0, y[
mc_proba("Ld_A_p_marg", pA_marg, ya > 0.75)
bande = np.abs(xa - 0.5) < 0.004
mc_proba("Ld_A_p", pA, ya[bande] > 0.75)
mc_moyenne("Ld_A_e", eA, ya[bande])
# Exemple B : f(x, y) = x exp(-x (1 + y)) sur x > 0, y > 0.
fB = xs * sp.exp(-xs * (1 + ys))
assert sp.integrate(sp.integrate(fB, (ys, 0, sp.oo)), (xs, 0, sp.oo)) == 1
fB_x = sp.integrate(fB, (ys, 0, sp.oo))
fB_y = sp.simplify(sp.integrate(fB, (xs, 0, sp.oo)))
assert sp.simplify(fB_x - sp.exp(-xs)) == 0 and sp.simplify(fB_y - 1 / (1 + ys) ** 2) == 0
assert sp.simplify(fB / fB_x - xs * sp.exp(-xs * ys)) == 0  # loi exponentielle de paramètre x
assert sp.simplify(fB / fB_y - (1 + ys) ** 2 * xs * sp.exp(-xs * (1 + ys))) == 0  # loi gamma (2, 1 + y)
pB = sp.integrate((fB / fB_x).subs(xs, 2), (ys, 1, sp.oo))
eB = sp.integrate(xs * (fB / fB_y).subs(ys, 1), (xs, 0, sp.oo))
pB_marg = sp.integrate(fB_y, (ys, 1, sp.oo))
assert sp.simplify(pB - sp.exp(-2)) == 0 and eB == 1 and pB_marg == R(1, 2)
V["Ld_B_p"] = math.exp(-2)
V["Ld_B_e"] = 1
V["Ld_B_p_marg"] = 0.5
xb_ = rng.exponential(1.0, n_a)
yb_ = rng.exponential(1.0, n_a) / xb_
mc_proba("Ld_B_p_marg", 0.5, yb_ > 1)
bande = np.abs(xb_ - 2) < 0.01
mc_proba("Ld_B_p", math.exp(-2), yb_[bande] > 1)
bande = np.abs(yb_ - 1) < 0.01
mc_moyenne("Ld_B_e", 1, xb_[bande])
# Conditionner par un événement de probabilité nulle : X, Y uniformes indépendantes, « X = Y » de deux façons.
e_diff = sp.integrate(xs * 1, (xs, 0, 1))  # sachant X - Y = 0 : densité 1 sur ]0, 1[
e_ratio = sp.integrate(xs * 2 * xs, (xs, 0, 1))  # sachant X / Y = 1 : densité 2x sur ]0, 1[
fraction("Ld_bk_diff", e_diff)
fraction("Ld_bk_ratio", e_ratio)
u1, u2 = rng.random(n_a), rng.random(n_a)
mc_moyenne("Ld_bk_diff", e_diff, u1[np.abs(u1 - u2) < 0.002])
mc_moyenne("Ld_bk_ratio", e_ratio, u1[np.abs(u1 / u2 - 1) < 0.002])
# Densité de (X, R = X / Y) : y = x / r, jacobien x / r^2 ; en r = 1 elle vaut x sur ]0, 1[.
r_s = sp.symbols("r", positive=True)
assert sp.diff(xs / r_s, r_s).subs(r_s, 1) == -xs

# =====================================================================================
# LEÇON, PARTIE 5 : estimation bayésienne de l'intensité d'une loi de Poisson
# =====================================================================================
def posterieur_liste(candidats, poids_a_priori, k, t):
    brut = [w * math.exp(-l * t) * (l * t) ** k / math.factorial(k) for l, w in zip(candidats, poids_a_priori)]
    return [b / sum(brut) for b in brut]


def simuler_liste(candidats, poids_a_priori, k, t, n_sim):
    """Tire l'intensité selon l'a priori, puis le comptage ; garde les tirages où le comptage vaut k."""
    p = np.array(poids_a_priori, dtype=float)
    lam_tire = rng.choice(np.array(candidats, dtype=float), p=p / p.sum(), size=n_sim)
    compt = rng.poisson(lam_tire * t)
    return lam_tire[compt == k]


cand = [1, 2, 3, 5]
k_e, t_e = 5, 2
post_u = posterieur_liste(cand, [1, 1, 1, 1], k_e, t_e)
post_i = posterieur_liste(cand, [1 / l for l in cand], k_e, t_e)
V["Le_noyau_unif"] = [l**k_e * math.exp(-l * t_e) for l in cand]
V["Le_noyau_inv"] = [l ** (k_e - 1) * math.exp(-l * t_e) for l in cand]
V["Le_post_unif"] = post_u
V["Le_post_inv"] = post_i
V["Le_map_unif"] = cand[int(np.argmax(post_u))]
V["Le_map_inv"] = cand[int(np.argmax(post_i))]
V["Le_emv"] = k_e / t_e
assert V["Le_map_unif"] == 3 and V["Le_map_inv"] == 2
V["Le_logmoy_23"] = (3 - 2) / math.log(3 / 2)
assert V["Le_logmoy_23"] < V["Le_emv"]
V["Le_prior_inv"] = [(1 / l) / sum(1 / c for c in cand) for l in cand]
garde = simuler_liste(cand, [1, 1, 1, 1], k_e, t_e, 2_000_000)
mc_proba("Le_post_unif_3", post_u[2], garde == 3)
garde = simuler_liste(cand, [1 / l for l in cand], k_e, t_e, 2_000_000)
mc_proba("Le_post_inv_2", post_i[1], garde == 2)
# Seuil entre deux candidats : la moyenne logarithmique. Candidats 2 et 6, 19 événements en 5 unités de temps.
V["Le_lm_seuil"] = (6 - 2) / math.log(6 / 2)
V["Le_lm_taux"] = 19 / 5
V["Le_lm_milieu"] = 4
V["Le_lm_geo"] = math.sqrt(12)
V["Le_lm_rapport"] = (6 / 2) ** 19 * math.exp(-(6 - 2) * 5)
assert V["Le_lm_geo"] < V["Le_lm_seuil"] < V["Le_lm_taux"] < V["Le_lm_milieu"] and V["Le_lm_rapport"] > 1
p2 = posterieur_liste([2, 6], [1, 1], 19, 5)
assert abs(p2[1] / p2[0] - V["Le_lm_rapport"]) < 1e-9
# Conjugaison gamma-Poisson : a priori gamma (2, 1), 7 événements en 3 heures.
al_s, be2_s, k_s, t2_s, l_s = sp.symbols("alpha beta k t lambda", positive=True)
noyau = l_s ** (al_s - 1) * sp.exp(-be2_s * l_s) * sp.exp(-l_s * t2_s) * l_s**k_s
assert sp.simplify(noyau / (l_s ** (al_s + k_s - 1) * sp.exp(-(be2_s + t2_s) * l_s))) == 1
assert sp.solve(sp.diff((al_s - 1) * sp.log(l_s) - be2_s * l_s, l_s), l_s) == [(al_s - 1) / be2_s]  # mode
th_s = sp.symbols("theta", positive=True)
assert sp.solve(sp.diff((al_s - 1) * sp.log(1 / th_s) - be2_s / th_s - 2 * sp.log(th_s), th_s), th_s) == [be2_s / (al_s + 1)]
a0, b0, k_g, t_g = 2, 1, 7, 3
al, be = a0 + k_g, b0 + t_g
V["Le_g_alpha"], V["Le_g_beta"] = al, be
fraction("Le_g_moy", Fraction(al, be))
fraction("Le_g_map", Fraction(al - 1, be))
fraction("Le_g_emv", Fraction(k_g, t_g))
fraction("Le_g_prior_moy", Fraction(a0, b0))
fraction("Le_g_prior_mode", Fraction(a0 - 1, b0))
fraction("Le_g_poids_prior", Fraction(b0, b0 + t_g))
fraction("Le_g_poids_obs", Fraction(t_g, b0 + t_g))
assert Fraction(b0, b0 + t_g) * Fraction(a0, b0) + Fraction(t_g, b0 + t_g) * Fraction(k_g, t_g) == Fraction(al, be)
fraction("Le_g_var", Fraction(al, be**2))
fraction("Le_imp_moy", Fraction(k_g, t_g))  # a priori 1/lambda : gamma (k, t)
fraction("Le_imp_map", Fraction(k_g - 1, t_g))
fraction("Le_plat_moy", Fraction(k_g + 1, t_g))  # a priori constant : gamma (k + 1, t)
fraction("Le_plat_map", Fraction(k_g, t_g))
fraction("Le_theta_mode", Fraction(be, al + 1))  # mode de 1/lambda sous la loi gamma (9, 4)
fraction("Le_theta_inv_map", Fraction(be, al - 1))
masse_post = sp.integrate(l_s ** (al - 1) * sp.exp(-be * l_s), (l_s, 0, sp.oo))
assert masse_post == sp.Rational(math.factorial(al - 1), be**al)
lam_g = rng.gamma(a0, 1 / b0, 6_000_000)
garde = lam_g[rng.poisson(lam_g * t_g) == k_g]
mc_moyenne("Le_g_moy", Fraction(al, be), garde)
mc_variance("Le_g_var", Fraction(al, be**2), garde)
mc_proba("Le_g_mediane_p", float(stats.gamma.cdf(2.0, al, scale=1 / be)), garde <= 2.0)
dire("Partie 5 : a posteriori sur une liste, moyenne logarithmique, conjugaison gamma-Poisson vérifiés.")

# Point de contrôle « posterieur » : distracteurs.
fraction("Ce_d_sans_duree", Fraction(al - 1, t_g))  # gamma (9, 3) : durée oubliée

# =====================================================================================
# LEÇON, PARTIE 6 : bâton coupé, espacements
# =====================================================================================
def esp_kieme(n, k):
    """Espérance du k-ième plus petit de n morceaux : (1/n) (1/n + 1/(n-1) + ... + 1/(n-k+1))."""
    return Fraction(1, n) * sum(Fraction(1, n - j) for j in range(k))


def esp_kieme_ie(n, k):
    """Même espérance par inclusion-exclusion : P(au moins r morceaux > c), avec E[min de j morceaux] = 1/(jn)."""
    r = n - k + 1
    return sum((-1) ** (j - r) * math.comb(j - 1, r - 1) * math.comb(n, j) * Fraction(1, j * n) for j in range(r, n + 1))


c_sym = sp.symbols("c", positive=True)
for n in range(2, 10):
    assert sum(esp_kieme(n, k) for k in range(1, n + 1)) == 1
    assert esp_kieme(n, 1) == Fraction(1, n * n)
    for k in range(1, n + 1):
        assert esp_kieme(n, k) == esp_kieme_ie(n, k)
    for j in range(1, n + 1):  # E[min de j morceaux donnés] = intégrale de (1 - jc)^(n-1) sur [0, 1/j]
        assert sp.integrate((1 - j * c_sym) ** (n - 1), (c_sym, 0, R(1, j))) == R(1, j * n)
fraction("Lf_3_min", esp_kieme(3, 1))
fraction("Lf_3_mil", esp_kieme(3, 2))
fraction("Lf_3_max", esp_kieme(3, 3))
fraction("Lf_2_min", esp_kieme(2, 1))
fraction("Lf_2_max", esp_kieme(2, 2))
fraction("Lf_triangle", 1 - 3 * Fraction(1, 2) ** 2)
fraction("Lf_poly4", 1 - 4 * Fraction(1, 2) ** 3)
V["Lf_seq"] = 2 * math.log(2) - 1
m_s = sp.symbols("m", positive=True)
assert sp.simplify(sp.integrate(2 * (1 / m_s - 1), (m_s, R(1, 2), 1)) - (2 * sp.log(2) - 1)) == 0
# Loi conjointe (n = 3) : densité 2 sur le triangle l1 > 0, l2 > 0, l1 + l2 < 1.
l1, l2 = sp.symbols("l_1 l_2", positive=True)
e_l1l2 = sp.integrate(sp.integrate(2 * l1 * l2, (l2, 0, 1 - l1)), (l1, 0, 1))
e_l1 = sp.integrate(sp.integrate(2 * l1, (l2, 0, 1 - l1)), (l1, 0, 1))
e_l1c = sp.integrate(sp.integrate(2 * l1**2, (l2, 0, 1 - l1)), (l1, 0, 1))
cov_l = e_l1l2 - e_l1**2
var_l = e_l1c - e_l1**2
assert e_l1 == R(1, 3) and cov_l == R(-1, 36) and var_l == R(1, 18)
fraction("Lf_3_corr", cov_l / var_l)


def morceaux(n, n_sim):
    """n morceaux d'un bâton de longueur 1 coupé par n - 1 points uniformes indépendants, dans l'ordre du bâton."""
    coupes = np.sort(rng.random((n_sim, n - 1)), axis=1)
    return np.diff(np.concatenate([np.zeros((n_sim, 1)), coupes, np.ones((n_sim, 1))], axis=1), axis=1)


m3 = morceaux(3, 1_000_000)
t3 = np.sort(m3, axis=1)
mc_moyenne("Lf_3_min", esp_kieme(3, 1), t3[:, 0])
mc_moyenne("Lf_3_mil", esp_kieme(3, 2), t3[:, 1])
mc_moyenne("Lf_3_max", esp_kieme(3, 3), t3[:, 2])
mc_proba("Lf_triangle", Fraction(1, 4), t3[:, 2] < 0.5)
mc_covariance("Lf_3_cov", cov_l, m3[:, 0], m3[:, 1])
m4 = morceaux(4, 1_000_000)
t4 = np.sort(m4, axis=1)
mc_proba("Lf_poly4", Fraction(1, 2), t4[:, 3] < 0.5)
# Loi conjointe : P(L_j > c_j pour tout j) = (1 - somme des c_j)^(n-1).
c_vec = np.array([0.1, 0.05, 0.2, 0.15])
V["Lf_joint_ex"] = (1 - c_vec.sum()) ** 3
mc_proba("Lf_joint_ex", V["Lf_joint_ex"], (m4 > c_vec).all(axis=1))
# Représentation par des exponentielles normalisées : mêmes moyennes des morceaux triés, et indépendance de la somme.
e4 = rng.exponential(1.0, (1_000_000, 4))
s4 = e4.sum(axis=1)
pe4 = np.sort(e4 / s4[:, None], axis=1)
for k in range(1, 5):
    mc_moyenne(f"Lf_4_expo_{k}", esp_kieme(4, k), pe4[:, k - 1])
    mc_moyenne(f"Lf_4_baton_{k}", esp_kieme(4, k), t4[:, k - 1])
mc_covariance("Lf_4_indep", 0, pe4[:, 0], s4)
mc_proba("Lf_joint_expo", V["Lf_joint_ex"], (e4 / s4[:, None] > c_vec).all(axis=1))
# Variante séquentielle : on coupe, puis on recoupe le plus grand morceau.
u_a, u_b = rng.random(1_000_000), rng.random(1_000_000)
grand = np.maximum(u_a, 1 - u_a)
p_seq = np.stack([1 - grand, grand * u_b, grand * (1 - u_b)], axis=1)
mc_proba("Lf_seq", V["Lf_seq"], p_seq.max(axis=1) < 0.5)
dire("Partie 6 : espérances des morceaux triés (formule, inclusion-exclusion, deux simulations) vérifiées.")

# =====================================================================================
# LEÇON, PARTIE 7 : conditionner sur la bonne information
# =====================================================================================
familles = list(product("GF", repeat=2))  # (aîné, cadet), quatre issues équiprobables
pr = Fraction(1, 4)
p_v1 = sum(pr for f in familles if f == ("F", "F")) / sum(pr for f in familles if f[0] == "F")
p_v2 = sum(pr for f in familles if f == ("G", "G")) / sum(pr for f in familles if "G" in f)
# Version 3 : un enfant choisi au hasard (probabilité 1/2 chacun, indépendamment des sexes) est un garçon.
num = sum(pr * Fraction(1, 2) for f in familles for i in (0, 1) if f[i] == "G" and f == ("G", "G"))
den = sum(pr * Fraction(1, 2) for f in familles for i in (0, 1) if f[i] == "G")
p_v3 = num / den
assert p_v1 == Fraction(1, 2) and p_v2 == Fraction(1, 3) and p_v3 == Fraction(1, 2)
fraction("Lg_v1", p_v1)
fraction("Lg_v2", p_v2)
fraction("Lg_v3", p_v3)
fraction("Lg_v3_pvu", den)  # probabilité que l'enfant rencontré soit un garçon
V["Lg_v3_nb_issues"] = 8
V["Lg_v3_nb_vu"] = sum(1 for f in familles for i in (0, 1) if f[i] == "G")
V["Lg_v3_nb_gg"] = sum(1 for f in familles for i in (0, 1) if f[i] == "G" and f == ("G", "G"))
# Version « au moins un garçon né un mardi » : 14 types d'enfant équiprobables.
types = list(product("GF", range(7)))
num = den = 0
for e1 in types:
    for e2 in types:
        if ("G", 1) in (e1, e2):
            den += 1
            num += e1[0] == "G" and e2[0] == "G"
p_mardi = Fraction(num, den)
assert p_mardi == Fraction(13, 27)
fraction("Lg_mardi", p_mardi)
V["Lg_mardi_nb"] = len(types) ** 2
q_s = sp.symbols("q", positive=True)
formule = sp.simplify((R(1, 4) * (1 - (1 - q_s) ** 2)) / (1 - (1 - q_s / 2) ** 2))
assert sp.simplify(formule - (2 - q_s) / (4 - q_s)) == 0
assert formule.subs(q_s, 1) == R(1, 3) and formule.subs(q_s, R(1, 7)) == R(13, 27) and sp.limit(formule, q_s, 0) == R(1, 2)
n_f = 2_000_000
sexe = rng.integers(0, 2, (n_f, 2))  # 1 = garçon ; colonne 0 = aîné
mc_proba("Lg_v1", p_v1, sexe[sexe[:, 0] == 0].sum(axis=1) == 0)
mc_proba("Lg_v2", p_v2, sexe[sexe.sum(axis=1) >= 1].sum(axis=1) == 2)
vu = rng.integers(0, 2, n_f)  # enfant rencontré
garcon_vu = sexe[np.arange(n_f), vu] == 1
mc_proba("Lg_v3", p_v3, sexe[garcon_vu].sum(axis=1) == 2)
jour = rng.integers(0, 7, (n_f, 2))
mardi = ((sexe == 1) & (jour == 1)).any(axis=1)
mc_proba("Lg_mardi", p_mardi, sexe[mardi].sum(axis=1) == 2)
dire("Partie 7 : les versions du problème des deux enfants vérifiées par énumération et par simulation.")

# =====================================================================================
# QUESTIONS
# =====================================================================================
# M5-02 : khi-deux d'adéquation à la loi uniforme sur 4 symboles, 200 tirages.
obs2 = [38, 62, 55, 45]
assert sum(obs2) == 200
c2 = sum(Fraction((o - 50) ** 2, 50) for o in obs2)
fraction("Q02_exact", c2)
V["Q02_somme_carres"] = sum((o - 50) ** 2 for o in obs2)
V["Q02_ddl"] = 3
V["Q02_q2"], V["Q02_q3"], V["Q02_q4"] = (float(stats.chi2.ppf(0.95, d)) for d in (2, 3, 4))
V["Q02_p"] = float(stats.chi2.sf(float(c2), 3))
V["Q02_d_sur_n"] = float(Fraction(V["Q02_somme_carres"], 200))
V["Q02_d_brut"] = V["Q02_somme_carres"]
assert V["Q02_q2"] < float(c2) < V["Q02_q3"] and V["Q02_d_sur_n"] < V["Q02_q3"] < V["Q02_d_brut"]
assert abs(stats.chisquare(obs2)[0] - float(c2)) < 1e-12
o_sim = rng.multinomial(200, [0.25] * 4, size=400_000)
c_sim = ((o_sim - 50) ** 2 / 50).sum(axis=1)
mc_moyenne("Q02_esp", 3, c_sim)
V["Q02_p_sim"] = float(np.mean(c_sim >= float(c2) - 1e-9))
assert abs(V["Q02_p_sim"] - V["Q02_p"]) < 0.01

# M5-03 : loi de Poisson ajustée, 200 intervalles, 300 événements.
lam3 = Fraction(300, 200)
pr3 = [math.exp(-1.5) * 1.5**j / math.factorial(j) for j in range(5)]
queue3 = 1 - sum(pr3[:4])
V["Q03_lambda"] = float(lam3)
V["Q03_exp"] = math.exp(-1.5)
V["Q03_somme"] = sum(1.5**j / math.factorial(j) for j in range(4))  # 1 + 1,5 + 1,125 + 0,5625
V["Q03_p_bas"] = sum(pr3[:4])
V["Q03_n_param"] = 1
V["Q03_exact"] = 200 * queue3
V["Q03_p_queue"] = queue3
V["Q03_att"] = [200 * p for p in pr3[:4]]
V["Q03_ddl"] = 5 - 1 - 1
V["Q03_d_egal4"] = 200 * pr3[4]
V["Q03_d_ddl_sans_estim"] = 5 - 1
V["Q03_d_ddl_classes"] = 5
assert abs(queue3 - stats.poisson.sf(3, 1.5)) < 1e-12 and round(V["Q03_exact"], 1) != round(V["Q03_d_egal4"], 1)
assert min(V["Q03_att"] + [V["Q03_exact"]]) >= 5
assert 200 * math.exp(-1.5) * 1.5**6 / math.factorial(6) < 1  # classe « 6 » seule : effectif attendu inférieur à 1
mc_proba("Q03_p_queue", queue3, rng.poisson(1.5, 2_000_000) >= 4)

# M5-04 : température et ventes, changement d'unités.
s_x4, s_y4, c4 = Fraction(5), Fraction(40), Fraction(120)
rho4 = c4 / (s_x4 * s_y4)
assert rho4 == Fraction(6, 10)
V["Q04_rho"] = float(rho4)
V["Q04_cov_f"] = float(Fraction(18, 10) * c4)
V["Q04_sx_f"] = float(Fraction(18, 10) * s_x4)
V["Q04_pente"] = float(c4 / s_x4**2)
V["Q04_pente_f"] = float(Fraction(18, 10) * c4 / (Fraction(18, 10) * s_x4) ** 2)
V["Q04_cov_ke"] = float(c4 / 1000)
V["Q04_d_cov_affine"] = float(Fraction(18, 10) * c4 + 32)
assert Fraction(18, 10) * c4 / (Fraction(18, 10) * s_x4 * s_y4) == rho4
xy4 = rng.multivariate_normal([15.0, 300.0], [[25.0, 120.0], [120.0, 1600.0]], size=400_000)
f4 = 1.8 * xy4[:, 0] + 32
mc_covariance("Q04_cov_f", V["Q04_cov_f"], f4, xy4[:, 1])
mc_covariance("Q04_cov_neg", -120, xy4[:, 0], -xy4[:, 1])
assert abs(np.corrcoef(f4, xy4[:, 1] / 1000)[0, 1] - 0.6) < 0.005
assert abs(np.polyfit(xy4[:, 0], xy4[:, 1], 1)[0] - V["Q04_pente"]) < 0.03

# M5-05 : cotes Z de trois examens.
z_a = Fraction(78 - 66, 8)
z_b = Fraction(1340 - 1100, 200)
z_c = Fraction(150 - 114, 20)  # (15 - 11,4) / 2
assert (z_a, z_b, z_c) == (Fraction(3, 2), Fraction(6, 5), Fraction(9, 5))
V["Q05_za"], V["Q05_zb"], V["Q05_zc"] = float(z_a), float(z_b), float(z_c)
V["Q05_d_ecart_b"] = 1340 - 1100
V["Q05_d_ecart_c"] = 3.6
V["Q05_d_pct_b"] = 100 * 1340 / 1600
V["Q05_pct_a"], V["Q05_pct_c"] = 78, 75
V["Q05_phi_c"] = float(stats.norm.cdf(1.8))
V["Q05_phi_a"] = float(stats.norm.cdf(1.5))
V["Q05_phi_b"] = float(stats.norm.cdf(1.2))
mc_proba("Q05_phi_c", V["Q05_phi_c"], rng.normal(11.4, 2.0, 1_000_000) <= 15)

# M5-06 : loi log-normale, ln X de loi normale centrée réduite.
V["Q06_med"] = 1
V["Q06_moy"] = math.exp(0.5)
V["Q06_mode"] = math.exp(-1)
V["Q06_d_mode"] = math.exp(-0.5)
V["Q06_skew"] = (math.e + 2) * math.sqrt(math.e - 1)
V["Q06_sigma"] = math.sqrt(math.e * (math.e - 1))
assert abs(V["Q06_skew"] - float(stats.lognorm.stats(1.0, moments="s"))) < 1e-9
x6 = np.exp(rng.normal(0, 1, 4_000_000))
mc_moyenne("Q06_moy", V["Q06_moy"], x6)
mc_proba("Q06_med_p", 0.5, x6 <= 1)
# Mode : la densité en exp(-1) dépasse la densité en des points voisins et en exp(-1/2).
dens6 = lambda v: stats.lognorm.pdf(v, 1.0)
assert dens6(V["Q06_mode"]) > max(dens6(V["Q06_mode"] * 0.98), dens6(V["Q06_mode"] * 1.02), dens6(V["Q06_d_mode"]))
h6 = 0.01
mc_proba("Q06_mode_p", stats.lognorm.cdf(V["Q06_mode"] + h6, 1.0) - stats.lognorm.cdf(V["Q06_mode"] - h6, 1.0), np.abs(x6 - V["Q06_mode"]) < h6)

# M5-07 : intensité parmi une liste, a priori proportionnel à 1/lambda, 3 événements en 2 heures.
cand7 = [0.5, 1, 2, 4]
post7 = posterieur_liste(cand7, [1 / l for l in cand7], 3, 2)
post7_u = posterieur_liste(cand7, [1, 1, 1, 1], 3, 2)
prior7 = [(1 / l) / sum(1 / c for c in cand7) for l in cand7]
assert int(np.argmax(post7)) == 1 and int(np.argmax(post7_u)) == 2 and int(np.argmax(prior7)) == 0
V["Q07_post"] = post7
V["Q07_exact"] = post7[1]
V["Q07_noyau"] = [l**2 * math.exp(-2 * l) for l in cand7]
V["Q07_somme_noyau"] = sum(V["Q07_noyau"])
V["Q07_d_unif_2"] = post7_u[2]
V["Q07_d_unif_1"] = post7_u[1]
V["Q07_d_prior"] = prior7[0]
V["Q07_emv"] = 3 / 2
V["Q07_e1"], V["Q07_e2"], V["Q07_e4"], V["Q07_e8"] = (math.exp(-v) for v in (1, 2, 4, 8))
V["Q07_noyau_unif"] = [l**3 * math.exp(-2 * l) for l in cand7]
V["Q07_prior"] = prior7
V["Q07_post_unif"] = post7_u
assert len({round(v, 3) for v in [post7[1], post7_u[2], post7_u[1], prior7[0]]}) == 4
assert abs(prior7[0] - 2 / 3.75) < 1e-15
garde = simuler_liste(cand7, [1 / l for l in cand7], 3, 2, 3_000_000)
mc_proba("Q07", post7[1], garde == 1)

# M5-08 : bâton coupé en 4 morceaux, deuxième plus petit.
fraction("Q08_exact", esp_kieme(4, 2))
fraction("Q08_d_min", esp_kieme(4, 1))
fraction("Q08_d_double", 2 * esp_kieme(4, 1))
fraction("Q08_d_rang", Fraction(2, 1 + 2 + 3 + 4))
fraction("Q08_3e", esp_kieme(4, 3))
fraction("Q08_max", esp_kieme(4, 4))
assert esp_kieme(4, 2) == Fraction(7, 48) and len({esp_kieme(4, 2), esp_kieme(4, 1), 2 * esp_kieme(4, 1), Fraction(1, 5)}) == 4
mc_moyenne("Q08", esp_kieme(4, 2), t4[:, 1])

# M5-09 : bâton coupé en 3 morceaux, affirmations.
fraction("Q09_max", esp_kieme(3, 3))
V["Q09_p_min"] = float((1 - 3 * Fraction(1, 10)) ** 2)
assert (1 - 3 * Fraction(1, 10)) ** 2 == Fraction(49, 100)
fraction("Q09_triangle", Fraction(1, 4))
fraction("Q09_d_triangle", Fraction(1, 2))
fraction("Q09_corr", cov_l / var_l)
m3b = morceaux(3, 1_000_000)
t3b = np.sort(m3b, axis=1)
mc_moyenne("Q09_max", esp_kieme(3, 3), t3b[:, 2])
mc_proba("Q09_p_min", V["Q09_p_min"], t3b[:, 0] > 0.1)
mc_proba("Q09_triangle", Fraction(1, 4), t3b[:, 2] < 0.5)
mc_proba("Q09_gauche", Fraction(4, 9), m3b[:, 0] > R(1, 3))  # P(L > 1/3) = (2/3)^2 pour chaque morceau
mc_proba("Q09_milieu", Fraction(4, 9), m3b[:, 1] > R(1, 3))
fraction("Q09_p_tiers", Fraction(4, 9))

# M5-10 : densité c (x + 2y) sur le carré unité.
masse10 = sp.integrate(sp.integrate(xs + 2 * ys, (xs, 0, 1)), (ys, 0, 1))
c10 = 1 / masse10
assert c10 == R(2, 3)
f10 = c10 * (xs + 2 * ys)
f10_x = sp.integrate(f10, (ys, 0, 1))
assert sp.simplify(f10_x - R(2, 3) * (xs + 1)) == 0
num10 = sp.integrate(f10.subs(xs, R(1, 4)), (ys, R(1, 2), 1))
p10 = num10 / f10_x.subs(xs, R(1, 4))
p10_marg = sp.integrate(sp.integrate(f10, (xs, 0, 1)), (ys, R(1, 2), 1))
assert p10 == R(7, 10) and num10 == R(7, 12) and p10_marg == R(2, 3)
fraction("Q10_c", c10)
fraction("Q10_exact", p10)
fraction("Q10_fx", f10_x.subs(xs, R(1, 4)))
fraction("Q10_d_marg", p10_marg)
fraction("Q10_d_joint", num10)
fraction("Q10_d_unif", Fraction(1, 2))
# Simulation par rejet : candidats uniformes sur le carré, acceptés avec probabilité (x + 2y) / 3.
cx, cy, cu = rng.random(n_a), rng.random(n_a), rng.random(n_a)
ok = cu < (cx + 2 * cy) / 3
x10, y10 = cx[ok], cy[ok]
mc_proba("Q10_marg", p10_marg, y10 > 0.5)
bande = np.abs(x10 - 0.25) < 0.004
mc_proba("Q10", p10, y10[bande] > 0.5)

# M5-11 : densité 2 exp(-x - y) sur 0 < x < y.
f11 = 2 * sp.exp(-xs - ys)
assert sp.integrate(sp.integrate(f11, (ys, xs, sp.oo)), (xs, 0, sp.oo)) == 1
f11_x = sp.integrate(f11, (ys, xs, sp.oo))
f11_y = sp.simplify(sp.integrate(f11, (xs, 0, ys)))
assert sp.simplify(f11_x - 2 * sp.exp(-2 * xs)) == 0 and sp.simplify(f11_y - 2 * sp.exp(-ys) * (1 - sp.exp(-ys))) == 0
assert sp.simplify(f11 / f11_x - sp.exp(-(ys - xs))) == 0  # sachant X = x, Y - x exponentielle de paramètre 1
e11_x = sp.integrate(xs * f11_x, (xs, 0, sp.oo))
e11_y = sp.integrate(ys * f11_y, (ys, 0, sp.oo))
assert e11_x == R(1, 2) and e11_y == R(3, 2)
fraction("Q11_ex", e11_x)
fraction("Q11_ey", e11_y)
V["Q11_p_cond"] = math.exp(-1)
V["Q11_p_y1"] = float(sp.integrate(f11_y, (ys, 1, sp.oo)))
V["Q11_d_p_y1"] = math.exp(-1)  # ce que donnerait une loi exponentielle de paramètre 1
assert abs(V["Q11_p_y1"] - (2 * math.exp(-1) - math.exp(-2))) < 1e-12
e2 = rng.exponential(1.0, (n_a, 2))
x11, y11 = e2.min(axis=1), e2.max(axis=1)  # la densité de (min, max) est 2 exp(-x - y) sur x < y
mc_moyenne("Q11_ex", e11_x, x11)
mc_moyenne("Q11_ey", e11_y, y11)
mc_proba("Q11_p_y1", V["Q11_p_y1"], y11 > 1)
bande = np.abs(x11 - 0.5) < 0.003
mc_proba("Q11_p_cond", math.exp(-1), y11[bande] > 1.5)
mc_covariance("Q11_cov", R(1, 4), x11, y11)  # Cov(min, max) = Var(min) = 1/4 : dépendance
fraction("Q11_cov", Fraction(1, 4))
cov11 = sp.integrate(sp.integrate(xs * ys * f11, (ys, xs, sp.oo)), (xs, 0, sp.oo)) - e11_x * e11_y
assert cov11 == R(1, 4)

# M5-12 : trois enfants, au moins une fille.
f3 = list(product("GF", repeat=3))
p12 = Fraction(sum(f == ("F", "F", "F") for f in f3), sum("F" in f for f in f3))
assert p12 == Fraction(1, 7)
fraction("Q12_exact", p12)
fraction("Q12_d_aine", Fraction(1, 4))
fraction("Q12_d_sans", Fraction(1, 8))
fraction("Q12_d_deux", Fraction(1, 3))
V["Q12_nb_issues"] = sum("F" in f for f in f3)
# Variante citée dans la correction : un enfant choisi au hasard est une fille.
num = sum(Fraction(1, 8) * Fraction(1, 3) for f in f3 for i in range(3) if f[i] == "F" and f == ("F", "F", "F"))
den = sum(Fraction(1, 8) * Fraction(1, 3) for f in f3 for i in range(3) if f[i] == "F")
assert num / den == Fraction(1, 4)
filles = rng.integers(0, 2, (n_f, 3))  # 1 = fille
mc_proba("Q12", p12, filles[filles.sum(axis=1) >= 1].sum(axis=1) == 3)
vu3 = rng.integers(0, 3, n_f)
mc_proba("Q12_hasard", Fraction(1, 4), filles[filles[np.arange(n_f), vu3] == 1].sum(axis=1) == 3)

# Arithmétique citée dans les corrections.
assert Fraction(144 + 144 + 25 + 25, 50) == c2 == Fraction(169, 25)
assert Fraction(1, 4) * (Fraction(1, 4) + Fraction(1, 3)) == Fraction(7, 48)
assert Fraction(2, 3) * (Fraction(1, 8) + Fraction(3, 4)) == Fraction(7, 12) and Fraction(7, 12) / Fraction(5, 6) == Fraction(7, 10)
assert 3 * (1 - Fraction(1, 2)) ** 2 == Fraction(3, 4) and 3 * Fraction(2, 5) ** 2 == Fraction(12, 25)

Path(__file__).with_name("seance_math_5.out.json").write_text(json.dumps(V, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
dire(f"{len(V)} valeurs exportées dans verif/seance_math_5.out.json.")
dire("Toutes les vérifications passent.")
Path(__file__).with_name("seance_math_5.out.txt").write_text("\n".join(LIGNES) + "\n", encoding="utf-8", newline="\n")
