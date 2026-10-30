"""Vérification de tous les nombres de la séance de mathématiques 3 (leçon L-MATH-3 et questions M3-01 à M3-08).

Thèmes : estimateurs (biais, variance, erreur quadratique moyenne), maximum de vraisemblance, intervalles de
confiance, tests et p-values, tests multiples (Bonferroni, Benjamini-Hochberg), inégalités de Markov,
Tchebychev, Jensen et Cauchy-Schwarz.

Principe :
  1. chaque valeur citée dans la leçon ou les questions est calculée ici (fractions exactes, sympy, ou quantiles
     de scipy pour les lois normale et de Student) ;
  2. chaque formule démontrée dans la leçon est contrôlée (énumération exacte, intégrale symbolique) ;
  3. chaque probabilité ou espérance est aussi estimée par simulation Monte-Carlo, dont l'écart à la valeur
     exacte doit rester inférieur à 4 erreurs standard.
Les distracteurs des questions sont calculés ici aussi.

Graine fixe : la sortie est identique à chaque exécution.
Sorties : verif/seance_math_3.out.json (lu par outils/construire.py) et verif/seance_math_3.out.txt (ce résumé).
"""
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import stats

rng = np.random.default_rng(20261007)
V = {}
LIGNES = []


def dire(texte):
    print(texte)
    LIGNES.append(texte)


def fraction(nom, f):
    """Exporte une fraction : valeur décimale, numérateur et dénominateur (forme irréductible)."""
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


def benjamini_hochberg(pvals, q):
    """Nombre de rejets de la procédure de Benjamini-Hochberg (pas à pas montante), en fractions exactes."""
    p = sorted(Fraction(x) for x in pvals)
    m = len(p)
    k = 0
    for i in range(1, m + 1):
        if p[i - 1] <= Fraction(i) * Fraction(q) / m:
            k = i
    return k


def bh_vectorise(pv, q):
    """Benjamini-Hochberg sur chaque ligne d'une matrice de p-values ; renvoie la matrice booléenne des rejets."""
    n_sim, m = pv.shape
    ordre = np.argsort(pv, axis=1)
    tries = np.take_along_axis(pv, ordre, axis=1)
    ok = tries <= q * np.arange(1, m + 1) / m
    # k = plus grand indice i (1 à m) tel que p_(i) <= i q / m, 0 s'il n'y en a pas
    k = np.where(ok.any(axis=1), m - np.argmax(ok[:, ::-1], axis=1), 0)
    rang = np.empty_like(ordre)
    np.put_along_axis(rang, ordre, np.arange(1, m + 1)[None, :].repeat(n_sim, axis=0), axis=1)
    return rang <= k[:, None]


z_sym = sp.symbols("z", real=True)
phi = sp.exp(-z_sym**2 / 2) / sp.sqrt(2 * sp.pi)

# =====================================================================================
# LEÇON, PARTIE 1 : biais, variance, erreur quadratique moyenne
# =====================================================================================
# E[SC] = (n - 1) sigma^2, sans hypothèse de normalité : énumération exacte, loi uniforme sur {0, 1, 3}.
valeurs = [0, 1, 3]
mu_d = Fraction(sum(valeurs), len(valeurs))
s2_d = sum((Fraction(v) - mu_d) ** 2 for v in valeurs) / len(valeurs)
for n_ in (2, 3, 4, 5):
    total = Fraction(0)
    for tup in itertools.product(valeurs, repeat=n_):
        moy = Fraction(sum(tup), n_)
        total += sum((Fraction(v) - moy) ** 2 for v in tup)
    assert total / len(valeurs) ** n_ == (n_ - 1) * s2_d
dire("E[SC] = (n-1) sigma^2 vérifié par énumération exacte (loi uniforme sur {0, 1, 3}, n = 2 à 5).")

# Quatrième moment de la loi normale centrée réduite, et identité de Stein utilisée dans la preuve.
assert sp.integrate(z_sym**4 * phi, (z_sym, -sp.oo, sp.oo)) == 3
assert sp.integrate(z_sym**2 * phi, (z_sym, -sp.oo, sp.oo)) == 1
assert sp.simplify(sp.diff(phi, z_sym) + z_sym * phi) == 0  # phi'(z) = -z phi(z)
for k_ in (1, 4, 9):
    assert abs(stats.chi2(k_).var() - 2 * k_) < 1e-9 and abs(stats.chi2(k_).moment(2) - (2 * k_ + k_**2)) < 1e-9
dire("E[Z^4] = 3, Var(khi-deux_k) = 2k vérifiés.")

# Erreur quadratique de c * SC sous normalité, en unités de sigma^4 : f(c) = c^2 (2k + k^2) - 2ck + 1, k = n - 1.
c_s, k_s = sp.symbols("c k", positive=True)
f_eqm = c_s**2 * (2 * k_s + k_s**2) - 2 * c_s * k_s + 1
assert sp.simplify(f_eqm.subs(c_s, 1 / k_s) - 2 / k_s) == 0
assert sp.simplify(f_eqm.subs(c_s, 1 / (k_s + 1)) - (2 * k_s + 1) / (k_s + 1) ** 2) == 0
assert sp.simplify(f_eqm.subs(c_s, 1 / (k_s + 2)) - 2 / (k_s + 2)) == 0
assert sp.solve(sp.diff(f_eqm, c_s), c_s) == [1 / (k_s + 2)]
assert sp.simplify(sp.diff(f_eqm, c_s, 2) - 2 * (2 * k_s + k_s**2)) == 0
dire("EQM de c SC : formule, valeurs en 1/(n-1), 1/n, 1/(n+1) et minimum en 1/(n+1) vérifiés symboliquement.")


def eqm_variance(n, d):
    """EQM de SC / d en unités de sigma^4, sous normalité (exact, fractions)."""
    k = n - 1
    c = Fraction(1, d)
    return c * c * (2 * k + k * k) - 2 * c * k + 1


n1 = 5
V["L1_n"] = n1
fraction("L1_eqm_nm1", eqm_variance(n1, n1 - 1))
fraction("L1_eqm_n", eqm_variance(n1, n1))
fraction("L1_eqm_np1", eqm_variance(n1, n1 + 1))
assert V["L1_eqm_nm1"] == 0.5 and eqm_variance(n1, n1) == Fraction(2 * n1 - 1, n1 * n1)
fraction("L1_biais_n", Fraction(-1, n1))
x = rng.standard_normal((400_000, n1))
sc = ((x - x.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
mc_moyenne("L1_eqm_nm1", V["L1_eqm_nm1"], (sc / (n1 - 1) - 1) ** 2)
mc_moyenne("L1_eqm_n", V["L1_eqm_n"], (sc / n1 - 1) ** 2)
mc_moyenne("L1_eqm_np1", V["L1_eqm_np1"], (sc / (n1 + 1) - 1) ** 2)
mc_moyenne("L1_biais_n", V["L1_biais_n"], sc / n1 - 1)
mc_moyenne("L1_biais_nm1", 0, sc / (n1 - 1) - 1)

# =====================================================================================
# LEÇON, PARTIE 2 : maximum de vraisemblance
# =====================================================================================
th = sp.symbols("theta", positive=True)
# Bernoulli : 7 succès sur 20.
ell = 7 * sp.log(th) + 13 * sp.log(1 - th)
assert sp.solve(sp.diff(ell, th), th) == [sp.Rational(7, 20)]
V["L2_bern_n"] = 20
V["L2_bern_k"] = 7
fraction("L2_bern_hat", Fraction(7, 20))
V["L2_bern_se"] = math.sqrt(0.35 * 0.65 / 20)
# Information de Fisher d'une Bernoulli : E[score^2] = -E[dérivée seconde] = 1/(theta (1 - theta)).
xs = sp.symbols("x")
logf = xs * sp.log(th) + (1 - xs) * sp.log(1 - th)
score = sp.diff(logf, th)
info1 = th * score.subs(xs, 1) ** 2 + (1 - th) * score.subs(xs, 0) ** 2
info2 = -(th * sp.diff(logf, th, 2).subs(xs, 1) + (1 - th) * sp.diff(logf, th, 2).subs(xs, 0))
assert sp.simplify(info1 - 1 / (th * (1 - th))) == 0 and sp.simplify(info2 - 1 / (th * (1 - th))) == 0
# Variance exacte de la proportion : theta (1 - theta) / n ; Monte-Carlo pour theta = 0,35, n = 20.
props = rng.binomial(20, 0.35, size=400_000) / 20
c_ = props - props.mean()
var_mc = (c_**2).mean()
controle_mc("L2_bern_var", 0.35 * 0.65 / 20, var_mc, np.sqrt(((c_**2 - var_mc) ** 2).mean() / len(props)))

# Normale : données 2, 4, 4, 5, 10.
donnees = [2, 4, 4, 5, 10]
m_s, v_s = sp.symbols("m v", positive=True)
loglik = sum(-sp.log(2 * sp.pi * v_s) / 2 - (d - m_s) ** 2 / (2 * v_s) for d in donnees)
sol = sp.solve([sp.diff(loglik, m_s), sp.diff(loglik, v_s)], [m_s, v_s], dict=True)
assert sol == [{m_s: 5, v_s: sp.Rational(36, 5)}]
V["L2_norm_moy"] = 5
V["L2_norm_sc"] = sum((d - 5) ** 2 for d in donnees)
fraction("L2_norm_emv", Fraction(36, 5))
fraction("L2_norm_s2", Fraction(36, 4))
assert V["L2_norm_sc"] == 36
# Identité de décomposition : somme des (x_i - m)^2 = SC + n (xbar - m)^2.
assert sp.expand(sum((d - m_s) ** 2 for d in donnees) - (36 + 5 * (5 - m_s) ** 2)) == 0

# Exponentielle : E[1/S] = lambda / (n - 1) pour S somme de n exponentielles indépendantes de paramètre lambda.
lam, t_s, s_s, x_s = sp.symbols("lambda t s x", positive=True)
assert sp.simplify(sp.integrate(lam * sp.exp(-lam * x_s) * sp.exp(-t_s * x_s), (x_s, 0, sp.oo), conds="none") - lam / (lam + t_s)) == 0
for n_ in (2, 3, 5):
    via_laplace = sp.integrate((lam / (lam + t_s)) ** n_, (t_s, 0, sp.oo))
    densite_gamma = lam**n_ * s_s ** (n_ - 1) * sp.exp(-lam * s_s) / sp.factorial(n_ - 1)
    via_gamma = sp.integrate(densite_gamma / s_s, (s_s, 0, sp.oo))
    assert sp.simplify(via_laplace - lam / (n_ - 1)) == 0 and sp.simplify(via_gamma - lam / (n_ - 1)) == 0
dire("E[1/S] = lambda/(n-1) vérifié par la transformée de Laplace et par la densité Gamma (n = 2, 3, 5).")
# Information de Fisher d'une exponentielle : -E[dérivée seconde de ln f] = 1/lambda^2.
assert sp.simplify(-sp.diff(sp.log(lam) - lam * x_s, lam, 2) - 1 / lam**2) == 0
V["L2_exp_n"] = 3
fraction("L2_exp_facteur", Fraction(3, 2))
xe = rng.exponential(1.0, size=(400_000, 3))
mc_moyenne("L2_exp_esp", 1.5, 1 / xe.mean(axis=1))

# =====================================================================================
# LEÇON, PARTIE 3 : intervalles de confiance
# =====================================================================================
V["L3_z975"] = float(stats.norm.ppf(0.975))
V["L3_t9"] = float(stats.t.ppf(0.975, 9))
# Réflexe cité dans la leçon (constaté sur une grille, non démontré) : t_{k; u} > z_u pour u > 1/2.
for k_ in range(1, 201):
    for u_ in (0.6, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995):
        assert stats.t.ppf(u_, k_) > stats.norm.ppf(u_)
n3, xbar3, s3 = 10, 5.2, 0.9
V["L3_n"] = n3
V["L3_xbar"] = xbar3
V["L3_s"] = s3
V["L3_demi_t"] = V["L3_t9"] * s3 / math.sqrt(n3)
V["L3_bas"] = xbar3 - V["L3_demi_t"]
V["L3_haut"] = xbar3 + V["L3_demi_t"]
V["L3_demi_z"] = V["L3_z975"] * s3 / math.sqrt(n3)
# Couverture exacte de l'intervalle t sous normalité : Monte-Carlo.
x = rng.normal(3.0, 2.0, size=(200_000, n3))
m_ = x.mean(axis=1)
s_ = x.std(axis=1, ddof=1)
demi = V["L3_t9"] * s_ / np.sqrt(n3)
mc_proba("L3_couv_t", 0.95, np.abs(m_ - 3.0) <= demi)
# Intervalle de Wald pour une proportion : couverture exacte par la loi binomiale.


def couverture_wald(n, theta, z):
    tot = 0.0
    for k in range(n + 1):
        h = k / n
        if abs(h - theta) <= z * math.sqrt(h * (1 - h) / n):
            tot += math.comb(n, k) * theta**k * (1 - theta) ** (n - k)
    return tot


for theta_, nom in [(0.1, "L3_wald_01"), (0.5, "L3_wald_05")]:
    V[nom] = couverture_wald(20, theta_, V["L3_z975"])
    k_sim = rng.binomial(20, theta_, size=400_000)
    h_sim = k_sim / 20
    mc_proba(nom, V[nom], np.abs(h_sim - theta_) <= V["L3_z975"] * np.sqrt(h_sim * (1 - h_sim) / 20))
V["L3_wald_n"] = 20
# Pour theta = 0,1 et n = 20, probabilité de k = 0 (intervalle réduit au point 0, qui exclut 0,1).
V["L3_wald_k0"] = 0.9**20

# =====================================================================================
# LEÇON, PARTIE 4 : tests, p-values, puissance
# =====================================================================================
n4, xbar4, s4 = 16, 0.5, 1.2
t_obs = Fraction(1, 2) * 4 / Fraction(6, 5)
assert t_obs == Fraction(5, 3)
fraction("L4_t", t_obs)
V["L4_n"] = n4
V["L4_ddl"] = n4 - 1
V["L4_p"] = float(2 * stats.t.sf(float(t_obs), n4 - 1))
V["L4_p_z"] = float(2 * stats.norm.sf(float(t_obs)))
V["L4_t15"] = float(stats.t.ppf(0.975, 15))
x = rng.normal(0.0, 1.7, size=(400_000, n4))
tt = x.mean(axis=1) / (x.std(axis=1, ddof=1) / np.sqrt(n4))
mc_proba("L4_p", V["L4_p"], np.abs(tt) >= float(t_obs))
# Uniformité de la p-value sous H0 (test t bilatéral, n = 10).
x = rng.normal(0.0, 1.0, size=(400_000, 10))
tt = x.mean(axis=1) / (x.std(axis=1, ddof=1) / np.sqrt(10))
pv = 2 * stats.t.sf(np.abs(tt), 9)
mc_proba("L4_unif_005", 0.05, pv <= 0.05)
mc_proba("L4_unif_020", 0.20, pv <= 0.20)
# Puissance du test z unilatéral : delta / sigma = 0,5, n = 25, niveau 0,05.
V["L4_z95"] = float(stats.norm.ppf(0.95))
V["L4_puiss_n"] = 25
V["L4_puiss_arg"] = math.sqrt(25) * 0.5 - V["L4_z95"]
V["L4_puiss"] = float(stats.norm.cdf(V["L4_puiss_arg"]))
x = rng.normal(0.5, 1.0, size=(400_000, 25))
mc_proba("L4_puiss", V["L4_puiss"], np.sqrt(25) * x.mean(axis=1) >= V["L4_z95"])
# Lecture bayésienne : 10 % d'effets réels, puissance 0,8, niveau 0,05.
num = Fraction(9, 10) * Fraction(1, 20)
den = num + Fraction(1, 10) * Fraction(4, 5)
fraction("L4_bayes", num / den)
fraction("L4_bayes_num", num)
fraction("L4_bayes_den", den)
assert num / den == Fraction(9, 25)
h0 = rng.random(2_000_000) < 0.9
rejet = np.where(h0, rng.random(2_000_000) < 0.05, rng.random(2_000_000) < 0.8)
mc_proba("L4_bayes", num / den, h0[rejet])

# =====================================================================================
# LEÇON, PARTIE 5 : tests multiples
# =====================================================================================
p_lecon = [Fraction(4, 1000), Fraction(21, 1000), Fraction(28, 1000), Fraction(35, 1000), Fraction(1, 2)]
V["L5_m"] = len(p_lecon)
V["L5_k"] = benjamini_hochberg(p_lecon, Fraction(5, 100))
V["L5_seuils"] = [float(Fraction(i, 1) * Fraction(5, 100) / 5) for i in range(1, 6)]
V["L5_bonf_seuil"] = 0.05 / 5
V["L5_bonf"] = sum(p <= Fraction(5, 100) / 5 for p in p_lecon)
assert V["L5_k"] == 4 and V["L5_bonf"] == 1
assert p_lecon[1] > Fraction(2 * 5, 100 * 5)  # p_(2) dépasse son seuil, et est pourtant rejetée
# FDR de BH sous indépendance : m = 20, m0 = 15 nulles uniformes, 5 alternatives de statistique N(3, 1).
m5, m0, q5 = 20, 15, 0.10
V["L5_fdr_m"] = m5
V["L5_fdr_m0"] = m0
V["L5_fdr_q"] = q5
V["L5_fdr_theo"] = m0 * q5 / m5
n_sim = 200_000
pv = np.empty((n_sim, m5))
pv[:, :m0] = rng.random((n_sim, m0))
pv[:, m0:] = stats.norm.sf(rng.normal(3.0, 1.0, size=(n_sim, m5 - m0)))
rej = bh_vectorise(pv, q5)
v_ = rej[:, :m0].sum(axis=1)
r_ = rej.sum(axis=1)
mc_moyenne("L5_fdr", V["L5_fdr_theo"], v_ / np.maximum(r_, 1))
V["L5_fwer_bh"] = float((v_ >= 1).mean())  # constaté : FWER de BH dans ce cadre (plus grand que la FDR)
assert V["L5_fwer_bh"] >= (v_ / np.maximum(r_, 1)).mean()
# Contrôle de bh_vectorise contre la version exacte sur 200 lignes.
for ligne in range(200):
    assert rej[ligne].sum() == benjamini_hochberg(pv[ligne].tolist(), Fraction(1, 10))
# Hypothèse nulle globale (m0 = m) : FDR = FWER = q.
pv0 = rng.random((n_sim, m5))
rej0 = bh_vectorise(pv0, q5)
mc_proba("L5_fwer_global", q5, rej0.any(axis=1))

# =====================================================================================
# LEÇON, PARTIE 6 : inégalités
# =====================================================================================
V["L6_markov"] = 2 / 10
assert sp.integrate(sp.exp(-x_s / 2) / 2, (x_s, 10, sp.oo)) == sp.exp(-5)
V["L6_exp"] = math.exp(-5)
mc_proba("L6_exp", V["L6_exp"], rng.exponential(2.0, size=2_000_000) >= 10)
# Tchebychev contre TCL : epsilon = 0,1, delta = 0,05.
fraction("L6_cheb_n", Fraction(1, 4) / (Fraction(1, 100) * Fraction(5, 100)))
V["L6_tcl_brut"] = (V["L3_z975"] * 0.5 / 0.1) ** 2
V["L6_tcl_n"] = math.ceil(V["L6_tcl_brut"])
assert V["L6_cheb_n"] == 500 and V["L6_tcl_n"] == 97
# Jensen : E[S] pour n = 2 sous la loi normale de variance 1 : S = |X1 - X2| / sqrt(2), X1 - X2 de loi N(0, 2).
u_s = sp.symbols("u", real=True)
dens2 = sp.exp(-u_s**2 / 4) / sp.sqrt(4 * sp.pi)
e_s = sp.integrate(sp.Abs(u_s) * dens2, (u_s, -sp.oo, sp.oo)) / sp.sqrt(2)
assert sp.simplify(e_s - sp.sqrt(2 / sp.pi)) == 0
V["L6_es"] = float(sp.sqrt(2 / sp.pi))
x = rng.standard_normal((400_000, 2))
mc_moyenne("L6_es", V["L6_es"], x.std(axis=1, ddof=1))
# Cauchy-Schwarz : Var(X) = 4, Var(Y) = 9.
V["L6_cs"] = math.sqrt(4 * 9)

# =====================================================================================
# POINTS DE CONTRÔLE
# =====================================================================================
V["Cic_bas"] = V["L3_bas"]
# Point « ic », option d : P(|Xbar' - Xbar| <= t S / sqrt(n)) = P(|T_{n-1}| <= t / sqrt(2)) < 0,95 (n = 10).
assert 2 * stats.t.cdf(V["L3_t9"] / math.sqrt(2), 9) - 1 < 0.95
V["Cic_haut"] = V["L3_haut"]

# =====================================================================================
# QUESTIONS
# =====================================================================================
# M3-01 : estimateur somme / (n + 1), n = 9, mu = 2, sigma^2 = 4.
n_, mu_, s2_ = 9, Fraction(2), Fraction(4)
biais = mu_ * n_ / (n_ + 1) - mu_
var_ = n_ * s2_ / (n_ + 1) ** 2
fraction("Q01_biais", biais)
fraction("Q01_var", var_)
fraction("Q01_exact", var_ + biais**2)
fraction("Q01_d_xbar", s2_ / n_)
fraction("Q01_d_moins", var_ - biais**2)
fraction("Q01_d_abs", var_ + abs(biais))
fraction("Q01_biais_carre", biais**2)
assert var_ + biais**2 == Fraction(2, 5)
# Le diviseur 10 bat Xbar si et seulement si 36/100 + mu^2/100 < 4/9, soit mu^2 < 76/9.
fraction("Q01_mu2_seuil", 100 * (s2_ / n_ - var_))
assert 100 * (s2_ / n_ - var_) == Fraction(76, 9)
x = rng.normal(2.0, 2.0, size=(400_000, n_))
mc_moyenne("Q01", V["Q01_exact"], (x.sum(axis=1) / (n_ + 1) - 2.0) ** 2)
mc_moyenne("Q01_xbar", V["Q01_d_xbar"], (x.mean(axis=1) - 2.0) ** 2)

# M3-02 : EMV du paramètre d'une loi exponentielle sur six durées.
durees = [Fraction(2, 10), Fraction(5, 10), Fraction(6, 10), Fraction(9, 10), Fraction(11, 10), Fraction(15, 10)]
somme = sum(durees)
assert somme == Fraction(48, 10)
ell_e = 6 * sp.log(lam) - lam * sp.Rational(48, 10)
assert sp.solve(sp.diff(ell_e, lam), lam) == [sp.Rational(5, 4)]
assert sp.diff(ell_e, lam, 2).subs(lam, sp.Rational(5, 4)) < 0
fraction("Q02_somme", somme)
fraction("Q02_moy", somme / 6)
fraction("Q02_exact", 6 / somme)
fraction("Q02_d_sans_biais", 5 / somme)
mediane = (durees[2] + durees[3]) / 2
fraction("Q02_mediane", mediane)
V["Q02_d_mediane"] = math.log(2) / float(mediane)

# M3-03 : E[1/Xbar] pour n = 5, lambda = 2.
fraction("Q03_exact", Fraction(5 * 2, 4))
fraction("Q03_d_inverse", Fraction(4, 5) * 2)
fraction("Q03_d_moyenne", Fraction(1, 2))
xe = rng.exponential(0.5, size=(400_000, 5))
mc_moyenne("Q03", V["Q03_exact"], 1 / xe.mean(axis=1))

# M3-04 : demi-largeur de l'intervalle t, n = 9, s = 1,5.
V["Q04_t8"] = float(stats.t.ppf(0.975, 8))
V["Q04_t9"] = float(stats.t.ppf(0.975, 9))
V["Q04_t8_unilat"] = float(stats.t.ppf(0.95, 8))
V["Q04_erreur_type"] = 1.5 / 3
V["Q04_exact"] = V["Q04_t8"] * 0.5
V["Q04_d_z"] = V["L3_z975"] * 0.5
V["Q04_d_t9"] = V["Q04_t9"] * 0.5
V["Q04_d_unilat"] = V["Q04_t8_unilat"] * 0.5
V["Q04_bas"] = 10.4 - V["Q04_exact"]
V["Q04_haut"] = 10.4 + V["Q04_exact"]
assert len({round(V[k], 3) for k in ["Q04_exact", "Q04_d_z", "Q04_d_t9", "Q04_d_unilat"]}) == 4
# L'énoncé donne les quantiles arrondis à 3 décimales : les réponses arrondies n'en dépendent pas.
for cle_q, cle_r in [("Q04_t8", "Q04_exact"), ("L3_z975", "Q04_d_z"), ("Q04_t9", "Q04_d_t9"), ("Q04_t8_unilat", "Q04_d_unilat")]:
    assert round(round(V[cle_q], 3) * 0.5, 3) == round(V[cle_r], 3)
x = rng.normal(10.0, 1.5, size=(400_000, 9))
demi = V["Q04_t8"] * x.std(axis=1, ddof=1) / 3
mc_proba("Q04_couv", 0.95, np.abs(x.mean(axis=1) - 10.0) <= demi)
demi_z = V["L3_z975"] * x.std(axis=1, ddof=1) / 3
V["Q04_couv_z"] = float(stats.t.cdf(V["L3_z975"], 8) - stats.t.cdf(-V["L3_z975"], 8))  # couverture exacte avec z
mc_proba("Q04_couv_z", V["Q04_couv_z"], np.abs(x.mean(axis=1) - 10.0) <= demi_z)

# M3-05 : 5 % d'effets réels, puissance 0,5, niveau 0,05.
num5 = Fraction(95, 100) * Fraction(5, 100)
den5 = num5 + Fraction(5, 100) * Fraction(1, 2)
fraction("Q05_exact", num5 / den5)
fraction("Q05_num", num5)
fraction("Q05_vrais", Fraction(5, 100) * Fraction(1, 2))
fraction("Q05_den", den5)
fraction("Q05_d_complement", 1 - num5 / den5)
assert num5 / den5 == Fraction(19, 29)
h0 = rng.random(2_000_000) < 0.95
rejet = np.where(h0, rng.random(2_000_000) < 0.05, rng.random(2_000_000) < 0.5)
mc_proba("Q05", num5 / den5, h0[rejet])

# M3-06 : Benjamini-Hochberg, 8 p-values, q = 0,10.
p06 = [Fraction(20, 100), Fraction(34, 1000), Fraction(80, 100), Fraction(1, 1000), Fraction(70, 1000),
       Fraction(45, 100), Fraction(48, 1000), Fraction(30, 1000)]
q06 = Fraction(1, 10)
V["Q06_exact"] = benjamini_hochberg(p06, q06)
V["Q06_seuils"] = [float(i * q06 / 8) for i in range(1, 9)]
V["Q06_tries"] = [float(p) for p in sorted(p06)]
tries06 = sorted(p06)
premier_echec = next(i for i in range(1, 9) if tries06[i - 1] > i * q06 / 8)
V["Q06_d_descendant"] = premier_echec - 1
V["Q06_d_individuel"] = sum(tries06[i - 1] <= i * q06 / 8 for i in range(1, 9))
V["Q06_d_brut"] = sum(p <= q06 for p in p06)
V["Q06_d_bonf"] = sum(p <= q06 / 8 for p in p06)
V["Q06_bonf_seuil"] = float(q06 / 8)
assert V["Q06_exact"] == 4 and V["Q06_d_descendant"] == 1 and V["Q06_d_individuel"] == 3
assert V["Q06_d_brut"] == 5 and V["Q06_d_bonf"] == 1

# M3-07 : diviseur optimal de SC sous normalité, n = 10.
n7 = 10
eqms = {d: eqm_variance(n7, d) for d in range(5, 20)}
assert min(eqms, key=eqms.get) == n7 + 1
for d in (9, 10, 11, 12):
    fraction(f"Q07_eqm_{d}", eqms[d])
V["Q07_exact"] = n7 + 1
V["Q07_d_sans_biais"] = n7 - 1
V["Q07_d_emv"] = n7
V["Q07_d_khi_n"] = n7 + 2
V["Q07_k"] = n7 - 1
assert eqms[11] < eqms[12] < eqms[10] < eqms[9]
# EQM si l'on croit SC de loi khi-deux à n degrés de liberté (moyenne connue) : minimum en 1/(n+2).
f_faux = c_s**2 * (2 * n7 + n7**2) - 2 * c_s * n7 + 1
assert sp.solve(sp.diff(f_faux, c_s), c_s) == [sp.Rational(1, n7 + 2)]
x = rng.standard_normal((400_000, n7))
sc = ((x - x.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
for d in (9, 10, 11, 12):
    mc_moyenne(f"Q07_eqm_{d}", eqms[d], (sc / d - 1) ** 2)

# M3-08 : taille d'échantillon par Tchebychev, epsilon = 0,05, delta = 0,05.
fraction("Q08_exact", Fraction(1, 4) / (Fraction(1, 400) * Fraction(1, 20)))
fraction("Q08_d_sigma1", Fraction(1) / (Fraction(1, 400) * Fraction(1, 20)))
fraction("Q08_d_eps", Fraction(1, 4) / (Fraction(1, 20) * Fraction(1, 20)))
V["Q08_tcl_brut"] = (V["L3_z975"] * 0.5 / 0.05) ** 2
V["Q08_d_tcl"] = math.ceil(V["Q08_tcl_brut"])
assert V["Q08_exact"] == 2000 and V["Q08_d_tcl"] == 385 and V["Q08_d_sigma1"] == 8000 and V["Q08_d_eps"] == 100
# Le pire cas theta = 1/2 : la probabilité réelle d'écart pour n = 2000, exacte par la loi binomiale.
V["Q08_p_reelle"] = float(stats.binom.cdf(900, 2000, 0.5) + stats.binom.sf(1099, 2000, 0.5))
assert 1e-6 < V["Q08_p_reelle"] < 1e-4  # « de l'ordre de 10^-5 » dans la correction
# Probabilité réelle au seuil TCL n = 385, theta = 1/2 : P(|K/385 - 1/2| >= 0,05).
bas8 = math.floor(385 * 0.45 + 1e-9)
haut8 = math.ceil(385 * 0.55 - 1e-9)
V["Q08_p_tcl"] = float(stats.binom.cdf(bas8, 385, 0.5) + stats.binom.sf(haut8 - 1, 385, 0.5))
k8 = rng.binomial(385, 0.5, size=1_000_000)
mc_proba("Q08_p_tcl", V["Q08_p_tcl"], np.abs(k8 / 385 - 0.5) >= 0.05 - 1e-12)

Path(__file__).with_name("seance_math_3.out.json").write_text(json.dumps(V, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
dire(f"{len(V)} valeurs exportées dans verif/seance_math_3.out.json.")
dire("Toutes les vérifications passent.")
Path(__file__).with_name("seance_math_3.out.txt").write_text("\n".join(LIGNES) + "\n", encoding="utf-8", newline="\n")
