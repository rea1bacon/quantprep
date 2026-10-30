"""Vérification de tous les nombres de la séance de ML 1 (leçon L-REGLIN-1 et questions F1-01 à F1-12).

Principe :
  1. chaque nombre affiché est calculé exactement (fractions, ou algèbre exacte avec sympy) ;
  2. chaque espérance ou variance est en plus estimée par simulation Monte-Carlo, et l'écart à l'exact doit rester
     inférieur à 4 erreurs standard ;
  3. chaque affirmation de ML de la leçon est contrôlée sur des données (identités algébriques sur des données
     aléatoires, conventions de scikit-learn, conditions d'optimalité du Lasso).
Les distracteurs et l'arithmétique citée dans les corrections sont aussi calculés ici.

Graine fixe : la sortie est identique à chaque exécution.
Sorties : verif/seance_ml_1.out.json (lu par outils/construire.py) et verif/seance_ml_1.out.txt (ce résumé).
"""
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp
from sklearn.linear_model import Lasso, Ridge

rng = np.random.default_rng(20261006)
V = {}
LIGNES = []


def dire(texte):
    print(texte)
    LIGNES.append(texte)


def fr(x):
    """Convertit un rationnel sympy (ou un entier, une Fraction) en Fraction."""
    x = sp.nsimplify(x) if not isinstance(x, (int, Fraction)) else x
    if isinstance(x, (int, Fraction)):
        return Fraction(x)
    assert x.is_rational, x
    return Fraction(int(x.p), int(x.q))


def fraction(nom, f):
    """Exporte une fraction : valeur décimale, numérateur et dénominateur (forme irréductible)."""
    f = fr(f)
    V[nom] = float(f)
    V[nom + "_num"] = f.numerator
    V[nom + "_den"] = f.denominator


def controle_mc(nom, exact, estimation, erreur_std, seuil=4.0):
    """Vérifie que l'estimation Monte-Carlo est à moins de `seuil` erreurs standard de la valeur exacte."""
    z = (float(estimation) - float(exact)) / float(erreur_std)
    assert abs(z) < seuil, f"{nom} : exact={float(exact)}, MC={estimation}, z={z}"
    V[nom + "_mc"] = float(estimation)
    dire(f"{nom:26s} exact={float(exact):.6f}  MC={float(estimation):.6f}  se={float(erreur_std):.6f}  z={z:+.2f}")


def mc_moyenne(nom, exact, echantillon):
    echantillon = np.asarray(echantillon, dtype=float)
    controle_mc(nom, exact, echantillon.mean(), echantillon.std(ddof=1) / np.sqrt(len(echantillon)))


def mc_variance(nom, exact, echantillon):
    """Variance empirique ; erreur standard par la méthode delta, estimée sur l'échantillon."""
    echantillon = np.asarray(echantillon, dtype=float)
    c = echantillon - echantillon.mean()
    v = (c**2).mean()
    controle_mc(nom, exact, v, np.sqrt(((c**2 - v) ** 2).mean() / len(echantillon)))


def mat(lignes):
    return sp.Matrix([[sp.Rational(Fraction(v).numerator, Fraction(v).denominator) for v in ligne] for ligne in lignes])


# =====================================================================================
# LEÇON, PARTIE 1 : moindres carrés, exemple à 4 points
# =====================================================================================
x1 = [0, 1, 2, 3]
y1 = [1, 3, 2, 5]
X = sp.Matrix([[1, v] for v in x1])
y = sp.Matrix(y1)
XtX = X.T * X
Xty = X.T * y
assert XtX == sp.Matrix([[4, 6], [6, 14]]) and Xty == sp.Matrix([11, 22])
det = XtX.det()
adj = XtX.adjugate()
inv = XtX.inv()
assert inv == adj / det
b = inv * Xty
yhat = X * b
e = y - yhat
H = X * inv * X.T
assert H.T == H and H * H == H and H.trace() == 2
assert X.T * e == sp.zeros(2, 1) and sum(e) == 0
assert (yhat.T * e)[0] == 0  # résidus orthogonaux aux valeurs ajustées
assert (y.T * e)[0] == (e.T * e)[0] != 0  # mais pas à y : y^T e = ||e||^2
b_np = np.linalg.lstsq(np.array(X, dtype=float), np.array(y1, dtype=float), rcond=None)[0]
assert np.allclose(b_np, [float(v) for v in b])
V["L1_n"], V["L1_p"] = 4, 2
V["L1_xtx11"], V["L1_xtx12"], V["L1_xtx22"] = int(XtX[0, 0]), int(XtX[0, 1]), int(XtX[1, 1])
V["L1_xty1"], V["L1_xty2"] = int(Xty[0]), int(Xty[1])
V["L1_det"] = int(det)
V["L1_adj11"], V["L1_adj12"], V["L1_adj22"] = int(adj[0, 0]), int(adj[0, 1]), int(adj[1, 1])
V["L1_adjxty1"], V["L1_adjxty2"] = int((adj * Xty)[0]), int((adj * Xty)[1])
fraction("L1_b0", b[0])
fraction("L1_b1", b[1])
for i in range(4):
    V[f"L1_yhat{i + 1}"] = float(yhat[i])
    V[f"L1_e{i + 1}"] = float(e[i])
V["L1_yhat"] = [float(v) for v in yhat]
V["L1_e"] = [float(v) for v in e]
V["L1_trH"] = int(H.trace())
fraction("L1_inv11", inv[0, 0])
fraction("L1_inv12", inv[0, 1])
fraction("L1_inv22", inv[1, 1])
V["L1_somme_e"] = float(sum(e))
V["L1_xe"] = float((X[:, 1].T * e)[0])
V["L1_ye"] = float((y.T * e)[0])
dire("Partie 1 : équations normales, matrice chapeau (symétrique, idempotente, trace 2), orthogonalité des résidus vérifiées en rationnels.")

# Deux colonnes égales : infinité de minimiseurs, mêmes valeurs ajustées.
Xd = sp.Matrix([[1, 1], [2, 2], [3, 3]])
yd = sp.Matrix([1, 2, 2])
assert Xd.rank() == 1 and (Xd.T * Xd).det() == 0
t_ = sp.symbols("t")
b_par = sp.Matrix([t_, sp.Rational(11, 14) - t_])  # toutes les solutions des équations normales
assert sp.simplify(Xd.T * Xd * b_par - Xd.T * yd) == sp.zeros(2, 1)
assert sp.simplify(Xd * b_par - Xd * b_par.subs(t_, 0)) == sp.zeros(3, 1)
# numpy.linalg.lstsq renvoie la solution de norme minimale : ici t = 11/28, coefficients égaux.
b_ls = np.linalg.lstsq(np.array(Xd, dtype=float), np.array(yd, dtype=float).ravel(), rcond=None)[0]
assert np.allclose(b_ls, [11 / 28, 11 / 28])

# =====================================================================================
# LEÇON, PARTIE 2 : modèle linéaire, hypothèses et violations
# =====================================================================================
scr = (e.T * e)[0]
s2 = scr / (4 - 2)
fraction("L2_scr", scr)
fraction("L2_s2", s2)
fraction("L2_var_pente", s2 * inv[1, 1])
V["L2_se_pente"] = math.sqrt(float(s2 * inv[1, 1]))
V["L2_t"] = float(b[1]) / V["L2_se_pente"]

# Sans biais, variance sigma^2 (X^T X)^{-1}, E[||e||^2] = sigma^2 (n - p) : simulation sur le plan à 4 points.
Xn = np.array(X, dtype=float)
beta_vrai = np.array([1.0, 2.0])
sigma = 1.5
nsim = 400_000
E_ = sigma * rng.standard_normal((nsim, 4))
Y_ = Xn.dot(beta_vrai) + E_
B_ = np.linalg.solve(Xn.T.dot(Xn), Xn.T.dot(Y_.T)).T
mc_moyenne("L2_sans_biais_pente", 2.0, B_[:, 1])
mc_variance("L2_var_pente_formule", sigma**2 * float(inv[1, 1]), B_[:, 1])
res_ = Y_ - B_.dot(Xn.T)
mc_moyenne("L2_esp_scr", sigma**2 * (4 - 2), (res_**2).sum(axis=1))

# Gauss-Markov : pour une matrice C quelconque avec C X = I, C C^T - (X^T X)^{-1} est semi-définie positive.
for _ in range(200):
    n_, p_ = 12, 3
    Xa = rng.standard_normal((n_, p_))
    D = rng.standard_normal((p_, n_))
    D = D - D.dot(Xa).dot(np.linalg.inv(Xa.T.dot(Xa))).dot(Xa.T)  # D X = 0
    C = np.linalg.inv(Xa.T.dot(Xa)).dot(Xa.T) + D
    assert np.allclose(C.dot(Xa), np.eye(p_))
    assert np.linalg.eigvalsh(C.dot(C.T) - np.linalg.inv(Xa.T.dot(Xa))).min() > -1e-10
dire("Gauss-Markov : C C^T - (X^T X)^{-1} semi-définie positive pour 200 estimateurs linéaires sans biais aléatoires.")
# Point de contrôle « gauss_markov » : un estimateur non linéaire sans biais peut faire mieux sans normalité.
# Modèle Y_i = beta + eps_i, eps_i uniformes sur [-1, 1], n = 10 : moyenne contre milieu de l'étendue.
n_c2 = 10
fraction("C2_var_moy", Fraction(1, 3) / n_c2)  # Var(uniforme sur [-1, 1]) = 1/3
u_, v_ = sp.symbols("u v")
dens_uv = sp.Rational(n_c2 * (n_c2 - 1), 2**n_c2) * (v_ - u_) ** (n_c2 - 2)  # densité jointe (min, max) sur [-1, 1]
var_mil = sp.integrate(sp.integrate(((u_ + v_) / 2) ** 2 * dens_uv, (u_, -1, v_)), (v_, -1, 1))
assert sp.integrate(sp.integrate(dens_uv, (u_, -1, v_)), (v_, -1, 1)) == 1
assert var_mil == sp.Rational(2, (n_c2 + 1) * (n_c2 + 2))
fraction("C2_var_mil", var_mil)
Uc = rng.uniform(-1, 1, (400_000, n_c2))
mc_variance("C2_var_moy", Fraction(1, 30), Uc.mean(axis=1))
mc_variance("C2_var_mil", var_mil, (Uc.min(axis=1) + Uc.max(axis=1)) / 2)
mc_moyenne("C2_mil_sans_biais", 0, (Uc.min(axis=1) + Uc.max(axis=1)) / 2)

# Frisch-Waugh et facteur d'inflation de la variance : [(X^T X)^{-1}]_jj = 1 / ||x_j tilde||^2 = 1 / (SCT_j (1 - R_j^2)).
for _ in range(100):
    n_ = 30
    Xa = np.column_stack([np.ones(n_), rng.standard_normal((n_, 3))])
    Xa[:, 2] += 2 * Xa[:, 1]
    ya = rng.standard_normal(n_)
    j = 2
    autres = np.delete(Xa, j, axis=1)
    tilde = Xa[:, j] - autres.dot(np.linalg.lstsq(autres, Xa[:, j], rcond=None)[0])
    sct_j = ((Xa[:, j] - Xa[:, j].mean()) ** 2).sum()
    r2_j = 1 - tilde.dot(tilde) / sct_j
    assert np.isclose(np.linalg.inv(Xa.T.dot(Xa))[j, j], 1 / tilde.dot(tilde))
    assert np.isclose(1 / tilde.dot(tilde), 1 / (sct_j * (1 - r2_j)))
    b_full = np.linalg.lstsq(Xa, ya, rcond=None)[0]
    assert np.isclose(b_full[j], tilde.dot(ya) / tilde.dot(tilde))
dire("Frisch-Waugh et formule du facteur d'inflation de la variance vérifiés sur 100 jeux aléatoires.")
V["L2_vif_r2"] = 0.99
V["L2_vif"] = 1 / (1 - 0.99)
V["L2_vif_se"] = math.sqrt(V["L2_vif"])
assert abs(V["L2_vif"] - 100) < 1e-9 and abs(V["L2_vif_se"] - 10) < 1e-9

# Variable omise : E[b] = beta + gamma (X^T X)^{-1} X^T z ; vérifié par simulation (même plan que la leçon).
n_ = 100
xo = rng.standard_normal(n_)
zo = 0.5 * xo + rng.standard_normal(n_)
Xo = np.column_stack([np.ones(n_), xo])
delta = np.linalg.lstsq(Xo, zo, rcond=None)[0][1]
Yo = 1 + 2 * xo + 3 * zo + rng.standard_normal((100_000, n_))
pentes_o = np.linalg.lstsq(Xo, Yo.T, rcond=None)[0][1]
mc_moyenne("L2_omise", 2 + 3 * delta, pentes_o)

# Erreur de mesure : pente limite beta Var(Z) / (Var(Z) + Var(U)) = 2 / 1,25.
fraction("L2_att", Fraction(2) * 1 / (1 + Fraction(1, 4)))
pentes_m = []
for _ in range(200):
    Z = rng.standard_normal(50_000)
    Vobs = Z + 0.5 * rng.standard_normal(50_000)
    Ym = 2 * Z + rng.standard_normal(50_000)
    pentes_m.append(Vobs.dot(Ym) / Vobs.dot(Vobs))
# La moyenne des pentes sur n = 50 000 approche la limite ; l'écart de biais en 1/n est très inférieur à l'erreur standard.
mc_moyenne("L2_att", V["L2_att"], pentes_m)

# Hétéroscédasticité (même plan que l'exemple de la leçon) : erreur-type usuelle trop petite, White proche du vrai.
n_ = 200
xh = rng.uniform(0, 1, n_)
Xh = np.column_stack([np.ones(n_), xh])
ecart = 0.1 + 4 * np.abs(xh - 0.5)
XtX_inv_h = np.linalg.inv(Xh.T.dot(Xh))
Yh = 1 + 2 * xh + ecart * rng.standard_normal((20_000, n_))
Bh = XtX_inv_h.dot(Xh.T.dot(Yh.T)).T
Rh = Yh - Bh.dot(Xh.T)
se_us = np.sqrt((Rh**2).sum(axis=1) / (n_ - 2) * XtX_inv_h[1, 1])
viande = np.einsum("si,ij,ik->sjk", Rh**2, Xh, Xh)
se_white = np.sqrt(np.einsum("j,sjk,k->s", XtX_inv_h[1], viande, XtX_inv_h[1]))
vrai_sd = math.sqrt(XtX_inv_h[1].dot((Xh * ecart[:, None] ** 2).T.dot(Xh)).dot(XtX_inv_h[1]))
mc_variance("L2_hetero_var", vrai_sd**2, Bh[:, 1])
assert se_us.mean() < 0.85 * vrai_sd and abs(se_white.mean() / vrai_sd - 1) < 0.05
dire(f"Hétéroscédasticité : écart-type exact {vrai_sd:.4f}, erreur-type usuelle moyenne {se_us.mean():.4f}, White {se_white.mean():.4f}.")

# Autocorrélation : Var vraie = sigma^2 / ||xc||^2 * (1 + 2 sum_h phi^h r_h) ; approximation (1 + phi rho) / (1 - phi rho).
phi = 0.8
V["L2_ac_phi"] = phi
V["L2_ac_rap_var"] = (1 + phi * phi) / (1 - phi * phi)
V["L2_ac_rap_se"] = math.sqrt(V["L2_ac_rap_var"])


def ar1(nsim, n, coef, generateur):
    u = generateur.standard_normal((nsim, n))
    z = np.empty((nsim, n))
    z[:, 0] = u[:, 0] / np.sqrt(1 - coef**2)
    for t in range(1, n):
        z[:, t] = coef * z[:, t - 1] + u[:, t]
    return z


n_ = 200
xa_ = ar1(1, n_, phi, rng)[0]
xc = xa_ - xa_.mean()
sig2 = 1 / (1 - phi**2)
cov = sig2 * phi ** np.abs(np.subtract.outer(np.arange(n_), np.arange(n_)))
var_exacte = xc.dot(cov).dot(xc) / xc.dot(xc) ** 2
r_h = np.array([xc[:-h].dot(xc[h:]) / xc.dot(xc) for h in range(1, n_)])
assert np.isclose(var_exacte, sig2 / xc.dot(xc) * (1 + 2 * np.sum(phi ** np.arange(1, n_) * r_h)))
eps_ = ar1(40_000, n_, phi, rng)
mc_variance("L2_ac_var", var_exacte, eps_.dot(xc) / xc.dot(xc))
assert var_exacte / (sig2 / xc.dot(xc)) > 1.5
dire(f"Autocorrélation : rapport des variances exact pour cette série {var_exacte / (sig2 / xc.dot(xc)):.3f}, approximation {V['L2_ac_rap_var']:.3f}.")

# =====================================================================================
# LEÇON, PARTIE 3 : R^2
# =====================================================================================
ybar = Fraction(sum(y1), 4)
sct = sum((Fraction(v) - ybar) ** 2 for v in y1)
sce = sum((fr(v) - ybar) ** 2 for v in yhat)
assert sct == sce + fr(scr)
r2 = 1 - fr(scr) / sct
r2aj = 1 - (fr(scr) / (4 - 2)) / (sct / (4 - 1))
corr = np.corrcoef(np.array(y1, float), np.array(yhat, float).ravel())[0, 1]
assert abs(corr**2 - float(r2)) < 1e-12
fraction("L3_ybar", ybar)
fraction("L3_sct", sct)
fraction("L3_sce", sce)
fraction("L3_scr", scr)
fraction("L3_r2", r2)
fraction("L3_r2aj", r2aj)
# Sans constante : x = (1, 2, 3), y = (3, 2, 1).
xn = [1, 2, 3]
yn = [3, 2, 1]
bn = Fraction(sum(a * c for a, c in zip(xn, yn)), sum(a * a for a in xn))
scrn = sum((c - bn * a) ** 2 for a, c in zip(xn, yn))
sctn = sum((c - Fraction(sum(yn), 3)) ** 2 for c in yn)
fraction("L3_nc_b", bn)
fraction("L3_nc_scr", scrn)
fraction("L3_nc_sct", sctn)
fraction("L3_nc_r2", 1 - scrn / sctn)
V["L3_nc_xy"] = sum(a * c for a, c in zip(xn, yn))
V["L3_nc_xx"] = sum(a * a for a in xn)
assert 1 - scrn / sctn < 0
# Identité : SCR_avant = SCR_après (1 + t^2 / (n - p - 1)) et R^2 ajusté croît si et seulement si t^2 > 1.
for _ in range(300):
    n_, p_ = 25, 3
    Xa = np.column_stack([np.ones(n_), rng.standard_normal((n_, p_ - 1))])
    xnew = rng.standard_normal(n_)
    ya = Xa[:, 1] + rng.standard_normal(n_) * rng.uniform(0.2, 3)
    ra = ya - Xa.dot(np.linalg.lstsq(Xa, ya, rcond=None)[0])
    Xb = np.column_stack([Xa, xnew])
    bb = np.linalg.lstsq(Xb, ya, rcond=None)[0]
    rb = ya - Xb.dot(bb)
    s2b = rb.dot(rb) / (n_ - p_ - 1)
    tstat = bb[-1] / math.sqrt(s2b * np.linalg.inv(Xb.T.dot(Xb))[-1, -1])
    assert np.isclose(ra.dot(ra), rb.dot(rb) * (1 + tstat**2 / (n_ - p_ - 1)))
    assert rb.dot(rb) <= ra.dot(ra) + 1e-12
    sct_a = ((ya - ya.mean()) ** 2).sum()
    aj_a = 1 - (ra.dot(ra) / (n_ - p_)) / (sct_a / (n_ - 1))
    aj_b = 1 - (rb.dot(rb) / (n_ - p_ - 1)) / (sct_a / (n_ - 1))
    assert (aj_b > aj_a) == (tstat**2 > 1)
dire("R^2 : ne décroît jamais ; R^2 ajusté croît si et seulement si t^2 > 1 (300 jeux aléatoires).")

# Expérience de la leçon (colonnes de bruit) : même code, pour contrôler le sens des conclusions.
g_ = np.random.default_rng(6)
x_app = g_.standard_normal(30)
y_app = 1 + x_app + g_.standard_normal(30)
x_tst = g_.standard_normal(1000)
y_tst = 1 + x_tst + g_.standard_normal(1000)
bruit_a = g_.standard_normal((30, 25))
bruit_t = g_.standard_normal((1000, 25))
r2s, r2ts = [], []
for k in [0, 5, 15, 25]:
    Xa = np.column_stack([np.ones(30), x_app, bruit_a[:, :k]])
    Xt = np.column_stack([np.ones(1000), x_tst, bruit_t[:, :k]])
    ba = np.linalg.lstsq(Xa, y_app, rcond=None)[0]
    r2s.append(1 - ((y_app - Xa.dot(ba)) ** 2).sum() / ((y_app - y_app.mean()) ** 2).sum())
    r2ts.append(1 - ((y_tst - Xt.dot(ba)) ** 2).sum() / ((y_tst - y_tst.mean()) ** 2).sum())
assert all(a <= c for a, c in zip(r2s, r2s[1:])) and r2ts[-1] < 0
dire("Expérience R^2 de la leçon : R^2 d'apprentissage croissant, R^2 de test négatif avec 25 colonnes de bruit.")

# =====================================================================================
# LEÇON, PARTIE 4 : Ridge
# =====================================================================================
G = mat([[1, Fraction(9, 10)], [Fraction(9, 10), 1]])
c = mat([[1], [Fraction(4, 5)]])
lam = sp.Rational(1, 10)
ols = G.inv() * c
rdg = (G + lam * sp.eye(2)).inv() * c
mu1, mu2 = sp.Rational(19, 10), sp.Rational(1, 10)
assert set(G.eigenvals()) == {mu1, mu2}
assert G * sp.Matrix([1, 1]) == mu1 * sp.Matrix([1, 1]) and G * sp.Matrix([1, -1]) == mu2 * sp.Matrix([1, -1])
a_, b_ = sp.Rational(9, 10), sp.Rational(1, 10)
assert a_ * sp.Matrix([1, 1]) + b_ * sp.Matrix([1, -1]) == c
assert ols == (a_ / mu1) * sp.Matrix([1, 1]) + (b_ / mu2) * sp.Matrix([1, -1])
assert rdg == (a_ / (mu1 + lam)) * sp.Matrix([1, 1]) + (b_ / (mu2 + lam)) * sp.Matrix([1, -1])
fraction("L4_mu1", mu1)
fraction("L4_mu2", mu2)
fraction("L4_a", a_)
fraction("L4_b", b_)
fraction("L4_ols1", ols[0])
fraction("L4_ols2", ols[1])
fraction("L4_rdg1", rdg[0])
fraction("L4_rdg2", rdg[1])
fraction("L4_ols_a", a_ / mu1)
fraction("L4_ols_b", b_ / mu2)
fraction("L4_rdg_a", a_ / (mu1 + lam))
fraction("L4_rdg_b", b_ / (mu2 + lam))
fraction("L4_f1", mu1 / (mu1 + lam))
fraction("L4_f2", mu2 / (mu2 + lam))
fraction("L4_mu1l", mu1 + lam)
fraction("L4_mu2l", mu2 + lam)
fraction("L4_kappa0", mu1 / mu2)
fraction("L4_kappa", (mu1 + lam) / (mu2 + lam))
fraction("L4_lam", lam)
Gn = np.array(G, dtype=float)
Xg = np.linalg.cholesky(Gn).T  # X^T X = G
yg = np.linalg.solve(Xg.T, np.array(c, dtype=float).ravel())  # X^T y = c
assert np.allclose(Xg.T.dot(Xg), Gn) and np.allclose(Xg.T.dot(yg), np.array(c, dtype=float).ravel())
assert np.allclose(Ridge(alpha=0.1, fit_intercept=False).fit(Xg, yg).coef_, np.array(rdg, dtype=float).ravel(), atol=1e-10)
dire("Ridge : exemple 2 x 2 exact, décomposition propre et scikit-learn (Ridge(alpha=lambda)) concordants.")

# Point de contrôle « conditionnement ».
V["C5_kappa0"] = 9 / 0.01
V["C5_kappa"] = (9 + 1) / (0.01 + 1)
V["C5_d_petit"] = 9 / (0.01 + 1)
V["C5_d_racine"] = math.sqrt(9 / 0.01)

# Ridge ne met jamais un coefficient exactement à zéro (sauf colonne nulle) ; Lasso oui.
for _ in range(50):
    Xa = rng.standard_normal((40, 8))
    ya = rng.standard_normal(40)
    for lam_ in [0.1, 10.0, 1e3, 1e6]:
        br = np.linalg.solve(Xa.T.dot(Xa) + lam_ * np.eye(8), Xa.T.dot(ya))
        assert np.all(br != 0)
        Aj = np.linalg.solve(Xa.T.dot(Xa) + lam_ * np.eye(8), Xa.T)
        assert np.all(np.abs(Aj).sum(axis=1) > 0)
Xz = rng.standard_normal((20, 3))
Xz[:, 1] = 0
bz = np.linalg.solve(Xz.T.dot(Xz) + 2.0 * np.eye(3), Xz.T.dot(rng.standard_normal(20)))
assert bz[1] == 0 and bz[0] != 0 and bz[2] != 0
# Interprétation bayésienne : gradient de -log a posteriori nul en la solution Ridge avec lambda = sigma^2 / tau^2.
s_, t2_ = 0.7, 1.3
Xa = rng.standard_normal((30, 4))
ya = rng.standard_normal(30)
br = np.linalg.solve(Xa.T.dot(Xa) + (s_**2 / t2_**2) * np.eye(4), Xa.T.dot(ya))
grad = -Xa.T.dot(ya - Xa.dot(br)) / s_**2 + br / t2_**2
assert np.allclose(grad, 0, atol=1e-10)
dire("Ridge : aucun zéro exact (50 jeux, 4 valeurs de lambda), zéro exact pour une colonne nulle ; MAP gaussien = Ridge.")

# =====================================================================================
# LEÇON, PARTIE 5 : Lasso
# =====================================================================================


def seuil_doux(z, l):
    return np.sign(z) * np.maximum(np.abs(z) - l, 0)


def seuil_doux_fr(z, l):
    return [(1 if v > 0 else -1) * max(abs(v) - l, Fraction(0)) for v in z]


z5 = [Fraction(3, 2), Fraction(-3, 10), Fraction(7, 10)]
l5 = Fraction(1, 2)
st5 = seuil_doux_fr(z5, l5)
rd5 = [v / (1 + l5) for v in z5]
V["L5_z"] = [float(v) for v in z5]
V["L5_lam"] = float(l5)
V["L5_lasso"] = [float(v) for v in st5]
V["L5_ridge"] = [float(v) for v in rd5]
# Vérification par scikit-learn sur une matrice à colonnes orthonormées : Lasso(alpha = lambda / n).
n_ = 20
Q, _ = np.linalg.qr(rng.standard_normal((n_, 3)))
y5 = Q.dot(np.array(V["L5_z"])) + (np.eye(n_) - Q.dot(Q.T)).dot(rng.standard_normal(n_))
assert np.allclose(Q.T.dot(y5), V["L5_z"])
l5_sk = Lasso(alpha=0.5 / n_, fit_intercept=False, tol=1e-14, max_iter=100_000).fit(Q, y5).coef_
assert np.allclose(l5_sk, V["L5_lasso"], atol=1e-9)
assert np.allclose(Ridge(alpha=0.5, fit_intercept=False).fit(Q, y5).coef_, V["L5_ridge"], atol=1e-10)
dire("Lasso orthonormal : seuillage doux confirmé par scikit-learn avec alpha = lambda / n ; Ridge = division par 1 + lambda.")

# Conditions d'optimalité, formule sur l'ensemble actif, et seuil lambda_max, sur des données aléatoires.
n_, p_ = 100, 10
Xl = rng.standard_normal((n_, p_))
yl = Xl[:, :3].dot([3.0, -2.0, 1.0]) + rng.standard_normal(n_)
lmax = np.abs(Xl.T.dot(yl)).max()
for lam_ in [5.0, 30.0, 0.999 * lmax, 1.001 * lmax]:
    bl = Lasso(alpha=lam_ / n_, fit_intercept=False, tol=1e-14, max_iter=1_000_000).fit(Xl, yl).coef_
    gl = Xl.T.dot(yl - Xl.dot(bl))
    act = bl != 0
    assert np.allclose(gl[act], lam_ * np.sign(bl[act]), atol=1e-6) and np.all(np.abs(gl[~act]) <= lam_ + 1e-6)
    if act.any():
        XA = Xl[:, act]
        assert np.allclose(np.linalg.solve(XA.T.dot(XA), XA.T.dot(yl) - lam_ * np.sign(bl[act])), bl[act], atol=1e-6)
    assert (not act.any()) == (lam_ >= lmax)
dire("Lasso : conditions d'optimalité, formule sur l'ensemble actif et seuil lambda_max = max |x_j^T y| vérifiés.")

# Point de contrôle « seuil ».
xty6 = np.array([0.3, -1.2, 0.5])
V["C6_xty"] = list(xty6)
V["C6_max"] = float(np.abs(xty6).max())
V["C6_l1"] = float(np.abs(xty6).sum())
V["C6_l2"] = float(np.sqrt((xty6**2).sum()))

# Colonnes dupliquées : toute répartition de même signe donne le même critère.
xd = rng.standard_normal(50)
Xd2 = np.column_stack([xd, xd])
yd2 = 2 * xd + rng.standard_normal(50)
bd = Lasso(alpha=10.0 / 50, fit_intercept=False, tol=1e-14, max_iter=100_000).fit(Xd2, yd2).coef_
crit = lambda bb: 0.5 * np.sum((yd2 - Xd2.dot(bb)) ** 2) + 10.0 * np.abs(bb).sum()
for t in [0.0, 0.25, 0.5, 1.0]:
    assert np.isclose(crit(np.array([t, 1 - t]) * bd.sum()), crit(bd))
dire("Lasso, colonnes dupliquées : solutions multiples de même critère.")

# =====================================================================================
# LEÇON, PARTIE 6 : biais, variance et lambda
# =====================================================================================
beta_bv = sp.Matrix([1, 0])
s2_bv = sp.Rational(1, 4)
x0 = sp.Matrix([1, -1])
V["L6_sigma2"] = float(s2_bv)
lams = [sp.Integer(0), sp.Rational(1, 10), sp.Rational(1, 2), sp.Integer(2)]
Xbv = np.linalg.cholesky(Gn).T
Ybv = Xbv.dot([1.0, 0.0]) + 0.5 * rng.standard_normal((400_000, 2))
for k, l_ in enumerate(lams):
    A = (G + l_ * sp.eye(2)).inv()
    esp = (x0.T * A * G * beta_bv)[0]
    biais2 = (esp - (x0.T * beta_bv)[0]) ** 2
    var = s2_bv * (x0.T * A * G * A * x0)[0]
    fraction(f"L6_l{k}", l_)
    fraction(f"L6_b{k}", biais2)
    fraction(f"L6_v{k}", var)
    fraction(f"L6_t{k}", s2_bv + biais2 + var)
    pred = Ybv.dot(np.linalg.solve(Gn + float(l_) * np.eye(2), Xbv.T).T).dot([1.0, -1.0])
    mc_variance(f"L6_v{k}", var, pred)
    mc_moyenne(f"L6_e{k}", esp, pred)
# Optimum en x0 : lambda = sigma^2 / theta_2^2 avec theta_2^2 = 1/2.
l_s = sp.symbols("l", nonnegative=True)
tot = s2_bv + ((x0.T * (G + l_s * sp.eye(2)).inv() * G * beta_bv)[0] - 1) ** 2 + s2_bv * (x0.T * (G + l_s * sp.eye(2)).inv() * G * (G + l_s * sp.eye(2)).inv() * x0)[0]
crit_l = sp.solve(sp.Eq(sp.diff(sp.simplify(tot), l_s), 0), l_s)
assert crit_l == [sp.Rational(1, 2)]
# Erreur quadratique des coefficients : MSE(lambda) = sum_j (lambda^2 theta_j^2 + sigma^2 mu_j) / (mu_j + lambda)^2.
theta2 = sp.Rational(1, 2)
mse = lambda l_: sum((l_**2 * theta2 + s2_bv * m) / (m + l_) ** 2 for m in [mu1, mu2])
fraction("L6_mse0", mse(0))
fraction("L6_mse_opt", mse(sp.Rational(1, 2)))
fraction("L6_lopt", s2_bv / theta2)
fraction("L6_theta2", theta2)
assert mse(0) == s2_bv * (G.inv()).trace()
# Dérivée de g_j(lambda) = (lambda^2 theta^2 + sigma^2 d^2) / (d^2 + lambda)^2 (Hoerl et Kennard).
ls, th, sg, dd = sp.symbols("lambda theta sigma d", positive=True)
gj = (ls**2 * th**2 + sg**2 * dd**2) / (dd**2 + ls) ** 2
assert sp.simplify(sp.diff(gj, ls) - 2 * dd**2 * (ls * th**2 - sg**2) / (dd**2 + ls) ** 3) == 0
dire("Biais-variance : tableau exact en x0 = (1, -1), optimum lambda = 1/2, dérivée de Hoerl et Kennard vérifiés.")

# =====================================================================================
# QUESTIONS
# =====================================================================================
# F1-01 : pente de la régression simple.
xq = [1, 2, 3, 4, 5]
yq = [2, 4, 5, 4, 5]
xb, yb = Fraction(sum(xq), 5), Fraction(sum(yq), 5)
sxy = sum((a - xb) * (c - yb) for a, c in zip(xq, yq))
sxx = sum((a - xb) ** 2 for a in xq)
syy = sum((c - yb) ** 2 for c in yq)
fraction("Q01_exact", sxy / sxx)
fraction("Q01_sxy", sxy)
fraction("Q01_sxx", sxx)
fraction("Q01_syy", syy)
fraction("Q01_xbar", xb)
fraction("Q01_ybar", yb)
fraction("Q01_const", yb - sxy / sxx * xb)
V["Q01_xy"] = sum(a * c for a, c in zip(xq, yq))
V["Q01_xx"] = sum(a * a for a in xq)
fraction("Q01_d_origine", Fraction(V["Q01_xy"], V["Q01_xx"]))
fraction("Q01_d_inverse", sxy / syy)
V["Q01_d_corr"] = float(sxy) / math.sqrt(float(sxx * syy))
V["Q01_dx"] = [float(v - xb) for v in xq]
V["Q01_dy"] = [float(v - yb) for v in yq]
assert np.isclose(np.polyfit(xq, yq, 1)[0], float(sxy / sxx))

# F1-02 : E[SCR] = sigma^2 (n - p).
V["Q02_exact"] = 2 * (50 - 4)
V["Q02_d_n"] = 2 * 50
V["Q02_d_p"] = 2 * 4
V["Q02_d_np1"] = 2 * (50 - 4 - 1)
V["Q02_nmp"] = 50 - 4
Xq2 = np.column_stack([np.ones(50), rng.standard_normal((50, 3))])
Hq2 = Xq2.dot(np.linalg.solve(Xq2.T.dot(Xq2), Xq2.T))
assert np.isclose(np.trace(Hq2), 4)
Eq2 = math.sqrt(2) * rng.standard_normal((100_000, 50))
Rq2 = Eq2 - Eq2.dot(Hq2)  # (I - H) epsilon, H symétrique
mc_moyenne("Q02", 92, (Rq2**2).sum(axis=1))

# F1-03 : hétéroscédasticité, x = (1, 2), Var(eps) = diag(1, 4), sans constante.
xh3 = sp.Matrix([1, 2])
Sig = sp.diag(1, 4)
xx3 = (xh3.T * xh3)[0]
var_mco = (xh3.T * Sig * xh3)[0] / xx3**2
H3 = xh3 * xh3.T / xx3
esp_scr3 = ((sp.eye(2) - H3) * Sig).trace()
esp_formule = esp_scr3 / (2 - 1) / xx3
var_mcp = 1 / (xh3.T * Sig.inv() * xh3)[0]
fraction("Q03_var_mco", var_mco)
fraction("Q03_formule", esp_formule)
fraction("Q03_var_mcp", var_mcp)
fraction("Q03_esp_scr", esp_scr3)
fraction("Q03_xSx", (xh3.T * Sig * xh3)[0])
fraction("Q03_trHS", (H3 * Sig).trace())
V["Q03_xx"] = int(xx3)
V["Q03_xx2"] = int(xx3**2)
V["Q03_trS"] = int(Sig.trace())
fraction("Q03_info", (xh3.T * Sig.inv() * xh3)[0])
E3 = rng.standard_normal((400_000, 2)) * np.array([1.0, 2.0])
Y3 = np.array([1.0, 2.0]) * 1.0 + E3  # beta = 1
b3 = Y3.dot([1.0, 2.0]) / 5
e3 = Y3 - np.outer(b3, [1.0, 2.0])
mc_variance("Q03_var_mco", var_mco, b3)
mc_moyenne("Q03_formule", esp_formule, (e3**2).sum(axis=1) / 1 / 5)
b3w = (Y3[:, 0] / 1 + 2 * Y3[:, 1] / 4) / (1 + 4 / 4)
mc_variance("Q03_var_mcp", var_mcp, b3w)
mc_moyenne("Q03_sans_biais", 1.0, b3)

# F1-04 : variable omise, pente de z sur (1, x) égale à 0,5.
V["Q04_exact"] = 2 + 3 * 0.5
V["Q04_d_vrai"] = 2
V["Q04_d_somme"] = 2 + 3
V["Q04_d_signe"] = 2 - 3 * 0.5
n_ = 40
x4 = rng.standard_normal(n_)
X4 = np.column_stack([np.ones(n_), x4])
w4 = rng.standard_normal(n_)
w4 -= X4.dot(np.linalg.lstsq(X4, w4, rcond=None)[0])  # w orthogonal à (1, x)
z4 = 0.5 * x4 + w4
assert np.isclose(np.linalg.lstsq(X4, z4, rcond=None)[0][1], 0.5)
Y4 = 1 + 2 * x4 + 3 * z4 + rng.standard_normal((200_000, n_))
mc_moyenne("Q04", 3.5, np.linalg.lstsq(X4, Y4.T, rcond=None)[0][1])

# F1-05 : facteur d'inflation, R_1^2 = 0,9.
V["Q05_exact"] = math.sqrt(1 / (1 - 0.9))
V["Q05_vif"] = 1 / (1 - 0.9)
V["Q05_d_var"] = 1 / (1 - 0.9)
V["Q05_d_r"] = math.sqrt(1 / (1 - 0.9**2))
V["Q05_d_inv"] = math.sqrt(1 / 0.9)
V["Q05_r2_r"] = 0.9**2
n_ = 60
x2_ = rng.standard_normal(n_)
x2_ -= x2_.mean()
w5 = rng.standard_normal(n_)
w5 -= np.column_stack([np.ones(n_), x2_]).dot(np.linalg.lstsq(np.column_stack([np.ones(n_), x2_]), w5, rcond=None)[0])
# x1 = a x2 + w, R^2 de x1 sur (1, x2) = a^2 ||x2||^2 / (a^2 ||x2||^2 + ||w||^2) = 0,9.
a5 = math.sqrt(9 * w5.dot(w5) / x2_.dot(x2_))
x1_ = a5 * x2_ + w5
X5 = np.column_stack([np.ones(n_), x1_, x2_])
sct1 = ((x1_ - x1_.mean()) ** 2).sum()
assert np.isclose(1 - w5.dot(w5) / sct1, 0.9)
rap5 = np.linalg.inv(X5.T.dot(X5))[1, 1] / (1 / sct1)
assert np.isclose(rap5, 10)
Y5 = X5.dot([0.0, 1.0, 1.0]) + rng.standard_normal((200_000, n_))
B5 = np.linalg.solve(X5.T.dot(X5), X5.T.dot(Y5.T))[1]
mc_variance("Q05_var", np.linalg.inv(X5.T.dot(X5))[1, 1], B5)

# F1-06 : R^2 ajusté.
n6, p6, r26 = 30, 4, Fraction(1, 2)
V["Q06_exact"] = float(1 - (1 - r26) * Fraction(n6 - 1, n6 - p6))
V["Q06_d_p5"] = float(1 - (1 - r26) * Fraction(n6 - 1, n6 - p6 - 1))
V["Q06_d_n"] = float(1 - (1 - r26) * Fraction(n6, n6 - p6))
V["Q06_d_inv"] = float(1 - (1 - r26) * Fraction(n6 - p6, n6 - 1))
V["Q06_nm1"] = n6 - 1
V["Q06_nmp"] = n6 - p6
fraction("Q06_rap", Fraction(n6 - 1, n6 - p6))

# F1-07 : ajout d'une variable de statistique t = 0,8.
n7, p7, scr7, sct7, t7 = 40, 3, 100.0, 200.0, 0.8
scr7b = scr7 / (1 + t7**2 / (n7 - p7 - 1))
V["Q07_scr_b"] = scr7b
V["Q07_r2_a"] = 1 - scr7 / sct7
V["Q07_r2_b"] = 1 - scr7b / sct7
V["Q07_aj_a"] = 1 - (scr7 / (n7 - p7)) / (sct7 / (n7 - 1))
V["Q07_aj_b"] = 1 - (scr7b / (n7 - p7 - 1)) / (sct7 / (n7 - 1))
V["Q07_aj_oubli"] = 1 - (scr7b / (n7 - p7)) / (sct7 / (n7 - 1))
V["Q07_r2_baisse"] = 1 - scr7 * (1 + t7**2 / (n7 - p7 - 1)) / sct7
V["Q07_ddl"] = n7 - p7 - 1
V["Q07_t2"] = t7**2
V["Q07_facteur"] = 1 + t7**2 / (n7 - p7 - 1)
assert V["Q07_r2_b"] > V["Q07_r2_a"] and V["Q07_aj_b"] < V["Q07_aj_a"] < V["Q07_aj_oubli"]
assert len({round(V[k], 3) for k in ["Q07_aj_a", "Q07_aj_b", "Q07_aj_oubli"]}) == 3

# F1-08 : facteur Ridge d^2 / (d^2 + lambda) pour d = 0,2, lambda = 1.
d8 = Fraction(1, 5)
fraction("Q08_exact", d8**2 / (d8**2 + 1))
fraction("Q08_d_lin", d8 / (d8 + 1))
fraction("Q08_d_compl", 1 / (d8**2 + 1))
fraction("Q08_d_orth", Fraction(1, 2))
V["Q08_f1"] = float(Fraction(9, 10))
V["Q08_f2"] = float(Fraction(1, 2))
fraction("Q08_d2", d8**2)
fraction("Q08_d2_plus_un", d8**2 + 1)

# F1-09 : a priori gaussien, sigma = 0,5, tau = 4.
s9, t9 = Fraction(1, 2), Fraction(4)
fraction("Q09_exact", s9**2 / t9**2)
fraction("Q09_d_inv", t9**2 / s9**2)
fraction("Q09_d_tau", 1 / t9**2)
fraction("Q09_s2", s9**2)
fraction("Q09_t2", t9**2)

# F1-10 : Lasso orthonormal, z = (2,5 ; -0,4 ; -1,1), lambda = 0,6.
z10 = [Fraction(5, 2), Fraction(-2, 5), Fraction(-11, 10)]
l10 = Fraction(3, 5)
V["Q10_z"] = [float(v) for v in z10]
V["Q10_lam"] = float(l10)
V["Q10_exact"] = [float(v) for v in seuil_doux_fr(z10, l10)]
V["Q10_d_ridge"] = [float(v / (1 + l10)) for v in z10]
V["Q10_d_dur"] = [float(v) if abs(v) > l10 else 0.0 for v in z10]
V["Q10_d_signe"] = [float(v - l10) for v in z10]
n_ = 30
Q10, _ = np.linalg.qr(rng.standard_normal((n_, 3)))
y10 = Q10.dot(V["Q10_z"]) + (np.eye(n_) - Q10.dot(Q10.T)).dot(rng.standard_normal(n_))
assert np.allclose(Lasso(alpha=0.6 / n_, fit_intercept=False, tol=1e-14, max_iter=100_000).fit(Q10, y10).coef_, V["Q10_exact"], atol=1e-9)

# F1-11 : seuil de scikit-learn, n = 200, X^T y = (4, -6, 2).
c11 = np.array([4.0, -6.0, 2.0])
V["Q11_exact"] = float(np.abs(c11).max() / 200)
V["Q11_d_signe"] = float(c11.max() / 200)
V["Q11_d_sans_n"] = float(np.abs(c11).max())
V["Q11_d_2n"] = float(np.abs(c11).max() / (2 * 200))
V["Q11_max"] = float(np.abs(c11).max())
X11 = rng.standard_normal((200, 3))
y11 = X11.dot(np.linalg.solve(X11.T.dot(X11), c11))
assert np.allclose(X11.T.dot(y11), c11)
assert np.all(Lasso(alpha=0.0300001, fit_intercept=False, tol=1e-14, max_iter=100_000).fit(X11, y11).coef_ == 0)
assert np.any(Lasso(alpha=0.0299, fit_intercept=False, tol=1e-14, max_iter=100_000).fit(X11, y11).coef_ != 0)
assert np.any(Lasso(alpha=0.025, fit_intercept=False, tol=1e-14, max_iter=100_000).fit(X11, y11).coef_ != 0)

# F1-12 : lambda optimal pour une coordonnée, X^T X = I, sigma^2 = 1, beta_1 = 2.
s12, b12 = Fraction(1), Fraction(2)
mse12 = lambda l_: (l_**2 * b12**2 + s12) / (1 + l_) ** 2
fraction("Q12_lopt", s12 / b12**2)
fraction("Q12_mse_opt", mse12(s12 / b12**2))
fraction("Q12_mse0", mse12(Fraction(0)))
fraction("Q12_d_inv_l", b12**2 / s12)
fraction("Q12_d_inv_mse", mse12(b12**2 / s12))
fraction("Q12_d_rac_l", Fraction(1, 2))
fraction("Q12_d_rac_mse", mse12(Fraction(1, 2)))
assert mse12(s12 / b12**2) == s12 * b12**2 / (s12 + b12**2)
grille = [Fraction(k, 100) for k in range(0, 501)]
assert min(grille, key=mse12) == Fraction(1, 4)
Z12 = 2 + rng.standard_normal(400_000)
mc_moyenne("Q12_mse_opt", mse12(Fraction(1, 4)), (Z12 / 1.25 - 2) ** 2)
mc_moyenne("Q12_mse0", 1, (Z12 - 2) ** 2)

Path(__file__).with_name("seance_ml_1.out.json").write_text(json.dumps(V, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
dire(f"{len(V)} valeurs exportées dans verif/seance_ml_1.out.json.")
dire("Toutes les vérifications passent.")
Path(__file__).with_name("seance_ml_1.out.txt").write_text("\n".join(LIGNES) + "\n", encoding="utf-8", newline="\n")
