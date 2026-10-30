"""Vérification de tous les nombres de l'examen blanc 1 (questions EB1-C1, EB1-M01 à EB1-M10, EB1-F01 à EB1-F05,
EB1-A01 à EB1-A08).

Principe :
  1. chaque valeur citée (bonne réponse, distracteur, étape de calcul) est calculée ici, en fractions exactes
     ou avec sympy quand c'est possible ;
  2. chaque probabilité ou espérance a une seconde vérification indépendante : simulation Monte-Carlo dont
     l'écart à l'exact doit rester inférieur à 4 erreurs standard, ou énumération complète ;
  3. chaque affirmation de ML est démontrée dans la correction et illustrée ici par une expérience
     (scikit-learn, descente de gradient, différences finies) ;
  4. pour la question de code : valeurs de l'exemple de l'énoncé, et preuve que les tests cachés attrapent
     les erreurs visées (décalage sur cadre non trié, pct_change, départage des égalités, pivot par date).

Graine fixe : la sortie est identique à chaque exécution.
Sorties : verif/examen_blanc_1.out.json (lu par outils/construire.py) et verif/examen_blanc_1.out.txt (ce résumé).
"""
import itertools
import json
import math
import sys
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from scipy import optimize, stats

RACINE = Path(__file__).resolve().parent.parent
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
    dire(f"{nom:24s} exact={float(exact):.6f}  MC={float(estimation):.6f}  se={float(erreur_std):.6f}  z={z:+.2f}")


def mc_moyenne(nom, exact, echantillon):
    echantillon = np.asarray(echantillon, dtype=float)
    controle_mc(nom, exact, echantillon.mean(), echantillon.std(ddof=1) / np.sqrt(len(echantillon)))


def mc_proba(nom, exact, indicatrices):
    indicatrices = np.asarray(indicatrices, dtype=float)
    p = float(exact)
    controle_mc(nom, exact, indicatrices.mean(), np.sqrt(p * (1 - p) / len(indicatrices)))


def mc_variance(nom, exact, echantillon):
    """Variance empirique ; erreur standard par la méthode delta, estimée sur l'échantillon."""
    x = np.asarray(echantillon, dtype=float)
    c = x - x.mean()
    var = (c**2).mean()
    controle_mc(nom, exact, var, np.sqrt(((c**2 - var) ** 2).mean() / len(x)))


N_SIM = 400_000

# =====================================================================================
# EB1-M01 : scrutin, A en tête tout au long du dépouillement (a = 7, b = 4)
# =====================================================================================
a_, b_ = 7, 4
mots = []
for pos_b in itertools.combinations(range(a_ + b_), b_):
    mots.append([-1 if j in pos_b else 1 for j in range(a_ + b_)])
cumuls = [np.cumsum(m) for m in mots]
nb_total = len(mots)
nb_bons = sum(all(s > 0 for s in c) for c in cumuls)
nb_large = sum(all(s >= 0 for s in c) for c in cumuls)
nb_debut_a = sum(m[0] == 1 for m in mots)
nb_debut_b = sum(m[0] == -1 for m in mots)
nb_mauvais_debut_a = sum(m[0] == 1 and any(s == 0 for s in c) for m, c in zip(mots, cumuls))
assert nb_total == math.comb(11, 4) and nb_debut_a == math.comb(10, 4) and nb_debut_b == math.comb(10, 3)
assert nb_mauvais_debut_a == nb_debut_b  # principe de réflexion : bijection
assert all(any(s == 0 for s in c) for m, c in zip(mots, cumuls) if m[0] == -1)  # tout mot qui commence par B repasse par 0
assert Fraction(nb_bons, nb_total) == Fraction(a_ - b_, a_ + b_) == Fraction(3, 11)
assert Fraction(nb_large, nb_total) == Fraction(a_ + 1 - b_, a_ + 1)
# Chaque mot correspond au même nombre d'ordres de bulletins distinguables (7! 4!) : les mots sont équiprobables.
V["M01_nb_total"] = nb_total
V["M01_nb_bons"] = nb_bons
V["M01_nb_debut_a"] = nb_debut_a
V["M01_nb_debut_b"] = nb_debut_b
fraction("M01_exact", Fraction(nb_bons, nb_total))
fraction("M01_d_premier", Fraction(a_, a_ + b_))
fraction("M01_d_large", Fraction(nb_large, nb_total))
fraction("M01_d_cond", Fraction(nb_bons, nb_debut_a))
bulletins = np.array([1] * a_ + [-1] * b_)
ordres = np.argsort(rng.random((N_SIM, a_ + b_)), axis=1)
chemins = np.cumsum(bulletins[ordres], axis=1)
mc_proba("M01", Fraction(3, 11), (chemins > 0).all(axis=1))

# =====================================================================================
# EB1-M02 : dé puis N pièces, variance totale
# =====================================================================================
loi_x = {}
for n in range(1, 7):
    for x in range(n + 1):
        loi_x[x] = loi_x.get(x, 0) + Fraction(1, 6) * Fraction(math.comb(n, x), 2**n)
assert sum(loi_x.values()) == 1
e_x = sum(x * p for x, p in loi_x.items())
var_x = sum(x * x * p for x, p in loi_x.items()) - e_x**2
e_n = Fraction(7, 2)
var_n = sum(Fraction(n * n, 6) for n in range(1, 7)) - e_n**2
assert var_n == Fraction(35, 12)
assert var_x == e_n / 4 + var_n / 4 == Fraction(77, 48)
fraction("M02_exact", var_x)
fraction("M02_e_var", e_n / 4)
fraction("M02_var_e", var_n / 4)
fraction("M02_var_n", var_n)
fraction("M02_d_demi", e_n / 4 + var_n / 2)
fraction("M02_e_x", e_x)
fraction("M02_e_n2", sum(Fraction(n * n, 6) for n in range(1, 7)))
fraction("M02_e_n_carre", e_n**2)
n_de = rng.integers(1, 7, N_SIM)
x_piles = rng.binomial(n_de, 0.5)
mc_variance("M02", var_x, x_piles)

# =====================================================================================
# EB1-M03 : processus de Poisson amincis, espérance conditionnelle du nombre total d'ordres
# =====================================================================================
lam, duree, p_gros = 3, 2, Fraction(1, 3)
mu = lam * duree  # 6 ordres en moyenne sur 2 minutes
# Calcul direct par la loi jointe : P(N = n, G = 1) = e^{-6} 6^n / n! * n (1/3) (2/3)^{n-1}, série sommée par sympy.
n_s = sp.symbols("n", integer=True, positive=True)
terme = sp.exp(-mu) * sp.Integer(mu) ** n_s / sp.factorial(n_s) * n_s * sp.Rational(1, 3) * sp.Rational(2, 3) ** (n_s - 1)
p_g1 = sp.simplify(sp.summation(terme, (n_s, 1, sp.oo)))
e_n_g1 = sp.simplify(sp.summation(n_s * terme, (n_s, 1, sp.oo)) / p_g1)
assert sp.simplify(p_g1 - 2 * sp.exp(-2)) == 0  # G de loi de Poisson de paramètre 6/3 = 2
assert e_n_g1 == 5
V["M03_exact"] = int(e_n_g1)
V["M03_mu"] = mu
V["M03_mu_gros"] = float(mu * p_gros)
V["M03_mu_petits"] = float(mu * (1 - p_gros))
V["M03_d_inconditionnel"] = mu
V["M03_d_inverse"] = int(1 / p_gros)
V["M03_d_petits"] = float(mu * (1 - p_gros))
n_poisson = rng.poisson(mu, 2_000_000)
g_gros = rng.binomial(n_poisson, float(p_gros))
mc_moyenne("M03", 5, n_poisson[g_gros == 1])
mc_proba("M03_p_g1", float(p_g1), g_gros == 1)

# =====================================================================================
# EB1-M04 : 10 lots identiques, 4 traders, entre 1 et 4 lots chacun
# =====================================================================================
force_brute = sum(1 for c in itertools.product(range(1, 5), repeat=4) if sum(c) == 10)
formule = math.comb(9, 3) - 4 * math.comb(5, 3)
assert force_brute == formule == 44
assert 2 * 4 > 6  # deux dépassements simultanés impossibles : y_i >= 4 et y_j >= 4 font déjà 8 > 6
V["M04_exact"] = force_brute
V["M04_sans_plafond"] = math.comb(9, 3)
V["M04_un_depassement"] = math.comb(5, 3)
V["M04_retires"] = 4 * math.comb(5, 3)
V["M04_d_plafond5"] = sum(1 for c in itertools.product(range(1, 6), repeat=4) if sum(c) == 10)
assert V["M04_d_plafond5"] == math.comb(9, 3) - 4 * math.comb(4, 3)
assert sum(1 for c in itertools.product(range(1, 11), repeat=4) if sum(c) == 10) == math.comb(9, 3)

# =====================================================================================
# EB1-M05 : Sherman-Morrison, somme des coefficients de (I + u u^T)^{-1}
# =====================================================================================
u = sp.Matrix([1, 2, 2])
un = sp.Matrix([1, 1, 1])
A = sp.eye(3) + u * u.T
A_inv = A.inv()
assert sp.simplify(A_inv - (sp.eye(3) - u * u.T / (1 + (u.T * u)[0]))) == sp.zeros(3, 3)
somme = (un.T * A_inv * un)[0]
assert somme == sp.Rational(1, 2)
fraction("M05_exact", Fraction(1, 2))
V["M05_utu"] = int((u.T * u)[0])
V["M05_1tu"] = int((un.T * u)[0])
V["M05_1tu2"] = int((un.T * u)[0] ** 2)
fraction("M05_d_sans_un", Fraction(3) - Fraction(25, 9))
fraction("M05_d_inverse_somme", Fraction(1, int((un.T * A * un)[0])))
V["M05_d_neumann"] = int((un.T * (sp.eye(3) - u * u.T) * un)[0])
V["M05_somme_A"] = int((un.T * A * un)[0])
assert abs(np.ones(3) @ np.linalg.inv(np.eye(3) + np.outer([1, 2, 2], [1, 2, 2])) @ np.ones(3) - 0.5) < 1e-12
assert all(ev > 0 for ev in A.eigenvals())  # A définie positive, donc A^{-1} aussi : forme quadratique > 0
V["M05_vp"] = sorted(int(ev) for ev in A.eigenvals())
assert A.eigenvals() == {1: 2, 10: 1}
V["M05_vp_simple"] = 1 + V["M05_utu"]

# =====================================================================================
# EB1-M06 : biais du maximum de vraisemblance du paramètre d'une loi exponentielle (n = 5)
# =====================================================================================
s_, l_ = sp.symbols("s lambda", positive=True)
n6 = 5
dens_gamma = l_**n6 * s_ ** (n6 - 1) * sp.exp(-l_ * s_) / sp.factorial(n6 - 1)  # loi de S = X_1 + ... + X_5
assert sp.simplify(sp.integrate(dens_gamma, (s_, 0, sp.oo))) == 1
e_inv_s = sp.simplify(sp.integrate(dens_gamma / s_, (s_, 0, sp.oo)))
assert sp.simplify(e_inv_s - l_ / (n6 - 1)) == 0
rapport = sp.simplify(n6 * e_inv_s / l_)
assert rapport == sp.Rational(5, 4)
# Le maximum de vraisemblance est bien 1/moyenne : dérivée de n ln(l) - l s nulle en l = n/s.
assert sp.solve(sp.diff(n6 * sp.log(l_) - l_ * s_, l_), l_) == [n6 / s_]
fraction("M06_exact", Fraction(5, 4))
fraction("M06_d_inverse", Fraction(4, 5))
fraction("M06_d_delta", Fraction(6, 5))
V["M06_n_moins_un"] = n6 - 1
lam6 = 2.0
xs6 = rng.exponential(1 / lam6, size=(N_SIM, n6))
mc_moyenne("M06", 1.25, (1 / xs6.mean(axis=1)) / lam6)

# =====================================================================================
# EB1-M07 : puissance d'un test unilatéral, valeur de mu où elle vaut 1/2
# =====================================================================================
n7 = 25
z95 = float(stats.norm.ppf(0.95))
z975 = float(stats.norm.ppf(0.975))
c7 = z95 / math.sqrt(n7)
assert abs(stats.norm.sf((c7 - c7) * math.sqrt(n7)) - 0.5) < 1e-15  # puissance en mu = c
V["M07_z95"] = z95
V["M07_z975"] = z975
V["M07_exact"] = c7
V["M07_d_sans_racine"] = z95
V["M07_d_bilateral"] = z975 / math.sqrt(n7)
V["M07_d_n"] = z95 / n7
for mu7, attendu, nom in [(c7, 0.5, "M07_puissance"), (0.0, 0.05, "M07_niveau")]:
    xbar = rng.normal(mu7, 1, size=(N_SIM, n7)).mean(axis=1)
    mc_proba(nom, attendu, xbar > c7)

# =====================================================================================
# EB1-M08 : critère de Kelly, pari +1 / -1 avec probabilité 0,6 / 0,4
# =====================================================================================
f_ = sp.symbols("f")
g8 = sp.Rational(3, 5) * sp.log(1 + f_) + sp.Rational(2, 5) * sp.log(1 - f_)
sol8 = sp.solve(sp.diff(g8, f_), f_)
assert sol8 == [sp.Rational(1, 5)]
assert sp.simplify(sp.diff(g8, f_, 2)) != 0 and all(sp.diff(g8, f_, 2).subs(f_, v) < 0 for v in [0, sp.Rational(1, 5), sp.Rational(9, 10)])
grille = np.linspace(0, 0.99, 99001)
g_num = 0.6 * np.log1p(grille) + 0.4 * np.log1p(-grille)
assert abs(grille[g_num.argmax()] - 0.2) < 1e-4
fraction("M08_exact", Fraction(1, 5))
V["M08_g_opt"] = float(g8.subs(f_, sp.Rational(1, 5)))
V["M08_g_un_demi"] = float(g8.subs(f_, sp.Rational(1, 2)))
fraction("M08_mu", Fraction(1, 5))
fraction("M08_var", Fraction(24, 25))
fraction("M08_d_merton", Fraction(1, 5) / Fraction(24, 25))
V["M08_d_proba"] = 0.6
V["M08_g_proba"] = float(g8.subs(f_, sp.Rational(3, 5)))
assert V["M08_g_proba"] < 0
assert V["M08_g_un_demi"] < 0 < V["M08_g_opt"]
gains = np.where(rng.random(N_SIM) < 0.6, 1.0, -1.0)
mc_moyenne("M08", V["M08_g_opt"], np.log1p(0.2 * gains))

# =====================================================================================
# EB1-M09 : chaîne de Markov de notation, temps moyen jusqu'au défaut
# =====================================================================================
eA, eB = sp.symbols("eA eB")
sol9 = sp.solve([sp.Eq(eA, 1 + sp.Rational(8, 10) * eA + sp.Rational(2, 10) * eB),
                 sp.Eq(eB, 1 + sp.Rational(3, 10) * eA + sp.Rational(5, 10) * eB)], [eA, eB])
assert sol9[eA] == sp.Rational(35, 2) and sol9[eB] == sp.Rational(25, 2)
# Méthode indépendante : matrice fondamentale (I - Q)^{-1} 1, Q restreinte aux états transitoires.
Q9 = sp.Matrix([[sp.Rational(8, 10), sp.Rational(2, 10)], [sp.Rational(3, 10), sp.Rational(5, 10)]])
t9 = (sp.eye(2) - Q9).inv() * sp.Matrix([1, 1])
assert t9[0] == sp.Rational(35, 2) and t9[1] == sp.Rational(25, 2)
fraction("M09_exact", Fraction(35, 2))
fraction("M09_eB", Fraction(25, 2))
V["M09_d_geom"] = 5
V["M09_d_somme"] = 10
V["M09_sortie_A"] = 5


def simuler_notation(n_sim):
    etat = np.zeros(n_sim, dtype=np.int64)  # 0 = A, 1 = B, 2 = défaut
    duree9 = np.zeros(n_sim, dtype=np.int64)
    while (etat != 2).any():
        actif9 = etat != 2
        u9 = rng.random(int(actif9.sum()))
        e9 = etat[actif9]
        nouveau = np.where(e9 == 0, np.where(u9 < 0.8, 0, 1), np.where(u9 < 0.3, 0, np.where(u9 < 0.8, 1, 2)))
        etat[actif9] = nouveau
        duree9[actif9] += 1
    return duree9


mc_moyenne("M09", 17.5, simuler_notation(N_SIM))

# =====================================================================================
# EB1-M10 : inégalité de Cantelli, borne optimale de P(X >= 3) pour E[X] = 0, Var(X) = 1
# =====================================================================================
t10 = 3
u_, t_ = sp.symbols("u t", positive=True)
borne_u = (1 + u_**2) / (t_ + u_) ** 2
u_opt = sp.solve(sp.diff(borne_u, u_), u_)
assert u_opt == [1 / t_]
assert sp.simplify(borne_u.subs(u_, 1 / t_) - 1 / (1 + t_**2)) == 0
# Loi à deux points qui atteint la borne : X = 3 avec probabilité 1/10, X = -1/3 avec probabilité 9/10.
p_haut, x_haut, x_bas = Fraction(1, 10), Fraction(3), Fraction(-1, 3)
assert p_haut * x_haut + (1 - p_haut) * x_bas == 0
assert p_haut * x_haut**2 + (1 - p_haut) * x_bas**2 == 1
fraction("M10_exact", Fraction(1, 1 + t10**2))
fraction("M10_d_tcheb", Fraction(1, t10**2))
fraction("M10_d_demi", Fraction(1, 2 * t10**2))
fraction("M10_d_markov", Fraction(1, t10))
fraction("M10_x_bas", x_bas)
# Programme linéaire indépendant : maximiser P(X >= 3) sur les lois portées par une grille de pas 1/3 sur [-10, 10].
grille10 = np.arange(-30, 31) / 3
res10 = optimize.linprog(-(grille10 >= 3 - 1e-12).astype(float), A_eq=np.vstack([np.ones_like(grille10), grille10, grille10**2]),
                         b_eq=[1, 0, 1], bounds=(0, None), method="highs")
assert res10.status == 0 and abs(-res10.fun - 0.1) < 1e-9
V["M10_lp"] = round(-res10.fun, 12)
x10 = np.where(rng.random(N_SIM) < 0.1, 3.0, -1 / 3)
mc_proba("M10", 0.1, x10 >= 3)
mc_moyenne("M10_moyenne", 0, x10)
# La borne « Tchebychev divisé par deux » est fausse pour cette loi : 1/10 > 1/18.
assert Fraction(1, 10) > Fraction(1, 18)

# =====================================================================================
# EB1-F01 : Ridge et Lasso avec deux colonnes identiques
# =====================================================================================
XtX = np.array([[1.0, 1.0], [1.0, 1.0]])
Xty = np.array([3.0, 3.0])
ridge = np.linalg.solve(XtX + np.eye(2), Xty)
assert np.allclose(ridge, [1, 1])
assert np.allclose(np.linalg.eigvalsh(XtX), [0, 2]) and np.allclose(np.linalg.eigvalsh(XtX + np.eye(2)), [1, 3])
V["F01_ridge"] = [float(x) for x in ridge]
V["F01_vp_xtx"] = [0, 2]
V["F01_vp_ridge"] = [1, 3]
# Données concrètes : x de norme 1, x^T y = 3.
x_col = np.full(4, 0.5)
y_f1 = 1.5 * np.ones(4) + 0.3 * np.array([1, -1, 1, -1])
assert abs(x_col @ x_col - 1) < 1e-15 and abs(x_col @ y_f1 - 3) < 1e-12
X_f1 = np.column_stack([x_col, x_col])


def critere_lasso(b, lam=1.0):
    return 0.5 * np.sum((y_f1 - X_f1 @ b) ** 2) + lam * np.abs(b).sum()


def critere_ridge(b, lam=1.0):
    return 0.5 * np.sum((y_f1 - X_f1 @ b) ** 2) + 0.5 * lam * np.sum(b**2)


valeurs_segment = [critere_lasso(np.array([t, 2 - t])) for t in np.linspace(0, 2, 21)]
assert max(valeurs_segment) - min(valeurs_segment) < 1e-12  # critère constant sur le segment
V["F01_crit_lasso"] = round(valeurs_segment[0], 10)
# Minimum global du Lasso : recherche sur une grille fine, puis comparaison au segment.
gr = np.linspace(-1, 3, 401)
P_, Q_ = np.meshgrid(gr, gr)
pred_grille = np.multiply.outer(P_ + Q_, x_col)  # X b = x (b1 + b2) puisque les deux colonnes valent x
meilleur = float((0.5 * ((y_f1 - pred_grille) ** 2).sum(axis=-1) + np.abs(P_) + np.abs(Q_)).min())
assert abs(meilleur - valeurs_segment[0]) < 1e-12
# Hors du segment (signes opposés), le critère est strictement plus grand.
assert critere_lasso(np.array([2.5, -0.5])) > valeurs_segment[0] + 0.5
# Ridge : (1, 1) est l'unique minimum, et (1,5 ; 0,5), de même somme, fait strictement plus.
assert critere_ridge(np.array([1.5, 0.5])) > critere_ridge(np.array([1.0, 1.0]))
from sklearn.linear_model import Lasso, Ridge  # noqa: E402

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sk_lasso = Lasso(alpha=1.0 / 4, fit_intercept=False, tol=1e-12, max_iter=100000).fit(X_f1, y_f1).coef_
sk_ridge = Ridge(alpha=1.0, fit_intercept=False).fit(X_f1, y_f1).coef_
assert np.allclose(sk_ridge, [1, 1])
assert abs(sk_lasso.sum() - 2) < 1e-8 and (sk_lasso >= -1e-12).all()
V["F01_sk_lasso"] = [round(float(x), 6) + 0.0 for x in sk_lasso]
V["F01_sk_ridge"] = [round(float(x), 6) + 0.0 for x in sk_ridge]
V["F01_d_orth"] = 1.5
V["F01_d_somme"] = 0.75
dire(f"F01 : Lasso de scikit-learn (descente par coordonnées) renvoie {V['F01_sk_lasso']}, critère {V['F01_crit_lasso']}")

# =====================================================================================
# EB1-F02 : régression logistique, coefficient ln 2, probabilité 0,2
# =====================================================================================
p0 = Fraction(1, 5)
cote0 = p0 / (1 - p0)
cote1 = 2 * cote0
p1 = cote1 / (1 + cote1)
assert p1 == Fraction(1, 3)
b0_ = math.log(float(cote0))
assert abs(1 / (1 + math.exp(-(b0_ + math.log(2)))) - 1 / 3) < 1e-15
fraction("F02_exact", p1)
fraction("F02_cote0", cote0)
fraction("F02_cote1", cote1)
V["F02_d_double"] = 0.4
V["F02_d_additif"] = 0.2 + math.log(2)
fraction("F02_d_cote", cote1)

# =====================================================================================
# EB1-F03 : validation croisée purgée avec embargo
# =====================================================================================
tous_t = set(range(1, 1001))
test = set(range(401, 601))
etiquette = {t: set(range(t + 1, t + 6)) for t in tous_t}
union_test = set().union(*(etiquette[t] for t in test))
assert min(union_test) == 402 and max(union_test) == 605
purges = {t for t in tous_t - test if etiquette[t] & union_test}
embargo = set(range(601, 611))
entrainement = tous_t - test - purges - embargo
V["F03_exact"] = len(entrainement)
V["F03_purges_avant"] = len({t for t in purges if t < 401})
V["F03_purges_apres"] = len({t for t in purges if t > 600})
V["F03_premier_purge"] = min(purges)
V["F03_dernier_purge_apres"] = max(purges)
assert sorted(purges) == [397, 398, 399, 400, 601, 602, 603, 604]
V["F03_restants_avant"] = len({t for t in entrainement if t < 401})
V["F03_restants_apres"] = len({t for t in entrainement if t > 600})
V["F03_union_min"] = min(union_test)
V["F03_union_max"] = max(union_test)
V["F03_d_rien"] = len(tous_t - test)
V["F03_d_purge_seule"] = len(tous_t - test - purges)
V["F03_d_embargo_seul"] = len(tous_t - test - embargo)
purges_jours_test = {t for t in tous_t - test if etiquette[t] & test}
V["F03_d_mauvais_intervalle"] = len(tous_t - test - purges_jours_test - embargo)
assert V["F03_exact"] == 786 and V["F03_d_purge_seule"] == 792 and V["F03_d_mauvais_intervalle"] == 785

# =====================================================================================
# EB1-F04 : AUC avec une égalité de scores
# =====================================================================================
pos = [0.9, 0.7, 0.4]
neg = [0.8, 0.4, 0.3, 0.1]
paires = Fraction(0)
for sp_ in pos:
    for sn_ in neg:
        paires += 1 if sp_ > sn_ else (Fraction(1, 2) if sp_ == sn_ else 0)
auc = paires / (len(pos) * len(neg))
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

y_auc = [1] * len(pos) + [0] * len(neg)
s_auc = pos + neg
assert abs(roc_auc_score(y_auc, s_auc) - float(auc)) < 1e-15
fpr, tpr, _ = roc_curve(y_auc, s_auc)
aire = sum((fpr[i + 1] - fpr[i]) * (tpr[i + 1] + tpr[i]) / 2 for i in range(len(fpr) - 1))
assert abs(aire - float(auc)) < 1e-15  # aire sous la courbe par la méthode des trapèzes
fraction("F04_exact", auc)
V["F04_paires"] = len(pos) * len(neg)
V["F04_gagnees"] = float(paires)
fraction("F04_d_strict", Fraction(sum(sp_ > sn_ for sp_ in pos for sn_ in neg), 12))
fraction("F04_d_large", Fraction(sum(sp_ >= sn_ for sp_ in pos for sn_ in neg), 12))
exact_05 = sum(s > 0.5 for s in pos) + sum(s <= 0.5 for s in neg)
fraction("F04_d_exactitude", Fraction(exact_05, 7))

# =====================================================================================
# EB1-F05 : k plus proches voisins, décomposition biais-variance
# =====================================================================================
f_voisins = [Fraction(1), Fraction(3, 2), Fraction(2), Fraction(3, 2)]
f_x0, sigma2, k5 = Fraction(1), Fraction(1), 4
biais = sum(f_voisins) / k5 - f_x0
erreur = sigma2 + biais**2 + sigma2 / k5
assert erreur == Fraction(3, 2)
fraction("F05_exact", erreur)
fraction("F05_biais", biais)
fraction("F05_biais2", biais**2)
fraction("F05_variance", sigma2 / k5)
fraction("F05_moy_voisins", sum(f_voisins) / k5)
fraction("F05_d_sans_bruit", biais**2 + sigma2 / k5)
fraction("F05_d_sans_variance", sigma2 + biais**2)
fraction("F05_d_variance_pleine", sigma2 + biais**2 + sigma2)
bruit = rng.standard_normal((N_SIM, k5))
pred = (np.array([float(v) for v in f_voisins]) + bruit).mean(axis=1)
y0 = 1 + rng.standard_normal(N_SIM)
mc_moyenne("F05", 1.5, (y0 - pred) ** 2)

# =====================================================================================
# EB1-A01 : descente de gradient sur une quadratique mal conditionnée
# =====================================================================================
lam_max, lam_min = 10.0, 1.0
eta_opt = 2 / (lam_max + lam_min)
fraction("A01_eta_max", Fraction(2, 10))
fraction("A01_eta_opt", Fraction(2, 11))
fraction("A01_facteur", Fraction(9, 11))
fraction("A01_d_facteur_1L", Fraction(9, 10))
fraction("A01_kappa", Fraction(10))
etas = np.linspace(0.001, 0.25, 249001)
facteurs = np.maximum(np.abs(1 - etas * lam_max), np.abs(1 - etas * lam_min))
assert abs(etas[facteurs.argmin()] - 2 / 11) < 1e-5 and abs(facteurs.min() - 9 / 11) < 1e-5
assert (facteurs[etas < 0.2 - 1e-9] < 1).all() and (facteurs[etas >= 0.2] >= 1).all()


def descente(eta, n_iter=200, w0=(1.0, 1.0)):
    w = np.array(w0)
    for _ in range(n_iter):
        w = w - eta * np.array([lam_max, lam_min]) * w
    return np.linalg.norm(w)


assert descente(0.19) < 1e-3 and descente(0.2) > 0.99 and descente(0.21) > 1e3
# Facteur observé sur 50 itérations au pas optimal, départ (1, 1) : (9/11)^50 exactement pour chaque coordonnée.
assert abs(descente(eta_opt, 50) - math.sqrt(2) * (9 / 11) ** 50) < 1e-15
V["A01_norme_50_opt"] = descente(eta_opt, 50)
V["A01_norme_50_1L"] = descente(0.1, 50)
assert V["A01_norme_50_opt"] < V["A01_norme_50_1L"]

# =====================================================================================
# EB1-A02 : gradient de l'entropie croisée par rapport aux scores (softmax)
# =====================================================================================
z2 = np.array([2.0, 1.0, 0.0])
p2 = np.exp(z2) / np.exp(z2).sum()
y2 = np.array([0.0, 1.0, 0.0])
grad2 = p2 - y2


def perte2(z):
    return -np.log(np.exp(z[1]) / np.exp(z).sum())


h2 = 1e-6
grad_df = np.array([(perte2(z2 + h2 * e) - perte2(z2 - h2 * e)) / (2 * h2) for e in np.eye(3)])
assert np.allclose(grad_df, grad2, atol=1e-8)
zs = sp.symbols("z1 z2 z3")
perte_sym = -sp.log(sp.exp(zs[1]) / sum(sp.exp(z) for z in zs))
for j in range(3):
    attendu = sp.exp(zs[j]) / sum(sp.exp(z) for z in zs) - (1 if j == 1 else 0)
    assert sp.simplify(sp.diff(perte_sym, zs[j]) - attendu) == 0
V["A02_p"] = [float(x) for x in p2]
V["A02_exact"] = [float(x) for x in grad2]
V["A02_d_signe"] = [float(-x) for x in grad2]
V["A02_d_p"] = [float(x) for x in p2]
V["A02_d_dp"] = [0.0, float(-1 / p2[1]), 0.0]
V["A02_somme_exp"] = float(np.exp(z2).sum())
V["A02_perte"] = float(perte2(z2))
V["A02_e"] = math.e
V["A02_e2"] = math.exp(2)
assert abs(grad2.sum()) < 1e-15  # la somme des coordonnées du gradient est nulle

# =====================================================================================
# EB1-A03 : attention par produit scalaire normalisé
# =====================================================================================
q3 = np.array([1.0, 1.0, 0.0, 0.0])
cles = np.array([[2.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 3.0], [1.0, -1.0, 0.0, 0.0]])
valeurs3 = np.array([10.0, 4.0, -2.0])


def sortie_attention(echelle):
    s3 = cles @ q3 / echelle
    w3 = np.exp(s3) / np.exp(s3).sum()
    return float(w3 @ valeurs3), s3, w3


sortie3, scores3, poids3 = sortie_attention(math.sqrt(4))
V["A03_produits"] = [float(x) for x in cles @ q3]
V["A03_scores"] = [float(x) for x in scores3]
V["A03_poids"] = [float(x) for x in poids3]
V["A03_exact"] = sortie3
V["A03_e2"] = math.exp(2)
assert abs(sortie3 - (10 * math.exp(2) + 4 - 2) / (math.exp(2) + 2)) < 1e-12
V["A03_d_sans_echelle"] = sortie_attention(1.0)[0]
V["A03_d_echelle_d"] = sortie_attention(4.0)[0]
V["A03_d_moyenne"] = float(valeurs3.mean())
# Pourquoi diviser par la racine de d : si q et k ont des coordonnées indépendantes centrées réduites, Var(q.k) = d.
for d_ in (4, 64):
    qk = (rng.standard_normal((N_SIM, d_)) * rng.standard_normal((N_SIM, d_))).sum(axis=1)
    mc_variance(f"A03_var_d{d_}", d_, qk)

# =====================================================================================
# EB1-A04 : boosting de gradient à une feuille, taux d'apprentissage 0,1, 20 étapes
# =====================================================================================
nu, m4, ybar = Fraction(1, 10), 20, 10
pred4 = Fraction(0)
for _ in range(m4):
    pred4 += nu * (ybar - pred4)  # la feuille unique prédit la moyenne des pseudo-résidus
assert pred4 == ybar * (1 - (1 - nu) ** m4)
V["A04_exact"] = float(pred4)
V["A04_d_residu"] = float(ybar * (1 - nu) ** m4)
V["A04_d_lineaire"] = float(m4 * nu * ybar)
V["A04_facteur"] = float((1 - nu) ** m4)
V["A04_d_plein"] = ybar
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402

y4 = 10 + rng.standard_normal(50)
y4 = y4 - y4.mean() + 10  # moyenne exactement 10 (à l'arrondi près)
gbr = GradientBoostingRegressor(loss="squared_error", learning_rate=0.1, n_estimators=20, init="zero",
                                max_depth=1, random_state=0).fit(np.zeros((50, 1)), y4)
V["A04_sklearn"] = round(float(gbr.predict(np.zeros((1, 1)))[0]), 9)
assert abs(V["A04_sklearn"] - V["A04_exact"]) < 1e-8

# =====================================================================================
# EB1-A05 : une itération de l'algorithme EM, mélange de deux gaussiennes de variance 1
# =====================================================================================
x5 = np.array([-1.0, 0.0, 3.0])
mu1, mu2 = 0.0, 2.0
d1 = stats.norm.pdf(x5, mu1, 1)
d2 = stats.norm.pdf(x5, mu2, 1)
r2 = d2 / (d1 + d2)  # poids égaux 1/2 : ils se simplifient
assert np.allclose(r2, 1 / (1 + np.exp(-(2 * x5 - 2))))  # logit = 2x - 2, dérivé dans la correction
nouveau_mu2 = float((r2 * x5).sum() / r2.sum())
nouveau_mu1 = float(((1 - r2) * x5).sum() / (1 - r2).sum())
V["A05_r2"] = [float(x) for x in r2]
V["A05_logits"] = [float(x) for x in 2 * x5 - 2]
V["A05_somme_r2"] = float(r2.sum())
V["A05_somme_r2x"] = float((r2 * x5).sum())
V["A05_exact"] = nouveau_mu2
V["A05_mu1"] = nouveau_mu1
V["A05_d_dur"] = 3.0
V["A05_e_moins2"] = math.exp(-2)
V["A05_e_moins4"] = math.exp(-4)
fraction("A05_d_moyenne", Fraction(2, 3))
V["A05_d_non_normalise"] = float((r2 * x5).sum())


def log_vrais(m1, m2):
    return float(np.log(0.5 * stats.norm.pdf(x5, m1, 1) + 0.5 * stats.norm.pdf(x5, m2, 1)).sum())


V["A05_logv_avant"] = log_vrais(mu1, mu2)
V["A05_logv_apres"] = log_vrais(nouveau_mu1, nouveau_mu2)
assert V["A05_logv_apres"] > V["A05_logv_avant"]  # propriété de monotonie d'EM, constatée ici

# =====================================================================================
# EB1-A06 : AIC et BIC
# =====================================================================================
n6m = 100
modeles = {"A": (3, -140.0), "B": (6, -136.0)}
aic = {m: 2 * k - 2 * ll for m, (k, ll) in modeles.items()}
bic = {m: k * math.log(n6m) - 2 * ll for m, (k, ll) in modeles.items()}
assert min(aic, key=aic.get) == "B" and min(bic, key=bic.get) == "A"
V["A06_aic_A"] = aic["A"]
V["A06_aic_B"] = aic["B"]
V["A06_bic_A"] = bic["A"]
V["A06_bic_B"] = bic["B"]
V["A06_ln_n"] = math.log(n6m)
V["A06_lr"] = 2 * (modeles["B"][1] - modeles["A"][1])
V["A06_seuil_bic"] = 3 * math.log(n6m)

# =====================================================================================
# EB1-A07 : étiquettes chevauchantes, variance de la moyenne
# =====================================================================================
n7m, h7 = 250, 10
c_s = [sum(1 for t in range(1, n7m + 1) if t + 1 <= s <= t + h7) for s in range(1, n7m + h7 + 1)]
somme_c2 = sum(c * c for c in c_s)
assert somme_c2 == (n7m - h7 + 1) * h7**2 + 2 * sum(j * j for j in range(1, h7))
rapport7 = Fraction(somme_c2, n7m * h7)
# Méthode indépendante : somme des covariances Cov(Y_s, Y_t) = sigma^2 max(h - |s - t|, 0).
idx = np.arange(n7m)
cov7 = np.maximum(h7 - np.abs(idx[:, None] - idx[None, :]), 0)
assert cov7.sum() == somme_c2
fraction("A07_exact", rapport7)
assert rapport7 == h7 - Fraction(h7**2 - 1, 3 * n7m)  # forme fermée h - (h^2 - 1)/(3n)
fraction("A07_correction", Fraction(h7**2 - 1, 3 * n7m))
V["A07_var_theorique"] = float(rapport7) * h7 / n7m
for n_, h_ in [(30, 4), (100, 7), (250, 10), (1000, 21)]:
    cc = [sum(1 for t in range(1, n_ + 1) if t + 1 <= s <= t + h_) for s in range(1, n_ + h_ + 1)]
    assert Fraction(sum(c * c for c in cc), n_ * h_) == h_ - Fraction(h_**2 - 1, 3 * n_)
V["A07_somme_c2"] = somme_c2
V["A07_nb_pleins"] = n7m - h7 + 1
V["A07_somme_carres"] = sum(j * j for j in range(1, h7))
V["A07_d_naif"] = 1
V["A07_d_limite"] = h7
V["A07_d_racine"] = math.sqrt(float(rapport7))
V["A07_correlation_3"] = float(Fraction(h7 - 3, h7))
poids7 = np.array(c_s, dtype=float) / n7m
moyennes7 = []
for _ in range(10):
    moyennes7.append(rng.standard_normal((20_000, n7m + h7)) @ poids7)
moyennes7 = np.concatenate(moyennes7)
mc_variance("A07", float(rapport7) * h7 / n7m, moyennes7)  # Var(Ybar) = rapport x h / n pour sigma = 1

# =====================================================================================
# EB1-A08 : processus gaussien, loi a posteriori en un point
# =====================================================================================
k01 = math.exp(-0.5)
bruit8, y8 = 0.5, 2.0
moy8 = k01 / (1 + bruit8) * y8
var8 = 1 - k01**2 / (1 + bruit8)
V["A08_k01"] = k01
V["A08_moyenne"] = moy8
V["A08_variance"] = var8
V["A08_d_moy_sans_bruit"] = k01 * y8
V["A08_d_var_sans_bruit"] = 1 - k01**2
V["A08_d_var_k"] = 1 - k01 / (1 + bruit8)
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import RBF  # noqa: E402

gpr = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=bruit8, optimizer=None, normalize_y=False)
gpr.fit(np.array([[0.0]]), np.array([y8]))
m_sk, s_sk = gpr.predict(np.array([[1.0]]), return_std=True)
assert abs(m_sk[0] - moy8) < 1e-12 and abs(s_sk[0] ** 2 - var8) < 1e-12
# Monte-Carlo : tirages de (f(0), f(1), bruit), régression de f(1) sur y ; pente et variance résiduelle.
cov8 = np.array([[1.0, k01], [k01, 1.0]])
f8 = rng.multivariate_normal([0.0, 0.0], cov8, size=N_SIM)
yy8 = f8[:, 0] + math.sqrt(bruit8) * rng.standard_normal(N_SIM)
pente = (f8[:, 1] @ yy8) / (yy8 @ yy8)
residus8 = f8[:, 1] - pente * yy8
var_res = float(residus8 @ residus8 / N_SIM)
controle_mc("A08_moyenne", moy8, pente * y8, y8 * math.sqrt(var_res / (yy8 @ yy8)))
mc_variance("A08_variance", var8, residus8)

# =====================================================================================
# EB1-C1 : question de code, exemple de l'énoncé et pouvoir des tests cachés
# =====================================================================================
DOSSIER = RACINE / "coding" / "examen1"
sys.path.insert(0, str(DOSSIER))
import tests as tests_c1  # noqa: E402
from _correction import rendement_long_short  # noqa: E402

ex = tests_c1.cadre(tests_c1.EXEMPLE)
V["C_exemple"] = rendement_long_short(ex, 1)
assert abs(V["C_exemple"] - (-0.15)) < 1e-12
d_ex = ex.sort_values(["actif", "date"])
g_ex = d_ex.groupby("actif")["prix"]
d_ex = d_ex.assign(passe=d_ex["prix"] / g_ex.shift(1) - 1, futur=g_ex.shift(-1) / d_ex["prix"] - 1)
for date_, nom in [("2026-01-06", "0106"), ("2026-01-07", "0107")]:
    lignes_d = d_ex[d_ex["date"] == date_].set_index("actif")
    V[f"C_passe_{nom}"] = [round(float(x), 12) + 0.0 for x in lignes_d["passe"]]
    V[f"C_futur_{nom}"] = [round(float(x), 12) + 0.0 for x in lignes_d["futur"]]
assert V["C_passe_0106"] == [0.1, -0.1, 0.05] and V["C_futur_0106"] == [-0.1, 0.2, 0.0]
assert V["C_passe_0107"] == [-0.1, 0.2, 0.0] and V["C_futur_0107"] == [0.0, 0.0, 0.1]
V["C_date_0106"] = round(float(-0.1 - 0.2), 12)
V["C_date_0107"] = 0.0
V["C_egalite"] = rendement_long_short(tests_c1.cadre(tests_c1.EGALITE), 1)
assert 110.0 / 100.0 == 55.0 / 50.0  # même réel 1,1, donc même flottant : l'égalité est exacte
assert abs(V["C_egalite"] - 0.2) < 1e-12


def variante_decalage_non_trie(df, k):
    d = df.copy()
    d["passe"] = d["prix"] / d.groupby("actif")["prix"].shift(1) - 1  # erreur : cadre non trié
    d["futur"] = d.groupby("actif")["prix"].shift(-1) / d["prix"] - 1
    d = d.dropna(subset=["passe", "futur"])
    r = [g.sort_values(["passe", "actif"])["futur"].iloc[-k:].mean() - g.sort_values(["passe", "actif"])["futur"].iloc[:k].mean()
         for _, g in d.groupby("date") if len(g) >= 2 * k]
    return float(np.mean(r)) if r else float("nan")


def variante_pct_change(df, k):
    d = df.sort_values(["actif", "date"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = d.assign(passe=d.groupby("actif")["prix"].pct_change())  # erreur : remplit les prix manquants
    d["futur"] = d.groupby("actif")["passe"].shift(-1)
    d = d.dropna(subset=["passe", "futur"])
    r = [g.sort_values(["passe", "actif"])["futur"].iloc[-k:].mean() - g.sort_values(["passe", "actif"])["futur"].iloc[:k].mean()
         for _, g in d.groupby("date") if len(g) >= 2 * k]
    return float(np.mean(r)) if r else float("nan")


def variante_nlargest(df, k):
    d = df.sort_values(["actif", "date"])
    g = d.groupby("actif")["prix"]
    d = d.assign(passe=d["prix"] / g.shift(1) - 1, futur=g.shift(-1) / d["prix"] - 1).dropna(subset=["passe", "futur"])
    r = [g.nlargest(k, "passe")["futur"].mean() - g.nsmallest(k, "passe")["futur"].mean()  # erreur : égalités mal départagées
         for _, g in d.groupby("date") if len(g) >= 2 * k]
    return float(np.mean(r)) if r else float("nan")


def variante_pivot(df, k):
    larg = df.pivot(index="date", columns="actif", values="prix").sort_index()  # erreur : date précédente du calendrier
    passe = (larg / larg.shift(1) - 1).stack().rename("passe")
    futur = (larg.shift(-1) / larg - 1).stack().rename("futur")
    d = pd.concat([passe, futur], axis=1).dropna().reset_index()
    r = [g.sort_values(["passe", "actif"])["futur"].iloc[-k:].mean() - g.sort_values(["passe", "actif"])["futur"].iloc[:k].mean()
         for _, g in d.groupby("date") if len(g) >= 2 * k]
    return float(np.mean(r)) if r else float("nan")


def variante_correcte_tri_par_date(df, k):
    # Variante correcte : trier par date suffit, groupby conserve l'ordre des lignes dans chaque groupe.
    d = df.sort_values("date")
    g = d.groupby("actif")["prix"]
    d = d.assign(passe=d["prix"] / g.shift(1) - 1, futur=g.shift(-1) / d["prix"] - 1).dropna(subset=["passe", "futur"])
    r = []
    for _, h in d.groupby("date"):
        if len(h) >= 2 * k:
            h = h.sort_values(["passe", "actif"])
            r.append(h["futur"].iloc[-k:].mean() - h["futur"].iloc[:k].mean())
    return float(np.mean(r)) if r else float("nan")


CAS_CACHES = tests_c1.cas_caches()


def score(fonction):
    cas = CAS_CACHES
    ok = 0
    for c in cas:
        try:
            ok += bool(tests_c1.comparer(fonction(*c["args"]), c["attendu"]))
        except Exception:  # noqa: BLE001 : une variante qui plante échoue au test
            pass
    return ok, len(cas)


for nom, fonction, doit_reussir in [
    ("C_var_decalage_non_trie", variante_decalage_non_trie, False),
    ("C_var_pct_change", variante_pct_change, False),
    ("C_var_nlargest", variante_nlargest, False),
    ("C_var_pivot", variante_pivot, False),
    ("C_var_tri_par_date", variante_correcte_tri_par_date, True),
]:
    reussis, total = score(fonction)
    assert (reussis == total) == doit_reussir, (nom, reussis, total)
    V[nom + "_reussis"] = reussis
    V[nom + "_total"] = total
    dire(f"{nom:28s} {reussis}/{total} tests cachés")
V["C_nb_caches"] = len(CAS_CACHES)
V["C_nb_visibles"] = len(tests_c1.cas_visibles())

# =====================================================================================
# Sorties
# =====================================================================================
Path(__file__).with_name("examen_blanc_1.out.json").write_bytes((json.dumps(V, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
dire(f"{len(V)} valeurs exportées dans verif/examen_blanc_1.out.json.")
dire("Toutes les vérifications passent.")
Path(__file__).with_name("examen_blanc_1.out.txt").write_bytes(("\n".join(LIGNES) + "\n").encode("utf-8"))  # fins de ligne Unix
