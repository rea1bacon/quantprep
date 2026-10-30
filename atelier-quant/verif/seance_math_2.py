"""Vérification de tous les nombres de la séance de mathématiques 2 (leçon L-MATH-2 et questions M2-01 à M2-08).

Thèmes : conditionnement (Bayes, espérance et variance totales), lois usuelles et leurs moments, absence de mémoire,
minimum d'exponentielles, processus de Poisson, collectionneur de coupons, dénombrement et inclusion-exclusion.

Chaque probabilité ou espérance a deux vérifications indépendantes :
  1. un calcul exact (fractions, énumération complète, sympy) ;
  2. une simulation Monte-Carlo, dont l'écart à l'exact doit rester inférieur à 4 erreurs standard.
Les formules générales de la leçon (moments du tableau, densité gamma, fonction de répartition d'Erlang,
inclusion-exclusion, dérangements) sont vérifiées symboliquement ou par énumération sur plusieurs cas.
Les distracteurs et l'arithmétique citée dans les corrections sont aussi calculés ici.

Graine fixe : la sortie est identique à chaque exécution.
Sorties : verif/seance_math_2.out.json (lu par outils/construire.py) et verif/seance_math_2.out.txt (ce résumé).
"""
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp

rng = np.random.default_rng(20261007)
V = {}
LIGNES = []
N_SIM = 400_000


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
    v = (c**2).mean()
    controle_mc(nom, exact, v, np.sqrt(((c**2 - v) ** 2).mean() / len(x)))


def poisson_pmf(mu, k):
    """P(N = k) pour N de loi de Poisson de paramètre mu (sympy, exact)."""
    return sp.exp(-mu) * mu**k / sp.factorial(k)


def poisson_cdf(mu, k):
    return sum(poisson_pmf(mu, j) for j in range(k + 1))


def derangements(n):
    """Nombre de permutations de n éléments sans point fixe, par la formule d'inclusion-exclusion."""
    return sum((-1) ** k * math.factorial(n) // math.factorial(k) for k in range(n + 1))


def surjections(k, n):
    """Nombre de suites de longueur k à valeurs dans n symboles où chaque symbole apparaît (inclusion-exclusion)."""
    return sum((-1) ** j * math.comb(n, j) * (n - j) ** k for j in range(n + 1))


# =====================================================================================
# FORMULES GÉNÉRALES DE LA LEÇON
# =====================================================================================
p_, lam_, mu_, x_, t_, u_ = sp.symbols("p lambda mu x t u", positive=True)
a_, b_ = sp.symbols("a b", real=True)
sig_ = sp.symbols("sigma", positive=True)
k_ = sp.symbols("k", integer=True, nonnegative=True)

# Bernoulli et binomiale (n de 1 à 8, p symbolique) : espérance n p, variance n p (1 - p).
for n in range(1, 9):
    e1 = sum(k * sp.binomial(n, k) * p_**k * (1 - p_) ** (n - k) for k in range(n + 1))
    e2 = sum(k * k * sp.binomial(n, k) * p_**k * (1 - p_) ** (n - k) for k in range(n + 1))
    assert sp.expand(e1 - n * p_) == 0 and sp.expand(e2 - e1**2 - n * p_ * (1 - p_)) == 0
# Géométrique sur {1, 2, ...} : espérance 1/p, second moment (2 - p)/p^2, variance (1 - p)/p^2.
for pv in [Fraction(1, 6), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(9, 10)]:
    pr = sp.Rational(pv.numerator, pv.denominator)
    loi = (1 - pr) ** (k_ - 1) * pr
    e1 = sp.summation(k_ * loi, (k_, 1, sp.oo))
    e2 = sp.summation(k_**2 * loi, (k_, 1, sp.oo))
    queue = sp.summation((1 - pr) ** k_, (k_, 0, sp.oo))  # E[G] = somme des P(G > k), k >= 0
    assert e1 == 1 / pr == queue and e2 == (2 - pr) / pr**2 and e2 - e1**2 == (1 - pr) / pr**2
# Second moment de la géométrique par le premier essai : E2 = p * 1 + (1 - p) E[(1 + G)^2].
E2 = sp.symbols("E2")
sol = sp.solve(sp.Eq(E2, p_ + (1 - p_) * (1 + 2 / p_ + E2)), E2)[0]
assert sp.simplify(sol - (2 - p_) / p_**2) == 0
# Poisson : espérance et variance lambda, moment factoriel lambda^2.
e1 = sp.simplify(sp.summation(k_ * poisson_pmf(lam_, k_), (k_, 0, sp.oo)))
ef = sp.simplify(sp.summation(k_ * (k_ - 1) * poisson_pmf(lam_, k_), (k_, 0, sp.oo)))
assert sp.simplify(e1 - lam_) == 0 and sp.simplify(ef - lam_**2) == 0 and sp.simplify(ef + e1 - e1**2 - lam_) == 0
# Uniforme sur [a, b] : (a + b)/2 et (b - a)^2/12.
dens = 1 / (b_ - a_)
e1 = sp.integrate(x_ * dens, (x_, a_, b_))
e2 = sp.integrate(x_**2 * dens, (x_, a_, b_))
assert sp.simplify(e1 - (a_ + b_) / 2) == 0 and sp.simplify(e2 - e1**2 - (b_ - a_) ** 2 / 12) == 0
# Exponentielle : 1/lambda et 1/lambda^2, par la densité et par la formule de la queue.
dens = lam_ * sp.exp(-lam_ * x_)
assert sp.integrate(dens, (x_, 0, sp.oo)) == 1
e1 = sp.integrate(x_ * dens, (x_, 0, sp.oo))
e2 = sp.integrate(x_**2 * dens, (x_, 0, sp.oo))
assert e1 == 1 / lam_ == sp.integrate(sp.exp(-lam_ * x_), (x_, 0, sp.oo))
assert e2 == 2 / lam_**2 == sp.integrate(2 * x_ * sp.exp(-lam_ * x_), (x_, 0, sp.oo))
# Normale : espérance mu, variance sigma^2 ; et la relation phi'(z) = -z phi(z) utilisée par la preuve.
y_ = sp.symbols("y", real=True)
dens = sp.exp(-((y_ - mu_) ** 2) / (2 * sig_**2)) / (sig_ * sp.sqrt(2 * sp.pi))
assert sp.simplify(sp.integrate(dens, (y_, -sp.oo, sp.oo))) == 1
assert sp.simplify(sp.integrate(y_ * dens, (y_, -sp.oo, sp.oo)) - mu_) == 0
assert sp.simplify(sp.integrate((y_ - mu_) ** 2 * dens, (y_, -sp.oo, sp.oo)) - sig_**2) == 0
phi = sp.exp(-(y_**2) / 2) / sp.sqrt(2 * sp.pi)
assert sp.simplify(sp.diff(phi, y_) + y_ * phi) == 0
dire("Moments du tableau des lois usuelles vérifiés (binomiale n <= 8, géométrique 5 valeurs de p, Poisson, uniforme, exponentielle, normale).")

# Somme aléatoire : E[S] = E[N] m, Var(S) = E[N] s2 + Var(N) m^2 (N indépendant des X_i), vérifié pour N de Poisson
# et binomiale par sommation exacte de E[S^2 | N = n] = n s2 + n^2 m^2.
m_, s2_ = sp.symbols("m s2", positive=True)
n_ = sp.symbols("n", integer=True, nonnegative=True)
es = sp.simplify(sp.summation(poisson_pmf(lam_, n_) * n_ * m_, (n_, 0, sp.oo)))
es2 = sp.simplify(sp.summation(poisson_pmf(lam_, n_) * (n_ * s2_ + n_**2 * m_**2), (n_, 0, sp.oo)))
assert sp.simplify(es - lam_ * m_) == 0 and sp.simplify(es2 - es**2 - (lam_ * s2_ + lam_ * m_**2)) == 0
for nn in range(1, 7):
    pmf = {j: sp.binomial(nn, j) * p_**j * (1 - p_) ** (nn - j) for j in range(nn + 1)}
    es = sum(w * j * m_ for j, w in pmf.items())
    es2 = sum(w * (j * s2_ + j * j * m_**2) for j, w in pmf.items())
    assert sp.expand(es2 - es**2 - (nn * p_ * s2_ + nn * p_ * (1 - p_) * m_**2)) == 0
dire("Formule de la variance d'une somme aléatoire vérifiée (N de Poisson symbolique, N binomial n <= 6).")

# Processus de Poisson : densité gamma par convolution (récurrence), fonction de répartition d'Erlang, E[S_k] = k/lambda.
s_ = sp.symbols("s", positive=True)
f = lam_ * sp.exp(-lam_ * s_)  # densité de S_1
for k in range(1, 7):
    assert sp.simplify(f - lam_**k * s_ ** (k - 1) * sp.exp(-lam_ * s_) / sp.factorial(k - 1)) == 0
    # P(S_k > t) = P(N(t) <= k - 1) : la dérivée de 1 - P(N(t) <= k-1) est la densité de S_k, et les deux valent 0 en 0.
    queue = sum(sp.exp(-lam_ * t_) * (lam_ * t_) ** j / sp.factorial(j) for j in range(k))
    assert sp.simplify(sp.diff(1 - queue, t_) - f.subs(s_, t_)) == 0 and sp.simplify((1 - queue).subs(t_, 0)) == 0
    assert sp.simplify(sp.integrate(queue, (t_, 0, sp.oo)) - k / lam_) == 0
    # Récurrence : densité de S_{k+1} = convolution de celle de S_k avec une exponentielle.
    f = sp.simplify(sp.integrate(f.subs(s_, u_) * lam_ * sp.exp(-lam_ * (s_ - u_)), (u_, 0, s_)))
for j in range(8):
    assert sp.integrate(u_**j * sp.exp(-u_), (u_, 0, sp.oo)) == sp.factorial(j)
# Somme de deux Poisson indépendantes : convolution et binôme.
a2, b2 = sp.symbols("a2 b2", positive=True)
for k in range(8):
    conv = sum(poisson_pmf(a2, j) * poisson_pmf(b2, k - j) for j in range(k + 1))
    assert sp.simplify(conv - poisson_pmf(a2 + b2, k)) == 0
dire("Densité gamma (k <= 6), fonction de répartition d'Erlang, E[S_k] = k/lambda et superposition (k <= 7) vérifiées.")

# Minimum d'exponentielles indépendantes : P(X_1 < X_2) = l1/(l1 + l2), et P(argmin = 1, min > t) = (l1/l) e^{-l t}.
l1, l2, l3 = sp.symbols("l1 l2 l3", positive=True)
assert sp.simplify(sp.integrate(l1 * sp.exp(-l1 * x_) * sp.exp(-l2 * x_), (x_, 0, sp.oo)) - l1 / (l1 + l2)) == 0
joint = sp.integrate(l1 * sp.exp(-l1 * x_) * sp.exp(-(l2 + l3) * x_), (x_, t_, sp.oo))
assert sp.simplify(joint - l1 / (l1 + l2 + l3) * sp.exp(-(l1 + l2 + l3) * t_)) == 0

# Inclusion-exclusion, dérangements, surjections, étoiles et barres : énumération sur de petits cas.
for n in range(1, 9):
    perms = list(itertools.permutations(range(n)))
    assert sum(all(s[i] != i for i in range(n)) for s in perms) == derangements(n)
for n in range(1, 5):
    for k in range(1, 7):
        assert sum(len(set(c)) == n for c in itertools.product(range(n), repeat=k)) == surjections(k, n)
for n in range(1, 6):
    for k in range(0, 6):
        assert len(list(itertools.combinations_with_replacement(range(n), k))) == math.comb(n + k - 1, k)
# Identité de développement : prod (1 - x_j) = somme sur les parties J de (-1)^{|J|} prod_{j dans J} x_j, m <= 5.
for m in range(1, 6):
    xs = sp.symbols(f"x1:{m + 1}")
    dev = sum((-1) ** len(J) * sp.Mul(*[xs[j] for j in J]) for r in range(m + 1) for J in itertools.combinations(range(m), r))
    assert sp.expand(sp.Mul(*[1 - xv for xv in xs]) - dev) == 0
# Encadrement des nombres harmoniques : ln(n + 1) <= H_n <= 1 + ln n.
for n in range(1, 200):
    hn = sum(1 / j for j in range(1, n + 1))
    assert math.log(n + 1) <= hn <= 1 + math.log(n) + 1e-12
# Reste de la série alternée : |d_n/n! - e^{-1}| <= 1/(n+1)!.
for n in range(1, 15):
    assert abs(derangements(n) / math.factorial(n) - math.exp(-1)) <= 1 / math.factorial(n + 1) + 1e-15
dire("Dérangements (n <= 8), surjections, étoiles et barres, développement du produit, bornes harmoniques vérifiés.")

# =====================================================================================
# LEÇON, PARTIE 1 : conditionnement
# =====================================================================================
# Exemple de Bayes : stratégie rentable a priori avec probabilité 0,1 ; backtest concluant avec probabilité 0,9 si
# rentable, 0,2 sinon.
pa, sens, faux = Fraction(1, 10), Fraction(9, 10), Fraction(2, 10)
fraction("La_joint", pa * sens)
fraction("La_faux", (1 - pa) * faux)
fraction("La_total", pa * sens + (1 - pa) * faux)
post = pa * sens / (pa * sens + (1 - pa) * faux)
assert post == Fraction(1, 3)
fraction("La_post", post)
vraie = rng.random(N_SIM) < float(pa)
concl = np.where(vraie, rng.random(N_SIM) < float(sens), rng.random(N_SIM) < float(faux))
mc_proba("La_post", post, vraie[concl])

# Somme aléatoire : N de Poisson de paramètre 3, montants d'espérance 10 et d'écart-type 5 (loi gamma de forme 4,
# d'échelle 2,5 pour la simulation), indépendants entre eux et de N.
lam_b, m_b, s2_b = 3, 10, 25
V["Lb_es"] = lam_b * m_b
V["Lb_intra"] = lam_b * s2_b
V["Lb_inter"] = lam_b * m_b**2
V["Lb_var"] = lam_b * s2_b + lam_b * m_b**2
nb = rng.poisson(lam_b, N_SIM)
assert nb.max() < 25
montants = rng.gamma(4.0, 2.5, size=(N_SIM, 25))
sb = np.where(np.arange(25)[None, :] < nb[:, None], montants, 0.0).sum(axis=1)
mc_moyenne("Lb_es", V["Lb_es"], sb)
mc_variance("Lb_var", V["Lb_var"], sb)

# Point de contrôle « vartot » : Y de Bernoulli(1/2) ; X sachant Y = 0 de loi N(0, 1), sachant Y = 1 de loi N(4, 1).
V["Cv_intra"] = 1
V["Cv_inter"] = float(Fraction(16, 4))  # Var(4 Y) = 16 Var(Y) = 16/4
V["Cv_exact"] = V["Cv_intra"] + V["Cv_inter"]
V["Cv_d_carre"] = float(16 * Fraction(1, 2)) + 1  # E[(4Y)^2] + 1, sans retrancher le carré de la moyenne
assert V["Cv_exact"] == 5
yv = rng.random(N_SIM) < 0.5
xv = np.where(yv, 4.0, 0.0) + rng.standard_normal(N_SIM)
mc_variance("Cv", 5, xv)

# =====================================================================================
# LEÇON, PARTIE 2 : lois usuelles ; exemple du dé lancé jusqu'au premier 6
# =====================================================================================
pg = Fraction(1, 6)
V["Lc_geo_e"] = float(1 / pg)
V["Lc_geo_e2"] = float((2 - pg) / pg**2)
V["Lc_geo_var"] = float((1 - pg) / pg**2)
assert (V["Lc_geo_e"], V["Lc_geo_e2"], V["Lc_geo_var"]) == (6, 66, 30)
g = rng.geometric(float(pg), N_SIM)  # numpy : loi géométrique sur {1, 2, ...}
mc_moyenne("Lc_geo_e", 6, g)
mc_variance("Lc_geo_var", 30, g)

# =====================================================================================
# LEÇON, PARTIE 3 : absence de mémoire, minimum d'exponentielles
# =====================================================================================
# Exemple : X exponentielle d'espérance 2 (lambda = 1/2), P(X > 5 | X > 3) = P(X > 2) = e^{-1}.
V["Ld_cond"] = math.exp(-1)
xe = rng.exponential(2.0, 4 * N_SIM)
mc_proba("Ld_cond", math.exp(-1), xe[xe > 3] > 5)
mc_moyenne("Ld_esp_cond", 5, xe[xe > 3])
V["Ld_esp_cond"] = 5

# Point de contrôle « memoire » : dé, aucun 6 lors des 4 premiers lancers ; espérance du nombre total de lancers.
V["Cm_total"] = 4 + 6
V["Cm_d_sans"] = 6
V["Cm_d_du"] = 6 - 4
V["Cm_d_echecs"] = 4 + 5
g2 = rng.geometric(1 / 6, 4 * N_SIM)
mc_moyenne("Cm_total", 10, g2[g2 > 4])

# Exemple : trois exponentielles indépendantes de paramètres 1, 2, 3.
rates = [1, 2, 3]
V["Le_lambda"] = sum(rates)
fraction("Le_emin", Fraction(1, sum(rates)))
fraction("Le_p1", Fraction(rates[0], sum(rates)))
fraction("Le_p3", Fraction(rates[2], sum(rates)))
V["Le_joint"] = (1 / 6) * math.exp(-6 * 0.2)
ee = rng.exponential(1.0, size=(N_SIM, 3)) / np.array(rates, dtype=float)
mn = ee.min(axis=1)
am = ee.argmin(axis=1)
mc_moyenne("Le_emin", Fraction(1, 6), mn)
mc_proba("Le_p1", Fraction(1, 6), am == 0)
mc_proba("Le_p3", Fraction(1, 2), am == 2)
mc_proba("Le_joint", V["Le_joint"], (am == 0) & (mn > 0.2))

# Point de contrôle « course » : paramètres 1, 2, 5 ; probabilité que celle de paramètre 2 soit la première.
fraction("Cc_exact", Fraction(2, 1 + 2 + 5))
fraction("Cc_d_moy", Fraction(1, 2) / (1 + Fraction(1, 2) + Fraction(1, 5)))
fraction("Cc_d_prod", Fraction(2, 3) * Fraction(2, 7))
ec = rng.exponential(1.0, size=(N_SIM, 3)) / np.array([1.0, 2.0, 5.0])
mc_proba("Cc", Fraction(1, 4), ec.argmin(axis=1) == 1)

# =====================================================================================
# LEÇON, PARTIE 4 : processus de Poisson
# =====================================================================================
lam_f, t_f = 2, Fraction(3, 2)
mu_f = lam_f * t_f
assert mu_f == 3
V["Lf_mu"] = float(mu_f)
V["Lf_p0"] = float(poisson_pmf(sp.Integer(3), 0))
V["Lf_p2"] = float(poisson_pmf(sp.Integer(3), 2))
fraction("Lf_p2_coef", Fraction(3**2, 2))
fraction("Lf_es3", Fraction(3, lam_f))
V["Lf_sup"] = 2 + 3
V["Lf_sup_p5"] = float(poisson_pmf(sp.Integer(5), 5))
fraction("Lf_premier", Fraction(2, 5))
fraction("Lf_unif", Fraction(1, 3))  # P(S_1 <= 0,5 | N(1,5) = 1) = 0,5/1,5


def simuler_poisson(lam, horizon, n_sim, n_max=60):
    """Instants d'arrivée d'un processus de Poisson simulé par des durées exponentielles indépendantes."""
    arrivees = np.cumsum(rng.exponential(1.0 / lam, size=(n_sim, n_max)), axis=1)
    assert (arrivees[:, -1] > horizon).all()
    return arrivees


arr = simuler_poisson(2.0, 1.5, N_SIM)
nf = (arr <= 1.5).sum(axis=1)
mc_proba("Lf_p0", V["Lf_p0"], nf == 0)
mc_proba("Lf_p2", V["Lf_p2"], nf == 2)
mc_moyenne("Lf_es3", 1.5, arr[:, 2])
mc_proba("Lf_unif", Fraction(1, 3), arr[nf == 1, 0] <= 0.5)
arr_a = simuler_poisson(2.0, 1.0, N_SIM)
arr_b = simuler_poisson(3.0, 1.0, N_SIM)
tot = (arr_a <= 1.0).sum(axis=1) + (arr_b <= 1.0).sum(axis=1)
mc_proba("Lf_sup_p5", V["Lf_sup_p5"], tot == 5)
mc_proba("Lf_premier", Fraction(2, 5), arr_a[:, 0] < arr_b[:, 0])

# =====================================================================================
# LEÇON, PARTIE 5 : collectionneur de coupons
# =====================================================================================
h6 = sum(Fraction(1, j) for j in range(1, 7))
fraction("Lg_h6", h6)
fraction("Lg_e", 6 * h6)
assert 6 * h6 == Fraction(147, 10)
V["Lg_ln7"] = math.log(7)
V["Lg_1ln6"] = 1 + math.log(6)
V["Lg_6ln6"] = 6 * math.log(6)
assert V["Lg_ln7"] <= float(h6) <= V["Lg_1ln6"]
V["Cg_d_carre"] = 36


def attente_collection(n_types, n_voulus, n_sim, longueur):
    """Nombre de tirages uniformes avec remise jusqu'à avoir vu n_voulus types distincts."""
    t = rng.integers(0, n_types, size=(n_sim, longueur))
    vu = np.zeros((n_sim, n_types), dtype=bool)
    distincts = np.zeros(n_sim, dtype=np.int64)
    fin = np.zeros(n_sim, dtype=np.int64)
    for j in range(longueur):
        nouveau = ~vu[np.arange(n_sim), t[:, j]]
        vu[np.arange(n_sim), t[:, j]] = True
        distincts += nouveau
        atteint = (distincts == n_voulus) & (fin == 0)
        fin[atteint] = j + 1
    assert (fin > 0).all()
    return fin


mc_moyenne("Lg_e", 6 * h6, attente_collection(6, 6, N_SIM // 2, 160))

# =====================================================================================
# LEÇON, PARTIE 6 : dénombrement, inclusion-exclusion, dérangements
# =====================================================================================
V["Lh_anag"] = math.factorial(6) // (math.factorial(2) * math.factorial(2))
assert V["Lh_anag"] == len(set(itertools.permutations("BANANE")))
V["Lh_eb"] = math.comb(5 + 3 - 1, 3 - 1)
assert V["Lh_eb"] == sum(1 for c in itertools.product(range(6), repeat=3) if sum(c) == 5)
# Deux dés : l'issue non ordonnée {1, 2} a deux fois la probabilité de {1, 1} ; les 21 issues non ordonnées
# ne sont pas équiprobables.
deux = list(itertools.product(range(1, 7), repeat=2))
fraction("Lh_mixte", Fraction(sum(sorted(c) == [1, 2] for c in deux), 36))
fraction("Lh_double", Fraction(sum(c == (1, 1) for c in deux), 36))
V["Lh_nb_non_ord"] = math.comb(6 + 2 - 1, 2)
assert V["Lh_nb_non_ord"] == len({tuple(sorted(c)) for c in deux})
# Trois couleurs, cinq tirages uniformes avec remise : probabilité de voir les trois couleurs.
V["Lh_c_tot"] = 3**5
V["Lh_c_un"] = 3 * 2**5
V["Lh_c_deux"] = 3 * 1**5
fraction("Lh_c_p", Fraction(surjections(5, 3), 3**5))
assert surjections(5, 3) == V["Lh_c_tot"] - V["Lh_c_un"] + V["Lh_c_deux"]
V["Lh_c_fav"] = surjections(5, 3)
cc = rng.integers(0, 3, size=(N_SIM, 5))
vues = np.stack([(cc == c).any(axis=1) for c in range(3)], axis=1).all(axis=1)
mc_proba("Lh_c_p", Fraction(surjections(5, 3), 3**5), vues)
# Dérangements.
V["Lh_d4"] = derangements(4)
fraction("Lh_p4", Fraction(derangements(4), math.factorial(4)))
V["Lh_p10"] = derangements(10) / math.factorial(10)
V["Lh_e_moins_un"] = math.exp(-1)
V["Lh_reste10"] = 1 / math.factorial(11)
assert derangements(4) == 9
perm = np.argsort(rng.random((N_SIM, 10)), axis=1)
mc_proba("Lh_p10", V["Lh_p10"], ~(perm == np.arange(10)).any(axis=1))

# Point de contrôle « derangement » : n = 4.
fraction("Cd_exact", Fraction(9, 24))
fraction("Cd_d_indep", Fraction(3, 4) ** 4)
V["Cd_d_lim"] = math.exp(-1)
V["Cd_d_boole"] = float(1 - 4 * Fraction(1, 4))
perm4 = np.argsort(rng.random((N_SIM, 4)), axis=1)
mc_proba("Cd", Fraction(3, 8), ~(perm4 == np.arange(4)).any(axis=1))

# =====================================================================================
# QUESTIONS
# =====================================================================================
# M2-01 : Bayes. A priori 5 % d'effets réels ; puissance 0,8 ; niveau 0,05.
pr1, pw, al = Fraction(5, 100), Fraction(8, 10), Fraction(5, 100)
fraction("Q01_joint", pr1 * pw)
fraction("Q01_faux", (1 - pr1) * al)
fraction("Q01_total", pr1 * pw + (1 - pr1) * al)
q01 = pr1 * pw / (pr1 * pw + (1 - pr1) * al)
assert q01 == Fraction(16, 35)
fraction("Q01_exact", q01)
V["Q01_d_inverse"] = float(pw)
V["Q01_d_joint"] = float(pr1 * pw)
fraction("Q01_d_sans_prior", pw / (pw + al))
eff = rng.random(N_SIM) < float(pr1)
rej = np.where(eff, rng.random(N_SIM) < float(pw), rng.random(N_SIM) < float(al))
mc_proba("Q01", q01, eff[rej])

# M2-02 : N binomiale (10 ; 0,4), X_i d'espérance 2 et de variance 9.
en, vn, m2, s2 = 4, Fraction(24, 10), 2, 9
assert en == 10 * Fraction(4, 10) and vn == 10 * Fraction(4, 10) * Fraction(6, 10)
V["Q02_en"] = en
V["Q02_vn"] = float(vn)
V["Q02_intra"] = en * s2
V["Q02_inter"] = float(vn * m2**2)
q02 = en * s2 + vn * m2**2
V["Q02_exact"] = float(q02)
V["Q02_es"] = en * m2
V["Q02_d_intra"] = en * s2
V["Q02_d_poisson"] = en * (s2 + m2**2)
V["Q02_d_mu"] = float(en * s2 + vn * m2)
# Contrôle exact indépendant : somme sur la loi de N de E[S^2 | N = n] = n s2 + n^2 m^2.
pmf = {j: Fraction(math.comb(10, j)) * Fraction(4, 10) ** j * Fraction(6, 10) ** (10 - j) for j in range(11)}
es = sum(w * j * m2 for j, w in pmf.items())
es2 = sum(w * (j * s2 + j * j * m2**2) for j, w in pmf.items())
assert es2 - es**2 == q02
n2 = rng.binomial(10, 0.4, N_SIM)
x2 = 2 + 3 * rng.standard_normal((N_SIM, 10))
s2v = np.where(np.arange(10)[None, :] < n2[:, None], x2, 0.0).sum(axis=1)
mc_variance("Q02", float(q02), s2v)

# M2-03 : durées de vie exponentielles d'espérances 1, 2, 4 ans ; probabilité que A soit le dernier en panne.
ra, rb, rc = Fraction(1), Fraction(1, 2), Fraction(1, 4)
tot3 = ra + rb + rc
q03 = rb / tot3 * rc / (ra + rc) + rc / tot3 * rb / (ra + rb)
q03_ie = 1 - ra / (ra + rb) - ra / (ra + rc) + ra / tot3  # inclusion-exclusion sur {A < B} et {A < C}
assert q03 == q03_ie == Fraction(11, 105)
xx = sp.symbols("xx", positive=True)
assert sp.integrate(sp.exp(-xx) * (1 - sp.exp(-xx / 2)) * (1 - sp.exp(-xx / 4)), (xx, 0, sp.oo)) == sp.Rational(11, 105)
fraction("Q03_exact", q03)
fraction("Q03_tot", tot3)
fraction("Q03_pb", rb / tot3)
fraction("Q03_pc", rc / tot3)
fraction("Q03_c_avant_a", rc / (ra + rc))
fraction("Q03_b_avant_a", rb / (ra + rb))
fraction("Q03_t1", rb / tot3 * rc / (ra + rc))
fraction("Q03_t2", rc / tot3 * rb / (ra + rb))
fraction("Q03_d_premier", ra / tot3)
fraction("Q03_d_moy", Fraction(1, 1 + 2 + 4))
fraction("Q03_d_prod", rb / (ra + rb) * rc / (ra + rc))
fraction("Q03_ab", ra / (ra + rb))
fraction("Q03_ac", ra / (ra + rc))
fraction("Q03_a_premier", ra / tot3)
vie = rng.exponential(1.0, size=(N_SIM, 3)) * np.array([1.0, 2.0, 4.0])
mc_proba("Q03", q03, vie.argmax(axis=1) == 0)
# Les événements {A > B} et {A > C} ont pour probabilités 1/3 et 1/5, mais leur intersection vaut 11/105, pas 1/15 :
# ils sont dépendants (positivement corrélés).
mc_proba("Q03_ba", rb / (ra + rb), vie[:, 0] > vie[:, 1])
mc_proba("Q03_ca", rc / (ra + rc), vie[:, 0] > vie[:, 2])
assert q03 > rb / (ra + rb) * rc / (ra + rc)
# Ordre des pannes : P(B en premier) = (1/2)/(7/4), puis course entre A et C relancée (absence de mémoire).
mc_proba("Q03_pb", rb / tot3, vie.argmin(axis=1) == 1)

# M2-04 : achats d'intensité 2 par minute, ventes d'intensité 1 par minute, fenêtre de 90 secondes.
mu4 = sp.Rational(3) * sp.Rational(3, 2)
assert mu4 == sp.Rational(9, 2)
V["Q04_mu"] = float(mu4)
V["Q04_exact"] = float(1 - poisson_cdf(mu4, 2))
V["Q04_somme"] = float(1 + mu4 + mu4**2 / 2)
V["Q04_e"] = float(sp.exp(-mu4))
V["Q04_d_achats"] = float(1 - poisson_cdf(sp.Integer(3), 2))
V["Q04_d_egal"] = float(poisson_pmf(mu4, 3))
V["Q04_d_strict"] = float(1 - poisson_cdf(mu4, 3))
V["Q04_mu_achats"] = 2 * 1.5
arr_a = simuler_poisson(2.0, 1.5, N_SIM)
arr_v = simuler_poisson(1.0, 1.5, N_SIM)
n4 = (arr_a <= 1.5).sum(axis=1) + (arr_v <= 1.5).sum(axis=1)
mc_proba("Q04", V["Q04_exact"], n4 >= 3)

# M2-05 : 10 types équiprobables, nombre moyen de tirages pour voir 5 types distincts.
q05 = sum(Fraction(10, 10 - k + 1) for k in range(1, 6))
fraction("Q05_exact", q05)
fraction("Q05_d_inverse", sum(Fraction(10, k) for k in range(1, 6)))
V["Q05_d_min"] = 5
fraction("Q05_d_moitie", sum(Fraction(10, k) for k in range(1, 11)) / 2)
fraction("Q05_g5", Fraction(10, 6))
mc_moyenne("Q05", q05, attente_collection(10, 5, N_SIM // 2, 80))

# M2-06 : 8 lancers d'un dé, probabilité de voir les six faces.
sur = surjections(8, 6)
assert sur == math.factorial(6) * int(sp.functions.combinatorial.numbers.stirling(8, 6))
q06 = Fraction(sur, 6**8)
fraction("Q06_exact", q06)
V["Q06_fav"] = sur
V["Q06_tot"] = 6**8
V["Q06_d_six"] = float(Fraction(math.factorial(6), 6**6))
V["Q06_d_tronque"] = float(1 - 6 * Fraction(5, 6) ** 8 + 15 * Fraction(4, 6) ** 8)
V["Q06_d_surcompte"] = float(Fraction(math.comb(8, 6) * math.factorial(6) * 36, 6**8))
V["Q06_s1"] = float(6 * Fraction(5, 6) ** 8)
V["Q06_s2"] = float(15 * Fraction(4, 6) ** 8)
V["Q06_s3"] = float(20 * Fraction(3, 6) ** 8)
assert V["Q06_s1"] > 1 and V["Q06_s3"] > V["Q06_exact"] / 2  # le terme d'ordre 3 n'est pas négligeable devant le résultat
# Énumération exacte des 6^8 suites.
grille = ((np.arange(6**8, dtype=np.int32)[:, None] // (6 ** np.arange(8, dtype=np.int32))[None, :]) % 6).astype(np.int8)
assert len(np.unique(grille.astype(np.int64) @ (6 ** np.arange(8, dtype=np.int64)))) == 6**8  # toutes les suites, une fois
presentes = np.zeros(len(grille), dtype=np.int64)
for c in range(6):
    presentes += (grille == c).any(axis=1)
assert int((presentes == 6).sum()) == sur
# Multiplicité du surcompte : motifs (3,1,1,1,1,1) comptés 3 fois, (2,2,1,1,1,1) comptés 4 fois.
n_311 = 6 * math.factorial(8) // math.factorial(3)
n_221 = math.comb(6, 2) * math.factorial(8) // (2 * 2)
assert n_311 + n_221 == sur and 3 * n_311 + 4 * n_221 == math.comb(8, 6) * math.factorial(6) * 36
del grille
d6 = rng.integers(0, 6, size=(N_SIM, 8))
mc_proba("Q06", q06, np.stack([(d6 == c).any(axis=1) for c in range(6)], axis=1).all(axis=1))

# M2-07 : 6 lettres, exactement 2 dans la bonne enveloppe.
q07 = Fraction(math.comb(6, 2) * derangements(4), math.factorial(6))
perms6 = list(itertools.permutations(range(6)))
assert Fraction(sum(sum(s[i] == i for i in range(6)) == 2 for s in perms6), len(perms6)) == q07
V["Q07_exact"] = float(q07)
fraction("Q07_exact_f", q07)
V["Q07_c62"] = math.comb(6, 2)
V["Q07_d4"] = derangements(4)
V["Q07_fav"] = math.comb(6, 2) * derangements(4)
V["Q07_tot"] = math.factorial(6)
V["Q07_d_binom"] = float(math.comb(6, 2) * Fraction(1, 6) ** 2 * Fraction(5, 6) ** 4)
V["Q07_d_poisson"] = math.exp(-1) / 2
V["Q07_d_paires"] = float(Fraction(math.comb(6, 2) * math.factorial(4), math.factorial(6)))
p7 = np.argsort(rng.random((N_SIM, 6)), axis=1)
mc_proba("Q07", q07, (p7 == np.arange(6)).sum(axis=1) == 2)
# Le distracteur « paires » est l'espérance du nombre de paires de lettres bien placées.
nf7 = (p7 == np.arange(6)).sum(axis=1)
mc_moyenne("Q07_d_paires", Fraction(1, 2), nf7 * (nf7 - 1) / 2)

# M2-08 : dé lancé jusqu'au premier 6 inclus, S = somme des faces ; variance de S.
# Méthode 1 : variance totale avec N géométrique de paramètre 1/6.
en8, vn8 = Fraction(6), Fraction(30)
var_u5 = Fraction(sum(x * x for x in range(1, 6)), 5) - Fraction(sum(range(1, 6)), 5) ** 2
assert var_u5 == 2
es8 = 3 * (en8 - 1) + 6
q08 = 2 * (en8 - 1) + 9 * vn8
# Méthode 2 : équations du premier lancer pour E[S] et E[S^2].
e1s, e2s = sp.symbols("e1s e2s")
sol = sp.solve([sp.Eq(e1s, sp.Rational(1, 6) * 6 + sp.Rational(1, 6) * sum(x + e1s for x in range(1, 6))),
                sp.Eq(e2s, sp.Rational(1, 6) * 36 + sp.Rational(1, 6) * sum(x * x + 2 * x * e1s + e2s for x in range(1, 6)))], [e1s, e2s])
assert sol[e1s] == es8 == 21 and sol[e2s] - sol[e1s] ** 2 == q08 == 280
V["Q08_exact"] = int(q08)
V["Q08_ex"] = 2 + 5 + 6  # exemple de l'énoncé : lancers 2, 5, 6
V["Q08_es"] = int(es8)
V["Q08_e2"] = int(sol[e2s])
V["Q08_var_u5"] = int(var_u5)
V["Q08_intra"] = int(2 * (en8 - 1))
V["Q08_inter"] = int(9 * vn8)
V["Q08_d_inter"] = int(9 * vn8)
V["Q08_d_indep"] = float(en8 * Fraction(35, 12) + vn8 * Fraction(7, 2) ** 2)
V["Q08_d_carre"] = int(2 * (en8 - 1) + 3 * vn8)
var_d6 = Fraction(sum(x * x for x in range(1, 7)), 6) - Fraction(7, 2) ** 2
assert var_d6 == Fraction(35, 12)
fraction("Q08_var_d6", var_d6)
V["Q08_moy_d6"] = float(Fraction(sum(range(1, 7)), 6))
V["Q08_moy_u5"] = float(Fraction(sum(range(1, 6)), 5))
V["Q08_e2_u5"] = float(Fraction(sum(x * x for x in range(1, 6)), 5))
g8 = rng.geometric(1 / 6, N_SIM)
assert g8.max() < 120
faces = rng.integers(1, 6, size=(N_SIM, 120))  # faces 1 à 5 pour les lancers qui précèdent le premier 6
s8 = np.where(np.arange(120)[None, :] < (g8 - 1)[:, None], faces, 0).sum(axis=1) + 6
mc_moyenne("Q08_es", 21, s8)
mc_variance("Q08", 280, s8)
# Simulation directe, sans la loi conditionnelle : lancers d'un dé jusqu'au premier 6.
de = rng.integers(1, 7, size=(N_SIM, 150))
premier6 = (de == 6).argmax(axis=1)
assert (de == 6).any(axis=1).all()
s8b = np.where(np.arange(150)[None, :] <= premier6[:, None], de, 0).sum(axis=1)
mc_variance("Q08_direct", 280, s8b)

Path(__file__).with_name("seance_math_2.out.json").write_text(json.dumps(V, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
dire(f"{len(V)} valeurs exportées dans verif/seance_math_2.out.json.")
dire("Toutes les vérifications passent.")
Path(__file__).with_name("seance_math_2.out.txt").write_text("\n".join(LIGNES) + "\n", encoding="utf-8", newline="\n")
