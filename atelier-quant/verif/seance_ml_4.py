"""Vérification de tous les nombres de la séance de ML 4 (leçon L-ML-4 et questions ML4-01 à ML4-08).

Principe :
  1. chaque valeur citée est calculée ici, en fractions exactes quand c'est possible ;
  2. chaque formule démontrée dans la leçon est contrôlée sur des cas numériques (itérations exactes,
     rayon spectral, recherche exhaustive, dérivées et intégrales calculées par sympy) ;
  3. chaque probabilité, espérance ou variance est aussi estimée par simulation Monte-Carlo, dont l'écart
     à l'exact doit rester inférieur à 4 erreurs standard.
Les expériences de la leçon elle-même sont des blocs exécutés à la construction (numpy 1.21.6) ; ce script
n'utilise que numpy, scipy, sympy et fractions. PyTorch n'est pas requis : s'il est installé, deux faits de
bibliothèque cités dans la leçon (remise à l'échelle de nn.Dropout, nombre de paramètres de nn.LSTM) sont
contrôlés en plus, sans rien exporter, pour que la sortie JSON ne dépende pas de sa présence.

Graine fixe : la sortie est identique à chaque exécution.
Sorties : verif/seance_ml_4.out.json (lu par outils/construire.py) et verif/seance_ml_4.out.txt (ce résumé).
"""
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar

rng = np.random.default_rng(20261005)
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


def scientifique(nom, x, chiffres=2):
    """Exporte x = mantisse x 10^exposant : nom_m (mantisse arrondie) et nom_e (exposant entier)."""
    e = math.floor(math.log10(abs(x)))
    m = round(x / 10**e, chiffres - 1)
    if abs(m) >= 10:
        m, e = m / 10, e + 1
    V[nom + "_m"] = m
    V[nom + "_e"] = e


def controle_mc(nom, exact, estimation, erreur_std, seuil=4.0):
    z = (float(estimation) - float(exact)) / float(erreur_std)
    assert abs(z) < seuil, f"{nom} : exact={float(exact)}, MC={estimation}, z={z}"
    V[nom + "_mc"] = float(estimation)
    V[nom + "_se"] = float(erreur_std)
    dire(f"{nom:24s} exact={float(exact):.6f}  MC={float(estimation):.6f}  se={float(erreur_std):.6f}  z={z:+.2f}")


def mc_moyenne(nom, exact, echantillon):
    echantillon = np.asarray(echantillon, dtype=float)
    controle_mc(nom, exact, echantillon.mean(), echantillon.std(ddof=1) / np.sqrt(len(echantillon)))


def iterations(q, cible=Fraction(1, 10**6)):
    """Plus petit entier t tel que q^t <= cible, pour 0 < q < 1."""
    t = math.ceil(math.log(float(cible)) / math.log(float(q)))
    assert q**t <= cible < q ** (t - 1) if isinstance(q, Fraction) else q**t <= float(cible) < q ** (t - 1)
    return t


# =====================================================================================
# PARTIE 1 : descente de gradient sur une quadratique
# =====================================================================================
def facteur(eta, vps):
    """Facteur de contraction garanti q(eta) = max_k |1 - eta lambda_k| (fractions exactes)."""
    return max(abs(1 - eta * l) for l in vps)


# (O2) et (O3) contrôlés sur des spectres tirés au hasard : le pas 2/(lmin + L) minimise q(eta) sur une grille,
# et l'erreur suit exactement (1 - eta lambda_k)^t dans chaque direction propre.
for _ in range(200):
    vps = sorted(Fraction(int(x), 4) for x in rng.integers(1, 80, int(rng.integers(2, 6))))
    lmin, lmax = vps[0], vps[-1]
    if lmin == lmax:
        continue
    eta_opt = Fraction(2) / (lmin + lmax)
    q_opt = facteur(eta_opt, vps)
    assert q_opt == (lmax - lmin) / (lmax + lmin)
    for k in range(1, 400):
        eta = Fraction(k, 400) * 2 / lmax * Fraction(3, 2)
        assert facteur(eta, vps) >= q_opt
        assert (facteur(eta, vps) < 1) == (eta < 2 / lmax)
dire("Descente de gradient : pas optimal 2/(lmin + L), facteur (L - lmin)/(L + lmin), convergence ssi eta < 2/L, 200 spectres.")

# Exemple de la leçon : valeurs propres 1 et 10.
O_LMIN, O_L = Fraction(1), Fraction(10)
V["O_lmin"], V["O_L"] = 1, 10
fraction("O_pas_max", 2 / O_L)
fraction("O_eta_opt", 2 / (O_LMIN + O_L))
fraction("O_q_opt", facteur(2 / (O_LMIN + O_L), [O_LMIN, O_L]))
fraction("O_q_invL", facteur(1 / O_L, [O_LMIN, O_L]))
fraction("O_q_021", facteur(Fraction(21, 100), [O_LMIN, O_L]))
assert V["O_q_021"] > 1
# Itérations exactes sur A = diag(1, 10), départ (1, 1), minimum en 0 : l'erreur vaut ((1 - eta)^t, (1 - 10 eta)^t).
eta = 2 / (O_LMIN + O_L)
e = [Fraction(1), Fraction(1)]
for t in range(1, 31):
    e = [(1 - eta * O_LMIN) * e[0], (1 - eta * O_L) * e[1]]
    norme2 = e[0] ** 2 + e[1] ** 2
    assert norme2 <= V["O_q_opt"] ** (2 * t) * 2 + 1e-30
dire("Exemple diag(1, 10) : erreur après t pas majorée par q^t fois l'erreur initiale, 30 pas exacts.")
# Nombre d'itérations pour diviser l'erreur par 10^6.
V["O_it_opt10"] = iterations(Fraction(9, 11))
# Conditionnement 100 : descente de gradient au pas optimal contre momentum.
K100 = 100
fraction("O_q100", Fraction(K100 - 1, K100 + 1))
V["O_it_gd100"] = iterations(Fraction(K100 - 1, K100 + 1))
fraction("O_mom100", Fraction(int(math.isqrt(K100)) - 1, int(math.isqrt(K100)) + 1))
V["O_it_mom100"] = iterations(Fraction(9, 11))
V["O_mom10"] = (math.sqrt(10) - 1) / (math.sqrt(10) + 1)
dire(f"Conditionnement 100 : {V['O_it_gd100']} pas (gradient) contre {V['O_it_mom100']} pas (momentum) pour 10^-6.")


# Momentum (boule pesante) : pour beta et eta optimaux, toutes les racines ont le module sqrt(beta).
def rayon_momentum(lmin, lmax, beta, eta, n=2001):
    r = 0.0
    for lam in np.linspace(lmin, lmax, n):
        m = np.array([[1 + beta - eta * lam, -beta], [1.0, 0.0]])
        r = max(r, max(abs(np.linalg.eigvals(m))))
    return r


for lmin, lmax in [(1.0, 10.0), (1.0, 100.0), (0.3, 7.0)]:
    kap = lmax / lmin
    beta = ((math.sqrt(kap) - 1) / (math.sqrt(kap) + 1)) ** 2
    eta = 4 / (math.sqrt(lmax) + math.sqrt(lmin)) ** 2
    # (1 - sqrt(beta))^2 = eta lmin et (1 + sqrt(beta))^2 = eta lmax
    assert abs((1 - math.sqrt(beta)) ** 2 - eta * lmin) < 1e-12 and abs((1 + math.sqrt(beta)) ** 2 - eta * lmax) < 1e-12
    assert abs(rayon_momentum(lmin, lmax, beta, eta) - math.sqrt(beta)) < 1e-6
dire("Momentum : rayon spectral = sqrt(beta) = (sqrt(kappa) - 1)/(sqrt(kappa) + 1) sur tout le spectre, 3 cas.")

# Gradient stochastique sur f(theta) = a theta^2 / 2 : Theta_t = (1 - eta a) Theta_{t-1} - eta Z_t.
def var_stationnaire(a, s2, eta):
    return eta**2 * s2 / (1 - (1 - eta * a) ** 2)


a_, s2_, eta_ = Fraction(1), Fraction(1), Fraction(1, 10)
fraction("O_sgd_var", var_stationnaire(a_, s2_, eta_))
assert var_stationnaire(a_, s2_, eta_) == eta_ * s2_ / (a_ * (2 - eta_ * a_))
fraction("O_sgd_approx", eta_ * s2_ / (2 * a_))
fraction("O_sgd_lot10", var_stationnaire(a_, s2_ / 10, eta_))
# Simulation : 40000 chaînes partant de 0, 300 pas ; variance exacte au pas t : v_inf (1 - (1 - eta a)^(2t)).
T_SGD = 300
theta = np.zeros(40000)
for _ in range(T_SGD):
    theta = (1 - 0.1) * theta - 0.1 * rng.normal(size=theta.size)
exact_t = var_stationnaire(a_, s2_, eta_) * (1 - (1 - eta_ * a_) ** (2 * T_SGD))
mc_moyenne("O_sgd_var", exact_t, theta**2)

# Calendrier géométrique : somme des pas finie, donc déplacement total borné.
fraction("O_geo_somme", Fraction(1, 10) / (1 - Fraction(9, 10)))
assert abs(sum(0.1 * 0.9**t for t in range(2000)) - V["O_geo_somme"]) < 1e-12

# =====================================================================================
# PARTIE 2 : réseaux
# =====================================================================================
x = sp.symbols("x", real=True)
dens = sp.exp(-x**2 / 2) / sp.sqrt(2 * sp.pi)
moment_relu = sp.integrate(x**2 * dens, (x, 0, sp.oo))
moyenne_relu = sp.integrate(x * dens, (x, 0, sp.oo))
var_relu = sp.simplify(moment_relu - moyenne_relu**2)
assert sp.simplify(moment_relu - sp.Rational(1, 2)) == 0
assert sp.simplify(var_relu - (sp.Rational(1, 2) - 1 / (2 * sp.pi))) == 0
V["R_var_relu"] = float(var_relu)
# Moment d'ordre 2 de ReLU(A) = E[A^2]/2 pour toute loi symétrique : gaussienne et Laplace (variance 2 pour b = 1).
g = rng.normal(size=400000)
mc_moyenne("R_relu_gauss", 0.5, np.maximum(g, 0) ** 2)
lap = rng.laplace(size=400000)
mc_moyenne("R_relu_laplace", 1.0, np.maximum(lap, 0) ** 2)
mc_moyenne("R_var_relu", float(var_relu), (np.maximum(g, 0) - np.maximum(g, 0).mean()) ** 2)
# Facteurs de variance après 50 couches ReLU.
scientifique("R_lecun50", 0.5**50)
scientifique("R_trois50", 1.5**50)
V["R_profondeur"] = 50
# Sigmoïde : dérivée maximale 1/4 en 0.
z = sp.symbols("z", real=True)
sig = 1 / (1 + sp.exp(-z))
dsig = sp.diff(sig, z)
assert sp.simplify(dsig - sig * (1 - sig)) == 0
assert sp.solve(sp.diff(dsig, z), z) == [0] and dsig.subs(z, 0) == sp.Rational(1, 4)
fraction("R_sig_max", dsig.subs(z, 0))
fraction("R_sig_borne10", Fraction(2, 4) ** 10)
# Connexions résiduelles : 50 blocs de jacobienne de norme au plus 0,01.
V["R_res_borne"] = 0.99**50
assert 0.99**50 >= 1 - 50 * 0.01  # Bernoulli
V["R_res_bernoulli"] = 1 - 50 * 0.01
scientifique("R_plain", 0.01**50)
# Contrôle : plus petite valeur singulière d'un produit de 50 matrices I + J, ||J|| = 0,01.
prod = np.eye(20)
for _ in range(50):
    j = rng.normal(size=(20, 20))
    j *= 0.01 / np.linalg.norm(j, 2)
    prod = (np.eye(20) + j) @ prod
assert np.linalg.svd(prod, compute_uv=False).min() >= 0.99**50 - 1e-12
# Dropout : p = 0,5, h = 3 (leçon) ; sortie D h / (1 - p).
for nom, p_, h_ in [("R_do", Fraction(1, 2), 3)]:
    fraction(nom + "_var", h_**2 * p_ / (1 - p_))
    fraction(nom + "_garde", 1 / (1 - p_))
    d = rng.random(400000) >= float(p_)
    sortie = d * h_ / (1 - float(p_))
    mc_moyenne(nom + "_esp", h_, sortie)
    # Pour p = 1/2, la sortie vaut 0 ou 2h : l'écart au carré à h vaut h^2 sur chaque tirage, sans aléa.
    assert np.all((sortie - h_) ** 2 == h_**2) and h_**2 * p_ / (1 - p_) == h_**2

# =====================================================================================
# PARTIE 3 : architectures
# =====================================================================================
def influence(noyaux, dilatations):
    """Ensemble des retards s >= 0 tels que x_{t-s} influence la sortie de la pile causale (calcul direct)."""
    ens = {0}
    for k, d in zip(noyaux, dilatations):
        ens = {s + d * i for s in ens for i in range(k)}
    return ens


def champ(noyaux, dilatations):
    return 1 + sum(d * (k - 1) for k, d in zip(noyaux, dilatations))


# Contrôle de la formule et de la condition sans trou sur des piles tirées au hasard.
for _ in range(500):
    m = int(rng.integers(1, 6))
    ks = [int(v) for v in rng.integers(2, 5, m)]
    ds = [int(v) for v in rng.integers(1, 9, m)]
    ens = influence(ks, ds)
    assert max(ens) + 1 == champ(ks, ds)
    sans_trou = all(ds[l] <= champ(ks[:l], ds[:l]) for l in range(m))
    if sans_trou:
        assert len(ens) == champ(ks, ds)
dire("Champ réceptif 1 + somme d (k - 1), et absence de trou si d_l <= R_(l-1), vérifiés sur 500 piles.")
# Vérification indépendante par convolution d'une impulsion (numpy).
def impulsion(ks, ds, longueur=200):
    sig_ = np.zeros(longueur)
    sig_[0] = 1.0
    for k, d in zip(ks, ds):
        noyau = np.zeros(d * (k - 1) + 1)
        noyau[::d] = 1.0
        sig_ = np.convolve(sig_, noyau)[:longueur]
    return int((sig_ > 0).sum())


V["A_champ_ex"] = champ([3] * 4, [1, 2, 4, 8])
assert impulsion([3] * 4, [1, 2, 4, 8]) == V["A_champ_ex"] == len(influence([3] * 4, [1, 2, 4, 8]))
V["A_wavenet"] = champ([2] * 10, [2**l for l in range(10)])
assert len(influence([2] * 10, [2**l for l in range(10)])) == V["A_wavenet"]
V["A_trou_span"] = champ([2, 2], [1, 4])
V["A_trou_nb"] = len(influence([2, 2], [1, 4]))
V["A_trou_ens"] = sorted(influence([2, 2], [1, 4]))
assert V["A_trou_nb"] < V["A_trou_span"]
# LSTM : facteur le long de la cellule, portes d'oubli à 0,99 sur 100 pas.
V["A_oubli100"] = 0.99**100
scientifique("A_rnn100", 0.5**100)
# Paramètres d'un LSTM, d = 10, h = 20, un biais par porte.
def params_lstm(d, h, biais=1):
    return 4 * (h * (h + d) + biais * h)


V["A_lstm_ex"] = params_lstm(10, 20)
# Attention : Var(q . k) = d pour des coordonnées indépendantes, centrées, de variance 1.
V["A_d"] = 64
V["A_std_brut"] = math.sqrt(64)
q_ = rng.normal(size=(100000, 64))
k_ = rng.normal(size=(100000, 64))
mc_moyenne("A_var_score", 64, (q_ * k_).sum(axis=1) ** 2)
V["A_cout_x"] = 4  # doubler T multiplie T^2 par 4
assert (2 * 7) ** 2 == V["A_cout_x"] * 7**2
# Jacobienne du softmax : dp_i/ds_j = p_i (delta_ij - p_j), contrôlée par différences finies.
for _ in range(50):
    s = rng.normal(size=5) * 3
    p = np.exp(s - s.max()) / np.exp(s - s.max()).sum()
    jac = np.diag(p) - np.outer(p, p)
    num = np.zeros((5, 5))
    for jj in range(5):
        dlt = np.zeros(5)
        dlt[jj] = 1e-6
        sp_ = s + dlt
        sm_ = s - dlt
        pp = np.exp(sp_ - sp_.max()) / np.exp(sp_ - sp_.max()).sum()
        pm = np.exp(sm_ - sm_.max()) / np.exp(sm_ - sm_.max()).sum()
        num[:, jj] = (pp - pm) / 2e-6
    assert np.allclose(jac, num, atol=1e-7)
dire("Jacobienne du softmax p_i (delta_ij - p_j) vérifiée par différences finies, 50 cas.")

# =====================================================================================
# PARTIE 4 : noyaux
# =====================================================================================
# Noyau polynomial (x . x' + 1)^2 en dimension 2 = produit scalaire de 6 caractéristiques.
def psi(v):
    r2 = math.sqrt(2)
    return np.array([1.0, r2 * v[0], r2 * v[1], v[0] ** 2, v[1] ** 2, r2 * v[0] * v[1]])


for _ in range(200):
    u, w = rng.normal(size=2), rng.normal(size=2)
    assert abs((u @ w + 1) ** 2 - psi(u) @ psi(w)) < 1e-10
V["N_dim_poly"] = 6
dire("Noyau polynomial de degré 2 en dimension 2 : 6 caractéristiques explicites, 200 cas.")
# Ridge à noyau = ridge sur les caractéristiques explicites (identité de poussée).
for _ in range(50):
    n_, lam_ = int(rng.integers(3, 12)), float(rng.uniform(0.05, 3))
    X_ = rng.normal(size=(n_, 2))
    y_ = rng.normal(size=n_)
    Psi = np.array([psi(v) for v in X_])
    b = np.linalg.solve(Psi.T @ Psi + lam_ * np.eye(6), Psi.T @ y_)
    K_ = Psi @ Psi.T
    alpha = np.linalg.solve(K_ + lam_ * np.eye(n_), y_)
    xt = rng.normal(size=2)
    assert abs(psi(xt) @ b - np.array([(v @ xt + 1) ** 2 for v in X_]) @ alpha) < 1e-8
    assert np.allclose(np.linalg.solve(Psi.T @ Psi + lam_ * np.eye(6), Psi.T), Psi.T @ np.linalg.inv(K_ + lam_ * np.eye(n_)))
dire("Ridge à noyau : prédictions identiques à la ridge sur caractéristiques explicites, 50 cas.")


def rbf(u, w):
    return math.exp(-((u - w) ** 2) / 2)


def gp(xs, ys, s2, xt):
    K_ = np.array([[rbf(u, w) for w in xs] for u in xs])
    s_ = np.array([rbf(u, xt) for u in xs])
    m = s_ @ np.linalg.solve(K_ + s2 * np.eye(len(xs)), np.array(ys, dtype=float))
    v = rbf(xt, xt) - s_ @ np.linalg.solve(K_ + s2 * np.eye(len(xs)), s_)
    return m, v, K_, s_


def mc_gp(nom, xs, s2, xt, m_exact_fn, v_exact, n=200000):
    """Simulation du modèle : F ~ GP(0, rbf), Y_i = F(x_i) + bruit. Le résidu F(x*) - s^T (K + s2 I)^-1 Y
    a pour variance la variance a posteriori et n'est pas corrélé à Y."""
    pts = list(xs) + [xt]
    C = np.array([[rbf(u, w) for w in pts] for u in pts])
    Lc = np.linalg.cholesky(C + 1e-12 * np.eye(len(pts)))
    F = rng.normal(size=(n, len(pts))) @ Lc.T
    Y = F[:, :-1] + math.sqrt(s2) * rng.normal(size=(n, len(xs)))
    _, _, K_, s_ = gp(xs, [0] * len(xs), s2, xt)
    coef = np.linalg.solve(K_ + s2 * np.eye(len(xs)), s_)
    res = F[:, -1] - Y @ coef
    mc_moyenne(nom, v_exact, res**2)
    for i in range(len(xs)):
        prod_ = res * Y[:, i]
        assert abs(prod_.mean()) < 4 * prod_.std(ddof=1) / math.sqrt(n)


# Exemple de la leçon : x = (-1, 0, 1), y = (0, 1, 0), bruit 0,1, point 0,5.
m_l, v_l, _, _ = gp([-1, 0, 1], [0, 1, 0], 0.1, 0.5)
V["N_gp_moy"], V["N_gp_var"] = float(m_l), float(v_l)
mc_gp("N_gp_var", [-1, 0, 1], 0.1, 0.5, None, float(v_l))

# SVM à marge souple, une variable : objectif 1/2 w^2 + C somme max(0, 1 - y (w x + b)).
SVM_X = [-2.0, -1.0, 0.5, 1.0, 2.0]
SVM_Y = [-1, -1, -1, 1, 1]


def charniere_min_b(w):
    """Minimum en b de la somme des pertes charnière : fonction convexe affine par morceaux de b, minimale
    en l'un de ses points de cassure b = y_i - w x_i."""
    cands = [yi - w * xi for xi, yi in zip(SVM_X, SVM_Y)]
    vals = [sum(max(0.0, 1 - yi * (w * xi + b)) for xi, yi in zip(SVM_X, SVM_Y)) for b in cands]
    k = int(np.argmin(vals))
    return vals[k], cands[k]


def svm(C):
    # L'objectif en w (après minimisation en b) est convexe : une recherche ternaire converge vers son minimum.
    obj = lambda w: 0.5 * w * w + C * charniere_min_b(w)[0]
    gauche, droite = 0.0, 20.0
    for _ in range(300):
        m1, m2 = gauche + (droite - gauche) / 3, droite - (droite - gauche) / 3
        if obj(m1) <= obj(m2):
            droite = m2
        else:
            gauche = m1
    w = (gauche + droite) / 2
    # Contrôle : aucune valeur de w sur une grille fine ne fait mieux.
    grille = np.linspace(0, 20, 20001)
    meilleur = min(0.5 * g_ * g_ + C * charniere_min_b(g_)[0] for g_ in grille)
    assert 0.5 * w * w + C * charniere_min_b(w)[0] <= meilleur + 1e-9
    return w, 0.5 * w * w + C * charniere_min_b(w)[0], charniere_min_b(w)[0]


for nom, C in [("N_svm_c01", 0.1), ("N_svm_c1", 1.0), ("N_svm_c100", 100.0)]:
    w, obj, hinge = svm(C)
    V[nom + "_w"] = round(w, 6)
    V[nom + "_marge"] = round(2 / w, 6)
    V[nom + "_charniere"] = round(hinge, 6)
    dire(f"SVM C = {C} : w = {w:.6f}, largeur 2/w = {2 / w:.6f}, somme des charnières = {hinge:.6f}")
assert V["N_svm_c01_marge"] > V["N_svm_c1_marge"] > V["N_svm_c100_marge"]
assert abs(V["N_svm_c100_w"] - 4) < 1e-5 and V["N_svm_c100_charniere"] < 1e-5  # marge dure : w = 4

# =====================================================================================
# PARTIE 5 : EM pour un mélange de deux gaussiennes
# =====================================================================================
def normale(v, mu, s2):
    return math.exp(-((v - mu) ** 2) / (2 * s2)) / math.sqrt(2 * math.pi * s2)


def log_vrais(xs, om, mu, s2):
    return sum(math.log(om[0] * normale(v, mu[0], s2[0]) + om[1] * normale(v, mu[1], s2[1])) for v in xs)


def etape_em(xs, om, mu, s2):
    r = []
    for v in xs:
        p1, p2 = om[0] * normale(v, mu[0], s2[0]), om[1] * normale(v, mu[1], s2[1])
        r.append(p1 / (p1 + p2))
    n1 = sum(r)
    n2 = len(xs) - n1
    mu1 = sum(ri * v for ri, v in zip(r, xs)) / n1
    mu2 = sum((1 - ri) * v for ri, v in zip(r, xs)) / n2
    v1 = sum(ri * (v - mu1) ** 2 for ri, v in zip(r, xs)) / n1
    v2 = sum((1 - ri) * (v - mu2) ** 2 for ri, v in zip(r, xs)) / n2
    return r, (n1 / len(xs), n2 / len(xs)), (mu1, mu2), (v1, v2)


# Monotonie contrôlée sur des données et des départs tirés au hasard.
for _ in range(100):
    xs = list(rng.normal(size=int(rng.integers(4, 30))) * 2 + rng.integers(0, 2, 1) * 3)
    om, mu, s2 = (0.5, 0.5), (float(rng.normal()), float(rng.normal()) + 1), (1.0, 1.0)
    ell = log_vrais(xs, om, mu, s2)
    for _ in range(30):
        _, om, mu, s2 = etape_em(xs, om, mu, s2)
        if min(s2) < 1e-8:
            break
        nouv = log_vrais(xs, om, mu, s2)
        assert nouv >= ell - 1e-9
        ell = nouv
dire("EM : log-vraisemblance croissante à chaque itération, 100 jeux de données.")
# Étape M : contrôle par maximisation numérique de Q en mu1 et s2_1 (dérivée nulle).
xs_l = [-1.0, 0.0, 4.0, 5.0]
r_l, om_l, mu_l, s2_l = etape_em(xs_l, (0.5, 0.5), (0.0, 3.0), (1.0, 1.0))
V["E_r"] = [round(v, 4) for v in r_l]
V["E_om1"], V["E_mu1"], V["E_mu2"], V["E_s21"], V["E_s22"] = om_l[0], mu_l[0], mu_l[1], s2_l[0], s2_l[1]
V["E_ell0"] = log_vrais(xs_l, (0.5, 0.5), (0.0, 3.0), (1.0, 1.0))
V["E_ell1"] = log_vrais(xs_l, om_l, mu_l, s2_l)
assert V["E_ell1"] > V["E_ell0"]
Qmu = lambda m: sum(ri * (-(v - m) ** 2 / (2 * s2_l[0])) for ri, v in zip(r_l, xs_l))
assert abs(minimize_scalar(lambda m: -Qmu(m), bounds=(-5, 5), method="bounded", options={"xatol": 1e-12}).x - mu_l[0]) < 1e-6
# Dégénérescence : mu1 = x_1 et sigma1 -> 0, la log-vraisemblance tend vers l'infini.
V["E_deg"] = [round(log_vrais(xs_l, (0.5, 0.5), (-1.0, 3.0), (s ** 2, 1.0)), 2) for s in (1e-1, 1e-3, 1e-6)]
assert V["E_deg"][0] < V["E_deg"][1] < V["E_deg"][2]

# =====================================================================================
# POINTS DE CONTRÔLE
# =====================================================================================
# Pas : valeurs propres 0,5 et 4.
fraction("C_pas_max", Fraction(2, 4))
fraction("C_pas_lmin", Fraction(2) / Fraction(1, 2))
fraction("C_pas_opt", Fraction(2) / (Fraction(1, 2) + 4))
fraction("C_pas_invL", Fraction(1, 4))
# Attention : d = 256.
V["C_att_d"] = 256
V["C_att_std"] = math.sqrt(256)

# =====================================================================================
# QUESTIONS
# =====================================================================================
# ML4-01 : valeurs propres 2 et 18.
vps1 = [Fraction(2), Fraction(18)]
fraction("Q01_eta", Fraction(2) / (vps1[0] + vps1[1]))
fraction("Q01_q", facteur(Fraction(2) / (vps1[0] + vps1[1]), vps1))
fraction("Q01_d_invL_eta", 1 / vps1[1])
fraction("Q01_d_invL_q", facteur(1 / vps1[1], vps1))
fraction("Q01_d_max_eta", 2 / vps1[1])
fraction("Q01_d_max_qlent", abs(1 - 2 / vps1[1] * vps1[0]))
fraction("Q01_d_max_q", facteur(2 / vps1[1], vps1))
fraction("Q01_d_lmin_eta", 1 / vps1[0])
fraction("Q01_d_lmin_q", facteur(1 / vps1[0], vps1))
assert V["Q01_d_max_q"] == 1 and V["Q01_d_lmin_q"] > 1
assert all(facteur(Fraction(k, 1000), vps1) >= facteur(Fraction(1, 10), vps1) for k in range(1, 1000))
V["Q01_kappa"] = 9

# ML4-02 : a = 2, sigma2 = 0,5, eta = 0,1, Theta_0 = 3.
a2, s22, eta2 = Fraction(2), Fraction(1, 2), Fraction(1, 10)
fraction("Q02_exact", var_stationnaire(a2, s22, eta2))
fraction("Q02_d_approx", eta2 * s22 / (2 * a2))
fraction("Q02_d_sans_carre", eta2**2 * s22 / (1 - (1 - eta2 * a2)))
fraction("Q02_d_un_pas", eta2**2 * s22)
fraction("Q02_facteur", (1 - eta2 * a2) ** 2)
assert len({V["Q02_exact"], V["Q02_d_approx"], V["Q02_d_sans_carre"], V["Q02_d_un_pas"]}) == 4
theta = np.full(40000, 3.0)
for _ in range(200):
    theta = (1 - 0.2) * theta - 0.1 * math.sqrt(0.5) * rng.normal(size=theta.size)
# Au pas 200, l'espérance 3 x 0,8^200 est négligeable devant l'erreur standard ; variance exacte au pas 200 :
var200 = var_stationnaire(a2, s22, eta2) * (1 - (1 - eta2 * a2) ** 400)
assert 3 * 0.8**200 < 1e-15
mc_moyenne("Q02", var200, theta**2)

# ML4-03 : 10 couches ReLU, variance des poids 1/n, pré-activations initiales gaussiennes de variance 1.
fraction("Q03_exact", Fraction(1, 2) ** 10)
fraction("Q03_d_onze", Fraction(1, 2) ** 11)
V["Q03_d_var_relu"] = float(var_relu) ** 10
V["Q03_inv"] = 2**10
# Simulation : réseaux de largeur 10, poids gaussiens indépendants, un réseau neuf par tirage.
N3, n3 = 40000, 10
a = rng.normal(size=(N3, n3))
for _ in range(10):
    W = rng.normal(size=(N3, n3, n3)) / math.sqrt(n3)
    a = np.einsum("bij,bj->bi", W, np.maximum(a, 0))
mc_moyenne("Q03", V["Q03_exact"], (a**2).mean(axis=1))

# ML4-04 : dropout p = 0,2, h = 2.
p4, h4 = Fraction(1, 5), 2
fraction("Q04_exact", h4**2 * p4 / (1 - p4))
fraction("Q04_d_sans_echelle", h4**2 * p4 * (1 - p4))
fraction("Q04_d_moment2", h4**2 / (1 - p4))
fraction("Q04_d_inverse", h4**2 * (1 - p4) / p4)
fraction("Q04_echelle", 1 / (1 - p4))
fraction("Q04_haut", h4 / (1 - p4))
d4 = rng.random(400000) >= 0.2
sortie4 = d4 * 2 / 0.8
mc_moyenne("Q04_esp", 2, sortie4)
mc_moyenne("Q04", V["Q04_exact"], (sortie4 - 2) ** 2)

# ML4-05 : noyau 3 avec dilatations 1, 2, 4, 8, puis noyau 2 avec dilatation 16.
ks5, ds5 = [3, 3, 3, 3, 2], [1, 2, 4, 8, 16]
V["Q05_exact"] = len(influence(ks5, ds5))
assert V["Q05_exact"] == champ(ks5, ds5) == impulsion(ks5, ds5)
V["Q05_avant"] = champ(ks5[:4], ds5[:4])
assert ds5[4] <= V["Q05_avant"]
V["Q05_d_sans_un"] = champ(ks5, ds5) - 1
V["Q05_d_k"] = sum(d * k for k, d in zip(ks5, ds5))
V["Q05_d_sans_dil"] = champ(ks5, [1] * 5)

# ML4-06 : LSTM, d = 32, h = 64.
V["Q06_exact"] = params_lstm(32, 64)
V["Q06_d_sans_biais"] = params_lstm(32, 64, biais=0)
V["Q06_d_rnn"] = params_lstm(32, 64) // 4
V["Q06_d_trois"] = 3 * (64 * (64 + 32) + 64)
V["Q06_deux_biais"] = params_lstm(32, 64, biais=2)
V["Q06_bloc"] = 64 * (64 + 32) + 64

# ML4-07 : processus gaussien, x = (0, 1), y = (1, 0), bruit 0,5, point 0.
m7, v7, K7, s7 = gp([0, 1], [1, 0], 0.5, 0)
V["Q07_exact"] = float(m7)
V["Q07_var"] = float(v7)
V["Q07_a"] = math.exp(-0.5)
V["Q07_a2"] = math.exp(-1)
V["Q07_num"] = 1.5 - math.exp(-1)
V["Q07_den"] = 2.25 - math.exp(-1)
assert abs(V["Q07_num"] / V["Q07_den"] - V["Q07_exact"]) < 1e-12
V["Q07_d_interp"] = float(gp([0, 1], [1, 0], 1e-14, 0)[0])
V["Q07_d_sigma"] = float(gp([0, 1], [1, 0], math.sqrt(0.5), 0)[0])
V["Q07_d_seul"] = 1 / 1.5
assert abs(V["Q07_d_interp"] - 1) < 1e-9
V["Q07_d_interp"] = 1
mc_gp("Q07_var", [0, 1], 0.5, 0, None, float(v7))
# Moyenne a posteriori par simulation : régression de F(0) sur Y, puis prédiction en y = (1, 0).
pts7 = [0, 1, 0]
C7 = np.array([[rbf(u, w) for w in pts7[:2]] for u in pts7[:2]])
L7 = np.linalg.cholesky(C7)
F7 = rng.normal(size=(200000, 2)) @ L7.T
Y7 = F7 + math.sqrt(0.5) * rng.normal(size=(200000, 2))
coef7, *_ = np.linalg.lstsq(Y7, F7[:, 0], rcond=None)
assert abs(coef7 @ np.array([1.0, 0.0]) - V["Q07_exact"]) < 0.01

# ML4-08 : EM, x = (0, 1, 5), départ omega = (1/2, 1/2), mu = (0, 4), variances 1.
xs8 = [0.0, 1.0, 5.0]
r8, om8, mu8, s28 = etape_em(xs8, (0.5, 0.5), (0.0, 4.0), (1.0, 1.0))
V["Q08_exact"] = mu8[0]
V["Q08_r"] = [round(v, 4) for v in r8]
V["Q08_r1"] = r8[1]
V["Q08_r3"] = r8[2]
V["Q08_n1"] = sum(r8)
V["Q08_d_dur"] = 0.5
V["Q08_d_sur_n"] = sum(ri * v for ri, v in zip(r8, xs8)) / 3
V["Q08_d_moyenne"] = 2
V["Q08_mu2"] = mu8[1]
assert len({round(V[k], 3) for k in ["Q08_exact", "Q08_d_dur", "Q08_d_sur_n", "Q08_d_moyenne"]}) == 4
# Responsabilités : contrôle par la forme logistique, r = 1 / (1 + exp(-(mu1 - mu2) x + (mu1^2 - mu2^2)/2)) à variances égales.
for v, ri in zip(xs8, r8):
    assert abs(ri - 1 / (1 + math.exp(((v - 0) ** 2 - (v - 4) ** 2) / 2))) < 1e-12
V["Q08_ell0"] = log_vrais(xs8, (0.5, 0.5), (0.0, 4.0), (1.0, 1.0))
V["Q08_ell1"] = log_vrais(xs8, om8, mu8, s28)
assert V["Q08_ell1"] >= V["Q08_ell0"]

# =====================================================================================
# Faits de bibliothèque, contrôlés seulement si PyTorch est installé (rien n'est exporté).
# =====================================================================================
try:
    import torch

    torch.manual_seed(0)
    couche = torch.nn.Dropout(p=0.2)
    couche.train()
    sortie_t = couche(torch.ones(100000))
    valeurs = set(torch.unique(sortie_t).tolist())
    assert valeurs <= {0.0, 1.25} and abs(float((sortie_t == 0).float().mean()) - 0.2) < 0.01
    couche.eval()
    assert torch.equal(couche(torch.ones(10)), torch.ones(10))
    lstm = torch.nn.LSTM(32, 64)
    assert sum(p.numel() for p in lstm.parameters()) == V["Q06_deux_biais"]
    print("PyTorch présent : nn.Dropout (zéro avec probabilité p, sinon x / (1 - p)) et nn.LSTM (deux biais) vérifiés.")
except ImportError:
    print("PyTorch absent : contrôles de bibliothèque ignorés.")

Path(__file__).with_name("seance_ml_4.out.json").write_text(json.dumps(V, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
dire(f"{len(V)} valeurs exportées dans verif/seance_ml_4.out.json.")
dire("Toutes les vérifications passent.")
Path(__file__).with_name("seance_ml_4.out.txt").write_text("\n".join(LIGNES) + "\n", encoding="utf-8", newline="\n")
