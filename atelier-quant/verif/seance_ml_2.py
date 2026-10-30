"""Vérification de tous les nombres de la séance de ML 2 (leçon L-ARBRES-1 et questions M1-01 à M1-12).

Principe :
  1. chaque valeur citée est calculée ici, en fractions exactes quand c'est possible ;
  2. chaque formule démontrée dans la leçon est contrôlée sur des cas numériques (identités exactes,
     comparaison à une recherche exhaustive, dérivées calculées par sympy) ;
  3. chaque probabilité ou espérance est aussi estimée par simulation Monte-Carlo, dont l'écart à l'exact
     doit rester inférieur à 4 erreurs standard.
Les expériences avec scikit-learn sont dans la leçon elle-même (blocs exécutés à la construction) ; ce script
n'utilise que numpy, sympy et fractions, pour donner la même sortie quelle que soit la version de scikit-learn.

Graine fixe : la sortie est identique à chaque exécution.
Sorties : verif/seance_ml_2.out.json (lu par outils/construire.py) et verif/seance_ml_2.out.txt (ce résumé).
"""
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp

rng = np.random.default_rng(20261006)
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
    z = (float(estimation) - float(exact)) / float(erreur_std)
    assert abs(z) < seuil, f"{nom} : exact={float(exact)}, MC={estimation}, z={z}"
    V[nom + "_mc"] = float(estimation)
    V[nom + "_se"] = float(erreur_std)
    dire(f"{nom:24s} exact={float(exact):.6f}  MC={float(estimation):.6f}  se={float(erreur_std):.6f}  z={z:+.2f}")


def mc_moyenne(nom, exact, echantillon):
    echantillon = np.asarray(echantillon, dtype=float)
    controle_mc(nom, exact, echantillon.mean(), echantillon.std(ddof=1) / np.sqrt(len(echantillon)))


# =====================================================================================
# Outils exacts : impuretés et gains
# =====================================================================================
def gini(etiquettes):
    """Impureté de Gini d'un ensemble d'étiquettes 0/1, exacte : 1 - somme des carrés des proportions."""
    if len(etiquettes) == 0:
        return Fraction(0)
    p = Fraction(sum(etiquettes), len(etiquettes))
    return 1 - p**2 - (1 - p) ** 2


def entropie(etiquettes):
    p = sum(etiquettes) / len(etiquettes)
    return 0.0 if p in (0, 1) else -(p * math.log(p) + (1 - p) * math.log(1 - p))


def gain(impurete, parent, gauche, droite):
    n = len(parent)
    return impurete(parent) - Fraction(len(gauche), n) * impurete(gauche) - Fraction(len(droite), n) * impurete(droite)


def gain_flottant(impurete, parent, gauche, droite):
    n = len(parent)
    return impurete(parent) - len(gauche) / n * impurete(gauche) - len(droite) / n * impurete(droite)


def variance_pop(v):
    v = [Fraction(x) for x in v]
    moy = sum(v) / len(v)
    return sum((x - moy) ** 2 for x in v) / len(v)


# Contrôle des résultats R1 à R3 de la leçon sur des cas tirés au hasard (valeurs entières, calcul exact).
for _ in range(300):
    m = int(rng.integers(2, 12))
    yv = [Fraction(int(x)) for x in rng.integers(-5, 6, m)]
    k = int(rng.integers(1, m))
    L, R = yv[:k], yv[k:]
    moy = sum(yv) / m
    moyL, moyR = sum(L) / len(L), sum(R) / len(R)
    # R1 : décomposition de la somme des carrés autour d'une constante c quelconque.
    c = Fraction(int(rng.integers(-7, 8)), 3)
    assert sum((y - c) ** 2 for y in yv) == sum((y - moy) ** 2 for y in yv) + m * (moy - c) ** 2
    # R2 : gain de variance = somme pondérée des carrés des écarts entre moyennes.
    g = variance_pop(yv) - Fraction(len(L), m) * variance_pop(L) - Fraction(len(R), m) * variance_pop(R)
    assert g == Fraction(len(L), m) * (moyL - moy) ** 2 + Fraction(len(R), m) * (moyR - moy) ** 2 >= 0
    # R3 : Gini = somme des variances des indicatrices ; même forme de gain.
    e = [int(y > 0) for y in yv]
    assert gini(e) == variance_pop(e) + variance_pop([1 - x for x in e])
    assert gain(gini, e, e[:k], e[k:]) >= 0 and gain_flottant(entropie, e, e[:k], e[k:]) >= -1e-12
dire("R1 à R3 (décomposition, gain de variance, Gini comme variance des indicatrices) vérifiés sur 300 cas.")

# R5 : une feuille impure dont une variable a des valeurs distinctes admet une coupure de gain > 0.
for _ in range(300):
    m = int(rng.integers(2, 10))
    yv = [Fraction(int(x)) for x in rng.integers(0, 3, m)]
    if len(set(yv)) == 1:
        continue
    meilleurs = max(variance_pop(yv) - Fraction(k, m) * variance_pop(yv[:k]) - Fraction(m - k, m) * variance_pop(yv[k:]) for k in range(1, m))
    assert meilleurs > 0
dire("R5 : toute feuille impure (valeurs distinctes) admet une coupure de gain strictement positif, 300 cas.")

# =====================================================================================
# LEÇON, PARTIE 1 : l'exemple jouet (x = 1..8)
# =====================================================================================
Y_JOUET = [0, 0, 1, 0, 1, 1, 1, 1]
n_j = len(Y_JOUET)
fraction("A_gini_parent", gini(Y_JOUET))
gains_gini = {k: gain(gini, Y_JOUET, Y_JOUET[:k], Y_JOUET[k:]) for k in range(1, n_j)}
gains_ent = {k: gain_flottant(entropie, Y_JOUET, Y_JOUET[:k], Y_JOUET[k:]) for k in range(1, n_j)}
k_gini = max(gains_gini, key=gains_gini.get)
k_ent = max(gains_ent, key=gains_ent.get)
assert sorted(gains_gini.values())[-1] > sorted(gains_gini.values())[-2]  # meilleure coupure unique
assert sorted(gains_ent.values())[-1] > sorted(gains_ent.values())[-2]
V["A_seuil_gini"] = k_gini + 0.5
V["A_seuil_ent"] = k_ent + 0.5
fraction("A_gain_gini", gains_gini[k_gini])
V["A_gain_ent"] = gains_ent[k_ent]
fraction("A_gini_gauche", gini(Y_JOUET[:k_gini]))
fraction("A_gini_droite", gini(Y_JOUET[k_gini:]))
fraction("A_gini_enfants", Fraction(k_gini, n_j) * gini(Y_JOUET[:k_gini]) + Fraction(n_j - k_gini, n_j) * gini(Y_JOUET[k_gini:]))
V["A_n_gauche"] = k_gini
V["A_pos_gauche"] = sum(Y_JOUET[:k_gini])
fraction("A_prop_gauche", Fraction(sum(Y_JOUET[:k_gini]), k_gini))
dire(f"Exemple jouet : meilleure coupure Gini en {k_gini + 0.5} (gain {gains_gini[k_gini]}), entropie en {k_ent + 0.5}.")

# =====================================================================================
# LEÇON, PARTIE 2 : bootstrap et variance d'une moyenne
# =====================================================================================
for n_ in [10, 100, 1000]:
    exact = (1 - 1 / n_) ** n_
    assert exact <= math.exp(-1)  # (1 - 1/n)^n <= 1/e pour tout n
    V[f"B_abs_{n_}"] = exact
    # Simulation : proportion des observations absentes d'un échantillon bootstrap.
    tirages = rng.integers(0, n_, size=(4000, n_))
    absentes = np.array([n_ - len(np.unique(t)) for t in tirages]) / n_
    mc_moyenne(f"B_abs_{n_}", exact, absentes)
V["B_inv_e"] = math.exp(-1)
V["B_in_lim"] = 1 - math.exp(-1)
V["B_distincts_100"] = 100 * (1 - 0.99**100)
V["B_nb_oob"] = 500 * 0.999**1000
V["B_jamais_oob"] = (1 - 0.999**1000) ** 20
assert V["B_jamais_oob"] < 2e-4

# Variance de la moyenne de B prédicteurs de variance sigma2 et de corrélation rho.
SIGMA2, RHO = Fraction(2), Fraction(1, 2)


def var_moyenne(b, sigma2=SIGMA2, rho=RHO):
    return rho * sigma2 + (1 - rho) * sigma2 / b


# Contrôle de la formule par la matrice de covariance complète : 1' C 1 / B^2.
for b in [1, 2, 7, 10, 100]:
    c = np.full((b, b), float(RHO * SIGMA2))
    np.fill_diagonal(c, float(SIGMA2))
    assert abs(c.sum() / b**2 - float(var_moyenne(b))) < 1e-12
for b in [1, 10, 100]:
    V[f"B_var_{b}"] = float(var_moyenne(b))
V["B_plancher"] = float(RHO * SIGMA2)
# Simulation : T_b = sqrt(rho sigma2) Z0 + sqrt((1 - rho) sigma2) Z_b, Z indépendantes N(0, 1).
for b in [10, 100]:
    z0 = rng.normal(size=(20000, 1))
    zb = rng.normal(size=(20000, b))
    t = math.sqrt(float(RHO * SIGMA2)) * z0 + math.sqrt(float((1 - RHO) * SIGMA2)) * zb
    moy = t.mean(axis=1)
    est = moy.var(ddof=1)
    # Erreur standard de la variance empirique d'une gaussienne : variance * sqrt(2 / (N - 1)).
    controle_mc(f"B_var_{b}", var_moyenne(b), est, float(var_moyenne(b)) * math.sqrt(2 / (len(moy) - 1)))

# Point de contrôle : n = 3, probabilité d'être absente.
fraction("Cb_abs", Fraction(2, 3) ** 3)
fraction("Cb_d_present", 1 - Fraction(2, 3) ** 3)
fraction("Cb_d_cube", Fraction(1, 3) ** 3)
# Point de contrôle : sigma2 = 1, rho = 0,5, nombre minimal d'arbres pour (1 - rho) sigma2 / B <= 0,01.
V["Cp_B_min"] = math.ceil(Fraction(1, 2) / Fraction(1, 100))
assert Fraction(1, 2) / V["Cp_B_min"] <= Fraction(1, 100) < Fraction(1, 2) / (V["Cp_B_min"] - 1)
V["Cp_d_rho"] = math.ceil(Fraction(1, 2) / Fraction(1, 100) * Fraction(1, 2))  # confusion avec rho sigma2 / B
V["Cp_d_inv"] = math.ceil(1 / Fraction(1, 100))  # oubli du facteur (1 - rho)

# =====================================================================================
# LEÇON, PARTIE 4 : boosting
# =====================================================================================
# Projection : pour r quelconque et une partition, r_chapeau = moyennes par feuille,
# ||r - eta r_chapeau||^2 = ||r||^2 - (2 eta - eta^2) ||r_chapeau||^2.
for _ in range(200):
    m = int(rng.integers(3, 15))
    r = rng.normal(size=m)
    feuilles = rng.integers(0, 3, m)
    rc = np.array([r[feuilles == f].mean() for f in feuilles])
    eta = float(rng.uniform(0.01, 1.99))
    assert abs(np.sum((r - eta * rc) ** 2) - (np.sum(r**2) - (2 * eta - eta**2) * np.sum(rc**2))) < 1e-9
dire("Identité de projection du boosting quadratique vérifiée sur 200 cas.")
ETA = 0.1
V["D_eta"] = ETA
V["D_facteur"] = (1 - ETA) ** 2
V["D_m_1pct"] = math.ceil(math.log(0.01) / math.log((1 - ETA) ** 2))
assert (1 - ETA) ** (2 * V["D_m_1pct"]) <= 0.01 < (1 - ETA) ** (2 * (V["D_m_1pct"] - 1))
# Point de contrôle : y = (1, 2, 6), f0 = moyenne, résidus.
yc = [1, 2, 6]
V["Cr_moy"] = sum(yc) / len(yc)
V["Cr_res"] = [y - V["Cr_moy"] for y in yc]
V["Cr_d_med"] = [y - 2 for y in yc]  # écarts à la médiane, 2
V["Cr_d_signe"] = [V["Cr_moy"] - y for y in yc]  # gradient au lieu de son opposé
assert sorted(yc)[1] == 2

# Dérivées des pertes (sympy).
f, yy, delta = sp.symbols("f y delta", real=True)
pi_f = 1 / (1 + sp.exp(-f))
perte_log = -(yy * sp.log(pi_f) + (1 - yy) * sp.log(1 - pi_f))
g_log = sp.simplify(sp.diff(perte_log, f) - (pi_f - yy))
h_log = sp.simplify(sp.diff(perte_log, f, 2) - pi_f * (1 - pi_f))
assert g_log == 0 and h_log == 0
assert sp.simplify(sp.diff(pi_f, f) - pi_f * (1 - pi_f)) == 0
perte_quad = (yy - f) ** 2 / 2
assert sp.simplify(-sp.diff(perte_quad, f) - (yy - f)) == 0 and sp.diff(perte_quad, f, 2) == 1
assert sp.diff(perte_quad, f, 3) == 0
dire("Dérivées vérifiées par sympy : logistique g = pi - y, h = pi (1 - pi) ; quadratique g = f - y, h = 1.")
# Huber : pseudo-résidu borné par delta (contrôle numérique).
for d_ in [0.5, 1.0, 2.0]:
    res = np.linspace(-10, 10, 2001)
    hub = np.where(np.abs(res) <= d_, res**2 / 2, d_ * (np.abs(res) - d_ / 2))
    pseudo = np.gradient(hub, res)  # dérivée par rapport au résidu = moins la dérivée par rapport à f
    assert np.max(np.abs(pseudo)) <= d_ + 1e-9 and np.allclose(pseudo[np.abs(res) < d_ - 0.02], res[np.abs(res) < d_ - 0.02])

# =====================================================================================
# LEÇON, PARTIE 5 : XGBoost sur l'exemple jouet, perte logistique, score courant 0
# =====================================================================================
LAM = Fraction(1)
pi0 = Fraction(1, 2)
g_j = [pi0 - y for y in Y_JOUET]
h_j = [pi0 * (1 - pi0)] * n_j


def score(G, H, lam):
    return G * G / (H + lam)


def gain_xgb(gl, hl, gr, hr, lam, gam=0):
    return Fraction(1, 2) * (score(sum(gl), sum(hl), lam) + score(sum(gr), sum(hr), lam)
                             - score(sum(gl) + sum(gr), sum(hl) + sum(hr), lam)) - gam


V["E_g_pos"] = float(pi0 - 1)
V["E_g_neg"] = float(pi0 - 0)
V["E_h"] = float(pi0 * (1 - pi0))
V["E_G"] = float(sum(g_j))
V["E_H"] = float(sum(h_j))
gains_xgb = {k: gain_xgb(g_j[:k], h_j[:k], g_j[k:], h_j[k:], LAM) for k in range(1, n_j)}
k_x = max(gains_xgb, key=gains_xgb.get)
assert sorted(gains_xgb.values())[-1] > sorted(gains_xgb.values())[-2]
assert k_x == k_gini
V["E_seuil"] = k_x + 0.5
fraction("E_gain", gains_xgb[k_x])
GL, HL, GR, HR = sum(g_j[:k_x]), sum(h_j[:k_x]), sum(g_j[k_x:]), sum(h_j[k_x:])
V["E_GL"], V["E_HL"], V["E_GR"], V["E_HR"] = float(GL), float(HL), float(GR), float(HR)
fraction("E_sL", score(GL, HL, LAM))
fraction("E_sR", score(GR, HR, LAM))
fraction("E_sP", score(GL + GR, HL + HR, LAM))
fraction("E_wL", -GL / (HL + LAM))
fraction("E_wR", -GR / (HR + LAM))
ETA_X = Fraction(3, 10)
V["E_eta"] = float(ETA_X)
fraction("E_eta_wR", ETA_X * (-GR / (HR + LAM)))
V["E_pi_new"] = 1 / (1 + math.exp(-float(ETA_X * (-GR / (HR + LAM)))))
# Gain négatif avec lambda > 0, sans gamma : dernière coupure de l'exemple.
k_neg = min(gains_xgb, key=gains_xgb.get)
assert gains_xgb[k_neg] < 0
V["E_seuil_neg"] = k_neg + 0.5
V["E_gain_neg"] = float(gains_xgb[k_neg])
V["E_gain_neg_l0"] = float(gain_xgb(g_j[:k_neg], h_j[:k_neg], g_j[k_neg:], h_j[k_neg:], 0))
assert V["E_gain_neg_l0"] > 0
# Avec lambda = 0, le gain est toujours >= 0 (Cauchy-Schwarz) : contrôle exhaustif sur des cas aléatoires.
for _ in range(500):
    m = int(rng.integers(2, 9))
    gg = [Fraction(int(x), 4) for x in rng.integers(-8, 9, m)]
    hh = [Fraction(int(x), 4) for x in rng.integers(1, 9, m)]
    k = int(rng.integers(1, m))
    assert gain_xgb(gg[:k], hh[:k], gg[k:], hh[k:], 0) >= 0
# Poids optimal : contrôle par minimisation numérique du critère quadratique.
for _ in range(100):
    G_, H_, lam_ = float(rng.normal()), float(rng.uniform(0.1, 3)), float(rng.uniform(0, 2))
    grille = np.linspace(-20, 20, 400001)
    crit = G_ * grille + 0.5 * (H_ + lam_) * grille**2
    assert abs(grille[crit.argmin()] - (-G_ / (H_ + lam_))) < 2e-4
    assert abs(crit.min() - (-0.5 * G_**2 / (H_ + lam_))) < 1e-6
# Lien avec la perte quadratique : gain (lambda = 0, gamma = 0) = la moitié de la baisse de la somme des carrés.
for _ in range(200):
    m = int(rng.integers(2, 12))
    res = [Fraction(int(x)) for x in rng.integers(-6, 7, m)]
    k = int(rng.integers(1, m))
    sc = lambda v: sum((x - sum(v) / len(v)) ** 2 for x in v)
    g_ = [-x for x in res]
    assert gain_xgb(g_[:k], [1] * k, g_[k:], [1] * (m - k), 0) == Fraction(1, 2) * (sc(res) - sc(res[:k]) - sc(res[k:]))
dire("XGBoost : poids optimal, valeur optimale, gain >= 0 pour lambda = 0, lien avec la somme des carrés vérifiés.")
# Un pas de Newton contre le minimum exact, feuille de droite (4 observations d'étiquette 1), lambda = 1.
n_droite = n_j - k_x
assert all(y == 1 for y in Y_JOUET[k_x:])
w = 0.0
for _ in range(60):  # Newton sur la vraie perte pénalisée : sum log(1 + e^{-w}) + w^2 / 2
    p_ = 1 / (1 + math.exp(-w))
    d1 = n_droite * (p_ - 1) + float(LAM) * w
    d2 = n_droite * p_ * (1 - p_) + float(LAM)
    w -= d1 / d2
V["E_w_exact"] = w
assert abs(n_droite * (1 / (1 + math.exp(-w)) - 1) + w) < 1e-12
V["E_n_droite"] = n_droite
# min_child_weight : nombre minimal d'observations de même probabilité prédite pour atteindre une somme de h égale à 1.
for nom, p_ in [("E_n_099", Fraction(99, 100)), ("E_n_05", Fraction(1, 2))]:
    hval = p_ * (1 - p_)
    V[nom] = math.ceil(1 / hval)
    V[nom.replace("_n_", "_h_")] = float(hval)
    assert V[nom] * hval >= 1 > (V[nom] - 1) * hval

# =====================================================================================
# LEÇON, PARTIE 7 : biais de l'importance par impureté sous l'hypothèse nulle
# =====================================================================================
M_NUL, R_NUL = 100, 4000
best_bin, best_cont = [], []
for _ in range(R_NUL):
    y_ = rng.integers(0, 2, M_NUL)
    xb = rng.integers(0, 2, M_NUL)
    # variable binaire : une seule coupure possible
    if 0 < xb.sum() < M_NUL:
        e = [int(v) for v in y_]
        best_bin.append(float(gain(gini, e, [int(v) for v in y_[xb == 0]], [int(v) for v in y_[xb == 1]])))
    else:
        best_bin.append(0.0)
    # variable continue (valeurs distinctes) : M_NUL - 1 coupures, ordre aléatoire des étiquettes
    ordre = y_[rng.permutation(M_NUL)]
    cum = np.cumsum(ordre)[:-1]
    k = np.arange(1, M_NUL)
    pl, pr, p = cum / k, (ordre.sum() - cum) / (M_NUL - k), ordre.mean()
    gp = 2 * p * (1 - p)
    gains_c = gp - k / M_NUL * 2 * pl * (1 - pl) - (M_NUL - k) / M_NUL * 2 * pr * (1 - pr)
    best_cont.append(float(gains_c.max()))
best_bin, best_cont = np.array(best_bin), np.array(best_cont)
V["F_m"] = M_NUL
V["F_R"] = R_NUL
V["F_bin_moy"] = float(best_bin.mean())
V["F_cont_moy"] = float(best_cont.mean())
V["F_rapport"] = float(best_cont.mean() / best_bin.mean())
V["F_prop"] = float(np.mean(best_cont > best_bin))
V["F_cont_se"] = float(best_cont.std(ddof=1) / np.sqrt(R_NUL))
dire(f"Hypothèse nulle, m = {M_NUL} : meilleur gain moyen binaire {best_bin.mean():.5f}, continue {best_cont.mean():.5f}.")
# Contrôle exact du cas binaire : sous l'hypothèse nulle, E[gain de Gini d'une coupure fixe] se calcule par énumération.
# Pour une coupure de tailles (k, m - k) et des étiquettes i.i.d. de Bernoulli(1/2), on somme sur les nombres de 1 de chaque côté.


def gini_comptes(uns, taille):
    p = Fraction(uns, taille)
    return 2 * p * (1 - p)


def esperance_gain_fixe(k, m):
    tot = Fraction(0)
    for a in range(k + 1):
        for b in range(m - k + 1):
            pa = Fraction(math.comb(k, a) * math.comb(m - k, b), 2**m)
            g_ = gini_comptes(a + b, m) - Fraction(k, m) * gini_comptes(a, k) - Fraction(m - k, m) * gini_comptes(b, m - k)
            tot += pa * g_
    return tot


assert esperance_gain_fixe(3, 7) == sum(  # contrôle de la version par comptes contre la version par listes
    Fraction(math.comb(3, a) * math.comb(4, b), 2**7) * gain(gini, [1] * (a + b) + [0] * (7 - a - b), [1] * a + [0] * (3 - a), [1] * b + [0] * (4 - b))
    for a in range(4) for b in range(5))


# Espérance exacte pour la variable binaire : on moyenne sur la taille k du groupe x = 0, binomiale(m, 1/2) (k = 0 ou m : gain nul).
esp_bin = sum(Fraction(math.comb(M_NUL, k), 2**M_NUL) * esperance_gain_fixe(k, M_NUL) for k in range(1, M_NUL))
V["F_bin_exact"] = float(esp_bin)
mc_moyenne("F_bin", esp_bin, best_bin)

# =====================================================================================
# QUESTIONS
# =====================================================================================
# M1-01 : gain de Gini, noeud de 10 observations (6 positives), gauche 4 (1 positive), droite 6 (5 positives).
par = [1] * 6 + [0] * 4
gau = [1] + [0] * 3
dro = [1] * 5 + [0]
fraction("Q01_parent", gini(par))
fraction("Q01_gauche", gini(gau))
fraction("Q01_droite", gini(dro))
fraction("Q01_enfants", Fraction(4, 10) * gini(gau) + Fraction(6, 10) * gini(dro))
fraction("Q01_exact", gain(gini, par, gau, dro))
fraction("Q01_d_non_pondere", gini(par) - (gini(gau) + gini(dro)) / 2)
fraction("Q01_d_demi", gain(gini, par, gau, dro) / 2)
assert len({V["Q01_exact"], V["Q01_d_non_pondere"], V["Q01_d_demi"], V["Q01_enfants"]}) == 4

# M1-02 : invariance monotone. Valeurs -3, -1, 2, 4 ; transformation x -> x^2.
xs = [-3, -1, 2, 4]
V["Q02_carres"] = [x * x for x in xs]
ordre_x = sorted(range(4), key=lambda i: xs[i])
ordre_x2 = sorted(range(4), key=lambda i: xs[i] ** 2)
parts = lambda ordre: {frozenset(ordre[:k]) for k in range(1, 4)}
V["Q02_nb_coupures"] = len(parts(ordre_x))
V["Q02_communes"] = len(parts(ordre_x) & parts(ordre_x2))
for t in [lambda x: 2 * x + 5, math.exp, lambda x: x**3]:
    assert sorted(range(4), key=lambda i: t(xs[i])) == ordre_x

# M1-03 : nombre moyen d'observations distinctes dans un échantillon bootstrap, n = 8.
n3 = 8
p_abs = Fraction(n3 - 1, n3) ** n3
V["Q03_exact"] = float(n3 * (1 - p_abs))
V["Q03_d_lim"] = n3 * (1 - math.exp(-1))
V["Q03_d_oob"] = float(n3 * p_abs)
V["Q03_d_un"] = float(1 - p_abs)
V["Q03_p_abs"] = float(p_abs)
tirages = rng.integers(0, n3, size=(200000, n3))
distincts = (np.sort(tirages, axis=1)[:, 1:] != np.sort(tirages, axis=1)[:, :-1]).sum(axis=1) + 1
mc_moyenne("Q03", V["Q03_exact"], distincts)

# M1-04 : deux mesures exactes de variance (B = 5 : 0,5 ; B = 20 : 0,35) -> rho.
v5, v20 = Fraction(1, 2), Fraction(35, 100)
b_coef = (v5 - v20) / (Fraction(1, 5) - Fraction(1, 20))
a_coef = v5 - b_coef / 5
sigma2_q4 = a_coef + b_coef
V["Q04_diff"] = float(v5 - v20)
assert (Fraction(1, 5) - Fraction(1, 20)) == v5 - v20  # d'où b = 1
fraction("Q04_b", b_coef)
fraction("Q04_a", a_coef)
fraction("Q04_sigma2", sigma2_q4)
fraction("Q04_rho", a_coef / sigma2_q4)
fraction("Q04_d_un_moins", 1 - a_coef / sigma2_q4)
assert a_coef + b_coef / 20 == v20

# M1-05 : biais^2 = 0,1 ; Var(Y) = 0,5 ; sigma2 = 1 ; rho = 0,2.
b2, vy, s2, r5 = Fraction(1, 10), Fraction(1, 2), Fraction(1), Fraction(1, 5)
V["Q05_lim"] = float(b2 + r5 * s2 + vy)
V["Q05_B1"] = float(b2 + s2 + vy)
V["Q05_sans_rho"] = float(b2 + vy)
V["Q05_sans_bruit"] = float(b2 + r5 * s2)
V["Q05_B100"] = float(b2 + r5 * s2 + (1 - r5) * s2 / 100 + vy)

# M1-06 : bagging (sigma2 = 1, rho = 0,6) contre forêt (sigma2 = 1,3, rho = 0,3).
s_bag, r_bag, s_rf, r_rf = Fraction(1), Fraction(6, 10), Fraction(13, 10), Fraction(3, 10)
vb = lambda b: r_bag * s_bag + (1 - r_bag) * s_bag / b
vr = lambda b: r_rf * s_rf + (1 - r_rf) * s_rf / b
b_seuil = ((1 - r_rf) * s_rf - (1 - r_bag) * s_bag) / (r_bag * s_bag - r_rf * s_rf)
V["Q06_b_seuil"] = float(b_seuil)
V["Q06_B_min"] = math.floor(b_seuil) + 1
assert all((vr(b) < vb(b)) == (b >= V["Q06_B_min"]) for b in range(1, 200))
V["Q06_bag_plancher"] = float(r_bag * s_bag)
V["Q06_rf_plancher"] = float(r_rf * s_rf)
V["Q06_bag_1"], V["Q06_rf_1"] = float(vb(1)), float(vr(1))
V["Q06_bag_2"], V["Q06_rf_2"] = float(vb(2)), float(vr(2))
V["Q06_bag_3"], V["Q06_rf_3"] = float(vb(3)), float(vr(3))
V["Q06_rf_b"] = float((1 - r_rf) * s_rf)
V["Q06_bag_b"] = float((1 - r_bag) * s_bag)
V["Q06_num"] = float((1 - r_rf) * s_rf - (1 - r_bag) * s_bag)
V["Q06_den"] = float(r_bag * s_bag - r_rf * s_rf)

# M1-07 : pseudo-résidu logistique, score ln 3, y = 0.
f7 = math.log(3)
pi7 = 1 / (1 + math.exp(-f7))
assert abs(pi7 - 0.75) < 1e-15
V["Q07_pi"] = pi7
V["Q07_r"] = 0 - pi7
V["Q07_d_score"] = 0 - f7
V["Q07_h"] = pi7 * (1 - pi7)

# M1-08 : poids de feuille logistique, étiquettes (1, 1, 1, 1, 0), score courant 0, lambda = 0,5.
y8 = [1, 1, 1, 1, 0]
g8 = [Fraction(1, 2) - y for y in y8]
h8 = [Fraction(1, 4)] * 5
lam8 = Fraction(1, 2)
V["Q08_G"], V["Q08_H"] = float(sum(g8)), float(sum(h8))
fraction("Q08_exact", -sum(g8) / (sum(h8) + lam8))
fraction("Q08_d_sans_lambda", -sum(g8) / sum(h8))
fraction("Q08_d_h_un", -sum(g8) / (5 + lam8))
fraction("Q08_d_moyenne", -sum(g8) / 5)

# M1-09 : gain d'une coupure, perte quadratique, résidus (2, 2, 3) | (-1, -2), lambda = 1, gamma = 1.
rl, rr = [2, 2, 3], [-1, -2]
gl9, gr9 = [Fraction(-x) for x in rl], [Fraction(-x) for x in rr]
V["Q09_GL"], V["Q09_GR"] = float(sum(gl9)), float(sum(gr9))
V["Q09_G"] = float(sum(gl9) + sum(gr9))
V["Q09_HL"], V["Q09_HR"] = len(rl), len(rr)
fraction("Q09_sL", score(sum(gl9), 3, 1))
fraction("Q09_sR", score(sum(gr9), 2, 1))
fraction("Q09_sP", score(sum(gl9) + sum(gr9), 5, 1))
V["Q09_exact"] = float(gain_xgb(gl9, [1] * 3, gr9, [1] * 2, 1, 1))
V["Q09_d_sans_demi"] = float(2 * gain_xgb(gl9, [1] * 3, gr9, [1] * 2, 1, 0) - 1)
V["Q09_d_sans_lambda"] = float(gain_xgb(gl9, [1] * 3, gr9, [1] * 2, 0, 1))
V["Q09_d_sans_gamma"] = float(gain_xgb(gl9, [1] * 3, gr9, [1] * 2, 1, 0))
V["Q09_wL"] = float(-sum(gl9) / (3 + 1))
V["Q09_wR"] = float(-sum(gr9) / (2 + 1))

# M1-10 : min_child_weight = 1, perte logistique, probabilité prédite 0,8.
p10 = Fraction(8, 10)
h10 = p10 * (1 - p10)
fraction("Q10_h", h10)
V["Q10_inv"] = float(1 / h10)
V["Q10_exact"] = math.ceil(1 / h10)
V["Q10_d_inv_un_moins"] = float(1 / (1 - p10))
V["Q10_d_max"] = 4
assert V["Q10_exact"] * h10 >= 1 > (V["Q10_exact"] - 1) * h10


Path(__file__).with_name("seance_ml_2.out.json").write_text(json.dumps(V, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
dire(f"{len(V)} valeurs exportées dans verif/seance_ml_2.out.json.")
dire("Toutes les vérifications passent.")
Path(__file__).with_name("seance_ml_2.out.txt").write_text("\n".join(LIGNES) + "\n", encoding="utf-8", newline="\n")
