"""Vérification de tous les nombres de la séance de ML 3 (leçon L-ML-3 et questions ML3-01 à ML3-08).

Principe :
  1. chaque valeur citée est calculée ici, en fractions exactes quand c'est possible ;
  2. chaque formule démontrée dans la leçon est contrôlée sur des cas numériques (dérivées par sympy,
     gradient contre différences finies, hessienne semi-définie positive, AUC par paires contre aire
     de la courbe ROC en escalier, purge et embargo par énumération des dates) ;
  3. chaque probabilité ou espérance est aussi estimée par simulation Monte-Carlo, dont l'écart à l'exact
     doit rester inférieur à 4 erreurs standard.
Les expériences avec scikit-learn sont dans la leçon elle-même (blocs exécutés à la construction) ; ce script
n'utilise que numpy, sympy et fractions, pour donner la même sortie quelle que soit la version de scikit-learn.

Graine fixe : la sortie est identique à chaque exécution.
Sorties : verif/seance_ml_3.out.json (lu par outils/construire.py) et verif/seance_ml_3.out.txt (ce résumé).
"""
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp

rng = np.random.default_rng(20261008)
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


def sigma(t):
    return 1.0 / (1.0 + np.exp(-t))


def logloss(X, y, b):
    """Log-loss moyenne, écrite sous la forme stable (1/n) somme [ln(1 + e^t) - y t]."""
    t = X.dot(b)
    return float(np.mean(np.logaddexp(0.0, t) - y * t))


def gradient(X, y, b):
    return X.T.dot(sigma(X.dot(b)) - y) / len(y)


def hessienne(X, y, b):
    p = sigma(X.dot(b))
    return (X * (p * (1 - p))[:, None]).T.dot(X) / len(y)


def newton(X, y, iterations=50):
    b = np.zeros(X.shape[1])
    for _ in range(iterations):
        b = b - np.linalg.solve(hessienne(X, y, b), gradient(X, y, b))
    assert np.max(np.abs(gradient(X, y, b))) < 1e-12
    return b


# =====================================================================================
# LEÇON, PARTIE 1 : régression logistique
# =====================================================================================
# Dérivées symboliques : sigma' = sigma (1 - sigma) ; d/dt [ln(1 + e^t) - y t] = sigma(t) - y ; dérivée seconde sigma (1 - sigma).
t_s, y_s = sp.symbols("t y", real=True)
sig_s = 1 / (1 + sp.exp(-t_s))
assert sp.simplify(sp.diff(sig_s, t_s) - sig_s * (1 - sig_s)) == 0
perte_s = sp.log(1 + sp.exp(t_s)) - y_s * t_s
assert sp.simplify(sp.diff(perte_s, t_s) - (sig_s - y_s)) == 0
assert sp.simplify(sp.diff(perte_s, t_s, 2) - sig_s * (1 - sig_s)) == 0
# Les deux écritures de la perte coïncident : -[y ln sigma + (1 - y) ln(1 - sigma)] = ln(1 + e^t) - y t.
for yv in (0, 1):
    ecart = -(yv * sp.log(sig_s) + (1 - yv) * sp.log(1 - sig_s)) - perte_s.subs(y_s, yv)
    assert all(abs(float(ecart.subs(t_s, tv))) < 1e-12 for tv in (-3.7, -0.2, 0.0, 1.3, 5.1))
dire("sympy : sigma' = sigma(1 - sigma), dérivée de la perte = sigma - y, dérivée seconde = sigma(1 - sigma).")

# Gradient et hessienne contre différences finies, hessienne semi-définie positive, sur 100 cas aléatoires.
for _ in range(100):
    n_, p_ = int(rng.integers(3, 30)), int(rng.integers(1, 5))
    X_ = np.column_stack([np.ones(n_), rng.normal(size=(n_, p_))])
    y_ = rng.integers(0, 2, n_).astype(float)
    b_ = rng.normal(size=p_ + 1)
    eps = 1e-6
    g_num = np.array([(logloss(X_, y_, b_ + eps * e) - logloss(X_, y_, b_ - eps * e)) / (2 * eps) for e in np.eye(p_ + 1)])
    assert np.max(np.abs(g_num - gradient(X_, y_, b_))) < 1e-7
    h_num = np.array([(gradient(X_, y_, b_ + eps * e) - gradient(X_, y_, b_ - eps * e)) / (2 * eps) for e in np.eye(p_ + 1)])
    assert np.max(np.abs(h_num - hessienne(X_, y_, b_))) < 1e-7
    assert np.linalg.eigvalsh(hessienne(X_, y_, b_)).min() > -1e-12
dire("Gradient X^T(pi - y)/n et hessienne X^T W X/n contrôlés par différences finies ; hessienne SDP, 100 cas.")

# Exemple chiffré : x = (-1, 0, 1, 2), y = (0, 1, 0, 1), avec constante.
X1 = np.column_stack([np.ones(4), [-1.0, 0.0, 1.0, 2.0]])
Y1 = np.array([0.0, 1.0, 0.0, 1.0])
V["L1_ln2"] = math.log(2)
assert abs(logloss(X1, Y1, np.zeros(2)) - math.log(2)) < 1e-15
g0 = [Fraction(1, 4) * sum(Fraction(int(X1[i, j])) * (Fraction(1, 2) - Fraction(int(Y1[i]))) for i in range(4)) for j in range(2)]
V["L1_g0"] = [float(v) for v in g0]
V["L1_g0_1"], V["L1_g0_2"] = float(g0[0]), float(g0[1])
V["L1_xg"] = float(sum(Fraction(int(X1[i, 1])) * (Fraction(1, 2) - Fraction(int(Y1[i]))) for i in range(4)))
assert np.allclose(gradient(X1, Y1, np.zeros(2)), V["L1_g0"])
xtx = X1.T.dot(X1)
V["L1_xtx11"], V["L1_xtx12"], V["L1_xtx22"] = float(xtx[0, 0]), float(xtx[0, 1]), float(xtx[1, 1])
h0 = [[Fraction(int(xtx[i, j]), 16) for j in range(2)] for i in range(2)]
V["L1_h11"], V["L1_h12"], V["L1_h22"] = float(h0[0][0]), float(h0[0][1]), float(h0[1][1])
assert np.allclose(hessienne(X1, Y1, np.zeros(2)), np.array(h0, dtype=float))
V["L1_poids0"] = 0.25  # sigma(0)(1 - sigma(0))
assert float(sigma(0.0) * (1 - sigma(0.0))) == V["L1_poids0"]
fraction("L1_hfact", Fraction(1, 4) * Fraction(1, 4))  # poids 1/4, moyenne sur n = 4
# Premier pas de Newton, exact : b1 = -H^{-1} g.
H0 = sp.Matrix(2, 2, lambda i, j: sp.Rational(h0[i][j].numerator, h0[i][j].denominator))
pas = -H0.inv() * sp.Matrix([sp.Rational(g0[0].numerator, g0[0].denominator), sp.Rational(g0[1].numerator, g0[1].denominator)])
V["L1_newton1"] = [float(v) for v in pas]
b_hat1 = newton(X1, Y1)
V["L1_bhat"] = [float(v) for v in b_hat1]
V["L1_bhat0"], V["L1_bhat1"] = float(b_hat1[0]), float(b_hat1[1])
pi_hat1 = sigma(X1.dot(b_hat1))
V["L1_pihat"] = [float(v) for v in pi_hat1]
V["L1_somme_pihat"] = float(pi_hat1.sum())
assert abs(pi_hat1.sum() - Y1.sum()) < 1e-12  # condition du premier ordre sur la constante
V["L1_somme_y"] = float(Y1.sum())
V["L1_loss_hat"] = logloss(X1, Y1, b_hat1)
assert V["L1_loss_hat"] < math.log(2)
V["L1_or_pente"] = math.exp(b_hat1[1])
dire(f"Exemple logistique : b_chapeau = {b_hat1}, log-loss {V['L1_loss_hat']:.6f} < ln 2.")

# Rapport de cotes : coefficient 0,7.
B_OR = 0.7
V["L1_beta_or"] = B_OR
V["L1_or"] = math.exp(B_OR)
for nom, p0 in [("L1_p05", Fraction(1, 2)), ("L1_p09", Fraction(9, 10))]:
    cote0 = p0 / (1 - p0)
    cote1 = float(cote0) * math.exp(B_OR)
    V[nom] = cote1 / (1 + cote1)
    V[nom + "_cote"] = cote1
    V[nom + "_cote0"] = float(cote0)
    assert abs(V[nom] - sigma(math.log(float(cote0)) + B_OR)) < 1e-15
V["L1_p05_hausse"] = V["L1_p05"] - 0.5
V["L1_p09_hausse"] = V["L1_p09"] - 0.9

# Séparation parfaite : x = (-2, -1, 1, 2), y = (0, 0, 1, 1), direction b = c (0, 1).
Xs = np.column_stack([np.ones(4), [-2.0, -1.0, 1.0, 2.0]])
Ys = np.array([0.0, 0.0, 1.0, 1.0])
for c in (1, 5, 10):
    V[f"L1_sep_c{c}"] = logloss(Xs, Ys, np.array([0.0, float(c)]))
    exact = 0.5 * (math.log1p(math.exp(-2 * c)) + math.log1p(math.exp(-c)))
    assert abs(V[f"L1_sep_c{c}"] - exact) < 1e-15
assert V["L1_sep_c1"] > V["L1_sep_c5"] > V["L1_sep_c10"] > 0
dire(f"Séparation : log-loss en c (0, 1) pour c = 1, 5, 10 : {V['L1_sep_c1']:.6f}, {V['L1_sep_c5']:.6f}, {V['L1_sep_c10']:.2e}.")

# =====================================================================================
# LEÇON, PARTIE 2 : métriques
# =====================================================================================
VP, FN, FP, VN = 60, 40, 90, 810
n2 = VP + FN + FP + VN
V["L2_n"], V["L2_pos"] = n2, VP + FN
fraction("L2_prec", Fraction(VP, VP + FP))
fraction("L2_rap", Fraction(VP, VP + FN))
fraction("L2_f1", Fraction(2 * VP, 2 * VP + FP + FN))
f1_moy = 2 * Fraction(VP, VP + FP) * Fraction(VP, VP + FN) / (Fraction(VP, VP + FP) + Fraction(VP, VP + FN))
assert f1_moy == Fraction(2 * VP, 2 * VP + FP + FN)  # moyenne harmonique = 2 VP / (2 VP + FP + FN)
fraction("L2_exact", Fraction(VP + VN, n2))
fraction("L2_const", Fraction(FP + VN, n2))
fraction("L2_tfp", Fraction(FP, FP + VN))
fraction("L2_prev", Fraction(VP + FN, n2))
# Précision et prévalence : TVP = 0,9, TFP = 0,05.
TVP_R, TFP_R = Fraction(9, 10), Fraction(1, 20)
for nom, q in [("L2_prec_rare", Fraction(1, 100)), ("L2_prec_eq", Fraction(1, 2))]:
    fraction(nom, q * TVP_R / (q * TVP_R + (1 - q) * TFP_R))
    fraction(nom + "_vp", q * TVP_R)
    fraction(nom + "_fp", (1 - q) * TFP_R)
fraction("L2_fp_par_vp", (Fraction(99, 100) * TFP_R) / (Fraction(1, 100) * TVP_R))
# Contrôle de la formule de la précision par simulation (prévalence 1 %).
N_MC = 400000
Yr = rng.uniform(size=N_MC) < 0.01
pred_r = np.where(Yr, rng.uniform(size=N_MC) < 0.9, rng.uniform(size=N_MC) < 0.05)
vp_r = np.logical_and(pred_r, Yr)
# La précision empirique est un rapport : on contrôle séparément les deux proportions qui la composent.
mc_moyenne("L2_prec_rare_vp", float(Fraction(1, 100) * TVP_R), vp_r)
mc_moyenne("L2_prec_rare_fp", float(Fraction(99, 100) * TFP_R), np.logical_and(pred_r, ~Yr))
V["L2_prec_rare_mc_rapport"] = float(vp_r.sum() / pred_r.sum())

# AUC : positifs (0,9 ; 0,5 ; 0,3), négatifs (0,8 ; 0,6 ; 0,4 ; 0,2).


def auc_paires(pos, neg):
    """Proportion de paires (positif, négatif) bien ordonnées, ex aequo comptés 1/2 (fractions exactes)."""
    total = Fraction(0)
    for a in pos:
        for b in neg:
            total += 1 if a > b else (Fraction(1, 2) if a == b else 0)
    return total / (len(pos) * len(neg))


def auc_escalier(pos, neg):
    """Aire exacte sous la courbe ROC empirique : seuils décroissants, ex aequo traités par un segment (trapèze)."""
    seuils = sorted(set(pos) | set(neg), reverse=True)
    x0, y0, aire = Fraction(0), Fraction(0), Fraction(0)
    for s in seuils:
        x1 = x0 + Fraction(sum(b == s for b in neg), len(neg))
        y1 = y0 + Fraction(sum(a == s for a in pos), len(pos))
        aire += (x1 - x0) * (y0 + y1) / 2
        x0, y0 = x1, y1
    return aire


POS2 = [Fraction(9, 10), Fraction(5, 10), Fraction(3, 10)]
NEG2 = [Fraction(8, 10), Fraction(6, 10), Fraction(4, 10), Fraction(2, 10)]
fraction("L2_auc", auc_paires(POS2, NEG2))
assert auc_paires(POS2, NEG2) == auc_escalier(POS2, NEG2)
V["L2_auc_bien"] = int(auc_paires(POS2, NEG2) * 12)
V["L2_auc_paires"] = len(POS2) * len(NEG2)
V["L2_auc_par_pos"] = [sum(a > b for b in NEG2) for a in POS2]
# Le théorème (aire = proportion de paires) sur 500 cas aléatoires, avec et sans ex aequo.
for _ in range(500):
    m1, m0 = int(rng.integers(1, 8)), int(rng.integers(1, 8))
    pos = [Fraction(int(v), 6) for v in rng.integers(0, 7, m1)]
    neg = [Fraction(int(v), 6) for v in rng.integers(0, 7, m0)]
    assert auc_paires(pos, neg) == auc_escalier(pos, neg)
    # Invariance par transformation strictement croissante (cube) et score opposé.
    assert auc_paires([a ** 3 for a in pos], [b ** 3 for b in neg]) == auc_paires(pos, neg)
    assert auc_paires([-a for a in pos], [-b for b in neg]) == 1 - auc_paires(pos, neg)
dire("AUC = proportion de paires bien ordonnées (ex aequo 1/2) = aire de la ROC en escalier, 500 cas ; invariance monotone.")

# Règles de score propres, Y de Bernoulli(q) avec q = 0,3.
q_s = sp.Rational(3, 10)
r_s = sp.symbols("r", positive=True)
brier_esp = q_s * (r_s - 1) ** 2 + (1 - q_s) * r_s ** 2
assert sp.expand(brier_esp - ((r_s - q_s) ** 2 + q_s * (1 - q_s))) == 0
log_esp = -(q_s * sp.log(r_s) + (1 - q_s) * sp.log(1 - r_s))
assert sp.solve(sp.diff(log_esp, r_s), r_s) == [q_s]
assert sp.simplify(sp.diff(log_esp, r_s, 2) - (q_s / r_s ** 2 + (1 - q_s) / (1 - r_s) ** 2)) == 0
fraction("L2_q", Fraction(3, 10))
fraction("L2_brier_q", Fraction(3, 10) * Fraction(7, 10))
fraction("L2_brier_05", Fraction(3, 10) * Fraction(7, 10) + (Fraction(1, 2) - Fraction(3, 10)) ** 2)
fraction("L2_brier_ecart", (Fraction(1, 2) - Fraction(3, 10)) ** 2)
V["L2_ll_q"] = -(0.3 * math.log(0.3) + 0.7 * math.log(0.7))
V["L2_ll_05"] = math.log(2)
Yb = (rng.uniform(size=N_MC) < 0.3).astype(float)
# La prévision 0,5 a un score de Brier constant, (0,5 - y)^2 = 0,25 pour y = 0 ou 1 : pas de simulation utile.
assert np.all((0.5 - Yb) ** 2 == 0.25) and Fraction(V["L2_brier_05_num"], V["L2_brier_05_den"]) == Fraction(1, 4)
mc_moyenne("L2_brier_q", V["L2_brier_q"], (0.3 - Yb) ** 2)
mc_moyenne("L2_ll_q", V["L2_ll_q"], -(Yb * math.log(0.3) + (1 - Yb) * math.log(0.7)))
# Seuil de décision par les coûts : prédire 1 si et seulement si pi > c_FP / (c_FP + c_FN).
fraction("L2_seuil", Fraction(1, 1 + 4))

# =====================================================================================
# LEÇON, PARTIE 3 : plus proches voisins, échelle, dimension
# =====================================================================================
Q3 = (30, 50000)
A3 = (60, 50500)
B3 = (31, 52000)
ET = (10, 10000)  # écarts-types d'apprentissage supposés : 10 ans, 10 000 euros
for nom, P in [("A", A3), ("B", B3)]:
    d2 = sum(Fraction(a - b) ** 2 for a, b in zip(P, Q3))
    V[f"L3_d{nom}_2"] = float(d2)
    V[f"L3_d{nom}"] = math.sqrt(d2)
    z2 = sum((Fraction(a - b) / s) ** 2 for a, b, s in zip(P, Q3, ET))
    fraction(f"L3_z{nom}_2", z2)
    V[f"L3_z{nom}"] = math.sqrt(z2)
assert V["L3_dA"] < V["L3_dB"] and V["L3_zB"] < V["L3_zA"]
V["L3_dA_age"], V["L3_dA_rev"] = (A3[0] - Q3[0]) ** 2, (A3[1] - Q3[1]) ** 2

# Concentration des distances, coordonnées uniformes sur [0, 1].
u, w = sp.symbols("u w", real=True)
m2 = sp.integrate(sp.integrate((u - w) ** 2, (u, 0, 1)), (w, 0, 1))
m4 = sp.integrate(sp.integrate((u - w) ** 4, (u, 0, 1)), (w, 0, 1))
assert m2 == sp.Rational(1, 6) and m4 == sp.Rational(1, 15)
c4 = m4 - m2 ** 2
fraction("L3_m2", Fraction(1, 6))
fraction("L3_m4", Fraction(1, 15))
fraction("L3_c", Fraction(int(c4.p), int(c4.q)))
coef = math.sqrt(float(c4)) / float(m2)  # rapport écart-type / espérance de D^2 multiplié par racine de p
fraction("L3_coef2", Fraction(int(c4.p), int(c4.q)) / Fraction(1, 36))
V["L3_coef"] = coef
for p in (2, 10, 100, 1000):
    V[f"L3_ratio_{p}"] = coef / math.sqrt(p)
# Simulation : rapport écart-type / espérance de D^2 pour p = 100.
D2 = ((rng.uniform(size=(40000, 100)) - rng.uniform(size=(40000, 100))) ** 2).sum(axis=1)
mc_moyenne("L3_esp_D2_100", 100 / 6, D2)
V["L3_esp_100"] = 100 / 6
V["L3_var_100"] = 100 * float(c4)
# Erreur standard de la variance empirique : sqrt((m4c - var^2) / N), avec m4c le moment centré d'ordre 4 empirique.
var_emp = D2.var(ddof=1)
m4c = np.mean((D2 - D2.mean()) ** 4)
controle_mc("L3_var_D2_100", 100 * float(c4), var_emp, math.sqrt((m4c - var_emp ** 2) / len(D2)))
# Tchebychev, p = 1000, epsilon = 0,1 : borne coef^2 / (p eps^2).
fraction("L3_cheb_1000", Fraction(int(c4.p), int(c4.q)) * 36 / (1000 * Fraction(1, 100)))
# Côté du sous-cube qui contient 1 % du volume.
for p in (1, 10, 100):
    V[f"L3_arete_{p}"] = 0.01 ** (1 / p)

# =====================================================================================
# LEÇON, PARTIE 4 : validation, purge et embargo
# =====================================================================================
# Corrélation de deux étiquettes à 5 jours décalées de k jours (rendements i.i.d. de variance v) : (5 - k)/5.
for k in range(0, 6):
    fraction(f"L4_corr_{k}", Fraction(max(5 - k, 0), 5))
r_mc = rng.normal(size=(200000, 6))
e0 = r_mc[:, 0:5].sum(axis=1)
e1 = r_mc[:, 1:6].sum(axis=1)
# Covariance estimée par la moyenne des produits (espérances nulles connues) ; exacte : 4 (v = 1).
mc_moyenne("L4_cov_1", 4, e0 * e1)


def purge_embargo(N, a, b, h, L):
    """Dates 0..N-1 ; bloc de test [a, b] ; étiquette de t : rendements t+1..t+h ; variables de t : rendements t-L+1..t.
    Renvoie (purgées, embargo, apprentissage) par énumération directe des règles."""
    rend_test = set(range(a + 1, b + h + 1))  # rendements qui entrent dans au moins une étiquette de test
    purgees, embargo, app = [], [], []
    for t in range(N):
        if a <= t <= b:
            continue
        if rend_test & set(range(t + 1, t + h + 1)):
            purgees.append(t)
        elif t > b and rend_test & set(range(t - L + 1, t + 1)):
            embargo.append(t)
        else:
            app.append(t)
    return purgees, embargo, app


pu, em, ap = purge_embargo(1000, 400, 599, 5, 20)
V["L4_purge_av"] = [min(t for t in pu if t < 400), max(t for t in pu if t < 400)]
V["L4_purge_ap"] = [min(t for t in pu if t > 599), max(t for t in pu if t > 599)]
V["L4_emb"] = [min(em), max(em)]
V["L4_n_purge"], V["L4_n_emb"], V["L4_n_app"] = len(pu), len(em), len(ap)
(V["L4_pav1"], V["L4_pav2"]), (V["L4_pap1"], V["L4_pap2"]), (V["L4_emb1"], V["L4_emb2"]) = V["L4_purge_av"], V["L4_purge_ap"], V["L4_emb"]
V["L4_n_sans"] = 1000 - 200
V["L4_hm1"] = 5 - 1
V["L4_Lp4"] = 20 + 5 - 1
assert V["L4_emb2"] == 599 + V["L4_Lp4"]
assert V["L4_purge_av"] == [396, 399] and V["L4_purge_ap"] == [600, 603] and V["L4_emb"] == [604, 623]
assert len(pu) == 2 * (5 - 1) and len(em) == 20 and len(ap) == 1000 - 200 - 8 - 20
# Formules générales : purge h - 1 de chaque côté, embargo L après ; contrôle sur des cas aléatoires.
for _ in range(200):
    N = int(rng.integers(200, 600))
    h, L = int(rng.integers(1, 12)), int(rng.integers(1, 40))
    a = int(rng.integers(h + L, N // 2))
    b = int(rng.integers(a, N - h - L - 1))
    pu_, em_, ap_ = purge_embargo(N, a, b, h, L)
    assert pu_ == list(range(a - h + 1, a)) + list(range(b + 1, b + h))
    assert em_ == list(range(b + h, b + h + L))
dire(f"Purge et embargo (N = 1000, test 400..599, h = 5, L = 20) : purgées {pu}, embargo {V['L4_emb']}, apprentissage {len(ap)}.")
# Écart entre moyenne globale et moyenne d'apprentissage (standardiser avant de découper), données i.i.d.
# m_glob - m_app = (n_test / n)(m_test - m_app), de variance s^2 (n_test/n)^2 (1/n_test + 1/n_app).
n_app_, n_test_ = 800, 200
var_ecart = (n_test_ / 1000) ** 2 * (1 / n_test_ + 1 / n_app_)
V["L4_sd_ecart"] = math.sqrt(var_ecart)
Z = rng.normal(size=(20000, 1000))
ecarts = Z.mean(axis=1) - Z[:, :n_app_].mean(axis=1)
mc_moyenne("L4_var_ecart", var_ecart, ecarts ** 2)

# =====================================================================================
# QUESTIONS
# =====================================================================================
# ML3-01 : coefficient 0,4, probabilité de départ 0,2.
b01, p01 = 0.4, Fraction(1, 5)
cote01 = float(p01 / (1 - p01))
V["Q01_cote0"] = cote01
V["Q01_or"] = math.exp(b01)
V["Q01_cote1"] = cote01 * math.exp(b01)
V["Q01_exact"] = V["Q01_cote1"] / (1 + V["Q01_cote1"])
assert abs(V["Q01_exact"] - sigma(math.log(cote01) + b01)) < 1e-15
V["Q01_d_prob"] = float(p01) * math.exp(b01)
V["Q01_d_lin"] = float(p01 + Fraction(2, 5) * p01 * (1 - p01))
V["Q01_d_sig"] = float(sigma(b01))
V["Q01_derivee"] = float(p01 * (1 - p01))
assert len({round(V[k], 3) for k in ["Q01_exact", "Q01_d_prob", "Q01_d_lin", "Q01_d_sig"]}) == 4

# ML3-02 : gradient de la log-loss moyenne, x = (-1, 0, 2), y = (0, 1, 1), b = (0, 1).
X2 = np.column_stack([np.ones(3), [-1.0, 0.0, 2.0]])
Y2 = np.array([0.0, 1.0, 1.0])
b2 = np.array([0.0, 1.0])
pi2 = sigma(X2.dot(b2))
V["Q02_pi"] = [float(v) for v in pi2]
V["Q02_res"] = [float(v) for v in pi2 - Y2]
g2 = gradient(X2, Y2, b2)
V["Q02_exact"] = [float(v) for v in g2]
V["Q02_d_signe"] = [float(-v) for v in g2]
V["Q02_d_sans_n"] = [float(v) for v in 3 * g2]
W2 = pi2 * (1 - pi2)
V["Q02_d_poids"] = [float(v) for v in X2.T.dot(W2 * (pi2 - Y2)) / 3]
V["Q02_somme"] = float((pi2 - Y2).sum())
V["Q02_somme_x"] = float(X2[:, 1].dot(pi2 - Y2))
eps = 1e-6
g_num2 = np.array([(logloss(X2, Y2, b2 + eps * e) - logloss(X2, Y2, b2 - eps * e)) / (2 * eps) for e in np.eye(2)])
assert np.max(np.abs(g_num2 - g2)) < 1e-8
V["Q02_loss_b"] = logloss(X2, Y2, b2)
V["Q02_loss_pas"] = logloss(X2, Y2, b2 - 0.1 * g2)
assert V["Q02_loss_pas"] < V["Q02_loss_b"]

# ML3-03 : existence du minimiseur, x = (1, 2, 3, 4), avec constante.
X3 = np.column_stack([np.ones(4), [1.0, 2.0, 3.0, 4.0]])
jeux = {"a": [0, 1, 0, 1], "b": [0, 0, 1, 1], "c": [0, 0, 0, 1], "d": [1, 1, 1, 1]}
b3 = newton(X3, np.array(jeux["a"], dtype=float))
V["Q03_bhat"] = [float(v) for v in b3]
V["Q03_loss_hat"] = logloss(X3, np.array(jeux["a"], dtype=float), b3)
assert np.linalg.eigvalsh(hessienne(X3, np.array(jeux["a"], dtype=float), b3)).min() > 0
# Jeux séparés : log-loss le long d'une direction séparante, pour c = 1, 10.
directions = {"b": np.array([-2.5, 1.0]), "c": np.array([-3.5, 1.0]), "d": np.array([1.0, 0.0])}
for k_, d_ in directions.items():
    yk = np.array(jeux[k_], dtype=float)
    t_ = X3.dot(d_)
    assert np.all((t_ > 0) == (yk == 1))  # la direction sépare strictement les deux classes (ou classe unique)
    for c in (1, 10):
        V[f"Q03_{k_}_c{c}"] = logloss(X3, yk, c * d_)
    assert V[f"Q03_{k_}_c10"] < V[f"Q03_{k_}_c1"]
    # Le gradient ne s'annule jamais : sa composante sur d_ est strictement négative en tout point c d_.
    assert gradient(X3, yk, 10 * d_).dot(d_) < 0
dire(f"ML3-03 : jeu a, b_chapeau = {b3} ; jeux b, c, d : log-loss tend vers 0 sans l'atteindre.")

# ML3-04 : n = 10 000, prévalence 2 %, TVP 0,8, TFP 0,05.
n4, pos4 = 10000, 200
vp4 = pos4 * Fraction(8, 10)
fp4 = (n4 - pos4) * Fraction(1, 20)
fn4, vn4 = pos4 - vp4, (n4 - pos4) - fp4
V["Q04_vp"], V["Q04_fp"], V["Q04_fn"], V["Q04_vn"] = int(vp4), int(fp4), int(fn4), int(vn4)
fraction("Q04_prec", vp4 / (vp4 + fp4))
fraction("Q04_exact", (vp4 + vn4) / n4)
fraction("Q04_const", Fraction(n4 - pos4, n4))
fraction("Q04_f1", 2 * vp4 / (2 * vp4 + fp4 + fn4))
fraction("Q04_d_spec", 1 - Fraction(1, 20))
fraction("Q04_rap", Fraction(8, 10))
V["Q04_neg"] = n4 - pos4
V["Q04_pp"] = int(vp4 + fp4)
V["Q04_vp_vn"] = int(vp4 + vn4)
assert V["Q04_exact"] < V["Q04_const"]

# ML3-05 : AUC avec un ex aequo.
POS5 = [Fraction(9, 10), Fraction(6, 10), Fraction(4, 10)]
NEG5 = [Fraction(7, 10), Fraction(4, 10), Fraction(3, 10), Fraction(1, 10)]
fraction("Q05_exact", auc_paires(POS5, NEG5))
assert auc_paires(POS5, NEG5) == auc_escalier(POS5, NEG5)
V["Q05_bien"] = sum(a > b for a in POS5 for b in NEG5)
V["Q05_egal"] = sum(a == b for a in POS5 for b in NEG5)
V["Q05_paires"] = len(POS5) * len(NEG5)
V["Q05_par_pos"] = [sum(a > b for b in NEG5) for a in POS5]
V["Q05_p1"], V["Q05_p2"], V["Q05_p3"] = V["Q05_par_pos"]
V["Q05_bien_egal"] = V["Q05_bien"] + V["Q05_egal"]
V["Q05_demi"] = V["Q05_bien"] + V["Q05_egal"] / 2
V["Q05_ex_vp"] = sum(s > Fraction(1, 2) for s in POS5)
V["Q05_ex_vn"] = sum(s <= Fraction(1, 2) for s in NEG5)
fraction("Q05_d_zero", Fraction(V["Q05_bien"], 12))
fraction("Q05_d_un", Fraction(V["Q05_bien"] + V["Q05_egal"], 12))
pred5 = [s > Fraction(1, 2) for s in POS5] + [s > Fraction(1, 2) for s in NEG5]
vrai5 = [True] * 3 + [False] * 4
fraction("Q05_d_exactitude", Fraction(sum(p == v for p, v in zip(pred5, vrai5)), 7))
V["Q05_somme_num"] = V["Q05_bien"] * 2 + V["Q05_egal"]
assert len({V["Q05_exact"], V["Q05_d_zero"], V["Q05_d_un"], V["Q05_d_exactitude"]}) == 4

# ML3-06 : deux groupes de même taille, probabilités vraies 0,1 et 0,5 ; modèle A (vraies probabilités), B (0,3 partout).
q1, q2 = Fraction(1, 10), Fraction(1, 2)
fraction("Q06_brier_A", (q1 * (1 - q1) + q2 * (1 - q2)) / 2)
fraction("Q06_brier_B", ((Fraction(3, 10) - q1) ** 2 + q1 * (1 - q1) + (Fraction(3, 10) - q2) ** 2 + q2 * (1 - q2)) / 2)
fraction("Q06_calib_B", (q1 + q2) / 2)
assert (q1 + q2) / 2 == Fraction(3, 10)
pp2 = q2 / (q1 + q2)  # P(positif vient du groupe 2)
pn1 = (1 - q1) / ((1 - q1) + (1 - q2))  # P(négatif vient du groupe 1)
auc_A = pp2 * pn1 + Fraction(1, 2) * (pp2 * (1 - pn1) + (1 - pp2) * pn1)
fraction("Q06_auc_A", auc_A)
fraction("Q06_pp2", pp2)
fraction("Q06_pn1", pn1)
fraction("Q06_pp1", 1 - pp2)
fraction("Q06_pn2", 1 - pn1)
fraction("Q06_d_brier_moy", Fraction(3, 10) * Fraction(7, 10))
# Simulation : 400 000 observations.
grp = rng.uniform(size=N_MC) < 0.5
q_vrai = np.where(grp, 0.5, 0.1)
Y6 = (rng.uniform(size=N_MC) < q_vrai).astype(float)
mc_moyenne("Q06_brier_A", V["Q06_brier_A"], (q_vrai - Y6) ** 2)
mc_moyenne("Q06_brier_B", V["Q06_brier_B"], (0.3 - Y6) ** 2)
# AUC de A par simulation : paires tirées au hasard (un positif, un négatif), ex aequo comptés 1/2.
ip = rng.choice(np.flatnonzero(Y6 == 1), 200000)
iN = rng.choice(np.flatnonzero(Y6 == 0), 200000)
conc = (q_vrai[ip] > q_vrai[iN]) + 0.5 * (q_vrai[ip] == q_vrai[iN])
mc_moyenne("Q06_auc_A", float(auc_A), conc)
mc_moyenne("Q06_calib_B", 0.3, Y6)

# ML3-07 : coordonnées gaussiennes N(0, 1) indépendantes, p = 50.
p7 = 50
x_, xp = sp.symbols("x xp", real=True)
# X - X' suit N(0, 2) ; moments d'une N(0, 2) : E[Z^2] = 2, E[Z^4] = 3 * 2^2 = 12.
z_ = sp.symbols("z", real=True)
dens = sp.exp(-z_ ** 2 / 4) / sp.sqrt(4 * sp.pi)
mz2 = sp.integrate(z_ ** 2 * dens, (z_, -sp.oo, sp.oo))
mz4 = sp.integrate(z_ ** 4 * dens, (z_, -sp.oo, sp.oo))
assert mz2 == 2 and mz4 == 12
V["Q07_m2"], V["Q07_m4"], V["Q07_var1"] = 2, 12, int(mz4 - mz2 ** 2)
V["Q07_m2sq"] = int(mz2 ** 2)
V["Q07_esp"] = 2 * p7
V["Q07_var"] = 8 * p7
fraction("Q07_ratio2", Fraction(8 * p7, (2 * p7) ** 2))
V["Q07_ratio"] = math.sqrt(8 * p7) / (2 * p7)
fraction("Q07_borne", Fraction(8 * p7, (2 * p7) ** 2) / Fraction(1, 4))
V["Q07_d_var2_ratio"] = math.sqrt(2 * p7) / (2 * p7)
fraction("Q07_d_var2_borne", Fraction(2 * p7, (2 * p7) ** 2) / Fraction(1, 4))
fraction("Q07_d_carre_ratio", Fraction(8 * p7, (2 * p7) ** 2))
fraction("Q07_d_carre_borne", Fraction(8 * p7, (2 * p7) ** 2) ** 2 / Fraction(1, 4))
V["Q07_d_lin_borne"] = V["Q07_ratio"] / 0.5
assert len({(round(V["Q07_ratio"], 3), round(V["Q07_borne"], 3)), (round(V["Q07_d_var2_ratio"], 3), round(V["Q07_d_var2_borne"], 3)),
            (round(V["Q07_d_carre_ratio"], 3), round(V["Q07_d_carre_borne"], 4)), (round(V["Q07_ratio"], 3), round(V["Q07_d_lin_borne"], 3))}) == 4
D7 = ((rng.normal(size=(100000, p7)) - rng.normal(size=(100000, p7))) ** 2).sum(axis=1)
mc_moyenne("Q07_esp", 2 * p7, D7)
var7 = D7.var(ddof=1)
m4c7 = np.mean((D7 - D7.mean()) ** 4)
controle_mc("Q07_var", 8 * p7, var7, math.sqrt((m4c7 - var7 ** 2) / len(D7)))
V["Q07_freq"] = float(np.mean(np.abs(D7 / (2 * p7) - 1) >= 0.5))
assert V["Q07_freq"] <= V["Q07_borne"]

# ML3-08 : 1500 dates, test 600..899, horizon 10, fenêtre des variables 60.
pu8, em8, ap8 = purge_embargo(1500, 600, 899, 10, 60)
V["Q08_exact"] = len(ap8)
V["Q08_n_purge"], V["Q08_n_emb"] = len(pu8), len(em8)
V["Q08_purge_av"] = [min(t for t in pu8 if t < 600), max(t for t in pu8 if t < 600)]
V["Q08_purge_ap"] = [min(t for t in pu8 if t > 899), max(t for t in pu8 if t > 899)]
V["Q08_emb"] = [min(em8), max(em8)]
(V["Q08_pav1"], V["Q08_pav2"]), (V["Q08_pap1"], V["Q08_pap2"]), (V["Q08_emb1"], V["Q08_emb2"]) = V["Q08_purge_av"], V["Q08_purge_ap"], V["Q08_emb"]
V["Q08_r1"], V["Q08_r2"] = 600 + 1, 899 + 10  # premier et dernier rendements des étiquettes de test
V["Q08_n_pav"] = sum(t < 600 for t in pu8)
V["Q08_n_pap"] = sum(t > 899 for t in pu8)
V["Q08_t0"] = 600
V["Q08_avant_purge"] = V["Q08_pav1"] - 1
assert V["Q08_emb2"] == V["Q08_r2"] + 59 and V["Q08_pav1"] + 10 == V["Q08_r1"]
V["Q08_sans"] = 1500 - 300
V["Q08_d_purge_seule"] = 1500 - 300 - len(pu8)
V["Q08_d_purge_h"] = 1500 - 300 - 2 * 10 - len(em8)
V["Q08_d_emb_deux"] = 1500 - 300 - len(pu8) - 2 * len(em8)
assert len({V["Q08_exact"], V["Q08_d_purge_seule"], V["Q08_d_purge_h"], V["Q08_d_emb_deux"]}) == 4
dire(f"ML3-08 : purgées {len(pu8)}, embargo {len(em8)} ({V['Q08_emb']}), apprentissage {len(ap8)}.")

Path(__file__).with_name("seance_ml_3.out.json").write_text(json.dumps(V, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
dire(f"{len(V)} valeurs exportées dans verif/seance_ml_3.out.json.")
dire("Toutes les vérifications passent.")
Path(__file__).with_name("seance_ml_3.out.txt").write_text("\n".join(LIGNES) + "\n", encoding="utf-8", newline="\n")
