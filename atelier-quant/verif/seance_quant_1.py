"""Vérification de tous les nombres de la séance quantitative 1 (leçon L-QUANT-1 et questions Q1-01 à Q1-08).

Principe :
  1. chaque valeur citée est calculée ici, en fractions exactes quand c'est possible ;
  2. chaque formule démontrée dans la leçon est contrôlée sur des cas numériques (recherche exhaustive,
     comparaison à une optimisation numérique, identités exactes) ;
  3. chaque probabilité ou espérance est aussi estimée par simulation Monte-Carlo, dont l'écart à l'exact
     doit rester inférieur à 4 erreurs standard ; une formule seulement asymptotique (erreur-type du ratio
     de Sharpe) est comparée à la simulation avec une tolérance relative annoncée ;
  4. les affirmations qualitatives sur l'expérience de double descente (pic au seuil d'interpolation,
     erreur d'entraînement nulle au-delà) sont des assertions sur la même expérience que celle de la leçon.

Graine fixe : la sortie est identique à chaque exécution.
Sorties : verif/seance_quant_1.out.json (lu par outils/construire.py) et verif/seance_quant_1.out.txt.
"""
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, optimize, stats

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
    dire(f"{nom:26s} exact={float(exact):.6f}  MC={float(estimation):.6f}  se={float(erreur_std):.6f}  z={z:+.2f}")


def mc_moyenne(nom, exact, echantillon):
    echantillon = np.asarray(echantillon, dtype=float)
    controle_mc(nom, exact, echantillon.mean(), echantillon.std(ddof=1) / np.sqrt(len(echantillon)))


def mc_variance(nom, exact, echantillon):
    """Variance d'un échantillon gaussien : erreur standard de la variance empirique, var * sqrt(2 / (R - 1))."""
    echantillon = np.asarray(echantillon, dtype=float)
    v = echantillon.var(ddof=1)
    controle_mc(nom, exact, v, float(exact) * math.sqrt(2 / (len(echantillon) - 1)))


# =====================================================================================
# PARTIE 1 : AIC et BIC
# =====================================================================================
# Seuil où la pénalité par paramètre du BIC (ln n) dépasse celle de l'AIC (2).
V["L_e2"] = math.exp(2)
V["L_n_seuil"] = min(n for n in range(1, 100) if math.log(n) > 2)
assert math.log(V["L_n_seuil"] - 1) < 2 < math.log(V["L_n_seuil"])
V["L_ln100"] = math.log(100)
V["L_ln1000"] = math.log(1000)
dire(f"BIC plus sévère que AIC dès n = {V['L_n_seuil']} (e^2 = {V['L_e2']:.4f}).")

# Optimisme de l'erreur d'entraînement : E[RSS test] - E[RSS entraînement] = 2 k sigma^2 (plan fixe, modèle juste).
n_o, k_o, s2_o, R_o = 30, 4, 1.0, 40000
Xo = rng.standard_normal((n_o, k_o))
Ho = Xo @ np.linalg.solve(Xo.T @ Xo, Xo.T)
assert abs(np.trace(Ho) - k_o) < 1e-10
mu_o = Xo @ np.array([1.0, -2.0, 0.5, 3.0])
E = rng.standard_normal((R_o, n_o))
Enew = rng.standard_normal((R_o, n_o))
Yo = mu_o + E
ajuste = Yo @ Ho                     # H symétrique : (H y) en ligne
rss_entr = ((Yo - ajuste) ** 2).sum(axis=1)
rss_test = ((mu_o + Enew - ajuste) ** 2).sum(axis=1)
V["L_opt_n"], V["L_opt_k"] = n_o, k_o
V["L_opt_exact"] = 2 * k_o * s2_o
mc_moyenne("L_opt", V["L_opt_exact"], rss_test - rss_entr)
mc_moyenne("L_rss_entr", (n_o - k_o) * s2_o, rss_entr)
mc_moyenne("L_rss_test", (n_o + k_o) * s2_o, rss_test)

# Probabilité que l'AIC (resp. le BIC) ajoute une variable inutile : P(chi2_1 > 2), P(chi2_1 > ln n).
V["L_p_aic"] = math.erfc(1.0)                     # P(|Z| > sqrt 2) = erfc(1)
assert abs(V["L_p_aic"] - stats.chi2.sf(2, 1)) < 1e-12
for n_b in [100, 1000]:
    V[f"L_p_bic_{n_b}"] = math.erfc(math.sqrt(math.log(n_b) / 2))
    assert abs(V[f"L_p_bic_{n_b}"] - stats.chi2.sf(math.log(n_b), 1)) < 1e-12
# Simulation directe : régression à plan fixe, sigma^2 = 1 connu, une variable de plus sans effet.
n_s, R_s = 100, 40000
X0 = np.column_stack([np.ones(n_s), rng.standard_normal(n_s)])
X1 = np.column_stack([X0, rng.standard_normal(n_s)])
H0 = X0 @ np.linalg.solve(X0.T @ X0, X0.T)
H1 = X1 @ np.linalg.solve(X1.T @ X1, X1.T)
Ys = X0 @ np.array([1.0, 2.0]) + rng.standard_normal((R_s, n_s))
deux_delta_ell = ((Ys - Ys @ H0) ** 2).sum(axis=1) - ((Ys - Ys @ H1) ** 2).sum(axis=1)   # = 2 (l1 - l0) avec sigma^2 = 1
assert deux_delta_ell.min() > -1e-9
mc_moyenne("L_p_aic", V["L_p_aic"], deux_delta_ell > 2)
mc_moyenne("L_p_bic_100", V["L_p_bic_100"], deux_delta_ell > math.log(n_s))
mc_moyenne("L_chi2_moyenne", 1.0, deux_delta_ell)

# Q1-01 : n = 200, le modèle B a 3 paramètres de plus et une log-vraisemblance maximale plus grande de 5.
n1, dk1, dl1 = 200, 3, 5
V["Q01_n"], V["Q01_dk"], V["Q01_dl"] = n1, dk1, dl1
V["Q01_pen_aic"] = 2 * dk1
V["Q01_pen_bic"] = dk1 * math.log(n1)
V["Q01_ln_n"] = math.log(n1)
V["Q01_gain"] = 2 * dl1
V["Q01_daic"] = 2 * dk1 - 2 * dl1          # AIC_B - AIC_A
V["Q01_dbic"] = dk1 * math.log(n1) - 2 * dl1
assert V["Q01_daic"] < 0 < V["Q01_dbic"]
dire(f"Q1-01 : AIC_B - AIC_A = {V['Q01_daic']}, BIC_B - BIC_A = {V['Q01_dbic']:.4f}.")

# =====================================================================================
# PARTIE 2 : double descente (même expérience que la leçon, assertions qualitatives)
# =====================================================================================
def double_descente(profil):
    g = np.random.default_rng(0)
    n, P, sigma, R = 40, 200, 0.5, 100
    j = np.arange(1, P + 1)
    beta = 1.0 / j if profil == "concentre" else np.ones(P)
    beta = beta / np.linalg.norm(beta)
    ps = [2, 5, 10, 20, 30, 38, 40, 42, 50, 80, 120, 200]
    Xt = g.standard_normal((5000, P))
    yt = Xt @ beta + sigma * g.standard_normal(5000)
    test = {p: [] for p in ps}
    entr = {p: [] for p in ps}
    smin = {p: [] for p in ps}
    for _ in range(R):
        X = g.standard_normal((n, P))
        y = X @ beta + sigma * g.standard_normal(n)
        for p in ps:
            b = np.linalg.lstsq(X[:, :p], y, rcond=None)[0]
            test[p].append(np.mean((Xt[:, :p] @ b - yt) ** 2))
            entr[p].append(np.mean((X[:, :p] @ b - y) ** 2))
            smin[p].append(np.linalg.svd(X[:, :p], compute_uv=False).min())
    med = {p: float(np.median(test[p])) for p in ps}
    return n, ps, med, {p: float(np.max(entr[p])) for p in ps}, {p: float(np.median(smin[p])) for p in ps}


for profil in ["concentre", "reparti"]:
    n_dd, ps, med, entr_max, smin = double_descente(profil)
    pic = max(ps, key=lambda p: med[p])
    assert pic == n_dd, (profil, pic)
    assert all(entr_max[p] < 1e-20 for p in ps if p >= n_dd)
    assert all(entr_max[p] > 1e-6 for p in ps if p < n_dd)
    assert min(ps, key=lambda p: smin[p]) == n_dd
    sous = [p for p in ps if p < n_dd]
    sur = [p for p in ps if p > n_dd]
    meilleur_sous, meilleur_sur = min(sous, key=lambda p: med[p]), min(sur, key=lambda p: med[p])
    # Après le pic, l'erreur décroît (seconde descente) jusqu'au meilleur modèle sur-paramétré.
    apres = [p for p in sur if p <= meilleur_sur]
    assert all(med[a] > med[b] for a, b in zip(apres, apres[1:]))
    if profil == "concentre":
        assert med[meilleur_sous] < med[meilleur_sur]       # le premier minimum reste le meilleur
        assert meilleur_sous not in (sous[0], sous[-1])     # vraie première descente puis remontée
    else:
        assert med[meilleur_sur] < med[meilleur_sous]       # le sur-paramétré bat tout sous-paramétré
    dire(f"Double descente ({profil}) : pic en p = {pic}, meilleur sous-paramétré p = {meilleur_sous} "
         f"({med[meilleur_sous]:.3f}), meilleur sur-paramétré p = {meilleur_sur} ({med[meilleur_sur]:.3f}).")
V["L_dd_n"] = 40

# Bruit amplifié : sigma^2 tr((X^T X)^-1) = sigma^2 somme 1/s_j^2 ; norme minimale de la pseudo-inverse.
for (nn, pp) in [(40, 20), (40, 39), (40, 60)]:
    X = rng.standard_normal((nn, pp))
    s = np.linalg.svd(X, compute_uv=False)
    Xp = np.linalg.pinv(X)
    assert np.isclose(np.trace(Xp @ Xp.T), np.sum(1 / s**2))
    y = rng.standard_normal(nn)
    b = Xp @ y
    if pp > nn:   # interpolation : X b = y, et b est de norme minimale parmi les interpolateurs
        assert np.allclose(X @ b, y)
        noyau = np.linalg.svd(X)[2][nn:].T
        h = noyau @ rng.standard_normal(pp - nn)
        assert np.isclose(np.linalg.norm(b + h) ** 2, np.linalg.norm(b) ** 2 + np.linalg.norm(h) ** 2)
        assert np.allclose(b, np.linalg.lstsq(X, y, rcond=None)[0])
dire("Formule du bruit sigma^2 somme 1/s_j^2, interpolation et norme minimale vérifiées.")

# =====================================================================================
# PARTIE 3 : importance par permutation d'un modèle linéaire, variables dupliquées
# =====================================================================================
# Importance (hausse d'erreur quadratique quand X_j est remplacée par une copie indépendante) = 2 b_j^2 Var(X_j).
V["Q02_exact"] = 2 * 0.5**2 * 1.0
V["Q02_A"] = 2 * 1.0**2 * 1.0
V["Q02_d_moitie"] = V["Q02_A"] / 2
V["Q02_d_sans2"] = 0.5**2 * 1.0
n_p = 400000
X1p = rng.standard_normal(n_p)
eps = rng.standard_normal(n_p)
Yp = X1p + eps
copie = rng.permutation(X1p)
base_B = (Yp - (0.5 * X1p + 0.5 * X1p)) ** 2
perm_B = (Yp - (0.5 * copie + 0.5 * X1p)) ** 2
perm_A = (Yp - copie) ** 2
mc_moyenne("Q02_exact", V["Q02_exact"], perm_B - base_B)
mc_moyenne("Q02_A", V["Q02_A"], perm_A - (Yp - X1p) ** 2)

# =====================================================================================
# PARTIE 4 : valeurs de Shapley
# =====================================================================================
def shapley(v, p):
    """Valeurs de Shapley exactes par la formule des coalitions ; v : fonction d'un frozenset."""
    phi = []
    for j in range(p):
        autres = [k for k in range(p) if k != j]
        total = Fraction(0)
        for t in range(p):
            for S in itertools.combinations(autres, t):
                S = frozenset(S)
                poids = Fraction(math.factorial(t) * math.factorial(p - t - 1), math.factorial(p))
                total += poids * (v(S | {j}) - v(S))
        phi.append(total)
    return phi


# Lemme : la somme des poids sur les coalitions qui ne contiennent pas j vaut 1.
for p in range(1, 9):
    assert sum(math.comb(p - 1, t) * Fraction(math.factorial(t) * math.factorial(p - t - 1), math.factorial(p)) for t in range(p)) == 1
dire("Somme des poids de Shapley égale à 1 pour p = 1 à 8.")

# Exemple linéaire : f = 1 + 2 x1 - 3 x2 + 0,5 x3, E[X] = (1, 0, 2), x = (2, 1, 2), variables indépendantes.
b_lin = [Fraction(2), Fraction(-3), Fraction(1, 2)]
mu_lin = [Fraction(1), Fraction(0), Fraction(2)]
x_lin = [Fraction(2), Fraction(1), Fraction(2)]
v_lin = lambda S: 1 + sum(b_lin[k] * (x_lin[k] if k in S else mu_lin[k]) for k in range(3))
phi_lin = shapley(v_lin, 3)
assert phi_lin == [b_lin[k] * (x_lin[k] - mu_lin[k]) for k in range(3)]
V["L_sh_phi"] = [float(f) for f in phi_lin]
V["L_sh_fx"] = float(v_lin(frozenset({0, 1, 2})))
V["L_sh_ef"] = float(v_lin(frozenset()))
V["L_sh_somme"] = float(sum(phi_lin))
assert sum(phi_lin) == v_lin(frozenset({0, 1, 2})) - v_lin(frozenset())
for k in range(3):
    V[f"L_sh_phi{k + 1}"] = float(phi_lin[k])

# Piège : X1 = X2 (corrélation parfaite), f = x1, valeur conditionnelle E[f(X) | X_S = x_S], x1 = 3, E[X1] = 1.
mu_c, x_c = Fraction(1), Fraction(3)
v_cond = lambda S: x_c if S else mu_c        # connaître X1 ou X2 revient à connaître X1
v_interv = lambda S: x_c if 0 in S else mu_c  # valeur interventionnelle : X2 n'entre pas dans f
phi_cond, phi_interv = shapley(v_cond, 2), shapley(v_interv, 2)
V["L_sh_cond"] = [float(f) for f in phi_cond]
V["L_sh_interv"] = [float(f) for f in phi_interv]
assert phi_cond == [Fraction(1), Fraction(1)] and phi_interv == [Fraction(2), Fraction(0)]

# Q1-03 : f = x1 x2, X1 et X2 indépendantes d'espérances 1 et 2, point x = (3, 4).
m1, m2, a1, a2 = Fraction(1), Fraction(2), Fraction(3), Fraction(4)
v3 = lambda S: (a1 if 0 in S else m1) * (a2 if 1 in S else m2)
phi3 = shapley(v3, 2)
assert phi3[0] == (a1 - m1) * (m2 + a2) / 2 and phi3[1] == (a2 - m2) * (m1 + a1) / 2
V["Q03_exact"] = float(phi3[0])
V["Q03_phi2"] = float(phi3[1])
V["Q03_fx"], V["Q03_ef"] = float(a1 * a2), float(m1 * m2)
V["Q03_diff"] = float(a1 * a2 - m1 * m2)
V["Q03_d_dernier"] = float(v3(frozenset({0, 1})) - v3(frozenset({1})))   # j ajouté en dernier
V["Q03_d_premier"] = float(v3(frozenset({0})) - v3(frozenset()))         # j ajouté en premier
V["Q03_d_moitie"] = float((a1 * a2 - m1 * m2) / 2)
V["Q03_v1"] = float(v3(frozenset({0})))
V["Q03_v2"] = float(v3(frozenset({1})))
V["Q03_c1"] = V["Q03_d_premier"]
V["Q03_c2"] = V["Q03_d_dernier"]
# Simulation : Shapley = espérance de la contribution marginale sur un ordre d'arrivée uniforme.
contrib = []
for _ in range(20000):
    ordre = rng.permutation(2)
    avant = frozenset(ordre[: list(ordre).index(0)])
    contrib.append(float(v3(avant | {0}) - v3(avant)))
mc_moyenne("Q03_exact", V["Q03_exact"], contrib)

# =====================================================================================
# PARTIE 5 : coefficient d'information
# =====================================================================================
def rangs(v):
    return stats.rankdata(v)


# Formule 1 - 6 somme d^2 / (N (N^2 - 1)) contre corrélation de Pearson des rangs, sur des permutations.
for N in range(2, 9):
    for _ in range(50):
        a, b = rng.permutation(N) + 1, rng.permutation(N) + 1
        d2 = int(((a - b) ** 2).sum())
        rho = 1 - Fraction(6 * d2, N * (N * N - 1))
        assert abs(float(rho) - np.corrcoef(a, b)[0, 1]) < 1e-12
dire("Formule de Spearman sans égalités vérifiée sur 350 couples de permutations.")

# Q1-04 : six actifs.
sig4 = [0.8, -0.3, 1.5, 0.1, -1.2, 0.6]
ren4 = [1.2, -0.5, 0.9, 0.4, -0.8, -0.2]
ra, rb = rangs(sig4), rangs(ren4)
V["Q04_rangs_signal"] = [int(x) for x in ra]
V["Q04_rangs_rend"] = [int(x) for x in rb]
d2_4 = int(((ra - rb) ** 2).sum())
N4 = 6
V["Q04_d2"] = d2_4
V["Q04_N3"] = N4 * (N4 * N4 - 1)
fraction("Q04_exact", 1 - Fraction(6 * d2_4, N4 * (N4 * N4 - 1)))
assert abs(V["Q04_exact"] - stats.spearmanr(sig4, ren4)[0]) < 1e-12
V["Q04_d_pearson"] = float(np.corrcoef(sig4, ren4)[0, 1])
V["Q04_d_sans6"] = float(1 - Fraction(d2_4, N4 * (N4 * N4 - 1)))
V["Q04_d_cube"] = float(1 - Fraction(6 * d2_4, N4**3))
dire(f"Q1-04 : rangs {V['Q04_rangs_signal']} et {V['Q04_rangs_rend']}, somme d^2 = {d2_4}, IC = {V['Q04_exact']:.6f}.")

# Sous l'hypothèse nulle : E[IC] = 0 et Var(IC) = 1/(N - 1) exactement (rangs sans égalités).
for N in [3, 4, 5]:
    tous = [np.corrcoef(np.arange(N), q)[0, 1] for q in itertools.permutations(range(N))]
    assert abs(np.mean(tous)) < 1e-12 and abs(np.var(tous) - 1 / (N - 1)) < 1e-12
dire("Var(IC) = 1/(N - 1) sous l'hypothèse nulle, vérifiée par énumération pour N = 3, 4, 5.")
N_ic = 10
ics = np.array([np.corrcoef(np.arange(N_ic), rng.permutation(N_ic))[0, 1] for _ in range(40000)])
mc_moyenne("L_ic_moy0", 0.0, ics)
controle_mc("L_ic_var10", 1 / (N_ic - 1), ics.var(ddof=1), np.sqrt(np.var((ics - ics.mean()) ** 2, ddof=1) / len(ics)))
V["L_ic_se500"] = 1 / math.sqrt(499)
V["L_ic_se500_250"] = 1 / math.sqrt(499 * 250)

# Loi fondamentale, version démontrée : N paris gaussiens indépendants, position = signal.
c_lf, N_lf = 0.05, 100
V["L_lf_c"], V["L_lf_N"] = c_lf, N_lf
V["L_lf_ir"] = c_lf * math.sqrt(N_lf) / math.sqrt(1 + c_lf**2)
V["L_lf_approx"] = c_lf * math.sqrt(N_lf)
V["L_lf_ir_an"] = V["L_lf_ir"] * math.sqrt(12)
R_lf = 100000
G = rng.standard_normal((R_lf, N_lf))
Ebr = rng.standard_normal((R_lf, N_lf))
rend = c_lf * G + math.sqrt(1 - c_lf**2) * Ebr          # omega = 1
pnl = (G * rend).sum(axis=1)
mc_moyenne("L_lf_moy", c_lf * N_lf, pnl)
mc_variance_ok = pnl.var(ddof=1)
se_var = np.sqrt(np.var((pnl - pnl.mean()) ** 2, ddof=1) / R_lf)
controle_mc("L_lf_var", N_lf * (1 + c_lf**2), mc_variance_ok, se_var)
V["L_lf_ir_mc"] = float(pnl.mean() / pnl.std(ddof=1))

# =====================================================================================
# PARTIE 6 : étiquettes qui se chevauchent
# =====================================================================================
def somme_c2(T, h):
    """Somme des carrés des nombres de fenêtres contenant chaque rendement quotidien."""
    c = np.zeros(T + h - 1, dtype=np.int64)
    for t in range(T):
        c[t : t + h] += 1
    s = int((c**2).sum())
    assert s == (T - h + 1) * h * h + 2 * sum(k * k for k in range(1, h)), (T, h)
    return s


for (T, h, nom) in [(1000, 5, "L_ch"), (1000, 20, "Q07")]:
    s = somme_c2(T, h)
    V[f"{nom}_T"], V[f"{nom}_h"] = T, h
    V[f"{nom}_somme_c2"] = s
    fraction(f"{nom}_rapport_var", Fraction(s, T * h))
    V[f"{nom}_rapport_se"] = math.sqrt(s / (T * h))
    V[f"{nom}_corr_voisins"] = (h - 1) / h
    # Simulation : rendements quotidiens N(0, 1), variance de la moyenne des étiquettes.
    R_ch = 20000
    moyennes = np.empty(R_ch)
    for bloc in range(0, R_ch, 2000):
        x = rng.standard_normal((2000, T + h - 1))
        cs = np.concatenate([np.zeros((2000, 1)), np.cumsum(x, axis=1)], axis=1)
        etiq = cs[:, h : h + T] - cs[:, 0:T]           # étiquette t = somme des rendements t .. t + h - 1
        moyennes[bloc : bloc + 2000] = etiq.mean(axis=1)
    mc_variance(f"{nom}_var_moyenne", s / T**2, moyennes)
    V[f"{nom}_var_naive"] = h / T
    V[f"{nom}_var_moyenne_exact"] = s / T**2
V["Q07_d_corr"] = V["Q07_corr_voisins"]
V["Q07_racine_h"] = math.sqrt(V["Q07_h"])
V["Q07_cle"] = V["Q07_h"] ** 2 * (V["Q07_T"] - V["Q07_h"] + 1)
V["Q07_bords"] = 2 * sum(k * k for k in range(1, V["Q07_h"]))
V["Q07_Th"] = V["Q07_T"] * V["Q07_h"]
assert V["Q07_cle"] + V["Q07_bords"] == V["Q07_somme_c2"]

# =====================================================================================
# PARTIE 7 : non-stationnarité, rupture de moyenne
# =====================================================================================
n_ns, d_ns = 100, 2.0
V["L_ns_n"], V["L_ns_delta"] = n_ns, d_ns
fraction("L_ns_es2", 1 + Fraction(n_ns) * Fraction(4) / (4 * (n_ns - 1)))
Xns = rng.standard_normal((40000, n_ns))
Xns[:, n_ns // 2 :] += d_ns
mc_moyenne("L_ns_es2", V["L_ns_es2"], Xns.var(axis=1, ddof=1))

# =====================================================================================
# PARTIE 8 : biais de sélection, espérance du maximum de N normales
# =====================================================================================
def esp_max(N, loi=stats.norm):
    f = lambda x: x * N * loi.pdf(x) * loi.cdf(x) ** (N - 1)
    val, err = integrate.quad(f, -15, 15, limit=400, points=[-2, 0, 1, 2, 3, 4], epsabs=1e-13)
    assert err < 1e-7, err
    return val


def minorant(N):
    L = lambda a: a * (1 - stats.norm.cdf(a) ** N) - stats.norm.pdf(a) * stats.norm.cdf(a) ** (N - 1)
    res = optimize.minimize_scalar(lambda a: -L(a), bounds=(0, 6), method="bounded")
    return L(res.x), res.x


for N in [10, 50, 100, 1000]:
    e = esp_max(N)
    haut = math.sqrt(2 * math.log(N))
    bas, a_opt = minorant(N)
    assert bas < e < haut
    V[f"L_max_{N}"] = e
    V[f"L_max_haut_{N}"] = haut
    V[f"L_max_bas_{N}"] = bas
    V[f"L_max_a_{N}"] = a_opt
    tirages = np.max(rng.standard_normal((20000, N)), axis=1)
    mc_moyenne(f"L_max_{N}", e, tirages)
    dire(f"N = {N:4d} : minorant {bas:.4f} < E[max] {e:.4f} < majorant {haut:.4f}")

# Q1-05 : 50 stratégies sans effet, 4 ans de rendements quotidiens (n = 1008).
N5, T5, n5 = 50, 4, 1008
V["Q05_N"], V["Q05_T"] = N5, T5
V["Q05_emax"] = esp_max(N5)
V["Q05_exact"] = V["Q05_emax"] / math.sqrt(T5)
V["Q05_d_sans_T"] = V["Q05_emax"]
V["Q05_d_borne"] = math.sqrt(2 * math.log(N5)) / math.sqrt(T5)
V["Q05_borne_z"] = math.sqrt(2 * math.log(N5))
V["Q05_student"] = esp_max(N5, stats.t(df=n5 - 1)) / math.sqrt(T5)   # loi exacte : Student à n - 1 ddl
assert abs(V["Q05_student"] - V["Q05_exact"]) < 2e-3
# Simulation directe de la sélection : 2000 répétitions de 50 stratégies sur 1008 jours.
meilleurs = []
for _ in range(2000):
    r = rng.standard_normal((N5, n5)) * 0.01
    sr = np.sqrt(252) * r.mean(axis=1) / r.std(axis=1, ddof=1)
    meilleurs.append(sr.max())
mc_moyenne("Q05_student", V["Q05_student"], meilleurs)

# =====================================================================================
# PARTIE 9 : erreur-type du ratio de Sharpe (Lo, 2002), rendements i.i.d. normaux
# =====================================================================================
def se_sharpe_annuel(sr_an, q, T):
    return math.sqrt((1 + sr_an**2 / (2 * q)) / T)


def simule_sharpe(sr_periode, n, R, loi="normale"):
    out = []
    for bloc in range(0, R, 5000):
        m = min(5000, R - bloc)
        if loi == "normale":
            x = sr_periode + rng.standard_normal((m, n))
        else:   # exponentielle recentrée : asymétrie 2, kurtosis 9, écart-type 1
            x = sr_periode + rng.exponential(1.0, (m, n)) - 1.0
        out.append(x.mean(axis=1) / x.std(axis=1, ddof=1))
    return np.concatenate(out)


# Vérification sympy de la méthode delta : gradient de mu / sqrt(s2).
mu_s, s2_s, g3, g4 = sp.symbols("mu s2 g3 g4", positive=True)
gfun = mu_s / sp.sqrt(s2_s)
grad = [sp.diff(gfun, mu_s), sp.diff(gfun, s2_s)]
cov = sp.Matrix([[s2_s, g3 * s2_s ** sp.Rational(3, 2)], [g3 * s2_s ** sp.Rational(3, 2), (g4 - 1) * s2_s**2]])
var_delta = sp.simplify((sp.Matrix([grad]) * cov * sp.Matrix(grad))[0])
sr_sym = mu_s / sp.sqrt(s2_s)
assert sp.simplify(var_delta - (1 - g3 * sr_sym + (g4 - 1) / 4 * sr_sym**2)) == 0
assert sp.simplify(var_delta.subs({g3: 0, g4: 3}) - (1 + sr_sym**2 / 2)) == 0
dire("Méthode delta (sympy) : variance asymptotique 1 - g3 SR + (g4 - 1) SR^2 / 4 ; cas normal 1 + SR^2 / 2.")

# Exemple de la leçon : quotidien, 4 ans, Sharpe annualisé 1.
V["L_sr_se_4ans"] = se_sharpe_annuel(1.0, 252, 4)
V["L_sr_se_4ans_simple"] = 1 / math.sqrt(4)
sim = simule_sharpe(1 / math.sqrt(252), 1008, 20000) * math.sqrt(252)
V["L_sr_se_4ans_mc"] = float(sim.std(ddof=1))
assert abs(sim.std(ddof=1) / V["L_sr_se_4ans"] - 1) < 0.02
# Contrôle asymptotique à grand n, Sharpe par période élevé : tolérance relative 2 %.
for sr_p, nn in [(0.5, 2000)]:
    sim = simule_sharpe(sr_p, nn, 20000)
    formule = math.sqrt((1 + sr_p**2 / 2) / nn)
    assert abs(sim.std(ddof=1) / formule - 1) < 0.02, (sim.std(ddof=1), formule)
    V["L_sr_asym_rel"] = float(sim.std(ddof=1) / formule - 1)
    sim = simule_sharpe(sr_p, nn, 20000, loi="expo")
    formule_g = math.sqrt((1 - 2 * sr_p + (9 - 1) / 4 * sr_p**2) / nn)
    assert abs(sim.std(ddof=1) / formule_g - 1) < 0.03, (sim.std(ddof=1), formule_g)
    V["L_sr_asym_expo_rel"] = float(sim.std(ddof=1) / formule_g - 1)
dire(f"Sharpe : écart relatif simulation / formule, cas normal {V['L_sr_asym_rel']:+.4f}, cas exponentiel {V['L_sr_asym_expo_rel']:+.4f}.")

# Q1-06 : mensuel, 10 ans, Sharpe annualisé 1,2.
q6, T6, sr6 = 12, 10, 1.2
V["Q06_exact"] = se_sharpe_annuel(sr6, q6, T6)
V["Q06_terme"] = sr6**2 / (2 * q6)
V["Q06_d_sans_sr"] = 1 / math.sqrt(T6)
V["Q06_d_annuel"] = math.sqrt((1 + sr6**2 / 2) / T6)
V["Q06_d_periode"] = math.sqrt((1 + (sr6 / math.sqrt(q6)) ** 2 / 2) / (q6 * T6))
sim = simule_sharpe(sr6 / math.sqrt(q6), q6 * T6, 40000) * math.sqrt(q6)
V["Q06_mc"] = float(sim.std(ddof=1))
assert abs(V["Q06_mc"] / V["Q06_exact"] - 1) < 0.03
dire(f"Q1-06 : formule {V['Q06_exact']:.4f}, simulation (n = 120) {V['Q06_mc']:.4f}.")

# =====================================================================================
# PARTIE 10 : optimisation (Taylor, Kelly, portefeuille de variance minimale)
# =====================================================================================
# Croissance logarithmique : R = +0,2 avec probabilité 0,4, -0,1 avec probabilité 0,6.
w = sp.symbols("w", real=True)
gk = sp.Rational(4, 10) * sp.log(1 + sp.Rational(2, 10) * w) + sp.Rational(6, 10) * sp.log(1 - sp.Rational(1, 10) * w)
w_exact = sp.solve(sp.diff(gk, w), w)
assert w_exact == [1]
assert all(sp.diff(gk, w, 2).subs(w, t) < 0 for t in [-4, 0, 1, 4.9])
mu_k = Fraction(4, 10) * Fraction(2, 10) + Fraction(6, 10) * Fraction(-1, 10)
m2_k = Fraction(4, 10) * Fraction(2, 10) ** 2 + Fraction(6, 10) * Fraction(1, 10) ** 2
fraction("L_k_mu", mu_k)
fraction("L_k_m2", m2_k)
fraction("L_k_approx", mu_k / m2_k)
V["L_k_exact"] = 1
V["L_k_g_exact"] = float(gk.subs(w, 1))
V["L_k_g_approx"] = float(gk.subs(w, sp.Rational(mu_k.numerator, mu_k.denominator) / sp.Rational(m2_k.numerator, m2_k.denominator)))
assert V["L_k_g_approx"] < V["L_k_g_exact"]
dire(f"Kelly : w exact 1, approximation d'ordre 2 {V['L_k_approx']:.6f}.")


def variance_minimale(Sigma):
    Sigma = sp.Matrix(Sigma)
    un = sp.ones(Sigma.shape[0], 1)
    s_inv_un = Sigma.inv() * un
    denom = (un.T * s_inv_un)[0]
    return s_inv_un / denom, 1 / denom


# Exemple de la leçon : trois actifs.
sig = [sp.Rational(1, 10), sp.Rational(15, 100), sp.Rational(2, 10)]
rho = {(0, 1): sp.Rational(3, 10), (0, 2): 0, (1, 2): sp.Rational(5, 10)}
Sigma3 = sp.zeros(3, 3)
for i in range(3):
    for j in range(3):
        r = 1 if i == j else rho[(min(i, j), max(i, j))]
        Sigma3[i, j] = r * sig[i] * sig[j]
w3, var3 = variance_minimale(Sigma3)
for i in range(3):
    fraction(f"L_pf_w{i + 1}", Fraction(int(sp.fraction(w3[i])[0]), int(sp.fraction(w3[i])[1])))
fraction("L_pf_var", Fraction(int(sp.fraction(var3)[0]), int(sp.fraction(var3)[1])))
V["L_pf_vol"] = math.sqrt(float(var3))
# Contrôle : minimisation numérique sous contrainte, et condition de Lagrange 2 Sigma w = lambda 1.
S3 = np.array(Sigma3.tolist(), dtype=float)
res = optimize.minimize(lambda v: v @ S3 @ v, np.ones(3) / 3, constraints=[{"type": "eq", "fun": lambda v: v.sum() - 1}], tol=1e-14)
assert np.allclose(res.x, np.array(w3.tolist(), dtype=float).ravel(), atol=1e-6)
grad3 = 2 * S3 @ np.array(w3.tolist(), dtype=float).ravel()
assert np.allclose(grad3, grad3[0])
V["L_pf_lambda"] = float(grad3[0])
assert abs(V["L_pf_lambda"] - 2 * float(var3)) < 1e-12
# Covariances de l'exemple, pour l'énoncé.
V["L_pf_cov"] = [[float(Sigma3[i, j]) for j in range(3)] for i in range(3)]
dire(f"Portefeuille de variance minimale (3 actifs) : w = {[round(float(x), 6) for x in w3]}, variance {float(var3):.6f}.")

# Q1-08 : deux actifs, sigma = (0,1 ; 0,2), corrélation 0,8.
s1, s2, r12 = sp.Rational(1, 10), sp.Rational(2, 10), sp.Rational(8, 10)
w8, var8 = variance_minimale([[s1**2, r12 * s1 * s2], [r12 * s1 * s2, s2**2]])
fraction("Q08_exact", Fraction(int(sp.fraction(w8[1])[0]), int(sp.fraction(w8[1])[1])))
fraction("Q08_w1", Fraction(int(sp.fraction(w8[0])[0]), int(sp.fraction(w8[0])[1])))
assert V["Q08_exact"] < 0
fraction("Q08_num", Fraction(1, 100) - Fraction(8, 10) * Fraction(1, 10) * Fraction(2, 10))
fraction("Q08_den", Fraction(1, 100) + Fraction(4, 100) - 2 * Fraction(8, 10) * Fraction(1, 10) * Fraction(2, 10))
fraction("Q08_d_invvar", Fraction(1, 4) / (Fraction(1, 1) + Fraction(1, 4)))
fraction("Q08_d_invvol", Fraction(1, 2) / (Fraction(1, 1) + Fraction(1, 2)))
V["Q08_vol"] = math.sqrt(float(var8))
fraction("Q08_var", Fraction(int(sp.fraction(var8)[0]), int(sp.fraction(var8)[1])))
grille = np.linspace(-2, 2, 400001)
var_grille = grille**2 * 0.04 + (1 - grille) ** 2 * 0.01 + 2 * grille * (1 - grille) * 0.016
assert abs(grille[np.argmin(var_grille)] - V["Q08_exact"]) < 2e-5
V["Q08_var_invvar"] = 0.2**2 * 0.04 + 0.8**2 * 0.01 + 2 * 0.2 * 0.8 * 0.016
assert V["Q08_var_invvar"] > float(var8)
dire(f"Q1-08 : w2 = {V['Q08_exact']:.6f}, w1 = {V['Q08_w1']:.6f}, volatilité minimale {V['Q08_vol']:.6f}.")

# Point de contrôle : rupture de moyenne, n = 50, Delta = 1.
fraction("C_ns", 1 + Fraction(50) * 1 / (4 * 49))
fraction("C_ns_sans_n", 1 + Fraction(1, 4))
mc_moyenne("C_ns", V["C_ns"], np.concatenate([rng.standard_normal((40000, 25)), 1 + rng.standard_normal((40000, 25))], axis=1).var(axis=1, ddof=1))

# Point de contrôle : seuils de pénalité (ln n > 2 pour le BIC contre l'AIC ; ln n > 1, distracteur).
V["C_n_ln1"] = min(n for n in range(1, 100) if math.log(n) > 1)

# Point de contrôle : Shapley linéaire, f = 3 + 2 x1 - x2, E[X] = (0, 1), x = (1, 3), variables indépendantes.
b_c, mu_c2, x_c2 = [Fraction(2), Fraction(-1)], [Fraction(0), Fraction(1)], [Fraction(1), Fraction(3)]
phi_c = shapley(lambda S: 3 + sum(b_c[k] * (x_c2[k] if k in S else mu_c2[k]) for k in range(2)), 2)
V["C_sh_phi2"] = float(phi_c[1])
V["C_sh_phi1"] = float(phi_c[0])
V["C_sh_d_sans_mu"] = float(b_c[1] * x_c2[1])
V["C_sh_d_coef"] = float(b_c[1])
assert phi_c[1] == b_c[1] * (x_c2[1] - mu_c2[1])

# Exemple de la leçon, coefficient d'information sur cinq actifs.
sig_l = [0.3, -1.0, 2.1, 0.7, -0.2]
ren_l = [0.5, -1.1, 0.2, 1.4, -0.3]
ra_l, rb_l = rangs(sig_l), rangs(ren_l)
V["L_ic_rangs_s"] = [int(x) for x in ra_l]
V["L_ic_rangs_r"] = [int(x) for x in rb_l]
V["L_ic_d"] = [int(x) for x in ra_l - rb_l]
V["L_ic_d2"] = int(((ra_l - rb_l) ** 2).sum())
V["L_ic_N3"] = 5 * 24
fraction("L_ic_ex", 1 - Fraction(6 * V["L_ic_d2"], 5 * 24))
assert abs(V["L_ic_ex"] - stats.spearmanr(sig_l, ren_l)[0]) < 1e-12

# Exemple de la leçon, biais de sélection : m = 100 stratégies, T = 10 ans.
V["L_sel_ex"] = V["L_max_100"] / math.sqrt(10)
V["L_sel_haut"] = V["L_max_haut_100"] / math.sqrt(10)
V["Q08_ratio"] = 0.1 / 0.2

# Résultats intermédiaires cités dans les corrections.
V["Q04_d"] = [a - b for a, b in zip(V["Q04_rangs_signal"], V["Q04_rangs_rend"])]
V["Q04_6d2"] = 6 * V["Q04_d2"]
V["Q04_N_cube"] = 6**3
V["Q04_se_nulle"] = 1 / math.sqrt(5)
fraction("Q06_theta2", Fraction(144, 100) / 12)
fraction("Q06_var_an", 12 * (1 + Fraction(144, 100) / 12 / 2) / 120)
assert abs(math.sqrt(V["Q06_var_an"]) - V["Q06_exact"]) < 1e-12
V["Q07_nb_plein"] = V["Q07_T"] - V["Q07_h"] + 1
V["Q07_h2"] = V["Q07_h"] ** 2
fraction("Q08_s1c", Fraction(1, 100))
fraction("Q08_s2c", Fraction(4, 100))
fraction("Q08_cov", Fraction(8, 10) * Fraction(1, 10) * Fraction(2, 10))
fraction("Q08_somme", Fraction(5, 100))
fraction("Q08_2cov", 2 * Fraction(8, 10) * Fraction(1, 10) * Fraction(2, 10))

Path(__file__).with_name("seance_quant_1.out.json").write_text(json.dumps(V, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
dire(f"{len(V)} valeurs exportées dans verif/seance_quant_1.out.json.")
dire("Toutes les vérifications passent.")
Path(__file__).with_name("seance_quant_1.out.txt").write_text("\n".join(LIGNES) + "\n", encoding="utf-8", newline="\n")
