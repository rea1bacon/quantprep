"""Vérification de tous les nombres de la séance de mathématiques 4 (leçon L-MATH-4 et questions M4-01 à M4-08).

Thèmes : chaînes de Markov à états finis, martingales et théorème d'arrêt, processus AR(1), mouvement brownien,
lemme d'Itô et mouvement brownien géométrique.

Chaque probabilité ou espérance a deux vérifications indépendantes :
  1. un calcul exact (fractions, système linéaire résolu en rationnels par sympy, programmation dynamique exacte,
     intégrale ou série symbolique) ;
  2. une simulation Monte-Carlo qui suit les règles de l'énoncé (lancers, points, pas de la marche, chocs), dont
     l'écart à l'exact doit rester inférieur à 4 erreurs standard.
Les énoncés généraux de la leçon sont aussi contrôlés : inverse de I - Q égal à la série de Neumann, propriété de
martingale (calcul symbolique), formules de l'AR(1) par deux méthodes exactes, lemme d'Itô sur les exemples.
Les distracteurs et l'arithmétique citée dans les corrections sont calculés ici.

Graine fixe : la sortie est identique à chaque exécution.
Sorties : verif/seance_math_4.out.json (lu par outils/construire.py) et verif/seance_math_4.out.txt (ce résumé).
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


def vers_fraction(x):
    x = sp.Rational(x)
    return Fraction(int(x.p), int(x.q))


def fraction(nom, f):
    """Exporte une fraction : valeur décimale, numérateur et dénominateur (forme irréductible)."""
    f = vers_fraction(f) if isinstance(f, sp.Basic) else Fraction(f)
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


# =====================================================================================
# Outils pour les chaînes de Markov
# =====================================================================================
def stationnaire(P):
    """Loi stationnaire par résolution exacte de pi (P - I) = 0 et somme = 1 ; exige une solution unique."""
    k = P.shape[0]
    pis = sp.symbols(f"pi0:{k}")
    eqs = list(sp.Matrix([pis]) * (P - sp.eye(k))) + [sum(pis) - 1]
    sol = sp.solve(eqs, pis, dict=True)
    assert len(sol) == 1
    return [sol[0][s] for s in pis]


def absorption(Q, Rm):
    """N = (I - Q)^{-1}, d = N 1, H = N R, en rationnels ; contrôle de la série de Neumann en flottants."""
    m = Q.shape[0]
    assert (sp.eye(m) - Q).det() != 0
    N = (sp.eye(m) - Q).inv()
    # Série de Neumann : somme des Q^t, t < 3000, comparée à N.
    qf = np.array(Q.tolist(), dtype=float)
    somme, puissance = np.zeros((m, m)), np.eye(m)
    for _ in range(3000):
        somme += puissance
        puissance = puissance @ qf
    assert np.allclose(somme, np.array(N.tolist(), dtype=float), atol=1e-10)
    # Équations du premier pas, résolues indépendamment par sympy.solve.
    ds = sp.symbols(f"d0:{m}")
    sol = sp.solve([ds[i] - 1 - sum(Q[i, j] * ds[j] for j in range(m)) for i in range(m)], ds, dict=True)[0]
    d = N * sp.ones(m, 1)
    assert all(sol[ds[i]] == d[i] for i in range(m))
    H = N * Rm
    assert all(sum(H[i, :]) == 1 for i in range(m))  # absorption certaine
    return N, d, H


def matrice_complete(Q, Rm):
    """Matrice de transition complète : états transitoires puis absorbants."""
    m, a = Q.shape[0], Rm.shape[1]
    P = sp.zeros(m + a, m + a)
    P[:m, :m] = Q
    P[:m, m:] = Rm
    P[m:, m:] = sp.eye(a)
    return np.array(P.tolist(), dtype=float)


def simuler_absorption(P, depart, n_sim):
    """Simule la chaîne jusqu'à absorption. Renvoie (état final, nombre de pas)."""
    k = P.shape[0]
    absorbant = np.array([P[i, i] == 1.0 for i in range(k)])
    cum = np.cumsum(P, axis=1)
    cum[:, -1] = 1.0
    etat = np.full(n_sim, depart)
    pas = np.zeros(n_sim, dtype=np.int64)
    actif = ~absorbant[etat]
    while actif.any():
        idx = np.flatnonzero(actif)
        u = rng.random(len(idx))
        etat[idx] = (u[:, None] >= cum[etat[idx]]).sum(axis=1)
        pas[idx] += 1
        actif = ~absorbant[etat]
    return etat, pas


def simuler_marche(depart, bas, haut, p, n_sim):
    """Marche de pas +1 (probabilité p) ou -1, arrêtée en bas ou en haut. Renvoie (finit en haut, nombre de pas)."""
    pos = np.full(n_sim, depart, dtype=np.int64)
    pas = np.zeros(n_sim, dtype=np.int64)
    actif = (pos > bas) & (pos < haut)
    while actif.any():
        k = int(actif.sum())
        pos[actif] += np.where(rng.random(k) < p, 1, -1)
        pas[actif] += 1
        actif = (pos > bas) & (pos < haut)
    return pos == haut, pas


def h_formule(i, n, p):
    """Probabilité de finir en n en partant de i, barrières 0 et n (formule de l'outil 2 de la séance 1)."""
    p = Fraction(p)
    q = 1 - p
    if p == q:
        return Fraction(i, n)
    r = q / p
    return (1 - r**i) / (1 - r**n)


def marche_comme_chaine(n, p):
    """La marche entre 0 et n comme chaîne absorbante : transitoires 1..n-1, absorbants 0 puis n."""
    p = R(p.numerator, p.denominator)
    q = 1 - p
    Q = sp.zeros(n - 1, n - 1)
    Rm = sp.zeros(n - 1, 2)
    for k, i in enumerate(range(1, n)):
        if i + 1 == n:
            Rm[k, 1] = p
        else:
            Q[k, k + 1] = p
        if i - 1 == 0:
            Rm[k, 0] = q
        else:
            Q[k, k - 1] = q
    return Q, Rm


def attente_motif(motif, n_sim, longueur=400):
    """Nombre de lancers d'une pièce équilibrée jusqu'à la première apparition du motif (1 = pile, 0 = face)."""
    l_ = len(motif)
    x = rng.integers(0, 2, size=(n_sim, longueur), dtype=np.int8)
    ok = np.ones((n_sim, longueur - l_ + 1), dtype=bool)
    for k, c in enumerate(motif):
        ok &= x[:, k:longueur - l_ + 1 + k] == c
    assert ok.any(axis=1).all()  # la probabilité de ne pas voir le motif en 400 lancers est négligeable
    return ok.argmax(axis=1) + l_


# =====================================================================================
# LEÇON, PARTIE 1 : chaînes de Markov
# =====================================================================================
# 1a. Deux régimes : calme -> agité avec probabilité a = 0,1, agité -> calme avec b = 0,4.
a_, b_ = R(1, 10), R(4, 10)
P2 = sp.Matrix([[1 - a_, a_], [b_, 1 - b_]])
pi2 = stationnaire(P2)
assert pi2 == [b_ / (a_ + b_), a_ / (a_ + b_)] == [R(4, 5), R(1, 5)]
lam = 1 - a_ - b_
V["La_pi1"] = float(pi2[0])
V["La_pi2"] = float(pi2[1])
V["La_lambda"] = float(lam)
# Convergence : u_t = P(agité à la date t) partant de calme vaut pi2 (1 - lambda^t) ; contrôle par puissances exactes.
for t in range(0, 12):
    loi = sp.Matrix([[1, 0]]) * P2**t
    assert loi[1] == pi2[1] * (1 - lam**t)
u3 = pi2[1] * (1 - lam**3)
fraction("La_u3", u3)
V["La_lambda3"] = float(lam**3)
etats = np.zeros(400_000, dtype=np.int64)
cum2 = np.cumsum(np.array(P2.tolist(), dtype=float), axis=1)
cum2[:, -1] = 1.0
for _ in range(3):
    etats = (rng.random(len(etats))[:, None] >= cum2[etats]).sum(axis=1)
mc_proba("La_u3", u3, etats == 1)

# 1b. Attendre PFP : états transitoires « vide », « P », « PF ».
Q_pfp = sp.Matrix([[R(1, 2), R(1, 2), 0], [0, R(1, 2), R(1, 2)], [R(1, 2), 0, 0]])
R_pfp = sp.Matrix([[0], [0], [R(1, 2)]])
N_pfp, d_pfp, H_pfp = absorption(Q_pfp, R_pfp)
assert list(d_pfp) == [10, 8, 6]
V["La_pfp_d0"], V["La_pfp_d1"], V["La_pfp_d2"] = int(d_pfp[0]), int(d_pfp[1]), int(d_pfp[2])
V["La_pfp_inv_p"] = 2**3  # 1 / P(PFP à une position donnée)
mc_moyenne("La_pfp", 10, attente_motif([1, 0, 1], 200_000))
mc_moyenne("La_pp", 6, attente_motif([1, 1], 200_000))  # comparaison citée : motif PP
V["La_pp"] = 6
d_pp = absorption(sp.Matrix([[R(1, 2), R(1, 2)], [R(1, 2), 0]]), sp.Matrix([[0], [R(1, 2)]]))[1]
assert d_pp[0] == 6

# 1c. Crédit : transitoires S (sain), F (fragile) ; absorbants D (défaut), R (remboursé).
Q_cr = sp.Matrix([[R(7, 10), R(2, 10)], [R(3, 10), R(4, 10)]])
R_cr = sp.Matrix([[0, R(1, 10)], [R(2, 10), R(1, 10)]])
N_cr, d_cr, H_cr = absorption(Q_cr, R_cr)
fraction("La_cr_det", (sp.eye(2) - Q_cr).det())
for (i, j), nom in {(0, 0): "La_cr_n_ss", (0, 1): "La_cr_n_sf", (1, 0): "La_cr_n_fs", (1, 1): "La_cr_n_ff"}.items():
    fraction(nom, N_cr[i, j])
fraction("La_cr_d_s", d_cr[0])
fraction("La_cr_d_f", d_cr[1])
fraction("La_cr_h_sd", H_cr[0, 0])
fraction("La_cr_h_sr", H_cr[0, 1])
fraction("La_cr_h_fd", H_cr[1, 0])
fraction("La_cr_h_fr", H_cr[1, 1])
assert H_cr == sp.Matrix([[R(1, 3), R(2, 3)], [R(1, 2), R(1, 2)]]) and list(d_cr) == [R(20, 3), 5]
P_cr = matrice_complete(Q_cr, R_cr)
fin, pas = simuler_absorption(P_cr, 0, 300_000)
mc_proba("La_cr_h_sd", H_cr[0, 0], fin == 2)
mc_moyenne("La_cr_d_s", d_cr[0], pas)
fin, pas = simuler_absorption(P_cr, 1, 300_000)
mc_proba("La_cr_h_fd", H_cr[1, 0], fin == 2)
mc_moyenne("La_cr_d_f", d_cr[1], pas)

# =====================================================================================
# LEÇON, PARTIE 2 : martingales
# =====================================================================================
p_s, z_s, y_s = sp.symbols("p z y", positive=True)
q_s = 1 - p_s
r_s = q_s / p_s
# r^Z est une martingale : E[r^X] = p r + q / r = 1, pour tout p dans ]0, 1[.
assert sp.simplify(p_s * r_s + q_s / r_s - 1) == 0
# r = p/q ne convient pas : E[(p/q)^X] - 1 = (p - q)^2 / (p q), nul seulement pour p = 1/2.
assert sp.simplify(p_s * (p_s / q_s) + q_s * (q_s / p_s) - 1 - (p_s - q_s) ** 2 / (p_s * q_s)) == 0
# Z^2 - t pour p = 1/2 : E[(z + X)^2] = z^2 + 1.
assert sp.expand(R(1, 2) * (z_s + 1) ** 2 + R(1, 2) * (z_s - 1) ** 2 - z_s**2 - 1) == 0
# Z - (p - q) t : E[z + X - (p - q)] = z.
assert sp.simplify(p_s * (z_s + 1) + q_s * (z_s - 1) - (p_s - q_s) - z_s) == 0
# Cas général : (Z - (p-q) t)^2 - (1 - (p-q)^2) t, avec D = X - (p - q) d'espérance 0.
dp, dm = 1 - (p_s - q_s), -1 - (p_s - q_s)
assert sp.simplify(p_s * (y_s + dp) ** 2 + q_s * (y_s + dm) ** 2 - y_s**2 - (1 - (p_s - q_s) ** 2)) == 0
dire("Martingales de la marche vérifiées symboliquement (r^Z, Z^2 - t, Z - (p-q)t, forme générale).")

# Application 1 : ruine par martingale, p = 2/3, départ 2, barrières 0 et 5 (même exemple que l'outil 2).
p_b = Fraction(2, 3)
r_b = (1 - p_b) / p_b
h_mart = (1 - r_b**2) / (1 - r_b**5)  # h r^n + (1 - h) = r^i
assert h_mart * r_b**5 + (1 - h_mart) == r_b**2
Q_w, R_w = marche_comme_chaine(5, p_b)
_, d_w, H_w = absorption(Q_w, R_w)
assert vers_fraction(H_w[1, 1]) == h_mart == Fraction(24, 31)
fraction("Lb_h", h_mart)
fraction("Lb_r", r_b)
haut_, pas_ = simuler_marche(2, 0, 5, float(p_b), 200_000)
mc_proba("Lb_h", h_mart, haut_)
mc_moyenne("Lb_mart_r", r_b**2, np.where(haut_, float(r_b**5), 1.0))  # E[r^{Z_T}] = r^i

# Application 2 : durée symétrique, n i - i^2 = i (n - i).
Q_s, R_s = marche_comme_chaine(5, Fraction(1, 2))
_, d_sym, _ = absorption(Q_s, R_s)
assert d_sym[1] == 5 * 2 - 2**2 == 6
V["Lb_d_sym"] = int(d_sym[1])
V["Lb_borne_sym"] = 5**2 - 2**2
haut_, pas_ = simuler_marche(2, 0, 5, 0.5, 200_000)
mc_moyenne("Lb_d_sym", 6, pas_)
z_fin = np.where(haut_, 5, 0)
mc_moyenne("Lb_mart_z2", 0, z_fin**2 - pas_ - 2**2)  # E[Z_T^2 - T] = i^2

# Contre-exemple : marche symétrique partie de 0, T = premier passage en 1, horizon t = 100.
t_ce = 100
loi = {0: Fraction(1)}  # loi de Z_t restreinte à {T > t}
for _ in range(t_ce):
    nouvelle = {}
    for z, pr in loi.items():
        for z2 in (z - 1, z + 1):
            if z2 < 1:
                nouvelle[z2] = nouvelle.get(z2, Fraction(0)) + pr / 2
    loi = nouvelle
p_survie = sum(loi.values())
e_survie = sum(z * pr for z, pr in loi.items())
assert e_survie + (1 - p_survie) * 1 == 0  # E[Z_{T ^ t}] = 0 exactement
assert p_survie == Fraction(math.comb(t_ce, t_ce // 2), 2**t_ce)  # principe de réflexion, méthode indépendante
V["Lb_ce_t"] = t_ce
V["Lb_ce_pT"] = float(p_survie)
V["Lb_ce_pT_compl"] = float(1 - p_survie)
V["Lb_ce_cond"] = float(e_survie / p_survie)
n_ce = 200_000
traj = np.cumsum(rng.choice(np.array([-1, 1], dtype=np.int16), size=(n_ce, t_ce)), axis=1, dtype=np.int16)
survie = traj.max(axis=1) <= 0
z_arrete = np.where(survie, traj[:, -1], 1)
mc_proba("Lb_ce_pT", p_survie, survie)
mc_moyenne("Lb_ce_e", 0, z_arrete)
mc_moyenne("Lb_ce_cond", e_survie / p_survie, traj[survie, -1])

# =====================================================================================
# LEÇON, PARTIE 3 : AR(1)
# =====================================================================================
phi_s, sig_s, c_s = sp.symbols("phi sigma c", real=True)
k_s, h_s = sp.symbols("k h", integer=True, nonnegative=True)


def ar1_exact(c, phi, sigma2, h):
    """Moyenne, variance et autocovariance au retard h, par les formules de la leçon."""
    mu = c / (1 - phi)
    g0 = sigma2 / (1 - phi**2)
    return mu, g0, phi**h * g0


def ar1_serie(phi, sigma2, h):
    """Autocovariance par la représentation en somme de chocs : sigma^2 somme_k phi^k phi^(k+h)."""
    return sp.summation(sigma2 * phi**k_s * phi ** (k_s + h), (k_s, 0, sp.oo))


for phi_ in [R(4, 5), R(3, 5), R(1, 2), R(-1, 2), R(-9, 10)]:
    for h_ in range(0, 6):
        assert sp.simplify(ar1_serie(phi_, 1, h_) - ar1_exact(0, phi_, 1, h_)[2]) == 0
dire("Autocovariances de l'AR(1) : formule phi^h sigma^2/(1-phi^2) conforme à la série des chocs (5 valeurs de phi, h <= 5).")


def simuler_ar1(c, phi, sigma, n_sim, n_pas, y0=None, chocs="normal"):
    """Trajectoires de l'AR(1) ; départ dans la loi stationnaire (normale) si y0 est None."""
    mu = c / (1 - phi)
    g0 = sigma**2 / (1 - phi**2)
    y = np.empty((n_sim, n_pas + 1))
    y[:, 0] = rng.normal(mu, np.sqrt(g0), n_sim) if y0 is None else y0
    for t in range(1, n_pas + 1):
        if chocs == "normal":
            e = rng.normal(0, sigma, n_sim)
        else:  # uniforme de même variance : seules l'espérance et la variance des chocs comptent
            e = rng.uniform(-np.sqrt(3) * sigma, np.sqrt(3) * sigma, n_sim)
        y[:, t] = c + phi * y[:, t - 1] + e
    return y


# Exemple : c = 1, phi = 0,8, sigma = 1.
c1, phi1, s1 = R(1), R(4, 5), R(1)
mu1, g01, g31 = ar1_exact(c1, phi1, s1**2, 3)
assert mu1 == 5 and g01 == R(25, 9)
V["Lc_mu"] = int(mu1)
fraction("Lc_g0", g01)
V["Lc_rho3"] = float(phi1**3)
fraction("Lc_g3", g31)
V["Lc_un_moins_phi"] = float(1 - phi1)
V["Lc_un_moins_phi2"] = float(1 - phi1**2)
y0_ = 10
ey3 = mu1 + phi1**3 * (y0_ - mu1)
vy3 = g01 * (1 - phi1**6)
V["Lc_ecart0"] = int(y0_ - mu1)
V["Lc_ey3"] = float(ey3)
fraction("Lc_vy3", vy3)
V["Lc_demi_vie"] = math.log(0.5) / math.log(0.8)
assert 0.8 ** V["Lc_demi_vie"] - 0.5 < 1e-15
y = simuler_ar1(1.0, 0.8, 1.0, 200_000, 3, chocs="uniforme")
mc_moyenne("Lc_mu", mu1, y[:, 3])
mc_variance("Lc_g0", g01, y[:, 3])
mc_covariance("Lc_g3", g31, y[:, 0], y[:, 3])
y = simuler_ar1(1.0, 0.8, 1.0, 200_000, 3, y0=10.0)
mc_moyenne("Lc_ey3", ey3, y[:, 3])
mc_variance("Lc_vy3", vy3, y[:, 3])

# Marche aléatoire : corrélation de Y_100 et Y_110.
V["Lc_rw_corr"] = math.sqrt(100 / 110)
e = rng.normal(0, 1, (100_000, 110))
w = np.cumsum(e, axis=1)
xs_, ys_ = w[:, 99], w[:, 109]
mc_covariance("Lc_rw_cov", 100, xs_, ys_)
V["Lc_rw_corr_mc"] = float(np.corrcoef(xs_, ys_)[0, 1])
assert abs(V["Lc_rw_corr_mc"] - V["Lc_rw_corr"]) < 0.005

# Point de contrôle « ar1 » : phi = -0,5, autocorrélation au retard 3.
phi_c = R(-1, 2)
V["Cc_rho3"] = float(phi_c**3)
V["Cc_d_signe"] = float(abs(phi_c**3))
V["Cc_d_produit"] = float(3 * phi_c)
V["Cc_d_rho1"] = float(phi_c)
y = simuler_ar1(0.0, -0.5, 1.0, 200_000, 3)
rho_mc = np.corrcoef(y[:, 0], y[:, 3])[0, 1]
assert abs(rho_mc - (-0.125)) < 0.01
V["Cc_rho3_mc"] = float(rho_mc)

# =====================================================================================
# LEÇON, PARTIE 4 : mouvement brownien, Itô, brownien géométrique
# =====================================================================================
x_s, th_s = sp.symbols("x theta", real=True)
v_s, t_s = sp.symbols("v t", positive=True)
densite = sp.exp(-x_s**2 / (2 * v_s)) / sp.sqrt(2 * sp.pi * v_s)
assert sp.simplify(sp.integrate(sp.exp(th_s * x_s) * densite, (x_s, -sp.oo, sp.oo)) - sp.exp(th_s**2 * v_s / 2)) == 0
assert sp.simplify(sp.integrate(x_s**4 * densite, (x_s, -sp.oo, sp.oo)) - 3 * v_s**2) == 0
# Coefficient de theta^4 dans exp(theta^2 v / 2) : v^2 / 8, donc E[G^4] = 24 v^2 / 8 = 3 v^2.
assert sp.series(sp.exp(th_s**2 * v_s / 2), th_s, 0, 6).removeO().coeff(th_s, 4) == v_s**2 / 8
dire("Fonction génératrice de la loi normale et quatrième moment vérifiés symboliquement.")
V["Ld_ew1"] = math.exp(0.5)
w1 = rng.normal(0, 1, 400_000)
mc_moyenne("Ld_ew1", math.exp(0.5), np.exp(w1))
# Route Itô : m(t) = 1 + (1/2) int_0^t m(s) ds, soit m' = m/2, m(0) = 1.
m_f = sp.Function("m")
sol = sp.dsolve(sp.Eq(m_f(t_s).diff(t_s), m_f(t_s) / 2), m_f(t_s), ics={m_f(0): 1})
assert sp.simplify(sol.rhs - sp.exp(t_s / 2)) == 0

# Variation quadratique sur [0, 1], subdivision régulière : E = 1, Var = 2/n.
for n_qv, n_traj, nom in [(100, 40_000, "Ld_qv_sd100"), (10_000, 4_000, "Ld_qv_sd10000")]:
    V[nom] = math.sqrt(2 / n_qv)
    vs = []
    for _ in range(n_traj // 1000):
        dw = rng.normal(0, math.sqrt(1 / n_qv), (1000, n_qv))
        vs.append((dw**2).sum(axis=1))
    vs = np.concatenate(vs)
    mc_moyenne(nom + "_moy", 1, vs)
    mc_variance(nom + "_var", 2 / n_qv, vs)
    V[nom + "_mc"] = float(vs.std())
# Intégrale de Itô de W contre W : somme à gauche contre (W_1^2 - 1)/2, n = 100.
n_i = 100
dw = rng.normal(0, math.sqrt(1 / n_i), (40_000, n_i))
wg = np.concatenate([np.zeros((40_000, 1)), np.cumsum(dw, axis=1)[:, :-1]], axis=1)  # W au point gauche
somme_gauche = (wg * dw).sum(axis=1)
w_fin = dw.sum(axis=1)
assert np.allclose(somme_gauche, (w_fin**2 - (dw**2).sum(axis=1)) / 2)  # identité algébrique exacte
ecart = somme_gauche - (w_fin**2 - 1) / 2
V["Ld_ito_n"] = n_i
V["Ld_ito_rms"] = 1 / math.sqrt(2 * n_i)
mc_moyenne("Ld_ito_ms", 1 / (2 * n_i), ecart**2)  # E[écart^2] = Var(V_n)/4 = 2/(4n)
V["Ld_ito_rms_mc"] = float(np.sqrt((ecart**2).mean()))
mc_moyenne("Ld_ito_moy", 0, somme_gauche)  # espérance nulle de l'intégrale stochastique

# Brownien géométrique : s0 = 100, mu = 0,08, sigma = 0,2, t = 1.
s0, mu_g, sg = 100.0, 0.08, 0.2
f_s = 100 * sp.exp((R(8, 100) - R(2, 100)) * t_s + R(1, 5) * x_s)
assert sp.simplify(f_s.diff(t_s) + f_s.diff(x_s, 2) / 2 - R(8, 100) * f_s) == 0  # dérive de Itô = mu f
assert sp.simplify(f_s.diff(x_s) - R(1, 5) * f_s) == 0  # terme en dW = sigma f
V["Ld_gbm_e"] = s0 * math.exp(mu_g)
V["Ld_gbm_med"] = s0 * math.exp(mu_g - sg**2 / 2)
V["Ld_gbm_drift_log"] = mu_g - sg**2 / 2
V["Ld_gbm_z"] = (mu_g - sg**2 / 2) / sg
V["Ld_gbm_p"] = 0.5 * (1 + math.erf(V["Ld_gbm_z"] / math.sqrt(2)))
assert abs(V["Ld_gbm_p"] - float(sp.Rational(1, 2) * (1 + sp.erf(sp.Rational(3, 10) / sp.sqrt(2))))) < 1e-15
e_exact = sp.integrate(f_s.subs(t_s, 1) * densite.subs(v_s, 1), (x_s, -sp.oo, sp.oo))
assert abs(float(e_exact) - V["Ld_gbm_e"]) < 1e-9
w1 = rng.normal(0, 1, 400_000)
s1 = s0 * np.exp(mu_g - sg**2 / 2 + sg * w1)
mc_moyenne("Ld_gbm_e", V["Ld_gbm_e"], s1)
mc_proba("Ld_gbm_p", V["Ld_gbm_p"], s1 > s0)
mc_proba("Ld_gbm_med", 0.5, s1 <= V["Ld_gbm_med"])
# Schéma d'Euler de l'équation dS = mu S dt + sigma S dW, même bruit que la solution fermée.
n_eu, n_traj = 1000, 20_000
dw = rng.normal(0, math.sqrt(1 / n_eu), (n_traj, n_eu))
s_eu = np.full(n_traj, s0)
for k in range(n_eu):
    s_eu = s_eu * (1 + mu_g / n_eu + sg * dw[:, k])
s_ex = s0 * np.exp(mu_g - sg**2 / 2 + sg * dw.sum(axis=1))
V["Ld_euler_err"] = float(np.mean(np.abs(s_eu - s_ex) / s_ex))
assert V["Ld_euler_err"] < 0.005
dire(f"Schéma d'Euler contre solution fermée, même bruit, 1000 pas : écart relatif moyen {V['Ld_euler_err']:.2e}")

# =====================================================================================
# QUESTIONS
# =====================================================================================
# M4-01 : deux régimes, a = 0,15 (calme -> agité), b = 0,35 (agité -> calme).
a1, b1 = R(15, 100), R(35, 100)
P_q1 = sp.Matrix([[1 - a1, a1], [b1, 1 - b1]])
pi_q1 = stationnaire(P_q1)
assert pi_q1[1] == a1 / (a1 + b1) == R(3, 10)
V["Q01_exact"] = float(pi_q1[1])
V["Q01_somme"] = float(a1 + b1)
V["Q01_lambda"] = float(1 - a1 - b1)
V["Q01_d_inverse"] = float(pi_q1[0])
V["Q01_d_ratio"] = float(a1 / b1)
V["Q01_d_un_pas"] = float(a1)
u60 = (sp.Matrix([[1, 0]]) * P_q1**60)[1]
assert abs(float(u60) - 0.3) < 1e-15
cum = np.cumsum(np.array(P_q1.tolist(), dtype=float), axis=1)
cum[:, -1] = 1.0
etats = np.zeros(200_000, dtype=np.int64)
for _ in range(60):
    etats = (rng.random(len(etats))[:, None] >= cum[etats]).sum(axis=1)
mc_proba("Q01", u60, etats == 1)

# M4-02 : dé équilibré, deux 6 consécutifs.
Q_66 = sp.Matrix([[R(5, 6), R(1, 6)], [R(5, 6), 0]])
R_66 = sp.Matrix([[0], [R(1, 6)]])
_, d_66, _ = absorption(Q_66, R_66)
assert list(d_66) == [42, 36]
V["Q02_exact"] = int(d_66[0])
V["Q02_e1"] = int(d_66[1])
V["Q02_d_inverse"] = 36
V["Q02_d_somme"] = 6 + 6
V["Q02_d_blocs"] = 2 * 36
V["Q02_borne"] = 2 * 36


def attente_deux_six(n_sim):
    precedent6 = np.zeros(n_sim, dtype=bool)
    n = np.zeros(n_sim, dtype=np.int64)
    actif = np.ones(n_sim, dtype=bool)
    while actif.any():
        idx = np.flatnonzero(actif)
        de = rng.integers(1, 7, size=len(idx))
        n[idx] += 1
        fini = precedent6[idx] & (de == 6)
        precedent6[idx] = de == 6
        actif[idx[fini]] = False
    return n


mc_moyenne("Q02", 42, attente_deux_six(200_000))

# M4-03 : tennis, égalité, serveur gagne chaque point avec probabilité 0,6.
p3 = R(3, 5)
q3 = 1 - p3
Q_t = sp.Matrix([[0, p3, q3], [q3, 0, 0], [p3, 0, 0]])  # égalité, avantage serveur, avantage receveur
R_t = sp.Matrix([[0, 0], [p3, 0], [0, q3]])  # jeu gagné, jeu perdu
_, d_t, H_t = absorption(Q_t, R_t)
assert H_t[0, 0] == p3**2 / (p3**2 + q3**2) == R(9, 13)
assert vers_fraction(H_t[0, 0]) == h_formule(2, 4, Fraction(3, 5))  # écart de points : marche entre -2 et +2
fraction("Q03_exact", H_t[0, 0])
fraction("Q03_duree", d_t[0])
assert d_t[0] == 2 / (p3**2 + q3**2)
V["Q03_p2"] = float(p3**2)
V["Q03_q2"] = float(q3**2)
V["Q03_2pq"] = float(2 * p3 * q3)
V["Q03_den"] = float(1 - 2 * p3 * q3)
V["Q03_d_p"] = float(p3)
V["Q03_d_deux"] = float(p3**2)
V["Q03_d_un_retour"] = float(p3**2 + 2 * p3 * q3 * p3**2)
ecart_pts = np.zeros(300_000, dtype=np.int64)
nb_pts = np.zeros(300_000, dtype=np.int64)
actif = np.ones(300_000, dtype=bool)
while actif.any():
    idx = np.flatnonzero(actif)
    ecart_pts[idx] += np.where(rng.random(len(idx)) < 0.6, 1, -1)
    nb_pts[idx] += 1
    actif = np.abs(ecart_pts) < 2
mc_proba("Q03", H_t[0, 0], ecart_pts == 2)
mc_moyenne("Q03_duree", d_t[0], nb_pts)

# M4-04 : atteindre +3 sans barrière basse, p = 0,45.
p4 = Fraction(9, 20)
r4 = (1 - p4) / p4
assert r4 == Fraction(11, 9)
lim4 = r4 ** (-3)
fraction("Q04_exact", lim4)
fraction("Q04_r", r4)
fraction("Q04_p_sur_q", 1 / r4)
m_b = sp.symbols("m", positive=True, integer=True)
r4s = R(11, 9)
h_m = (1 - r4s ** (-m_b)) / (r4s**3 - r4s ** (-m_b))
assert sp.limit(h_m, m_b, sp.oo) == R(729, 1331)
m_bas = 60
h60 = h_formule(m_bas, m_bas + 3, p4)  # barrière basse à -60 : positions décalées de 60
assert h60 == (1 - r4 ** (-m_bas)) / (r4**3 - r4 ** (-m_bas))
assert 0 < lim4 - h60 < Fraction(1, 10**5)
V["Q04_m"] = m_bas
V["Q04_h60"] = float(h60)
V["Q04_d_trois"] = float(p4**3)
V["Q04_d_un"] = 1
V["Q04_d_exposant"] = float(1 / r4)
haut_, _ = simuler_marche(m_bas, 0, m_bas + 3, float(p4), 200_000)
mc_proba("Q04", h60, haut_)

# M4-05 : AR(1), c = 1, phi = 0,6, sigma = 2 : Cov(Y_t, Y_{t+2}).
mu5, g05, g25 = ar1_exact(R(1), R(3, 5), R(4), 2)
assert g25 == R(9, 4) and g05 == R(25, 4)
assert sp.simplify(ar1_serie(R(3, 5), 4, 2) - g25) == 0
V["Q05_exact"] = float(g25)
V["Q05_g0"] = float(g05)
V["Q05_phi2"] = float(R(3, 5) ** 2)
V["Q05_un_moins_phi2"] = float(1 - R(3, 5) ** 2)
V["Q05_mu"] = float(mu5)
V["Q05_d_var"] = float(g05)
V["Q05_d_rho"] = float(R(3, 5) ** 2)
V["Q05_d_mauvais_g0"] = float(R(3, 5) ** 2 * 4 / (1 - R(3, 5)))
y = simuler_ar1(1.0, 0.6, 2.0, 300_000, 2)
mc_covariance("Q05", g25, y[:, 0], y[:, 2])

# M4-06 : AR(1), phi = 0,5, sigma^2 = 1 : Var(Y_1 + Y_2 + Y_3).
phi6 = R(1, 2)
gam = [ar1_exact(0, phi6, 1, h)[2] for h in range(3)]
var6 = sum(gam[abs(s - t)] for s in range(3) for t in range(3))  # somme de la matrice de covariance
assert var6 == gam[0] * (3 + 4 * phi6 + 2 * phi6**2) == R(22, 3)
fraction("Q06_exact", var6)
fraction("Q06_g0", gam[0])
V["Q06_phi2"] = float(phi6**2)
fraction("Q06_g1", gam[1])
fraction("Q06_g2", gam[2])
fraction("Q06_croises", 2 * (2 * gam[1] + gam[2]))
V["Q06_d_indep"] = float(3 * gam[0])
fraction("Q06_d_une_fois", 3 * gam[0] + 2 * gam[1] + gam[2])
V["Q06_d_sigma"] = 3.0
y = simuler_ar1(0.0, 0.5, 1.0, 300_000, 2)
mc_variance("Q06", var6, y.sum(axis=1))

# M4-07 : Var(W_1 + W_2 + W_3).
temps = [1, 2, 3]
var7 = sum(min(s, t) for s in temps for t in temps)
assert var7 == 3**2 * 1 + 2**2 * 1 + 1**2 * 1 == 14
V["Q07_exact"] = var7
V["Q07_diag"] = sum(temps)
V["Q07_croises"] = var7 - sum(temps)
V["Q07_d_indep"] = sum(temps)
V["Q07_d_trois_w2"] = 3**2 * 2
V["Q07_d_produit"] = sum(s * t if s != t else s for s in temps for t in temps)
assert V["Q07_d_produit"] == 28
inc = rng.normal(0, 1, (400_000, 3))
wt = np.cumsum(inc, axis=1)
mc_variance("Q07", 14, wt.sum(axis=1))

# M4-08 : Var(S_1), s0 = 100, mu = 0,05, sigma = 0,2.
mu8, s8 = 0.05, 0.2
var8 = 100**2 * math.exp(2 * mu8) * (math.exp(s8**2) - 1)
f8 = 100 * sp.exp(R(3, 100) + R(1, 5) * x_s)
e1 = sp.integrate(f8 * densite.subs(v_s, 1), (x_s, -sp.oo, sp.oo))
e2 = sp.integrate(f8**2 * densite.subs(v_s, 1), (x_s, -sp.oo, sp.oo))
assert abs(float(e2 - e1**2) - var8) < 1e-8
V["Q08_exact"] = var8
V["Q08_drift_log"] = mu8 - s8**2 / 2
V["Q08_es"] = 100 * math.exp(mu8)
V["Q08_es2"] = 100**2 * math.exp(2 * mu8 + s8**2)
V["Q08_expo1"] = 2 * (mu8 - s8**2 / 2)
V["Q08_expo2"] = 2 * s8**2
V["Q08_expo_tot"] = 2 * mu8 + s8**2
V["Q08_expo_moy"] = 2 * mu8
V["Q08_s2"] = s8**2
V["Q08_s2_demi"] = s8**2 / 2
V["Q08_deux_sigma"] = 2 * s8
V["Q08_d_naif"] = 100**2 * s8**2
V["Q08_d_sans_derive"] = 100**2 * (math.exp(s8**2) - 1)
V["Q08_d_mediane"] = 100**2 * math.exp(2 * mu8 - s8**2) * (math.exp(s8**2) - 1)
arrondis = [round(V[k]) for k in ["Q08_exact", "Q08_d_naif", "Q08_d_sans_derive", "Q08_d_mediane"]]
assert len(set(arrondis)) == 4, arrondis
w1 = rng.normal(0, 1, 1_000_000)
s1 = 100 * np.exp(mu8 - s8**2 / 2 + s8 * w1)
mc_variance("Q08", var8, s1)

# Arithmétique citée dans les corrections.
assert R(15, 100) / R(50, 100) == R(3, 10)
assert R(36, 100) / R(52, 100) == R(9, 13) and 2 / R(52, 100) == R(50, 13)
assert R(4) / R(64, 100) == R(25, 4) and R(36, 100) * R(25, 4) == R(9, 4)
assert R(4, 3) * 3 + 2 * (2 * R(2, 3) + R(1, 3)) == R(22, 3)
assert abs(V["Q08_es2"] - V["Q08_es"] ** 2 - var8) < 1e-8

Path(__file__).with_name("seance_math_4.out.json").write_text(json.dumps(V, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
dire(f"{len(V)} valeurs exportées dans verif/seance_math_4.out.json.")
dire("Toutes les vérifications passent.")
Path(__file__).with_name("seance_math_4.out.txt").write_text("\n".join(LIGNES) + "\n", encoding="utf-8", newline="\n")
