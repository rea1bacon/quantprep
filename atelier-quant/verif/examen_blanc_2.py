"""Vérification de tous les nombres de l'examen blanc 2 (questions EB2-M01 à EB2-A08, exemple de EB2-C1).

Principe :
  1. chaque valeur citée est calculée ici, en fractions exactes ou par sympy quand c'est possible ;
  2. chaque probabilité ou espérance est aussi estimée par simulation Monte-Carlo, dont l'écart à l'exact
     doit rester inférieur à 4 erreurs standard ;
  3. chaque affirmation de ML est illustrée par une expérience numérique (numpy seul, pour que la sortie
     ne dépende pas de la version de scikit-learn) ;
  4. pour la question de code, on vérifie l'exemple de l'énoncé, le comportement de merge_asof cité dans la
     correction, et que les tests cachés attrapent chacune des erreurs visées.

Graine fixe : la sortie est identique à chaque exécution.
Sorties : verif/examen_blanc_2.out.json (lu par outils/construire.py) et verif/examen_blanc_2.out.txt (ce résumé).
"""
import itertools
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import mpmath
import numpy as np
import pandas as pd
import sympy as sp

RACINE = Path(__file__).resolve().parent.parent
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


def mc_proba(nom, exact, indicatrices):
    indicatrices = np.asarray(indicatrices, dtype=float)
    p = float(exact)
    controle_mc(nom, exact, indicatrices.mean(), np.sqrt(p * (1 - p) / len(indicatrices)))


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def phi(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


# =====================================================================================
# EB2-M01 : chaîne de Markov, temps de retour
# =====================================================================================
F = Fraction
P = [[F(8, 10), F(2, 10), F(0)], [F(1, 10), F(7, 10), F(2, 10)], [F(0), F(5, 10), F(5, 10)]]  # C, N, A
assert all(sum(l) == 1 for l in P)
# Loi stationnaire : système linéaire en rationnels.
pc, pn, pa = sp.symbols("pc pn pa")
Ps = sp.Matrix([[sp.Rational(x.numerator, x.denominator) for x in l] for l in P])
sol = sp.solve(list(sp.Matrix([[pc, pn, pa]]) * Ps - sp.Matrix([[pc, pn, pa]])) + [pc + pn + pa - 1], [pc, pn, pa], dict=True)[0]
pi = [F(str(sol[s])) for s in (pc, pn, pa)]
assert pi == [F(5, 19), F(10, 19), F(4, 19)]
# Temps d'atteinte de A par le premier pas : hC = 1 + 0,8 hC + 0,2 hN ; hN = 1 + 0,1 hC + 0,7 hN.
hc, hn = sp.symbols("hc hn")
s2 = sp.solve([hc - 1 - sp.Rational(8, 10) * hc - sp.Rational(2, 10) * hn,
               hn - 1 - sp.Rational(1, 10) * hc - sp.Rational(7, 10) * hn], [hc, hn], dict=True)[0]
hC, hN = F(str(s2[hc])), F(str(s2[hn]))
assert hC == 5 + hN and hN == F(15, 2) and hC == F(25, 2)
mA = 1 + P[2][1] * hN + P[2][2] * 0
assert mA == 1 / pi[2] == F(19, 4)  # premier pas et lemme de Kac concordent
fraction("M01_exact", mA)
fraction("M01_hN", hN)
fraction("M01_hC", hC)
fraction("M01_piA", pi[2])
fraction("M01_piC", pi[0])
fraction("M01_piN", pi[1])
fraction("M01_d_sejour", 1 / (1 - P[2][2]))
fraction("M01_d_un_plus_hN", 1 + hN)
V["M01_d_uniforme"] = 3
# Minoration utilisée pour la finitude : de C, atteindre A en 2 pas a pour probabilité 0,2 x 0,2.
fraction("M01_p_deux_pas", P[0][1] * P[1][2])
# Monte-Carlo : temps de retour en A, chaînes simulées en parallèle.
Pf = np.array([[float(x) for x in l] for l in P])
cum = Pf.cumsum(axis=1)
n_sim = 200_000
etat = np.full(n_sim, 2)
duree = np.zeros(n_sim, dtype=np.int64)
actif = np.ones(n_sim, dtype=bool)
while actif.any():
    idx = np.nonzero(actif)[0]
    u = rng.random(len(idx))
    etat[idx] = (u[:, None] > cum[etat[idx]]).sum(axis=1)
    duree[idx] += 1
    actif[idx] = etat[idx] != 2
mc_moyenne("M01", mA, duree)

# =====================================================================================
# EB2-M02 : urne de Pólya, loi du nombre de rouges
# =====================================================================================
def proba_suite_polya(suite):
    """Probabilité exacte d'une suite de tirages (1 = rouge) depuis 1 rouge et 1 bleue."""
    r, b, p = 1, 1, F(1)
    for x in suite:
        p *= F(r if x else b, r + b)
        if x:
            r += 1
        else:
            b += 1
    return p


n_p = 8
suites = list(itertools.product([0, 1], repeat=n_p))
loi_rouges = {k: sum(proba_suite_polya(s) for s in suites if sum(s) == k) for k in range(n_p + 1)}
assert all(v == F(1, n_p + 1) for v in loi_rouges.values())
# Échangeabilité : toutes les suites à 3 rouges ont la même probabilité 3! 5! / 9!.
assert {proba_suite_polya(s) for s in suites if sum(s) == 3} == {F(math.factorial(3) * math.factorial(5), math.factorial(9))}
fraction("M02_exact", loi_rouges[3])
fraction("M02_d_binomiale", F(math.comb(8, 3), 2**8))
fraction("M02_d_huit", F(1, 8))
fraction("M02_d_suite", F(math.factorial(3) * math.factorial(5), math.factorial(9)))
V["M02_C83"] = math.comb(8, 3)
n_sim = 400_000
rouges = np.ones(n_sim)
total = 2.0
nb_rouges = np.zeros(n_sim)
for _ in range(n_p):
    tire_rouge = rng.random(n_sim) < rouges / total
    rouges += tire_rouge
    nb_rouges += tire_rouge
    total += 1
mc_proba("M02", loi_rouges[3], nb_rouges == 3)

# =====================================================================================
# EB2-M03 : anagrammes avec contrainte d'adjacence
# =====================================================================================
mot = "ARBITRAGE"
assert sorted(mot) == sorted("AABEGIRRT")
anag = set(itertools.permutations(mot))
total_anag = len(anag)
sans_rr = sum(1 for a in anag if "RR" not in "".join(a))
assert total_anag == math.factorial(9) // 4 and total_anag - sans_rr == math.factorial(8) // 2
V["M03_exact"] = sans_rr
V["M03_total"] = total_anag
V["M03_adjacents"] = total_anag - sans_rr
V["M03_d_a_distincts"] = math.factorial(9) // 2 - math.factorial(8)

# =====================================================================================
# EB2-M04 : variance totale, Poisson de paramètre exponentiel
# =====================================================================================
lam, k = sp.symbols("lam k", positive=True)
loi_k = sp.simplify(sp.integrate(sp.exp(-lam) * lam**k / sp.factorial(k) * sp.Rational(1, 4) * sp.exp(-lam / 4), (lam, 0, sp.oo)))
for kk in range(6):
    assert sp.nsimplify(loi_k.subs(k, kk)) == sp.Rational(1, 5) * sp.Rational(4, 5) ** kk
kk = sp.symbols("kk", integer=True, nonnegative=True)
loi = sp.Rational(1, 5) * sp.Rational(4, 5) ** kk
m1 = sp.summation(kk * loi, (kk, 0, sp.oo))
m2 = sp.summation(kk**2 * loi, (kk, 0, sp.oo))
assert m1 == 4 and m2 - m1**2 == 20
V["M04_exact"] = 20
V["M04_E"] = 4
V["M04_var_lambda"] = 16
V["M04_d_somme_moyennes"] = 8
V["M04_E2"] = int(m2)
n_sim = 400_000
L = rng.exponential(4.0, size=n_sim)
N = rng.poisson(L)
mc_moyenne("M04_E", 4, N)
mc_moyenne("M04_E2", m2, N.astype(float) ** 2)

# =====================================================================================
# EB2-M05 : maximum de deux exponentielles
# =====================================================================================
t = sp.symbols("t", positive=True)
esp_max = sp.integrate(1 - (1 - sp.exp(-3 * t)) * (1 - sp.exp(-2 * t)), (t, 0, sp.oo))
assert esp_max == sp.Rational(19, 30) == sp.Rational(1, 3) + sp.Rational(1, 2) - sp.Rational(1, 5)
fraction("M05_exact", F(19, 30))
fraction("M05_d_max_moyennes", F(1, 2))
fraction("M05_d_somme", F(5, 6))
fraction("M05_d_min", F(1, 5))
n_sim = 400_000
A_ = rng.exponential(1 / 3, size=n_sim)
B_ = rng.exponential(1 / 2, size=n_sim)
mc_moyenne("M05", F(19, 30), np.maximum(A_, B_))

# =====================================================================================
# EB2-M06 : conditionnement d'une matrice non symétrique (cisaillement)
# =====================================================================================
A6 = sp.Matrix([[1, 1], [0, 1]])
vp_ata = sorted((A6.T * A6).eigenvals().keys(), key=lambda v: float(v))
assert [sp.simplify(v - w) for v, w in zip(vp_ata, [(3 - sp.sqrt(5)) / 2, (3 + sp.sqrt(5)) / 2])] == [0, 0]
kappa = sp.sqrt(vp_ata[1] / vp_ata[0])
assert sp.simplify(kappa - (3 + sp.sqrt(5)) / 2) == 0
V["M06_exact"] = float(kappa)
V["M06_vp_max"] = float(vp_ata[1])
V["M06_vp_min"] = float(vp_ata[0])
V["M06_d_valeurs_propres"] = 1
V["M06_d_carre"] = float(kappa**2)
inv6 = np.linalg.inv(np.array([[1.0, 1.0], [0.0, 1.0]]))
V["M06_d_infini"] = float(np.abs([[1, 1], [0, 1]]).sum(axis=1).max() * np.abs(inv6).sum(axis=1).max())
assert V["M06_d_infini"] == 4
assert abs(np.linalg.cond(np.array([[1.0, 1.0], [0.0, 1.0]])) - float(kappa)) < 1e-12

# =====================================================================================
# EB2-M07 : méthode delta, écart-type de ln(moyenne)
# =====================================================================================
mu7, s7, n7 = 2, 2, 100
V["M07_exact"] = s7 / (mu7 * math.sqrt(n7))
V["M07_d_sans_derivee"] = s7 / math.sqrt(n7)
V["M07_d_variance"] = V["M07_exact"] ** 2
V["M07_d_derivee_seconde"] = s7 / (mu7**2 * math.sqrt(n7))
V["M07_sd_moyenne"] = s7 / math.sqrt(n7)
# Illustration : loi exponentielle d'espérance 2 (variance 4), écart-type simulé de ln(moyenne de 100).
moy7 = rng.exponential(2.0, size=(200_000, n7)).mean(axis=1)
V["M07_sd_sim"] = round(float(np.log(moy7).std(ddof=1)), 4)
assert abs(V["M07_sd_sim"] - V["M07_exact"]) < 0.005
dire(f"M07 : écart-type simulé de ln(moyenne) = {V['M07_sd_sim']} (méthode delta : {V['M07_exact']})")

# =====================================================================================
# EB2-M08 : test d'une proportion, p-value bilatérale
# =====================================================================================
n_j, succes = 400, 220
p_chapeau = F(succes, n_j)
se0 = math.sqrt(0.25 / n_j)
z = float(p_chapeau - F(1, 2)) / se0
assert abs(z - 2) < 1e-12
pval = 2 * (1 - Phi(2.0))
V["M08_phat"] = float(p_chapeau)
V["M08_se"] = se0
V["M08_z"] = 2
V["M08_exact"] = pval
V["M08_d_unilaterale"] = pval / 2
V["M08_d_complement"] = 1 - pval
V["M08_d_div_n_z"] = float(p_chapeau - F(1, 2)) / (0.25 / n_j)
V["M08_d_div_n"] = 2 * (1 - Phi(V["M08_d_div_n_z"]))
assert V["M08_d_div_n"] < 5e-4
# Contrôle de 2(1 - Phi(2)) par une intégrale sympy, puis Monte-Carlo sur des normales.
zz = sp.symbols("zz", real=True)
assert abs(float(2 * sp.integrate(sp.exp(-zz**2 / 2) / sp.sqrt(2 * sp.pi), (zz, 2, sp.oo))) - pval) < 1e-12
mc_proba("M08", pval, np.abs(rng.standard_normal(2_000_000)) >= 2)
# Pour information : p-value binomiale exacte du même test (sans approximation normale).
p_binom = F(sum(math.comb(n_j, j) for j in range(n_j + 1) if abs(j - 200) >= 20), 2**n_j)
V["M08_binom"] = float(p_binom)
mc_proba("M08_binom", p_binom, np.abs(rng.binomial(n_j, 0.5, size=1_000_000) - 200) >= 20)

# =====================================================================================
# EB2-M09 : espérance tronquée gaussienne
# =====================================================================================
exact_m09 = sp.integrate((zz - 1) * sp.exp(-zz**2 / 2) / sp.sqrt(2 * sp.pi), (zz, 1, sp.oo))
v09 = float(exact_m09)
assert abs(v09 - (phi(1) - (1 - Phi(1)))) < 1e-12
V["M09_exact"] = v09
V["M09_phi1"] = phi(1)
V["M09_queue"] = 1 - Phi(1)
V["M09_Phi1"] = Phi(1)
V["M09_d_jensen"] = 0
V["M09_E_zplus"] = phi(0)
Z_ = rng.standard_normal(2_000_000)
mc_moyenne("M09", v09, np.maximum(Z_ - 1, 0))

# =====================================================================================
# EB2-M10 : variance de l'intégrale du mouvement brownien
# =====================================================================================
s_, t_ = sp.symbols("s_ t_", positive=True)
var_I = sp.integrate(sp.integrate(sp.Min(s_, t_), (s_, 0, 1)), (t_, 0, 1))
assert var_I == sp.Rational(1, 3)
assert sp.integrate((1 - s_) ** 2, (s_, 0, 1)) == sp.Rational(1, 3)  # isométrie d'Itô
assert sp.integrate(s_, (s_, 0, 1)) == sp.Rational(1, 2)
assert 2 * sp.integrate(sp.integrate(s_, (s_, 0, t_)), (t_, 0, 1)) == sp.Rational(1, 3)
fraction("M10_exact", F(1, 3))
fraction("M10_d_second_moment", F(1, 2))
V["M10_d_un"] = 1
fraction("M10_d_moitie", F(1, 6))


def var_riemann(N):
    """Variance exacte de (1/N) somme_{k=1}^{N} W_{k/N} : (1/N^3) somme_{j,k} min(j, k)."""
    return F(sum(min(j, kk_) for j in range(1, N + 1) for kk_ in range(1, N + 1)), N**3)


for N_ in (10, 100, 400):
    dire(f"M10 : variance de la somme de Riemann à {N_} pas = {float(var_riemann(N_)):.6f} (limite 1/3)")
N_ = 200
v200 = var_riemann(N_)
assert abs(float(v200) - 1 / 3) < 1 / N_
V["M10_var_riemann_200"] = float(v200)
paquets = []
for _ in range(10):
    W_ = np.cumsum(rng.standard_normal((20_000, N_)) / np.sqrt(N_), axis=1)
    paquets.append(W_.mean(axis=1))
I_ = np.concatenate(paquets)
mc_moyenne("M10_riemann_200", v200, I_**2)

# =====================================================================================
# EB2-F01 : log-loss du meilleur prédicteur constant
# =====================================================================================
q1 = sp.Rational(1, 5)
pp = sp.symbols("pp", positive=True)
ll = -(q1 * sp.log(pp) + (1 - q1) * sp.log(1 - pp))
assert sp.solve(sp.diff(ll, pp), pp) == [q1]
V["F01_exact"] = float(ll.subs(pp, q1))
V["F01_d_demi"] = math.log(2)
V["F01_d_negatifs"] = -math.log(0.8)
V["F01_d_brier"] = 0.2 * 0.8
grille_p = np.linspace(0.01, 0.99, 9801)
pertes = -(0.2 * np.log(grille_p) + 0.8 * np.log(1 - grille_p))
assert abs(grille_p[pertes.argmin()] - 0.2) < 1e-4

# =====================================================================================
# EB2-F02 : R² hors échantillon négatif
# =====================================================================================
y_test = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
pred = 4.0
scr = float(((y_test - pred) ** 2).sum())
sct = float(((y_test - y_test.mean()) ** 2).sum())
V["F02_scr"] = scr
V["F02_sct"] = sct
V["F02_exact"] = 1 - scr / sct
V["F02_d_zero"] = 0
V["F02_d_signe"] = -(1 - scr / sct)
V["F02_d_sans_un"] = -scr / sct
assert V["F02_exact"] == -0.5

# =====================================================================================
# EB2-F03 : seuil de décision optimal avec coûts asymétriques
# =====================================================================================
c_fn, c_fp = 9, 1
fraction("F03_exact", F(c_fp, c_fp + c_fn))
V["F03_d_defaut"] = 0.5
fraction("F03_d_inverse", F(c_fn, c_fp + c_fn))
fraction("F03_d_rapport", F(c_fp, c_fn))
# Expérience : probabilités calibrées uniformes, étiquettes tirées selon elles ; coût moyen selon le seuil.
p3 = rng.random(1_000_000)
y3 = rng.random(1_000_000) < p3
seuils = np.round(np.arange(0.02, 0.99, 0.02), 2)
couts = [float(np.where(p3 > s_, ~y3 * c_fp, y3 * c_fn).mean()) for s_ in seuils]
V["F03_exp_meilleur_seuil"] = float(seuils[int(np.argmin(couts))])
assert abs(V["F03_exp_meilleur_seuil"] - 0.1) <= 0.02
V["F03_exp_cout_01"] = round(couts[list(seuils).index(0.1)], 3)
V["F03_exp_cout_05"] = round(couts[list(seuils).index(0.5)], 3)
dire(f"F03 : coût moyen simulé {V['F03_exp_cout_01']} au seuil 0,1 contre {V['F03_exp_cout_05']} au seuil 0,5")

# =====================================================================================
# EB2-F04 : malédiction de la dimension
# =====================================================================================
arete = 0.01 ** (1 / 10)
V["F04_exact"] = arete
V["F04_d_fraction"] = 0.01
V["F04_d_div"] = 0.001
V["F04_d_dim2"] = 0.1
V["F04_arete_d100"] = 0.01 ** (1 / 100)
pts = rng.random((400_000, 10))
dedans = np.all((pts >= 0.2) & (pts <= 0.2 + arete), axis=1)
mc_proba("F04", 0.01, dedans)

# =====================================================================================
# EB2-F05 : validation croisée un-contre-tous en régression linéaire, formule du levier
# =====================================================================================
x5 = np.array([0.0, 1.0, 2.0, 3.0, 9.0])
X5 = np.column_stack([np.ones(5), x5])
H5 = X5 @ np.linalg.inv(X5.T @ X5) @ X5.T
xb = F(15, 5)
sxx5 = sum((F(int(v)) - xb) ** 2 for v in x5)
h55 = F(1, 5) + (F(9) - xb) ** 2 / sxx5
assert abs(H5[4, 4] - float(h55)) < 1e-12 and h55 == F(23, 25)
fraction("F05_h", h55)
V["F05_sxx"] = int(sxx5)
V["F05_moy_x"] = int(xb)
fraction("F05_exact", F(2, 5) / (1 - h55))
fraction("F05_d_residu", F(2, 5))
fraction("F05_d_sans_1n", F(2, 5) / (1 - (F(9) - xb) ** 2 / sxx5))
fraction("F05_d_1n", F(2, 5) / (1 - F(1, 5)))
# Données réalisant l'énoncé : y = 5 e_5 a un résidu 0,4 en x = 9 ; contrôle par réajustement explicite,
# puis identité e_i / (1 - h_ii) vérifiée sur des données aléatoires pour chaque observation.
y5 = np.array([0, 0, 0, 0, 5.0])
res5 = y5 - H5 @ y5
assert abs(res5[4] - 0.4) < 1e-12
b_sans = np.linalg.lstsq(X5[:4], y5[:4], rcond=None)[0]
assert abs((y5[4] - X5[4] @ b_sans) - V["F05_exact"]) < 1e-9
Xr = np.column_stack([np.ones(30), rng.standard_normal((30, 3))])
yr = rng.standard_normal(30)
Hr = Xr @ np.linalg.inv(Xr.T @ Xr) @ Xr.T
er = yr - Hr @ yr
for i in range(30):
    m_ = np.arange(30) != i
    bi = np.linalg.lstsq(Xr[m_], yr[m_], rcond=None)[0]
    assert abs((yr[i] - Xr[i] @ bi) - er[i] / (1 - Hr[i, i])) < 1e-9

# =====================================================================================
# EB2-A01 : boosting avec perte absolue, valeur de feuille = médiane
# =====================================================================================
r = [F(-2), F(1, 2), F(1), F(4), F(10)]
perte = lambda w: sum(abs(x - w) for x in r)
grille_w = [F(i, 100) for i in range(-500, 1201)]
meilleur = min(grille_w, key=perte)
assert meilleur == 1 and all(perte(w) > perte(F(1)) for w in grille_w if w != 1)
V["A01_exact"] = 1
fraction("A01_d_moyenne", sum(r) / 5)
fraction("A01_d_signes", F(sum(1 if x > 0 else -1 for x in r), 5))
fraction("A01_perte_mediane", perte(F(1)))
fraction("A01_perte_moyenne", perte(sum(r) / 5))

# =====================================================================================
# EB2-A02 : LSTM, chemin de gradient de la cellule
# =====================================================================================
V["A02_exact"] = 0.99**50
V["A02_rnn"] = 0.9**50
V["A02_d_lineaire"] = 1 - 50 * 0.01
V["A02_d_somme"] = 50 * 0.99
c_, eps_ = 1.0, 1e-6
cellule = lambda c0: (0.99**50) * c0 + sum(0.99**k for k in range(50)) * 0.3  # i_t g_t = 0,3 constant
assert abs((cellule(c_ + eps_) - cellule(c_)) / eps_ - V["A02_exact"]) < 1e-6

# =====================================================================================
# EB2-A03 : normalisation par lot, calcul d'une sortie
# =====================================================================================
lot = np.array([1.0, 2.0, 3.0, 6.0])
m3, v3 = lot.mean(), lot.var()
assert m3 == 3 and v3 == 3.5
V["A03_moyenne"] = float(m3)
V["A03_var"] = float(v3)
V["A03_exact"] = 2 * (6 - m3) / math.sqrt(v3) + 1
V["A03_d_n_moins_1"] = 2 * (6 - m3) / math.sqrt(lot.var(ddof=1)) + 1
V["A03_var_n_moins_1"] = float(lot.var(ddof=1))
V["A03_d_sans_affine"] = (6 - m3) / math.sqrt(v3)
V["A03_d_variance"] = 2 * (6 - m3) / v3 + 1

# =====================================================================================
# EB2-A04 : momentum, déplacement avec un gradient constant
# =====================================================================================
beta4, eta4 = F(9, 10), F(1, 100)
v4, depl, vitesses = F(0), F(0), []
for _ in range(3):
    v4 = beta4 * v4 + 1
    vitesses.append(v4)
    depl += eta4 * v4
assert vitesses == [1, F(19, 10), F(271, 100)] and depl == F(561, 10000)
V["A04_exact"] = float(depl)
V["A04_v2"] = float(vitesses[1])
V["A04_v3"] = float(vitesses[2])
V["A04_d_sans_moment"] = float(3 * eta4)
V["A04_d_dernier"] = float(eta4 * vitesses[2])
V["A04_d_limite"] = float(3 * eta4 / (1 - beta4))
V["A04_pas_limite"] = float(eta4 / (1 - beta4))

# =====================================================================================
# EB2-A05 : dimension minimale de l'espace de caractéristiques du noyau (1 + x.z)^2
# =====================================================================================
d = 10
V["A05_exact"] = 1 + d + d * (d + 1) // 2
assert V["A05_exact"] == 66 == math.comb(d + 2, 2)
V["A05_d_ordonne"] = 1 + d + d * d
V["A05_d_degre2"] = d * (d + 1) // 2
V["A05_d_carres"] = 1 + 2 * d
V["A05_lineaires"] = d
V["A05_carres"] = d
V["A05_croises"] = d * (d - 1) // 2
Xk = rng.standard_normal((300, d))
K = (1 + Xk @ Xk.T) ** 2
vs = np.linalg.svd(K, compute_uv=False)
rang = int((vs > vs[0] * 1e-10).sum())
assert rang == 66
V["A05_rang_gram"] = rang


def phi_poly(x):
    i, j = np.triu_indices(d, k=1)
    return np.concatenate([[1.0], math.sqrt(2) * x, x**2, math.sqrt(2) * x[i] * x[j]])


Phi_ = np.array([phi_poly(x) for x in Xk])
assert Phi_.shape[1] == 66 and np.allclose(Phi_ @ Phi_.T, K)

# =====================================================================================
# EB2-A06 : SVM à marge dure, largeur de la marge
# =====================================================================================
pts6 = np.array([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
y6 = np.array([1.0, 1.0, -1.0])
w6, b6 = np.array([1.0, 1.0]), -1.0
assert np.allclose(y6 * (pts6 @ w6 + b6), 1.0)  # réalisable, les trois contraintes saturées
# Optimalité : recherche sur une grille fine de (w1, w2, b) ; aucun point réalisable n'a une norme plus petite.
g6 = np.linspace(-3, 3, 161)
W1, W2, B6 = np.meshgrid(g6, g6, g6, indexing="ij")
realisable = ((2 * W1 + B6) >= 1 - 1e-12) & ((2 * W2 + B6) >= 1 - 1e-12) & (-B6 >= 1 - 1e-12)
assert np.sqrt(W1**2 + W2**2)[realisable].min() >= math.sqrt(2) - 1e-12
V["A06_norme_w"] = math.sqrt(2)
V["A06_exact"] = 2 / math.sqrt(2)
V["A06_d_demi"] = 1 / math.sqrt(2)
V["A06_d_points"] = 2.0
V["A06_d_positifs"] = float(np.linalg.norm(pts6[0] - pts6[1]))

# =====================================================================================
# EB2-A07 : moyenne mobile exponentielle, demi-vie
# =====================================================================================
lam7 = 0.94
V["A07_exact"] = math.log(0.5) / math.log(lam7)
V["A07_d_inverse"] = 1 / (1 - lam7)
V["A07_d_age"] = lam7 / (1 - lam7)
V["A07_d_lineaire"] = 0.5 / (1 - lam7)
V["A07_poids0"] = 1 - lam7
V["A07_moins_ln"] = -math.log(lam7)
V["A07_approx"] = math.log(2) / (1 - lam7)
k7 = sp.symbols("k7", integer=True, nonnegative=True)
assert sp.summation((1 - sp.Rational(94, 100)) * sp.Rational(94, 100) ** k7, (k7, 0, sp.oo)) == 1
assert sp.summation(k7 * (1 - sp.Rational(94, 100)) * sp.Rational(94, 100) ** k7, (k7, 0, sp.oo)) == sp.Rational(94, 6)
assert abs(lam7 ** V["A07_exact"] - 0.5) < 1e-12

# =====================================================================================
# EB2-A08 : ratio de Sharpe d'une combinaison de deux stratégies corrélées
# =====================================================================================
rho8 = sp.Rational(1, 2)
sr8 = 1 / sp.sqrt((1 + rho8) / 2)
V["A08_exact"] = float(sr8)
V["A08_d_moyenne"] = 1
V["A08_d_independance"] = math.sqrt(2)
V["A08_d_somme"] = 2
V["A08_var_comb"] = float((1 + rho8) / 2)
V["A08_limite"] = float(1 / sp.sqrt(rho8))
# Simulation : rendements annuels gaussiens d'espérance 1, d'écart-type 1, de corrélation 0,5.
Z8 = rng.standard_normal((2_000_000, 2))
R1 = 1 + Z8[:, 0]
R2 = 1 + 0.5 * Z8[:, 0] + math.sqrt(0.75) * Z8[:, 1]
comb = (R1 + R2) / 2
mc_moyenne("A08_moyenne", 1, comb)
mc_moyenne("A08_var", (1 + rho8) / 2, (comb - 1) ** 2)

# =====================================================================================
# EB2-C1 : exemple de l'énoncé, comportement de merge_asof, tests qui attrapent les erreurs visées
# =====================================================================================
DOSSIER = RACINE / "coding" / "examen2"
sys.path.insert(0, str(DOSSIER))
import tests as tests_c1  # noqa: E402
from _communs import cadre_cotations, cadre_transactions, comparer  # noqa: E402
from _correction import desequilibre  # noqa: E402

ex = desequilibre(cadre_transactions(tests_c1.EXEMPLE_T), cadre_cotations(tests_c1.EXEMPLE_C))
assert list(ex.index) == ["A", "B", "C"] and np.isnan(ex["C"])
assert abs(ex["A"] - 3 / 7) < 1e-15 and abs(ex["B"] - 0.2) < 1e-15
assert (100.5 + 99.5) / 2 == 100 and (50 + 50.5) / 2 == 50.25 and (51 + 51.5) / 2 == 51.25
fraction("C1_A", F(10 - 4, 10 + 4))
fraction("C1_B", F(3 - 2, 3 + 2))
# merge_asof : tolérance incluse, égalité exclue avec allow_exact_matches=False.
gauche = pd.DataFrame({"temps": np.array([1000, 6000, 6001], dtype="int64"), "actif": ["A"] * 3})
droite = pd.DataFrame({"tc": np.array([1000], dtype="int64"), "actif": ["A"], "achat": [1.0]})
mm = pd.merge_asof(gauche, droite, left_on="temps", right_on="tc", by="actif", allow_exact_matches=False, tolerance=5000)
assert np.isnan(mm["achat"][0]) and mm["achat"][1] == 1.0 and np.isnan(mm["achat"][2])
dire("merge_asof : écart 5000 retenu (tolérance incluse), 5001 rejeté, égalité exclue (pandas " + pd.__version__ + ")")


def sans_filtrage(t, c):
    """Erreur visée : jointure sur toutes les cotations, puis cotation invalide = non classée."""
    v = c.rename(columns={"temps": "tc"}).sort_values("tc")
    m = pd.merge_asof(t.sort_values("temps"), v, left_on="temps", right_on="tc", by="actif",
                      allow_exact_matches=False, tolerance=5000)
    ok = m["achat"] < m["vente"]
    mil = (m["achat"] + m["vente"]) / 2
    va = m["quantite"].where(ok & (m["prix"] > mil), 0).groupby(m["actif"]).sum()
    vv = m["quantite"].where(ok & (m["prix"] < mil), 0).groupby(m["actif"]).sum()
    tot = va + vv
    return ((va - vv) / tot.where(tot > 0)).reindex(sorted(t["actif"].unique())).astype(float)


def variante(exact=False, tolerance=5000, par_actif=True, egalite_vente=False):
    def f(t, c):
        if len(t) == 0:
            return pd.Series(dtype=float)
        v = c.loc[c["achat"] < c["vente"]].rename(columns={"temps": "tc"}).sort_values("tc")
        if not par_actif:
            v = v.drop(columns="actif")
        m = pd.merge_asof(t.sort_values("temps"), v, left_on="temps", right_on="tc",
                          by="actif" if par_actif else None, allow_exact_matches=exact, tolerance=tolerance)
        mil = (m["achat"] + m["vente"]) / 2
        va = m["quantite"].where(m["prix"] > mil, 0)
        vv = m["quantite"].where((m["prix"] <= mil) if egalite_vente else (m["prix"] < mil), 0)
        va, vv = va.groupby(m["actif"]).sum(), vv.groupby(m["actif"]).sum()
        tot = va + vv
        return ((va - vv) / tot.where(tot > 0)).reindex(sorted(t["actif"].unique())).astype(float)
    return f


caches = tests_c1.cas_caches()
erreurs = {
    "cotation de même instant acceptée": variante(exact=True),
    "sans tolérance de 5 s": variante(tolerance=None),
    "tolérance stricte (moins de 5000 ms)": variante(tolerance=4999),
    "cotation d'un autre actif": variante(par_actif=False),
    "prix au milieu compté à la vente": variante(egalite_vente=True),
    "cotations invalides non filtrées avant la jointure": sans_filtrage,
}
for nom, f in erreurs.items():
    rates = []
    for cas_ in caches:
        try:
            ok = comparer(f(*cas_["args"]), cas_["attendu"])
        except Exception:  # noqa: BLE001 : une erreur d'exécution compte comme un test raté
            ok = False
        if not ok:
            rates.append(cas_["nom"])
    assert rates, f"aucun test caché n'attrape l'erreur : {nom}"
    dire(f"C1 : « {nom} » échoue à {len(rates)} test(s) caché(s), dont « {rates[0]} »")
V["C1_nb_caches"] = len(caches)

Path(__file__).with_name("examen_blanc_2.out.json").write_bytes((json.dumps(V, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
dire(f"{len(V)} valeurs exportées dans verif/examen_blanc_2.out.json.")
dire("Toutes les vérifications passent.")
Path(__file__).with_name("examen_blanc_2.out.txt").write_bytes(("\n".join(LIGNES) + "\n").encode("utf-8"))  # fins de ligne Unix
