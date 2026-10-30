"""Vérification de tous les nombres de la séance de mathématiques 1 (leçon L-PROBAS-1 et questions S1M-01 à S1M-12).

Chaque probabilité ou espérance a deux vérifications indépendantes :
  1. un calcul exact (fractions, énumération complète, système linéaire résolu en rationnels, ou sympy) ;
  2. une simulation Monte-Carlo, dont l'écart à l'exact doit rester inférieur à 4 erreurs standard.
Les distracteurs et l'arithmétique citée dans les corrections sont aussi calculés ici.

Graine fixe : la sortie est identique à chaque exécution.
Sorties : verif/seance_math_1.out.json (lu par outils/construire.py) et verif/seance_math_1.out.txt (ce résumé).
"""
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp

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


def resoudre_premier_pas(bas, haut, p, cible):
    """Résout en rationnels le système du premier pas sur {bas, ..., haut}.

    cible = "proba" : h(bas) = 0, h(haut) = 1, h(i) = p h(i+1) + q h(i-1) ;
    cible = "duree" : d(bas) = d(haut) = 0, d(i) = 1 + p d(i+1) + q d(i-1).
    Renvoie un dictionnaire position -> valeur exacte. La matrice est inversible (unicité, principe du maximum).
    """
    p = Fraction(p)
    q = 1 - p
    etats = list(range(bas, haut + 1))
    m = sp.zeros(len(etats), len(etats))
    b = sp.zeros(len(etats), 1)
    for k, i in enumerate(etats):
        m[k, k] = 1
        if i in (bas, haut):
            b[k] = 1 if (cible == "proba" and i == haut) else 0
        else:
            m[k, k + 1] = -sp.Rational(p.numerator, p.denominator)
            m[k, k - 1] = -sp.Rational(q.numerator, q.denominator)
            b[k] = 0 if cible == "proba" else 1
    assert m.det() != 0
    x = m.LUsolve(b)
    return {i: Fraction(int(sp.fraction(x[k])[0]), int(sp.fraction(x[k])[1])) for k, i in enumerate(etats)}


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
    """Probabilité de finir en n en partant de i, barrières 0 et n, formule fermée."""
    p = Fraction(p)
    q = 1 - p
    if p == q:
        return Fraction(i, n)
    r = q / p
    return (1 - r**i) / (1 - r**n)


def d_formule(i, n, p):
    """Espérance du temps d'arrêt, formule fermée."""
    p = Fraction(p)
    q = 1 - p
    if p == q:
        return Fraction(i * (n - i))
    return (n * h_formule(i, n, p) - i) / (p - q)


# Contrôle préalable des formules fermées contre le système linéaire, sur une grille de cas.
for n_ in range(2, 9):
    for p_ in [Fraction(1, 2), Fraction(2, 3), Fraction(2, 5), Fraction(3, 10), Fraction(7, 10)]:
        hs = resoudre_premier_pas(0, n_, p_, "proba")
        ds = resoudre_premier_pas(0, n_, p_, "duree")
        for i_ in range(n_ + 1):
            assert hs[i_] == h_formule(i_, n_, p_) and ds[i_] == d_formule(i_, n_, p_), (n_, p_, i_)
dire("Formules fermées de h et d conformes au système linéaire pour n de 2 à 8 et 5 valeurs de p.")

# =====================================================================================
# LEÇON, OUTIL 1 : linéarité de l'espérance, indicatrices
# =====================================================================================
# Points fixes d'une permutation uniforme : énumération complète pour n = 2 à 7.
for n_ in range(2, 8):
    perms = list(itertools.permutations(range(n_)))
    nb = [sum(s[i] == i for i in range(n_)) for s in perms]
    moy = Fraction(sum(nb), len(perms))
    e2 = Fraction(sum(x * x for x in nb), len(perms))
    assert moy == 1 and e2 == 2 and e2 - moy**2 == 1
    # Somme des variances des indicatrices : n (1/n)(1 - 1/n) = 1 - 1/n, différente de Var(N) = 1.
    assert n_ * Fraction(1, n_) * (1 - Fraction(1, n_)) == 1 - Fraction(1, n_)
perms4 = list(itertools.permutations(range(4)))
nb4 = [sum(s[i] == i for i in range(4)) for s in perms4]
V["La_pf_nb_perm"] = len(perms4)
V["La_pf_moy_enum"] = float(Fraction(sum(nb4), len(perms4)))
V["La_pf_e2_enum"] = float(Fraction(sum(x * x for x in nb4), len(perms4)))
V["La_pf_var_enum"] = V["La_pf_e2_enum"] - V["La_pf_moy_enum"] ** 2
# Indicatrices de deux points fixes : dépendantes (n = 4).
p1 = Fraction(sum(s[0] == 0 for s in perms4), len(perms4))
p12 = Fraction(sum(s[0] == 0 and s[1] == 1 for s in perms4), len(perms4))
assert p1 == Fraction(1, 4) and p12 == Fraction(1, 12) and p12 != p1 * p1
fraction("La_pf_p1", p1)
fraction("La_pf_p12", p12)
fraction("La_pf_p1_carre", p1 * p1)
# Monte-Carlo, n = 10 : moyenne et second moment du nombre de points fixes.
n_sim = 400_000
perm = np.argsort(rng.random((n_sim, 10)), axis=1)
nb_fixes = (perm == np.arange(10)).sum(axis=1)
V["La_pf_n_mc"] = 10
mc_moyenne("La_pf_esp_n10", 1, nb_fixes)
mc_moyenne("La_pf_e2_n10", 2, nb_fixes**2)

# Motif PP dans 10 lancers : énumération des 2^10 suites et Monte-Carlo.
L = 10
suites = list(itertools.product([0, 1], repeat=L))  # 1 = pile
compte = [sum(s[j] == 1 and s[j + 1] == 1 for j in range(L - 1)) for s in suites]
esp_motif = Fraction(sum(compte), len(suites))
assert esp_motif == Fraction(L - 1, 4)
V["La_motif_pos"] = L - 1
fraction("La_motif_p", Fraction(1, 4))
fraction("La_motif_esp", esp_motif)
joint = Fraction(sum(s[0] == s[1] == s[2] == 1 for s in suites), len(suites))
assert joint == Fraction(1, 8) and joint != Fraction(1, 16)
fraction("La_motif_joint", joint)
fraction("La_motif_prod", Fraction(1, 4) ** 2)
lancers = rng.integers(0, 2, size=(n_sim, L))
mc_moyenne("La_motif_esp", esp_motif, (lancers[:, :-1] & lancers[:, 1:]).sum(axis=1))
mc_proba("La_motif_joint", joint, lancers[:, 0] & lancers[:, 1] & lancers[:, 2])

# Point de contrôle « urne » : 4 boules tirées sans remise parmi 10 numérotées de 1 à 10.
sous_ensembles = list(itertools.combinations(range(1, 11), 4))
esp_somme = Fraction(sum(sum(c) for c in sous_ensembles), len(sous_ensembles))
assert esp_somme == Fraction(4, 10) * 55 == 22
V["Ca_nb_sous_ens"] = len(sous_ensembles)
V["Ca_somme_tot"] = sum(range(1, 11))
V["Ca_exact"] = float(esp_somme)
V["Ca_d_un"] = float(Fraction(sum(range(1, 11)), 10))
V["Ca_d_moy_fausse"] = 4 * (10 // 2)
fraction("Ca_p_tiree", Fraction(4, 10))
tirages = np.argsort(rng.random((n_sim, 10)), axis=1)[:, :4] + 1
mc_moyenne("Ca", esp_somme, tirages.sum(axis=1))

# =====================================================================================
# LEÇON, OUTIL 2 : conditionnement sur le premier pas
# =====================================================================================
# Exemple : départ 2, barrières 0 et 5.
h_sym = resoudre_premier_pas(0, 5, Fraction(1, 2), "proba")[2]
d_sym = resoudre_premier_pas(0, 5, Fraction(1, 2), "duree")[2]
assert h_sym == Fraction(2, 5) and d_sym == 6
V["Lb_sym_h"] = float(h_sym)
V["Lb_sym_d"] = int(d_sym)
haut_, pas_ = simuler_marche(2, 0, 5, 0.5, 200_000)
mc_proba("Lb_sym_h", h_sym, haut_)
mc_moyenne("Lb_sym_d", d_sym, pas_)

p_b = Fraction(2, 3)
r_b = (1 - p_b) / p_b
assert r_b == Fraction(1, 2)
h_b = resoudre_premier_pas(0, 5, p_b, "proba")[2]
d_b = resoudre_premier_pas(0, 5, p_b, "duree")[2]
assert h_b == (1 - r_b**2) / (1 - r_b**5) == Fraction(24, 31)
assert d_b == (5 * h_b - 2) / (p_b - (1 - p_b)) == Fraction(174, 31)
V["Lb_r2_inv"] = int(1 / r_b**2)
V["Lb_r5_inv"] = int(1 / r_b**5)
fraction("Lb_un_moins_r2", 1 - r_b**2)
fraction("Lb_un_moins_r5", 1 - r_b**5)
fraction("Lb_h", h_b)
fraction("Lb_d", d_b)
fraction("Lb_5h", 5 * h_b)
fraction("Lb_5h_moins_2", 5 * h_b - 2)
fraction("Lb_p_moins_q", p_b - (1 - p_b))
haut_, pas_ = simuler_marche(2, 0, 5, float(p_b), 200_000)
mc_proba("Lb_h", h_b, haut_)
mc_moyenne("Lb_d", d_b, pas_)
# Identité de Wald sur la même simulation : E[Z_T] - i = (p - q) E[T].
z_final = np.where(haut_, 5, 0)
mc_moyenne("Lb_wald", 0, (z_final - 2) - float(p_b - (1 - p_b)) * pas_)

# Barrière haute infinie, p = 2/3, départ 2 : probabilité de ruine r^2 = 1/4.
fraction("Lb_ruine_inf", r_b**2)
x_ = sp.symbols("x", positive=True)
ruine_n = lambda n_: (sp.Rational(1, 2) ** 2 - sp.Rational(1, 2) ** n_) / (1 - sp.Rational(1, 2) ** n_)
assert sp.limit((sp.Rational(1, 4) - sp.Rational(1, 2) ** x_) / (1 - sp.Rational(1, 2) ** x_), x_, sp.oo) == sp.Rational(1, 4)
haut_, _ = simuler_marche(2, 0, 60, float(p_b), 200_000)  # barrière 60 : ruine avant 60 = 1/4 à 1e-17 près
mc_proba("Lb_ruine_inf", float(ruine_n(60)), ~haut_)
assert abs(float(ruine_n(60)) - 0.25) < 1e-15

# Attente du motif PF (pièce équilibrée) : e0 = 1 + e1/2 + e0/2, e1 = 1 + e1/2 (+ 0 si F).
e0, e1 = sp.symbols("e0 e1")
sol = sp.solve([sp.Eq(e0, 1 + e1 / 2 + e0 / 2), sp.Eq(e1, 1 + e1 / 2)], [e0, e1])
assert sol[e0] == 4 and sol[e1] == 2
V["Lb_pf_e0"] = int(sol[e0])
V["Lb_pf_e1"] = int(sol[e1])
V["Lb_pf_borne"] = 2 * 4  # blocs disjoints de 2 lancers, chacun égal à PF avec probabilité 1/4


def attente_motif(motif, n_sim, longueur=200):
    """Nombre de lancers jusqu'à la première apparition du motif (1 = pile, 0 = face)."""
    l_ = len(motif)
    x = rng.integers(0, 2, size=(n_sim, longueur))
    ok = np.ones((n_sim, longueur - l_ + 1), dtype=bool)
    for k, c in enumerate(motif):
        ok &= x[:, k:longueur - l_ + 1 + k] == c
    assert ok.any(axis=1).all()  # la probabilité de ne pas voir le motif en 200 lancers est négligeable
    return ok.argmax(axis=1) + l_


mc_moyenne("Lb_pf", 4, attente_motif([1, 0], 200_000))

# Point de contrôle « derive » : p = 0,6, départ 5, barrières 0 et 10.
h_c = resoudre_premier_pas(0, 10, Fraction(3, 5), "proba")[5]
r_c = Fraction(2, 5) / Fraction(3, 5)
assert h_c == 1 / (1 + r_c**5) == Fraction(243, 275)
fraction("Cb_h", h_c)
fraction("Cb_r5", r_c**5)
fraction("Cb_d_inverse", 1 - h_c)
assert resoudre_premier_pas(0, 10, Fraction(2, 5), "proba")[5] == 1 - h_c  # p et q échangés
V["Cb_d_cinq"] = float(Fraction(3, 5) ** 5)
haut_, _ = simuler_marche(5, 0, 10, 0.6, 200_000)
mc_proba("Cb_h", h_c, haut_)

# =====================================================================================
# LEÇON, OUTIL 3 : statistiques d'ordre de lois uniformes
# =====================================================================================
a_, b_, t_ = sp.symbols("a b t", nonnegative=True)
for a in range(0, 6):
    for b in range(0, 6):
        assert sp.integrate(t_**a * (1 - t_) ** b, (t_, 0, 1)) == sp.Rational(math.factorial(a) * math.factorial(b), math.factorial(a + b + 1))
for n_ in range(1, 9):
    for k in range(1, n_ + 1):
        # Par la formule de la queue et la binomiale (preuve de la leçon)...
        queue = sp.integrate(sum(sp.binomial(n_, j) * t_**j * (1 - t_) ** (n_ - j) for j in range(k)), (t_, 0, 1))
        # ... et par la densité bêta, méthode indépendante.
        dens = sp.factorial(n_) / (sp.factorial(k - 1) * sp.factorial(n_ - k)) * t_ ** (k - 1) * (1 - t_) ** (n_ - k)
        assert queue == sp.integrate(t_ * dens, (t_, 0, 1)) == sp.Rational(k, n_ + 1)
dire("E[U_(k)] = k/(n+1) vérifié pour 1 <= k <= n <= 8 par deux intégrations exactes ; intégrale bêta vérifiée pour a, b <= 5.")
n5 = 5
fraction("Lc_k2", Fraction(2, n5 + 1))
fraction("Lc_max", Fraction(n5, n5 + 1))
fraction("Lc_min", Fraction(1, n5 + 1))
fraction("Lc_etendue", Fraction(n5 - 1, n5 + 1))
fraction("Lc_ecart", Fraction(1, n5 + 1))
V["Lc_p_max"] = float(Fraction(9, 10) ** 5)
fraction("Lc_deux", Fraction(1, 3))
u = np.sort(rng.random((400_000, n5)), axis=1)
mc_moyenne("Lc_k2", Fraction(2, 6), u[:, 1])
mc_moyenne("Lc_max", Fraction(5, 6), u[:, 4])
mc_moyenne("Lc_min", Fraction(1, 6), u[:, 0])
mc_moyenne("Lc_etendue", Fraction(4, 6), u[:, 4] - u[:, 0])
mc_moyenne("Lc_ecart", Fraction(1, 6), u[:, 3] - u[:, 2])
mc_proba("Lc_p_max", V["Lc_p_max"], u[:, 4] <= 0.9)
deux = rng.random((400_000, 2))
mc_moyenne("Lc_deux", Fraction(1, 3), np.abs(deux[:, 0] - deux[:, 1]))
# Contre-exemple : maximum de deux exponentielles de paramètre 1.
xs, ys = sp.symbols("xs ys", positive=True)
e_min = sp.integrate(sp.exp(-2 * xs), (xs, 0, sp.oo))  # P(min > x) = e^{-2x}
e_max = sp.integrate(sp.integrate(sp.Max(xs, ys) * sp.exp(-xs - ys), (xs, 0, sp.oo)), (ys, 0, sp.oo))
assert e_min == sp.Rational(1, 2) and sp.simplify(e_max - sp.Rational(3, 2)) == 0
V["Lc_exp_min"] = 0.5
V["Lc_exp_max"] = 1.5
V["Lc_exp_quantile"] = math.log(3)  # F^{-1}(2/3) pour F(x) = 1 - e^{-x}
assert abs(1 - math.exp(-math.log(3)) - 2 / 3) < 1e-15
ex = rng.exponential(1.0, size=(400_000, 2))
mc_moyenne("Lc_exp_max", 1.5, ex.max(axis=1))
mc_moyenne("Lc_exp_min", 0.5, ex.min(axis=1))

# Point de contrôle « ordre » : 7 uniformes, deuxième plus grand = U_(6).
fraction("Cc_exact", Fraction(6, 8))
fraction("Cc_d_petit", Fraction(2, 8))
fraction("Cc_d_n", Fraction(6, 7))
fraction("Cc_d_max", Fraction(7, 8))
u7 = np.sort(rng.random((400_000, 7)), axis=1)
mc_moyenne("Cc", Fraction(6, 8), u7[:, 5])

# =====================================================================================
# LEÇON, OUTIL 4 : complémentaire, Boole, Bonferroni
# =====================================================================================
m_, p_ = 50, Fraction(5, 100)
exact_d = 1 - (1 - p_) ** m_
V["Ld_exact"] = float(exact_d)
V["Ld_aucun"] = float((1 - p_) ** m_)
V["Ld_mp"] = float(m_ * p_)
V["Ld_poisson"] = 1 - math.exp(-float(m_ * p_))
V["Ld_ecart"] = V["Ld_exact"] - V["Ld_poisson"]
V["Ld_borne"] = float(m_ * p_**2) * math.exp(-float(m_ * p_))
V["Ld_mp2"] = float(m_ * p_**2)
assert 0 <= V["Ld_ecart"] <= V["Ld_borne"]
V["Ld_seuil_bonf"] = float(Fraction(5, 100) / m_)
V["Ld_fwer_bonf"] = float(1 - (1 - Fraction(5, 100) / m_) ** m_)
assert V["Ld_fwer_bonf"] <= 0.05
rej = rng.random((200_000, m_))
mc_proba("Ld_exact", exact_d, (rej < 0.05).any(axis=1))
mc_proba("Ld_fwer_bonf", V["Ld_fwer_bonf"], (rej < 0.001).any(axis=1))
# Encadrement 1 - e^{-mp} <= 1 - (1-p)^m <= 1 - e^{-mp} + m p^2 e^{-mp}, pour 0 <= p <= 1/2, sur une grille.
for m in [1, 2, 5, 10, 50, 200, 1000]:
    for p in np.linspace(0, 0.5, 51):
        vrai = 1 - (1 - p) ** m
        assert 1 - math.exp(-m * p) - 1e-15 <= vrai <= 1 - math.exp(-m * p) + m * p * p * math.exp(-m * p) + 1e-15
        # Boole surestime 1 - e^{-mp} d'au plus (mp)^2 / 2, et l'exact d'au plus autant.
        assert m * p - vrai <= (m * p) ** 2 / 2 + 1e-12
# Inégalités élémentaires utilisées dans la preuve, sur une grille.
for p in np.linspace(0, 0.5, 501):
    assert math.log1p(-p) >= -p - p * p - 1e-15
for y in np.linspace(0, 20, 2001):
    assert math.exp(-y) >= 1 - y and math.exp(-y) <= 1 - y + y * y / 2 + 1e-15
dire("Encadrement de 1 - (1-p)^m et inégalités de la preuve vérifiés sur une grille.")

# Point de contrôle « bonferroni » : 10 tests.
V["Cd_seuil"] = 0.05 / 10
V["Cd_fwer_sans"] = float(1 - Fraction(95, 100) ** 10)
V["Cd_borne_sans"] = 10 * 0.05
rej10 = rng.random((200_000, 10))
mc_proba("Cd_fwer_sans", V["Cd_fwer_sans"], (rej10 < 0.05).any(axis=1))

# =====================================================================================
# QUESTIONS
# =====================================================================================
n_sim = 400_000

# S1M-01 : motif PFP dans 12 lancers.
L = 12
motif = (1, 0, 1)
suites = list(itertools.product([0, 1], repeat=L))
compte = [sum(s[j:j + 3] == motif for j in range(L - 2)) for s in suites]
q01 = Fraction(sum(compte), len(suites))
assert q01 == Fraction(L - 2, 8)
assert sum(1 for j in range(3) if (1, 0, 1, 0, 1)[j:j + 3] == motif) == 2  # PFPFP contient 2 occurrences
V["Q01_pos"] = L - 2
V["Q01_p_den"] = 2**3
V["Q01_exact"] = float(q01)
V["Q01_d_12pos"] = float(Fraction(L, 8))
V["Q01_d_au_moins_un"] = float(1 - Fraction(7, 8) ** (L - 2))
V["Q01_d_proba1"] = float(Fraction(1, 8))
p_au_moins_un = Fraction(sum(c >= 1 for c in compte), len(suites))
V["Q01_ex"] = sum(1 for j in range(3) if (1, 0, 1, 0, 1)[j:j + 3] == motif)
V["La_motif_ex"] = sum(1 for j in range(2) if (1, 1, 1)[j:j + 2] == (1, 1))
assert V["Q01_ex"] == 2 and V["La_motif_ex"] == 2
V["Q01_p_au_moins_un_vraie"] = float(p_au_moins_un)  # probabilité exacte d'au moins une occurrence, par énumération
dire(f"S1M-01 : P(au moins une occurrence) exacte = {float(p_au_moins_un):.6f}, calcul sous indépendance supposée = {V['Q01_d_au_moins_un']:.6f}")
x = rng.integers(0, 2, size=(n_sim, L))
occ = (x[:, :-2] == 1) & (x[:, 1:-1] == 0) & (x[:, 2:] == 1)
mc_moyenne("Q01", q01, occ.sum(axis=1))

# S1M-02 : 10 tirages avec remise parmi 20 numéros, nombre de numéros distincts.
N_, T_ = 20, 10
q02 = N_ * (1 - Fraction(N_ - 1, N_) ** T_)
# Méthode exacte indépendante : loi du nombre K de numéros distincts, P(K = k) = C(N,k) k! S(T,k) / N^T.
loi = {k: Fraction(math.comb(N_, k) * math.factorial(k) * int(sp.functions.combinatorial.numbers.stirling(T_, k)), N_**T_) for k in range(1, T_ + 1)}
assert sum(loi.values()) == 1 and sum(k * pk for k, pk in loi.items()) == q02
V["Q02_exact"] = float(q02)
V["Q02_p_absent"] = float(Fraction(N_ - 1, N_) ** T_)
V["Q02_d_absents"] = float(N_ * Fraction(N_ - 1, N_) ** T_)
V["Q02_nb_paires"] = math.comb(T_, 2)
V["Q02_d_paires"] = float(T_ - Fraction(math.comb(T_, 2), N_))
V["Q02_d_uniques"] = float(T_ * Fraction(N_ - 1, N_) ** (T_ - 1))
V["Q02_ratio"] = float(Fraction(N_ - 1, N_))
# Vérification du distracteur « numéros obtenus une seule fois » par Monte-Carlo, et du résultat principal.
t = rng.integers(0, N_, size=(n_sim, T_))
tri = np.sort(t, axis=1)
distincts = 1 + (np.diff(tri, axis=1) != 0).sum(axis=1)
mc_moyenne("Q02", q02, distincts)
comptes = np.zeros((n_sim, N_), dtype=np.int64)
np.add.at(comptes, (np.repeat(np.arange(n_sim), T_), t.ravel()), 1)
mc_moyenne("Q02_d_uniques", V["Q02_d_uniques"], (comptes == 1).sum(axis=1))

# S1M-03 : variance du nombre de points fixes, n = 10.
n_ = 10
e2 = Fraction(1) + n_ * (n_ - 1) * Fraction(1, n_ * (n_ - 1))
q03 = e2 - 1
assert q03 == 1
V["Q03_exact"] = int(q03)
V["Q03_e2"] = int(e2)
V["Q03_nb_couples"] = n_ * (n_ - 1)
V["Q03_d_indep"] = float(n_ * Fraction(1, n_) * (1 - Fraction(1, n_)))
V["Q03_d_e2"] = int(e2)
V["Q03_d_paires"] = float(1 + math.comb(n_, 2) * Fraction(1, n_ * (n_ - 1)) - 1)
V["Q03_nb_paires"] = math.comb(n_, 2)
fraction("Q03_e2_paires", 1 + math.comb(n_, 2) * Fraction(1, n_ * (n_ - 1)))
perm = np.argsort(rng.random((n_sim, n_)), axis=1)
nf = (perm == np.arange(n_)).sum(axis=1).astype(float)
# Variance empirique : erreur standard par la méthode delta, estimée sur l'échantillon.
c = nf - nf.mean()
var_mc = (c**2).mean()
se_var = np.sqrt(((c**2 - var_mc) ** 2).mean() / n_sim)
controle_mc("Q03", 1, var_mc, se_var)

# S1M-04 : marche symétrique, départ 2, barrières -3 et 6.
d04 = resoudre_premier_pas(-3, 6, Fraction(1, 2), "duree")[2]
assert d04 == (2 - (-3)) * (6 - 2) == 20
V["Q04_exact"] = int(d04)
V["Q04_dist_bas"] = 2 - (-3)
V["Q04_dist_haut"] = 6 - 2
V["Q04_largeur"] = 6 - (-3)
V["Q04_i_translate"] = 2 - (-3)
V["Q04_d_produit"] = 2 * 6
V["Q04_d_sans_translation"] = 2 * (6 - 2)
V["Q04_d_min"] = 6 - 2
_, pas_ = simuler_marche(2, -3, 6, 0.5, 200_000)
mc_moyenne("Q04", d04, pas_)

# S1M-05 et S1M-06 : joueur, départ 3, barrières 0 et 6, gain avec probabilité 0,4.
p5 = Fraction(2, 5)
r5 = (1 - p5) / p5
h05 = resoudre_premier_pas(0, 6, p5, "proba")[3]
d06 = resoudre_premier_pas(0, 6, p5, "duree")[3]
assert h05 == (1 - r5**3) / (1 - r5**6) == 1 / (1 + r5**3) == Fraction(8, 35)
assert d06 == (6 * h05 - 3) / (p5 - (1 - p5)) == Fraction(57, 7)
V["Q05_r"] = float(r5)
fraction("Q05_r3", r5**3)
fraction("Q05_r6", r5**6)
fraction("Q05_un_moins_r3", 1 - r5**3)
fraction("Q05_un_moins_r6", 1 - r5**6)
fraction("Q05_un_plus_r3", 1 + r5**3)
fraction("Q05_exact", h05)
V["Q05_d_sym"] = 3 / 6
fraction("Q05_d_inverse", resoudre_premier_pas(0, 6, 1 - p5, "proba")[3])
assert V["Q05_d_inverse"] == float(1 - h05)
V["Q05_d_trois"] = float(p5**3)
haut_, pas_ = simuler_marche(3, 0, 6, float(p5), 400_000)
mc_proba("Q05", h05, haut_)
fraction("Q06_exact", d06)
fraction("Q06_nh", 6 * h05)
fraction("Q06_nh_moins_i", 6 * h05 - 3)
V["Q06_p_moins_q"] = float(p5 - (1 - p5))
V["Q06_d_sym"] = 3 * (6 - 3)
V["Q06_d_une_barriere"] = float(3 / (1 - 2 * p5))
fraction("Q06_d_signe", (3 + 6 * h05) / ((1 - p5) - p5))
fraction("Q06_borne", Fraction(6) / p5**6)  # majoration par blocs : n / p^n
mc_moyenne("Q06", d06, pas_)
assert float(d06) <= V["Q06_borne"] and float(d06) <= V["Q06_d_une_barriere"]
# Durée moyenne jusqu'à la ruine sans barrière haute (p < 1/2) : i / (q - p) = 15.
# Limite exacte de d_n(3) quand n tend vers l'infini, puis simulation avec une barrière haute inaccessible.
assert abs(float(d_formule(3, 200, p5)) - 15) < 1e-12
assert all(d_formule(3, n_, p5) <= d_formule(3, n_ + 1, p5) for n_ in range(4, 60))  # T_n croît avec n
_, pas_ = simuler_marche(3, 0, 10**6, float(p5), 200_000)
mc_moyenne("Q06_d_une_barriere", 15, pas_)
V["Q02_ex_distincts"] = len({4, 7, 4, 4, 12, 7, 1, 1, 9, 20})
assert V["Q02_ex_distincts"] == 6
V["Q03_n2"] = 10 * 10

# S1M-07 : attente de deux piles consécutifs.
e0, e1 = sp.symbols("e0 e1")
sol = sp.solve([sp.Eq(e0, 1 + e1 / 2 + e0 / 2), sp.Eq(e1, 1 + sp.Rational(1, 2) * 0 + e0 / 2)], [e0, e1])  # après P en état 1 : arrêt (coût restant 0)
assert sol[e0] == 6 and sol[e1] == 4
V["Q07_exact"] = int(sol[e0])
V["Q07_e1"] = int(sol[e1])
V["Q07_d_inverse"] = 4  # 1 / P(PP) = 1 / (1/4)
V["Q07_d_blocs"] = 2 * 4
V["Q07_d_naif"] = 2 + 1
V["Q07_borne"] = 2 * 4
mc_moyenne("Q07", 6, attente_motif([1, 1], 400_000))

# S1M-08 : 5 uniformes, deuxième plus grand = U_(4).
fraction("Q08_exact", Fraction(4, 6))
fraction("Q08_d_petit", Fraction(2, 6))
fraction("Q08_d_n", Fraction(4, 5))
fraction("Q08_d_max", Fraction(5, 6))
u5 = np.sort(rng.random((n_sim, 5)), axis=1)
mc_moyenne("Q08", Fraction(4, 6), u5[:, 3])

# S1M-09 : 6 uniformes sur [0, 2], étendue.
q09 = 2 * Fraction(6 - 1, 6 + 1)
fraction("Q09_exact", q09)
fraction("Q09_d_echelle", Fraction(5, 7))
fraction("Q09_d_max", 2 * Fraction(6, 7))
fraction("Q09_d_n", 2 * Fraction(5, 6))
fraction("Q09_e_max", 2 * Fraction(6, 7))
fraction("Q09_e_min", 2 * Fraction(1, 7))
y_ = sp.symbols("y_", positive=True)
assert sp.integrate((1 - (y_ / 2) ** 6) - (1 - y_ / 2) ** 6, (y_, 0, 2)) == sp.Rational(10, 7)  # E[max] - E[min] par la formule de la queue
u6 = 2 * rng.random((n_sim, 6))
mc_moyenne("Q09", q09, u6.max(axis=1) - u6.min(axis=1))

# S1M-10 : 4 uniformes, toutes dans [0,2 ; 0,9].
q10 = Fraction(7, 10) ** 4
V["Q10_exact"] = float(q10)
V["Q10_largeur"] = float(Fraction(9, 10) - Fraction(2, 10))
V["Q10_d_indep"] = float(Fraction(9, 10) ** 4 * Fraction(8, 10) ** 4)
V["Q10_d_diff"] = float(Fraction(9, 10) ** 4 - Fraction(2, 10) ** 4)
V["Q10_d_un"] = float(Fraction(7, 10))
V["Q10_p_max"] = float(Fraction(9, 10) ** 4)
V["Q10_p_min"] = float(Fraction(8, 10) ** 4)
u4 = rng.random((n_sim, 4))
mc_proba("Q10", q10, (u4.min(axis=1) >= 0.2) & (u4.max(axis=1) <= 0.9))

# S1M-11 : 200 signaux au niveau 0,5 %.
m11, p11 = 200, Fraction(5, 1000)
q11 = 1 - (1 - p11) ** m11
V["Q11_exact"] = float(q11)
V["Q11_aucun"] = float((1 - p11) ** m11)
V["Q11_mp"] = float(m11 * p11)
V["Q11_poisson"] = 1 - math.exp(-float(m11 * p11))
V["Q11_mp2"] = float(m11 * p11**2)
V["Q11_borne"] = float(m11 * p11**2) * math.exp(-float(m11 * p11))
V["Q11_haut"] = V["Q11_poisson"] + V["Q11_borne"]
V["Q11_e_moins_un"] = math.exp(-1)
V["Q11_d_boole"] = float(m11 * p11)
V["Q11_d_un"] = float(p11)
assert V["Q11_poisson"] <= V["Q11_exact"] <= V["Q11_haut"] < 0.635 and round(V["Q11_exact"], 2) == round(V["Q11_poisson"], 2) == 0.63
rej = rng.random((200_000, m11)) < float(p11)
mc_proba("Q11", q11, rej.any(axis=1))

# S1M-12 : 25 tests, Bonferroni contre Šidák, dépendance quelconque.
m12, alpha = 25, 0.05
t_bonf = alpha / m12
t_sidak = 1 - (1 - alpha) ** (1 / m12)
assert t_sidak > t_bonf and m12 * t_sidak > alpha and m12 * t_sidak <= 1
assert abs(1 - (1 - t_sidak) ** m12 - alpha) < 1e-15
V["Q12_bonf"] = t_bonf
V["Q12_sidak"] = t_sidak
V["Q12_fwer_sidak_disjoint"] = m12 * t_sidak
V["Q12_fwer_005_indep"] = float(1 - Fraction(95, 100) ** m12)
V["Q12_borne_bonf"] = m12 * t_bonf
# Construction de rejets disjoints : U uniforme, P_j = partie fractionnaire de U - (j-1) t ; chaque P_j est uniforme.
U = rng.random(400_000)
for t_niv, nom in [(t_sidak, "Q12_disjoint_sidak"), (t_bonf, "Q12_disjoint_bonf")]:
    pv = np.mod(U[:, None] - np.arange(m12)[None, :] * t_niv, 1.0)
    # Chaque test rejette à tort avec probabilité t : contrôle de P(P_j <= t) pour j = 1 et j = 25.
    for j in [0, m12 - 1]:
        mc_proba(f"{nom}_marg{j + 1}", t_niv, pv[:, j] <= t_niv)
    rejets = pv <= t_niv
    assert (rejets.sum(axis=1) <= 1).all()  # jamais deux rejets simultanés
    mc_proba(nom, m12 * t_niv, rejets.any(axis=1))
pv_ind = rng.random((200_000, m12))
mc_proba("Q12_indep_005", V["Q12_fwer_005_indep"], (pv_ind <= 0.05).any(axis=1))
mc_proba("Q12_indep_sidak", alpha, (pv_ind <= t_sidak).any(axis=1))

# Nombres dérivés des données, cités dans les textes.
V["Q11_q"] = float(1 - p11)
V["Q10_q_min"] = float(1 - Fraction(2, 10))
V["Q12_q"] = float(1 - Fraction(5, 100))
V["Ld_q"] = float(1 - p_)
V["Cd_q"] = float(1 - Fraction(5, 100))
V["Q06_q_moins_p"] = float((1 - p5) - p5)

# Arithmétique citée dans les corrections.
assert Fraction(6 * 8, 35) == 6 * h05 and Fraction(48, 35) - 3 == Fraction(-57, 35)
assert Fraction(-57, 35) / Fraction(-1, 5) == Fraction(57, 7)
assert (Fraction(27, 8) - 1) == Fraction(19, 8) and (Fraction(729, 64) - 1) == Fraction(665, 64)
assert Fraction(19, 8) / Fraction(665, 64) == Fraction(8, 35)
assert Fraction(3, 4) / Fraction(31, 32) == Fraction(24, 31) and (5 * Fraction(24, 31) - 2) * 3 == Fraction(174, 31)

Path(__file__).with_name("seance_math_1.out.json").write_text(json.dumps(V, indent=1, ensure_ascii=False), encoding="utf-8")
dire(f"{len(V)} valeurs exportées dans verif/seance_math_1.out.json.")
dire("Toutes les vérifications passent.")
Path(__file__).with_name("seance_math_1.out.txt").write_text("\n".join(LIGNES) + "\n", encoding="utf-8")
