"""Vérification de tous les nombres du test en ligne du 9 octobre 2026 (page à repasser et débrief).

Numérotation : Qnn = question nn du test (1 = code, 2 à 11 = mathématiques, 12 à 16 = ML fondamental,
17 à 24 = ML avancé). Les questions 14, 15 et 18 n'ont pas été capturées : les nombres Q14, Q15 et Q18
sont ceux des questions de remplacement.

Chaque probabilité ou espérance a deux vérifications indépendantes : un calcul exact (fractions, énumération
ou sympy) et une simulation Monte-Carlo dont l'écart à l'exact doit rester inférieur à 4 erreurs standard.
Les formes de tenseurs sont calculées par formule puis contrôlées par une vraie convolution numpy sur un
petit tenseur. Les affirmations de ML sont accompagnées d'une expérience.

Graine fixe : la sortie est identique à chaque exécution.
Sorties : verif/test_9_octobre.out.json (lu par outils/construire.py) et verif/test_9_octobre.out.txt.
"""
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, special, stats

rng = np.random.default_rng(20261009)
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
    dire(f"{nom:22s} exact={float(exact):.6f}  MC={float(estimation):.6f}  se={float(erreur_std):.6f}  z={z:+.2f}")


def mc_moyenne(nom, exact, echantillon):
    echantillon = np.asarray(echantillon, dtype=float)
    controle_mc(nom, exact, echantillon.mean(), echantillon.std(ddof=1) / np.sqrt(len(echantillon)))


def mc_proba(nom, exact, indicatrices):
    indicatrices = np.asarray(indicatrices, dtype=float)
    p = float(exact)
    controle_mc(nom, exact, indicatrices.mean(), np.sqrt(p * (1 - p) / len(indicatrices)))


N_SIM = 1_000_000

# =====================================================================================
# BILAN : ce que montrent les captures, comparé aux réponses attendues établies plus bas.
# "vue" = options cochées sur la capture (dans l'ordre de la capture, A = première option) ; None = rien de coché.
# =====================================================================================
ATTENDUES = {
    2: "ABCD", 3: "D", 4: "AD", 5: "C", 6: "AC", 7: "D", 8: "D", 9: "B", 10: "B", 11: "C", 12: "B", 13: "A",
    16: "BC", 17: "ABC", 19: "D", 20: "AC", 21: "ACD", 22: "B", 23: "ABD", 24: "A",
}
VUES = {
    1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None, 11: None, 12: None,
    13: None, 16: None, 17: None, 19: None, 20: None, 21: None, 22: None, 23: None, 24: None,
}
MANQUANTES = [14, 15, 18]
assert sorted(list(VUES) + MANQUANTES) == list(range(1, 25))


def verdict(q):
    vue = VUES[q]
    if vue is None:
        return "non visible"
    att = ATTENDUES[q]
    if vue == att:
        return "juste"
    if set(vue) < set(att):
        return "incomplète"
    return "fausse"


VERDICTS = {q: verdict(q) for q in VUES}
V["B_total"] = 24
V["B_capturees"] = len(VUES)
V["B_manquantes"] = len(MANQUANTES)
V["B_visibles"] = sum(v != "non visible" for v in VERDICTS.values())
V["B_non_visibles"] = sum(v == "non visible" for v in VERDICTS.values())
V["B_justes"] = sum(v == "juste" for v in VERDICTS.values())
V["B_fausses"] = sum(v == "fausse" for v in VERDICTS.values())
V["B_incompletes"] = sum(v == "incomplète" for v in VERDICTS.values())
V["B_multiples_visibles"] = sum(1 for q, v in VERDICTS.items() if v != "non visible" and len(ATTENDUES[q]) > 1)
V["B_multiples_justes"] = sum(1 for q, v in VERDICTS.items() if v == "juste" and len(ATTENDUES[q]) > 1)
V["B_uniques_visibles"] = V["B_visibles"] - V["B_multiples_visibles"]
V["B_uniques_justes"] = V["B_justes"] - V["B_multiples_justes"]
assert V["B_visibles"] == V["B_justes"] + V["B_fausses"] + V["B_incompletes"]
assert V["B_capturees"] + V["B_manquantes"] == 24
for q in sorted(VERDICTS):
    dire(f"Q{q:02d} : capture = {VUES[q] or '-':4s} attendue = {ATTENDUES.get(q, '(code)'):5s} verdict = {VERDICTS[q]}")
dire(f"Bilan : {V['B_visibles']} réponses visibles, {V['B_justes']} justes, {V['B_fausses']} fausses, {V['B_incompletes']} incomplètes ; "
     f"{V['B_non_visibles']} non visibles ; {V['B_manquantes']} captures manquantes.")

# =====================================================================================
# Q2 : tests d'adéquation. Expérience : Kolmogorov-Smirnov avec paramètres connus ou estimés.
# =====================================================================================
n_ks, rep_ks = 50, 4000
rej_vrais = rej_estimes = 0
for _ in range(rep_ks):
    x = rng.normal(0.0, 1.0, n_ks)
    rej_vrais += stats.kstest(x, "norm", args=(0.0, 1.0)).pvalue < 0.05
    rej_estimes += stats.kstest(x, "norm", args=(x.mean(), x.std(ddof=1))).pvalue < 0.05
V["Q02_n"] = n_ks
V["Q02_rep"] = rep_ks
V["Q02_ks_vrais"] = rej_vrais / rep_ks
V["Q02_ks_estimes"] = rej_estimes / rep_ks
se = math.sqrt(0.05 * 0.95 / rep_ks)
assert abs(V["Q02_ks_vrais"] - 0.05) < 4 * se and V["Q02_ks_estimes"] < 0.005
dire(f"Q02 : taux de rejet à 5 % du test KS sous H0, paramètres connus {V['Q02_ks_vrais']:.4f}, paramètres estimés {V['Q02_ks_estimes']:.4f}")
# Khi-deux d'adéquation à une loi de Poisson dont le paramètre est estimé : degrés de liberté = classes - 1 - 1.
n_chi, rep_chi, rej_bon, rej_naif = 200, 4000, 0, 0
for _ in range(rep_chi):
    x = rng.poisson(3.0, n_chi)
    lam = x.mean()
    bornes = [0, 1, 2, 3, 4, 5, 6]  # classes 0-1, 2, 3, 4, 5, 6 et plus
    obs = np.array([(x <= 1).sum(), (x == 2).sum(), (x == 3).sum(), (x == 4).sum(), (x == 5).sum(), (x >= 6).sum()])
    p = np.array([stats.poisson.cdf(1, lam), *[stats.poisson.pmf(k, lam) for k in (2, 3, 4, 5)], stats.poisson.sf(5, lam)])
    stat = ((obs - n_chi * p) ** 2 / (n_chi * p)).sum()
    rej_bon += stats.chi2.sf(stat, len(obs) - 2) < 0.05
    rej_naif += stats.chi2.sf(stat, len(obs) - 1) < 0.05
V["Q02_chi_classes"] = 6
V["Q02_chi_ddl"] = 6 - 1 - 1
V["Q02_chi_bon"] = rej_bon / rep_chi
V["Q02_chi_naif"] = rej_naif / rep_chi
assert abs(V["Q02_chi_bon"] - 0.05) < 0.02 and V["Q02_chi_naif"] < V["Q02_chi_bon"]
dire(f"Q02 : khi-deux Poisson, taux de rejet à 5 % avec {V['Q02_chi_ddl']} ddl {V['Q02_chi_bon']:.4f}, avec 5 ddl {V['Q02_chi_naif']:.4f}")

# =====================================================================================
# Q4 : corrélation et covariance (calcul exact sur de petites lois uniformes).
# =====================================================================================
def cov_exacte(xs, ys):
    n = len(xs)
    mx, my = Fraction(sum(xs), n), Fraction(sum(ys), n)
    return sum((Fraction(a) - mx) * (Fraction(b) - my) for a, b in zip(xs, ys)) / n


xs, ys = [1, 2, 3, 4], [2, 1, 4, 3]
c0, vx, vy = cov_exacte(xs, ys), cov_exacte(xs, xs), cov_exacte(ys, ys)
xs_cm = [100 * a for a in xs]
c1, vx1 = cov_exacte(xs_cm, ys), cov_exacte(xs_cm, xs_cm)
assert c1 == 100 * c0 and vx1 == 100**2 * vx
fraction("Q04_cov", c0)
fraction("Q04_cov_cm", c1)
rho0 = float(c0) / math.sqrt(float(vx * vy))
rho1 = float(c1) / math.sqrt(float(vx1 * vy))
assert abs(rho0 - rho1) < 1e-15 and abs(rho0 - 0.6) < 1e-15
V["Q04_corr"] = round(rho0, 12)
V["Q04_corr_cm"] = round(rho1, 12)
V["Q04_corr_oppose"] = round(float(cov_exacte([-a for a in xs], ys)) / math.sqrt(float(vx * vy)), 12)
assert V["Q04_corr_oppose"] == -V["Q04_corr"]
# Dépendance non linéaire : X uniforme sur {-1, 0, 1}, Y = X^2.
x3 = [-1, 0, 1]
y3 = [a * a for a in x3]
assert cov_exacte(x3, y3) == 0
V["Q04_cov_carre"] = 0
xm = rng.integers(-1, 2, N_SIM)
mc_moyenne("Q04_cov_carre", 0, (xm - xm.mean()) * (xm**2 - (xm**2).mean()))

# =====================================================================================
# Q5 : crises (loi de Poisson, a priori proportionnel à 1/N).
# =====================================================================================
candidats = [2, 4, 6, 8]
k_obs = 6
H = sum(Fraction(1, n) for n in candidats)
fraction("Q05_H", H)
jointes = {}
for n in candidats:
    prior = Fraction(1, n) / H
    vrais = math.exp(-n) * n**k_obs / math.factorial(k_obs)
    jointes[n] = float(prior) * vrais
    V[f"Q05_prior_{n}"] = float(prior)
    V[f"Q05_vrais_{n}"] = vrais
    V[f"Q05_jointe_{n}"] = jointes[n]
    V[f"Q05_g_{n}"] = n ** (k_obs - 1) * math.exp(-n)  # N^5 e^{-N}, proportionnel à la probabilité jointe
V["Q05_exact"] = max(jointes, key=jointes.get)
assert V["Q05_exact"] == 6
V["Q05_rapport_6_4"] = jointes[6] / jointes[4]
V["Q05_vrais_max"] = max(candidats, key=lambda n: V[f"Q05_vrais_{n}"])
assert V["Q05_vrais_max"] == 6
# Maximum de g(N) = N^5 e^{-N} sur les réels positifs : N = 5 (dérivée nulle), vérifié par sympy.
Ns = sp.symbols("N", positive=True)
assert sp.solve(sp.diff(Ns**5 * sp.exp(-Ns), Ns), Ns) == [5]
V["Q05_max_continu"] = 5
V["Q05_g_5"] = 5**5 * math.exp(-5)
assert V["Q05_g_5"] > V["Q05_g_6"] > V["Q05_g_4"]
# Autre lecture : valeurs = taux par an, moyenne 10 fois plus grande sur 10 ans.
alt = {lam: (1 / lam) * math.exp(-10 * lam) * (10 * lam) ** k_obs for lam in candidats}
V["Q05_alt"] = max(alt, key=alt.get)
assert V["Q05_alt"] == 2
V["Q05_alt_moy_2"] = 10 * 2
# Simulation : N tiré selon l'a priori, puis nombre de crises selon Poisson(N).
priors = np.array([float(Fraction(1, n) / H) for n in candidats])
tir = rng.choice(candidats, size=2 * N_SIM, p=priors)
crises = rng.poisson(tir)
for n in candidats:
    mc_proba(f"Q05_jointe_{n}", jointes[n], (tir == n) & (crises == k_obs))

# =====================================================================================
# Q6 : asymétrie positive. Loi log-normale (exact), et contre-exemple discret.
# =====================================================================================
V["Q06_ln_mode"] = math.exp(-1)
V["Q06_ln_mediane"] = 1.0
V["Q06_ln_moyenne"] = math.exp(0.5)
V["Q06_ln_asym"] = (math.e + 2) * math.sqrt(math.e - 1)
assert V["Q06_ln_mode"] < V["Q06_ln_mediane"] < V["Q06_ln_moyenne"] and V["Q06_ln_asym"] > 0
# Contrôle du mode par sympy : maximum de la densité log-normale.
xs_ = sp.symbols("x", positive=True)
dens = sp.exp(-sp.log(xs_) ** 2 / 2) / (xs_ * sp.sqrt(2 * sp.pi))
assert sp.simplify(sp.diff(dens, xs_).subs(xs_, sp.exp(-1))) == 0
ln = rng.lognormal(0.0, 1.0, 4 * N_SIM)
mc_moyenne("Q06_ln_moyenne", V["Q06_ln_moyenne"], ln)
mc_proba("Q06_ln_mediane_p", 0.5, ln <= 1.0)
# Loi exponentielle de paramètre 1 : mode 0, médiane ln 2, moyenne 1, asymétrie 2.
V["Q06_exp_mediane"] = math.log(2)
# Contre-exemple : loi sur {0, 1, 5}.
loi = {0: Fraction(45, 100), 1: Fraction(47, 100), 5: Fraction(8, 100)}
assert sum(loi.values()) == 1
moy = sum(k * p for k, p in loi.items())
var = sum((k - moy) ** 2 * p for k, p in loi.items())
m3 = sum((k - moy) ** 3 * p for k, p in loi.items())
mediane = min(k for k in loi if sum(p for j, p in loi.items() if j <= k) >= Fraction(1, 2))
mode = max(loi, key=loi.get)
assert m3 > 0 and moy < mediane and moy < mode and mediane == 1 and mode == 1
V["Q06_ce_moyenne"] = float(moy)
V["Q06_ce_mediane"] = mediane
V["Q06_ce_mode"] = mode
V["Q06_ce_asym"] = float(m3) / float(var) ** 1.5
V["Q06_ce_p0"], V["Q06_ce_p1"], V["Q06_ce_p5"] = 0.45, 0.47, 0.08
dire(f"Q06 : contre-exemple, moyenne {float(moy):.2f} < médiane = mode = 1, asymétrie {V['Q06_ce_asym']:.3f} > 0")

# =====================================================================================
# Q7 : corde de longueur 1 coupée en deux points uniformes indépendants, espérance du plus petit morceau.
# =====================================================================================
t, a, b = sp.symbols("t a b", nonnegative=True)
# P(les trois morceaux dépassent t) : points ordonnés a < b, a > t, b - a > t, 1 - b > t ; facteur 2 pour l'ordre.
survie = sp.expand(2 * sp.integrate(sp.integrate(1, (b, a + t, 1 - t)), (a, t, 1 - 2 * t)))
assert sp.simplify(survie - (1 - 3 * t) ** 2) == 0
e_min = sp.integrate((1 - 3 * t) ** 2, (t, 0, sp.Rational(1, 3)))
assert e_min == sp.Rational(1, 9)
# P(le plus grand morceau <= t) pour t dans [1/2, 1] : 1 - 3 (1 - t)^2 ; pour t dans [1/3, 1/2] : (3t - 1)^2.
e_max = sp.Rational(1, 3) + sp.integrate(1 - (3 * t - 1) ** 2, (t, sp.Rational(1, 3), sp.Rational(1, 2))) + sp.integrate(3 * (1 - t) ** 2, (t, sp.Rational(1, 2), 1))
assert e_max == sp.Rational(11, 18)
e_mid = 1 - e_min - e_max
assert e_mid == sp.Rational(5, 18)
fraction("Q07_exact", Fraction(1, 9))
fraction("Q07_max", Fraction(11, 18))
fraction("Q07_milieu", Fraction(5, 18))
fraction("Q07_d_moyen", Fraction(1, 3))
fraction("Q07_d_sixieme", Fraction(1, 6))
fraction("Q07_d_huitieme", Fraction(1, 8))
fraction("Q07_deux_morceaux", Fraction(1, 4))
u = np.sort(rng.random((N_SIM, 2)), axis=1)
morceaux = np.sort(np.stack([u[:, 0], u[:, 1] - u[:, 0], 1 - u[:, 1]], axis=1), axis=1)
mc_moyenne("Q07_exact", Fraction(1, 9), morceaux[:, 0])
mc_moyenne("Q07_milieu", Fraction(5, 18), morceaux[:, 1])
mc_moyenne("Q07_max", Fraction(11, 18), morceaux[:, 2])
mc_proba("Q07_survie_01", (1 - 0.3) ** 2, morceaux[:, 0] > 0.1)
V["Q07_survie_01"] = (1 - 0.3) ** 2
# Formule générale E[min] = 1/m^2 pour m morceaux (m - 1 coupes uniformes indépendantes) : contrôle par simulation pour m = 2 à 5.
for m_ in range(2, 6):
    coupes = np.sort(rng.random((N_SIM // 2, m_ - 1)), axis=1)
    bords = np.concatenate([np.zeros((len(coupes), 1)), coupes, np.ones((len(coupes), 1))], axis=1)
    mc_moyenne(f"Q07_general_{m_}", Fraction(1, m_ * m_), np.diff(bords, axis=1).min(axis=1))
V["Q07_general_max"] = 5
# Autre modèle (une coupe uniforme, puis une coupe uniforme du morceau de droite) : simulation seule, pour montrer que la réponse dépend du modèle.
u1, u2 = rng.random(N_SIM), rng.random(N_SIM)
seq = np.minimum(np.minimum(u1, (1 - u1) * u2), (1 - u1) * (1 - u2))
V["Q07_sequentiel_mc"] = float(seq.mean())
V["Q07_sequentiel_se"] = float(seq.std(ddof=1) / math.sqrt(N_SIM))
assert abs(V["Q07_sequentiel_mc"] - 1 / 9) > 10 * V["Q07_sequentiel_se"]
dire(f"Q07 : modèle séquentiel, E[min] estimée {V['Q07_sequentiel_mc']:.5f} (se {V['Q07_sequentiel_se']:.5f}), différente de 1/9")

# =====================================================================================
# Q8 : densité jointe f(x, y) = exp(-y/x) exp(-x) / x sur x > 0, y > 0.
# =====================================================================================
x, y = sp.symbols("x y", positive=True)
f = sp.exp(-y / x) * sp.exp(-x) / x
f_x = sp.integrate(f, (y, 0, sp.oo))
assert sp.simplify(f_x - sp.exp(-x)) == 0
assert sp.integrate(f_x, (x, 0, sp.oo)) == 1
cond = sp.simplify(sp.integrate(f, (y, 1, sp.oo)) / f_x)
assert sp.simplify(cond - sp.exp(-1 / x)) == 0
V["Q08_x_ex"] = 2
V["Q08_val_x2"] = math.exp(-0.5)
V["Q08_d_plus_x2"] = math.exp(0.5)
V["Q08_d_joint_x2"] = math.exp(-2 - 0.5)
# Contrôle numérique indépendant : intégration de la densité jointe sur une bande étroite autour de x = 2.
num, _ = integrate.dblquad(lambda yy, xx: math.exp(-yy / xx - xx) / xx, 1.9, 2.1, 1.0, np.inf)
den, _ = integrate.dblquad(lambda yy, xx: math.exp(-yy / xx - xx) / xx, 1.9, 2.1, 0.0, np.inf)
V["Q08_bande"] = num / den
assert abs(V["Q08_bande"] - V["Q08_val_x2"]) < 2e-3
# Probabilité non conditionnelle : P(Y > 1) = intégrale de exp(-x - 1/x) = 2 K_1(2).
p_y1 = 2 * special.kv(1, 2.0)
quad_y1, _ = integrate.quad(lambda xx: math.exp(-xx - 1 / xx), 0, np.inf)
assert abs(p_y1 - quad_y1) < 1e-9
V["Q08_p_y1"] = float(p_y1)
X = rng.exponential(1.0, 4 * N_SIM)
Y = X * rng.exponential(1.0, 4 * N_SIM)  # sachant X = x, Y est exponentielle de moyenne x
mc_proba("Q08_p_y1", p_y1, Y > 1)
bande = (X > 1.9) & (X < 2.1)
mc_proba("Q08_bande", V["Q08_bande"], Y[bande] > 1)
mc_moyenne("Q08_esp_y_bande", float(integrate.quad(lambda xx: xx * math.exp(-xx), 1.9, 2.1)[0] / integrate.quad(lambda xx: math.exp(-xx), 1.9, 2.1)[0]), Y[bande])

# =====================================================================================
# Q9 : constante de normalisation de c x^2 (1 + y) sur [0, 3]^2.
# =====================================================================================
ix = sp.integrate(x**2, (x, 0, 3))
iy = sp.integrate(1 + y, (y, 0, 3))
assert ix == 9 and iy == sp.Rational(15, 2)
total = Fraction(9) * Fraction(15, 2)
c = 1 / total
assert c == Fraction(2, 135)
V["Q09_ix"] = 9
fraction("Q09_iy", Fraction(15, 2))
fraction("Q09_total", total)
fraction("Q09_exact", c)
for nom, val in [("Q09_d_65", Fraction(2, 65)), ("Q09_d_97", Fraction(3, 97)), ("Q09_d_10", Fraction(1, 10))]:
    fraction(nom, val)
    fraction(nom + "_masse", val * total)
    assert val * total != 1
pts = 3 * rng.random((N_SIM, 2))
mc_moyenne("Q09_total", total, 9 * pts[:, 0] ** 2 * (1 + pts[:, 1]))  # aire 9 fois la moyenne de l'intégrande

# =====================================================================================
# Q10 : deux enfants.
# =====================================================================================
familles = list(itertools.product("FG", repeat=2))  # (aîné, cadet), F = fille, G = garçon, 4 cas équiprobables
cas1 = [f_ for f_ in familles if f_[0] == "F"]
p1 = Fraction(sum(f_ == ("F", "F") for f_ in cas1), len(cas1))
cas2 = [f_ for f_ in familles if "G" in f_]
p2 = Fraction(sum(f_ == ("G", "G") for f_ in cas2), len(cas2))
assert p1 == Fraction(1, 2) and p2 == Fraction(1, 3)
fraction("Q10_p1", p1)
fraction("Q10_p2", p2)
V["Q10_cas1"] = len(cas1)
V["Q10_cas2"] = len(cas2)
# Autre protocole : on observe un enfant tiré au hasard parmi les deux, et c'est un garçon.
num_alt = sum(Fraction(1, 4) * Fraction(sum(e == "G" for e in f_), 2) for f_ in familles if f_ == ("G", "G"))
den_alt = sum(Fraction(1, 4) * Fraction(sum(e == "G" for e in f_), 2) for f_ in familles)
fraction("Q10_p2_alt", num_alt / den_alt)
assert num_alt / den_alt == Fraction(1, 2)
enf = rng.integers(0, 2, size=(N_SIM, 2))  # 1 = garçon
mc_proba("Q10_p1", p1, enf[enf[:, 0] == 0][:, 1] == 0)
mc_proba("Q10_p2", p2, enf[enf.sum(axis=1) >= 1].sum(axis=1) == 2)
choix = rng.integers(0, 2, N_SIM)
vu_garcon = enf[np.arange(N_SIM), choix] == 1
mc_proba("Q10_p2_alt", Fraction(1, 2), enf[vu_garcon].sum(axis=1) == 2)

# =====================================================================================
# Q11 : comparaison de deux notes par l'écart réduit.
# =====================================================================================
z_m = Fraction(70 - 60, 15)
z_s = Fraction(72 - 68, 6)
assert z_m == z_s == Fraction(2, 3)
fraction("Q11_z_maths", z_m)
fraction("Q11_z_sciences", z_s)
V["Q11_ecart_maths"] = 70 - 60
V["Q11_ecart_sciences"] = 72 - 68
V["Q11_rang"] = float(stats.norm.cdf(2 / 3))
V["Q11_rang_pct"] = 100 * V["Q11_rang"]
V["Q11_rapport_maths"] = 70 / 60
V["Q11_rapport_sciences"] = 72 / 68
g = rng.normal(size=N_SIM)
mc_proba("Q11_rang_maths", V["Q11_rang"], 60 + 15 * g < 70)
g = rng.normal(size=N_SIM)
mc_proba("Q11_rang_sciences", V["Q11_rang"], 68 + 6 * g < 72)

# =====================================================================================
# Q12 : formes de tenseurs. Formule, puis vraie convolution numpy sur un petit tenseur.
# =====================================================================================
def taille_sortie(n, k, p, s):
    """Taille de sortie d'une convolution ou d'un pooling : floor((n + 2p - k) / s) + 1."""
    return (n + 2 * p - k) // s + 1


def conv2d(entree, noyaux, p):
    """Convolution (corrélation croisée) de pas 1. entree : (h, l, c) ; noyaux : (k, k, c, f) ; marge de zéros p."""
    k = noyaux.shape[0]
    e = np.pad(entree, ((p, p), (p, p), (0, 0)))
    h, l = e.shape[0] - k + 1, e.shape[1] - k + 1
    sortie = np.zeros((h, l, noyaux.shape[3]))
    for i in range(k):
        for j in range(k):
            sortie += np.tensordot(e[i:i + h, j:j + l, :], noyaux[i, j], axes=([2], [0]))
    return sortie


def max_pool(entree, k=2, s=2):
    h, l = (entree.shape[0] - k) // s + 1, (entree.shape[1] - k) // s + 1
    blocs = np.stack([entree[i:i + s * h:s, j:j + s * l:s, :] for i in range(k) for j in range(k)])
    return blocs.max(axis=0)


def reseau(n, f1, f2, p):
    """Formes par formule : conv1 (f1 filtres), conv2 (f2 filtres), concaténation de conv1 et conv2 en profondeur, pooling 2x2 de pas 2."""
    n1 = taille_sortie(n, 3, p, 1)
    n2 = taille_sortie(n1, 3, p, 1)
    if n1 != n2:
        return None, n1, n2
    return (taille_sortie(n2, 2, 0, 2), taille_sortie(n2, 2, 0, 2), f1 + f2), n1, n2


forme_fig, _, _ = reseau(256, 64, 128, 1)  # figure, marge « same »
forme_txt, _, _ = reseau(256, 64, 64, 1)  # liste écrite, marge « same »
forme_valid, v1, v2 = reseau(256, 64, 128, 0)  # sans marge
assert forme_fig == (128, 128, 192) and forme_txt == (128, 128, 128) and forme_valid is None and (v1, v2) == (254, 252)
V["Q12_cote"] = forme_fig[0]
V["Q12_prof_figure"] = forme_fig[2]
V["Q12_prof_liste"] = forme_txt[2]
V["Q12_valid_1"] = v1
V["Q12_valid_2"] = v2
V["Q12_avant_pool"] = 256
# Vraie convolution, modèle réduit : entrée 16 x 16 x 3, 4 filtres puis 8 filtres.
petit = rng.normal(size=(16, 16, 3))
k1 = rng.normal(size=(3, 3, 3, 4))
k2 = rng.normal(size=(3, 3, 4, 8))
a1 = conv2d(petit, k1, 1)
a2 = conv2d(a1, k2, 1)
sortie = max_pool(np.concatenate([a1, a2], axis=2))
assert a1.shape == (16, 16, 4) and a2.shape == (16, 16, 8) and sortie.shape == (8, 8, 12)
assert sortie.shape == reseau(16, 4, 8, 1)[0]
# Contrôle de conv2d contre scipy sur un canal.
from scipy import signal  # noqa: E402
ref = signal.correlate2d(petit[:, :, 0], k1[:, :, 0, 0], mode="same") + signal.correlate2d(petit[:, :, 1], k1[:, :, 1, 0], mode="same") \
    + signal.correlate2d(petit[:, :, 2], k1[:, :, 2, 0], mode="same")
assert np.allclose(ref, a1[:, :, 0])
b1 = conv2d(petit, k1, 0)
b2 = conv2d(b1, k2, 0)
assert b1.shape == (14, 14, 4) and b2.shape == (12, 12, 8)
try:
    np.concatenate([b1, b2], axis=2)
    echec_concat = False
except ValueError:
    echec_concat = True
assert echec_concat
V["Q12_petit_n"] = 16
V["Q12_petit_f1"], V["Q12_petit_f2"] = 4, 8
V["Q12_petit_cote"], V["Q12_petit_prof"] = sortie.shape[0], sortie.shape[2]
V["Q12_petit_valid_1"], V["Q12_petit_valid_2"] = b1.shape[0], b2.shape[0]
# Nombre de paramètres (noyaux et biais) des deux convolutions de la figure.
V["Q12_param_1"] = 3 * 3 * 3 * 64 + 64
V["Q12_param_2"] = 3 * 3 * 64 * 128 + 128
dire(f"Q12 : marge same, figure {forme_fig}, liste {forme_txt} ; sans marge {v1} contre {v2} : concaténation impossible ; modèle réduit {sortie.shape}")

# =====================================================================================
# Q13 : réglages d'entraînement d'une segmentation binaire, 2000 paires.
# =====================================================================================
reglages = {"A": (1600, 200, 200, 100, 16), "B": (1600, 200, 200, 150, 10), "C": (1000, 800, 200, 100, 8), "D": (1000, 800, 200, 50, 20)}
for lettre, (tr, va, te, pas, lot) in reglages.items():
    assert tr + va + te == 2000
    V[f"Q13_vus_{lettre}"] = pas * lot
    V[f"Q13_part_train_{lettre}"] = 100 * tr / 2000
    V[f"Q13_part_val_{lettre}"] = 100 * va / 2000
assert V["Q13_vus_A"] == 1600 and V["Q13_vus_B"] == 1500 and V["Q13_vus_C"] == 800 and V["Q13_vus_D"] == 1000
V["Q13_manque_B"] = 1600 - V["Q13_vus_B"]
V["Q13_manque_C"] = 1000 - V["Q13_vus_C"]
V["Q13_part_test"] = 100 * 200 / 2000
# Entropie croisée à 2 classes avec softmax = entropie croisée binaire avec sigmoïde de la différence des scores.
s = rng.normal(size=(1000, 2))
etiq = rng.integers(0, 2, 1000)
soft = np.exp(s) / np.exp(s).sum(axis=1, keepdims=True)
cat = -np.log(soft[np.arange(1000), etiq])
sig = 1 / (1 + np.exp(-(s[:, 1] - s[:, 0])))
binaire = -(etiq * np.log(sig) + (1 - etiq) * np.log(1 - sig))
assert np.allclose(cat, binaire)
dire("Q13 : entropie croisée catégorielle à 2 sorties softmax = entropie croisée binaire sur la sigmoïde de la différence des scores (1000 cas).")

# =====================================================================================
# Q14 (remplacement) : matrice de confusion.
# =====================================================================================
n_tot, n_pos, n_alertes, n_vp = 1000, 50, 80, 30
fp, fn = n_alertes - n_vp, n_pos - n_vp
vn = n_tot - n_vp - fp - fn
assert vn == 900
V["Q14_fp"], V["Q14_fn"], V["Q14_vn"] = fp, fn, vn
fraction("Q14_rappel", Fraction(n_vp, n_pos))
fraction("Q14_precision", Fraction(n_vp, n_alertes))
fraction("Q14_exactitude", Fraction(n_vp + vn, n_tot))
fraction("Q14_f1", 2 * Fraction(n_vp, n_alertes) * Fraction(n_vp, n_pos) / (Fraction(n_vp, n_alertes) + Fraction(n_vp, n_pos)))
fraction("Q14_d_fpr", Fraction(fp, fp + vn))
fraction("Q14_tout_negatif", Fraction(n_tot - n_pos, n_tot))
# Simulation : on tire une transaction frauduleuse au hasard, probabilité qu'elle ait été signalée.
fraudes = np.array([1] * n_vp + [0] * fn)
mc_proba("Q14_rappel", Fraction(n_vp, n_pos), rng.choice(fraudes, N_SIM))

# =====================================================================================
# Q15 (remplacement) : k plus proches voisins, exactitude sur l'ensemble d'entraînement.
# =====================================================================================
n0, n1_ = 120, 80
pts_knn = rng.normal(size=(n0 + n1_, 2))
assert len({tuple(p_) for p_ in pts_knn.round(12)}) == n0 + n1_  # points distincts
classes = np.array([0] * n0 + [1] * n1_)
d2 = ((pts_knn[:, None, :] - pts_knn[None, :, :]) ** 2).sum(axis=2)
ordre = np.argsort(d2, axis=1, kind="stable")


def exactitude_knn(k):
    votes = classes[ordre[:, :k]].mean(axis=1)
    assert not np.any(votes == 0.5)
    return float(((votes > 0.5).astype(int) == classes).mean())


V["Q15_n"] = n0 + n1_
V["Q15_n0"], V["Q15_n1"] = n0, n1_
V["Q15_k1"] = exactitude_knn(1)
V["Q15_kn"] = exactitude_knn(n0 + n1_)
V["Q15_k15"] = exactitude_knn(15)
assert V["Q15_k1"] == 1.0 and V["Q15_kn"] == n0 / (n0 + n1_) == 0.6
V["Q15_d_minoritaire"] = n1_ / (n0 + n1_)
V["Q15_d_hasard"] = 0.5
dire(f"Q15 : exactitude d'entraînement k=1 : {V['Q15_k1']}, k=15 : {V['Q15_k15']:.3f}, k=n : {V['Q15_kn']}")

# =====================================================================================
# Q18 (remplacement) : dropout inversé.
# =====================================================================================
p_drop, act = Fraction(1, 5), 3
garde = 1 - p_drop
facteur = 1 / garde
esp = garde * act * facteur
var_ = garde * (act * facteur) ** 2 - esp**2
assert esp == 3 and var_ == Fraction(9, 4) and var_ == act**2 * p_drop / garde
V["Q18_p"] = float(p_drop)
V["Q18_garde"] = float(garde)
V["Q18_facteur"] = float(facteur)
V["Q18_valeur_gardee"] = float(act * facteur)
V["Q18_esp"] = float(esp)
V["Q18_var"] = float(var_)
V["Q18_d_esp_sans"] = float(garde * act)
V["Q18_d_var_sans"] = float(garde * act**2 - (garde * act) ** 2)
V["Q18_d_var_p"] = float(act**2 * p_drop * garde)
masque = rng.random(N_SIM) < float(garde)
sortie_d = masque * act * float(facteur)
mc_moyenne("Q18_esp", esp, sortie_d)
cd = sortie_d - sortie_d.mean()
controle_mc("Q18_var", var_, (cd**2).mean(), math.sqrt(((cd**2 - (cd**2).mean()) ** 2).mean() / N_SIM))

# =====================================================================================
# Q20 : arbres de décision. Expérience : la profondeur ne dépend pas du nombre de variables.
# =====================================================================================
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import ElasticNet, Lasso, Ridge  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

gen = np.random.RandomState(20261009)
Xa = gen.normal(size=(500, 2))
ya = gen.randint(0, 2, 500)  # étiquettes sans lien avec les variables
arbre = DecisionTreeClassifier(random_state=0).fit(Xa, ya)
V["Q20_nb_variables"] = 2
V["Q20_n"] = 500
V["Q20_profondeur"] = int(arbre.get_depth())
V["Q20_exactitude_train"] = float(arbre.score(Xa, ya))
assert V["Q20_profondeur"] > 2 and V["Q20_exactitude_train"] == 1.0
dire(f"Q20 : arbre non limité sur 2 variables numériques et 500 points : profondeur {V['Q20_profondeur']}, exactitude d'entraînement {V['Q20_exactitude_train']}")

# =====================================================================================
# Q21 : importances. Une variable de pur bruit reçoit une importance par impureté non nulle.
# =====================================================================================
n_imp = 600
x_utile = gen.normal(size=n_imp)
bruit_continu = gen.normal(size=n_imp)
bruit_binaire = gen.randint(0, 2, n_imp).astype(float)
y_imp = (x_utile + 1.5 * gen.normal(size=n_imp) > 0).astype(int)
Xi = np.column_stack([x_utile, bruit_continu, bruit_binaire])
foret = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xi[:400], y_imp[:400])
mdi = foret.feature_importances_
perm = permutation_importance(foret, Xi[400:], y_imp[400:], n_repeats=30, random_state=0).importances_mean
V["Q21_mdi_utile"], V["Q21_mdi_bruit_continu"], V["Q21_mdi_bruit_binaire"] = float(mdi[0]), float(mdi[1]), float(mdi[2])
V["Q21_perm_utile"], V["Q21_perm_bruit_continu"], V["Q21_perm_bruit_binaire"] = float(perm[0]), float(perm[1]), float(perm[2])
assert mdi[1] > 0.25 and mdi[1] > 5 * mdi[2] and perm[1] < 0.02 and perm[2] < 0.02 and perm[0] > 0.05
dire(f"Q21 : importance par impureté (utile, bruit continu, bruit binaire) = {mdi.round(3)} ; par permutation sur données de test = {perm.round(3)}")

# =====================================================================================
# Q22 (et Q19) : régularisation contre surapprentissage. Polynôme de degré 10 sur 30 points.
# =====================================================================================
def base(xv, degre):
    return np.polynomial.legendre.legvander(xv, degre)


x_tr = np.sort(gen.uniform(-1, 1, 30))
x_te = gen.uniform(-1, 1, 5000)
vraie = lambda xv: np.sin(3 * xv)  # noqa: E731
y_tr = vraie(x_tr) + 0.3 * gen.normal(size=30)
y_te = vraie(x_te) + 0.3 * gen.normal(size=5000)


def erreurs(degre, lam):
    A = base(x_tr, degre)
    penal = lam * np.eye(degre + 1)
    penal[0, 0] = 0.0  # la constante n'est pas pénalisée
    coef = np.linalg.solve(A.T @ A + penal, A.T @ y_tr) if lam > 0 else np.linalg.lstsq(A, y_tr, rcond=None)[0]
    return float(((A @ coef - y_tr) ** 2).mean()), float(((base(x_te, degre) @ coef - y_te) ** 2).mean())


V["Q22_n"] = 30
V["Q22_degre"] = 10
V["Q22_degre_plus"] = 15
V["Q22_lambda"] = 0.1
V["Q22_bruit_var"] = 0.3**2
V["Q22_train_sans"], V["Q22_test_sans"] = erreurs(10, 0.0)
V["Q22_train_ridge"], V["Q22_test_ridge"] = erreurs(10, 0.1)
V["Q22_train_plus"], V["Q22_test_plus"] = erreurs(15, 0.0)
assert V["Q22_train_plus"] < V["Q22_train_sans"] < V["Q22_train_ridge"]
assert V["Q22_test_ridge"] < V["Q22_test_sans"] < V["Q22_test_plus"]
dire(f"Q22 : degré 10 sans pénalité, erreur train {V['Q22_train_sans']:.4f} test {V['Q22_test_sans']:.4f} ; ridge train {V['Q22_train_ridge']:.4f} "
     f"test {V['Q22_test_ridge']:.4f} ; degré 15 train {V['Q22_train_plus']:.4f} test {V['Q22_test_plus']:.4f}")

# =====================================================================================
# Q23 : ACP. Parts de variance exactes à partir de matrices de covariance connues.
# =====================================================================================
def part_premiere(cov):
    vp = np.linalg.eigvalsh(np.asarray(cov, dtype=float))
    return float(vp.max() / vp.sum())


V["Q23_part_egales"] = part_premiere(np.eye(2))
V["Q23_facteur"] = 100
V["Q23_var_apres"] = 100**2
V["Q23_part_echelle"] = part_premiere(np.diag([100.0**2, 1.0]))
assert abs(V["Q23_part_echelle"] - 10000 / 10001) < 1e-12 and V["Q23_part_egales"] == 0.5
V["Q23_part_echelle_pct"] = 100 * V["Q23_part_echelle"]
V["Q23_dim_iso"] = 10
V["Q23_part_iso"] = part_premiere(np.eye(10))
V["Q23_part_iso_pct"] = 100 * V["Q23_part_iso"]
assert abs(V["Q23_part_iso"] - 0.1) < 1e-12
# Orthogonalité et combinaisons linéaires, sur un échantillon : composantes = données centrées fois vecteurs propres.
Z = gen.normal(size=(300, 4)) @ gen.normal(size=(4, 4))
Zc = Z - Z.mean(axis=0)
vp, W = np.linalg.eigh(Zc.T @ Zc / (len(Z) - 1))
scores = Zc @ W
assert np.allclose(W.T @ W, np.eye(4)) and np.allclose(np.cov(scores.T), np.diag(vp))
dire(f"Q23 : part de la 1re composante, variances égales {V['Q23_part_egales']}, une variable multipliée par 100 : {V['Q23_part_echelle']:.6f}, "
     f"10 variables non corrélées de même variance : {V['Q23_part_iso']:.2f}")

# =====================================================================================
# Q24 : Lasso, Ridge, Elastic Net en grande dimension (plus de variables que d'observations).
# =====================================================================================
n_l, p_l, utiles = 50, 200, 5
Xl = gen.normal(size=(n_l, p_l))
beta = np.zeros(p_l)
beta[:utiles] = [3.0, -2.0, 2.0, -3.0, 2.5]
yl = Xl @ beta + 0.5 * gen.normal(size=n_l)
lasso = Lasso(alpha=0.3, max_iter=100000).fit(Xl, yl)
ridge = Ridge(alpha=1.0).fit(Xl, yl)
enet = ElasticNet(alpha=0.3, l1_ratio=0.5, max_iter=100000).fit(Xl, yl)
V["Q24_n"], V["Q24_p"], V["Q24_utiles"] = n_l, p_l, utiles
V["Q24_nz_lasso"] = int((lasso.coef_ != 0).sum())
V["Q24_nz_ridge"] = int((ridge.coef_ != 0).sum())
V["Q24_nz_enet"] = int((enet.coef_ != 0).sum())
V["Q24_lasso_retrouve"] = int((lasso.coef_[:utiles] != 0).sum())
assert V["Q24_nz_ridge"] == p_l and V["Q24_lasso_retrouve"] == utiles and V["Q24_nz_lasso"] < V["Q24_nz_enet"] < p_l
assert V["Q24_nz_lasso"] <= n_l
dire(f"Q24 : coefficients non nuls sur {p_l} : Lasso {V['Q24_nz_lasso']}, Elastic Net {V['Q24_nz_enet']}, Ridge {V['Q24_nz_ridge']}")

# Budgets de la page à repasser.
V["T_code"], V["T_math"], V["T_mlf"], V["T_mla"] = 30, 28, 9, 15
V["T_total"] = V["T_code"] + V["T_math"] + V["T_mlf"] + V["T_mla"]
assert V["T_total"] == 82

Path(__file__).with_name("test_9_octobre.out.json").write_text(json.dumps(V, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
dire(f"{len(V)} valeurs exportées dans verif/test_9_octobre.out.json.")
dire("Toutes les vérifications passent.")
Path(__file__).with_name("test_9_octobre.out.txt").write_text("\n".join(LIGNES) + "\n", encoding="utf-8", newline="\n")
