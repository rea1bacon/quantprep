"""Vérification de tous les nombres de la séance d'algèbre linéaire 1 (leçon L-ALGEBRE-1 et questions AL-01 à AL-12).

Méthode :
  1. calcul exact (sympy, fractions) de chaque valeur citée dans les textes ;
  2. contrôle numérique indépendant (numpy) de chaque affirmation de la leçon et de chaque réponse ;
  3. pour les affirmations « avec probabilité 1 » et les espérances : simulation à graine fixe
     (écart à l'exact inférieur à 4 erreurs standard pour les espérances).
Les exemples numériques de la leçon elle-même sont exécutés à la construction (blocs python-exec) ;
ce script vérifie à part les énoncés généraux qu'ils illustrent.

Graine fixe et valeurs arrondies à 12 chiffres significatifs : la sortie est identique à chaque exécution.
Sorties : verif/seance_algebre_1.out.json (lu par outils/construire.py) et verif/seance_algebre_1.out.txt.
"""
import itertools
import json
import os
from fractions import Fraction
from pathlib import Path

# Un seul fil de calcul pour l'algèbre linéaire : résultats identiques d'une exécution à l'autre,
# et pas de ralentissement quand d'autres calculs occupent la machine.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import sympy as sp

rng = np.random.default_rng(20261006)
V = {}
LIGNES = []


def dire(texte):
    print(texte)
    LIGNES.append(texte)


def val(nom, x):
    """Exporte une valeur numérique, arrondie à 12 chiffres significatifs (sortie reproductible)."""
    x = float(x)
    V[nom] = int(x) if x == int(x) else float(f"{x:.12g}")


def fraction(nom, f):
    """Exporte une fraction : valeur, numérateur et dénominateur (forme irréductible)."""
    f = Fraction(f)
    val(nom, f)
    V[nom + "_num"] = f.numerator
    V[nom + "_den"] = f.denominator


def frac_sp(x):
    x = sp.nsimplify(x)
    return Fraction(int(sp.fraction(x)[0]), int(sp.fraction(x)[1]))


def controle_mc(nom, exact, echantillon, seuil=4.0):
    echantillon = np.asarray(echantillon, dtype=float)
    m, se = echantillon.mean(), echantillon.std(ddof=1) / np.sqrt(len(echantillon))
    z = (m - float(exact)) / se
    assert abs(z) < seuil, f"{nom} : exact={float(exact)}, MC={m}, z={z}"
    val(nom + "_mc", m)
    val(nom + "_se", se)
    dire(f"{nom:22s} exact={float(exact):.6f}  MC={m:.6f}  se={se:.6f}  z={z:+.2f}")


def vp_min(m):
    return float(np.linalg.eigvalsh(np.asarray(m, dtype=float)).min())


def equicorr(n, rho):
    return (1 - rho) * sp.eye(n) + rho * sp.ones(n, n)


def rang_cov(x):
    """Rang de la covariance empirique centrée d'un tableau T x N (tolérance relative standard de numpy)."""
    y = x - x.mean(axis=0)
    return int(np.linalg.matrix_rank(y.T.dot(y) / (len(x) - 1)))


# ============================================================ LEÇON
# --- Théorème spectral, Rayleigh : contrôle sur des matrices symétriques aléatoires.
for _ in range(50):
    b = rng.standard_normal((6, 6))
    a = (b + b.T) / 2
    lam, q = np.linalg.eigh(a)
    assert np.allclose(q.T.dot(q), np.eye(6)) and np.allclose(q.dot(np.diag(lam)).dot(q.T), a)
    x = rng.standard_normal((6, 1000))
    r = np.einsum("ik,ij,jk->k", x, a, x) / np.einsum("ik,ik->k", x, x)
    assert r.max() <= lam[-1] + 1e-12 and r.min() >= lam[0] - 1e-12
    assert np.isclose(np.trace(a), lam.sum()) and np.isclose(np.linalg.det(a), lam.prod())
dire("Théorème spectral, encadrement de Rayleigh, trace = somme et déterminant = produit des valeurs propres : 50 matrices 6 x 6.")

# --- Équicorrélation : valeurs propres 1 + (n-1) rho (simple) et 1 - rho (multiplicité n-1), pour n de 2 à 8.
rho = sp.symbols("rho")
for n in range(2, 9):
    c = equicorr(n, rho)
    # Base de vecteurs propres : le vecteur 1 et les n - 1 vecteurs e_1 - e_j, libres (matrice de rang n).
    un = sp.ones(n, 1)
    assert sp.expand(c * un - (1 + (n - 1) * rho) * un) == sp.zeros(n, 1)
    base = [un]
    for j in range(1, n):
        v = sp.zeros(n, 1)
        v[0], v[j] = 1, -1
        assert sp.expand(c * v - (1 - rho) * v) == sp.zeros(n, 1)
        base.append(v)
    assert sp.Matrix.hstack(*base).rank() == n
    borne = sp.Rational(-1, n - 1)
    assert vp_min(equicorr(n, borne)) > -1e-12 and vp_min(equicorr(n, borne - sp.Rational(1, 100))) < 0
dire("Équicorrélation n x n : valeurs propres 1 + (n-1) rho et 1 - rho, borne -1/(n-1), vérifiées pour n de 2 à 8.")
# Exemple de la leçon : n = 5, rho = -3/10.
fraction("L_eq5_borne", Fraction(-1, 4))
r5 = Fraction(-3, 10)
fraction("L_eq5_vp1", 1 + 4 * r5)
fraction("L_eq5_vp2", 1 - r5)
fraction("L_eq5_var", 5 + 5 * 4 * r5)
assert abs(vp_min(equicorr(5, sp.Rational(-3, 10))) - float(1 + 4 * r5)) < 1e-12

# --- Développement de det(A + t I) en somme de mineurs principaux, sur une matrice entière 4 x 4.
t = sp.symbols("t")
a4 = sp.Matrix(4, 4, lambda i, j: int(rng.integers(-5, 6)))
a4 = a4 + a4.T
poly = sp.Poly((a4 + t * sp.eye(4)).det(), t)
for k in range(5):
    e_k = sum((a4.extract(list(s), list(s)).det() for s in itertools.combinations(range(4), k)), sp.Integer(0)) if k else 1
    assert poly.coeff_monomial(t ** (4 - k)) == e_k
dire("det(A + t I) = somme sur k de t^(n-k) fois la somme des mineurs principaux d'ordre k : vérifié sur une matrice 4 x 4.")

# --- Contre-exemple aux mineurs principaux dominants.
m_ce = sp.Matrix([[1, 1, 0], [1, 1, 0], [0, 0, -1]])
dominants = [m_ce[:k, :k].det() for k in (1, 2, 3)]
assert dominants == [1, 0, 0]
V["L_ce_dom"] = [int(d) for d in dominants]
vps = sorted(m_ce.eigenvals(multiple=True), reverse=True)
V["L_ce_vp"] = [int(v) for v in vps]
assert min(vps) < 0
val("L_ce_mineur33", m_ce[2, 2])
dire(f"Contre-exemple : mineurs dominants {dominants}, valeurs propres {vps}.")

# --- Sylvester strict : mineurs dominants > 0 <=> définie positive, sur 2000 matrices aléatoires.
accord = 0
for _ in range(2000):
    b = rng.integers(-3, 4, size=(3, 3)).astype(float)
    a = b + b.T + rng.integers(0, 6) * np.eye(3)
    dom = all(np.linalg.det(a[:k, :k]) > 1e-9 for k in (1, 2, 3))
    dp = np.linalg.eigvalsh(a).min() > 1e-9
    assert dom == dp
    accord += dp
dire(f"Critère de Sylvester vérifié sur 2000 matrices 3 x 3 entières ({accord} définies positives).")

# --- Projection : matrice de centrage et matrice chapeau.
for tt in (2, 5, 8):
    mt = sp.eye(tt) - sp.ones(tt, tt) / tt
    assert mt == mt.T and mt * mt == mt and mt.trace() == tt - 1 and mt.rank() == tt - 1
mt = np.eye(50) - np.ones((50, 50)) / 50
assert np.allclose(mt.dot(mt), mt) and np.linalg.matrix_rank(mt) == 49 and np.isclose(np.trace(mt), 49)
x_h = rng.standard_normal((40, 4))
h = x_h.dot(np.linalg.solve(x_h.T.dot(x_h), x_h.T))
assert np.allclose(h, h.T) and np.allclose(h.dot(h), h) and np.isclose(np.trace(h), 4)
dire("Matrice de centrage (trace T - 1) et matrice chapeau (trace p) : symétriques, idempotentes.")

# --- Rang de la covariance empirique : min(N, T - 1) avec probabilité 1, pour plusieurs lois à densité.
cas_rang = [(30, 50), (50, 50), (51, 50), (80, 50), (2, 7)]
for tt, nn in cas_rang:
    for loi in ("normale", "student", "uniforme"):
        for _ in range(5):
            if loi == "normale":
                x = rng.standard_normal((tt, nn)).dot(rng.standard_normal((nn, nn)))
            elif loi == "student":
                x = rng.standard_t(3, size=(tt, nn))
            else:
                x = rng.random((tt, nn))
            assert rang_cov(x) == min(nn, tt - 1), (tt, nn, loi)
dire("Rang de la covariance empirique = min(N, T - 1) : 75 tirages (lois normale corrélée, Student à 3 degrés, uniforme).")
# Une loi sans densité peut faire baisser le rang : rendements entiers pris dans {-1, 0, 1}.
egal = sum(rang_cov(rng.integers(-1, 2, size=(4, 3)).astype(float)) < 3 for _ in range(2000))
val("L_rang_discret_part", egal / 2000)
dire(f"Loi discrète sur {{-1, 0, 1}}, T = 4, N = 3 : rang < 3 dans {egal} tirages sur 2000.")

# --- Dispersion des valeurs propres empiriques : E[lambda_max(S)] >= lambda_max(Sigma), E[lambda_min(S)] <= lambda_min(Sigma).
nn, tt = 20, 40
lmax, lmin = [], []
for _ in range(2000):
    x = rng.standard_normal((tt, nn))
    vp = np.linalg.eigvalsh(np.cov(x.T))
    lmax.append(vp[-1])
    lmin.append(vp[0])
val("L_disp_lmax_moy", np.mean(lmax))
val("L_disp_lmin_moy", np.mean(lmin))
assert np.mean(lmax) - 4 * np.std(lmax) / np.sqrt(2000) > 1 and np.mean(lmin) + 4 * np.std(lmin) / np.sqrt(2000) < 1
dire(f"Sigma = I, N = {nn}, T = {tt} : E[lambda_max(S)] ~ {np.mean(lmax):.3f} > 1, E[lambda_min(S)] ~ {np.mean(lmin):.3f} < 1.")

# --- Eckart-Young : aucune matrice de rang k aléatoire ne fait mieux que la SVD tronquée ; erreur = racine des s_j^2 omis.
for _ in range(20):
    x = rng.standard_normal((8, 5))
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    for k in (1, 2, 3):
        xk = (u[:, :k] * s[:k]).dot(vt[:k])
        err = np.linalg.norm(x - xk)
        assert np.isclose(err, np.sqrt((s[k:] ** 2).sum())) and np.isclose(np.linalg.norm(x - xk, 2), s[k])
        for _ in range(50):
            b = rng.standard_normal((8, k)).dot(rng.standard_normal((k, 5)))
            b = xk + 0.1 * b  # perturbation : rang au plus 2k, on reprojette au rang k
            ub, sb, vbt = np.linalg.svd(b, full_matrices=False)
            bk = (ub[:, :k] * sb[:k]).dot(vbt[:k])
            assert np.linalg.norm(x - bk) >= err - 1e-12 and np.linalg.norm(x - bk, 2) >= s[k] - 1e-12
dire("Eckart-Young (normes de Frobenius et spectrale) : 20 matrices 8 x 5, k = 1 à 3, 50 concurrents de rang k chacune.")

# --- Gradients : différences finies.
for _ in range(20):
    a = rng.standard_normal((4, 4))
    x0 = rng.standard_normal(4)
    bb = rng.standard_normal(4)
    gq = (a + a.T).dot(x0)
    gm = 2 * a.T.dot(a.dot(x0) - bb)
    for j in range(4):
        e = np.zeros(4)
        e[j] = 1e-6
        assert abs((x0 + e).dot(a).dot(x0 + e) - (x0 - e).dot(a).dot(x0 - e) - 2e-6 * gq[j]) < 1e-9
        f = lambda z: np.sum((a.dot(z) - bb) ** 2)  # noqa: E731
        assert abs(f(x0 + e) - f(x0 - e) - 2e-6 * gm[j]) < 1e-8
dire("Gradients de x'Ax et de ||Ax - b||^2 : différences finies sur 20 cas.")

# --- ACP : rendements standardisés => matrice de corrélation ; diviseur T ou T - 1 sans effet sur directions et parts.
x = rng.standard_normal((60, 5)).dot(rng.standard_normal((5, 5))) * np.array([1, 2, 5, 10, 0.5])
y = x - x.mean(axis=0)
z = y / y.std(axis=0)  # diviseur T
assert np.allclose(z.T.dot(z) / 60, np.corrcoef(x.T))
z1 = y / y.std(axis=0, ddof=1)
assert np.allclose(z1.T.dot(z1) / 59, np.corrcoef(x.T))
dire("Rendements standardisés : Z'Z/T = matrice de corrélation (diviseurs cohérents), vérifié.")

# --- Modèle à un facteur : Sigma = s_f^2 b b' + s_e^2 I ; b est vecteur propre, valeur propre s_f^2 |b|^2 + s_e^2.
bet = rng.uniform(0.5, 1.5, 8)
sig = 0.01**2 * np.outer(bet, bet) + 0.02**2 * np.eye(8)
lam, q = np.linalg.eigh(sig)
assert np.isclose(lam[-1], 0.01**2 * bet.dot(bet) + 0.02**2) and np.allclose(lam[:-1], 0.02**2)
assert np.isclose(abs(q[:, -1].dot(bet / np.linalg.norm(bet))), 1)
# Variances idiosyncratiques inégales : la première composante n'est plus exactement b.
sig2 = 0.01**2 * np.outer(bet, bet) + np.diag(rng.uniform(0.01, 0.03, 8) ** 2)
q2 = np.linalg.eigh(sig2)[1][:, -1]
cos2 = abs(q2.dot(bet / np.linalg.norm(bet)))
assert cos2 < 1 - 1e-6
dire(f"Un facteur : b vecteur propre exact si variances idiosyncratiques égales ; sinon cosinus {cos2:.4f} < 1.")
# Perron : matrice à coefficients > 0 => premier vecteur propre de signe constant.
for _ in range(200):
    b = rng.random((6, 6)) + 0.01
    a = b + b.T
    q1 = np.linalg.eigh(a)[1][:, -1]
    assert (q1 > 0).all() or (q1 < 0).all()
dire("Coefficients tous > 0 : premier vecteur propre de signe constant sur 200 matrices.")

# --- Ledoit-Wolf : valeurs propres (1 - d) lambda + d mu, conditionnement décroissant en d ; formule de sklearn.
lam_s = np.array([5.0, 2.0, 0.5, 0.0])
mu = lam_s.mean()
kap = [((1 - d) * lam_s.max() + d * mu) / ((1 - d) * lam_s.min() + d * mu) for d in np.linspace(0.01, 1, 100)]
assert all(k1 > k2 for k1, k2 in zip(kap, kap[1:])) and np.isclose(kap[-1], 1)
from sklearn.covariance import LedoitWolf  # noqa: E402
import sklearn  # noqa: E402

for tt, nn in [(40, 50), (200, 50), (30, 10)]:
    x = rng.standard_normal((tt, nn)).dot(np.diag(rng.uniform(0.5, 2, nn)))
    y = x - x.mean(axis=0)
    s = y.T.dot(y) / tt
    m_ = np.trace(s) / nn
    d2 = np.sum((s - m_ * np.eye(nn)) ** 2) / nn
    b2 = sum(np.sum((np.outer(r, r) - s) ** 2) for r in y) / nn / tt**2
    delta = min(b2, d2) / d2
    lw = LedoitWolf().fit(x)
    assert np.isclose(lw.shrinkage_, delta) and np.allclose(lw.covariance_, (1 - delta) * s + delta * m_ * np.eye(nn))
dire(f"Ledoit-Wolf : intensité recalculée = shrinkage_ de sklearn {sklearn.__version__} sur 3 jeux de données.")

# ============================================================ POINTS DE CONTRÔLE
fraction("C_eq4_vp", 1 + 3 * Fraction(-2, 5))     # n = 4, rho = -0,4
fraction("C_eq4_borne", Fraction(-1, 3))
val("C_centrage_trace", 50 - 1)                   # T = 50
vp_c = [Fraction(48, 10), Fraction(24, 10), Fraction(8, 10), Fraction(0)]
mu_c = sum(vp_c) / 4
val("C_lw_mu", mu_c)
val("C_lw_vp_min", min((1 - Fraction(1, 4)) * v + Fraction(1, 4) * mu_c for v in vp_c))
val("C_lw_d_id", min((1 - Fraction(1, 4)) * v + Fraction(1, 4) for v in vp_c))
val("C_lw_d_inv", min(Fraction(1, 4) * v + (1 - Fraction(1, 4)) * mu_c for v in vp_c))
val("C_eq4_ex", 1 + 3 * Fraction(-3, 10))
val("L_coef_distincts", 500 * 501 // 2)
val("L_acp_rho", Fraction(36, 100))
assert Fraction(6, 10) ** 2 / (Fraction(6, 10) ** 2 + Fraction(8, 10) ** 2) == Fraction(36, 100)

# ============================================================ QUESTIONS
# AL-01 : équicorrélation 5 x 5, plus petite valeur de rho.
fraction("Q01_borne", Fraction(-1, 4))
fraction("Q01_d_n", Fraction(-1, 5))
fraction("Q01_d_3", Fraction(-1, 2))
val("Q01_vp_moins_demi", 1 + 4 * Fraction(-1, 2))
val("Q01_vp_moins_un", 1 + 4 * (-1))
val("Q01_mult", 5 - 1)
assert abs(vp_min(equicorr(5, sp.Rational(-1, 4)))) < 1e-12
assert vp_min(equicorr(5, sp.Rational(-1, 5))) > 0

# AL-02 : corrélation minimale de X2 et X3 quand corr(X1, X2) = corr(X1, X3) = 0,8.
a_, c_ = sp.symbols("a c")
c3 = sp.Matrix([[1, a_, a_], [a_, 1, c_], [a_, c_, 1]])
assert sp.factor(c3.det() - (1 - c_) * (1 + c_ - 2 * a_**2)) == 0
a02 = sp.Rational(4, 5)
cmin = 2 * a02**2 - 1
fraction("Q02_exact", frac_sp(cmin))
fraction("Q02_d_produit", frac_sp(a02**2))
fraction("Q02_d_equi", Fraction(-1, 2))
assert abs(vp_min(c3.subs({a_: a02, c_: cmin}))) < 1e-12
assert vp_min(c3.subs({a_: a02, c_: cmin - sp.Rational(1, 100)})) < 0
val("Q02_vp_c0", vp_min(c3.subs({a_: a02, c_: 0})))
val("Q02_vp_cdemi", vp_min(c3.subs({a_: a02, c_: sp.Rational(-1, 2)})))
fraction("Q02_a2", frac_sp(a02**2))
fraction("Q02_2a2", frac_sp(2 * a02**2))
# Simulation : un vecteur gaussien de corrélation C(0,8 ; 0,28) existe ; corrélations empiriques.
cm = np.array(c3.subs({a_: a02, c_: cmin}).tolist(), dtype=float)
w, qq = np.linalg.eigh(cm)
racine = qq.dot(np.diag(np.sqrt(np.clip(w, 0, None))))
g = rng.standard_normal((400_000, 3)).dot(racine.T)
controle_mc("Q02_corr23", float(cmin), (g[:, 1] * g[:, 2]))
dire("AL-02 : det = (1 - c)(1 + c - 2a^2) vérifié symboliquement ; borne 2a^2 - 1 atteinte.")

# AL-03 : mineurs dominants nuls, mineur principal {2, 3} négatif.
a03 = sp.Matrix([[0, 0, 0], [0, 2, 3], [0, 3, 1]])
assert [a03[:k, :k].det() for k in (1, 2, 3)] == [0, 0, 0]
val("Q03_mineur23", a03.extract([1, 2], [1, 2]).det())
vp03 = sorted(a03.eigenvals(multiple=True), key=float)
assert sp.simplify(vp03[0] - (3 - sp.sqrt(37)) / 2) == 0
val("Q03_vp_neg", float(vp03[0]))
val("Q03_racine", 37)
x03 = np.array([0, 1.0, -1.0])
val("Q03_forme", x03.dot(np.array(a03.tolist(), dtype=float)).dot(x03))

# AL-04 : N = 300, T = 120 : nombre de valeurs propres nulles.
n04, t04 = 300, 120
val("Q04_exact", n04 - (t04 - 1))
val("Q04_d_sans_centrage", n04 - t04)
val("Q04_rang", t04 - 1)
assert rang_cov(rng.standard_normal((t04, n04))) == t04 - 1

# AL-05 : démoyennage transversal, N = 100, T = 250.
n05, t05 = 100, 250
val("Q05_exact", n05 - 1)
val("Q05_d_t", t05 - 1)
val("Q05_d_double", n05 - 2)
for _ in range(3):
    x = rng.standard_normal((t05, n05)).dot(rng.standard_normal((n05, n05)))
    xd = x - x.mean(axis=1, keepdims=True)
    assert rang_cov(xd) == n05 - 1
    assert np.allclose(np.cov(xd.T).dot(np.ones(n05)), 0)
dire("AL-05 : rang 99 après démoyennage transversal (3 tirages), vecteur 1 dans le noyau.")

# AL-06 : E[somme des carrés des résidus] = (T - p) sigma^2, T = 100, p = 4.
t06, p06 = 100, 4
val("Q06_exact", t06 - p06)
val("Q06_d_p", p06)
val("Q06_d_double", t06 - p06 - 1)
x06 = np.column_stack([np.ones(t06), rng.standard_normal((t06, 3))])
h06 = x06.dot(np.linalg.solve(x06.T.dot(x06), x06.T))
assert np.isclose(np.trace(np.eye(t06) - h06), t06 - p06)
eps = rng.standard_normal((50_000, t06)) * 0.02
res = eps - eps.dot(h06)  # (I - H) eps, H symétrique
controle_mc("Q06_rss_sur_sigma2", t06 - p06, (res**2).sum(axis=1) / 0.02**2)

# AL-07 : conditionnement, valeurs propres 4, 1, 0,04 ; perturbation relative 0,1 %.
vp07 = [Fraction(4), Fraction(1), Fraction(1, 25)]
kappa07 = max(vp07) / min(vp07)
val("Q07_kappa", kappa07)
fraction("Q07_exact", kappa07 * Fraction(1, 1000))
fraction("Q07_d_inv", Fraction(1, 1000) / min(vp07))
fraction("Q07_d_racine", Fraction(1, 1000) * 10)
fraction("Q07_d_max", Fraction(1, 1000) * max(vp07))
assert Fraction(10) ** 2 == kappa07
s07 = np.diag([4.0, 1.0, 0.04])
m07 = np.array([1.0, 0, 0])
dm07 = np.array([0, 0, 1e-3])
w07 = np.linalg.solve(s07, m07)
dw07 = np.linalg.solve(s07, m07 + dm07) - w07
assert np.isclose(np.linalg.norm(dw07) / np.linalg.norm(w07), float(kappa07) * 1e-3)
for _ in range(2000):  # borne jamais dépassée
    q = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    s_ = q.dot(s07).dot(q.T)
    mm, dm = rng.standard_normal(3), rng.standard_normal(3)
    dm *= 1e-3 * np.linalg.norm(mm) / np.linalg.norm(dm)
    ww = np.linalg.solve(s_, mm)
    assert np.linalg.norm(np.linalg.solve(s_, dm)) / np.linalg.norm(ww) <= float(kappa07) * 1e-3 * (1 + 1e-9)
dire("AL-07 : borne kappa * 0,1 % atteinte par l'exemple diagonal, jamais dépassée sur 2000 cas.")

# AL-08 : gradient de x'Ax, A = [[1, 2], [0, 3]], x = (1, -1).
a08 = sp.Matrix([[1, 2], [0, 3]])
x08 = sp.Matrix([1, -1])
xs = sp.symbols("x1 x2")
f08 = (sp.Matrix(xs).T * a08 * sp.Matrix(xs))[0]
grad = [sp.diff(f08, v).subs({xs[0]: 1, xs[1]: -1}) for v in xs]
assert sp.Matrix(grad) == (a08 + a08.T) * x08
V["Q08_exact"] = [int(g_) for g_ in grad]
V["Q08_d_2ax"] = [int(g_) for g_ in 2 * a08 * x08]
V["Q08_d_2atx"] = [int(g_) for g_ in 2 * a08.T * x08]
V["Q08_d_ax"] = [int(g_) for g_ in a08 * x08]
V["Q08_ax"] = [int(g_) for g_ in a08 * x08]
V["Q08_atx"] = [int(g_) for g_ in a08.T * x08]
sym08 = a08 + a08.T
for (i_, j_), nom in {(0, 0): "11", (0, 1): "12", (1, 1): "22"}.items():
    val("Q08_s" + nom, sym08[i_, j_])
    val("Q08_h" + nom, sym08[i_, j_] / 2)
assert sym08[0, 1] == sym08[1, 0] == a08[0, 1] + a08[1, 0]

# AL-09 : Eckart-Young, valeurs singulières 4, 3, 2, 1, rang 2.
s09 = [4, 3, 2, 1]
val("Q09_somme_carres", sum(v**2 for v in s09[2:]))
val("Q09_exact", np.sqrt(sum(v**2 for v in s09[2:])))
val("Q09_d_spectrale", s09[2])
val("Q09_d_somme", sum(s09[2:]))
u09 = np.linalg.qr(rng.standard_normal((6, 4)))[0]
v09 = np.linalg.qr(rng.standard_normal((4, 4)))[0]
x09 = (u09 * np.array(s09, dtype=float)).dot(v09.T)
u_, s_, vt_ = np.linalg.svd(x09)
assert np.allclose(s_, s09) and np.isclose(np.linalg.norm(x09 - (u_[:, :2] * s_[:2]).dot(vt_[:2])), V["Q09_exact"])

# AL-10 : ACP sur rendements standardisés, N = 5, quatre plus grandes valeurs propres 2,6 ; 1,1 ; 0,7 ; 0,4.
vp10 = [Fraction(26, 10), Fraction(11, 10), Fraction(7, 10), Fraction(4, 10)]
lam5 = 5 - sum(vp10)
fraction("Q10_lam5", lam5)
assert lam5 >= 0 and lam5 <= min(vp10)
fraction("Q10_somme2", vp10[0] + vp10[1])
fraction("Q10_exact", (vp10[0] + vp10[1]) / 5)
fraction("Q10_d_somme4", sum(vp10))
fraction("Q10_d_quatre", (vp10[0] + vp10[1]) / sum(vp10))
fraction("Q10_d_premiere", vp10[0] / 5)
# L'énoncé est réalisable : on construit une matrice de corrélation 5 x 5 ayant exactement ces valeurs propres.
# Départ Q diag(lambda) Q' (Q orthogonale aléatoire), puis rotations planes (i, j) qui ramènent un coefficient
# diagonal à 1 sans changer le spectre (méthode de Bendel et Mickey, 1978).
spec10 = np.array([float(v) for v in vp10 + [lam5]])
q10 = np.linalg.qr(rng.standard_normal((5, 5)))[0]
a10 = q10.dot(np.diag(spec10)).dot(q10.T)
for _ in range(10):
    dg = np.diag(a10)
    if np.allclose(dg, 1, atol=1e-13):
        break
    i, j = int(np.argmin(dg)), int(np.argmax(dg))
    # Rotation d'angle th dans le plan (i, j) : nouveau a_ii = c^2 a_ii + 2 c s a_ij + s^2 a_jj = 1, avec t = s / c.
    pa, pb, pc = a10[j, j] - 1, 2 * a10[i, j], a10[i, i] - 1
    tt_ = (-pb + np.sqrt(pb * pb - 4 * pa * pc)) / (2 * pa)
    th = np.arctan(tt_)
    rot = np.eye(5)
    rot[i, i] = rot[j, j] = np.cos(th)
    rot[i, j], rot[j, i] = -np.sin(th), np.sin(th)
    a10 = rot.T.dot(a10).dot(rot)
assert np.allclose(np.diag(a10), 1) and np.allclose(np.linalg.eigvalsh(a10), np.sort(spec10))
dire("AL-10 : une matrice de corrélation 5 x 5 de valeurs propres 2,6 ; 1,1 ; 0,7 ; 0,4 ; 0,2 existe (construction explicite).")

# AL-11 : un facteur, N = 20, sigma_F = 1 %, sigma_E = 2 %, beta = 1.
n11 = 20
sf2, se2 = Fraction(1, 10000), Fraction(4, 10000)
rho11 = sf2 / (sf2 + se2)
fraction("Q11_rho", rho11)
fraction("Q11_lam1", 1 + (n11 - 1) * rho11)
fraction("Q11_exact", (1 + (n11 - 1) * rho11) / n11)
fraction("Q11_d_inv", Fraction(1, n11))
fraction("Q11_d_n", (1 + n11 * rho11) / n11)
fraction("Q11_var_tot", sf2 + se2)
sig11 = sp.Matrix(n11, n11, lambda i, j: sp.Rational(sf2.numerator, sf2.denominator) + (sp.Rational(se2.numerator, se2.denominator) if i == j else 0))
d11 = sp.diag(*[1 / sp.sqrt(sig11[i, i]) for i in range(n11)])
corr11 = d11 * sig11 * d11
assert corr11[0, 1] == sp.Rational(rho11.numerator, rho11.denominator)
vp11 = np.linalg.eigvalsh(np.array(corr11.tolist(), dtype=float))
assert np.isclose(vp11[-1] / n11, float(V["Q11_exact"]))
# Simulation : part de la première composante de la corrélation empirique, T = 20000 (estimateur convergent).
f11 = rng.standard_normal((20_000, 1)) * 0.01
r11 = f11 + rng.standard_normal((20_000, n11)) * 0.02
part = np.linalg.eigvalsh(np.corrcoef(r11.T))[-1] / n11
assert abs(part - float(V["Q11_exact"])) < 0.01
val("Q11_part_simulee", part)
dire(f"AL-11 : part exacte {float(V['Q11_exact']):.4f}, part empirique (T = 20000) {part:.4f}.")

# AL-12 : Ledoit-Wolf, valeurs propres de S 6 ; 3 ; 1 ; 0 ; delta = 1/5.
vp12 = [Fraction(6), Fraction(3), Fraction(1), Fraction(0)]
mu12 = sum(vp12) / 4
d12 = Fraction(1, 5)
fraction("Q12_mu", mu12)
nouv = [(1 - d12) * v + d12 * mu12 for v in vp12]
V["Q12_nouv"] = [float(v) for v in nouv]
fraction("Q12_nouv_max", max(nouv))
fraction("Q12_nouv_min", min(nouv))
fraction("Q12_exact", max(nouv) / min(nouv))
inv = [d12 * v + (1 - d12) * mu12 for v in vp12]
fraction("Q12_d_inverse", max(inv) / min(inv))
cible_i = [(1 - d12) * v + d12 for v in vp12]
fraction("Q12_d_identite", max(cible_i) / min(cible_i))
fraction("Q12_d_id_max", max(cible_i))
fraction("Q12_d_id_min", min(cible_i))
q12 = np.linalg.qr(rng.standard_normal((4, 4)))[0]
s12 = q12.dot(np.diag([float(v) for v in vp12])).dot(q12.T)
sh = (1 - 0.2) * s12 + 0.2 * np.trace(s12) / 4 * np.eye(4)
assert np.isclose(np.linalg.cond(sh), float(V["Q12_exact"]))
assert np.allclose(np.linalg.eigvalsh(sh), sorted(float(v) for v in nouv))

Path(__file__).with_name("seance_algebre_1.out.json").write_text(json.dumps(V, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
dire(f"{len(V)} valeurs exportées dans verif/seance_algebre_1.out.json.")
dire("Toutes les vérifications passent.")
Path(__file__).with_name("seance_algebre_1.out.txt").write_text("\n".join(LIGNES) + "\n", encoding="utf-8", newline="\n")
