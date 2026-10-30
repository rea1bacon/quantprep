"""Vérification de tous les nombres de la séance de ML 5 (leçon L-ML-5 et questions ML5-01 à ML5-14).

Principe :
  1. chaque valeur citée est calculée ici, en fractions exactes quand c'est possible ;
  2. chaque formule démontrée dans la leçon est contrôlée sur des cas numériques : formes de sortie comparées à
     une vraie convolution numpy (et à scipy quand le pas et la dilatation valent 1), champ réceptif mesuré par
     perturbation de l'entrée, convolution transposée contrôlée comme adjointe, équations de Bellman résolues
     exactement puis par itération sur la valeur, optimum de l'elastic net retrouvé par minimisation numérique ;
  3. chaque probabilité ou espérance est aussi estimée par simulation Monte-Carlo, dont l'écart à l'exact doit
     rester inférieur à 4 erreurs standard.
Les expériences avec scikit-learn sont dans la leçon elle-même (blocs exécutés à la construction).

Graine fixe : la sortie est identique à chaque exécution.
Sorties : verif/seance_ml_5.out.json (lu par outils/construire.py) et verif/seance_ml_5.out.txt (ce résumé).
"""
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar
from scipy.signal import correlate2d

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
    dire(f"{nom:24s} exact={float(exact):.6f}  MC={float(estimation):.6f}  se={float(erreur_std):.6f}  z={z:+.2f}")


def mc_moyenne(nom, exact, echantillon):
    echantillon = np.asarray(echantillon, dtype=float)
    controle_mc(nom, exact, echantillon.mean(), echantillon.std(ddof=1) / np.sqrt(len(echantillon)))


# =====================================================================================
# A. Arithmétique des réseaux convolutifs
# =====================================================================================
def taille(n, k, p=0, s=1, d=1):
    """Formule (C1) : taille de sortie d'une convolution ou d'un pooling le long d'un axe."""
    assert n + 2 * p >= d * (k - 1) + 1
    return (n + 2 * p - d * (k - 1) - 1) // s + 1


def conv2d(x, w, p=0, s=1, d=1):
    """Vraie convolution (corrélation croisée) : x de forme (H, W, C), w de forme (k, k, C, F).

    La sortie est construite position par position, tant que la fenêtre tient dans l'entrée complétée :
    sa forme ne doit rien à la formule (C1), qu'elle sert à contrôler.
    """
    k, f = w.shape[0], w.shape[3]
    xp = np.pad(x, ((p, p), (p, p), (0, 0)))
    etendue = d * (k - 1) + 1
    lignes = []
    i = 0
    while i + etendue <= xp.shape[0]:
        ligne = []
        j = 0
        while j + etendue <= xp.shape[1]:
            fenetre = xp[i:i + etendue:d, j:j + etendue:d, :]
            ligne.append([np.sum(fenetre * w[:, :, :, c]) for c in range(f)])
            j += s
        lignes.append(ligne)
        i += s
    return np.array(lignes)


def maxpool2d(x, k, s, p=0):
    xp = np.pad(x, ((p, p), (p, p), (0, 0)), constant_values=-np.inf)
    lignes = []
    i = 0
    while i + k <= xp.shape[0]:
        ligne = []
        j = 0
        while j + k <= xp.shape[1]:
            ligne.append(xp[i:i + k, j:j + k, :].max(axis=(0, 1)))
            j += s
        lignes.append(ligne)
        i += s
    return np.array(lignes)


# Contrôle de (C1) sur 300 configurations tirées au hasard, contre la vraie convolution.
nb = 0
for _ in range(300):
    k, s, d, p = int(rng.integers(1, 6)), int(rng.integers(1, 4)), int(rng.integers(1, 4)), int(rng.integers(0, 4))
    n = int(rng.integers(d * (k - 1) + 1, d * (k - 1) + 14))
    c, f = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    y = conv2d(rng.normal(size=(n, n, c)), rng.normal(size=(k, k, c, f)), p, s, d)
    assert y.shape == (taille(n, k, p, s, d), taille(n, k, p, s, d), f)
    nb += 1
dire(f"(C1) forme de sortie : {nb} configurations concordent avec une vraie convolution numpy.")
# Valeurs comparées à scipy (pas 1, dilatation 1), conventions valid et same.
for _ in range(20):
    n, k = int(rng.integers(5, 10)), int(rng.choice([1, 3, 5]))
    x, w = rng.normal(size=(n, n, 1)), rng.normal(size=(k, k, 1, 1))
    assert np.allclose(conv2d(x, w)[:, :, 0], correlate2d(x[:, :, 0], w[:, :, 0, 0], mode="valid"))
    assert np.allclose(conv2d(x, w, p=(k - 1) // 2)[:, :, 0], correlate2d(x[:, :, 0], w[:, :, 0, 0], mode="same"))
# Convention same : sortie plafond(n / s), avec un remplissage total max((plafond(n/s) - 1) s + d (k - 1) + 1 - n, 0).
for n, k, s, d in itertools.product(range(5, 30), [1, 2, 3, 5], [1, 2, 3], [1, 2]):
    cible = math.ceil(n / s)
    total = max((cible - 1) * s + d * (k - 1) + 1 - n, 0)
    assert (n + total - d * (k - 1) - 1) // s + 1 == cible
dire("Convention same : le remplissage total indiqué donne bien plafond(n / s).")


def parametres(k, c_in, f, biais=True):
    return f * (k * k * c_in + (1 if biais else 0))


# Exemple de la leçon : image 96 x 96 x 3, deux branches parallèles, concaténation, max-pooling.
V["LC_n"], V["LC_fA"], V["LC_fB"] = 96, 16, 24
V["LC_pA"], V["LC_pB"] = (3 - 1) // 2, (5 - 1) // 2
assert taille(96, 3, V["LC_pA"]) == 96 and taille(96, 5, V["LC_pB"]) == 96
V["LC_canaux"] = 16 + 24
V["LC_pool"] = taille(96, 2, 0, 2)
V["LC_validA"], V["LC_validB"] = taille(96, 3), taille(96, 5)
V["LC_paramsA"], V["LC_paramsB"] = parametres(3, 3, 16), parametres(5, 3, 24)
V["LC_params"] = V["LC_paramsA"] + V["LC_paramsB"]
V["LC_seq_params"] = parametres(5, 16, 24)          # lecture en série : la seconde couche lit 16 canaux
V["LC_pas2"] = taille(96, 7, 3, 2)
V["LC_dil"] = taille(96, 3, 0, 1, 2)
V["LC_impair"] = taille(97, 2, 0, 2)
V["LC_etendue_dil"] = 2 * (3 - 1) + 1
# Même enchaînement sur un petit tenseur 12 x 12 x 3, par une vraie convolution.
x = rng.normal(size=(12, 12, 3))
a = conv2d(x, rng.normal(size=(3, 3, 3, 16)), p=1)
b = conv2d(x, rng.normal(size=(5, 5, 3, 24)), p=2)
z = maxpool2d(np.concatenate([a, b], axis=2), 2, 2)
assert a.shape == (12, 12, 16) and b.shape == (12, 12, 24) and z.shape == (6, 6, 40)
assert conv2d(x, rng.normal(size=(3, 3, 3, 16))).shape[0] != conv2d(x, rng.normal(size=(5, 5, 3, 24))).shape[0]
# Convolutions 1 x 1 avant une convolution coûteuse (module à branches parallèles).
V["LI_direct"] = parametres(5, 192, 32)
V["LI_reduit"] = parametres(1, 192, 16) + parametres(5, 16, 32)
V["LI_rapport"] = V["LI_direct"] / V["LI_reduit"]
# Saut par concaténation dans un décodeur : 64 + 64 canaux lus par une convolution 3 x 3 à 64 filtres.
V["LU_canaux"] = 64 + 64
V["LU_params"] = parametres(3, 128, 64)
V["LU_params_sans_saut"] = parametres(3, 64, 64)


def champ(couches):
    """(C3) : couches = liste de (k, s, d) ; renvoie les champs réceptifs successifs et le saut final."""
    r, saut, etapes = 1, 1, []
    for k, s, d in couches:
        r += d * (k - 1) * saut
        saut *= s
        etapes.append(r)
    return etapes, saut


def champ_mesure(couches, n=200):
    """Champ réceptif mesuré : on perturbe chaque entrée et on regarde si une sortie centrale change (dimension 1)."""
    def reseau(x):
        for k, s, d in couches:
            w = np.ones(k)
            x = np.array([np.sum(x[i:i + d * (k - 1) + 1:d] * w) for i in range(0, len(x) - d * (k - 1), s)])
        return x
    base = reseau(np.zeros(n))
    centre = len(base) // 2
    touche = []
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        if reseau(e)[centre] != base[centre]:
            touche.append(i)
    return max(touche) - min(touche) + 1, len(touche)


lecon_champ = [(3, 1, 1), (2, 2, 1), (3, 1, 1), (2, 2, 1), (3, 1, 1)]
etapes, saut = champ(lecon_champ)
V["LR_etapes"], V["LR_final"], V["LR_saut"] = etapes, etapes[-1], saut
assert champ_mesure(lecon_champ) == (etapes[-1], etapes[-1])
V["LR_sans_pas"] = 1 + sum(k - 1 for k, _, _ in lecon_champ)
for _ in range(40):
    couches = [(int(rng.integers(1, 4)), int(rng.integers(1, 3)), 1) for _ in range(int(rng.integers(1, 5)))]
    assert champ_mesure(couches)[0] == champ(couches)[0][-1]
dire(f"(C3) champ réceptif avec pas : formule contrôlée par perturbation sur 41 réseaux ; exemple {etapes}.")


def taille_transposee(n, k, p=0, s=1, d=1, op=0):
    return (n - 1) * s - 2 * p + d * (k - 1) + op + 1


def conv1d(x, w, s):
    return np.array([np.sum(x[i:i + len(w)] * w) for i in range(0, len(x) - len(w) + 1, s)])


def conv1d_transposee(y, w, s):
    """Chaque entrée dépose le noyau, multiplié par sa valeur, à des positions espacées de s."""
    out = np.zeros((len(y) - 1) * s + len(w))
    for j, v in enumerate(y):
        out[j * s:j * s + len(w)] += v * w
    return out


for _ in range(100):
    k, s = int(rng.integers(1, 6)), int(rng.integers(1, 4))
    m = int(rng.integers(k, k + 12))
    m = (m - k) // s * s + k                      # tailles pour lesquelles la convolution n'ignore aucune entrée
    w, x = rng.normal(size=k), rng.normal(size=m)
    y = rng.normal(size=taille(m, k, 0, s))
    t = conv1d_transposee(y, w, s)
    assert len(t) == taille_transposee(len(y), k, 0, s) == m
    assert np.isclose(np.dot(conv1d(x, w, s), y), np.dot(x, t))      # adjointe : <conv(x), y> = <x, convT(y)>
dire("Convolution transposée : forme (n - 1) s + k et identité d'adjonction contrôlées sur 100 cas.")
V["LT_double"] = taille_transposee(24, 2, 0, 2)
V["LT_k3"] = taille_transposee(24, 3, 1, 2, 1, 1)
V["LT_k4"] = taille_transposee(24, 4, 1, 2)
V["LT_k3_sans"] = taille_transposee(24, 3, 0, 2)
assert V["LT_double"] == V["LT_k3"] == V["LT_k4"] == 48
# Les tailles 47 et 48 sont toutes deux envoyées sur 24 par la convolution k = 3, p = 1, s = 2 : d'où output_padding.
assert taille(47, 3, 1, 2) == taille(48, 3, 1, 2) == 24
V["LT_m1"], V["LT_m2"] = 47, 48

# ML5-01 : deux branches sur une image 128 x 128 x 3, concaténation, max-pooling.
qa = taille(128, 5, 2, 2)
qb = taille(taille(128, 3, 1, 2), 1)
assert qa == qb
V["Q01_brancheA"], V["Q01_brancheB"] = qa, qb
V["Q01_canaux"] = 12 + 20
V["Q01_sortie"] = taille(qa, 2, 0, 2)
V["Q01_d_produit"] = 12 * 20
V["Q01_d_entree"] = 12 + 20 + 3
x = rng.normal(size=(16, 16, 3))                                     # même enchaînement, tailles divisées par 8
a = conv2d(x, rng.normal(size=(5, 5, 3, 12)), p=2, s=2)
b = conv2d(maxpool2d(x, 3, 2, 1), rng.normal(size=(1, 1, 3, 20)))
assert maxpool2d(np.concatenate([a, b], axis=2), 2, 2).shape == (V["Q01_sortie"] // 8, V["Q01_sortie"] // 8, V["Q01_canaux"])

# ML5-02 : quelles sorties peut-on concaténer ou additionner (entrée 64 x 64 x 16).
V["Q02_a1"], V["Q02_a2"] = taille(64, 3, 1), taille(64, 5, 2)
V["Q02_a_canaux"] = 32 + 8
V["Q02_c1"], V["Q02_c2"] = taille(64, 3, 1, 2), taille(64, 2, 0, 2)
V["Q02_c_canaux"] = 32 + 16
V["Q02_d1"], V["Q02_d2"] = taille(64, 3), taille(64, 5)
assert V["Q02_a1"] == V["Q02_a2"] == 64 and V["Q02_c1"] == V["Q02_c2"] == 32 and V["Q02_d1"] != V["Q02_d2"]

# =====================================================================================
# B. Mettre en place un entraînement
# =====================================================================================
n_tot = 1500
V["LE_n"], V["LE_app"], V["LE_val"], V["LE_test"] = n_tot, int(n_tot * 0.70), int(n_tot * 0.15), int(n_tot * 0.15)
assert V["LE_app"] + V["LE_val"] + V["LE_test"] == n_tot
V["LE_lot"] = 32
V["LE_pas"] = math.ceil(V["LE_app"] / 32)
V["LE_quotient"], V["LE_reste"] = divmod(V["LE_app"], 32)
V["LE_epoques"] = 40
V["LE_maj"] = V["LE_epoques"] * V["LE_pas"]
V["LE_pas_faux"] = math.ceil(n_tot / 32)                 # erreur : taille totale au lieu de la taille d'apprentissage
V["LE_exemples_faux"] = V["LE_pas_faux"] * 32
V["LE_pas_val"] = math.ceil(V["LE_val"] / 32)
V["LE_reste_val"] = V["LE_val"] % 32
V["LE_se"] = math.sqrt(0.9 * 0.1 / V["LE_test"])
V["LE_se_x4"] = math.sqrt(0.9 * 0.1 / (4 * V["LE_test"]))
# Contrôle par un vrai découpage en lots.
lots = [list(range(V["LE_app"]))[i:i + 32] for i in range(0, V["LE_app"], 32)]
assert len(lots) == V["LE_pas"] and len(lots[-1]) == V["LE_reste"] and len([l for l in lots if len(l) == 32]) == V["LE_quotient"]


def sig(z):
    return 1.0 / (1.0 + math.exp(-z))


# Pertes : gradient par rapport au logit de l'entropie croisée et de l'erreur quadratique derrière une sigmoïde.
zs, ys, ps = sp.symbols("z y p")
pi_s = 1 / (1 + sp.exp(-zs))
ce = -(ys * sp.log(pi_s) + (1 - ys) * sp.log(1 - pi_s))
assert sp.simplify(sp.diff(ce, zs) - (pi_s - ys)) == 0
assert sp.simplify(sp.diff((pi_s - ys) ** 2 / 2, zs) - (pi_s - ys) * pi_s * (1 - pi_s)) == 0
V["LP_z"] = -4
V["LP_pi"] = sig(-4)
V["LP_ce"] = V["LP_pi"] - 1
V["LP_mse"] = (V["LP_pi"] - 1) * V["LP_pi"] * (1 - V["LP_pi"])
V["LP_facteur"] = V["LP_pi"] * (1 - V["LP_pi"])
# Softmax : exemple, gradient p - y par différences finies, et cas de deux classes.
logits = np.array([2.0, 1.0, -1.0])
proba = np.exp(logits) / np.exp(logits).sum()
V["LS_p"] = [float(v) for v in proba]
V["LS_perte"] = float(-math.log(proba[1]))
for _ in range(50):
    zv = rng.normal(size=4)
    c = int(rng.integers(0, 4))
    f = lambda u: -math.log(np.exp(u[c]) / np.exp(u).sum())
    num = np.array([(f(zv + 1e-6 * np.eye(4)[j]) - f(zv - 1e-6 * np.eye(4)[j])) / 2e-6 for j in range(4)])
    assert np.allclose(num, np.exp(zv) / np.exp(zv).sum() - np.eye(4)[c], atol=1e-6)
    z0, z1 = rng.normal(size=2)
    assert math.isclose(math.exp(z1) / (math.exp(z0) + math.exp(z1)), sig(z1 - z0))
dire("Pertes : gradient p - y du softmax (différences finies) et softmax à deux classes = sigmoïde de la différence.")

# Dice et IoU.
def iou(vp, fp, fn):
    return Fraction(vp, vp + fp + fn)


def dice(vp, fp, fn):
    return Fraction(2 * vp, 2 * vp + fp + fn)


for _ in range(1000):
    a = rng.random(60) < rng.random()
    b = rng.random(60) < rng.random()
    vp, fp, fn = int(np.sum(a & b)), int(np.sum(a & ~b)), int(np.sum(~a & b))
    if vp + fp + fn == 0:
        continue
    j, dc = iou(vp, fp, fn), dice(vp, fp, fn)
    assert dc == 2 * j / (1 + j) and j == dc / (2 - dc) and dc >= j and (dc == j) == (j in (0, 1))
    assert dc == Fraction(2 * int(np.sum(a & b)), int(a.sum() + b.sum()))
dire("(S1) Dice = 2 IoU / (1 + IoU), Dice >= IoU avec égalité en 0 et 1 seulement : contrôlé sur des masques tirés au hasard.")
V["LD_vp"], V["LD_fp"], V["LD_fn"], V["LD_pixels"] = 30, 10, 20, 400
V["LD_vn"] = 400 - 30 - 10 - 20
fraction("LD_iou", iou(30, 10, 20))
fraction("LD_dice", dice(30, 10, 20))
V["LD_exactitude"] = (30 + V["LD_vn"]) / 400
V["LD_fond"] = (400 - 30 - 20) / 400                    # exactitude du masque « tout en fond »
V["LD_objet"] = (30 + 20) / 400
# Moyennes sur deux images : le classement par IoU moyen et par Dice moyen peut s'inverser.
jx, jy = [Fraction(1, 2), Fraction(1, 2)], [Fraction(1, 5), Fraction(41, 50)]
versd = lambda j: 2 * j / (1 + j)
fraction("LD_x_iou", sum(jx) / 2)
fraction("LD_y_iou", sum(jy) / 2)
fraction("LD_x_dice", sum(versd(j) for j in jx) / 2)
fraction("LD_y_dice", sum(versd(j) for j in jy) / 2)
fraction("LD_y_dice1", versd(jy[0]))
fraction("LD_y_dice2", versd(jy[1]))
assert V["LD_y_iou"] > V["LD_x_iou"] and V["LD_y_dice"] < V["LD_x_dice"]

# ML5-03 : nombre de mises à jour.
V["Q03_app"], V["Q03_val"], V["Q03_test"] = int(2400 * 0.75), int(2400 * 0.15), int(2400 * 0.10)
assert V["Q03_app"] + V["Q03_val"] + V["Q03_test"] == 2400
V["Q03_pas"] = math.ceil(V["Q03_app"] / 48)
V["Q03_reste"] = V["Q03_app"] % 48
V["Q03_quotient"] = V["Q03_app"] // 48
V["Q03_exact"] = 25 * V["Q03_pas"]
V["Q03_d_plancher"] = 25 * (V["Q03_app"] // 48)
V["Q03_d_total"] = 25 * math.ceil(2400 / 48)
V["Q03_d_epoque"] = V["Q03_pas"]
assert len({V["Q03_exact"], V["Q03_d_plancher"], V["Q03_d_total"], V["Q03_d_epoque"]}) == 4
# ML5-04 : segmentation, 4 % de pixels d'objet.
V["Q04_fond"] = 1 - 0.04
# ML5-05 : Dice et IoU à partir des tailles.
pred, vrai, inter = 50, 70, 40
fraction("Q05_iou", Fraction(inter, pred + vrai - inter))
fraction("Q05_dice", Fraction(2 * inter, pred + vrai))
assert V["Q05_dice"] == float(dice(inter, pred - inter, vrai - inter)) and V["Q05_iou"] == float(iou(inter, pred - inter, vrai - inter))
fraction("Q05_d_sans2", Fraction(inter, pred + vrai))
fraction("Q05_d_union", Fraction(inter, pred + vrai))
V["Q05_union"] = pred + vrai - inter
fraction("Q05_precision", Fraction(inter, pred))
fraction("Q05_rappel", Fraction(inter, vrai))

# =====================================================================================
# Déséquilibre de classes : stratification et pondération
# =====================================================================================
def hypergeom_pmf(k, N, K, n):
    return Fraction(math.comb(K, k) * math.comb(N - K, n - k), math.comb(N, n))


def hypergeom_var(N, K, n):
    q = Fraction(K, N)
    return n * q * (1 - q) * Fraction(N - n, N - 1)


N, K, n = 1000, 30, 200
V["LSt_N"], V["LSt_K"], V["LSt_n"] = N, K, n
V["LSt_moy"] = float(Fraction(n * K, N))
V["LSt_var"] = float(hypergeom_var(N, K, n))
V["LSt_sd"] = math.sqrt(V["LSt_var"])
p2 = sum(hypergeom_pmf(k, N, K, n) for k in range(3))
V["LSt_p2"] = float(p2)
assert sum(hypergeom_pmf(k, N, K, n) for k in range(K + 1)) == 1
assert sum(k * hypergeom_pmf(k, N, K, n) for k in range(K + 1)) == Fraction(n * K, N)
assert sum(k * k * hypergeom_pmf(k, N, K, n) for k in range(K + 1)) - Fraction(n * K, N) ** 2 == hypergeom_var(N, K, n)
tirages = rng.hypergeometric(K, N - K, n, size=400000)
mc_moyenne("LSt_p2", V["LSt_p2"], tirages <= 2)
mc_moyenne("LSt_var", V["LSt_var"], (tirages - V["LSt_moy"]) ** 2)
# Perte pondérée : minimiseur de -w q ln r - (1 - q) ln(1 - r).
ws, qs, rs = sp.symbols("w q r", positive=True)
perte_p = -ws * qs * sp.log(rs) - (1 - qs) * sp.log(1 - rs)
r_opt = ws * qs / (ws * qs + 1 - qs)
assert sp.simplify(sp.diff(perte_p, rs).subs(rs, r_opt)) == 0
assert sp.simplify(r_opt / (1 - r_opt) - ws * qs / (1 - qs)) == 0
fraction("LW_r9", Fraction(9, 10) / (Fraction(9, 10) + Fraction(9, 10)))
fraction("LW_r4", Fraction(4, 10) / (Fraction(4, 10) + Fraction(9, 10)))
fraction("LW_seuil9", Fraction(1, 1 + 9))
V["LW_bal_pos"] = 1000 / (2 * 100)
V["LW_bal_neg"] = 1000 / (2 * 900)
V["LW_bal_rapport"] = V["LW_bal_pos"] / V["LW_bal_neg"]
# ML5-06 : 500 observations, 20 positives, 5 plis de 100.
V["Q06_moy"] = float(Fraction(100 * 20, 500))
V["Q06_var"] = float(hypergeom_var(500, 20, 100))
V["Q06_sd"] = math.sqrt(V["Q06_var"])
V["Q06_p1"] = float(sum(hypergeom_pmf(k, 500, 20, 100) for k in range(2)))
tirages = rng.hypergeometric(20, 480, 100, size=400000)
mc_moyenne("Q06_var", V["Q06_var"], (tirages - 4.0) ** 2)
mc_moyenne("Q06_p1", V["Q06_p1"], tirages <= 1)
fraction("Q06_cote", Fraction(3 * 1, 4) / (Fraction(3 * 1, 4) + Fraction(3, 4)))   # w = 3, q = 1/4 : r = 1/2

# =====================================================================================
# D. Apprentissage par renforcement
# =====================================================================================
def resoudre(P, r, gamma):
    """Résout exactement v = r + gamma P v (fractions), par élimination de Gauss."""
    m = len(r)
    A = [[(1 if i == j else 0) - gamma * P[i][j] for j in range(m)] + [r[i]] for i in range(m)]
    for i in range(m):
        piv = next(k for k in range(i, m) if A[k][i] != 0)
        A[i], A[piv] = A[piv], A[i]
        A[i] = [x / A[i][i] for x in A[i]]
        for k in range(m):
            if k != i:
                A[k] = [x - A[k][i] * y for x, y in zip(A[k], A[i])]
    return [A[i][m] for i in range(m)]


def etudier(mdp, gamma):
    """mdp[s][a] = (récompense, {état suivant: probabilité}). Renvoie v*, q*, la politique optimale et toutes les politiques."""
    etats = sorted(mdp)
    toutes = {}
    for choix in itertools.product(*[sorted(mdp[s]) for s in etats]):
        P = [[Fraction(mdp[s][a][1].get(t, 0)) for t in etats] for s, a in zip(etats, choix)]
        r = [Fraction(mdp[s][a][0]) for s, a in zip(etats, choix)]
        toutes[choix] = resoudre(P, r, gamma)
    meilleure = max(toutes, key=lambda c: toutes[c])
    v = toutes[meilleure]
    assert all(all(v[i] >= w[i] for i in range(len(etats))) for w in toutes.values())      # optimale en chaque état
    q = {(s, a): Fraction(mdp[s][a][0]) + gamma * sum(Fraction(pr) * v[etats.index(t)] for t, pr in mdp[s][a][1].items())
         for s in etats for a in mdp[s]}
    for i, s in enumerate(etats):                                                          # équation de Bellman d'optimalité
        assert v[i] == max(q[(s, a)] for a in mdp[s])
    return v, q, meilleure, toutes


def iteration_valeur(mdp, gamma, nb):
    etats = sorted(mdp)
    v = [Fraction(0)] * len(etats)
    suite = [v]
    for _ in range(nb):
        v = [max(Fraction(mdp[s][a][0]) + gamma * sum(Fraction(pr) * v[etats.index(t)] for t, pr in mdp[s][a][1].items())
                 for a in mdp[s]) for s in etats]
        suite.append(v)
    return suite


demi = Fraction(1, 2)
MDP_LECON = {
    1: {"prudent": (1, {1: 1}), "risque": (0, {1: demi, 2: demi})},
    2: {"attendre": (2, {2: 1}), "encaisser": (4, {1: 1})},
}
g9 = Fraction(9, 10)
v, q, pol, toutes = etudier(MDP_LECON, g9)
assert pol == ("risque", "attendre")
fraction("RL_v1", v[0])
fraction("RL_v2", v[1])
fraction("RL_q1_prudent", q[(1, "prudent")])
fraction("RL_q1_risque", q[(1, "risque")])
fraction("RL_q2_attendre", q[(2, "attendre")])
fraction("RL_q2_encaisser", q[(2, "encaisser")])
fraction("RL_pp_v1", toutes[("prudent", "attendre")][0])
fraction("RL_pe_v2", toutes[("prudent", "encaisser")][1])
fraction("RL_re_v1", toutes[("risque", "encaisser")][0])
fraction("RL_re_v2", toutes[("risque", "encaisser")][1])
suite = iteration_valeur(MDP_LECON, g9, 200)
for k in (1, 2, 3):
    V[f"RL_it{k}"] = [float(x) for x in suite[k]]
ecart0 = max(abs(a - b) for a, b in zip(suite[0], v))
for k in range(201):                                    # contraction : écart au plus gamma^k fois l'écart initial
    assert max(abs(a - b) for a, b in zip(suite[k], v)) <= g9**k * ecart0
V["RL_ecart0"] = float(ecart0)
V["RL_it_precision"] = next(k for k in range(201) if max(abs(a - b) for a, b in zip(suite[k], v)) < Fraction(1, 100))
V["RL_borne_precision"] = math.ceil(math.log(0.01 / float(ecart0)) / math.log(0.9))
assert V["RL_it_precision"] <= V["RL_borne_precision"]
v5, q5, pol5, _ = etudier(MDP_LECON, demi)
assert pol5 == ("prudent", "encaisser")
V["RL_g5_v1"], V["RL_g5_v2"] = float(v5[0]), float(v5[1])
fraction("RL_g5_q1_risque", q5[(1, "risque")])
V["RL_g5_q2_attendre"] = float(q5[(2, "attendre")])
# Retours.
V["LG_ex"] = 1 + 0.9 * 0 + 0.9**2 * 4
V["LG_const"] = float(2 / (1 - g9))
V["LG_borne"] = float(4 / (1 - g9))
# Simulation de la politique optimale depuis l'état 1 (retour tronqué à 400 pas : reste au plus 0,9^400 x 40).
nb_ep, horizon = 40000, 400
etat = np.ones(nb_ep, dtype=int)
retour = np.zeros(nb_ep)
for t in range(horizon):
    en1 = etat == 1
    retour += 0.9**t * np.where(en1, 0.0, 2.0)          # risqué en 1 : récompense 0 ; attendre en 2 : récompense 2
    passe = en1 & (rng.random(nb_ep) < 0.5)
    etat = np.where(passe, 2, etat)
assert 0.9**horizon * 40 < 1e-12
mc_moyenne("RL_v1", V["RL_v1"], retour)

# ML5-08 : autre processus à deux états, gamma = 0,8.
MDP_Q = {
    1: {"garder": (2, {1: 1}), "changer": (0, {2: 1})},
    2: {"continuer": (4, {1: demi, 2: demi})},
}
g8 = Fraction(4, 5)
v, q, pol, toutes = etudier(MDP_Q, g8)
assert pol == ("changer", "continuer")
fraction("Q08_exact", v[0])
fraction("Q08_v2", v[1])
fraction("Q08_d_garder_toujours", toutes[("garder", "continuer")][0])
fraction("Q08_d_q_garder", q[(1, "garder")])
suite = iteration_valeur(MDP_Q, g8, 300)
assert abs(float(suite[-1][0]) - V["Q08_exact"]) < 1e-12
assert len({V["Q08_exact"], V["Q08_v2"], V["Q08_d_garder_toujours"], V["Q08_d_q_garder"]}) == 4
nb_ep, horizon = 40000, 200
etat = np.ones(nb_ep, dtype=int)
retour = np.zeros(nb_ep)
for t in range(horizon):
    en1 = etat == 1
    retour += 0.8**t * np.where(en1, 0.0, 4.0)
    retourne = (~en1) & (rng.random(nb_ep) < 0.5)
    etat = np.where(en1, 2, np.where(retourne, 1, 2))
mc_moyenne("Q08_exact", V["Q08_exact"], retour)


# Gradient de politique sur un bandit à deux actions : pi(1) = sig(theta), récompenses déterministes r1 et r0.
def reinforce(pi, r1, r0, b):
    """Espérance et variance exactes de g = (R - b) d/dtheta ln pi(A)."""
    pi, r1, r0, b = Fraction(pi), Fraction(r1), Fraction(r0), Fraction(b)
    g1, g0 = (r1 - b) * (1 - pi), -(r0 - b) * pi
    esp = pi * g1 + (1 - pi) * g0
    return esp, pi * g1**2 + (1 - pi) * g0**2 - esp**2


th, bs, r1s, r0s = sp.symbols("theta b r1 r0")
pis = 1 / (1 + sp.exp(-th))
esp_s = pis * (r1s - bs) * sp.diff(sp.log(pis), th) + (1 - pis) * (r0s - bs) * sp.diff(sp.log(1 - pis), th)
assert sp.simplify(esp_s - sp.diff(pis * r1s + (1 - pis) * r0s, th)) == 0          # sans biais, quel que soit b
e0, v0 = reinforce(demi, 11, 10, 0)
e1, v1 = reinforce(demi, 11, 10, Fraction(21, 2))
e2, v2 = reinforce(demi, 11, 10, 10)
assert e0 == e1 == e2
fraction("PG_esp", e0)
fraction("PG_var0", v0)
fraction("PG_var_b", v1)
fraction("PG_var_10", v2)
V["PG_g1"], V["PG_g0"] = float(11 * demi), float(-10 * demi)
act = rng.random(400000) < 0.5
g = np.where(act, 11 * 0.5, -10 * 0.5)
mc_moyenne("PG_esp", V["PG_esp"], g)
# Les deux valeurs de g, 5,5 et -5, sont à la même distance de l'espérance 0,25 : (g - 0,25)^2 est constant,
# et la variance s'en déduit sans erreur d'échantillonnage.
assert np.allclose((g - V["PG_esp"]) ** 2, V["PG_var0"])
# ML5-10 : pi = 1/5, r1 = 6, r0 = 1.
cinq = Fraction(1, 5)
e0, v0 = reinforce(cinq, 6, 1, 0)
eb, vb = reinforce(cinq, 6, 1, 2)
assert e0 == eb == cinq * (1 - cinq) * 5
fraction("Q10_esp", e0)
fraction("Q10_var0", v0)
fraction("Q10_var_b", vb)
V["Q10_moyR"] = float(cinq * 6 + (1 - cinq) * 1)
V["Q10_g1"], V["Q10_g0"] = float(6 * (1 - cinq)), float(-1 * cinq)
V["Q10_gb1"], V["Q10_gb0"] = float((6 - 2) * (1 - cinq)), float(-(1 - 2) * cinq)
V["Q10_d_m2_0"] = float(v0 + e0**2)
V["Q10_d_m2_b"] = float(vb + eb**2)
fraction("Q10_b_opt", 6 * (1 - cinq) + 1 * cinq)
assert reinforce(cinq, 6, 1, 5)[1] == 0
act = rng.random(400000) < 0.2
for nom, b, esp, var in [("Q10_var0", 0.0, e0, v0), ("Q10_var_b", 2.0, eb, vb)]:
    g = np.where(act, (6 - b) * 0.8, -(1 - b) * 0.2)
    mc_moyenne(nom.replace("var", "esp"), float(esp), g)
    mc_moyenne(nom, float(var), (g - float(esp)) ** 2)

# Bandit : blocage de la stratégie gloutonne (bras 1 de moyenne 0,8 ; bras 2 de moyenne 0,5).
fraction("BD_blocage", (1 - Fraction(4, 5)) * demi)
V["BD_ecart"] = float(Fraction(4, 5) - demi)
V["BD_T"] = 1000
V["BD_regret_min"] = float((1 - Fraction(4, 5)) * demi * Fraction(3, 10) * (1000 - 2))
nb_run, T = 20000, 300
somme = np.zeros((nb_run, 2))
compte = np.zeros((nb_run, 2))
x1, x2 = rng.random(nb_run) < 0.8, rng.random(nb_run) < 0.5
somme[:, 0], somme[:, 1] = x1, x2
compte[:] = 1
lignes_idx = np.arange(nb_run)
for t in range(2, T):
    bras = np.argmax(somme / compte, axis=1)             # égalité : bras 1 (indice 0)
    gain = rng.random(nb_run) < np.where(bras == 0, 0.8, 0.5)
    somme[lignes_idx, bras] += gain
    compte[lignes_idx, bras] += 1
jamais = compte[:, 0] == 1                               # le bras 1 n'a plus jamais été tiré
assert np.all(jamais[(~x1) & x2])                        # l'événement de la preuve implique le blocage
mc_moyenne("BD_blocage", V["BD_blocage"], (~x1) & x2)
V["BD_blocage_obs"] = float(jamais.mean())
assert V["BD_blocage_obs"] >= V["BD_blocage_mc"]
ts, ns = sp.symbols("t n", positive=True)
assert sp.simplify(sp.exp(-2 * ns * (sp.sqrt(2 * sp.log(ts) / ns)) ** 2) - ts**-4) == 0
V["BD_bonus"] = math.sqrt(2 * math.log(100) / 10)

# =====================================================================================
# E. Réflexes : elastic net, ACP
# =====================================================================================
def seuil_doux(z, t):
    return math.copysign(max(abs(z) - t, 0.0), z) + 0.0        # + 0.0 : pas de zéro négatif à l'affichage


def elastic_orthonormal(z, lam, alpha):
    """Colonnes orthonormées : minimiseur de (1/2)(z - b)^2 + lam (alpha |b| + (1 - alpha) b^2 / 2), coordonnée par coordonnée."""
    return [seuil_doux(x, lam * alpha) / (1 + lam * (1 - alpha)) + 0.0 for x in z]


def elastic_numerique(z, lam, alpha):
    res = []
    for x in z:
        f = lambda b: 0.5 * (x - b) ** 2 + lam * (alpha * abs(b) + (1 - alpha) * b * b / 2)
        cands = [minimize_scalar(f, bounds=(-10, 0), method="bounded", options={"xatol": 1e-12}).x,
                 minimize_scalar(f, bounds=(0, 10), method="bounded", options={"xatol": 1e-12}).x, 0.0]
        res.append(min(cands, key=f))
    return res


for _ in range(200):
    z = list(rng.normal(scale=2, size=3))
    lam, alpha = float(rng.uniform(0.1, 3)), float(rng.uniform(0, 1))
    assert np.allclose(elastic_orthonormal(z, lam, alpha), elastic_numerique(z, lam, alpha), atol=1e-5)
dire("Elastic net, colonnes orthonormées : seuillage doux puis division, retrouvé par minimisation numérique (200 cas).")
zl = [3.0, -0.5, 1.2]
V["EN_z"] = zl
V["EN_seuil"], V["EN_div"] = 1.0 * 0.5, 1 + 1.0 * 0.5
V["EN_en"] = elastic_orthonormal(zl, 1.0, 0.5)
V["EN_lasso"] = elastic_orthonormal(zl, 1.0, 1.0)
V["EN_ridge"] = elastic_orthonormal(zl, 1.0, 0.0)
# ML5-13.
zq = [4.0, -1.5, 0.6]
V["Q13_seuil"], V["Q13_div"] = 2.0 * 0.5, 1 + 2.0 * 0.5
V["Q13_exact"] = elastic_orthonormal(zq, 2.0, 0.5)
assert np.allclose(V["Q13_exact"], elastic_numerique(zq, 2.0, 0.5), atol=1e-6)
V["Q13_d_sans_div"] = [seuil_doux(x, 1.0) for x in zq]
V["Q13_d_sans_seuil"] = [x / 2.0 for x in zq]
V["Q13_d_seuil_lambda"] = [seuil_doux(x, 2.0) / 2.0 + 0.0 for x in zq]
assert len({tuple(V[c]) for c in ["Q13_exact", "Q13_d_sans_div", "Q13_d_sans_seuil", "Q13_d_seuil_lambda"]}) == 4
# ACP : parts de variance.
vp = [Fraction(11, 5), Fraction(1), Fraction(1, 2), Fraction(3, 10)]
assert sum(vp) == 4
V["Q12_part1"] = float(vp[0] / 4)
V["Q12_part2"] = float((vp[0] + vp[1]) / 4)
V["Q12_part1_pct"] = float(100 * vp[0] / 4)
V["Q12_part2_pct"] = float(100 * (vp[0] + vp[1]) / 4)
V["ACP_min10"] = 1 / 10
# Une matrice de corrélation de spectre donné existe : conjugaison par une matrice orthogonale à coefficients +-1/2.
Hd = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]) / 2.0
C = Hd.dot(np.diag([float(x) for x in vp])).dot(Hd.T)
assert np.allclose(np.diag(C), 1) and np.allclose(sorted(np.linalg.eigvalsh(C)), sorted(float(x) for x in vp))
dire("ACP : une matrice de corrélation 4 x 4 de valeurs propres 2,2 ; 1 ; 0,5 ; 0,3 existe (construite explicitement).")

# =====================================================================================
# Valeurs diverses de la leçon et des points de contrôle
# =====================================================================================
# Point de contrôle « forme » : entrée 60 x 60 x 8 ; conv 5 x 5, 12 filtres, remplissage 2, pas 2 ; max-pooling 2 x 2 de pas 2.
V["C_forme_n"] = taille(60, 5, 2, 2)
assert V["C_forme_n"] == taille(60, 2, 0, 2)
V["C_forme_c"] = 12 + 8
V["C_forme_sans"] = taille(60, 5, 0, 2)
V["C_forme_prod"] = 12 * 8
# Point de contrôle « pas » : 5000 images, 80 % en apprentissage, lots de 64.
V["C_pas_app"] = int(5000 * 0.8)
V["C_pas"] = math.ceil(V["C_pas_app"] / 64)
V["C_pas_plancher"] = V["C_pas_app"] // 64
V["C_pas_reste"] = V["C_pas_app"] % 64
V["C_pas_total"] = math.ceil(5000 / 64)
assert len({V["C_pas"], V["C_pas_plancher"], V["C_pas_total"], V["C_pas_app"]}) == 4
# Écart-type d'initialisation de He pour 512 entrées (séance de ML 4, partie 2.2).
V["LA_he"] = math.sqrt(2 / 512)
# Arbre et ou exclusif : gains de Gini à la racine, en population.
gini2 = lambda p: 2 * p * (1 - p)
fraction("T_gini_parent", gini2(demi))
fraction("T_gini_x3", gini2(Fraction(3, 5)))
fraction("T_gain_x3", gini2(demi) - gini2(Fraction(3, 5)))
fraction("T_gain_x1", gini2(demi) - gini2(demi))
xa, xb = rng.integers(0, 2, 400000), rng.integers(0, 2, 400000)
yx = xa ^ xb
xc = np.where(rng.random(400000) < 0.6, yx, 1 - yx)
for nom, col, exact in [("T_gain_x3", xc, V["T_gain_x3"]), ("T_gain_x1", xa, 0.0)]:
    g0, g1 = yx[col == 0], yx[col == 1]
    obs = gini2(yx.mean()) - len(g0) / len(yx) * gini2(g0.mean()) - len(g1) / len(yx) * gini2(g1.mean())
    assert abs(obs - exact) < 2e-3, (nom, obs)
    V[nom + "_obs"] = float(obs)
# Stratégie epsilon-gloutonne à deux bras (0,8 et 0,5), epsilon = 0,1 : regret espéré par tirage au moins (eps / K) x écart.
V["BD_eps_pas"] = 0.1 / 2 * 0.3
V["BD_eps_T"] = 0.1 / 2 * 0.3 * (5000 - 2)

# =====================================================================================
# Contrôles de bibliothèque, facultatifs : n'écrivent rien dans les sorties conservées.
# =====================================================================================
try:
    import torch

    xt = torch.zeros(1, 3, 37, 41)
    for k, p, s, d in [(3, 0, 1, 1), (3, 1, 1, 1), (5, 2, 2, 1), (3, 1, 2, 2), (7, 3, 2, 1), (4, 1, 3, 2)]:
        assert tuple(torch.nn.Conv2d(3, 8, k, stride=s, padding=p, dilation=d)(xt).shape) == (1, 8, taille(37, k, p, s, d), taille(41, k, p, s, d))
    assert sum(t.numel() for t in torch.nn.Conv2d(3, 16, 3).parameters()) == V["LC_paramsA"]
    assert tuple(torch.nn.MaxPool2d(2)(xt).shape) == (1, 3, 18, 20)
    for k, p, s, op in [(2, 0, 2, 0), (3, 1, 2, 1), (4, 1, 2, 0), (3, 0, 2, 0)]:
        assert torch.nn.ConvTranspose2d(3, 8, k, stride=s, padding=p, output_padding=op)(xt).shape[2] == taille_transposee(37, k, p, s, 1, op)
    c0 = torch.nn.Conv2d(3, 8, 3)
    assert (c0.stride, c0.padding, c0.dilation) == ((1, 1), (0, 0), (1, 1))
    assert tuple(torch.nn.Conv2d(3, 8, 4, padding="same")(xt).shape) == (1, 8, 37, 41)
    try:
        torch.nn.Conv2d(3, 8, 3, stride=2, padding="same")
        raise AssertionError("padding='same' accepté avec un pas de 2")
    except ValueError:
        pass
    assert torch.nn.MaxPool2d(2).stride == 2 and tuple(torch.nn.MaxPool2d(2, ceil_mode=True)(xt).shape) == (1, 3, 19, 21)
    assert tuple(torch.cat([torch.zeros(1, 4, 8, 8), torch.zeros(1, 6, 8, 8)], dim=1).shape) == (1, 10, 8, 8)
    assert tuple((torch.zeros(1, 4, 8, 8) + torch.zeros(1, 1, 8, 8)).shape) == (1, 4, 8, 8)       # diffusion
    import inspect

    defauts = {o.__name__: inspect.signature(o.__init__).parameters["lr"].default
               for o in [torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW, torch.optim.RMSprop, torch.optim.Adagrad]}
    print("PyTorch, pas d'apprentissage par défaut :", defauts)
    if torch.__version__.startswith("2.4.1"):
        assert defauts == {"SGD": 0.001, "Adam": 0.001, "AdamW": 0.001, "RMSprop": 0.01, "Adagrad": 0.01}
    from torch.utils.data import DataLoader

    assert len(DataLoader(list(range(1400)), batch_size=32)) == 44 and len(DataLoader(list(range(1400)), batch_size=32, drop_last=True)) == 43
    zt = torch.tensor([[2.0, 0.5, -1.0]])
    assert abs(float(torch.nn.CrossEntropyLoss()(zt, torch.tensor([0]))) + float(torch.log_softmax(zt, 1)[0, 0])) < 1e-6
    print(f"PyTorch {torch.__version__} présent : formes de Conv2d, MaxPool2d, ConvTranspose2d, concaténation, nombre de paramètres, "
          "DataLoader et CrossEntropyLoss vérifiés.")
except ImportError:
    print("PyTorch absent : contrôles de bibliothèque ignorés.")

try:
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import inspect

    import keras

    xk = keras.ops.zeros((1, 37, 41, 3))
    ck = keras.layers.Conv2D(8, 3)
    assert (ck.strides, ck.padding, ck.dilation_rate) == ((1, 1), "valid", (1, 1))
    for k, s in [(3, 1), (3, 2), (4, 3)]:
        assert tuple(keras.layers.Conv2D(8, k, strides=s, padding="same")(xk).shape) == (1, math.ceil(37 / s), math.ceil(41 / s), 8)
    assert tuple(keras.layers.Conv2D(8, 5, strides=2)(xk).shape) == (1, taille(37, 5, 0, 2), taille(41, 5, 0, 2), 8)
    mp = keras.layers.MaxPooling2D()
    assert mp.pool_size == (2, 2) and mp.strides == (2, 2) and tuple(mp(xk).shape) == (1, 18, 20, 3)
    for k, s in [(2, 2), (3, 2), (4, 2)]:
        assert tuple(keras.layers.Conv2DTranspose(8, k, strides=s, padding="valid")(xk).shape)[1] == (37 - 1) * s + k
        assert tuple(keras.layers.Conv2DTranspose(8, k, strides=s, padding="same")(xk).shape)[1] == 37 * s
    assert keras.layers.Concatenate().axis == -1
    defauts = {o.__name__: inspect.signature(o.__init__).parameters["learning_rate"].default
               for o in [keras.optimizers.SGD, keras.optimizers.Adam, keras.optimizers.AdamW, keras.optimizers.RMSprop, keras.optimizers.Adagrad]}
    print("Keras, pas d'apprentissage par défaut :", defauts)
    if keras.__version__.startswith("3.12"):
        assert defauts == {"SGD": 0.01, "Adam": 0.001, "AdamW": 0.001, "RMSprop": 0.001, "Adagrad": 0.001}
    for perte in [keras.losses.BinaryCrossentropy, keras.losses.CategoricalCrossentropy, keras.losses.SparseCategoricalCrossentropy]:
        assert inspect.signature(perte.__init__).parameters["from_logits"].default is False
    pk = np.array([[0.2, 0.7, 0.1]], dtype="float32")
    assert abs(float(keras.losses.CategoricalCrossentropy()(np.array([[0.0, 1.0, 0.0]], dtype="float32"), pk))
               - float(keras.losses.SparseCategoricalCrossentropy()(np.array([1]), pk))) < 1e-6

    class Compteur(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.pas, self.courant = [], 0

        def on_train_batch_end(self, batch, logs=None):
            self.courant += 1

        def on_epoch_end(self, epoch, logs=None):
            self.pas.append(self.courant)
            self.courant = 0

    import warnings

    Xk, yk = np.zeros((1050, 4), dtype="float32"), np.zeros((1050, 1), dtype="float32")
    modele = keras.Sequential([keras.Input((4,)), keras.layers.Dense(1)])
    modele.compile(loss="mse", optimizer="sgd")
    observes = []
    for arguments in [{"batch_size": 32}, {}, {"batch_size": 32, "steps_per_epoch": 10}, {"batch_size": 32, "steps_per_epoch": 50}]:
        compteur = Compteur()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modele.fit(Xk, yk, epochs=1, verbose=0, callbacks=[compteur], **arguments)
        observes.append(compteur.pas[0])
    assert observes == [33, 33, 10, 33], observes
    print(f"Keras {keras.__version__} présent : formes de Conv2D, MaxPooling2D, Conv2DTranspose, pertes et pas par époque de fit vérifiés.")
except ImportError:
    print("Keras absent : contrôles de bibliothèque ignorés.")

Path(__file__).with_name("seance_ml_5.out.json").write_text(json.dumps(V, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
dire(f"{len(V)} valeurs exportées dans verif/seance_ml_5.out.json.")
dire("Toutes les vérifications passent.")
Path(__file__).with_name("seance_ml_5.out.txt").write_text("\n".join(LIGNES) + "\n", encoding="utf-8", newline="\n")
