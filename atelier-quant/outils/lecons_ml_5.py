"""Leçon de la séance de ML 5 : arithmétique des réseaux convolutifs, mise en place d'un entraînement,
biais et variance en image de cible, apprentissage par renforcement, réflexes pour les affirmations à cocher.

Format : voir outils/lecons.py (élément de type "lecon", contenu en Markdown réduit, points de contrôle
[[controle:IDENT]] sur une ligne seule, option "a" correcte, une explication par option).
Les blocs python-exec sont exécutés à la construction par le noyau de l'environnement .venv-hackerrank
(numpy 1.21.6, scipy 1.8.1, scikit-learn 1.1.1, sans PyTorch ni Keras), dans le dossier coding/lecons ; leur
sortie réelle est insérée. Graines fixées.
Les autres nombres sont des jetons @CLE@ remplacés par les valeurs de verif/seance_ml_5.out.json.
"""

LECON_ML_5 = {
    "id": "L-ML-5", "type": "lecon", "section": "lecon",
    "theme": "Leçon : réseaux convolutifs, entraînement, apprentissage par renforcement, réflexes de choix multiple",
    "difficulte": 4, "temps_cible_s": 3900, "dossier": "coding/lecons",
    "contenu": r"""## Leçon : réseaux convolutifs, entraînement, renforcement, affirmations à cocher

Durée visée : 65 minutes pour tout lire, preuves comprises ; environ 30 minutes en ne lisant que les énoncés, les exemples et les pièges. La leçon a cinq parties indépendantes : tu peux les lire en plusieurs fois.

Ce qui est déjà démontré ailleurs est cité par le titre de la séance et n'est pas redémontré. Les expériences sont des blocs de code exécutés à la construction de cette page, avec numpy 1.21.6, scipy 1.8.1 et scikit-learn 1.1.1 (environnement `.venv-hackerrank`, sans PyTorch ni Keras) : la sortie affichée est la vraie sortie. Les autres nombres viennent du script `verif/seance_ml_5.py`. Les comportements de PyTorch et de Keras cités ont été exécutés le 9 octobre 2026 dans un environnement séparé (PyTorch 2.4.1, Keras 3.12.2) : pour toute autre version, ils sont à vérifier.

**Conventions de notation**, valables pour toute la leçon :

- une majuscule désigne une variable aléatoire, une minuscule une valeur ou un paramètre fixe, le gras un vecteur ou une matrice ;
- exceptions assumées, parce que ce sont les notations universelles : \(B\) (taille d'un lot), \(K\) (nombre de classes ou de bras), \(T\) (nombre de pas de temps ou de tirages), \(L\) (majorant de courbure), \(N\) et \(N_1\) (tailles d'une population et de sa classe positive) et \(M\) (nombre de couches) sont des entiers ou des réels fixes ; \(J(\boldsymbol{\theta})\) est une fonction fixe du paramètre ; VP, FP, FN, VN sont des effectifs observés (séance de ML 3) ;
- \(\lfloor u \rfloor\) est la partie entière de \(u\) (le plus grand entier inférieur ou égal à \(u\)), \(\lceil u \rceil\) son plafond (le plus petit entier supérieur ou égal à \(u\)) ;
- \(\mathrm{sig}(z) = 1/(1 + e^{-z})\) est la sigmoïde ; \(I_A\) vaut 1 si l'événement \(A\) a lieu, 0 sinon ; \(\lVert \mathbf{v} \rVert_\infty\) est la plus grande valeur absolue des coordonnées de \(\mathbf{v}\) ;
- en partie 4, \(\mathcal{S}\) et \(\mathcal{A}\) sont des ensembles finis (états, actions) ; \(S_t\), \(A_t\), \(R_t\), \(G_t\) sont aléatoires ; \(s\), \(a\), \(r\) désignent des valeurs.

## Partie 1 : arithmétique des réseaux convolutifs

### 1.1 Forme de sortie d'une convolution

**Le problème.** Avant d'entraîner un réseau convolutif, il faut que les formes des tenseurs s'emboîtent : chaque couche attend un nombre de canaux précis, et deux tenseurs ne se concatènent ou ne s'additionnent que si certaines dimensions coïncident. Une question type donne une image, des couches, et demande la forme finale.

**Les définitions.** Une image, ou une carte d'activations, est un tenseur de forme (hauteur, largeur, canaux). On raisonne sur un seul axe spatial ; l'autre se traite de la même façon, indépendamment. Sur cet axe :

- \(n\) est la taille de l'entrée, dont les positions sont numérotées de 0 à \(n - 1\) ;
- le **remplissage** (padding) \(p \ge 0\) est le nombre de zéros ajoutés de chaque côté : l'entrée complétée a \(n + 2p\) positions ;
- \(k \ge 1\) est la taille du noyau (nombre de poids le long de l'axe) ;
- la **dilatation** \(d \ge 1\) est l'écart entre deux positions lues par des poids voisins ; le noyau dilaté couvre une **étendue** \(e = d(k - 1) + 1\) positions ;
- le **pas** (stride) \(s \ge 1\) est le déplacement du noyau entre deux sorties : la sortie numéro \(j\) (à partir de 0) lit les positions \(js, js + d, \dots, js + d(k - 1)\) de l'entrée complétée.

**L'intuition.** On pose le noyau tout à gauche, puis on le décale de \(s\) tant qu'il tient dans l'entrée complétée : la taille de sortie est le nombre de poses.

**L'énoncé** (C1). Si \(n + 2p \ge e\), la taille de sortie le long de l'axe est

\(\displaystyle m = \left\lfloor \frac{n + 2p - d(k - 1) - 1}{s} \right\rfloor + 1.\)

**La preuve.**

1. La pose numéro \(j\) couvre les positions \(js\) à \(js + e - 1\) de l'entrée complétée (définition du pas et de l'étendue).
2. Elle tient dans l'entrée complétée si et seulement si sa dernière position existe : \(js + e - 1 \le n + 2p - 1\), soit \(j \le (n + 2p - e)/s\).
3. \(j\) est un entier positif ou nul : les poses possibles sont \(j = 0, 1, \dots, \lfloor (n + 2p - e)/s \rfloor\), soit \(\lfloor (n + 2p - e)/s \rfloor + 1\) poses. On remplace \(e\) par \(d(k - 1) + 1\).

**Deux conventions de remplissage.**

- « valid » : aucun remplissage, \(p = 0\). Pour \(s = d = 1\), \(m = n - k + 1\) : un noyau \(3 \times 3\) retire 2 à chaque dimension spatiale.
- « same » : le remplissage est choisi pour que \(m = \lceil n/s \rceil\). Pour \(s = 1\), il faut \(2p = d(k - 1)\) : \(p = d(k - 1)/2\) de chaque côté quand \(d(k - 1)\) est pair (pour \(d = 1\) : \(k\) impair, \(p = 1\) pour \(k = 3\), \(p = 2\) pour \(k = 5\)), sinon un zéro de plus d'un côté que de l'autre. Pour \(s > 1\), le nombre total de zéros à ajouter est \(\max\big((\lceil n/s \rceil - 1)s + e - n, 0\big)\) : (C1) donne alors \(\lceil n/s \rceil\) (contrôlé par le script sur toutes les combinaisons d'une grille).

**Canaux et paramètres** (C2). Une couche de convolution à \(f\) filtres, appliquée à une entrée de \(c\) canaux avec un noyau \(k \times k\), a une sortie de \(f\) canaux et \(f\,(k^2 c + 1)\) paramètres avec biais. Preuve : chaque filtre est un bloc de \(k \times k \times c\) poids, qui lit **tous** les canaux d'entrée à la fois et produit un canal de sortie, plus un biais ; il y a \(f\) filtres. Deux conséquences : le nombre de canaux de sortie ne dépend que du nombre de filtres, pas du nombre de canaux d'entrée ; le nombre de paramètres ne dépend pas de la taille spatiale de l'image.

**Exemple.** Entrée de taille 96. Noyau 7, remplissage 3, pas 2 : \(m = \lfloor (96 + 6 - 7)/2 \rfloor + 1 = @LC_pas2@\). Noyau 3 de dilatation 2 sans remplissage : étendue \(@LC_etendue_dil@\), \(m = @LC_dil@\). Convolution \(3 \times 3\) à 16 filtres sur une image à 3 canaux : \(16 \times (9 \times 3 + 1) = @LC_paramsA@\) paramètres. Vérification par une vraie convolution numpy sur un tenseur \(12 \times 12 \times 3\), la fonction ne connaissant pas la formule :

```python-exec
import numpy as np
from scipy.signal import correlate2d
np.set_printoptions(precision=4, suppress=True)

def conv2d(x, w, p=0, s=1, d=1):
    # x : (hauteur, largeur, canaux) ; w : (k, k, canaux, filtres) ; p zéros de chaque côté, pas s, dilatation d
    k, F = w.shape[0], w.shape[3]
    xp = np.pad(x, ((p, p), (p, p), (0, 0)))
    e = d * (k - 1) + 1                                    # étendue du noyau dilaté
    lignes, colonnes = range(0, xp.shape[0] - e + 1, s), range(0, xp.shape[1] - e + 1, s)
    return np.array([[[np.sum(xp[i:i + e:d, j:j + e:d, :] * w[:, :, :, f]) for f in range(F)] for j in colonnes] for i in lignes])

def taille(n, k, p=0, s=1, d=1):                           # formule (C1)
    return (n + 2 * p - d * (k - 1) - 1) // s + 1

rng = np.random.default_rng(0)
x = rng.normal(size=(12, 12, 3))                           # petite « image » 12 x 12 à 3 canaux
for k, p, s, d in [(3, 0, 1, 1), (3, 1, 1, 1), (5, 2, 1, 1), (3, 1, 2, 1), (3, 0, 1, 2), (7, 3, 2, 1)]:
    y = conv2d(x, rng.normal(size=(k, k, 3, 4)), p, s, d)
    print(f"k={k} p={p} s={s} d={d} : forme obtenue {y.shape}, formule {taille(12, k, p, s, d)}")
w = rng.normal(size=(3, 3, 1, 1))
print("mêmes valeurs que scipy, valid :", np.allclose(conv2d(x[:, :, :1], w)[:, :, 0], correlate2d(x[:, :, 0], w[:, :, 0, 0], mode="valid")))
print("mêmes valeurs que scipy, same  :", np.allclose(conv2d(x[:, :, :1], w, p=1)[:, :, 0], correlate2d(x[:, :, 0], w[:, :, 0, 0], mode="same")))
```

**Correspondance avec les bibliothèques** (exécuté le 9 octobre 2026, hors de cette page) :

- PyTorch 2.4.1, `nn.Conv2d(c, f, k, stride=1, padding=0, dilation=1)` : tenseurs dans l'ordre (lot, canaux, hauteur, largeur) ; les valeurs par défaut sont celles écrites ici ; la forme de sortie suit (C1) sur les six configurations essayées ; `padding='same'` est refusé dès que le pas dépasse 1.
- Keras 3.12.2, `Conv2D(f, k, strides=1, padding='valid')` : tenseurs dans l'ordre (lot, hauteur, largeur, canaux) ; `'valid'` par défaut ; `'same'` donne \(\lceil n/s \rceil\), y compris pour un pas supérieur à 1.
- La façon dont Keras répartit un nombre impair de zéros entre les deux côtés n'a pas été exécutée : à vérifier.

**Cas limites.** \(k = 1\) : \(m = \lfloor (n + 2p - 1)/s \rfloor + 1\), soit \(n\) pour \(s = 1\), \(p = 0\) ; une convolution \(1 \times 1\) ne change que le nombre de canaux. \(n + 2p < e\) : aucune pose, la couche n'est pas applicable. \(s > 1\) : les \((n + 2p - e) \bmod s\) dernières positions ne sont lues par aucune pose.

**Pièges.** Oublier le « + 1 » ; arrondir au plus proche au lieu de prendre la partie entière ; compter \(p\) une seule fois alors qu'il est ajouté des deux côtés ; croire que le nombre de filtres modifie la taille spatiale ; croire que le nombre de canaux d'entrée se retrouve en sortie.

### 1.2 Pooling

**La définition.** Une couche de pooling fait glisser une fenêtre \(k \times k\) avec un pas \(s\) sur **chaque canal séparément** et renvoie le maximum (max-pooling) ou la moyenne de la fenêtre.

**L'énoncé** (C1 bis). La taille spatiale de sortie suit (C1) avec les mêmes \(k\), \(p\), \(s\) ; le nombre de canaux est inchangé ; la couche n'a aucun paramètre. Pour une fenêtre \(2 \times 2\) de pas 2 sans remplissage : \(m = \lfloor (n - 2)/2 \rfloor + 1 = \lfloor n/2 \rfloor\). Preuve : la preuve de (C1) ne dépend que des positions lues, pas de l'opération appliquée à la fenêtre.

**Cas limite.** \(n\) impair : la dernière ligne et la dernière colonne ne sont lues par aucune fenêtre ; une carte de taille 97 donne \(@LC_impair@\). Dans PyTorch 2.4.1 et Keras 3.12.2, le pas par défaut d'un max-pooling est la taille de sa fenêtre, et `nn.MaxPool2d` a une option `ceil_mode=True` qui arrondit au plafond (exécuté le 9 octobre 2026).

### 1.3 Concaténer ou additionner

**Le problème.** Les architectures modernes ne sont pas de simples chaînes : des branches se rejoignent. Il y a deux façons de joindre deux tenseurs, avec des conditions différentes.

**L'énoncé** (C3). Soit deux tenseurs de formes \((n_1, n_2, c)\) et \((n_1', n_2', c')\).

1. Leur **concaténation selon les canaux** existe si et seulement si \(n_1 = n_1'\) et \(n_2 = n_2'\) ; sa forme est \((n_1, n_2, c + c')\). Les nombres de canaux sont libres et s'additionnent.
2. Leur **addition** terme à terme (connexion résiduelle, séance de ML 4, partie 2.6) existe si et seulement si les trois dimensions coïncident ; la forme ne change pas.

Preuve : ce sont les définitions. Concaténer, c'est mettre bout à bout, en chaque position spatiale, les \(c\) valeurs de l'un et les \(c'\) valeurs de l'autre : il faut les mêmes positions. Additionner, c'est sommer les valeurs de même position et de même canal : il faut aussi les mêmes canaux.

**Exemple entièrement calculé.** Image \(96 \times 96 \times 3\). Deux branches **partent toutes deux de l'image** : A, convolution \(3 \times 3\) à 16 filtres ; B, convolution \(5 \times 5\) à 24 filtres ; pas 1, convention « same ». On concatène selon les canaux, puis on applique un max-pooling \(2 \times 2\) de pas 2.

1. Branche A : \(p = @LC_pA@\), taille \(96 + 2 - 3 + 1 = 96\) : forme \(96 \times 96 \times 16\). Paramètres : \(@LC_paramsA@\).
2. Branche B : \(p = @LC_pB@\), taille \(96 + 4 - 5 + 1 = 96\) : forme \(96 \times 96 \times 24\). Paramètres : \(24 \times (25 \times 3 + 1) = @LC_paramsB@\).
3. Concaténation : mêmes dimensions spatiales, donc possible ; forme \(96 \times 96 \times @LC_canaux@\).
4. Max-pooling : \(\lfloor 96/2 \rfloor = @LC_pool@\), canaux inchangés, aucun paramètre : forme finale \(@LC_pool@ \times @LC_pool@ \times @LC_canaux@\). Total : \(@LC_params@\) paramètres.

Avec la convention « valid », A donnerait une taille \(@LC_validA@\) et B une taille \(@LC_validB@\) : la concaténation serait impossible sans rogner A ou compléter B. Et l'addition de A et B est impossible dans les deux conventions, puisque \(16 \ne 24\).

```python-exec
def maxpool(x, k=2, s=2):
    return np.array([[x[i:i + k, j:j + k, :].max(axis=(0, 1)) for j in range(0, x.shape[1] - k + 1, s)] for i in range(0, x.shape[0] - k + 1, s)])

a = conv2d(x, rng.normal(size=(3, 3, 3, 16)), p=1)        # branche A : 3 x 3, 16 filtres, remplissage 1
b = conv2d(x, rng.normal(size=(5, 5, 3, 24)), p=2)        # branche B : 5 x 5, 24 filtres, remplissage 2
c = np.concatenate([a, b], axis=2)                         # concaténation selon l'axe des canaux
print("branches :", a.shape, b.shape, "; concaténation :", c.shape, "; après max-pooling 2 x 2 de pas 2 :", maxpool(c).shape)
print("max-pooling 2 x 2 de pas 2 d'une carte 5 x 5 :", maxpool(x[:5, :5, :]).shape)
a_valid, b_valid = conv2d(x, rng.normal(size=(3, 3, 3, 16))), conv2d(x, rng.normal(size=(5, 5, 3, 24)))
print("les mêmes branches sans remplissage :", a_valid.shape, b_valid.shape)
```

```python-exec-erreur
np.concatenate([a_valid, b_valid], axis=2)
```

```python-exec-erreur
a + b
```

**Trois architectures à reconnaître** (faits tirés des articles cités, non exécutés ici).

- **Branches parallèles** (module Inception : Szegedy et al., « Going Deeper with Convolutions », CVPR 2015) : la même entrée passe par des convolutions de tailles différentes et par un pooling de pas 1, toutes remplies pour garder la taille spatiale, puis les sorties sont concaténées. Une convolution \(1 \times 1\) placée avant une convolution coûteuse réduit le nombre de canaux lus : sur 192 canaux, une convolution \(5 \times 5\) à 32 filtres a \(@LI_direct@\) paramètres ; précédée d'une convolution \(1 \times 1\) à 16 filtres, l'ensemble en a \(@LI_reduit@\), soit \(@LI_rapport:1m@\) fois moins (calcul par C2).
- **Sauts par concaténation** (U-Net : Ronneberger, Fischer et Brox, « U-Net: Convolutional Networks for Biomedical Image Segmentation », MICCAI 2015) : un encodeur réduit la taille spatiale par pooling, un décodeur la rétablit, et chaque étage du décodeur concatène sa carte avec celle de l'encodeur de même taille. La couche suivante lit donc la somme des canaux : avec 64 et 64 canaux, une convolution \(3 \times 3\) à 64 filtres lit \(@LU_canaux@\) canaux et a \(@LU_params@\) paramètres, contre \(@LU_params_sans_saut@\) sans le saut. L'article d'origine utilise des convolutions sans remplissage et rogne la carte de l'encodeur avant de concaténer.
- **Sauts par addition** (ResNet : He, Zhang, Ren et Sun, « Deep Residual Learning for Image Recognition », CVPR 2016) : l'entrée du bloc est additionnée à sa sortie, ce qui exige la même forme ; quand le bloc change le nombre de canaux ou la taille spatiale, une des options de l'article est de faire passer le saut par une convolution \(1 \times 1\) qui rétablit la forme.

**Pièges.**

1. « Deux convolutions appliquées à l'image puis concaténées » décrit deux branches **parallèles**. En série, il n'y a qu'une sortie, rien à concaténer, et la seconde couche lit les canaux de la première : la convolution \(5 \times 5\) à 24 filtres aurait alors \(24 \times (25 \times 16 + 1) = @LC_seq_params@\) paramètres au lieu de \(@LC_paramsB@\).
2. La concaténation additionne les canaux ; elle ne les multiplie pas et ne garde pas le plus grand.
3. Le pooling ne touche pas aux canaux.
4. Une bibliothèque peut additionner deux tenseurs de formes différentes par diffusion (broadcasting) : PyTorch 2.4.1 additionne un tenseur à 1 canal et un tenseur à 4 canaux sans erreur. Ce n'est pas une connexion résiduelle, et (C3) parle de l'addition terme à terme.

### 1.4 Champ réceptif avec des pas

**Le problème.** De combien de pixels d'entrée dépend une sortie ? La séance de ML 4 (partie 3.1, résultat C1) donne \(1 + \sum_l d_l(k_l - 1)\) quand tous les pas valent 1. Un pas ou un pooling change le compte.

**L'énoncé** (C4). On empile \(M\) couches (convolutions ou poolings) de tailles \(k_l\), de pas \(s_l\) et de dilatations \(d_l\). On pose \(r_0 = 1\), \(j_0 = 1\), puis, pour \(l = 1, \dots, M\) : \(r_l = r_{l-1} + d_l(k_l - 1)\,j_{l-1}\) et \(j_l = j_{l-1}\,s_l\). Alors une sortie de la couche \(l\) ne dépend que d'une fenêtre de \(r_l\) positions consécutives de l'entrée (son **champ réceptif**), et les fenêtres de deux sorties voisines sont décalées de \(j_l\) positions (le **saut**).

**La preuve**, par récurrence sur \(l\).

1. \(l = 0\) : une « sortie » est une entrée ; fenêtre de taille 1, décalage 1.
2. Supposons le résultat vrai au rang \(l - 1\). Une sortie de la couche \(l\) lit \(k_l\) sorties de la couche \(l - 1\), séparées de \(d_l\) rangs. Leurs fenêtres ont chacune \(r_{l-1}\) positions, et la fenêtre de la dernière est décalée de \(d_l(k_l - 1)\,j_{l-1}\) positions par rapport à celle de la première (hypothèse de récurrence : \(j_{l-1}\) par rang). La réunion tient dans une fenêtre de \(r_{l-1} + d_l(k_l - 1)\,j_{l-1}\) positions.
3. Passer à la sortie voisine de la couche \(l\) avance de \(s_l\) rangs dans la couche \(l - 1\), donc de \(s_l\,j_{l-1}\) positions dans l'entrée.

**Exemple.** Convolution 3, pooling 2 de pas 2, convolution 3, pooling 2 de pas 2, convolution 3 (dilatation 1 partout) : champs réceptifs successifs \(@LR_etapes@\), saut final \(@LR_saut@\). Sans tenir compte des pas, on aurait trouvé \(@LR_sans_pas@\). Mesure directe en dimension 1 : on modifie une entrée à la fois et on regarde si la sortie centrale change.

```python-exec
def conv1d(x, w, s=1):
    return np.array([np.sum(x[i:i + len(w)] * w) for i in range(0, len(x) - len(w) + 1, s)])

def reseau(x):                                             # conv 3 ; pooling 2 de pas 2 ; conv 3 ; pooling 2 de pas 2 ; conv 3
    for k, s in [(3, 1), (2, 2), (3, 1), (2, 2), (3, 1)]:
        x = conv1d(x, np.ones(k), s)                       # seule la fenêtre lue compte ici : un max-pooling lit la même
    return x

centre = len(reseau(np.zeros(100))) // 2
influents = [i for i in range(100) if reseau(np.eye(100)[i])[centre] != 0]
print("entrées qui modifient la sortie centrale : de", influents[0], "à", influents[-1], ", soit", len(influents))
```

**Pièges.** Le pas d'une couche n'agit pas sur son propre terme, mais sur ceux des couches **suivantes** (c'est \(j_{l-1}\), pas \(j_l\), qui multiplie). (C4) donne l'étendue de la fenêtre ; toutes ses positions sont réellement lues quand les fenêtres réunies à l'étape 2 se touchent ou se chevauchent, c'est-à-dire quand \(d_l\,j_{l-1} \le r_{l-1}\) (même argument que le résultat C2 de la séance de ML 4).

### 1.5 Convolution transposée

**Le problème.** Un décodeur doit **augmenter** la taille spatiale. La convolution transposée est une couche apprise qui le fait.

**La définition.** Chaque valeur d'entrée dépose le noyau, multiplié par cette valeur, dans la sortie ; deux dépôts voisins sont décalés de \(s\) ; les dépôts qui se recouvrent s'additionnent. Un remplissage \(p\) retire \(p\) positions de chaque côté du résultat.

**L'énoncé** (C5). Pour une entrée de taille \(n\) : \(m = (n - 1)\,s - 2p + d(k - 1) + 1\), plus un éventuel complément \(o\) avec \(0 \le o < s\) (argument `output_padding` de PyTorch). C'est la plus petite taille qu'une convolution de mêmes \(k\), \(p\), \(s\), \(d\) envoie sur \(n\).

**La preuve.** Le dernier dépôt commence à la position \((n - 1)s\) et couvre \(e = d(k - 1) + 1\) positions : la sortie a \((n - 1)s + e\) positions, moins \(2p\). Réciproquement, par (C1), une convolution envoie une taille \(m'\) sur \(n\) si et seulement si \(\lfloor (m' + 2p - e)/s \rfloor = n - 1\), soit \((n - 1)s \le m' + 2p - e < ns\) : les tailles possibles sont \(m' = (n - 1)s - 2p + e + o\) avec \(0 \le o < s\), et la plus petite correspond à \(o = 0\).

**Exemple.** Entrée de taille 24. Noyau 2, pas 2 : \(m = @LT_double@\), la taille double. Noyau 4, pas 2, \(p = 1\) : \(@LT_k4@\). Noyau 3, pas 2, \(p = 1\) : \(@LT_m1@\), ou \(@LT_m2@\) avec \(o = 1\) ; les deux tailles sont envoyées sur 24 par la convolution correspondante, d'où le besoin de l'argument \(o\). Noyau 3, pas 2, \(p = 0\) : \(@LT_k3_sans@\). Ces formes ont été retrouvées avec `nn.ConvTranspose2d` (PyTorch 2.4.1) ; `Conv2DTranspose` de Keras 3.12.2 donne \(ns\) avec `padding='same'` et \((n - 1)s + k\) avec `'valid'` sur les cas essayés (\(k \ge s\)).

```python-exec
def conv1d_transposee(y, w, s):
    t = np.zeros((len(y) - 1) * s + len(w))
    for j, v in enumerate(y):
        t[j * s:j * s + len(w)] += v * w                   # chaque entrée dépose le noyau, tous les s
    return t

w1 = np.array([1.0, 2.0, 3.0])
print("transposée de (1, 10, 100), noyau (1, 2, 3), pas 2 :", conv1d_transposee(np.array([1.0, 10.0, 100.0]), w1, 2))
xx, yy = rng.normal(size=9), rng.normal(size=4)
print("tailles : convolution", conv1d(xx, w1, 2).shape, "; transposée", conv1d_transposee(yy, w1, 2).shape)
print("<conv(x), y> =", np.dot(conv1d(xx, w1, 2), yy).round(6), "; <x, transposée(y)> =", np.dot(xx, conv1d_transposee(yy, w1, 2)).round(6))
```

La dernière ligne explique le nom : si la convolution est l'application linéaire de matrice \(\mathbf{A}\), la convolution transposée de même noyau est celle de matrice \(\mathbf{A}^\top\), puisque \((\mathbf{A}\mathbf{x})^\top \mathbf{y} = \mathbf{x}^\top (\mathbf{A}^\top \mathbf{y})\).

**Piège.** Elle rétablit la **forme**, pas les valeurs : \(\mathbf{A}^\top\) n'est pas l'inverse de \(\mathbf{A}\). L'appeler « déconvolution » est un abus de langage.

[[controle:forme]]

## Partie 2 : mettre en place un entraînement

### 2.1 Trois ensembles, trois rôles

Les rôles (l'apprentissage ajuste les poids, la validation choisit les hyperparamètres et le moment d'arrêt, le test mesure une fois à la fin) et les fuites d'information sont traités dans la séance de ML 3, partie 4. Trois compléments pour des images.

1. **L'unité de découpage est l'exemple entier**, ici la paire (image, masque), jamais le pixel : les pixels d'une même image sont dépendants, et en placer dans deux ensembles est une fuite. Si plusieurs images viennent d'une même source (même patient, même scène, images voisines d'une vidéo), on découpe par source, pour la même raison.
2. **Les proportions sont un choix, pas une règle.** 70/15/15 ou 80/10/10 sont des conventions. Ce qui fixe la taille du test est la précision voulue : une proportion \(q\) mesurée sur \(n_{\text{test}}\) exemples indépendants a pour erreur standard \(\sqrt{q(1 - q)/n_{\text{test}}}\) (variance d'une moyenne de variables de Bernoulli indépendantes).
3. Les trois ensembles sont disjoints, et leurs tailles somment à la taille totale : c'est le premier contrôle de cohérence d'un énoncé.

Exemple : \(@LE_n@\) paires, découpées en 70 %, 15 % et 15 % : \(@LE_app@\), \(@LE_val@\) et \(@LE_test@\). Une exactitude de 0,9 mesurée sur \(@LE_test@\) images a une erreur standard de \(@LE_se:2m@\) ; il faudrait quatre fois plus d'images de test pour la ramener à \(@LE_se_x4:2m@\).

### 2.2 Époque, lot, itération

**Les définitions.** Un **lot** (batch) est un paquet de \(B\) exemples d'apprentissage sur lequel on calcule un gradient moyen. Une **itération** (ou pas d'entraînement) est une mise à jour des poids, faite avec un lot. Une **époque** est un passage sur tous les exemples d'apprentissage, chacun utilisé une fois.

**L'énoncé** (E1). Avec \(n_{\text{app}}\) exemples d'apprentissage et des lots de taille \(B\), une époque compte \(\lceil n_{\text{app}}/B \rceil\) itérations si l'on garde le dernier lot incomplet, \(\lfloor n_{\text{app}}/B \rfloor\) si on l'écarte. Un entraînement de \(n_e\) époques fait \(n_e\) fois ce nombre de mises à jour.

**La preuve.** Division euclidienne : \(n_{\text{app}} = uB + v\) avec \(0 \le v < B\). On forme \(u\) lots complets, et les \(v\) exemples restants forment un lot incomplet si \(v > 0\) : \(u + 1 = \lceil n_{\text{app}}/B \rceil\) lots dans ce cas, \(u\) si \(v = 0\) ou si l'on écarte le reste.

**Exemple.** \(n_{\text{app}} = @LE_app@\), \(B = @LE_lot@\) : \(@LE_app@ = @LE_quotient@ \times 32 + @LE_reste@\), donc \(@LE_pas@\) itérations par époque, la dernière sur un lot de \(@LE_reste@\) exemples ; \(@LE_epoques@\) époques font \(@LE_maj@\) mises à jour. C'est la taille de l'**apprentissage** qui compte, pas la taille totale : avec \(@LE_n@\) on aurait écrit \(@LE_pas_faux@\).

**Ce qui se passe avec une valeur incohérente.** Certaines interfaces demandent le nombre \(u'\) de pas par époque. Ce nombre fixe ce que l'outil appelle « époque », soit \(u' B\) exemples consommés :

- \(u' B < n_{\text{app}}\) : une « époque » n'est plus un passage complet. Tout ce qui est compté en époques (calendrier de pas d'apprentissage, patience de l'arrêt précoce, fréquence de la validation) se déclenche plus tôt qu'on ne le croit ;
- \(u' B > n_{\text{app}}\) : il faut plus d'exemples qu'il n'y en a ; soit les données s'épuisent, soit des exemples repassent dans la même « époque ». Avec \(u' = @LE_pas_faux@\) et \(B = 32\), on demande \(@LE_exemples_faux@\) exemples alors qu'il y en a \(@LE_app@\) ;
- pour la validation, on veut évaluer chaque exemple exactement une fois : \(\lceil n_{\text{val}}/B \rceil\) pas, ici \(@LE_pas_val@\) (le dernier lot contient \(@LE_reste_val@\) exemple) ; avec moins de pas, des exemples ne sont jamais évalués.

Comportements exécutés le 9 octobre 2026 : avec Keras 3.12.2, `fit` sur 1050 exemples en tableaux numpy avec `batch_size=32` fait 33 pas par époque, et 32 est la taille de lot par défaut ; `steps_per_epoch=10` fait 10 pas par époque ; `steps_per_epoch=50` s'arrête à 33 pas avec un avertissement d'épuisement des données. Avec PyTorch 2.4.1, un `DataLoader` de 1400 exemples et `batch_size=32` fournit 44 lots, et 43 avec `drop_last=True`. Avec d'autres sources de données (générateurs, jeux répétés à l'infini), le comportement est à vérifier.

**Piège.** La taille de lot agit sur deux choses à la fois : le bruit du gradient, dont la variance est divisée par \(B\) (séance de ML 4, partie 1.3), et le nombre de mises à jour par époque, divisé par \(B\). Comparer deux tailles de lot « à nombre d'époques égal » compare donc aussi des nombres de mises à jour différents.

### 2.3 Le pas d'apprentissage

**Le problème.** Quel ordre de grandeur donner au pas d'apprentissage \(\eta\) ? Il n'existe pas de valeur universelle, et l'on peut dire pourquoi.

**L'énoncé** (P1). Soit \(f\) une fonction deux fois continûment dérivable dont la hessienne a, en tout point, toutes ses valeurs propres inférieures ou égales à \(L > 0\). Alors un pas de descente de gradient \(\boldsymbol{\theta}' = \boldsymbol{\theta} - \eta\,\nabla f(\boldsymbol{\theta})\) vérifie \(f(\boldsymbol{\theta}') \le f(\boldsymbol{\theta}) - \eta\,(1 - \tfrac{L\eta}{2})\,\lVert \nabla f(\boldsymbol{\theta}) \rVert^2\). Pour \(0 < \eta < 2/L\), la perte ne peut donc pas augmenter.

**La preuve.**

1. Formule de Taylor avec reste de Lagrange (théorème général, admis) le long du segment : il existe un point \(\boldsymbol{\xi}\) du segment tel que \(f(\boldsymbol{\theta} + \mathbf{h}) = f(\boldsymbol{\theta}) + \nabla f(\boldsymbol{\theta})^\top \mathbf{h} + \tfrac12 \mathbf{h}^\top \nabla^2 f(\boldsymbol{\xi})\,\mathbf{h}\).
2. Les valeurs propres de la hessienne sont au plus \(L\), donc \(\mathbf{h}^\top \nabla^2 f(\boldsymbol{\xi})\,\mathbf{h} \le L\,\lVert \mathbf{h} \rVert^2\) (quotient de Rayleigh, séance d'algèbre linéaire, section 1).
3. Avec \(\mathbf{h} = -\eta\,\nabla f(\boldsymbol{\theta})\) : \(f(\boldsymbol{\theta}') \le f(\boldsymbol{\theta}) - \eta\,\lVert \nabla f \rVert^2 + \tfrac{L\eta^2}{2}\lVert \nabla f \rVert^2\).

**Conséquence : l'échelle du bon pas est \(1/L\), et \(L\) dépend du problème.** Multiplier toutes les variables d'entrée par 10 multiplie par 100 la hessienne des moindres carrés, \(\mathbf{X}^\top \mathbf{X}/n\), donc divise par 100 le pas admissible ; sommer la perte sur les \(n\) exemples au lieu de la moyenner multiplie \(L\) par \(n\). Pour la régression logistique, la hessienne de la log-loss moyenne est \(\tfrac1n \mathbf{X}^\top \mathbf{W}\mathbf{X}\) avec des poids \(\pi_i(1 - \pi_i) \le \tfrac14\) (séance de ML 3, partie 1), donc \(L\) peut être pris égal au quart de la plus grande valeur propre de \(\mathbf{X}^\top \mathbf{X}/n\). Expérience : descente de gradient sur une régression logistique, 200 pas, pour des pas d'apprentissage espacés d'un facteur 10.

```python-exec
from scipy.special import expit                            # sigmoïde calculée sans dépassement

rng = np.random.default_rng(1)
n, p = 400, 5
X = rng.normal(size=(n, p)) * np.array([1.0, 1.0, 1.0, 1.0, 3.0])
y = (rng.random(n) < expit(X.dot(np.array([1.0, -1.0, 0.5, 0.0, 0.3])))).astype(float)
L = np.linalg.eigvalsh(X.T.dot(X) / n).max() / 4           # majorant de la plus grande valeur propre de la hessienne
print("pas limite garanti 2 / L :", round(2 / L, 3))

def perte_finale(eta, pas=200):
    b = np.zeros(p)
    for _ in range(pas):
        b = b - eta * X.T.dot(expit(X.dot(b)) - y) / n     # gradient de la log-loss moyenne
    z = X.dot(b)
    return np.mean(np.logaddexp(0, z) - y * z)

print("log-loss au départ :", round(np.log(2), 4))
for eta in [1e-4, 1e-3, 1e-2, 1e-1, 2 / L, 3.0, 30.0]:
    print(f"pas {eta:8.4f} : log-loss après 200 pas {perte_finale(eta):.4f}")
```

Lecture : trop petit, le pas laisse la perte presque à sa valeur de départ ; au-delà de la limite, la perte finit **au-dessus** de sa valeur de départ. Entre les deux, plusieurs ordres de grandeur conviennent, à des vitesses différentes.

**Valeurs par défaut des bibliothèques : des faits de version, pas des vérités.** Lues dans les signatures le 9 octobre 2026 :

| Optimiseur | PyTorch 2.4.1 | Keras 3.12.2 |
|---|---|---|
| SGD | 0,001 | 0,01 |
| Adam | 0,001 | 0,001 |
| AdamW | 0,001 | 0,001 |
| RMSprop | 0,01 | 0,001 |
| Adagrad | 0,01 | 0,001 |

Les deux bibliothèques ne s'accordent ni sur SGD ni sur RMSprop : une valeur par défaut est un point de départ, à régler sur la validation. Ce que l'on peut affirmer sans bibliothèque :

- avec Adam, le premier pas déplace chaque poids d'environ \(\eta\), quelle que soit l'échelle du gradient (séance de ML 4, partie 1.4) ; or une initialisation de He sur une couche de 512 entrées tire des poids d'écart-type \(\sqrt{2/512} = @LA_he:4m@\) (séance de ML 4, partie 2.2). Un pas \(\eta = 1\) ou \(\eta = 10\) remplace donc les poids par des valeurs sans rapport avec l'initialisation dès la première mise à jour ;
- une perte qui augmente ou devient infinie signale un pas trop grand ; une perte qui décroît à peine sur des centaines de pas, un pas trop petit (ou un gradient qui disparaît, séance de ML 4, partie 2.3).

### 2.4 Choisir la perte et l'activation de sortie

**Le principe** (V1). On choisit un modèle probabiliste de la cible sachant la sortie du réseau, et la perte est l'opposé de sa log-vraisemblance. L'activation de sortie est celle qui transforme un réel quelconque en paramètre valide de ce modèle.

1. **Régression** : cible réelle, modèle « sortie plus bruit gaussien de variance fixe ». L'opposé de la log-densité vaut \((y - \hat y)^2/(2\sigma^2)\) plus une constante : erreur quadratique, sortie linéaire (aucune activation).
2. **Classification binaire** : loi de Bernoulli de paramètre \(\pi = \mathrm{sig}(z)\), où \(z\) est la sortie brute (le **logit**). Perte : entropie croisée binaire \(-y \ln \pi - (1 - y)\ln(1 - \pi)\) (séance de ML 3, partie 1).
3. **\(K\) classes exclusives** : \(K\) logits \(z_1, \dots, z_K\), probabilités \(p_j = e^{z_j}/\sum_{i} e^{z_i}\) (**softmax**). Perte : entropie croisée catégorielle \(-\ln p_y\), où \(y\) est la vraie classe.
4. **\(K\) étiquettes non exclusives** (multi-étiquette) : \(K\) lois de Bernoulli, donc \(K\) sigmoïdes et la somme de \(K\) entropies croisées binaires. Un softmax serait faux : il impose aux probabilités de sommer à 1.
5. **Segmentation** : une classification **par pixel**. La carte de sortie a la taille spatiale du masque et 1 canal (sigmoïde, deux classes) ou \(K\) canaux (softmax, \(K\) classes) ; la perte est la moyenne sur les pixels de l'entropie croisée du pixel.

**L'énoncé.**

1. (V2) Pour le softmax et l'entropie croisée catégorielle \(\ell = -\ln p_y\) : \(\partial \ell/\partial z_j = p_j - I_{\{j = y\}}\).
2. (V3) Avec deux classes de logits \(z_0\) et \(z_1\) : \(p_1 = \mathrm{sig}(z_1 - z_0)\). Deux canaux suivis d'un softmax et un canal suivi d'une sigmoïde décrivent donc la même famille de modèles, avec la même perte.
3. (V4) Derrière une sigmoïde, le gradient par rapport au logit vaut \(\pi - y\) pour l'entropie croisée et \((\pi - y)\,\pi(1 - \pi)\) pour la perte quadratique \(\tfrac12(\pi - y)^2\).

**La preuve.**

1. (V2) \(\ell = -z_y + \ln \sum_i e^{z_i}\). La dérivée du premier terme par rapport à \(z_j\) est \(-I_{\{j = y\}}\) ; celle du second est \(e^{z_j}/\sum_i e^{z_i} = p_j\).
2. (V3) \(p_1 = e^{z_1}/(e^{z_0} + e^{z_1})\) ; on divise haut et bas par \(e^{z_1}\) : \(1/(1 + e^{-(z_1 - z_0)})\).
3. (V4) \(\mathrm{sig}' = \mathrm{sig}(1 - \mathrm{sig})\). Entropie croisée : \(\partial \ell/\partial z = \big(-y/\pi + (1 - y)/(1 - \pi)\big)\pi(1 - \pi) = \pi - y\). Quadratique : \((\pi - y)\,\pi(1 - \pi)\) par la règle de la chaîne.

**Exemples.** Logits \((2 ; 1 ; -1)\) : probabilités \(@LS_p:4m@\) ; si la vraie classe est la deuxième, la perte vaut \(@LS_perte:4m@\). Sigmoïde saturée du mauvais côté, \(z = -4\) et \(y = 1\) : \(\pi = @LP_pi:4m@\) ; gradient de l'entropie croisée \(@LP_ce:4m@\), gradient de la perte quadratique \(@LP_mse:4m@\), soit \(@LP_facteur:4m@\) fois le premier : la perte quadratique n'apprend presque plus quand le modèle est très sûr de lui et se trompe. Contrôle de (V3) sur une carte \(4 \times 4\) :

```python-exec
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

rng = np.random.default_rng(2)
z = rng.normal(size=(4, 4, 2))                             # carte 4 x 4, deux canaux de logits
p_softmax = softmax(z)[:, :, 1]
p_sigmoide = expit(z[:, :, 1] - z[:, :, 0])
masque = (rng.random((4, 4)) < 0.3).astype(float)
ce = -np.mean(np.log(np.where(masque == 1, p_softmax, 1 - p_softmax)))
bce = -np.mean(masque * np.log(p_sigmoide) + (1 - masque) * np.log(1 - p_sigmoide))
print("softmax à deux canaux = sigmoïde de la différence :", np.allclose(p_softmax, p_sigmoide))
print("entropie croisée catégorielle par pixel :", ce.round(6), "; binaire :", bce.round(6))
```

**Pièges.**

1. Appliquer l'activation deux fois. Dans PyTorch 2.4.1, `CrossEntropyLoss` attend des **logits** et des indices de classe (elle calcule elle-même le logarithme du softmax), et `BCEWithLogitsLoss` attend des logits ; dans Keras 3.12.2, les pertes `BinaryCrossentropy`, `CategoricalCrossentropy` et `SparseCategoricalCrossentropy` ont `from_logits=False` par défaut, donc attendent des probabilités (exécuté le 9 octobre 2026).
2. Confondre multi-classe (une seule classe par exemple, softmax) et multi-étiquette (plusieurs possibles, sigmoïdes).
3. Format des étiquettes : indice de classe ou vecteur indicateur (un 1 à la place de la classe, des 0 ailleurs). Dans Keras 3.12.2, `SparseCategoricalCrossentropy` prend des indices et `CategoricalCrossentropy` des vecteurs indicateurs, pour la même valeur de perte (exécuté le 9 octobre 2026).
4. Une erreur quadratique reste le bon choix pour une cible réelle, même comprise entre 0 et 1 ; c'est la nature de la cible (classe ou quantité) qui décide.

### 2.5 Dice et IoU

**Le problème.** En segmentation, quand l'objet occupe une petite part de l'image, l'exactitude par pixel récompense le masque vide. On mesure donc le recouvrement entre le masque prédit et le vrai masque.

**Les définitions.** \(A\) est l'ensemble des pixels prédits « objet », \(B_0\) l'ensemble des pixels réellement « objet » (deux ensembles fixes, notés en majuscules comme le veut l'usage pour des ensembles), \(\lvert \cdot \rvert\) le nombre de pixels. Avec les effectifs de la séance de ML 3 : \(\mathrm{VP} = \lvert A \cap B_0 \rvert\), \(\mathrm{FP} = \lvert A \rvert - \mathrm{VP}\), \(\mathrm{FN} = \lvert B_0 \rvert - \mathrm{VP}\).

- **IoU** (intersection sur union, indice de Jaccard) : \(\mathrm{IoU} = \dfrac{\lvert A \cap B_0 \rvert}{\lvert A \cup B_0 \rvert} = \dfrac{\mathrm{VP}}{\mathrm{VP} + \mathrm{FP} + \mathrm{FN}}\).
- **Dice** : \(\mathrm{Dice} = \dfrac{2\,\lvert A \cap B_0 \rvert}{\lvert A \rvert + \lvert B_0 \rvert} = \dfrac{2\,\mathrm{VP}}{2\,\mathrm{VP} + \mathrm{FP} + \mathrm{FN}}\) : c'est le score F1 de la classe « objet » (séance de ML 3, partie 2).

**L'énoncé** (S1). Si \(A \cup B_0\) n'est pas vide : \(\mathrm{Dice} = \dfrac{2\,\mathrm{IoU}}{1 + \mathrm{IoU}}\) et \(\mathrm{IoU} = \dfrac{\mathrm{Dice}}{2 - \mathrm{Dice}}\). En conséquence \(\mathrm{Dice} \ge \mathrm{IoU}\), avec égalité seulement en 0 et en 1, et sur **une** image les deux mesures classent deux prédictions dans le même ordre.

**La preuve.**

1. On note \(i = \mathrm{VP}\) et \(u = \mathrm{VP} + \mathrm{FP} + \mathrm{FN} > 0\), de sorte que \(\mathrm{IoU} = i/u\) et \(\lvert A \rvert + \lvert B_0 \rvert = 2\,\mathrm{VP} + \mathrm{FP} + \mathrm{FN} = i + u\).
2. \(\mathrm{Dice} = 2i/(i + u)\) ; en divisant haut et bas par \(u\) : \(2\,\mathrm{IoU}/(1 + \mathrm{IoU})\). On résout en \(\mathrm{IoU}\) pour la formule inverse.
3. \(\mathrm{Dice} - \mathrm{IoU} = \mathrm{IoU}\,(1 - \mathrm{IoU})/(1 + \mathrm{IoU}) \ge 0\), nul si et seulement si \(\mathrm{IoU}\) vaut 0 ou 1. Et \(t \mapsto 2t/(1 + t)\) est strictement croissante sur \([0, 1]\) (dérivée \(2/(1 + t)^2 > 0\)) : même ordre.

**Exemple.** Image de \(@LD_pixels@\) pixels, VP = \(@LD_vp@\), FP = \(@LD_fp@\), FN = \(@LD_fn@\) : \(\mathrm{IoU} = \tfrac{@LD_iou_num@}{@LD_iou_den@}\), \(\mathrm{Dice} = \tfrac{@LD_dice_num@}{@LD_dice_den@}\). L'exactitude par pixel vaut \(@LD_exactitude:3m@\), et le masque « tout en fond » obtient \(@LD_fond:3m@\) d'exactitude avec un Dice nul, parce que l'objet n'occupe que \(@LD_objet:3m@\) de l'image.

```python-exec
vrai = np.zeros((20, 20), dtype=bool)
vrai[4:9, 3:13] = True                                     # 50 pixels d'objet
pred = np.zeros((20, 20), dtype=bool)
pred[6:10, 3:13] = True                                    # 40 pixels prédits
vp, fp, fn = np.sum(pred & vrai), np.sum(pred & ~vrai), np.sum(~pred & vrai)
iou, dice = vp / (vp + fp + fn), 2 * vp / (2 * vp + fp + fn)
print("VP, FP, FN :", vp, fp, fn, "; IoU :", iou, "; Dice :", round(dice, 4), "; 2 IoU / (1 + IoU) :", round(2 * iou / (1 + iou), 4))
print("exactitude par pixel :", np.mean(pred == vrai), "; masque « tout en fond » : exactitude", np.mean(~vrai), "et Dice 0")
```

**Cas limites.** Masque vrai vide et prédiction vide : \(0/0\) ; il faut une convention (score 1, ou un petit terme ajouté en haut et en bas), à lire dans la définition utilisée. Aucun recouvrement : les deux valent 0. Les vrais négatifs n'entrent dans aucune des deux formules.

**Pièges.**

1. « Même ordre » vaut image par image, pas pour des **moyennes** : la fonction \(2t/(1 + t)\) n'est pas affine. Deux images, modèle X d'IoU \((\tfrac12 ; \tfrac12)\), modèle Y d'IoU \((\tfrac15 ; \tfrac{41}{50})\) : l'IoU moyen préfère Y (\(@LD_y_iou:2m@\) contre \(@LD_x_iou:2m@\)), le Dice moyen préfère X (\(@LD_x_dice:3m@\) contre \(@LD_y_dice:3m@\)).
2. Comme perte, on utilise une version dérivable : \(1 - \dfrac{2\sum_i \pi_i y_i}{\sum_i \pi_i + \sum_i y_i}\), où \(\pi_i\) est la probabilité prédite au pixel \(i\) et \(y_i\) son étiquette. Ce n'est pas une log-vraisemblance : elle n'a pas la justification (V1), et les probabilités qu'elle produit n'ont pas de raison d'être calibrées.

### 2.6 Classes déséquilibrées : stratifier, pondérer, mesurer

**Le problème.** Quand une classe est rare, trois choses se dérèglent : la métrique (l'exactitude récompense le classifieur constant, séance de ML 3, partie 2), le découpage (un ensemble de validation peut ne contenir presque aucun exemple rare) et l'apprentissage (la classe rare pèse peu dans la perte). À chacune son remède, et le mot « variance » y désigne deux choses qu'il faut séparer :

- la **variance du modèle** : ses prédictions changent beaucoup d'un jeu d'apprentissage à l'autre (partie 3) ; remèdes : régularisation, plus de données, modèle plus simple ;
- la **variance de l'évaluation** : la métrique mesurée change beaucoup d'un découpage à l'autre ; remèdes : stratification, plus de plis, ensemble de test plus grand.

**L'énoncé.**

1. (St1) Dans une population de \(N\) exemples dont \(N_1\) positifs (proportion \(q = N_1/N\)), on tire un ensemble de test de \(n\) exemples, uniformément et sans remise. Le nombre \(X\) de positifs du test vérifie \(\mathbb{E}[X] = nq\) et \(\mathrm{Var}(X) = nq(1 - q)\,\dfrac{N - n}{N - 1}\). Avec un découpage **stratifié** (on tire séparément \(nq\) positifs parmi les positifs et \(n(1 - q)\) négatifs parmi les négatifs, quand ces nombres sont entiers), \(X = nq\) : variance nulle.
2. (W1) Soit un point où la probabilité vraie d'être positif vaut \(q \in ]0, 1[\). La prévision \(r\) qui minimise l'entropie croisée espérée **pondérée**, \(-w\,q \ln r - (1 - q)\ln(1 - r)\) avec un poids \(w > 0\) sur les positifs, est \(r^* = \dfrac{wq}{wq + 1 - q}\) : la cote \(r^*/(1 - r^*)\) vaut \(w\) fois la vraie cote \(q/(1 - q)\).

**La preuve.**

1. (St1) \(X = \sum_{i=1}^n I_i\), où \(I_i\) vaut 1 si le \(i\)-ième exemple tiré est positif. Par symétrie, chaque tirage est uniforme sur la population : \(\mathbb{E}[I_i] = q\), d'où \(\mathbb{E}[X] = nq\) et \(\mathrm{Var}(I_i) = q(1 - q)\).
2. Pour \(i \ne j\) : \(\mathbb{P}(I_i = I_j = 1) = \dfrac{N_1(N_1 - 1)}{N(N - 1)}\), donc \(\mathrm{Cov}(I_i, I_j) = q\,\dfrac{N_1 - 1}{N - 1} - q^2 = -\dfrac{q(1 - q)}{N - 1}\) (on réduit au même dénominateur avec \(N_1 = qN\)).
3. \(\mathrm{Var}(X) = n\,q(1 - q) - n(n - 1)\dfrac{q(1 - q)}{N - 1} = nq(1 - q)\,\dfrac{N - n}{N - 1}\) : somme des \(n\) variances et des \(n(n - 1)\) covariances.
4. (W1) \(h(r) = -wq \ln r - (1 - q)\ln(1 - r)\) a pour dérivée \(-wq/r + (1 - q)/(1 - r)\), nulle si et seulement si \(wq(1 - r) = (1 - q)r\), soit \(r = wq/(wq + 1 - q)\) ; sa dérivée seconde \(wq/r^2 + (1 - q)/(1 - r)^2\) est strictement positive : minimum unique. Enfin \(r^*/(1 - r^*) = wq/(1 - q)\).

**Exemple.** \(N = @LSt_N@\), \(@LSt_K@\) positifs, test de \(@LSt_n@\) : en moyenne \(@LSt_moy@\) positifs, variance \(@LSt_var:2m@\), écart-type \(@LSt_sd:2m@\) ; la probabilité d'avoir 2 positifs ou moins vaut \(@LSt_p2:4m@\) (loi hypergéométrique, calcul exact ; simulation du script : @LSt_p2_mc:4@, erreur standard @LSt_p2_se:4@). Une métrique de rappel calculée sur 2 positifs ne prend que les valeurs 0, 0,5 et 1. Stratifié, le test contient exactement \(@LSt_moy@\) positifs.

```python-exec
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

y = np.r_[np.ones(30, dtype=int), np.zeros(970, dtype=int)]
X = np.arange(1000).reshape(-1, 1)
sans = np.array([train_test_split(X, y, test_size=200, random_state=g)[3].sum() for g in range(2000)])
avec = np.array([train_test_split(X, y, test_size=200, random_state=g, stratify=y)[3].sum() for g in range(2000)])
print("sans stratification : moyenne", sans.mean(), "; écart-type", sans.std().round(3), "; part des découpages avec 2 positifs ou moins :", np.mean(sans <= 2))
print("avec stratification : valeurs observées", sorted(set(avec)))
print("5 plis mélangés :", [int(y[t].sum()) for _, t in KFold(5, shuffle=True, random_state=0).split(X)],
      "; 5 plis stratifiés :", [int(y[t].sum()) for _, t in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)])
```

Pondération : \(q = 0{,}1\) et \(w = 9\) donnent \(r^* = \tfrac{@LW_r9_num@}{@LW_r9_den@}\) ; \(w = 4\) donne \(\tfrac{@LW_r4_num@}{@LW_r4_den@}\). Pondérer revient à déplacer le seuil : \(r^* > \tfrac12\) si et seulement si \(q > 1/(1 + w)\), soit \(q > \tfrac{@LW_seuil9_num@}{@LW_seuil9_den@}\) pour \(w = 9\). Avec 100 positifs sur 1000, les poids « balanced » de scikit-learn, \(n/(2 n_c)\) pour une classe d'effectif \(n_c\), valent \(@LW_bal_pos@\) et \(@LW_bal_neg:3m@\), de rapport \(@LW_bal_rapport@\). Expérience : régression logistique sur des données dont le vrai modèle est logistique, de constante \(-3\) et de pente \(1{,}5\).

```python-exec
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

rng = np.random.default_rng(2)
n = 20000
x = rng.normal(size=(n, 1))
y = (rng.random(n) < expit(-3.0 + 1.5 * x[:, 0])).astype(int)
poids = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
print("proportion de positifs :", y.mean(), "; poids « balanced » :", poids.round(3), "; logarithme de leur rapport :", np.log(poids[1] / poids[0]).round(3))
for cw in [None, "balanced"]:
    m = LogisticRegression(C=1e6, class_weight=cw).fit(x, y)
    pr = m.predict_proba(x)[:, 1]
    pos = pr > 0.5
    print(f"class_weight={cw}: constante {m.intercept_[0]:.3f}, pente {m.coef_[0, 0]:.3f}, probabilité moyenne {pr.mean():.3f}, "
          f"rappel {np.mean(pos[y == 1]):.3f}, précision {np.mean(y[pos]):.3f}")
```

Lecture : la pondération laisse la pente presque inchangée et décale la constante d'environ le logarithme du rapport des poids, comme l'annonce (W1) ; la probabilité moyenne prédite n'est plus la proportion de positifs ; le rappel monte, la précision baisse. La pondération ne crée pas d'information : elle change le point de fonctionnement.

**Cas limites.** \(w = 1\) : \(r^* = q\), la règle de score est propre (séance de ML 3, partie 2). Remplacer chaque positif par \(w\) copies (sur-échantillonnage, \(w\) entier) donne exactement la perte pondérée de poids \(w\) : même effet. Ne garder qu'une fraction \(1/w\) des négatifs multiplie aussi la cote par \(w\) en espérance, mais jette des données.

**Pièges.**

1. Rééquilibrer (pondérer, dupliquer, sous-échantillonner) se fait sur l'**apprentissage seul**, après le découpage : dupliquer avant de découper place des copies du même exemple des deux côtés, et un ensemble de validation rééquilibré ne mesure plus la performance à la vraie proportion.
2. La stratification ne change ni le biais ni la variance du modèle : elle stabilise la composition des ensembles, donc l'évaluation.
3. Après pondération, les probabilités ne sont plus calibrées : pour les utiliser comme probabilités, on divise la cote prédite par \(w\).
4. Une régularisation plus forte répond à une variance du modèle (écart entre erreur d'apprentissage et de validation), pas au déséquilibre lui-même.

[[controle:pas]]

## Partie 3 : biais et variance en image de cible

### 3.1 Ce que représente la cible

**Le problème.** Une figure classique montre quatre cibles de tir, avec des impacts groupés ou dispersés, centrés ou décalés, et demande laquelle est « fort biais, faible variance ». Il faut savoir ce que sont le centre et les impacts.

**Les définitions.** On fixe un point \(\mathbf{x}_0\). Le **centre** de la cible est la vraie valeur à prédire en ce point, \(m_0 = \mathbb{E}[Y_0]\). Chaque **impact** est la prédiction \(\hat Y_0\) d'un modèle entraîné sur **un** jeu d'apprentissage ; tirer un nouveau jeu d'apprentissage de la même loi donne un nouvel impact. Le **biais** est l'écart entre le centre du nuage d'impacts et le centre de la cible, \(\mathbb{E}[\hat Y_0] - m_0\) ; la **variance** est la dispersion du nuage, \(\mathrm{Var}(\hat Y_0)\).

**L'énoncé.** \(\mathbb{E}[(Y_0 - \hat Y_0)^2] = \sigma^2 + \text{biais}^2 + \text{variance}\), où \(\sigma^2\) est la variance du bruit de \(Y_0\) : démontré dans la séance de ML 1, partie 6, énoncé 1. Le bruit n'est pas sur la cible : même un nuage parfaitement groupé au centre laisse l'erreur \(\sigma^2\).

| Image | Biais | Variance | Situation |
|---|---|---|---|
| impacts groupés au centre | faible | faible | l'idéal ; l'erreur restante est le bruit |
| impacts groupés, loin du centre | fort | faible | modèle trop rigide : sous-apprentissage |
| impacts dispersés autour du centre | faible | forte | modèle trop souple pour la quantité de données : surapprentissage |
| impacts dispersés, loin du centre | fort | forte | mauvais modèle et trop peu de données |

**Exemple entièrement calculé par simulation.** La vraie fonction est \(\sin(3x)\), le bruit a pour écart-type 0,3, on vise \(x_0 = 0{,}5\). Chaque ligne tire 2000 jeux d'apprentissage, ajuste un polynôme par moindres carrés (avec une pénalité Ridge sur tous les coefficients à la dernière ligne), et résume les 2000 impacts.

```python-exec
rng = np.random.default_rng(3)
f = lambda u: np.sin(3 * u)                                # la vraie fonction
x0, sigma = 0.5, 0.3                                       # point visé, écart-type du bruit

def tirs(degre, n, lam=0.0, R=2000):
    # R jeux d'apprentissage indépendants ; renvoie les R prédictions en x0 (les R impacts sur la cible)
    preds = np.empty(R)
    for r in range(R):
        x = rng.uniform(-1, 1, n)
        yv = f(x) + sigma * rng.normal(size=n)
        A = np.vander(x, degre + 1)                        # colonnes x^degre, ..., x, 1
        b = np.linalg.solve(A.T.dot(A) + lam * np.eye(degre + 1), A.T.dot(yv))
        preds[r] = np.vander([x0], degre + 1).dot(b)[0]
    return preds

for nom, degre, n, lam in [("degré 1, n = 30", 1, 30, 0.0), ("degré 1, n = 300", 1, 300, 0.0), ("degré 4, n = 30", 4, 30, 0.0),
                           ("degré 12, n = 30", 12, 30, 0.0), ("degré 12, n = 300", 12, 300, 0.0), ("degré 12, n = 30, ridge 0,01", 12, 30, 0.01)]:
    t = tirs(degre, n, lam)
    print(f"{nom:28s}: biais {t.mean() - f(x0):+.3f}, écart-type {t.std():.3f}, erreur quadratique {np.mean((t - f(x0)) ** 2):.4f}")
```

Lecture, ligne par ligne :

1. degré 1 : impacts groupés loin du centre (fort biais, faible variance) ;
2. dix fois plus de données : le nuage se resserre, il ne se recentre pas ;
3. degré 4 : groupé près du centre ;
4. degré 12 avec 30 points : centré mais très dispersé (faible biais, forte variance) ;
5. dix fois plus de données : la dispersion s'effondre ;
6. même degré 12 et mêmes 30 points, avec une petite pénalité : la dispersion retombe au niveau du degré 4, pour un biais qui reste petit.

### 3.2 De la lecture des erreurs au remède

**Reconnaître la situation sans voir la cible.** On ne dispose que d'un jeu de données, donc d'un seul impact. On lit deux erreurs, mesurées avec la même perte : l'erreur d'apprentissage et l'erreur de validation.

- Les deux sont élevées et proches : le modèle n'arrive pas à suivre même ses données d'apprentissage ; c'est le signe d'un **fort biais**.
- L'erreur d'apprentissage est faible et l'erreur de validation nettement plus élevée : le modèle suit le bruit de son échantillon ; c'est le signe d'une **forte variance**.

Ce sont des indices, pas des théorèmes : « élevée » se juge par rapport au bruit irréductible ou à une référence, et un écart peut aussi venir d'une différence de loi entre apprentissage et validation.

| Remède | Effet sur le biais | Effet sur la variance | Où c'est démontré ou montré |
|---|---|---|---|
| plus de données d'apprentissage | inchangé | baisse | lignes 2 et 5 ci-dessus ; variance des MCO en \(\sigma^2(\mathbf{X}^\top \mathbf{X})^{-1}\), séance de ML 1, partie 2 |
| régularisation plus forte (L1, L2, dropout, arbres moins profonds, arrêt précoce) | monte | baisse | Ridge : séance de ML 1, partie 6 ; ligne 6 ci-dessus |
| modèle plus complexe (plus de variables, de degrés, de couches, de profondeur) | baisse | monte | lignes 1, 3, 4 ci-dessus ; arbre : séance de ML 2, partie 1 |
| moyenne de modèles (bagging, forêt) | inchangé | baisse | séance de ML 2, parties 2 et 3 |
| boosting d'arbres peu profonds | baisse | peut monter avec le nombre d'arbres | séance de ML 2, partie 4 |

**Cas limite.** La colonne « modèle plus complexe » décrit le régime où le nombre de paramètres reste inférieur au nombre d'observations. Au-delà du seuil d'interpolation, l'erreur de test peut redescendre (double descente : séance quantitative, partie 2).

**Pièges.**

1. Les impacts ne sont pas les observations d'un jeu de données : ce sont les prédictions de modèles entraînés sur des jeux différents.
2. « Forte variance » ne veut pas dire « données bruitées » : c'est la sensibilité du modèle à son échantillon.
3. Augmenter la complexité n'est jamais un remède au surapprentissage dans le régime classique : c'est le remède au sous-apprentissage.
4. Ajouter des données ne corrige pas un biais.

[[controle:cible]]

## Partie 4 : apprentissage par renforcement

### 4.1 Le cadre : processus de décision markovien

**Le problème.** Un agent agit dans un environnement, étape après étape ; chaque action rapporte une récompense et change la situation. Personne ne lui donne la bonne action : il n'a que les récompenses. Comment définir, puis trouver, un bon comportement ?

**Les définitions.** Un **processus de décision markovien** est la donnée de :

- un ensemble fini d'**états** \(\mathcal{S}\) et un ensemble fini d'**actions** \(\mathcal{A}\) ;
- des probabilités de **transition** \(p(s' \mid s, a)\) : la probabilité d'arriver dans l'état \(s'\) quand on fait l'action \(a\) dans l'état \(s\) ; elles ne dépendent que de \(s\) et \(a\), pas du passé (propriété de Markov) ;
- une **récompense** \(r(s, a)\), nombre réel reçu en faisant \(a\) dans \(s\), avec \(\lvert r(s, a) \rvert \le r_{\max}\) ;
- un **facteur d'actualisation** \(\gamma \in [0, 1[\).

Une **politique** \(\pi\) donne, pour chaque état \(s\), une loi \(\pi(a \mid s)\) sur les actions. À partir d'un état \(S_0\), l'agent produit une trajectoire aléatoire \(S_0, A_0, R_1, S_1, A_1, R_2, \dots\), où \(A_t\) suit \(\pi(\cdot \mid S_t)\), \(R_{t+1} = r(S_t, A_t)\) et \(S_{t+1}\) suit \(p(\cdot \mid S_t, A_t)\). Le **retour** depuis l'instant \(t\) est la somme actualisée des récompenses futures :

\(\displaystyle G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{i \ge 0} \gamma^i R_{t+1+i}.\)

La série converge, et \(\lvert G_t \rvert \le r_{\max}/(1 - \gamma)\) (série géométrique). Les **fonctions de valeur** d'une politique sont \(v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]\) et \(q_\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]\), où \(\mathbb{E}_\pi\) désigne l'espérance quand les actions suivent \(\pi\). L'objectif est une politique qui maximise \(v_\pi(s)\) en tout état : la **récompense** est immédiate, le **retour** est cumulé, et c'est l'espérance du retour que l'on maximise.

**Exemple.** Récompenses 1, 0, 4 puis 0 ensuite, \(\gamma = 0{,}9\) : \(G_0 = 1 + 0{,}9 \times 0 + 0{,}81 \times 4 = @LG_ex:2m@\). Récompense 2 à chaque pas : \(G_0 = 2/(1 - 0{,}9) = @LG_const@\).

**Ce qui distingue ce cadre de l'apprentissage supervisé.**

1. Pas d'étiquette : la récompense évalue l'action faite, elle ne dit pas quelle était la bonne.
2. Les données dépendent de l'agent : ce qu'il observe dépend de ses actions ; les observations ne sont ni indépendantes ni de même loi.
3. Les conséquences sont différées : une action peut ne payer que bien plus tard, à travers les états qu'elle rend accessibles.
4. Il faut explorer : une action jamais essayée a une valeur inconnue (partie 4.5).

### 4.2 Équations de Bellman et itération sur la valeur

**L'énoncé.**

1. (Be1) Pour toute politique \(\pi\) et tout état \(s\) : \(\displaystyle v_\pi(s) = \sum_a \pi(a \mid s)\Big[r(s, a) + \gamma \sum_{s'} p(s' \mid s, a)\,v_\pi(s')\Big]\), et le crochet vaut \(q_\pi(s, a)\). Sous forme matricielle, \(\mathbf{v}_\pi = \mathbf{r}_\pi + \gamma\,\mathbf{P}_\pi \mathbf{v}_\pi\), où \(\mathbf{r}_\pi\) a pour coordonnées \(\sum_a \pi(a \mid s)\,r(s, a)\) et \(\mathbf{P}_\pi\) pour coefficients \(\sum_a \pi(a \mid s)\,p(s' \mid s, a)\) ; ce système a une solution unique.
2. (Be2) Pour un vecteur \(\mathbf{v}\) indexé par les états, on pose \((\mathcal{T}\mathbf{v})(s) = \max_a \big[r(s, a) + \gamma \sum_{s'} p(s' \mid s, a)\,v(s')\big]\). Alors \(\lVert \mathcal{T}\mathbf{v} - \mathcal{T}\mathbf{w} \rVert_\infty \le \gamma\,\lVert \mathbf{v} - \mathbf{w} \rVert_\infty\) ; \(\mathcal{T}\) a un unique point fixe \(\mathbf{v}^*\) (**équation de Bellman d'optimalité** \(\mathbf{v}^* = \mathcal{T}\mathbf{v}^*\)) ; \(v^*(s) \ge v_\pi(s)\) pour toute politique \(\pi\) et tout état \(s\), avec égalité pour la politique **gloutonne** qui choisit en chaque état une action réalisant le maximum.
3. (Be3) **Itération sur la valeur** : \(\mathbf{v}_{i+1} = \mathcal{T}\mathbf{v}_i\) vérifie \(\lVert \mathbf{v}_i - \mathbf{v}^* \rVert_\infty \le \gamma^i\,\lVert \mathbf{v}_0 - \mathbf{v}^* \rVert_\infty\).

**La preuve.**

1. \(G_t = R_{t+1} + \gamma\,G_{t+1}\) : on met \(\gamma\) en facteur dans la série à partir du deuxième terme.
2. Par linéarité de l'espérance conditionnelle : \(v_\pi(s) = \mathbb{E}_\pi[R_{t+1} \mid S_t = s] + \gamma\,\mathbb{E}_\pi[G_{t+1} \mid S_t = s]\). Le premier terme vaut \(\sum_a \pi(a \mid s)\,r(s, a)\) (on conditionne par \(A_t\)).
3. Pour le second, on conditionne par \(A_t = a\) puis \(S_{t+1} = s'\), de probabilités \(\pi(a \mid s)\) et \(p(s' \mid s, a)\). Sachant \(S_{t+1} = s'\), la suite de la trajectoire ne dépend plus de \(s\) ni de \(a\) (propriété de Markov, et la politique ne regarde que l'état courant) : \(\mathbb{E}_\pi[G_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s'] = v_\pi(s')\). D'où (Be1).
4. Unicité : si \((\mathbf{I} - \gamma\mathbf{P}_\pi)\mathbf{u} = \mathbf{0}\), alors \(\lVert \mathbf{u} \rVert_\infty = \gamma\,\lVert \mathbf{P}_\pi \mathbf{u} \rVert_\infty \le \gamma\,\lVert \mathbf{u} \rVert_\infty\), car chaque ligne de \(\mathbf{P}_\pi\) est faite de nombres positifs de somme 1 ; comme \(\gamma < 1\), \(\mathbf{u} = \mathbf{0}\) : la matrice est inversible.
5. (Be2), contraction. Pour deux fonctions \(g\) et \(h\) sur \(\mathcal{A}\) : \(g(a) \le h(a) + \max_{a'} \lvert g(a') - h(a') \rvert\) pour tout \(a\), donc \(\max g \le \max h + \max \lvert g - h \rvert\), et de même en échangeant les rôles : \(\lvert \max g - \max h \rvert \le \max \lvert g - h \rvert\). On l'applique aux deux crochets : leur différence vaut \(\gamma \sum_{s'} p(s' \mid s, a)\,(v(s') - w(s'))\), de valeur absolue au plus \(\gamma\,\lVert \mathbf{v} - \mathbf{w} \rVert_\infty\).
6. Point fixe. Existence : théorème du point fixe de Banach pour une application contractante de \(\mathbb{R}^{\lvert \mathcal{S} \rvert}\) (théorème général, admis). Unicité : deux points fixes vérifient \(\lVert \mathbf{v} - \mathbf{w} \rVert_\infty \le \gamma\,\lVert \mathbf{v} - \mathbf{w} \rVert_\infty\), donc sont égaux. (Be3) : récurrence sur \(i\) avec \(\mathbf{w} = \mathbf{v}^* = \mathcal{T}\mathbf{v}^*\).
7. Optimalité. Soit \(\pi^*\) la politique gloutonne pour \(\mathbf{v}^*\). L'équation \(\mathbf{v}^* = \mathcal{T}\mathbf{v}^*\) s'écrit \(\mathbf{v}^* = \mathbf{r}_{\pi^*} + \gamma\,\mathbf{P}_{\pi^*}\mathbf{v}^*\) : par l'unicité de l'étape 4, \(\mathbf{v}^* = \mathbf{v}_{\pi^*}\). Pour une politique \(\pi\) quelconque, notons \(\mathcal{T}_\pi \mathbf{v} = \mathbf{r}_\pi + \gamma\,\mathbf{P}_\pi \mathbf{v}\). Une moyenne étant inférieure au maximum, \(\mathcal{T}_\pi \mathbf{v}^* \le \mathcal{T}\mathbf{v}^* = \mathbf{v}^*\) coordonnée par coordonnée ; \(\mathcal{T}_\pi\) conserve les inégalités (coefficients de \(\mathbf{P}_\pi\) positifs), donc \(\mathcal{T}_\pi^i \mathbf{v}^* \le \mathbf{v}^*\) pour tout \(i\) ; et \(\mathcal{T}_\pi^i \mathbf{v}^* \to \mathbf{v}_\pi\) (même argument de contraction, de point fixe \(\mathbf{v}_\pi\)). À la limite, \(\mathbf{v}_\pi \le \mathbf{v}^*\).

**Exemple entièrement calculé.** Deux états, \(\gamma = 0{,}9\).

| État | Action | Récompense | État suivant |
|---|---|---|---|
| 1 | prudent | 1 | 1 |
| 1 | risqué | 0 | 1 ou 2, avec probabilité 1/2 chacun |
| 2 | attendre | 2 | 2 |
| 2 | encaisser | 4 | 1 |

Politique (risqué en 1, attendre en 2), par (Be1) :

1. État 2 : \(v(2) = 2 + 0{,}9\,v(2)\), donc \(v(2) = @RL_v2@\).
2. État 1 : \(v(1) = 0 + 0{,}9\,\big(\tfrac12 v(1) + \tfrac12 \times 20\big)\), donc \(0{,}55\,v(1) = 9\) et \(v(1) = \tfrac{@RL_v1_num@}{@RL_v1_den@} \approx @RL_v1:3m@\).
3. Est-elle optimale ? On vérifie (Be2) : en 1, prudent vaudrait \(1 + 0{,}9 \times \tfrac{@RL_v1_num@}{@RL_v1_den@} = \tfrac{@RL_q1_prudent_num@}{@RL_q1_prudent_den@} < \tfrac{@RL_q1_risque_num@}{@RL_q1_risque_den@}\) ; en 2, encaisser vaudrait \(4 + 0{,}9 \times \tfrac{@RL_v1_num@}{@RL_v1_den@} = \tfrac{@RL_q2_encaisser_num@}{@RL_q2_encaisser_den@} < 20\). Chaque maximum est atteint par l'action de la politique : \(\mathbf{v}^* = (\tfrac{@RL_v1_num@}{@RL_v1_den@} ; 20)\).

À comparer : la politique (prudent, attendre) donne \(v(1) = @RL_pp_v1@\) ; (prudent, encaisser) donne \(v(2) = @RL_pe_v2@\) ; (risqué, encaisser) donne \(v(1) = \tfrac{@RL_re_v1_num@}{@RL_re_v1_den@} \approx @RL_re_v1:2m@\). Le script résout les quatre systèmes en fractions et simule la politique optimale depuis l'état 1 : retour moyen @RL_v1_mc:3@, erreur standard @RL_v1_se:3@. Itération sur la valeur depuis \(\mathbf{v}_0 = (0 ; 0)\) : \(\mathbf{v}_1 = @RL_it1:0m@\), \(\mathbf{v}_2 = @RL_it2:2m@\), \(\mathbf{v}_3 = @RL_it3:4m@\) ; l'écart initial vaut \(@RL_ecart0@\), et il passe sous 0,01 après \(@RL_it_precision@\) itérations (la borne (Be3) en garantit au plus \(@RL_borne_precision@\)).

```python-exec
# mdp[état][action] = (récompense, {état suivant: probabilité})
mdp = {1: {"prudent": (1.0, {1: 1.0}), "risqué": (0.0, {1: 0.5, 2: 0.5})},
       2: {"attendre": (2.0, {2: 1.0}), "encaisser": (4.0, {1: 1.0})}}

def q_de(v, s, a, gamma):
    r, suivants = mdp[s][a]
    return r + gamma * sum(pr * v[t] for t, pr in suivants.items())

def iteration_valeur(gamma, nb=200):
    v = {1: 0.0, 2: 0.0}
    for k in range(1, nb + 1):
        v = {s: max(q_de(v, s, a, gamma) for a in mdp[s]) for s in mdp}
        if k in (1, 2, 3, 10, 50, nb):
            glouton = [max(mdp[s], key=lambda a: q_de(v, s, a, gamma)) for s in mdp]
            print(f"k = {k:3d} : v = ({v[1]:.4f} ; {v[2]:.4f}) ; politique gloutonne : {glouton[0]}, {glouton[1]}")
    return v

v = iteration_valeur(0.9)
print("180/11 =", round(180 / 11, 4))
print("avec gamma = 0,5 :")
v5 = iteration_valeur(0.5, 60)
```

**Cas limite : le facteur d'actualisation fait partie du problème.** Avec \(\gamma = 0{,}5\), la politique optimale devient (prudent, encaisser), de valeurs \((@RL_g5_v1@ ; @RL_g5_v2@)\) : risqué vaudrait \(\tfrac{@RL_g5_q1_risque_num@}{@RL_g5_q1_risque_den@}\) en 1 et attendre \(@RL_g5_q2_attendre:1m@\) en 2. Un \(\gamma\) petit rend l'agent myope (\(\gamma = 0\) : il maximise la récompense immédiate) ; un \(\gamma\) proche de 1 lui fait accepter une récompense nulle aujourd'hui pour un état meilleur demain.

**Pièges.** (Be1) est un système **linéaire** (une politique fixée) ; (Be2) contient un maximum et n'est pas linéaire. L'itération sur la valeur demande de connaître \(p\) et \(r\) : c'est de la planification, pas de l'apprentissage par essais.

### 4.3 Q-learning

**Le problème.** L'agent ne connaît ni \(p\) ni \(r\) : il observe seulement des transitions \((s, a, r, s')\). Peut-il quand même trouver \(q^*\), la fonction de valeur d'action de la politique optimale ?

**L'énoncé.** \(q^*(s, a) = r(s, a) + \gamma \sum_{s'} p(s' \mid s, a) \max_{a'} q^*(s', a')\) (c'est (Be2) écrit pour \(q^*\), puisque \(v^*(s') = \max_{a'} q^*(s', a')\)). Le **Q-learning** tient un tableau \(Q(s, a)\) et, après chaque transition observée \((s, a, r, s')\), fait

\(\displaystyle Q(s, a) \leftarrow Q(s, a) + \alpha\,\big[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\big],\)

où \(\alpha \in ]0, 1]\) est un pas. Le crochet est l'écart entre une **cible**, estimation à un tirage du membre de droite de l'équation, et la valeur courante.

- La cible utilise \(\max_{a'}\), quelle que soit l'action que l'agent fera réellement en \(s'\) : le Q-learning apprend \(q^*\) en suivant une **autre** politique (méthode **hors politique**, off-policy).
- **SARSA** remplace le maximum par \(Q(s', a')\) pour l'action \(a'\) réellement choisie : il apprend la valeur de la politique suivie (méthode **sur politique**, on-policy).
- Convergence (Watkins et Dayan, « Q-learning », Machine Learning, 1992 ; théorème admis) : avec un tableau, si chaque couple (état, action) est visité une infinité de fois et si les pas vérifient les conditions de Robbins et Monro (séance de ML 4, partie 1.3), \(Q\) converge vers \(q^*\) avec probabilité 1.

**Expérience.** Sur l'exemple à deux états, le comportement est entièrement aléatoire (action tirée à pile ou face). Les deux règles voient donc les mêmes données ; seule la cible change.

```python-exec
couples = [(1, "prudent"), (1, "risqué"), (2, "attendre"), (2, "encaisser")]

def apprendre(regle, gamma=0.9, pas=300000, graine=0):
    rng = np.random.default_rng(graine)
    hasard = rng.random((pas, 2))
    Q, N = dict.fromkeys(couples, 0.0), dict.fromkeys(couples, 0)
    s, a = 1, "prudent"
    for u1, u2 in hasard:
        r, suivants = mdp[s][a]
        etats = list(suivants)
        s2 = etats[0] if u1 < suivants[etats[0]] else etats[1]
        a2 = list(mdp[s2])[int(u2 < 0.5)]                  # comportement : action tirée à pile ou face, sans regarder Q
        if regle == "Q-learning":
            cible = r + gamma * max(Q[(s2, b)] for b in mdp[s2])
        else:                                              # SARSA : l'action réellement choisie ensuite
            cible = r + gamma * Q[(s2, a2)]
        N[(s, a)] += 1
        Q[(s, a)] += (cible - Q[(s, a)]) / N[(s, a)] ** 0.7     # pas décroissant, propre à chaque couple
        s, a = s2, a2
    return np.array([Q[c] for c in couples])

q_etoile = np.array([q_de(v, s, a, 0.9) for s, a in couples])    # valeurs exactes, par la partie 4.2
P, r = np.zeros((4, 4)), np.array([mdp[s][a][0] for s, a in couples])
for i, (s, a) in enumerate(couples):                       # politique uniforme : système linéaire (Be1) sur les couples
    for t, pr in mdp[s][a][1].items():
        for j, (s2, a2) in enumerate(couples):
            if s2 == t:
                P[i, j] = pr * 0.5
q_uniforme = np.linalg.solve(np.eye(4) - 0.9 * P, r)
print("couples (état, action)            :", couples)
print("q* exact                          :", q_etoile.round(3))
print("Q-learning                        :", apprendre("Q-learning").round(3))
print("q de la politique uniforme, exact :", q_uniforme.round(3))
print("SARSA                             :", apprendre("SARSA").round(3))
```

Lecture : avec le même comportement aléatoire, le Q-learning retrouve les valeurs **optimales** et SARSA celles de la politique **suivie**. Dans les deux tableaux, l'action de plus grande valeur est la même en chaque état sur cet exemple ; ce n'est pas une propriété générale.

**Pièges.** « Hors politique » ne veut pas dire « sans exploration » : il faut encore visiter tous les couples. Avec un réseau de neurones à la place du tableau, le théorème de convergence ne s'applique plus.

### 4.4 Gradient de politique et REINFORCE

**Le problème.** Au lieu d'apprendre des valeurs puis d'en déduire les actions, on peut paramétrer directement la politique, \(\pi_{\boldsymbol{\theta}}(a \mid s)\), et monter le gradient du retour espéré. Difficulté : le retour dépend de \(\boldsymbol{\theta}\) à travers la loi des trajectoires, que l'on ne connaît pas.

**Le cadre.** Épisodes de longueur fixe \(T\). Une trajectoire est \(\tau = (s_0, a_0, s_1, \dots, a_{T-1}, s_T)\), son retour \(g(\tau) = \sum_{t=0}^{T-1} \gamma^t\,r(s_t, a_t)\), et sa probabilité \(\mathbb{P}_{\boldsymbol{\theta}}(\tau) = \mu(s_0)\prod_{t=0}^{T-1} \pi_{\boldsymbol{\theta}}(a_t \mid s_t)\,p(s_{t+1} \mid s_t, a_t)\), où \(\mu\) est la loi de l'état initial. On suppose \(\pi_{\boldsymbol{\theta}}(a \mid s) > 0\) et dérivable en \(\boldsymbol{\theta}\) (par exemple un softmax). L'objectif est \(J(\boldsymbol{\theta}) = \mathbb{E}_{\boldsymbol{\theta}}[g(\tau)] = \sum_\tau \mathbb{P}_{\boldsymbol{\theta}}(\tau)\,g(\tau)\), somme finie. On note \(G_t = \sum_{i=t}^{T-1} \gamma^{i-t} R_{i+1}\) le retour à partir de \(t\).

**L'énoncé.**

1. (PG1) \(\displaystyle \nabla J(\boldsymbol{\theta}) = \mathbb{E}_{\boldsymbol{\theta}}\Big[g(\tau) \sum_{t=0}^{T-1} \nabla \ln \pi_{\boldsymbol{\theta}}(A_t \mid S_t)\Big]\) : ni \(p\) ni \(r\) ne sont dérivés.
2. (PG2) Pour toute fonction \(h\) du passé jusqu'à \(S_t\) inclus (états et actions avant \(t\), et \(S_t\)) : \(\mathbb{E}_{\boldsymbol{\theta}}\big[h\,\nabla \ln \pi_{\boldsymbol{\theta}}(A_t \mid S_t)\big] = \mathbf{0}\).
3. (PG3) En conséquence, pour toute **ligne de base** \(b(s)\) qui ne dépend pas de l'action : \(\displaystyle \nabla J(\boldsymbol{\theta}) = \mathbb{E}_{\boldsymbol{\theta}}\Big[\sum_{t=0}^{T-1} \gamma^t\,\big(G_t - b(S_t)\big)\,\nabla \ln \pi_{\boldsymbol{\theta}}(A_t \mid S_t)\Big]\).

**La preuve.**

1. Astuce du logarithme : pour une fonction strictement positive, \(\nabla \mathbb{P}_{\boldsymbol{\theta}}(\tau) = \mathbb{P}_{\boldsymbol{\theta}}(\tau)\,\nabla \ln \mathbb{P}_{\boldsymbol{\theta}}(\tau)\). Donc \(\nabla J = \sum_\tau \mathbb{P}_{\boldsymbol{\theta}}(\tau)\,g(\tau)\,\nabla \ln \mathbb{P}_{\boldsymbol{\theta}}(\tau)\), qui est une espérance.
2. \(\ln \mathbb{P}_{\boldsymbol{\theta}}(\tau) = \ln \mu(s_0) + \sum_t \ln \pi_{\boldsymbol{\theta}}(a_t \mid s_t) + \sum_t \ln p(s_{t+1} \mid s_t, a_t)\). Seul le terme du milieu dépend de \(\boldsymbol{\theta}\) : c'est (PG1). (Les trajectoires de probabilité nulle ne comptent pas dans la somme.)
3. (PG2) Sachant le passé et \(S_t = s\), \(A_t\) suit \(\pi_{\boldsymbol{\theta}}(\cdot \mid s)\) : \(\mathbb{E}[\nabla \ln \pi_{\boldsymbol{\theta}}(A_t \mid s)] = \sum_a \pi_{\boldsymbol{\theta}}(a \mid s)\,\dfrac{\nabla \pi_{\boldsymbol{\theta}}(a \mid s)}{\pi_{\boldsymbol{\theta}}(a \mid s)} = \nabla \sum_a \pi_{\boldsymbol{\theta}}(a \mid s) = \nabla 1 = \mathbf{0}\). On multiplie par \(h\), fixé sachant le passé, puis on prend l'espérance.
4. (PG3) Dans (PG1), on écrit \(g(\tau) = \sum_{i < t} \gamma^i R_{i+1} + \gamma^t G_t\) devant le terme d'indice \(t\). La première somme est une fonction du passé (les récompenses d'avant \(t\) ne dépendent que des états et actions d'avant \(t\)) : par (PG2), sa contribution est nulle. De même \(\gamma^t b(S_t)\) est une fonction du passé : on peut la retrancher.

**REINFORCE** (Williams, 1992) : on joue un épisode avec \(\pi_{\boldsymbol{\theta}}\), puis \(\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \sum_t \gamma^t (G_t - b(S_t))\,\nabla \ln \pi_{\boldsymbol{\theta}}(A_t \mid S_t)\) : une montée de gradient stochastique, dont l'estimateur est sans biais par (PG3). La méthode est sur politique (les épisodes doivent venir de la politique courante) et son estimateur a une forte variance, que la ligne de base réduit sans introduire de biais. Quand \(b\) est une estimation apprise de \(v_\pi\), on parle d'**acteur-critique**.

**Exemple entièrement calculé.** Un seul état, un seul pas (\(T = 1\)), deux actions ; \(\pi_\theta(1) = \mathrm{sig}(\theta)\), noté \(\pi\). L'action 1 rapporte 11, l'action 0 rapporte 10. Les dérivées sont \(\tfrac{d}{d\theta}\ln \pi = 1 - \pi\) et \(\tfrac{d}{d\theta}\ln(1 - \pi) = -\pi\). Pour \(\theta = 0\), \(\pi = \tfrac12\) :

1. \(J = 11\pi + 10(1 - \pi)\), de dérivée \(\pi(1 - \pi)(11 - 10) = \tfrac{@PG_esp_num@}{@PG_esp_den@}\).
2. Sans ligne de base, l'estimateur vaut \(11 \times \tfrac12 = @PG_g1:1m@\) avec probabilité \(\tfrac12\) et \(-10 \times \tfrac12 = @PG_g0@\) avec probabilité \(\tfrac12\) : espérance \(\tfrac{@PG_esp_num@}{@PG_esp_den@}\), variance \(@PG_var0:4m@\).
3. Avec \(b = 10\) : valeurs \(\tfrac12\) et 0, même espérance, variance \(@PG_var_10:4m@\). Avec \(b = 10{,}5\) : valeurs \(\tfrac14\) et \(\tfrac14\), variance \(@PG_var_b@\).

```python-exec
rng = np.random.default_rng(5)

def gradients(theta, b, nb=200000):
    pi = expit(theta)
    A = rng.random(nb) < pi                                # action 1 avec probabilité pi
    R = np.where(A, 11.0, 10.0)
    score = np.where(A, 1 - pi, -pi)                       # dérivée de ln pi(A) par rapport à theta
    return (R - b) * score

for b in [0.0, 10.0, 10.5]:
    g = gradients(0.0, b)
    print(f"ligne de base {b:4.1f} : moyenne {g.mean():.4f}, variance {g.var():.4f}")

def reinforce(b, alpha=0.05, pas=400, runs=2000):
    theta = np.zeros(runs)                                 # 2000 apprentissages indépendants, menés en parallèle
    for _ in range(pas):
        pi = expit(theta)
        A = rng.random(runs) < pi
        theta = theta + alpha * (np.where(A, 11.0, 10.0) - b) * np.where(A, 1 - pi, -pi)
    return expit(theta)

for b in [0.0, 10.5]:
    pi = reinforce(b)
    print(f"ligne de base {b:4.1f} : pi(1) après 400 pas, moyenne {pi.mean():.3f}, écart-type {pi.std():.3f}, part des apprentissages avec pi(1) < 0,5 : {np.mean(pi < 0.5):.3f}")
```

Lecture : la moyenne du gradient estimé ne dépend pas de la ligne de base, sa variance si. Sans ligne de base, une part notable des apprentissages est partie dans le mauvais sens après 400 pas.

**Pièges.** Le gradient de politique n'a besoin ni d'un modèle de l'environnement, ni d'une récompense dérivable : seule la politique est dérivée. Il converge au mieux vers un maximum local de \(J\). Il s'applique à des actions continues (politique gaussienne), là où le maximum sur les actions du Q-learning devient lui-même un problème d'optimisation.

### 4.5 Exploration contre exploitation

**Le problème.** Pour savoir qu'une action est bonne, il faut l'essayer ; chaque essai d'une action médiocre coûte. Le **bandit** à \(K\) bras isole ce dilemme : un seul état, \(K\) actions, la récompense du bras \(i\) est tirée d'une loi de moyenne \(\mu_i\) inconnue. Le **regret** après \(T\) tirages est \(T \max_i \mu_i\) moins la somme des moyennes des bras tirés.

**L'énoncé.**

1. (X1) **Stratégie gloutonne** (tirer chaque bras une fois, puis toujours celui de meilleure moyenne observée) sur deux bras de Bernoulli de moyennes 0,8 et 0,5 : avec probabilité au moins \(\tfrac{@BD_blocage_num@}{@BD_blocage_den@}\), elle ne tire plus jamais le meilleur bras. Son regret espéré croît donc linéairement : au moins \(@BD_regret_min:2m@\) après \(@BD_T@\) tirages.
2. (X2) **Epsilon-gloutonne** (avec probabilité \(\epsilon\), un bras tiré uniformément au hasard ; sinon le meilleur observé) : tous les bras sont tirés une infinité de fois, mais avec \(\epsilon\) constant le regret espéré par tirage reste au moins \(\tfrac{\epsilon}{K}\sum_i (\max_j \mu_j - \mu_i)\) : linéaire aussi.
3. (X3) **UCB1** (Auer, Cesa-Bianchi et Fischer, « Finite-time Analysis of the Multiarmed Bandit Problem », Machine Learning, 2002) : au tirage \(t\), choisir le bras qui maximise \(\bar x_i + \sqrt{2 \ln t / n_i}\), où \(\bar x_i\) est la moyenne observée du bras \(i\) et \(n_i\) son nombre de tirages (valeurs constatées au moment du choix). Pour des récompenses dans \([0, 1]\), son regret espéré croît au plus comme \(\ln T\) (théorème admis).

**La preuve.**

1. (X1) Si le premier tirage du bras de moyenne 0,8 donne 0 et celui du bras de moyenne 0,5 donne 1, événement de probabilité \(0{,}2 \times 0{,}5\), les moyennes observées sont 0 et 1. La stratégie tire le second bras ; sa moyenne observée reste strictement positive (il a au moins un succès), alors que celle du premier bras, qui n'est plus tiré, reste 0. Par récurrence, le premier bras n'est plus jamais tiré. Chaque tirage du second bras coûte \(0{,}8 - 0{,}5 = @BD_ecart:1m@\) de regret : au moins \(0{,}1 \times 0{,}3 \times (T - 2)\) en espérance.
2. (X2) À chaque tirage, avec probabilité \(\epsilon\) le bras est uniforme sur les \(K\) bras : le bras \(i\) est tiré avec probabilité au moins \(\epsilon/K\), et coûte alors \(\max_j \mu_j - \mu_i\).
3. Origine du terme d'UCB1 : l'inégalité de Hoeffding (théorème général, admis) borne par \(e^{-2nu^2}\) la probabilité qu'une moyenne de \(n\) variables indépendantes à valeurs dans \([0, 1]\) dépasse son espérance de \(u\). Avec \(u = \sqrt{2 \ln t / n}\), la borne vaut \(t^{-4}\) : l'indice est une borne supérieure de confiance de \(\mu_i\), d'autant plus large que le bras a été peu tiré. Exemple : \(t = 100\), \(n_i = 10\) donne un terme de \(@BD_bonus:3m@\).

**Expérience.** Les deux bras de (X1), 5000 tirages, 500 parties par stratégie, \(\epsilon = 0{,}1\). (X2) garantit ici un regret d'au moins \(@BD_eps_T:1m@\) pour la stratégie epsilon-gloutonne.

```python-exec
def bandit(strategie, T=5000, runs=500, eps=0.1, graine=0):
    rng = np.random.default_rng(graine)
    mu = np.array([0.8, 0.5])                              # probabilités de gain des deux bras
    S, N = np.zeros((runs, 2)), np.zeros((runs, 2))        # gains cumulés et nombres de tirages, par bras
    lignes, regret = np.arange(runs), np.zeros(runs)
    for t in range(T):
        if t < 2:
            bras = np.full(runs, t)                        # un tirage de chaque bras pour commencer
        elif strategie == "gloutonne":
            bras = np.argmax(S / N, axis=1)
        elif strategie == "epsilon-gloutonne":
            bras = np.where(rng.random(runs) < eps, rng.integers(0, 2, runs), np.argmax(S / N, axis=1))
        else:
            bras = np.argmax(S / N + np.sqrt(2 * np.log(t + 1) / N), axis=1)
        S[lignes, bras] += rng.random(runs) < mu[bras]
        N[lignes, bras] += 1
        regret += mu.max() - mu[bras]
    return regret.mean(), np.mean(N[:, 0] < N[:, 1])

for strategie in ["gloutonne", "epsilon-gloutonne", "UCB1"]:
    reg, rate = bandit(strategie)
    print(f"{strategie:18s}: regret moyen après 5000 tirages {reg:7.1f} ; part des parties où le moins bon bras est le plus tiré : {rate:.3f}")
```

Lecture : la stratégie gloutonne finit sur le moins bon bras dans une part des parties au moins égale à la borne de (X1), et son regret est le plus grand ; le regret de la stratégie epsilon-gloutonne dépasse la borne de (X2), qu'elle ne peut pas battre tant que \(\epsilon\) reste constant ; UCB1 a le plus petit regret des trois dans cette expérience.

**Cas limites.** \(\epsilon = 0\) : stratégie gloutonne. \(\epsilon = 1\) : exploration pure, aucune exploitation. Un \(\epsilon_t\) qui décroît vers 0 assez lentement pour que \(\sum_t \epsilon_t = \infty\) continue de tirer chaque bras une infinité de fois tout en exploitant de plus en plus.

### 4.6 Affirmations fausses fréquentes

1. « L'agent a besoin d'exemples étiquetés de la bonne action. » Faux : il n'a que des récompenses (4.1).
2. « L'agent maximise la récompense immédiate. » Faux : l'espérance du retour cumulé actualisé ; l'exemple à deux états choisit une récompense nulle en 1 (4.2).
3. « Un \(\gamma\) proche de 0 rend l'agent prévoyant. » Faux : c'est l'inverse (4.2, cas limite).
4. « Le Q-learning est une méthode sur politique. » Faux : sa cible prend le maximum, pas l'action suivie (4.3).
5. « Les méthodes de gradient de politique apprennent une fonction de valeur et choisissent l'action par un maximum. » Faux : elles optimisent directement les paramètres de la politique ; c'est le Q-learning qui passe par les valeurs. Un acteur-critique combine les deux (4.4).
6. « Le gradient de politique exige de connaître ou de dériver les transitions. » Faux : (PG1) ne dérive que la politique.
7. « Retrancher une ligne de base biaise le gradient. » Faux si elle ne dépend pas de l'action : (PG3).
8. « Une fois les valeurs estimées, explorer ne sert plus. » Faux : (X1), une estimation pessimiste d'un bon bras n'est jamais corrigée si on ne le tire plus.
9. « Une stratégie epsilon-gloutonne à \(\epsilon\) constant finit par ne plus rien perdre. » Faux : (X2).
10. « Récompense et retour sont la même chose. » Faux : le retour est la somme actualisée des récompenses futures.

[[controle:rl]]

## Partie 5 : réflexes pour les affirmations à cocher

### 5.1 Lire une affirmation

Dans une question à plusieurs réponses, chaque affirmation se juge seule. Quand la notation est tout ou rien, comme sur ce site, une seule erreur de lecture suffit à perdre la question. La méthode tient en quatre gestes.

1. **Repérer le quantificateur.** « Toujours », « jamais », « garantit », « nécessairement », « seulement » : l'affirmation est universelle, un seul contre-exemple la réfute. « Peut », « en espérance », « sous telle hypothèse » : un seul exemple, ou une hypothèse, suffit à la rendre vraie.
2. **Chercher le contre-exemple dans les cas limites** : variables corrélées ou identiques ; plus de variables que d'observations ; changement d'unité d'une variable ; classe rare ; pénalité nulle ou infinie ; interaction pure entre deux variables.
3. **Ne pas réfuter par réflexe.** Certaines affirmations universelles sont vraies, parce que ce sont des théorèmes : R² ne baisse jamais quand on ajoute une variable (séance de ML 1, partie 3) ; la solution Ridge est unique pour tout \(\lambda > 0\) (séance de ML 1, partie 4) ; un gain de Gini est toujours positif ou nul (séance de ML 2, partie 1) ; les directions principales sont toujours orthogonales (5.4).
4. **Séparer « vrai » et « vrai pour la raison donnée ».** « Le Lasso est préférable en grande dimension car il a une forme fermée » : conclusion défendable, raison fausse, affirmation fausse.

Les fiches qui suivent donnent, pour chaque thème, les faits exacts avec leurs conditions. Ce qui est démontré dans une autre séance est cité ; le reste est démontré ou montré ici.

### 5.2 Arbres de décision

**Ce que choisit un nœud** (T1). À chaque nœud, l'algorithme CART choisit le couple (variable, seuil) dont la coupure réduit le plus l'impureté **des observations de ce nœud** (séance de ML 2, partie 1) ; à la racine, ce sont toutes les observations d'apprentissage. Ce critère évalue chaque variable **seule**, pour **une** coupure : la variable de la racine est celle de plus grande réduction d'impureté en une coupure, ce qui n'est pas « la variable la plus importante » en un autre sens (importance par permutation, effet sur les prédictions de l'arbre entier).

Contre-exemple démontré. \(X_1\) et \(X_2\) sont deux variables binaires indépendantes et équiprobables, \(Y = 1\) si et seulement si \(X_1 \ne X_2\) (ou exclusif), et \(X_3\) est égale à \(Y\) avec probabilité 0,6, à \(1 - Y\) sinon.

1. Sachant \(X_1 = 0\) comme sachant \(X_1 = 1\), \(\mathbb{P}(Y = 1) = \tfrac12\) : les deux enfants ont les proportions du parent, le gain de Gini de \(X_1\) est \(@T_gain_x1@\) (de même pour \(X_2\)).
2. Sachant \(X_3\), la proportion de la classe majoritaire est 0,6 : Gini des enfants \(2 \times 0{,}6 \times 0{,}4 = \tfrac{@T_gini_x3_num@}{@T_gini_x3_den@}\), gain \(\tfrac12 - \tfrac{@T_gini_x3_num@}{@T_gini_x3_den@} = \tfrac{@T_gain_x3_num@}{@T_gain_x3_den@} > 0\).
3. La racine coupe donc selon \(X_3\) (en population ; le script retrouve ces gains sur 400000 tirages : @T_gain_x1_obs:4@ et @T_gain_x3_obs:4@). Pourtant \(X_1\) et \(X_2\) déterminent \(Y\) exactement, et \(X_3\) ne donne que 60 % d'exactitude.

```python-exec
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier

rng = np.random.default_rng(6)
n = 4000
x1, x2 = rng.integers(0, 2, n), rng.integers(0, 2, n)
y = x1 ^ x2                                                # ou exclusif : y vaut 1 si et seulement si x1 et x2 diffèrent
x3 = np.where(rng.random(n) < 0.6, y, 1 - y)               # x3 : égale à y avec probabilité 0,6
X = np.column_stack([x1, x2, x3])
app, val = slice(0, 2000), slice(2000, n)
arbre = DecisionTreeClassifier(random_state=0).fit(X[app], y[app])
print("variable testée à la racine : x" + str(arbre.tree_.feature[0] + 1), "; profondeur :", arbre.get_depth(), "; feuilles :", arbre.get_n_leaves())
print("importances par impureté                 :", arbre.feature_importances_.round(3))
perm = permutation_importance(arbre, X[val], y[val], n_repeats=10, random_state=0)
print("importances par permutation (validation) :", perm.importances_mean.round(3))
souche = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X[app], y[app])
print("exactitude en validation : arbre complet", arbre.score(X[val], y[val]), "; arbre de profondeur 1", souche.score(X[val], y[val]).round(3))
```

Lecture : la racine teste \(x_3\) ; l'importance par permutation place \(x_1\) et \(x_2\) à égalité, loin devant \(x_3\) ; l'importance par impureté, elle, attribue presque tout à **une seule** des deux variables de l'ou exclusif, celle qui est coupée en dernier, alors que leurs rôles sont symétriques. Trois « importances », trois réponses.

**Variables numériques, profondeur** (T2, T3).

- Une variable numérique est utilisée telle quelle : les coupures candidates sont des seuils placés entre deux valeurs observées consécutives, sans aucune mise en classes préalable, et la même variable peut resservir plus bas avec un autre seuil (séance de ML 2, partie 1, définitions). Ni standardisation ni transformation monotone ne changent l'arbre appris sur l'apprentissage (résultat R6 de la même partie).
- Un arbre binaire de profondeur \(D\) a au plus \(2^D\) feuilles (récurrence : chaque niveau double au plus le nombre de nœuds). Il utilise au plus \(2^D - 1\) coupures, donc pas forcément toutes les variables.
- Plus profond : biais plus faible, variance plus forte. Avec la même règle de choix et les égalités de gain départagées de la même façon, l'arbre de profondeur \(D + 1\) raffine celui de profondeur \(D\), et son impureté d'apprentissage pondérée ne peut pas être plus grande, chaque gain étant positif ou nul (résultats R2 à R4) ; l'erreur de validation, elle, n'est pas monotone (expérience de la séance de ML 2, partie 1).

```python-exec
xc = rng.normal(size=(300, 1))
yc = (xc[:, 0] > 0.37).astype(int)
a1 = DecisionTreeClassifier(random_state=0).fit(xc, yc)
tri = np.sort(xc[:, 0])
print("seuil appris :", a1.tree_.threshold[0].round(4), "; valeurs d'entraînement qui encadrent 0,37 :", tri[tri <= 0.37].max().round(4), "et", tri[tri > 0.37].min().round(4))
Xb, yb = rng.normal(size=(500, 3)), rng.integers(0, 2, 500)
for prof in [1, 2, 3, 4]:
    print("profondeur maximale", prof, ": feuilles", DecisionTreeClassifier(max_depth=prof, random_state=0).fit(Xb, yb).get_n_leaves(), ", au plus", 2 ** prof)
```

Fait de version : dans scikit-learn 1.1.1, `DecisionTreeClassifier` n'accepte que des nombres ; une variable qualitative doit être encodée avant.

```python-exec-erreur
DecisionTreeClassifier().fit(np.array([["rouge"], ["bleu"], ["rouge"]]), [0, 1, 0])
```

### 5.3 Importances de variables

Trois mesures, trois questions différentes. On les explique avant de les résumer.

**Par réduction d'impureté** : somme des gains des coupures faites sur la variable, pondérés par la part des observations qui atteignent le nœud. Elle est calculée sur l'apprentissage ; elle est biaisée en faveur des variables qui offrent beaucoup de coupures (continues, ou qualitatives à nombreuses modalités), et une variable de pur bruit reçoit une importance positive dès qu'elle sert à couper (séance de ML 2, partie 7, résultat I1) ; elle dépend de l'ordre des coupures (expérience ci-dessus).

**Par permutation** : baisse du score quand on mélange les valeurs de la variable sur des données de validation. Elle vaut pour tout modèle ; son espérance est nulle pour une variable indépendante du reste (résultat I2 de la même partie). Limites : entre variables corrélées, l'importance se dilue ou se déplace selon le modèle choisi, et la permutation fabrique des combinaisons de valeurs absentes des données (séance quantitative, partie 3, résultat P1).

**Par les coefficients d'un modèle linéaire.**

Énoncé (Co1). Dans une régression par moindres carrés avec constante, de matrice de rang plein, si l'on remplace la variable \(x_j\) par \(c\,x_j\) (\(c \ne 0\), changement d'unité), le coefficient \(\hat b_j\) devient \(\hat b_j/c\), les autres coefficients et les prédictions ne changent pas.

Preuve. Tout vecteur de coefficients \(\mathbf{b}\) pour les anciennes variables correspond au vecteur \(\mathbf{b}'\), égal à \(\mathbf{b}\) sauf \(b_j' = b_j/c\), pour les nouvelles, avec les mêmes valeurs ajustées (\(b_j x_j = (b_j/c)(c\,x_j)\)), donc la même somme des carrés des résidus. Les deux problèmes ont donc des minimiseurs qui se correspondent, et chacun est unique (rang plein : séance de ML 1, partie 1).

Conséquences : la valeur absolue d'un coefficient dépend de l'unité, donc ne classe pas des variables d'unités différentes. Le produit \(\hat b_j\) fois l'écart-type de \(x_j\) ne dépend pas de l'unité ; son carré, multiplié par 2, est l'importance par permutation d'un modèle linéaire (séance quantitative, résultat P1). Autres limites : entre variables corrélées, les coefficients sont instables (facteur d'inflation de la variance, séance de ML 1, partie 2) ; un coefficient mesure un effet « toutes autres variables du modèle fixées », et peut changer de signe quand on en ajoute une (biais de variable omise, même partie) ; avec une pénalité, changer l'unité change aussi la solution (séance de ML 1, partie 4, pièges).

```python-exec
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(7)
n = 500
distance_km = rng.uniform(1, 50, n)
pieces = rng.integers(1, 6, n).astype(float)
prix = 200 - 4.0 * distance_km + 30.0 * pieces + rng.normal(0, 10, n)
for nom, Xr in [("distance en km", np.column_stack([distance_km, pieces])), ("distance en m", np.column_stack([1000 * distance_km, pieces]))]:
    m = LinearRegression().fit(Xr, prix)
    print(f"{nom:15s}: coefficients {m.coef_.round(4)} ; coefficients fois écart-type de la variable {(m.coef_ * Xr.std(axis=0)).round(2)}")
```

Lecture : en valeur absolue, le coefficient brut de la distance est le plus petit des deux, et mille fois plus petit encore en mètres ; multiplié par l'écart-type, c'est la distance qui pèse le plus, dans les deux unités.

| Mesure | Données utilisées | Limite principale | Valable pour |
|---|---|---|---|
| réduction d'impureté | apprentissage | biais vers les variables à nombreuses valeurs ; positive même pour du bruit | arbres et ensembles d'arbres |
| permutation | validation | variables corrélées : dilution, combinaisons irréalistes | tout modèle |
| coefficient linéaire | apprentissage | dépend de l'unité ; instable entre variables corrélées | modèles linéaires |

### 5.4 Analyse en composantes principales

Les preuves sont dans la séance d'algèbre linéaire, sections 1, 5 et 9. Faits exacts, avec leur statut :

1. **Réduction de dimension** : garder les \(k\) premières composantes donne la meilleure approximation de rang \(k\) des données centrées, au sens des moindres carrés (Eckart-Young, section 5). Vrai sans condition.
2. **Orthogonalité** : les directions principales sont des vecteurs propres orthonormés d'une matrice symétrique ; les composantes (les nouvelles variables) sont non corrélées (section 9, point 2). Vrai sans condition. « Indépendantes » est faux en général : non corrélé n'implique indépendant que pour un vecteur gaussien.
3. **Combinaisons linéaires** : chaque composante est une combinaison linéaire de **toutes** les variables centrées, \(G_k = \mathbf{q}_k^\top(\mathbf{R} - \mathbf{m})\). L'ACP ne sélectionne pas de variables.
4. **Standardisation** : l'ACP n'est pas invariante par changement d'unité. Sur la covariance, une variable de grande variance domine ; standardiser revient à faire l'ACP de la matrice de corrélation (section 9, point 4). « Il faut toujours standardiser » est faux : si les variables ont la même unité et que leurs variances portent une information, la covariance est le bon objet. « Standardiser est sans effet » est faux aussi.
5. **Part de variance** : la part de la composante \(k\) est \(\lambda_k/\sum_i \lambda_i\). La première a la plus grande part, comprise entre \(1/p\) (pour \(p\) variables : plus grande valeur propre au moins égale à la moyenne) et 1. « Elle explique toujours la majorité de la variance » est faux : pour 10 variables non corrélées de même variance, sa part est \(@ACP_min10:1m@\).
6. **Non supervisée** : l'ACP n'utilise pas la cible. Les premières composantes ne sont pas forcément les plus utiles pour prédire.
7. **Linéaire** : elle ne voit que des directions ; le signe de chaque direction est arbitraire.

```python-exec
from sklearn.decomposition import PCA

rng = np.random.default_rng(8)
n = 2000
F = rng.normal(size=n)
X = np.column_stack([F + 0.5 * rng.normal(size=n), F + 0.5 * rng.normal(size=n), 0.1 * rng.normal(size=n), 20 * rng.normal(size=n)])
acp = PCA().fit(X)
print("données brutes, parts de variance :", acp.explained_variance_ratio_.round(4))
print("données brutes, première direction :", acp.components_[0].round(3))
Z = (X - X.mean(axis=0)) / X.std(axis=0)
acz = PCA().fit(Z)
print("données standardisées, parts de variance :", acz.explained_variance_ratio_.round(4))
print("données standardisées, première direction :", acz.components_[0].round(3))
G = acp.transform(X)                                       # valeurs des composantes, données brutes
print("directions orthonormées :", np.allclose(acp.components_.dot(acp.components_.T), np.eye(4)),
      "; composantes = données centrées fois directions :", np.allclose(G, (X - X.mean(axis=0)).dot(acp.components_.T)),
      "; plus grande corrélation entre deux composantes :", np.abs(np.corrcoef(G, rowvar=False) - np.eye(4)).max().round(6))
cible = X[:, 2] + 0.01 * rng.normal(size=n)                # une cible qui ne dépend que de la variable de plus petite variance
print("carré de la corrélation entre la cible et chaque composante :", [round(np.corrcoef(G[:, k], cible)[0, 1] ** 2, 3) for k in range(4)])
```

Lecture : sur les données brutes, la première composante est presque la quatrième variable, celle d'écart-type 20 ; standardisées, les données donnent une première direction portée par les deux variables corrélées. Les directions sont orthonormées et les composantes non corrélées dans tous les cas. La cible, qui ne dépend que de la variable de plus petite variance, est portée par la **dernière** composante : la garder ou non ne se décide pas sur la variance expliquée.

### 5.5 L1, L2, elastic net en grande dimension

**Le problème.** Beaucoup plus de variables que d'observations, et l'on veut un modèle lisible : peu de variables retenues. Quelle pénalité ?

**Les définitions.** Avec les notations de la séance de ML 1 (parties 4 et 5), l'**elastic net** (Zou et Hastie, « Regularization and variable selection via the elastic net », Journal of the Royal Statistical Society B, 2005) minimise, pour \(\lambda > 0\) et \(\alpha \in [0, 1]\) :

\(\displaystyle \tfrac12 \lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2 + \lambda\Big[\alpha\,\lVert \mathbf{b} \rVert_1 + \tfrac{1 - \alpha}{2}\,\lVert \mathbf{b} \rVert^2\Big].\)

\(\alpha = 1\) : Lasso ; \(\alpha = 0\) : Ridge.

**Ce qui est déjà démontré** (séance de ML 1). Ridge : solution unique quel que soit le rang, aucun coefficient exactement nul sauf cas de probabilité nulle, deux colonnes identiques reçoivent le même coefficient. Lasso : un coefficient ne peut être non nul que si \(\lvert g_j \rvert = \lambda\), où \(g_j = \mathbf{x}_j^\top(\mathbf{y} - \mathbf{X}\hat{\mathbf{b}})\) est le produit scalaire de la colonne \(j\) avec le résidu ; tous sont nuls dès que \(\lambda \ge \lambda_{\max}\) ; en position générale, au plus \(\min(n, p)\) coefficients non nuls (admis là-bas) ; avec deux colonnes identiques, une infinité de solutions, et entre colonnes très corrélées un choix instable.

**L'énoncé.**

1. (EN1) Pour \(\alpha < 1\), le critère de l'elastic net est strictement convexe : la solution est unique, même si \(p > n\), et deux colonnes identiques reçoivent le même coefficient.
2. (EN2) Pour \(0 < \alpha < 1\), le coefficient \(j\) de la solution est nul si et seulement si \(\lvert g_j \rvert \le \lambda\alpha\) : il y a des zéros exacts, comme pour le Lasso.
3. (EN3) Colonnes orthonormées, \(\mathbf{z} = \mathbf{X}^\top \mathbf{y}\) : \(\hat b_j = \dfrac{\mathrm{signe}(z_j)\max(\lvert z_j \rvert - \lambda\alpha, 0)}{1 + \lambda(1 - \alpha)}\) : un seuillage doux, puis une division.

**La preuve.**

1. (EN1) Le terme \(\tfrac{\lambda(1 - \alpha)}{2}\lVert \mathbf{b} \rVert^2\) est strictement convexe et les deux autres sont convexes : la somme est strictement convexe, donc a au plus un minimiseur ; elle en a un (fonction continue qui tend vers l'infini avec \(\lVert \mathbf{b} \rVert\), comme pour le Lasso). Si \(\mathbf{x}_1 = \mathbf{x}_2\), échanger \(b_1\) et \(b_2\) dans la solution ne change pas le critère : on obtient un minimiseur, égal à la solution par unicité, donc \(b_1 = b_2\).
2. (EN2) On écrit le critère comme un Lasso sur des données augmentées : avec \(\tilde{\mathbf{X}}\) formée de \(\mathbf{X}\) au-dessus de \(\sqrt{\lambda(1 - \alpha)}\,\mathbf{I}_p\) et \(\tilde{\mathbf{y}}\) formé de \(\mathbf{y}\) suivi de \(p\) zéros, \(\tfrac12\lVert \tilde{\mathbf{y}} - \tilde{\mathbf{X}}\mathbf{b} \rVert^2 = \tfrac12\lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2 + \tfrac{\lambda(1 - \alpha)}{2}\lVert \mathbf{b} \rVert^2\). Les conditions d'optimalité du Lasso (séance de ML 1, partie 5, énoncé 2) s'appliquent avec le seuil \(\lambda\alpha\) et \(\tilde g_j = g_j - \lambda(1 - \alpha)\hat b_j\), qui vaut \(g_j\) quand \(\hat b_j = 0\). Réciproquement, si \(\hat b_j \ne 0\) : \(\lvert g_j \rvert = \lambda\alpha + \lambda(1 - \alpha)\lvert \hat b_j \rvert > \lambda\alpha\).
3. (EN3) Comme dans le cas orthonormal du Lasso (même partie, preuve, étape 6), le critère se découple en \(\sum_j \big[\tfrac12(z_j - b_j)^2 + \lambda\alpha\lvert b_j \rvert + \tfrac{\lambda(1 - \alpha)}{2}b_j^2\big]\) plus une constante. Pour une coordonnée, les conditions d'optimalité donnent : \(b_j = 0\) si \(\lvert z_j \rvert \le \lambda\alpha\) ; sinon \(z_j - b_j - \lambda(1 - \alpha)b_j = \lambda\alpha\,\mathrm{signe}(b_j)\), soit \(b_j = (z_j - \lambda\alpha\,\mathrm{signe}(z_j))/(1 + \lambda(1 - \alpha))\).

**Exemple.** \(\mathbf{z} = @EN_z:1m@\), \(\lambda = 1\), \(\alpha = \tfrac12\) : seuil \(@EN_seuil:1m@\), diviseur \(@EN_div:1m@\), solution \(@EN_en:3m@\). Lasso (\(\alpha = 1\)) : \(@EN_lasso:1m@\). Ridge (\(\alpha = 0\)) : \(@EN_ridge:2m@\), aucun zéro. Le script retrouve ces trois solutions par minimisation numérique.

```python-exec
from sklearn.linear_model import ElasticNet, Lasso, Ridge

rng = np.random.default_rng(9)
n, p = 60, 300
X = rng.normal(size=(n, p))
facteurs = rng.normal(size=(n, 3))
for g in range(3):                                         # trois groupes de 5 variables presque identiques
    X[:, 5 * g:5 * g + 5] = facteurs[:, [g]] + 0.05 * rng.normal(size=(n, 5))
X = (X - X.mean(axis=0)) / X.std(axis=0)
y = X[:, :15].dot(np.r_[np.full(5, 1.0), np.full(5, -1.0), np.full(5, 0.5)]) + rng.normal(size=n)
y = y - y.mean()
modeles = {"Lasso": Lasso(alpha=0.3, max_iter=100000), "Ridge": Ridge(alpha=10.0), "Elastic net": ElasticNet(alpha=0.6, l1_ratio=0.5, max_iter=100000)}
for nom, m in modeles.items():
    c = m.fit(X, y).coef_
    print(f"{nom:12s}: {np.sum(c != 0):3d} coefficients non nuls sur {p} ; dans chaque groupe utile : {[int(np.sum(c[5 * g:5 * g + 5] != 0)) for g in range(3)]} ; ailleurs : {int(np.sum(c[15:] != 0))}")
x = rng.normal(size=n)
X2, y2 = np.column_stack([x, x]), 2 * x + 0.1 * rng.normal(size=n)
print("deux colonnes identiques : Lasso", Lasso(alpha=0.1, fit_intercept=False).fit(X2, y2).coef_.round(3),
      "; Ridge", Ridge(alpha=1.0, fit_intercept=False).fit(X2, y2).coef_.round(3),
      "; elastic net", ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=False, tol=1e-12, max_iter=100000).fit(X2, y2).coef_.round(3))
Qm, _ = np.linalg.qr(rng.normal(size=(n, 3)))              # trois colonnes orthonormées
z = np.array([3.0, -0.5, 1.2])
yo = Qm.dot(z)                                             # alors Qm^T yo = z
lam, alpha = 1.0, 0.5
en = ElasticNet(alpha=lam / n, l1_ratio=alpha, fit_intercept=False, tol=1e-12, max_iter=100000).fit(Qm, yo).coef_
print("colonnes orthonormées : scikit-learn", en.round(4) + 0.0, "; seuillage doux puis division", (np.sign(z) * np.maximum(np.abs(z) - lam * alpha, 0) / (1 + lam * (1 - alpha))).round(4) + 0.0)
```

Lecture : Ridge garde les 300 variables ; le Lasso en garde peu, mais seulement une partie de chaque groupe de variables presque identiques ; l'elastic net garde les groupes entiers, au prix de plus de variables retenues. Sur deux colonnes identiques : le Lasso de scikit-learn charge une seule colonne, Ridge et l'elastic net partagent. La dernière ligne établit la correspondance avec scikit-learn 1.1.1 : `ElasticNet(alpha=a, l1_ratio=u)` correspond à \(\lambda = n\,a\) et \(\alpha = u\) dans le critère de cette leçon.

**Réponse à la question.** Pour un modèle lisible en grande dimension : une pénalité qui annule des coefficients, donc L1, ou l'elastic net si des variables corrélées doivent être retenues ou écartées ensemble. L2 seule stabilise mais ne sélectionne rien. Conditions à rappeler : les variables sont standardisées et la constante non pénalisée (séance de ML 1) ; \(\lambda\) se choisit par validation ; la sélection suppose que peu de variables comptent vraiment, et l'ensemble retenu peut changer d'un échantillon à l'autre quand les variables sont corrélées.

### 5.6 Remèdes au surapprentissage

Le surapprentissage est une forte variance (partie 3) : erreur d'apprentissage faible, erreur de validation nettement plus élevée. Tout remède réduit la sensibilité du modèle à son échantillon :

- plus de données, ou des données augmentées par des transformations qui ne changent pas l'étiquette (pour des images : retournements, petites translations) ;
- une régularisation plus forte : pénalité L1 ou L2 (séance de ML 1), dropout et décroissance des poids (séance de ML 4), arbres moins profonds ou feuilles plus grandes (séance de ML 2) ;
- un modèle plus simple : moins de variables, moins de paramètres ;
- l'arrêt précoce : on arrête l'entraînement quand l'erreur de validation cesse de baisser (boosting : séance de ML 2, partie 4) ;
- la moyenne de modèles : bagging, forêt aléatoire (séance de ML 2, parties 2 et 3).

Ce qui l'aggrave dans le régime classique : ajouter des couches, des variables ou de la profondeur ; baisser \(\lambda\) ; entraîner plus longtemps sans surveiller la validation. Ce sont les remèdes du **sous**-apprentissage. Et ce qui n'y change rien : la stratification (elle agit sur l'évaluation, partie 2.6), le choix de la métrique.

[[controle:affirmations]]

## Réflexes en 30 secondes

| Indice dans l'énoncé | Premier geste |
|---|---|
| forme après une convolution ou un pooling | par axe : \(\lfloor (n + 2p - d(k - 1) - 1)/s \rfloor + 1\) ; « same » : \(\lceil n/s \rceil\) ; canaux = nombre de filtres ; pooling : canaux inchangés |
| nombre de paramètres d'une convolution | \(f(k^2 c + 1)\), indépendant de la taille de l'image ; pooling : 0 |
| concaténer, additionner | concaténer : mêmes dimensions spatiales, canaux additionnés ; additionner : même forme |
| champ réceptif avec des pas | \(r_l = r_{l-1} + d_l(k_l - 1)\,j_{l-1}\), \(j_l = j_{l-1}s_l\) |
| convolution transposée | \((n - 1)s - 2p + d(k - 1) + 1\) |
| pas par époque | \(\lceil n_{\text{app}}/B \rceil\), avec la taille d'**apprentissage** ; mises à jour = époques fois pas |
| pas d'apprentissage | échelle \(1/L\) ; défauts de bibliothèque entre 0,001 et 0,01 ; trop grand : la perte monte |
| perte et sortie | régression : quadratique, linéaire ; binaire : entropie croisée, sigmoïde ; \(K\) classes : entropie croisée, softmax ; multi-étiquette : \(K\) sigmoïdes ; segmentation : par pixel |
| Dice, IoU | \(\mathrm{IoU} = \mathrm{VP}/(\mathrm{VP} + \mathrm{FP} + \mathrm{FN})\) ; \(\mathrm{Dice} = 2\,\mathrm{IoU}/(1 + \mathrm{IoU}) = \mathrm{F1}\) |
| classe rare | stratifier le découpage ; pondérer : cote multipliée par \(w\) ; juger par précision, rappel, Dice ; rééquilibrer l'apprentissage seul |
| cible biais-variance | centre du nuage : biais ; dispersion : variance ; écart apprentissage-validation : variance |
| retour, valeur | \(G_t = R_{t+1} + \gamma G_{t+1}\) ; \(v = r + \gamma \sum p\,v\) ; optimalité : maximum sur les actions |
| Q-learning | cible \(r + \gamma \max_{a'} Q(s', a')\), hors politique ; SARSA : action suivie, sur politique |
| gradient de politique | \(\mathbb{E}[(G_t - b)\,\nabla \ln \pi]\) ; pas de modèle ; ligne de base : même espérance, moins de variance |
| exploration | gloutonne : blocage possible ; epsilon constant : regret linéaire ; UCB : \(\bar x_i + \sqrt{2\ln t/n_i}\) |
| « la racine de l'arbre » | meilleure réduction d'impureté en une coupure, une variable à la fois ; pas « la plus importante » |
| importance d'une variable | impureté : biais, apprentissage ; permutation : corrélations ; coefficient : unité |
| ACP | orthogonales, non corrélées ; combinaisons de toutes les variables ; dépend de l'échelle ; non supervisée |
| grande dimension, modèle lisible | L1 ; elastic net si groupes corrélés ; L2 ne sélectionne pas |
| surapprentissage | régulariser, simplifier, plus de données, arrêt précoce ; jamais plus de complexité |

Contrôles de vraisemblance en 5 secondes :

- une taille de sortie est un entier, au plus \(\lceil (n + 2p)/s \rceil\) ; après concaténation, le nombre de canaux est la somme ;
- \(0 \le \mathrm{IoU} \le \mathrm{Dice} \le 1\) ;
- une valeur d'état est comprise entre \(-r_{\max}/(1 - \gamma)\) et \(r_{\max}/(1 - \gamma)\), et \(v^* \ge v_\pi\) pour toute politique ;
- une part de variance expliquée est entre 0 et 1, et les parts somment à 1.

## À savoir refaire sans notes

- (C1) par le compte des poses ; le nombre de paramètres (C2) ; l'exemple à deux branches, concaténation puis pooling, et pourquoi « valid » l'empêche.
- (C4) par récurrence sur un exemple à deux poolings ; la taille (C5) d'une convolution transposée.
- (E1) par la division euclidienne ; (P1) par Taylor.
- Le gradient \(p_j - I_{\{j = y\}}\) du softmax ; le softmax à deux classes égal à une sigmoïde ; le lien entre Dice et IoU.
- (St1) par les indicatrices et leur covariance ; (W1) par la dérivée.
- La lecture des quatre cibles et le tableau des remèdes.
- (Be1) par conditionnement ; la contraction de (Be2) ; l'exemple à deux états résolu à la main.
- La mise à jour du Q-learning et ce qui la distingue de SARSA.
- (PG1) par l'astuce du logarithme ; (PG2) par \(\nabla \sum_a \pi = \mathbf{0}\) ; l'exemple à deux actions avec et sans ligne de base.
- Le blocage de la stratégie gloutonne (X1).
- Le contre-exemple de l'ou exclusif pour la racine d'un arbre ; (Co1) ; (EN1) et (EN3).
""",
    "controles": {
        "forme": {
            "question": r"Une carte de forme \(60 \times 60 \times 8\) entre dans deux branches. Branche A : convolution \(5 \times 5\), 12 filtres, remplissage 2, pas 2. Branche B : max-pooling \(2 \times 2\) de pas 2, sans remplissage. On concatène les deux sorties selon les canaux. Quelle est la forme du résultat ?",
            "options": [
                {"id": "a", "texte": r"\(@C_forme_n@ \times @C_forme_n@ \times @C_forme_c@\)",
                 "explication": r"Oui : A donne \(\lfloor (60 + 4 - 5)/2 \rfloor + 1 = @C_forme_n@\) et 12 canaux ; B donne \(\lfloor 60/2 \rfloor = @C_forme_n@\) et garde ses 8 canaux ; les canaux s'additionnent."},
                {"id": "b", "texte": r"\(@C_forme_n@ \times @C_forme_n@ \times 12\)",
                 "explication": r"Les 8 canaux de la branche B ont été oubliés : le pooling ne supprime pas les canaux, et la concaténation les ajoute aux 12 de la branche A."},
                {"id": "c", "texte": r"\(@C_forme_sans@ \times @C_forme_sans@ \times @C_forme_c@\)",
                 "explication": r"C'est la taille de A **sans** remplissage, \(\lfloor (60 - 5)/2 \rfloor + 1\). Avec cette taille, la concaténation avec B, de taille \(@C_forme_n@\), serait d'ailleurs impossible."},
                {"id": "d", "texte": r"\(@C_forme_n@ \times @C_forme_n@ \times @C_forme_prod@\)",
                 "explication": r"Les nombres de canaux ont été multipliés. Concaténer met les canaux bout à bout : \(12 + 8\), pas \(12 \times 8\)."},
            ],
        },
        "pas": {
            "question": r"Un jeu de 5000 images est découpé en 80 % d'apprentissage, 10 % de validation et 10 % de test. On entraîne avec des lots de 64 images, en gardant le dernier lot incomplet. Combien d'itérations compte une époque ?",
            "options": [
                {"id": "a", "texte": "@C_pas@",
                 "explication": r"Oui : \(@C_pas_app@ = @C_pas_plancher@ \times 64 + @C_pas_reste@\), donc \(@C_pas_plancher@\) lots complets et un lot de \(@C_pas_reste@\) images : \(\lceil @C_pas_app@/64 \rceil\)."},
                {"id": "b", "texte": "@C_pas_plancher@",
                 "explication": r"C'est la partie entière : elle écarte le dernier lot de \(@C_pas_reste@\) images, alors que l'énoncé dit qu'on le garde."},
                {"id": "c", "texte": "@C_pas_total@",
                 "explication": r"C'est \(\lceil 5000/64 \rceil\) : la taille totale a été utilisée, alors que seules les images d'apprentissage servent aux mises à jour."},
                {"id": "d", "texte": "@C_pas_app@",
                 "explication": r"C'est le nombre d'images d'apprentissage : il y aurait une itération par image seulement avec des lots de taille 1."},
            ],
        },
        "cible": {
            "question": "Un modèle a une erreur d'apprentissage très faible et une erreur de validation nettement plus élevée, les deux ensembles venant de la même loi. Quelle cible lui correspond, et quel remède ?",
            "options": [
                {"id": "a", "texte": "Impacts dispersés autour du centre ; régulariser davantage ou ajouter des données d'apprentissage.",
                 "explication": "Oui : le modèle suit le bruit de son échantillon, donc ses prédictions changent beaucoup d'un échantillon à l'autre (forte variance). Régularisation et données supplémentaires resserrent le nuage."},
                {"id": "b", "texte": "Impacts groupés loin du centre ; rendre le modèle plus complexe.",
                 "explication": "C'est le fort biais, qui se lit sur deux erreurs élevées et proches : le modèle n'ajusterait même pas ses données d'apprentissage. Ici l'erreur d'apprentissage est très faible."},
                {"id": "c", "texte": "Impacts dispersés autour du centre ; rendre le modèle plus complexe.",
                 "explication": "La cible est la bonne, le remède est à l'envers : plus de complexité augmente encore la variance dans le régime classique. C'est le remède du sous-apprentissage."},
                {"id": "d", "texte": "Impacts groupés au centre ; il n'y a rien à corriger.",
                 "explication": "Ce serait le cas si l'erreur de validation était proche de l'erreur d'apprentissage et du bruit irréductible. L'écart entre les deux erreurs signale au contraire une forte variance."},
            ],
        },
        "rl": {
            "question": r"Après avoir observé une transition (état \(s\), action \(a\), récompense \(r\), état suivant \(s'\)), vers quelle cible le Q-learning déplace-t-il \(Q(s, a)\) ?",
            "options": [
                {"id": "a", "texte": r"\(r + \gamma \max_{a'} Q(s', a')\)",
                 "explication": r"Oui : c'est une estimation à un tirage du membre de droite de l'équation de Bellman d'optimalité. Le maximum ne dépend pas de l'action que l'agent fera en \(s'\) : la méthode est hors politique."},
                {"id": "b", "texte": r"\(r + \gamma\,Q(s', a')\), où \(a'\) est l'action que l'agent choisit réellement en \(s'\)",
                 "explication": r"C'est la cible de SARSA, méthode sur politique : elle estime la valeur de la politique suivie, exploration comprise, et non la valeur optimale."},
                {"id": "c", "texte": r"Le retour complet \(G_t\) observé jusqu'à la fin de l'épisode",
                 "explication": r"C'est une cible de Monte-Carlo, celle qu'utilise REINFORCE. Le Q-learning n'attend pas la fin de l'épisode : il s'appuie sur sa propre estimation en \(s'\)."},
                {"id": "d", "texte": r"La récompense \(r\) seule",
                 "explication": r"Ce serait le cas \(\gamma = 0\), un agent myope. Pour \(\gamma > 0\), la cible doit contenir la valeur actualisée de l'état suivant."},
            ],
        },
        "affirmations": {
            "question": "Laquelle de ces affirmations est vraie sans condition supplémentaire ?",
            "options": [
                {"id": "a", "texte": "Les directions principales d'une ACP sont deux à deux orthogonales.",
                 "explication": "Oui : ce sont des vecteurs propres orthonormés d'une matrice symétrique (théorème spectral), que les données soient standardisées ou non."},
                {"id": "b", "texte": "La première composante principale explique toujours plus de la moitié de la variance totale.",
                 "explication": "Un contre-exemple suffit : pour des variables non corrélées de même variance, toutes les valeurs propres sont égales et la part de la première est un sur le nombre de variables."},
                {"id": "c", "texte": "La variable testée à la racine d'un arbre de décision est toujours celle dont l'importance par permutation est la plus grande.",
                 "explication": "La racine maximise la réduction d'impureté d'une seule coupure, une variable à la fois. Dans l'exemple de l'ou exclusif, elle teste la variable la moins utile au sens de la permutation."},
                {"id": "d", "texte": "Parmi des variables presque identiques, le Lasso retient toujours tout le groupe.",
                 "explication": "C'est le comportement de l'elastic net. Le Lasso n'en garde qu'une partie, et son choix peut changer d'un échantillon à l'autre ; avec des colonnes identiques, sa solution n'est même pas unique."},
            ],
        },
    },
    "cellules_initiales": [
        "import numpy as np\nnp.set_printoptions(precision=4, suppress=True)\n\n# Essaie ici les formules de la leçon, en les modifiant.\ndef taille(n, k, p=0, s=1, d=1):\n    return (n + 2 * p - d * (k - 1) - 1) // s + 1\n\nfor k, p, s in [(3, 0, 1), (3, 1, 1), (5, 2, 2), (2, 0, 2)]:\n    print(k, p, s, taille(96, k, p, s))",
    ],
    "verif": {"script": "verif/seance_ml_5.py"},
}
