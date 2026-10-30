"""Leçon d'algèbre linéaire de la séance d'algèbre 1 : spectre, semi-définie positivité, SVD, ACP et covariance.

Format : voir outils/lecons.py (élément de type "lecon", contenu en Markdown réduit, points de contrôle
[[controle:IDENT]] sur une ligne seule, option "a" correcte, une explication par option).
Les blocs ```python-exec sont exécutés à la construction par le noyau de l'environnement des notebooks
(numpy 1.21.6, scikit-learn 1.1.1), dans le dossier coding/lecons, et leur vraie sortie est insérée.
Aucun autre nombre calculé n'est écrit à la main : chaque jeton @CLE@, @CLE:3@ ou @CLE:3m@ est remplacé
par la valeur calculée par verif/seance_algebre_1.py (fichier verif/seance_algebre_1.out.json).
"""

LECON_ALGEBRE = {
    "id": "L-ALGEBRE-1", "type": "lecon", "section": "lecon",
    "theme": "Leçon algèbre linéaire : spectre, ACP et covariance",
    "difficulte": 4, "temps_cible_s": 3300, "dossier": "coding/lecons",
    "contenu": r"""## Leçon : spectre, semi-définie positivité, ACP et covariance

Durée visée : 55 minutes. La leçon répond notamment à deux questions du test de positionnement : la condition pour qu'une matrice d'équicorrélation \(3 \times 3\) soit semi-définie positive (DIAG-M4), et le rang d'une covariance empirique quand il y a moins de dates que d'actifs (DIAG-A6). Elle traite aussi l'ACP d'une matrice de rendements centrés et divisés par leur volatilité.

La leçon reconstruit tout à partir d'un seul outil, le **théorème spectral**, dans cet ordre : matrices symétriques, semi-définie positivité, trace, rang et projections, mineurs principaux, décomposition en valeurs singulières, conditionnement, gradients, covariance empirique, ACP, rétrécissement de Ledoit-Wolf. Chaque notion suit le même plan : le problème, l'intuition, l'énoncé, la preuve, l'exemple calculé, les cas limites, les pièges. Les exemples sont exécutés avec numpy 1.21.6 et scikit-learn 1.1.1, les versions de l'environnement des notebooks de la plateforme ; les autres nombres viennent du script `verif/seance_algebre_1.py`, qui vérifie aussi chaque énoncé général sur de nombreux cas.

**Conventions de notation**, valables pour toute la leçon :

- une majuscule désigne une variable aléatoire, une minuscule une valeur ou un paramètre fixe ; le gras désigne un vecteur ou une matrice ;
- un vecteur fixe est en minuscule grasse (\(\mathbf{x}\)), un vecteur aléatoire en majuscule grasse (\(\mathbf{R}\)) ; les vecteurs sont des colonnes ;
- une matrice est toujours en majuscule grasse (\(\mathbf{A}\)), par usage : le texte précise chaque fois qu'une matrice est aléatoire ;
- \(\mathbf{x}^\top\) est la transposée, \(\lVert \mathbf{x} \rVert = \sqrt{\mathbf{x}^\top \mathbf{x}}\) la norme euclidienne, \(\mathbf{I}_n\) (ou \(\mathbf{I}\)) la matrice identité \(n \times n\), \(\mathbf{1}\) le vecteur dont toutes les coordonnées valent 1, \(\mathbf{e}_j\) le vecteur dont la coordonnée \(j\) vaut 1 et les autres 0 ;
- \(\mathbb{E}\) désigne une espérance, \(\mathrm{Var}\) une variance, \(\mathrm{Cov}\) une covariance, \(\mathrm{tr}\) la trace (somme des coefficients diagonaux) et \(\det\) le déterminant.

## 1. Matrices symétriques : le théorème spectral

### Le problème

Une matrice de covariance est symétrique. Pour savoir si une matrice symétrique est une covariance possible, pour l'inverser sans risque, ou pour trouver les directions de plus grande variance d'un portefeuille, il faut la décrire dans une base où elle agit simplement. Le théorème spectral donne cette base.

### L'intuition

Une matrice symétrique \(\mathbf{A}\) est un étirement pur : il existe des axes perpendiculaires le long desquels \(\mathbf{A}\) multiplie simplement chaque vecteur par un nombre réel, sans le faire tourner. Ces axes sont les vecteurs propres, ces nombres les valeurs propres.

### L'énoncé

Soit \(n \ge 1\) un entier et \(\mathbf{A}\) une matrice réelle \(n \times n\) symétrique (\(\mathbf{A}^\top = \mathbf{A}\)). Un réel \(\lambda\) est une **valeur propre** de \(\mathbf{A}\), et un vecteur non nul \(\mathbf{q} \in \mathbb{R}^n\) un **vecteur propre** associé, si \(\mathbf{A}\mathbf{q} = \lambda\,\mathbf{q}\).

1. Toute valeur propre complexe de \(\mathbf{A}\) est réelle.
2. Deux vecteurs propres associés à deux valeurs propres différentes sont orthogonaux.
3. Théorème spectral : il existe une matrice \(\mathbf{Q}\) de taille \(n \times n\), **orthogonale** (\(\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}\) : ses colonnes \(\mathbf{q}_1, \dots, \mathbf{q}_n\) forment une base orthonormée), et des réels \(\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n\), tels que \(\mathbf{A}\mathbf{q}_k = \lambda_k \mathbf{q}_k\) pour chaque \(k\). Avec \(\boldsymbol{\Lambda}\) la matrice diagonale de coefficients \(\lambda_1, \dots, \lambda_n\) : \(\mathbf{A} = \mathbf{Q}\,\boldsymbol{\Lambda}\,\mathbf{Q}^\top = \sum_{k=1}^{n} \lambda_k\,\mathbf{q}_k \mathbf{q}_k^\top\).
4. Conséquences : \(\mathrm{tr}(\mathbf{A}) = \lambda_1 + \dots + \lambda_n\) et \(\det(\mathbf{A}) = \lambda_1 \lambda_2 \cdots \lambda_n\).
5. **Quotient de Rayleigh.** Pour \(\mathbf{x} \ne \mathbf{0}\), on pose \(r(\mathbf{x}) = \dfrac{\mathbf{x}^\top \mathbf{A}\,\mathbf{x}}{\mathbf{x}^\top \mathbf{x}}\). Alors \(\lambda_n \le r(\mathbf{x}) \le \lambda_1\) pour tout \(\mathbf{x} \ne \mathbf{0}\) ; le maximum \(\lambda_1\) est atteint en \(\mathbf{x} = \mathbf{q}_1\), et les vecteurs qui l'atteignent sont exactement les vecteurs propres non nuls associés à \(\lambda_1\). Autrement dit : \(\lambda_1 = \max_{\lVert \mathbf{x} \rVert = 1} \mathbf{x}^\top \mathbf{A}\,\mathbf{x}\).

### La preuve

On note \(\bar{z}\) le conjugué d'un nombre complexe \(z\), \(\bar{\mathbf{z}}\) le vecteur des conjugués, et on admet un seul résultat général d'algèbre : tout polynôme non constant à coefficients complexes a une racine complexe (théorème de d'Alembert-Gauss). Appliqué au polynôme \(\det(\mathbf{A} - z\,\mathbf{I})\), de degré \(n\), il donne une valeur propre complexe \(\lambda\) et un vecteur complexe non nul \(\mathbf{z}\) avec \(\mathbf{A}\mathbf{z} = \lambda\,\mathbf{z}\).

1. **Valeurs propres réelles.** Soit \(\mathbf{A}\mathbf{z} = \lambda\,\mathbf{z}\) avec \(\mathbf{z}\) complexe non nul. Le nombre \(s = \bar{\mathbf{z}}^\top \mathbf{A}\,\mathbf{z} = \sum_{i,j} a_{ij}\,\bar{z}_i z_j\), où \(a_{ij}\) est le coefficient \((i, j)\) de \(\mathbf{A}\), est réel : son conjugué vaut \(\sum_{i,j} a_{ij}\,z_i \bar{z}_j\), qui est la même somme après échange des noms \(i\) et \(j\), puisque \(a_{ij} = a_{ji}\) et que \(a_{ij}\) est réel. Or \(s = \lambda\,\bar{\mathbf{z}}^\top \mathbf{z} = \lambda \sum_i \lvert z_i \rvert^2\), avec \(\sum_i \lvert z_i \rvert^2 > 0\). Donc \(\lambda = s / \sum_i \lvert z_i \rvert^2\) est réel. Un vecteur propre réel s'en déduit : la partie réelle ou la partie imaginaire de \(\mathbf{z}\), l'une des deux étant non nulle, vérifie \(\mathbf{A}\mathbf{x} = \lambda\,\mathbf{x}\) car \(\mathbf{A}\) et \(\lambda\) sont réels.
2. **Orthogonalité.** Si \(\mathbf{A}\mathbf{u} = \alpha\,\mathbf{u}\) et \(\mathbf{A}\mathbf{v} = \beta\,\mathbf{v}\) avec \(\alpha \ne \beta\) : \(\alpha\,\mathbf{u}^\top \mathbf{v} = (\mathbf{A}\mathbf{u})^\top \mathbf{v} = \mathbf{u}^\top \mathbf{A}^\top \mathbf{v} = \mathbf{u}^\top \mathbf{A}\,\mathbf{v} = \beta\,\mathbf{u}^\top \mathbf{v}\), donc \((\alpha - \beta)\,\mathbf{u}^\top \mathbf{v} = 0\) et \(\mathbf{u}^\top \mathbf{v} = 0\).
3. **Base orthonormée, par récurrence sur \(n\).** Pour \(n = 1\), \(\mathbf{q}_1 = 1\) convient. Pour \(n \ge 2\), l'étape 1 donne une valeur propre réelle \(\lambda\) et un vecteur propre réel, que l'on divise par sa norme : \(\mathbf{q}\), unitaire. Soit \(W = \{\mathbf{x} : \mathbf{q}^\top \mathbf{x} = 0\}\), de dimension \(n - 1\). \(W\) est stable par \(\mathbf{A}\) : si \(\mathbf{x} \in W\), \(\mathbf{q}^\top \mathbf{A}\,\mathbf{x} = (\mathbf{A}\mathbf{q})^\top \mathbf{x} = \lambda\,\mathbf{q}^\top \mathbf{x} = 0\). Soit \(\mathbf{B}\) la matrice \(n \times (n-1)\) dont les colonnes forment une base orthonormée de \(W\) (\(\mathbf{B}^\top \mathbf{B} = \mathbf{I}_{n-1}\)). La matrice \(\mathbf{B}^\top \mathbf{A}\,\mathbf{B}\), de taille \(n - 1\), est symétrique ; par hypothèse de récurrence, elle a une base orthonormée de vecteurs propres \(\mathbf{y}_1, \dots, \mathbf{y}_{n-1}\). Si \(\mathbf{B}^\top \mathbf{A}\,\mathbf{B}\,\mathbf{y} = \mu\,\mathbf{y}\), alors \(\mathbf{A}\mathbf{B}\mathbf{y} \in W\) (stabilité), donc \(\mathbf{A}\mathbf{B}\mathbf{y} = \mathbf{B}\mathbf{z}\) pour un vecteur \(\mathbf{z}\) ; en multipliant à gauche par \(\mathbf{B}^\top\), \(\mathbf{z} = \mathbf{B}^\top \mathbf{A}\,\mathbf{B}\,\mathbf{y} = \mu\,\mathbf{y}\), d'où \(\mathbf{A}(\mathbf{B}\mathbf{y}) = \mu\,(\mathbf{B}\mathbf{y})\). Les vecteurs \(\mathbf{B}\mathbf{y}_1, \dots, \mathbf{B}\mathbf{y}_{n-1}\) sont donc des vecteurs propres de \(\mathbf{A}\), orthonormés (car \((\mathbf{B}\mathbf{y}_j)^\top \mathbf{B}\mathbf{y}_k = \mathbf{y}_j^\top \mathbf{y}_k\)) et dans \(W\) ; en y ajoutant \(\mathbf{q}\), on obtient une base orthonormée de \(\mathbb{R}^n\) formée de vecteurs propres de \(\mathbf{A}\). On range enfin les valeurs propres par ordre décroissant. L'égalité \(\mathbf{A}\mathbf{Q} = \mathbf{Q}\boldsymbol{\Lambda}\) (colonne par colonne) donne \(\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top\) en multipliant à droite par \(\mathbf{Q}^\top = \mathbf{Q}^{-1}\).
4. **Trace et déterminant.** Pour deux matrices \(\mathbf{M}\) (\(p \times q\)) et \(\mathbf{N}\) (\(q \times p\)) : \(\mathrm{tr}(\mathbf{M}\mathbf{N}) = \sum_{i}\sum_{j} m_{ij} n_{ji} = \mathrm{tr}(\mathbf{N}\mathbf{M})\). Donc \(\mathrm{tr}(\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top) = \mathrm{tr}(\boldsymbol{\Lambda}\mathbf{Q}^\top \mathbf{Q}) = \mathrm{tr}(\boldsymbol{\Lambda})\). Pour le déterminant, multiplicatif : \(\det(\mathbf{Q})^2 = \det(\mathbf{Q}^\top \mathbf{Q}) = 1\), donc \(\det(\mathbf{A}) = \det(\mathbf{Q})\det(\boldsymbol{\Lambda})\det(\mathbf{Q}^\top) = \det(\boldsymbol{\Lambda}) = \lambda_1 \cdots \lambda_n\).
5. **Rayleigh.** On décompose \(\mathbf{x} = \sum_k c_k \mathbf{q}_k\), avec \(c_k = \mathbf{q}_k^\top \mathbf{x}\). Par orthonormalité, \(\mathbf{x}^\top \mathbf{x} = \sum_k c_k^2\) et \(\mathbf{x}^\top \mathbf{A}\,\mathbf{x} = \sum_k \lambda_k c_k^2\). Comme \(\lambda_n \le \lambda_k \le \lambda_1\) pour chaque \(k\), \(\lambda_n \sum_k c_k^2 \le \sum_k \lambda_k c_k^2 \le \lambda_1 \sum_k c_k^2\) : c'est l'encadrement. Égalité à droite : \(\sum_k (\lambda_1 - \lambda_k)\,c_k^2 = 0\), somme de termes positifs ou nuls, donc \(c_k = 0\) dès que \(\lambda_k < \lambda_1\) : \(\mathbf{x}\) est une combinaison des \(\mathbf{q}_k\) de valeur propre \(\lambda_1\), c'est-à-dire un vecteur propre associé à \(\lambda_1\). Réciproquement, un tel vecteur donne \(r(\mathbf{x}) = \lambda_1\).

### Exemple calculé

```python-exec
import numpy as np
np.set_printoptions(precision=4, suppress=True)
A = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
lam, Q = np.linalg.eigh(A)          # valeurs propres croissantes, vecteurs propres en colonnes
print("valeurs propres :", lam)
print("Q^T Q = I :", np.allclose(Q.T.dot(Q), np.eye(3)), "; Q diag(lam) Q^T = A :", np.allclose(Q.dot(np.diag(lam)).dot(Q.T), A))
print("trace :", np.trace(A), "= somme :", lam.sum().round(10), "; det :", round(np.linalg.det(A), 10), "= produit :", lam.prod().round(10))
rng = np.random.default_rng(0)
x = rng.standard_normal((3, 100000))  # 100000 vecteurs tirés au hasard, en colonnes
rayleigh = np.einsum("ik,ij,jk->k", x, A, x) / (x ** 2).sum(axis=0)
print("Rayleigh sur 100000 vecteurs : min", rayleigh.min().round(4), "; max", rayleigh.max().round(4))
```

Les quotients de Rayleigh restent entre la plus petite et la plus grande valeur propre, et s'en approchent sans les dépasser. Attention : `np.linalg.eigh` range les valeurs propres par ordre **croissant**, l'inverse de la convention de l'énoncé.

### Cas limites et pièges

- Valeur propre multiple : les vecteurs propres d'une même valeur propre ne sont pas uniques (toute base orthonormée du sous-espace propre convient), et un vecteur propre n'est jamais défini qu'à son signe près. Deux logiciels peuvent donc rendre des \(\mathbf{Q}\) différents, tous justes.
- Matrice non symétrique : les valeurs propres peuvent être complexes (une rotation du plan d'un quart de tour n'a aucun vecteur propre réel), et les vecteurs propres ne sont pas orthogonaux en général. Rien de cette section ne s'applique.
- `np.linalg.eig` sur une matrice symétrique peut rendre des vecteurs non orthonormés quand une valeur propre est multiple ; `np.linalg.eigh` est fait pour les matrices symétriques.

[[controle:spectral]]

## 2. Semi-définie positivité, covariance et corrélation

### Le problème

On propose une matrice de corrélation, par exemple avec des corrélations négatives entre plusieurs actifs. Est-elle possible ? Une matrice symétrique à diagonale 1 et coefficients entre −1 et 1 ne l'est pas toujours : il faut en plus qu'aucun portefeuille n'ait une variance négative.

### L'énoncé

Une matrice symétrique réelle \(\mathbf{A}\) de taille \(n \times n\) est **semi-définie positive** (SDP) si \(\mathbf{x}^\top \mathbf{A}\,\mathbf{x} \ge 0\) pour tout \(\mathbf{x} \in \mathbb{R}^n\), et **définie positive** (DP) si \(\mathbf{x}^\top \mathbf{A}\,\mathbf{x} > 0\) pour tout \(\mathbf{x} \ne \mathbf{0}\).

1. Pour \(\mathbf{A}\) symétrique, les trois propriétés suivantes sont équivalentes : (a) \(\mathbf{A}\) est SDP ; (b) toutes ses valeurs propres sont positives ou nulles ; (c) il existe une matrice réelle \(\mathbf{B}\) (d'un nombre quelconque de lignes, \(n\) colonnes) telle que \(\mathbf{A} = \mathbf{B}^\top \mathbf{B}\). De même, \(\mathbf{A}\) est DP si et seulement si toutes ses valeurs propres sont strictement positives.
2. Soit \(\mathbf{R}\) un vecteur aléatoire de \(\mathbb{R}^n\) dont les coordonnées ont un moment d'ordre 2 fini, \(\mathbf{m} = \mathbb{E}[\mathbf{R}]\) sa moyenne et \(\boldsymbol{\Sigma} = \mathbb{E}[(\mathbf{R} - \mathbf{m})(\mathbf{R} - \mathbf{m})^\top]\) sa matrice de covariance (coefficient \((i, j)\) : \(\mathrm{Cov}(R_i, R_j)\)). Alors \(\boldsymbol{\Sigma}\) est SDP, et pour tout \(\mathbf{w} \in \mathbb{R}^n\), \(\mathrm{Var}(\mathbf{w}^\top \mathbf{R}) = \mathbf{w}^\top \boldsymbol{\Sigma}\,\mathbf{w}\).
3. Si de plus chaque variance \(\sigma_{ii}\) (coefficient diagonal de \(\boldsymbol{\Sigma}\)) est non nulle, la **matrice de corrélation** \(\mathbf{C}\), de coefficients \(c_{ij} = \sigma_{ij}/\sqrt{\sigma_{ii}\sigma_{jj}}\), est SDP et de diagonale 1. Réciproquement, toute matrice SDP de diagonale 1 est la matrice de corrélation d'un vecteur aléatoire.
4. **Équicorrélation.** Pour \(n \ge 2\) et un réel \(\rho\), soit \(\mathbf{C}_\rho = (1 - \rho)\,\mathbf{I} + \rho\,\mathbf{1}\mathbf{1}^\top\), la matrice \(n \times n\) de diagonale 1 dont tous les autres coefficients valent \(\rho\). Ses valeurs propres sont \(1 + (n - 1)\rho\), de vecteur propre \(\mathbf{1}\), et \(1 - \rho\), de multiplicité \(n - 1\) (vecteurs propres : tous les vecteurs de somme nulle). Donc \(\mathbf{C}_\rho\) est SDP si et seulement si \(-\dfrac{1}{n-1} \le \rho \le 1\).

### La preuve

1. **(a) implique (b)** : si \(\mathbf{A}\mathbf{q} = \lambda\,\mathbf{q}\) avec \(\lVert \mathbf{q} \rVert = 1\), alors \(0 \le \mathbf{q}^\top \mathbf{A}\,\mathbf{q} = \lambda\). **(b) implique (c)** : avec \(\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top\) (section 1), on pose \(\mathbf{B} = \boldsymbol{\Lambda}^{1/2}\mathbf{Q}^\top\), où \(\boldsymbol{\Lambda}^{1/2}\) est diagonale de coefficients \(\sqrt{\lambda_k}\), définis car \(\lambda_k \ge 0\) ; alors \(\mathbf{B}^\top \mathbf{B} = \mathbf{Q}\boldsymbol{\Lambda}^{1/2}\boldsymbol{\Lambda}^{1/2}\mathbf{Q}^\top = \mathbf{A}\). **(c) implique (a)** : \(\mathbf{x}^\top \mathbf{B}^\top \mathbf{B}\,\mathbf{x} = \lVert \mathbf{B}\mathbf{x} \rVert^2 \ge 0\). Cas DP : avec \(c_k = \mathbf{q}_k^\top \mathbf{x}\), \(\mathbf{x}^\top \mathbf{A}\,\mathbf{x} = \sum_k \lambda_k c_k^2\) (section 1, étape 5) ; si tous les \(\lambda_k > 0\) et \(\mathbf{x} \ne \mathbf{0}\), un \(c_k\) au moins est non nul et la somme est strictement positive ; la réciproque est l'argument de (a) implique (b) avec des inégalités strictes.
2. **Covariance.** \(\mathbf{w}^\top \mathbf{R} - \mathbf{w}^\top \mathbf{m} = \mathbf{w}^\top (\mathbf{R} - \mathbf{m})\), nombre réel égal à sa transposée \((\mathbf{R} - \mathbf{m})^\top \mathbf{w}\). Donc \(\mathrm{Var}(\mathbf{w}^\top \mathbf{R}) = \mathbb{E}[\mathbf{w}^\top (\mathbf{R} - \mathbf{m})(\mathbf{R} - \mathbf{m})^\top \mathbf{w}] = \mathbf{w}^\top \boldsymbol{\Sigma}\,\mathbf{w}\), par linéarité de l'espérance (\(\mathbf{w}\) est fixe). Une variance est positive ou nulle : \(\boldsymbol{\Sigma}\) est SDP.
3. **Corrélation.** Soit \(\mathbf{D}\) la matrice diagonale de coefficients \(1/\sqrt{\sigma_{ii}}\) : \(\mathbf{C} = \mathbf{D}\boldsymbol{\Sigma}\mathbf{D}\), et \(\mathbf{x}^\top \mathbf{C}\,\mathbf{x} = (\mathbf{D}\mathbf{x})^\top \boldsymbol{\Sigma}\,(\mathbf{D}\mathbf{x}) \ge 0\). Réciproque : si \(\mathbf{C}\) est SDP de diagonale 1, on écrit \(\mathbf{C} = \mathbf{B}^\top \mathbf{B}\) (point 1, \(\mathbf{B}\) de taille \(n \times n\)) ; soit \(\mathbf{G}\) un vecteur aléatoire de \(n\) variables indépendantes de loi normale centrée réduite, de covariance \(\mathbf{I}\), et \(\mathbf{R} = \mathbf{B}^\top \mathbf{G}\). Par linéarité, \(\mathrm{Cov}(\mathbf{R}) = \mathbb{E}[\mathbf{B}^\top \mathbf{G}\mathbf{G}^\top \mathbf{B}] = \mathbf{B}^\top \mathbf{I}\,\mathbf{B} = \mathbf{C}\) : les variances valent 1, donc les corrélations sont les covariances, c'est-à-dire les coefficients de \(\mathbf{C}\).
4. **Équicorrélation.** \(\mathbf{1}\mathbf{1}^\top \mathbf{1} = \mathbf{1}\,(\mathbf{1}^\top \mathbf{1}) = n\,\mathbf{1}\), donc \(\mathbf{C}_\rho \mathbf{1} = (1 - \rho + n\rho)\,\mathbf{1}\). Si \(\mathbf{1}^\top \mathbf{x} = 0\), alors \(\mathbf{1}\mathbf{1}^\top \mathbf{x} = \mathbf{0}\) et \(\mathbf{C}_\rho \mathbf{x} = (1 - \rho)\,\mathbf{x}\) ; les vecteurs de somme nulle forment un sous-espace de dimension \(n - 1\). On a ainsi \(n\) valeurs propres, comptées avec multiplicité, et le point 1 conclut : \(1 + (n - 1)\rho \ge 0\) et \(1 - \rho \ge 0\).

Seconde preuve de la borne inférieure, à retenir car elle se refait en 20 secondes. Si \(R_1, \dots, R_n\) ont toutes une variance 1 et des corrélations deux à deux égales à \(\rho\), alors \(\mathrm{Var}(R_1 + \dots + R_n) = \mathbf{1}^\top \mathbf{C}_\rho \mathbf{1} = n + n(n-1)\rho\) (les \(n\) variances, plus les \(n(n-1)\) covariances des couples ordonnés d'indices distincts). Cette variance doit être positive ou nulle : \(\rho \ge -1/(n-1)\).

### Exemple calculé

\(n = 5\) : la borne est \(\rho \ge \tfrac{@L_eq5_borne_num@}{@L_eq5_borne_den@}\). Avec \(\rho = -0{,}3\), la valeur propre simple vaut \(1 + 4\rho = @L_eq5_vp1:1m@ < 0\), la valeur propre multiple \(1 - \rho = @L_eq5_vp2:1m@\), et le portefeuille \(\mathbf{1}\) aurait une variance \(5 + 20\rho = @L_eq5_var@\) : impossible.

```python-exec
def equicorr(n, rho):
    return (1 - rho) * np.eye(n) + rho * np.ones((n, n))

for rho in [-0.3, -0.25, -0.2, 0.5]:
    print("n = 5, rho =", rho, ": valeurs propres", np.linalg.eigvalsh(equicorr(5, rho)).round(6) + 0.0)
```

### Cas limites

- \(\rho = -1/(n-1)\) : la matrice est SDP mais pas DP ; la somme des \(n\) variables est constante (variance nulle).
- \(\rho = 1\) : toutes les variables sont égales à une constante près ; valeur propre \(n\) une fois et 0 avec multiplicité \(n - 1\).
- Quand \(n\) grandit, la borne \(-1/(n-1)\) tend vers 0 : beaucoup de variables ne peuvent pas être toutes nettement anticorrélées deux à deux. Pour \(n = 2\), la borne est −1 : la seule condition est \(\lvert \rho \rvert \le 1\).

### Pièges

1. Croire que des coefficients diagonaux égaux à 1 et des corrélations entre −1 et 1 suffisent : c'est la condition pour \(n = 2\) seulement.
2. Retenir \(-1/n\) au lieu de \(-1/(n-1)\) : contrôle avec \(n = 2\), où la borne doit valoir −1.
3. Une matrice non symétrique : \(\mathbf{x}^\top \mathbf{A}\,\mathbf{x} = \mathbf{x}^\top \tfrac{\mathbf{A} + \mathbf{A}^\top}{2}\mathbf{x}\), car \(\mathbf{x}^\top \mathbf{A}\,\mathbf{x}\) est un nombre égal à sa transposée \(\mathbf{x}^\top \mathbf{A}^\top \mathbf{x}\) ; le signe de la forme ne dépend que de la partie symétrique, et les valeurs propres de \(\mathbf{A}\) elle-même ne disent rien.

[[controle:equi]]

## 3. Trace, rang, inverse et projections orthogonales

### Rang et inverse d'une matrice symétrique

Le **rang** d'une matrice est la dimension de son image (l'ensemble des \(\mathbf{A}\mathbf{x}\)) ; le **noyau** est l'ensemble des \(\mathbf{x}\) tels que \(\mathbf{A}\mathbf{x} = \mathbf{0}\). Pour une matrice à \(n\) colonnes, rang + dimension du noyau = \(n\) (théorème du rang, admis : résultat général d'algèbre linéaire).

1. Pour \(\mathbf{A}\) symétrique, le rang est le nombre de valeurs propres non nulles, comptées avec multiplicité. Preuve : \(\mathbf{A}\mathbf{x} = \sum_k \lambda_k c_k \mathbf{q}_k\) avec \(c_k = \mathbf{q}_k^\top \mathbf{x}\) ; l'image est donc contenue dans l'espace engendré par les \(\mathbf{q}_k\) tels que \(\lambda_k \ne 0\), et elle contient chacun d'eux (\(\mathbf{q}_k = \mathbf{A}(\mathbf{q}_k / \lambda_k)\)) ; ces vecteurs sont orthonormés, donc libres, et leur nombre est la dimension de l'image.
2. \(\mathbf{A}\) symétrique est inversible si et seulement si aucune valeur propre n'est nulle, et alors \(\mathbf{A}^{-1} = \mathbf{Q}\boldsymbol{\Lambda}^{-1}\mathbf{Q}^\top\) : on vérifie \(\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top \mathbf{Q}\boldsymbol{\Lambda}^{-1}\mathbf{Q}^\top = \mathbf{I}\). Les valeurs propres de l'inverse sont les \(1/\lambda_k\), avec les mêmes vecteurs propres.
3. Pour toute matrice réelle \(\mathbf{B}\) : \(\mathbf{B}^\top \mathbf{B}\) et \(\mathbf{B}\) ont le même noyau, donc le même rang. Preuve : si \(\mathbf{B}^\top \mathbf{B}\,\mathbf{v} = \mathbf{0}\), alors \(0 = \mathbf{v}^\top \mathbf{B}^\top \mathbf{B}\,\mathbf{v} = \lVert \mathbf{B}\mathbf{v} \rVert^2\), donc \(\mathbf{B}\mathbf{v} = \mathbf{0}\) ; la réciproque est immédiate. Le théorème du rang conclut (même nombre de colonnes).

### Projections orthogonales : le problème

Régresser un vecteur \(\mathbf{y}\) sur des variables, c'est chercher le point le plus proche de \(\mathbf{y}\) dans un sous-espace ; centrer des données, c'est retirer leur composante le long de \(\mathbf{1}\). Ces deux opérations sont des projections orthogonales, et leur trace donne un nombre de degrés de liberté.

### L'énoncé

Soit \(V\) un sous-espace de \(\mathbb{R}^n\) de dimension \(r \ge 1\), \(\mathbf{b}_1, \dots, \mathbf{b}_r\) une base orthonormée de \(V\), \(\mathbf{B}\) la matrice \(n \times r\) de colonnes \(\mathbf{b}_k\) (donc \(\mathbf{B}^\top \mathbf{B} = \mathbf{I}_r\)), et \(\mathbf{P} = \mathbf{B}\mathbf{B}^\top\).

1. Pour tout \(\mathbf{x}\), \(\mathbf{P}\mathbf{x} \in V\), \(\mathbf{x} - \mathbf{P}\mathbf{x}\) est orthogonal à \(V\), et \(\mathbf{P}\mathbf{x}\) est l'unique point de \(V\) le plus proche de \(\mathbf{x}\). \(\mathbf{P}\) est la **projection orthogonale** sur \(V\).
2. \(\mathbf{P}\) est symétrique et idempotente (\(\mathbf{P}^2 = \mathbf{P}\)).
3. Ses valeurs propres valent 0 ou 1 ; la valeur 1 a pour multiplicité \(r\). Donc \(\mathrm{tr}(\mathbf{P}) = \mathrm{rang}(\mathbf{P}) = r\).
4. Réciproquement, toute matrice symétrique et idempotente est la projection orthogonale sur son image.

### La preuve

1. \(\mathbf{P}\mathbf{x} = \sum_k (\mathbf{b}_k^\top \mathbf{x})\,\mathbf{b}_k \in V\), et \(\mathbf{B}^\top (\mathbf{x} - \mathbf{P}\mathbf{x}) = \mathbf{B}^\top \mathbf{x} - \mathbf{B}^\top \mathbf{B}\mathbf{B}^\top \mathbf{x} = \mathbf{0}\) : \(\mathbf{x} - \mathbf{P}\mathbf{x}\) est orthogonal à chaque \(\mathbf{b}_k\), donc à \(V\). Pour \(\mathbf{v} \in V\), \(\mathbf{P}\mathbf{x} - \mathbf{v} \in V\) est orthogonal à \(\mathbf{x} - \mathbf{P}\mathbf{x}\), et Pythagore donne \(\lVert \mathbf{x} - \mathbf{v} \rVert^2 = \lVert \mathbf{x} - \mathbf{P}\mathbf{x} \rVert^2 + \lVert \mathbf{P}\mathbf{x} - \mathbf{v} \rVert^2\), minimal si et seulement si \(\mathbf{v} = \mathbf{P}\mathbf{x}\).
2. \(\mathbf{P}^\top = (\mathbf{B}\mathbf{B}^\top)^\top = \mathbf{B}\mathbf{B}^\top\) et \(\mathbf{P}^2 = \mathbf{B}(\mathbf{B}^\top \mathbf{B})\mathbf{B}^\top = \mathbf{P}\).
3. Si \(\mathbf{P}\mathbf{v} = \lambda\,\mathbf{v}\) avec \(\mathbf{v} \ne \mathbf{0}\) : \(\lambda\,\mathbf{v} = \mathbf{P}\mathbf{v} = \mathbf{P}^2 \mathbf{v} = \lambda^2 \mathbf{v}\), donc \(\lambda^2 = \lambda\), \(\lambda \in \{0, 1\}\). Les vecteurs fixés par \(\mathbf{P}\) sont exactement ceux de \(V\) (si \(\mathbf{v} \in V\), \(\mathbf{v} = \sum_k c_k \mathbf{b}_k\) et \(\mathbf{P}\mathbf{v} = \mathbf{v}\) ; si \(\mathbf{P}\mathbf{v} = \mathbf{v}\), \(\mathbf{v} \in V\) par le point 1), donc la valeur propre 1 a pour sous-espace propre \(V\), de dimension \(r\). La trace est la somme des valeurs propres (section 1), le rang leur nombre non nul (ci-dessus) : les deux valent \(r\).
4. Si \(\mathbf{P}\) est symétrique et idempotente, pour tous \(\mathbf{x}, \mathbf{y}\) : \((\mathbf{P}\mathbf{y})^\top (\mathbf{x} - \mathbf{P}\mathbf{x}) = \mathbf{y}^\top (\mathbf{P}\mathbf{x} - \mathbf{P}^2 \mathbf{x}) = 0\). Donc \(\mathbf{x} - \mathbf{P}\mathbf{x}\) est orthogonal à l'image de \(\mathbf{P}\), et l'argument de Pythagore du point 1 montre que \(\mathbf{P}\mathbf{x}\) est le point de l'image le plus proche de \(\mathbf{x}\).

### Deux projections à connaître

- **Centrage.** \(\mathbf{u} = \mathbf{1}/\sqrt{T}\) est une base orthonormée de la droite engendrée par \(\mathbf{1} \in \mathbb{R}^T\) ; la projection sur cette droite est \(\mathbf{u}\mathbf{u}^\top = \mathbf{1}\mathbf{1}^\top / T\), qui remplace chaque coordonnée par la moyenne. La matrice de centrage \(\mathbf{M}_T = \mathbf{I}_T - \mathbf{1}\mathbf{1}^\top / T\) est symétrique et idempotente (on développe \(\mathbf{M}_T^2\) avec \(\mathbf{1}^\top \mathbf{1} = T\)) : c'est la projection sur l'orthogonal de \(\mathbf{1}\), de trace \(T - 1\), donc de rang \(T - 1\). Ce nombre reviendra en section 8.
- **Régression.** Soit \(\mathbf{A}\) une matrice \(T \times p\) de rang \(p\) (les \(p\) variables explicatives en colonnes, constante comprise). \(\mathbf{A}^\top \mathbf{A}\) est inversible (même rang \(p\) que \(\mathbf{A}\), point 3 ci-dessus), et la **matrice chapeau** \(\mathbf{H} = \mathbf{A}(\mathbf{A}^\top \mathbf{A})^{-1}\mathbf{A}^\top\) est symétrique et idempotente (vérification directe), d'image l'espace engendré par les colonnes de \(\mathbf{A}\) (\(\mathbf{H}\mathbf{A} = \mathbf{A}\), et toute image \(\mathbf{H}\mathbf{y}\) est de la forme \(\mathbf{A}\mathbf{z}\)). Sa trace : \(\mathrm{tr}(\mathbf{A}(\mathbf{A}^\top \mathbf{A})^{-1}\mathbf{A}^\top) = \mathrm{tr}((\mathbf{A}^\top \mathbf{A})^{-1}\mathbf{A}^\top \mathbf{A}) = \mathrm{tr}(\mathbf{I}_p) = p\). La matrice des résidus \(\mathbf{I} - \mathbf{H}\) a pour trace \(T - p\).

```python-exec
T, p = 8, 3
rng = np.random.default_rng(1)
Ap = np.column_stack([np.ones(T), rng.standard_normal((T, p - 1))])   # constante + 2 variables
H = Ap.dot(np.linalg.inv(Ap.T.dot(Ap))).dot(Ap.T)
print("H symétrique :", np.allclose(H, H.T), "; idempotente :", np.allclose(H.dot(H), H))
print("valeurs propres de H :", np.linalg.eigvalsh(H).round(6) + 0.0)
print("trace :", np.trace(H).round(6), "; rang :", np.linalg.matrix_rank(H))
Mc = np.eye(T) - np.ones((T, T)) / T
print("centrage : trace", np.trace(Mc).round(6), "; rang", np.linalg.matrix_rank(Mc))
```

### Pièges

1. Une matrice idempotente non symétrique est une projection **oblique** : ses valeurs propres valent aussi 0 ou 1, mais \(\mathbf{P}\mathbf{x}\) n'est pas le point le plus proche.
2. La trace d'une matrice quelconque n'est pas son rang ; l'égalité est propre aux projections.

[[controle:projection]]

## 4. Mineurs principaux : le bon critère, et le piège

### Le problème

Calculer les valeurs propres d'une matrice \(3 \times 3\) à la main est long. Des déterminants de sous-matrices permettent de conclure plus vite, à condition de choisir les bons.

### Définitions

Pour une matrice \(\mathbf{A}\) de taille \(n \times n\) et un ensemble non vide d'indices \(J \subset \{1, \dots, n\}\), la **sous-matrice principale** \(\mathbf{A}_J\) garde les lignes **et** les colonnes d'indices dans \(J\) ; son déterminant est un **mineur principal** (il y en a \(2^n - 1\)). Les **mineurs principaux dominants** sont ceux des \(J = \{1, \dots, k\}\), pour \(k = 1, \dots, n\) : les déterminants des coins supérieurs gauches.

### L'énoncé

Soit \(\mathbf{A}\) symétrique réelle \(n \times n\).

1. \(\mathbf{A}\) est SDP si et seulement si **tous** ses mineurs principaux sont positifs ou nuls.
2. Critère de Sylvester : \(\mathbf{A}\) est DP si et seulement si ses \(n\) mineurs principaux **dominants** sont strictement positifs.
3. Piège : des mineurs principaux dominants tous positifs ou nuls **ne suffisent pas** à rendre \(\mathbf{A}\) SDP.

### La preuve

1. **Lemme.** Si \(\mathbf{A}\) est SDP (resp. DP), chaque \(\mathbf{A}_J\) l'est : pour \(\mathbf{y} \in \mathbb{R}^{\lvert J \rvert}\), on complète \(\mathbf{y}\) par des zéros hors de \(J\) en un vecteur \(\mathbf{x}\), non nul si \(\mathbf{y}\) ne l'est pas, et \(\mathbf{y}^\top \mathbf{A}_J \mathbf{y} = \mathbf{x}^\top \mathbf{A}\,\mathbf{x}\). Son déterminant est alors le produit de ses valeurs propres (section 1), positives ou nulles (resp. strictement positives) par la section 2. D'où le sens direct de 1 et de 2.
2. **Développement de \(\det(\mathbf{A} + t\,\mathbf{I})\).** Soit \(t\) un réel et \(e_k\) la somme des mineurs principaux d'ordre \(k\) (\(e_0 = 1\)). Alors \(\det(\mathbf{A} + t\,\mathbf{I}) = \sum_{k=0}^{n} e_k\,t^{n-k}\). En effet, la colonne \(j\) de \(\mathbf{A} + t\,\mathbf{I}\) est \(\mathbf{a}_j + t\,\mathbf{e}_j\), où \(\mathbf{a}_j\) est la colonne \(j\) de \(\mathbf{A}\). Le déterminant est linéaire par rapport à chaque colonne : on le développe en \(2^n\) termes, un par ensemble \(K\) de colonnes où l'on choisit \(t\,\mathbf{e}_j\). Le terme de \(K\) vaut \(t^{\lvert K \rvert}\) fois le déterminant de la matrice dont les colonnes \(j \in K\) sont \(\mathbf{e}_j\) et les autres \(\mathbf{a}_j\) ; en développant par rapport à chaque colonne \(\mathbf{e}_j\), dont le seul 1 est sur la diagonale (signe \((-1)^{j+j} = +1\)), on supprime la ligne et la colonne \(j\) : il reste le mineur principal de \(\mathbf{A}\) sur le complémentaire \(J\) de \(K\). En regroupant selon \(k = \lvert J \rvert\), le coefficient de \(t^{n-k}\) est \(e_k\).
3. **Réciproque de 1.** Si tous les mineurs principaux sont positifs ou nuls, tous les \(e_k\) le sont, et pour \(t > 0\), \(\det(\mathbf{A} + t\,\mathbf{I}) \ge t^n > 0\). Si \(\mathbf{A}\) avait une valeur propre \(\lambda < 0\), de vecteur propre \(\mathbf{q}\), alors \((\mathbf{A} + t\,\mathbf{I})\mathbf{q} = \mathbf{0}\) pour \(t = -\lambda > 0\), et ce déterminant serait nul. Donc toutes les valeurs propres sont positives ou nulles, et \(\mathbf{A}\) est SDP (section 2).
4. **Réciproque de 2, par récurrence sur \(n\).** Pour \(n = 1\), \(\mathbf{A} = (a_{11})\) avec \(a_{11} > 0\). Pour \(n \ge 2\), le coin \(\mathbf{A}_{n-1}\) de taille \(n - 1\) a des mineurs dominants strictement positifs : il est DP par hypothèse de récurrence. Supposons que \(\mathbf{A}\) ait deux valeurs propres négatives ou nulles (comptées avec multiplicité), de vecteurs propres orthonormés \(\mathbf{q}\) et \(\mathbf{q}'\). Le plan qu'ils engendrent (dimension 2) et l'hyperplan \(\{x_n = 0\}\) (dimension \(n - 1\)) se coupent hors de \(\mathbf{0}\), car \(2 + (n - 1) > n\). Un vecteur non nul \(\mathbf{x} = c\,\mathbf{q} + c'\,\mathbf{q}'\) de cette intersection vérifie \(\mathbf{x}^\top \mathbf{A}\,\mathbf{x} = \lambda c^2 + \lambda' c'^2 \le 0\) ; mais \(\mathbf{x} = (\mathbf{y}, 0)\) avec \(\mathbf{y} \ne \mathbf{0}\), donc \(\mathbf{x}^\top \mathbf{A}\,\mathbf{x} = \mathbf{y}^\top \mathbf{A}_{n-1}\mathbf{y} > 0\). Contradiction : \(\mathbf{A}\) a au plus une valeur propre négative ou nulle. S'il y en avait exactement une, le produit des valeurs propres, \(\det(\mathbf{A})\), serait négatif ou nul, contre l'hypothèse. Toutes sont donc strictement positives : \(\mathbf{A}\) est DP.
5. **Pourquoi le piège.** Avec des inégalités larges, l'étape 4 casse : le coin \(\mathbf{A}_{n-1}\) n'est plus que SDP, et \(\mathbf{y}^\top \mathbf{A}_{n-1}\mathbf{y} \ge 0\) ne contredit plus \(\le 0\).

### Exemple calculé : le contre-exemple

Soit \(\mathbf{M}\) la matrice \(3 \times 3\) de lignes \((1, 1, 0)\), \((1, 1, 0)\), \((0, 0, -1)\). Ses mineurs principaux dominants valent \(@L_ce_dom@\), tous positifs ou nuls ; ses valeurs propres valent \(@L_ce_vp@\). Elle n'est pas SDP : le mineur principal d'indice \(J = \{3\}\), non dominant, vaut \(@L_ce_mineur33@\), et \(\mathbf{e}_3^\top \mathbf{M}\,\mathbf{e}_3 = @L_ce_mineur33@\).

```python-exec
M = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
print("mineurs dominants :", [round(float(np.linalg.det(M[:k, :k])), 10) + 0.0 for k in (1, 2, 3)])
print("valeurs propres :", np.linalg.eigvalsh(M).round(6) + 0.0)
e3 = np.array([0.0, 0.0, 1.0])
print("x^T M x pour x = e3 :", e3.dot(M).dot(e3))
```

Un contre-exemple encore plus petit, \(2 \times 2\) : la matrice diagonale de coefficients 0 et −1 a ses deux mineurs dominants nuls.

### Pièges

1. Appliquer Sylvester avec des inégalités larges (piège 3 de l'énoncé).
2. Oublier qu'un coefficient diagonal est un mineur principal d'ordre 1 : un coefficient diagonal strictement négatif interdit d'être SDP, en une seconde.
3. Pour une matrice \(2 \times 2\) symétrique, SDP équivaut à : les deux coefficients diagonaux et le déterminant sont positifs ou nuls (ce sont tous ses mineurs principaux).

[[controle:mineurs]]

## 5. Décomposition en valeurs singulières

### Le problème

Une matrice de données \(\mathbf{X}\) de taille \(T \times N\) (\(T\) dates, \(N\) actifs) n'est pas carrée : elle n'a pas de valeurs propres. On veut quand même ses directions principales, son rang, et sa meilleure approximation par une matrice de petit rang (quelques facteurs).

### L'énoncé

Soit \(\mathbf{X}\) une matrice réelle \(T \times N\) de rang \(r \ge 1\).

1. Il existe des réels \(s_1 \ge s_2 \ge \dots \ge s_r > 0\) (les **valeurs singulières**), des vecteurs orthonormés \(\mathbf{u}_1, \dots, \mathbf{u}_r\) de \(\mathbb{R}^T\) et des vecteurs orthonormés \(\mathbf{v}_1, \dots, \mathbf{v}_r\) de \(\mathbb{R}^N\) tels que \(\displaystyle \mathbf{X} = \sum_{k=1}^{r} s_k\,\mathbf{u}_k \mathbf{v}_k^\top\). Sous forme matricielle, \(\mathbf{X} = \mathbf{U}\mathbf{S}\mathbf{V}^\top\), avec \(\mathbf{U}\) (\(T \times r\)) et \(\mathbf{V}\) (\(N \times r\)) de colonnes \(\mathbf{u}_k\) et \(\mathbf{v}_k\), et \(\mathbf{S}\) diagonale de coefficients \(s_k\). Par convention, \(s_k = 0\) pour \(k > r\).
2. \(\mathbf{X}^\top \mathbf{X} = \mathbf{V}\mathbf{S}^2\mathbf{V}^\top\) et \(\mathbf{X}\mathbf{X}^\top = \mathbf{U}\mathbf{S}^2\mathbf{U}^\top\) : ces deux matrices ont les **mêmes valeurs propres non nulles**, \(s_1^2, \dots, s_r^2\) ; la première a en plus \(N - r\) valeurs propres nulles, la seconde \(T - r\). Le rang de \(\mathbf{X}\) est le nombre de valeurs singulières non nulles.
3. Norme d'opérateur : \(\max_{\lVert \mathbf{x} \rVert = 1} \lVert \mathbf{X}\mathbf{x} \rVert = s_1\). Norme de Frobenius : \(\lVert \mathbf{X} \rVert_F^2 = \sum_{t,i} x_{ti}^2 = \mathrm{tr}(\mathbf{X}^\top \mathbf{X}) = s_1^2 + \dots + s_r^2\).
4. **Eckart-Young.** Pour \(1 \le k < r\), soit \(\mathbf{X}_k = \sum_{j=1}^{k} s_j\,\mathbf{u}_j \mathbf{v}_j^\top\). Pour toute matrice \(\mathbf{B}\) de taille \(T \times N\) et de rang au plus \(k\) : \(\lVert \mathbf{X} - \mathbf{B} \rVert_F \ge \lVert \mathbf{X} - \mathbf{X}_k \rVert_F = \sqrt{s_{k+1}^2 + \dots + s_r^2}\), et de même en norme d'opérateur, avec la valeur \(s_{k+1}\). La SVD tronquée est la meilleure approximation de rang \(k\).

### La preuve

1. **Construction.** \(\mathbf{X}^\top \mathbf{X}\) est symétrique, SDP (forme \(\mathbf{B}^\top \mathbf{B}\), section 2), de rang \(r\) (section 3, point 3). Par le théorème spectral, elle a une base orthonormée de vecteurs propres \(\mathbf{v}_1, \dots, \mathbf{v}_N\), de valeurs propres \(\lambda_1 \ge \dots \ge \lambda_r > 0 = \lambda_{r+1} = \dots = \lambda_N\). On pose \(s_k = \sqrt{\lambda_k}\) et \(\mathbf{u}_k = \mathbf{X}\mathbf{v}_k / s_k\) pour \(k \le r\). Alors \(\mathbf{u}_j^\top \mathbf{u}_k = \mathbf{v}_j^\top \mathbf{X}^\top \mathbf{X}\,\mathbf{v}_k / (s_j s_k) = \lambda_k\,\mathbf{v}_j^\top \mathbf{v}_k / (s_j s_k)\), qui vaut 1 si \(j = k\) et 0 sinon : les \(\mathbf{u}_k\) sont orthonormés.
2. **Égalité.** Les deux membres \(\mathbf{X}\) et \(\sum_k s_k \mathbf{u}_k \mathbf{v}_k^\top\) sont des applications linéaires ; il suffit qu'ils coïncident sur la base des \(\mathbf{v}_j\). Pour \(j \le r\), le second membre envoie \(\mathbf{v}_j\) sur \(s_j \mathbf{u}_j = \mathbf{X}\mathbf{v}_j\). Pour \(j > r\), il l'envoie sur \(\mathbf{0}\), et \(\lVert \mathbf{X}\mathbf{v}_j \rVert^2 = \mathbf{v}_j^\top \mathbf{X}^\top \mathbf{X}\,\mathbf{v}_j = \lambda_j = 0\).
3. **Point 2.** \(\mathbf{X}^\top \mathbf{X} = \mathbf{V}\mathbf{S}\mathbf{U}^\top \mathbf{U}\mathbf{S}\mathbf{V}^\top = \mathbf{V}\mathbf{S}^2\mathbf{V}^\top\) car \(\mathbf{U}^\top \mathbf{U} = \mathbf{I}_r\) ; de même pour \(\mathbf{X}\mathbf{X}^\top\). Les \(\mathbf{v}_k\) (resp. \(\mathbf{u}_k\)) sont vecteurs propres de valeur propre \(s_k^2\), et l'orthogonal de leur espace engendré est le noyau.
4. **Point 3.** \(\lVert \mathbf{X}\mathbf{x} \rVert^2 = \mathbf{x}^\top \mathbf{X}^\top \mathbf{X}\,\mathbf{x}\), dont le maximum sur les vecteurs unitaires est la plus grande valeur propre \(s_1^2\) (Rayleigh, section 1). La norme de Frobenius : \(\sum_{t,i} x_{ti}^2\) est la trace de \(\mathbf{X}^\top \mathbf{X}\), somme de ses valeurs propres.
5. **Eckart-Young en norme d'opérateur.** On utilise la base orthonormée complète \(\mathbf{v}_1, \dots, \mathbf{v}_N\) de l'étape 1, avec \(s_j = 0\) pour \(j > r\) ; si \(k + 1 > N\), la borne cherchée vaut 0 et il n'y a rien à montrer. Soit \(\mathbf{B}\) de rang au plus \(k\) : son noyau est de dimension au moins \(N - k\). L'espace engendré par \(\mathbf{v}_1, \dots, \mathbf{v}_{k+1}\) est de dimension \(k + 1\) ; comme \((N - k) + (k + 1) > N\), les deux se coupent hors de \(\mathbf{0}\). Pour un vecteur unitaire \(\mathbf{w} = \sum_{j \le k+1} c_j \mathbf{v}_j\) de cette intersection : \(\lVert (\mathbf{X} - \mathbf{B})\mathbf{w} \rVert^2 = \lVert \mathbf{X}\mathbf{w} \rVert^2 = \sum_{j \le k+1} s_j^2 c_j^2 \ge s_{k+1}^2\). Donc la norme d'opérateur de \(\mathbf{X} - \mathbf{B}\) est au moins \(s_{k+1}\), valeur atteinte par \(\mathbf{X} - \mathbf{X}_k = \sum_{j > k} s_j \mathbf{u}_j \mathbf{v}_j^\top\), dont la plus grande valeur singulière est \(s_{k+1}\).
6. **Eckart-Young en norme de Frobenius.** On note \(s_i(\mathbf{M})\) la \(i\)-ième valeur singulière d'une matrice \(\mathbf{M}\). Montrons \(s_i(\mathbf{X} - \mathbf{B}) \ge s_{k+i}(\mathbf{X})\) pour tout \(i \ge 1\). Soit \(\mathbf{C}_i\) la SVD tronquée de \(\mathbf{X} - \mathbf{B}\) au rang \(i - 1\) : par l'étape 5 appliquée à \(\mathbf{X} - \mathbf{B}\), \(\lVert (\mathbf{X} - \mathbf{B}) - \mathbf{C}_i \rVert = s_i(\mathbf{X} - \mathbf{B})\) en norme d'opérateur. Or \(\mathbf{B} + \mathbf{C}_i\) est de rang au plus \(k + i - 1\) (l'image d'une somme est dans la somme des images), donc l'étape 5 appliquée à \(\mathbf{X}\) donne \(\lVert \mathbf{X} - (\mathbf{B} + \mathbf{C}_i) \rVert \ge s_{k+i}(\mathbf{X})\). En sommant les carrés (point 3) : \(\lVert \mathbf{X} - \mathbf{B} \rVert_F^2 = \sum_i s_i(\mathbf{X} - \mathbf{B})^2 \ge \sum_{i \ge 1} s_{k+i}(\mathbf{X})^2 = \lVert \mathbf{X} - \mathbf{X}_k \rVert_F^2\).

### Exemple calculé

```python-exec
rng = np.random.default_rng(2)
X = rng.standard_normal((6, 4)).dot(np.diag([3.0, 2.0, 1.0, 0.0]))   # 4e colonne nulle : rang 3
U, s, Vt = np.linalg.svd(X, full_matrices=False)
print("valeurs singulières :", s + 0.0)
print("carrés :", s ** 2 + 0.0)
print("valeurs propres de X^T X (4 x 4) :", np.clip(np.linalg.eigvalsh(X.T.dot(X)), 0, None)[::-1] + 0.0)
print("valeurs propres de X X^T (6 x 6) :", np.clip(np.linalg.eigvalsh(X.dot(X.T)), 0, None)[::-1] + 0.0)
print("rang :", np.linalg.matrix_rank(X))
X1 = s[0] * np.outer(U[:, 0], Vt[0])
print("erreur de rang 1 (Frobenius) :", np.linalg.norm(X - X1).round(6), "; racine de s2^2 + s3^2 + s4^2 :", np.sqrt((s[1:] ** 2).sum()).round(6))
print("erreur de rang 1 (opérateur) :", np.linalg.norm(X - X1, 2).round(6), "; s2 :", s[1].round(6))
```

### Pièges

1. Les valeurs propres de \(\mathbf{X}^\top \mathbf{X}\) sont les **carrés** des valeurs singulières, pas les valeurs singulières.
2. \(\mathbf{X}^\top \mathbf{X}\) (\(N \times N\)) et \(\mathbf{X}\mathbf{X}^\top\) (\(T \times T\)) n'ont pas la même taille : seules leurs valeurs propres **non nulles** coïncident.
3. L'erreur de Frobenius de la meilleure approximation de rang \(k\) est la **racine de la somme des carrés** des valeurs singulières omises, pas leur somme.

[[controle:svd]]

## 6. Conditionnement et sensibilité d'un système linéaire

### Le problème

Les poids de variance minimale ou de moyenne-variance s'obtiennent en résolvant \(\boldsymbol{\Sigma}\,\mathbf{w} = \mathbf{b}\). Si \(\mathbf{b}\) est connu avec une petite erreur relative, de combien l'erreur relative sur \(\mathbf{w}\) peut-elle être plus grande ?

### L'énoncé

Soit \(\mathbf{A}\) une matrice \(n \times n\) inversible, de valeurs singulières \(s_1 \ge \dots \ge s_n > 0\). Son **conditionnement** est \(\kappa(\mathbf{A}) = s_1 / s_n\) ; pour \(\mathbf{A}\) symétrique DP, les valeurs singulières sont les valeurs propres et \(\kappa(\mathbf{A}) = \lambda_{\max} / \lambda_{\min}\). Soient \(\mathbf{b} \ne \mathbf{0}\), \(\mathbf{x}\) la solution de \(\mathbf{A}\mathbf{x} = \mathbf{b}\), \(\boldsymbol{\delta}_b\) une perturbation du second membre et \(\boldsymbol{\delta}_x\) la perturbation induite : \(\mathbf{A}(\mathbf{x} + \boldsymbol{\delta}_x) = \mathbf{b} + \boldsymbol{\delta}_b\). Alors

\(\displaystyle \frac{\lVert \boldsymbol{\delta}_x \rVert}{\lVert \mathbf{x} \rVert} \le \kappa(\mathbf{A})\,\frac{\lVert \boldsymbol{\delta}_b \rVert}{\lVert \mathbf{b} \rVert},\)

et l'égalité est atteinte pour \(\mathbf{b}\) proportionnel à \(\mathbf{u}_1\) et \(\boldsymbol{\delta}_b\) proportionnel à \(\mathbf{u}_n\).

### La preuve

1. Ici \(r = n\), donc \(\mathbf{U}\) et \(\mathbf{V}\) sont carrées orthogonales et \(\mathbf{A}^{-1} = \mathbf{V}\mathbf{S}^{-1}\mathbf{U}^\top\) (vérification : \(\mathbf{U}\mathbf{S}\mathbf{V}^\top \mathbf{V}\mathbf{S}^{-1}\mathbf{U}^\top = \mathbf{I}\)). Les valeurs singulières de \(\mathbf{A}^{-1}\) sont les \(1/s_k\), et sa norme d'opérateur vaut \(1/s_n\) (section 5, point 3).
2. Par linéarité, \(\mathbf{A}\boldsymbol{\delta}_x = \boldsymbol{\delta}_b\), donc \(\lVert \boldsymbol{\delta}_x \rVert = \lVert \mathbf{A}^{-1}\boldsymbol{\delta}_b \rVert \le \lVert \boldsymbol{\delta}_b \rVert / s_n\).
3. \(\lVert \mathbf{b} \rVert = \lVert \mathbf{A}\mathbf{x} \rVert \le s_1 \lVert \mathbf{x} \rVert\), donc \(1/\lVert \mathbf{x} \rVert \le s_1 / \lVert \mathbf{b} \rVert\).
4. Le produit des inégalités des étapes 2 et 3, à membres positifs, donne l'énoncé. Égalité : \(\mathbf{b} = \mathbf{u}_1\) donne \(\mathbf{x} = \mathbf{v}_1/s_1\), égalité à l'étape 3 ; \(\boldsymbol{\delta}_b = \varepsilon\,\mathbf{u}_n\) donne \(\boldsymbol{\delta}_x = \varepsilon\,\mathbf{v}_n / s_n\), égalité à l'étape 2.

### Exemple calculé

```python-exec
A2 = np.array([[1.0, 1.0], [1.0, 1.0001]])
s2 = np.linalg.svd(A2, compute_uv=False)
print("valeurs singulières :", f"{s2[0]:.6f}", "et", f"{s2[1]:.3e}", "; conditionnement :", (s2[0] / s2[-1]).round(1), "=", np.linalg.cond(A2).round(1))
b = np.array([2.0, 2.0001])
db = np.array([0.0, 0.0001])
x0 = np.linalg.solve(A2, b)
x1 = np.linalg.solve(A2, b + db)
print("solution :", x0.round(6) + 0.0, "; solution perturbée :", x1.round(6) + 0.0)
eb = np.linalg.norm(db) / np.linalg.norm(b)
ex = np.linalg.norm(x1 - x0) / np.linalg.norm(x0)
print("erreur relative sur b :", f"{eb:.3e}", "; sur x :", f"{ex:.3e}", "; amplification :", round(ex / eb, 1))
```

L'amplification observée reste sous le conditionnement, comme le garantit l'énoncé ; elle ne l'atteint pas, car \(\mathbf{b}\) n'est pas exactement le long de \(\mathbf{u}_1\).

### Cas limites et pièges

- \(\kappa \ge 1\) toujours, et \(\kappa = 1\) pour une matrice orthogonale ou un multiple de \(\mathbf{I}\).
- Une matrice singulière a un conditionnement infini : c'est le cas de toute covariance empirique quand \(T \le N\) (section 8).
- Le conditionnement est le rapport des valeurs singulières **extrêmes** ; la plus petite seule, \(1/s_n\), ne borne que l'erreur **absolue** \(\lVert \boldsymbol{\delta}_x \rVert \le \lVert \boldsymbol{\delta}_b \rVert / s_n\).

## 7. Calcul matriciel : trois gradients

### Le problème

Moindres carrés, ridge, variance minimale : chaque fois on annule le gradient d'une fonction de \(\mathbf{x}\). Il faut les trois gradients de base sans hésiter, et savoir les redériver composante par composante.

### L'énoncé

Pour une fonction \(f\) de \(\mathbb{R}^n\) dans \(\mathbb{R}\), dérivable, le **gradient** \(\nabla f(\mathbf{x})\) est le vecteur colonne des dérivées partielles \(\partial f / \partial x_k\). Soient \(\mathbf{a} \in \mathbb{R}^n\), \(\mathbf{A}\) une matrice \(n \times n\) quelconque, \(\mathbf{M}\) une matrice \(m \times n\) et \(\mathbf{b} \in \mathbb{R}^m\).

1. \(\nabla (\mathbf{a}^\top \mathbf{x}) = \mathbf{a}\).
2. \(\nabla (\mathbf{x}^\top \mathbf{A}\,\mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\,\mathbf{x}\), qui vaut \(2\mathbf{A}\mathbf{x}\) si \(\mathbf{A}\) est symétrique.
3. \(\nabla \lVert \mathbf{M}\mathbf{x} - \mathbf{b} \rVert^2 = 2\,\mathbf{M}^\top (\mathbf{M}\mathbf{x} - \mathbf{b})\).
4. Conséquence : \(\mathbf{x}^\star\) minimise \(\lVert \mathbf{M}\mathbf{x} - \mathbf{b} \rVert^2\) si et seulement si \(\mathbf{M}^\top \mathbf{M}\,\mathbf{x}^\star = \mathbf{M}^\top \mathbf{b}\) (**équations normales**).

### La preuve

1. \(\mathbf{a}^\top \mathbf{x} = \sum_i a_i x_i\), de dérivée \(a_k\) par rapport à \(x_k\).
2. \(\mathbf{x}^\top \mathbf{A}\,\mathbf{x} = \sum_i \sum_j a_{ij} x_i x_j\). La dérivée par rapport à \(x_k\) reçoit les termes où \(i = k\), soit \(\sum_j a_{kj} x_j = (\mathbf{A}\mathbf{x})_k\), et ceux où \(j = k\), soit \(\sum_i a_{ik} x_i = (\mathbf{A}^\top \mathbf{x})_k\) ; le terme \(i = j = k\), \(a_{kk} x_k^2\), de dérivée \(2 a_{kk} x_k\), est bien compté une fois dans chaque somme.
3. \(\lVert \mathbf{M}\mathbf{x} - \mathbf{b} \rVert^2 = \mathbf{x}^\top \mathbf{M}^\top \mathbf{M}\,\mathbf{x} - 2\,\mathbf{b}^\top \mathbf{M}\,\mathbf{x} + \mathbf{b}^\top \mathbf{b}\) (les deux termes croisés sont des nombres égaux, transposés l'un de l'autre). Par 2 avec la matrice symétrique \(\mathbf{M}^\top \mathbf{M}\), et 1 avec \(\mathbf{a} = \mathbf{M}^\top \mathbf{b}\) : gradient \(2\mathbf{M}^\top \mathbf{M}\,\mathbf{x} - 2\mathbf{M}^\top \mathbf{b}\).
4. Si \(\mathbf{x}^\star\) vérifie les équations normales, pour tout \(\mathbf{x}\) on écrit \(\mathbf{M}\mathbf{x} - \mathbf{b} = \mathbf{M}(\mathbf{x} - \mathbf{x}^\star) + (\mathbf{M}\mathbf{x}^\star - \mathbf{b})\) ; le produit scalaire des deux morceaux vaut \((\mathbf{x} - \mathbf{x}^\star)^\top \mathbf{M}^\top (\mathbf{M}\mathbf{x}^\star - \mathbf{b}) = 0\), donc \(\lVert \mathbf{M}\mathbf{x} - \mathbf{b} \rVert^2 = \lVert \mathbf{M}(\mathbf{x} - \mathbf{x}^\star) \rVert^2 + \lVert \mathbf{M}\mathbf{x}^\star - \mathbf{b} \rVert^2 \ge \lVert \mathbf{M}\mathbf{x}^\star - \mathbf{b} \rVert^2\) : c'est un minimum, sans théorie de la convexité. Réciproquement, en un minimum d'une fonction dérivable sur \(\mathbb{R}^n\), le gradient est nul. Si \(\mathbf{M}\) est de rang \(n\), \(\mathbf{M}^\top \mathbf{M}\) est inversible (section 3) et \(\mathbf{x}^\star = (\mathbf{M}^\top \mathbf{M})^{-1}\mathbf{M}^\top \mathbf{b}\) ; alors \(\mathbf{M}\mathbf{x}^\star = \mathbf{H}\mathbf{b}\), la projection de la section 3.

### Exemple calculé : différences finies

```python-exec
rng = np.random.default_rng(3)
A3 = rng.standard_normal((3, 3))        # non symétrique
b3 = rng.standard_normal(3)
x3 = rng.standard_normal(3)
f = lambda z: z.dot(A3).dot(z)
g = lambda z: ((A3.dot(z) - b3) ** 2).sum()
h = 1e-6
df = np.array([(f(x3 + h * e) - f(x3 - h * e)) / (2 * h) for e in np.eye(3)])
dg = np.array([(g(x3 + h * e) - g(x3 - h * e)) / (2 * h) for e in np.eye(3)])
print("x^T A x    : différences finies", df, "; (A + A^T) x", (A3 + A3.T).dot(x3), "; 2 A x (faux)", 2 * A3.dot(x3))
print("|Ax - b|^2 : différences finies", dg, "; 2 A^T (Ax - b)", 2 * A3.T.dot(A3.dot(x3) - b3))
```

### Pièges

1. Écrire \(2\mathbf{A}\mathbf{x}\) pour une matrice non symétrique (dernière colonne de la première ligne ci-dessus).
2. Écrire \(2\mathbf{M}(\mathbf{M}\mathbf{x} - \mathbf{b})\) : contrôle des dimensions, le gradient doit être dans \(\mathbb{R}^n\), et \(\mathbf{M}\mathbf{x} - \mathbf{b}\) est dans \(\mathbb{R}^m\) ; seule \(\mathbf{M}^\top\) le ramène dans \(\mathbb{R}^n\).
3. Ridge : \(\nabla \big(\lVert \mathbf{M}\mathbf{x} - \mathbf{b} \rVert^2 + \alpha \lVert \mathbf{x} \rVert^2\big) = 2\mathbf{M}^\top (\mathbf{M}\mathbf{x} - \mathbf{b}) + 2\alpha\,\mathbf{x}\), pour un réel \(\alpha > 0\) ; d'où \((\mathbf{M}^\top \mathbf{M} + \alpha\,\mathbf{I})\,\mathbf{x}^\star = \mathbf{M}^\top \mathbf{b}\), système toujours inversible car les valeurs propres de \(\mathbf{M}^\top \mathbf{M} + \alpha\,\mathbf{I}\) sont celles de \(\mathbf{M}^\top \mathbf{M}\), positives ou nulles, augmentées de \(\alpha\).

## 8. Covariance empirique en grande dimension

### Le problème

Avec 500 actifs et un an de données quotidiennes (environ 250 dates), on veut estimer une covariance \(500 \times 500\), soit \(500 \times 501 / 2 = @L_coef_distincts@\) coefficients distincts, à partir de 250 vecteurs. L'estimateur empirique est alors singulier, et même quand il ne l'est pas, ses valeurs propres extrêmes sont faussées.

### Le cadre

Soient \(T \ge 2\) et \(N \ge 1\) des entiers, et \(\mathbf{X}_1, \dots, \mathbf{X}_T\) des vecteurs aléatoires de \(\mathbb{R}^N\) (les rendements des \(N\) actifs aux \(T\) dates). On note \(\mathbf{X}\) la matrice aléatoire \(T \times N\) de lignes \(\mathbf{X}_t^\top\), \(\bar{\mathbf{X}} = \tfrac{1}{T}\sum_t \mathbf{X}_t\) la moyenne, \(\mathbf{Y} = \mathbf{M}_T \mathbf{X}\) la matrice centrée, de lignes \((\mathbf{X}_t - \bar{\mathbf{X}})^\top\) (\(\mathbf{M}_T\) est la matrice de centrage de la section 3), et

\(\displaystyle \mathbf{S} = \frac{1}{T - 1}\sum_{t=1}^{T} (\mathbf{X}_t - \bar{\mathbf{X}})(\mathbf{X}_t - \bar{\mathbf{X}})^\top = \frac{\mathbf{Y}^\top \mathbf{Y}}{T - 1},\)

la covariance empirique, matrice aléatoire \(N \times N\).

**Absence de biais.** Si les \(\mathbf{X}_t\) sont indépendants, de même moyenne \(\mathbf{m}\) et de même covariance \(\boldsymbol{\Sigma}\), alors \(\mathbb{E}[\mathbf{S}] = \boldsymbol{\Sigma}\) ; c'est la raison du diviseur \(T - 1\). Preuve : avec \(\mathbf{L}_t = \mathbf{X}_t - \mathbf{m}\) et \(\bar{\mathbf{L}} = \bar{\mathbf{X}} - \mathbf{m}\), on développe \(\sum_t (\mathbf{L}_t - \bar{\mathbf{L}})(\mathbf{L}_t - \bar{\mathbf{L}})^\top = \sum_t \mathbf{L}_t \mathbf{L}_t^\top - T\,\bar{\mathbf{L}}\bar{\mathbf{L}}^\top\) (les termes croisés valent \(-T\,\bar{\mathbf{L}}\bar{\mathbf{L}}^\top\) chacun, car \(\sum_t \mathbf{L}_t = T\,\bar{\mathbf{L}}\)). Or \(\mathbb{E}[\mathbf{L}_t \mathbf{L}_t^\top] = \boldsymbol{\Sigma}\), et \(\mathbb{E}[\bar{\mathbf{L}}\bar{\mathbf{L}}^\top] = \tfrac{1}{T^2}\sum_{s,t} \mathbb{E}[\mathbf{L}_s \mathbf{L}_t^\top] = \boldsymbol{\Sigma}/T\), les termes \(s \ne t\) étant nuls par indépendance (espérance d'un produit de facteurs indépendants centrés). L'espérance de la somme vaut donc \(T\boldsymbol{\Sigma} - \boldsymbol{\Sigma} = (T - 1)\boldsymbol{\Sigma}\).

Une loi sur \(\mathbb{R}^N\) **a une densité** \(f\) si \(\mathbb{P}(\mathbf{Z} \in E) = \int_E f\) pour tout ensemble mesurable \(E\) ; elle ne charge alors aucun ensemble de mesure de Lebesgue nulle.

### L'énoncé

1. Toujours : \(\mathrm{rang}(\mathbf{S}) \le \min(N, T - 1)\).
2. Si \(\mathbf{X}_1, \dots, \mathbf{X}_T\) sont indépendants et ont chacun une densité, alors avec probabilité 1, \(\mathrm{rang}(\mathbf{S}) = \min(N, T - 1)\).
3. Conséquence : si \(T \le N\), \(\mathbf{S}\) a au moins \(N - T + 1\) valeurs propres nulles (exactement, avec probabilité 1, sous l'hypothèse de 2) ; elle n'est pas inversible, et tout calcul qui utilise \(\mathbf{S}^{-1}\) (variance minimale, distance de Mahalanobis, moindres carrés généralisés) est impossible tel quel.
4. Biais des valeurs propres extrêmes : si les \(\mathbf{X}_t\) sont indépendants, de même moyenne et de même covariance \(\boldsymbol{\Sigma}\), de valeurs propres extrêmes \(\lambda_{\max}(\boldsymbol{\Sigma})\) et \(\lambda_{\min}(\boldsymbol{\Sigma})\), alors \(\mathbb{E}[\lambda_{\max}(\mathbf{S})] \ge \lambda_{\max}(\boldsymbol{\Sigma})\) et \(\mathbb{E}[\lambda_{\min}(\mathbf{S})] \le \lambda_{\min}(\boldsymbol{\Sigma})\) : la plus grande valeur propre empirique est en moyenne trop grande, la plus petite trop petite.

### La preuve du rang

1. **Majoration.** \(\mathrm{rang}(\mathbf{S}) = \mathrm{rang}(\mathbf{Y})\) (section 3, point 3). L'image de \(\mathbf{Y} = \mathbf{M}_T \mathbf{X}\) est contenue dans celle de \(\mathbf{M}_T\), de dimension \(T - 1\) (section 3) ; et \(\mathbf{Y}\) a \(N\) colonnes. D'où \(\mathrm{rang}(\mathbf{S}) \le \min(N, T - 1)\).
2. **Réduction à des différences.** On pose \(\mathbf{D}_t = \mathbf{X}_t - \mathbf{X}_T\) pour \(t = 1, \dots, T - 1\), et \(\mathbf{D}_T = \mathbf{0}\). Le rang de \(\mathbf{Y}\) est la dimension de l'espace engendré par ses lignes, et cet espace est celui des \(\mathbf{D}_t\) : d'une part \(\mathbf{X}_t - \bar{\mathbf{X}} = \tfrac{1}{T}\sum_{s=1}^{T} (\mathbf{X}_t - \mathbf{X}_s) = \tfrac{1}{T}\sum_{s=1}^{T} (\mathbf{D}_t - \mathbf{D}_s)\) ; d'autre part \(\mathbf{D}_t = (\mathbf{X}_t - \bar{\mathbf{X}}) - (\mathbf{X}_T - \bar{\mathbf{X}})\).
3. **Un sous-espace strict est de mesure nulle.** Soit \(W\) un sous-espace de \(\mathbb{R}^N\) de dimension au plus \(N - 1\). Il est contenu dans un hyperplan \(\{\mathbf{z} : \mathbf{c}^\top \mathbf{z} = 0\}\) avec \(\mathbf{c}\) unitaire. Une matrice orthogonale qui envoie \(\mathbf{c}\) sur \(\mathbf{e}_1\) (on complète \(\mathbf{c}\) en base orthonormée) envoie cet hyperplan sur \(\{0\} \times \mathbb{R}^{N-1}\) et conserve la mesure de Lebesgue (son déterminant vaut ±1, formule du changement de variables, admise). Or \(\{0\} \times \mathbb{R}^{N-1}\) est de mesure nulle, par le théorème de Fubini (admis) : la mesure d'un point de \(\mathbb{R}\) est nulle. Donc si \(\mathbf{Z}\) a une densité, \(\mathbb{P}(\mathbf{Z} \in W) = 0\).
4. **Lemme.** Si \(k \le N\) vecteurs aléatoires \(\mathbf{Z}_1, \dots, \mathbf{Z}_k\) de \(\mathbb{R}^N\) sont indépendants et ont chacun une densité, ils sont linéairement indépendants avec probabilité 1. Preuve : la famille est liée seulement si l'un des \(\mathbf{Z}_j\) tombe dans l'espace \(W_{j-1}\) engendré par \(\mathbf{Z}_1, \dots, \mathbf{Z}_{j-1}\) (le premier \(j\) pour lequel la sous-famille \(\mathbf{Z}_1, \dots, \mathbf{Z}_j\) est liée convient). \(W_{j-1}\) est de dimension au plus \(j - 1 \le N - 1\). Comme \(\mathbf{Z}_j\) est indépendant de \(\mathbf{Z}_1, \dots, \mathbf{Z}_{j-1}\), on peut fixer ceux-ci : par Fubini, \(\mathbb{P}(\mathbf{Z}_j \in W_{j-1})\) est l'intégrale, contre la loi de \((\mathbf{Z}_1, \dots, \mathbf{Z}_{j-1})\), de la probabilité que \(\mathbf{Z}_j\) tombe dans un sous-espace strict fixé, nulle par l'étape 3. La réunion de ces \(k\) événements de probabilité nulle est de probabilité nulle.
5. **Conclusion.** Les \(\mathbf{D}_t\) ne sont pas indépendants : ils partagent \(\mathbf{X}_T\). On fixe donc \(\mathbf{X}_T = \mathbf{c}\) : comme \(\mathbf{X}_1, \dots, \mathbf{X}_{T-1}\) sont indépendants entre eux et de \(\mathbf{X}_T\), les vecteurs \(\mathbf{X}_t - \mathbf{c}\), \(t \le T - 1\), sont indépendants, et chacun a une densité (translatée de celle de \(\mathbf{X}_t\)). Si \(T - 1 \le N\), le lemme avec \(k = T - 1\) montre qu'ils sont libres avec probabilité 1 ; leur espace engendré est de dimension \(T - 1\). Si \(T - 1 > N\), le lemme appliqué à \(\mathbf{X}_1 - \mathbf{c}, \dots, \mathbf{X}_N - \mathbf{c}\) (\(k = N\)) montre que ces \(N\) vecteurs sont libres, donc engendrent \(\mathbb{R}^N\). Dans les deux cas, la dimension vaut \(\min(N, T - 1)\) avec probabilité conditionnelle 1, pour tout \(\mathbf{c}\), et en intégrant sur la loi de \(\mathbf{X}_T\) (Fubini), avec probabilité 1. L'étape 2 conclut.

### La preuve du biais des valeurs propres extrêmes

Soit \(\mathbf{q}\) un vecteur propre unitaire de \(\boldsymbol{\Sigma}\) associé à \(\lambda_{\max}(\boldsymbol{\Sigma})\), vecteur **fixe**. Pour chaque tirage, \(\lambda_{\max}(\mathbf{S}) \ge \mathbf{q}^\top \mathbf{S}\,\mathbf{q}\) (Rayleigh, section 1). On prend l'espérance, qui conserve les inégalités (\(\lambda_{\max}(\mathbf{S})\) est positive et majorée par \(\mathrm{tr}(\mathbf{S})\), d'espérance finie), et par linéarité et absence de biais \(\mathbb{E}[\mathbf{q}^\top \mathbf{S}\,\mathbf{q}] = \mathbf{q}^\top \mathbb{E}[\mathbf{S}]\,\mathbf{q} = \mathbf{q}^\top \boldsymbol{\Sigma}\,\mathbf{q} = \lambda_{\max}(\boldsymbol{\Sigma})\). Même argument pour le minimum avec l'inégalité inverse. La faute de raisonnement à éviter : prendre la plus grande valeur propre empirique pour un estimateur sans biais de la vraie, alors qu'on choisit la direction **après** avoir vu les données.

### Exemples calculés

Rang, pour des rendements gaussiens indépendants :

```python-exec
rng = np.random.default_rng(4)
for T, N in [(30, 50), (50, 50), (51, 50), (80, 50)]:
    Xr = rng.standard_normal((T, N))
    S = np.cov(Xr, rowvar=False)          # centrage et diviseur T - 1
    print("T =", T, ", N =", N, ": rang", np.linalg.matrix_rank(S), "; min(N, T - 1) =", min(N, T - 1))
```

Dispersion des valeurs propres empiriques, quand la vraie covariance est l'identité (toutes les vraies valeurs propres valent 1) :

```python-exec
N, T = 100, 200
Xr = rng.standard_normal((T, N))
vp = np.linalg.eigvalsh(np.cov(Xr, rowvar=False))
print("empiriques : plus petite", vp.min().round(3), "; plus grande", vp.max().round(3), "; moyenne", vp.mean().round(3))
c = N / T
print("bornes de Marchenko-Pastur :", round((1 - c ** 0.5) ** 2, 3), "et", round((1 + c ** 0.5) ** 2, 3))
```

Lecture : la moyenne des valeurs propres empiriques est juste (c'est \(\mathrm{tr}(\mathbf{S})/N\), d'espérance \(\mathrm{tr}(\boldsymbol{\Sigma})/N = 1\)), mais elles s'étalent largement autour de 1. Les bornes affichées viennent d'un résultat asymptotique **admis**, cité pour situer l'expérience et non utilisé ailleurs (Marčenko et Pastur, 1967, « Distribution of eigenvalues for some sets of random matrices », Mathematics of the USSR-Sbornik) : sous des hypothèses précises (coordonnées indépendantes, centrées, de variance 1, et \(N/T\) qui tend vers une constante \(c \in \,]0, 1]\)), la proportion de valeurs propres de \(\mathbf{S}\) dans tout intervalle converge vers celle d'une loi de support \([(1 - \sqrt{c})^2, (1 + \sqrt{c})^2]\). Le script de vérification confirme le biais moyen démontré ci-dessus sur 2000 échantillons (\(N = 20\), \(T = 40\), \(\boldsymbol{\Sigma} = \mathbf{I}\)) : moyenne de la plus grande valeur propre \(@L_disp_lmax_moy:3m@\), de la plus petite \(@L_disp_lmin_moy:3m@\).

### Cas limites et pièges

1. Oublier le centrage : sans soustraction de la moyenne (moyenne connue, supposée nulle), \(\tfrac{1}{T}\sum_t \mathbf{X}_t \mathbf{X}_t^\top\) est de rang \(\min(N, T)\), et non \(\min(N, T - 1)\) ; la preuve est la même, avec les \(\mathbf{X}_t\) eux-mêmes à la place des \(\mathbf{D}_t\).
2. L'hypothèse de densité compte : des rendements à valeurs discrètes (prix sur une grille de tick, rendements exactement nuls les jours sans cotation) peuvent faire tomber le rang. Exemple du script : \(T = 4\), \(N = 3\), coordonnées indépendantes uniformes sur \(\{-1, 0, 1\}\) ; le rang est inférieur à 3 dans une proportion \(@L_rang_discret_part:3m@\) des tirages.
3. \(T > N\) ne suffit pas à avoir une estimation fiable : \(\mathbf{S}\) est alors inversible avec probabilité 1, mais dans l'exemple ci-dessus (\(N/T = 1/2\)), son conditionnement est le rapport des valeurs propres extrêmes affichées, alors que celui de \(\boldsymbol{\Sigma} = \mathbf{I}\) vaut 1.
4. Une colonne constante, ou un actif combinaison linéaire d'autres (un indice et ses composants), fait aussi chuter le rang : l'hypothèse de densité est alors fausse.

## 9. Analyse en composantes principales

### Le problème

Les rendements de \(N\) actifs sont corrélés. On cherche quelques portefeuilles qui résument la plus grande part possible de leur variabilité totale, sans redondance entre eux.

### L'intuition

Le portefeuille de poids \(\mathbf{w}\) (vecteur unitaire) a pour variance \(\mathbf{w}^\top \boldsymbol{\Sigma}\,\mathbf{w}\) : c'est un quotient de Rayleigh. La direction de variance maximale est donc le premier vecteur propre de \(\boldsymbol{\Sigma}\), et la suivante, cherchée orthogonale à la première, est le deuxième vecteur propre.

### L'énoncé

Soit \(\mathbf{R}\) un vecteur aléatoire de \(\mathbb{R}^N\) de moyenne \(\mathbf{m}\) et de covariance \(\boldsymbol{\Sigma}\), avec \(\boldsymbol{\Sigma} = \sum_k \lambda_k \mathbf{q}_k \mathbf{q}_k^\top\), \(\lambda_1 \ge \dots \ge \lambda_N \ge 0\) (section 1).

1. \(\max_{\lVert \mathbf{w} \rVert = 1} \mathrm{Var}(\mathbf{w}^\top \mathbf{R}) = \lambda_1\), atteint en \(\mathbf{w} = \mathbf{q}_1\). Pour \(k \ge 2\), parmi les \(\mathbf{w}\) unitaires orthogonaux à \(\mathbf{q}_1, \dots, \mathbf{q}_{k-1}\), le maximum de la variance vaut \(\lambda_k\), atteint en \(\mathbf{q}_k\).
2. Les **composantes principales** \(G_k = \mathbf{q}_k^\top (\mathbf{R} - \mathbf{m})\) (variables aléatoires réelles) sont non corrélées entre elles, de variances \(\lambda_k\). La variance totale \(\sum_i \mathrm{Var}(R_i) = \mathrm{tr}(\boldsymbol{\Sigma}) = \sum_k \lambda_k\), et la **part de variance expliquée** par la composante \(k\) est \(\lambda_k / \mathrm{tr}(\boldsymbol{\Sigma})\).
3. **ACP empirique et SVD.** Sur des données \(\mathbf{X}\) (\(T \times N\)), centrées en \(\mathbf{Y}\) (section 8), de SVD \(\mathbf{Y} = \sum_k s_k \mathbf{u}_k \mathbf{v}_k^\top\) : \(\mathbf{S} = \mathbf{Y}^\top \mathbf{Y}/(T - 1)\) a pour vecteurs propres les \(\mathbf{v}_k\) et pour valeurs propres \(s_k^2/(T - 1)\) ; les valeurs des composantes aux \(T\) dates sont les coordonnées de \(\mathbf{Y}\mathbf{v}_k = s_k \mathbf{u}_k\).
4. **Rendements standardisés.** Soit \(\hat{\sigma}_i > 0\) l'écart-type empirique de l'actif \(i\), calculé avec le même diviseur \(d\) (\(d = T\) ou \(d = T - 1\)) que la matrice ci-dessous, \(\hat{\mathbf{D}}\) la matrice diagonale de coefficients \(\hat{\sigma}_i\), et \(\mathbf{Z} = \mathbf{Y}\hat{\mathbf{D}}^{-1}\) (rendements centrés, divisés par leur volatilité). Alors \(\mathbf{Z}^\top \mathbf{Z}/d\) est la **matrice de corrélation empirique**. L'ACP des rendements standardisés est l'ACP de la corrélation : ses valeurs propres ont pour somme \(N\), et la part de la composante \(k\) est \(\lambda_k / N\).
5. **Modèle à un facteur.** Soit \(R_i = \beta_i F + E_i\) pour \(i = 1, \dots, N\), où \(F\) (le facteur) a pour variance \(\sigma_F^2 > 0\), les \(E_i\) (parts propres à chaque actif) sont non corrélés entre eux et avec \(F\), de même variance \(\sigma_E^2 > 0\), et \(\boldsymbol{\beta} = (\beta_1, \dots, \beta_N)\) est un vecteur fixe non nul. Alors \(\boldsymbol{\Sigma} = \sigma_F^2 \boldsymbol{\beta}\boldsymbol{\beta}^\top + \sigma_E^2 \mathbf{I}\), la première direction principale est \(\boldsymbol{\beta}/\lVert \boldsymbol{\beta} \rVert\) **exactement**, de valeur propre \(\sigma_F^2 \lVert \boldsymbol{\beta} \rVert^2 + \sigma_E^2\), et toutes les autres valeurs propres valent \(\sigma_E^2\). Si de plus tous les \(\beta_i\) valent 1, la matrice de corrélation est l'équicorrélation \(\mathbf{C}_\rho\) avec \(\rho = \sigma_F^2/(\sigma_F^2 + \sigma_E^2)\) ; la première direction est \(\mathbf{1}/\sqrt{N}\) (poids égaux) et sa part vaut \((1 + (N - 1)\rho)/N\).
6. **Signe des poids.** Si toutes les corrélations sont strictement positives, la première direction principale de la matrice de corrélation peut être choisie à coordonnées toutes strictement positives.

### La preuve

1. \(\mathrm{Var}(\mathbf{w}^\top \mathbf{R}) = \mathbf{w}^\top \boldsymbol{\Sigma}\,\mathbf{w}\) (section 2) ; Rayleigh (section 1) donne le premier point. Pour \(\mathbf{w}\) unitaire orthogonal à \(\mathbf{q}_1, \dots, \mathbf{q}_{k-1}\) : \(\mathbf{w} = \sum_{j \ge k} c_j \mathbf{q}_j\) avec \(\sum_{j \ge k} c_j^2 = 1\), donc \(\mathbf{w}^\top \boldsymbol{\Sigma}\,\mathbf{w} = \sum_{j \ge k} \lambda_j c_j^2 \le \lambda_k\), égalité en \(\mathbf{w} = \mathbf{q}_k\).
2. \(\mathrm{Cov}(G_j, G_k) = \mathbf{q}_j^\top \boldsymbol{\Sigma}\,\mathbf{q}_k = \lambda_k\,\mathbf{q}_j^\top \mathbf{q}_k\), qui vaut \(\lambda_k\) si \(j = k\) et 0 sinon. La variance totale est la somme des coefficients diagonaux, donc la trace, égale à la somme des valeurs propres (section 1).
3. \(\mathbf{Y}^\top \mathbf{Y} = \sum_k s_k^2 \mathbf{v}_k \mathbf{v}_k^\top\) (section 5, point 2), et \(\mathbf{Y}\mathbf{v}_k = s_k \mathbf{u}_k\) par orthonormalité des \(\mathbf{v}_j\). Numériquement, on calcule la SVD de \(\mathbf{Y}\) sans former \(\mathbf{Y}^\top \mathbf{Y}\), dont le conditionnement est le carré de celui de \(\mathbf{Y}\) (ses valeurs singulières sont les \(s_k^2\)).
4. Le coefficient \((i, j)\) de \(\mathbf{Z}^\top \mathbf{Z}/d\) vaut \(\dfrac{1}{d}\sum_t \dfrac{y_{ti}\,y_{tj}}{\hat{\sigma}_i \hat{\sigma}_j}\), où \(y_{ti}\) est le coefficient de \(\mathbf{Y}\) ; c'est la covariance empirique de diviseur \(d\) divisée par le produit des écarts-types de même diviseur, c'est-à-dire la corrélation empirique (les diviseurs se simplifient). Ses coefficients diagonaux valent 1, donc sa trace vaut \(N\). Si les diviseurs diffèrent (écarts-types en \(T\), matrice en \(T - 1\)), la matrice est multipliée par \(T/(T - 1)\) : mêmes vecteurs propres, mêmes parts.
5. \(\mathrm{Cov}(R_i, R_j) = \beta_i \beta_j \sigma_F^2 + \mathrm{Cov}(E_i, E_j)\) par bilinéarité de la covariance et non-corrélation de \(F\) et des \(E_i\) ; d'où \(\boldsymbol{\Sigma}\). Ensuite \(\boldsymbol{\Sigma}\boldsymbol{\beta} = \sigma_F^2 \boldsymbol{\beta}(\boldsymbol{\beta}^\top \boldsymbol{\beta}) + \sigma_E^2 \boldsymbol{\beta}\), et pour \(\mathbf{x}\) orthogonal à \(\boldsymbol{\beta}\), \(\boldsymbol{\Sigma}\mathbf{x} = \sigma_E^2 \mathbf{x}\) : \(N\) valeurs propres en tout, la plus grande étant \(\sigma_F^2 \lVert \boldsymbol{\beta} \rVert^2 + \sigma_E^2\). Avec tous les \(\beta_i = 1\) : chaque variance vaut \(\sigma_F^2 + \sigma_E^2\), chaque covariance \(\sigma_F^2\), d'où la corrélation \(\rho\) commune, et la section 2 donne la valeur propre \(1 + (N - 1)\rho\), de vecteur \(\mathbf{1}\).
6. Soit \(\mathbf{q}\) un vecteur propre unitaire de la plus grande valeur propre \(\lambda_1\) de la corrélation \(\mathbf{C}\), et \(\lvert \mathbf{q} \rvert\) le vecteur des valeurs absolues de ses coordonnées, unitaire lui aussi. Comme tous les \(c_{ij} > 0\) : \(\lvert \mathbf{q} \rvert^\top \mathbf{C}\,\lvert \mathbf{q} \rvert = \sum_{i,j} c_{ij} \lvert q_i \rvert \lvert q_j \rvert \ge \sum_{i,j} c_{ij} q_i q_j = \lambda_1\). Par Rayleigh, \(\lvert \mathbf{q} \rvert\) atteint donc le maximum et est un vecteur propre associé à \(\lambda_1\) (section 1, point 5). Sa coordonnée \(i\) vérifie \(\lambda_1 \lvert q_i \rvert = \sum_j c_{ij} \lvert q_j \rvert > 0\) (somme de termes positifs dont l'un au moins est non nul), et \(\lambda_1 \ge c_{11} = 1 > 0\) : toutes les coordonnées sont strictement positives.

### Exemples calculés

Quatre actifs construits comme \(0{,}6\,F + 0{,}8\,E_i\) (corrélation théorique commune \(0{,}6^2 = @L_acp_rho:2m@\)), puis multipliés par des échelles, le quatrième étant cinq fois plus volatil que les autres :

```python-exec
rng = np.random.default_rng(5)
T, N = 500, 4
F = rng.standard_normal((T, 1))
R = (0.6 * F + 0.8 * rng.standard_normal((T, N))) * np.array([0.01, 0.01, 0.01, 0.05])
Y = R - R.mean(axis=0)
Z = Y / Y.std(axis=0)                     # rendements centrés, divisés par leur volatilité (diviseur T)
C = Z.T.dot(Z) / T
print("Z^T Z / T = corrélation empirique :", np.allclose(C, np.corrcoef(R, rowvar=False)))
lam_c, Q_c = np.linalg.eigh(np.cov(R, rowvar=False))
lam_r, Q_r = np.linalg.eigh(C)
q_c, q_r = Q_c[:, -1] * np.sign(Q_c[:, -1].sum()), Q_r[:, -1] * np.sign(Q_r[:, -1].sum())   # signe fixé
print("ACP de la covariance, 1re direction :", q_c.round(3), "; part :", (lam_c[-1] / lam_c.sum()).round(3))
print("ACP de la corrélation, 1re direction :", q_r.round(3), "; part :", (lam_r[-1] / N).round(3))
U, s, Vt = np.linalg.svd(Z, full_matrices=False)
print("SVD de Z : même direction :", np.allclose(np.abs(Vt[0]), np.abs(Q_r[:, -1])), "; s1^2 / T = lambda1 :", np.isclose(s[0] ** 2 / T, lam_r[-1]))
```

L'ACP de la covariance est dominée par l'actif le plus volatil, simplement à cause de son échelle ; l'ACP des rendements standardisés retrouve des poids presque égaux, la structure commune.

Modèle à un facteur, avec \(\sigma_F = 0{,}01\) et \(\sigma_E = 0{,}02\), puis avec des variances propres inégales :

```python-exec
beta = np.array([0.5, 0.8, 1.0, 1.2, 1.5])
Sig = 0.01 ** 2 * np.outer(beta, beta) + 0.02 ** 2 * np.eye(5)
lam, Q = np.linalg.eigh(Sig)
q1 = Q[:, -1] * np.sign(Q[:, -1].sum())
print("valeurs propres x 10000 :", (lam * 1e4).round(4))
print("(sigma_F^2 |beta|^2 + sigma_E^2) x 10000 :", ((0.01 ** 2 * beta.dot(beta) + 0.02 ** 2) * 1e4).round(4))
print("1re direction :", q1.round(4), "; beta / |beta| :", (beta / np.linalg.norm(beta)).round(4))
Sig2 = 0.01 ** 2 * np.outer(beta, beta) + np.diag([0.01, 0.03, 0.01, 0.03, 0.02]) ** 2
q2 = np.linalg.eigh(Sig2)[1][:, -1]
q2 = q2 * np.sign(q2.sum())
print("variances propres inégales :", q2.round(4), "; cosinus avec beta :", (q2.dot(beta) / np.linalg.norm(beta)).round(4))
```

### Ce que l'on peut dire du « facteur de marché »

Démontré : dans un modèle à un facteur avec des variances propres égales, la première composante est exactement le vecteur des expositions \(\boldsymbol{\beta}\) normalisé ; si de plus les expositions sont égales, c'est le portefeuille équipondéré, et sa part tend vers \(\rho\) quand \(N\) grandit. Si toutes les corrélations sont positives, la première composante est un portefeuille à poids tous de même signe (point 6). Non démontré ici : avec des variances propres inégales, la première composante n'est qu'une approximation de \(\boldsymbol{\beta}\) (cosinus inférieur à 1 dans l'exemple). Et l'affirmation empirique selon laquelle, sur de vraies actions, la première composante ressemble à l'indice de marché et explique une part donnée de la variance n'est pas établie dans ce programme : aucun chiffre n'est donné, faute de source vérifiée.

### Pièges

1. Faire l'ACP de la covariance sur des actifs de volatilités très différentes et l'interpréter comme une structure de dépendance : elle mesure surtout les échelles.
2. Diviser par la somme des valeurs propres **affichées** quand on n'en connaît que quelques-unes : pour une corrélation, la somme de toutes vaut \(N\).
3. Interpréter le signe d'un vecteur propre : il est arbitraire.
4. Oublier de centrer avant la SVD : la première direction capte alors la moyenne, pas la variance.
5. Avec \(T \le N\), \(N - T + 1\) valeurs propres au moins sont nulles (section 8) : les dernières composantes ne sont pas estimées du tout.

[[controle:acp]]

## 10. Rétrécissement de Ledoit-Wolf

### Le problème

La section 8 montre que \(\mathbf{S}\) est singulière si \(T \le N\), et mal conditionnée si \(N/T\) n'est pas petit, avec des valeurs propres trop dispersées. On veut un estimateur toujours inversible, bien conditionné, et plus proche de \(\boldsymbol{\Sigma}\) que \(\mathbf{S}\).

### L'intuition

On mélange \(\mathbf{S}\), sans biais mais bruitée, avec une cible \(\mathbf{F}\) très simple, biaisée mais peu bruitée. Avec la cible \(\mu\,\mathbf{I}\), le mélange rapproche chaque valeur propre de leur moyenne : il corrige exactement la dispersion excessive de la section 8.

### L'énoncé

Soit \(\mathbf{S}\) une matrice symétrique SDP \(N \times N\), de décomposition \(\mathbf{S} = \sum_k \lambda_k \mathbf{q}_k \mathbf{q}_k^\top\), \(\mu = \mathrm{tr}(\mathbf{S})/N\) la moyenne de ses valeurs propres (supposée strictement positive), \(\delta \in [0, 1]\) l'**intensité de rétrécissement**, et

\(\displaystyle \hat{\boldsymbol{\Sigma}}_\delta = (1 - \delta)\,\mathbf{S} + \delta\,\mu\,\mathbf{I}.\)

1. \(\hat{\boldsymbol{\Sigma}}_\delta\) a les **mêmes vecteurs propres** que \(\mathbf{S}\), de valeurs propres \((1 - \delta)\lambda_k + \delta\mu\) ; sa trace est celle de \(\mathbf{S}\).
2. Pour \(\delta > 0\), toutes ses valeurs propres sont au moins \(\delta\mu > 0\) : \(\hat{\boldsymbol{\Sigma}}_\delta\) est DP, donc inversible, même si \(\mathbf{S}\) est singulière.
3. Si \(\lambda_{\max} > \lambda_{\min}\), le conditionnement \(\kappa(\delta) = \dfrac{(1 - \delta)\lambda_{\max} + \delta\mu}{(1 - \delta)\lambda_{\min} + \delta\mu}\) décroît strictement de \(\kappa(\mathbf{S})\) (infini si \(\lambda_{\min} = 0\)) à 1 quand \(\delta\) va de 0 à 1.
4. Ce que fait `sklearn.covariance.LedoitWolf` (version 1.1.1, code source lu et recalculé ci-dessous) : \(\mathbf{S}\) est la covariance empirique de diviseur \(T\) (et non \(T - 1\)) des lignes centrées \(\mathbf{y}_t\) des données observées (valeurs fixes, d'où la minuscule), la cible est \(\mu\,\mathbf{I}\), et l'intensité vaut \(\delta = \min(\bar{b}^2, d^2)/d^2\) avec \(d^2 = \lVert \mathbf{S} - \mu\,\mathbf{I} \rVert_F^2 / N\) et \(\bar{b}^2 = \dfrac{1}{N\,T^2}\sum_{t=1}^{T} \lVert \mathbf{y}_t \mathbf{y}_t^\top - \mathbf{S} \rVert_F^2\). La documentation de scikit-learn renvoie pour cette formule à Ledoit et Wolf (2004), « A well-conditioned estimator for large-dimensional covariance matrices », Journal of Multivariate Analysis, volume 88, numéro 2, pages 365 à 411.

### La preuve

1. \(\hat{\boldsymbol{\Sigma}}_\delta \mathbf{q}_k = (1 - \delta)\lambda_k \mathbf{q}_k + \delta\mu\,\mathbf{q}_k\). Trace : \((1 - \delta)\,\mathrm{tr}(\mathbf{S}) + \delta\mu N = \mathrm{tr}(\mathbf{S})\).
2. \(\lambda_k \ge 0\) (section 2) et \(1 - \delta \ge 0\), donc \((1 - \delta)\lambda_k + \delta\mu \ge \delta\mu > 0\) ; la section 2 (cas DP) conclut.
3. Les valeurs propres extrêmes de \(\hat{\boldsymbol{\Sigma}}_\delta\) sont les images de \(\lambda_{\max}\) et \(\lambda_{\min}\), car \(\lambda \mapsto (1 - \delta)\lambda + \delta\mu\) est croissante. On note \(n(\delta) = \lambda_{\max} + \delta(\mu - \lambda_{\max})\) et \(m(\delta) = \lambda_{\min} + \delta(\mu - \lambda_{\min})\), strictement positif pour \(\delta > 0\). La dérivée de \(n/m\) a le signe de \(n' m - n m' = (\mu - \lambda_{\max})\,m(\delta) - (\mu - \lambda_{\min})\,n(\delta)\). Les termes en \(\delta\) se compensent exactement (\((\mu - \lambda_{\max})(\mu - \lambda_{\min})\,\delta\) des deux côtés), et il reste \((\mu - \lambda_{\max})\lambda_{\min} - (\mu - \lambda_{\min})\lambda_{\max} = \mu\,(\lambda_{\min} - \lambda_{\max}) < 0\). En \(\delta = 1\), toutes les valeurs propres valent \(\mu\) et \(\kappa = 1\).
4. Le point 4 décrit un code et se vérifie en le recalculant (ci-dessous). Ce qui est **admis**, et non vérifié ici : le théorème de l'article, selon lequel cette intensité estimée est asymptotiquement optimale, parmi les combinaisons de \(\mathbf{S}\) et de \(\mu\,\mathbf{I}\), pour une perte quadratique de Frobenius normalisée, sous les hypothèses de l'article, y compris quand \(N\) et \(T\) grandissent ensemble.

### Exemple calculé

\(N = 50\) actifs, \(T = 40\) dates, vraie covariance diagonale d'écarts-types répartis de 0,5 à 2 :

```python-exec
from sklearn.covariance import LedoitWolf
import sklearn
rng = np.random.default_rng(6)
N, T = 50, 40
sd = np.linspace(0.5, 2.0, N)
Xr = rng.standard_normal((T, N)) * sd
Y = Xr - Xr.mean(axis=0)
S = Y.T.dot(Y) / T                                   # diviseur T, comme scikit-learn
mu = np.trace(S) / N
d2 = ((S - mu * np.eye(N)) ** 2).sum() / N
b2 = sum(((np.outer(y, y) - S) ** 2).sum() for y in Y) / N / T ** 2
delta = min(b2, d2) / d2
lw = LedoitWolf().fit(Xr)
print("sklearn", sklearn.__version__, ": shrinkage_ =", round(lw.shrinkage_, 4), "; recalcul :", round(delta, 4))
print("même matrice :", np.allclose(lw.covariance_, (1 - delta) * S + delta * mu * np.eye(N)))
vs, vl = np.linalg.eigvalsh(S), np.linalg.eigvalsh(lw.covariance_)
print("S : rang", np.linalg.matrix_rank(S), "; plus petite valeur propre", vs.min().round(6) + 0.0)
print("Ledoit-Wolf : plus petite valeur propre", vl.min().round(4), "; égale à delta * mu :", np.isclose(vl.min(), delta * mu), "; conditionnement", (vl.max() / vl.min()).round(2))
print("mêmes vecteurs propres (5 premiers) :", np.allclose(np.abs(np.linalg.eigh(S)[1][:, -5:]), np.abs(np.linalg.eigh(lw.covariance_)[1][:, -5:])))
Sigma = np.diag(sd ** 2)
print("distance de Frobenius à la vraie covariance : S", np.linalg.norm(S - Sigma).round(3), "; Ledoit-Wolf", np.linalg.norm(lw.covariance_ - Sigma).round(3))
```

Sur cet exemple, l'estimateur rétréci est plus proche de la vraie covariance que \(\mathbf{S}\). C'est une illustration sur un tirage, pas une garantie : la garantie de l'article est asymptotique.

### Cas limites et pièges

1. \(\delta = 0\) redonne \(\mathbf{S}\), \(\delta = 1\) donne \(\mu\,\mathbf{I}\) : tous les actifs de même variance et non corrélés.
2. Avec la cible \(\mu\,\mathbf{I}\), les **vecteurs propres ne changent pas** : l'ACP (directions et ordre des composantes) est la même avant et après rétrécissement ; seules les valeurs propres, donc les parts et le conditionnement, changent. Le rétrécissement ne corrige pas le bruit des directions.
3. Rétrécir vers \(\mathbf{I}\) au lieu de \(\mu\,\mathbf{I}\) : si les variances sont de l'ordre de \(10^{-4}\) (rendements quotidiens), la cible \(\mathbf{I}\) est absurde d'échelle ; \(\mu\) remet la cible à l'échelle des données.
4. Le diviseur : scikit-learn utilise \(T\), `np.cov` utilise \(T - 1\) par défaut. Pour comparer, recalculer avec le même diviseur.
5. D'autres cibles existent (par exemple une corrélation constante, ou un modèle à un facteur) ; l'énoncé 1 sur les vecteurs propres est propre à une cible multiple de \(\mathbf{I}\).

[[controle:lw]]

## Réflexes en 30 secondes

| Indice dans l'énoncé | Premier geste |
|---|---|
| « pour quels \(\rho\) cette matrice est-elle une corrélation » | valeurs propres ; équicorrélation : \(1 + (n-1)\rho \ge 0\) et \(1 - \rho \ge 0\) ; sinon tous les mineurs principaux |
| corrélations imposées pour deux paires, la troisième demandée | déterminant \(3 \times 3\) positif ou nul, et vérifier les mineurs \(2 \times 2\) |
| mineurs dominants positifs ou nuls | ne conclut rien : chercher un coefficient diagonal ou un mineur non dominant négatif |
| rang d'une covariance empirique, \(T\) dates, \(N\) actifs | \(\min(N, T - 1)\) avec centrage, \(\min(N, T)\) sans ; après démoyennage transversal (somme nulle sur les actifs à chaque date), le rang est au plus \(N - 1\) |
| trace d'une projection, degrés de liberté | trace = rang = dimension de l'image ; résidus d'une régression à \(p\) colonnes : \(T - p\) |
| valeurs singulières données | valeurs propres de \(\mathbf{X}^\top \mathbf{X}\) : les carrés ; erreur de rang \(k\) : racine de la somme des carrés omis |
| rendements « centrés et divisés par leur volatilité » | ACP de la corrélation ; somme des valeurs propres \(= N\) ; part \(= \lambda_k / N\) |
| sensibilité de \(\mathbf{w}\) solution de \(\mathbf{A}\mathbf{w} = \mathbf{b}\) | erreur relative multipliée par au plus \(\lambda_{\max}/\lambda_{\min}\) |
| gradient d'une forme quadratique | \((\mathbf{A} + \mathbf{A}^\top)\mathbf{x}\) ; moindres carrés : \(2\mathbf{M}^\top (\mathbf{M}\mathbf{x} - \mathbf{b})\) |
| Ledoit-Wolf, cible \(\mu\,\mathbf{I}\) | valeurs propres \((1 - \delta)\lambda + \delta\mu\), vecteurs propres inchangés |

Contrôles de vraisemblance en 5 secondes :

- une matrice SDP a une diagonale positive ou nulle et \(\lvert a_{ij} \rvert \le \sqrt{a_{ii} a_{jj}}\) (mineur \(2 \times 2\)) ;
- la somme des valeurs propres est la trace ; une part de variance est entre 0 et 1 ;
- un rang ne dépasse ni le nombre de lignes ni le nombre de colonnes.

## À savoir refaire sans notes

- Les valeurs propres d'une matrice symétrique sont réelles, les vecteurs propres de valeurs distinctes orthogonaux, et l'encadrement de Rayleigh avec son cas d'égalité.
- Les équivalences SDP (valeurs propres, forme quadratique, \(\mathbf{B}^\top \mathbf{B}\)), la positivité d'une covariance, et les valeurs propres de l'équicorrélation avec la borne \(-1/(n-1)\) par les deux preuves.
- Le critère des mineurs principaux, le critère de Sylvester, et le contre-exemple des mineurs dominants.
- Projection orthogonale : symétrique, idempotente, valeurs propres 0 et 1, trace = rang ; trace de la matrice chapeau.
- La construction de la SVD à partir de \(\mathbf{X}^\top \mathbf{X}\), et Eckart-Young en norme d'opérateur.
- La borne de conditionnement et son cas d'égalité.
- Les trois gradients composante par composante, et les équations normales.
- Le rang \(\min(N, T - 1)\) de la covariance empirique, majoration et égalité presque sûre (différences \(\mathbf{D}_t\), mesure nulle d'un sous-espace strict, lemme de liberté) ; le biais des valeurs propres extrêmes.
- L'ACP par Rayleigh, par SVD, l'égalité entre ACP des rendements standardisés et ACP de la corrélation, le modèle à un facteur.
- Les valeurs propres et le conditionnement de \((1 - \delta)\mathbf{S} + \delta\mu\,\mathbf{I}\).
""",
    "controles": {
        "spectral": {
            "question": r"Soit \(\mathbf{A}\) une matrice symétrique réelle \(3 \times 3\), \(\mathbf{u}\) un vecteur propre de valeur propre 2 et \(\mathbf{v}\) un vecteur propre de valeur propre 5. Que vaut \(\mathbf{u}^\top \mathbf{v}\) ?",
            "options": [
                {"id": "a", "texte": r"0, quelle que soit \(\mathbf{A}\).",
                 "explication": r"Oui : \(2\,\mathbf{u}^\top \mathbf{v} = (\mathbf{A}\mathbf{u})^\top \mathbf{v} = \mathbf{u}^\top \mathbf{A}\,\mathbf{v} = 5\,\mathbf{u}^\top \mathbf{v}\) grâce à la symétrie, donc \(3\,\mathbf{u}^\top \mathbf{v} = 0\)."},
                {"id": "b", "texte": r"On ne peut pas le savoir sans connaître \(\mathbf{A}\).",
                 "explication": r"La symétrie suffit : deux vecteurs propres de valeurs propres différentes d'une matrice symétrique sont toujours orthogonaux (section 1, point 2)."},
                {"id": "c", "texte": r"1 si \(\mathbf{u}\) et \(\mathbf{v}\) sont unitaires.",
                 "explication": r"\(\mathbf{u}^\top \mathbf{v} = 1\) avec deux vecteurs unitaires voudrait dire \(\mathbf{u} = \mathbf{v}\) (égalité dans Cauchy-Schwarz), impossible puisqu'ils n'ont pas la même valeur propre."},
                {"id": "d", "texte": r"Un nombre non nul, car \(\mathbf{A}\) mélange les directions.",
                 "explication": r"Une matrice symétrique ne mélange pas ses directions propres : elle étire chacune séparément. C'est pour une matrice non symétrique que des vecteurs propres peuvent ne pas être orthogonaux."},
            ],
        },
        "equi": {
            "question": r"Quatre variables ont toutes une variance 1 et des corrélations deux à deux égales à \(-0{,}4\). Cette configuration est-elle possible ?",
            "options": [
                {"id": "a", "texte": r"Non : la valeur propre \(1 + 3\rho\) vaudrait \(@C_eq4_vp:1m@\), et la somme des quatre variables aurait une variance négative.",
                 "explication": r"Oui : pour \(n = 4\), la borne est \(\rho \ge \tfrac{@C_eq4_borne_num@}{@C_eq4_borne_den@}\), et \(-0{,}4\) est en dessous. \(\mathrm{Var}(R_1 + \dots + R_4) = 4 + 12\rho\) serait négative."},
                {"id": "b", "texte": r"Oui, car \(\lvert \rho \rvert \le 1\).",
                 "explication": r"\(\lvert \rho \rvert \le 1\) ne suffit que pour deux variables. Pour quatre, il faut \(\rho \ge \tfrac{@C_eq4_borne_num@}{@C_eq4_borne_den@}\)."},
                {"id": "c", "texte": r"Oui, car \(-0{,}4 \ge -1/2\).",
                 "explication": r"\(-1/2\) est la borne pour trois variables (\(n = 3\), DIAG-M4). Pour quatre variables, elle vaut \(-1/(n-1) = \tfrac{@C_eq4_borne_num@}{@C_eq4_borne_den@}\)."},
                {"id": "d", "texte": "Non, car une matrice de corrélation ne peut avoir aucun coefficient négatif.",
                 "explication": r"Des corrélations négatives sont possibles : par exemple \(\rho = -0{,}3\) convient pour quatre variables, car \(1 + 3 \times (-0{,}3) = @C_eq4_ex:1m@ > 0\)."},
            ],
        },
        "mineurs": {
            "question": r"Les mineurs principaux dominants d'une matrice symétrique \(3 \times 3\) valent 2, 0 et 0. Que peut-on conclure ?",
            "options": [
                {"id": "a", "texte": "Rien sans les autres mineurs principaux : elle peut être SDP ou non.",
                 "explication": r"Oui : la matrice diagonale de coefficients 2, 0, 0 est SDP, et la matrice diagonale de coefficients 2, 0, −1 ne l'est pas ; toutes deux ont les mineurs dominants 2, 0, 0."},
                {"id": "b", "texte": "Elle est SDP, puisque ses mineurs dominants sont positifs ou nuls.",
                 "explication": r"Le critère des mineurs **dominants** n'existe qu'avec des inégalités strictes (Sylvester, pour DP). Contre-exemple : la diagonale 2, 0, −1."},
                {"id": "c", "texte": "Elle n'est pas SDP, puisqu'un mineur dominant est nul.",
                 "explication": r"Un mineur nul n'interdit pas d'être SDP : la matrice diagonale de coefficients 2, 0, 0 est SDP (valeurs propres 2, 0, 0)."},
                {"id": "d", "texte": "Elle est définie positive.",
                 "explication": r"Une matrice DP a un déterminant strictement positif (produit de valeurs propres strictement positives) ; ici il est nul."},
            ],
        },
        "projection": {
            "question": r"Soit \(\mathbf{M} = \mathbf{I}_{50} - \mathbf{1}\mathbf{1}^\top / 50\), la matrice de centrage de 50 observations. Que valent ses valeurs propres et sa trace ?",
            "options": [
                {"id": "a", "texte": r"Valeur propre 0 une fois (vecteur \(\mathbf{1}\)), valeur propre 1 avec multiplicité @C_centrage_trace@ ; trace @C_centrage_trace@.",
                 "explication": r"Oui : \(\mathbf{M}\) est la projection orthogonale sur l'orthogonal de \(\mathbf{1}\), de dimension @C_centrage_trace@ ; trace = rang = dimension de l'image."},
                {"id": "b", "texte": r"Toutes ses valeurs propres valent \(1 - 1/50\) ; trace @C_centrage_trace@.",
                 "explication": r"La trace est juste (50 coefficients diagonaux égaux à \(1 - 1/50\)), mais les valeurs propres d'une projection valent 0 ou 1 : \(\mathbf{M}\mathbf{1} = \mathbf{0}\), donc 0 est valeur propre."},
                {"id": "c", "texte": "Valeurs propres 0 et 1 ; trace 1.",
                 "explication": r"Trace 1 est celle de la projection sur la droite engendrée par \(\mathbf{1}\), \(\mathbf{1}\mathbf{1}^\top/50\). \(\mathbf{M}\) est la projection complémentaire, de trace \(50 - 1\)."},
                {"id": "d", "texte": "Elle est inversible, car ses coefficients diagonaux sont non nuls ; trace 50.",
                 "explication": r"Des coefficients diagonaux non nuls ne garantissent pas l'inversibilité : \(\mathbf{M}\mathbf{1} = \mathbf{0}\), donc \(\mathbf{M}\) est singulière. Et sa trace vaut \(50\,(1 - 1/50)\), pas 50."},
            ],
        },
        "svd": {
            "question": r"Une matrice \(\mathbf{X}\) de taille \(10 \times 3\) est de rang 3. Combien de valeurs propres non nulles a la matrice \(\mathbf{X}\mathbf{X}^\top\), de taille \(10 \times 10\) ?",
            "options": [
                {"id": "a", "texte": "3",
                 "explication": r"Oui : \(\mathbf{X}\mathbf{X}^\top\) et \(\mathbf{X}^\top \mathbf{X}\) ont les mêmes valeurs propres non nulles, les carrés des 3 valeurs singulières non nulles ; les 7 autres valeurs propres de \(\mathbf{X}\mathbf{X}^\top\) sont nulles."},
                {"id": "b", "texte": "10",
                 "explication": r"Le rang de \(\mathbf{X}\mathbf{X}^\top\) est celui de \(\mathbf{X}\), au plus 3 (nombre de colonnes) : une matrice \(10 \times 10\) de rang 3 a 7 valeurs propres nulles."},
                {"id": "c", "texte": "7",
                 "explication": r"7 est le nombre de valeurs propres **nulles** de \(\mathbf{X}\mathbf{X}^\top\) (dimension de son noyau, \(10 - 3\))."},
                {"id": "d", "texte": "Entre 3 et 10 selon les valeurs de \(\mathbf{X}\).",
                 "explication": r"Le nombre est fixé par le rang : \(\mathrm{rang}(\mathbf{X}\mathbf{X}^\top) = \mathrm{rang}(\mathbf{X}^\top) = 3\) (section 3, point 3, appliqué à \(\mathbf{B} = \mathbf{X}^\top\))."},
            ],
        },
        "acp": {
            "question": r"On exprime les rendements d'un seul des \(N\) actifs en points de base (multiplication par \(10^4\)), les autres restant en fraction. Quelle ACP est inchangée ?",
            "options": [
                {"id": "a", "texte": "L'ACP des rendements standardisés (de la corrélation) ; celle de la covariance change.",
                 "explication": r"Oui : la corrélation est invariante par changement d'échelle d'une variable, la covariance non. L'actif changé d'échelle a une variance multipliée par \(10^8\) et accapare la première composante de la covariance."},
                {"id": "b", "texte": "Les deux sont inchangées, car les vecteurs propres ne dépendent pas des unités.",
                 "explication": r"Multiplier une coordonnée par \(a\) remplace \(\boldsymbol{\Sigma}\) par \(\mathbf{D}\boldsymbol{\Sigma}\mathbf{D}\), avec \(\mathbf{D}\) diagonale non multiple de \(\mathbf{I}\) : ce n'est pas une rotation, les vecteurs propres changent."},
                {"id": "c", "texte": "Les deux changent.",
                 "explication": r"La matrice de corrélation, donc l'ACP des rendements standardisés, ne dépend pas des échelles : chaque variable est divisée par son propre écart-type."},
                {"id": "d", "texte": "L'ACP de la covariance est inchangée ; celle de la corrélation change.",
                 "explication": r"C'est l'inverse : la standardisation supprime précisément l'effet d'échelle, que la covariance conserve."},
            ],
        },
        "lw": {
            "question": r"Une covariance empirique \(\mathbf{S}\) \(4 \times 4\) a pour valeurs propres 4,8 ; 2,4 ; 0,8 ; 0, donc \(\mu = \mathrm{tr}(\mathbf{S})/4 = @C_lw_mu@\). On forme \((1 - \delta)\,\mathbf{S} + \delta\,\mu\,\mathbf{I}\) avec \(\delta = 1/4\). Que vaut sa plus petite valeur propre ?",
            "options": [
                {"id": "a", "texte": "@C_lw_vp_min:1@",
                 "explication": r"Oui : \((1 - \delta) \times 0 + \delta\mu = \tfrac14 \times @C_lw_mu@\). Le rétrécissement rend la matrice inversible."},
                {"id": "b", "texte": r"0, car \(\mathbf{S}\) est singulière et le rétrécissement ne change pas le noyau.",
                 "explication": r"Les vecteurs propres sont inchangés, mais la valeur propre associée au noyau devient \(\delta\mu > 0\) : le noyau disparaît."},
                {"id": "c", "texte": "@C_lw_d_id:2@",
                 "explication": r"C'est \(\delta \times 1\) : le rétrécissement vers \(\mathbf{I}\) au lieu de \(\mu\,\mathbf{I}\). Ici \(\mu = @C_lw_mu@\)."},
                {"id": "d", "texte": "@C_lw_d_inv:1@",
                 "explication": r"C'est \((1 - \delta)\,\mu\) : les poids de \(\mathbf{S}\) et de la cible ont été échangés. Le poids \(\delta\) va à la cible \(\mu\,\mathbf{I}\)."},
            ],
        },
    },
    "cellules_initiales": [
        "import numpy as np\nnp.set_printoptions(precision=4, suppress=True)\n\n# Essaie ici les exemples de la leçon, en les modifiant.\nA = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])\nnp.linalg.eigh(A)",
    ],
    "verif": {"script": "verif/seance_algebre_1.py"},
}
