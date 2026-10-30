"""Leçon de la séance de ML 1 : régression linéaire, Ridge et Lasso, au niveau démonstration.

Format : voir outils/lecons.py (élément de type "lecon", contenu en Markdown réduit, points de contrôle
[[controle:IDENT]] sur une ligne seule, option "a" correcte, une explication par option).
Aucun nombre calculé n'est écrit à la main : chaque jeton @CLE@, @CLE:3@ ou @CLE:3m@ est remplacé par la valeur
calculée par verif/seance_ml_1.py (fichier verif/seance_ml_1.out.json), et les blocs ```python-exec sont exécutés
à la construction par le noyau de l'environnement des notebooks (scikit-learn 1.1.1, numpy 1.21.6), qui insère leur
vraie sortie. Graines aléatoires fixées dans chaque bloc.
"""

LECON_REGLIN = {
    "id": "L-REGLIN-1", "type": "lecon", "section": "lecon",
    "theme": "Leçon régression linéaire : moindres carrés, hypothèses, R², Ridge, Lasso",
    "difficulte": 4, "temps_cible_s": 3000, "dossier": "coding/lecons",
    "contenu": r"""## Leçon : régression linéaire, Ridge et Lasso

Durée visée : 50 minutes. La régression linéaire, Ridge et Lasso y sont traités au niveau démonstration, jusqu'à la question « existe-t-il une forme fermée pour Ridge et pour Lasso ? ». La leçon a six parties : moindres carrés, hypothèses du modèle, R², Ridge, Lasso, biais et variance. Chacune suit le même ordre : le problème, l'intuition, l'énoncé, la preuve, un exemple chiffré, les cas limites, les pièges. Chaque affirmation de machine learning a une preuve et une expérience ; le notebook de droite permet de relancer et de modifier chaque expérience. La partie « Réflexes en 30 secondes » est celle à relire juste avant un test ou un entretien.

Conventions de notation, valables pour toute la leçon. Une majuscule non grasse désigne une variable aléatoire réelle, une minuscule une valeur ou un paramètre fixe. En gras, une minuscule est un vecteur colonne et une majuscule une matrice ; pour un vecteur, la majuscule grasse signale qu'il est aléatoire (\(\mathbf{Y}\) aléatoire, \(\mathbf{y}\) sa valeur observée). Une matrice en gras est fixe sauf mention contraire. Les lettres grecques n'ont pas de casse utile : on dit en mots si elles sont aléatoires ; le bruit \(\boldsymbol{\varepsilon}\) et tout estimateur à chapeau calculé à partir de \(\mathbf{Y}\) le sont. Exceptions d'usage : le coefficient de détermination R² et les sommes de carrés SCT, SCE, SCR gardent leur notation habituelle, et sont des nombres calculés sur des données observées. Pour gagner de la place, un vecteur colonne est écrit en ligne : \((0, 1, 2, 3)\).

Symboles communs : \(^\top\) est la transposée ; \(\lVert \mathbf{v} \rVert = \sqrt{\mathbf{v}^\top \mathbf{v}}\) la norme euclidienne et \(\lVert \mathbf{v} \rVert_1 = \sum_j \lvert v_j \rvert\) la norme L1 ; \(\mathbf{I}\) la matrice identité, \(\mathbf{I}_p\) quand sa taille \(p \times p\) compte ; \(\mathbf{1}\) le vecteur dont toutes les coordonnées valent 1 ; \(\mathrm{tr}\) la trace, somme des coefficients diagonaux. \(\mathbb{E}\), \(\mathrm{Var}\) et \(\mathrm{Cov}\) désignent espérance, variance et covariance ; pour un vecteur aléatoire \(\mathbf{Z}\), \(\mathrm{Var}(\mathbf{Z}) = \mathbb{E}[(\mathbf{Z} - \mathbb{E}\mathbf{Z})(\mathbf{Z} - \mathbb{E}\mathbf{Z})^\top]\) est sa matrice de covariance. Une matrice symétrique \(\mathbf{M}\) est semi-définie positive si \(\mathbf{v}^\top \mathbf{M} \mathbf{v} \ge 0\) pour tout vecteur \(\mathbf{v}\), définie positive si l'inégalité est stricte pour \(\mathbf{v} \ne \mathbf{0}\). Tous les nombres hors des sorties de code sont calculés par `verif/seance_ml_1.py`.

Le premier bloc prépare l'affichage ; il est exécuté une fois, et les blocs suivants réutilisent ses variables.

```python-exec
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score
np.set_printoptions(precision=4, suppress=True)
print("prêt")
```

## Partie 1 : les moindres carrés

### Le problème

On observe \(n\) individus. Pour chacun, on connaît \(p\) variables explicatives et une réponse. On cherche les coefficients d'une combinaison linéaire des variables qui approche le mieux la réponse, au sens de la somme des carrés des écarts. Trois questions : la solution existe-t-elle, est-elle unique, a-t-elle une formule ?

### Notations

- \(n \ge 1\) et \(p \ge 1\) : entiers, nombre d'observations et nombre de colonnes.
- \(\mathbf{X}\) : matrice fixe \(n \times p\). Sa ligne \(i\) contient les variables de l'observation \(i\) ; sa colonne \(j\), notée \(\mathbf{x}_j \in \mathbb{R}^n\), contient la variable \(j\) pour toutes les observations. Si le modèle a une constante, c'est une colonne égale à \(\mathbf{1}\), comptée dans \(p\).
- \(\mathbf{y} \in \mathbb{R}^n\) : les réponses observées ; \(\mathbf{b} \in \mathbb{R}^p\) : un vecteur de coefficients candidat, variable du problème d'optimisation.
- \(s(\mathbf{b}) = \lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2\) : la somme des carrés des résidus du candidat \(\mathbf{b}\).
- \(\mathrm{Im}(\mathbf{X}) = \{\mathbf{X}\mathbf{b} : \mathbf{b} \in \mathbb{R}^p\}\) : l'image de \(\mathbf{X}\), sous-espace de \(\mathbb{R}^n\) engendré par les colonnes. \(\ker(\mathbf{X}) = \{\mathbf{h} \in \mathbb{R}^p : \mathbf{X}\mathbf{h} = \mathbf{0}\}\) : son noyau. Le rang de \(\mathbf{X}\) est la dimension de son image ; il vaut au plus \(\min(n, p)\), et le théorème du rang dit que rang plus dimension du noyau égale \(p\).

### L'intuition

Quand \(\mathbf{b}\) parcourt \(\mathbb{R}^p\), le vecteur \(\mathbf{X}\mathbf{b}\) parcourt le sous-espace \(\mathrm{Im}(\mathbf{X})\). Minimiser \(s\), c'est chercher le point de ce sous-espace le plus proche de \(\mathbf{y}\) : le pied de la perpendiculaire abaissée de \(\mathbf{y}\). L'écart \(\mathbf{y} - \mathbf{X}\mathbf{b}\) doit donc être orthogonal au sous-espace, c'est-à-dire à chaque colonne : \(\mathbf{x}_j^\top (\mathbf{y} - \mathbf{X}\mathbf{b}) = 0\) pour tout \(j\). Écrites ensemble, ces \(p\) équations sont les équations normales.

### L'énoncé

1. Un vecteur \(\mathbf{b}\) minimise \(s\) si et seulement s'il vérifie les **équations normales** \(\mathbf{X}^\top \mathbf{X}\,\mathbf{b} = \mathbf{X}^\top \mathbf{y}\). Elles ont toujours au moins une solution : le minimum existe pour toute matrice \(\mathbf{X}\) et tout \(\mathbf{y}\).
2. Le minimiseur est unique si et seulement si \(\mathbf{X}\) est de rang \(p\) (colonnes linéairement indépendantes, ce qui exige \(n \ge p\)), ce qui équivaut à l'inversibilité de \(\mathbf{X}^\top \mathbf{X}\). Il vaut alors \(\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}\) : c'est la **forme fermée** des moindres carrés ordinaires (MCO ; OLS, ordinary least squares, en anglais).
3. Tous les minimiseurs donnent les mêmes **valeurs ajustées** \(\hat{\mathbf{y}}\), égales à \(\mathbf{X}\mathbf{b}\) pour n'importe quel minimiseur \(\mathbf{b}\) : c'est la projection orthogonale de \(\mathbf{y}\) sur \(\mathrm{Im}(\mathbf{X})\). Le vecteur des **résidus** \(\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}\) vérifie \(\mathbf{X}^\top \mathbf{e} = \mathbf{0}\) : il est orthogonal à chaque colonne, donc à \(\hat{\mathbf{y}}\).
4. Si \(\mathbf{X}\) est de rang \(p\), \(\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}\), où la **matrice chapeau** \(\mathbf{H} = \mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top\), de taille \(n \times n\), est symétrique (\(\mathbf{H}^\top = \mathbf{H}\)), idempotente (\(\mathbf{H}^2 = \mathbf{H}\)) et de trace \(p\). La matrice \(\mathbf{I} - \mathbf{H}\) est symétrique, idempotente, de trace \(n - p\), et \(\mathbf{e} = (\mathbf{I} - \mathbf{H})\mathbf{y}\).
5. Si \(\mathbf{1}\) est une colonne de \(\mathbf{X}\), la somme des résidus est nulle, et la moyenne des valeurs ajustées égale celle des \(y_i\).

### La preuve

1. **Identité de développement (D).** Pour tous \(\mathbf{b}, \mathbf{h} \in \mathbb{R}^p\), on pose \(\mathbf{r} = \mathbf{y} - \mathbf{X}\mathbf{b}\). En développant \(\lVert \mathbf{u} - \mathbf{v} \rVert^2 = \lVert \mathbf{u} \rVert^2 - 2\,\mathbf{v}^\top \mathbf{u} + \lVert \mathbf{v} \rVert^2\) avec \(\mathbf{u} = \mathbf{r}\), \(\mathbf{v} = \mathbf{X}\mathbf{h}\), et \((\mathbf{X}\mathbf{h})^\top \mathbf{r} = \mathbf{h}^\top \mathbf{X}^\top \mathbf{r}\) : \(s(\mathbf{b} + \mathbf{h}) = s(\mathbf{b}) - 2\,\mathbf{h}^\top \mathbf{X}^\top \mathbf{r} + \lVert \mathbf{X}\mathbf{h} \rVert^2\).
2. **Condition suffisante.** Si \(\mathbf{b}\) vérifie les équations normales, \(\mathbf{X}^\top \mathbf{r} = \mathbf{0}\), et (D) donne \(s(\mathbf{b} + \mathbf{h}) = s(\mathbf{b}) + \lVert \mathbf{X}\mathbf{h} \rVert^2 \ge s(\mathbf{b})\) pour tout \(\mathbf{h}\) : \(\mathbf{b}\) est un minimiseur global, et \(\mathbf{b} + \mathbf{h}\) en est un autre si et seulement si \(\mathbf{X}\mathbf{h} = \mathbf{0}\).
3. **Condition nécessaire.** Soit \(\mathbf{b}\) un minimiseur, \(\mathbf{g} = \mathbf{X}^\top \mathbf{r}\), et \(t > 0\) un réel. (D) avec \(\mathbf{h} = t\,\mathbf{g}\) donne \(s(\mathbf{b} + t\,\mathbf{g}) - s(\mathbf{b}) = -2t\,\lVert \mathbf{g} \rVert^2 + t^2 \lVert \mathbf{X}\mathbf{g} \rVert^2\). Si \(\mathbf{g} \ne \mathbf{0}\), ce nombre est strictement négatif pour \(t\) assez petit (pour tout \(t\) si \(\mathbf{X}\mathbf{g} = \mathbf{0}\), pour \(t < 2 \lVert \mathbf{g} \rVert^2 / \lVert \mathbf{X}\mathbf{g} \rVert^2\) sinon), ce qui contredit la minimalité. Donc \(\mathbf{g} = \mathbf{0}\).
4. **Existence.** Il faut que \(\mathbf{X}^\top \mathbf{y}\) soit dans l'image de \(\mathbf{X}^\top \mathbf{X}\). D'abord \(\ker(\mathbf{X}^\top \mathbf{X}) = \ker(\mathbf{X})\) : si \(\mathbf{X}^\top \mathbf{X}\mathbf{v} = \mathbf{0}\), alors \(0 = \mathbf{v}^\top \mathbf{X}^\top \mathbf{X}\mathbf{v} = \lVert \mathbf{X}\mathbf{v} \rVert^2\), donc \(\mathbf{X}\mathbf{v} = \mathbf{0}\) ; l'inclusion inverse est immédiate. Par le théorème du rang, \(\mathbf{X}^\top \mathbf{X}\) a donc le rang de \(\mathbf{X}\), qui est aussi celui de \(\mathbf{X}^\top\). Or \(\mathrm{Im}(\mathbf{X}^\top \mathbf{X}) \subset \mathrm{Im}(\mathbf{X}^\top)\), puisque \(\mathbf{X}^\top \mathbf{X}\mathbf{v} = \mathbf{X}^\top (\mathbf{X}\mathbf{v})\) ; deux sous-espaces emboîtés de même dimension sont égaux. Donc \(\mathbf{X}^\top \mathbf{y} \in \mathrm{Im}(\mathbf{X}^\top) = \mathrm{Im}(\mathbf{X}^\top \mathbf{X})\).
5. **Unicité.** Par l'étape 2, les minimiseurs sont les \(\mathbf{b} + \mathbf{h}\), \(\mathbf{h} \in \ker(\mathbf{X})\), pour un minimiseur \(\mathbf{b}\) fixé. Il y en a un seul si et seulement si \(\ker(\mathbf{X}) = \{\mathbf{0}\}\), c'est-à-dire rang \(p\) par le théorème du rang ; par l'étape 4, c'est dire que la matrice carrée \(\mathbf{X}^\top \mathbf{X}\), de taille \(p \times p\), est de rang \(p\), donc inversible. Les équations normales donnent alors la forme fermée.
6. **Projection.** Deux minimiseurs diffèrent d'un \(\mathbf{h}\) avec \(\mathbf{X}\mathbf{h} = \mathbf{0}\) : mêmes valeurs ajustées. \(\hat{\mathbf{y}}\) est dans \(\mathrm{Im}(\mathbf{X})\), et \(\mathbf{X}^\top (\mathbf{y} - \hat{\mathbf{y}}) = \mathbf{0}\) dit que \(\mathbf{y} - \hat{\mathbf{y}}\) est orthogonal à chaque colonne, donc à toute combinaison des colonnes, en particulier à \(\hat{\mathbf{y}}\) : c'est la définition de la projection orthogonale.
7. **Matrice chapeau.** \(\mathbf{H}^\top = \mathbf{X}\big((\mathbf{X}^\top \mathbf{X})^{-1}\big)^\top \mathbf{X}^\top = \mathbf{H}\), car l'inverse d'une matrice symétrique est symétrique. \(\mathbf{H}^2 = \mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1} (\mathbf{X}^\top \mathbf{X}) (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top = \mathbf{H}\). Pour \(\mathbf{A}\) de taille \(n \times p\) et \(\mathbf{B}\) de taille \(p \times n\), \(\mathrm{tr}(\mathbf{A}\mathbf{B}) = \sum_{i, k} a_{ik} b_{ki} = \mathrm{tr}(\mathbf{B}\mathbf{A})\) ; donc \(\mathrm{tr}(\mathbf{H}) = \mathrm{tr}\big((\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{X}\big) = \mathrm{tr}(\mathbf{I}_p) = p\). Enfin \((\mathbf{I} - \mathbf{H})^2 = \mathbf{I} - 2\mathbf{H} + \mathbf{H}^2 = \mathbf{I} - \mathbf{H}\), de trace \(n - p\).
8. **Constante.** Si \(\mathbf{1}\) est une colonne, la coordonnée correspondante de \(\mathbf{X}^\top \mathbf{e} = \mathbf{0}\) s'écrit \(\sum_i e_i = 0\), d'où \(\sum_i \hat{y}_i = \sum_i y_i\).

### Exemple chiffré

Quatre observations (\(n = @L1_n@\)), une constante et une variable (\(p = @L1_p@\)) : \(\mathbf{x}_1 = \mathbf{1}\), \(\mathbf{x}_2 = (0, 1, 2, 3)\) et \(\mathbf{y} = (1, 3, 2, 5)\).

1. \(\mathbf{X}^\top \mathbf{X} = \begin{pmatrix} @L1_xtx11@ & @L1_xtx12@ \\ @L1_xtx12@ & @L1_xtx22@ \end{pmatrix}\) (nombre d'observations, somme des \(x_{i2}\), somme de leurs carrés) et \(\mathbf{X}^\top \mathbf{y} = (@L1_xty1@, @L1_xty2@)\) (somme des \(y_i\), somme des \(x_{i2} y_i\)).
2. Déterminant \(@L1_det@\), donc \((\mathbf{X}^\top \mathbf{X})^{-1} = \tfrac{1}{@L1_det@} \begin{pmatrix} @L1_adj11@ & @L1_adj12@ \\ @L1_adj12@ & @L1_adj22@ \end{pmatrix}\).
3. \(\hat{\boldsymbol{\beta}} = \tfrac{1}{@L1_det@} (@L1_adjxty1@, @L1_adjxty2@) = (@L1_b0:1m@ ; @L1_b1:1m@)\) : constante et pente valent toutes deux \(@L1_b0:1m@\).
4. \(\hat{\mathbf{y}} = @L1_yhat:1m@\) et \(\mathbf{e} = @L1_e:1m@\). Contrôles : \(\sum_i e_i = @L1_somme_e@\) ; \(\mathbf{x}_2^\top \mathbf{e} = 0 \times (@L1_e1:1m@) + 1 \times @L1_e2:1m@ + 2 \times (@L1_e3:1m@) + 3 \times @L1_e4:1m@ = @L1_xe@\) ; \(\mathrm{tr}(\mathbf{H}) = @L1_trH@\). En revanche \(\mathbf{y}^\top \mathbf{e} = \lVert \mathbf{e} \rVert^2 = @L1_ye:1m@\), non nul : les résidus ne sont pas orthogonaux à \(\mathbf{y}\).

```python-exec
x2 = np.array([0.0, 1.0, 2.0, 3.0])
y = np.array([1.0, 3.0, 2.0, 5.0])
X = np.column_stack([np.ones(4), x2])
b_normales = np.linalg.solve(X.T.dot(X), X.T.dot(y))
b_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
e = y - X.dot(b_normales)
H = X.dot(np.linalg.inv(X.T.dot(X))).dot(X.T)
print("b par les équations normales :", b_normales)
print("b par lstsq (SVD)            :", b_lstsq)
print("résidus e                    :", e)
print("X^T e                        :", X.T.dot(e))
print("H symétrique :", np.allclose(H, H.T), "; idempotente :", np.allclose(H.dot(H), H), "; trace :", round(np.trace(H), 10))
```

### Cas limites

- Colonnes liées (rang inférieur à \(p\)) : une infinité de minimiseurs, mêmes valeurs ajustées. Exemple : deux colonnes égales \(\mathbf{x}_1 = \mathbf{x}_2\) ; seule la somme \(b_1 + b_2\) est déterminée. `np.linalg.lstsq` renvoie alors la solution de plus petite norme (ici \(b_1 = b_2\)), justifiée en partie 4.
- \(p > n\) : le rang est au plus \(n < p\), le minimiseur n'est jamais unique.
- \(n = p\) et \(\mathbf{X}\) inversible : \(\mathrm{Im}(\mathbf{X}) = \mathbb{R}^n\), donc \(\hat{\mathbf{y}} = \mathbf{y}\), \(\mathbf{e} = \mathbf{0}\) et \(\mathbf{H} = \mathbf{I}\) : ajustement parfait, qui ne dit rien de la qualité de prédiction.
- Constante seule (\(\mathbf{X} = \mathbf{1}\)) : \(\hat{\beta} = \mathbf{1}^\top \mathbf{y} / n\), la moyenne des \(y_i\).

### Pièges

1. Les résidus sont orthogonaux aux colonnes et à \(\hat{\mathbf{y}}\), pas à \(\mathbf{y}\) : \(\mathbf{y}^\top \mathbf{e} = (\hat{\mathbf{y}} + \mathbf{e})^\top \mathbf{e} = \lVert \mathbf{e} \rVert^2\).
2. « La somme des résidus est nulle » exige que \(\mathbf{1}\) soit dans \(\mathrm{Im}(\mathbf{X})\), par exemple une colonne constante.
3. Pour calculer, on ne forme pas l'inverse : `np.linalg.solve` sur les équations normales, ou mieux `np.linalg.lstsq` qui travaille sur \(\mathbf{X}\) (par SVD) et évite d'élever le conditionnement au carré (partie 4).
4. Oublier la colonne constante force l'hyperplan ajusté à passer par l'origine.

[[controle:residus]]

## Partie 2 : le modèle statistique et ce que garantit chaque hypothèse

### Le problème

La partie 1 est de la géométrie : aucune probabilité. Pour dire si \(\hat{\boldsymbol{\beta}}\) est proche des vrais coefficients, avec quelle précision, et pour tester, il faut un modèle qui décrit comment les données sont produites. Chaque hypothèse du modèle garantit une propriété précise ; quand elle tombe, c'est cette propriété qui tombe, et pas les autres.

### Le modèle

\(\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}\), où \(\mathbf{X}\) est la matrice fixe \(n \times p\) de la partie 1, \(\boldsymbol{\beta} \in \mathbb{R}^p\) le vecteur fixe et inconnu des vrais coefficients, \(\boldsymbol{\varepsilon}\) un vecteur aléatoire de \(\mathbb{R}^n\) (le bruit), et \(\mathbf{Y}\) le vecteur aléatoire des réponses, dont \(\mathbf{y}\) est une réalisation. \(\sigma^2 > 0\) est un réel fixe. L'estimateur \(\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{Y}\) est désormais un vecteur aléatoire, de même que les résidus \(\hat{\boldsymbol{\varepsilon}} = \mathbf{Y} - \mathbf{X}\hat{\boldsymbol{\beta}}\).

| Hypothèse | Énoncé |
|---|---|
| (A1) rang | \(\mathbf{X}\) est de rang \(p\) |
| (A2) exogénéité | \(\mathbb{E}[\boldsymbol{\varepsilon}] = \mathbf{0}\) : le modèle est bien spécifié, \(\mathbb{E}[\mathbf{Y}] = \mathbf{X}\boldsymbol{\beta}\) |
| (A3) homoscédasticité et non-corrélation | \(\mathrm{Var}(\boldsymbol{\varepsilon}) = \sigma^2 \mathbf{I}\) : chaque \(\varepsilon_i\) a la variance \(\sigma^2\), et \(\mathrm{Cov}(\varepsilon_i, \varepsilon_m) = 0\) pour \(i \ne m\) |
| (A4) normalité | \(\boldsymbol{\varepsilon}\) suit la loi normale \(\mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})\) |

Plan aléatoire : si \(\mathbf{X}\) est aléatoire, on remplace (A2) par \(\mathbb{E}[\boldsymbol{\varepsilon} \mid \mathbf{X}] = \mathbf{0}\) et (A3) par \(\mathrm{Var}(\boldsymbol{\varepsilon} \mid \mathbf{X}) = \sigma^2 \mathbf{I}\) ; les résultats ci-dessous valent alors conditionnellement à \(\mathbf{X}\), avec les mêmes preuves.

### L'énoncé

1. Sous (A1) et (A2) : \(\mathbb{E}[\hat{\boldsymbol{\beta}}] = \boldsymbol{\beta}\) (estimateur sans biais).
2. Sous (A1), (A2) et (A3) : \(\mathrm{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2 (\mathbf{X}^\top \mathbf{X})^{-1}\), et, si \(n > p\), \(\hat{\sigma}^2 = \lVert \hat{\boldsymbol{\varepsilon}} \rVert^2 / (n - p)\) vérifie \(\mathbb{E}[\hat{\sigma}^2] = \sigma^2\).
3. **Théorème de Gauss-Markov**, sous (A1), (A2) et (A3). Soit \(\mathbf{C}\) une matrice fixe \(p \times n\) telle que \(\tilde{\boldsymbol{\beta}} = \mathbf{C}\mathbf{Y}\) soit sans biais pour toute valeur de \(\boldsymbol{\beta}\). Alors \(\mathrm{Var}(\tilde{\boldsymbol{\beta}}) - \mathrm{Var}(\hat{\boldsymbol{\beta}})\) est semi-définie positive ; en particulier, pour tout vecteur fixe \(\mathbf{a} \in \mathbb{R}^p\), \(\mathrm{Var}(\mathbf{a}^\top \tilde{\boldsymbol{\beta}}) \ge \mathrm{Var}(\mathbf{a}^\top \hat{\boldsymbol{\beta}})\). Les MCO sont le meilleur estimateur linéaire sans biais (BLUE, best linear unbiased estimator).
4. Sous (A1) à (A4) : \(\hat{\boldsymbol{\beta}}\) suit la loi \(\mathcal{N}\big(\boldsymbol{\beta}, \sigma^2 (\mathbf{X}^\top \mathbf{X})^{-1}\big)\), et pour chaque \(j\), \(T_j = (\hat{\beta}_j - \beta_j) / \sqrt{\hat{\sigma}^2 [(\mathbf{X}^\top \mathbf{X})^{-1}]_{jj}}\) suit une loi de Student à \(n - p\) degrés de liberté : c'est la base des tests t et des intervalles de confiance exacts. De plus, les MCO sont l'estimateur du maximum de vraisemblance de \(\boldsymbol{\beta}\).

### La preuve

1. **Sans biais.** \(\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top (\mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}) = \boldsymbol{\beta} + \mathbf{A}\boldsymbol{\varepsilon}\), avec \(\mathbf{A} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top\) fixe, de taille \(p \times n\). Par linéarité de l'espérance, coordonnée par coordonnée, et (A2) : \(\mathbb{E}[\hat{\boldsymbol{\beta}}] = \boldsymbol{\beta} + \mathbf{A}\,\mathbb{E}[\boldsymbol{\varepsilon}] = \boldsymbol{\beta}\). Seules (A1), pour que \(\mathbf{A}\) existe, et (A2) ont servi.
2. **Règle de calcul.** Pour une matrice fixe \(\mathbf{M}\) et un vecteur aléatoire \(\mathbf{Z}\) : \(\mathbf{M}\mathbf{Z} - \mathbb{E}[\mathbf{M}\mathbf{Z}] = \mathbf{M}(\mathbf{Z} - \mathbb{E}\mathbf{Z})\), donc \(\mathrm{Var}(\mathbf{M}\mathbf{Z}) = \mathbb{E}[\mathbf{M}(\mathbf{Z} - \mathbb{E}\mathbf{Z})(\mathbf{Z} - \mathbb{E}\mathbf{Z})^\top \mathbf{M}^\top] = \mathbf{M}\,\mathrm{Var}(\mathbf{Z})\,\mathbf{M}^\top\) par linéarité. Avec (A3) : \(\mathrm{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2 \mathbf{A}\mathbf{A}^\top = \sigma^2 (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{X} (\mathbf{X}^\top \mathbf{X})^{-1} = \sigma^2 (\mathbf{X}^\top \mathbf{X})^{-1}\).
3. **Variance résiduelle.** \(\hat{\boldsymbol{\varepsilon}} = (\mathbf{I} - \mathbf{H})\mathbf{Y} = (\mathbf{I} - \mathbf{H})\boldsymbol{\varepsilon}\), car \((\mathbf{I} - \mathbf{H})\mathbf{X} = \mathbf{X} - \mathbf{X} = \mathbf{0}\). Avec \(\mathbf{M} = \mathbf{I} - \mathbf{H}\), symétrique et idempotente (partie 1) : \(\lVert \hat{\boldsymbol{\varepsilon}} \rVert^2 = \boldsymbol{\varepsilon}^\top \mathbf{M}^\top \mathbf{M}\boldsymbol{\varepsilon} = \boldsymbol{\varepsilon}^\top \mathbf{M}\boldsymbol{\varepsilon}\). Un nombre est sa propre trace, et \(\mathrm{tr}(\mathbf{A}\mathbf{B}) = \mathrm{tr}(\mathbf{B}\mathbf{A})\) : \(\boldsymbol{\varepsilon}^\top \mathbf{M}\boldsymbol{\varepsilon} = \mathrm{tr}(\mathbf{M}\boldsymbol{\varepsilon}\boldsymbol{\varepsilon}^\top)\). L'espérance passe à travers la trace, qui est linéaire, et \(\mathbb{E}[\boldsymbol{\varepsilon}\boldsymbol{\varepsilon}^\top] = \mathrm{Var}(\boldsymbol{\varepsilon}) = \sigma^2 \mathbf{I}\) car \(\mathbb{E}\boldsymbol{\varepsilon} = \mathbf{0}\) : \(\mathbb{E}\lVert \hat{\boldsymbol{\varepsilon}} \rVert^2 = \sigma^2\,\mathrm{tr}(\mathbf{I} - \mathbf{H}) = \sigma^2 (n - p)\).
4. **Gauss-Markov.** \(\mathbb{E}[\mathbf{C}\mathbf{Y}] = \mathbf{C}\mathbf{X}\boldsymbol{\beta}\) doit valoir \(\boldsymbol{\beta}\) pour tout \(\boldsymbol{\beta}\) ; en prenant pour \(\boldsymbol{\beta}\) chaque vecteur de la base canonique, \(\mathbf{C}\mathbf{X} = \mathbf{I}_p\). On pose \(\mathbf{D} = \mathbf{C} - \mathbf{A}\) ; alors \(\mathbf{D}\mathbf{X} = \mathbf{I}_p - \mathbf{I}_p = \mathbf{0}\). Par l'étape 2 : \(\mathrm{Var}(\mathbf{C}\mathbf{Y}) = \sigma^2 \mathbf{C}\mathbf{C}^\top = \sigma^2 (\mathbf{A}\mathbf{A}^\top + \mathbf{A}\mathbf{D}^\top + \mathbf{D}\mathbf{A}^\top + \mathbf{D}\mathbf{D}^\top)\). Or \(\mathbf{A}\mathbf{D}^\top = (\mathbf{X}^\top \mathbf{X})^{-1} (\mathbf{D}\mathbf{X})^\top = \mathbf{0}\), et \(\mathbf{D}\mathbf{A}^\top\) est sa transposée. Donc \(\mathrm{Var}(\tilde{\boldsymbol{\beta}}) - \mathrm{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2 \mathbf{D}\mathbf{D}^\top\), et \(\mathbf{a}^\top \mathbf{D}\mathbf{D}^\top \mathbf{a} = \lVert \mathbf{D}^\top \mathbf{a} \rVert^2 \ge 0\).
5. **Normalité.** \(\hat{\boldsymbol{\beta}} = \boldsymbol{\beta} + \mathbf{A}\boldsymbol{\varepsilon}\) est une transformation affine d'un vecteur gaussien, donc gaussienne (admis), d'espérance et de variance données par les étapes 1 et 2. La loi de Student est admise : elle repose sur l'indépendance de \(\hat{\boldsymbol{\beta}}\) et \(\hat{\boldsymbol{\varepsilon}}\), qui sont non corrélés (\(\mathbf{A}(\mathbf{I} - \mathbf{H}) = \mathbf{A} - \mathbf{A} = \mathbf{0}\), car \(\mathbf{A}\mathbf{H} = \mathbf{A}\)) et conjointement gaussiens, et sur la loi du khi-deux à \(n - p\) degrés de liberté de \(\lVert \hat{\boldsymbol{\varepsilon}} \rVert^2 / \sigma^2\).
6. **Maximum de vraisemblance.** Sous (A4), la densité de \(\mathbf{Y}\) en \(\mathbf{y}\), pour des paramètres \(\mathbf{b}\) et \(\sigma^2\), vaut \((2\pi\sigma^2)^{-n/2} \exp\big(-s(\mathbf{b})/(2\sigma^2)\big)\). Son logarithme est \(-\tfrac{n}{2} \ln(2\pi\sigma^2) - s(\mathbf{b})/(2\sigma^2)\) : à \(\sigma^2\) fixé, la maximiser en \(\mathbf{b}\) revient à minimiser \(s(\mathbf{b})\).

### Exemple chiffré, suite

Sur l'exemple à quatre points : \(\lVert \mathbf{e} \rVert^2 = @L2_scr:1m@\), donc la valeur observée de \(\hat{\sigma}^2\) est \(@L2_scr:1m@ / (4 - 2) = @L2_s2:2m@\). Variance estimée de la pente : \(@L2_s2:2m@ \times [(\mathbf{X}^\top \mathbf{X})^{-1}]_{22} = @L2_s2:2m@ \times @L1_inv22:1m@ = @L2_var_pente:2m@\), d'où une erreur-type de \(@L2_se_pente:4m@\) et une statistique \(t = @L1_b1:1m@ / @L2_se_pente:4m@ \approx @L2_t:3m@\), à comparer à une loi de Student à 2 degrés de liberté sous (A4).

L'expérience suivante compare les MCO à un autre estimateur linéaire sans biais de la pente, la pente entre le premier et le dernier point, \((Y_n - Y_1)/(x_{n2} - x_{12})\). Les deux sont sans biais ; Gauss-Markov prédit que les MCO ont la plus petite variance.

```python-exec
rng = np.random.default_rng(0)
n, sigma = 20, 1.0
x2 = np.linspace(0, 1, n)
X = np.column_stack([np.ones(n), x2])
Y = X.dot(np.array([1.0, 2.0])) + sigma * rng.standard_normal((20000, n))   # une ligne par échantillon simulé
B = np.linalg.solve(X.T.dot(X), X.T.dot(Y.T)).T                           # MCO pour chaque échantillon
extremes = (Y[:, -1] - Y[:, 0]) / (x2[-1] - x2[0])
s2 = ((Y - B.dot(X.T)) ** 2).sum(axis=1) / (n - 2)
print("moyenne des pentes MCO       :", round(B[:, 1].mean(), 4), "(vraie pente : 2)")
print("moyenne des pentes extrêmes  :", round(extremes.mean(), 4))
print("variance MCO simulée         :", round(B[:, 1].var(), 4), "; formule sigma^2 [(X^T X)^-1]_22 :", round(sigma**2 * np.linalg.inv(X.T.dot(X))[1, 1], 4))
print("variance extrêmes simulée    :", round(extremes.var(), 4), "; formule 2 sigma^2 / (x_n - x_1)^2 :", round(2 * sigma**2 / (x2[-1] - x2[0]) ** 2, 4))
print("moyenne de sigma^2 chapeau   :", round(s2.mean(), 4), "(sigma^2 = 1)")
```

[[controle:gauss_markov]]

### Les violations, une par une

**Hétéroscédasticité** : (A3) tombe parce que les variances diffèrent, \(\mathrm{Var}(\boldsymbol{\varepsilon}) = \boldsymbol{\Sigma}\), matrice diagonale de coefficients \(\sigma_1^2, \dots, \sigma_n^2\) non tous égaux.

1. Les MCO restent sans biais : l'étape 1 n'utilise que (A1) et (A2).
2. Par l'étape 2, \(\mathrm{Var}(\hat{\boldsymbol{\beta}}) = \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \boldsymbol{\Sigma}\mathbf{X} (\mathbf{X}^\top \mathbf{X})^{-1}\) (forme « sandwich »), différente de \(\sigma^2 (\mathbf{X}^\top \mathbf{X})^{-1}\) sauf si \(\mathbf{X}^\top \boldsymbol{\Sigma}\mathbf{X} = \sigma^2 \mathbf{X}^\top \mathbf{X}\) pour un réel \(\sigma^2\) : les erreurs-types usuelles, les tests t et les intervalles sont faux.
3. Sens de l'erreur, en régression simple avec constante. Notons \(\bar{x}\) la moyenne des \(x_{i2}\) et \(S_{xx} = \sum_i (x_{i2} - \bar{x})^2\). Les équations normales donnent \(\hat{\beta}_2 = \sum_i (x_{i2} - \bar{x}) Y_i / S_{xx}\) (première équation : \(\hat{\beta}_1 = \bar{Y} - \hat{\beta}_2 \bar{x}\), où \(\bar{Y}\) est la moyenne des \(Y_i\) ; on la reporte dans la seconde), donc \(\hat{\beta}_2 - \beta_2 = \sum_i c_i \varepsilon_i\) avec \(c_i = (x_{i2} - \bar{x}) / S_{xx}\). Alors \(\mathrm{Var}(\hat{\beta}_2) = \sum_i c_i^2 \sigma_i^2 = \big(\sum_i w_i \sigma_i^2\big) / S_{xx}\), où les poids \(w_i = (x_{i2} - \bar{x})^2 / S_{xx}\) sont positifs et de somme 1. La formule usuelle remplace les \(\sigma_i^2\) par une variance commune, estimée par \(\hat{\sigma}^2\). Par l'étape 3 avec \(\boldsymbol{\Sigma}\) à la place de \(\sigma^2 \mathbf{I}\), \(\mathbb{E}[\hat{\sigma}^2] = \mathrm{tr}((\mathbf{I} - \mathbf{H})\boldsymbol{\Sigma}) / (n - 2) = \sum_i (1 - h_{ii})\,\sigma_i^2 / (n - 2)\), où les \(h_{ii}\), coefficients diagonaux de \(\mathbf{H}\), sont positifs et de somme 2 : quand \(n\) est grand devant 2 et qu'aucun \(h_{ii}\) n'est proche de 1, c'est presque la moyenne simple des \(\sigma_i^2\). La formule usuelle **sous-estime** alors la variance quand le bruit est plus fort loin de \(\bar{x}\) (moyenne des \(\sigma_i^2\) pondérée par les \(w_i\) supérieure à leur moyenne simple), et la surestime dans le cas contraire.
4. Gauss-Markov ne s'applique plus. Si les \(\sigma_i\) sont connus, les moindres carrés pondérés (MCP), qui minimisent \(\sum_i (y_i - (\mathbf{X}\mathbf{b})_i)^2 / \sigma_i^2\), sont le meilleur estimateur linéaire sans biais : ce sont les MCO du modèle transformé \(Y_i / \sigma_i = (\mathbf{X}\boldsymbol{\beta})_i / \sigma_i + \varepsilon_i / \sigma_i\), dont le bruit vérifie (A3) avec \(\sigma^2 = 1\), et Gauss-Markov s'y applique.
5. Si les \(\sigma_i\) sont inconnus : on garde les MCO et l'on remplace \(\boldsymbol{\Sigma}\) dans le sandwich par la matrice diagonale des \(e_i^2\) (erreurs-types de White) ; leur convergence quand \(n\) grandit, sous des conditions de moments, est admise.

```python-exec
rng = np.random.default_rng(1)
n = 200
x2 = rng.uniform(0, 1, n)
X = np.column_stack([np.ones(n), x2])
ecart = 0.1 + 4 * np.abs(x2 - 0.5)                  # écart-type du bruit, plus grand loin du centre
XtX_inv = np.linalg.inv(X.T.dot(X))
Y = 1 + 2 * x2 + ecart * rng.standard_normal((20000, n))
B = XtX_inv.dot(X.T.dot(Y.T)).T
R = Y - B.dot(X.T)
se_usuelle = np.sqrt((R**2).sum(axis=1) / (n - 2) * XtX_inv[1, 1])
viande = np.einsum("si,ij,ik->sjk", R**2, X, X)        # X^T diag(e_i^2) X pour chaque échantillon
se_white = np.sqrt(np.einsum("j,sjk,k->s", XtX_inv[1], viande, XtX_inv[1]))
print("moyenne des pentes            :", round(B[:, 1].mean(), 4), "(vraie pente : 2)")
print("écart-type réel des pentes    :", round(B[:, 1].std(), 4))
print("erreur-type usuelle, moyenne  :", round(se_usuelle.mean(), 4))
print("erreur-type de White, moyenne :", round(se_white.mean(), 4))
```

Lecture : la moyenne des pentes reste à la vraie valeur (sans biais), l'erreur-type usuelle est plus petite que l'écart-type réel (le bruit est plus fort aux extrémités, là où les poids \(w_i\) sont grands), et l'erreur-type de White s'en approche.

**Autocorrélation** : (A3) tombe parce que des bruits sont corrélés, \(\mathrm{Cov}(\varepsilon_i, \varepsilon_m) \ne 0\) pour certains \(i \ne m\), par exemple entre dates voisines d'une série temporelle.

1. Les MCO restent sans biais, et \(\mathrm{Var}(\hat{\boldsymbol{\beta}}) = \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top\) avec \(\boldsymbol{\Sigma} = \mathrm{Var}(\boldsymbol{\varepsilon})\) non diagonale : la formule usuelle est fausse.
2. Cas précis : régression simple avec constante, observations indexées par le temps, bruit stationnaire de type AR(1), \(\mathrm{Cov}(\varepsilon_i, \varepsilon_m) = \sigma^2 \phi^{\lvert i - m \rvert}\) avec \(\phi\) réel, \(\lvert \phi \rvert < 1\). Avec les \(c_i\) ci-dessus, \(\mathrm{Var}(\hat{\beta}_2) = \sum_{i, m} c_i c_m \sigma^2 \phi^{\lvert i - m \rvert}\) ; en regroupant les couples selon l'écart \(h = \lvert i - m \rvert\) : \(\mathrm{Var}(\hat{\beta}_2) = \dfrac{\sigma^2}{S_{xx}} \Big(1 + 2 \sum_{h \ge 1} \phi^h \rho_h\Big)\), où \(\rho_h = \sum_i (x_{i2} - \bar{x})(x_{i+h,2} - \bar{x}) / S_{xx}\) (somme sur \(i\) de 1 à \(n - h\)) est l'autocorrélation empirique du régresseur au retard \(h\).
3. Conséquence : si \(\phi > 0\) et si tous les \(\rho_h\) sont positifs (régresseur lui-même positivement autocorrélé), la vraie variance dépasse \(\sigma^2 / S_{xx}\) : la formule usuelle, même avec la vraie \(\sigma^2\), sous-estime la variance, et les tests rejettent trop. Ordre de grandeur : pour un régresseur AR(1) de coefficient \(\phi\) lui aussi et une longue série, \(\rho_h\) est proche de \(\phi^h\) (admis), et le facteur vaut environ \((1 + \phi^2)/(1 - \phi^2)\) ; pour \(\phi = @L2_ac_phi:1m@\), environ \(@L2_ac_rap_var:2m@\) sur la variance, soit \(@L2_ac_rap_se:2m@\) sur l'erreur-type.
4. Une source de cette autocorrélation en finance quantitative : des étiquettes qui se chevauchent. Un rendement sur 5 jours calculé chaque jour partage 4 rendements quotidiens avec celui du lendemain ; si les rendements quotidiens sont non corrélés et de même variance \(v\), leur covariance vaut \(4v\) pour une variance \(5v\), donc une corrélation de 4/5.
5. Remèdes : erreurs-types de Newey-West, qui pondèrent les autocovariances empiriques des \(e_i\) (convergence admise), ou une modélisation explicite de la dépendance.

```python-exec
rng = np.random.default_rng(2)
def ar1(nsim, n, phi):
    u = rng.standard_normal((nsim, n))
    z = np.empty((nsim, n))
    z[:, 0] = u[:, 0] / np.sqrt(1 - phi**2)           # loi stationnaire
    for t in range(1, n):
        z[:, t] = phi * z[:, t - 1] + u[:, t]
    return z
n, phi = 200, 0.8
x2 = ar1(1, n, phi)[0]
xc = x2 - x2.mean()
eps = ar1(20000, n, phi)
pentes = eps.dot(xc) / xc.dot(xc)                      # pente estimée moins la vraie pente
sigma2 = 1 / (1 - phi**2)                              # variance marginale du bruit
print("écart-type réel des pentes             :", round(pentes.std(), 4))
print("formule usuelle sqrt(sigma^2 / S_xx)   :", round(np.sqrt(sigma2 / xc.dot(xc)), 4))
rho_h = np.array([xc[:-h].dot(xc[h:]) for h in range(1, n)]) / xc.dot(xc)
facteur = 1 + 2 * np.sum(phi ** np.arange(1, n) * rho_h)
print("formule exacte de l'étape 2            :", round(np.sqrt(sigma2 / xc.dot(xc) * facteur), 4))
```

Lecture : l'écart-type réel dépasse la formule usuelle, et la formule exacte de l'étape 2, calculée avec les autocorrélations empiriques \(\rho_h\) de cette série, le retrouve. Le rapport observé dépend de la série tirée : sur 200 dates, les \(\rho_h\) empiriques peuvent être plus petits que \(\phi^h\), et l'ordre de grandeur de l'étape 3 ne vaut que pour une série longue.

**Variable omise.** Le vrai modèle est \(\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \gamma\,\mathbf{z} + \boldsymbol{\varepsilon}\), avec \(\mathbf{z} \in \mathbb{R}^n\) un vecteur fixe (la variable omise), \(\gamma\) un réel fixe et \(\mathbb{E}[\boldsymbol{\varepsilon}] = \mathbf{0}\) ; on régresse \(\mathbf{Y}\) sur \(\mathbf{X}\) seul. On note \(\boldsymbol{\delta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{z}\) les coefficients MCO de la régression de \(\mathbf{z}\) sur \(\mathbf{X}\).

1. \(\mathbb{E}[\hat{\boldsymbol{\beta}}] = \mathbf{A}(\mathbf{X}\boldsymbol{\beta} + \gamma\,\mathbf{z}) = \boldsymbol{\beta} + \gamma\,\boldsymbol{\delta}\). Le biais du coefficient \(j\) vaut \(\gamma\,\delta_j\) : il a le signe du produit de l'effet de la variable omise et de sa pente sur la variable \(j\).
2. Le biais est nul si et seulement si \(\gamma = 0\) ou \(\mathbf{X}^\top \mathbf{z} = \mathbf{0}\) (variable omise orthogonale aux colonnes ; avec constante, de moyenne nulle et non corrélée empiriquement aux autres variables).
3. Ce biais ne disparaît pas quand \(n\) grandit tant que \(\boldsymbol{\delta}\) ne tend pas vers 0 : il ne se corrige pas avec plus de données, seulement en incluant \(\mathbf{z}\).

```python-exec
rng = np.random.default_rng(3)
n = 100
x2 = rng.standard_normal(n)
z = 0.5 * x2 + rng.standard_normal(n)                 # z corrélée à x2
X = np.column_stack([np.ones(n), x2])
delta = np.linalg.lstsq(X, z, rcond=None)[0][1]        # pente de z sur (1, x2) dans l'échantillon
Y = 1 + 2 * x2 + 3 * z + rng.standard_normal((20000, n))
pentes = np.linalg.lstsq(X, Y.T, rcond=None)[0][1]
print("delta (pente de z sur x2)      :", round(delta, 4))
print("moyenne des pentes estimées    :", round(pentes.mean(), 4))
print("prédiction 2 + 3 * delta       :", round(2 + 3 * delta, 4))
```

**Endogénéité.** Plan aléatoire (la matrice \(\mathbf{X}\) est alors aléatoire), et (A2) conditionnelle tombe : \(\mathbb{E}[\boldsymbol{\varepsilon} \mid \mathbf{X}] \ne \mathbf{0}\) ; le cas qui pose problème est celui d'une variable explicative corrélée au bruit de la même observation. Causes classiques : variable omise corrélée aux régresseurs, erreur de mesure sur un régresseur, causalité dans les deux sens. Conséquence, démontrée ci-dessous sur l'erreur de mesure : un biais qui ne disparaît pas quand \(n\) grandit (estimateur non convergent).

Exemple démontré, l'erreur de mesure. Pour \(i = 1, \dots, n\), indépendamment : \(Y_i = \beta Z_i + \varepsilon_i\), mais on observe \(W_i = Z_i + U_i\) au lieu de \(Z_i\). \(Z_i\), \(U_i\), \(\varepsilon_i\) sont indépendantes, centrées, de variances \(v_Z > 0\), \(v_U \ge 0\) et \(\sigma^2\) ; \(\beta\) est un réel fixe. La pente MCO sans constante \(\hat{\beta} = \sum_i W_i Y_i / \sum_i W_i^2\) converge presque sûrement vers \(\beta\,v_Z / (v_Z + v_U)\).

1. \(\mathbb{E}[W_i Y_i] = \mathbb{E}[(Z_i + U_i)(\beta Z_i + \varepsilon_i)] = \beta\,v_Z\) : les trois autres produits sont des produits de variables indépendantes centrées, d'espérance nulle.
2. \(\mathbb{E}[W_i^2] = v_Z + 2\,\mathbb{E}[Z_i U_i] + v_U = v_Z + v_U\).
3. Par la loi forte des grands nombres (admise), \(\tfrac1n \sum_i W_i Y_i\) et \(\tfrac1n \sum_i W_i^2\) convergent presque sûrement vers ces espérances ; le quotient converge vers le quotient des limites, car la limite du dénominateur est non nulle.
4. Lecture : \(Y_i = \beta W_i + (\varepsilon_i - \beta U_i)\), et le bruit composite est corrélé au régresseur observé : \(\mathrm{Cov}(W_i, \varepsilon_i - \beta U_i) = -\beta\,v_U \ne 0\) si \(\beta \ne 0\) et \(v_U > 0\). La pente est tirée vers 0 (biais d'atténuation). Avec \(\beta = 2\), \(v_Z = 1\), \(v_U = 0{,}25\) : limite \(@L2_att:1m@\).
5. Remède : variables instrumentales (non traité ici).

```python-exec
rng = np.random.default_rng(4)
n = 200000
Z = rng.standard_normal(n)                     # vraie variable, variance 1
W = Z + 0.5 * rng.standard_normal(n)           # mesurée avec une erreur de variance 0,25
Y = 2 * Z + rng.standard_normal(n)
print("pente de Y sur la variable mesurée :", round(W.dot(Y) / W.dot(W), 4))
print("limite 2 x 1 / (1 + 0,25)          :", 2 / 1.25)
```

**Multicolinéarité.** Parfaite, c'est la violation de (A1) : les coefficients ne sont pas identifiables (infinité de minimiseurs), mais les valeurs ajustées restent uniques (partie 1). Imparfaite (colonnes presque liées) : (A1) à (A3) tiennent, donc les MCO restent sans biais et BLUE, mais leur variance est grande. Formule exacte :

Soit \(j\) l'indice d'une colonne, \(\mathbf{X}_{-j}\) la matrice des autres colonnes, supposée contenir la constante, et \(\tilde{\mathbf{x}}_j\) le résidu de la régression MCO de \(\mathbf{x}_j\) sur \(\mathbf{X}_{-j}\). On note \(\mathrm{SCT}_j = \sum_i (x_{ij} - \bar{x}_j)^2\), où \(\bar{x}_j\) est la moyenne de la colonne \(j\), et \(R_j^2 = 1 - \lVert \tilde{\mathbf{x}}_j \rVert^2 / \mathrm{SCT}_j\), le coefficient de détermination de cette régression auxiliaire (partie 3). Alors, sous (A1) à (A3) :

\(\displaystyle \mathrm{Var}(\hat{\beta}_j) = \frac{\sigma^2}{\lVert \tilde{\mathbf{x}}_j \rVert^2} = \frac{\sigma^2}{\mathrm{SCT}_j\,(1 - R_j^2)}.\)

Le facteur \(1/(1 - R_j^2)\) est le **facteur d'inflation de la variance** (VIF, variance inflation factor) : la variance est multipliée par ce facteur par rapport à une colonne de même dispersion non corrélée aux autres.

Preuve (théorème de Frisch-Waugh) :

1. Soit \(\mathbf{P}\) la matrice de projection orthogonale sur le complément orthogonal de \(\mathrm{Im}(\mathbf{X}_{-j})\), c'est-à-dire \(\mathbf{I}\) moins la matrice chapeau de \(\mathbf{X}_{-j}\) ; elle est symétrique, idempotente (partie 1), et \(\tilde{\mathbf{x}}_j = \mathbf{P}\mathbf{x}_j\), \(\mathbf{P}\mathbf{X}_{-j} = \mathbf{0}\).
2. On écrit \(\mathbf{y} = \mathbf{X}_{-j}\hat{\mathbf{b}}_{-j} + \hat{b}_j \mathbf{x}_j + \mathbf{e}\), décomposition des MCO sur toutes les colonnes. On applique \(\mathbf{P}\) : \(\mathbf{P}\mathbf{y} = \hat{b}_j \tilde{\mathbf{x}}_j + \mathbf{P}\mathbf{e}\), et \(\mathbf{P}\mathbf{e} = \mathbf{e}\) car \(\mathbf{e}\) est orthogonal aux colonnes de \(\mathbf{X}_{-j}\).
3. On multiplie à gauche par \(\tilde{\mathbf{x}}_j^\top\) : \(\tilde{\mathbf{x}}_j^\top \mathbf{e} = \mathbf{x}_j^\top \mathbf{P}\mathbf{e} = \mathbf{x}_j^\top \mathbf{e} = 0\) et \(\tilde{\mathbf{x}}_j^\top \mathbf{P}\mathbf{y} = \mathbf{x}_j^\top \mathbf{P}^2 \mathbf{y} = \tilde{\mathbf{x}}_j^\top \mathbf{y}\). D'où \(\hat{b}_j = \tilde{\mathbf{x}}_j^\top \mathbf{y} / \lVert \tilde{\mathbf{x}}_j \rVert^2\) : le coefficient \(j\) est la pente de \(\mathbf{y}\) sur la partie de \(\mathbf{x}_j\) que les autres colonnes n'expliquent pas.
4. Dans le modèle, \(\hat{\beta}_j = \tilde{\mathbf{x}}_j^\top \mathbf{Y} / \lVert \tilde{\mathbf{x}}_j \rVert^2\), et l'étape 2 de la preuve générale donne \(\mathrm{Var}(\hat{\beta}_j) = \sigma^2 \lVert \tilde{\mathbf{x}}_j \rVert^2 / \lVert \tilde{\mathbf{x}}_j \rVert^4\). Comme \(\mathbf{X}_{-j}\) contient la constante, \(\mathrm{SCT}_j\) est la somme des carrés totale de la régression auxiliaire et \(\lVert \tilde{\mathbf{x}}_j \rVert^2 = \mathrm{SCT}_j (1 - R_j^2)\) par définition de \(R_j^2\). En comparant avec l'énoncé 2, on obtient aussi \([(\mathbf{X}^\top \mathbf{X})^{-1}]_{jj} = 1/\lVert \tilde{\mathbf{x}}_j \rVert^2\).

Exemple : \(R_j^2 = @L2_vif_r2:2m@\) donne un VIF de \(@L2_vif:0m@\) et une erreur-type multipliée par \(@L2_vif_se:0m@\).

```python-exec
rng = np.random.default_rng(5)
n = 100
x3 = rng.standard_normal(n)
for rho in [0.0, 0.9, 0.99]:
    x2 = rho * x3 + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x2, x3])
    autres = X[:, [0, 2]]
    tilde = x2 - autres.dot(np.linalg.lstsq(autres, x2, rcond=None)[0])
    R2_j = 1 - tilde.dot(tilde) / ((x2 - x2.mean()) ** 2).sum()
    print(f"rho={rho:4.2f}  R2_j={R2_j:.4f}  VIF={1 / (1 - R2_j):7.2f}  [(X^T X)^-1]_22={np.linalg.inv(X.T.dot(X))[1, 1]:.5f}  1/norme(tilde)^2={1 / tilde.dot(tilde):.5f}")
```

[[controle:omise]]

### Pièges

1. « Hétéroscédasticité ou autocorrélation implique un biais » : faux, seules les variances et les tests sont faux ; c'est (A2) qui garantit l'absence de biais.
2. « La multicolinéarité biaise les coefficients » : faux, elle les rend imprécis. Les valeurs ajustées aléatoires \(\hat{\mathbf{Y}} = \mathbf{H}\mathbf{Y}\), elles, n'en souffrent pas : \(\mathrm{Var}(\hat{\mathbf{Y}}) = \sigma^2 \mathbf{H}\mathbf{H}^\top = \sigma^2 \mathbf{H}\), de trace \(\sigma^2 p\) quelle que soit la colinéarité.
3. Gauss-Markov ne dit pas que les MCO ont la plus petite erreur quadratique : la comparaison est limitée aux estimateurs **linéaires et sans biais**. Un estimateur biaisé comme Ridge peut faire mieux (partie 6).
4. La normalité (A4) ne sert ni au caractère sans biais ni à Gauss-Markov ; elle sert aux lois exactes des tests en petit échantillon.
5. Ajouter des données ne corrige ni la variable omise ni l'endogénéité.

## Partie 3 : R² et R² ajusté

### Le problème

Mesurer la part de la variabilité de \(\mathbf{y}\) que le modèle restitue, et comparer des modèles qui n'ont pas le même nombre de colonnes.

### Définitions

\(\bar{y} = \tfrac1n \sum_i y_i\). Somme des carrés totale \(\mathrm{SCT} = \lVert \mathbf{y} - \bar{y}\,\mathbf{1} \rVert^2\), expliquée \(\mathrm{SCE} = \lVert \hat{\mathbf{y}} - \bar{y}\,\mathbf{1} \rVert^2\), résiduelle \(\mathrm{SCR} = \lVert \mathbf{e} \rVert^2\). Coefficient de détermination \(R^2 = 1 - \mathrm{SCR}/\mathrm{SCT}\) et R² ajusté \(R^2_{\mathrm{aj}} = 1 - \dfrac{\mathrm{SCR}/(n - p)}{\mathrm{SCT}/(n - 1)}\), où \(p\) compte la constante. On suppose \(\mathrm{SCT} > 0\).

### L'intuition

\(\mathrm{SCT}\) est l'erreur du modèle le plus bête, la constante \(\bar{y}\) ; \(\mathrm{SCR}\) est l'erreur du modèle. R² mesure la fraction de l'erreur du modèle bête que le modèle supprime. Le R² ajusté remplace les deux sommes par des estimateurs sans biais de variance, ce qui fait payer chaque colonne ajoutée.

### L'énoncé

1. Avec une constante dans \(\mathbf{X}\) : \(\mathrm{SCT} = \mathrm{SCE} + \mathrm{SCR}\), donc \(0 \le R^2 \le 1\) et \(R^2 = \mathrm{SCE}/\mathrm{SCT}\) ; de plus \(R^2\) est le carré de la corrélation empirique entre \(\mathbf{y}\) et \(\hat{\mathbf{y}}\).
2. Ajouter une colonne ne fait jamais décroître R² sur l'échantillon d'apprentissage. Précisément, si \(\mathbf{x}_{\mathrm{new}} \notin \mathrm{Im}(\mathbf{X})\) et \(\tilde{\mathbf{x}} = (\mathbf{I} - \mathbf{H})\mathbf{x}_{\mathrm{new}}\) est son résidu sur les anciennes colonnes, alors \(\mathrm{SCR}_{\mathrm{nouveau}} = \mathrm{SCR}_{\mathrm{ancien}} - c^2 \lVert \tilde{\mathbf{x}} \rVert^2\), où \(c = \tilde{\mathbf{x}}^\top \mathbf{y} / \lVert \tilde{\mathbf{x}} \rVert^2\) est le coefficient de la nouvelle colonne dans le nouveau modèle.
3. Le R² ajusté augmente si et seulement si \(t^2 > 1\), où \(t\) est la statistique t de la nouvelle colonne dans le nouveau modèle (supposé avec \(n > p + 1\) et \(\mathrm{SCR}_{\mathrm{nouveau}} > 0\)).
4. R² peut être négatif : sans constante, ou hors de l'échantillon d'apprentissage.

### La preuve

1. \(\mathbf{y} - \bar{y}\,\mathbf{1} = (\hat{\mathbf{y}} - \bar{y}\,\mathbf{1}) + \mathbf{e}\). Le premier terme est dans \(\mathrm{Im}(\mathbf{X})\), puisque \(\mathbf{1}\) en fait partie ; \(\mathbf{e}\) est orthogonal à \(\mathrm{Im}(\mathbf{X})\). Pythagore donne \(\mathrm{SCT} = \mathrm{SCE} + \mathrm{SCR}\). Pour la corrélation : la moyenne des \(\hat{y}_i\) est \(\bar{y}\) (partie 1, énoncé 5), et \((\mathbf{y} - \bar{y}\,\mathbf{1})^\top (\hat{\mathbf{y}} - \bar{y}\,\mathbf{1}) = (\hat{\mathbf{y}} - \bar{y}\,\mathbf{1} + \mathbf{e})^\top (\hat{\mathbf{y}} - \bar{y}\,\mathbf{1}) = \mathrm{SCE}\), d'où un carré de corrélation \(\mathrm{SCE}^2 / (\mathrm{SCT} \cdot \mathrm{SCE}) = \mathrm{SCE}/\mathrm{SCT}\) (si \(\mathrm{SCE} > 0\)).
2. On note \(\mathbf{X}^{+}\) la matrice augmentée de la colonne \(\mathbf{x}_{\mathrm{new}} = \mathbf{H}\mathbf{x}_{\mathrm{new}} + \tilde{\mathbf{x}}\) ; son image est la somme orthogonale de \(\mathrm{Im}(\mathbf{X})\) et de la droite engendrée par \(\tilde{\mathbf{x}}\), qui est non nul et orthogonal à \(\mathrm{Im}(\mathbf{X})\). Le candidat \(\mathbf{q} = \mathbf{H}\mathbf{y} + c\,\tilde{\mathbf{x}}\) est dans cette image, et \(\mathbf{y} - \mathbf{q} = \mathbf{e} - c\,\tilde{\mathbf{x}}\) est orthogonal à \(\mathrm{Im}(\mathbf{X})\) (comme \(\mathbf{e}\) et \(\tilde{\mathbf{x}}\)) et à \(\tilde{\mathbf{x}}\) dès que \(\tilde{\mathbf{x}}^\top \mathbf{e} = c\,\lVert \tilde{\mathbf{x}} \rVert^2\). Or \(\tilde{\mathbf{x}}^\top \mathbf{e} = \mathbf{x}_{\mathrm{new}}^\top (\mathbf{I} - \mathbf{H})^2 \mathbf{y} = \tilde{\mathbf{x}}^\top \mathbf{y}\), ce qui est la définition de \(c\). Donc \(\mathbf{q}\) est la projection, et le nouveau résidu est \(\mathbf{e} - c\,\tilde{\mathbf{x}}\) ; en développant, \(\lVert \mathbf{e} - c\,\tilde{\mathbf{x}} \rVert^2 = \lVert \mathbf{e} \rVert^2 - 2c\,\tilde{\mathbf{x}}^\top \mathbf{e} + c^2 \lVert \tilde{\mathbf{x}} \rVert^2 = \lVert \mathbf{e} \rVert^2 - c^2 \lVert \tilde{\mathbf{x}} \rVert^2\). \(\mathrm{SCT}\) ne change pas : R² ne décroît pas, et il croît strictement si \(c \ne 0\). Enfin \(\mathbf{q} = \mathbf{X}(\cdots) + c\,\mathbf{x}_{\mathrm{new}}\), donc \(c\) est bien le nouveau coefficient (c'est Frisch-Waugh, partie 2).
3. Par Frisch-Waugh, la variance estimée du nouveau coefficient est \(\hat{\sigma}_+^2 / \lVert \tilde{\mathbf{x}} \rVert^2\), avec \(\hat{\sigma}_+^2 = \mathrm{SCR}_{\mathrm{nouveau}} / (n - p - 1)\) ; donc \(t^2 = c^2 \lVert \tilde{\mathbf{x}} \rVert^2 / \hat{\sigma}_+^2\), et l'étape 2 s'écrit \(\mathrm{SCR}_{\mathrm{ancien}} = \mathrm{SCR}_{\mathrm{nouveau}} \big(1 + t^2/(n - p - 1)\big)\). Le R² ajusté augmente si et seulement si \(\mathrm{SCR}_{\mathrm{nouveau}}/(n - p - 1) < \mathrm{SCR}_{\mathrm{ancien}}/(n - p)\), c'est-à-dire \((n - p)\,\mathrm{SCR}_{\mathrm{nouveau}} < (n - p - 1)\,\mathrm{SCR}_{\mathrm{nouveau}} + t^2\,\mathrm{SCR}_{\mathrm{nouveau}}\), soit \(1 < t^2\).
4. Sans constante, la décomposition de l'étape 1 n'a plus lieu. Exemple : \(\mathbf{x} = (1, 2, 3)\), \(\mathbf{y} = (3, 2, 1)\), régression sans constante : pente \(@L3_nc_xy@/@L3_nc_xx@ = \tfrac{@L3_nc_b_num@}{@L3_nc_b_den@}\), \(\mathrm{SCR} = \tfrac{@L3_nc_scr_num@}{@L3_nc_scr_den@}\), \(\mathrm{SCT} = @L3_nc_sct@\), donc \(R^2 = \tfrac{@L3_nc_r2_num@}{@L3_nc_r2_den@} \approx @L3_nc_r2:2m@\). Hors échantillon, avec \(R^2_{\mathrm{test}} = 1 - \sum_i (y_i - \hat{y}_i)^2 / \sum_i (y_i - \bar{y}_{\mathrm{test}})^2\) sur les points de test, \(R^2_{\mathrm{test}} < 0\) exactement quand le modèle prédit moins bien que la moyenne des réponses de test, que rien n'interdit.

### Exemple chiffré

Exemple à quatre points de la partie 1 : \(\bar{y} = @L3_ybar:2m@\), \(\mathrm{SCT} = @L3_sct:2m@\), \(\mathrm{SCE} = @L3_sce:2m@\), \(\mathrm{SCR} = @L3_scr:1m@\), et l'on vérifie \(@L3_sce:2m@ + @L3_scr:1m@ = @L3_sct:2m@\). Donc \(R^2 = 1 - @L3_scr:1m@/@L3_sct:2m@ \approx @L3_r2:3m@\) et \(R^2_{\mathrm{aj}} = 1 - (@L3_scr:1m@/2)/(@L3_sct:2m@/3) \approx @L3_r2aj:3m@\).

L'expérience suivante ajoute à une régression simple (30 observations d'apprentissage, 1000 de test) des colonnes de bruit pur, indépendantes de tout.

```python-exec
rng = np.random.default_rng(6)
x_app = rng.standard_normal(30)
y_app = 1 + x_app + rng.standard_normal(30)
x_tst = rng.standard_normal(1000)
y_tst = 1 + x_tst + rng.standard_normal(1000)
bruit_app = rng.standard_normal((30, 25))
bruit_tst = rng.standard_normal((1000, 25))
for k in [0, 5, 15, 25]:
    Xa = np.column_stack([np.ones(30), x_app, bruit_app[:, :k]])
    Xt = np.column_stack([np.ones(1000), x_tst, bruit_tst[:, :k]])
    b = np.linalg.lstsq(Xa, y_app, rcond=None)[0]
    p = Xa.shape[1]
    R2 = r2_score(y_app, Xa.dot(b))
    R2aj = 1 - (1 - R2) * (30 - 1) / (30 - p)
    print(f"{k:2d} colonnes de bruit, p={p:2d} : R2={R2:.3f}  R2 ajusté={R2aj:.3f}  R2 test={r2_score(y_tst, Xt.dot(b)):.3f}")
```

Lecture : le R² d'apprentissage croît à chaque ajout (énoncé 2) ; le R² ajusté ne suit pas ; le R² de test devient négatif quand le nombre de colonnes approche le nombre d'observations. `r2_score` de scikit-learn calcule \(1 - \mathrm{SCR}/\mathrm{SCT}\) avec la moyenne des réponses passées en premier argument : sur des données de test, c'est \(R^2_{\mathrm{test}}\).

### Cas limites

- \(R^2 = 1\) si et seulement si \(\mathbf{e} = \mathbf{0}\) (par exemple \(n = p\) et \(\mathbf{X}\) inversible) ; \(R^2 = 0\) si et seulement si \(\hat{\mathbf{y}} = \bar{y}\,\mathbf{1}\).
- Régression simple avec constante : R² est le carré de la corrélation empirique entre \(\mathbf{x}_2\) et \(\mathbf{y}\), puisque \(\hat{\mathbf{y}}\) est une fonction affine de \(\mathbf{x}_2\) de pente non nulle (si la pente est nulle, les deux valent 0).

### Pièges

1. Comparer des modèles par leur R² d'apprentissage favorise toujours le plus gros : utiliser le R² ajusté, un critère pénalisé ou une erreur hors échantillon.
2. \(R^2_{\mathrm{aj}}\) peut être négatif, et n'est pas le carré de quoi que ce soit.
3. Un R² élevé ne dit rien des hypothèses (A2) ou (A3), ni de causalité.

[[controle:r2]]

## Partie 4 : Ridge

### Le problème

Quand \(\mathbf{X}^\top \mathbf{X}\) n'est pas inversible (\(p > n\), colonnes liées), les MCO n'ont pas de solution unique ; quand elle est presque singulière (colonnes presque liées), leur variance explose (VIF, partie 2). Idée : pénaliser la taille des coefficients.

### Définitions

Pour un réel \(\lambda > 0\) fixé, le critère Ridge est \(r_\lambda(\mathbf{b}) = \tfrac12 \lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2 + \tfrac{\lambda}{2} \lVert \mathbf{b} \rVert^2\). Les facteurs \(\tfrac12\) ne changent pas le minimiseur ; `Ridge(alpha=a)` de scikit-learn minimise \(\lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2 + a\,\lVert \mathbf{b} \rVert^2\), qui a le même minimiseur pour \(a = \lambda\) (multiplier \(r_\lambda\) par 2).

**Décomposition en valeurs singulières** (SVD, singular value decomposition), d'existence admise : si \(\mathbf{X}\) est de rang \(q \ge 1\), \(\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^\top\), où \(\mathbf{U}\) (\(n \times q\)) a des colonnes orthonormées \(\mathbf{u}_1, \dots, \mathbf{u}_q\), \(\mathbf{V}\) (\(p \times q\)) des colonnes orthonormées \(\mathbf{v}_1, \dots, \mathbf{v}_q\), et \(\mathbf{D}\) est diagonale de coefficients \(d_1 \ge \dots \ge d_q > 0\), les valeurs singulières. Comme \(\mathbf{U}^\top \mathbf{U} = \mathbf{I}_q\) : \(\mathbf{X}^\top \mathbf{X} = \mathbf{V}\mathbf{D}^2 \mathbf{V}^\top\), donc \(\mathbf{X}^\top \mathbf{X}\,\mathbf{v}_j = d_j^2\,\mathbf{v}_j\) et \(\mathbf{X}\mathbf{v}_j = d_j\,\mathbf{u}_j\) ; les autres valeurs propres de \(\mathbf{X}^\top \mathbf{X}\), s'il y en a (\(q < p\)), sont nulles.

**Conditionnement.** Pour une matrice symétrique définie positive \(\mathbf{M}\), \(\kappa(\mathbf{M})\) est le rapport de sa plus grande à sa plus petite valeur propre. Il mesure de combien une erreur relative sur le second membre de \(\mathbf{M}\mathbf{b} = \mathbf{c}\) peut être amplifiée sur la solution (au plus d'un facteur \(\kappa\), admis). Pour \(\mathbf{X}\) de rang \(p\), on pose \(\kappa(\mathbf{X}) = d_1/d_p\).

### L'intuition

Les données déterminent mal certaines directions de \(\mathbb{R}^p\) : celles où les colonnes varient peu ensemble, les \(\mathbf{v}_j\) de petite valeur singulière. Les MCO divisent par \(d_j\) dans ces directions et amplifient le bruit. Ridge ajoute \(\lambda\) à chaque \(d_j^2\) : les directions bien déterminées (\(d_j^2\) grand devant \(\lambda\)) sont presque intactes, les directions mal déterminées sont écrasées.

### L'énoncé

1. **Forme fermée.** Pour tout \(\lambda > 0\), quels que soient \(n\), \(p\) et le rang de \(\mathbf{X}\), \(\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}\) est définie positive, donc inversible, et \(r_\lambda\) a un unique minimiseur \(\hat{\boldsymbol{\beta}}_\lambda = (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}\).
2. **Lecture par la SVD.** \(\displaystyle \hat{\boldsymbol{\beta}}_\lambda = \sum_{j=1}^q \frac{d_j}{d_j^2 + \lambda} (\mathbf{u}_j^\top \mathbf{y})\,\mathbf{v}_j\) et \(\displaystyle \mathbf{X}\hat{\boldsymbol{\beta}}_\lambda = \sum_{j=1}^q \frac{d_j^2}{d_j^2 + \lambda} (\mathbf{u}_j^\top \mathbf{y})\,\mathbf{u}_j\). Pour \(q = p\), les MCO correspondent à des facteurs égaux à 1 : Ridge multiplie la composante des valeurs ajustées selon \(\mathbf{u}_j\) par \(d_j^2/(d_j^2 + \lambda)\), d'autant plus petit que \(d_j\) est petit.
3. **Valeurs propres et conditionnement.** \(\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}\) a les mêmes vecteurs propres que \(\mathbf{X}^\top \mathbf{X}\), et ses valeurs propres sont celles de \(\mathbf{X}^\top \mathbf{X}\) augmentées de \(\lambda\). Si \(\mu_{\max}\) et \(\mu_{\min}\) sont la plus grande et la plus petite valeur propre de \(\mathbf{X}^\top \mathbf{X}\) (\(\mu_{\min} = 0\) possible), \(\kappa(\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}) = (\mu_{\max} + \lambda)/(\mu_{\min} + \lambda)\), fini, strictement décroissant en \(\lambda\) si \(\mu_{\min} < \mu_{\max}\), et de limite 1. Pour \(\mathbf{X}\) de rang \(p\), \(\kappa(\mathbf{X}^\top \mathbf{X}) = \kappa(\mathbf{X})^2\).
4. **Interprétation bayésienne.** Si \(\mathbf{B}\) est un vecteur aléatoire de loi a priori \(\mathcal{N}(\mathbf{0}, \tau^2 \mathbf{I})\), et si, sachant \(\mathbf{B} = \mathbf{b}\), \(\mathbf{Y}\) suit \(\mathcal{N}(\mathbf{X}\mathbf{b}, \sigma^2 \mathbf{I})\) (\(\sigma, \tau > 0\) fixés), alors la loi a posteriori de \(\mathbf{B}\) sachant \(\mathbf{Y} = \mathbf{y}\) est gaussienne, de moyenne et de mode \(\hat{\boldsymbol{\beta}}_\lambda\) avec \(\lambda = \sigma^2/\tau^2\).
5. **Interprétation géométrique.** \(\hat{\boldsymbol{\beta}}_\lambda\) minimise aussi \(\lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2\) sous la contrainte \(\lVert \mathbf{b} \rVert^2 \le t\), avec \(t = \lVert \hat{\boldsymbol{\beta}}_\lambda \rVert^2\) : c'est le point où les ellipsoïdes de niveau de \(s\) touchent la boule.
6. **Ridge ne met jamais un coefficient exactement à zéro, aux conditions suivantes.** Fixons \(\lambda > 0\) et \(j\). Il existe un vecteur fixe \(\mathbf{w}_j \in \mathbb{R}^n\) tel que \(\hat{\beta}_{\lambda, j} = \mathbf{w}_j^\top \mathbf{y}\) pour tout \(\mathbf{y}\), et \(\mathbf{w}_j = \mathbf{0}\) si et seulement si la colonne \(\mathbf{x}_j\) est nulle. Donc : si \(\mathbf{x}_j = \mathbf{0}\), le coefficient est nul pour tout \(\mathbf{y}\) ; sinon, il est nul exactement quand \(\mathbf{y}\) appartient à l'hyperplan \(\{\mathbf{v} : \mathbf{w}_j^\top \mathbf{v} = 0\}\), ce qui a probabilité 0 dès que \(\mathbf{Y}\) a une densité sur \(\mathbb{R}^n\), par exemple sous (A4). Enfin le vecteur entier \(\hat{\boldsymbol{\beta}}_\lambda\) est nul si et seulement si \(\mathbf{X}^\top \mathbf{y} = \mathbf{0}\), quel que soit \(\lambda\) : il n'existe pas de seuil de \(\lambda\) qui annule les coefficients, contrairement au Lasso.

### La preuve

1. **Forme fermée.** Pour \(\mathbf{v} \ne \mathbf{0}\) : \(\mathbf{v}^\top (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})\mathbf{v} = \lVert \mathbf{X}\mathbf{v} \rVert^2 + \lambda \lVert \mathbf{v} \rVert^2 > 0\) : définie positive, donc noyau nul, donc inversible. En divisant (D) par 2 et avec \(\tfrac{\lambda}{2}\lVert \mathbf{b} + \mathbf{h} \rVert^2 = \tfrac{\lambda}{2}\lVert \mathbf{b} \rVert^2 + \lambda\,\mathbf{h}^\top \mathbf{b} + \tfrac{\lambda}{2}\lVert \mathbf{h} \rVert^2\) : \(r_\lambda(\mathbf{b} + \mathbf{h}) = r_\lambda(\mathbf{b}) - \mathbf{h}^\top \big(\mathbf{X}^\top \mathbf{y} - (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})\mathbf{b}\big) + \tfrac12 \lVert \mathbf{X}\mathbf{h} \rVert^2 + \tfrac{\lambda}{2} \lVert \mathbf{h} \rVert^2\). Pour \(\mathbf{b} = \hat{\boldsymbol{\beta}}_\lambda\), le terme linéaire est nul et le reste est strictement positif pour \(\mathbf{h} \ne \mathbf{0}\) : minimiseur global unique.
2. **SVD.** \((\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})\mathbf{v}_j = (d_j^2 + \lambda)\mathbf{v}_j\). Appliquée au vecteur proposé \(\mathbf{b}^\star = \sum_j \frac{d_j}{d_j^2 + \lambda} (\mathbf{u}_j^\top \mathbf{y})\mathbf{v}_j\), la matrice donne \(\sum_j d_j (\mathbf{u}_j^\top \mathbf{y})\mathbf{v}_j = \mathbf{V}\mathbf{D}\mathbf{U}^\top \mathbf{y} = \mathbf{X}^\top \mathbf{y}\) : \(\mathbf{b}^\star\) est la solution, unique par l'étape 1. Avec \(\mathbf{X}\mathbf{v}_j = d_j \mathbf{u}_j\), on obtient les valeurs ajustées. Pour \(q = p\) et \(\lambda = 0\), la même vérification vaut avec \(\mathbf{X}^\top \mathbf{X}\) inversible : facteurs \(1/d_j\) sur les coefficients, 1 sur les valeurs ajustées.
3. **Conditionnement.** Si \(\mathbf{X}^\top \mathbf{X}\mathbf{v} = \mu\,\mathbf{v}\), alors \((\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})\mathbf{v} = (\mu + \lambda)\mathbf{v}\), et toutes les valeurs propres sont obtenues ainsi (base orthonormée de vecteurs propres de la matrice symétrique \(\mathbf{X}^\top \mathbf{X}\), théorème spectral admis). La dérivée de \(\lambda \mapsto (\mu_{\max} + \lambda)/(\mu_{\min} + \lambda)\) vaut \((\mu_{\min} - \mu_{\max})/(\mu_{\min} + \lambda)^2 < 0\). Les valeurs propres de \(\mathbf{X}^\top \mathbf{X}\) étant les \(d_j^2\), \(\kappa(\mathbf{X}^\top \mathbf{X}) = d_1^2/d_p^2\).
4. **Bayes.** Par la formule de Bayes, la densité a posteriori est proportionnelle à \(\exp\big(-\lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2/(2\sigma^2)\big)\exp\big(-\lVert \mathbf{b} \rVert^2/(2\tau^2)\big) = \exp\big(-r_\lambda(\mathbf{b})/\sigma^2\big)\) avec \(\lambda = \sigma^2/\tau^2\). Notons \(\mathbf{M} = \mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}\). En complétant le carré avec \(\mathbf{M}\hat{\boldsymbol{\beta}}_\lambda = \mathbf{X}^\top \mathbf{y}\) : \(2\,r_\lambda(\mathbf{b}) = \mathbf{b}^\top \mathbf{M}\mathbf{b} - 2\,\mathbf{b}^\top \mathbf{M}\hat{\boldsymbol{\beta}}_\lambda + \lVert \mathbf{y} \rVert^2 = (\mathbf{b} - \hat{\boldsymbol{\beta}}_\lambda)^\top \mathbf{M}(\mathbf{b} - \hat{\boldsymbol{\beta}}_\lambda) + \text{constante}\). La densité a posteriori est donc proportionnelle à celle de \(\mathcal{N}(\hat{\boldsymbol{\beta}}_\lambda, \sigma^2 \mathbf{M}^{-1})\), dont la moyenne et le mode valent \(\hat{\boldsymbol{\beta}}_\lambda\).
5. **Contrainte.** Soit \(\mathbf{b}\) avec \(\lVert \mathbf{b} \rVert^2 \le t\). Par optimalité de \(\hat{\boldsymbol{\beta}}_\lambda\) pour \(r_\lambda\) : \(\tfrac12 \lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2 \ge \tfrac12 \lVert \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}_\lambda \rVert^2 + \tfrac{\lambda}{2}(t - \lVert \mathbf{b} \rVert^2) \ge \tfrac12 \lVert \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}_\lambda \rVert^2\), la dernière inégalité car \(\lambda > 0\) et \(t - \lVert \mathbf{b} \rVert^2 \ge 0\).
6. **Zéros.** On note \(\mathbf{k}_j\) le \(j\)-ième vecteur de la base canonique de \(\mathbb{R}^p\) et \(\mathbf{M} = \mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}\). \(\hat{\beta}_{\lambda, j} = \mathbf{k}_j^\top \mathbf{M}^{-1} \mathbf{X}^\top \mathbf{y} = \mathbf{w}_j^\top \mathbf{y}\) avec \(\mathbf{w}_j = \mathbf{X}\mathbf{M}^{-1}\mathbf{k}_j\) (\(\mathbf{M}^{-1}\) est symétrique). Si \(\mathbf{x}_j = \mathbf{X}\mathbf{k}_j = \mathbf{0}\) : \(\mathbf{M}\mathbf{k}_j = \lambda \mathbf{k}_j\), donc \(\mathbf{M}^{-1}\mathbf{k}_j = \mathbf{k}_j/\lambda\) et \(\mathbf{w}_j = \mathbf{x}_j/\lambda = \mathbf{0}\). Réciproquement, si \(\mathbf{w}_j = \mathbf{0}\), le vecteur \(\mathbf{v} = \mathbf{M}^{-1}\mathbf{k}_j\) vérifie \(\mathbf{X}\mathbf{v} = \mathbf{0}\), donc \(\mathbf{k}_j = \mathbf{M}\mathbf{v} = \lambda \mathbf{v}\), et \(\mathbf{x}_j = \mathbf{X}\mathbf{k}_j = \lambda \mathbf{X}\mathbf{v} = \mathbf{0}\). Un hyperplan de \(\mathbb{R}^n\) est de mesure de Lebesgue nulle, donc de probabilité nulle pour une loi à densité. Enfin \(\hat{\boldsymbol{\beta}}_\lambda = \mathbf{M}^{-1}\mathbf{X}^\top \mathbf{y}\) est nul si et seulement si \(\mathbf{X}^\top \mathbf{y} = \mathbf{0}\), puisque \(\mathbf{M}^{-1}\) est inversible. Mécanisme : la dérivée de la pénalité \(\tfrac{\lambda}{2} b_j^2\) vaut \(\lambda b_j\), qui tend vers 0 avec \(b_j\) ; près de 0, la pénalité ne pousse plus, et le moindre gain d'ajustement suffit à quitter 0.

### Exemple chiffré

Deux variables de norme 1 dont le produit scalaire vaut 0,9 : \(\mathbf{X}^\top \mathbf{X} = \begin{pmatrix} 1 & 0{,}9 \\ 0{,}9 & 1 \end{pmatrix}\), et \(\mathbf{X}^\top \mathbf{y} = (1 ; 0{,}8)\). Les solutions MCO et Ridge ne dépendent des données qu'à travers ces deux quantités.

1. Vecteurs propres : \((1, 1)\) de valeur propre \(@L4_mu1:1m@\), et \((1, -1)\) de valeur propre \(@L4_mu2:1m@\). Conditionnement \(@L4_kappa0@\).
2. \(\mathbf{X}^\top \mathbf{y} = @L4_a:1m@\,(1, 1) + @L4_b:1m@\,(1, -1)\).
3. MCO : on divise chaque composante par sa valeur propre : \(\tfrac{@L4_a:1m@}{@L4_mu1:1m@}(1, 1) + \tfrac{@L4_b:1m@}{@L4_mu2:1m@}(1, -1) \approx (@L4_ols1:3m@ ; @L4_ols2:3m@)\). Les deux coefficients, de signes opposés, viennent surtout de la direction mal déterminée \((1, -1)\), la différence des deux variables.
4. Ridge, \(\lambda = @L4_lam:1m@\) : on divise par la valeur propre augmentée de \(\lambda\) : \(\tfrac{@L4_a:1m@}{@L4_mu1l:1m@}(1, 1) + \tfrac{@L4_b:1m@}{@L4_mu2l:1m@}(1, -1) = (@L4_rdg1:2m@ ; @L4_rdg2:2m@)\).
5. Facteurs de rétrécissement par direction : \(@L4_mu1:1m@/@L4_mu1l:1m@ = @L4_f1:2m@\) selon \((1, 1)\), \(@L4_mu2:1m@/@L4_mu2l:1m@ = @L4_f2:1m@\) selon \((1, -1)\). Conditionnement : \(@L4_mu1l:1m@/@L4_mu2l:1m@ = @L4_kappa@\) au lieu de \(@L4_kappa0@\).

```python-exec
rng = np.random.default_rng(7)
n, p, lam = 50, 5, 3.0
X = rng.standard_normal((n, p))
X[:, 4] = X[:, 3] + 0.01 * rng.standard_normal(n)       # deux colonnes presque égales
y = X.dot(np.array([1.0, -1.0, 0.5, 2.0, 0.0])) + rng.standard_normal(n)
forme_fermee = np.linalg.solve(X.T.dot(X) + lam * np.eye(p), X.T.dot(y))
U, d, Vt = np.linalg.svd(X, full_matrices=False)
par_svd = Vt.T.dot(d / (d**2 + lam) * U.T.dot(y))
print("forme fermée              :", forme_fermee)
print("par la SVD                :", par_svd)
print("Ridge(alpha=lam) sklearn  :", Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_)
print("MCO                       :", np.linalg.lstsq(X, y, rcond=None)[0])
print("valeurs singulières d     :", d)
print("facteurs d^2/(d^2 + lam)  :", d**2 / (d**2 + lam))
mu = np.linalg.eigvalsh(X.T.dot(X))
print("conditionnement de X^T X           :", round(mu.max() / mu.min(), 1), "; kappa(X)^2 :", round((d[0] / d[-1]) ** 2, 1))
print("conditionnement de X^T X + lam I   :", round((mu.max() + lam) / (mu.min() + lam), 1))
# Avec constante : scikit-learn centre X et y et ne pénalise pas la constante.
Xc, yc = X - X.mean(axis=0), y - y.mean()
print("fit_intercept=True        :", Ridge(alpha=lam).fit(X, y).coef_)
print("données centrées          :", np.linalg.solve(Xc.T.dot(Xc) + lam * np.eye(p), Xc.T.dot(yc)))
```

Lecture : les trois calculs Ridge coïncident ; les MCO donnent aux deux dernières colonnes, presque égales, des coefficients instables, alors que Ridge les partage ; le facteur de la plus petite valeur singulière est proche de 0.

### Cas limites

- \(\lambda \to 0^+\) : si \(\mathbf{X}\) est de rang \(p\), \(\hat{\boldsymbol{\beta}}_\lambda\) tend vers les MCO. Sinon, il tend vers \(\sum_j (\mathbf{u}_j^\top \mathbf{y}/d_j)\,\mathbf{v}_j\), la solution des moindres carrés de plus petite norme : elle vérifie les équations normales (étape 2 de la preuve avec \(\lambda = 0\)) et elle est orthogonale à \(\ker(\mathbf{X})\), car les \(\mathbf{v}_j\) le sont ; toute autre solution s'écrit elle plus \(\mathbf{h} \in \ker(\mathbf{X})\), de norme au carré augmentée de \(\lVert \mathbf{h} \rVert^2\) par Pythagore. C'est ce que renvoie `np.linalg.lstsq`.
- \(\lambda \to \infty\) : \(\hat{\boldsymbol{\beta}}_\lambda \approx \mathbf{X}^\top \mathbf{y}/\lambda \to \mathbf{0}\), sans l'atteindre si \(\mathbf{X}^\top \mathbf{y} \ne \mathbf{0}\).
- Deux colonnes identiques : par unicité, elles reçoivent le même coefficient (échanger les deux coordonnées d'une solution donne une solution, qui est donc la même).
- Colonnes orthonormées (\(\mathbf{X}^\top \mathbf{X} = \mathbf{I}\)) : \(\hat{\boldsymbol{\beta}}_\lambda = \mathbf{X}^\top \mathbf{y}/(1 + \lambda)\), une simple division.

### Pièges

1. Ridge n'est pas invariant par changement d'échelle : multiplier une colonne par 10 change sa pénalité effective. On standardise les colonnes avant, avec les moyennes et écarts-types de l'apprentissage seul.
2. On ne pénalise pas la constante : on centre \(\mathbf{X}\) et \(\mathbf{y}\), ce que fait scikit-learn avec `fit_intercept=True` (vérifié ci-dessus).
3. Conventions d'échelle de \(\lambda\) : lire la définition du critère (facteurs \(\tfrac12\), division par \(n\)).
4. Ridge ne sélectionne pas de variables.

[[controle:conditionnement]]

## Partie 5 : Lasso

### Le problème

On veut un modèle avec peu de coefficients non nuls : sélection de variables, lisibilité, robustesse quand une partie des variables est inutile. Ridge ne met jamais de zéro exact.

### Définition

Pour \(\lambda > 0\) : \(\ell_\lambda(\mathbf{b}) = \tfrac12 \lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2 + \lambda \lVert \mathbf{b} \rVert_1\). `Lasso(alpha=a)` de scikit-learn minimise \(\tfrac{1}{2n} \lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2 + a\,\lVert \mathbf{b} \rVert_1\) : même minimiseur pour \(a = \lambda/n\) (multiplier par \(n\)). On note \(\mathrm{signe}(u)\) le signe d'un réel non nul \(u\), et pour un vecteur \(\mathbf{b}\), \(\mathbf{g}(\mathbf{b}) = \mathbf{X}^\top (\mathbf{y} - \mathbf{X}\mathbf{b})\) : sa coordonnée \(g_j\) est le produit scalaire de la colonne \(j\) avec le résidu, c'est-à-dire le gain d'ajustement par unité de déplacement de \(b_j\).

### L'intuition

En une dimension, près de 0, la pénalité L2 \(\tfrac{\lambda}{2} b^2\) a une pente \(\lambda b\) qui s'annule : aucune force ne retient en 0. La pénalité L1 \(\lambda \lvert b \rvert\) a une pente \(\pm \lambda\) aussi près de 0 qu'on veut : un coefficient reste en 0 tant que le gain d'ajustement à le déplacer, \(\lvert g_j \rvert\), ne dépasse pas \(\lambda\). Géométriquement, la boule L1 \(\{\lVert \mathbf{b} \rVert_1 \le t\}\) a des coins sur les axes, et les ellipsoïdes de niveau de \(s\) la touchent en un coin, où des coordonnées sont nulles, pour toute une plage de positions de \(\mathbf{y}\) ; la boule L2 est ronde, sans coin, et le contact n'a lieu sur un axe que pour un ensemble de \(\mathbf{y}\) de probabilité nulle (partie 4, énoncé 6).

### L'énoncé

1. **Existence.** \(\ell_\lambda\) atteint son minimum (fonction continue et \(\ell_\lambda(\mathbf{b}) \ge \lambda \lVert \mathbf{b} \rVert_1 \to \infty\) quand \(\lVert \mathbf{b} \rVert \to \infty\) ; le minimum d'une telle fonction est atteint, admis).
2. **Conditions d'optimalité.** \(\hat{\mathbf{b}}\) minimise \(\ell_\lambda\) si et seulement si, avec \(\mathbf{g} = \mathbf{g}(\hat{\mathbf{b}})\), pour chaque \(j\) : \(g_j = \lambda\,\mathrm{signe}(\hat{b}_j)\) si \(\hat{b}_j \ne 0\), et \(\lvert g_j \rvert \le \lambda\) si \(\hat{b}_j = 0\).
3. Tous les minimiseurs ont les mêmes valeurs ajustées \(\mathbf{X}\hat{\mathbf{b}}\) et la même norme \(\lVert \hat{\mathbf{b}} \rVert_1\).
4. **Seuil.** \(\mathbf{0}\) est un minimiseur si et seulement si \(\lambda \ge \lambda_{\max} = \max_j \lvert \mathbf{x}_j^\top \mathbf{y} \rvert\), et c'est alors l'unique minimiseur.
5. **Cas orthonormal** (\(\mathbf{X}^\top \mathbf{X} = \mathbf{I}\), donc \(n \ge p\)). Avec \(\mathbf{z} = \mathbf{X}^\top \mathbf{y}\) (les MCO), l'unique solution est le **seuillage doux** \(\hat{b}_j = \mathrm{signe}(z_j) \max(\lvert z_j \rvert - \lambda, 0)\) (et 0 si \(z_j = 0\)). Ridge, dans le même cas, divise : \(z_j/(1 + \lambda)\).
6. **Pas de forme fermée en général.** Si l'on connaît l'ensemble actif \(\mathcal{A} = \{j : \hat{b}_j \ne 0\}\) et les signes \(\mathbf{s}_{\mathcal{A}}\), et si la matrice \(\mathbf{X}_{\mathcal{A}}\) des colonnes actives est de rang plein, alors \(\hat{\mathbf{b}}_{\mathcal{A}} = (\mathbf{X}_{\mathcal{A}}^\top \mathbf{X}_{\mathcal{A}})^{-1} (\mathbf{X}_{\mathcal{A}}^\top \mathbf{y} - \lambda\,\mathbf{s}_{\mathcal{A}})\) et les autres coordonnées sont nulles. Mais \(\mathcal{A}\) et les signes sont inconnus : chaque coordonnée peut être négative, nulle ou positive, soit \(3^p\) configurations, et la bonne n'est donnée par aucune formule ; on la trouve par un algorithme (descente par coordonnées, comme scikit-learn, ou LARS). Formules fermées dans les cas particuliers \(p = 1\) et \(\mathbf{X}^\top \mathbf{X}\) diagonale (seuillage doux de \(\mathbf{x}_j^\top \mathbf{y}\) au niveau \(\lambda\), divisé par \(\lVert \mathbf{x}_j \rVert^2\)).
7. **Interprétation bayésienne.** Si les coordonnées de \(\mathbf{B}\) sont indépendantes de loi de Laplace, de densité \(\tfrac{1}{2\tau} e^{-\lvert b \rvert/\tau}\) (\(\tau > 0\)), et si \(\mathbf{Y}\) sachant \(\mathbf{B} = \mathbf{b}\) suit \(\mathcal{N}(\mathbf{X}\mathbf{b}, \sigma^2 \mathbf{I})\), le mode a posteriori est la solution du Lasso avec \(\lambda = \sigma^2/\tau\). La moyenne a posteriori, elle, n'est pas la solution du Lasso.
8. **Non-unicité possible.** Avec deux colonnes égales, et plus largement quand \(\ker(\mathbf{X})\) n'est pas réduit à \(\mathbf{0}\) (ce qui est toujours le cas si \(p > n\)), le minimiseur peut ne pas être unique. Quand les colonnes sont « en position générale », ce qui a lieu avec probabilité 1 si \(\mathbf{X}\) a une densité, la solution est unique et a au plus \(\min(n, p)\) coefficients non nuls (Tibshirani, 2013, admis).

### La preuve

1. **Développement.** En divisant (D) par 2 : \(\tfrac12 \lVert \mathbf{y} - \mathbf{X}(\hat{\mathbf{b}} + \mathbf{h}) \rVert^2 = \tfrac12 \lVert \mathbf{y} - \mathbf{X}\hat{\mathbf{b}} \rVert^2 - \mathbf{h}^\top \mathbf{g} + \tfrac12 \lVert \mathbf{X}\mathbf{h} \rVert^2\).
2. **Conditions suffisantes.** On suppose les conditions vérifiées, et \(\mathbf{b} = \hat{\mathbf{b}} + \mathbf{h}\) quelconque. Pour chaque \(j\), \(\lambda \lvert b_j \rvert \ge \lambda \lvert \hat{b}_j \rvert + g_j h_j\). Si \(\hat{b}_j \ne 0\), avec \(s = \mathrm{signe}(\hat{b}_j)\) et \(g_j = \lambda s\) : \(\lambda \lvert b_j \rvert \ge \lambda s\,b_j = \lambda s\,\hat{b}_j + \lambda s\,h_j = \lambda \lvert \hat{b}_j \rvert + g_j h_j\). Si \(\hat{b}_j = 0\) : \(\lambda \lvert b_j \rvert = \lambda \lvert h_j \rvert \ge \lvert g_j \rvert \lvert h_j \rvert \ge g_j h_j\). En sommant sur \(j\) et avec le développement : \(\ell_\lambda(\mathbf{b}) - \ell_\lambda(\hat{\mathbf{b}}) \ge -\mathbf{h}^\top \mathbf{g} + \tfrac12 \lVert \mathbf{X}\mathbf{h} \rVert^2 + \mathbf{h}^\top \mathbf{g} \ge 0\).
3. **Conditions nécessaires.** Soit \(\hat{\mathbf{b}}\) un minimiseur, \(j\) fixé, et \(f(u) = \ell_\lambda(\hat{\mathbf{b}} + u\,\mathbf{k}_j)\) pour \(u\) réel (\(\mathbf{k}_j\) : vecteur de base, partie 4). Par le développement, \(f(u) - f(0) = -u\,g_j + \tfrac12 u^2 \lVert \mathbf{x}_j \rVert^2 + \lambda(\lvert \hat{b}_j + u \rvert - \lvert \hat{b}_j \rvert)\), positif pour tout \(u\). Si \(\hat{b}_j \ne 0\), pour \(\lvert u \rvert < \lvert \hat{b}_j \rvert\) le dernier terme vaut \(\lambda\,\mathrm{signe}(\hat{b}_j)\,u\) : \(f\) est dérivable en 0, de dérivée \(-g_j + \lambda\,\mathrm{signe}(\hat{b}_j)\), qui doit être nulle en un minimum. Si \(\hat{b}_j = 0\) : \(f(u) - f(0) = \lvert u \rvert (\lambda - g_j\,\mathrm{signe}(u)) + \tfrac12 u^2 \lVert \mathbf{x}_j \rVert^2\) ; en divisant par \(\lvert u \rvert\) et en faisant tendre \(u\) vers 0 par valeurs positives puis négatives : \(\lambda - g_j \ge 0\) et \(\lambda + g_j \ge 0\), soit \(\lvert g_j \rvert \le \lambda\).
4. **Mêmes valeurs ajustées.** Pour tous vecteurs \(\mathbf{u}_1, \mathbf{u}_2\) : \(\tfrac12 (\lVert \mathbf{y} - \mathbf{u}_1 \rVert^2 + \lVert \mathbf{y} - \mathbf{u}_2 \rVert^2) - \lVert \mathbf{y} - \tfrac{\mathbf{u}_1 + \mathbf{u}_2}{2} \rVert^2 = \tfrac14 \lVert \mathbf{u}_1 - \mathbf{u}_2 \rVert^2\) (identité du parallélogramme). Si deux minimiseurs \(\mathbf{b}_1, \mathbf{b}_2\) avaient \(\mathbf{X}\mathbf{b}_1 \ne \mathbf{X}\mathbf{b}_2\), leur milieu aurait une partie quadratique strictement plus petite que la moyenne des deux, et une norme L1 au plus égale à la moyenne (inégalité triangulaire) : un critère strictement plus petit, contradiction. Mêmes valeurs ajustées et même critère imposent la même norme L1.
5. **Seuil.** Pour \(\hat{\mathbf{b}} = \mathbf{0}\), \(\mathbf{g} = \mathbf{X}^\top \mathbf{y}\), et les conditions se réduisent à \(\lvert \mathbf{x}_j^\top \mathbf{y} \rvert \le \lambda\) pour tout \(j\). Unicité : par l'étape 4, tout minimiseur a la norme L1 de \(\mathbf{0}\), donc est nul.
6. **Cas orthonormal.** Par (D) en \(\mathbf{z}\) avec \(\mathbf{h} = \mathbf{b} - \mathbf{z}\) : \(\mathbf{X}^\top (\mathbf{y} - \mathbf{X}\mathbf{z}) = \mathbf{z} - \mathbf{z} = \mathbf{0}\) et \(\lVert \mathbf{X}\mathbf{h} \rVert^2 = \mathbf{h}^\top \mathbf{X}^\top \mathbf{X}\mathbf{h} = \lVert \mathbf{h} \rVert^2\), donc \(\ell_\lambda(\mathbf{b}) = \tfrac12 \lVert \mathbf{y} - \mathbf{X}\mathbf{z} \rVert^2 + \sum_j \big(\tfrac12 (z_j - b_j)^2 + \lambda \lvert b_j \rvert\big)\) : le problème se découple en \(p\) problèmes d'une variable. Pour l'un d'eux, \(g = z_j - b_j\) et les conditions de l'étape 2 donnent : si \(b_j \ne 0\), \(b_j = z_j - \lambda\,\mathrm{signe}(b_j)\), possible seulement si \(\lvert z_j \rvert > \lambda\), avec le signe de \(z_j\) ; si \(b_j = 0\), \(\lvert z_j \rvert \le \lambda\). Les deux cas s'excluent : la solution est unique et c'est le seuillage doux.
7. **Forme sur l'ensemble actif.** Les conditions de l'étape 2 sur les coordonnées actives s'écrivent \(\mathbf{X}_{\mathcal{A}}^\top (\mathbf{y} - \mathbf{X}_{\mathcal{A}}\hat{\mathbf{b}}_{\mathcal{A}}) = \lambda\,\mathbf{s}_{\mathcal{A}}\), système linéaire dont on tire \(\hat{\mathbf{b}}_{\mathcal{A}}\). Ce qui empêche une formule générale : les conditions sur les coordonnées nulles sont des inégalités, et le terme \(\lambda\,\mathbf{s}_{\mathcal{A}}\) dépend des signes ; la non-dérivabilité de \(\lvert \cdot \rvert\) en 0 interdit d'écrire un seul système linéaire valable pour tout \(\mathbf{y}\). Les termes non diagonaux de \(\mathbf{X}^\top \mathbf{X}\) couplent les coordonnées : le gain \(g_j\) de chacune dépend des autres, et le seuil de l'une change quand une autre entre ou sort. Sans couplage (\(\mathbf{X}^\top \mathbf{X}\) diagonale), le problème se découple comme à l'étape 6.
8. **Bayes.** Moins le logarithme de la densité a posteriori vaut, à une constante près, \(\lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2/(2\sigma^2) + \lVert \mathbf{b} \rVert_1/\tau = \tfrac{1}{\sigma^2}\big(\tfrac12 \lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2 + \tfrac{\sigma^2}{\tau} \lVert \mathbf{b} \rVert_1\big)\) : son minimiseur, le mode, est la solution du Lasso.
9. **Non-unicité.** Si \(\mathbf{x}_1 = \mathbf{x}_2\), le critère ne dépend de \((b_1, b_2)\) qu'à travers \(b_1 + b_2\) pour l'ajustement, et \(\lvert b_1 \rvert + \lvert b_2 \rvert \ge \lvert b_1 + b_2 \rvert\), avec égalité si \(b_1\) et \(b_2\) ont le même signe. Si un minimiseur a \(b_1 + b_2 = m \ne 0\), tous les partages \((u\,m, (1 - u)\,m)\), \(u \in [0, 1]\), donnent le même critère : une infinité de minimiseurs.

### Exemple chiffré

Colonnes orthonormées, \(\mathbf{z} = \mathbf{X}^\top \mathbf{y} = @L5_z:1m@\), \(\lambda = @L5_lam:1m@\). Lasso : \(@L5_lasso:1m@\) (la deuxième coordonnée, de valeur absolue sous \(\lambda\), est mise à 0 ; les autres perdent \(\lambda\) en valeur absolue). Ridge, même \(\lambda\) : \(@L5_ridge:3m@\), aucun zéro.

L'expérience suivante compte les coefficients exactement nuls de Ridge et du Lasso (100 observations, 10 colonnes, 3 utiles), pour plusieurs \(\lambda\), avec la correspondance `alpha = lambda / n` du Lasso de scikit-learn.

```python-exec
rng = np.random.default_rng(8)
n, p = 100, 10
X = rng.standard_normal((n, p))
y = X[:, :3].dot(np.array([3.0, -2.0, 1.0])) + rng.standard_normal(n)
print("lambda_max = max_j |x_j^T y| =", round(np.abs(X.T.dot(y)).max(), 2))
for lam in [1.0, 10.0, 100.0, 1000.0]:
    r = Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_
    l = Lasso(alpha=lam / n, fit_intercept=False, tol=1e-12, max_iter=1000000).fit(X, y).coef_
    print(f"lambda={lam:6.0f}  zéros Ridge : {np.sum(r == 0):2d}   zéros Lasso : {np.sum(l == 0):2d}   plus petit |coef| Ridge : {np.abs(r).min():.2e}")
```

Contrôle des conditions d'optimalité sur la solution de scikit-learn pour \(\lambda = 30\), puis formule sur l'ensemble actif :

```python-exec
lam = 30.0
b = Lasso(alpha=lam / n, fit_intercept=False, tol=1e-12, max_iter=1000000).fit(X, y).coef_
g = X.T.dot(y - X.dot(b))
for j in range(p):
    print(f"j={j}  {'actif' if b[j] != 0 else 'nul  '}  b_j={b[j]: .4f}   g_j = x_j^T (y - X b) = {g[j]: .4f}")
A = b != 0
XA = X[:, A]
print("formule sur l'ensemble actif :", np.linalg.solve(XA.T.dot(XA), XA.T.dot(y) - lam * np.sign(b[A])))
print("coefficients actifs sklearn  :", b[A])
```

Lecture : les coordonnées actives ont \(g_j = \pm 30\), du signe de \(b_j\) ; les coordonnées nulles ont \(\lvert g_j \rvert \le 30\) ; la formule sur l'ensemble actif redonne la solution, une fois l'ensemble connu.

Deux colonnes identiques : la solution de scikit-learn et un partage différent ont le même critère.

```python-exec
x = rng.standard_normal(n)
X2 = np.column_stack([x, x])
y2 = 2 * x + rng.standard_normal(n)
lam = 10.0
b_sk = Lasso(alpha=lam / n, fit_intercept=False, tol=1e-12, max_iter=1000000).fit(X2, y2).coef_
def critere(b):
    return 0.5 * np.sum((y2 - X2.dot(b)) ** 2) + lam * np.abs(b).sum()
partage = np.array([b_sk.sum() / 2, b_sk.sum() / 2])
print("solution scikit-learn :", b_sk, "; critère :", round(critere(b_sk), 6))
print("partage en deux       :", partage, "; critère :", round(critere(partage), 6))
```

### Cas limites

- \(\lambda \ge \lambda_{\max}\) : solution nulle, unique.
- \(p = 1\) : \(\hat{b} = \mathrm{signe}(c)\max(\lvert c \rvert - \lambda, 0)/\lVert \mathbf{x}_1 \rVert^2\) avec \(c = \mathbf{x}_1^\top \mathbf{y}\), par les conditions d'optimalité (étape 2) avec une seule coordonnée, où \(g = c - \hat{b}\,\lVert \mathbf{x}_1 \rVert^2\).
- Colonnes identiques : une infinité de partages sont optimaux, et scikit-learn en renvoie un qui met tout le poids sur une seule colonne (expérience ci-dessus). Colonnes très corrélées mais distinctes : la solution est unique en position générale, mais la colonne retenue peut changer avec une petite perturbation des données.

### Pièges

1. `Lasso(alpha=a)` de scikit-learn correspond à \(\lambda = n\,a\) dans la convention de cette leçon ; `Ridge(alpha=a)` à \(\lambda = a\), sans facteur \(n\).
2. « Pas de forme fermée » ne veut pas dire « pas de solution » ni « pas unique » : la solution existe toujours, et elle est unique dans le cas orthonormal ou en position générale.
3. Les coefficients actifs sont rétrécis : dans le cas orthonormal, ils perdent exactement \(\lambda\) en valeur absolue. On peut réestimer par MCO sur les variables sélectionnées.
4. Comme Ridge, le Lasso exige des colonnes standardisées et une constante non pénalisée.

[[controle:seuil]]

## Partie 6 : biais, variance et rôle de lambda

### Le problème

Pourquoi accepter un estimateur biaisé ? Parce que l'erreur qui compte, l'erreur de prédiction ou l'erreur quadratique sur les coefficients, se décompose en biais au carré plus variance, et qu'une petite hausse du biais peut acheter une forte baisse de la variance.

### Notations

On se place dans le modèle de la partie 2 sous (A1), (A2), (A3). \(\mathbf{x}_0 \in \mathbb{R}^p\) est un point fixe ; la réponse future en ce point est \(Y_0 = \mathbf{x}_0^\top \boldsymbol{\beta} + \varepsilon_0\), où \(\varepsilon_0\) est centrée, de variance \(\sigma^2\), indépendante de \(\boldsymbol{\varepsilon}\). La prédiction Ridge est \(\hat{Y}_0 = \mathbf{x}_0^\top \hat{\boldsymbol{\beta}}_\lambda\), aléatoire à travers \(\mathbf{Y}\). On note \(\mu_1, \dots, \mu_p > 0\) les valeurs propres de \(\mathbf{X}^\top \mathbf{X}\), \(\mathbf{v}_1, \dots, \mathbf{v}_p\) des vecteurs propres orthonormés associés, et \(\theta_j = \mathbf{v}_j^\top \boldsymbol{\beta}\) la coordonnée du vrai \(\boldsymbol{\beta}\) selon \(\mathbf{v}_j\).

### L'énoncé

1. **Décomposition.** Pour tout prédicteur \(\hat{Y}_0\) fonction des seules données d'apprentissage, de variance finie, et avec \(m_0 = \mathbb{E}[Y_0]\) : \(\mathbb{E}[(Y_0 - \hat{Y}_0)^2] = \sigma^2 + (\mathbb{E}[\hat{Y}_0] - m_0)^2 + \mathrm{Var}(\hat{Y}_0)\) : bruit irréductible, biais au carré, variance.
2. **Ridge.** \(\mathbb{E}[\hat{\boldsymbol{\beta}}_\lambda] - \boldsymbol{\beta} = -\lambda (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \boldsymbol{\beta}\) et \(\mathrm{Var}(\hat{\boldsymbol{\beta}}_\lambda) = \sigma^2 (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{X} (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1}\). Selon \(\mathbf{v}_j\) : biais \(-\lambda \theta_j/(\mu_j + \lambda)\), variance \(\sigma^2 \mu_j/(\mu_j + \lambda)^2\), et les coordonnées selon des \(\mathbf{v}_j\) différents sont non corrélées. Quand \(\lambda\) croît, chaque biais croît en valeur absolue (si \(\theta_j \ne 0\)) et chaque variance décroît.
3. **Ridge bat les MCO (Hoerl et Kennard).** \(\displaystyle \mathbb{E}\lVert \hat{\boldsymbol{\beta}}_\lambda - \boldsymbol{\beta} \rVert^2 = \sum_{j=1}^p \frac{\lambda^2 \theta_j^2 + \sigma^2 \mu_j}{(\mu_j + \lambda)^2}\). Chaque terme décroît strictement sur \([0, \sigma^2/\theta_j^2]\) et croît ensuite (décroît sur tout \([0, \infty[\) si \(\theta_j = 0\)). Donc tout \(\lambda\) de \(]0, \sigma^2/\max_j \theta_j^2]\) donne une erreur quadratique strictement plus petite que les MCO (\(\lambda = 0\)).
4. **Rôle de \(\lambda\).** \(\lambda = 0\) : MCO, sans biais, variance maximale. \(\lambda \to \infty\) : variance tendant vers 0, biais tendant vers \(-\boldsymbol{\beta}\). Le meilleur \(\lambda\) dépend de \(\boldsymbol{\beta}\) et \(\sigma^2\), inconnus : on le choisit sur des données non utilisées pour l'ajustement (validation, validation croisée), jamais sur l'erreur d'apprentissage, qui est minimale en \(\lambda = 0\) puisque les MCO minimisent \(s\).

### La preuve

1. \(Y_0 - \hat{Y}_0 = \varepsilon_0 + (m_0 - \mathbb{E}\hat{Y}_0) + (\mathbb{E}\hat{Y}_0 - \hat{Y}_0)\). On développe le carré et on prend l'espérance. Termes croisés : \(\mathbb{E}[\varepsilon_0]\,(m_0 - \mathbb{E}\hat{Y}_0) = 0\) ; \(\mathbb{E}[\varepsilon_0 (\mathbb{E}\hat{Y}_0 - \hat{Y}_0)] = \mathbb{E}[\varepsilon_0]\,\mathbb{E}[\mathbb{E}\hat{Y}_0 - \hat{Y}_0] = 0\) par indépendance, \(\hat{Y}_0\) ne dépendant que de \(\boldsymbol{\varepsilon}\) ; \((m_0 - \mathbb{E}\hat{Y}_0)\,\mathbb{E}[\mathbb{E}\hat{Y}_0 - \hat{Y}_0] = 0\). Restent \(\sigma^2\), le biais au carré et la variance.
2. Avec \(\mathbf{M} = \mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}\) : \(\hat{\boldsymbol{\beta}}_\lambda = \mathbf{M}^{-1} \mathbf{X}^\top (\mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon})\). Espérance : \(\mathbf{M}^{-1} \mathbf{X}^\top \mathbf{X}\boldsymbol{\beta} = \mathbf{M}^{-1} (\mathbf{M} - \lambda \mathbf{I})\boldsymbol{\beta} = \boldsymbol{\beta} - \lambda \mathbf{M}^{-1}\boldsymbol{\beta}\). Variance : règle de calcul de la partie 2 avec la matrice \(\mathbf{M}^{-1}\mathbf{X}^\top\). Comme \(\mathbf{M}^{-1}\mathbf{v}_j = \mathbf{v}_j/(\mu_j + \lambda)\) et \(\mathbf{X}^\top \mathbf{X}\mathbf{v}_j = \mu_j \mathbf{v}_j\) : \(\mathbf{v}_j^\top (\text{biais}) = -\lambda \theta_j/(\mu_j + \lambda)\) et \(\mathbf{v}_j^\top \mathrm{Var}(\hat{\boldsymbol{\beta}}_\lambda)\mathbf{v}_k = \sigma^2 \mu_j/(\mu_j + \lambda)^2\) si \(j = k\), 0 sinon. Monotonie : \(\lambda/(\mu_j + \lambda)\) croît et \(\mu_j/(\mu_j + \lambda)^2\) décroît en \(\lambda \ge 0\).
3. \(\mathbb{E}\lVert \hat{\boldsymbol{\beta}}_\lambda - \boldsymbol{\beta} \rVert^2 = \lVert \mathbb{E}\hat{\boldsymbol{\beta}}_\lambda - \boldsymbol{\beta} \rVert^2 + \mathrm{tr}\,\mathrm{Var}(\hat{\boldsymbol{\beta}}_\lambda)\) (énoncé 1 coordonnée par coordonnée, sans le terme de bruit) ; on calcule norme et trace dans la base orthonormée des \(\mathbf{v}_j\), ce qui donne la somme annoncée. Pour un terme \(g_j(\lambda) = (\lambda^2 \theta_j^2 + \sigma^2 \mu_j)/(\mu_j + \lambda)^2\), la dérivée vaut \(\dfrac{2\lambda \theta_j^2 (\mu_j + \lambda) - 2(\lambda^2 \theta_j^2 + \sigma^2 \mu_j)}{(\mu_j + \lambda)^3} = \dfrac{2\mu_j (\lambda \theta_j^2 - \sigma^2)}{(\mu_j + \lambda)^3}\), du signe de \(\lambda \theta_j^2 - \sigma^2\).
4. Limites : facteurs \(\lambda/(\mu_j + \lambda) \to 1\) et \(\mu_j/(\mu_j + \lambda)^2 \to 0\).

### Exemple chiffré

Même \(\mathbf{X}^\top \mathbf{X}\) qu'en partie 4 (valeurs propres \(@L4_mu1:1m@\) et \(@L4_mu2:1m@\)), vrai \(\boldsymbol{\beta} = (1, 0)\), \(\sigma^2 = @L6_sigma2:2m@\). Les vecteurs propres normés sont \((1, 1)/\sqrt{2}\) et \((1, -1)/\sqrt{2}\), donc \(\theta_1^2 = \theta_2^2 = @L6_theta2:1m@\), et chaque terme est minimal en \(\lambda = \sigma^2/\theta_j^2 = @L6_lopt:1m@\). Erreur quadratique des coefficients : \(@L6_mse0:3m@\) pour les MCO, \(@L6_mse_opt:3m@\) pour \(\lambda = @L6_lopt:1m@\).

Prédiction en \(\mathbf{x}_0 = (1, -1)\), la direction mal déterminée (\(\mathbf{x}_0^\top \boldsymbol{\beta} = 1\)) :

| \(\lambda\) | bruit | biais au carré | variance | erreur totale |
|---|---|---|---|---|
| \(@L6_l0@\) | \(@L6_sigma2:2m@\) | \(@L6_b0:4m@\) | \(@L6_v0:4m@\) | \(@L6_t0:4m@\) |
| \(@L6_l1:1m@\) | \(@L6_sigma2:2m@\) | \(@L6_b1:4m@\) | \(@L6_v1:4m@\) | \(@L6_t1:4m@\) |
| \(@L6_l2:1m@\) | \(@L6_sigma2:2m@\) | \(@L6_b2:4m@\) | \(@L6_v2:4m@\) | \(@L6_t2:4m@\) |
| \(@L6_l3@\) | \(@L6_sigma2:2m@\) | \(@L6_b3:4m@\) | \(@L6_v3:4m@\) | \(@L6_t3:4m@\) |

L'erreur totale est minimale en \(\lambda = @L6_lopt:1m@\) (dérivée calculée par le script). Simulation de ces quatre lignes :

```python-exec
G = np.array([[1.0, 0.9], [0.9, 1.0]])
X = np.linalg.cholesky(G).T                     # matrice 2 x 2 telle que X^T X = G
beta, sigma, x0 = np.array([1.0, 0.0]), 0.5, np.array([1.0, -1.0])
rng = np.random.default_rng(9)
Y = X.dot(beta) + sigma * rng.standard_normal((200000, 2))
for lam in [0.0, 0.1, 0.5, 2.0]:
    A = np.linalg.solve(X.T.dot(X) + lam * np.eye(2), X.T)     # beta_lambda = A y
    pred = Y.dot(A.T).dot(x0)
    biais2 = (pred.mean() - x0.dot(beta)) ** 2
    print(f"lambda={lam:3.1f}  biais^2={biais2:.4f}  variance={pred.var():.4f}  erreur totale={sigma**2 + biais2 + pred.var():.4f}")
```

### Cas limites

- \(\sigma = 0\) : les MCO sont exacts, et \(\sigma^2/\theta_j^2 = 0\) : aucun \(\lambda > 0\) n'aide.
- \(\boldsymbol{\beta}\) aligné sur une direction bien déterminée (\(\mu_j\) grand) : le biais selon \(\mathbf{v}_j\), \(-\lambda \theta_j/(\mu_j + \lambda)\), reste petit tant que \(\lambda\) est petit devant \(\mu_j\), et le gain de variance vient des directions de petite valeur propre.
- Le Lasso obéit au même arbitrage, sans formule fermée du biais et de la variance.

### Pièges

1. Choisir \(\lambda\) sur l'erreur d'apprentissage donne toujours \(\lambda = 0\).
2. La standardisation et le choix de \(\lambda\) doivent être faits à l'intérieur de chaque pli de validation, avec les seules données d'apprentissage du pli ; sinon de l'information des données de validation fuit dans l'ajustement.
3. Gauss-Markov n'est pas contredit : Ridge est biaisé, hors de la classe où les MCO sont optimaux.

[[controle:biais_variance]]

## Réflexes en 30 secondes

| Indice dans l'énoncé | Réflexe |
|---|---|
| « forme fermée de Ridge ? » | oui : \((\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}\), unique pour tout \(\lambda > 0\), quel que soit le rang |
| « forme fermée du Lasso ? » | non en général : non-dérivabilité en 0 et couplage par \(\mathbf{X}^\top \mathbf{X}\) ; oui si \(\mathbf{X}^\top \mathbf{X}\) diagonale (seuillage doux) ou \(p = 1\) |
| « MCO unique ? » | si et seulement si \(\mathbf{X}\) est de rang \(p\) |
| « matrice chapeau » | symétrique, idempotente, trace \(p\) ; \(\mathbb{E}\lVert \hat{\boldsymbol{\varepsilon}} \rVert^2 = \sigma^2 (n - p)\) |
| hétéroscédasticité, autocorrélation | sans biais ; variance usuelle fausse ; plus BLUE ; White, Newey-West |
| variable omise | biais \(\gamma\,\boldsymbol{\delta}\) ; signe de l'effet fois signe de la pente |
| erreur de mesure | atténuation \(\beta\,v_Z/(v_Z + v_U)\) |
| multicolinéarité | sans biais ; variance multipliée par \(1/(1 - R_j^2)\) |
| ajout d'une variable | R² ne baisse pas ; R² ajusté monte si et seulement si \(t^2 > 1\) |
| pourquoi des zéros avec L1 | pente \(\lambda\) en 0 ; nul tant que \(\lvert g_j \rvert \le \lambda\) ; coins de la boule |
| a priori | gaussien : Ridge, \(\lambda = \sigma^2/\tau^2\) ; Laplace : Lasso (mode), \(\lambda = \sigma^2/\tau\) |
| scikit-learn | `Ridge(alpha)` : \(\lambda = \) alpha ; `Lasso(alpha)` : \(\lambda = n \times\) alpha |
| choisir \(\lambda\) | biais contre variance ; validation hors échantillon |

## À savoir refaire sans notes

- Les équations normales par l'identité (D), l'existence par \(\ker(\mathbf{X}^\top \mathbf{X}) = \ker(\mathbf{X})\), l'unicité par le rang, et les propriétés de \(\mathbf{H}\).
- Sans biais, \(\mathrm{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2 (\mathbf{X}^\top \mathbf{X})^{-1}\), \(\mathbb{E}\lVert \hat{\boldsymbol{\varepsilon}} \rVert^2 = \sigma^2 (n - p)\) par la trace, et Gauss-Markov par \(\mathbf{C} = \mathbf{A} + \mathbf{D}\).
- Le biais de variable omise, l'atténuation par erreur de mesure, la formule du VIF par Frisch-Waugh.
- \(\mathrm{SCT} = \mathrm{SCE} + \mathrm{SCR}\), la non-décroissance de R², et le critère \(t^2 > 1\) du R² ajusté.
- Ridge : forme fermée, lecture SVD, conditionnement, a priori gaussien, absence de zéros exacts et sa condition.
- Lasso : conditions d'optimalité (les deux sens), seuil \(\lambda_{\max}\), seuillage doux, formule sur l'ensemble actif et raison de l'absence de forme fermée.
- Décomposition biais-variance et dérivée de Hoerl et Kennard.
""",
    "controles": {
        "residus": {
            "question": r"Régression MCO **avec constante**, \(\mathbf{X}\) de rang \(p < n\). Laquelle de ces affirmations est toujours vraie ?",
            "options": [
                {"id": "a", "texte": r"La somme des résidus est nulle, et le vecteur des résidus est orthogonal au vecteur des valeurs ajustées.",
                 "explication": r"Oui : \(\mathbf{X}^\top \mathbf{e} = \mathbf{0}\) donne la somme nulle (ligne de la constante) et l'orthogonalité à toute combinaison des colonnes, dont \(\hat{\mathbf{y}}\)."},
                {"id": "b", "texte": r"Le vecteur des résidus est orthogonal au vecteur \(\mathbf{y}\) des réponses.",
                 "explication": r"Faux : \(\mathbf{y}^\top \mathbf{e} = (\hat{\mathbf{y}} + \mathbf{e})^\top \mathbf{e} = \lVert \mathbf{e} \rVert^2\), nul seulement si l'ajustement est parfait. Dans l'exemple à quatre points, \(\mathbf{y}^\top \mathbf{e} = @L1_ye:1m@\)."},
                {"id": "c", "texte": r"La somme des résidus serait nulle aussi sans la constante.",
                 "explication": r"Pas toujours vrai : la somme nulle vient de la ligne \(\mathbf{1}^\top \mathbf{e} = 0\) des équations normales, qui n'existe que si \(\mathbf{1}\) est dans l'image de \(\mathbf{X}\)."},
                {"id": "d", "texte": r"La matrice chapeau \(\mathbf{H}\) est inversible.",
                 "explication": r"Faux : l'image de \(\mathbf{H}\) est \(\mathrm{Im}(\mathbf{X})\), de dimension \(p\), donc \(\mathbf{H}\) n'est pas inversible dès que \(p < n\). Elle n'est inversible que si \(p = n\), auquel cas \(\mathbf{H} = \mathbf{I}\)."},
            ],
        },
        "gauss_markov": {
            "question": r"Sous (A1), (A2) et (A3), **sans** supposer la normalité, laquelle de ces affirmations est un énoncé exact ?",
            "options": [
                {"id": "a", "texte": r"Pour tout vecteur fixe \(\mathbf{a}\) et tout estimateur \(\tilde{\boldsymbol{\beta}} = \mathbf{C}\mathbf{Y}\) linéaire et sans biais, \(\mathrm{Var}(\mathbf{a}^\top \hat{\boldsymbol{\beta}}) \le \mathrm{Var}(\mathbf{a}^\top \tilde{\boldsymbol{\beta}})\).",
                 "explication": r"Oui : c'est Gauss-Markov, prouvé par \(\mathbf{C} = \mathbf{A} + \mathbf{D}\) avec \(\mathbf{D}\mathbf{X} = \mathbf{0}\), qui donne une différence de variances égale à \(\sigma^2 \mathbf{D}\mathbf{D}^\top\)."},
                {"id": "b", "texte": r"\(\hat{\boldsymbol{\beta}}\) a la plus petite variance parmi tous les estimateurs sans biais, linéaires ou non.",
                 "explication": r"Faux sans normalité. Exemple : \(Y_i = \beta + \varepsilon_i\), \(i = 1, \dots, 10\), bruits indépendants uniformes sur \([-1, 1]\). La moyenne (MCO) a pour variance \(@C2_var_moy:4m@\) ; le milieu de l'étendue \((\min_i Y_i + \max_i Y_i)/2\), sans biais par symétrie mais non linéaire, a pour variance \(@C2_var_mil:4m@\) (calcul exact et simulation dans le script)."},
                {"id": "c", "texte": r"\(\hat{\boldsymbol{\beta}}\) a la plus petite erreur quadratique moyenne parmi tous les estimateurs linéaires en \(\mathbf{Y}\).",
                 "explication": r"Faux : Ridge est linéaire en \(\mathbf{Y}\), biaisé, et peut avoir une erreur quadratique strictement plus petite (partie 6, Hoerl et Kennard). Gauss-Markov ne compare qu'aux estimateurs sans biais."},
                {"id": "d", "texte": r"Le théorème n'est valable que si les bruits sont gaussiens.",
                 "explication": r"Faux : la preuve n'utilise que \(\mathbb{E}[\boldsymbol{\varepsilon}] = \mathbf{0}\) et \(\mathrm{Var}(\boldsymbol{\varepsilon}) = \sigma^2 \mathbf{I}\). La normalité sert aux lois exactes des tests."},
            ],
        },
        "omise": {
            "question": r"Vrai modèle : \(Y_i = \beta_1 + \beta_2 x_{i2} + \gamma z_i + \varepsilon_i\), avec \(\gamma > 0\), bruits centrés, plan fixe. Dans l'échantillon, la régression MCO de \(\mathbf{z}\) sur \((\mathbf{1}, \mathbf{x}_2)\) a une pente \(\delta > 0\). On régresse \(\mathbf{Y}\) sur \((\mathbf{1}, \mathbf{x}_2)\) seulement. Que vaut l'espérance de la pente estimée ?",
            "options": [
                {"id": "a", "texte": r"\(\beta_2 + \gamma\,\delta\) : la pente est surestimée.",
                 "explication": r"Oui : \(\mathbb{E}[\hat{\boldsymbol{\beta}}] = \boldsymbol{\beta} + \gamma\,\boldsymbol{\delta}\), et \(\gamma\,\delta > 0\). La variable \(\mathbf{x}_2\) « capte » une partie de l'effet de \(\mathbf{z}\), qui varie avec elle."},
                {"id": "b", "texte": r"\(\beta_2\), car les bruits sont centrés.",
                 "explication": r"Le bruit du modèle estimé n'est plus \(\varepsilon_i\) mais \(\gamma z_i + \varepsilon_i\), qui n'est pas centré et varie avec \(\mathbf{x}_2\) : l'hypothèse (A2) est violée pour le modèle estimé."},
                {"id": "c", "texte": r"\(\beta_2 - \gamma\,\delta\) : la pente est sous-estimée.",
                 "explication": r"Erreur de signe : \(\mathbb{E}[\hat{\boldsymbol{\beta}}] = \mathbf{A}(\mathbf{X}\boldsymbol{\beta} + \gamma\,\mathbf{z}) = \boldsymbol{\beta} + \gamma\,\mathbf{A}\mathbf{z}\), et \(\mathbf{A}\mathbf{z} = \boldsymbol{\delta}\) s'ajoute."},
                {"id": "d", "texte": r"\(\beta_2 + \gamma\) : tout l'effet de \(\mathbf{z}\) passe dans la pente.",
                 "explication": r"Seule la partie de \(\mathbf{z}\) alignée sur \(\mathbf{x}_2\) passe dans la pente, dans la proportion \(\delta\). Si \(\mathbf{z}\) était orthogonale à \((\mathbf{1}, \mathbf{x}_2)\), il n'y aurait aucun biais."},
            ],
        },
        "r2": {
            "question": r"On ajoute à une régression avec constante une colonne de bruit pur, tirée indépendamment de tout le reste. Qu'est-ce qui est vrai **à coup sûr** sur l'échantillon d'apprentissage ?",
            "options": [
                {"id": "a", "texte": r"Le R² ne diminue pas.",
                 "explication": r"Oui : \(\mathrm{SCR}\) diminue de \(c^2 \lVert \tilde{\mathbf{x}} \rVert^2 \ge 0\) et \(\mathrm{SCT}\) ne change pas. Le minimum sur un ensemble plus grand ne peut pas être plus grand."},
                {"id": "b", "texte": r"Le R² diminue, car la colonne n'a aucun pouvoir explicatif.",
                 "explication": r"Faux : sur l'échantillon d'apprentissage, la nouvelle colonne s'ajuste au bruit ; elle ne peut pas faire monter \(\mathrm{SCR}\), puisque le coefficient 0 reste disponible."},
                {"id": "c", "texte": r"Le R² ajusté augmente aussi.",
                 "explication": r"Pas à coup sûr : il augmente si et seulement si la statistique t de la nouvelle colonne vérifie \(t^2 > 1\). Pour une colonne de bruit, cela arrive pour une partie seulement des tirages."},
                {"id": "d", "texte": r"Le R² sur des données de test augmente aussi.",
                 "explication": r"Pas garanti : le coefficient de la colonne de bruit est ajusté au bruit d'apprentissage ; sur des données nouvelles, il ajoute de la variance de prédiction sans réduire le biais, et l'expérience de la partie 3 montre un R² de test qui baisse jusqu'à devenir négatif."},
            ],
        },
        "conditionnement": {
            "question": r"\(\mathbf{X}^\top \mathbf{X}\) a pour valeurs propres 9 et 0,01. Quel est le conditionnement de \(\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}\) pour \(\lambda = 1\) ?",
            "options": [
                {"id": "a", "texte": r"\((@C5_kappa:2m@)\), contre \(@C5_kappa0:0@\) sans pénalité.",
                 "explication": r"Oui : les valeurs propres deviennent \(9 + 1\) et \(0{,}01 + 1\), d'où \(10/1{,}01 \approx @C5_kappa:2m@\)."},
                {"id": "b", "texte": r"\(@C5_kappa0:0@\), inchangé, car ajouter \(\lambda \mathbf{I}\) ne change pas les vecteurs propres.",
                 "explication": r"Les vecteurs propres ne changent pas, mais toutes les valeurs propres augmentent de \(\lambda\), et le rapport \((9 + 1)/(0{,}01 + 1)\) est bien plus petit que \(9/0{,}01\)."},
                {"id": "c", "texte": r"\((@C5_d_petit:2m@)\)",
                 "explication": r"C'est \(9/(0{,}01 + 1)\) : \(\lambda\) n'a été ajouté qu'à la petite valeur propre. Il s'ajoute à toutes."},
                {"id": "d", "texte": r"\(@C5_d_racine:0@\)",
                 "explication": r"C'est \(\sqrt{9/0{,}01}\), le conditionnement de \(\mathbf{X}\) sans pénalité (\(\kappa(\mathbf{X}^\top \mathbf{X}) = \kappa(\mathbf{X})^2\)). La question porte sur \(\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}\)."},
            ],
        },
        "seuil": {
            "question": r"Lasso \(\tfrac12 \lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2 + \lambda \lVert \mathbf{b} \rVert_1\) avec \(\mathbf{X}^\top \mathbf{y} = @C6_xty:1m@\). Pour quels \(\lambda\) la solution est-elle le vecteur nul ?",
            "options": [
                {"id": "a", "texte": r"Exactement pour \(\lambda \ge @C6_max:1m@\), le plus grand \(\lvert \mathbf{x}_j^\top \mathbf{y} \rvert\).",
                 "explication": r"Oui : en \(\mathbf{b} = \mathbf{0}\), \(\mathbf{g} = \mathbf{X}^\top \mathbf{y}\), et les conditions d'optimalité demandent \(\lvert g_j \rvert \le \lambda\) pour chaque \(j\)."},
                {"id": "b", "texte": r"Exactement pour \(\lambda \ge @C6_l1:1m@\), la somme des \(\lvert \mathbf{x}_j^\top \mathbf{y} \rvert\).",
                 "explication": r"Cette condition est suffisante mais pas nécessaire : pour \(\lambda\) entre \(@C6_max:1m@\) et \(@C6_l1:1m@\), chaque \(\lvert g_j \rvert\) est déjà sous \(\lambda\), et la solution est nulle."},
                {"id": "c", "texte": r"Exactement pour \(\lambda \ge @C6_l2:3m@\), la norme euclidienne de \(\mathbf{X}^\top \mathbf{y}\).",
                 "explication": r"Les conditions portent sur chaque coordonnée séparément (norme du maximum), pas sur la norme euclidienne ; pour \(\lambda\) entre \(@C6_max:1m@\) et \(@C6_l2:3m@\), la solution est déjà nulle."},
                {"id": "d", "texte": r"Pour aucun \(\lambda\) fini : la solution tend vers 0 sans l'atteindre.",
                 "explication": r"C'est le comportement de Ridge. La pénalité L1 garde une pente \(\lambda\) en 0, ce qui crée un seuil fini au-delà duquel tout est nul."},
            ],
        },
        "biais_variance": {
            "question": r"Ridge, \(\mathbf{X}\) de rang \(p\), \(\sigma > 0\), \(\boldsymbol{\beta} \ne \mathbf{0}\). Quand \(\lambda\) augmente de 0 vers l'infini, que se passe-t-il ?",
            "options": [
                {"id": "a", "texte": r"La norme du biais croît de 0 vers \(\lVert \boldsymbol{\beta} \rVert\), la variance décroît vers 0, et l'erreur quadratique commence par décroître.",
                 "explication": r"Oui : facteurs \(\lambda/(\mu_j + \lambda)\) croissants sur le biais, \(\mu_j/(\mu_j + \lambda)^2\) décroissants sur la variance, et chaque terme de l'erreur décroît tant que \(\lambda < \sigma^2/\theta_j^2\) (Hoerl et Kennard)."},
                {"id": "b", "texte": r"Le biais et la variance décroissent tous les deux.",
                 "explication": r"Le biais vaut \(-\lambda (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1}\boldsymbol{\beta}\) : nul en \(\lambda = 0\), il croît en norme. C'est cet arbitrage qui fait exister un \(\lambda\) optimal."},
                {"id": "c", "texte": r"Le biais décroît et la variance croît.",
                 "explication": r"C'est l'inverse : \(\lambda = 0\) donne les MCO, sans biais et de variance maximale ; la pénalité rétrécit vers 0, ce qui réduit la variance et crée du biais."},
                {"id": "d", "texte": r"L'erreur quadratique croît dès \(\lambda > 0\), car les MCO sont sans biais et optimaux.",
                 "explication": r"Les MCO ne sont optimaux que parmi les estimateurs linéaires sans biais. La dérivée de l'erreur en \(\lambda = 0\) vaut \(-2\sigma^2 \sum_j 1/\mu_j^2 < 0\) : un petit \(\lambda\) fait toujours mieux."},
            ],
        },
    },
    "cellules_initiales": [
        "import numpy as np\nfrom sklearn.linear_model import Lasso, Ridge\nfrom sklearn.metrics import r2_score\nnp.set_printoptions(precision=4, suppress=True)",
        "# Essaie ici les expériences de la leçon, en les modifiant.\nrng = np.random.default_rng(0)\nX = rng.standard_normal((50, 5))\ny = X.dot(np.array([1.0, 0.0, 0.0, -2.0, 0.5])) + rng.standard_normal(50)\nfor lam in [1.0, 10.0, 100.0]:\n    print(lam, Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_, Lasso(alpha=lam / 50, fit_intercept=False).fit(X, y).coef_)",
    ],
    "verif": {"script": "verif/seance_ml_1.py"},
}
