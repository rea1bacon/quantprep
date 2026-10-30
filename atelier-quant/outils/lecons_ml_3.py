"""Leçon express de la séance de ML 3 : régression logistique, métriques de classification, plus proches voisins,
mise à l'échelle, malédiction de la dimension, validation (fuites, validation croisée temporelle, purge, embargo).

Format : voir outils/lecons.py (élément de type "lecon", contenu en Markdown réduit, points de contrôle
[[controle:IDENT]] sur une ligne seule, option "a" correcte, une explication par option).
Aucun nombre calculé n'est écrit à la main : chaque jeton @CLE@, @CLE:3@ ou @CLE:3m@ est remplacé par la valeur
calculée par verif/seance_ml_3.py (fichier verif/seance_ml_3.out.json), et les blocs ```python-exec sont exécutés
à la construction par le noyau de l'environnement .venv-hackerrank (scikit-learn 1.1.1, numpy 1.21.6, scipy 1.8.1),
qui insère leur vraie sortie. Graines aléatoires fixées dans chaque bloc.
La régression linéaire et la décomposition biais-variance sont dans la leçon de la séance de ML 1
(outils/lecons_reglin.py) ; les arbres dans celle de la séance de ML 2 (outils/lecons_arbres.py).
"""

LECON_ML_3 = {
    "id": "L-ML-3", "type": "lecon", "section": "lecon",
    "theme": "Leçon express : régression logistique, métriques, plus proches voisins, validation",
    "difficulte": 4, "temps_cible_s": 1800, "dossier": "coding/lecons",
    "contenu": r"""## Leçon express : classification, métriques et validation

Durée visée : 30 minutes. Quatre parties : la régression logistique, les métriques de classification, les plus proches voisins avec la mise à l'échelle et la malédiction de la dimension, puis la validation (fuites d'information, validation croisée temporelle, purge et embargo). Chaque résultat suit l'ordre : le problème, l'énoncé, la preuve, un exemple chiffré, les cas limites, les pièges. La régression linéaire et la décomposition biais-variance sont supposées connues : leçon de la séance de ML 1 (`outils/lecons_reglin.py`), parties 1 et 6.

**Conventions de notation**, valables pour toute la leçon :

- une majuscule non grasse désigne une variable aléatoire, une minuscule une valeur ou un paramètre fixe ; en gras, une minuscule est un vecteur colonne, une majuscule une matrice fixe ;
- les données comptent \(n\) observations ; l'observation \(i\) a un vecteur fixe \(\mathbf{x}_i \in \mathbb{R}^p\) de \(p\) variables, dont la première vaut 1 quand le modèle a une constante, et une étiquette \(y_i \in \{0, 1\}\), valeur observée de la variable aléatoire \(Y_i\) ; \(\mathbf{X}\) est la matrice \(n \times p\) de lignes \(\mathbf{x}_i^\top\), et \(\mathbf{y}\) le vecteur des \(y_i\) ;
- exceptions d'usage, déclarées : les effectifs VP, FP, FN, VN de la matrice de confusion, les taux TVP et TFP, l'AUC et le score de Brier gardent leur nom habituel, et sont des nombres calculés sur des données observées.

Tous les nombres hors des sorties de code sont calculés par `verif/seance_ml_3.py`. Le premier bloc prépare les outils ; les suivants réutilisent ses variables.

```python-exec
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_classif
np.set_printoptions(precision=4, suppress=True)

def voisins(F_app, y_app, F_test, k):
    # moyenne des réponses des k plus proches voisins (distance euclidienne) de chaque ligne de F_test
    d2 = ((F_test[:, None, :] - F_app[None, :, :]) ** 2).sum(axis=2)
    proches = np.argsort(d2, axis=1, kind="stable")[:, :k]   # égalités départagées par l'ordre des lignes
    return y_app[proches].mean(axis=1)

print("prêt")
```

La fonction `voisins` calcule la prédiction des plus proches voisins directement avec numpy : la règle est visible en trois lignes, et le calcul est plus rapide que la classe de scikit-learn dans cet environnement.

## Partie 1 : la régression logistique

### Le problème

On veut prédire une étiquette binaire et, surtout, sa **probabilité**. Les moindres carrés sur des étiquettes 0 ou 1 peuvent prédire hors de \([0, 1]\), et supposent une variance du bruit constante, alors qu'une variable de Bernoulli de paramètre \(\pi\) a pour variance \(\pi(1 - \pi)\), qui dépend de \(\pi\).

### Notations et modèle

- \(\sigma(t) = 1/(1 + e^{-t})\), pour \(t\) réel : la fonction logistique, strictement croissante de \(\mathbb{R}\) dans \(]0, 1[\).
- \(\mathbf{b} \in \mathbb{R}^p\) : un vecteur de coefficients candidat ; \(t_i = \mathbf{x}_i^\top \mathbf{b}\) : le score linéaire de l'observation \(i\) ; \(\pi_i(\mathbf{b}) = \sigma(t_i)\) : la probabilité prédite ; \(\boldsymbol{\pi}(\mathbf{b})\) : le vecteur des \(\pi_i(\mathbf{b})\).
- La **cote** d'une probabilité \(\pi \in ]0, 1[\) est \(\pi/(1 - \pi)\) ; son logarithme est le **logit**.
- **Modèle** : les \(Y_i\) sont indépendantes, et \(Y_i\) suit la loi de Bernoulli de paramètre \(\sigma(\mathbf{x}_i^\top \boldsymbol{\beta})\), où \(\boldsymbol{\beta} \in \mathbb{R}^p\) est fixe et inconnu. Comme \(\ln\big(\sigma(t)/(1 - \sigma(t))\big) = t\), le modèle dit que le logit de la probabilité est linéaire en \(\mathbf{x}\).
- **Vraisemblance** : \(\prod_i \pi_i(\mathbf{b})^{y_i} (1 - \pi_i(\mathbf{b}))^{1 - y_i}\). **Log-loss** (entropie croisée), moins le logarithme de la vraisemblance divisé par \(n\) : \(\displaystyle \ell(\mathbf{b}) = -\frac{1}{n} \sum_{i=1}^n \big[y_i \ln \pi_i(\mathbf{b}) + (1 - y_i) \ln(1 - \pi_i(\mathbf{b}))\big] = \frac{1}{n} \sum_{i=1}^n \big[\ln(1 + e^{t_i}) - y_i t_i\big]\). La seconde forme vient de \(\ln \sigma(t) = t - \ln(1 + e^t)\) et \(\ln(1 - \sigma(t)) = -\ln(1 + e^t)\). Maximiser la vraisemblance, c'est minimiser \(\ell\).
- \(\mathbf{W}(\mathbf{b})\) : la matrice diagonale \(n \times n\) de coefficients \(\pi_i(\mathbf{b})(1 - \pi_i(\mathbf{b}))\).

### L'énoncé

1. **Gradient** : \(\nabla \ell(\mathbf{b}) = \tfrac{1}{n} \mathbf{X}^\top (\boldsymbol{\pi}(\mathbf{b}) - \mathbf{y})\). Même forme que les moindres carrés, avec la probabilité prédite à la place de la valeur ajustée.
2. **Hessienne** : \(\nabla^2 \ell(\mathbf{b}) = \tfrac{1}{n} \mathbf{X}^\top \mathbf{W}(\mathbf{b}) \mathbf{X}\), semi-définie positive : \(\ell\) est **convexe**. Si \(\mathbf{X}\) est de rang \(p\), elle est définie positive : \(\ell\) est strictement convexe, a **au plus un** minimiseur, et tout point où le gradient s'annule est ce minimiseur global.
3. **Pas de forme fermée** : l'équation \(\mathbf{X}^\top (\boldsymbol{\pi}(\mathbf{b}) - \mathbf{y}) = \mathbf{0}\) n'est pas linéaire en \(\mathbf{b}\) (à cause de \(\sigma\)). On la résout par itérations : méthode de Newton \(\mathbf{b} \leftarrow \mathbf{b} - (\nabla^2 \ell)^{-1} \nabla \ell\) (appelée aussi moindres carrés repondérés itérés, IRLS), ou descente de gradient ; leur convergence est admise.
4. **Avec une constante**, au minimiseur \(\hat{\boldsymbol{\beta}}\) : \(\sum_i \pi_i(\hat{\boldsymbol{\beta}}) = \sum_i y_i\). La probabilité moyenne prédite sur l'échantillon d'apprentissage égale la fréquence observée.
5. **Rapport de cotes** : si \(x_j\) augmente de 1, les autres variables fixées, la cote prédite est multipliée par \(e^{\beta_j}\), quel que soit le point de départ. La probabilité, elle, change d'une quantité qui dépend du point de départ : sa dérivée par rapport à \(x_j\) vaut \(\beta_j \pi (1 - \pi)\).
6. **Séparation parfaite** : s'il existe \(\mathbf{b}^\star\) tel que \(\mathbf{x}_i^\top \mathbf{b}^\star > 0\) pour tout \(i\) avec \(y_i = 1\) et \(\mathbf{x}_i^\top \mathbf{b}^\star < 0\) pour tout \(i\) avec \(y_i = 0\), alors \(\ell\) n'a **pas** de minimiseur : sa borne inférieure est 0, non atteinte, et toute suite \((\mathbf{b}_m)\) telle que \(\ell(\mathbf{b}_m) \to 0\) vérifie \(\lVert \mathbf{b}_m \rVert \to \infty\).

### La preuve

1. **Gradient.** La dérivée de \(t \mapsto \ln(1 + e^t) - y t\) vaut \(e^t/(1 + e^t) - y = \sigma(t) - y\), et \(\partial t_i / \partial b_j = x_{ij}\). Par dérivation composée, \(\partial \ell / \partial b_j = \tfrac1n \sum_i (\pi_i - y_i) x_{ij}\) : c'est la coordonnée \(j\) de \(\tfrac1n \mathbf{X}^\top (\boldsymbol{\pi} - \mathbf{y})\).
2. **Hessienne.** \(\sigma'(t) = e^{-t}/(1 + e^{-t})^2 = \sigma(t)(1 - \sigma(t))\). En dérivant l'étape 1 par rapport à \(b_k\) : \(\partial^2 \ell / \partial b_j \partial b_k = \tfrac1n \sum_i \pi_i (1 - \pi_i) x_{ij} x_{ik}\), coefficient \((j, k)\) de \(\tfrac1n \mathbf{X}^\top \mathbf{W}\mathbf{X}\).
3. **Signe.** Pour \(\mathbf{v} \in \mathbb{R}^p\) : \(\mathbf{v}^\top \mathbf{X}^\top \mathbf{W}\mathbf{X}\mathbf{v} = \sum_i \pi_i (1 - \pi_i) (\mathbf{x}_i^\top \mathbf{v})^2 \ge 0\). Chaque poids \(\pi_i(1 - \pi_i)\) est strictement positif, car \(\sigma\) prend ses valeurs dans \(]0, 1[\) ; la somme est donc nulle si et seulement si \(\mathbf{X}\mathbf{v} = \mathbf{0}\), ce qui force \(\mathbf{v} = \mathbf{0}\) quand \(\mathbf{X}\) est de rang \(p\).
4. **Convexité et minimum global.** Formule de Taylor avec reste intégral, pour \(\mathbf{h} \in \mathbb{R}^p\) : \(\ell(\mathbf{b} + \mathbf{h}) = \ell(\mathbf{b}) + \nabla \ell(\mathbf{b})^\top \mathbf{h} + \int_0^1 (1 - u)\,\mathbf{h}^\top \nabla^2 \ell(\mathbf{b} + u\mathbf{h})\,\mathbf{h}\,du\). Par l'étape 3, l'intégrale est positive, strictement si \(\mathbf{h} \ne \mathbf{0}\) et \(\mathbf{X}\) de rang \(p\). Si \(\nabla \ell(\mathbf{b}) = \mathbf{0}\), on obtient \(\ell(\mathbf{b} + \mathbf{h}) \ge \ell(\mathbf{b})\) pour tout \(\mathbf{h}\), avec inégalité stricte pour \(\mathbf{h} \ne \mathbf{0}\) dans le cas de rang \(p\) : minimum global, unique.
5. **Constante.** La coordonnée du gradient associée à la colonne \(\mathbf{1}\) est \(\tfrac1n \sum_i (\pi_i - y_i)\), nulle au minimiseur.
6. **Cotes.** La cote prédite au point \(\mathbf{x}\) vaut \(e^{\mathbf{x}^\top \boldsymbol{\beta}}\). Remplacer \(x_j\) par \(x_j + 1\) multiplie l'exposant par \(e^{\beta_j}\). Pour la probabilité : \(\partial \sigma(\mathbf{x}^\top \boldsymbol{\beta}) / \partial x_j = \beta_j\,\sigma'(\mathbf{x}^\top \boldsymbol{\beta}) = \beta_j \pi (1 - \pi)\).
7. **Séparation.** On pose \(\mathbf{b} = c\,\mathbf{b}^\star\), \(c > 0\), et \(a_i = \lvert \mathbf{x}_i^\top \mathbf{b}^\star \rvert > 0\). Si \(y_i = 1\), le terme \(i\) vaut \(\ln(1 + e^{t_i}) - t_i = \ln(1 + e^{-t_i}) = \ln(1 + e^{-c a_i})\) ; si \(y_i = 0\), il vaut \(\ln(1 + e^{t_i}) = \ln(1 + e^{-c a_i})\). Chacun tend vers 0 quand \(c \to \infty\), donc la borne inférieure de \(\ell\) est 0. Or pour tout \(\mathbf{b}\), chaque terme s'écrit \(\ln(1 + e^{\pm t_i}) > 0\), donc \(\ell(\mathbf{b}) > 0\) : la borne n'est pas atteinte. Enfin, si une suite \((\mathbf{b}_m)\) avec \(\ell(\mathbf{b}_m) \to 0\) avait une sous-suite bornée, elle aurait une sous-suite convergente (Bolzano-Weierstrass), de limite \(\mathbf{b}_\infty\) avec \(\ell(\mathbf{b}_\infty) = 0\) par continuité : contradiction.

### Exemple chiffré

Quatre observations, une constante et une variable : \(x = (-1, 0, 1, 2)\), \(\mathbf{y} = (0, 1, 0, 1)\). Pas de séparation : l'étiquette 1 en \(x = 0\) est encadrée par des étiquettes 0.

1. En \(\mathbf{b} = \mathbf{0}\), toutes les probabilités valent \(\tfrac12\) et \(\ell(\mathbf{0}) = \ln 2 \approx @L1_ln2:4m@\).
2. Gradient : \(\tfrac14 \mathbf{X}^\top (\tfrac12 \mathbf{1} - \mathbf{y}) = @L1_g0:2m@\) (constante : deux étiquettes 1 sur quatre, somme nulle ; pente : \(\tfrac14 \big((-1)\tfrac12 + 0 + 1 \cdot \tfrac12 + 2 \cdot (-\tfrac12)\big)\)). Négatif sur la pente : augmenter la pente fait baisser la perte.
3. Hessienne : les poids \(\pi_i(1 - \pi_i)\) valent tous \(@L1_poids0:2m@\), donc \(\nabla^2 \ell(\mathbf{0}) = \tfrac{@L1_hfact_num@}{@L1_hfact_den@} \mathbf{X}^\top \mathbf{X} = \begin{pmatrix} @L1_h11:3m@ & @L1_h12:3m@ \\ @L1_h12:3m@ & @L1_h22:3m@ \end{pmatrix}\), avec \(\mathbf{X}^\top \mathbf{X} = \begin{pmatrix} @L1_xtx11@ & @L1_xtx12@ \\ @L1_xtx12@ & @L1_xtx22@ \end{pmatrix}\). Premier pas de Newton : \(\mathbf{b} = @L1_newton1:1m@\).
4. Le minimiseur est \(\hat{\boldsymbol{\beta}} \approx @L1_bhat:3m@\), de log-loss \(@L1_loss_hat:4m@ < \ln 2\). Probabilités ajustées \(@L1_pihat:3m@\), de somme \(@L1_somme_pihat@\), égale au nombre d'étiquettes 1 (énoncé 4). Rapport de cotes de la pente : \(e^{@L1_bhat1:3m@} \approx @L1_or_pente:2m@\).

```python-exec
x = np.array([-1.0, 0.0, 1.0, 2.0])
y = np.array([0, 1, 0, 1])
X = np.column_stack([np.ones(4), x])
def sigma(t):
    return 1 / (1 + np.exp(-t))
b = np.zeros(2)
for etape in range(5):
    pi = sigma(X.dot(b))
    g = X.T.dot(pi - y) / 4
    H = X.T.dot(X * (pi * (1 - pi))[:, None]) / 4
    perte = np.mean(np.logaddexp(0, X.dot(b)) - y * X.dot(b))
    print(f"itération {etape} : b = {b}, log-loss = {perte:.6f}, gradient = {g}")
    b = b - np.linalg.solve(H, g)
m = LogisticRegression(penalty="none", tol=1e-10, max_iter=1000).fit(x.reshape(-1, 1), y)
print("scikit-learn sans pénalité : constante", m.intercept_, "pente", m.coef_[0])
print("somme des probabilités ajustées :", round(sigma(X.dot(b)).sum(), 10), "; somme des y :", y.sum())
```

Rapport de cotes, coefficient \(\beta_j = @L1_beta_or:1m@\), soit un facteur \(e^{@L1_beta_or:1m@} \approx @L1_or:3m@\) sur la cote. Depuis \(\pi = 0{,}5\) (cote 1), la cote passe à \(@L1_p05_cote:3m@\) et la probabilité à \(@L1_p05:3m@\), soit \(+@L1_p05_hausse:3m@\). Depuis \(\pi = 0{,}9\) (cote \(@L1_p09_cote0:0m@\)), la cote passe à \(@L1_p09_cote:2m@\) et la probabilité à \(@L1_p09:3m@\), soit \(+@L1_p09_hausse:3m@\) seulement : même coefficient, effets différents sur la probabilité.

### Cas limites

- **Séparation** : \(x = (-2, -1, 1, 2)\), \(\mathbf{y} = (0, 0, 1, 1)\), \(\mathbf{b}^\star = (0, 1)\). Le long de \(c\,\mathbf{b}^\star\), la log-loss vaut \(@L1_sep_c1:4m@\) pour \(c = 1\), \(@L1_sep_c5:4m@\) pour \(c = 5\), \(@L1_sep_c10:7m@\) pour \(c = 10\) : elle tend vers 0, et la pente estimée part à l'infini. Quasi-séparation (inégalités larges, avec au moins une égalité) : le minimiseur peut aussi ne pas exister (non démontré ici).
- **Toutes les étiquettes égales** : c'est une séparation, par la direction de la constante (\(\mathbf{b}^\star = (1, 0, \dots, 0)\) si toutes valent 1) ; pas de minimiseur.
- **Pénalité L2**, qui est le défaut de scikit-learn (vérifié ci-dessous sous scikit-learn 1.1.1 : `penalty="l2"`, `C=1.0`). Notons \(\tilde{\mathbf{x}}_i\) les variables de l'observation \(i\) sans la constante, \(\mathbf{w}\) leurs coefficients et \(w_0\) la constante, de sorte que \(t_i = w_0 + \tilde{\mathbf{x}}_i^\top \mathbf{w}\). `LogisticRegression` minimise \(\tfrac12 \lVert \mathbf{w} \rVert^2 + C \sum_i \big[\ln(1 + e^{t_i}) - y_i t_i\big]\) : la constante n'est pas pénalisée avec le solveur par défaut `lbfgs` (les conditions du premier ordre sont contrôlées ci-dessous). Si les deux classes sont présentes, ce critère a un unique minimiseur, même en cas de séparation. Stricte convexité : dans une direction \((\mathbf{v}, v_0)\), la dérivée seconde vaut \(\lVert \mathbf{v} \rVert^2 + C \sum_i \pi_i(1 - \pi_i)(v_0 + \tilde{\mathbf{x}}_i^\top \mathbf{v})^2\), strictement positive si \(\mathbf{v} \ne \mathbf{0}\), et égale à \(C\,v_0^2 \sum_i \pi_i(1 - \pi_i) > 0\) si \(\mathbf{v} = \mathbf{0}\) et \(v_0 \ne 0\). Existence : le critère tend vers l'infini quand \((\mathbf{w}, w_0)\) s'éloigne (la pénalité contrôle \(\mathbf{w}\) ; à \(\mathbf{w}\) borné, une constante très grande rend infinie la perte des étiquettes 0, une constante très négative celle des étiquettes 1), et une fonction continue qui tend vers l'infini atteint son minimum (admis, comme dans la leçon de ML 1). Les coefficients grandissent avec \(C\).

```python-exec
print({k: v for k, v in LogisticRegression().get_params().items() if k in ("penalty", "C", "solver")})
xs = np.array([[-2.0], [-1.0], [1.0], [2.0]])
ys = np.array([0, 0, 1, 1])
for C in [1.0, 1e2, 1e4, 1e6]:
    m = LogisticRegression(C=C, tol=1e-12, max_iter=100000).fit(xs, ys)
    p = m.predict_proba(xs)[:, 1]
    cond_w = m.coef_[0, 0] + C * np.sum((p - ys) * xs[:, 0])     # dérivée du critère en w
    cond_c = C * np.sum(p - ys)                                 # dérivée en la constante
    print(f"C = {C:9.0f} : pente = {m.coef_[0, 0]:8.4f} ; conditions du premier ordre : {cond_w: .1e}, {cond_c: .1e}")
```

### Pièges

1. \(e^{\beta_j}\) multiplie la **cote**, pas la probabilité.
2. Comparer des coefficients n'a de sens qu'à échelles de variables comparables ; avec pénalité, l'échelle change aussi la solution (partie 3).
3. Des coefficients énormes, ou un avertissement de non-convergence sans pénalité, signalent une séparation : le maximum de vraisemblance n'existe pas, ce n'est pas un modèle « très sûr ».
4. L'énoncé 4 donne une calibration **en moyenne** sur l'apprentissage, pas une calibration par niveau de probabilité (partie 2), ni hors échantillon.

[[controle:convexite]]

## Partie 2 : les métriques de classification

### Matrice de confusion

Un classifieur produit un score réel \(s_i\) par observation (par exemple \(\pi_i\)) ; avec un seuil \(u\), il prédit 1 si \(s_i > u\). On compte VP (vrais positifs : prédit 1, étiquette 1), FP (prédit 1, étiquette 0), FN (prédit 0, étiquette 1), VN (prédit 0, étiquette 0). Définitions :

- **précision** \(= \mathrm{VP}/(\mathrm{VP} + \mathrm{FP})\) : part de vrais positifs parmi les prédictions positives ;
- **rappel**, ou taux de vrais positifs, \(\mathrm{TVP} = \mathrm{VP}/(\mathrm{VP} + \mathrm{FN})\) : part des positifs détectés ;
- taux de faux positifs \(\mathrm{TFP} = \mathrm{FP}/(\mathrm{FP} + \mathrm{VN})\) : part des négatifs pris pour des positifs ;
- **exactitude** (accuracy) \(= (\mathrm{VP} + \mathrm{VN})/n\) ;
- **F1**, moyenne harmonique de la précision \(P\) et du rappel \(R\) : \(2PR/(P + R) = 2\mathrm{VP}/(2\mathrm{VP} + \mathrm{FP} + \mathrm{FN})\). Preuve : en remplaçant \(P\) et \(R\) par leurs définitions, \(2PR = 2\mathrm{VP}^2/\big((\mathrm{VP} + \mathrm{FP})(\mathrm{VP} + \mathrm{FN})\big)\) et \(P + R = \mathrm{VP}(2\mathrm{VP} + \mathrm{FP} + \mathrm{FN})/\big((\mathrm{VP} + \mathrm{FP})(\mathrm{VP} + \mathrm{FN})\big)\) ; on divise.

Exemple : \(n = @L2_n@\), dont \(@L2_pos@\) positifs ; VP = 60, FN = 40, FP = 90, VN = 810. Précision \(@L2_prec:2m@\), rappel \(@L2_rap:2m@\), F1 \(@L2_f1:2m@\), TFP \(@L2_tfp:2m@\), exactitude \(@L2_exact:2m@\). Le classifieur constant « toujours 0 » a une exactitude de \(@L2_const:2m@\), **meilleure**, avec un rappel nul : sous déséquilibre, l'exactitude récompense l'ignorance de la classe rare. Elle vaut \(1 - q\) pour le classifieur constant, où \(q\) est la proportion de positifs (la prévalence).

### Pourquoi la précision dépend de la prévalence, et pas la courbe ROC

Au niveau de la population : \(q = \mathbb{P}(Y = 1)\), \(\mathrm{TVP} = \mathbb{P}(\text{prédit } 1 \mid Y = 1)\), \(\mathrm{TFP} = \mathbb{P}(\text{prédit } 1 \mid Y = 0)\). Par la formule de Bayes :

\(\displaystyle \text{précision} = \mathbb{P}(Y = 1 \mid \text{prédit } 1) = \frac{q\,\mathrm{TVP}}{q\,\mathrm{TVP} + (1 - q)\,\mathrm{TFP}}.\)

TVP et TFP sont conditionnels à la classe : ils ne dépendent pas de \(q\), et la courbe ROC non plus. La précision en dépend. Exemple, TVP = 0,9 et TFP = 0,05 : pour \(q = 0{,}01\), précision \(= @L2_prec_rare_vp:3m@/(@L2_prec_rare_vp:3m@ + @L2_prec_rare_fp:4m@) = \tfrac{@L2_prec_rare_num@}{@L2_prec_rare_den@} \approx @L2_prec_rare:3m@\) : \(@L2_fp_par_vp:1m@\) fausses alertes par vraie ; pour \(q = 0{,}5\), \(\tfrac{@L2_prec_eq_num@}{@L2_prec_eq_den@} \approx @L2_prec_eq:3m@\). Même point ROC, conséquences opposées. La **courbe précision-rappel** (précision en fonction du rappel quand le seuil varie) montre ce que la ROC cache quand les positifs sont rares : le nombre de faux positifs par vrai positif. Sa référence au hasard est la droite horizontale de hauteur \(q\), celle de la ROC la diagonale.

### Courbe ROC et AUC

Quand le seuil \(u\) descend de \(+\infty\) à \(-\infty\), le point \((\mathrm{TFP}(u), \mathrm{TVP}(u))\) va de \((0, 0)\) à \((1, 1)\) ; en joignant les points successifs par des segments, on obtient la **courbe ROC**, et l'**AUC** est l'aire sous cette courbe.

**Énoncé.** On a \(n_+\) positifs de scores \(s_1^+, \dots, s_{n_+}^+\) et \(n_-\) négatifs de scores \(s_1^-, \dots, s_{n_-}^-\), tous distincts. Alors \(\displaystyle \mathrm{AUC} = \frac{1}{n_+ n_-} \#\{(i, k) : s_i^+ > s_k^-\} = \mathbb{P}(S^+ > S^-)\), où \(S^+\) est tiré uniformément parmi les scores positifs et \(S^-\), indépendamment, parmi les scores négatifs.

**Preuve.**

1. Les scores étant distincts, chaque fois que \(u\) franchit un score, une seule observation passe en « prédit 1 ». Si c'est un positif, TVP augmente de \(1/n_+\) : pas vertical. Si c'est un négatif, TFP augmente de \(1/n_-\) : pas horizontal. La courbe est un escalier.
2. L'aire est la somme, sur les pas horizontaux, de la largeur \(1/n_-\) multipliée par la hauteur de la courbe à ce moment (les pas verticaux n'ont pas de largeur).
3. Le pas horizontal du négatif \(k\) a lieu quand \(u\) franchit \(s_k^-\) ; la hauteur vaut alors la part des positifs déjà prédits 1, c'est-à-dire de score supérieur à \(s_k^-\) : \(\#\{i : s_i^+ > s_k^-\}/n_+\).
4. En sommant sur \(k\) : \(\mathrm{AUC} = \sum_k \tfrac{1}{n_-} \cdot \tfrac{\#\{i : s_i^+ > s_k^-\}}{n_+}\), la proportion de paires bien ordonnées.

**Ex aequo** entre un positif et un négatif : ils passent ensemble, le pas est un segment diagonal, dont l'aire est la moitié du rectangle : chaque paire ex aequo compte \(\tfrac12\). C'est la convention de `roc_auc_score`, et le script de vérification contrôle l'égalité entre aire et décompte sur 500 cas tirés au hasard.

**Conséquences.** L'AUC ne dépend que de l'**ordre** des scores : elle est invariante par toute transformation strictement croissante, donc aveugle à la calibration. Un score qui ne dépend pas de l'étiquette, à loi continue, donne \(\tfrac12\) en espérance (par symétrie, \(\mathbb{P}(S^+ > S^-) = \mathbb{P}(S^- > S^+)\)). Le score opposé donne \(1 - \mathrm{AUC}\).

Exemple : positifs \((0{,}9 ; 0{,}5 ; 0{,}3)\), négatifs \((0{,}8 ; 0{,}6 ; 0{,}4 ; 0{,}2)\). Chaque positif bat respectivement \(@L2_auc_par_pos@\) négatifs, soit \(@L2_auc_bien@\) paires sur \(@L2_auc_paires@\) : \(\mathrm{AUC} = \tfrac{@L2_auc_num@}{@L2_auc_den@} \approx @L2_auc:3m@\).

```python-exec
pos = np.array([0.9, 0.5, 0.3]); neg = np.array([0.8, 0.6, 0.4, 0.2])
s = np.concatenate([pos, neg]); etiquettes = np.r_[np.ones(3), np.zeros(4)]
print("proportion de paires bien ordonnées :", np.mean(pos[:, None] > neg[None, :]))
print("roc_auc_score                       :", roc_auc_score(etiquettes, s))
print("après la transformation s -> s**3   :", roc_auc_score(etiquettes, s**3))
```

### Log-loss, score de Brier, calibration

Pour des probabilités prédites \(r_i \in [0, 1]\) : log-loss \(-\tfrac1n \sum_i [y_i \ln r_i + (1 - y_i)\ln(1 - r_i)]\) (partie 1), score de Brier \(\tfrac1n \sum_i (r_i - y_i)^2\).

**Énoncé : ce sont des règles de score propres.** Si \(Y\) suit la loi de Bernoulli de paramètre \(q \in ]0, 1[\), l'espérance du score d'une prévision \(r\) est minimale en \(r = q\), et seulement là.

**Preuve.** Brier : \(\mathbb{E}[(r - Y)^2] = q(r - 1)^2 + (1 - q)r^2 = (r - q)^2 + q(1 - q)\), en développant : minimum unique en \(r = q\), et l'excès vaut exactement \((r - q)^2\). Log : \(f(r) = -q \ln r - (1 - q)\ln(1 - r)\) a pour dérivée \(-q/r + (1 - q)/(1 - r)\), nulle seulement en \(r = q\), et pour dérivée seconde \(q/r^2 + (1 - q)/(1 - r)^2 > 0\) : minimum unique. Conséquence : un prévisionniste qui minimise l'un de ces scores en espérance a intérêt à annoncer sa vraie probabilité.

Exemple, \(q = @L2_q:1m@\) : Brier espéré \(@L2_brier_q:2m@\) pour \(r = q\) et \(@L2_brier_05:2m@\) pour \(r = 0{,}5\), écart \((0{,}5 - 0{,}3)^2 = @L2_brier_ecart:2m@\) ; log-loss espérée \(@L2_ll_q:4m@\) contre \(\ln 2 \approx @L2_ll_05:4m@\). Différence de comportement : le score de Brier reste entre 0 et 1 ; la log-loss d'une prévision \(r = 0\) suivie de \(y = 1\) est infinie, et punit très fortement les erreurs confiantes.

**Calibration.** Un modèle est calibré si, pour tout niveau \(r\), parmi les observations auxquelles il attribue la probabilité \(r\), la proportion de positifs est \(r\) : \(\mathbb{P}(Y = 1 \mid \text{prévision} = r) = r\). Le **diagramme de fiabilité** se construit en regroupant les prévisions par tranches, puis en traçant la fréquence observée de positifs contre la prévision moyenne de chaque tranche ; calibré signifie sur la diagonale. L'expérience suivante compare des probabilités vraies \(\pi\) et leur carré \(\pi^2\) : même ordre, donc même AUC, mais \(\pi^2\) sous-estime les probabilités, ce que voient le Brier, la log-loss et le diagramme.

```python-exec
rng = np.random.default_rng(0)
pi = rng.uniform(0, 1, 100000)                       # probabilités vraies
Y = (rng.uniform(size=100000) < pi).astype(int)
tranches = np.linspace(0, 1, 6)
for nom, r in [("pi", pi), ("pi^2", pi**2)]:
    print(f"{nom:4s} : AUC {roc_auc_score(Y, r):.4f}  Brier {brier_score_loss(Y, r):.4f}  log-loss {log_loss(Y, r):.4f}")
    k = np.digitize(r, tranches[1:-1])
    print("       (prévision moyenne, fréquence observée) par tranche :", [(round(r[k == j].mean(), 2), round(Y[k == j].mean(), 2)) for j in range(5)])
```

### Choisir la métrique

Le choix découle de ce que la décision utilise :

- la **valeur** de la probabilité (dimensionner une position, comme la mise par croissance logarithmique d'un système de trading, ou publier un risque) : log-loss ou Brier, et diagramme de fiabilité ; une erreur de calibration devient directement une erreur de taille ;
- seulement l'**ordre** (prendre les 10 % de scores les plus élevés, construire un portefeuille long-court) : AUC, ou corrélation de rang entre signal et rendement (coefficient d'information) ;
- une décision binaire à **coûts asymétriques** \(c_{\mathrm{FP}}\) et \(c_{\mathrm{FN}}\) avec une probabilité calibrée \(\pi\) : prédire 1 coûte \((1 - \pi)c_{\mathrm{FP}}\) en espérance, prédire 0 coûte \(\pi\,c_{\mathrm{FN}}\), d'où le seuil \(u^\star = c_{\mathrm{FP}}/(c_{\mathrm{FP}} + c_{\mathrm{FN}})\) ; pour \(c_{\mathrm{FP}} = 1\), \(c_{\mathrm{FN}} = 4\), \(u^\star = @L2_seuil:1m@\), et non 0,5 ;
- des **positifs rares** dont chaque alerte coûte : courbe précision-rappel, précision à rappel fixé ;
- l'exactitude seulement avec des classes de tailles comparables, et toujours comparée au classifieur constant.

### Pièges

1. Précision et rappel ne se comparent pas d'un jeu de données à l'autre si la prévalence change ; TVP et TFP, si.
2. Une AUC élevée ne dit rien de la calibration ; un modèle calibré peut avoir une AUC de \(\tfrac12\) (prévision constante égale à \(q\)).
3. F1 ignore les VN : il ne convient pas quand les deux classes comptent.

[[controle:auc]]

## Partie 3 : plus proches voisins, échelle, dimension

### Les k plus proches voisins

Pour un point \(\mathbf{x}\), on cherche les \(k\) observations d'apprentissage les plus proches au sens de la distance euclidienne \(\lVert \mathbf{x} - \mathbf{x}_i \rVert = \big(\sum_j (x_j - x_{ij})^2\big)^{1/2}\) (\(k\) entier fixé, égalités départagées par une règle arbitraire). En régression on prédit la moyenne de leurs réponses, en classification la classe majoritaire, ou la proportion de 1 comme probabilité. Aucun paramètre n'est ajusté : tout est dans le choix de \(k\) et de la distance.

\(k\) règle l'arbitrage biais-variance (leçon de la séance de ML 1, partie 6). Si les \(\mathbf{x}_i\) sont fixes et les réponses ont un bruit indépendant de variance \(\sigma^2\), la prédiction en \(\mathbf{x}\) est la moyenne de \(k\) réponses de voisins fixés, de variance \(\sigma^2/k\) : la variance baisse avec \(k\), mais les voisins s'éloignent, et le biais monte. Cas extrêmes : \(k = 1\) donne une erreur d'apprentissage nulle si les \(\mathbf{x}_i\) sont distincts (chaque point est son propre voisin) ; \(k = n\) prédit partout la même moyenne.

### Pourquoi l'échelle compte pour les distances, pas pour les arbres

**Distances.** La variable \(j\) contribue \((x_j - x_{ij})^2\) au carré de la distance, dans **son** unité : la multiplier par \(c\) multiplie sa contribution par \(c^2\). Exemple : point \(Q\) = (30 ans, 50 000 euros), \(A\) = (60 ans, 50 500 euros), \(B\) = (31 ans, 52 000 euros). Brut : \(\lVert Q - A \rVert^2 = @L3_dA_age@ + @L3_dA_rev@\), \(\lVert Q - A \rVert \approx @L3_dA:1m@\) ; \(\lVert Q - B \rVert \approx @L3_dB:1m@\) : le plus proche est \(A\), qui a trente ans de plus, parce que l'euro écrase l'année. Après division par les écarts-types d'apprentissage (supposés de 10 ans et 10 000 euros) : \(\lVert Q - A \rVert \approx @L3_zA:3m@\) et \(\lVert Q - B \rVert \approx @L3_zB:3m@\) : le plus proche devient \(B\). On standardise donc avant un algorithme à distances (plus proches voisins, k-moyennes, noyaux), avec les moyennes et écarts-types de l'**apprentissage seul** (partie 4).

**Arbres.** Une coupure « \(x_j \le u\) » sépare les observations d'apprentissage selon l'**ordre** des valeurs de \(x_j\). Une transformation \(g\) strictement croissante de la variable \(j\) conserve cet ordre : mêmes partitions possibles, mêmes gains, même arbre, le seuil \(u\) devenant \(g(u)\) (leçon de la séance de ML 2, `outils/lecons_arbres.py`, partie 1). Précision sur les points nouveaux : scikit-learn place le seuil au milieu de deux valeurs consécutives ; pour \(g\) affine de pente positive (une standardisation), le milieu est transporté exactement, et les prédictions sont identiques partout ; pour \(g\) non affine, elles sont identiques sur l'apprentissage, et peuvent différer pour un point nouveau tombé entre deux valeurs consécutives.

**Régression logistique.** Sans pénalité, remplacer \(x_j\) par \(c\,x_j\) (\(c \ne 0\)) et \(b_j\) par \(b_j/c\) laisse chaque \(t_i\) inchangé : mêmes prédictions. Avec la pénalité L2, \(b_j^2\) devient \(b_j^2/c^2\) : la pénalité effective change, et les prédictions aussi.

```python-exec
rng = np.random.default_rng(1)
n = 2000
X = rng.standard_normal((n, 2))
y = (X[:, 0] + 0.5 * rng.standard_normal(n) > 0).astype(int)   # seule la variable 0 est utile
X[:, 1] *= 1000                                                # variable inutile, en grandes unités
app, tst = slice(0, 1000), slice(1000, None)
moy, et = X[app].mean(axis=0), X[app].std(axis=0)              # statistiques de l'apprentissage seul
Z = (X - moy) / et
for nom, D in [("brut", X), ("standardisé", Z)]:
    knn = voisins(D[app], y[app], D[tst], 15) > 0.5               # vote majoritaire des 15 voisins
    arbre = DecisionTreeClassifier(max_depth=4, random_state=0).fit(D[app], y[app])
    print(f"{nom:12s} : exactitude test kNN {np.mean(knn == y[tst]):.3f} ; arbre {arbre.score(D[tst], y[tst]):.3f}")
a1 = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X[app], y[app]).predict(X[tst])
a2 = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Z[app], y[app]).predict(Z[tst])
print("prédictions de l'arbre identiques sur le test :", bool(np.all(a1 == a2)))
seule = voisins(Z[app][:, :1], y[app], Z[tst][:, :1], 15) > 0.5            # kNN sur la seule variable utile
print(f"kNN sur la seule variable utile : exactitude test {np.mean(seule == y[tst]):.3f}")
Z20 = np.column_stack([Z[:, :1], rng.standard_normal((n, 20))])             # plus 20 variables de bruit, déjà réduites
avec20 = voisins(Z20[app], y[app], Z20[tst], 15) > 0.5
print(f"kNN avec 20 variables de bruit standardisées : exactitude test {np.mean(avec20 == y[tst]):.3f}")
```

### La malédiction de la dimension : concentration des distances

**Énoncé.** Soient \(\mathbf{X}\) et \(\mathbf{X}'\) deux vecteurs aléatoires de \(\mathbb{R}^p\) (exception d'usage : en gras majuscule, aléatoires), dont les \(2p\) coordonnées sont indépendantes et de même loi, avec un moment d'ordre 4 fini et une variance non nulle. On note \(D^2 = \lVert \mathbf{X} - \mathbf{X}' \rVert^2\), \(m = \mathbb{E}[(X_1 - X_1')^2]\) et \(v = \mathrm{Var}\big((X_1 - X_1')^2\big)\). Alors \(\mathbb{E}[D^2] = p\,m\), \(\mathrm{Var}(D^2) = p\,v\), et pour tout \(\varepsilon > 0\) :

\(\displaystyle \mathbb{P}\Big(\Big\lvert \frac{D^2}{\mathbb{E}[D^2]} - 1 \Big\rvert \ge \varepsilon\Big) \le \frac{v}{p\,m^2\,\varepsilon^2},\)

borne qui tend vers 0 quand \(p \to \infty\), \(\varepsilon\) étant fixé.

Le rapport écart-type sur espérance de \(D^2\) vaut \(\sqrt{v}/(m\sqrt{p})\) : toutes les distances entre points deviennent relativement égales.

**Preuve.**

1. \(D^2 = \sum_{j=1}^p (X_j - X_j')^2\) est une somme de \(p\) variables indépendantes de même loi (chacune ne dépend que de la paire \((X_j, X_j')\), et les paires sont indépendantes) : linéarité de l'espérance, et variance d'une somme de variables indépendantes égale à la somme des variances.
2. **Inégalité de Markov** : pour \(V \ge 0\) et \(a > 0\), \(V \ge a\,I_{\{V \ge a\}}\), où \(I_A\) est l'indicatrice de l'événement \(A\), d'où \(\mathbb{E}[V] \ge a\,\mathbb{P}(V \ge a)\).
3. **Tchebychev** : Markov appliquée à \(V = (D^2 - \mathbb{E}D^2)^2\) et \(a = \varepsilon^2 (\mathbb{E}D^2)^2\) donne \(\mathbb{P}(\lvert D^2 - \mathbb{E}D^2 \rvert \ge \varepsilon\,\mathbb{E}D^2) \le p\,v/(\varepsilon^2 p^2 m^2)\).

Cas uniforme sur \([0, 1]\) : \(m = \tfrac16\), \(\mathbb{E}[(X_1 - X_1')^4] = \tfrac{1}{15}\), donc \(v = \tfrac{@L3_c_num@}{@L3_c_den@}\) (intégrales calculées par sympy), et le rapport vaut \(\sqrt{v}/(m\sqrt{p}) \approx @L3_coef:3m@/\sqrt{p}\) : \(@L3_ratio_2:3m@\) pour \(p = 2\), \(@L3_ratio_10:3m@\) pour \(p = 10\), \(@L3_ratio_100:3m@\) pour \(p = 100\), \(@L3_ratio_1000:3m@\) pour \(p = 1000\). Pour \(p = 1000\) et \(\varepsilon = 0{,}1\), la borne de Tchebychev vaut \(@L3_cheb_1000:2m@\). Simulation pour \(p = 100\) : espérance \(@L3_esp_D2_100_mc:3m@\) et variance \(@L3_var_D2_100_mc:3m@\), pour des valeurs exactes \(100\,m \approx @L3_esp_100:3m@\) et \(100\,v \approx @L3_var_100:3m@\) (script, écarts sous 4 erreurs standard).

L'expérience suivante mesure, pour un point requête et 500 points uniformes, l'écart relatif entre la plus grande et la plus petite distance (constat sur un tirage, pas une preuve) :

```python-exec
rng = np.random.default_rng(2)
for p in [2, 10, 100, 1000]:
    points, requete = rng.uniform(size=(500, p)), rng.uniform(size=p)
    d = np.sqrt(((points - requete) ** 2).sum(axis=1))
    print(f"p = {p:4d} : distance min {d.min():7.3f}, max {d.max():7.3f}, (max - min)/min = {(d.max() - d.min()) / d.min():7.3f}")
```

**Conditions, et ce qui casse le résultat.** L'énoncé exige des coordonnées indépendantes qui apportent chacune une part comparable de variance. Contre-exemple : les \(p\) coordonnées égales à une même variable uniforme \(U\) ; alors \(D^2 = p\,(U - U')^2\), dont le rapport écart-type sur espérance vaut \(\sqrt{v}/m \approx @L3_coef:3m@\) pour tout \(p\) : aucune concentration. Des données réelles dont la variance est portée par quelques facteurs (les rendements de nombreuses actions, par exemple) ont une dimension effective plus petite que \(p\) : on s'attend à une concentration plus faible que dans le cas indépendant (non démontré ici ; le contre-exemple en est le cas extrême).

**Conséquence pour les plus proches voisins** : quand les distances se concentrent, le « plus proche » voisin n'est guère plus proche que les autres, et la prédiction moyenne les réponses d'observations qui ne ressemblent guère plus au point que les autres. Autre face du même phénomène : pour capturer 1 % d'un nuage uniforme dans un sous-cube, il faut un côté de \(0{,}01^{1/p}\), soit \(@L3_arete_10:2m@\) pour \(p = 10\) et \(@L3_arete_100:3m@\) pour \(p = 100\) : un voisinage qui contient assez de points n'est plus local.

### Pièges

1. Standardiser avec des statistiques calculées sur tout l'échantillon est une fuite (partie 4).
2. Une variable inutile en grandes unités ruine les plus proches voisins (expérience ci-dessus). Une fois standardisée, une seule variable de bruit ne coûte rien de mesurable dans cette expérience ; vingt variables de bruit standardisées, qui ajoutent chacune à la distance un terme de même ordre que la variable utile, dégradent nettement l'exactitude. Standardiser ne dispense pas de sélectionner les variables.
3. « La malédiction de la dimension rend les distances inutiles » n'est vrai que sous l'hypothèse d'une dimension effective grande.

[[controle:echelle]]

## Partie 4 : validation et fuites d'information

### Trois ensembles, trois rôles

L'**apprentissage** ajuste les paramètres ; la **validation** choisit les hyperparamètres et le modèle ; le **test** mesure une fois, à la fin, la performance du modèle choisi. Pourquoi ne pas annoncer l'erreur de validation du modèle retenu : si \(\hat{E}_1, \dots, \hat{E}_M\) sont les erreurs de validation (aléatoires) de \(M\) modèles, \(\min_m \hat{E}_m \le \hat{E}_{m'}\) pour chaque \(m'\), donc \(\mathbb{E}[\min_m \hat{E}_m] \le \min_{m'} \mathbb{E}[\hat{E}_{m'}]\) : le minimum est biaisé vers l'optimisme, d'autant plus que \(M\) est grand et les estimations bruitées. C'est un biais de sélection ; ton CV mentionne un biais de sélection révélé par l'audit de ton système de trading : demande-toi s'il relevait de ce mécanisme (garder le meilleur de plusieurs essais) et sache l'expliquer.

### La fuite d'information

**Définition** : une fuite a lieu quand une étape quelconque de l'ajustement (prétraitement, sélection de variables, choix d'hyperparamètres, ajustement) utilise des données d'évaluation. L'évaluation n'est alors plus indépendante de l'ajustement, et son estimation est biaisée, vers l'optimisme dans les exemples ci-dessous.

**Exemple 1 : standardiser avant de découper.** La moyenne utilisée pour centrer contient les lignes de test. Pour des données i.i.d. de variance 1, l'écart entre moyenne globale et moyenne d'apprentissage vaut \((n_{\mathrm{test}}/n)(\bar{x}_{\mathrm{test}} - \bar{x}_{\mathrm{app}})\), d'écart-type \((n_{\mathrm{test}}/n)\sqrt{1/n_{\mathrm{test}} + 1/n_{\mathrm{app}}}\), soit \(@L4_sd_ecart:4m@\) pour \(n = 1000\) et \(n_{\mathrm{test}} = 200\) (contrôlé par simulation dans le script) : petit. En série temporelle avec tendance ou changement de régime, la moyenne globale contient le futur, et cette formule ne borne plus rien : c'est un lookahead (usage d'une information future).

**Exemple 2 : sélectionner les variables sur tout l'échantillon.** 50 observations (25 de chaque classe), 5000 variables de bruit pur, étiquettes indépendantes des variables : sur une observation nouvelle d'étiquette équiprobable, indépendante de l'apprentissage, tout classifieur a une exactitude de 50 % exactement, puisque sa prédiction est indépendante de l'étiquette. On sélectionne les 100 variables les plus liées à l'étiquette, puis on évalue un plus proche voisin par validation croisée à 5 plis, soit avec la sélection faite une fois sur tout l'échantillon, soit refaite à l'intérieur de chaque pli, sur l'apprentissage du pli seul (avec scikit-learn, c'est ce que fait un `Pipeline` passé à `cross_val_score`).

```python-exec
cv = KFold(5, shuffle=True, random_state=0)
def exactitude_cv(X, y, selection_globale):
    masque_global = SelectKBest(f_classif, k=100).fit(X, y).get_support()      # voit les 50 observations
    bons = 0
    for app, test in cv.split(X):
        masque = masque_global if selection_globale else SelectKBest(f_classif, k=100).fit(X[app], y[app]).get_support()
        pred = voisins(X[app][:, masque], y[app], X[test][:, masque], 1) > 0.5
        bons += np.sum(pred == y[test])
    return bons / len(y)
resultats = []
for graine in range(10):
    rng = np.random.default_rng(100 + graine)
    X = rng.standard_normal((50, 5000))
    y = rng.permutation(np.r_[np.zeros(25, int), np.ones(25, int)])
    resultats.append((exactitude_cv(X, y, True), exactitude_cv(X, y, False)))
resultats = np.array(resultats)
print("exactitude moyenne sur 10 jeux, sélection sur tout l'échantillon / dans chaque pli :", np.round(resultats.mean(axis=0), 3))
print("erreur standard de ces moyennes                                                  :", np.round(resultats.std(axis=0, ddof=1) / np.sqrt(10), 3))
```

Mécanisme : les variables retenues sont, par construction, corrélées à l'étiquette **dans tout l'échantillon**, plis de validation compris ; la validation croisée mesure alors cette corrélation d'échantillon, pas une capacité de prédiction. La seconde valeur est même sous 50 % de plusieurs erreurs standard : avec des effectifs de classes fixés (25 et 25), un pli de validation riche en 1 laisse un apprentissage pauvre en 1, et inversement, ce qui défavorise un classifieur qui reproduit les proportions de son apprentissage. Ce biais pessimiste de la validation croisée sous l'hypothèse nulle est constaté ici, pas démontré dans cette leçon ; l'écart qui compte est celui entre les deux valeurs. Règle : toute étape qui regarde \(\mathbf{y}\), ou la loi des variables (standardisation, imputation, ACP), fait partie du modèle et s'ajuste dans chaque pli, sur l'apprentissage du pli seul.

### Validation croisée

On partage les observations en \(K\) plis ; pour chaque pli, on ajuste sur les \(K - 1\) autres et on évalue sur lui ; on moyenne les \(K\) scores. Cela estime la performance de la **procédure** ajustée sur une fraction \((K - 1)/K\) des données, et suppose des observations indépendantes et interchangeables. Pour choisir un hyperparamètre **et** annoncer une performance, il faut un test séparé, ou une validation croisée imbriquée (le choix refait dans chaque pli extérieur).

### Séries temporelles : l'exemple des étiquettes à 5 jours

Données quotidiennes. \(R_t\) est le rendement du jour \(t\) (aléatoire). L'étiquette de la date \(t\) est le rendement des 5 jours suivants, \(Y_t = R_{t+1} + \dots + R_{t+5}\) (horizon \(h = 5\)). Les variables de la date \(t\) utilisent les rendements \(R_{t-L+1}, \dots, R_t\) (fenêtre \(L = 20\)) : elles sont connues en \(t\), l'étiquette l'est en \(t + 5\).

**Étiquettes qui se chevauchent.** Si les \(R_t\) sont non corrélés, de même variance \(w > 0\), alors pour \(0 \le k \le 5\) : \(\mathrm{Corr}(Y_t, Y_{t+k}) = (5 - k)/5\), et 0 au-delà. Preuve : les deux sommes partagent \(5 - k\) rendements ; la covariance vaut \((5 - k)\,w\) (les termes croisés de rendements distincts sont nuls), chaque variance vaut \(5w\). Donc \(@L4_corr_1:1m@\) pour deux dates voisines, \(@L4_corr_4:1m@\) à 4 jours d'écart.

Une validation croisée à plis **mélangés** pose deux problèmes : elle apprend sur des dates postérieures au test, ce qui n'existe pas en temps réel ; et la date voisine d'une date de test, présente dans l'apprentissage, a une étiquette corrélée à \(@L4_corr_1:1m@\) avec l'étiquette de test, et des variables presque identiques : un modèle qui mémorise est récompensé. Première correction : des plis formés de **blocs de dates contiguës** ; le bloc de test est \(\{t_0, \dots, t_1\}\).

**Purge** : on retire de l'apprentissage toute date dont l'étiquette partage au moins un rendement avec une étiquette de test. Les étiquettes de test utilisent \(R_{t_0+1}, \dots, R_{t_1+5}\) ; celle de \(t\) utilise \(R_{t+1}, \dots, R_{t+5}\) ; elles se rencontrent si et seulement si \(t + 5 \ge t_0 + 1\) et \(t + 1 \le t_1 + 5\), soit \(t_0 - @L4_hm1@ \le t \le t_1 + @L4_hm1@\). La purge retire donc les \(h - 1 = @L4_hm1@\) dates avant le bloc et les @L4_hm1@ dates après. Ce qu'elle retire : le canal **étiquettes partagées**.

**Embargo** : on retire en plus, **après** le bloc, toute date dont les variables utilisent un rendement d'une étiquette de test. La fenêtre de \(t\) rencontre \(\{t_0 + 1, \dots, t_1 + 5\}\) si et seulement si \(t - L + 1 \le t_1 + 5\), soit \(t \le t_1 + L + 4\) : l'embargo retire les \(L\) dates \(t_1 + 5, \dots, t_1 + L + 4\), les précédentes étant déjà purgées. Ce qu'il retire : les variables construites avec la période de test. Avant le bloc, rien de tel : les variables n'utilisent que le passé, ce qui serait connu en temps réel. Purge et embargo de longueur \(L\) garantissent ensemble qu'aucun rendement entrant dans une étiquette de test n'entre dans les données d'apprentissage, ni par les étiquettes, ni par les variables. Avec des rendements indépendants, la purge supprime le seul canal démontré ici ; l'embargo est une marge contre une dépendance au-delà de l'horizon (volatilité en grappes, régimes), et sa longueur est un choix.

Exemple : 1000 dates numérotées de 0 à 999, 5 blocs de 200, test = dates 400 à 599, \(h = 5\), \(L = 20\). Purge : dates @L4_pav1@ à @L4_pav2@ et @L4_pap1@ à @L4_pap2@, soit @L4_n_purge@ dates ; embargo : dates @L4_emb1@ à @L4_emb2@, soit @L4_n_emb@ ; il reste \(@L4_n_app@\) dates d'apprentissage au lieu de \(@L4_n_sans@\).

**Validation progressive** (walk-forward : apprentissage sur le passé seul, comme ton backtest) : pas d'embargo nécessaire, puisqu'aucune date d'apprentissage n'est postérieure au test. La purge reste indispensable : à la date de décision \(t_0\), les étiquettes des dates \(t_0 - @L4_hm1@, \dots, t_0 - 1\) utilisent des rendements postérieurs à \(t_0\), inconnus ; les utiliser est un lookahead.

Expérience : rendements quotidiens indépendants, donc **aucune** prévisibilité réelle ; variables : moyennes des rendements sur 120 et 250 jours ; modèle : 5 plus proches voisins ; mesure : coefficient d'information, corrélation de rang (Spearman) entre prédiction et étiquette hors échantillon. Toute valeur nettement positive est une fuite.

```python-exec
T, h = 2000, 5
def donnees(graine):
    rng = np.random.default_rng(graine)
    r = rng.standard_normal(T + 250 + h)
    cumul = np.concatenate([[0.0], np.cumsum(r)])
    t = np.arange(250, T + 250)
    f1 = (cumul[t + 1] - cumul[t + 1 - 120]) / 120          # moyenne des 120 derniers rendements, t inclus
    f2 = (cumul[t + 1] - cumul[t + 1 - 250]) / 250
    etiq = cumul[t + 1 + h] - cumul[t + 1]                  # somme des rendements t+1 à t+5
    return np.column_stack([f1 / f1.std(), f2 / f2.std()]), etiq
def plis_blocs(purge, embargo, L=250, K=5):
    idx = np.arange(T)
    for t0 in range(0, T, T // K):
        t1 = t0 + T // K - 1
        garde = (idx < t0) | (idx > t1)
        if purge:
            garde &= ~((idx >= t0 - (h - 1)) & (idx <= t1 + h - 1))
        if embargo:
            garde &= ~((idx >= t1 + h) & (idx <= t1 + h + L - 1))
        yield idx[garde], idx[t0:t1 + 1]
def ic(F, etiq, plis):
    pred = np.empty(T)
    for app, test in plis:
        pred[test] = voisins(F[app], etiq[app], F[test], 5)
    return spearmanr(pred, etiq).correlation
res = []
for graine in range(5):
    F, etiq = donnees(graine)
    res.append([ic(F, etiq, KFold(5, shuffle=True, random_state=0).split(F)), ic(F, etiq, plis_blocs(False, False)), ic(F, etiq, plis_blocs(True, True))])
res = np.array(res)
print("IC moyen sur 5 jeux, plis mélangés / blocs / blocs purgés avec embargo :", np.round(res.mean(axis=0), 3))
print("erreur standard de ces moyennes                                       :", np.round(res.std(axis=0, ddof=1) / np.sqrt(5), 3))
```

Validation progressive, réapprentissage tous les 5 jours, avec et sans purge des \(h - 1\) dernières dates :

```python-exec
def progressive(F, etiq, purge, pas=5, debut=500):
    preds, vraies = [], []
    for t0 in range(debut, T, pas):
        fin = t0 - (h - 1) if purge else t0               # dates d'apprentissage : 0 à fin - 1
        preds.append(voisins(F[:fin], etiq[:fin], F[t0:t0 + pas], 5)); vraies.append(etiq[t0:t0 + pas])
    return spearmanr(np.concatenate(preds), np.concatenate(vraies)).correlation
res = np.array([[progressive(*donnees(g), False), progressive(*donnees(g), True)] for g in range(5)])
print("IC moyen sur 5 jeux, sans purge / avec purge :", np.round(res.mean(axis=0), 3))
print("erreur standard de ces moyennes              :", np.round(res.std(axis=0, ddof=1) / np.sqrt(5), 3))
```

Lecture : les plis mélangés font apparaître un signal là où il n'y en a pas ; les blocs en suppriment l'essentiel. Dans ces blocs, les dates de bord sont peu nombreuses (\(2(h - 1)\) purgées par bloc), et l'écart entre blocs avec et sans purge est de l'ordre de l'erreur standard : cette expérience ne le distingue pas. En validation progressive avec réapprentissage fréquent, chaque date de test est une date de bord, et l'absence de purge produit un coefficient d'information positif ; la purge le ramène vers 0.

### Pièges

1. Mélanger les plis d'une série temporelle à étiquettes chevauchantes gonfle la performance, même sans aucune variable du futur.
2. La purge dépend de l'horizon de l'étiquette (\(h - 1\) dates de chaque côté), l'embargo de la fenêtre des variables, et seulement après le bloc.
3. Toute étape qui regarde les données (standardisation, sélection, choix de \(k\) ou de \(\lambda\)) s'ajuste dans le pli.
4. Erreur de validation du meilleur de \(M\) modèles : estimation optimiste ; garder un test intact.

[[controle:selection]]

[[controle:purge]]

## Réflexes en 30 secondes

| Indice dans l'énoncé | Réflexe |
|---|---|
| gradient de la log-loss | \(\tfrac1n \mathbf{X}^\top (\boldsymbol{\pi} - \mathbf{y})\) ; hessienne \(\tfrac1n \mathbf{X}^\top \mathbf{W}\mathbf{X}\), poids \(\pi_i(1 - \pi_i)\) |
| convexité de la logistique | hessienne SDP ; stricte si \(\mathbf{X}\) de rang \(p\) ; pas de forme fermée |
| \(e^{\beta_j}\) | facteur sur la cote, pas sur la probabilité ; \(\partial \pi / \partial x_j = \beta_j \pi(1 - \pi)\) |
| coefficients qui explosent | séparation : pas de maximum de vraisemblance ; pénalité L2 (défaut de scikit-learn) |
| avec constante, au minimum | somme des \(\pi_i\) = somme des \(y_i\) |
| AUC | proportion de paires (positif, négatif) bien ordonnées, ex aequo \(\tfrac12\) ; invariante par transformation croissante |
| classes déséquilibrées | exactitude trompeuse ; précision \(= q\,\mathrm{TVP}/(q\,\mathrm{TVP} + (1 - q)\mathrm{TFP})\) ; courbe précision-rappel |
| Brier, log-loss | règles propres ; Brier espéré \(= (r - q)^2 + q(1 - q)\) |
| seuil à coûts asymétriques | \(c_{\mathrm{FP}}/(c_{\mathrm{FP}} + c_{\mathrm{FN}})\) sur une probabilité calibrée |
| plus proches voisins | standardiser (statistiques d'apprentissage) ; \(k\) petit : variance \(\sigma^2/k\) grande |
| arbres et échelle | invariants par transformation strictement croissante de chaque variable |
| grande dimension | \(D^2\) : espérance \(p\,m\), variance \(p\,v\), rapport en \(1/\sqrt{p}\) ; Tchebychev ; exige des coordonnées indépendantes |
| fuite | toute étape qui voit le test ; sélection sur tout l'échantillon : performance illusoire |
| étiquettes à \(h\) jours | corrélation \((h - k)/h\) ; purge \(h - 1\) dates de chaque côté ; embargo \(L\) dates après |

## À savoir refaire sans notes

- Les deux écritures de la log-loss, le gradient, la hessienne, sa positivité et la convexité par Taylor.
- La somme des probabilités ajustées avec constante, le rapport de cotes, la non-existence du minimum sous séparation.
- F1 en fonction de VP, FP, FN ; la précision par la formule de Bayes ; l'AUC comme proportion de paires (preuve par l'escalier).
- Brier et log-loss comme règles propres ; le seuil par les coûts.
- La concentration de \(D^2\) avec Markov et Tchebychev, et le contre-exemple des coordonnées égales.
- Purge et embargo sur l'exemple à 5 jours : quelles dates, et pourquoi.
""",
    "controles": {
        "convexite": {
            "question": r"Régression logistique sans pénalité, \(\mathbf{X}\) de rang \(p\). Quelle affirmation sur la log-loss moyenne \(\ell\) est vraie ?",
            "options": [
                {"id": "a", "texte": r"\(\ell\) est strictement convexe, son gradient vaut \(\tfrac1n \mathbf{X}^\top (\boldsymbol{\pi} - \mathbf{y})\), et son minimiseur, s'il existe, est unique mais n'a pas de forme fermée.",
                 "explication": r"Oui : hessienne \(\tfrac1n \mathbf{X}^\top \mathbf{W}\mathbf{X}\) définie positive au rang \(p\) ; l'équation du premier ordre n'est pas linéaire en \(\mathbf{b}\) ; et l'existence peut échouer (séparation)."},
                {"id": "b", "texte": r"\(\ell\) a toujours un unique minimiseur, puisqu'elle est strictement convexe.",
                 "explication": r"La stricte convexité donne au plus un minimiseur, pas son existence. Sous séparation parfaite, \(\ell\) tend vers 0 sans l'atteindre : aucun minimiseur (partie 1, énoncé 6)."},
                {"id": "c", "texte": r"\(\ell\) n'est pas convexe, car la fonction \(\sigma\) ne l'est pas : on peut tomber dans un minimum local.",
                 "explication": r"\(\sigma\) n'est ni convexe ni concave, mais \(\ell\) est une somme de \(\ln(1 + e^{t}) - y t\), convexes en \(t\) (dérivée seconde \(\sigma(1 - \sigma) > 0\)), composées avec \(t = \mathbf{x}_i^\top \mathbf{b}\) linéaire : convexe. Pas de minimum local non global."},
                {"id": "d", "texte": r"Le minimiseur vaut \((\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}\), comme pour les moindres carrés.",
                 "explication": r"C'est la solution des moindres carrés sur les étiquettes 0 ou 1 (modèle de probabilité linéaire), pas le minimiseur de la log-loss, qui annule \(\mathbf{X}^\top (\boldsymbol{\pi}(\mathbf{b}) - \mathbf{y})\) avec \(\boldsymbol{\pi}\) non linéaire en \(\mathbf{b}\)."},
            ],
        },
        "auc": {
            "question": r"Un modèle a une AUC de 0,8 sur un échantillon de test. On remplace chaque probabilité prédite \(r\) par \(r^2\). Que deviennent l'AUC et le score de Brier ?",
            "options": [
                {"id": "a", "texte": r"L'AUC reste 0,8 ; le score de Brier change en général.",
                 "explication": r"Oui : \(r \mapsto r^2\) est strictement croissante sur \([0, 1]\), donc l'ordre des scores, et la proportion de paires bien ordonnées, ne changent pas. Le Brier dépend des valeurs : dans l'expérience de calibration de la leçon, il augmente nettement."},
                {"id": "b", "texte": r"L'AUC baisse, car les probabilités ne sont plus calibrées.",
                 "explication": r"L'AUC ne voit que l'ordre des scores, pas leurs valeurs : elle est aveugle à la calibration. C'est pour cela qu'on la complète par le Brier, la log-loss ou un diagramme de fiabilité."},
                {"id": "c", "texte": r"L'AUC et le Brier restent inchangés, car le classement est le même.",
                 "explication": r"Le Brier est la moyenne de \((r_i - y_i)^2\) : il dépend des valeurs prédites, pas seulement de leur ordre. Une transformation croissante qui fausse les probabilités le dégrade en général."},
                {"id": "d", "texte": r"L'AUC devient \(0{,}8^2\), car chaque score est élevé au carré.",
                 "explication": r"L'AUC est une proportion de paires bien ordonnées, pas une fonction des valeurs des scores. Élever les scores au carré ne change aucune comparaison entre deux scores positifs."},
            ],
        },
        "echelle": {
            "question": r"On multiplie par 1000 l'une des variables (par exemple on passe de milliers d'euros à euros), sans rien changer d'autre, et on réajuste. Quelles prédictions sont à coup sûr inchangées, sur l'apprentissage comme sur des points nouveaux ?",
            "options": [
                {"id": "a", "texte": r"Celles d'un arbre de décision.",
                 "explication": r"Oui : multiplier par 1000 est affine croissant ; l'ordre des valeurs, les partitions possibles et les gains sont les mêmes, et les seuils au milieu de deux valeurs sont transportés exactement. L'expérience de la partie 3 le constate."},
                {"id": "b", "texte": r"Celles des 15 plus proches voisins avec la distance euclidienne.",
                 "explication": r"La contribution de la variable à la distance est multipliée par \(1000^2\) : elle domine toutes les autres, et les voisins changent (exemple \(Q\), \(A\), \(B\) de la partie 3)."},
                {"id": "c", "texte": r"Celles d'une régression logistique avec la pénalité L2 par défaut de scikit-learn.",
                 "explication": r"Sans pénalité, le coefficient serait divisé par 1000 et les prédictions inchangées. Avec la pénalité \(\tfrac12 \lVert \mathbf{w} \rVert^2\), le coefficient plus petit est moins pénalisé : la solution, et les prédictions, changent."},
                {"id": "d", "texte": r"Aucune des trois : tout modèle dépend de l'unité des variables.",
                 "explication": r"Les arbres ne dépendent que de l'ordre des valeurs de chaque variable, et la régression logistique sans pénalité est elle aussi invariante (changement \(b_j \to b_j/1000\))."},
            ],
        },
        "selection": {
            "question": r"On dispose de 60 observations et 2000 variables. On garde les 20 variables les plus corrélées à \(\mathbf{y}\) sur les 60 observations, puis on estime l'erreur d'une régression par validation croisée à 5 plis sur ces 20 variables. Quel est le défaut ?",
            "options": [
                {"id": "a", "texte": r"La sélection a vu les observations de validation : l'erreur estimée est optimiste ; il faut refaire la sélection dans chaque pli, sur l'apprentissage du pli.",
                 "explication": r"Oui : les variables retenues sont corrélées à \(\mathbf{y}\) dans tout l'échantillon, plis de validation compris. L'expérience de la partie 4, avec des variables de bruit pur, donne une exactitude très supérieure au hasard avec cette procédure, et proche du hasard avec la sélection dans le pli."},
                {"id": "b", "texte": r"Aucun : la sélection n'utilise que les corrélations, pas le modèle de régression.",
                 "explication": r"Toute étape qui regarde \(\mathbf{y}\) fait partie de l'ajustement. Que ce soit une corrélation ou une régression ne change rien : l'information des plis de validation a servi à choisir les variables."},
                {"id": "c", "texte": r"5 plis sont trop peu ; avec 60 plis (une observation par pli), l'estimation serait sans biais.",
                 "explication": r"Le nombre de plis ne corrige pas la fuite : avec une observation par pli, la sélection a toujours été faite en voyant cette observation. Le défaut est l'ordre des étapes, pas le nombre de plis."},
                {"id": "d", "texte": r"Il fallait standardiser les variables avant la sélection, sur tout l'échantillon.",
                 "explication": r"La corrélation ne dépend pas de l'échelle, donc standardiser ne change pas la sélection ; et standardiser sur tout l'échantillon serait une seconde fuite, plus petite."},
            ],
        },
        "purge": {
            "question": r"Étiquettes : rendement des 5 jours suivants ; variables : rendements des 20 derniers jours ; validation croisée par blocs de dates contiguës. Que retirent la purge et l'embargo de l'ensemble d'apprentissage ?",
            "options": [
                {"id": "a", "texte": r"La purge : les @L4_hm1@ dates de chaque côté du bloc de test, dont l'étiquette partage un rendement avec une étiquette de test. L'embargo : les 20 dates suivantes après le bloc, dont les variables utilisent des rendements de la période de test.",
                 "explication": r"Oui : chevauchement si et seulement si \(t_0 - @L4_hm1@ \le t \le t_1 + @L4_hm1@\) pour la purge ; fenêtre des variables qui rencontre \(\{t_0 + 1, \dots, t_1 + 5\}\) si et seulement si \(t \le t_1 + @L4_Lp4@\) pour l'embargo (exemple de la leçon : dates @L4_emb1@ à @L4_emb2@)."},
                {"id": "b", "texte": r"La purge : les 5 dates de chaque côté du bloc. L'embargo : les 20 dates de chaque côté.",
                 "explication": r"Deux erreurs. La date \(t_0 - 5\) a pour étiquette \(R_{t_0 - 4} + \dots + R_{t_0}\), qui ne partage rien avec les étiquettes de test (elles commencent à \(R_{t_0 + 1}\)) : la purge retire \(h - 1 = @L4_hm1@\) dates. Et avant le bloc, les variables n'utilisent que le passé : pas d'embargo de ce côté."},
                {"id": "c", "texte": r"La purge : les dates postérieures au bloc de test. L'embargo : les dates antérieures.",
                 "explication": r"La purge agit des deux côtés (les étiquettes des @L4_hm1@ dates précédant le bloc contiennent des rendements de la période de test). L'embargo ne concerne que l'après-bloc."},
                {"id": "d", "texte": r"Rien, dès que les plis sont des blocs contigus : seul le mélange des dates crée une fuite.",
                 "explication": r"Les blocs suppriment l'essentiel de la fuite, mais pas aux bords : la date \(t_0 - 1\) a une étiquette corrélée à \(@L4_corr_1:1m@\) avec celle de \(t_0\). En validation progressive à réapprentissage fréquent, toutes les dates de test sont des bords (expérience de la partie 4)."},
            ],
        },
    },
    "cellules_initiales": [
        "import numpy as np\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import roc_auc_score, brier_score_loss, log_loss\nnp.set_printoptions(precision=4, suppress=True)",
        "# Essaie ici les expériences de la leçon, en les modifiant.\nrng = np.random.default_rng(0)\npi = rng.uniform(0, 1, 10000)\nY = (rng.uniform(size=10000) < pi).astype(int)\nfor r in [pi, pi**2, np.sqrt(pi)]:\n    print(roc_auc_score(Y, r), brier_score_loss(Y, r))",
    ],
    "verif": {"script": "verif/seance_ml_3.py"},
}
