"""Leçon de la séance de ML 2 : arbres de décision, bagging, forêts aléatoires, boosting de gradient et XGBoost.

Format : voir outils/lecons.py (élément de type "lecon", contenu en Markdown réduit, points de contrôle
[[controle:IDENT]] sur une ligne seule, option "a" correcte, une explication par option).
Les blocs python-exec sont exécutés à la construction par le noyau de l'environnement .venv-hackerrank
(scikit-learn 1.1.1, numpy 1.21.6, sans XGBoost) ; leur sortie réelle est insérée. Graines fixées.
Les autres nombres sont des jetons @CLE@ remplacés par les valeurs de verif/seance_ml_2.out.json.
"""

LECON_ARBRES = {
    "id": "L-ARBRES-1", "type": "lecon", "section": "lecon",
    "theme": "Leçon arbres : forêts aléatoires, boosting et XGBoost",
    "difficulte": 4, "temps_cible_s": 2700, "dossier": "coding/lecons",
    "contenu": r"""## Leçon : arbres, forêts aléatoires, boosting et XGBoost

Durée visée : 45 minutes. La leçon répond notamment à deux questions du test de positionnement, la variance d'une moyenne d'arbres corrélés (DIAG-F4) et le poids optimal d'une feuille XGBoost (DIAG-A1), et à la question « comparer forêt aléatoire et XGBoost ». Elle reconstruit tout depuis un arbre seul, en sept parties, chacune dans l'ordre : le problème, l'intuition, l'énoncé, la preuve, un exemple chiffré, les cas limites, les pièges.

Les expériences sont des blocs de code exécutés à la construction de cette page, avec scikit-learn 1.1.1 et numpy 1.21.6 (environnement des notebooks de la plateforme) : la sortie affichée sous chaque bloc est la vraie sortie. Les autres nombres sont calculés, et contrôlés par simulation, par le script `verif/seance_ml_2.py`.

**Conventions de notation**, valables pour toute la leçon :

- une majuscule désigne une variable aléatoire, une minuscule une valeur ou un paramètre fixe, le gras un vecteur ;
- les données d'entraînement comptent \(n\) observations ; l'observation \(i\) est formée d'un vecteur \(\mathbf{x}_i\) de \(p\) variables réelles, dont la \(j\)-ième vaut \(x_{ij}\), et d'une cible \(y_i\) : un réel en régression, une étiquette 0 ou 1 en classification binaire. Dans les parties 1, 4, 5 et 6 ces données sont observées, donc fixes : minuscules ;
- dans les parties 2, 3 et 7, on étudie ce qui varie quand l'échantillon et les tirages de l'algorithme changent : les prédictions deviennent des variables aléatoires, en majuscules ;
- \(\lvert S \rvert\) est le nombre d'éléments d'un ensemble \(S\), et \(I_A\) l'indicatrice d'un événement \(A\) (1 si \(A\) se produit, 0 sinon) ;
- exceptions assumées, parce que ce sont les notations universelles et celles du test de positionnement : \(B\), le nombre d'arbres, est un entier fixe ; \(G\) et \(H\), des sommes de dérivées, sont des nombres fixes.

## Partie 1 : l'arbre de décision

### Le problème

On veut prédire \(y\) à partir de \(\mathbf{x}\) sans imposer de forme à la relation : ni linéarité, ni liste d'interactions (une interaction est un effet d'une variable qui dépend de la valeur d'une autre). Un arbre découpe l'espace des \(\mathbf{x}\) en régions rectangulaires et prédit une constante par région.

### L'intuition

On pose des questions « \(x_j \le s\) ? », une à la fois. Chaque question est choisie pour que les deux groupes qu'elle crée soient le plus homogènes possible en \(y\), puis on recommence dans chaque groupe. La procédure est **gloutonne** : chaque question est la meilleure au moment où on la pose, sans garantie que l'arbre entier soit le meilleur arbre possible.

### Les définitions

- Un **nœud** est un ensemble \(S\) d'indices d'observations. La racine contient les \(n\) observations ; une **feuille** est un nœud que l'on ne coupe plus.
- Une **coupure** de \(S\) est un couple \((j, s)\) formé d'une variable \(j\) et d'un seuil réel \(s\). Elle partage \(S\) en \(L = \{i \in S : x_{ij} \le s\}\) et \(R = \{i \in S : x_{ij} > s\}\), tous deux non vides. Seuls comptent les seuils qui donnent des partages différents : on prend \(s\) strictement entre deux valeurs consécutives de la variable \(j\) dans \(S\) (scikit-learn prend le milieu). Une variable qui prend \(k\) valeurs distinctes dans \(S\) offre donc \(k - 1\) coupures candidates.
- **Prédiction d'une feuille** \(S\) : en régression, la moyenne \(\bar y_S = \frac{1}{\lvert S \rvert}\sum_{i \in S} y_i\) ; en classification, la proportion \(\hat p_c(S)\) des observations de \(S\) qui sont de classe \(c\), pour chaque classe \(c\), et la classe majoritaire si l'on veut une étiquette.
- **Impureté** \(\mathrm{imp}(S)\), qui mesure l'hétérogénéité de \(y\) dans \(S\) :
- régression, la variance : \(\mathrm{imp}(S) = \frac{1}{\lvert S \rvert}\sum_{i \in S} (y_i - \bar y_S)^2\) ;
- classification, l'indice de Gini : \(\mathrm{imp}(S) = 1 - \sum_c \hat p_c(S)^2\), la somme portant sur les classes ; avec deux classes de proportions \(\hat p\) et \(1 - \hat p\), il vaut \(2\hat p\,(1 - \hat p)\) ;
- classification, l'entropie : \(\mathrm{imp}(S) = -\sum_c \hat p_c(S) \ln \hat p_c(S)\), avec la convention \(0 \ln 0 = 0\).
- **Gain** de la coupure : \(\mathrm{gain} = \mathrm{imp}(S) - \frac{\lvert L \rvert}{\lvert S \rvert}\,\mathrm{imp}(L) - \frac{\lvert R \rvert}{\lvert S \rvert}\,\mathrm{imp}(R)\), la baisse d'impureté, chaque enfant pesant selon sa taille. L'algorithme choisit, parmi toutes les variables et tous les seuils candidats, la coupure de gain maximal ; il recommence dans \(L\) et dans \(R\), jusqu'à un critère d'arrêt : profondeur maximale (nombre de questions entre la racine et une feuille), taille minimale d'une feuille, gain minimal.

### L'énoncé

1. (R1) Pour tout réel \(c\) : \(\sum_{i \in S} (y_i - c)^2 = \sum_{i \in S} (y_i - \bar y_S)^2 + \lvert S \rvert\,(\bar y_S - c)^2\). La moyenne de la feuille est donc la constante qui minimise la somme des carrés des erreurs sur la feuille.
2. (R2) Pour la variance : \(\mathrm{gain} = \frac{\lvert L \rvert}{\lvert S \rvert}(\bar y_L - \bar y_S)^2 + \frac{\lvert R \rvert}{\lvert S \rvert}(\bar y_R - \bar y_S)^2 \ge 0\), nul si et seulement si \(\bar y_L = \bar y_R\).
3. (R3) L'indice de Gini est la somme, sur les classes, des variances des indicatrices de classe : \(1 - \sum_c \hat p_c^2 = \sum_c \hat p_c\,(1 - \hat p_c)\). Son gain est la somme des gains de variance de ces indicatrices : il est positif ou nul.
4. (R4) Le gain d'entropie est positif ou nul.
5. (R5) Si une variable prend des valeurs deux à deux distinctes sur tout l'échantillon d'entraînement, toute feuille où \(y\) n'est pas constant admet une coupure de gain de variance strictement positif. Un arbre de régression que l'on fait croître tant qu'une telle coupure existe s'arrête avec des feuilles où \(y\) est constant : son erreur d'entraînement est nulle.
6. (R6) Si l'on remplace une variable \(x_j\) par \(\phi(x_j)\), où \(\phi\) est une fonction strictement croissante (logarithme d'une variable positive, cube, rang), chaque nœud a les mêmes partages candidats avec les mêmes gains. Si les égalités de gain sont départagées de la même façon, l'arbre réalise la même partition des observations d'entraînement et leur donne les mêmes prédictions. Sur un **nouveau** point, la prédiction peut changer, mais seulement si sa valeur tombe strictement entre deux valeurs d'entraînement consécutives qui encadrent un seuil. Si \(\phi\) est affine croissante (standardisation), rien ne change, même sur un nouveau point.

### La preuve

1. (R1) On écrit \(y_i - c = (y_i - \bar y_S) + (\bar y_S - c)\) et on développe le carré. Le double produit vaut \(2\,(\bar y_S - c)\sum_{i \in S}(y_i - \bar y_S) = 0\), car la somme des écarts à la moyenne est nulle par définition de \(\bar y_S\). Le second terme est minimal, et nul, pour \(c = \bar y_S\).
2. (R2) On coupe en deux la somme \(\lvert S \rvert\,\mathrm{imp}(S) = \sum_{i \in S}(y_i - \bar y_S)^2\), sur \(L\) et sur \(R\), et on applique (R1) à chaque morceau avec \(c = \bar y_S\) : \(\sum_{i \in L}(y_i - \bar y_S)^2 = \lvert L \rvert\,\mathrm{imp}(L) + \lvert L \rvert\,(\bar y_L - \bar y_S)^2\), de même pour \(R\). On additionne et on divise par \(\lvert S \rvert\). Cas d'égalité : \(\bar y_S\) est la moyenne de \(\bar y_L\) et \(\bar y_R\) pondérée par \(\lvert L \rvert\) et \(\lvert R \rvert\), donc les deux carrés sont nuls si et seulement si \(\bar y_L = \bar y_S = \bar y_R\).
3. (R3) L'indicatrice de la classe \(c\) vaut 1 sur une proportion \(\hat p_c\) des observations de \(S\) et 0 sur les autres. Sa moyenne est \(\hat p_c\) et sa variance \(\hat p_c\,(1 - \hat p_c)^2 + (1 - \hat p_c)\,\hat p_c^2 = \hat p_c\,(1 - \hat p_c)\). En sommant sur les classes, avec \(\sum_c \hat p_c = 1\) : \(\sum_c \hat p_c\,(1 - \hat p_c) = 1 - \sum_c \hat p_c^2\). Le gain de Gini est donc la somme sur les classes des gains de variance des indicatrices, chacun positif ou nul par (R2).
4. (R4) La fonction \(\varphi(u) = -u \ln u\), prolongée par \(\varphi(0) = 0\), a pour dérivée seconde \(-1/u < 0\) sur \(]0, 1]\) : elle y est concave, et, étant continue en 0, elle l'est sur \([0, 1]\) (l'inégalité de concavité passe à la limite). Les proportions du parent sont la moyenne pondérée de celles des enfants : \(\hat p_c(S) = \frac{\lvert L \rvert}{\lvert S \rvert}\hat p_c(L) + \frac{\lvert R \rvert}{\lvert S \rvert}\hat p_c(R)\). La définition de la concavité donne, pour chaque classe, \(\varphi(\hat p_c(S)) \ge \frac{\lvert L \rvert}{\lvert S \rvert}\varphi(\hat p_c(L)) + \frac{\lvert R \rvert}{\lvert S \rvert}\varphi(\hat p_c(R))\) ; on somme sur les classes.
5. (R5) Dans la feuille \(S\), on range les observations selon la variable aux valeurs distinctes : \(y_{(1)}, y_{(2)}, \dots, y_{(\lvert S \rvert)}\). Les coupures qui isolent les \(k\) premières, pour \(k = 1, \dots, \lvert S \rvert - 1\), sont toutes réalisables, car les valeurs sont distinctes. Si toutes avaient un gain nul, (R2) donnerait pour chaque \(k\) une moyenne des \(k\) premières égale à \(\bar y_S\) : pour \(k = 1\), \(y_{(1)} = \bar y_S\) ; pour \(k = 2\), \(y_{(2)} = \bar y_S\) ; et ainsi de suite jusqu'à \(k = \lvert S \rvert - 1\). La dernière valeur vaudrait aussi \(\bar y_S\), puisque la moyenne de toutes vaut \(\bar y_S\) : \(y\) serait constant sur \(S\), contradiction. Comme chaque coupure crée deux nœuds non vides, il y a au plus \(n - 1\) coupures : la croissance s'arrête, et elle ne s'arrête que sur des feuilles de \(y\) constant, où la moyenne prédit exactement chaque observation.
6. (R6) Le partage \(\{i \in S : x_{ij} \le s\}\) ne dépend que de l'ordre des valeurs \(x_{ij}\) : les partages candidats sont « les \(k\) plus petites valeurs d'un côté, les autres de l'autre ». Une fonction strictement croissante conserve l'ordre, donc la liste des partages ; et le gain d'un partage ne dépend que des \(y_i\) de chaque côté. Les mêmes partages ont les mêmes gains, la même coupure est choisie à la racine, puis, de proche en proche, à chaque nœud. Pour un nouveau point : le seuil vaut \((v_k + v_{k+1})/2\) sur l'échelle d'origine et \((\phi(v_k) + \phi(v_{k+1}))/2\) sur l'échelle transformée, où \(v_k < v_{k+1}\) sont les deux valeurs qui l'encadrent ; une valeur strictement entre \(v_k\) et \(v_{k+1}\) peut tomber à gauche de l'un et à droite de l'autre. Si \(\phi\) est affine, \(\phi\) du milieu est le milieu des images, et les deux seuils se correspondent exactement.

### Exemple chiffré

Huit observations, une seule variable, \(x_i = i\) pour \(i = 1, \dots, 8\), et les étiquettes \(y = (0, 0, 1, 0, 1, 1, 1, 1)\). Cet exemple jouet resservira pour XGBoost en partie 5.

```python-exec
import numpy as np

x_jouet = np.arange(1, 9)
y_jouet = np.array([0, 0, 1, 0, 1, 1, 1, 1])

def gini(e):
    p = e.mean()
    return 2 * p * (1 - p)

def entropie(e):
    p = e.mean()
    return 0.0 if p in (0.0, 1.0) else -(p * np.log(p) + (1 - p) * np.log(1 - p))

def gain(imp, e, gauche):
    g, d = e[gauche], e[~gauche]
    return imp(e) - len(g) / len(e) * imp(g) - len(d) / len(e) * imp(d)

for s in np.arange(1.5, 8, 1.0):
    print(f"seuil {s}: gain de Gini {gain(gini, y_jouet, x_jouet <= s):.4f}, gain d'entropie {gain(entropie, y_jouet, x_jouet <= s):.4f}")
```

Les deux critères choisissent ici le même seuil, \(s = @A_seuil_gini:1m@\). Calcul à la main du gain de Gini :

1. Parent : 5 étiquettes 1 sur 8, Gini \(2 \times \tfrac58 \times \tfrac38 = \tfrac{@A_gini_parent_num@}{@A_gini_parent_den@}\).
2. Gauche : @A_n_gauche@ observations dont @A_pos_gauche@ de classe 1, proportion \(\tfrac{@A_prop_gauche_num@}{@A_prop_gauche_den@}\), Gini \(\tfrac{@A_gini_gauche_num@}{@A_gini_gauche_den@}\). Droite : 4 observations toutes de classe 1, Gini @A_gini_droite@.
3. Gain : \(\tfrac{@A_gini_parent_num@}{@A_gini_parent_den@} - \tfrac48 \times \tfrac{@A_gini_gauche_num@}{@A_gini_gauche_den@} - \tfrac48 \times @A_gini_droite@ = \tfrac{@A_gain_gini_num@}{@A_gain_gini_den@} \approx @A_gain_gini:3m@\).

Profondeur et sur-apprentissage, puis invariance, sur une régression à deux variables bruitées :

```python-exec
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(0)
def tirer(n):  # deux variables uniformes entre 0,1 et 3 ; cible = signal + bruit gaussien d'écart-type 0,6
    X = rng.uniform(0.1, 3.0, size=(n, 2))
    return X, np.sin(2 * X[:, 0]) + X[:, 1] + rng.normal(0, 0.6, n)

X, y = tirer(200)       # entraînement
Xn, yn = tirer(5000)    # nouveaux points
for prof in [2, 4, 8, None]:
    arbre = DecisionTreeRegressor(max_depth=prof, random_state=0).fit(X, y)
    print(f"profondeur maximale {prof}: {arbre.get_n_leaves()} feuilles, erreur quadratique "
          f"entraînement {np.mean((arbre.predict(X) - y) ** 2):.4f}, nouveaux points {np.mean((arbre.predict(Xn) - yn) ** 2):.4f}")
```

Lecture : sans limite de profondeur, l'erreur d'entraînement est nulle, comme l'annonce (R5) (des valeurs tirées d'une loi continue sont distinctes avec probabilité 1), et l'erreur sur les nouveaux points est la plus élevée des quatre arbres. Le bruit a pour variance \(0{,}6^2\), plancher qu'aucune prédiction ne peut battre en moyenne sur de nouveaux points : le meilleur des quatre arbres, de profondeur 4, en reste au-dessus, et l'arbre sans limite en est le plus loin.

```python-exec
phi = lambda Z: np.column_stack([np.exp(Z[:, 0]), Z[:, 1] ** 3])   # deux transformations strictement croissantes
a = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)
b = DecisionTreeRegressor(max_depth=4, random_state=0).fit(phi(X), y)
print("mêmes prédictions sur l'entraînement :", np.array_equal(a.predict(X), b.predict(phi(X))))
print("part des nouveaux points prédits différemment :", np.mean(a.predict(Xn) != b.predict(phi(Xn))))
m, e = X.mean(axis=0), X.std(axis=0)
c = DecisionTreeRegressor(max_depth=4, random_state=0).fit((X - m) / e, y)
print("standardisation, part des nouveaux points prédits différemment :", np.mean(a.predict(Xn) != c.predict((Xn - m) / e)))
```

### Cas limites

- Feuille d'une seule observation, ou de \(y\) constant : impureté nulle, aucun gain possible.
- Variable binaire : une seule coupure candidate ; variable continue aux valeurs distinctes : \(\lvert S \rvert - 1\). Ce déséquilibre est la source du biais de la partie 7.
- Transformation strictement décroissante : mêmes partages, gauche et droite échangées, même partition.
- Mettre les variables à l'échelle ne sert à rien pour un arbre (R6), contrairement aux méthodes fondées sur des distances, comme les \(k\) plus proches voisins, dont les voisins changent quand on change l'unité d'une variable.

### Pièges

1. Une erreur d'entraînement nulle ne dit rien de l'erreur de test : (R5) la garantit pour un arbre sans limite, bruit compris.
2. Oublier la pondération par \(\lvert L \rvert / \lvert S \rvert\) : une petite feuille pure compterait autant qu'une grande.
3. Gini à deux classes : \(2\hat p\,(1 - \hat p)\), et non \(\hat p\,(1 - \hat p)\). Le facteur 2 ne change pas la coupure choisie, mais il change la valeur du gain.
4. Une coupure n'utilise qu'une variable : les frontières de l'arbre sont faites de morceaux parallèles aux axes. Une frontière oblique, comme \(x_1 + x_2 = 1\), n'est jamais reproduite exactement, seulement approchée par un escalier.

[[controle:profondeur]]

## Partie 2 : le bagging

### Le problème

Un arbre profond suit les données de près : changer l'échantillon d'entraînement change fortement ses prédictions (l'expérience de la partie 3 mesure cette variance : elle dépasse de loin celle de la forêt). On voudrait moyenner de nombreux arbres entraînés sur des échantillons différents, mais on ne dispose que d'un échantillon.

### L'intuition

On fabrique des échantillons en tirant au hasard, avec remise, dans l'échantillon disponible : c'est le **bootstrap**. Les arbres obtenus ne sont pas indépendants, puisqu'ils viennent des mêmes données : la moyenne réduit la variance, mais moins que \(1/B\). Une formule dit exactement de combien.

### Les définitions

- **Échantillon bootstrap** : \(n\) tirages indépendants, chacun uniforme parmi les \(n\) observations, avec remise. \(N_i\) est le nombre de fois où l'observation \(i\) est tirée.
- **Bagging** (bootstrap aggregating) : on tire \(B\) échantillons bootstrap indépendants, on entraîne un arbre sur chacun, et l'on prédit par la moyenne des \(B\) prédictions (en classification, par la moyenne des probabilités ou par un vote).
- Une observation est **hors sac** (OOB, out of bag) pour l'arbre \(b\) si elle est absente de son échantillon bootstrap.
- En un point \(\mathbf{x}\) fixé, \(T_b\) est la prédiction de l'arbre \(b\) : une variable aléatoire, à travers deux sources de hasard, l'échantillon d'entraînement et les tirages de l'algorithme. \(\bar T_B = \frac1B\sum_{b=1}^B T_b\) est la prédiction de l'ensemble.

### L'énoncé

1. (B1) \(\mathbb{P}(N_i = 0) = (1 - 1/n)^n\). Cette probabilité est au plus \(1/e\) pour tout \(n \ge 1\) et tend vers \(1/e\) quand \(n \to \infty\). Le nombre moyen d'observations distinctes d'un échantillon bootstrap est \(n\,\big(1 - (1 - 1/n)^n\big)\).
2. (B2) Soient \(T_1, \dots, T_B\) des variables aléatoires réelles de même variance finie \(\sigma^2 > 0\) et de même corrélation \(\rho\) deux à deux. Alors \(\mathrm{Var}(\bar T_B) = \rho\,\sigma^2 + (1 - \rho)\,\sigma^2/B\). Si de plus elles ont la même espérance, \(\mathbb{E}[\bar T_B] = \mathbb{E}[T_1]\) : la moyenne a le **même biais** qu'un arbre seul.
3. (B3) Les arbres d'un bagging ou d'une forêt vérifient ces hypothèses : ils sont produits par la même procédure, à partir des mêmes données, avec des tirages indépendants et de même loi ; échanger les tirages de deux arbres ne change pas la loi de l'ensemble, donc tous les \(T_b\) ont la même loi, et tous les couples \((T_b, T_{b'})\) avec \(b \ne b'\) la même loi jointe, d'où une variance et une corrélation communes.

### La preuve

1. (B1) Un tirage manque l'observation \(i\) avec probabilité \(1 - 1/n\) ; les \(n\) tirages sont indépendants : \((1 - 1/n)^n\).
2. Majoration. La fonction \(u \mapsto \ln(1 - u) + u\) vaut 0 en 0 et a pour dérivée \(-\tfrac{1}{1-u} + 1 = -\tfrac{u}{1-u}\), strictement négative sur \(]0, 1[\) ; donc \(\ln(1 - u) < -u\) pour \(0 < u < 1\). Avec \(u = 1/n\) et \(n \ge 2\) : \(n \ln(1 - 1/n) < -1\), donc \((1 - 1/n)^n < e^{-1}\) ; pour \(n = 1\), la probabilité est nulle. La majoration est donc stricte pour tout \(n\).
3. Limite. \(\ln(1 - u)/u \to -1\) quand \(u \to 0\) (c'est la dérivée de \(\ln\) en 1, au signe près), donc \(n \ln(1 - 1/n) \to -1\), et l'exponentielle est continue.
4. Nombre d'observations distinctes : c'est \(\sum_{i=1}^n I_{\{N_i \ge 1\}}\), d'espérance \(n\,\big(1 - (1 - 1/n)^n\big)\) par linéarité de l'espérance, qui ne demande aucune indépendance entre les termes (leçon de probabilités de la séance de mathématiques 1, outil 1).
5. (B2) Les produits \((T_b - \mathbb{E}T_b)(T_{b'} - \mathbb{E}T_{b'})\) sont intégrables, car \(\lvert uv \rvert \le (u^2 + v^2)/2\) et les variances sont finies. On développe le carré : \(\mathrm{Var}\big(\sum_b T_b\big) = \mathbb{E}\Big[\big(\sum_b (T_b - \mathbb{E}T_b)\big)^2\Big] = \sum_b \sum_{b'} \mathrm{Cov}(T_b, T_{b'})\), par linéarité de l'espérance.
6. La double somme contient \(B\) termes diagonaux égaux à \(\sigma^2\) et \(B(B-1)\) couples ordonnés \(b \ne b'\), chacun égal à \(\rho\,\sigma^2\) (définition de la corrélation). Diviser la somme par \(B\) divise la variance par \(B^2\) : \(\mathrm{Var}(\bar T_B) = \frac{\sigma^2}{B} + \frac{(B-1)\,\rho\,\sigma^2}{B} = \rho\,\sigma^2 + \frac{(1 - \rho)\,\sigma^2}{B}\).
7. Espérance : linéarité, \(\mathbb{E}[\bar T_B] = \frac1B \sum_b \mathbb{E}[T_b] = \mathbb{E}[T_1]\).

### Exemple chiffré

- Variance avec \(\sigma^2 = 2\) et \(\rho = 0{,}5\) : \(B = 1\) donne @B_var_1@, \(B = 10\) donne \(@B_var_10:2m@\), \(B = 100\) donne \(@B_var_100:2m@\) ; le plancher \(\rho\,\sigma^2\) vaut @B_plancher@. Passer de 10 à 100 arbres ne gagne que \(@B_var_10:2m@ - @B_var_100:2m@\). Simulation (script, vecteurs gaussiens de corrélation 0,5) : @B_var_10_mc:3@ pour \(B = 10\).
- Proportion d'observations absentes : \(n = 10\) donne @B_abs_10:3@, \(n = 100\) donne @B_abs_100:3@, \(n = 1000\) donne @B_abs_1000:4@, et \(1/e \approx @B_inv_e:4m@\). Simulation pour \(n = 100\) : @B_abs_100_mc:3@.
- Un échantillon bootstrap de 100 observations en contient @B_distincts_100:1@ distinctes en moyenne, mais il compte toujours 100 tirages.
- Avec \(n = 1000\) et \(B = 500\) arbres, chaque observation est hors sac pour \(500 \times 0{,}999^{1000} \approx @B_nb_oob:1m@\) arbres en moyenne (linéarité). La probabilité qu'une observation ne soit hors sac pour aucun de \(B = 20\) arbres vaut \((1 - 0{,}999^{1000})^{20} \approx @B_jamais_oob:6m@\).

### L'erreur hors sac

Pour chaque observation \(i\), on moyenne seulement les prédictions des arbres pour lesquels \(i\) est hors sac. Aucun de ces arbres n'a vu \(i\) : c'est une prédiction hors échantillon, comme sur un jeu de validation, obtenue sans mettre de données de côté. L'erreur moyenne de ces prédictions est l'**erreur OOB**. Elle décrit une forêt plus petite que la vraie, d'environ \(@B_abs_1000:3m@\,B\) arbres pour \(n\) grand ; la partie 3 montre que l'erreur quadratique espérée décroît avec le nombre d'arbres, donc l'écart est faible quand \(B\) est grand. Expérience en partie 3.

### Cas limites

- \(\rho = 1\) : moyenner est inutile, la variance reste \(\sigma^2\). \(\rho = 0\) : la variance \(\sigma^2/B\) tend vers 0.
- Une variance est positive pour tout \(B\), donc \(\rho\,\sigma^2 + (1 - \rho)\,\sigma^2/B \ge 0\), soit \(\rho \ge -1/(B-1)\) pour \(B \ge 2\). Une forêt peut contenir autant d'arbres qu'on veut, avec la même loi pour chaque couple : sa corrélation \(\rho\) est donc positive ou nulle.
- \(n = 1\) : l'échantillon bootstrap est toujours l'unique observation.

### Pièges

1. Diviser \(\sigma^2\) par \(B\) comme si les arbres étaient indépendants.
2. Placer \(\rho\) dans le terme en \(1/B\) : c'est lui qui gouverne le terme qui **ne** décroît **pas**.
3. \(1/e\) est une limite et un majorant ; pour \(n = 10\), la proportion d'absentes est plus faible, @B_abs_10:3@.
4. Le bagging ne réduit pas le biais : \(\mathbb{E}[\bar T_B] = \mathbb{E}[T_1]\).

[[controle:bootstrap]]

[[controle:plancher]]

## Partie 3 : la forêt aléatoire

### Le problème

Le plancher \(\rho\,\sigma^2\) ne baisse pas quand on ajoute des arbres. En bagging, deux échantillons bootstrap partagent une grande partie de leurs observations ; si une variable domine, tous les arbres la choisissent pour leurs premières coupures et se ressemblent : \(\rho\) est grand.

### L'intuition

On force la diversité : à chaque coupure, l'arbre ne peut choisir que parmi un sous-ensemble de variables tiré au hasard. Une variable donnée n'est disponible, à chaque coupure, qu'avec probabilité `max_features`\(/p\) : une variable dominante manque à une part des coupures, les arbres diffèrent davantage et \(\rho\) baisse. Le prix : chaque arbre est moins bon, sa variance \(\sigma^2\) et son biais peuvent augmenter. Pour \(B\) grand, la formule (B2) dit ce qui compte : \(\rho\,\sigma^2\), plus le biais.

### Les définitions

Une **forêt aléatoire** est un bagging d'arbres dans lequel, à chaque coupure, on tire uniformément sans remise `max_features` variables parmi les \(p\), et l'on cherche la meilleure coupure parmi elles seulement. Avec `max_features` égal à \(p\), c'est le bagging.

### Ce qui est démontré, et ce qui ne l'est pas

- Démontré : (B2) et (B3) s'appliquent aux arbres d'une forêt ; la variance de la forêt tend vers \(\rho\,\sigma^2\) quand \(B \to \infty\).
- Non démontré, et faux en général : « réduire `max_features` réduit l'erreur ». C'est un réglage du compromis entre biais et variance, à choisir par validation ou par l'erreur OOB. L'expérience suivante mesure les trois effets.

Expérience. Données `make_friedman1` de scikit-learn : 10 variables uniformes sur \([0, 1]\), cible \(10 \sin(\pi x_1 x_2) + 20\,(x_3 - 0{,}5)^2 + 10\,x_4 + 5\,x_5\) plus un bruit gaussien (ici d'écart-type 1), les variables 6 à 10 sans effet (la formule est contrôlée par l'assertion du code). On tire 40 échantillons d'entraînement indépendants de 200 observations ; sur chacun, une forêt de 50 arbres ; on mesure, en 20 points fixés, la variance \(\sigma^2\) d'un arbre et la corrélation \(\rho\) entre deux arbres sur les 40 échantillons, la variance de la forêt et son biais au carré.

```python-exec
from sklearn.datasets import make_friedman1
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def vraie(X):  # fonction de régression de make_friedman1
    return 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]

X0, _ = make_friedman1(n_samples=20, n_features=10, random_state=1)   # 20 points fixés
assert np.allclose(make_friedman1(n_samples=20, n_features=10, noise=0.0, random_state=1)[1], vraie(X0))
R, B = 40, 50
for mf in [10, 3, 1]:
    P = np.empty((R, B, len(X0)))   # P[r, b, k] : arbre b de la forêt entraînée sur l'échantillon r, au point k
    for r in range(R):
        Xr, yr = make_friedman1(n_samples=200, n_features=10, noise=1.0, random_state=100 + r)
        foret = RandomForestRegressor(n_estimators=B, max_features=mf, random_state=r).fit(Xr, yr)
        P[r] = [arbre.predict(X0) for arbre in foret.estimators_]
    sigma2 = P.var(axis=0, ddof=1).mean(axis=0)
    rho = np.array([np.corrcoef(P[:, :, k].T)[np.triu_indices(B, 1)].mean() for k in range(len(X0))])
    moyenne = P.mean(axis=1)
    print(f"max_features = {mf:2d}: sigma2 {sigma2.mean():5.2f}, rho {rho.mean():.3f}, "
          f"variance de la forêt {moyenne.var(axis=0, ddof=1).mean():.3f} (formule {np.mean(rho * sigma2 + (1 - rho) * sigma2 / B):.3f}), "
          f"biais au carré {np.mean((moyenne.mean(axis=0) - vraie(X0)) ** 2):.3f}")
print("max_features par défaut :", RandomForestRegressor().max_features, "en régression,", RandomForestClassifier().max_features, "en classification")
```

Lecture : quand `max_features` baisse de 10 à 1, \(\rho\) baisse, \(\sigma^2\) monte, la variance de la forêt baisse, et le biais au carré monte. Sur ces données, la somme du biais au carré et de la variance n'est pas meilleure avec `max_features` égal à 1 qu'avec le bagging : le tirage de variables est un réglage, pas une amélioration garantie. La formule (B2), calculée avec les \(\sigma^2\) et \(\rho\) estimés, retrouve la variance mesurée. Dernière ligne : en scikit-learn 1.1.1, la forêt de **régression** utilise par défaut toutes les variables, c'est-à-dire du bagging.

### Ajouter des arbres : l'énoncé

On fixe un point \(\mathbf{x}\). La cible en ce point est une variable aléatoire \(Y\), indépendante de l'échantillon d'entraînement et des tirages de la forêt, d'espérance \(\mu\) et de variance \(\mathrm{Var}(Y)\) (le bruit, que nul modèle ne prédit). Avec les notations de (B2) :

\(\displaystyle \mathbb{E}\big[(\bar T_B - Y)^2\big] = \big(\mathbb{E}[T_1] - \mu\big)^2 + \rho\,\sigma^2 + \frac{(1 - \rho)\,\sigma^2}{B} + \mathrm{Var}(Y).\)

Ce qui est vrai :

- cette erreur quadratique espérée ne croît pas avec \(B\), et elle décroît strictement si \(\rho < 1\) ; en moyennant sur les points \(\mathbf{x}\), l'erreur de test espérée ne croît pas non plus ;
- la même conclusion vaut à échantillon d'entraînement fixé, en ne prenant l'espérance que sur les tirages de la forêt : sachant les données, les arbres sont indépendants et de même loi, la formule s'applique avec une corrélation nulle et la variance conditionnelle d'un arbre.

Ce qui est faux :

- « la courbe d'erreur de test baisse à chaque arbre ajouté » : l'énoncé porte sur une espérance ; une forêt donnée fluctue ;
- « une forêt ne peut pas sur-apprendre » : la limite \((\mathbb{E}[T_1] - \mu)^2 + \rho\,\sigma^2\) dépend de la profondeur des arbres et de `max_features`, que \(B\) ne règle pas ;
- l'argument ne couvre pas le vote majoritaire avec l'erreur de classement, qui n'est pas l'écart quadratique d'une moyenne ; il couvre la moyenne des probabilités avec la perte quadratique (score de Brier).

### Ajouter des arbres : la preuve

1. On écrit \(\bar T_B - Y = (\bar T_B - \mathbb{E}\bar T_B) + (\mathbb{E}\bar T_B - \mu) + (\mu - Y)\) et on développe le carré.
2. Doubles produits : le premier terme est centré et le deuxième constant, d'où une espérance nulle ; le premier et le troisième sont indépendants et centrés, donc l'espérance de leur produit est le produit de leurs espérances, nul ; le deuxième est constant et le troisième centré.
3. Il reste \(\mathrm{Var}(\bar T_B) + (\mathbb{E}\bar T_B - \mu)^2 + \mathrm{Var}(Y)\). On remplace \(\mathrm{Var}(\bar T_B)\) par (B2) et \(\mathbb{E}\bar T_B\) par \(\mathbb{E}[T_1]\).
4. Le seul terme qui dépend de \(B\) est \((1 - \rho)\,\sigma^2/B\), positif et décroissant en \(B\) (strictement si \(\rho < 1\)).

Expérience : une forêt de 500 arbres sur des données plus bruitées, l'erreur de test de la forêt formée des \(B\) premiers arbres, et l'erreur OOB.

```python-exec
from sklearn.metrics import r2_score

Xa, ya = make_friedman1(n_samples=200, n_features=10, noise=5.0, random_state=7)    # entraînement
Xt, yt = make_friedman1(n_samples=5000, n_features=10, noise=5.0, random_state=8)   # test
foret = RandomForestRegressor(n_estimators=500, max_features=3, oob_score=True, random_state=0).fit(Xa, ya)
arbres = np.array([arbre.predict(Xt) for arbre in foret.estimators_])
cumul = np.cumsum(arbres, axis=0) / np.arange(1, 501)[:, None]     # ligne B - 1 : forêt des B premiers arbres
erreur_rf = ((cumul - yt) ** 2).mean(axis=1)
for b in [1, 10, 50, 100, 200, 500]:
    print(f"B = {b:3d}: erreur quadratique de test {erreur_rf[b - 1]:.2f}")
print("hausses d'un arbre au suivant :", int((np.diff(erreur_rf) > 0).sum()), "sur 499")
print(f"R2 hors sac {foret.oob_score_:.3f} ; R2 de test {r2_score(yt, foret.predict(Xt)):.3f}")
```

Lecture : la tendance est à la baisse, puis au plateau, mais la courbe réalisée remonte d'un arbre au suivant à de nombreuses reprises : c'est l'espérance qui décroît, pas chaque réalisation. Le \(R^2\) hors sac est proche du \(R^2\) de test, sans aucune donnée mise de côté.

### Cas limites

- `max_features` égal à 1 : la variable de chaque coupure est tirée au hasard ; seul le seuil est optimisé.
- \(B\) petit : le terme \((1 - \rho)\,\sigma^2/B\) domine, et une forêt dont les arbres ont un \(\sigma^2\) plus grand peut avoir une variance plus grande que le bagging ; pour \(B = 1\), c'est le cas dès que son \(\sigma^2\) est plus grand.

### Pièges

1. Croire que la forêt réduit le biais : elle réduit la variance, et le tirage de variables peut augmenter le biais.
2. Laisser la valeur par défaut de `max_features` en régression avec scikit-learn et croire faire une forêt aléatoire.
3. Régler \(B\) par validation comme un hyperparamètre sensible : il suffit qu'il soit assez grand ; le coût est le temps de calcul.

## Partie 4 : le boosting de gradient

### Le problème

La forêt réduit la variance d'arbres profonds, pas leur biais. Le boosting fait l'inverse : il part d'un modèle simple, biaisé, et corrige ses erreurs une à une, avec une suite d'arbres peu profonds, chacun entraîné sur ce que les précédents ont manqué.

### L'intuition

Avec la perte quadratique : on ajuste un arbre aux résidus, on ajoute une fraction de cet arbre au modèle, on recalcule les résidus, et l'on recommence. Pour une autre perte, on remplace le résidu par la direction dans laquelle chaque prédiction doit bouger pour faire baisser la perte le plus vite : l'opposé de la dérivée de la perte.

### Les définitions

- La **perte** \(\ell(y, f)\) mesure l'erreur d'une prédiction \(f\), un réel, pour la cible \(y\) ; elle est dérivable en \(f\). En régression, \(f\) prédit \(y\) ; en classification, \(f\) est un score, le logarithme de la cote, et la probabilité prédite est \(\pi = 1/(1 + e^{-f})\).
- Perte quadratique : \(\ell(y, f) = \tfrac12 (y - f)^2\). Perte logistique, pour \(y \in \{0, 1\}\) : \(\ell(y, f) = -\big[y \ln \pi + (1 - y)\ln(1 - \pi)\big]\), avec \(\pi = 1/(1 + e^{-f})\).
- **Algorithme.** \(f_0\) est la constante qui minimise \(\sum_i \ell(y_i, c)\) en \(c\) (la moyenne des \(y_i\) pour la perte quadratique, par (R1)). Pour \(m = 1, 2, \dots, B\) :
1. pseudo-résidus : \(r_i = -\frac{\partial \ell}{\partial f}\big(y_i, f_{m-1}(\mathbf{x}_i)\big)\) pour chaque observation \(i\) ;
2. on ajuste un arbre de régression \(u_m\), de profondeur limitée, aux couples \((\mathbf{x}_i, r_i)\), avec le critère de variance de la partie 1 ; on peut ensuite recalculer la valeur de chaque feuille pour minimiser la perte de ses observations (c'est ce que précise XGBoost, partie 5) ;
3. mise à jour : \(f_m = f_{m-1} + \eta\,u_m\), où \(\eta \in \,]0, 1]\) est le **taux d'apprentissage**.

### L'énoncé

1. (D1) Perte quadratique : \(r_i = y_i - f_{m-1}(\mathbf{x}_i)\), le résidu. Perte logistique : \(r_i = y_i - \pi_i\), où \(\pi_i\) est la probabilité prédite ; de plus la dérivée seconde de la perte logistique en \(f\) vaut \(\pi\,(1 - \pi)\) (elle servira en partie 5).
2. (D2) Le boosting est une descente de gradient sur le vecteur des \(n\) prédictions d'entraînement, dont chaque pas est remplacé par l'arbre qui l'approche le mieux au sens des moindres carrés.
3. (D3) Perte quadratique, valeurs de feuilles égales aux moyennes des résidus, \(0 < \eta < 2\) : la somme des carrés des résidus d'entraînement ne croît jamais. Si de plus chaque arbre isole chaque observation dans sa propre feuille, elle est multipliée par \((1 - \eta)^2\) à chaque arbre et tend vers 0 : le modèle reproduit les données d'entraînement, bruit compris.
4. Conséquence, à la différence de la forêt : le nombre d'arbres peut sur-apprendre. Il se règle par **arrêt précoce** : on garde le nombre d'arbres qui minimise l'erreur sur un jeu de validation.

### La preuve

1. (D1) Perte quadratique : \(\frac{\partial}{\partial f}\tfrac12(y - f)^2 = f - y\), dont l'opposé est \(y - f\). Perte logistique : \(\frac{d\pi}{df} = \frac{e^{-f}}{(1 + e^{-f})^2} = \pi\,(1 - \pi)\). Donc \(\frac{\partial \ell}{\partial f} = -y\,\frac{\pi(1-\pi)}{\pi} + (1 - y)\,\frac{\pi(1-\pi)}{1 - \pi} = -y\,(1 - \pi) + (1 - y)\,\pi = \pi - y\), et en dérivant une seconde fois, \(\frac{\partial^2 \ell}{\partial f^2} = \pi\,(1 - \pi)\).
2. (D2) La perte d'entraînement ne dépend de \(f\) qu'à travers le vecteur \(\mathbf{v} = (f(\mathbf{x}_1), \dots, f(\mathbf{x}_n))\) : \(\mathcal{L}(\mathbf{v}) = \sum_i \ell(y_i, v_i)\). La composante \(i\) de son gradient est \(\frac{\partial \ell}{\partial f}(y_i, v_i) = -r_i\). Un pas de descente de gradient de longueur \(\eta\) remplacerait \(\mathbf{v}\) par \(\mathbf{v} + \eta\,\mathbf{r}\), où \(\mathbf{r} = (r_1, \dots, r_n)\). Mais un vecteur de \(n\) nombres ne prédit rien en un nouveau point : on le remplace par l'arbre \(u_m\), fonction de \(\mathbf{x}\), le plus proche de \(\mathbf{r}\) aux points d'entraînement au sens des moindres carrés, que construit l'étape 2 de l'algorithme.
3. (D3) Notations : \(\mathbf{r}\) est le vecteur des résidus avant l'arbre \(m\), \(\hat{\mathbf{r}}\) le vecteur des prédictions de cet arbre aux points d'entraînement, \(\langle \mathbf{a}, \mathbf{b} \rangle = \sum_i a_i b_i\) le produit scalaire et \(\lVert \mathbf{a} \rVert^2 = \langle \mathbf{a}, \mathbf{a} \rangle\). Dans chaque feuille \(S\), \(\hat r_i\) vaut la moyenne \(\bar r_S\) des résidus de la feuille.
4. \(\langle \mathbf{r} - \hat{\mathbf{r}}, \hat{\mathbf{r}} \rangle = \sum_{\text{feuilles } S} \bar r_S \sum_{i \in S}(r_i - \bar r_S) = 0\), car la somme des écarts à la moyenne est nulle dans chaque feuille. Donc \(\langle \mathbf{r}, \hat{\mathbf{r}} \rangle = \lVert \hat{\mathbf{r}} \rVert^2\).
5. Les nouveaux résidus sont \(y_i - f_m(\mathbf{x}_i) = r_i - \eta\,\hat r_i\). En développant : \(\lVert \mathbf{r} - \eta\,\hat{\mathbf{r}} \rVert^2 = \lVert \mathbf{r} \rVert^2 - 2\eta\,\langle \mathbf{r}, \hat{\mathbf{r}} \rangle + \eta^2 \lVert \hat{\mathbf{r}} \rVert^2 = \lVert \mathbf{r} \rVert^2 - \eta\,(2 - \eta)\,\lVert \hat{\mathbf{r}} \rVert^2\), et \(\eta\,(2 - \eta) > 0\) pour \(0 < \eta < 2\).
6. Si chaque observation est seule dans sa feuille, \(\hat{\mathbf{r}} = \mathbf{r}\), et les nouveaux résidus valent \((1 - \eta)\,\mathbf{r}\) : la somme des carrés est multipliée par \((1 - \eta)^2\).

### Exemple chiffré

Avec \(\eta = @D_eta:1m@\) et des arbres qui isolent chaque observation, la somme des carrés des résidus d'entraînement est multipliée par \((1 - \eta)^2 = @D_facteur:2m@\) à chaque arbre ; elle passe sous 1 % de sa valeur initiale après @D_m_1pct@ arbres. Expérience sur les données de la partie 3, arbres de profondeur 3 :

```python-exec
from sklearn.ensemble import GradientBoostingRegressor

for eta in [0.1, 0.3]:
    gb = GradientBoostingRegressor(n_estimators=1000, learning_rate=eta, max_depth=3, random_state=0).fit(Xa, ya)
    test = np.array([np.mean((p - yt) ** 2) for p in gb.staged_predict(Xt)])
    entr = np.array([np.mean((p - ya) ** 2) for p in gb.staged_predict(Xa)])
    print(f"eta = {eta}: erreur de test minimale {test.min():.2f} après {test.argmin() + 1} arbres ; "
          f"après 1000 arbres : test {test[-1]:.2f}, entraînement {entr[-1]:.2e} ; "
          f"erreur d'entraînement jamais croissante : {bool(np.all(np.diff(entr) <= 1e-12))}")
print(f"forêt de la partie 3, mêmes données, 500 arbres : test {erreur_rf[-1]:.2f}")
```

Lecture : l'erreur d'entraînement ne croît jamais, comme l'annonce (D3), et devient quasi nulle ; l'erreur de test passe par un minimum puis remonte : le boosting a sur-appris avec trop d'arbres. Avec un taux d'apprentissage plus grand, le minimum arrive après moins d'arbres. Ces deux constats valent pour ces données ; seule la décroissance de l'erreur d'entraînement est un théorème.

### Cas limites

- Arbres de profondeur 1 (une seule coupure) : chaque arbre ne dépend que d'une variable, donc le modèle est une somme de fonctions d'une variable chacune, sans aucune interaction. De même, un arbre de profondeur \(d\) fait intervenir au plus \(d\) variables sur chaque chemin de la racine à une feuille.
- \(\eta = 1\), perte quadratique et arbre qui isole chaque observation : un seul arbre reproduit exactement les données d'entraînement.

### Pièges

1. Avec la perte quadratique, le pseudo-résidu est le résidu : une étiquette aberrante garde un grand pseudo-résidu tant qu'elle n'est pas ajustée, et les arbres suivants lui consacrent des feuilles. La perte de Huber, quadratique pour un résidu de valeur absolue au plus \(\delta\) et linéaire au-delà, a un pseudo-résidu borné par \(\delta\) (contrôlé numériquement par le script).
2. Perte logistique : le pseudo-résidu \(y - \pi\) est sur l'échelle des probabilités, mais l'arbre s'ajoute au score \(f\).
3. Avec \((y - f)^2\) sans le facteur \(\tfrac12\), la dérivée vaut \(2\,(f - y)\) : le pseudo-résidu est le double du résidu.
4. Plus d'arbres n'est pas toujours mieux, à la différence de la forêt.

[[controle:residus]]

## Partie 5 : XGBoost

### Le problème

Le boosting de la partie 4 ajuste chaque arbre aux pseudo-résidus par les moindres carrés, puis recalcule éventuellement les valeurs des feuilles par une seconde minimisation. XGBoost (Chen et Guestrin, 2016) remplace ces deux étapes par un seul critère : une approximation d'ordre 2 de la perte, plus une pénalité explicite sur la complexité de l'arbre. Ce critère donne, sous forme fermée et pour toute perte deux fois dérivable, la valeur de chaque feuille et le gain de chaque coupure.

### Les définitions

On construit l'arbre numéro \(m\).

- \(\hat y_i = f_{m-1}(\mathbf{x}_i)\) est la prédiction courante de l'observation \(i\), somme des arbres déjà construits ; pour la perte logistique, c'est un score, pas une probabilité.
- \(g_i = \frac{\partial \ell}{\partial f}(y_i, \hat y_i)\) et \(h_i = \frac{\partial^2 \ell}{\partial f^2}(y_i, \hat y_i)\) sont les dérivées première et seconde de la perte par rapport à la prédiction, évaluées en la prédiction courante. Par définition, \(g_i = -r_i\).
- Pour un ensemble \(S\) d'observations : \(G_S = \sum_{i \in S} g_i\) et \(H_S = \sum_{i \in S} h_i\).
- Le nouvel arbre ajoute à la prédiction de chaque observation de la feuille \(S\) une même valeur \(w_S\), le **poids** de la feuille.
- Objectif régularisé : \(\sum_i \ell(y_i, \hat y_i + w_{S(i)}) + \gamma \times (\text{nombre de feuilles}) + \tfrac12 \lambda \sum_S w_S^2\), où \(S(i)\) est la feuille de l'observation \(i\), \(\lambda \ge 0\) pénalise les poids et \(\gamma \ge 0\) chaque feuille.
- Développement de Taylor à l'ordre 2 en \(w\), autour de \(\hat y_i\) : \(\ell(y_i, \hat y_i + w) = \ell(y_i, \hat y_i) + g_i\,w + \tfrac12 h_i\,w^2 + o(w^2)\), où \(o(w^2)\) est un terme négligeable devant \(w^2\) quand \(w \to 0\) (formule de Taylor-Young, valable si \(\ell\) est deux fois dérivable en \(f\)). On supprime les termes \(\ell(y_i, \hat y_i)\), qui ne dépendent pas de l'arbre, et le reste ; on regroupe par feuille. Critère approché :

\(\displaystyle \sum_{\text{feuilles } S} \Big[G_S\,w_S + \tfrac12 (H_S + \lambda)\,w_S^2\Big] + \gamma \times (\text{nombre de feuilles}).\)

### L'énoncé

1. (X1) Si \(H_S + \lambda > 0\), le poids optimal est \(w_S^\star = -\dfrac{G_S}{H_S + \lambda}\), et la valeur optimale du critère, pour une structure d'arbre donnée, est \(-\tfrac12 \sum_S \dfrac{G_S^2}{H_S + \lambda} + \gamma \times (\text{nombre de feuilles})\).
2. (X2) Couper la feuille \(S\) en \(L\) et \(R\) fait baisser le critère du **gain** écrit ci-dessous, après la liste. La coupure améliore le critère si et seulement si ce gain est strictement positif ; XGBoost choisit, à chaque nœud, la coupure de gain maximal.
3. (X3) Si \(\lambda = 0\), \(\gamma = 0\) et \(H_L, H_R > 0\), le gain est positif ou nul. Si \(\lambda > 0\), le crochet peut être négatif.
4. (X4) Perte quadratique : \(g_i = \hat y_i - y_i = -r_i\) et \(h_i = 1\). Le développement est exact, \(w_S^\star = \dfrac{\sum_{i \in S} r_i}{\lvert S \rvert + \lambda}\) est la moyenne des résidus rétrécie par le facteur \(\lvert S \rvert/(\lvert S \rvert + \lambda)\), et, pour \(\lambda = \gamma = 0\), le gain est la moitié de la baisse de la somme des carrés des résidus : la même coupure que le critère de variance de la partie 1.
5. (X5) Perte logistique : \(g_i = \pi_i - y_i\) et \(h_i = \pi_i\,(1 - \pi_i) \in \,]0, \tfrac14]\), où \(\pi_i = 1/(1 + e^{-\hat y_i})\). Le poids \(w_S^\star\) est exactement un pas de la méthode de Newton sur la perte pénalisée de la feuille, partant de \(w = 0\) ; ce n'est pas son minimum exact.
6. (X6) Rôle des réglages. \(\lambda\) rétrécit les poids vers 0 (\(\lvert w_S^\star \rvert\) décroît avec \(\lambda\)) et diminue les gains. \(\gamma\) est le gain minimal exigé pour couper. Le taux d'apprentissage \(\eta\) (nommé `eta`) multiplie le poids : on ajoute \(\eta\,w_S^\star\). `min_child_weight` interdit une coupure dont un enfant aurait \(H_L\) ou \(H_R\) inférieur à sa valeur : pour la perte quadratique c'est un nombre minimal d'observations ; pour la perte logistique, il faut au moins \(1/(\pi(1 - \pi))\) observations de probabilité prédite \(\pi\) pour atteindre une somme égale à 1.

Le gain de (X2) :

\(\displaystyle \mathrm{gain} = \frac12 \left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma.\)

La définition de `min_child_weight` (somme minimale des \(h_i\) dans chaque enfant) et les valeurs par défaut citées en partie 6 ont été vérifiées le 5 octobre 2026 sur XGBoost 3.2.0, dans un environnement séparé, car XGBoost n'est pas installé dans l'environnement des notebooks de la plateforme. Expérience : perte logistique, prédiction initiale 0,5, donc \(h_i = 0{,}5 \times 0{,}5 = 0{,}25\) pour chaque observation ; une coupure qui laisse 4 observations dans chaque enfant (somme des \(h_i\) égale à 1) est acceptée avec `min_child_weight` égal à 1 et refusée avec 1,01.

### La preuve

1. (X1) Chaque poids \(w_S\) n'apparaît que dans le terme de sa feuille : on minimise chaque terme séparément. \(q(w) = G\,w + \tfrac12 (H + \lambda)\,w^2\) est un polynôme du second degré de coefficient dominant positif quand \(H + \lambda > 0\), donc strictement convexe ; \(q'(w) = G + (H + \lambda)\,w\) s'annule en \(w^\star = -G/(H + \lambda)\), et \(q(w^\star) = -\frac{G^2}{H + \lambda} + \frac12\frac{G^2}{H + \lambda} = -\frac12\frac{G^2}{H + \lambda}\). On somme sur les feuilles.
2. (X2) Avant la coupure, la feuille \(S\) contribue \(-\tfrac12 G_S^2/(H_S + \lambda) + \gamma\) ; après, \(-\tfrac12 G_L^2/(H_L + \lambda) - \tfrac12 G_R^2/(H_R + \lambda) + 2\gamma\). Comme \(L\) et \(R\) partagent \(S\), \(G_S = G_L + G_R\) et \(H_S = H_L + H_R\). Le gain est la différence « avant moins après ».
3. (X3) Avec \(\lambda = 0\), notons \(a = G_L\), \(b = G_R\), \(u = H_L > 0\), \(v = H_R > 0\). Au dénominateur commun : \(\frac{a^2}{u} + \frac{b^2}{v} - \frac{(a + b)^2}{u + v} = \frac{a^2 v (u + v) + b^2 u (u + v) - (a + b)^2 u v}{u v (u + v)} = \frac{(a v - b u)^2}{u v (u + v)} \ge 0\), en développant le numérateur. Avec \(\lambda > 0\), contre-exemple numérique ci-dessous.
4. (X4) \(\frac{\partial}{\partial f}\tfrac12 (y - f)^2 = f - y\), dérivée seconde 1, dérivée troisième nulle : le reste de Taylor est nul. Poids : \(-G_S/(H_S + \lambda) = \sum_{i \in S} r_i/(\lvert S \rvert + \lambda)\). Gain : par (R1) avec \(c = 0\), la somme des carrés des résidus de \(S\) autour de leur moyenne vaut \(\sum_{i \in S} r_i^2 - \lvert S \rvert\,\bar r_S^2 = \sum_{i \in S} r_i^2 - G_S^2/\lvert S \rvert\). La baisse due à la coupure vaut donc \(G_L^2/\lvert L \rvert + G_R^2/\lvert R \rvert - G_S^2/\lvert S \rvert\), car les \(\sum r_i^2\) se compensent : c'est le double du gain avec \(H = \lvert \cdot \rvert\) et \(\lambda = \gamma = 0\).
5. (X5) Les dérivées sont celles de (D1). \(\pi\,(1 - \pi) \le \tfrac14\) car \(\tfrac14 - \pi + \pi^2 = (\pi - \tfrac12)^2 \ge 0\), et \(\pi\,(1 - \pi) > 0\) car \(0 < \pi < 1\). La perte pénalisée de la feuille est \(\psi(w) = \sum_{i \in S} \ell(y_i, \hat y_i + w) + \tfrac12 \lambda w^2\) ; \(\psi'(0) = G_S\) et \(\psi''(0) = H_S + \lambda\). Un pas de Newton partant de 0 donne \(-\psi'(0)/\psi''(0) = w_S^\star\).
6. (X6) \(n\) observations de même probabilité prédite \(\pi\) ont \(H = n\,\pi\,(1 - \pi)\), qui atteint 1 si et seulement si \(n \ge 1/(\pi(1 - \pi))\).

### Exemple chiffré

L'exemple jouet de la partie 1, perte logistique, prédiction courante \(\hat y_i = 0\) pour toutes les observations, donc \(\pi_i = \tfrac12\) ; \(\lambda = 1\), \(\gamma = 0\).

1. Dérivées : \(g_i = \pi_i - y_i\) vaut \(@E_g_neg:1m@\) si \(y_i = 0\) et \(@E_g_pos:1m@\) si \(y_i = 1\) ; \(h_i = @E_h:2m@\) pour tous.
2. Meilleure coupure, au seuil \(@E_seuil:1m@\) : \(G_L = @E_GL@\), \(H_L = @E_HL@\), \(G_R = @E_GR@\), \(H_R = @E_HR@\).
3. Gain : \(\tfrac12 \big[\tfrac{(@E_GL@)^2}{@E_HL@ + 1} + \tfrac{(@E_GR@)^2}{@E_HR@ + 1} - \tfrac{(@E_GL@ + (@E_GR@))^2}{@E_HL@ + @E_HR@ + 1}\big] = \tfrac12 \big[@E_sL:1m@ + @E_sR@ - \tfrac{@E_sP_num@}{@E_sP_den@}\big] = \tfrac{@E_gain_num@}{@E_gain_den@} \approx @E_gain:3m@\).
4. Poids : \(w_L = @E_wL:1m@\) et \(w_R = @E_wR@\). Avec \(\eta = @E_eta:1m@\), on ajoute \(@E_eta_wR:1m@\) au score des observations de droite, dont la probabilité prédite passe de 0,5 à \(1/(1 + e^{-0{,}3}) \approx @E_pi_new:3m@\).
5. Newton contre minimum exact, feuille de droite (@E_n_droite@ observations d'étiquette 1) : le poids XGBoost vaut @E_wR@, le minimum exact de la perte pénalisée \(@E_w_exact:4m@\).
6. Gain négatif sans \(\gamma\) : la coupure au seuil \(@E_seuil_neg:1m@\) a un gain \(@E_gain_neg:4m@\) avec \(\lambda = 1\), et \(@E_gain_neg_l0:4m@\) avec \(\lambda = 0\).

Les mêmes calculs en numpy, sans XGBoost :

```python-exec
def poids(g, h, lam):
    return -g.sum() / (h.sum() + lam)

def gain_xgb(g, h, gauche, lam, gam):
    s = lambda G, H: G ** 2 / (H + lam)
    return 0.5 * (s(g[gauche].sum(), h[gauche].sum()) + s(g[~gauche].sum(), h[~gauche].sum()) - s(g.sum(), h.sum())) - gam

score = np.zeros(8)                         # prédiction courante (score) de chaque observation
pi = 1 / (1 + np.exp(-score))               # probabilité prédite
g, h = pi - y_jouet, pi * (1 - pi)          # dérivées première et seconde de la perte logistique
seuils = np.arange(1.5, 8, 1.0)
for s in seuils:
    print(f"seuil {s}: gain (lambda = 1, gamma = 0) {gain_xgb(g, h, x_jouet <= s, 1.0, 0.0):+.4f}")
meilleur = max(seuils, key=lambda s: gain_xgb(g, h, x_jouet <= s, 1.0, 0.0))
gauche = x_jouet <= meilleur
print("seuil retenu :", meilleur, "; poids gauche", poids(g[gauche], h[gauche], 1.0), "; poids droite", poids(g[~gauche], h[~gauche], 1.0))

yd = y_jouet[~gauche]   # feuille de droite : méthode de Newton sur la vraie perte pénalisée (lambda = 1)
w = 0.0
for etape in range(1, 6):
    p = 1 / (1 + np.exp(-w))
    w = w - (np.sum(p - yd) + 1.0 * w) / (len(yd) * p * (1 - p) + 1.0)
    print(f"Newton, étape {etape}: w = {w:.6f}")
```

La première étape de Newton est le poids XGBoost ; les suivantes convergent vers le minimum exact.

### Cas limites

- \(\lambda \to \infty\) : tous les poids tendent vers 0, l'arbre n'apprend plus rien.
- Pour une perte convexe en \(f\) (\(h_i \ge 0\)), \(H_S + \lambda = 0\) n'arrive qu'avec \(\lambda = 0\) et des \(h_i\) tous nuls, ce qui est impossible pour la perte logistique (\(h_i > 0\)) et pour la perte quadratique (\(h_i = 1\)). Pour une perte non convexe, \(H_S + \lambda\) peut être négatif ou nul, et (X1) ne s'applique pas.
- `min_child_weight` égal à 1 et perte logistique : avec \(\pi = \tfrac12\), \(h = \tfrac14\) et il faut @E_n_05@ observations ; avec \(\pi = 0{,}99\), \(h = @E_h_099:4m@\) et il en faut @E_n_099@. Les zones où le modèle est déjà très confiant deviennent difficiles à couper.

### Pièges

1. Le signe : \(w^\star = -G/(H + \lambda)\), et \(g_i\) est l'opposé du résidu.
2. Diviser par le nombre d'observations au lieu de \(H + \lambda\) : réflexe de la perte quadratique, faux pour la perte logistique.
3. Dériver par rapport à la probabilité \(\pi\) au lieu du score \(f\).
4. \(\gamma\) se retranche une fois par coupure (une feuille de plus), pas une fois par feuille de l'arbre.
5. Le facteur \(\tfrac12\) du gain.
6. Lire `min_child_weight` comme un nombre d'observations : ce n'en est un que pour la perte quadratique.

[[controle:newton]]

## Partie 6 : comparer forêt aléatoire et gradient boosting

Chaque ligne renvoie à une partie démontrée ou à une expérience de cette leçon.

| Critère | Forêt aléatoire | Gradient boosting, XGBoost | Justification |
|---|---|---|---|
| Ce que la combinaison réduit | la variance : la moyenne garde le biais d'un arbre, et sa variance vaut \(\rho\,\sigma^2 + (1 - \rho)\,\sigma^2/B\) | le biais : chaque arbre corrige les pseudo-résidus, et l'erreur d'entraînement ne croît jamais (perte quadratique) | (B2), (D3) |
| Arbres de base | profonds, faible biais à moyenner | peu profonds, petites corrections successives | défauts imprimés ci-dessous |
| Nombre d'arbres | l'erreur quadratique espérée ne croît pas avec \(B\) : prendre \(B\) assez grand | peut sur-apprendre : arrêt précoce sur un jeu de validation | partie 3, partie 4 et leurs expériences |
| Construction | arbres indépendants sachant les données : construction parallèle | séquentielle : l'arbre \(m\) dépend de \(f_{m-1}\) ; XGBoost parallélise la recherche des coupures à l'intérieur d'un arbre | (B3), algorithme de la partie 4 ; Chen et Guestrin, 2016 |
| Réglages sensibles | `max_features`, taille minimale des feuilles | \(B\), \(\eta\) et la profondeur, liés entre eux ; \(\lambda\), \(\gamma\), `min_child_weight` | partie 3, (X6) |
| Validation intégrée | erreur hors sac | sans sous-échantillonnage, chaque arbre voit toutes les observations : aucune n'est hors sac | partie 2 |
| Étiquettes aberrantes, perte quadratique | chaque arbre profond les ajuste, la moyenne réduit la variance qui en résulte | leur pseudo-résidu reste grand et attire les arbres suivants ; perte de Huber pour borner | partie 4, piège 1 |

```python-exec
from sklearn.ensemble import GradientBoostingClassifier

for modele in [RandomForestRegressor(), RandomForestClassifier(), GradientBoostingRegressor(), GradientBoostingClassifier()]:
    print(f"{type(modele).__name__}: n_estimators {modele.n_estimators}, max_depth {modele.max_depth}, "
          f"max_features {modele.max_features}, learning_rate {getattr(modele, 'learning_rate', 'sans objet')}")
```

Valeurs par défaut de XGBoost, lues dans la configuration d'un modèle entraîné sans paramètre sous XGBoost 3.2.0 (vérification du 5 octobre 2026, environnement séparé) : `max_depth` 6, `eta` 0,3, `lambda` 1, `gamma` 0 et `min_child_weight` 1.

Réponse d'entretien, en cinq phrases :

1. Les deux combinent des arbres, mais la forêt les moyenne en parallèle pour réduire la variance d'arbres profonds, alors que le boosting les ajoute en série pour réduire le biais d'arbres peu profonds.
2. La variance d'une moyenne de \(B\) arbres corrélés vaut \(\rho\,\sigma^2 + (1 - \rho)\,\sigma^2/B\) : la forêt tire des variables à chaque coupure pour baisser \(\rho\), au prix d'un \(\sigma^2\) et d'un biais plus grands, compromis réglé par `max_features`.
3. Le boosting est une descente de gradient dans l'espace des fonctions ; XGBoost utilise en plus la dérivée seconde de la perte, ce qui donne le poids de feuille \(-G/(H + \lambda)\) et un gain de coupure exact pour le critère approché, avec des pénalités \(\lambda\) et \(\gamma\).
4. Ajouter des arbres à une forêt ne fait pas croître l'erreur quadratique espérée, alors qu'un boosting sur-apprend avec trop d'arbres : on le règle par arrêt précoce, avec un taux d'apprentissage et une profondeur choisis par validation.
5. La forêt demande peu de réglages, et \(B\) n'a pas à être réglé finement ; le boosting en demande davantage, et l'on compare les deux par validation sur le problème posé.

## Partie 7 : importance des variables

### Le problème

Quelles variables le modèle utilise-t-il ? Deux mesures sont courantes, et la plus utilisée a un biais connu.

### Les définitions

- **Importance par impureté** (MDI, mean decrease in impurity ; `feature_importances_` de scikit-learn) : pour chaque variable, la somme, sur les nœuds qui coupent selon elle, de la part des observations qui atteignent le nœud multipliée par le gain de la coupure ; normalisée pour sommer à 1 dans chaque arbre, puis moyennée sur les arbres et normalisée de nouveau. Elle est calculée sur les données d'entraînement de chaque arbre.
- **Importance par permutation** : sur un jeu de validation, le score du modèle moins son score quand on permute au hasard les valeurs de la variable \(j\), ce qui coupe son lien avec la cible et les autres variables ; moyennée sur plusieurs permutations.

### L'énoncé

1. (I1) Hypothèse nulle : les étiquettes sont indépendantes, de même loi, et indépendantes des variables. Alors le meilleur gain d'une variable continue aux valeurs distinctes est, en espérance, au moins celui d'une variable binaire. Et comme tout gain est positif ou nul (R2 à R4), une variable sans aucun lien avec la cible reçoit une importance par impureté positive dès qu'elle sert à couper.
2. (I2) Si, pour chaque observation, la variable \(j\) est indépendante du couple formé par la cible et les autres variables, et si les observations de validation sont indépendantes et de même loi, l'importance par permutation de \(j\) est d'espérance nulle, quel que soit le modèle, pourvu qu'il ait été entraîné sur d'autres données.
3. (I3) Si deux variables portent la même information, la permutation de l'une laisse l'autre intacte : (I2) ne s'applique pas, et l'importance de chacune est diluée.

### La preuve

1. (I1) Sachant les valeurs des deux variables, les étiquettes restent indépendantes et de même loi : le gain d'un partage fixé de tailles \(k\) et \(m - k\) (\(m\) observations dans le nœud) a une loi qui ne dépend que de \(k\), pas des indices placés de chaque côté.
2. La variable binaire offre un seul partage, de tailles \(k\) et \(m - k\), où \(k\) est le nombre d'observations où elle vaut 0 (si \(k = 0\) ou \(k = m\), aucun partage, gain nul).
3. La variable continue offre les partages « \(k'\) plus petites contre les autres » pour tout \(k' = 1, \dots, m - 1\), en particulier un partage de mêmes tailles \(k\) et \(m - k\). Son meilleur gain est au moins le gain de ce partage, qui a, sachant les valeurs, la même loi que le gain de la variable binaire. On prend l'espérance sachant les valeurs, puis l'espérance.
4. (I2) Notons \(\mathbf{C}\) le vecteur des valeurs de validation de la variable \(j\) et \(\mathbf{D}\) le reste des données de validation (cible et autres variables). Les lignes étant indépendantes et de même loi, et la variable \(j\) indépendante du reste dans chaque ligne, \(\mathbf{C}\) est indépendant de \(\mathbf{D}\) et ses coordonnées sont indépendantes et de même loi. Permuter les coordonnées d'un tel vecteur, avec une permutation tirée indépendamment, ne change pas sa loi ni son indépendance avec \(\mathbf{D}\) : les données permutées ont la même loi que les données d'origine. Le score après permutation a donc la même loi que le score avant, et leur différence est d'espérance nulle (le score, par exemple l'exactitude, est borné, donc intégrable).

### Exemple chiffré

Hypothèse nulle, nœud de @F_m@ observations, étiquettes indépendantes de loi de Bernoulli de paramètre 1/2, @F_R@ tirages (script) : le meilleur gain de Gini vaut en moyenne @F_bin_moy:4@ pour une variable binaire (valeur exacte @F_bin_exact:4@) et @F_cont_moy:4@ pour une variable continue, soit @F_rapport:1@ fois plus ; la variable continue l'emporte dans une proportion @F_prop:3@ des tirages.

Expérience : une variable binaire informative (la cible lui est égale avec probabilité 0,75), une variable continue de pur bruit, une variable binaire de pur bruit ; 1000 observations d'entraînement, 3000 de validation. Colonnes dans cet ordre : informative, bruit continu, bruit binaire.

```python-exec
from sklearn.inspection import permutation_importance

rng = np.random.default_rng(3)
n = 4000
info = rng.integers(0, 2, n)                                  # binaire, informative
cible = np.where(rng.random(n) < 0.75, info, 1 - info)        # égale à info avec probabilité 0,75
bruit_continu = rng.normal(size=n)                            # bruit, valeurs toutes distinctes
bruit_binaire = rng.integers(0, 2, n)                         # bruit, deux valeurs
Xi = np.column_stack([info, bruit_continu, bruit_binaire])
ent, val = slice(0, 1000), slice(1000, n)
for feuille_min in [1, 50]:
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=feuille_min, random_state=0).fit(Xi[ent], cible[ent])
    perm = permutation_importance(rf, Xi[val], cible[val], n_repeats=5, random_state=0)
    print(f"min_samples_leaf = {feuille_min}: impureté {rf.feature_importances_.round(3)}, permutation {perm.importances_mean.round(3)}")

Xd = np.column_stack([info, info, bruit_continu])             # la variable informative en double
rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xd[ent], cible[ent])
perm = permutation_importance(rf, Xd[val], cible[val], n_repeats=5, random_state=0)
print("informative en double : permutation", perm.importances_mean.round(3))
```

Lecture : avec des feuilles d'une observation, l'importance par impureté place le bruit continu **devant** la variable informative ; la permutation, sur la validation, donne au bruit une importance proche de 0 (fluctuation autour de son espérance nulle, d'après (I2)), et une importance nette à la variable informative. Imposer 50 observations par feuille réduit le biais de l'importance par impureté sans l'annuler. Dernière ligne : dédoublée, la variable informative voit son importance par permutation partagée entre ses deux copies (I3).

### Cas limites et pièges

1. L'importance par impureté est mesurée sur l'entraînement : elle récompense l'ajustement au bruit, d'autant plus que la variable offre de coupures.
2. Une importance par permutation peut être légèrement négative sur un jeu de validation fini : c'est une fluctuation, pas un effet.
3. Variables liées entre elles : la permutation crée aussi des combinaisons de valeurs qui n'existent pas dans les données, et l'importance se dilue entre variables redondantes.
4. Ajouter des arbres ne corrige pas le biais de l'importance par impureté : il est présent dans chaque arbre, et la moyenne conserve l'espérance.

[[controle:importance]]

## Réflexes en 30 secondes

| Indice dans l'énoncé | Premier geste |
|---|---|
| gain d'une coupure, Gini, entropie | impureté du parent moins la moyenne **pondérée par les tailles** des impuretés des enfants ; Gini à deux classes \(2\hat p(1 - \hat p)\) |
| bootstrap, hors sac, observations distinctes | \((1 - 1/n)^n\), au plus \(1/e \approx @B_inv_e:3m@\) ; distinctes : \(n\,(1 - (1 - 1/n)^n)\) |
| variance d'une moyenne de \(B\) arbres | \(\rho\,\sigma^2 + (1 - \rho)\,\sigma^2/B\) ; plancher \(\rho\,\sigma^2\) |
| ajouter des arbres | forêt : l'erreur quadratique espérée ne croît pas ; boosting : peut sur-apprendre, arrêt précoce |
| pseudo-résidu | quadratique \(y - f\) ; logistique \(y - \pi\) |
| poids de feuille XGBoost | \(g\), \(h\) d'abord ; \(-G/(H + \lambda)\) ; logistique \(h = \pi(1 - \pi)\), quadratique \(h = 1\) |
| gain XGBoost | \(\tfrac12[\,G_L^2/(H_L + \lambda) + G_R^2/(H_R + \lambda) - G^2/(H + \lambda)\,] - \gamma\) |
| `min_child_weight` | somme des \(h_i\) de l'enfant, pas un nombre d'observations |
| importance d'une variable | impureté : biaisée vers les variables à nombreuses valeurs ; permutation sur validation |

Contrôles de vraisemblance en 5 secondes :

- un gain de Gini ou d'entropie est positif ou nul ; un gain XGBoost avec \(\lambda > 0\) peut être négatif ;
- la variance d'une moyenne d'arbres est entre \(\rho\,\sigma^2\) et \(\sigma^2\) (pour \(\rho \ge 0\)) ;
- si les pseudo-résidus d'une feuille ont une somme positive, son poids est positif ;
- pour \(H > 0\), un poids XGBoost est, en valeur absolue, au plus égal à celui sans régularisation (\(\lvert G \rvert/(H + \lambda) \le \lvert G \rvert/H\)).

## À savoir refaire sans notes

- (R1), (R2) et le Gini comme somme de variances d'indicatrices ; (R5), l'arbre sans limite qui atteint une erreur d'entraînement nulle.
- \((1 - 1/n)^n \le 1/e\) et sa limite ; la variance \(\rho\,\sigma^2 + (1 - \rho)\,\sigma^2/B\) par la double somme de covariances ; la décomposition de l'erreur quadratique espérée de la forêt.
- Les pseudo-résidus quadratique et logistique, avec la dérivée \(\pi(1 - \pi)\) ; la décroissance de l'erreur d'entraînement du boosting quadratique.
- Le poids \(-G/(H + \lambda)\), la valeur optimale, le gain avec \(\gamma\), et le gain positif pour \(\lambda = 0\) par la mise au dénominateur commun.
- La comparaison en cinq phrases de la partie 6.
""",
    "controles": {
        "profondeur": {
            "question": r"On entraîne un arbre de régression sans limite de profondeur, en coupant tant qu'il existe une coupure de gain strictement positif, sur 500 observations. La première variable prend 500 valeurs deux à deux distinctes ; la cible contient un bruit de variance 1. Que vaut l'erreur quadratique moyenne de l'arbre sur ses 500 observations d'entraînement ?",
            "options": [
                {"id": "a", "texte": "0, quel que soit le bruit.",
                 "explication": r"Oui : par (R5), toute feuille où \(y\) n'est pas constant peut encore être coupée avec un gain positif ; la croissance s'arrête sur des feuilles de \(y\) constant, prédit exactement. Ce zéro mesure l'ajustement au bruit, pas la qualité du modèle."},
                {"id": "b", "texte": "Environ 1, la variance du bruit, qu'aucun modèle ne peut prédire.",
                 "explication": r"C'est l'erreur minimale, en moyenne, sur de **nouvelles** observations. Sur l'entraînement, l'arbre apprend le bruit lui-même, valeur par valeur."},
                {"id": "c", "texte": "On ne peut pas le savoir : un algorithme glouton ne garantit pas la meilleure partition.",
                 "explication": r"L'algorithme glouton ne trouve pas forcément le meilleur arbre, mais il atteint des feuilles pures : (R5) garantit une coupure de gain strictement positif tant qu'une feuille est impure."},
                {"id": "d", "texte": "Strictement positive, car certaines feuilles finales contiennent plusieurs observations.",
                 "explication": r"Une feuille finale de plusieurs observations a forcément un \(y\) constant, sinon (R5) permettrait encore de la couper : sa moyenne prédit chacune exactement."},
            ],
        },
        "bootstrap": {
            "question": r"On tire un échantillon bootstrap à partir de \(n = 3\) observations (3 tirages uniformes, indépendants, avec remise). Quelle est la probabilité qu'une observation donnée soit hors sac ?",
            "options": [
                {"id": "a", "texte": r"\(\tfrac{@Cb_abs_num@}{@Cb_abs_den@}\)",
                 "explication": r"Oui : chaque tirage manque l'observation avec probabilité \(2/3\), et les 3 tirages sont indépendants : \((2/3)^3\)."},
                {"id": "b", "texte": r"\(\tfrac{@Cb_d_present_num@}{@Cb_d_present_den@}\)",
                 "explication": r"C'est la probabilité que l'observation soit tirée au moins une fois, le complémentaire de la réponse."},
                {"id": "c", "texte": r"\(\tfrac{@Cb_d_cube_num@}{@Cb_d_cube_den@}\)",
                 "explication": r"C'est \((1/3)^3\), la probabilité que l'observation soit tirée aux trois tirages, pas qu'elle soit absente."},
                {"id": "d", "texte": r"\(1/e \approx @B_inv_e:3m@\)",
                 "explication": r"C'est la limite quand \(n \to \infty\). Pour tout \(n\) fini, \((1 - 1/n)^n\) est strictement plus petit que \(1/e\) (partie 2, preuve, étape 2)."},
            ],
        },
        "plancher": {
            "question": r"Des arbres ont une variance \(\sigma^2 = 1\) et une corrélation \(\rho = 0{,}5\) deux à deux. Combien en faut-il au minimum pour que la variance de leur moyenne dépasse son plancher \(\rho\,\sigma^2\) d'au plus 0,01 ?",
            "options": [
                {"id": "a", "texte": "@Cp_B_min@",
                 "explication": r"Oui : l'excès est \((1 - \rho)\,\sigma^2/B = 0{,}5/B\), au plus 0,01 si et seulement si \(B \ge @Cp_B_min@\)."},
                {"id": "b", "texte": "@Cp_d_rho@",
                 "explication": r"C'est le résultat de \(\rho\,\sigma^2/B \le 0{,}01\) : la corrélation a été placée dans le terme en \(1/B\). Le terme qui décroît est \((1 - \rho)\,\sigma^2/B\)."},
                {"id": "c", "texte": "@Cp_d_inv@",
                 "explication": r"C'est le résultat de \(\sigma^2/B \le 0{,}01\), qui oublie le facteur \(1 - \rho\) : seule la part non corrélée de la variance est divisée par \(B\)."},
                {"id": "d", "texte": "Aucun nombre fini : la variance n'atteint jamais son plancher.",
                 "explication": r"Elle ne l'atteint jamais, mais la question demande de s'en approcher à 0,01 près, ce qui arrive dès que \(0{,}5/B \le 0{,}01\)."},
            ],
        },
        "residus": {
            "question": r"Boosting de gradient avec la perte \(\tfrac12 (y - f)^2\), sur trois observations de cibles \(y = (1, 2, 6)\). Le modèle initial \(f_0\) est la constante qui minimise la perte totale. Quelles cibles reçoit le premier arbre ?",
            "options": [
                {"id": "a", "texte": "@Cr_res@",
                 "explication": r"Oui : \(f_0\) est la moyenne, @Cr_moy@, et le pseudo-résidu de la perte quadratique est le résidu \(y_i - f_0\)."},
                {"id": "b", "texte": "@Cr_d_signe@",
                 "explication": r"C'est le gradient \(f_0 - y_i\), pas son opposé : l'arbre doit pointer dans la direction qui fait **baisser** la perte."},
                {"id": "c", "texte": "(1 ; 2 ; 6)",
                 "explication": r"Ce serait le cas si \(f_0 = 0\). Ici \(f_0\) est la constante optimale, la moyenne des cibles, qu'il faut retrancher."},
                {"id": "d", "texte": "@Cr_d_med@",
                 "explication": r"Ce sont les écarts à la médiane, qui est la constante optimale pour la perte absolue \(\lvert y - f \rvert\). Pour la perte quadratique, c'est la moyenne (R1)."},
            ],
        },
        "newton": {
            "question": r"Pour quelle perte le poids XGBoost \(-G/(H + \lambda)\) est-il le minimum exact de la perte pénalisée de la feuille, et non une approximation ?",
            "options": [
                {"id": "a", "texte": "La perte quadratique : son développement de Taylor à l'ordre 2 est exact.",
                 "explication": r"Oui : sa dérivée troisième est nulle, le reste de Taylor aussi, et le critère approché est la vraie perte pénalisée (X4)."},
                {"id": "b", "texte": r"La perte logistique, car \(h_i = \pi_i(1 - \pi_i)\) se calcule exactement.",
                 "explication": r"Calculer \(h_i\) exactement ne rend pas le développement exact : la perte logistique n'est pas un polynôme de degré 2 en \(w\). Le poids est un pas de Newton ; sur l'exemple de la leçon, @E_wR@ contre un minimum exact de @E_w_exact:4@."},
                {"id": "c", "texte": "Toute perte deux fois dérivable, puisque le minimum d'un polynôme du second degré se calcule exactement.",
                 "explication": r"Le minimum du polynôme est exact, mais le polynôme n'est qu'une approximation de la perte, sauf quand la perte est elle-même de degré 2 en \(w\)."},
                {"id": "d", "texte": r"Aucune : la pénalité \(\tfrac12 \lambda w^2\) empêche d'atteindre le minimum.",
                 "explication": r"La pénalité fait partie de la fonction minimisée ; elle n'introduit aucune approximation. Pour la perte quadratique, \(-G/(H + \lambda)\) minimise exactement perte plus pénalité."},
            ],
        },
        "importance": {
            "question": "Laquelle de ces mesures est calculée sur les données d'entraînement de chaque arbre, et peut donc récompenser l'ajustement au bruit ?",
            "options": [
                {"id": "a", "texte": "L'importance par impureté (`feature_importances_` de scikit-learn).",
                 "explication": r"Oui : elle additionne les gains des coupures, mesurés sur les données qui ont servi à les choisir ; tout gain est positif ou nul, même pour une variable de pur bruit (I1)."},
                {"id": "b", "texte": "L'importance par permutation sur un jeu de validation.",
                 "explication": r"Elle mesure la dégradation du score sur des données que le modèle n'a pas vues ; pour une variable indépendante de tout le reste, son espérance est nulle (I2)."},
                {"id": "c", "texte": "L'erreur hors sac de la forêt.",
                 "explication": r"Elle est calculée, pour chaque observation, par les seuls arbres qui ne l'ont pas vue : c'est une mesure hors échantillon, et ce n'est pas une importance de variable."},
                {"id": "d", "texte": "Les deux importances, puisqu'elles sont calculées sur le même modèle.",
                 "explication": r"Le modèle est le même, mais les données d'évaluation diffèrent : la permutation se fait sur la validation, et c'est ce qui la protège du biais d'ajustement."},
            ],
        },
    },
    "verif": {"script": "verif/seance_ml_2.py"},
}
