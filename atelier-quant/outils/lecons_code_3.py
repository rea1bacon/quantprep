"""Leçon de la séance de code 3 : une chaîne complète de classification avec scikit-learn, du CSV sale au fichier
de prédictions.

Format : voir outils/lecons.py. Les blocs python-exec sont exécutés à la construction par le noyau du notebook
(.venv-hackerrank : Python 3.9, pandas 1.4.2, scikit-learn 1.1.1), dans coding/seance3, et leur vraie sortie est
insérée. Les fichiers abonnes_train.csv et abonnes_test.csv (données synthétiques) sont écrits par
verif/seance_code_3.py, qui vérifie aussi chaque fait d'interface cité ici, dans cette version et dans celle de la
machine. Aucun nombre calculé n'est écrit à la main : ils sont tous dans les sorties des blocs exécutés.
"""

LECON_SKLEARN = {
    "id": "L-SKLEARN-1", "type": "lecon", "section": "lecon",
    "theme": "Leçon scikit-learn : du CSV sale au fichier de prédictions",
    "difficulte": 3, "temps_cible_s": 1800, "dossier": "coding/seance3",
    "contenu": r"""## 1. La tâche et la règle unique

Durée visée : 30 minutes. Lis dans l'ordre, exécute les exemples dans le notebook de droite si tu veux les modifier, et réponds à chaque point de contrôle avant de continuer. Les sorties affichées sous chaque exemple sont les vraies sorties de Python 3.9, pandas 1.4.2 et scikit-learn 1.1.1, les versions du notebook.

Conventions de cette leçon : une minuscule désigne une valeur observée, donc fixe. Exception d'usage déclarée : `X` (le tableau des variables explicatives) et `y` (la colonne cible) sont les noms consacrés par scikit-learn ; ce `X` majuscule est un tableau de valeurs observées, pas une variable aléatoire.

### Le problème

Rien n'y est difficile mathématiquement. Ce qui coûte les points, c'est le temps perdu à chercher une fonction, et les fautes silencieuses : une ligne de test supprimée, une colonne mal alignée, une estimation de l'exactitude trop optimiste.

Cette leçon donne une chaîne complète, à savoir écrire sans documentation en 25 à 30 minutes.

### L'intuition

Il y a deux jeux de lignes. Sur le jeu d'**apprentissage**, on connaît la cible : on peut y **ajuster** des choses (verbe `fit` : calculer une médiane, une moyenne, une liste de modalités, des coefficients). Sur le jeu de **test**, on ne connaît pas la cible : on ne peut qu'**appliquer** ce qui a été ajusté (verbes `transform` et `predict`). Pour savoir ce que vaudra le modèle sur le test, on rejoue cette situation à l'intérieur de l'apprentissage : on met des lignes de côté, on fait comme si leur cible était inconnue, puis on compare.

### L'énoncé de la règle

**Règle unique.** Toute quantité calculée à partir des données et utilisée ensuite pour transformer ou pour prédire (valeur de remplacement d'un manquant, moyenne et écart-type d'une mise à l'échelle, liste des modalités d'un encodage, variables retenues, coefficients du modèle) est calculée sur des lignes d'apprentissage seulement. Pendant une validation, « lignes d'apprentissage » veut dire : les lignes qui ne sont pas mises de côté à ce moment-là.

Une violation de cette règle s'appelle une **fuite** : de l'information des lignes mises de côté, ou du test, entre dans ce qui est ajusté, et l'exactitude estimée n'est plus celle d'un modèle placé devant des lignes nouvelles.

Le nettoyage ligne par ligne (retirer une unité, convertir un mot en nombre) n'apprend rien des autres lignes : il s'applique à l'identique aux deux jeux, par une même fonction, et ne crée aucune fuite.

L'ordre des étapes, qui est aussi le plan de la leçon :

1. Lire les fichiers et les inspecter (partie 2).
2. Nettoyer chaque colonne sale, par une fonction appliquée aux deux jeux (partie 3).
3. Traiter les valeurs manquantes, sans jamais retirer de ligne du test (partie 4).
4. Encoder les catégories (partie 5).
5. Mettre à l'échelle si le modèle en a besoin, et tout ranger dans une chaîne (partie 6).
6. Mettre des lignes de côté, fixer un point de comparaison, mesurer (partie 7).
7. Comparer deux ou trois modèles (partie 8).
8. Réajuster sur tout l'apprentissage, prédire le test, écrire le fichier, le relire (partie 9).

## 2. Lire et inspecter

### Le problème

Les saletés ne sont pas annoncées, ou le sont à moitié. Trois minutes d'inspection disent quelles colonnes sont sales, où sont les manquants, et si les classes sont équilibrées. Sans cette inspection, le premier `fit` échoue sur un message obscur, ou réussit sur des données fausses.

### Exemple

Les fichiers de la leçon décrivent des abonnés d'un service (données synthétiques) ; la cible `resilie` vaut 1 si l'abonné a résilié.

```python-exec
import numpy as np
import pandas as pd

train = pd.read_csv("abonnes_train.csv")
test = pd.read_csv("abonnes_test.csv")
print(train.shape, test.shape)
train.head(8)
```

Premier geste : les types. Une colonne `object` contient des chaînes ; si elle devrait être numérique, elle est sale.

```python-exec
train.dtypes
```

`age`, `revenu` et `appels` sont en `object` alors que ce sont des nombres : à nettoyer. Deuxième geste : les valeurs manquantes déclarées, dans les deux jeux.

```python-exec
print(train.isna().sum())
print(test.isna().sum())
```

Troisième geste : la cible, les modalités des colonnes de texte, et les bornes des colonnes numériques.

```python-exec
print(train["resilie"].value_counts(normalize=True))
print(train["forfait"].unique())
print(train["appels"].unique())
print(train["age"].unique()[:12])
train.describe()
```

Ce que cette sortie apprend : les deux classes ont des parts voisines ; `forfait` a trois modalités écrites de plusieurs façons ; `appels` mélange chiffres et mots ; `age` porte parfois une unité ou le mot `inconnu` ; et le minimum de `anciennete_mois` est négatif, ce qui est impossible pour une durée.

### Cas limites

- `isna()` ne compte que les manquants **déclarés** (cellule vide du CSV). Le mot `inconnu` dans `age` et la valeur négative dans `anciennete_mois` sont des manquants **déguisés** : ils n'apparaissent dans ce décompte qu'après le nettoyage.
- `describe()` ne résume par défaut que les colonnes numériques d'un cadre qui en contient : les colonnes sales en `object` n'y figurent pas. Leur absence du tableau est elle-même un signal.
- Un champ qui contient une virgule est écrit entre guillemets dans un CSV, et `read_csv` le relit correctement comme une seule valeur.

### Pièges

1. Ne regarder que `head()` : les cinq premières lignes peuvent être propres.
2. Inspecter l'apprentissage et pas le test : une saleté ou une modalité peut n'exister que dans le test.
3. Oublier l'équilibre des classes : il fixe le point de comparaison de la partie 7.

## 3. Convertir une colonne numérique sale

### Le problème

Une colonne numérique lue en `object` contient des chaînes : nombres avec unité, séparateur de milliers, virgule décimale, nombres en lettres, texte libre. Il faut en tirer une colonne de flottants, en sachant exactement quelles valeurs n'ont pas pu être converties.

### L'intuition

Deux temps : d'abord ramener le texte à une écriture de nombre (retirer ce qui est en trop), ensuite convertir en demandant que tout ce qui reste illisible devienne manquant. Puis **compter** les manquants avant et après : la différence est le nombre de valeurs que le nettoyage n'a pas comprises, et il faut aller les voir.

### L'énoncé

`pd.to_numeric(s, errors="coerce")` rend une série numérique de même index que `s` : chaque valeur convertible en nombre est convertie, toute autre devient `NaN`. Sans `errors="coerce"`, la valeur par défaut `errors="raise"` lève `ValueError` à la première valeur illisible.

```python-exec
s = pd.Series(["12", "1 200", "3,5", "douze", None])
pd.to_numeric(s, errors="coerce")
```

```python-exec-erreur
pd.to_numeric(s)
```

Lecture : seul `"12"` est une écriture de nombre pour pandas. L'espace des milliers, la virgule décimale et le mot donnent tous `NaN` : `coerce` ne nettoie rien, il rend seulement l'échec visible et comptable.

### Exemple : séparateurs, unité, virgule décimale

```python-exec
revenu = (train["revenu"].str.replace(" ", "", regex=False)
          .str.replace("€", "", regex=False)
          .str.replace(",", ".", regex=False))
revenu_num = pd.to_numeric(revenu, errors="coerce")
print("manquants avant :", train["revenu"].isna().sum(), "; après :", revenu_num.isna().sum())
pd.DataFrame({"brut": train["revenu"], "propre": revenu_num}).head(8)
```

Le nombre de manquants est le même avant et après : toutes les valeurs présentes ont été comprises. `regex=False` demande un remplacement de texte littéral ; c'est à écrire explicitement, car un point ou une parenthèse ont un sens spécial dans une expression régulière.

### Exemple : extraire par expression régulière

`s.str.extract(motif)` rend un DataFrame avec une colonne par groupe entre parenthèses du motif, et `NaN` sur les lignes où le motif ne correspond pas. Le motif ci-dessous se lit de gauche à droite : l'accent circonflexe ancre le début de la chaîne ; le groupe entre parenthèses capture une suite de chiffres ; la parenthèse qui commence par `?:` accepte, sans le capturer, le suffixe facultatif ` ans` ; l'antislash suivi d'un Z majuscule ancre la fin de la chaîne.

```python-exec
age = train["age"].str.extract(r"^(\d+)(?: ans)?\Z")[0]
print("valeurs non comprises :", train["age"][age.isna()].unique())
age_num = pd.to_numeric(age)
age_num.describe()[["count", "min", "max"]]
```

La seule valeur non comprise est `inconnu` : c'est bien un manquant. L'ancre de fin s'écrit plus couramment avec le symbole dollar ; celle utilisée ici désigne strictement la fin de la chaîne et a le même effet sur ces données.

### Exemple : nombres en lettres

Quand les valeurs possibles sont peu nombreuses, un dictionnaire suffit. `map` rend `NaN` pour toute clé absente du dictionnaire : on complète donc par la conversion numérique des autres valeurs.

```python-exec
MOTS = {"zéro": 0, "un": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5}
appels_num = train["appels"].map(MOTS).fillna(pd.to_numeric(train["appels"], errors="coerce"))
print("manquants :", appels_num.isna().sum(), "; valeurs :", sorted(appels_num.unique()))
```

Pour des nombres composés reliés par un trait d'union, la valeur est la somme des mots :

```python-exec
VALEURS = {"vingt": 20, "trente": 30, "deux": 2, "huit": 8}
pd.Series(["trente-deux", "vingt", "vingt-huit"]).map(lambda t: sum(VALEURS[m] for m in t.split("-")))
```

Ici un mot absent de `VALEURS` lève `KeyError` : c'est voulu, une erreur visible vaut mieux qu'une valeur fausse.

### Exemple : deux informations dans une colonne

Le motif ci-dessous se lit : n'importe quelle suite de caractères (premier groupe), puis un séparateur parmi les trois placés entre crochets, puis un chiffre final (second groupe). Comme `.*` prend le plus de caractères possible, c'est le **dernier** séparateur qui coupe.

```python-exec
c = pd.Series(["32.0;2", "trente-deux_3", "28.0,1"])
c.str.extract(r"^(.*)[;,_](\d)\Z")
```

### Exemple : valeurs impossibles et modalités mal écrites

Une valeur impossible (durée négative, hauteur négative) n'est pas une information : on la rend manquante. `s.where(condition)` garde les valeurs où la condition est vraie et met `NaN` ailleurs.

```python-exec
anciennete = train["anciennete_mois"].where(train["anciennete_mois"] >= 0)
print("négatives :", (train["anciennete_mois"] < 0).sum(), "; manquantes avant :", train["anciennete_mois"].isna().sum(), "; après :", anciennete.isna().sum())
print(train["forfait"].str.strip().str.lower().value_counts())
```

### Exemple : la fonction de nettoyage, appliquée aux deux jeux

```python-exec
def nettoyer(df):
    d = df.copy()
    d["age"] = pd.to_numeric(d["age"].astype(str).str.strip().str.extract(r"^(\d+)(?: ans)?\Z")[0], errors="coerce")
    revenu = d["revenu"].astype(str).str.replace(" ", "", regex=False).str.replace("€", "", regex=False).str.replace(",", ".", regex=False)
    d["revenu"] = pd.to_numeric(revenu, errors="coerce")
    appels = d["appels"].astype(str).str.strip()
    d["appels"] = appels.map(MOTS).fillna(pd.to_numeric(appels, errors="coerce"))
    d["anciennete_mois"] = d["anciennete_mois"].where(d["anciennete_mois"] >= 0)
    d["forfait"] = d["forfait"].str.strip().str.lower()
    return d

a, b = nettoyer(train), nettoyer(test)
print(a.dtypes)
print(a.isna().sum())
```

Les quatre colonnes numériques sont en `float64`, et les manquants déguisés sont maintenant comptés.

### Cas limites

L'accesseur `.str` n'existe que sur une colonne de chaînes. Si une colonne « sale » ne l'est pas dans l'un des fichiers (par exemple `appels` tout en chiffres dans le test), `read_csv` la lit comme numérique et `.str` échoue :

```python-exec-erreur
pd.Series([3, 0, 5]).str.strip()
```

D'où `astype(str)` en tête de chaque traitement de texte dans `nettoyer` : un nombre devient son écriture, et une valeur manquante devient le texte `nan`, que `to_numeric` reconvertit en `NaN`.

[[controle:coerce]]

### Pièges

1. Un motif sans ancres : `str.extract` cherche le motif **n'importe où** dans la chaîne, et tire donc un nombre d'une valeur qui devrait être rejetée.
2. Retirer la virgule décimale au lieu de la remplacer par un point : la valeur est multipliée par une puissance de dix, sans erreur.
3. Nettoyer l'apprentissage dans des cellules éparses, puis oublier une étape sur le test. Une seule fonction, deux appels.
4. Prendre la valeur absolue d'une valeur négative : rien dans les données ne dit que seul le signe est faux. Sans indication de l'énoncé, la valeur est manquante.

## 4. Valeurs manquantes

### Le problème

La régression logistique et la forêt aléatoire de scikit-learn 1.1.1 refusent une entrée qui contient un `NaN` (`ValueError`). Il faut donc soit retirer des lignes, soit remplacer les valeurs manquantes.

### L'intuition

Retirer une ligne d'apprentissage coûte un exemple ; c'est acceptable s'il en reste beaucoup. Retirer une ligne de **test** est toujours une faute : le fichier de prédictions doit contenir une ligne par identifiant du test, et une ligne absente est une prédiction manquante, donc un fichier invalide ou une réponse fausse. Les valeurs manquantes du test se **remplacent**.

### L'énoncé

**Suppression.** `a.dropna()` rend les lignes sans aucune valeur manquante ; `a.dropna(subset=[...])` ne regarde que les colonnes citées. À faire **après** le nettoyage, et sur l'apprentissage seulement.

**Imputation.** `SimpleImputer(strategy="median")` calcule, à l'appel de `fit`, la médiane de chaque colonne sur les lignes qu'on lui donne (attribut `statistics_`) ; `transform` remplace ensuite chaque `NaN` par la médiane de sa colonne. La stratégie par défaut est `"mean"`. D'après la règle unique, `fit` ne reçoit que des lignes d'apprentissage, et le test est transformé avec ces mêmes médianes.

### Exemple

```python-exec
from sklearn.impute import SimpleImputer

NUM = ["age", "revenu", "appels", "anciennete_mois"]
print(len(a), "lignes d'apprentissage, dont", len(a.dropna()), "sans aucune valeur manquante")
print(len(b), "lignes de test, dont", len(b.dropna()), "sans aucune valeur manquante")
imputer = SimpleImputer(strategy="median").fit(a[NUM])
print(dict(zip(NUM, imputer.statistics_)))
print("manquants du test après transformation :", int(np.isnan(imputer.transform(b[NUM])).sum()))
```

Un `dropna` sur le test aurait fait disparaître la différence entre les deux nombres de la deuxième ligne : autant de prédictions absentes du fichier.

### Cas limites

- **Moyenne ou médiane.** La médiane ne dépend pas des valeurs extrêmes ; c'est le choix prudent quand on n'a pas le temps de regarder la distribution.
- **Modèle qui accepte les manquants.** `HistGradientBoostingClassifier` accepte les `NaN` tels quels (vérifié) ; d'après sa documentation, il apprend à chaque coupure de quel côté envoyer les lignes dont la valeur manque. L'imputation reste nécessaire pour les autres modèles.
- **Colonne catégorielle.** `SimpleImputer(strategy="most_frequent")` remplace par la modalité la plus fréquente ; on peut aussi créer une modalité `manquant` avec `fillna("manquant")`.
- **L'énoncé impose la suppression.** Si la consigne demande de retirer les lignes incomplètes, elle vise l'apprentissage ; le test garde toutes ses lignes, et il faut quand même prévoir d'y remplacer les manquants.

[[controle:test]]

### Pièges

1. `dropna()` avant le nettoyage : les manquants déguisés (`inconnu`, valeur négative) ne sont pas encore des `NaN`, les lignes restent.
2. `fillna(0)` par réflexe : zéro est une valeur comme une autre, ici un âge ou un revenu nul, très loin des valeurs observées.
3. Calculer la médiane sur l'apprentissage et le test réunis : c'est contraire à la règle unique, et impossible à reproduire le jour où une seule ligne nouvelle arrive.

## 5. Encoder les catégories

### Le problème

Un modèle de scikit-learn calcule sur des nombres. Une colonne de texte fait échouer l'ajustement :

```python-exec-erreur
from sklearn.linear_model import LogisticRegression

LogisticRegression().fit(a[["forfait"]], a["resilie"])
```

### L'intuition

Une catégorie sans ordre (un forfait, un matériau, un numéro de ligne de production) n'est pas une quantité : la remplacer par 0, 1, 2 dirait au modèle que la modalité 2 est « deux fois » la modalité 1. On la remplace par autant de colonnes qu'il y a de modalités, chacune valant 1 si la ligne a cette modalité et 0 sinon : c'est l'encodage **un-parmi-n** (one-hot). Une catégorie **ordonnée** (bas, moyen, haut) peut être remplacée par son rang : c'est l'encodage **ordinal**.

### L'énoncé

`OneHotEncoder` apprend à l'appel de `fit` la liste triée des modalités de chaque colonne (attribut `categories_`), puis `transform` produit une colonne par modalité apprise. Trois faits à connaître, vérifiés sous scikit-learn 1.1.1 :

- par défaut `handle_unknown="error"` : une modalité absente de l'apprentissage fait échouer `transform` ; avec `handle_unknown="ignore"`, elle donne une ligne de zéros ;
- la sortie est une matrice creuse ; `.toarray()` la rend dense ;
- pour obtenir directement une sortie dense, le paramètre s'appelle `sparse=False` dans cette version. Le nom `sparse_output`, qu'utilisent les versions plus récentes, y lève `TypeError`.

### Exemple

```python-exec
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown="ignore").fit(a[["forfait"]])
print(enc.categories_)
petit = pd.DataFrame({"forfait": ["plus", "pro", "base"]})
enc.transform(petit).toarray()
```

La modalité `pro` n'existe pas dans l'apprentissage : sa ligne est nulle, le modèle la traite comme « aucun des forfaits connus ». Avec le réglage par défaut :

```python-exec-erreur
OneHotEncoder().fit(a[["forfait"]]).transform(petit)
```

### Exemple : le piège de get_dummies

`pd.get_dummies` crée une colonne par modalité **présente dans le cadre qu'on lui donne**. Appliqué séparément aux deux jeux, il donne des colonnes différentes dès qu'une modalité n'existe que d'un côté :

```python-exec
print(list(pd.get_dummies(a[["forfait"]]).columns))
print(list(pd.get_dummies(b[["forfait"]]).columns))
```

Le modèle ajusté sur les colonnes de la première ligne ne peut pas prédire sur celles de la seconde. La réparation : imposer au test les colonnes de l'apprentissage, dans le même ordre, en remplissant de zéros.

```python-exec
x_a = pd.get_dummies(a[["forfait"]], dtype=float)
x_b = pd.get_dummies(b[["forfait"]], dtype=float).reindex(columns=x_a.columns, fill_value=0.0)
print(list(x_b.columns), x_b.shape)
```

`reindex(columns=...)` retire la colonne de la modalité inconnue, ajouterait à zéro celle d'une modalité absente du test, et remet l'ordre. C'est l'objet de l'exercice 2.

### Exemple : encodage ordinal

```python-exec
from sklearn.preprocessing import OrdinalEncoder

niveaux = pd.DataFrame({"niveau": ["bas", "haut", "moyen", "bas"]})
print(OrdinalEncoder().fit_transform(niveaux).ravel())
print(OrdinalEncoder(categories=[["bas", "moyen", "haut"]]).fit_transform(niveaux).ravel())
```

Par défaut, les rangs suivent l'ordre alphabétique (`bas`, `haut`, `moyen`), qui n'est pas l'ordre voulu : il faut donner l'ordre par `categories`.

### Cas limites

- **Un numéro qui est une étiquette.** Une colonne d'entiers peut être une catégorie (numéro d'atelier, code postal). La citer parmi les colonnes catégorielles de l'encodeur suffit : `OneHotEncoder` accepte des entiers.
- **Modèles à base d'arbres.** Un arbre peut isoler une modalité codée par un entier en deux coupures ; l'encodage ordinal d'une catégorie sans ordre y est donc tolérable, alors qu'il fausse une régression logistique. L'encodage un-parmi-n reste correct pour tous.

[[controle:encodage]]

### Pièges

1. `get_dummies` séparément sur l'apprentissage et le test, sans réalignement.
2. `get_dummies` sur la concaténation des deux jeux : les colonnes sont alignées, mais elles dépendent du test, contre la règle unique, et une modalité propre au test crée une colonne toujours nulle à l'apprentissage.
3. Uniformiser les modalités **après** l'encodage : `Plus` et `plus` sont alors deux colonnes.

## 6. Mise à l'échelle, Pipeline et ColumnTransformer

### Le problème

Deux questions liées. Quels modèles ont besoin de variables à la même échelle ? Et comment garantir, sans y penser à chaque ligne de code, que l'imputation, l'échelle et l'encodage ne sont ajustés que sur des lignes d'apprentissage, y compris à l'intérieur d'une validation croisée ?

### L'énoncé : la standardisation

Pour une colonne dont les valeurs d'apprentissage sont \(x_1, \dots, x_n\), on note \(m\) leur moyenne et \(s\) leur écart-type (à dénominateur \(n\) dans `StandardScaler`, avec \(s > 0\)). La valeur standardisée d'une valeur \(x\), qu'elle vienne de l'apprentissage ou du test, est

\(\displaystyle z = \frac{x - m}{s}.\)

Les nombres \(m\) et \(s\) sont ajustés (`fit`) sur l'apprentissage et réutilisés tels quels sur le test.

**Qui en a besoin.** Les modèles dont le résultat dépend de l'unité des variables : régression logistique **pénalisée** (celle de scikit-learn l'est par défaut : pénalité \(\ell_2\), `C=1.0`), k plus proches voisins, SVM, réseaux de neurones, ACP. **Qui n'en a pas besoin** : arbres de décision, forêts aléatoires, boosting d'arbres.

### Preuve : un arbre ne dépend pas de l'échelle, une régression pénalisée si

1. Un arbre découpe par des règles de la forme \(x_j \le t\), où \(x_j\) est la valeur d'une variable et \(t\) un seuil.
2. Soit \(g(x) = (x - m)/s\) avec \(s > 0\) : \(g\) est strictement croissante, donc \(x_j \le t\) équivaut à \(g(x_j) \le g(t)\).
3. Toute séparation des lignes d'apprentissage réalisable par un seuil sur \(x_j\) l'est donc par un seuil sur \(g(x_j)\), avec les mêmes lignes de chaque côté, donc la même impureté. L'arbre choisit les mêmes séparations.
4. scikit-learn place le seuil au milieu de deux valeurs observées consécutives \(u < v\). Comme \(g\) est affine, \(g\bigl(\tfrac{u+v}{2}\bigr) = \tfrac{g(u)+g(v)}{2}\) : le seuil appris sur la variable transformée est l'image du seuil d'origine, et toute ligne nouvelle tombe du même côté. Les prédictions sont identiques, aux arrondis du calcul flottant près.
5. Pour une transformation croissante **non affine** (un logarithme), les points 2 et 3 restent vrais mais pas le 4 : les prédictions coïncident sur les lignes d'apprentissage et peuvent différer pour une ligne nouvelle située entre deux valeurs observées.
6. Régression logistique pénalisée : elle minimise la perte logistique plus un terme proportionnel à la somme des carrés des coefficients. Si l'on divise une colonne par \(c > 1\), il faut multiplier son coefficient par \(c\) pour garder les mêmes prédictions, ce qui multiplie sa part de la pénalité par \(c^2\). Le compromis entre perte et pénalité change, donc le minimum aussi : les prédictions dépendent de l'unité. Sans pénalité, et lorsque le minimum de la perte existe, le même raisonnement donne au contraire les mêmes prédictions.

### Expérience

Les mêmes données, en euros et en mois, puis en milliers d'euros et en années. La pénalité est renforcée (`C=0.01`) pour que l'effet soit visible.

```python-exec
from sklearn.tree import DecisionTreeClassifier

complet = a[["revenu", "anciennete_mois", "resilie"]].dropna()
x = complet[["revenu", "anciennete_mois"]].to_numpy()
cible = complet["resilie"].to_numpy()
x_k = x / [1000.0, 12.0]
for nom, modele in [("arbre", DecisionTreeClassifier(max_depth=4, random_state=0)),
                    ("régression logistique pénalisée", LogisticRegression(C=0.01, max_iter=5000))]:
    p1 = modele.fit(x, cible).predict_proba(x)[:, 1]
    p2 = modele.fit(x_k, cible).predict_proba(x_k)[:, 1]
    print(nom, ": écart maximal entre les probabilités prédites =", round(float(np.abs(p1 - p2).max()), 4))
```

### L'énoncé : Pipeline et ColumnTransformer

`Pipeline([(nom, étape), ..., (nom, modèle)])` enchaîne des transformations et un modèle final en un seul objet. Son `fit(X, y)` ajuste chaque étape sur les lignes reçues, dans l'ordre, chacune recevant la sortie de la précédente ; son `predict(X)` applique les transformations **déjà ajustées**, sans les réajuster, puis le modèle. Passé à une validation croisée, l'objet entier est réajusté sur les seules lignes d'apprentissage de chaque pli : la règle unique est respectée par construction.

`ColumnTransformer([(nom, transformation, colonnes), ...])` applique une transformation différente à chaque liste de colonnes et colle les résultats. Deux réglages par défaut à connaître : `remainder="drop"`, les colonnes citées nulle part sont **retirées** (pratique pour l'identifiant, dangereux pour une variable oubliée) ; `sparse_threshold=0.3`, la sortie est une matrice creuse quand la part de valeurs non nulles est sous ce seuil, et `sparse_threshold=0` force une sortie dense.

### Exemple

```python-exec
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CAT = ["forfait"]

def chaine(modele):
    preparation = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("echelle", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ], sparse_threshold=0)
    return Pipeline([("preparation", preparation), ("modele", modele)])

X, y = a[NUM + CAT], a["resilie"]
modele = chaine(LogisticRegression(max_iter=1000)).fit(X, y)
print(modele.named_steps["preparation"].get_feature_names_out())
print(modele.predict(b[NUM + CAT])[:10])
```

Une seule ligne ajuste l'imputation, l'échelle, l'encodage et le modèle sur l'apprentissage ; une seule ligne applique le tout au test, modalité inconnue comprise.

### Expérience : une fuite qui se voit

Cent lignes, deux mille variables tirées au hasard, une cible tirée à pile ou face : rien n'est prévisible, et tout modèle a une exactitude attendue de 1/2 sur des lignes nouvelles. On garde les vingt variables les plus liées à la cible, de deux façons.

```python-exec
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)
bruit = rng.standard_normal((100, 2000))
pile_ou_face = rng.integers(0, 2, 100)

choisies = SelectKBest(f_classif, k=20).fit_transform(bruit, pile_ou_face)      # sélection sur TOUTES les lignes
avec_fuite = cross_val_score(LogisticRegression(max_iter=1000), choisies, pile_ou_face, cv=5).mean()
dans_la_chaine = Pipeline([("selection", SelectKBest(f_classif, k=20)), ("modele", LogisticRegression(max_iter=1000))])
sans_fuite = cross_val_score(dans_la_chaine, bruit, pile_ou_face, cv=5).mean()
print("sélection avant la validation croisée :", round(avec_fuite, 3))
print("sélection dans la chaîne :", round(sans_fuite, 3))
```

La première estimation est très au-dessus de 1/2 alors qu'il n'y a rien à apprendre : les vingt variables ont été choisies en regardant aussi la cible des lignes de validation, parmi deux mille candidates dont certaines lui ressemblent par hasard. Dans la seconde, la sélection est refaite à chaque pli sans ces lignes.

Repère pour lire ces deux nombres : pour un modèle fixé dont l'exactitude vraie est 1/2, l'exactitude mesurée sur cent lignes indépendantes a pour écart-type \(\sqrt{0{,}25/100} = 0{,}05\). La seconde estimation est à moins de deux écarts-types de 1/2, ce qui est compatible avec le hasard ; la première en est à plus de cinq.

À retenir. Constaté ci-dessus : une étape qui **utilise la cible** (ici une sélection de variables ; de même un encodage d'une catégorie par la moyenne de la cible) peut fabriquer une exactitude de toutes pièces. Argument, non mesuré ici : une étape qui n'utilise pas la cible (médiane, moyenne, écart-type) fuit beaucoup moins, parce que ces statistiques changent peu quand on leur retire un cinquième de lignes tirées au hasard parmi des centaines. La chaîne supprime les deux cas, sans avoir à juger lequel est grave.

[[controle:fuite]]

### Pièges

1. `fit_transform` sur le test : c'est réajuster l'échelle sur le test. Sur le test, seulement `transform`, ou mieux `predict` de la chaîne.
2. `HistGradientBoostingClassifier` refuse une matrice creuse : `sparse_threshold=0` dans le `ColumnTransformer`.
3. Une variable oubliée dans les listes de colonnes est retirée sans message (`remainder="drop"`) : afficher `get_feature_names_out()` une fois.
4. Ajouter une variable calculée ligne par ligne (un produit de deux colonnes) se fait dans la fonction de nettoyage, avant la chaîne : aucune fuite, rien n'est appris.

## 7. Valider : découpage, point de comparaison, métriques

### Le problème

La cible du test est inconnue : la seule façon de savoir ce que vaut le modèle, et de choisir entre deux modèles, est de l'estimer sur des lignes d'apprentissage mises de côté. Et un nombre seul ne dit rien : une exactitude de 0,90 est un vrai résultat si les classes sont équilibrées, et ne vaut pas mieux que la règle « toujours la même classe » si 90 % des lignes sont de cette classe.

### L'énoncé

**Découpage stratifié.** `train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)` mélange les lignes et en met 20 % de côté, en conservant dans chaque partie la proportion de chaque classe de `y` (à une ligne près). Sans `test_size`, la part mise de côté est de 25 %. Fixer `random_state` rend le découpage reproductible.

**Validation croisée à \(k\) plis.** Les lignes sont réparties en \(k\) paquets ; tour à tour, chaque paquet est mis de côté, le modèle est ajusté sur les \(k - 1\) autres et noté sur lui. On obtient \(k\) notes ; leur moyenne estime la performance, leur écart-type donne l'ordre de grandeur de l'incertitude. `cross_val_score(modele, X, y, cv=5)` fait cela ; pour un classifieur, les plis sont stratifiés et la note est l'exactitude.

**Point de comparaison.** Prédire pour toutes les lignes la classe la plus fréquente de l'apprentissage. Si une part \(p\) des lignes de validation est de classe 1, cette règle a pour exactitude \(p\) si elle prédit 1 et \(1 - p\) si elle prédit 0. Un modèle ne vaut que par son écart à ce nombre.

**Métriques.** Sur \(n\) lignes de validation, on note \(\mathrm{VP}\) le nombre de lignes de classe 1 prédites 1 (vrais positifs), \(\mathrm{FP}\) le nombre de lignes de classe 0 prédites 1 (faux positifs), \(\mathrm{FN}\) le nombre de lignes de classe 1 prédites 0 (faux négatifs) et \(\mathrm{VN}\) le nombre de lignes de classe 0 prédites 0 (vrais négatifs) ; leur somme est \(n\).

| Métrique | Définition | Répond à |
|---|---|---|
| exactitude | \((\mathrm{VP} + \mathrm{VN})/n\) | quelle part des lignes est bien classée |
| précision | \(\mathrm{VP}/(\mathrm{VP} + \mathrm{FP})\) | parmi les lignes prédites 1, quelle part l'est vraiment |
| rappel | \(\mathrm{VP}/(\mathrm{VP} + \mathrm{FN})\) | parmi les lignes de classe 1, quelle part est retrouvée |
| F1 | \(2\,\mathrm{VP}/(2\,\mathrm{VP} + \mathrm{FP} + \mathrm{FN})\) | moyenne harmonique de la précision et du rappel |
| AUC | probabilité qu'une ligne de classe 1 tirée au hasard ait un score plus élevé qu'une ligne de classe 0 tirée au hasard, une égalité comptant pour 1/2 | le modèle ordonne-t-il bien, quel que soit le seuil |

**Laquelle.** D'abord celle que l'énoncé annonce. Sinon : l'exactitude quand les classes sont à peu près équilibrées et que les deux erreurs coûtent autant ; le F1, la précision ou le rappel quand la classe 1 est rare ; l'AUC pour comparer des modèles sans fixer de seuil. L'AUC se calcule sur des **scores** (`predict_proba(X)[:, 1]`), pas sur les classes prédites.

### Preuve : la formule du F1

1. Par définition, le F1 est la moyenne harmonique de la précision \(q\) et du rappel \(r\) : \(\mathrm{F1} = 2qr/(q + r)\), définie quand \(q + r > 0\).
2. Avec \(q = \mathrm{VP}/(\mathrm{VP} + \mathrm{FP})\) et \(r = \mathrm{VP}/(\mathrm{VP} + \mathrm{FN})\), on multiplie numérateur et dénominateur par \((\mathrm{VP} + \mathrm{FP})(\mathrm{VP} + \mathrm{FN})/\mathrm{VP}\), ce qui suppose \(\mathrm{VP} > 0\) : le numérateur devient \(2\,\mathrm{VP}\), le dénominateur \((\mathrm{VP} + \mathrm{FN}) + (\mathrm{VP} + \mathrm{FP})\).
3. Donc \(\mathrm{F1} = 2\,\mathrm{VP}/(2\,\mathrm{VP} + \mathrm{FP} + \mathrm{FN})\). Les vrais négatifs n'y figurent pas : c'est ce qui rend le F1 insensible au nombre de lignes de la classe fréquente bien classées.
4. Si \(\mathrm{VP} = 0\), la formule finale vaut 0 dès que \(\mathrm{FP} + \mathrm{FN} > 0\) : c'est la convention retenue.

### Exemple

```python-exec
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

X_app, X_val, y_app, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
print("part de résiliations : ensemble", round(y.mean(), 4), "; apprentissage", round(y_app.mean(), 4), "; validation", round(y_val.mean(), 4))

classe_majoritaire = int(y_app.mean() >= 0.5)
print("classe majoritaire :", classe_majoritaire, "; son exactitude en validation :", round((y_val == classe_majoritaire).mean(), 4))

modele = chaine(LogisticRegression(max_iter=1000)).fit(X_app, y_app)
classes = modele.predict(X_val)
scores = modele.predict_proba(X_val)[:, 1]
print("exactitude :", round(accuracy_score(y_val, classes), 4), "; F1 :", round(f1_score(y_val, classes), 4), "; AUC :", round(roc_auc_score(y_val, scores), 4))
```

```python-exec
plis = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
notes = cross_val_score(chaine(LogisticRegression(max_iter=1000)), X_app, y_app, cv=plis)
print(np.round(notes, 4), "; moyenne", round(notes.mean(), 4), "; écart-type", round(notes.std(), 4))
```

L'exactitude dépasse largement celle de la classe majoritaire : le modèle a appris quelque chose. Les cinq notes ne sont pas égales : leur dispersion dit qu'un écart du même ordre entre deux modèles ne prouve rien.

### Exemple : classe rare

Mille lignes dont cinquante de classe 1, et la règle « toujours 0 » :

```python-exec
y_rare = np.array([1] * 50 + [0] * 950)
toujours_zero = np.zeros(1000, dtype=int)
print("exactitude :", accuracy_score(y_rare, toujours_zero), "; F1 :", f1_score(y_rare, toujours_zero, zero_division=0))
```

Une exactitude élevée et un F1 nul : la règle ne retrouve aucune ligne de classe 1.

### Cas limites

- **Incertitude d'une exactitude.** Mesurée sur \(n\) lignes de validation indépendantes, l'exactitude d'un modèle déjà ajusté est la moyenne de \(n\) indicatrices « ligne bien classée », de même espérance \(e\) : son écart-type est \(\sqrt{e(1 - e)/n}\). Deux modèles dont les exactitudes diffèrent de moins que cet écart-type ne sont pas départagés.
- **Données datées.** Si les lignes de test sont postérieures aux lignes d'apprentissage, un découpage au hasard est trop optimiste : on met de côté les lignes les plus récentes.
- **Choisir puis estimer.** Si les mêmes lignes servent à choisir parmi plusieurs modèles et à annoncer l'exactitude du gagnant, le nombre annoncé est biaisé vers le haut, d'autant plus que les candidats sont nombreux. Remède : choisir par validation croisée sur une partie, estimer sur une partie jamais utilisée.

[[controle:metrique]]

### Pièges

1. Annoncer une exactitude sans celle de la classe majoritaire.
2. `roc_auc_score(y, modele.predict(X))` : sur des classes, l'ordre fin des scores est perdu.
3. Mesurer sur les lignes qui ont servi à ajuster : c'est l'exactitude d'apprentissage, toujours flatteuse pour un modèle flexible.

## 8. Choisir le modèle

### Le problème

Avec 25 à 30 minutes, il n'y a le temps que pour deux ou trois candidats aux réglages par défaut. Lesquels, et que demande chacun en préparation ?

### L'énoncé

Trois candidats couvrent le besoin, tous présents dans scikit-learn 1.1.1 (XGBoost n'est pas installé dans le notebook).

| Modèle | Préparation exigée | Ce qu'il représente | À savoir en 1.1.1 |
|---|---|---|---|
| `LogisticRegression` | imputation, mise à l'échelle, un-parmi-n | une somme d'effets, un par colonne ; pas d'interaction sans colonne produit | 100 itérations par défaut : `max_iter=1000` |
| `RandomForestClassifier` | imputation, encodage ; pas d'échelle | non-linéarités et interactions | refuse les `NaN` ; 100 arbres par défaut |
| `HistGradientBoostingClassifier` | encodage ; ni échelle ni imputation | non-linéarités et interactions | accepte les `NaN` ; refuse une matrice creuse ; pas de paramètre `class_weight` |

Garder la même préparation pour les trois permet de n'écrire qu'une seule fonction `chaine`. L'échelle est sans effet sur les deux derniers (preuve de la partie 6) ; l'imputation, elle, remplace le traitement des manquants propre au boosting par histogrammes, ce qui peut changer son résultat dans un sens ou dans l'autre : à mesurer s'il reste du temps.

**Règle de choix.** Comparer par validation croisée sur les mêmes plis ; si l'écart entre deux candidats est plus petit que l'écart-type entre plis, prendre le plus simple.

### Exemple

```python-exec
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

candidats = {
    "régression logistique": chaine(LogisticRegression(max_iter=1000)),
    "forêt aléatoire": chaine(RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=0)),
    "boosting par histogrammes": chaine(HistGradientBoostingClassifier(random_state=0)),
}
for nom, candidat in candidats.items():
    notes = cross_val_score(candidat, X_app, y_app, cv=plis)
    print(f"{nom:27s} {notes.mean():.4f} (écart-type entre plis {notes.std():.4f})")
```

Ce classement est propre à ce jeu : ses données ont été fabriquées par un modèle logistique sans interaction, le cas le plus favorable à la régression logistique. Il n'y a pas de classement valable partout ; c'est la raison de la comparaison.

### Cas limites

- **Interaction.** Si l'effet d'une variable dépend d'une autre (la pression use d'autant plus que la machine tourne longtemps), la régression logistique ne peut le représenter que si on lui donne la colonne produit ; les modèles à base d'arbres le représentent sans aide. « Ajouter une variable » sert d'abord au modèle linéaire.
- **Classe rare.** `class_weight="balanced"` existe pour `LogisticRegression` et `RandomForestClassifier` ; il donne à chaque classe un poids inversement proportionnel à son nombre de lignes.

### Pièges

1. Régler des hyperparamètres avant d'avoir un fichier valide : la chaîne simple d'abord.
2. Comparer des modèles sur des plis différents : les écarts mélangent alors l'effet du modèle et celui du découpage.

## 9. Écrire le fichier, le gabarit complet, le temps

### Le problème

Une prédiction juste dans un fichier mal formé vaut zéro. Le format est imposé : nom du fichier, noms et ordre des colonnes, une ligne par identifiant du test, valeurs attendues.

### L'énoncé

1. Réajuster le modèle retenu sur **toutes** les lignes d'apprentissage : les lignes de validation ont fini de servir et sont des exemples de plus. C'est un usage, pas un théorème ; l'exactitude estimée juste avant est celle du modèle ajusté sur 80 % des lignes, pas exactement celle du modèle final.
2. Prédire le test **dans l'ordre de ses lignes**, et prendre les identifiants dans le cadre de test d'origine.
3. Écrire avec `to_csv(nom, index=False)` : sans `index=False`, pandas ajoute l'index en première colonne.
4. **Relire** le fichier écrit et contrôler : colonnes, nombre de lignes, identifiants, valeurs. Comparer au fichier d'exemple s'il y en a un.

### Exemple : le gabarit complet

Tout ce qui précède, d'un seul tenant. Seule la fonction `nettoyer` (partie 3) et les deux listes de colonnes changent d'un jeu à l'autre.

```python-exec
import os
import tempfile

# 1. Lire, nettoyer (une fonction, deux appels)
train, test = pd.read_csv("abonnes_train.csv"), pd.read_csv("abonnes_test.csv")
a, b = nettoyer(train), nettoyer(test)
NUM, CAT, CIBLE = ["age", "revenu", "appels", "anciennete_mois"], ["forfait"], "resilie"
X, y, X_test = a[NUM + CAT], a[CIBLE], b[NUM + CAT]

# 2. Mettre de côté, point de comparaison
X_app, X_val, y_app, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
print("classe majoritaire :", round(max(y_val.mean(), 1 - y_val.mean()), 4))

# 3. Comparer les candidats par validation croisée, sur les mêmes plis
plis = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
candidats = {"logistique": chaine(LogisticRegression(max_iter=1000)), "boosting": chaine(HistGradientBoostingClassifier(random_state=0))}
notes = {nom: cross_val_score(c, X_app, y_app, cv=plis).mean() for nom, c in candidats.items()}
meilleur = max(notes, key=notes.get)
print({nom: round(v, 4) for nom, v in notes.items()}, "; retenu :", meilleur)

# 4. Estimer hors échantillon, sur les lignes jamais utilisées
print("exactitude de validation :", round(candidats[meilleur].fit(X_app, y_app).score(X_val, y_val), 4))

# 5. Réajuster sur tout, prédire, écrire, relire
final = candidats[meilleur].fit(X, y)
chemin = os.path.join(tempfile.gettempdir(), "predictions_lecon.csv")   # dans une épreuve : le nom imposé, dossier courant
pd.DataFrame({"id": test["id"], CIBLE: final.predict(X_test)}).to_csv(chemin, index=False)
relu = pd.read_csv(chemin)
print(relu.shape, list(relu.columns), relu["id"].tolist() == test["id"].tolist(), sorted(relu[CIBLE].unique()))
print("part de résiliations prédites :", round(relu[CIBLE].mean(), 4), "; observée à l'apprentissage :", round(y.mean(), 4))
relu.head(3)
```

Le dernier contrôle, la part de classe 1 prédite comparée à la part observée, attrape les catastrophes : un fichier presque tout à 0 ou tout à 1 signale une colonne mal alignée ou un nettoyage oublié sur le test.

### Cas limites

- **Identifiants.** Ils viennent de `test["id"]`, jamais d'un `range` : l'identifiant n'est pas le numéro de ligne.
- **Type des valeurs.** Si l'on attend 0 ou 1, écrire des entiers : `predict` rend des classes, `predict_proba` des probabilités, et un fichier de probabilités n'est pas un fichier de classes.
- **Nom du fichier.** Recopier le nom de l'énoncé caractère par caractère : le fichier d'exemple et le fichier à rendre n'ont pas forcément le même nom.

### Gestion du temps

Répartition proposée pour 25 à 30 minutes, à ajuster en s'entraînant :

| Minutes | Étape | Produit |
|---|---|---|
| 0 à 3 | lire, `dtypes`, `isna().sum()`, `value_counts`, `describe` | la liste des colonnes sales |
| 3 à 11 | fonction `nettoyer`, appliquée aux deux jeux | deux cadres propres, manquants comptés |
| 11 à 16 | `chaine` avec une régression logistique, point de comparaison, **premier fichier écrit et relu** | un fichier valide |
| 16 à 22 | validation croisée, deuxième modèle, variable ajoutée | le meilleur candidat |
| 22 à 27 | réajuster, réécrire, relire | le fichier final |
| 27 à 30 | relire l'énoncé : nom du fichier, colonnes, consignes non traitées | rien d'oublié |

L'ordre compte plus que les durées : un fichier valide existe dès la minute 16, et tout le reste ne fait que l'améliorer.

### Pièges

1. Trier ou filtrer le test en cours de route, puis coller les prédictions sur les identifiants d'origine : les lignes ne correspondent plus.
2. Écrire le fichier une seule fois, à la dernière minute.
3. Ne pas relire : `index=False` oublié et colonnes mal nommées se voient en cinq secondes.

## Réflexes en 30 secondes

| Besoin | Écriture |
|---|---|
| colonnes sales | `df.dtypes`, puis `df.isna().sum()` et `df.describe()` |
| texte vers nombre, illisible vers manquant | `pd.to_numeric(s, errors="coerce")` |
| retirer espaces, unité, virgule décimale | `s.str.replace(" ", "", regex=False)`, de même pour l'unité, puis la virgule vers un point |
| nombres en lettres | `s.map(dictionnaire).fillna(pd.to_numeric(s, errors="coerce"))` |
| valeur impossible vers manquant | `s.where(s >= 0)` |
| modalités uniformes | `s.str.strip().str.lower()` |
| colonne peut-être numérique | `s.astype(str)` avant `.str` |
| imputation | `SimpleImputer(strategy="median")`, dans la chaîne |
| un-parmi-n sans erreur au test | `OneHotEncoder(handle_unknown="ignore")` |
| aligner des `get_dummies` | `x_test.reindex(columns=x_train.columns, fill_value=0.0)` |
| sortie dense pour le boosting | `ColumnTransformer([...], sparse_threshold=0)` |
| mettre de côté | `train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)` |
| point de comparaison | `max(y.mean(), 1 - y.mean())` |
| comparer des modèles | `cross_val_score(modele, X, y, cv=plis).mean()` |
| AUC | `roc_auc_score(y, modele.predict_proba(X)[:, 1])` |
| écrire | `pd.DataFrame({"id": test["id"], cible: predictions}).to_csv(nom, index=False)` |

Les deux motifs d'extraction de la partie 3, à savoir réécrire :

```python
s.str.extract(r"^(\d+)(?: ans)?\Z")[0]      # un entier, suffixe facultatif, chaîne entière
s.str.extract(r"^(.*)[;,_](\d)\Z")          # deux informations, coupées au dernier séparateur
```

## À savoir refaire sans notes

1. Écrire de mémoire la fonction `chaine` : `ColumnTransformer` avec imputation et échelle pour les colonnes numériques, un-parmi-n pour les catégories, puis le modèle.
2. Écrire de mémoire le gabarit de la partie 9, de `read_csv` à la relecture du fichier, en moins de dix minutes.
3. Énoncer la règle unique et dire, pour chacune des opérations suivantes, si elle l'enfreint : médiane calculée sur les deux jeux réunis, `get_dummies` sur la concaténation, sélection de variables avant la validation croisée, nettoyage ligne par ligne des deux jeux.
4. Dire pourquoi on ne retire jamais de ligne du test, et ce qu'on fait à la place.
5. Refaire la preuve qu'un arbre ne dépend pas d'une mise à l'échelle affine, et dire où elle cesse de valoir pour un logarithme.
6. Redonner les définitions de l'exactitude, de la précision, du rappel et du F1, et retrouver la formule du F1 sans vrais négatifs.
7. Citer de mémoire les trois faits de version : `sparse=False` pour l'encodeur, pas de `class_weight` pour le boosting par histogrammes, forêt aléatoire qui refuse les `NaN`.
""",
    "controles": {
        "coerce": {
            "question": 'Que rend `pd.to_numeric(pd.Series(["7", "7 %", "sept"]), errors="coerce")` ?',
            "options": [
                {"id": "a", "texte": "Les valeurs 7, `NaN`, `NaN`.",
                 "explication": "Oui : seule la première chaîne est une écriture de nombre. `coerce` transforme les deux autres en manquants, sans rien nettoyer ; il y a @L_coerce_manquants@ manquants à aller regarder."},
                {"id": "b", "texte": "Les valeurs 7, 7, `NaN`.",
                 "explication": "`to_numeric` ne retire ni unité ni symbole : `7 %` n'est pas une écriture de nombre. Il faut d'abord retirer le symbole avec `str.replace`, puis convertir."},
                {"id": "c", "texte": "Une erreur `ValueError`.",
                 "explication": "C'est le comportement par défaut, `errors=\"raise\"`. Avec `errors=\"coerce\"`, les valeurs illisibles deviennent `NaN` au lieu d'arrêter le calcul."},
                {"id": "d", "texte": "Les valeurs 7, 7, 7.",
                 "explication": "pandas ne lit pas les nombres écrits en lettres, dans aucune langue : il faut un dictionnaire et `map`, comme dans l'exemple des appels."},
            ],
        },
        "test": {
            "question": "Après nettoyage, quelques lignes du jeu de test ont un revenu manquant. Que faire avant de prédire ?",
            "options": [
                {"id": "a", "texte": "Remplacer ces revenus par la médiane des revenus de l'apprentissage.",
                 "explication": "Oui : chaque identifiant du test garde sa ligne et reçoit une prédiction, et la valeur de remplacement est ajustée sur l'apprentissage seul, comme le veut la règle unique."},
                {"id": "b", "texte": "Retirer ces lignes avec `dropna()`, comme pour l'apprentissage.",
                 "explication": "Le fichier de prédictions doit contenir une ligne par identifiant du test : chaque ligne retirée est une prédiction absente, donc un fichier invalide ou des réponses comptées fausses."},
                {"id": "c", "texte": "Remplacer ces revenus par la médiane des revenus du test.",
                 "explication": "La transformation dépendrait du lot de test : elle n'est pas celle avec laquelle le modèle a été ajusté, et elle est impossible à appliquer à une ligne isolée. La valeur de remplacement s'ajuste sur l'apprentissage."},
                {"id": "d", "texte": "Remplacer ces revenus par 0.",
                 "explication": "Zéro est une valeur de revenu très éloignée de toutes celles que le modèle a vues : il extrapole, et la prédiction de ces lignes est tirée vers un extrême sans raison."},
            ],
        },
        "encodage": {
            "question": "L'apprentissage a les forfaits `base`, `plus`, `premium` ; le test contient en plus `pro`. Quelle écriture donne au test les mêmes colonnes qu'à l'apprentissage, sans erreur ?",
            "options": [
                {"id": "a", "texte": "`OneHotEncoder(handle_unknown=\"ignore\")` ajusté sur l'apprentissage, puis `transform` sur le test.",
                 "explication": "Oui : les colonnes sont celles des modalités apprises, et une ligne `pro` reçoit des zéros partout."},
                {"id": "b", "texte": "`pd.get_dummies(train)` d'un côté, `pd.get_dummies(test)` de l'autre.",
                 "explication": "`get_dummies` crée une colonne par modalité présente dans le cadre reçu : le test aurait une colonne `forfait_pro` que le modèle ne connaît pas, et il lui manquerait celle de toute modalité absente du test."},
                {"id": "c", "texte": "`OneHotEncoder()` ajusté sur l'apprentissage, puis `transform` sur le test.",
                 "explication": "Par défaut `handle_unknown=\"error\"` : `transform` lève `ValueError` à la première ligne `pro`."},
                {"id": "d", "texte": "`OneHotEncoder(handle_unknown=\"ignore\")` avec `fit_transform` sur l'apprentissage, puis `fit_transform` sur le test.",
                 "explication": "Le second `fit_transform` réapprend les modalités sur le test : les colonnes du test deviennent celles de ses propres modalités, dans un autre ordre et en autre nombre que celles de l'apprentissage."},
            ],
        },
        "fuite": {
            "question": "Laquelle de ces écritures estime l'exactitude sans fuite ?",
            "options": [
                {"id": "a", "texte": "`cross_val_score(Pipeline([(\"echelle\", StandardScaler()), (\"modele\", LogisticRegression())]), X, y, cv=5)`",
                 "explication": "Oui : à chaque pli, la chaîne entière est réajustée sur les lignes d'apprentissage du pli ; la moyenne et l'écart-type de l'échelle ne voient jamais les lignes mises de côté."},
                {"id": "b", "texte": "`cross_val_score(LogisticRegression(), StandardScaler().fit_transform(X), y, cv=5)`",
                 "explication": "L'échelle est ajustée une fois sur toutes les lignes, lignes de validation de chaque pli comprises. La fuite est faible pour une échelle, mais c'est la même écriture qui devient massive avec une sélection de variables."},
                {"id": "c", "texte": "Ajuster la chaîne sur `X`, puis calculer son exactitude sur ce même `X`.",
                 "explication": "C'est l'exactitude d'apprentissage : aucune ligne n'est mise de côté, et un modèle flexible peut y atteindre 1 sans rien valoir sur des lignes nouvelles."},
                {"id": "d", "texte": "Ajuster l'échelle sur l'apprentissage et le test réunis, puis faire la validation croisée sur l'apprentissage.",
                 "explication": "L'échelle dépend alors du test et des lignes de validation : deux violations de la règle unique, même si les colonnes sont bien alignées."},
            ],
        },
        "metrique": {
            "question": "Sur des lignes de validation dont 90 % sont de classe 0, un modèle obtient une exactitude de @L_exactitude_majoritaire_90:2@. Que conclure ?",
            "options": [
                {"id": "a", "texte": "Rien de bon : c'est exactement l'exactitude de la règle qui prédit toujours 0.",
                 "explication": "Oui : prédire toujours la classe majoritaire donne ici @L_exactitude_majoritaire_90:2@. Il faut regarder le rappel, la précision ou le F1 de la classe 1 pour savoir si le modèle en retrouve."},
                {"id": "b", "texte": "Le modèle est bon : il classe bien neuf lignes sur dix.",
                 "explication": "Une exactitude se lit par rapport au point de comparaison. Ici le point de comparaison vaut déjà @L_exactitude_majoritaire_90:2@ : le modèle n'apporte rien de mesurable par cette métrique."},
                {"id": "c", "texte": "Le modèle retrouve 90 % des lignes de classe 1.",
                 "explication": "C'est confondre l'exactitude et le rappel. Un modèle qui prédit toujours 0 a cette exactitude avec un rappel nul."},
                {"id": "d", "texte": "Son F1 vaut aussi @L_exactitude_majoritaire_90:2@.",
                 "explication": "Le F1 ne compte pas les vrais négatifs : il peut être nul avec cette exactitude, comme pour la règle « toujours 0 » de l'exemple de la classe rare."},
            ],
        },
    },
    "cellules_initiales": [
        "import numpy as np\nimport pandas as pd\n\ntrain = pd.read_csv(\"abonnes_train.csv\")\ntest = pd.read_csv(\"abonnes_test.csv\")\ntrain.head()",
        "# Essaie ici les exemples de la leçon, en les modifiant.\npd.to_numeric(pd.Series([\"12\", \"1 200\", \"3,5\", \"douze\", None]), errors=\"coerce\")",
        "train.dtypes",
    ],
    "verif": {"script": "verif/seance_code_3.py"},
}
