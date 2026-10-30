"""Séance de code 3 : une chaîne complète de classification avec scikit-learn, du CSV sale au fichier de prédictions.

Leçon : outils/lecons_code_3.py (L-SKLEARN-1). Exercices, corrections et tests : coding/seance3/.
Valeurs des exemples et des questions, preuve que les tests cachés attrapent les erreurs visées, mesures qui fixent
les seuils de l'exercice 3 et faits d'interface de la leçon : verif/seance_code_3.py.
Dans chaque question à réponse unique, l'option "a" est la bonne ; S3-Q3 est à plusieurs réponses ("correctes").
"""
import lecons_code_3
import questions_diagnostic as diag

ID = "seance-code-3"
TITRE = "Séance de code 3 : une chaîne complète de classification avec scikit-learn"
VERIF = "verif/seance_code_3.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [
    {"id": "lecon", "nom": "Leçon", "budget_min": 30},
    {"id": "code", "nom": "Coding, data science", "budget_min": 45},
    {"id": "mlf", "nom": "Machine Learning fondamental : pièges d'une chaîne", "budget_min": 7},
]
PRESENTATION = (
    "D'abord une leçon de 30 minutes, avec cinq points de contrôle corrigés immédiatement et un notebook pour essayer "
    "les exemples : lire et inspecter deux fichiers CSV, convertir des colonnes numériques sales, traiter les valeurs "
    "manquantes sans jamais retirer de ligne du jeu de test, encoder les catégories, ranger la préparation et le "
    "modèle dans une chaîne pour éviter les fuites, valider, choisir un modèle, écrire le fichier de prédictions et "
    "le relire. Ensuite trois exercices de code, du plus guidé au plus complet : une fonction de nettoyage, une "
    "fonction qui aligne les colonnes encodées de l'apprentissage et du test, et une fonction `predire(train, test)` "
    "jugée par son exactitude sur des étiquettes cachées. Enfin quatre questions à choix sur les pièges, dont une à "
    "plusieurs réponses.\n\n"
    "Règle de travail : écris une solution simple, lance les tests visibles, **soumets tout de suite**, puis améliore "
    "et soumets à nouveau. Un code jamais soumis ne rapporte rien. Règle d'abandon : au-delà du double du temps cible "
    "d'un exercice, soumets ce que tu as et passe au suivant ; au-delà du double du temps cible d'une question à "
    "choix, réponds « Je ne sais pas » et passe à la suivante."
)
AVERTISSEMENTS = [
    "Temps indicatif : 12 minutes pour l'exercice 1, 8 pour le 2, 25 pour le 3, puis 7 minutes pour les quatre "
    "questions à choix.",
    "Le notebook s'exécute sous Python 3.9, pandas 1.4.2, numpy 1.21.6 et scikit-learn 1.1.1 ; PyTorch et XGBoost n'y "
    "sont pas installés. Les faits de version cités dans la leçon sont vérifiés dans cet environnement.",
    "Toutes les données de la séance sont synthétiques et inventées pour l'entraînement.",
    "La question S3-Q3 est à plusieurs réponses, notée tout ou rien : il faut cocher exactement l'ensemble juste.",
]

CELLULE_IMPORTS = "import numpy as np\nimport pandas as pd"


def fonctionnement(visibles, caches, limite):
    return (
        "\n\n**Fonctionnement.** Notebook sous Python 3.9, pandas 1.4.2 et scikit-learn 1.1.1. Tolérance de comparaison "
        f"des nombres : `1e-9` ; limite de temps : {limite} secondes par test. **Tests visibles** exécute {visibles} tests "
        f"visibles ; **Soumettre** exécute {caches} tests cachés et n'affiche que leur résultat.\n"
    )


CODES = [
    {
        "id": "S3-E1", "type": "code", "section": "code", "theme": "pandas : nettoyer des colonnes sales", "difficulte": 2, "temps_cible_s": 720,
        "enonce": r"""## Exercice 1 : nettoyer un cadre sale

Définis `nettoyer(df)`.

**Entrée.** `df` est un DataFrame qui contient au moins les colonnes `age`, `revenu`, `appels`, `anciennete_mois` et `forfait`, et éventuellement d'autres colonnes (par exemple `id` ou `resilie`). Son index est quelconque, sans doublon ; il peut n'avoir aucune ligne. Dans ces cinq colonnes, chaque valeur est une chaîne, un nombre, ou manquante (`NaN`). Attention : une colonne dont toutes les valeurs sont des nombres ou sont manquantes n'est pas une colonne de chaînes pour pandas.

**Sortie.** Un **nouveau** DataFrame, de même index et de mêmes colonnes dans le même ordre ; `df` n'est pas modifié ; les colonnes autres que les cinq citées sont recopiées telles quelles. Les colonnes `age`, `revenu`, `appels` et `anciennete_mois` sont des colonnes de flottants. Règles, valeur par valeur :

| Colonne | Si la valeur est une chaîne | Si la valeur est un nombre |
|---|---|---|
| `age` | on retire les espaces de début et de fin ; s'il reste un nombre écrit en chiffres, éventuellement suivi d'un espace et de `ans`, la valeur est ce nombre ; sinon `NaN` | conservé |
| `revenu` | on retire tous les espaces et le symbole `€`, on remplace la virgule par un point ; s'il reste un nombre écrit en chiffres, la valeur est ce nombre ; sinon `NaN` | conservé |
| `appels` | on retire les espaces de début et de fin ; `zéro`, `un`, `deux`, `trois`, `quatre`, `cinq` valent 0, 1, 2, 3, 4, 5 ; un nombre écrit en chiffres vaut ce nombre ; sinon `NaN` | conservé |
| `anciennete_mois` | (jamais une chaîne) | `NaN` s'il est strictement négatif, conservé sinon |
| `forfait` | espaces de début et de fin retirés, puis mise en minuscules | (jamais un nombre) |

Dans les cinq colonnes, une valeur manquante reste manquante. « Un nombre écrit en chiffres » désigne une suite de chiffres, éventuellement suivie d'un point et d'une autre suite de chiffres (`1980`, `3100.50`) ; les tests ne contiennent aucune autre écriture de nombre (ni signe, ni exposant).

**Exemple.**

| index | id | age | revenu | appels | anciennete_mois | forfait |
|---|---|---|---|---|---|---|
| 0 | 11 | `34 ans` | `2 450 €` | `trois` | 12.0 | ` Premium ` |
| 1 | 12 | `inconnu` | `1980` | `0` | -1.0 | `base` |
| 2 | 13 | ` 51` | NaN | `cinq` | NaN | `PLUS` |
| 3 | 14 | `27` | `3 100,50` | `2` | 40.0 | `plus` |

Sortie :

| index | id | age | revenu | appels | anciennete_mois | forfait |
|---|---|---|---|---|---|---|
| 0 | 11 | 34.0 | 2450.0 | 3.0 | 12.0 | `premium` |
| 1 | 12 | NaN | 1980.0 | 0.0 | NaN | `base` |
| 2 | 13 | 51.0 | NaN | 5.0 | NaN | `plus` |
| 3 | 14 | 27.0 | 3100.5 | 2.0 | 40.0 | `plus` |

La sortie contient @E1_exemple_manquants@ valeurs numériques manquantes : l'âge `inconnu`, l'ancienneté négative, et les deux valeurs déjà manquantes.

**Ce que vise l'exercice.** Les gestes de la partie 3 de la leçon : `str.replace`, `str.extract` avec un motif ancré, `map` par dictionnaire, `pd.to_numeric(errors="coerce")`, `where`. Les tests cachés contiennent un cadre vide, des textes illisibles, des colonnes entièrement manquantes, et des colonnes sales qui se trouvent être déjà numériques (leçon, partie 3, cas limites).
""" + fonctionnement(2, 9, 20),
        "dossier": "coding/seance3", "module_tests": "coding/seance3/tests_e1.py", "fonction": "nettoyer",
        "cellules_initiales": [
            CELLULE_IMPORTS + "\n\nexemple = pd.DataFrame(\n    [(11, \"34 ans\", \"2 450 €\", \"trois\", 12.0, \" Premium \"), (12, \"inconnu\", \"1980\", \"0\", -1.0, \"base\"),\n     (13, \" 51\", np.nan, \"cinq\", np.nan, \"PLUS\"), (14, \"27\", \"3 100,50\", \"2\", 40.0, \"plus\")],\n    columns=[\"id\", \"age\", \"revenu\", \"appels\", \"anciennete_mois\", \"forfait\"])\nexemple",
            "def nettoyer(df: pd.DataFrame) -> pd.DataFrame:\n    d = df.copy()\n    # À compléter\n    return d",
            "nettoyer(exemple)",
            "# Fichier d'exploration : les abonnés de la leçon.\nnettoyer(pd.read_csv(\"abonnes_train.csv\")).head()",
        ],
        "tests_visibles": 2, "tests_caches": 9,
        "fichier_correction": "coding/seance3/_correction.py",
        "intro_correction": "Correction commentée des trois exercices : les exercices 1 et 2 en version pandas et en version sans bibliothèque externe, l'exercice 3 avec scikit-learn seulement",
        "verif": {"script": "coding/seance3/tests_e1.py"},
    },
    {
        "id": "S3-E2", "type": "code", "section": "code", "theme": "pandas : aligner les colonnes encodées de l'apprentissage et du test", "difficulte": 3, "temps_cible_s": 480,
        "enonce": r"""## Exercice 2 : aligner l'encodage un-parmi-n

Définis `aligner(train, test, colonnes)`, qui renvoie un couple `(x_train, x_test)`.

**Entrées.** `train` et `test` sont deux DataFrame qui ont les mêmes colonnes, dans le même ordre. `train` a au moins une ligne ; `test` peut n'en avoir aucune. Leurs index sont quelconques, sans doublon. `colonnes` est une liste non vide de noms de colonnes **catégorielles** : leurs valeurs sont des chaînes ou des entiers, ou sont manquantes (`NaN`). Toutes les autres colonnes sont numériques.

**Sortie.** Deux DataFrame, `x_train` de même index que `train` et `x_test` de même index que `test`, qui ont **exactement les mêmes colonnes, dans le même ordre** :

1. d'abord les colonnes non catégorielles, dans leur ordre d'origine, inchangées ;
2. puis, pour chaque colonne `c` de `colonnes`, dans l'ordre de la liste : une colonne nommée `c_v` pour chaque valeur `v` non manquante que prend `c` **dans `train`**, ces valeurs étant rangées par ordre croissant (alphabétique pour des chaînes, numérique pour des entiers). Le nom est formé de `c`, d'un tiret bas et de l'écriture de `v` : `forfait_plus`, `ligne_2`.

La colonne `c_v` est une colonne de flottants : 1.0 sur les lignes où `c` vaut `v`, 0.0 sur toutes les autres. En conséquence : une valeur du test que `train` ne contient pas ne crée aucune colonne, et sa ligne a 0.0 dans toutes les colonnes de `c` ; une valeur de `train` absente du test donne dans `x_test` une colonne de 0.0 ; une valeur manquante a 0.0 dans toutes les colonnes de `c`. Ni `train` ni `test` ne sont modifiés.

**Exemple**, avec `colonnes = ["forfait"]`.

`train` :

| index | revenu | forfait |
|---|---|---|
| 0 | 2450.0 | plus |
| 1 | 1980.0 | base |
| 2 | 3100.0 | plus |
| 3 | 2700.0 | premium |

`test` :

| index | revenu | forfait |
|---|---|---|
| 0 | 2000.0 | base |
| 1 | 2600.0 | pro |
| 2 | 3300.0 | base |

`x_train` :

| index | revenu | forfait_base | forfait_plus | forfait_premium |
|---|---|---|---|---|
| 0 | 2450.0 | 0.0 | 1.0 | 0.0 |
| 1 | 1980.0 | 1.0 | 0.0 | 0.0 |
| 2 | 3100.0 | 0.0 | 1.0 | 0.0 |
| 3 | 2700.0 | 0.0 | 0.0 | 1.0 |

`x_test` :

| index | revenu | forfait_base | forfait_plus | forfait_premium |
|---|---|---|---|---|
| 0 | 2000.0 | 1.0 | 0.0 | 0.0 |
| 1 | 2600.0 | 0.0 | 0.0 | 0.0 |
| 2 | 3300.0 | 1.0 | 0.0 | 0.0 |

Les deux cadres ont @E2_colonnes_train@ colonnes. Un `pd.get_dummies(test, columns=["forfait"])` seul n'en aurait que @E2_colonnes_test_naif@ : `revenu`, `forfait_base` et `forfait_pro`, inutilisables par un modèle ajusté sur `x_train`.

**Ce que vise l'exercice.** Le piège de `get_dummies` (leçon, partie 5) et sa réparation : l'apprentissage fixe les colonnes, le test s'y conforme. Trois lignes suffisent.
""" + fonctionnement(2, 8, 20),
        "dossier": "coding/seance3", "module_tests": "coding/seance3/tests_e2.py", "fonction": "aligner",
        "cellules_initiales": [
            CELLULE_IMPORTS + "\n\ntrain = pd.DataFrame({\"revenu\": [2450.0, 1980.0, 3100.0, 2700.0], \"forfait\": [\"plus\", \"base\", \"plus\", \"premium\"]})\ntest = pd.DataFrame({\"revenu\": [2000.0, 2600.0, 3300.0], \"forfait\": [\"base\", \"pro\", \"base\"]})\npd.get_dummies(train, columns=[\"forfait\"]), pd.get_dummies(test, columns=[\"forfait\"])",
            "def aligner(train: pd.DataFrame, test: pd.DataFrame, colonnes):\n    # À compléter : renvoyer (x_train, x_test)\n    return train, test",
            "aligner(train, test, [\"forfait\"])",
        ],
        "tests_visibles": 2, "tests_caches": 8,
        "fichier_correction": "coding/seance3/_correction.py",
        "intro_correction": "Correction commentée des trois exercices : les exercices 1 et 2 en version pandas et en version sans bibliothèque externe, l'exercice 3 avec scikit-learn seulement",
        "verif": {"script": "coding/seance3/tests_e2.py"},
    },
    {
        "id": "S3-E3", "type": "code", "section": "code", "theme": "scikit-learn : nettoyer, encoder, entraîner, prédire", "difficulte": 4, "temps_cible_s": 1500,
        "enonce": r"""## Exercice 3 : de deux cadres sales à des prédictions

Définis `predire(train, test)`.

**Entrées.**

- `train` : DataFrame aux colonnes `id`, `age`, `revenu`, `appels`, `anciennete_mois`, `forfait`, écrites comme dans l'exercice 1 (mêmes saletés, mêmes règles de lecture), et `resilie`, la cible, qui vaut 0 ou 1 et ne manque jamais.
- `test` : les mêmes colonnes sans `resilie`, au moins une ligne, index quelconque.

Ce que les tests peuvent contenir : des valeurs manquantes, déclarées ou déguisées, dans `age`, `revenu` et `anciennete_mois`, **y compris dans `test`** ; dans `test`, un forfait `pro` absent de `train` ; une colonne `appels` entièrement en chiffres, donc lue comme une colonne d'entiers. `forfait` et `appels` ne manquent jamais. Les données sont synthétiques : `resilie` dépend des cinq variables, et l'identifiant ne porte aucune information.

**Sortie.** Une suite de `len(test)` valeurs 0 ou 1 (liste, tableau numpy ou `Series`) : la prédiction de `resilie` pour chaque ligne de `test`, **dans l'ordre des lignes de `test`** (la première valeur pour la première ligne, quel que soit l'index).

**Notation.** Chaque test fournit un couple (`train`, `test`) et garde pour lui les vraies valeurs de `resilie` sur `test`. Il est réussi si la sortie a le bon format et si son **exactitude**, la proportion de lignes de `test` bien classées, atteint le seuil du test. Deux tests ne contrôlent que le format (une ligne et cinq lignes de test). Les @E3_tests_exactitude@ autres ont pour seuil @E3_seuil:2@, sauf celui dont l'apprentissage n'a que 200 lignes, de seuil @E3_seuil_petit:2@.

Repères mesurés sur ces @E3_tests_exactitude@ tests (script `verif/seance_code_3.py`) : prédire toujours la classe majoritaire donne au plus @E3_majoritaire_max:3@ ; une régression logistique sur la seule colonne déjà numérique, manquants mis à 0, au plus @E3_baclee_max:3@ ; la correction (nettoyage de l'exercice 1, imputation par la médiane, mise à l'échelle, encodage un-parmi-n, régression logistique) entre @E3_correction_min:3@ et @E3_correction_max:3@.

**Fichiers d'exploration.** `abonnes_train.csv` et `abonnes_test.csv`, les fichiers de la leçon, sont dans le répertoire courant ; le second contient des forfaits `pro`. Sers-t'en pour valider ta fonction par un découpage de `abonnes_train.csv`.

**Ce que vise l'exercice.** La chaîne complète de la leçon : la fonction de nettoyage appliquée aux deux cadres, puis un `Pipeline` qui contient un `ColumnTransformer` (imputation et échelle des colonnes numériques, `OneHotEncoder(handle_unknown="ignore")` pour le forfait) et un classifieur. Ne retire aucune ligne de `test`.
""" + fonctionnement(2, 8, 30),
        "dossier": "coding/seance3", "module_tests": "coding/seance3/tests_e3.py", "fonction": "predire",
        "cellules_initiales": [
            CELLULE_IMPORTS + "\n\ntrain = pd.read_csv(\"abonnes_train.csv\")\ntest = pd.read_csv(\"abonnes_test.csv\")\ntrain.head()",
            "def nettoyer(df: pd.DataFrame) -> pd.DataFrame:\n    d = df.copy()\n    # Ta fonction de l'exercice 1\n    return d",
            "def predire(train: pd.DataFrame, test: pd.DataFrame):\n    # À compléter : renvoyer une prédiction 0 ou 1 par ligne de test, dans l'ordre\n    return np.zeros(len(test), dtype=int)",
            "# Validation maison : mettre de côté une partie de abonnes_train.csv et mesurer l'exactitude de predire\npredictions = predire(train, test)\nlen(predictions), predictions[:10]",
        ],
        "tests_visibles": 2, "tests_caches": 8,
        "fichier_correction": "coding/seance3/_correction.py",
        "intro_correction": "Correction commentée des trois exercices : les exercices 1 et 2 en version pandas et en version sans bibliothèque externe, l'exercice 3 avec scikit-learn seulement",
        "verif": {"script": "coding/seance3/tests_e3.py"},
    },
]

LECONS = [lecons_code_3.LECON_SKLEARN]

QUESTIONS = [
    {
        "id": "S3-Q1", "section": "mlf", "theme": "Mise à l'échelle ajustée sur l'apprentissage seul", "difficulte": 3, "temps_cible_s": 120,
        "enonce": r"""Une variable vaut 2, 4, 6 et 8 sur les quatre lignes d'apprentissage, et 20 sur l'unique ligne de test. On la standardise avec `StandardScaler`, qui retranche la moyenne puis divise par l'écart-type à dénominateur \(n\) (le nombre de lignes sur lesquelles il est ajusté).

Dans une chaîne correcte, où l'échelle est ajustée sur l'apprentissage seul, quelle valeur standardisée le modèle reçoit-il pour la ligne de test, à quatre décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q1_z_juste:4@",
             "explication": r"Oui : sur l'apprentissage, la moyenne vaut @Q1_moyenne_app@ et la variance @Q1_variance_app@, donc \(z = (20 - @Q1_moyenne_app@)/\sqrt{@Q1_variance_app@}\)."},
            {"id": "b", "texte": "@Q1_z_fuite:4@",
             "explication": r"C'est la valeur obtenue en ajustant l'échelle sur les cinq lignes, test compris : moyenne @Q1_moyenne_tout@, variance @Q1_variance_tout@. La ligne de test a servi à fixer sa propre échelle : c'est une fuite, et la valeur extrême 20 paraît alors banale."},
            {"id": "c", "texte": "@Q1_z_ddof1:4@",
             "explication": r"C'est le résultat avec l'écart-type à dénominateur \(n - 1\), celui que rend `Series.std()` de pandas. `StandardScaler` divise par \(n\), comme le précise l'énoncé."},
            {"id": "d", "texte": "@Q1_z_test_seul@",
             "explication": r"C'est ce que donne `fit_transform` appliqué à la ligne de test seule : sa moyenne est elle-même, la valeur centrée est nulle. Sur le test, on applique `transform` avec la moyenne et l'écart-type de l'apprentissage."},
        ],
        "solution": r"""Réponse : @Q1_z_juste:4@.

Étape 1 : l'échelle est ajustée sur les quatre lignes d'apprentissage. Leur moyenne est \((2 + 4 + 6 + 8)/4 = @Q1_moyenne_app@\) ; leur variance à dénominateur 4 est \((9 + 1 + 1 + 9)/4 = @Q1_variance_app@\).

Étape 2 : la ligne de test est transformée avec ces deux nombres, sans réajustement : \(z = (20 - @Q1_moyenne_app@)/\sqrt{@Q1_variance_app@} \approx @Q1_z_juste:4m@\).

Étape 3, ce que change la fuite : ajustée sur les cinq lignes, l'échelle aurait pour moyenne @Q1_moyenne_tout@ et pour variance @Q1_variance_tout@, et la même ligne vaudrait \(12/\sqrt{@Q1_variance_tout@} \approx @Q1_z_fuite:4m@\). Le script recalcule les quatre valeurs par fractions et par `StandardScaler`.""",
    },
    {
        "id": "S3-Q2", "section": "mlf", "theme": "Modalité inconnue au test, encodage un-parmi-n", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""La colonne `forfait` vaut `base`, `plus`, `premium`, `plus` sur les quatre lignes d'apprentissage, et `plus`, `pro` sur les deux lignes de test. On ajuste `OneHotEncoder(handle_unknown="ignore")` sur la colonne d'apprentissage, puis on l'applique (`transform`) à la colonne de test.

Combien de colonnes a la matrice obtenue pour le test, et que contient la ligne du forfait `pro` ?""",
        "options": [
            {"id": "a", "texte": "@Q2_colonnes@ colonnes ; la ligne `pro` ne contient que des zéros.",
             "explication": "Oui : les colonnes sont celles des modalités apprises (`base`, `plus`, `premium`), et `handle_unknown=\"ignore\"` code une modalité inconnue par une ligne nulle, de somme @Q2_somme_ligne_pro@."},
            {"id": "b", "texte": "@Q2_colonnes_concat@ colonnes ; la ligne `pro` a un 1 dans la colonne `pro`.",
             "explication": "Ce serait le résultat d'un encodage ajusté sur l'apprentissage et le test réunis. `transform` n'ajoute jamais de colonne : il applique les modalités apprises par `fit`, et `pro` n'en fait pas partie."},
            {"id": "c", "texte": "@Q2_colonnes_dummies_test@ colonnes ; la ligne `pro` a un 1 dans la colonne `pro`.",
             "explication": "C'est ce que donne `pd.get_dummies` appliqué au test seul : une colonne par modalité présente dans le test, `plus` et `pro`. Ces colonnes ne sont pas celles sur lesquelles le modèle a été ajusté."},
            {"id": "d", "texte": "Aucune : `transform` lève une erreur à la ligne `pro`.",
             "explication": "C'est le comportement du réglage par défaut, `handle_unknown=\"error\"`. L'énoncé précise `handle_unknown=\"ignore\"`, qui remplace l'erreur par une ligne de zéros."},
        ],
        "solution": r"""Réponse : @Q2_colonnes@ colonnes, et la ligne `pro` ne contient que des zéros.

Étape 1 : `fit` apprend la liste triée des modalités de l'apprentissage : `base`, `plus`, `premium`. La valeur `plus`, présente deux fois, ne compte qu'une fois : @Q2_colonnes@ modalités, donc @Q2_colonnes@ colonnes.

Étape 2 : `transform` produit ces @Q2_colonnes@ colonnes pour toute ligne. La ligne `plus` du test a un 1 dans la deuxième colonne.

Étape 3 : `pro` n'est pas une modalité apprise. Avec `handle_unknown="ignore"`, sa ligne est nulle ; avec le réglage par défaut, `transform` lèverait `ValueError`. Le script le vérifie sous scikit-learn 1.1.1, ainsi que les @Q2_colonnes_dummies_test@ colonnes de `get_dummies` sur le test seul et les @Q2_colonnes_concat@ colonnes d'un encodage des deux jeux réunis.""",
    },
    {
        "id": "S3-Q3", "section": "mlf", "theme": "Fuites et lignes de test : ce qui est permis", "difficulte": 4, "temps_cible_s": 120,
        "conceptuelle": True, "correctes": ["a", "d"],
        "enonce": r"""On dispose d'un jeu d'apprentissage, dont la cible est connue, et d'un jeu de test, dont il faut prédire la cible pour **chaque** ligne. On veut aussi une estimation honnête de l'exactitude du modèle sur des lignes nouvelles.

Parmi les pratiques suivantes, lesquelles sont correctes ? (Plusieurs réponses possibles ; il faut cocher exactement les bonnes.)""",
        "options": [
            {"id": "a", "texte": "Placer l'imputation, la mise à l'échelle et le modèle dans un `Pipeline`, et passer ce `Pipeline` à `cross_val_score` sur le jeu d'apprentissage.",
             "explication": "Correct : à chaque pli, la chaîne entière est réajustée sur les seules lignes d'apprentissage du pli ; médianes, moyennes et écarts-types ne voient jamais les lignes qui servent à noter."},
            {"id": "b", "texte": "Appliquer `dropna()` au jeu de test après nettoyage, comme au jeu d'apprentissage, puis prédire les lignes restantes.",
             "explication": "Incorrect : chaque ligne retirée du test est une prédiction absente du fichier. Les valeurs manquantes du test se remplacent, par une valeur ajustée sur l'apprentissage ; on ne retire des lignes que de l'apprentissage."},
            {"id": "c", "texte": "Garder les variables les plus corrélées à la cible sur tout le jeu d'apprentissage, puis estimer l'exactitude du modèle par validation croisée sur ces variables.",
             "explication": "Incorrect : la sélection a utilisé la cible des lignes qui serviront ensuite à noter chaque pli. L'expérience de la partie 6 de la leçon montre qu'on obtient ainsi une exactitude élevée sur des variables de pur bruit. La sélection doit être une étape de la chaîne."},
            {"id": "d", "texte": "Une fois le modèle choisi par validation, le réajuster sur toutes les lignes d'apprentissage avant de prédire le test.",
             "explication": "Correct : les lignes de validation ont fini de servir à estimer ; aucune information du test n'entre dans ce réajustement, et l'estimation déjà faite reste celle d'un modèle ajusté sur un peu moins de lignes."},
        ],
        "solution": r"""Réponse : le `Pipeline` passé à `cross_val_score`, et le réajustement final sur toutes les lignes d'apprentissage.

La règle qui tranche les quatre cas (leçon, partie 1) : toute quantité ajustée sur des données et utilisée pour transformer ou prédire ne doit venir que de lignes d'apprentissage, c'est-à-dire, pendant une validation, des lignes qui ne sont pas mises de côté à ce moment-là.

- Le `Pipeline` dans la validation croisée respecte la règle par construction.
- Le réajustement final n'utilise que des lignes d'apprentissage : il est permis, et c'est l'usage.
- `dropna()` sur le test n'est pas une fuite, c'est une faute de format : le fichier n'a plus une ligne par identifiant.
- La sélection de variables faite avant la validation croisée enfreint la règle : elle a vu la cible des lignes de validation.""",
    },
    {
        "id": "S3-Q4", "section": "mlf", "theme": "Exactitude, F1 et classe rare", "difficulte": 4, "temps_cible_s": 150,
        "enonce": r"""Sur @Q4_n@ lignes de validation, @Q4_positifs@ sont de classe 1. Le modèle A prédit 0 pour toutes les lignes. Le modèle B prédit 1 pour @Q4_vp@ lignes de classe 1 et pour @Q4_fp@ lignes de classe 0, et 0 pour toutes les autres.

L'exactitude est la proportion de lignes bien classées ; le F1 de la classe 1 est la moyenne harmonique de la précision (part des lignes prédites 1 qui sont de classe 1) et du rappel (part des lignes de classe 1 prédites 1). L'exactitude du modèle A vaut @Q4_exactitude_A:2@.

Que valent l'exactitude et le F1 du modèle B, à quatre décimales ?""",
        "options": [
            {"id": "a", "texte": "Exactitude @Q4_exactitude_B:4@ et F1 @Q4_f1_B:4@",
             "explication": r"Oui : \((@Q4_vp@ + @Q4_vn@)/@Q4_n@\) pour l'exactitude, et \(2 \times @Q4_vp@/(2 \times @Q4_vp@ + @Q4_fp@ + @Q4_fn@) = \tfrac{@Q4_f1_num@}{@Q4_f1_den@}\) pour le F1. B a une exactitude plus faible que A, mais c'est le seul des deux qui retrouve des lignes de classe 1."},
            {"id": "b", "texte": "Exactitude @Q4_exactitude_B:4@ et F1 @Q4_moyenne_arithmetique:4@",
             "explication": r"C'est la moyenne arithmétique de la précision @Q4_precision_B:4@ et du rappel @Q4_rappel_B:4@. Le F1 est leur moyenne **harmonique**, toujours inférieure ou égale, et tirée vers la plus petite des deux."},
            {"id": "c", "texte": "Exactitude @Q4_rappel_B:4@ et F1 @Q4_f1_B:4@",
             "explication": r"@Q4_rappel_B:4@ est le rappel, \(@Q4_vp@/@Q4_positifs@\), la part des lignes de classe 1 retrouvées. L'exactitude compte toutes les lignes bien classées, vrais négatifs compris."},
            {"id": "d", "texte": "Exactitude @Q4_exactitude_B:4@ et F1 @Q4_precision_B:4@",
             "explication": r"@Q4_precision_B:4@ est la précision seule, \(@Q4_vp@/(@Q4_vp@ + @Q4_fp@)\). Le F1 combine la précision et le rappel ; il ne coïncide avec la précision que si les deux sont égaux."},
        ],
        "solution": r"""Réponse : exactitude @Q4_exactitude_B:4@ et F1 @Q4_f1_B:4@.

Étape 1 : les quatre cases. Vrais positifs \(\mathrm{VP} = @Q4_vp@\) ; faux positifs \(\mathrm{FP} = @Q4_fp@\) ; faux négatifs \(\mathrm{FN} = @Q4_positifs@ - @Q4_vp@ = @Q4_fn@\) ; vrais négatifs \(\mathrm{VN} = @Q4_n@ - @Q4_vp@ - @Q4_fp@ - @Q4_fn@ = @Q4_vn@\).

Étape 2 : exactitude \(= (\mathrm{VP} + \mathrm{VN})/@Q4_n@ = @Q4_exactitude_B:2m@\), inférieure à celle du modèle A, @Q4_exactitude_A:2@, qui est la part de la classe 0.

Étape 3 : précision \(= @Q4_vp@/(@Q4_vp@ + @Q4_fp@) \approx @Q4_precision_B:4m@\), rappel \(= @Q4_vp@/@Q4_positifs@ = @Q4_rappel_B:1m@\), et \(\mathrm{F1} = 2\,\mathrm{VP}/(2\,\mathrm{VP} + \mathrm{FP} + \mathrm{FN}) = \tfrac{@Q4_f1_num@}{@Q4_f1_den@} \approx @Q4_f1_B:4m@\) (leçon, partie 7).

Étape 4, ce qu'il faut en retenir : le modèle A n'a aucun vrai positif, son rappel et son F1 sont nuls. Avec une classe rare, l'exactitude classe A devant B alors que A ne détecte rien : il faut lire le rappel, la précision ou le F1.""",
    },
]
