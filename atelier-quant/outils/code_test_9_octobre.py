"""Question de code T9-C1 : une chaîne complète de classification, sur le modèle de la question de code du test en
ligne du 9 octobre 2026 (notebook, train.csv, test.csv, sample_submission.csv, prédictions dans submissions.csv).

Le jeu de données est synthétique et inventé (coding/test9oct/generer_donnees.py) : ce n'est pas celui du test.
Évaluation : coding/test9oct/tests_test9oct.py, fonction evaluer(mode, espace), qui juge le fichier submissions.csv.
Mesures et seuils : verif/code_test_9_octobre.py. Les nombres de l'énoncé sont lus dans sa sortie, pas écrits à la main.

Ce module ne définit que CODE (et le texte INTRO_CORRECTION) : il se branche dans le module d'un test ou d'une séance
par `CODES = [code_test_9_octobre.CODE]`.
"""
import json
from pathlib import Path

_V = json.loads((Path(__file__).resolve().parent.parent / "verif" / "code_test_9_octobre.out.json").read_text(encoding="utf-8"))
_S = [f"{s:.2f}".replace(".", ",") for s in _V["seuils"]]
assert len(_S) == 4

INTRO_CORRECTION = (
    "Correction commentée, étape par étape, avec pandas et scikit-learn. Il n'y a qu'une version : entraîner un "
    "classifieur n'a pas d'équivalent raisonnable sans bibliothèque dans le temps de l'épreuve"
)

ENONCE = r"""## Prédire la panne d'un compresseur

**Origine.** Cette question reprend le genre de la question de code du test en ligne du 9 octobre 2026 : un notebook, un jeu d'apprentissage et un jeu de test en CSV, des colonnes mal formées, un classifieur binaire à entraîner, un fichier de prédictions à rendre. Le jeu de données, lui, est **synthétique et inventé pour cet entraînement** : ce n'est pas celui du test, et ses colonnes, ses valeurs et ses saletés sont différentes.

### Contexte

Un atelier relève les caractéristiques de ses compresseurs d'air. La colonne cible `panne` vaut 1 si le compresseur est tombé en panne dans les 90 jours qui ont suivi le relevé, 0 sinon. Tu dois entraîner un modèle sur `train.csv` et prédire `panne` pour chaque compresseur de `test.csv`.

### Fichiers du répertoire courant

| Fichier | Contenu |
|---|---|
| `train.csv` | N_TRAIN lignes : les colonnes ci-dessous et la cible `panne` |
| `test.csv` | N_TEST lignes : les mêmes colonnes, sans `panne` |
| `sample_submission.csv` | exemple du format attendu, 5 lignes aux valeurs arbitraires |
| `submissions.csv` | **à écrire** : tes prédictions |

### Colonnes

| Colonne | Contenu | Saletés |
|---|---|---|
| `id` | identifiant entier du compresseur, unique | aucune |
| `debit_et_ligne` | chaîne `<débit><séparateur><ligne>` : le débit nominal en m³/h, puis le numéro de la ligne de production | voir ci-dessous |
| `pression_bar` | pression de service, en bars (nombre) | aucune |
| `temperature_c` | température de refoulement, en degrés Celsius (nombre) | peut manquer |
| `anciennete_ans` | années écoulées depuis la mise en service (nombre) | peut manquer ; valeurs négatives |
| `heures_semaine` | heures de fonctionnement par semaine (entier de 20 à 168) | aucune |
| `materiau` | matériau du carter : `acier`, `fonte` ou `alu` | majuscules et espaces parasites |
| `panne` | cible, 0 ou 1 (dans `train.csv` seulement) | aucune |

Règles de lecture, valables pour les deux fichiers :

- **`debit_et_ligne`.** Le débit est un entier pair de 20 à 40. Il est écrit soit en chiffres avec une décimale (`32.0`), soit en lettres minuscules, les mots reliés par un trait d'union (`trente-deux`, `vingt`, `quarante`). Le séparateur est l'un des trois caractères `;` `,` `_`. La ligne est un seul chiffre, de 1 à 4 : c'est une **étiquette**, les lignes de production ne sont pas ordonnées. Exemples : `32.0;2`, `trente-deux_3`, `vingt-huit,1`.
- **`anciennete_ans`.** Une valeur strictement négative est une erreur de saisie : la vraie valeur est perdue, la traiter comme **manquante**.
- **`materiau`.** `ACIER`, `Acier` et ` acier ` (avec des espaces) désignent tous `acier` ; de même pour les deux autres matériaux.

### Travail demandé

1. **Nettoyer** les deux jeux selon les règles ci-dessus : deux colonnes propres à la place de `debit_et_ligne` (le débit en nombre, la ligne), anciennetés négatives rendues manquantes, matériaux uniformisés.
2. **Valeurs manquantes.** Dans le jeu d'apprentissage, retirer toute ligne qui a au moins une valeur manquante après nettoyage. Dans le jeu de test, **ne retirer aucune ligne** : chaque `id` de `test.csv` doit recevoir une prédiction, il faut donc y remplacer les valeurs manquantes.
3. **Ajouter au moins une variable** construite à partir des colonnes existantes (par exemple le produit ou le rapport de deux colonnes numériques), et la garder si elle améliore l'exactitude estimée.
4. **Encoder** les variables catégorielles là où c'est nécessaire.
5. **Entraîner un classifieur binaire** de `panne`, et **estimer son exactitude hors échantillon** sur des lignes d'apprentissage mises de côté.
6. **Écrire `submissions.csv`** dans le répertoire courant : exactement deux colonnes, `id` puis `panne`, avec la ligne d'en-tête ; une ligne par `id` de `test.csv`, chacun une seule fois, dans un ordre quelconque ; `panne` vaut 0 ou 1 (les écritures `0.0` et `1.0` sont acceptées ; `True`, `False` ou une probabilité ne le sont pas).

### Notation

Seul le fichier `submissions.csv` est jugé : les étapes 1 à 5 ne sont pas contrôlées une à une, elles se voient dans l'exactitude obtenue. L'**exactitude** est la proportion des N_TEST lignes de test dont la classe prédite est égale à la classe réelle, que tu ne connais pas.

- **Tests visibles** : 5 contrôles de format (fichier lisible, colonnes, nombre de lignes, identifiants, valeurs). Ils ne disent rien de la qualité des prédictions.
- **Soumettre** : 5 contrôles cachés. Le premier reprend le format complet ; les quatre autres demandent une exactitude au moins égale à SEUIL_1, SEUIL_2, SEUIL_3 puis SEUIL_4. Prédire toujours la classe la plus fréquente ne passe aucun de ces quatre seuils. Le dernier est placé au-dessus de ce qu'obtiennent sur ce jeu, après un nettoyage complet mais sans variable ajoutée, une régression logistique et un boosting par histogrammes aux réglages par défaut (mesures de `verif/code_test_9_octobre.py`).
- La note est la proportion de contrôles cachés réussis à la dernière soumission. Un fichier jamais soumis ne rapporte rien.

### Fonctionnement

- Notebook sous Python 3.9, pandas 1.4.2, numpy 1.21.6 et scikit-learn 1.1.1. PyTorch et XGBoost ne sont pas installés.
- Temps conseillé : 25 à 30 minutes. Vise d'abord une chaîne complète et simple qui écrit un fichier valide, soumets, puis améliore.
- Un `submissions.csv` écrit lors d'un essai précédent reste dans le dossier : il est jugé tant que tu ne l'as pas réécrit.
- Raccourcis : Maj+Entrée exécute la cellule et passe à la suivante, Ctrl+Entrée l'exécute sans changer de cellule.
"""
for _marque, _valeur in [("N_TRAIN", str(_V["n_train"])), ("N_TEST", str(_V["n_test"]))] + [(f"SEUIL_{k + 1}", s) for k, s in enumerate(_S)]:
    assert _marque in ENONCE
    ENONCE = ENONCE.replace(_marque, _valeur)

CODE = {
    "id": "T9-C1",
    "type": "code",
    "section": "code",
    "theme": "scikit-learn : chaîne complète de classification, du CSV sale au fichier de prédictions",
    "difficulte": 4,
    "temps_cible_s": 1800,
    "enonce": ENONCE,
    "dossier": "coding/test9oct",
    "module_tests": "coding/test9oct/tests_test9oct.py",
    "cellules_initiales": [
        "import numpy as np\nimport pandas as pd\n\n# Le jeu d'apprentissage est déjà chargé ci-dessous.\ndata = pd.read_csv(\"train.csv\")\ndata.head()",
        "test = pd.read_csv(\"test.csv\")\npd.read_csv(\"sample_submission.csv\")",
        "# Exploration et nettoyage\n",
        "# Variable ajoutée, encodage, modèle, validation\n",
        "# Écriture de submissions.csv, puis relecture pour contrôle\n",
    ],
    "tests_visibles": 5,
    "tests_caches": 5,
    "fichier_correction": "coding/test9oct/correction_test9oct.py",
    "intro_correction": INTRO_CORRECTION,
    "verif": {"script": "coding/test9oct/tests_test9oct.py"},
}
