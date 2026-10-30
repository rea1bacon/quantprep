"""Séance de code 1 : trois gestes pandas (décalage par groupe, rangs par date, corrélation par date), puis la
question de code du test de positionnement, ic_moyen, qui les assemble.

Valeurs des exemples : verif/seance_code_1.py.
"""
import lecons
import questions_diagnostic as diag

ID = "seance-code-1"
TITRE = "Séance de code 1 : groupby, décalage, rangs"
VERIF = "verif/seance_code_1.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [{"id": "lecon", "nom": "Leçon", "budget_min": 20}, {"id": "code", "nom": "Coding, data science", "budget_min": 50}]
PRESENTATION = (
    "D'abord une leçon de 20 minutes, avec des points de contrôle corrigés immédiatement et un notebook pour essayer "
    "les exemples. Ensuite quatre exercices courts, du plus simple au plus complet. Les trois premiers isolent chacun un geste "
    "nécessaire à la question de code du test de positionnement ; le quatrième est cette question elle-même.\n\n"
    "Règle de travail, à appliquer aussi dans un test en ligne : écris une solution simple, lance les tests visibles, "
    "**soumets tout de suite**, puis améliore et soumets à nouveau. Un code jamais soumis ne rapporte rien."
)
AVERTISSEMENTS = [
    "Temps indicatif : 8 minutes pour chacun des trois premiers exercices, 20 minutes pour le quatrième. "
    "Dans les examens blancs, la question de code dispose de 24 minutes.",
]

CELLULE_IMPORTS = "import numpy as np\nimport pandas as pd"

CODES = [
    {
        "id": "S1-E1", "type": "code", "section": "code", "theme": "pandas : décalage par groupe", "difficulte": 2, "temps_cible_s": 480,
        "enonce": """## Exercice 1 : rendement futur par actif

Définis `rendement_futur(df)`.

**Entrée.** Un DataFrame aux colonnes `date` (chaîne `AAAA-MM-JJ`), `actif` (chaîne), `prix` (nombre strictement positif ou `NaN`). Au plus une ligne par couple (`date`, `actif`). Lignes dans un ordre **quelconque** ; l'index est quelconque mais sans doublon (ce n'est pas forcément 0, 1, 2, ...).

**Sortie.** Une `Series` de flottants **ayant le même index que `df`** : pour la ligne d'étiquette `i`, d'actif `a` et de prix `p`, la valeur est `q / p - 1`, où `q` est le prix de la ligne suivante de `a` dans l'ordre chronologique (la ligne de `a` à la plus petite date strictement postérieure). `NaN` si cette ligne n'existe pas ou si `p` ou `q` est `NaN`.

**Exemple.**

| index | date | actif | prix | sortie |
|---|---|---|---|---|
| 0 | 2026-01-06 | A | 110 | 99/110 - 1 = -0,1 |
| 1 | 2026-01-05 | A | 100 | 110/100 - 1 = 0,1 |
| 2 | 2026-01-05 | B | 50 | 55/50 - 1 = 0,1 |
| 3 | 2026-01-07 | A | 99 | NaN |
| 4 | 2026-01-07 | B | 55 | NaN |

Les valeurs sont égales à l'arrondi flottant près (tolérance `1e-9`). La ligne 2 a pour suivante la ligne 4 : l'actif B n'a pas de ligne le 2026-01-06.

**Ce que vise l'exercice.** `shift(-1)` donne la ligne suivante **dans l'ordre où les lignes se trouvent**. Pour qu'elle soit la suivante dans le temps, il faut décaler un cadre trié, et ensuite revenir à l'index d'origine.
""",
        "dossier": "coding/seance1", "module_tests": "coding/seance1/tests_e1.py", "fonction": "rendement_futur",
        "cellules_initiales": [
            CELLULE_IMPORTS + "\n\nexemple = pd.DataFrame(\n    [(\"2026-01-06\", \"A\", 110.0), (\"2026-01-05\", \"A\", 100.0), (\"2026-01-05\", \"B\", 50.0),\n     (\"2026-01-07\", \"A\", 99.0), (\"2026-01-07\", \"B\", 55.0)],\n    columns=[\"date\", \"actif\", \"prix\"])\nexemple",
            "def rendement_futur(df: pd.DataFrame) -> pd.Series:\n    # À compléter\n    return pd.Series(np.nan, index=df.index)",
            "rendement_futur(exemple)",
        ],
        "tests_visibles": 2, "tests_caches": 8,
        "fichier_correction": "coding/seance1/_correction.py",
        "verif": {"script": "coding/seance1/tests_e1.py"},
    },
    {
        "id": "S1-E2", "type": "code", "section": "code", "theme": "pandas : rang par groupe", "difficulte": 2, "temps_cible_s": 480,
        "enonce": """## Exercice 2 : rang transversal

Définis `rang_transversal(df)`.

**Entrée.** Un DataFrame aux colonnes `date`, `actif`, `signal` (nombre ou `NaN`). Lignes dans un ordre quelconque, index quelconque sans doublon.

**Sortie.** Une `Series` de flottants ayant le même index que `df` : pour chaque ligne, le rang de son signal parmi les signaux **non manquants de la même date**, par ordre croissant, à partir de 1 ; les valeurs égales reçoivent la **moyenne** des rangs qu'elles occupent. `NaN` si le signal de la ligne est `NaN`.

**Exemple.**

| index | date | actif | signal | sortie |
|---|---|---|---|---|
| 0 | 2026-01-05 | A | 3 | 2,5 |
| 1 | 2026-01-05 | B | 1 | 1 |
| 2 | 2026-01-05 | C | 3 | 2,5 |
| 3 | 2026-01-06 | A | NaN | NaN |
| 4 | 2026-01-06 | B | 2 | 1 |

Le 2026-01-05, les deux signaux égaux à 3 occupent les rangs 2 et 3 : chacun reçoit (2 + 3)/2 = 2,5.

**Ce que vise l'exercice.** Le rang par date s'obtient en une ligne, sans boucle ni `argsort` : c'est le geste de base de toute normalisation transversale (rang, z-score) et du coefficient d'information.
""",
        "dossier": "coding/seance1", "module_tests": "coding/seance1/tests_e2.py", "fonction": "rang_transversal",
        "cellules_initiales": [
            CELLULE_IMPORTS + "\n\nexemple = pd.DataFrame(\n    [(\"2026-01-05\", \"A\", 3.0), (\"2026-01-05\", \"B\", 1.0), (\"2026-01-05\", \"C\", 3.0),\n     (\"2026-01-06\", \"A\", np.nan), (\"2026-01-06\", \"B\", 2.0)],\n    columns=[\"date\", \"actif\", \"signal\"])\nexemple",
            "def rang_transversal(df: pd.DataFrame) -> pd.Series:\n    # À compléter\n    return pd.Series(np.nan, index=df.index)",
            "rang_transversal(exemple)",
        ],
        "tests_visibles": 2, "tests_caches": 8,
        "fichier_correction": "coding/seance1/_correction.py",
        "verif": {"script": "coding/seance1/tests_e2.py"},
    },
    {
        "id": "S1-E3", "type": "code", "section": "code", "theme": "pandas : statistique par groupe", "difficulte": 3, "temps_cible_s": 480,
        "enonce": """## Exercice 3 : corrélation par date

Définis `correlation_par_date(df)`.

**Entrée.** Un DataFrame aux colonnes `date`, `x`, `y` (nombres ou `NaN`), lignes dans un ordre quelconque.

**Sortie.** Une `Series` de flottants dont l'index est la liste **triée** des dates distinctes de `df`. Pour chaque date, on garde les lignes où `x` et `y` sont tous deux non manquants ; la valeur est la corrélation de Pearson entre `x` et `y` sur ces lignes, ou `NaN` s'il y en a moins de 3, si tous les `x` sont égaux ou si tous les `y` sont égaux. Un DataFrame vide donne une `Series` vide.

**Exemple.**

| date | x | y |
|---|---|---|
| 2026-01-06 | 1 | 5 |
| 2026-01-05 | 1 | 2 |
| 2026-01-05 | 2 | 4 |
| 2026-01-05 | 3 | 7 |
| 2026-01-06 | 2 | 3 |
| 2026-01-06 | NaN | 1 |
| 2026-01-07 | 4 | 1 |
| 2026-01-07 | 5 | 1 |
| 2026-01-07 | 6 | 1 |

Sortie : 2026-01-05 : @E3_correlation_0105:4@ (à 4 décimales) ; 2026-01-06 : NaN (2 lignes valides seulement) ; 2026-01-07 : NaN (tous les `y` égaux).

Calcul du 2026-01-05 : les moyennes sont 2 et 13/3 ; somme des produits des écarts @E3_cov_num:0@ ; sommes des carrés des écarts @E3_sxx:0@ pour `x` et 38/3 pour `y` ; corrélation 5 / racine(2 × 38/3).

**Ce que vise l'exercice.** Appliquer une fonction à chaque groupe et récupérer une valeur par groupe : c'est ce qui manquait pour passer de « un groupby » à « un coefficient d'information par date ». Attention aux versions : en pandas 1.4.2, `groupby("date").apply(f)` passe aussi la colonne `date` à `f`.
""",
        "dossier": "coding/seance1", "module_tests": "coding/seance1/tests_e3.py", "fonction": "correlation_par_date",
        "cellules_initiales": [
            CELLULE_IMPORTS + "\n\nexemple = pd.DataFrame(\n    [(\"2026-01-06\", 1.0, 5.0), (\"2026-01-05\", 1.0, 2.0), (\"2026-01-05\", 2.0, 4.0), (\"2026-01-05\", 3.0, 7.0),\n     (\"2026-01-06\", 2.0, 3.0), (\"2026-01-06\", np.nan, 1.0), (\"2026-01-07\", 4.0, 1.0), (\"2026-01-07\", 5.0, 1.0),\n     (\"2026-01-07\", 6.0, 1.0)],\n    columns=[\"date\", \"x\", \"y\"])\nexemple",
            "def correlation_par_date(df: pd.DataFrame) -> pd.Series:\n    # À compléter\n    return pd.Series(dtype=float)",
            "correlation_par_date(exemple)",
        ],
        "tests_visibles": 1, "tests_caches": 7,
        "fichier_correction": "coding/seance1/_correction.py",
        "verif": {"script": "coding/seance1/tests_e3.py"},
    },
    dict(diag.CODE, id="S1-E4", theme="pandas : coefficient d'information (reprise du test de positionnement)", temps_cible_s=1200,
         enonce=diag.CODE["enonce"].replace("## Coefficient d'information moyen",
                                            "## Exercice 4 : coefficient d'information moyen (reprise du test de positionnement)\n\n"
                                            "Les exercices 1 à 3 en sont les briques : rendement futur, rangs par date, corrélation par date.", 1)),
]

LECONS = [lecons.LECON_PANDAS]

QUESTIONS = []
