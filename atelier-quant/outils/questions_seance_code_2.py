"""Séance de code 2 : fenêtres glissantes par actif, rééchantillonnage et merge_asof, normalisation transversale,
mini-backtest sans lookahead.

Leçon : outils/lecons_code_2.py (L-PANDAS-2). Exercices, corrections et tests : coding/seance2/.
Valeurs des exemples et preuve que les tests cachés attrapent les erreurs visées : verif/seance_code_2.py.
Affirmations de la leçon sur pandas, vérifiées en 1.4.2 et en 2.3.3 : verif/lecon_pandas_2.py.
"""
import lecons_code_2
import questions_diagnostic as diag

ID = "seance-code-2"
TITRE = "Séance de code 2 : fenêtres glissantes, alignement temporel, normalisation, backtest"
VERIF = "verif/seance_code_2.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [{"id": "lecon", "nom": "Leçon", "budget_min": 20}, {"id": "code", "nom": "Coding, data science", "budget_min": 50}]
PRESENTATION = (
    "D'abord une leçon de 20 minutes, avec cinq points de contrôle corrigés immédiatement et un notebook pour essayer "
    "les exemples. Ensuite quatre exercices, du plus guidé au plus complet : une moyenne glissante par actif, un passage "
    "en fréquence mensuelle avec la dernière information disponible, une normalisation par date avec winsorisation, et "
    "un mini-backtest long-court dont on calcule le ratio de Sharpe. Chaque exercice a ses tests visibles et des tests "
    "cachés qui contiennent des cas limites : cadre vide, une seule ligne, valeurs manquantes, égalités, index non "
    "trivial, grande taille.\n\n"
    "Règle de travail, à appliquer aussi dans un test en ligne : écris une solution simple, lance les tests visibles, "
    "**soumets tout de suite**, puis améliore et soumets à nouveau. Un code jamais soumis ne rapporte rien. Au-delà du "
    "double du temps cible d'un exercice, soumets ce que tu as et passe au suivant : c'est la règle d'abandon à suivre dans un test chronométré."
)
AVERTISSEMENTS = [
    "Temps indicatif : 8 minutes pour l'exercice 1, 12 pour le 2, 10 pour le 3, 20 pour le 4. Dans les "
    "examens blancs, la question de code dispose de 24 minutes.",
    "Le notebook de cette séance s'exécute sous Python 3.9 et pandas 1.4.2, l'environnement des notebooks de la "
    "plateforme.",
    "Les points de contrôle de la leçon sont en choix multiple à une seule bonne réponse.",
]

CELLULE_IMPORTS = "import numpy as np\nimport pandas as pd"
FONCTIONNEMENT = (
    "\n\n**Fonctionnement.** Notebook sous Python 3.9 et pandas 1.4.2. Tolérance de comparaison des "
    "nombres : `1e-9` ; limite de temps : 10 secondes par test. **Tests visibles** exécute les tests de l'énoncé ; "
    "**Soumettre** exécute les tests cachés et n'affiche que leurs noms et leur résultat.\n"
)

CODES = [
    {
        "id": "S2-E1", "type": "code", "section": "code", "theme": "pandas : fenêtre glissante par actif", "difficulte": 2, "temps_cible_s": 480,
        "enonce": r"""## Exercice 1 : moyenne glissante par actif

Définis `moyenne_glissante(df, fenetre, min_periodes)`.

**Entrée.** `df` est un DataFrame aux colonnes `date` (chaîne `AAAA-MM-JJ`), `actif` (chaîne) et `rendement` (nombre ou `NaN`). Au plus une ligne par couple (`date`, `actif`) ; lignes dans un ordre **quelconque** ; index quelconque sans doublon. `fenetre` et `min_periodes` sont deux entiers vérifiant 1 ≤ `min_periodes` ≤ `fenetre`.

**Sortie.** Une `Series` de flottants **ayant le même index que `df`**. Pour la ligne d'actif `a` et de date `d` : on prend les lignes de `a` dont la date est inférieure ou égale à `d`, et parmi elles les `fenetre` plus récentes (toutes s'il y en a moins), la ligne elle-même comprise. Si au moins `min_periodes` d'entre elles ont un rendement non manquant, la valeur est la moyenne arithmétique de ces rendements non manquants ; sinon, `NaN`.

La fenêtre compte des **lignes de l'actif**, pas des jours du calendrier : une date sans ligne pour `a` n'occupe aucune place dans la fenêtre de `a`. Les lignes des autres actifs n'interviennent jamais.

**Exemple**, avec `fenetre = 3` et `min_periodes = 2`.

| index | date | actif | rendement | sortie |
|---|---|---|---|---|
| 0 | 2026-01-07 | A | 3 | (1 + 3)/2 = @E1_ex_0@ |
| 1 | 2026-01-05 | B | 10 | NaN (une seule ligne) |
| 2 | 2026-01-05 | A | 1 | NaN (une seule ligne) |
| 3 | 2026-01-08 | A | 5 | (3 + 5)/2 = @E1_ex_3@ |
| 4 | 2026-01-06 | A | NaN | NaN (une seule valeur non manquante, 1) |
| 5 | 2026-01-06 | B | 20 | (10 + 20)/2 = @E1_ex_5@ |

Ligne 3 : les trois lignes de A les plus récentes jusqu'au 2026-01-08 sont celles des 2026-01-06 (`NaN`), 2026-01-07 (3) et 2026-01-08 (5) ; deux valeurs non manquantes, donc la moyenne de 3 et 5. Ligne 0 : la fenêtre contient 1, `NaN` et 3.

**Ce que vise l'exercice.** Trier, grouper par actif, régler `min_periods`, puis revenir à l'index d'origine malgré l'index à deux niveaux que renvoie `groupby(...).rolling(...)` (leçon, section 2). Sur l'exemple trié par date, une fenêtre sans `groupby` donnerait à la ligne 0 la valeur @E1_fuite_ligne_0@, moyenne de B le 2026-01-06 et de A le 2026-01-07 : une fuite entre actifs.
""" + FONCTIONNEMENT.replace("les tests de l'énoncé", "2 tests visibles").replace("les tests cachés", "8 tests cachés"),
        "dossier": "coding/seance2", "module_tests": "coding/seance2/tests_e1.py", "fonction": "moyenne_glissante",
        "cellules_initiales": [
            CELLULE_IMPORTS + "\n\nexemple = pd.DataFrame(\n    [(\"2026-01-07\", \"A\", 3.0), (\"2026-01-05\", \"B\", 10.0), (\"2026-01-05\", \"A\", 1.0),\n     (\"2026-01-08\", \"A\", 5.0), (\"2026-01-06\", \"A\", np.nan), (\"2026-01-06\", \"B\", 20.0)],\n    columns=[\"date\", \"actif\", \"rendement\"])\nexemple",
            "def moyenne_glissante(df: pd.DataFrame, fenetre: int, min_periodes: int) -> pd.Series:\n    # À compléter\n    return pd.Series(np.nan, index=df.index)",
            "moyenne_glissante(exemple, 3, 2)",
        ],
        "tests_visibles": 2, "tests_caches": 8,
        "fichier_correction": "coding/seance2/_correction.py",
        "verif": {"script": "coding/seance2/tests_e1.py"},
    },
    {
        "id": "S2-E2", "type": "code", "section": "code", "theme": "pandas : rééchantillonnage et merge_asof", "difficulte": 4, "temps_cible_s": 720,
        "enonce": r"""## Exercice 2 : dernier prix du mois et dernier signal disponible

Définis `mensualiser(prix, signaux)`.

**Entrées.**

- `prix` : DataFrame aux colonnes `date` (chaîne `AAAA-MM-JJ`), `actif` (chaîne), `prix` (nombre strictement positif ou `NaN`). Au plus une ligne par couple (`date`, `actif`).
- `signaux` : DataFrame aux colonnes `date` (chaîne `AAAA-MM-JJ`, n'importe quel jour, week-ends compris), `actif` (chaîne), `signal` (nombre, jamais `NaN`). Au plus une ligne par couple (`date`, `actif`). Un signal daté du jour `j` est publié **après la clôture** de `j`.

Les deux cadres ont des lignes dans un ordre quelconque, un index quelconque, et peuvent être vides (zéro ligne, colonnes présentes).

**Sortie.** Un DataFrame avec **une ligne par couple (actif, mois civil)** tel que l'actif a au moins un prix non manquant dans ce mois, et exactement les colonnes suivantes, dans cet ordre :

| Colonne | Contenu |
|---|---|
| `actif` | l'actif |
| `fin_de_mois` | le dernier jour civil du mois, chaîne `AAAA-MM-JJ` (par exemple 2026-02-28) |
| `date` | la plus grande date du mois où l'actif a un prix **non manquant** |
| `prix` | ce prix |
| `signal` | le signal de l'actif dont la date est la plus grande date **strictement antérieure** à `date` ; `NaN` s'il n'y en a pas |

Lignes triées par `actif` puis par `fin_de_mois` (ordre croissant des chaînes). L'index n'est pas vérifié. Les colonnes de dates peuvent être des chaînes `AAAA-MM-JJ` ou des `Timestamp`. Entrée sans aucun prix non manquant : un DataFrame vide avec ces cinq colonnes.

L'idée : la décision mensuelle est prise à la clôture de la dernière observation du mois, et ne peut utiliser qu'un signal publié avant cette clôture, donc daté d'un jour strictement antérieur.

**Exemple.**

`prix` :

| date | actif | prix |
|---|---|---|
| 2026-01-30 | A | 11 |
| 2026-03-31 | B | NaN |
| 2026-01-29 | A | 10 |
| 2026-01-30 | B | 20 |
| 2026-02-27 | A | 12 |
| 2026-03-30 | B | 21 |

`signaux` :

| date | actif | signal |
|---|---|---|
| 2026-01-30 | A | 0.7 |
| 2026-03-30 | B | 2.0 |
| 2026-01-29 | A | 0.5 |
| 2026-01-31 | B | -1.0 |

Sortie :

| actif | fin_de_mois | date | prix | signal |
|---|---|---|---|---|
| A | 2026-01-31 | 2026-01-30 | 11 | 0.5 |
| A | 2026-02-28 | 2026-02-27 | 12 | 0.7 |
| B | 2026-01-31 | 2026-01-30 | 20 | NaN |
| B | 2026-03-31 | 2026-03-30 | 21 | -1.0 |

Lecture : en janvier, A a pour dernière observation le 2026-01-30 ; son signal du même jour est exclu, celui du 29 est retenu. En février, le signal du 30 janvier est le dernier strictement antérieur au 27 février. B n'a aucun prix en février : pas de ligne. En mars, le prix du 31 manque, la dernière observation est donc le 30, et le signal du 30 est exclu : il reste celui du 31 janvier (un samedi). B en janvier : aucun signal de B avant le 30 janvier.

**Ce que vise l'exercice.** `resample("M")` par actif après avoir retiré les prix manquants, `dropna` des mois vides, puis `merge_asof` avec `by` et `allow_exact_matches=False` sur des colonnes converties en dates (leçon, sections 3 et 4). Une solution par `groupby` sur le mois et une recherche à la main est aussi acceptée, si elle tient dans le temps limite.
""" + FONCTIONNEMENT.replace("les tests de l'énoncé", "2 tests visibles").replace("les tests cachés", "8 tests cachés"),
        "dossier": "coding/seance2", "module_tests": "coding/seance2/tests_e2.py", "fonction": "mensualiser",
        "cellules_initiales": [
            CELLULE_IMPORTS + "\n\nprix = pd.DataFrame(\n    [(\"2026-01-30\", \"A\", 11.0), (\"2026-03-31\", \"B\", np.nan), (\"2026-01-29\", \"A\", 10.0),\n     (\"2026-01-30\", \"B\", 20.0), (\"2026-02-27\", \"A\", 12.0), (\"2026-03-30\", \"B\", 21.0)],\n    columns=[\"date\", \"actif\", \"prix\"])\nsignaux = pd.DataFrame(\n    [(\"2026-01-30\", \"A\", 0.7), (\"2026-03-30\", \"B\", 2.0), (\"2026-01-29\", \"A\", 0.5), (\"2026-01-31\", \"B\", -1.0)],\n    columns=[\"date\", \"actif\", \"signal\"])\nprix, signaux",
            "def mensualiser(prix: pd.DataFrame, signaux: pd.DataFrame) -> pd.DataFrame:\n    # À compléter\n    return pd.DataFrame(columns=[\"actif\", \"fin_de_mois\", \"date\", \"prix\", \"signal\"])",
            "mensualiser(prix, signaux)",
        ],
        "tests_visibles": 2, "tests_caches": 8,
        "fichier_correction": "coding/seance2/_correction.py",
        "verif": {"script": "coding/seance2/tests_e2.py"},
    },
    {
        "id": "S2-E3", "type": "code", "section": "code", "theme": "pandas : normalisation transversale et winsorisation", "difficulte": 3, "temps_cible_s": 600,
        "enonce": r"""## Exercice 3 : winsorisation, z-score et rang par date

Définis `normaliser(df, q_bas, q_haut)`.

**Entrée.** `df` est un DataFrame aux colonnes `date` (chaîne), `actif` (chaîne), `signal` (nombre ou `NaN`), lignes dans un ordre quelconque, index quelconque sans doublon, éventuellement vide. `q_bas` et `q_haut` sont deux nombres vérifiant 0 ≤ `q_bas` < `q_haut` ≤ 1.

**Calcul, date par date.** Pour une date, soit \(n\) le nombre de ses signaux **non manquants**, et \(x_{(1)} \le \dots \le x_{(n)}\) ces signaux triés.

1. **Quantiles.** Pour un niveau \(q\), on pose \(h = (n - 1)\,q\), \(i = \lfloor h \rfloor\) et \(f = h - i\) ; le quantile vaut \(x_{(i+1)} + f\,\bigl(x_{(i+2)} - x_{(i+1)}\bigr)\), ou \(x_{(i+1)}\) si \(f = 0\). On note \(b\) le quantile de niveau `q_bas` et \(c\) celui de niveau `q_haut`.
2. **Winsorisation.** Le signal winsorisé d'une ligne de signal \(x\) est \(w = \min(\max(x, b), c)\).
3. **z-score.** \(z = (w - \bar w)/s\), où \(\bar w\) est la moyenne des \(n\) valeurs winsorisées de la date et \(s\) leur écart-type à dénominateur \(n - 1\). `NaN` si \(n < 2\) ou si les \(n\) valeurs winsorisées sont toutes égales.
4. **Rang.** Rang du signal **brut** parmi les \(n\) signaux non manquants de la date, par ordre croissant à partir de 1, les valeurs égales recevant la moyenne des rangs qu'elles occupent, puis divisé par \(n\).

**Sortie.** Un DataFrame **ayant le même index que `df`**, aux colonnes `w`, `z`, `rang` dans cet ordre, de flottants. Une ligne de signal `NaN` a `NaN` dans les trois colonnes.

**Exemple.**

| index | date | actif | signal |
|---|---|---|---|
| 0 | 2026-01-05 | A | 3 |
| 1 | 2026-01-06 | A | 7 |
| 2 | 2026-01-05 | B | 100 |
| 3 | 2026-01-05 | C | 1 |
| 4 | 2026-01-07 | A | 5 |
| 5 | 2026-01-05 | D | 4 |
| 6 | 2026-01-05 | E | NaN |
| 7 | 2026-01-05 | F | 2 |
| 8 | 2026-01-07 | B | 5 |

Avec `q_bas = 0.25` et `q_haut = 0.75` : le 2026-01-05, les signaux triés sont 1, 2, 3, 4, 100, donc \(n = 5\) ; \(h = @E3_q25_h:0@\) donne \(b = @E3_q25:0@\), \(h = @E3_q75_h:0@\) donne \(c = @E3_q75:0@\). Valeurs winsorisées : A 3, B 4, C 2, D 4, F 2, de moyenne 3 et d'écart-type 1 (somme des carrés des écarts 4, divisée par 4).

| index | w | z | rang |
|---|---|---|---|
| 0 | 3 | 0 | 0.6 |
| 1 | 7 | NaN (une seule valeur) | 1 |
| 2 | 4 | 1 | 1 |
| 3 | 2 | -1 | 0.2 |
| 4 | 5 | NaN (valeurs égales) | 0.75 |
| 5 | 4 | 1 | 0.8 |
| 6 | NaN | NaN | NaN |
| 7 | 2 | -1 | 0.4 |
| 8 | 5 | NaN (valeurs égales) | 0.75 |

Avec `q_bas = 0.1` et `q_haut = 0.9`, le 2026-01-05 : \(h = @E3_q10_h:1m@\), \(b = 1 + @E3_q10_h:1m@ \times (2 - 1) = @E3_q10:1m@\) ; \(h = @E3_q90_h:1m@\), \(c = 4 + @E3_q90_f:1m@ \times (100 - 4) = @E3_q90:1m@\). Valeurs winsorisées : A 3, B @E3_q90:1@, C @E3_q10:1@, D 4, F 2, de moyenne @E3_ex10_moyenne:2@ et d'écart-type @E3_ex10_et:4@ ; z-scores : A @E3_ex10_z_A:4@, B @E3_ex10_z_B:4@, C @E3_ex10_z_C:4@, D @E3_ex10_z_D:4@, F @E3_ex10_z_F:4@ (à 4 décimales). Les rangs ne changent pas.

**Ce que vise l'exercice.** Toutes les statistiques sont **par date** (leçon, section 5). Pièges couverts par les tests cachés : quantiles calculés sur tout le cadre, écart-type à dénominateur \(n\) (qui donnerait à B le z-score @E3_ex10_z_B_ddof0:4@ au lieu de @E3_ex10_z_B:4@), rang calculé sur le signal winsorisé, valeurs toutes égales dont l'écart-type calculé n'est pas exactement nul.
""" + FONCTIONNEMENT.replace("les tests de l'énoncé", "2 tests visibles").replace("les tests cachés", "8 tests cachés"),
        "dossier": "coding/seance2", "module_tests": "coding/seance2/tests_e3.py", "fonction": "normaliser",
        "cellules_initiales": [
            CELLULE_IMPORTS + "\n\nexemple = pd.DataFrame(\n    [(\"2026-01-05\", \"A\", 3.0), (\"2026-01-06\", \"A\", 7.0), (\"2026-01-05\", \"B\", 100.0), (\"2026-01-05\", \"C\", 1.0),\n     (\"2026-01-07\", \"A\", 5.0), (\"2026-01-05\", \"D\", 4.0), (\"2026-01-05\", \"E\", np.nan), (\"2026-01-05\", \"F\", 2.0),\n     (\"2026-01-07\", \"B\", 5.0)],\n    columns=[\"date\", \"actif\", \"signal\"])\nexemple",
            "def normaliser(df: pd.DataFrame, q_bas: float, q_haut: float) -> pd.DataFrame:\n    # À compléter\n    return pd.DataFrame(np.nan, index=df.index, columns=[\"w\", \"z\", \"rang\"])",
            "normaliser(exemple, 0.25, 0.75)",
        ],
        "tests_visibles": 2, "tests_caches": 8,
        "fichier_correction": "coding/seance2/_correction.py",
        "verif": {"script": "coding/seance2/tests_e3.py"},
    },
    {
        "id": "S2-E4", "type": "code", "section": "code", "theme": "pandas : mini-backtest long-court sans lookahead", "difficulte": 5, "temps_cible_s": 1200,
        "enonce": r"""## Exercice 4 : mini-backtest long-court et ratio de Sharpe

Définis `backtest(df, nb_groupes)`, qui renvoie un couple `(rendements, sharpe)`.

**Entrée.** `df` est un DataFrame aux colonnes `date` (chaîne `AAAA-MM-JJ`), `actif` (chaîne), `prix` (nombre strictement positif ou `NaN`) et `signal` (nombre ou `NaN`). Au plus une ligne par couple (`date`, `actif`) ; lignes dans un ordre quelconque ; index quelconque ; éventuellement vide. `nb_groupes` est un entier au moins égal à 2.

**Conventions.**

1. **Calendrier.** Les dates du calendrier sont les dates distinctes de `df`, triées. Pour une date \(t\) du calendrier autre que la dernière, \(t+1\) désigne la date **suivante du calendrier** (pas la ligne suivante de l'actif).
2. **Éligibles.** Un actif est éligible en \(t\) si `df` a une ligne (\(t\), actif) dont le prix **et** le signal sont non manquants. Rien d'autre n'intervient : en particulier, l'existence d'un prix en \(t+1\) ne compte pas. On note \(n_t\) le nombre d'éligibles et \(k_t\) le quotient de la division entière de \(n_t\) par `nb_groupes`.
3. **Positions.** On classe les éligibles par signal croissant ; à signal égal, par nom d'actif croissant. Les \(k_t\) premiers sont **vendus** (poids \(-1/k_t\) chacun), les \(k_t\) derniers **achetés** (poids \(+1/k_t\) chacun), les autres ont un poids nul. Si \(k_t = 0\), aucune position.
4. **Rendement réalisé.** Un actif en position en \(t\), de prix \(p_t\), rapporte \(p_{t+1}/p_t - 1\), où \(p_{t+1}\) est son prix en \(t+1\) ; ce rendement vaut 0 si l'actif n'a pas de ligne en \(t+1\) ou si son prix y est `NaN`.
5. **Rendement du portefeuille** en \(t\) : la somme des poids multipliés par les rendements réalisés, soit la moyenne des rendements achetés moins la moyenne des rendements vendus ; 0 si \(k_t = 0\).
6. **Sharpe.** Pour la série des \(m\) rendements du portefeuille, de moyenne \(\bar r\) et d'écart-type \(s\) à dénominateur \(m - 1\) : \(\sqrt{252}\,\bar r / s\) (252 dates par an, par convention). `NaN` si \(m < 2\) ou si les \(m\) rendements sont tous égaux.

**Sortie.** `rendements` : une `Series` de flottants indexée par les dates \(t\) du calendrier **sauf la dernière**, dans l'ordre chronologique (index en chaînes `AAAA-MM-JJ` ou en `Timestamp`) ; la valeur en \(t\) est le rendement du portefeuille décidé en \(t\) et réalisé de \(t\) à \(t+1\). `sharpe` : un `float`. Calendrier de moins de deux dates : `Series` vide et `NaN`.

**Exemple**, avec `nb_groupes = 2`.

| date | actif | prix | signal |
|---|---|---|---|
| 2026-01-05 | A | 100 | 0.9 |
| 2026-01-05 | B | 50 | 0.1 |
| 2026-01-05 | C | 20 | -0.5 |
| 2026-01-05 | D | 10 | 0.3 |
| 2026-01-06 | A | 110 | 0.2 |
| 2026-01-06 | B | 50 | 0.2 |
| 2026-01-06 | C | 19 | 0.4 |
| 2026-01-06 | D | NaN | 1.0 |
| 2026-01-07 | A | 99 | NaN |
| 2026-01-07 | B | 55 | 0.0 |
| 2026-01-07 | C | 19 | -0.1 |
| 2026-01-07 | D | 12 | 0.5 |

(Dans les tests, ces lignes sont mélangées.)

- 2026-01-05 : 4 éligibles, \(k = 2\). Ordre : C, B, D, A. Vendus C et B, achetés D et A. Rendements : A \(110/100 - 1 = @E4_r_A:2m@\), D 0 (prix manquant le 2026-01-06), B 0, C \(19/20 - 1 = @E4_r_C:2m@\). Portefeuille : \(@E4_long_0105:3m@ - (@E4_court_0105:4m@) = @E4_pf_0105:4m@\).
- 2026-01-06 : D n'est pas éligible (prix manquant), 3 éligibles, \(k = 1\). A et B ont le même signal 0.2 : A est classé avant B. Vendu A, acheté C. Rendements : C 0, A \(99/110 - 1 = @E4_r_A_0106:1m@\). Portefeuille : \(0 - (@E4_r_A_0106:1m@) = @E4_pf_0106:1m@\).
- 2026-01-07 : dernière date, pas de rendement.

Sortie : `rendements` vaut @E4_pf_0105@ au 2026-01-05 et @E4_pf_0106@ au 2026-01-06 ; moyenne @E4_moyenne@, écart-type @E4_et:6@, `sharpe` \(= \sqrt{252} \times @E4_moyenne:4m@ / @E4_et:6m@ \approx @E4_sharpe:4m@\). Ce Sharpe absurde vient de deux dates seulement.

Si l'on avait écarté D le 2026-01-05 parce que son prix du lendemain manque, on aurait eu 3 éligibles, \(k = 1\), et le rendement @E4_pf_0105_lookahead@ : la sélection aurait utilisé une information du 2026-01-06.

**Fichier d'exploration.** `panel.csv` (@panel_lignes@ lignes, @panel_dates@ dates, @panel_actifs@ actifs, mêmes colonnes) contient un signal synthétique fabriqué avec une part du rendement du lendemain. Avec `nb_groupes = 5`, la correction y donne un Sharpe de @E4_sharpe_panel:4@ : de quoi contrôler ton résultat, et un exemple de ce qu'un lookahead produit.

**Ce que vise l'exercice.** Le format large par `pivot`, le rendement de \(t\) à \(t+1\) par `shift(-1)` sur le calendrier, la sélection avec la seule information de \(t\), le classement avec départage, puis le Sharpe annualisé (leçon, section 6). Commence par une version simple et lente, une boucle sur les dates, puis vectorise si le temps le permet.
""" + FONCTIONNEMENT.replace("les tests de l'énoncé", "2 tests visibles").replace("les tests cachés", "8 tests cachés"),
        "dossier": "coding/seance2", "module_tests": "coding/seance2/tests_e4.py", "fonction": "backtest",
        "cellules_initiales": [
            CELLULE_IMPORTS + "\n\nexemple = pd.DataFrame(\n    [(\"2026-01-05\", \"A\", 100.0, 0.9), (\"2026-01-05\", \"B\", 50.0, 0.1), (\"2026-01-05\", \"C\", 20.0, -0.5),\n     (\"2026-01-05\", \"D\", 10.0, 0.3), (\"2026-01-06\", \"A\", 110.0, 0.2), (\"2026-01-06\", \"B\", 50.0, 0.2),\n     (\"2026-01-06\", \"C\", 19.0, 0.4), (\"2026-01-06\", \"D\", np.nan, 1.0), (\"2026-01-07\", \"A\", 99.0, np.nan),\n     (\"2026-01-07\", \"B\", 55.0, 0.0), (\"2026-01-07\", \"C\", 19.0, -0.1), (\"2026-01-07\", \"D\", 12.0, 0.5)],\n    columns=[\"date\", \"actif\", \"prix\", \"signal\"])\npanel = pd.read_csv(\"panel.csv\")\nexemple",
            "def backtest(df: pd.DataFrame, nb_groupes: int):\n    # À compléter : renvoyer (rendements, sharpe)\n    return pd.Series(dtype=float), float(\"nan\")",
            "backtest(exemple, 2)",
            "backtest(panel, 5)[1]",
        ],
        "tests_visibles": 2, "tests_caches": 8,
        "fichier_correction": "coding/seance2/_correction.py",
        "verif": {"script": "coding/seance2/tests_e4.py"},
    },
]

LECONS = [lecons_code_2.LECON_PANDAS_2]

QUESTIONS = []
