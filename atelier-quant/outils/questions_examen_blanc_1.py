"""Examen blanc 1 : format d'un test en ligne de recherche quantitative (1 question de code, 10 de mathématiques, 5 de ML fondamental, 8 de ML avancé).

Questions toutes nouvelles par rapport aux banques existantes (test de positionnement, séances de code 1, mathématiques 1,
ML 1 et 2, algèbre 1). Pas de leçon. Toutes les valeurs : verif/examen_blanc_1.py.
Dans chaque question, l'option "a" est la bonne ; l'ordre d'affichage est mélangé par la page.
"""
import questions_diagnostic as diag

ID = "examen-blanc-1"
TITRE = "Examen blanc 1"
VERIF = "verif/examen_blanc_1.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0, "pause": False, "je_ne_sais_pas": False}
SECTIONS = [
    {"id": "code", "nom": "Coding, data science", "budget_min": 24},
    {"id": "math", "nom": "Mathématiques", "budget_min": 25},
    {"id": "mlf", "nom": "Machine Learning fondamental", "budget_min": 7},
    {"id": "mla", "nom": "Machine Learning avancé", "budget_min": 16},
]
PRESENTATION = (
    "Examen blanc au format d'un test en ligne de recherche quantitative : 1 question de code, 10 de mathématiques, "
    "5 de ML fondamental, 8 de ML avancé. Passe-le en conditions réelles : d'une traite, seul, sans notes, sans "
    "recherche et sans pause. Le chronomètre affiche 72 minutes, la somme des budgets de section (code 24, "
    "mathématiques 25, ML fondamental 7, ML avancé 16). Le format de référence dure 82 minutes : les 10 minutes "
    "d'écart sont la marge à garder, et l'objectif est de finir avant la fin du chronomètre.\n\n"
    "Ordre conseillé : la question de code d'abord, en soumettant tôt une solution simple et correcte, puis les "
    "questions à choix multiple. Règle d'abandon : au-delà du double du temps cible d'une question, choisis la réponse "
    "la plus plausible après élimination des options impossibles, passe à la suivante, et reviens-y avec la marge s'il "
    "en reste. Il n'y a ni pause ni « Je ne sais pas » : sans pénalité, une question laissée vide est un point perdu à coup sûr."
)
AVERTISSEMENTS = [
    "Les questions de mathématiques et de ML sont présentées ici en choix multiple à une seule bonne réponse. Cherche "
    "la réponse avant de lire les options, pour t'entraîner aussi au format à réponse numérique.",
    "Ici il n'y a pas de pénalité et le retour en arrière est permis dans tout l'examen : ce sont les règles de cette "
    "plateforme, pas celles de tout test en ligne.",
    "Calculs : quand une valeur d'exponentielle, de logarithme ou de quantile est nécessaire, elle est donnée dans "
    "l'énoncé. Avant un test réel, vérifie si une calculatrice ou un brouillon sont autorisés.",
]

LECONS = []

CODES = [{
    "id": "EB1-C1", "type": "code", "section": "code", "theme": "pandas : décalages par actif, tri par date, stratégie long-short",
    "difficulte": 4, "temps_cible_s": 1200,
    "enonce": """## Rendement moyen d'une stratégie long-short de momentum

Une stratégie de momentum achète, à chaque date, les actifs qui viennent de le plus monter et vend ceux qui viennent de le plus baisser, puis mesure ce que chaque position rapporte jusqu'à la date suivante de l'actif.

Dans le notebook, définis la fonction `rendement_long_short(df, k)` qui renvoie un `float`.

### Entrée

`df` est un DataFrame pandas dont les colonnes sont exactement :

| Colonne | Type | Contenu |
|---|---|---|
| `date` | chaîne au format `AAAA-MM-JJ` | date de l'observation |
| `actif` | chaîne | identifiant de l'actif |
| `prix` | nombre, strictement positif ou `NaN` | prix de clôture de l'actif à cette date |

`k` est un entier au moins égal à 1.

Garanties : au plus une ligne par couple (`date`, `actif`) ; lignes dans un ordre **quelconque**, index quelconque ; le DataFrame peut être vide (zéro ligne, mais les trois colonnes présentes) ; tous les actifs n'ont pas nécessairement une ligne à chaque date.

### Calcul demandé

1. **Lignes précédente et suivante.** Pour une ligne (date `d`, actif `a`), la ligne précédente est la ligne de `a` dont la date est la plus grande date strictement inférieure à `d` parmi les lignes de `a` ; la ligne suivante est la ligne de `a` dont la date est la plus petite date strictement supérieure à `d`. Si `a` n'a pas de ligne à une date, on passe directement à sa ligne disponible suivante.
2. **Rendement passé** d'une ligne : `p / p_prec - 1`, où `p` est son prix et `p_prec` celui de sa ligne précédente. Il est manquant (`NaN`) si la ligne précédente n'existe pas, ou si `p` ou `p_prec` est `NaN`.
3. **Rendement futur** d'une ligne : `p_suiv / p - 1`, où `p_suiv` est le prix de sa ligne suivante. Il est manquant si la ligne suivante n'existe pas, ou si `p` ou `p_suiv` est `NaN`.
4. **Lignes exploitables.** Une ligne est exploitable si son rendement passé et son rendement futur sont tous deux non manquants.
5. **Rendement d'une date.** Pour chaque date, on ne garde que ses lignes exploitables. S'il y en a moins de `2k`, la date est ignorée. Sinon, on range ces lignes par rendement passé croissant et, à rendement passé égal, par identifiant d'actif croissant (ordre des chaînes de Python). La **jambe vendeuse** est formée des `k` premières lignes, la **jambe acheteuse** des `k` dernières. Le rendement de la date est la moyenne des rendements futurs de la jambe acheteuse moins la moyenne des rendements futurs de la jambe vendeuse.
6. **Résultat.** La moyenne arithmétique des rendements des dates non ignorées, sous forme de `float` ; `float("nan")` si toutes les dates sont ignorées.

Les égalités de rendement passé sont des égalités exactes entre flottants calculés par la formule du point 2 : par exemple `110 / 100 - 1` et `55 / 50 - 1` sont égaux. Tolérance de comparaison : `1e-9`. Limite de temps : 10 secondes par test.

### Exemple, avec `k = 1`

Le DataFrame a 12 lignes, une par case de prix de ce tableau :

| date | prix de A | prix de B | prix de C |
|---|---|---|---|
| 2026-01-05 | 100 | 50 | 20 |
| 2026-01-06 | 110 | 45 | 21 |
| 2026-01-07 | 99 | 54 | 21 |
| 2026-01-08 | 99 | 54 | 23,1 |

- Le 2026-01-05 n'a aucun rendement passé et le 2026-01-08 aucun rendement futur : ces deux dates n'ont aucune ligne exploitable et sont ignorées.
- 2026-01-06 : rendements passés de (A, B, C) = @C_passe_0106@, rendements futurs = @C_futur_0106@. Jambe acheteuse A, jambe vendeuse B : rendement de la date -0,1 - 0,2 = @C_date_0106@.
- 2026-01-07 : rendements passés = @C_passe_0107@, rendements futurs = @C_futur_0107@. Jambe acheteuse B, jambe vendeuse A : rendement @C_date_0107@.
- Résultat : (-0,3 + 0) / 2 = @C_exemple:2@ (à l'arrondi flottant près).

Exemple d'égalité (deuxième test visible) : si A passe de 100 à 110 et B de 50 à 55, leurs rendements passés sont égaux ; dans l'ordre du point 5, A vient avant B, donc avec `k = 1` c'est B, et non A, qui forme la jambe acheteuse si ces deux rendements sont les plus élevés.

### Fonctionnement

- Le notebook s'exécute avec Python 3.9 et pandas 1.4.2, l'environnement des notebooks de la plateforme.
- Le fichier `prix_panel.csv` (mêmes colonnes) est dans le répertoire courant pour explorer : `pd.read_csv("prix_panel.csv")`.
- **Tests visibles** exécute les @C_nb_visibles@ tests visibles. **Soumettre** exécute les @C_nb_caches@ tests cachés : seul le nombre de réussites s'affiche, et la dernière soumission compte. Un code jamais soumis ne rapporte rien.
- Budget : 24 minutes au plus, objectif 20. Écris d'abord une solution simple et correcte, lance les tests visibles, **soumets**, puis améliore et soumets à nouveau.
""",
    "dossier": "coding/examen1",
    "module_tests": "coding/examen1/tests.py",
    "fonction": "rendement_long_short",
    "cellules_initiales": [
        "import numpy as np\nimport pandas as pd\n\ndf = pd.read_csv(\"prix_panel.csv\")\ndf.head()",
        "def rendement_long_short(df: pd.DataFrame, k: int) -> float:\n    # À compléter\n    return float(\"nan\")",
        "rendement_long_short(df, 2)",
    ],
    "tests_visibles": 3,
    "tests_caches": 14,
    "fichier_correction": "coding/examen1/_correction.py",
    "verif": {"script": "coding/examen1/tests.py"},
}]

QUESTIONS = [
    # ================================================================== MATHÉMATIQUES
    {
        "id": "EB1-M01", "section": "math", "theme": "Marche aléatoire, théorème du scrutin (réflexion)", "difficulte": 4, "temps_cible_s": 150,
        "enonce": r"""Une élection oppose deux candidats, A et B. A obtient 7 voix et B 4 voix, soit 11 bulletins. On dépouille les bulletins un par un, dans un ordre tiré uniformément au hasard : les \(11!\) ordres possibles des 11 bulletins, physiquement distincts, sont équiprobables.

Quelle est la probabilité que A ait strictement plus de voix que B à chaque instant du dépouillement, c'est-à-dire après le dépouillement de chacun des 11 bulletins ?""",
        "options": [
            {"id": "a", "texte": "@M01_exact_num@/@M01_exact_den@", "explication": r"Oui : \((a - b)/(a + b)\), par le principe de réflexion."},
            {"id": "b", "texte": "@M01_d_premier_num@/@M01_d_premier_den@", "explication": r"C'est la probabilité que le premier bulletin soit pour A : une condition nécessaire, pas suffisante. Parmi les @M01_nb_debut_a@ suites qui commencent par A, @M01_nb_debut_b@ repassent par l'égalité."},
            {"id": "c", "texte": "@M01_d_large_num@/@M01_d_large_den@", "explication": r"C'est la probabilité que A ne soit jamais **derrière** B, égalités permises, \((a + 1 - b)/(a + 1)\). La question exige que A soit strictement devant : une égalité en cours de dépouillement suffit à échouer."},
            {"id": "d", "texte": "@M01_d_cond_num@/@M01_d_cond_den@", "explication": r"C'est \(@M01_nb_bons@/@M01_nb_debut_a@\), la probabilité conditionnelle sachant que le premier bulletin est pour A. Il faut encore multiplier par la probabilité que le premier bulletin soit pour A, \(7/11\)."},
        ],
        "solution": r"""Réponse : \(\tfrac{a - b}{a + b} = \tfrac{@M01_exact_num@}{@M01_exact_den@} \approx @M01_exact:3m@\).

Notations : \(a = 7\), \(b = 4\). On code le dépouillement par un mot de 11 lettres A ou B ; \(S_j\) est l'avance de A sur B après \(j\) bulletins (nombre de A moins nombre de B parmi les \(j\) premières lettres), avec \(S_0 = 0\) et \(S_{11} = 3\). L'événement demandé est « \(S_j > 0\) pour \(j = 1, \dots, 11\) ».

Étape 1, les mots sont équiprobables : chaque mot provient d'exactement \(7!\,4!\) ordres des bulletins (on permute entre eux les bulletins de A, et ceux de B). Il y a \(\binom{11}{4} = @M01_nb_total@\) mots.

Étape 2 : un bon mot commence par A, sinon \(S_1 = -1\). Il y a \(\binom{10}{4} = @M01_nb_debut_a@\) mots qui commencent par A et \(\binom{10}{3} = @M01_nb_debut_b@\) qui commencent par B.

Étape 3, réflexion : un mot qui commence par B passe par \(S_j = 0\), car \(S\) part de \(-1\), finit en 3 et varie de 1 à chaque pas. Pour un mot qui touche 0, soit \(\tau \ge 1\) le premier \(j\) tel que \(S_j = 0\). Échanger A et B dans les \(\tau\) premières lettres remplace \(S_j\) par \(-S_j\) pour \(j \le \tau\) et ne change pas \(S_j\) après \(\tau\) (puisque \(S_\tau = 0\)) ; le premier zéro reste \(\tau\). Cette opération envoie les mots qui commencent par A et touchent 0 sur les mots qui commencent par B, et elle est sa propre réciproque : c'est une bijection. Il y a donc @M01_nb_debut_b@ mauvais mots qui commencent par A.

Étape 4 : bons mots \(= @M01_nb_debut_a@ - @M01_nb_debut_b@ = @M01_nb_bons@\), et \(\mathbb{P} = @M01_nb_bons@/@M01_nb_total@ = \tfrac{@M01_exact_num@}{@M01_exact_den@}\). Le même calcul donne \((a - b)/(a + b)\) pour tous \(a > b\) (théorème du scrutin).

Vérification (script) : énumération des @M01_nb_total@ mots, et simulation : @M01_mc:4@.""",
    },
    {
        "id": "EB1-M02", "section": "math", "theme": "Espérance et variance conditionnelles, variance totale", "difficulte": 3, "temps_cible_s": 120,
        "enonce": r"""On lance un dé équilibré à 6 faces ; on note \(N\) le résultat. On lance ensuite \(N\) pièces équilibrées, indépendantes entre elles et indépendantes du dé, et on note \(X\) le nombre de piles obtenus.

Que vaut \(\mathrm{Var}(X)\), arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@M02_exact:3@", "explication": r"Oui : \(\mathbb{E}[N]/4 + \mathrm{Var}(N)/4\)."},
            {"id": "b", "texte": "@M02_e_var:3@", "explication": r"C'est \(\mathbb{E}[\mathrm{Var}(X \mid N)] = \mathbb{E}[N]/4\) seulement. Il manque la variance due au hasard de \(N\) lui-même, \(\mathrm{Var}(\mathbb{E}[X \mid N]) = \mathrm{Var}(N)/4\)."},
            {"id": "c", "texte": "@M02_var_e:3@", "explication": r"C'est \(\mathrm{Var}(\mathbb{E}[X \mid N]) = \mathrm{Var}(N)/4\) seulement. Il manque la variance du nombre de piles à \(N\) fixé, dont l'espérance est \(\mathbb{E}[N]/4\)."},
            {"id": "d", "texte": "@M02_d_demi:3@", "explication": r"Le facteur 1/2 de \(\mathbb{E}[X \mid N] = N/2\) n'a pas été élevé au carré : \(\mathrm{Var}(N/2) = \mathrm{Var}(N)/4\), pas \(\mathrm{Var}(N)/2\)."},
        ],
        "solution": r"""Réponse : \(\mathrm{Var}(X) = \tfrac{@M02_exact_num@}{@M02_exact_den@} \approx @M02_exact:3m@\).

Étape 1, loi conditionnelle : sachant \(N = n\), \(X\) suit la loi binomiale de paramètres \(n\) et 1/2, donc \(\mathbb{E}[X \mid N] = N/2\) et \(\mathrm{Var}(X \mid N) = N/4\).

Étape 2, formule de la variance totale : \(\mathrm{Var}(X) = \mathbb{E}[\mathrm{Var}(X \mid N)] + \mathrm{Var}(\mathbb{E}[X \mid N])\). Preuve : \(\mathbb{E}[X^2] = \mathbb{E}\big[\mathbb{E}[X^2 \mid N]\big] = \mathbb{E}\big[\mathrm{Var}(X \mid N)\big] + \mathbb{E}\big[\mathbb{E}[X \mid N]^2\big]\), et \(\mathbb{E}[X]^2 = \mathbb{E}\big[\mathbb{E}[X \mid N]\big]^2\) ; la différence des deux derniers termes est \(\mathrm{Var}(\mathbb{E}[X \mid N])\).

Étape 3, moments du dé : \(\mathbb{E}[N] = 7/2\), \(\mathbb{E}[N^2] = \tfrac{@M02_e_n2_num@}{@M02_e_n2_den@}\), \(\mathrm{Var}(N) = \tfrac{@M02_e_n2_num@}{@M02_e_n2_den@} - \tfrac{@M02_e_n_carre_num@}{@M02_e_n_carre_den@} = \tfrac{@M02_var_n_num@}{@M02_var_n_den@}\).

Étape 4 : \(\mathbb{E}[N]/4 = \tfrac{@M02_e_var_num@}{@M02_e_var_den@}\) et \(\mathrm{Var}(N)/4 = \tfrac{@M02_var_e_num@}{@M02_var_e_den@}\), d'où \(\mathrm{Var}(X) = \tfrac{@M02_exact_num@}{@M02_exact_den@} \approx @M02_exact:3m@\).

Vérification (script) : loi exacte de \(X\) en fractions, et simulation : @M02_mc:3@.""",
    },
    {
        "id": "EB1-M03", "section": "math", "theme": "Processus de Poisson, amincissement", "difficulte": 4, "temps_cible_s": 150,
        "enonce": r"""Des ordres arrivent sur un carnet selon un processus de Poisson d'intensité 3 ordres par minute. Indépendamment des instants d'arrivée et des autres ordres, chaque ordre est « gros » avec probabilité 1/3 et « petit » avec probabilité 2/3. Pendant les 2 premières minutes, on n'observe que les gros ordres : il en est arrivé exactement 1.

Sachant cette observation, quelle est l'espérance du nombre total d'ordres, gros et petits, arrivés pendant ces 2 minutes ?""",
        "options": [
            {"id": "a", "texte": "@M03_exact@", "explication": r"Oui : le gros ordre observé, plus les petits ordres, dont le nombre est indépendant de celui des gros et d'espérance @M03_mu_petits:0@."},
            {"id": "b", "texte": "@M03_d_inconditionnel@", "explication": r"C'est l'espérance sans tenir compte de l'observation, \(3 \times 2\). Savoir qu'il n'y a eu qu'un gros ordre, contre @M03_mu_gros:0@ en moyenne, change l'espérance du nombre total."},
            {"id": "c", "texte": "@M03_d_inverse@", "explication": r"C'est \(1/(1/3)\) : un gros ordre « représenterait » 3 ordres. Ce raisonnement à rebours est faux : par l'amincissement, le nombre de petits ordres est indépendant du nombre de gros, et son espérance reste @M03_mu_petits:0@."},
            {"id": "d", "texte": "@M03_d_petits:0@", "explication": r"C'est l'espérance du nombre de petits ordres seulement : il faut ajouter le gros ordre observé."},
        ],
        "solution": r"""Réponse : \(1 + @M03_mu_petits:0m@ = @M03_exact@\).

Notations : \(N\) est le nombre total d'ordres sur les 2 premières minutes, \(G\) le nombre de gros ordres, \(P = N - G\) le nombre de petits.

Étape 1 : \(N\) suit la loi de Poisson de paramètre \(\mu = 3 \times 2 = @M03_mu@\), et sachant \(N = n\), \(G\) suit la loi binomiale de paramètres \(n\) et 1/3.

Étape 2, amincissement : \(\mathbb{P}(G = g, P = r) = \mathbb{P}(N = g + r)\binom{g + r}{g}(1/3)^g (2/3)^r = e^{-\mu}\dfrac{\mu^{g + r}}{(g + r)!} \cdot \dfrac{(g + r)!}{g!\,r!}(1/3)^g (2/3)^r = \Big[e^{-\mu/3}\dfrac{(\mu/3)^g}{g!}\Big]\Big[e^{-2\mu/3}\dfrac{(2\mu/3)^r}{r!}\Big]\), en écrivant \(e^{-\mu} = e^{-\mu/3}e^{-2\mu/3}\). La loi jointe est le produit d'une fonction de \(g\) et d'une fonction de \(r\) : \(G\) et \(P\) sont indépendantes, de lois de Poisson de paramètres @M03_mu_gros:0@ et @M03_mu_petits:0@.

Étape 3 : par indépendance, conditionner par \(G = 1\) ne change pas la loi de \(P\), donc \(\mathbb{E}[N \mid G = 1] = 1 + \mathbb{E}[P] = 1 + @M03_mu_petits:0m@ = @M03_exact@\).

Vérification (script) : série \(\sum_n n\,\mathbb{P}(N = n, G = 1)/\mathbb{P}(G = 1)\) sommée par sympy, et simulation : @M03_mc:3@.""",
    },
    {
        "id": "EB1-M04", "section": "math", "theme": "Dénombrement avec contraintes, inclusion-exclusion", "difficulte": 3, "temps_cible_s": 120,
        "enonce": r"""On répartit 10 lots identiques (indiscernables) entre 4 traders numérotés de 1 à 4. Une répartition est la donnée du nombre de lots reçus par chaque trader ; deux répartitions sont différentes si au moins un trader reçoit un nombre de lots différent. Chaque trader doit recevoir au moins 1 lot et au plus 4 lots.

Combien y a-t-il de répartitions possibles ?""",
        "options": [
            {"id": "a", "texte": "@M04_exact@", "explication": r"Oui : \(\binom{9}{3} - 4\binom{5}{3}\), deux dépassements simultanés étant impossibles."},
            {"id": "b", "texte": "@M04_sans_plafond@", "explication": r"C'est \(\binom{9}{3}\), le nombre de répartitions avec au moins 1 lot chacun, sans le plafond de 4 lots."},
            {"id": "c", "texte": "@M04_d_plafond5@", "explication": r"Le plafond a été décalé d'une unité : on a retiré les répartitions où un trader reçoit au moins 6 lots au lieu d'au moins 5. C'est le nombre de répartitions avec un plafond de 5 lots."},
            {"id": "d", "texte": "@M04_retires@", "explication": r"C'est le nombre de répartitions **retirées**, \(4 \times \binom{5}{3}\), où un trader dépasse le plafond, pas le nombre de répartitions valides."},
        ],
        "solution": r"""Réponse : @M04_exact@.

Notations : \(x_i\) est le nombre de lots du trader \(i\) ; on compte les quadruplets d'entiers avec \(x_1 + x_2 + x_3 + x_4 = 10\) et \(1 \le x_i \le 4\). On pose \(y_i = x_i - 1\) : \(y_1 + y_2 + y_3 + y_4 = 6\) avec \(0 \le y_i \le 3\).

Étape 1, sans plafond (étoiles et barres) : une solution en entiers positifs ou nuls de \(y_1 + \dots + y_4 = 6\) est une suite de 6 étoiles et 3 barres ; il y en a \(\binom{9}{3} = @M04_sans_plafond@\).

Étape 2, dépassement d'un trader : les solutions avec \(y_1 \ge 4\) correspondent, par \(y_1' = y_1 - 4\), aux solutions de \(y_1' + y_2 + y_3 + y_4 = 2\), au nombre de \(\binom{5}{3} = @M04_un_depassement@\). Même nombre pour chacun des 4 traders.

Étape 3, inclusion-exclusion : deux dépassements simultanés demanderaient \(y_i + y_j \ge 8 > 6\), ce qui est impossible ; les termes d'ordre 2 et plus sont nuls. Le nombre cherché est \(@M04_sans_plafond@ - 4 \times @M04_un_depassement@ = @M04_exact@\).

Vérification (script) : énumération des quadruplets.""",
    },
    {
        "id": "EB1-M05", "section": "math", "theme": "Inverse d'une perturbation de rang 1 (Sherman-Morrison)", "difficulte": 4, "temps_cible_s": 150,
        "enonce": r"""Soient \(\mathbf{u} = (1, 2, 2)^\top \in \mathbb{R}^3\), \(\mathbf{I}\) la matrice identité \(3 \times 3\), \(\mathbf{1} = (1, 1, 1)^\top\) et \(\mathbf{A} = \mathbf{I} + \mathbf{u}\mathbf{u}^\top\).

Que vaut la somme de tous les coefficients de \(\mathbf{A}^{-1}\), c'est-à-dire \(\mathbf{1}^\top \mathbf{A}^{-1} \mathbf{1}\) ?""",
        "options": [
            {"id": "a", "texte": "@M05_exact_num@/@M05_exact_den@", "explication": r"Oui : \(3 - (\mathbf{1}^\top \mathbf{u})^2/(1 + \mathbf{u}^\top \mathbf{u}) = 3 - @M05_1tu2@/10\)."},
            {"id": "b", "texte": "@M05_d_sans_un_num@/@M05_d_sans_un_den@", "explication": r"On a divisé par \(\mathbf{u}^\top \mathbf{u} = @M05_utu@\) au lieu de \(1 + \mathbf{u}^\top \mathbf{u}\) dans la formule de Sherman-Morrison."},
            {"id": "c", "texte": "@M05_d_inverse_somme_num@/@M05_d_inverse_somme_den@", "explication": r"C'est \(1/(\mathbf{1}^\top \mathbf{A}\,\mathbf{1}) = 1/@M05_somme_A@\) : l'inverse de la somme des coefficients de \(\mathbf{A}\) n'est pas la somme des coefficients de \(\mathbf{A}^{-1}\)."},
            {"id": "d", "texte": "@M05_d_neumann@", "explication": r"C'est le résultat avec \(\mathbf{I} - \mathbf{u}\mathbf{u}^\top\) à la place de \(\mathbf{A}^{-1}\) : le développement \((\mathbf{I} + \mathbf{M})^{-1} \approx \mathbf{I} - \mathbf{M}\), valable seulement pour \(\mathbf{M}\) petite. Un nombre négatif est impossible : \(\mathbf{A}^{-1}\) est définie positive, donc \(\mathbf{1}^\top \mathbf{A}^{-1} \mathbf{1} > 0\)."},
        ],
        "solution": r"""Réponse : \(\mathbf{1}^\top \mathbf{A}^{-1} \mathbf{1} = 3 - @M05_1tu2@/10 = \tfrac{@M05_exact_num@}{@M05_exact_den@}\).

Étape 1, formule de Sherman-Morrison : pour tout \(\mathbf{v} \in \mathbb{R}^3\), avec \(c = \mathbf{v}^\top \mathbf{v}\) (un scalaire positif ou nul), \((\mathbf{I} + \mathbf{v}\mathbf{v}^\top)^{-1} = \mathbf{I} - \dfrac{\mathbf{v}\mathbf{v}^\top}{1 + c}\). Preuve : \((\mathbf{I} + \mathbf{v}\mathbf{v}^\top)\Big(\mathbf{I} - \dfrac{\mathbf{v}\mathbf{v}^\top}{1 + c}\Big) = \mathbf{I} + \mathbf{v}\mathbf{v}^\top - \dfrac{\mathbf{v}\mathbf{v}^\top + \mathbf{v}(\mathbf{v}^\top \mathbf{v})\mathbf{v}^\top}{1 + c}\), et le dernier numérateur vaut \((1 + c)\,\mathbf{v}\mathbf{v}^\top\) : le produit est \(\mathbf{I}\).

Étape 2 : \(\mathbf{u}^\top \mathbf{u} = 1 + 4 + 4 = @M05_utu@\) et \(\mathbf{1}^\top \mathbf{u} = @M05_1tu@\).

Étape 3 : \(\mathbf{1}^\top \mathbf{A}^{-1} \mathbf{1} = \mathbf{1}^\top \mathbf{1} - \dfrac{(\mathbf{1}^\top \mathbf{u})^2}{1 + \mathbf{u}^\top \mathbf{u}} = 3 - \dfrac{@M05_1tu2@}{10} = \tfrac{@M05_exact_num@}{@M05_exact_den@}\).

Contrôle par le spectre : \(\mathbf{A}\mathbf{u} = (1 + \mathbf{u}^\top \mathbf{u})\,\mathbf{u} = @M05_vp_simple@\,\mathbf{u}\), et \(\mathbf{A}\mathbf{x} = \mathbf{x}\) pour tout \(\mathbf{x}\) orthogonal à \(\mathbf{u}\) : les valeurs propres sont @M05_vp_simple@ et 1 (double), toutes positives, donc \(\mathbf{A}\) est inversible et définie positive, et \(\mathbf{A}^{-1}\) aussi. Lien avec la finance : une covariance à un facteur s'écrit diagonale plus rang 1, et cette formule l'inverse sans calcul matriciel lourd. Vérification (script) : inverse exacte par sympy et par numpy.""",
    },
    {
        "id": "EB1-M06", "section": "math", "theme": "Maximum de vraisemblance, biais (loi exponentielle)", "difficulte": 4, "temps_cible_s": 180,
        "enonce": r"""Soient \(X_1, \dots, X_5\) des variables aléatoires indépendantes de loi exponentielle de paramètre \(\lambda > 0\) inconnu (densité \(\lambda e^{-\lambda x}\) pour \(x \ge 0\), espérance \(1/\lambda\)), et \(\bar X = (X_1 + \dots + X_5)/5\). L'estimateur du maximum de vraisemblance de \(\lambda\) est \(\hat\lambda = 1/\bar X\).

Que vaut le rapport \(\mathbb{E}[\hat\lambda]/\lambda\) ?""",
        "options": [
            {"id": "a", "texte": "@M06_exact:2@", "explication": r"Oui : \(n/(n - 1) = 5/4\)."},
            {"id": "b", "texte": "1", "explication": r"L'invariance dit que \(1/\bar X\) est l'estimateur du maximum de vraisemblance de \(\lambda\) puisque \(\bar X\) est celui de \(1/\lambda\) ; elle ne dit rien du biais. Par l'inégalité de Jensen stricte (\(x \mapsto 1/x\) strictement convexe sur \(]0, +\infty[\), \(\bar X\) non constante), \(\mathbb{E}[1/\bar X] > 1/\mathbb{E}[\bar X] = \lambda\)."},
            {"id": "c", "texte": "@M06_d_inverse:2@", "explication": r"C'est \((n - 1)/n\), le rapport inversé, qui rappelle le biais d'un autre estimateur (la variance empirique divisée par \(n\)). Ici \(\hat\lambda\) surestime \(\lambda\) d'après l'inégalité de Jensen : le rapport dépasse 1."},
            {"id": "d", "texte": "@M06_d_delta:2@", "explication": r"C'est \(1 + 1/n\), l'approximation de Taylor à l'ordre 2 (méthode delta) : \(\mathbb{E}[g(\bar X)] \approx g(\mu) + \tfrac12 g''(\mu)\,\mathrm{Var}(\bar X)\) avec \(g(x) = 1/x\) et \(\mu = 1/\lambda\). Le calcul exact donne \(n/(n - 1)\)."},
        ],
        "solution": r"""Réponse : \(\mathbb{E}[\hat\lambda]/\lambda = n/(n - 1) = 5/@M06_n_moins_un@ = @M06_exact:2m@\).

Notations : \(n = 5\) et \(S = X_1 + \dots + X_n\), de sorte que \(\hat\lambda = n/S\).

Étape 0, c'est bien le maximum de vraisemblance : avec \(s\) la valeur observée de \(S\), la log-vraisemblance \(n \ln \lambda - \lambda s\) est strictement concave en \(\lambda\), de dérivée \(n/\lambda - s\), nulle en \(\lambda = n/s\).

Étape 1, loi de \(S\) : somme de \(n\) exponentielles indépendantes de paramètre \(\lambda\), elle a pour fonction génératrice des moments \(\big(\lambda/(\lambda - t)\big)^n\) pour \(t < \lambda\), celle de la loi Gamma de densité \(\lambda^n s^{n-1} e^{-\lambda s}/(n-1)!\) sur \(s > 0\).

Étape 2 : \(\mathbb{E}[1/S] = \int_0^\infty \dfrac{1}{s}\,\dfrac{\lambda^n s^{n-1} e^{-\lambda s}}{(n-1)!}\,ds = \dfrac{\lambda}{n - 1}\int_0^\infty \dfrac{\lambda^{n-1} s^{n-2} e^{-\lambda s}}{(n-2)!}\,ds = \dfrac{\lambda}{n - 1}\), car la dernière intégrande est la densité Gamma d'ordre \(n - 1\) (il faut \(n \ge 2\)).

Étape 3 : \(\mathbb{E}[\hat\lambda] = n\,\mathbb{E}[1/S] = \dfrac{n}{n - 1}\,\lambda\), soit \(@M06_exact:2m@\,\lambda\) pour \(n = 5\). L'estimateur \((n - 1)/S\) est sans biais ; \(\hat\lambda\) est asymptotiquement sans biais.

Vérification (script) : intégrale exacte par sympy, et simulation avec \(\lambda = 2\) : @M06_mc:3@.""",
    },
    {
        "id": "EB1-M07", "section": "math", "theme": "Tests : seuil, puissance", "difficulte": 3, "temps_cible_s": 120,
        "enonce": r"""Soient \(X_1, \dots, X_{25}\) des variables aléatoires indépendantes de loi normale de moyenne \(\mu\) inconnue et de variance connue égale à 1, et \(\bar X\) leur moyenne. On teste \(H_0 : \mu = 0\) contre \(H_1 : \mu > 0\) au niveau 5 % : on rejette \(H_0\) si \(\bar X > c\), où le seuil \(c\) est tel que \(\mathbb{P}(\bar X > c) = 0{,}05\) quand \(\mu = 0\). La puissance du test en \(\mu\) est la probabilité de rejeter \(H_0\) quand la vraie moyenne vaut \(\mu\).

On note \(z_q\) le réel tel que \(\mathbb{P}(Z \le z_q) = q\) pour \(Z\) de loi normale centrée réduite ; \(z_{0{,}95} \approx @M07_z95:3m@\) et \(z_{0{,}975} \approx @M07_z975:3m@\). Pour quelle valeur de \(\mu\) la puissance vaut-elle exactement 1/2 (arrondie à trois décimales) ?""",
        "options": [
            {"id": "a", "texte": "@M07_exact:3@", "explication": r"Oui : la puissance vaut 1/2 quand la vraie moyenne est égale au seuil \(c = z_{0{,}95}/5\)."},
            {"id": "b", "texte": "@M07_d_sans_racine:3@", "explication": r"C'est \(z_{0{,}95}\) sans la division par \(\sqrt{25}\) : l'écart-type de \(\bar X\) est \(1/\sqrt{n} = 1/5\), pas 1."},
            {"id": "c", "texte": "@M07_d_bilateral:3@", "explication": r"C'est \(z_{0{,}975}/5\), le seuil d'un test bilatéral au niveau 5 %. Le test est unilatéral : toute la probabilité 0,05 est dans la queue de droite."},
            {"id": "d", "texte": "@M07_d_n:3@", "explication": r"C'est \(z_{0{,}95}/25\) : on a divisé par \(n\) au lieu de \(\sqrt{n}\). La variance de \(\bar X\) est \(1/n\), son écart-type \(1/\sqrt{n}\)."},
        ],
        "solution": r"""Réponse : \(\mu = c = z_{0{,}95}/\sqrt{25} \approx @M07_exact:3m@\).

Étape 1, loi de la moyenne : \(\bar X\) suit la loi normale de moyenne \(\mu\) et de variance \(1/25\), d'écart-type \(1/5\).

Étape 2, seuil : sous \(H_0\), \(\mathbb{P}(\bar X > c) = \mathbb{P}(Z > 5c) = 0{,}05\), donc \(5c = z_{0{,}95}\) et \(c \approx @M07_z95:3m@/5 \approx @M07_exact:3m@\).

Étape 3, puissance : avec \(\Phi\) la fonction de répartition de la loi normale centrée réduite, \(\pi(\mu) = \mathbb{P}(\bar X > c) = \mathbb{P}\big(Z > 5\,(c - \mu)\big) = 1 - \Phi\big(5\,(c - \mu)\big)\). Comme \(\Phi(0) = 1/2\) et que \(\Phi\) est strictement croissante, \(\pi(\mu) = 1/2\) si et seulement si \(\mu = c\).

Lecture : un effet égal au seuil n'est détecté qu'une fois sur deux ; pour une puissance de 80 %, il faut \(\mu = c + z_{0{,}80}/5\), plus grand. Simulation (script) : puissance en \(\mu = c\) : @M07_puissance_mc:3@ ; niveau en \(\mu = 0\) : @M07_niveau_mc:3@.""",
    },
    {
        "id": "EB1-M08", "section": "math", "theme": "Optimisation : croissance logarithmique (critère de Kelly)", "difficulte": 3, "temps_cible_s": 120,
        "enonce": r"""À chaque tour, un investisseur mise une fraction \(f \in [0, 1[\) de sa richesse sur un pari dont le gain par euro misé est une variable aléatoire \(R\) valant \(+1\) avec probabilité 0,6 et \(-1\) avec probabilité 0,4 ; les tours sont indépendants. Sa richesse (aléatoire) après \(t\) tours, \(W_t\), vérifie \(W_{t+1} = W_t\,(1 + f R_{t+1})\), où \(R_{t+1}\) est le gain du tour \(t + 1\), de même loi que \(R\). Il choisit \(f\) pour maximiser le taux de croissance \(g(f) = \mathbb{E}[\ln(1 + fR)]\), limite presque sûre de \(\tfrac1t \ln(W_t/W_0)\) par la loi forte des grands nombres.

Quelle valeur de \(f\) maximise \(g\) ?""",
        "options": [
            {"id": "a", "texte": "@M08_exact:3@", "explication": r"Oui : \(f^\star = p - q = 0{,}6 - 0{,}4\)."},
            {"id": "b", "texte": "@M08_d_merton:3@", "explication": r"C'est \(\mathbb{E}[R]/\mathrm{Var}(R) = 0{,}2/0{,}96\), la règle du temps continu (Merton). Pour ce pari discret ce n'est qu'une approximation ; l'optimum exact annule \(g'\)."},
            {"id": "c", "texte": "@M08_d_proba:3@", "explication": r"C'est la probabilité de gain. Miser 60 % de sa richesse donne un taux de croissance négatif, \(g(0{,}6) \approx @M08_g_proba:3m@\) : la richesse tend presque sûrement vers 0."},
            {"id": "d", "texte": r"Aucune : \(g\) croît jusqu'à \(f \to 1\), car le pari a une espérance positive.", "explication": r"C'est l'espérance de la richesse, \(\mathbb{E}[1 + fR] = 1 + 0{,}2 f\), qui croît avec \(f\). Le logarithme pénalise les pertes : \(g(f) \to -\infty\) quand \(f \to 1\), car \(\ln(1 - f) \to -\infty\)."},
        ],
        "solution": r"""Réponse : \(f^\star = p - q = @M08_exact:1m@\), avec \(p = 0{,}6\) et \(q = 0{,}4\).

Étape 1 : \(g(f) = 0{,}6 \ln(1 + f) + 0{,}4 \ln(1 - f)\) sur \([0, 1[\).

Étape 2, concavité : \(g''(f) = -\dfrac{0{,}6}{(1 + f)^2} - \dfrac{0{,}4}{(1 - f)^2} < 0\), donc \(g\) est strictement concave, et un point où \(g'\) s'annule est son unique maximum.

Étape 3 : \(g'(f) = \dfrac{0{,}6}{1 + f} - \dfrac{0{,}4}{1 - f} = 0 \iff 0{,}6\,(1 - f) = 0{,}4\,(1 + f) \iff f = 0{,}2\). En général \(f^\star = p - q\) pour un pari à \(\pm 1\) avec \(p > q\) (critère de Kelly).

Étape 4, valeur : \(g(0{,}2) = 0{,}6 \ln 1{,}2 + 0{,}4 \ln 0{,}8 \approx @M08_g_opt:4m@\) par tour.

Comparaison : l'approximation \(\ln(1 + x) \approx x - x^2/2\) donne \(g(f) \approx f\,\mathbb{E}[R] - \tfrac12 f^2\,\mathbb{E}[R^2]\), maximale en \(\mathbb{E}[R]/\mathbb{E}[R^2] = 0{,}2\) : elle tombe juste ici parce que \(\mathbb{E}[R^2] = 1\), alors que la règle \(\mathbb{E}[R]/\mathrm{Var}(R) \approx @M08_d_merton:3m@\) ne tombe pas juste. Simulation (script) : moyenne de \(\ln(1 + 0{,}2\,R)\) : @M08_mc:4@.""",
    },
    {
        "id": "EB1-M09", "section": "math", "theme": "Chaîne de Markov, temps d'absorption", "difficulte": 3, "temps_cible_s": 120,
        "enonce": r"""Une entreprise a chaque année une note parmi A, B et D (défaut). D'une année à la suivante, la note évolue selon une chaîne de Markov homogène, de probabilités de transition :

| depuis | vers A | vers B | vers D |
|---|---|---|---|
| A | 0,8 | 0,2 | 0 |
| B | 0,3 | 0,5 | 0,2 |
| D | 0 | 0 | 1 |

La note est A aujourd'hui. Quelle est l'espérance du nombre d'années jusqu'au premier passage en D ?""",
        "options": [
            {"id": "a", "texte": "@M09_exact:1@", "explication": r"Oui : \(e_A = 5 + e_B\) et \(e_B = @M09_eB:1m@\)."},
            {"id": "b", "texte": "@M09_eB:1@", "explication": r"C'est l'espérance en partant de B, \(e_B\). Depuis A, il faut d'abord quitter A, ce qui prend en moyenne @M09_sortie_A@ ans."},
            {"id": "c", "texte": "@M09_d_somme@", "explication": r"C'est 5 ans pour quitter A plus 5 ans de B au défaut, comme si B ne remontait jamais en A. Depuis B, la note remonte en A avec probabilité 0,3 par an, ce qui allonge le temps restant à \(e_B = @M09_eB:1m@\)."},
            {"id": "d", "texte": "@M09_d_geom@", "explication": r"C'est \(1/0{,}2\), comme si le défaut avait la probabilité 0,2 chaque année. Depuis A, le défaut est impossible en un an : il faut passer par B."},
        ],
        "solution": r"""Réponse : \(e_A = @M09_exact:1m@\) ans.

Notations : \(e_A\) et \(e_B\) sont les espérances du nombre d'années jusqu'au premier passage en D, en partant de A et de B.

Étape 1, finitude : depuis A comme depuis B, la chaîne atteint D en au plus deux ans avec probabilité au moins \(0{,}2 \times 0{,}2\) (A, puis B, puis D). Par la propriété de Markov, les blocs successifs de deux ans échouent chacun avec probabilité au plus \(1 - 0{,}04\) quel que soit le passé : le nombre de blocs est majoré par une variable géométrique, d'espérance finie.

Étape 2, conditionnement sur la première année : \(e_A = 1 + 0{,}8\,e_A + 0{,}2\,e_B\) et \(e_B = 1 + 0{,}3\,e_A + 0{,}5\,e_B\), le passage en D laissant 0 année restante.

Étape 3 : la première équation donne \(e_A = 5 + e_B\) ; en reportant dans la seconde, \(0{,}5\,e_B = 1 + 0{,}3\,(5 + e_B)\), soit \(0{,}2\,e_B = 2{,}5\), d'où \(e_B = @M09_eB:1m@\) et \(e_A = @M09_exact:1m@\). Le système a une solution unique, et les vraies espérances, finies, le vérifient.

Forme matricielle : avec \(\mathbf{Q}\) la matrice des transitions entre les états A et B, \((e_A, e_B)^\top = (\mathbf{I} - \mathbf{Q})^{-1}\mathbf{1}\). Vérification (script) : résolution exacte par les deux méthodes, et simulation : @M09_mc:2@.""",
    },
    {
        "id": "EB1-M10", "section": "math", "theme": "Inégalités : Markov, Tchebychev, Cantelli", "difficulte": 5, "temps_cible_s": 180,
        "enonce": r"""On considère toutes les variables aléatoires réelles \(X\) d'espérance 0 et de variance 1, sans autre hypothèse sur leur loi.

Quelle est la plus petite constante \(c\) telle que \(\mathbb{P}(X \ge 3) \le c\) pour toutes ces variables ?""",
        "options": [
            {"id": "a", "texte": "@M10_exact_num@/@M10_exact_den@", "explication": r"Oui : borne de Cantelli \(1/(1 + t^2)\), atteinte par une loi à deux points."},
            {"id": "b", "texte": "@M10_d_tcheb_num@/@M10_d_tcheb_den@", "explication": r"C'est la borne de Tchebychev, \(\mathbb{P}(\lvert X \rvert \ge 3) \le 1/9\). Elle est valable mais n'est pas la plus petite : elle majore les deux queues à la fois, alors que la borne \(1/10\) est valable pour une seule queue et atteinte."},
            {"id": "c", "texte": "@M10_d_demi_num@/@M10_d_demi_den@", "explication": r"C'est la moitié de la borne de Tchebychev, ce qui suppose la loi symétrique. Sans symétrie, la loi qui vaut 3 avec probabilité 1/10 et \(-1/3\) avec probabilité 9/10 est d'espérance 0 et de variance 1, et donne \(\mathbb{P}(X \ge 3) = 1/10 > 1/18\)."},
            {"id": "d", "texte": "@M10_d_markov_num@/@M10_d_markov_den@", "explication": r"C'est l'inégalité de Markov appliquée à \(\lvert X \rvert\) avec \(\mathbb{E}\lvert X \rvert \le \sqrt{\mathbb{E}[X^2]} = 1\) (Cauchy-Schwarz) : valable, mais beaucoup plus faible, car elle n'exploite la variance qu'à travers \(\mathbb{E}\lvert X \rvert\)."},
        ],
        "solution": r"""Réponse : \(c = 1/(1 + 3^2) = \tfrac{@M10_exact_num@}{@M10_exact_den@}\) (inégalité de Cantelli, atteinte).

Notations : \(t = 3\), et \(u \ge 0\) un réel libre.

Étape 1, majoration pour toute \(X\) : si \(X \ge t\), alors \(X + u \ge t + u > 0\), donc \((X + u)^2 \ge (t + u)^2\). Par l'inégalité de Markov appliquée à la variable positive \((X + u)^2\) : \(\mathbb{P}(X \ge t) \le \mathbb{P}\big((X + u)^2 \ge (t + u)^2\big) \le \dfrac{\mathbb{E}[(X + u)^2]}{(t + u)^2} = \dfrac{1 + u^2}{(t + u)^2}\), car \(\mathbb{E}[(X + u)^2] = \mathbb{E}[X^2] + 2u\,\mathbb{E}[X] + u^2 = 1 + u^2\).

Étape 2, meilleur \(u\) : la dérivée de \(u \mapsto (1 + u^2)/(t + u)^2\) vaut \(\dfrac{2(tu - 1)}{(t + u)^3}\), négative puis positive, nulle en \(u = 1/t\), où la borne vaut \(\dfrac{1 + 1/t^2}{(t + 1/t)^2} = \dfrac{1}{1 + t^2} = \tfrac{@M10_exact_num@}{@M10_exact_den@}\).

Étape 3, la borne est atteinte : soit \(X = 3\) avec probabilité 1/10 et \(X = -1/3\) avec probabilité 9/10. Alors \(\mathbb{E}[X] = 3/10 - 3/10 = 0\), \(\mathbb{E}[X^2] = 9/10 + (1/9)(9/10) = 1\), et \(\mathbb{P}(X \ge 3) = 1/10\). Aucune constante plus petite ne convient.

Vérification (script) : programme linéaire sur les lois portées par une grille, d'optimum @M10_lp:3@, et simulation de la loi à deux points : @M10_mc:4@.""",
    },
    # ================================================================== ML FONDAMENTAL
    {
        "id": "EB1-F01", "section": "mlf", "theme": "Ridge et Lasso avec deux variables identiques", "difficulte": 4, "temps_cible_s": 105,
        "enonce": r"""Une matrice de variables \(\mathbf{X}\), de taille \(n \times 2\), a deux colonnes identiques, \(\mathbf{x}_1 = \mathbf{x}_2 = \mathbf{x}\), où \(\mathbf{x} \in \mathbb{R}^n\) est de norme euclidienne 1. Le vecteur des réponses \(\mathbf{y} \in \mathbb{R}^n\) vérifie \(\mathbf{x}^\top \mathbf{y} = 3\). Il n'y a pas de constante. Pour \(\mathbf{b} = (b_1, b_2)\), Ridge minimise \(\tfrac12 \lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2 + \tfrac{\lambda}{2}(b_1^2 + b_2^2)\) et le Lasso minimise \(\tfrac12 \lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2 + \lambda\,(\lvert b_1 \rvert + \lvert b_2 \rvert)\), avec \(\lambda = 1\) dans les deux cas.

Quelle affirmation est vraie ?""",
        "options": [
            {"id": "a", "texte": r"Ridge : @F01_ridge:0@, unique. Lasso : une infinité de minimiseurs, tous les \((b_1, b_2)\) à coordonnées positives ou nulles de somme 2.", "explication": r"Oui : Ridge est strictement convexe ; le critère du Lasso ne dépend que de \(b_1 + b_2\) sur le quadrant positif."},
            {"id": "b", "texte": r"Ridge : @F01_ridge:0@. Lasso : uniquement (2 ; 0) ou (0 ; 2), car la pénalité L1 met des coefficients exactement à zéro.", "explication": r"Le critère du Lasso prend la même valeur, @F01_crit_lasso:2@ dans l'expérience du script, en (2 ; 0), en (1 ; 1) et en tout point positif de somme 2. La descente par coordonnées de scikit-learn renvoie @F01_sk_lasso:0@ parce qu'elle met à jour \(b_1\) en premier : c'est un effet de l'algorithme, pas une propriété du minimum."},
            {"id": "c", "texte": r"Ridge : (@F01_d_orth:1@ ; @F01_d_orth:1@). Lasso : (1 ; 1), unique.", "explication": r"(1,5 ; 1,5) applique la formule \(\mathbf{x}_j^\top \mathbf{y}/(1 + \lambda)\) du cas à colonnes orthonormées. Ici \(\mathbf{X}^\top \mathbf{X}\) a des coefficients hors diagonale égaux à 1 : les deux colonnes portent la même information et se la partagent. Et le minimiseur du Lasso n'est pas unique."},
            {"id": "d", "texte": r"Ridge n'a pas de solution unique, car \(\mathbf{X}^\top \mathbf{X}\) n'est pas inversible. Lasso : (1 ; 1), unique.", "explication": r"\(\mathbf{X}^\top \mathbf{X}\) est singulière (valeurs propres @F01_vp_xtx@), mais \(\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}\) a pour valeurs propres @F01_vp_ridge@ : elle est inversible pour tout \(\lambda > 0\), et la solution Ridge est unique. C'est l'une des raisons d'être de Ridge."},
        ],
        "solution": r"""Réponse : Ridge a pour unique solution \((1 ; 1)\) ; le Lasso a une infinité de solutions, tous les \((b_1, b_2)\) positifs ou nuls de somme 2.

Étape 1, Ridge : le critère est strictement convexe (une fonction convexe plus \(\tfrac{\lambda}{2}\lVert \mathbf{b} \rVert^2\)), donc son minimum est unique et annule le gradient : \((\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})\,\mathbf{b} = \mathbf{X}^\top \mathbf{y}\), soit \(2b_1 + b_2 = 3\) et \(b_1 + 2b_2 = 3\), d'où \(b_1 = b_2 = 1\). Ridge répartit l'effet à parts égales entre les deux copies.

Étape 2, Lasso : \(\mathbf{X}\mathbf{b} = s\,\mathbf{x}\) avec \(s = b_1 + b_2\), et \(\lVert \mathbf{y} - s\mathbf{x} \rVert^2 = \lVert \mathbf{y} \rVert^2 - 2s\,\mathbf{x}^\top \mathbf{y} + s^2\). Par l'inégalité triangulaire, \(\lvert b_1 \rvert + \lvert b_2 \rvert \ge \lvert s \rvert\), avec égalité si et seulement si \(b_1\) et \(b_2\) sont de même signe ou nuls. À \(s\) fixé, le critère est donc minimal exactement en ces points, où il vaut \(h(s) = \tfrac12 \lVert \mathbf{y} \rVert^2 - 3s + \tfrac12 s^2 + \lvert s \rvert\).

Étape 3 : \(h\) est convexe ; pour \(s > 0\), \(h'(s) = s - 2\) s'annule en \(s = 2\) ; pour \(s < 0\), \(h'(s) = s - 4 < 0\). Le minimum est en \(s = 2\), et les minimiseurs sont les points de même signe de somme 2 : le segment de (2 ; 0) à (0 ; 2).

Expérience (script, \(n = 4\)) : le critère du Lasso vaut @F01_crit_lasso:2@ en tout point du segment et nulle part moins sur une grille ; scikit-learn renvoie @F01_sk_lasso:0@ pour le Lasso et @F01_sk_ridge:0@ pour Ridge. Avec deux variables identiques, le Lasso ne sait pas laquelle garder (le choix dépend de l'algorithme), alors que Ridge les garde ensemble.""",
    },
    {
        "id": "EB1-F02", "section": "mlf", "theme": "Régression logistique, interprétation d'un coefficient", "difficulte": 2, "temps_cible_s": 60,
        "enonce": r"""Une régression logistique modélise \(\mathbb{P}(Y = 1 \mid x) = \dfrac{1}{1 + e^{-(b_0 + b_1 x)}}\), où \(Y \in \{0, 1\}\) est l'étiquette, \(x\) une variable réelle, et \(b_0\), \(b_1\) les coefficients estimés. Le coefficient de \(x\) vaut \(b_1 = \ln 2\). Pour un client donné, la probabilité prédite est 0,2.

Si sa variable \(x\) augmente de 1, toutes choses égales par ailleurs, quelle devient la probabilité prédite (arrondie à trois décimales) ?""",
        "options": [
            {"id": "a", "texte": "@F02_exact:3@", "explication": r"Oui : la cote passe de 0,25 à 0,5, et la probabilité de 0,2 à 1/3."},
            {"id": "b", "texte": "@F02_d_double:3@", "explication": r"La probabilité a été doublée. C'est la **cote** \(p/(1 - p)\) qui est multipliée par \(e^{b_1} = 2\), pas la probabilité."},
            {"id": "c", "texte": "@F02_d_cote:3@", "explication": r"C'est la nouvelle cote, \(2 \times 0{,}25\), pas la nouvelle probabilité : il faut encore la convertir par \(p = \mathrm{cote}/(1 + \mathrm{cote})\)."},
            {"id": "d", "texte": "@F02_d_additif:3@", "explication": r"On a ajouté \(b_1 = \ln 2\) à la probabilité. \(b_1\) s'ajoute au logarithme de la cote, pas à la probabilité."},
        ],
        "solution": r"""Réponse : \(\tfrac{@F02_exact_num@}{@F02_exact_den@} \approx @F02_exact:3m@\).

Étape 1 : le modèle s'écrit \(\ln\dfrac{p}{1 - p} = b_0 + b_1 x\), où \(p = \mathbb{P}(Y = 1 \mid x)\) : le logarithme de la cote est linéaire en \(x\). Augmenter \(x\) de 1 ajoute \(b_1\) au logarithme de la cote, donc multiplie la cote par \(e^{b_1} = 2\).

Étape 2 : cote initiale \(0{,}2/0{,}8 = @F02_cote0:2m@\), nouvelle cote \(@F02_cote1:1m@\).

Étape 3 : \(p = @F02_cote1:1m@/(1 + @F02_cote1:1m@) = \tfrac{@F02_exact_num@}{@F02_exact_den@}\).

Piège : l'effet d'une unité de \(x\) sur la probabilité dépend du point de départ (il est le plus fort près de \(p = 1/2\), où la dérivée \(p(1 - p)\,b_1\) est maximale) ; seul l'effet sur la cote est constant.""",
    },
    {
        "id": "EB1-F03", "section": "mlf", "theme": "Validation croisée temporelle, purge et embargo", "difficulte": 4, "temps_cible_s": 105,
        "enonce": r"""Des observations quotidiennes sont numérotées \(t = 1, \dots, 1000\). L'étiquette de l'observation \(t\) est le rendement cumulé des jours \(t + 1\) à \(t + 5\) ; on appelle intervalle d'étiquette de \(t\) l'ensemble de jours \(\{t + 1, \dots, t + 5\}\). On fait une validation croisée en 5 blocs contigus de 200 observations, et l'on considère le pli dont le bloc de test est \(\{401, \dots, 600\}\). L'entraînement part de toutes les observations hors du bloc de test, puis :

1. purge : on retire toute observation dont l'intervalle d'étiquette a au moins un jour en commun avec la réunion des intervalles d'étiquette des observations de test ;
2. embargo : on retire en outre les 10 observations qui suivent immédiatement le bloc de test, \(t = 601, \dots, 610\).

Combien d'observations reste-t-il dans l'ensemble d'entraînement ?""",
        "options": [
            {"id": "a", "texte": "@F03_exact@", "explication": r"Oui : 4 observations purgées avant le bloc, puis 10 retirées après (la purge y est comprise dans l'embargo)."},
            {"id": "b", "texte": "@F03_d_purge_seule@", "explication": r"C'est la purge seule : 4 observations retirées avant le bloc et 4 après. L'embargo retire en plus les observations 605 à 610."},
            {"id": "c", "texte": "@F03_d_mauvais_intervalle@", "explication": r"La purge a été faite contre les jours du bloc de test (401 à 600) au lieu des jours couverts par ses étiquettes (@F03_union_min@ à @F03_union_max@). On retire alors aussi l'observation 396, dont l'intervalle d'étiquette s'arrête au jour 401 et ne partage aucun jour avec les étiquettes de test : la règle de l'énoncé ne la retire pas."},
            {"id": "d", "texte": "@F03_d_rien@", "explication": r"C'est l'entraînement sans purge ni embargo. Les étiquettes des observations @F03_premier_purge@ à 400 contiennent des rendements de jours qui figurent aussi dans les étiquettes de test : c'est une fuite d'information."},
        ],
        "solution": r"""Réponse : @F03_exact@ observations.

Étape 1 : la réunion des intervalles d'étiquette du bloc de test va du jour \(401 + 1 = @F03_union_min@\) au jour \(600 + 5 = @F03_union_max@\).

Étape 2, purge avant le bloc : l'observation \(t \le 400\) a pour intervalle \(\{t + 1, \dots, t + 5\}\), qui rencontre \(\{@F03_union_min@, \dots, @F03_union_max@\}\) si et seulement si \(t + 5 \ge @F03_union_min@\), soit \(t \ge @F03_premier_purge@\) : @F03_purges_avant@ observations retirées.

Étape 3, purge après le bloc : l'observation \(t \ge 601\) le rencontre si et seulement si \(t + 1 \le @F03_union_max@\), soit \(t \le @F03_dernier_purge_apres@\) : @F03_purges_apres@ observations, toutes déjà dans l'embargo 601 à 610.

Étape 4 : il reste @F03_restants_avant@ observations avant le bloc et @F03_restants_apres@ après, soit @F03_exact@.

Pourquoi un embargo en plus de la purge : les observations qui suivent le test ne partagent plus de jour d'étiquette avec lui au-delà de @F03_dernier_purge_apres@, mais leurs variables explicatives (rendements passés, par exemple) peuvent contenir des jours de la période de test. L'embargo de 10 jours est une marge de prudence fixée par l'utilisateur, pas une conséquence de la règle de purge. Vérification (script) : calcul par ensembles.""",
    },
    {
        "id": "EB1-F04", "section": "mlf", "theme": "Courbe ROC et AUC, égalités de scores", "difficulte": 3, "temps_cible_s": 75,
        "enonce": r"""Un classifieur attribue un score à 7 exemples de test. Les 3 exemples positifs ont les scores 0,9 ; 0,7 ; 0,4, et les 4 exemples négatifs les scores 0,8 ; 0,4 ; 0,3 ; 0,1. La courbe ROC est tracée en faisant varier un seuil \(s\) (un exemple est prédit positif si son score est au moins \(s\)) : chaque seuil donne un point (taux de faux positifs, taux de vrais positifs), et les points obtenus pour les seuils égaux aux scores observés, avec le point (0, 0), sont reliés par des segments de droite, comme le fait `roc_auc_score` de scikit-learn.

Quelle est l'aire sous cette courbe (AUC), arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@F04_exact:3@", "explication": r"Oui : \(@F04_gagnees:1m@\) couples bien ordonnés sur @F04_paires@, l'égalité comptant pour 1/2."},
            {"id": "b", "texte": "@F04_d_strict:3@", "explication": r"Le couple de scores égaux (0,4 et 0,4) a été compté comme mal ordonné. Le segment diagonal qui franchit ces deux exemples ensemble ajoute la moitié d'un rectangle : une égalité compte pour 1/2."},
            {"id": "c", "texte": "@F04_d_large:3@", "explication": r"Le couple de scores égaux a été compté comme bien ordonné ; il compte pour 1/2."},
            {"id": "d", "texte": "@F04_d_exactitude:3@", "explication": r"C'est l'exactitude au seuil 0,5 (5 exemples bien classés sur 7). L'AUC ne dépend d'aucun seuil particulier : elle résume tous les seuils à la fois."},
        ],
        "solution": r"""Réponse : \(@F04_gagnees:1m@/@F04_paires@ = \tfrac{@F04_exact_num@}{@F04_exact_den@} \approx @F04_exact:3m@\).

Étape 1, lecture par les couples : en abaissant le seuil, chaque négatif franchi déplace la courbe de \(1/4\) vers la droite, à une hauteur égale à la proportion de positifs déjà franchis, c'est-à-dire de score strictement plus élevé : il ajoute l'aire \(\tfrac14 \times \tfrac{m}{3}\), où \(m\) est le nombre de positifs qui le battent. Un positif et un négatif de même score sont franchis ensemble : le segment diagonal ajoute un trapèze, dont la part due à ce couple vaut la moitié de \(\tfrac14 \times \tfrac13\). L'AUC est donc la proportion des couples (positif, négatif) où le positif a le score le plus élevé, une égalité comptant pour 1/2.

Étape 2, décompte des \(3 \times 4 = @F04_paires@\) couples : le positif 0,9 bat les 4 négatifs ; 0,7 bat 0,4, 0,3 et 0,1 ; 0,4 bat 0,3 et 0,1 et égale 0,4. Total \(4 + 3 + 2 + 1/2 = @F04_gagnees:1m@\).

Étape 3 : \(\mathrm{AUC} = @F04_gagnees:1m@/@F04_paires@ \approx @F04_exact:3m@\). Lecture : c'est la probabilité qu'un positif tiré au hasard ait un score plus élevé qu'un négatif tiré au hasard, les égalités comptant pour moitié.

Vérification (script) : `roc_auc_score`, et aire par la méthode des trapèzes sur la courbe ROC.""",
    },
    {
        "id": "EB1-F05", "section": "mlf", "theme": "Décomposition biais-variance, k plus proches voisins", "difficulte": 3, "temps_cible_s": 75,
        "enonce": r"""On observe \(Y = f(x) + \varepsilon\), où \(f\) est une fonction inconnue, \(x\) une entrée fixée et \(\varepsilon\) un bruit centré de variance \(\sigma^2 = 1\), les bruits des différentes observations étant indépendants. Pour prédire en un point \(x_0\), la méthode des \(k = 4\) plus proches voisins renvoie la moyenne \(\hat f(x_0)\) des réponses des 4 points d'entraînement les plus proches de \(x_0\). Ces 4 points sont fixés, et \(f\) y vaut 1 ; 1,5 ; 2 ; 1,5. On a \(f(x_0) = 1\).

Une nouvelle observation \(Y_0 = f(x_0) + \varepsilon_0\) est tirée, avec \(\varepsilon_0\) de même loi que les bruits d'entraînement et indépendant d'eux. Que vaut l'erreur de prédiction espérée \(\mathbb{E}\big[(Y_0 - \hat f(x_0))^2\big]\) ?""",
        "options": [
            {"id": "a", "texte": "@F05_exact:2@", "explication": r"Oui : bruit irréductible 1, biais au carré @F05_biais2:2@, variance @F05_variance:2@."},
            {"id": "b", "texte": "@F05_d_sans_bruit:2@", "explication": r"C'est le biais au carré plus la variance de \(\hat f(x_0)\) : l'erreur d'estimation de \(f(x_0)\), pas l'erreur de prédiction de \(Y_0\). Le bruit \(\varepsilon_0\) de la nouvelle observation ajoute \(\sigma^2 = 1\), qu'aucun modèle ne peut réduire."},
            {"id": "c", "texte": "@F05_d_sans_variance:2@", "explication": r"Il manque la variance de la moyenne des 4 réponses bruitées, \(\sigma^2/k = 1/4\)."},
            {"id": "d", "texte": "@F05_d_variance_pleine:2@", "explication": r"La variance de la prédiction a été prise égale à \(\sigma^2\) au lieu de \(\sigma^2/k\) : la moyenne de 4 bruits indépendants a une variance 4 fois plus petite."},
        ],
        "solution": r"""Réponse : \(\sigma^2 + \mathrm{biais}^2 + \sigma^2/k = 1 + @F05_biais2:2m@ + @F05_variance:2m@ = @F05_exact:2m@\).

Étape 1, décomposition : \(Y_0 - \hat f(x_0) = \varepsilon_0 + \big(f(x_0) - \mathbb{E}[\hat f(x_0)]\big) + \big(\mathbb{E}[\hat f(x_0)] - \hat f(x_0)\big)\) : un bruit centré indépendant du reste, une constante, et une variable centrée. En développant le carré, les doubles produits sont d'espérance nulle (indépendance et centrage), d'où \(\mathbb{E}\big[(Y_0 - \hat f(x_0))^2\big] = \sigma^2 + \big(f(x_0) - \mathbb{E}[\hat f(x_0)]\big)^2 + \mathrm{Var}\big(\hat f(x_0)\big)\).

Étape 2, biais : \(\mathbb{E}[\hat f(x_0)] = (1 + 1{,}5 + 2 + 1{,}5)/4 = @F05_moy_voisins:1m@\), donc biais \(= @F05_moy_voisins:1m@ - 1 = @F05_biais:1m@\), et biais au carré \(@F05_biais2:2m@\).

Étape 3, variance : \(\hat f(x_0)\) est une constante plus la moyenne de 4 bruits indépendants de variance 1, donc \(\mathrm{Var}(\hat f(x_0)) = 1/4\).

Lecture : augmenter \(k\) réduit la variance \(\sigma^2/k\), mais fait entrer des voisins plus lointains, ce qui peut augmenter le biais. Simulation (script) : @F05_mc:3@.""",
    },
    # ================================================================== ML AVANCÉ
    {
        "id": "EB1-A01", "section": "mla", "theme": "Descente de gradient, pas et conditionnement", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""On minimise \(f(\mathbf{w}) = \tfrac12 \mathbf{w}^\top \mathbf{H}\,\mathbf{w}\) sur \(\mathbb{R}^2\), où \(\mathbf{H}\) est la matrice diagonale de coefficients diagonaux 10 et 1, par descente de gradient à pas fixe \(\eta > 0\) : \(\mathbf{w}_{k+1} = \mathbf{w}_k - \eta\,\nabla f(\mathbf{w}_k)\). On note \(\rho(\eta) = \max\big(\lvert 1 - 10\eta \rvert, \lvert 1 - \eta \rvert\big)\) le facteur de contraction, qui vérifie \(\lVert \mathbf{w}_{k+1} \rVert \le \rho(\eta)\,\lVert \mathbf{w}_k \rVert\).

Pour quels \(\eta\) la suite converge-t-elle vers \(\mathbf{0}\) depuis tout point de départ, et quel pas \(\eta^\star\) minimise \(\rho\) ?""",
        "options": [
            {"id": "a", "texte": r"\(0 < \eta < @A01_eta_max:1m@\) ; \(\eta^\star = 2/11\), avec \(\rho(\eta^\star) = 9/11 \approx @A01_facteur:3m@\).", "explication": r"Oui : \(\eta < 2/L\) avec \(L = 10\), et \(\eta^\star = 2/(L + \mu)\) égalise les deux facteurs."},
            {"id": "b", "texte": r"\(0 < \eta < 2\) ; \(\eta^\star = 2/11\), avec \(\rho(\eta^\star) = 9/11\).", "explication": r"La condition \(\eta < 2\) ne regarde que la plus petite valeur propre, 1. C'est la plus grande, 10, qui limite le pas : pour \(\eta > 0{,}2\), \(\lvert 1 - 10\eta \rvert > 1\) et la première coordonnée diverge en oscillant."},
            {"id": "c", "texte": r"\(0 < \eta < @A01_eta_max:1m@\) ; \(\eta^\star = 1/10\), avec \(\rho(\eta^\star) = @A01_d_facteur_1L:1m@\).", "explication": r"Le pas \(1/L = 1/10\) évite toute oscillation, mais il n'est pas optimal : la direction lente n'est réduite que d'un facteur 0,9 par pas. En acceptant une oscillation amortie sur la direction rapide, \(\eta = 2/11\) égalise les deux facteurs à \(9/11\)."},
            {"id": "d", "texte": r"\(0 < \eta \le @A01_eta_max:1m@\) ; \(\eta^\star = @A01_eta_max:1m@\), avec \(\rho(\eta^\star) = 0{,}8\).", "explication": r"En \(\eta = 0{,}2\), la première coordonnée est multipliée par \(1 - 10 \times 0{,}2 = -1\) à chaque pas : elle oscille sans converger, et \(\rho(0{,}2) = 1\). Le facteur 0,8 ne regarde que la direction lente."},
        ],
        "solution": r"""Réponse : convergence si et seulement si \(0 < \eta < @A01_eta_max:1m@\) ; pas optimal \(\eta^\star = 2/(10 + 1) = 2/11\), de facteur \(\rho(\eta^\star) = (10 - 1)/(10 + 1) = 9/11 \approx @A01_facteur:3m@\).

Notations : \(L = 10\) et \(\mu = 1\) sont la plus grande et la plus petite valeur propre de \(\mathbf{H}\), \(\kappa = L/\mu = @A01_kappa:0m@\) son conditionnement.

Étape 1 : \(\nabla f(\mathbf{w}) = \mathbf{H}\mathbf{w}\), donc \(\mathbf{w}_{k+1} = (\mathbf{I} - \eta\mathbf{H})\,\mathbf{w}_k\) : à chaque pas, la première coordonnée est multipliée par \(1 - 10\eta\) et la seconde par \(1 - \eta\).

Étape 2, convergence depuis tout départ : il faut et il suffit que \(\lvert 1 - 10\eta \rvert < 1\) et \(\lvert 1 - \eta \rvert < 1\), soit \(0 < \eta < 0{,}2\) et \(0 < \eta < 2\) : c'est \(L\) qui impose \(\eta < 2/L\).

Étape 3, pas optimal : sur \([1/10, 2/11]\), \(\lvert 1 - \eta \rvert = 1 - \eta\) décroît et \(\lvert 1 - 10\eta \rvert = 10\eta - 1\) croît ; avant, \(\rho = 1 - \eta\) décroît, après, \(\rho = 10\eta - 1\) croît. Le minimum est atteint quand \(1 - \eta = 10\eta - 1\), soit \(\eta^\star = 2/(L + \mu) = 2/11\), où \(\rho = 1 - 2/11 = 9/11 = (\kappa - 1)/(\kappa + 1)\).

Lecture : pour \(\kappa\) grand, \(-\ln \rho = \ln\dfrac{\kappa + 1}{\kappa - 1} \approx 2/\kappa\), donc le nombre d'itérations nécessaire pour diviser l'erreur par une constante donnée croît à peu près proportionnellement à \(\kappa\). Expérience (script) : après 50 itérations depuis \((1, 1)\), norme @A01_norme_50_opt:6@ au pas \(\eta^\star\), contre @A01_norme_50_1L:6@ au pas \(1/10\).""",
    },
    {
        "id": "EB1-A02", "section": "mla", "theme": "Rétropropagation : softmax et entropie croisée", "difficulte": 3, "temps_cible_s": 120,
        "enonce": r"""Un réseau de classification à 3 classes produit, pour une observation, le vecteur de scores (logits) \(\mathbf{z} = (2, 1, 0)\). Les probabilités prédites sont \(p_j = e^{z_j}/\sum_{k=1}^{3} e^{z_k}\) (softmax), et la perte d'entropie croisée est \(L = -\ln p_c\), où \(c\) est le numéro de la vraie classe ; ici \(c = 2\). On donne \(e \approx @A02_e:3m@\) et \(e^2 \approx @A02_e2:3m@\).

Que vaut le gradient \(\partial L/\partial \mathbf{z}\), coordonnées arrondies à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@A02_exact:3@", "explication": r"Oui : \(\mathbf{p} - \mathbf{y}\), avec \(\mathbf{y}\) l'indicatrice de la vraie classe."},
            {"id": "b", "texte": "@A02_d_signe:3@", "explication": r"C'est \(\mathbf{y} - \mathbf{p}\), l'opposé du gradient, c'est-à-dire la direction de descente. Augmenter le score de la vraie classe fait baisser la perte : la coordonnée correspondante du gradient doit être négative."},
            {"id": "c", "texte": "@A02_d_p:3@", "explication": r"C'est \(\mathbf{p}\) seul : il manque \(-1\) sur la coordonnée de la vraie classe, qui vient du terme \(-z_c\) dans \(L = -z_c + \ln \sum_k e^{z_k}\)."},
            {"id": "d", "texte": "@A02_d_dp:3@", "explication": r"C'est le gradient par rapport aux **probabilités**, \(\partial L/\partial p_2 = -1/p_2\). La question porte sur les scores \(\mathbf{z}\) : il faut encore composer par la jacobienne du softmax."},
        ],
        "solution": r"""Réponse : \(\partial L/\partial \mathbf{z} = \mathbf{p} - \mathbf{y} \approx @A02_exact:3m@\), où \(\mathbf{y} = (0, 1, 0)\) est l'indicatrice de la vraie classe.

Étape 1, réécriture : \(L = -\ln p_c = -z_c + \ln \sum_k e^{z_k}\).

Étape 2, dérivée : \(\dfrac{\partial}{\partial z_j} \ln \sum_k e^{z_k} = \dfrac{e^{z_j}}{\sum_k e^{z_k}} = p_j\), et \(\partial(-z_c)/\partial z_j\) vaut \(-1\) si \(j = c\), 0 sinon. Donc \(\partial L/\partial z_j = p_j - y_j\).

Étape 3, valeurs : \(\sum_k e^{z_k} = e^2 + e + 1 \approx @A02_somme_exp:3m@\), \(\mathbf{p} \approx @A02_p:3m@\), d'où le gradient. Contrôle : ses coordonnées somment à 0, car \(\sum_j p_j = \sum_j y_j = 1\).

Lecture : chaque coordonnée du gradient par rapport aux scores est comprise entre \(-1\) et 1 quelle que soit la confiance du modèle, alors que \(-1/p_c\) peut être arbitrairement grand. Vérification (script) : dérivation symbolique par sympy et différences finies.""",
    },
    {
        "id": "EB1-A03", "section": "mla", "theme": "Attention par produit scalaire normalisé", "difficulte": 3, "temps_cible_s": 120,
        "enonce": r"""Une tête d'attention reçoit une requête \(\mathbf{q} = (1, 1, 0, 0)\) et trois clés \(\mathbf{k}_1 = (2, 2, 0, 0)\), \(\mathbf{k}_2 = (0, 0, 3, 3)\), \(\mathbf{k}_3 = (1, -1, 0, 0)\), de dimension \(d = 4\), associées aux valeurs scalaires \(v_1 = 10\), \(v_2 = 4\), \(v_3 = -2\). Elle calcule les poids \(\alpha_j = \dfrac{\exp(\mathbf{q}^\top \mathbf{k}_j/\sqrt{d})}{\sum_{i=1}^{3} \exp(\mathbf{q}^\top \mathbf{k}_i/\sqrt{d})}\) et renvoie \(\sum_{j=1}^{3} \alpha_j v_j\). On donne \(e^2 \approx @A03_e2:3m@\).

Quelle est la sortie, arrondie à deux décimales ?""",
        "options": [
            {"id": "a", "texte": "@A03_exact:2@", "explication": r"Oui : scores (2, 0, 0) après division par \(\sqrt{4} = 2\), puis moyenne pondérée des valeurs."},
            {"id": "b", "texte": "@A03_d_sans_echelle:2@", "explication": r"C'est le résultat sans la division par \(\sqrt{d}\) : scores (4, 0, 0) au lieu de (2, 0, 0), et des poids trop concentrés sur la première clé."},
            {"id": "c", "texte": "@A03_d_echelle_d:2@", "explication": r"Les produits scalaires ont été divisés par \(d = 4\) au lieu de \(\sqrt{d} = 2\) : scores (1, 0, 0), et des poids trop uniformes."},
            {"id": "d", "texte": "@A03_d_moyenne:2@", "explication": r"C'est la moyenne simple des valeurs, qui correspondrait à des scores tous égaux. Ici la première clé est alignée avec la requête et reçoit le plus de poids."},
        ],
        "solution": r"""Réponse : \(\dfrac{10\,e^2 + 4 - 2}{e^2 + 2} \approx @A03_exact:2m@\).

Étape 1, produits scalaires : \(\mathbf{q}^\top \mathbf{k}_j\) vaut respectivement @A03_produits@ (la deuxième clé est orthogonale à la requête, et pour la troisième \(1 - 1 = 0\)).

Étape 2, normalisation : divisés par \(\sqrt{4} = 2\), les scores valent @A03_scores@.

Étape 3, softmax : les poids sont \(e^2/(e^2 + 2)\), \(1/(e^2 + 2)\) et \(1/(e^2 + 2)\), soit environ @A03_poids:3@.

Étape 4 : sortie \(= (10\,e^2 + 4 - 2)/(e^2 + 2) \approx @A03_exact:2m@\).

Pourquoi \(\sqrt{d}\) : si les coordonnées de \(\mathbf{q}\) et de \(\mathbf{k}\) sont indépendantes, centrées et de variance 1, alors \(\mathrm{Var}(\mathbf{q}^\top \mathbf{k}) = \sum_{i=1}^{d} \mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = d\), les termes croisés étant d'espérance nulle. Diviser par \(\sqrt{d}\) ramène la variance des scores à 1 quelle que soit la dimension. Sans cela, les scores grandissent avec \(d\), le softmax sature sur une seule clé, et ses dérivées \(\partial \alpha_j/\partial s_j = \alpha_j(1 - \alpha_j)\) (\(s_j\) le score de la clé \(j\)) deviennent très petites. Simulation (script) : variance du produit scalaire @A03_var_d4_mc:2@ pour \(d = 4\) et @A03_var_d64_mc:1@ pour \(d = 64\).""",
    },
    {
        "id": "EB1-A04", "section": "mla", "theme": "Boosting de gradient, taux d'apprentissage", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""Boosting de gradient pour la perte quadratique \(\tfrac12 (y - F)^2\), où \(y\) est la cible et \(F\) la prédiction, sur un jeu de données dont les cibles ont pour moyenne \(\bar y = 10\). Chaque arbre est réduit à une seule feuille : à l'étape \(m\), il prédit pour toutes les observations la même constante \(T_m\), égale à la moyenne des pseudo-résidus \(y_i - F_{m-1}(x_i)\). Le modèle part de \(F_0 = 0\) et se met à jour par \(F_m = F_{m-1} + \nu\,T_m\), avec un taux d'apprentissage \(\nu = 0{,}1\). On donne \(0{,}9^{20} \approx @A04_facteur:4m@\).

Que vaut la prédiction \(F_{20}\) après 20 étapes, arrondie à deux décimales ?""",
        "options": [
            {"id": "a", "texte": "@A04_exact:2@", "explication": r"Oui : chaque étape corrige 10 % de l'écart restant, qui vaut \(10 \times 0{,}9^{20}\) après 20 étapes."},
            {"id": "b", "texte": "@A04_d_plein:2@", "explication": r"C'est la limite quand le nombre d'étapes tend vers l'infini, atteinte dès la première étape avec \(\nu = 1\). Avec \(\nu = 0{,}1\), chaque étape ne corrige que 10 % de l'écart restant."},
            {"id": "c", "texte": "@A04_d_residu:2@", "explication": r"C'est l'écart **restant**, \(10 \times 0{,}9^{20}\), pas la prédiction."},
            {"id": "d", "texte": "@A04_d_lineaire:2@", "explication": r"C'est \(20 \times 0{,}1 \times 10\), comme si chaque étape ajoutait \(0{,}1 \times 10\). Or les pseudo-résidus diminuent au fil des étapes ; sans cela, la prédiction dépasserait la moyenne des cibles."},
        ],
        "solution": r"""Réponse : \(F_{20} = 10\,(1 - 0{,}9^{20}) \approx @A04_exact:2m@\).

Notations : \(r_m = \bar y - F_m\) est l'écart restant après \(m\) étapes ; \(F_m\) est la même constante pour toutes les observations.

Étape 1 : la moyenne des pseudo-résidus vaut \(\bar y - F_{m-1} = r_{m-1}\), donc \(T_m = r_{m-1}\), \(F_m = F_{m-1} + \nu\,r_{m-1}\), et \(r_m = (1 - \nu)\,r_{m-1}\).

Étape 2 : \(r_m = (1 - \nu)^m\,r_0 = 10 \times 0{,}9^m\), donc \(F_{20} = 10 - 10 \times 0{,}9^{20} \approx 10 - @A04_d_residu:2m@ = @A04_exact:2m@\).

Lecture : avec un petit taux d'apprentissage, il faut plus d'arbres pour s'approcher de l'ajustement ; le nombre d'étapes devient un paramètre de régularisation, à choisir par validation. Vérification (script) : `GradientBoostingRegressor` de scikit-learn, avec une seule variable constante (aucune coupure possible, donc une seule feuille) et `init="zero"`, donne @A04_sklearn:6@.""",
    },
    {
        "id": "EB1-A05", "section": "mla", "theme": "Algorithme EM, mélange de deux gaussiennes", "difficulte": 4, "temps_cible_s": 150,
        "enonce": r"""On ajuste par l'algorithme EM un mélange de deux lois normales sur \(\mathbb{R}\), de poids fixés à 1/2 chacune et de variances fixées à 1 ; seules les moyennes \(\mu_1\) et \(\mu_2\) sont estimées. Les données sont \(x_1 = -1\), \(x_2 = 0\), \(x_3 = 3\), et l'on part de \(\mu_1 = 0\), \(\mu_2 = 2\). Une itération calcule, pour chaque point, la responsabilité \(r_i\) : la probabilité a posteriori, avec les paramètres courants, que \(x_i\) provienne de la composante 2 (étape E) ; puis elle remplace \(\mu_2\) par \(\sum_i r_i x_i / \sum_i r_i\) (étape M ; \(\mu_1\) est mis à jour de la même façon avec les poids \(1 - r_i\)). On donne \(e^{-2} \approx @A05_e_moins2:4m@\) et \(e^{-4} \approx @A05_e_moins4:4m@\).

Que vaut \(\mu_2\) après une itération, arrondi à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@A05_exact:3@", "explication": r"Oui : moyenne des \(x_i\) pondérée par les responsabilités \(r_i = 1/(1 + e^{-(2x_i - 2)})\)."},
            {"id": "b", "texte": "@A05_d_dur:3@", "explication": r"C'est l'affectation dure des k-moyennes : chaque point à la moyenne la plus proche, et seul \(x_3 = 3\) est plus proche de 2 que de 0. EM affecte chaque point à chaque composante avec un poids entre 0 et 1 : les points \(-1\) et 0 tirent un peu \(\mu_2\) vers le bas."},
            {"id": "c", "texte": "@A05_d_non_normalise:3@", "explication": r"C'est \(\sum_i r_i x_i\) sans la division par \(\sum_i r_i \approx @A05_somme_r2:3m@\) : la nouvelle moyenne est une moyenne pondérée, dont les poids doivent être normalisés."},
            {"id": "d", "texte": "@A05_d_moyenne:3@", "explication": r"C'est la moyenne de toutes les données, qui correspondrait à des responsabilités toutes égales. Les responsabilités dépendent de la distance de chaque point aux deux moyennes courantes."},
        ],
        "solution": r"""Réponse : \(\mu_2 \approx @A05_exact:3m@\).

Étape 1, responsabilités, par la formule de Bayes (les poids égaux 1/2 se simplifient) : \(r_i = \dfrac{\varphi(x_i - 2)}{\varphi(x_i) + \varphi(x_i - 2)}\), où \(\varphi(u) = e^{-u^2/2}/\sqrt{2\pi}\) est la densité normale centrée réduite. En divisant numérateur et dénominateur par \(\varphi(x_i - 2)\) : \(r_i = 1/(1 + e^{-\ell_i})\), avec \(\ell_i = \tfrac12\big(x_i^2 - (x_i - 2)^2\big) = 2x_i - 2\).

Étape 2, valeurs : \(\ell = @A05_logits:0@\), donc \(r \approx @A05_r2:4@\).

Étape 3, étape M : \(\sum_i r_i \approx @A05_somme_r2:4m@\) et \(\sum_i r_i x_i \approx @A05_somme_r2x:4m@\) (le point 0 ne contribue pas), d'où \(\mu_2 \approx @A05_exact:3m@\). Cette formule maximise en \(\mu_2\) le terme \(\sum_i r_i \ln \varphi(x_i - \mu_2)\) de l'espérance de la log-vraisemblance complète, dont la dérivée \(\sum_i r_i (x_i - \mu_2)\) s'annule en la moyenne pondérée.

Contrôle (script) : avec la mise à jour simultanée de \(\mu_1\) (qui devient @A05_mu1:3@), la log-vraisemblance des données passe de @A05_logv_avant:3@ à @A05_logv_apres:3@. Que EM ne fasse jamais baisser la log-vraisemblance est un théorème général, admis ici et constaté sur cet exemple.""",
    },
    {
        "id": "EB1-A06", "section": "mla", "theme": "Sélection de modèles : AIC et BIC", "difficulte": 2, "temps_cible_s": 90,
        "enonce": r"""Deux modèles sont ajustés par maximum de vraisemblance sur les mêmes \(n = 100\) observations. Le modèle A a \(k = 3\) paramètres et une log-vraisemblance maximale \(\ln \hat L = -140\) ; le modèle B a \(k = 6\) paramètres et \(\ln \hat L = -136\). On utilise \(\mathrm{AIC} = 2k - 2\ln \hat L\) et \(\mathrm{BIC} = k \ln n - 2\ln \hat L\), et chaque critère retient le modèle de plus petite valeur. On donne \(\ln 100 \approx @A06_ln_n:3m@\).

Quels modèles sont retenus ?""",
        "options": [
            {"id": "a", "texte": "AIC retient B, BIC retient A.", "explication": r"Oui : AIC @A06_aic_B:0@ contre @A06_aic_A:0@, BIC @A06_bic_A:1@ contre @A06_bic_B:1@."},
            {"id": "b", "texte": "AIC et BIC retiennent B.", "explication": r"BIC pénalise chaque paramètre par \(\ln 100\) au lieu de 2 : les 3 paramètres supplémentaires coûtent \(3 \ln 100 \approx @A06_seuil_bic:1m@\), plus que le gain de @A06_lr:0@ en \(-2\ln \hat L\)."},
            {"id": "c", "texte": "AIC et BIC retiennent A.", "explication": r"Pour AIC, les 3 paramètres supplémentaires coûtent \(2 \times 3 = 6\), moins que le gain de @A06_lr:0@ en \(-2\ln \hat L\) : AIC vaut @A06_aic_B:0@ pour B contre @A06_aic_A:0@ pour A."},
            {"id": "d", "texte": "AIC retient A, BIC retient B.", "explication": r"Les rôles sont inversés : BIC pénalise plus fortement que AIC dès que \(\ln n > 2\), c'est-à-dire \(n \ge 8\) ; c'est donc BIC qui retient le modèle le plus simple."},
        ],
        "solution": r"""Réponse : AIC retient B (@A06_aic_B:0@ contre @A06_aic_A:0@) ; BIC retient A (@A06_bic_A:1@ contre @A06_bic_B:1@).

Étape 1, AIC : A : \(2 \times 3 + 280 = @A06_aic_A:0m@\) ; B : \(2 \times 6 + 272 = @A06_aic_B:0m@\).

Étape 2, BIC : A : \(3 \times @A06_ln_n:3m@ + 280 \approx @A06_bic_A:1m@\) ; B : \(6 \times @A06_ln_n:3m@ + 272 \approx @A06_bic_B:1m@\).

Lecture : passer de A à B baisse \(-2\ln \hat L\) de @A06_lr:0@. AIC accepte les 3 paramètres supplémentaires si ce gain dépasse \(2 \times 3 = 6\) ; BIC exige plus de \(3 \ln n \approx @A06_seuil_bic:1m@\). Résultats classiques, admis ici : AIC est construit comme une estimation de l'erreur de prédiction au sens de Kullback-Leibler, BIC comme une approximation de la probabilité a posteriori du modèle, et BIC retrouve asymptotiquement le vrai modèle quand il figure parmi les candidats. Si A est emboîté dans B, le test du rapport de vraisemblance compare la statistique \(@A06_lr:0m@\) aux quantiles de la loi du khi-deux à 3 degrés de liberté (asymptotiquement, sous les conditions usuelles de régularité).""",
    },
    {
        "id": "EB1-A07", "section": "mla", "theme": "Étiquettes chevauchantes, variance d'une moyenne", "difficulte": 5, "temps_cible_s": 150,
        "enonce": r"""Les rendements quotidiens \(R_1, R_2, \dots\) sont des variables aléatoires indépendantes de même loi, d'espérance 0 et de variance \(\sigma^2 > 0\). Avec \(h = 10\), l'étiquette de la date \(t\) est le rendement des \(h\) jours suivants, \(Y_t = R_{t+1} + \dots + R_{t+h}\), pour \(t = 1, \dots, n\) avec \(n = 250\) : deux étiquettes de dates proches partagent des jours. On estime l'espérance des étiquettes par \(\bar Y = (Y_1 + \dots + Y_n)/n\). Un analyste calcule la variance de \(\bar Y\) comme si les étiquettes étaient indépendantes, soit \(\mathrm{Var}(Y_1)/n\).

Que vaut le rapport \(\mathrm{Var}(\bar Y)\big/\big(\mathrm{Var}(Y_1)/n\big)\), arrondi à deux décimales ?""",
        "options": [
            {"id": "a", "texte": "@A07_exact:2@", "explication": r"Oui : \(h - (h^2 - 1)/(3n) = 10 - 99/750\)."},
            {"id": "b", "texte": "@A07_d_limite:2@", "explication": r"C'est la limite du rapport quand \(n \to \infty\) à \(h\) fixé. Pour \(n = 250\), les jours des deux bords sont couverts par moins de \(h\) étiquettes, et le rapport exact est \(h - (h^2 - 1)/(3n)\)."},
            {"id": "c", "texte": "@A07_d_racine:2@", "explication": r"C'est la racine carrée du rapport, le facteur d'erreur sur l'**écart-type** de \(\bar Y\). La question porte sur la variance."},
            {"id": "d", "texte": "@A07_d_naif:2@", "explication": r"C'est l'hypothèse d'indépendance de l'analyste. Or \(\mathrm{Cov}(Y_s, Y_t) = \sigma^2 (h - \lvert s - t \rvert)\) pour \(\lvert s - t \rvert < h\) : par exemple, deux étiquettes distantes de 3 jours ont une corrélation de @A07_correlation_3:1@."},
        ],
        "solution": r"""Réponse : \(h - \dfrac{h^2 - 1}{3n} = 10 - \dfrac{99}{750} = @A07_exact:3m@\), soit @A07_exact:2@.

Notations : pour chaque jour \(s\), \(c_s\) est le nombre d'étiquettes \(Y_1, \dots, Y_n\) qui contiennent \(R_s\).

Étape 1 : \(\sum_{t=1}^{n} Y_t = \sum_s c_s R_s\), donc, par indépendance des \(R_s\), \(\mathrm{Var}\big(\sum_t Y_t\big) = \sigma^2 \sum_s c_s^2\) ; et \(\mathrm{Var}(Y_1) = h\sigma^2\).

Étape 2, décompte : \(R_s\) est dans \(Y_t\) si et seulement si \(s - h \le t \le s - 1\). Les jours \(s = h + 1, \dots, n + 1\), au nombre de @A07_nb_pleins@, sont dans \(h\) étiquettes ; aux deux bords, \(c_s\) prend une fois chacune des valeurs \(1, 2, \dots, h - 1\). Donc \(\sum_s c_s^2 = @A07_nb_pleins@ \times 100 + 2 \times @A07_somme_carres@ = @A07_somme_c2@\).

Étape 3 : rapport \(= \dfrac{\sigma^2 \sum_s c_s^2/n^2}{h\sigma^2/n} = \dfrac{@A07_somme_c2@}{250 \times 10} = @A07_exact:3m@\). En général \(\sum_s c_s^2 = (n - h + 1)h^2 + (h - 1)h(2h - 1)/3\), d'où la forme \(h - (h^2 - 1)/(3n)\), qui tend vers \(h\).

Conséquence : à variance de \(Y_1\) connue, la statistique t de \(\bar Y\) calculée comme si les étiquettes étaient indépendantes est gonflée d'un facteur \(\sqrt{@A07_exact:3m@} \approx @A07_d_racine:2m@\). Il faut des erreurs standard qui tiennent compte de l'autocorrélation, ou ne garder qu'une étiquette sur \(h\). Simulation (script, \(\sigma = 1\)) : \(\mathrm{Var}(\bar Y) \approx @A07_mc:4@\), contre @A07_var_theorique:4@ en théorie.""",
    },
    {
        "id": "EB1-A08", "section": "mla", "theme": "Processus gaussiens, loi a posteriori", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Soit \(f\) un processus gaussien sur \(\mathbb{R}\), de moyenne nulle et de noyau \(k(x, x') = \exp\big(-(x - x')^2/2\big)\) : pour tous points \(x_1, \dots, x_m\), le vecteur \((f(x_1), \dots, f(x_m))\) est gaussien centré de matrice de covariance \(\big(k(x_i, x_j)\big)_{i,j}\). On observe \(Y = f(0) + \varepsilon\), où \(\varepsilon\) suit la loi normale centrée de variance 0,5, indépendante de \(f\), et la valeur observée est \(Y = 2\). On donne \(e^{-1/2} \approx @A08_k01:4m@\).

Quelles sont la moyenne et la variance de la loi de \(f(1)\) sachant \(Y = 2\), arrondies à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "moyenne @A08_moyenne:3@ ; variance @A08_variance:3@", "explication": r"Oui : moyenne \((c/v)\,y\) et variance \(1 - c^2/v\), avec \(c = e^{-1/2}\) et \(v = 1{,}5\)."},
            {"id": "b", "texte": "moyenne @A08_d_moy_sans_bruit:3@ ; variance @A08_d_var_sans_bruit:3@", "explication": r"C'est le résultat sans le bruit d'observation, comme si \(Y = f(0)\) : on a divisé par \(k(0, 0) = 1\) au lieu de \(k(0, 0) + 0{,}5\). Le bruit rend l'observation moins informative : la moyenne a posteriori est plus proche de 0 et la variance plus grande."},
            {"id": "c", "texte": "moyenne @A08_moyenne:3@ ; variance 1,000", "explication": r"C'est la variance a priori. L'observation en 0 renseigne aussi sur \(f(1)\), puisque \(f(0)\) et \(f(1)\) sont corrélées (covariance \(e^{-1/2}\)) : la variance a posteriori est plus petite, et elle ne dépend pas de la valeur observée."},
            {"id": "d", "texte": "moyenne @A08_moyenne:3@ ; variance @A08_d_var_k:3@", "explication": r"La covariance \(k(1, 0)\) n'a pas été élevée au carré dans la réduction de variance : la formule est \(k(1, 1) - k(1, 0)^2/\big(k(0, 0) + 0{,}5\big)\)."},
        ],
        "solution": r"""Réponse : moyenne \(\dfrac{e^{-1/2}}{1{,}5} \times 2 \approx @A08_moyenne:3m@\) ; variance \(1 - \dfrac{e^{-1}}{1{,}5} \approx @A08_variance:3m@\).

Notations : \(F_1 = f(1)\), \(c = \mathrm{Cov}(F_1, Y) = \mathrm{Cov}(f(1), f(0)) = k(1, 0) = e^{-1/2}\) (\(\varepsilon\) est indépendant de \(f\)), \(v = \mathrm{Var}(Y) = k(0, 0) + 0{,}5 = 1{,}5\).

Étape 1 : \((F_1, Y)\) est un vecteur gaussien centré, image linéaire du vecteur gaussien \((f(0), f(1), \varepsilon)\), de variances 1 et \(v\) et de covariance \(c\).

Étape 2, conditionnement gaussien : \(F_1 - (c/v)\,Y\) a une covariance nulle avec \(Y\) (\(c - (c/v)\,v = 0\)) ; le couple étant gaussien, elle est indépendante de \(Y\). Donc, sachant \(Y = y\), \(F_1\) suit la loi normale de moyenne \((c/v)\,y\) et de variance \(\mathrm{Var}\big(F_1 - (c/v)\,Y\big) = 1 - 2c^2/v + c^2/v = 1 - c^2/v\).

Étape 3, valeurs : moyenne \(@A08_k01:4m@ \times 2/1{,}5 \approx @A08_moyenne:3m@\), variance \(1 - @A08_k01:4m@^2/1{,}5 \approx @A08_variance:3m@\).

Lecture : la moyenne a posteriori \(k(1, 0)\,(k(0, 0) + 0{,}5)^{-1}\,y\) est aussi la prédiction de la régression ridge à noyau qui minimise \(\sum_i (y_i - g(x_i))^2 + \lambda \lVert g \rVert^2\) dans l'espace du noyau, avec \(\lambda = 0{,}5\). Vérification (script) : `GaussianProcessRegressor` de scikit-learn (`alpha=0.5`, noyau fixé), et simulation : moyenne @A08_moyenne_mc:3@, variance @A08_variance_mc:3@.""",
    },
]
