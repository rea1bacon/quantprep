"""Source des questions du test de positionnement.

Les textes sont des chaînes brutes : les antislashs LaTeX s'écrivent une seule fois.
Aucun nombre calculé n'est écrit à la main : un jeton @CLE@ ou @CLE:3@ (3 décimales)
ou @CLE:3m@ (3 décimales, en mode mathématique) est remplacé par la valeur de
verif/diagnostic.out.json. Seules les données des énoncés sont écrites en clair.

Dans chaque question, l'option d'identifiant "a" est la bonne ; l'ordre d'affichage
est mélangé par la page.
"""

ID = "diagnostic"
TITRE = "Test de positionnement"
VERIF = "verif/diagnostic.out.json"
OPTIONS = {"retour_arriere": True, "penalite": 0}
PRESENTATION = (
    "Ce test de positionnement mesure ton niveau initial sur les quatre domaines de la plateforme : code et science "
    "des données, mathématiques, ML fondamental et ML avancé. Passe-le d'une traite, sans notes ni recherche, comme "
    "un test en ligne.\n\n"
    "Le chronomètre global démarre au clic et ne s'arrête pas. Les temps par section sont indicatifs : le "
    "temps passé sur chaque question est enregistré."
)
AVERTISSEMENTS = [
    "Les questions de mathématiques et de ML sont présentées ici en choix multiple à une seule bonne réponse.",
    "Ici le retour en arrière est permis et il n'y a pas de pénalité : ce sont les règles de cette plateforme, pas "
    "celles de tout test en ligne.",
    "Si tu ne sais pas, choisis « Je ne sais pas » plutôt que de deviner : le but est de mesurer, pas de marquer.",
]

CONVENTIONS = (
    r"Conventions de notation : une majuscule désigne une variable aléatoire, une minuscule une valeur "
    r"ou un paramètre fixe, le gras un vecteur ou une matrice. \(\mathbb{P}\) désigne une probabilité, "
    r"\(\mathbb{E}\) une espérance, \(\mathrm{Var}\) une variance."
)

SECTIONS = [
    {"id": "code", "nom": "Coding, data science", "budget_min": 15},
    {"id": "math", "nom": "Mathématiques", "budget_min": 16},
    {"id": "mlf", "nom": "Machine Learning fondamental", "budget_min": 6},
    {"id": "mla", "nom": "Machine Learning avancé", "budget_min": 9},
]

CODE = {
    "id": "DIAG-C1",
    "type": "code",
    "section": "code",
    "theme": "pandas : groupby, décalage, rangs",
    "difficulte": 3,
    "temps_cible_s": 900,
    "enonce": """## Coefficient d'information moyen

Le coefficient d'information d'un signal, à une date donnée, est la corrélation de rang (corrélation de Spearman) entre la valeur du signal sur chaque actif à cette date et le rendement futur de chaque actif. Il mesure si le signal classe correctement les actifs.

Dans le notebook, définis la fonction `ic_moyen(df)` qui renvoie un `float`.

### Entrée

`df` est un DataFrame pandas dont les colonnes sont exactement :

| Colonne | Type | Contenu |
|---|---|---|
| `date` | chaîne au format `AAAA-MM-JJ` | date de l'observation |
| `actif` | chaîne | identifiant de l'actif |
| `prix` | nombre, strictement positif ou `NaN` | prix de clôture de l'actif à cette date |
| `signal` | nombre ou `NaN` | valeur du signal pour cet actif à cette date |

Garanties : au plus une ligne par couple (`date`, `actif`) ; lignes dans un ordre **quelconque** ; le DataFrame peut être vide (zéro ligne, mais les quatre colonnes présentes) ; tous les actifs n'ont pas nécessairement une ligne à chaque date.

### Calcul demandé

1. **Rendement futur.** Pour une ligne (date `d`, actif `a`) de prix `p`, on considère la ligne suivante du même actif `a` dans l'ordre chronologique, c'est-à-dire la ligne de `a` dont la date est la plus petite date strictement supérieure à `d` parmi les lignes de `a`. Si son prix est `q`, le rendement futur de la ligne est `q / p - 1`. Il est manquant (`NaN`) si la ligne suivante n'existe pas, ou si `p` ou `q` est `NaN`.
2. **Lignes valides.** Une ligne est valide si son signal et son rendement futur sont tous deux non manquants.
3. **Coefficient d'information d'une date.** Pour chaque date, on ne garde que ses lignes valides. On calcule le rang des signaux et le rang des rendements futurs parmi ces lignes, par ordre croissant, les valeurs égales recevant la **moyenne** des rangs qu'elles occupent (par exemple les valeurs 5, 7, 7, 9 ont les rangs 1 ; 2,5 ; 2,5 ; 4). Le coefficient d'information de la date est la corrélation de Pearson entre les rangs des signaux et les rangs des rendements futurs.
4. **Dates ignorées.** Une date est ignorée si elle a moins de 3 lignes valides, ou si tous ses signaux valides sont égaux, ou si tous ses rendements futurs valides sont égaux.
5. **Résultat.** La moyenne arithmétique des coefficients d'information des dates non ignorées, sous forme de `float` ; `float("nan")` si toutes les dates sont ignorées.

Les égalités entre valeurs sont des égalités exactes entre flottants ; les tests ne contiennent pas de valeurs presque égales. Tolérance de comparaison : `1e-9`. Limite de temps : 10 secondes par test.

### Exemple

| date | actif | prix | signal |
|---|---|---|---|
| 2026-01-05 | A | 100 | 0.3 |
| 2026-01-05 | B | 50 | -1.0 |
| 2026-01-05 | C | 20 | 2.0 |
| 2026-01-06 | A | 101 | 1.0 |
| 2026-01-06 | B | 49 | 2.0 |
| 2026-01-06 | C | 22 | 3.0 |

Rendements futurs au 2026-01-05 : A : 101/100 - 1 = 0,01 ; B : 49/50 - 1 = -0,02 ; C : 22/20 - 1 = 0,10. Les rangs des signaux (A : 2, B : 1, C : 3) coïncident avec ceux des rendements : le coefficient d'information du 2026-01-05 vaut 1. Le 2026-01-06 n'a aucun rendement futur, il est ignoré. La fonction renvoie `1.0`.

### Fonctionnement

- Le notebook s'exécute avec Python 3.9 et pandas 1.4.2, l'environnement des notebooks de la plateforme.
- Le fichier `prix_signaux.csv` (mêmes colonnes) est dans le répertoire courant pour explorer : `pd.read_csv("prix_signaux.csv")`.
- **Tests visibles** exécute les 3 tests de l'énoncé sur ta fonction. **Soumettre** exécute les 16 tests cachés : seul le nombre de réussites s'affiche, et la dernière soumission compte. Un code jamais soumis ne rapporte rien.
- Raccourcis : Maj+Entrée exécute la cellule et passe à la suivante, Ctrl+Entrée l'exécute sans changer de cellule.
- Autocomplétion : les suggestions apparaissent en tapant (ou avec Ctrl+Espace), l'aide aux paramètres après `(`, la documentation au survol d'un nom. Elle connaît les variables des cellules déjà exécutées. Dans une fonction, elle ne connaît le type d'un paramètre que s'il est annoté, d'où `df: pd.DataFrame` dans la cellule de départ.
""",
    "dossier": "coding/diagnostic",
    "module_tests": "coding/diagnostic/tests.py",
    "fonction": "ic_moyen",
    "cellules_initiales": [
        "import numpy as np\nimport pandas as pd\n\ndf = pd.read_csv(\"prix_signaux.csv\")\ndf.head()",
        "def ic_moyen(df: pd.DataFrame) -> float:\n    # À compléter\n    return float(\"nan\")",
        "ic_moyen(df)",
    ],
    "tests_visibles": 3,
    "tests_caches": 16,
    "fichier_correction": "coding/diagnostic/_correction.py",
    "verif": {"script": "coding/diagnostic/tests.py"},
}

CODES = [CODE]

QUESTIONS = [
    # ------------------------------------------------------------------ MATH
    {
        "id": "DIAG-M1", "section": "math", "theme": "Formule de Bayes", "difficulte": 2, "temps_cible_s": 120,
        "enonce": (
            r"Une maladie touche 2 % d'une population. Un test de dépistage a une sensibilité de 95 % (probabilité "
            r"que le test soit positif sachant que la personne est malade) et une spécificité de 90 % (probabilité "
            r"que le test soit négatif sachant que la personne n'est pas malade)."
            "\n\n"
            r"On tire une personne uniformément au hasard dans la population et son test est positif. Quelle est "
            r"la probabilité qu'elle soit malade, arrondie à trois décimales ?"
        ),
        "options": [
            {"id": "a", "texte": "@M1_exact:3@", "explication": ""},
            {"id": "b", "texte": "0,950", "explication": r"C'est la sensibilité, \(\mathbb{P}(T \mid M)\) : la probabilité d'un test positif sachant la maladie. La question demande la probabilité conditionnelle dans l'autre sens, \(\mathbb{P}(M \mid T)\)."},
            {"id": "c", "texte": "@M1_joint:3@", "explication": r"C'est la probabilité jointe \(\mathbb{P}(M \cap T)\) d'être malade et positif. Il manque la division par \(\mathbb{P}(T)\), qui traduit le fait qu'on sait déjà que le test est positif."},
            {"id": "d", "texte": "@M1_p_positif:3@", "explication": r"C'est \(\mathbb{P}(T)\), la probabilité qu'une personne quelconque soit positive : le dénominateur de la formule de Bayes, pas le résultat."},
        ],
        "solution": (
            r"Réponse : @M1_exact:3@, soit exactement 19/117."
            "\n\n"
            r"Notations : \(M\) est l'événement « la personne est malade », \(T\) l'événement « le test est positif », \(M^c\) le complémentaire de \(M\)."
            "\n\n"
            r"Étape 1, probabilité d'être malade et positif : \(\mathbb{P}(M \cap T) = \mathbb{P}(T \mid M)\,\mathbb{P}(M) = 0{,}95 \times 0{,}02 = @M1_joint:3m@\)."
            "\n\n"
            r"Étape 2, probabilité d'être sain et positif. La spécificité est \(\mathbb{P}(T^c \mid M^c) = 0{,}90\), donc \(\mathbb{P}(T \mid M^c) = 0{,}10\) et \(\mathbb{P}(M^c \cap T) = 0{,}10 \times 0{,}98 = @M1_faux_positifs:3m@\)."
            "\n\n"
            r"Étape 3, formule des probabilités totales, car \(M\) et \(M^c\) forment une partition : \(\mathbb{P}(T) = @M1_joint:3m@ + @M1_faux_positifs:3m@ = @M1_p_positif:3m@\)."
            "\n\n"
            r"Étape 4, définition de la probabilité conditionnelle : \(\mathbb{P}(M \mid T) = \mathbb{P}(M \cap T) / \mathbb{P}(T) = @M1_joint:3m@ / @M1_p_positif:3m@ \approx @M1_exact:3m@\)."
            "\n\n"
            r"Mécanisme à retenir : les personnes saines sont 49 fois plus nombreuses que les malades, donc même un taux de faux positifs de 10 % produit plus de positifs sains que de positifs malades."
        ),
    },
    {
        "id": "DIAG-M2", "section": "math", "theme": "Marche aléatoire, ruine du joueur", "difficulte": 3, "temps_cible_s": 150,
        "enonce": (
            r"Une particule part de la position 3 sur les entiers. À chaque pas, indépendamment des pas précédents, "
            r"elle se déplace de +1 avec probabilité 1/2 et de −1 avec probabilité 1/2. Elle s'arrête dès qu'elle "
            r"atteint la position 0 ou la position 10."
            "\n\n"
            r"Quelle est l'espérance du nombre de pas effectués jusqu'à l'arrêt ?"
        ),
        "options": [
            {"id": "a", "texte": "@M2_exact@", "explication": ""},
            {"id": "b", "texte": "@M2_d_depart_fois_barriere@", "explication": r"C'est \(3 \times 10\), le produit de la position de départ par la barrière haute. La formule correcte multiplie les distances aux deux barrières : \(3 \times (10 - 3)\)."},
            {"id": "c", "texte": "@M2_d_depart_carre@", "explication": r"C'est \(3^2\). Le carré de la distance n'apparaît que comme solution particulière de l'équation de récurrence ; il ne s'annule pas en 10 et ne respecte donc pas la condition au bord."},
            {"id": "d", "texte": "@M2_d_distance_haut@", "explication": r"C'est la distance à la barrière haute, c'est-à-dire le nombre minimal de pas pour atteindre 10, pas une espérance : la particule fait des allers-retours et peut aussi finir en 0."},
        ],
        "solution": (
            r"Réponse : @M2_exact@ pas."
            "\n\n"
            r"Notations : \(n = 10\) est la barrière haute, \(i \in \{0, \dots, n\}\) une position de départ, \(d(i)\) l'espérance du nombre de pas jusqu'à l'arrêt en partant de \(i\). On cherche \(d(3)\)."
            "\n\n"
            r"Étape 0, \(d(i)\) est finie. Depuis toute position, 10 pas consécutifs vers le haut provoquent l'arrêt, ce qui arrive avec probabilité \(2^{-10}\) sur chaque bloc de 10 pas, indépendamment d'un bloc à l'autre. Le nombre de blocs nécessaires est donc majoré par une variable géométrique, d'espérance finie."
            "\n\n"
            r"Étape 1, conditionnement sur le premier pas. Aux bords, \(d(0) = d(n) = 0\). Pour \(0 < i < n\), le premier pas coûte 1 et mène en \(i+1\) ou \(i-1\) avec probabilité 1/2 chacun ; les pas suivants étant indépendants du premier, la suite repart comme une marche neuve : \(d(i) = 1 + \tfrac12 d(i+1) + \tfrac12 d(i-1)\)."
            "\n\n"
            r"Étape 2, résolution. L'équation se réécrit \(d(i+1) - 2d(i) + d(i-1) = -2\). La suite \(-i^2\) est une solution particulière, car \(-(i+1)^2 + 2i^2 - (i-1)^2 = -2\). Les solutions de l'équation homogène (second membre nul) sont les suites affines \(a\,i + b\), car une différence seconde nulle signifie des accroissements constants. Donc \(d(i) = -i^2 + a\,i + b\)."
            "\n\n"
            r"Étape 3, conditions aux bords. \(d(0) = 0\) donne \(b = 0\) ; \(d(n) = 0\) donne \(a = n\). Donc \(d(i) = i\,(n - i)\) et \(d(3) = 3 \times 7 = @M2_exact@\)."
            "\n\n"
            r"Question suivante probable : la probabilité de finir en 10. Elle vaut \(i/n = @M2_p_haut_exact:1m@\) (même méthode, avec l'équation \(p(i) = \tfrac12 p(i+1) + \tfrac12 p(i-1)\), \(p(0) = 0\), \(p(n) = 1\))."
        ),
    },
    {
        "id": "DIAG-M3", "section": "math", "theme": "Linéarité de l'espérance, indicatrices", "difficulte": 3, "temps_cible_s": 120,
        "enonce": (
            r"On lance 5 dés équilibrés à 6 faces, indépendants. Quelle est l'espérance du nombre de faces "
            r"distinctes obtenues, arrondie à trois décimales ? Par exemple, le tirage 2, 2, 5, 6, 5 donne 3 faces distinctes."
        ),
        "options": [
            {"id": "a", "texte": "@M3_exact:3@", "explication": ""},
            {"id": "b", "texte": "@M3_faces_absentes:3@", "explication": r"C'est \(6 \times (5/6)^5\), l'espérance du nombre de faces qui n'apparaissent pas. Le nombre de faces distinctes est 6 moins cette quantité."},
            {"id": "c", "texte": "@M3_moins_paires:3@", "explication": r"C'est 5 moins l'espérance du nombre de paires de dés égaux, \(\binom{5}{2}/6\). Ce décompte est faux dès que trois dés montrent la même face : ils forment 3 paires alors qu'il n'y a que 2 dés en trop."},
            {"id": "d", "texte": "3,500", "explication": r"C'est l'espérance de la valeur d'un dé. La question porte sur le nombre de faces différentes, pas sur les valeurs."},
        ],
        "solution": (
            r"Réponse : \(6\,(1 - (5/6)^5) \approx @M3_exact:3m@\)."
            "\n\n"
            r"Notations : pour chaque face \(k \in \{1, \dots, 6\}\), \(I_k\) est la variable aléatoire qui vaut 1 si la face \(k\) apparaît au moins une fois parmi les 5 dés, 0 sinon. Le nombre de faces distinctes est \(N = I_1 + \dots + I_6\)."
            "\n\n"
            r"Étape 1 : la face \(k\) est absente si chacun des 5 dés, indépendants, montre une autre face, ce qui a pour probabilité \((5/6)^5\). Donc \(\mathbb{E}[I_k] = \mathbb{P}(I_k = 1) = 1 - (5/6)^5\)."
            "\n\n"
            r"Étape 2 : par linéarité de l'espérance, qui ne demande aucune indépendance entre les \(I_k\) (et ils ne sont pas indépendants), \(\mathbb{E}[N] = 6\,(1 - (5/6)^5) \approx @M3_exact:3m@\)."
        ),
    },
    {
        "id": "DIAG-M4", "section": "math", "theme": "Matrices semi-définies positives, corrélation", "difficulte": 4, "temps_cible_s": 150,
        "enonce": (
            r"Soit \(\rho\) un réel et \(\mathbf{C}\) la matrice \(3 \times 3\) dont les coefficients diagonaux valent 1 "
            r"et tous les autres valent \(\rho\). Pour quelles valeurs de \(\rho\) la matrice \(\mathbf{C}\) est-elle "
            r"semi-définie positive, c'est-à-dire telle que \(\mathbf{x}^\top \mathbf{C}\,\mathbf{x} \ge 0\) pour tout "
            r"vecteur \(\mathbf{x}\) de \(\mathbb{R}^3\) ? C'est la condition pour que \(\mathbf{C}\) soit la matrice "
            r"de corrélation de trois variables aléatoires."
        ),
        "options": [
            {"id": "a", "texte": r"\(-1/2 \le \rho \le 1\)", "explication": ""},
            {"id": "b", "texte": r"\(-1 \le \rho \le 1\)", "explication": r"Cette condition suffit pour une seule paire de variables, pas pour trois à la fois. Avec \(\rho = -1\), la valeur propre \(1 + 2\rho\) vaut @M4_vp_simple_rho_moins_un@ : trois variables ne peuvent pas être toutes parfaitement anticorrélées deux à deux."},
            {"id": "c", "texte": r"\(0 \le \rho \le 1\)", "explication": r"Trop restrictif : des corrélations négatives modérées sont possibles. Avec \(\rho = @M4_ex_rho:1m@\), les valeurs propres sont @M4_ex_vp_simple:1@ et @M4_ex_vp_double:1@, toutes positives."},
            {"id": "d", "texte": r"\(-1/3 \le \rho \le 1\)", "explication": r"La borne inférieure générale est \(-1/(n-1)\) pour une matrice \(n \times n\). \(-1/3\) correspond à \(n = 4\) ; ici \(n = 3\)."},
        ],
        "solution": (
            r"Réponse : \(-1/2 \le \rho \le 1\)."
            "\n\n"
            r"Notations : \(\mathbf{I}\) est la matrice identité \(3 \times 3\), \(\mathbf{1}\) le vecteur de \(\mathbb{R}^3\) dont les trois coordonnées valent 1."
            "\n\n"
            r"Étape 1, écriture utile : \(\mathbf{C} = (1 - \rho)\,\mathbf{I} + \rho\,\mathbf{1}\mathbf{1}^\top\), car \(\mathbf{1}\mathbf{1}^\top\) est la matrice dont tous les coefficients valent 1."
            "\n\n"
            r"Étape 2, valeurs propres. Pour \(\mathbf{x} = \mathbf{1}\) : \(\mathbf{1}\mathbf{1}^\top \mathbf{1} = 3\,\mathbf{1}\), donc \(\mathbf{C}\mathbf{1} = (1 - \rho + 3\rho)\,\mathbf{1} = (1 + 2\rho)\,\mathbf{1}\). Pour tout \(\mathbf{x}\) orthogonal à \(\mathbf{1}\) (somme des coordonnées nulle) : \(\mathbf{1}\mathbf{1}^\top \mathbf{x} = \mathbf{0}\), donc \(\mathbf{C}\mathbf{x} = (1 - \rho)\,\mathbf{x}\). Les valeurs propres sont donc \(1 + 2\rho\), simple, et \(1 - \rho\), double puisque l'orthogonal de \(\mathbf{1}\) est de dimension 2."
            "\n\n"
            r"Étape 3, lien avec la définition. \(\mathbf{C}\) est symétrique réelle, donc diagonalisable dans une base orthonormée \((\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3)\) de vecteurs propres, de valeurs propres \(\lambda_1, \lambda_2, \lambda_3\) (théorème spectral). En décomposant \(\mathbf{x}\) dans cette base : \(\mathbf{x}^\top \mathbf{C}\,\mathbf{x} = \sum_j \lambda_j\,(\mathbf{u}_j^\top \mathbf{x})^2\). Cette somme est positive pour tout \(\mathbf{x}\) si et seulement si tous les \(\lambda_j\) sont positifs ou nuls (prendre \(\mathbf{x} = \mathbf{u}_j\) pour la réciproque)."
            "\n\n"
            r"Étape 4 : \(1 + 2\rho \ge 0\) et \(1 - \rho \ge 0\) donnent \(-1/2 \le \rho \le 1\)."
            "\n\n"
            r"Généralisation, par le même calcul : pour \(n\) variables, les valeurs propres sont \(1 + (n-1)\rho\) et \(1 - \rho\), d'où \(-1/(n-1) \le \rho \le 1\)."
        ),
    },
    {
        "id": "DIAG-M5", "section": "math", "theme": "Processus AR(1), stationnarité", "difficulte": 2, "temps_cible_s": 90,
        "enonce": (
            r"Soit \((\varepsilon_t)\) une suite de variables aléatoires indépendantes, d'espérance 0 et de variance 1, "
            r"et \((X_t)\) un processus stationnaire au second ordre (son espérance et sa variance ne dépendent pas "
            r"de la date \(t\)) vérifiant, pour tout entier \(t\), \(X_t = 0{,}8\,X_{t-1} + \varepsilon_t\), avec "
            r"\(\varepsilon_t\) indépendant de \(X_{t-1}\)."
            "\n\n"
            r"Quelle est la variance de \(X_t\), arrondie à deux décimales ?"
        ),
        "options": [
            {"id": "a", "texte": "@M5_exact:2@", "explication": ""},
            {"id": "b", "texte": "@M5_d_un_moins_phi:2@", "explication": r"C'est \(1/(1 - \varphi)\) avec \(\varphi = 0{,}8\). Ce facteur s'applique à l'espérance quand l'équation contient une constante. Pour la variance, le coefficient \(\varphi\) intervient au carré."},
            {"id": "c", "texte": "@M5_d_un_sur_phi2:2@", "explication": r"C'est \(1/\varphi^2\), qui ne provient d'aucune équation du modèle : l'équation de stationnarité est \(v = \varphi^2 v + 1\)."},
            {"id": "d", "texte": "@M5_d_un_plus_phi2:2@", "explication": r"C'est \(1 + \varphi^2\), les deux premiers termes seulement de la série \(\sum_{k \ge 0} \varphi^{2k}\) : la variance cumule les chocs de toutes les dates passées, pas des deux dernières."},
        ],
        "solution": (
            r"Réponse : \(1/(1 - 0{,}8^2) \approx @M5_exact:2m@\)."
            "\n\n"
            r"Notations : \(\varphi = 0{,}8\) est le coefficient d'autorégression, \(v = \mathrm{Var}(X_t)\), qui ne dépend pas de \(t\) par stationnarité."
            "\n\n"
            r"Étape 1 : \(\varepsilon_t\) et \(X_{t-1}\) sont indépendants, donc la variance de la somme est la somme des variances : \(\mathrm{Var}(X_t) = \varphi^2\,\mathrm{Var}(X_{t-1}) + \mathrm{Var}(\varepsilon_t)\)."
            "\n\n"
            r"Étape 2 : par stationnarité, \(v = \varphi^2 v + 1\), donc \(v = 1/(1 - \varphi^2) = 1/0{,}36 \approx @M5_exact:2m@\)."
            "\n\n"
            r"Condition de validité : une solution stationnaire de cette forme existe si et seulement si \(|\varphi| < 1\) ; pour \(|\varphi| \ge 1\) l'équation \(v = \varphi^2 v + 1\) n'a pas de solution positive."
        ),
    },
    {
        "id": "DIAG-M6", "section": "math", "theme": "Statistiques d'ordre", "difficulte": 3, "temps_cible_s": 120,
        "enonce": (
            r"Soient \(U_1, U_2, U_3, U_4\) quatre variables aléatoires indépendantes de loi uniforme sur \([0, 1]\). "
            r"Quelle est l'espérance de \(\max(U_1, \dots, U_4) - \min(U_1, \dots, U_4)\) ?"
        ),
        "options": [
            {"id": "a", "texte": "3/5", "explication": ""},
            {"id": "b", "texte": "4/5", "explication": r"C'est l'espérance du maximum seul, \(n/(n+1)\) avec \(n = 4\). Il faut encore soustraire l'espérance du minimum, \(1/(n+1)\)."},
            {"id": "c", "texte": "1/2", "explication": r"C'est la valeur pour \(n = 3\) variables : \((n-1)/(n+1) = 2/4\)."},
            {"id": "d", "texte": "3/4", "explication": r"C'est \((n-1)/n\). Le bon dénominateur est \(n + 1\) : \(n\) points découpent le segment en \(n + 1\) intervalles de même longueur moyenne, et l'étendue en couvre \(n - 1\)."},
        ],
        "solution": (
            r"Réponse : 3/5."
            "\n\n"
            r"Notations : \(n = 4\), \(M = \max(U_1, \dots, U_n)\), \(L = \min(U_1, \dots, U_n)\)."
            "\n\n"
            r"Étape 1, loi du maximum. Pour \(x \in [0, 1]\), \(M \le x\) si et seulement si tous les \(U_i\) sont inférieurs à \(x\) ; par indépendance, \(\mathbb{P}(M \le x) = x^n\). La densité de \(M\) est donc \(n\,x^{n-1}\) et \(\mathbb{E}[M] = \int_0^1 x \cdot n\,x^{n-1}\,dx = n/(n+1) = 4/5\)."
            "\n\n"
            r"Étape 2, minimum par symétrie. Les variables \(1 - U_i\) sont aussi indépendantes et uniformes sur \([0, 1]\), et \(L = 1 - \max(1 - U_1, \dots, 1 - U_n)\). Donc \(\mathbb{E}[L] = 1 - n/(n+1) = 1/(n+1) = 1/5\)."
            "\n\n"
            r"Étape 3, linéarité : \(\mathbb{E}[M - L] = (n-1)/(n+1) = 3/5\)."
        ),
    },
    {
        "id": "DIAG-M7", "section": "math", "theme": "Tests multiples", "difficulte": 2, "temps_cible_s": 90,
        "enonce": (
            r"On réalise 20 tests statistiques indépendants, chacun au niveau 5 % : lorsque son hypothèse nulle est "
            r"vraie, chaque test la rejette avec probabilité exactement 0,05. Les 20 hypothèses nulles sont toutes vraies."
            "\n\n"
            r"Quelle est la probabilité qu'au moins un des 20 tests rejette son hypothèse nulle, arrondie à trois décimales ?"
        ),
        "options": [
            {"id": "a", "texte": "@M7_exact:3@", "explication": ""},
            {"id": "b", "texte": "@M7_d_boole:3@", "explication": r"C'est \(20 \times 0{,}05\), qui est l'espérance du nombre de rejets et un majorant de la probabilité cherchée (inégalité de Boole), pas cette probabilité : les événements de rejet ne sont pas disjoints."},
            {"id": "c", "texte": "0,050", "explication": r"C'est la probabilité de rejet d'un seul test. Répéter 20 tests multiplie les occasions de rejeter à tort."},
            {"id": "d", "texte": "@M7_aucun_rejet:3@", "explication": r"C'est \(0{,}95^{20}\), la probabilité qu'aucun test ne rejette : l'événement complémentaire de celui demandé."},
        ],
        "solution": (
            r"Réponse : \(1 - 0{,}95^{20} \approx @M7_exact:3m@\)."
            "\n\n"
            r"Étape 1 : passer au complémentaire. « Au moins un rejet » est le contraire de « aucun rejet »."
            "\n\n"
            r"Étape 2 : chaque test ne rejette pas avec probabilité 0,95 ; par indépendance, la probabilité qu'aucun des 20 ne rejette est \(0{,}95^{20} \approx @M7_aucun_rejet:3m@\)."
            "\n\n"
            r"Étape 3 : la probabilité cherchée est \(1 - @M7_aucun_rejet:3m@ = @M7_exact:3m@\)."
            "\n\n"
            r"Lien avec la recherche quantitative : tester 20 signaux sans effet réel au seuil de 5 % produit au moins une fausse découverte près de deux fois sur trois. C'est le mécanisme du surapprentissage de backtest."
        ),
    },
    {
        "id": "DIAG-M8", "section": "math", "theme": "Optimisation sous contrainte", "difficulte": 3, "temps_cible_s": 120,
        "enonce": (
            r"Deux actifs ont des rendements d'écarts-types (volatilités) 20 % et 10 %, et de corrélation nulle. "
            r"Un portefeuille investit une fraction \(w\) de sa valeur dans le premier actif et \(1 - w\) dans le "
            r"second, \(w\) étant un réel quelconque."
            "\n\n"
            r"Quelle valeur de \(w\) minimise la variance du rendement du portefeuille ?"
        ),
        "options": [
            {"id": "a", "texte": "@M8_exact:1@", "explication": ""},
            {"id": "b", "texte": "@M8_d_inverse:1@", "explication": r"Les poids sont inversés : c'est l'actif le moins volatil, le second, qui doit recevoir le poids 0,8."},
            {"id": "c", "texte": "1/3", "explication": r"C'est une pondération proportionnelle à l'inverse de la volatilité, \((1/0{,}2)/(1/0{,}2 + 1/0{,}1)\). La variance minimale s'obtient avec l'inverse de la variance, pas de l'écart-type."},
            {"id": "d", "texte": "0,5", "explication": r"L'équipondération ne minimise la variance que si les deux actifs ont la même variance."},
        ],
        "solution": (
            r"Réponse : \(w = @M8_exact:1m@\)."
            "\n\n"
            r"Notations : \(\sigma_1 = 0{,}20\) et \(\sigma_2 = 0{,}10\) sont les écarts-types des rendements \(R_1\) et \(R_2\) des deux actifs ; le rendement du portefeuille est \(w R_1 + (1 - w) R_2\)."
            "\n\n"
            r"Étape 1 : la corrélation étant nulle, la covariance est nulle et la variance du portefeuille est \(f(w) = w^2 \sigma_1^2 + (1 - w)^2 \sigma_2^2\)."
            "\n\n"
            r"Étape 2 : \(f\) est un polynôme du second degré de coefficient dominant \(\sigma_1^2 + \sigma_2^2 > 0\), donc strictement convexe : son unique minimum est le point où la dérivée s'annule. \(f'(w) = 2 w \sigma_1^2 - 2 (1 - w) \sigma_2^2 = 0\) donne \(w = \sigma_2^2 / (\sigma_1^2 + \sigma_2^2) = @M8_var2:2m@ / @M8_somme_var:2m@ = @M8_exact:1m@\)."
            "\n\n"
            r"Lecture : les poids sont proportionnels aux inverses des variances, \(1/\sigma_1^2 = 25\) et \(1/\sigma_2^2 = 100\). L'écart-type du portefeuille optimal vaut @M8_vol_min_pct:2@ %, moins que celui de chaque actif."
        ),
    },
    # ------------------------------------------------------- ML FONDAMENTAL
    {
        "id": "DIAG-F1", "section": "mlf", "theme": "Régularisation L1 et L2", "difficulte": 4, "temps_cible_s": 120,
        "enonce": (
            r"Soient \(\mathbf{y} \in \mathbb{R}^n\) le vecteur des observations et \(\mathbf{X}\) une matrice \(n \times 2\) "
            r"dont les colonnes sont orthonormées (\(\mathbf{X}^\top \mathbf{X} = \mathbf{I}\), matrice identité \(2 \times 2\)). "
            r"Pour \(\boldsymbol{\beta} = (\beta_1, \beta_2) \in \mathbb{R}^2\), on note \(\lVert \cdot \rVert\) la norme euclidienne et "
            r"\(\lVert \boldsymbol{\beta} \rVert_1 = |\beta_1| + |\beta_2|\). Avec \(\lambda = 1\), on minimise sur \(\boldsymbol{\beta}\) le critère du Lasso "
            r"\(\tfrac12 \lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert^2 + \lambda \lVert \boldsymbol{\beta} \rVert_1\)."
            "\n\n"
            r"L'estimateur des moindres carrés ordinaires, \(\mathbf{X}^\top \mathbf{y}\), vaut \((3 \,;\, 0{,}5)\). Quel est le minimiseur du critère du Lasso ?"
        ),
        "options": [
            {"id": "a", "texte": "@F1_lasso@", "explication": ""},
            {"id": "b", "texte": "@F1_ridge@", "explication": r"C'est la solution Ridge, obtenue avec la pénalité \(\tfrac{\lambda}{2} \lVert \boldsymbol{\beta} \rVert^2\) : chaque coefficient est divisé par \(1 + \lambda\), aucun n'est annulé. La pénalité L1 agit par soustraction, pas par division."},
            {"id": "c", "texte": "@F1_d_seuillage_dur@", "explication": r"C'est un seuillage dur : on annule les petits coefficients sans toucher aux grands. Le Lasso rétrécit aussi les coefficients conservés, de \(\lambda\)."},
            {"id": "d", "texte": "@F1_d_sans_troncature@", "explication": r"On a soustrait \(\lambda\) aux deux coefficients sans s'arrêter à zéro. Quand \(|b_j| \le \lambda\), le coefficient est exactement nul : dépasser zéro augmenterait à la fois l'erreur et la pénalité."},
        ],
        "solution": (
            r"Réponse : @F1_lasso@."
            "\n\n"
            r"Notations : \(\mathbf{b} = \mathbf{X}^\top \mathbf{y} = (b_1, b_2)\) est l'estimateur des moindres carrés ordinaires."
            "\n\n"
            r"Étape 1, le critère se sépare coordonnée par coordonnée. On écrit \(\mathbf{y} - \mathbf{X}\boldsymbol{\beta} = (\mathbf{y} - \mathbf{X}\mathbf{b}) + \mathbf{X}(\mathbf{b} - \boldsymbol{\beta})\). Le premier terme est orthogonal aux colonnes de \(\mathbf{X}\), car \(\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{b}) = \mathbf{b} - \mathbf{b} = \mathbf{0}\). Par le théorème de Pythagore, puis \(\lVert \mathbf{X}\mathbf{v} \rVert^2 = \mathbf{v}^\top \mathbf{X}^\top \mathbf{X} \mathbf{v} = \lVert \mathbf{v} \rVert^2\) : \(\lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert^2 = \lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2 + \lVert \mathbf{b} - \boldsymbol{\beta} \rVert^2\). Le premier terme ne dépend pas de \(\boldsymbol{\beta}\). Il reste à minimiser, pour chaque \(j\), la fonction d'une variable \(h_j(\beta) = \tfrac12 (\beta - b_j)^2 + \lambda |\beta|\)."
            "\n\n"
            r"Étape 2, minimisation de \(h_j\), convexe, pour \(b_j \ge 0\). Sur \(\beta > 0\), \(h_j'(\beta) = \beta - b_j + \lambda\) s'annule en \(\beta = b_j - \lambda\), qui n'est dans la zone \(\beta > 0\) que si \(b_j > \lambda\). Sur \(\beta < 0\), \(h_j'(\beta) = \beta - b_j - \lambda < 0\) : la fonction décroît jusqu'en 0, donc aucun minimum n'y est atteint. Si \(b_j \le \lambda\), \(h_j' > 0\) sur \(\beta > 0\) : la fonction décroît avant 0 et croît après, le minimum est en \(\beta = 0\). Le cas \(b_j < 0\) est symétrique."
            "\n\n"
            r"Étape 3, formule du seuillage doux : \(\hat\beta_j = \mathrm{signe}(b_j)\,\max(|b_j| - \lambda, 0)\). Ici \(\hat\beta_1 = 3 - 1 = 2\) et \(\hat\beta_2 = 0\) car \(0{,}5 \le 1\)."
            "\n\n"
            r"Condition de validité : cette formule fermée suppose les colonnes orthonormées. Pour une matrice \(\mathbf{X}\) quelconque, le terme quadratique couple les coordonnées et le Lasso n'a pas de forme fermée générale."
        ),
    },
    {
        "id": "DIAG-F2", "section": "mlf", "theme": "Régression linéaire, R² et R² ajusté", "difficulte": 3, "temps_cible_s": 90,
        "enonce": (
            r"On ajuste par moindres carrés ordinaires, avec constante, une régression d'une variable sur \(p\) variables "
            r"explicatives, à partir de \(n\) observations. On ajoute ensuite une variable explicative et on réajuste sur les mêmes observations."
            "\n\n"
            r"Définitions : \(R^2 = 1 - \mathrm{SCR}/\mathrm{SCT}\), calculé sur les observations d'ajustement, où SCR est la somme des carrés des résidus et SCT la somme des carrés des écarts de la variable expliquée à sa moyenne ; "
            r"\(R^2\) ajusté \(= 1 - (1 - R^2)\,(n - 1)/(n - k)\), où \(k\) est le nombre de coefficients estimés, constante comprise."
            "\n\n"
            r"Quelle affirmation est vraie quelle que soit la variable ajoutée ?"
        ),
        "options": [
            {"id": "a", "texte": r"Le \(R^2\) ne peut pas diminuer ; le \(R^2\) ajusté peut diminuer.", "explication": ""},
            {"id": "b", "texte": r"Ni le \(R^2\) ni le \(R^2\) ajusté ne peuvent diminuer.", "explication": r"Le \(R^2\) ajusté pénalise chaque coefficient supplémentaire par le facteur \((n-1)/(n-k)\). Dans l'expérience de vérification, l'ajout d'une variable de pur bruit le fait baisser @F2_r2_ajuste_baisse@ fois sur @F2_essais@."},
            {"id": "c", "texte": r"Le \(R^2\) augmente strictement.", "explication": r"Il peut rester inchangé : par exemple si la nouvelle variable est une combinaison linéaire des précédentes, l'espace des prédictions possibles est le même et SCR ne change pas."},
            {"id": "d", "texte": r"Le \(R^2\) calculé sur un échantillon de test indépendant ne peut pas diminuer.", "explication": r"La garantie ne vaut que sur l'échantillon d'ajustement, là où les coefficients minimisent SCR. Hors échantillon, une variable de bruit dégrade la prédiction : c'est le surapprentissage."},
        ],
        "solution": (
            r"Réponse : le \(R^2\) ne peut pas diminuer, le \(R^2\) ajusté peut diminuer."
            "\n\n"
            r"Étape 1, le \(R^2\). SCR est le minimum de la somme des carrés des résidus sur tous les vecteurs de coefficients. Le modèle augmenté contient l'ancien : il suffit de donner le coefficient 0 à la nouvelle variable. Minimiser sur un ensemble plus grand donne un minimum inférieur ou égal, donc SCR ne peut pas augmenter. SCT ne dépend pas du modèle. Donc \(R^2 = 1 - \mathrm{SCR}/\mathrm{SCT}\) ne peut pas diminuer."
            "\n\n"
            r"Étape 2, le \(R^2\) ajusté. Il s'écrit \(1 - \dfrac{\mathrm{SCR}/(n-k)}{\mathrm{SCT}/(n-1)}\). En passant de \(k\) à \(k+1\) coefficients, il diminue exactement quand \(\mathrm{SCR}_{\text{nouveau}}/(n-k-1) > \mathrm{SCR}_{\text{ancien}}/(n-k)\), c'est-à-dire quand la baisse relative de SCR est inférieure à \(1/(n-k)\)."
            "\n\n"
            r"Expérience (script de vérification) : \(n = 40\), une variable utile, puis ajout d'une variable de bruit pur, @F2_essais@ répétitions. Le \(R^2\) a baissé @F2_r2_baisse@ fois, le \(R^2\) ajusté @F2_r2_ajuste_baisse@ fois."
        ),
    },
    {
        "id": "DIAG-F3", "section": "mlf", "theme": "Métriques de classification", "difficulte": 2, "temps_cible_s": 75,
        "enonce": (
            r"Un classifieur binaire évalué sur 1000 exemples donne 30 vrais positifs, 20 faux positifs, "
            r"10 faux négatifs et 940 vrais négatifs. Quel est le score F1 de la classe positive, arrondi à trois décimales ?"
        ),
        "options": [
            {"id": "a", "texte": "@F3_f1:3@", "explication": ""},
            {"id": "b", "texte": "@F3_moyenne_arith:3@", "explication": r"C'est la moyenne arithmétique de la précision et du rappel. Le score F1 est leur moyenne harmonique, toujours inférieure ou égale, qui pénalise un déséquilibre entre les deux."},
            {"id": "c", "texte": "@F3_exactitude:3@", "explication": r"C'est l'exactitude (accuracy), la proportion de prédictions correctes. Elle est dominée par les 940 vrais négatifs, que le score F1 ignore."},
            {"id": "d", "texte": "@F3_precision:3@", "explication": r"C'est la précision seule, \(30/(30+20)\). Le score F1 la combine avec le rappel."},
        ],
        "solution": (
            r"Réponse : @F3_f1:3@, soit exactement 2/3."
            "\n\n"
            r"Notations : VP, FP, FN sont les nombres de vrais positifs, faux positifs et faux négatifs."
            "\n\n"
            r"Étape 1 : la précision est la proportion de vrais positifs parmi les exemples prédits positifs, \(\mathrm{VP}/(\mathrm{VP} + \mathrm{FP}) = 30/50 = @F3_precision:2m@\)."
            "\n\n"
            r"Étape 2 : le rappel est la proportion de vrais positifs parmi les exemples réellement positifs, \(\mathrm{VP}/(\mathrm{VP} + \mathrm{FN}) = 30/40 = @F3_rappel:2m@\)."
            "\n\n"
            r"Étape 3 : le score F1 est la moyenne harmonique des deux, \(2 \times @F3_precision:2m@ \times @F3_rappel:2m@ / (@F3_precision:2m@ + @F3_rappel:2m@) \approx @F3_f1:3m@\). Forme directe équivalente : \(2\,\mathrm{VP}/(2\,\mathrm{VP} + \mathrm{FP} + \mathrm{FN}) = 60/90\)."
        ),
    },
    {
        "id": "DIAG-F4", "section": "mlf", "theme": "Bagging, forêts aléatoires", "difficulte": 4, "temps_cible_s": 120,
        "enonce": (
            r"On dispose de 10 prédicteurs \(T_1, \dots, T_{10}\), variables aléatoires réelles représentant les prédictions "
            r"de 10 arbres en un même point. Chacun a pour variance 1 et la corrélation entre deux prédicteurs distincts vaut 0,3."
            "\n\n"
            r"Quelle est la variance de leur moyenne \((T_1 + \dots + T_{10})/10\) ?"
        ),
        "options": [
            {"id": "a", "texte": "@F4_exact:2@", "explication": ""},
            {"id": "b", "texte": "@F4_d_sigma2_sur_B:2@", "explication": r"C'est \(\sigma^2/B\), valable seulement pour des prédicteurs non corrélés. Ici les covariances entre arbres ajoutent un terme qui ne diminue pas avec le nombre d'arbres."},
            {"id": "c", "texte": "@F4_d_rho_sigma2:2@", "explication": r"C'est \(\rho\,\sigma^2\), la limite quand le nombre d'arbres tend vers l'infini. Avec 10 arbres, il reste le terme \((1 - \rho)\,\sigma^2/B\)."},
            {"id": "d", "texte": "@F4_d_rho_sigma2_sur_B:2@", "explication": r"C'est \(\rho\,\sigma^2/B\) : la corrélation a été placée dans le terme qui décroît avec \(B\), alors qu'elle gouverne le terme qui ne décroît pas."},
        ],
        "solution": (
            r"Réponse : \(0{,}3 + 0{,}7/10 = @F4_exact:2m@\)."
            "\n\n"
            r"Notations : \(B = 10\) est le nombre de prédicteurs, \(\sigma^2 = 1\) leur variance commune, \(\rho = 0{,}3\) leur corrélation deux à deux, donc \(\mathrm{Cov}(T_i, T_j) = \rho\,\sigma^2\) pour \(i \ne j\)."
            "\n\n"
            r"Étape 1 : la variance d'une somme est la somme de toutes les covariances, \(\mathrm{Var}\big(\sum_i T_i\big) = \sum_i \mathrm{Var}(T_i) + \sum_{i \ne j} \mathrm{Cov}(T_i, T_j) = B\,\sigma^2 + B(B-1)\,\rho\,\sigma^2\), car il y a \(B(B-1)\) couples ordonnés d'indices distincts."
            "\n\n"
            r"Étape 2 : diviser la somme par \(B\) divise la variance par \(B^2\) : \(\mathrm{Var}(\text{moyenne}) = \sigma^2/B + (B-1)\,\rho\,\sigma^2/B = \rho\,\sigma^2 + (1 - \rho)\,\sigma^2/B\)."
            "\n\n"
            r"Étape 3 : application numérique, \(0{,}3 + 0{,}7/10 = @F4_exact:2m@\)."
            "\n\n"
            r"Conséquence pour les forêts aléatoires : ajouter des arbres ne réduit que le second terme ; le plancher \(\rho\,\sigma^2\) ne baisse qu'en décorrélant les arbres, ce que font le rééchantillonnage des observations et le tirage d'un sous-ensemble de variables à chaque coupure."
        ),
    },
    # ------------------------------------------------------------ ML AVANCÉ
    {
        "id": "DIAG-A1", "section": "mla", "theme": "XGBoost, poids optimal d'une feuille", "difficulte": 4, "temps_cible_s": 120,
        "enonce": (
            r"Notations. Pour une observation \(i\) : \(y_i\) est la valeur observée (un réel), \(\hat y_i\) la prédiction "
            r"courante du modèle, c'est-à-dire la somme des arbres déjà construits, et \(\ell(y_i, \hat y) = \tfrac12 (y_i - \hat y)^2\) "
            r"la perte, vue comme une fonction de la prédiction \(\hat y\). On note \(g_i\) et \(h_i\) ses dérivées première et "
            r"seconde par rapport à la prédiction, évaluées en la prédiction courante : "
            r"\(g_i = \frac{\partial \ell}{\partial \hat y}(y_i, \hat y_i)\) et \(h_i = \frac{\partial^2 \ell}{\partial \hat y^2}(y_i, \hat y_i)\)."
            "\n\n"
            r"Un nouvel arbre ajoute à la prédiction de chaque observation d'une même feuille une même constante \(w\) (le poids "
            r"de la feuille). XGBoost choisit \(w\) en minimisant, sur les observations \(i\) de la feuille, le développement de "
            r"Taylor à l'ordre 2 de la perte \(\ell(y_i, \hat y_i + w)\) autour de \(\hat y_i\), plus une pénalité, soit "
            r"\(\sum_i \big[g_i\,w + \tfrac12 h_i\,w^2\big] + \tfrac12 \lambda\,w^2\), où \(\lambda \ge 0\) est le coefficient "
            r"de régularisation. Le terme d'ordre 0, \(\sum_i \ell(y_i, \hat y_i)\), est omis car il ne dépend pas de \(w\)."
            "\n\n"
            r"On prend \(\lambda = 1\). La feuille contient 3 observations dont les résidus \(y_i - \hat y_i\) valent 1, 2 et 3. "
            r"Que vaut le \(w\) optimal ?"
        ),
        "options": [
            {"id": "a", "texte": "@A1_exact:1@", "explication": ""},
            {"id": "b", "texte": "@A1_d_moyenne:0@", "explication": r"C'est la moyenne des résidus, la réponse sans régularisation (\(\lambda = 0\)). Avec \(\lambda = 1\), le dénominateur passe de 3 à 4 et le poids est rétréci vers zéro."},
            {"id": "c", "texte": "@A1_d_signe:1@", "explication": r"Erreur de signe : le gradient est \(g_i = \hat y_i - y_i\), l'opposé du résidu, et le poids optimal est \(-G/(H + \lambda)\). Des résidus positifs demandent d'augmenter la prédiction."},
            {"id": "d", "texte": "@A1_d_somme:0@", "explication": r"C'est la somme des résidus, \(-G\), sans la division par \(H + \lambda\)."},
        ],
        "solution": (
            r"Réponse : \(w = 6/4 = @A1_exact:1m@\)."
            "\n\n"
            r"Notations : \(G = \sum_i g_i\) et \(H = \sum_i h_i\), sommes sur les observations de la feuille."
            "\n\n"
            r"Étape 0, d'où vient le critère. Ajouter \(w\) à la prédiction change la perte de l'observation \(i\) en "
            r"\(\ell(y_i, \hat y_i + w)\). La formule de Taylor à l'ordre 2 en \(w\), autour de \(\hat y_i\), donne "
            r"\(\ell(y_i, \hat y_i + w) = \ell(y_i, \hat y_i) + g_i\,w + \tfrac12 h_i\,w^2 + o(w^2)\). Pour la perte quadratique, "
            r"le reste est nul : la dérivée troisième est nulle, donc le développement est exact. Pour une autre perte (logistique "
            r"par exemple), c'est une approximation, et XGBoost minimise l'approximation."
            "\n\n"
            r"Étape 1, forme générale. Le critère s'écrit \(G\,w + \tfrac12 (H + \lambda)\,w^2\), polynôme du second degré en \(w\), strictement convexe dès que \(H + \lambda > 0\). Sa dérivée \(G + (H + \lambda)\,w\) s'annule en \(w^\star = -G/(H + \lambda)\)."
            "\n\n"
            r"Étape 2, dérivées de la perte quadratique. \(\frac{\partial}{\partial \hat y} \tfrac12 (y_i - \hat y)^2 = -(y_i - \hat y) = \hat y - y_i\), donc \(g_i = \hat y_i - y_i\), l'opposé du résidu ; en dérivant une seconde fois, \(h_i = 1\)."
            "\n\n"
            r"Étape 3, application : \(G = -(1 + 2 + 3) = @A1_G:0m@\), \(H = @A1_H:0m@\), donc \(w^\star = 6/(3 + 1) = @A1_exact:1m@\)."
            "\n\n"
            r"Remarque : pour la perte quadratique, le développement au second ordre est exact, et \(w^\star\) est la moyenne des résidus rétrécie par le facteur \(H/(H + \lambda) = 3/4\)."
        ),
    },
    {
        "id": "DIAG-A2", "section": "mla", "theme": "Convolutions : padding, stride, dilatation", "difficulte": 3, "temps_cible_s": 90,
        "enonce": (
            r"Une couche de convolution à une dimension reçoit une séquence de longueur 100. Son noyau a 5 coefficients, "
            r"avec une dilatation de 2 (les 5 coefficients s'appliquent à des éléments espacés de 2 positions), un pas "
            r"(stride) de 2 (le noyau avance de 2 positions entre deux sorties) et une marge (padding) de 2 zéros ajoutés "
            r"de chaque côté de la séquence. Une sortie est produite pour chaque position où le noyau dilaté tient "
            r"entièrement dans la séquence étendue, en partant de son début."
            "\n\n"
            r"Quelle est la longueur de la séquence de sortie ?"
        ),
        "options": [
            {"id": "a", "texte": "@A2_exact@", "explication": ""},
            {"id": "b", "texte": "@A2_d_sans_dilatation@", "explication": r"C'est le résultat sans dilatation : un noyau de 5 coefficients contigus couvre 5 positions. Avec une dilatation de 2, il en couvre \(2 \times 4 + 1 = 9\)."},
            {"id": "c", "texte": "@A2_d_sans_pas@", "explication": r"C'est le résultat avec un pas de 1. Un pas de 2 ne garde qu'une position de départ sur deux."},
            {"id": "d", "texte": "@A2_d_sans_plus_un@", "explication": r"Il manque le \(+1\) : les positions de départ vont de 0 à @A2_dernier_depart_pair@ par pas de 2, ce qui fait \(@A2_dernier_depart_pair@/2 + 1\) positions, la position 0 comprise."},
        ],
        "solution": (
            r"Réponse : @A2_exact@."
            "\n\n"
            r"Notations : \(n = 100\) longueur d'entrée, \(k = 5\) taille du noyau, \(d = 2\) dilatation, \(s = 2\) pas, \(m = 2\) marge de chaque côté."
            "\n\n"
            r"Étape 1 : la séquence étendue a pour longueur \(n + 2m = @A2_longueur_etendue@\)."
            "\n\n"
            r"Étape 2 : entre le premier et le dernier coefficient du noyau dilaté il y a \(d\,(k-1)\) positions d'écart, donc le noyau couvre \(d\,(k-1) + 1 = @A2_portee@\) positions."
            "\n\n"
            r"Étape 3 : le noyau tient entièrement si sa position de départ, comptée à partir de 0, est au plus \(@A2_longueur_etendue@ - @A2_portee@ = @A2_dernier_debut@\). Les départs possibles sont les multiples de \(s = 2\) entre 0 et @A2_dernier_debut@, soit \(\lfloor @A2_dernier_debut@/2 \rfloor + 1 = @A2_exact@\) positions."
            "\n\n"
            r"Formule générale : \(\lfloor (n + 2m - d\,(k-1) - 1)/s \rfloor + 1\). Vérifiée par décompte direct et par PyTorch dans le script."
        ),
    },
    {
        "id": "DIAG-A3", "section": "mla", "theme": "Rétropropagation", "difficulte": 3, "temps_cible_s": 120,
        "enonce": (
            r"Un réseau à une entrée réelle \(x\) calcule \(a = \tanh(w_1 x)\), puis \(\hat y = w_2\,a\), où \(w_1\) et "
            r"\(w_2\) sont deux poids réels. La perte est \(\ell = \tfrac12 (\hat y - y)^2\), où \(y\) est la cible."
            "\n\n"
            r"Pour \(x = 2\), \(y = 1\), \(w_1 = 0\) et \(w_2 = 3\), que vaut \(\partial \ell / \partial w_1\) ?"
        ),
        "options": [
            {"id": "a", "texte": "@A3_dw1@", "explication": ""},
            {"id": "b", "texte": "0", "explication": r"C'est \(\partial \ell/\partial w_2 = (\hat y - y)\,a\), nul car \(a = \tanh(0) = 0\). Pour \(w_1\), la dérivée de tanh en 0 vaut 1, pas 0 : le gradient passe."},
            {"id": "c", "texte": "@A3_d_sans_x@", "explication": r"Il manque le facteur \(x = 2\), la dérivée de \(w_1 x\) par rapport à \(w_1\)."},
            {"id": "d", "texte": "@A3_d_signe@", "explication": r"Erreur de signe : la dérivée de \(\tfrac12 (\hat y - y)^2\) par rapport à \(\hat y\) est \(\hat y - y = -1\), pas \(y - \hat y\)."},
        ],
        "solution": (
            r"Réponse : @A3_dw1@."
            "\n\n"
            r"Étape 1, passe avant : \(w_1 x = 0\), \(a = \tanh(0) = 0\), \(\hat y = 3 \times 0 = 0\), \(\ell = \tfrac12 (0 - 1)^2\)."
            "\n\n"
            r"Étape 2, règle de la chaîne, de la sortie vers l'entrée : \(\dfrac{\partial \ell}{\partial w_1} = \dfrac{\partial \ell}{\partial \hat y} \cdot \dfrac{\partial \hat y}{\partial a} \cdot \dfrac{\partial a}{\partial (w_1 x)} \cdot \dfrac{\partial (w_1 x)}{\partial w_1} = (\hat y - y) \cdot w_2 \cdot \big(1 - \tanh^2(w_1 x)\big) \cdot x\), en utilisant \(\tanh' = 1 - \tanh^2\)."
            "\n\n"
            r"Étape 3, valeurs : \((0 - 1) \times 3 \times (1 - 0) \times 2 = @A3_dw1@\)."
            "\n\n"
            r"Par comparaison, \(\partial \ell/\partial w_2 = (\hat y - y)\,a = @A3_dw2@\) : avec \(w_1 = 0\), l'activation est nulle et \(w_2\) ne reçoit aucun gradient à ce pas."
        ),
    },
    {
        "id": "DIAG-A4", "section": "mla", "theme": "Optimisation : Adam", "difficulte": 4, "temps_cible_s": 120,
        "enonce": (
            r"L'algorithme Adam met à jour un paramètre réel \(\theta\) ainsi, à partir de \(m_0 = v_0 = 0\), où \(g_t\) est le gradient au pas \(t\) :"
            "\n\n"
            r"\(m_t = \beta_1 m_{t-1} + (1 - \beta_1)\,g_t\), \(\quad v_t = \beta_2 v_{t-1} + (1 - \beta_2)\,g_t^2\), \(\quad \hat m_t = m_t/(1 - \beta_1^t)\), \(\quad \hat v_t = v_t/(1 - \beta_2^t)\), \(\quad \theta_t = \theta_{t-1} - \eta\,\hat m_t/(\sqrt{\hat v_t} + \epsilon)\)."
            "\n\n"
            r"Avec le pas d'apprentissage \(\eta = 0{,}01\), \(\beta_1 = 0{,}9\), \(\beta_2 = 0{,}999\), \(\epsilon = 10^{-8}\) et un premier gradient \(g_1 = 100\), que vaut \(\theta_1 - \theta_0\), arrondi à quatre décimales ?"
        ),
        "options": [
            {"id": "a", "texte": "@A4_exact:4@", "explication": ""},
            {"id": "b", "texte": "@A4_d_sgd:4@", "explication": r"C'est le pas de la descente de gradient simple, \(-\eta\,g_1\). Adam divise par \(\sqrt{\hat v_1} = |g_1|\), ce qui efface l'échelle du gradient."},
            {"id": "c", "texte": "@A4_d_sans_correction:4@", "explication": r"C'est le résultat sans les corrections de biais : \(m_1 = 10\) et \(v_1 = 10\), d'où \(10/\sqrt{10}\). Les divisions par \(1 - \beta_1\) et \(1 - \beta_2\) compensent l'initialisation à zéro."},
            {"id": "d", "texte": "@A4_d_m_sans_correction:4@", "explication": r"La correction de biais a été appliquée à \(v_1\) mais pas à \(m_1\) : on a utilisé \(m_1 = 10\) au lieu de \(\hat m_1 = 100\)."},
        ],
        "solution": (
            r"Réponse : @A4_exact:4@, c'est-à-dire \(-\eta\) à \(10^{-10}\) près."
            "\n\n"
            r"Étape 1 : \(m_1 = (1 - \beta_1)\,g_1\) et \(v_1 = (1 - \beta_2)\,g_1^2\), puisque \(m_0 = v_0 = 0\)."
            "\n\n"
            r"Étape 2, corrections de biais au pas \(t = 1\) : \(\hat m_1 = m_1/(1 - \beta_1) = g_1\) et \(\hat v_1 = v_1/(1 - \beta_2) = g_1^2\)."
            "\n\n"
            r"Étape 3 : \(\theta_1 - \theta_0 = -\eta\,g_1/(|g_1| + \epsilon) = -0{,}01 \times 100/(100 + 10^{-8}) \approx @A4_exact:4m@\)."
            "\n\n"
            r"À retenir : au premier pas, Adam se déplace de \(\eta\) dans le sens opposé au gradient, quelle que soit l'amplitude de celui-ci (tant que \(|g_1|\) est grand devant \(\epsilon\)). Vérifié avec l'implémentation de PyTorch dans le script."
        ),
    },
    {
        "id": "DIAG-A5", "section": "mla", "theme": "Ratio de Sharpe et significativité", "difficulte": 4, "temps_cible_s": 120,
        "enonce": (
            r"On observe \(n = 1008\) rendements quotidiens d'une stratégie, soit 4 années de 252 jours. On note \(m\) "
            r"leur moyenne empirique et \(s\) leur écart-type empirique. Le ratio de Sharpe annualisé empirique, défini ici "
            r"par \(\sqrt{252}\;m/s\) (taux sans risque nul), vaut 1,0."
            "\n\n"
            r"Que vaut la statistique de Student \(t = m/(s/\sqrt{n})\) du test de l'hypothèse nulle « l'espérance des rendements est nulle » ?"
        ),
        "options": [
            {"id": "a", "texte": "@A5_exact:1@", "explication": ""},
            {"id": "b", "texte": "1,0", "explication": r"Le ratio de Sharpe annualisé n'est pas la statistique de Student : il ne dépend pas de la longueur de l'historique, alors que \(t\) croît comme la racine du nombre d'années."},
            {"id": "c", "texte": "@A5_d_annees:1@", "explication": r"Le ratio de Sharpe annualisé a été multiplié par le nombre d'années. Le bon facteur est sa racine carrée, \(\sqrt{4} = 2\)."},
            {"id": "d", "texte": "@A5_d_racine_n:1@", "explication": r"C'est \(\sqrt{1008} \times 1{,}0\) : le ratio annualisé a été multiplié par \(\sqrt{n}\) sans être d'abord ramené à l'échelle quotidienne par la division par \(\sqrt{252}\)."},
        ],
        "solution": (
            r"Réponse : \(t = @A5_exact:1m@\)."
            "\n\n"
            r"Étape 1 : \(t = (m/s)\,\sqrt{n}\) par définition."
            "\n\n"
            r"Étape 2 : le ratio de Sharpe quotidien est \(m/s = 1{,}0/\sqrt{252}\), d'après la définition du ratio annualisé."
            "\n\n"
            r"Étape 3 : \(t = \sqrt{n/252} \times 1{,}0 = \sqrt{4} = @A5_exact:1m@\). En général, \(t\) est égal au ratio de Sharpe annualisé multiplié par la racine carrée du nombre d'années."
            "\n\n"
            r"Interprétation, sous hypothèse : si les rendements sont indépendants et de même loi normale, \(t\) suit sous l'hypothèse nulle une loi de Student à \(n - 1\) degrés de liberté, et la p-value bilatérale vaut ici @A5_p_bilaterale:3@. Avec des rendements autocorrélés, ou une stratégie choisie parmi plusieurs essais, cette p-value n'est plus valable."
        ),
    },
    {
        "id": "DIAG-A6", "section": "mla", "theme": "Covariance en grande dimension", "difficulte": 4, "temps_cible_s": 120,
        "enonce": (
            r"On observe les rendements de 500 actifs à 250 dates. Pour \(t = 1, \dots, 250\), \(\mathbf{x}_t \in \mathbb{R}^{500}\) est le "
            r"vecteur des rendements observés à la date \(t\), et \(\bar{\mathbf{x}}\) la moyenne des 250 vecteurs. On estime la matrice "
            r"de covariance par l'estimateur empirique \(\mathbf{S} = \dfrac{1}{249} \sum_{t=1}^{250} (\mathbf{x}_t - \bar{\mathbf{x}})(\mathbf{x}_t - \bar{\mathbf{x}})^\top\), "
            r"matrice \(500 \times 500\)."
            "\n\n"
            r"Les 250 vecteurs sont des tirages indépendants d'une loi à densité sur \(\mathbb{R}^{500}\). Quel est, avec probabilité 1, le rang de \(\mathbf{S}\) ?"
        ),
        "options": [
            {"id": "a", "texte": "@A6_rang@", "explication": ""},
            {"id": "b", "texte": "250", "explication": r"Le centrage fait perdre une dimension : les 250 vecteurs centrés ont une somme nulle, ils sont donc liés et engendrent un espace de dimension au plus 249."},
            {"id": "c", "texte": "500", "explication": r"Le rang plein exigerait au moins 501 dates. Avec 250 dates, \(\mathbf{S}\) est une somme de 250 matrices de rang 1 et ne peut pas atteindre le rang 500."},
            {"id": "d", "texte": "499", "explication": r"Les rôles du nombre d'actifs et du nombre de dates ont été échangés : c'est le nombre de dates, moins un, qui borne le rang."},
        ],
        "solution": (
            r"Réponse : @A6_rang@."
            "\n\n"
            r"Notations : \(\mathbf{Y}\) est la matrice \(250 \times 500\) dont la ligne \(t\) est \((\mathbf{x}_t - \bar{\mathbf{x}})^\top\). Alors \(\mathbf{S} = \mathbf{Y}^\top \mathbf{Y}/249\)."
            "\n\n"
            r"Étape 1 : \(\mathbf{S}\) et \(\mathbf{Y}\) ont le même rang. En effet, si \(\mathbf{Y}^\top \mathbf{Y}\mathbf{v} = \mathbf{0}\), alors \(\mathbf{v}^\top \mathbf{Y}^\top \mathbf{Y}\mathbf{v} = \lVert \mathbf{Y}\mathbf{v} \rVert^2 = 0\), donc \(\mathbf{Y}\mathbf{v} = \mathbf{0}\) ; la réciproque est immédiate. Les deux matrices ont le même noyau dans \(\mathbb{R}^{500}\), donc le même rang par le théorème du rang."
            "\n\n"
            r"Étape 2 : la somme des lignes de \(\mathbf{Y}\) est \(\sum_t (\mathbf{x}_t - \bar{\mathbf{x}}) = \mathbf{0}\) par définition de la moyenne. Les 250 lignes sont liées, donc le rang de \(\mathbf{Y}\) est au plus 249."
            "\n\n"
            r"Étape 3, le rang vaut exactement @A6_rang@ avec probabilité 1. Notations : \(N = 500\), \(T = 250\), \(\mathbf{d}_t = \mathbf{x}_t - \mathbf{x}_T\) pour \(t = 1, \dots, T - 1\). Comme \(T - 1 = 249 \le N\), il suffit de montrer, en quatre points, que l'espace engendré par les lignes de \(\mathbf{Y}\) est celui des \(\mathbf{d}_t\), et que les \(\mathbf{d}_t\) sont presque sûrement linéairement indépendants."
            "\n\n"
            r"3a. Même espace engendré. Chaque \(\mathbf{x}_t - \bar{\mathbf{x}} = \tfrac{1}{T} \sum_{s} (\mathbf{x}_t - \mathbf{x}_s)\), et chaque \(\mathbf{x}_t - \mathbf{x}_s = \mathbf{d}_t - \mathbf{d}_s\) (avec \(\mathbf{d}_T = \mathbf{0}\)) : les lignes de \(\mathbf{Y}\) sont dans l'espace engendré par les \(\mathbf{d}_t\). Réciproquement, \(\mathbf{d}_t = (\mathbf{x}_t - \bar{\mathbf{x}}) - (\mathbf{x}_T - \bar{\mathbf{x}})\). Le rang de \(\mathbf{Y}\) est donc la dimension de l'espace engendré par \(\mathbf{d}_1, \dots, \mathbf{d}_{T-1}\)."
            "\n\n"
            r"3b. Un sous-espace strict \(V\) de \(\mathbb{R}^N\) est de mesure de Lebesgue nulle. Il est contenu dans un hyperplan \(\{\mathbf{z} : \mathbf{a}^\top \mathbf{z} = 0\}\) avec \(\mathbf{a}\) unitaire. Une rotation, qui conserve la mesure de Lebesgue (valeur absolue du déterminant égale à 1), envoie \(\mathbf{a}\) sur le premier vecteur de base, donc cet hyperplan sur \(\{0\} \times \mathbb{R}^{N-1}\), produit d'un point par \(\mathbb{R}^{N-1}\), de mesure \(0 \times \infty = 0\) par le théorème de Fubini. Conséquence : si \(\mathbf{Z}\) a une densité \(f\), \(\mathbb{P}(\mathbf{Z} \in V) = \int_V f = 0\)."
            "\n\n"
            r"3c. Lemme : si \(k \le N\) vecteurs aléatoires \(\mathbf{Z}_1, \dots, \mathbf{Z}_k\) sont indépendants et ont chacun une densité, ils sont linéairement indépendants avec probabilité 1. Par récurrence sur \(j\) : sachant \(\mathbf{Z}_1, \dots, \mathbf{Z}_{j-1}\), leur espace engendré \(V_{j-1}\) est de dimension au plus \(j - 1 < N\), donc un sous-espace strict ; \(\mathbf{Z}_j\) étant indépendant des précédents, \(\mathbb{P}(\mathbf{Z}_j \in V_{j-1} \mid \mathbf{Z}_1, \dots, \mathbf{Z}_{j-1}) = 0\) par 3b, et en prenant l'espérance, \(\mathbb{P}(\mathbf{Z}_j \in V_{j-1}) = 0\). Hors d'un événement de probabilité nulle, chaque \(\mathbf{Z}_j\) sort de l'espace engendré par les précédents, et la famille est libre."
            "\n\n"
            r"3d. Les \(\mathbf{d}_t\) ne sont pas indépendants, car ils partagent \(\mathbf{x}_T\). On conditionne donc par \(\mathbf{x}_T = \mathbf{c}\) : les \(\mathbf{d}_t = \mathbf{x}_t - \mathbf{c}\), \(t \le T - 1\), sont alors indépendants (les \(\mathbf{x}_t\) sont indépendants de \(\mathbf{x}_T\) et entre eux), et chacun a la densité translatée \(\mathbf{z} \mapsto f(\mathbf{z} + \mathbf{c})\). Par le lemme avec \(k = T - 1 = 249 \le 500\), ils sont libres avec probabilité conditionnelle 1, pour tout \(\mathbf{c}\) ; en intégrant sur la loi de \(\mathbf{x}_T\), avec probabilité 1. Avec 3a, le rang de \(\mathbf{Y}\), donc de \(\mathbf{S}\), vaut exactement @A6_rang@."
            "\n\n"
            r"Vérification numérique (script) : 250 tirages gaussiens dans \(\mathbb{R}^{500}\) donnent une matrice de rang @A6_rang@."
            "\n\n"
            r"Conséquence : \(\mathbf{S}\) a @A6_vp_nulles@ valeurs propres nulles et n'est pas inversible. Toute méthode qui utilise \(\mathbf{S}^{-1}\), comme le portefeuille de variance minimale, est inapplicable telle quelle : c'est ce que corrige le rétrécissement de type Ledoit-Wolf."
        ),
    },
]
