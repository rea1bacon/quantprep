"""Séance d'algèbre linéaire 1 : leçon (spectre, ACP, covariance), puis 12 questions à choix multiple.

Les questions comprennent des équivalents, jamais identiques, des deux questions d'algèbre linéaire du test de
positionnement : DIAG-M4 (équicorrélation 3 x 3 semi-définie positive) et DIAG-A6 (rang d'une covariance empirique
avec T < N). Elles couvrent toute la leçon L-ALGEBRE-1 (outils/lecons_algebre.py), dont l'ACP de rendements
standardisés.
Toutes les valeurs : verif/seance_algebre_1.py. Dans chaque question, l'option "a" est la bonne.
"""
import lecons_algebre
import questions_diagnostic as diag

ID = "seance-algebre-1"
TITRE = "Séance d'algèbre linéaire : spectre, ACP et covariance"
VERIF = "verif/seance_algebre_1.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [{"id": "lecon", "nom": "Leçon", "budget_min": 55}, {"id": "math", "nom": "Mathématiques", "budget_min": 26}]
PRESENTATION = (
    "D'abord une leçon d'environ 55 minutes : théorème spectral, semi-définie positivité, projections, mineurs "
    "principaux, SVD, conditionnement, gradients, covariance empirique en grande dimension, ACP et rétrécissement de "
    "Ledoit-Wolf, avec sept points de contrôle corrigés immédiatement et des exemples exécutés dans l'environnement "
    "des notebooks de la plateforme. Ensuite 12 questions à choix multiple couvrant toute la leçon, dont des équivalents "
    "des deux questions d'algèbre du test de positionnement (équicorrélation, rang d'une covariance empirique).\n\n"
    "Objectif de vitesse : chaque question se résout dans son temps cible avec les outils de la leçon. Commence par "
    "reconnaître l'outil (tableau « Réflexes en 30 secondes »), puis calcule. Au-delà du double du temps cible "
    "affiché, choisis « Je ne sais pas » et passe à la suivante : c'est la règle d'abandon à suivre dans un test chronométré."
)
AVERTISSEMENTS = [
    "Les questions de mathématiques sont présentées ici en choix multiple à une seule bonne réponse. Cherche la réponse avant de "
    "lire les options, pour t'entraîner aussi au format à réponse numérique.",
    "Ici le retour en arrière est permis et il n'y a pas de pénalité : ce sont les règles de cette plateforme, pas "
    "celles de tout test en ligne.",
]

LECONS = [lecons_algebre.LECON_ALGEBRE]
CODES = []

QUESTIONS = [
    # ------------------------------------------------- Semi-définie positivité
    {
        "id": "AL-01", "section": "math", "theme": "Équicorrélation, semi-définie positivité", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""Soit \(\rho\) un réel et \(\mathbf{C}\) la matrice \(5 \times 5\) dont les coefficients diagonaux valent 1 et tous les autres valent \(\rho\). Une matrice de diagonale 1 est la matrice de corrélation d'un vecteur aléatoire si et seulement si elle est semi-définie positive (\(\mathbf{x}^\top \mathbf{C}\,\mathbf{x} \ge 0\) pour tout \(\mathbf{x} \in \mathbb{R}^5\)).

Quelle est la plus petite valeur de \(\rho\) pour laquelle \(\mathbf{C}\) est semi-définie positive ?""",
        "options": [
            {"id": "a", "texte": r"\(@Q01_borne_num@/@Q01_borne_den@\)", "explication": r"Oui : la valeur propre simple \(1 + 4\rho\) s'annule en \(\rho = @Q01_borne_num@/@Q01_borne_den@\), et l'autre, \(1 - \rho\), est alors positive."},
            {"id": "b", "texte": r"\(@Q01_d_n_num@/@Q01_d_n_den@\)", "explication": r"C'est \(-1/n\) au lieu de \(-1/(n-1)\). Pour \(\rho = @Q01_borne_num@/@Q01_borne_den@\), plus petit, la matrice est encore semi-définie positive (valeur propre \(1 + 4\rho = 0\)) : \(@Q01_d_n_num@/@Q01_d_n_den@\) n'est donc pas la plus petite valeur. Contrôle avec \(n = 2\) : la formule \(-1/n\) donnerait \(-1/2\), alors que \(\rho = -1\) est possible pour deux variables."},
            {"id": "c", "texte": r"\(@Q01_d_3_num@/@Q01_d_3_den@\)", "explication": r"C'est la borne pour trois variables (\(n = 3\), question DIAG-M4 du test de positionnement). Pour \(n = 5\), en \(\rho = -1/2\), la valeur propre \(1 + 4\rho\) vaut \(@Q01_vp_moins_demi@ < 0\)."},
            {"id": "d", "texte": r"\(-1\)", "explication": r"\(\lvert \rho \rvert \le 1\) est la seule condition pour deux variables. Pour cinq, en \(\rho = -1\), la valeur propre \(1 + 4\rho\) vaut \(@Q01_vp_moins_un@\) : la somme des cinq variables aurait une variance négative."},
        ],
        "solution": r"""Réponse : \(\rho = @Q01_borne_num@/@Q01_borne_den@\).

Notations : \(\mathbf{I}\) est l'identité \(5 \times 5\), \(\mathbf{1}\) le vecteur de \(\mathbb{R}^5\) dont les coordonnées valent 1. Alors \(\mathbf{C} = (1 - \rho)\,\mathbf{I} + \rho\,\mathbf{1}\mathbf{1}^\top\).

Étape 1, valeurs propres : \(\mathbf{C}\mathbf{1} = (1 + 4\rho)\,\mathbf{1}\), car \(\mathbf{1}^\top \mathbf{1} = 5\) ; pour tout \(\mathbf{x}\) de somme nulle, \(\mathbf{C}\mathbf{x} = (1 - \rho)\,\mathbf{x}\), et ces vecteurs forment un sous-espace de dimension @Q01_mult@. Les valeurs propres sont donc \(1 + 4\rho\) (simple) et \(1 - \rho\) (multiplicité @Q01_mult@).

Étape 2 : une matrice symétrique est semi-définie positive si et seulement si toutes ses valeurs propres sont positives ou nulles (leçon, section 2). Il faut \(1 + 4\rho \ge 0\) et \(1 - \rho \ge 0\), soit \(@Q01_borne_num@/@Q01_borne_den@ \le \rho \le 1\).

Seconde voie, en 20 secondes : \(\mathrm{Var}(R_1 + \dots + R_5) = \mathbf{1}^\top \mathbf{C}\,\mathbf{1} = 5 + 20\rho \ge 0\).

Vérification (script) : valeurs propres exactes pour \(n\) de 2 à 8, plus petite valeur propre nulle en \(\rho = -1/(n-1)\) et négative juste en dessous.""",
    },
    {
        "id": "AL-02", "section": "math", "theme": "Matrice de corrélation, déterminant et mineurs", "difficulte": 4, "temps_cible_s": 150,
        "enonce": r"""Trois variables aléatoires \(X_1\), \(X_2\), \(X_3\) ont des variances finies et non nulles. On sait que \(\mathrm{corr}(X_1, X_2) = \mathrm{corr}(X_1, X_3) = 0{,}8\), où \(\mathrm{corr}\) désigne le coefficient de corrélation linéaire.

Quelle est la plus petite valeur possible de \(\mathrm{corr}(X_2, X_3)\) ?""",
        "options": [
            {"id": "a", "texte": "@Q02_exact:2@", "explication": r"Oui : le déterminant de la matrice de corrélation vaut \((1 - c)(1 + c - 2 \times 0{,}8^2)\), positif ou nul si et seulement si \(c \ge 2 \times 0{,}8^2 - 1\) (pour \(c < 1\))."},
            {"id": "b", "texte": "@Q02_d_produit:2@", "explication": r"C'est \(0{,}8^2\) : la corrélation obtenue quand \(X_2\) et \(X_3\) valent chacune \(0{,}8\,X_1\) plus un bruit indépendant de tout le reste (variables réduites). C'est une valeur possible, pas la plus petite : en \(c = @Q02_exact:2m@\), la matrice est encore semi-définie positive."},
            {"id": "c", "texte": "@Q02_d_equi:1@", "explication": r"C'est la borne \(-1/(n-1)\) de l'équicorrélation à trois variables, quand les trois corrélations sont égales. Ici deux corrélations sont imposées à 0,8 : en \(c = -0{,}5\), la plus petite valeur propre vaut \(@Q02_vp_cdemi:3m@ < 0\)."},
            {"id": "d", "texte": "0", "explication": r"Deux variables très corrélées à une même troisième ne peuvent pas être non corrélées entre elles : en \(c = 0\), la plus petite valeur propre de la matrice de corrélation vaut \(@Q02_vp_c0:3m@ < 0\)."},
        ],
        "solution": r"""Réponse : \(2 \times 0{,}8^2 - 1 = @Q02_exact:2m@\).

Notations : \(a = 0{,}8\), \(c = \mathrm{corr}(X_2, X_3)\), et \(\mathbf{C}\) la matrice de corrélation de \((X_1, X_2, X_3)\), de lignes \((1, a, a)\), \((a, 1, c)\), \((a, c, 1)\).

Étape 1 : une matrice de diagonale 1 est une matrice de corrélation si et seulement si elle est semi-définie positive (leçon, section 2), c'est-à-dire si et seulement si tous ses mineurs principaux sont positifs ou nuls (leçon, section 4).

Étape 2, mineurs d'ordre 1 et 2 : les coefficients diagonaux valent 1 ; \(1 - a^2 > 0\) deux fois ; \(1 - c^2 \ge 0\) exige \(-1 \le c \le 1\).

Étape 3, déterminant : en développant, \(\det(\mathbf{C}) = 1 + 2a^2 c - 2a^2 - c^2 = (1 - c)(1 + c - 2a^2)\). Pour \(c < 1\), il est positif ou nul si et seulement si \(c \ge 2a^2 - 1 = @Q02_2a2:2m@ - 1 = @Q02_exact:2m@\) ; pour \(c = 1\), il est nul.

Étape 4 : l'ensemble des valeurs possibles est donc \([@Q02_exact:2m@ ; 1]\), et la plus petite est \(@Q02_exact:2m@\). Elle est atteinte : la matrice correspondante est semi-définie positive, donc c'est la corrélation d'un vecteur gaussien (leçon, section 2, point 3).

Vérification (script) : factorisation symbolique du déterminant, plus petite valeur propre nulle en \(c = @Q02_exact:2m@\) et négative juste en dessous ; simulation d'un vecteur gaussien de cette corrélation : corrélation empirique de \(X_2\) et \(X_3\) égale à \(@Q02_corr23_mc:3m@\).""",
    },
    {
        "id": "AL-03", "section": "math", "theme": "Mineurs principaux, piège des mineurs dominants", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Soit \(\mathbf{A}\) la matrice symétrique \(3 \times 3\) de lignes \((0, 0, 0)\), \((0, 2, 3)\), \((0, 3, 1)\). Ses trois mineurs principaux dominants (les déterminants des sous-matrices formées des \(k\) premières lignes et des \(k\) premières colonnes, \(k = 1, 2, 3\)) sont tous nuls.

Quelle affirmation est vraie ?""",
        "options": [
            {"id": "a", "texte": r"\(\mathbf{A}\) n'est pas semi-définie positive : le mineur principal formé des lignes et colonnes 2 et 3 vaut @Q03_mineur23@.",
             "explication": r"Oui : une matrice semi-définie positive a **tous** ses mineurs principaux positifs ou nuls, et celui-ci est négatif. Contrôle direct : \(\mathbf{x} = (0, 1, -1)\) donne \(\mathbf{x}^\top \mathbf{A}\,\mathbf{x} = @Q03_forme@\)."},
            {"id": "b", "texte": r"\(\mathbf{A}\) est semi-définie positive, puisque ses mineurs principaux dominants sont tous positifs ou nuls.",
             "explication": r"Le critère des mineurs dominants (Sylvester) ne vaut qu'avec des inégalités strictes, et caractérise les matrices définies positives. Avec des inégalités larges, il faut tous les mineurs principaux."},
            {"id": "c", "texte": r"\(\mathbf{A}\) est semi-définie positive, puisqu'elle est symétrique et que ses coefficients diagonaux sont positifs ou nuls.",
             "explication": r"Une diagonale positive est nécessaire, pas suffisante : les coefficients hors diagonale peuvent créer une direction de variance négative, ici \((0, 1, -1)\)."},
            {"id": "d", "texte": r"\(\mathbf{A}\) n'est pas semi-définie positive, puisque son déterminant est nul.",
             "explication": r"La conclusion est juste mais la raison est fausse : un déterminant nul n'empêche pas d'être semi-définie positive (la matrice nulle, ou la matrice diagonale de coefficients 1 et 0, le sont)."},
        ],
        "solution": r"""Réponse : \(\mathbf{A}\) n'est pas semi-définie positive, car son mineur principal sur les indices 2 et 3 vaut \(2 \times 1 - 3 \times 3 = @Q03_mineur23@ < 0\).

Étape 1 : une matrice symétrique est semi-définie positive si et seulement si **tous** ses mineurs principaux (lignes et colonnes de mêmes indices, pour tout ensemble d'indices) sont positifs ou nuls (leçon, section 4). Les mineurs dominants ne suffisent pas.

Étape 2 : la sous-matrice principale sur les indices \(\{2, 3\}\) a pour lignes \((2, 3)\) et \((3, 1)\), de déterminant \(@Q03_mineur23@\).

Étape 3, certificat explicite : \(\mathbf{x} = (0, 1, -1)\) donne \(\mathbf{x}^\top \mathbf{A}\,\mathbf{x} = 2 - 2 \times 3 + 1 = @Q03_forme@ < 0\).

Valeurs propres (script, calcul exact) : 0 et \((3 \pm \sqrt{@Q03_racine@})/2\), dont une négative, \(\approx @Q03_vp_neg:3m@\).""",
    },
    # ------------------------------------------------- Covariance empirique
    {
        "id": "AL-04", "section": "math", "theme": "Rang d'une covariance empirique, T < N", "difficulte": 3, "temps_cible_s": 75,
        "enonce": r"""On observe les rendements de 300 actifs à 120 dates. Pour \(t = 1, \dots, 120\), \(\mathbf{X}_t \in \mathbb{R}^{300}\) est le vecteur aléatoire des rendements à la date \(t\) ; les 120 vecteurs sont indépendants et chacun a une densité sur \(\mathbb{R}^{300}\). Avec \(\bar{\mathbf{X}}\) leur moyenne, on forme la covariance empirique \(\mathbf{S} = \dfrac{1}{119}\sum_{t=1}^{120} (\mathbf{X}_t - \bar{\mathbf{X}})(\mathbf{X}_t - \bar{\mathbf{X}})^\top\), matrice \(300 \times 300\).

Combien de valeurs propres de \(\mathbf{S}\), comptées avec multiplicité, sont nulles avec probabilité 1 ?""",
        "options": [
            {"id": "a", "texte": "@Q04_exact@", "explication": r"Oui : le rang vaut \(\min(300, 120 - 1) = @Q04_rang@\), et \(300 - @Q04_rang@ = @Q04_exact@\) valeurs propres sont nulles."},
            {"id": "b", "texte": "@Q04_d_sans_centrage@", "explication": r"C'est \(300 - 120\), le résultat sans centrage. La soustraction de la moyenne lie les 120 vecteurs centrés (leur somme est nulle) : le rang tombe à \(@Q04_rang@\), et il y a une valeur propre nulle de plus."},
            {"id": "c", "texte": "@Q04_rang@", "explication": r"C'est le **rang** de \(\mathbf{S}\), c'est-à-dire le nombre de valeurs propres non nulles. Le nombre de valeurs propres nulles est la dimension du noyau, \(300 - @Q04_rang@\)."},
            {"id": "d", "texte": "0", "explication": r"Pour que \(\mathbf{S}\) n'ait aucune valeur propre nulle, il faudrait au moins \(300 + 1\) dates : avec 120 dates, son rang est au plus \(120 - 1\)."},
        ],
        "solution": r"""Réponse : @Q04_exact@.

Notations : \(\mathbf{Y}\) est la matrice aléatoire \(120 \times 300\) de lignes \((\mathbf{X}_t - \bar{\mathbf{X}})^\top\), de sorte que \(\mathbf{S} = \mathbf{Y}^\top \mathbf{Y}/119\).

Étape 1, majoration : \(\mathrm{rang}(\mathbf{S}) = \mathrm{rang}(\mathbf{Y})\) (même noyau : \(\mathbf{Y}^\top \mathbf{Y}\mathbf{v} = \mathbf{0}\) implique \(\lVert \mathbf{Y}\mathbf{v} \rVert^2 = 0\)), et les 120 lignes de \(\mathbf{Y}\) ont une somme nulle, donc \(\mathrm{rang}(\mathbf{Y}) \le 120 - 1 = @Q04_rang@\).

Étape 2, égalité avec probabilité 1 (leçon, section 8, démontré) : les différences \(\mathbf{X}_t - \mathbf{X}_{120}\), \(t \le 119\), engendrent le même espace que les lignes de \(\mathbf{Y}\) ; sachant \(\mathbf{X}_{120}\), ce sont @Q04_rang@ vecteurs indépendants à densité dans \(\mathbb{R}^{300}\), avec \(@Q04_rang@ \le 300\), donc libres avec probabilité 1, car un sous-espace strict est de mesure nulle. Le rang vaut @Q04_rang@.

Étape 3 : \(\mathbf{S}\) est symétrique ; son rang est le nombre de valeurs propres non nulles, donc \(300 - @Q04_rang@ = @Q04_exact@\) valeurs propres sont nulles.

Vérification (script) : tirage gaussien \(120 \times 300\), rang numérique @Q04_rang@.""",
    },
    {
        "id": "AL-05", "section": "math", "theme": "Rang d'une covariance après démoyennage transversal", "difficulte": 5, "temps_cible_s": 150,
        "enonce": r"""On observe les rendements de 100 actifs à 250 dates. Pour \(t = 1, \dots, 250\), \(\mathbf{X}_t \in \mathbb{R}^{100}\) est le vecteur aléatoire des rendements à la date \(t\) ; les 250 vecteurs sont indépendants et chacun a une densité sur \(\mathbb{R}^{100}\).

On retire d'abord, à chaque date, la moyenne des 100 rendements de cette date : \(\tilde{\mathbf{X}}_t = \mathbf{X}_t - \tfrac{1}{100}(\mathbf{1}^\top \mathbf{X}_t)\,\mathbf{1}\), où \(\mathbf{1}\) est le vecteur de \(\mathbb{R}^{100}\) dont les coordonnées valent 1. Puis, avec \(\bar{\tilde{\mathbf{X}}}\) la moyenne des \(\tilde{\mathbf{X}}_t\), on forme \(\tilde{\mathbf{S}} = \dfrac{1}{249}\sum_{t=1}^{250} (\tilde{\mathbf{X}}_t - \bar{\tilde{\mathbf{X}}})(\tilde{\mathbf{X}}_t - \bar{\tilde{\mathbf{X}}})^\top\).

Quel est le rang de \(\tilde{\mathbf{S}}\) avec probabilité 1 ?""",
        "options": [
            {"id": "a", "texte": "@Q05_exact@", "explication": r"Oui : chaque \(\tilde{\mathbf{X}}_t\) est orthogonal à \(\mathbf{1}\), donc \(\tilde{\mathbf{S}}\mathbf{1} = \mathbf{0}\) et le rang est au plus \(100 - 1 = @Q05_exact@\) ; dans l'hyperplan orthogonal à \(\mathbf{1}\), de dimension @Q05_exact@, les données ont une densité et \(T - 1 = @Q05_d_t@ \ge @Q05_exact@\)."},
            {"id": "b", "texte": "100", "explication": r"Avec 250 dates, la covariance des \(\mathbf{X}_t\) eux-mêmes serait de rang 100 avec probabilité 1. Mais le démoyennage transversal impose \(\mathbf{1}^\top \tilde{\mathbf{X}}_t = 0\) à chaque date : \(\mathbf{1}\) est dans le noyau de \(\tilde{\mathbf{S}}\)."},
            {"id": "c", "texte": "@Q05_d_t@", "explication": r"C'est \(T - 1\). Le rang ne peut pas dépasser la taille de la matrice, \(100\) ; la borne \(T - 1\) n'est active que quand \(T - 1\) est inférieur à la dimension."},
            {"id": "d", "texte": "@Q05_d_double@", "explication": r"On a retiré une unité pour chaque centrage. Le centrage temporel ne coûte rien ici : il borne le rang par \(T - 1 = @Q05_d_t@\), qui n'est pas la contrainte active. Seul le démoyennage transversal retire une dimension."},
        ],
        "solution": r"""Réponse : @Q05_exact@.

Notations : \(N = 100\), \(T = 250\) ; \(\mathbf{P} = \mathbf{I} - \mathbf{1}\mathbf{1}^\top/N\), projection orthogonale sur l'hyperplan \(H = \{\mathbf{z} : \mathbf{1}^\top \mathbf{z} = 0\}\), de dimension \(N - 1 = @Q05_exact@\), de sorte que \(\tilde{\mathbf{X}}_t = \mathbf{P}\mathbf{X}_t\) ; \(\mathbf{B}\) est une matrice \(N \times (N - 1)\) dont les colonnes forment une base orthonormée de \(H\) (donc \(\mathbf{B}^\top \mathbf{B} = \mathbf{I}_{N-1}\) et \(\mathbf{P} = \mathbf{B}\mathbf{B}^\top\)) ; \(\mathbf{W}_t = \mathbf{B}^\top \mathbf{X}_t \in \mathbb{R}^{N-1}\) sont les coordonnées de \(\tilde{\mathbf{X}}_t\) dans cette base, et \(\mathbf{S}_W\) la covariance empirique (diviseur 249) des \(\mathbf{W}_t\).

Étape 1, majoration : \(\mathbf{1}^\top \tilde{\mathbf{X}}_t = 0\) pour tout \(t\), donc aussi pour la moyenne, et \(\tilde{\mathbf{S}}\mathbf{1} = \mathbf{0}\) : rang au plus \(N - 1\).

Étape 2, réduction : \(\tilde{\mathbf{X}}_t = \mathbf{B}\mathbf{W}_t\), donc \(\tilde{\mathbf{S}} = \mathbf{B}\,\mathbf{S}_W \mathbf{B}^\top\) et \(\mathbf{S}_W = \mathbf{B}^\top \tilde{\mathbf{S}}\,\mathbf{B}\). Le rang d'un produit ne dépasse celui d'aucun facteur, donc \(\mathrm{rang}(\tilde{\mathbf{S}}) \le \mathrm{rang}(\mathbf{S}_W) \le \mathrm{rang}(\tilde{\mathbf{S}})\) : les deux rangs sont égaux.

Étape 3, densité : soit \(\mathbf{Q}\) la matrice orthogonale \(N \times N\) formée des colonnes de \(\mathbf{B}\) et de \(\mathbf{1}/\sqrt{N}\). \(\mathbf{Q}^\top \mathbf{X}_t\) a une densité (changement de variables orthogonal, de déterminant ±1, admis), et \(\mathbf{W}_t\) en est formé des \(N - 1\) premières coordonnées : une loi marginale d'une loi à densité a une densité (on intègre la densité jointe par rapport à la coordonnée retirée, théorème de Fubini). Les \(\mathbf{W}_t\) sont indépendants, comme fonctions de vecteurs indépendants.

Étape 4 : par le théorème de la leçon (section 8), \(\mathrm{rang}(\mathbf{S}_W) = \min(N - 1, T - 1) = @Q05_exact@\) avec probabilité 1, puisque \(T - 1 = @Q05_d_t@ \ge N - 1\).

Vérification (script) : trois tirages gaussiens corrélés, rang numérique @Q05_exact@ et \(\tilde{\mathbf{S}}\mathbf{1} = \mathbf{0}\).

Conséquence : une covariance de rendements démoyennés transversalement (rendements « neutres au marché » équipondérés) n'est jamais inversible, quel que soit \(T\).""",
    },
    # ------------------------------------------------- Projections, conditionnement, gradients, SVD
    {
        "id": "AL-06", "section": "math", "theme": "Projections orthogonales, trace et résidus", "difficulte": 3, "temps_cible_s": 120,
        "enonce": r"""Soit \(\mathbf{A}\) une matrice fixe \(100 \times 4\) de rang 4 (une colonne constante et trois variables explicatives), \(\mathbf{b} \in \mathbb{R}^4\) un vecteur fixe de coefficients inconnus, et \(\mathbf{U}\) un vecteur aléatoire de \(\mathbb{R}^{100}\) d'espérance nulle et de matrice de covariance \(\sigma^2 \mathbf{I}\), avec \(\sigma > 0\). On observe \(\mathbf{Y} = \mathbf{A}\mathbf{b} + \mathbf{U}\) et l'on calcule le vecteur des résidus des moindres carrés \(\hat{\mathbf{U}} = \mathbf{Y} - \mathbf{A}\hat{\mathbf{b}}\), avec \(\hat{\mathbf{b}} = (\mathbf{A}^\top \mathbf{A})^{-1}\mathbf{A}^\top \mathbf{Y}\).

Que vaut \(\mathbb{E}\big[\lVert \hat{\mathbf{U}} \rVert^2\big]\) ?""",
        "options": [
            {"id": "a", "texte": r"\(@Q06_exact@\,\sigma^2\)", "explication": r"Oui : \(\hat{\mathbf{U}} = (\mathbf{I} - \mathbf{H})\mathbf{U}\), et \(\mathbb{E}[\mathbf{U}^\top (\mathbf{I} - \mathbf{H})\mathbf{U}] = \sigma^2\,\mathrm{tr}(\mathbf{I} - \mathbf{H}) = \sigma^2 (100 - 4)\)."},
            {"id": "b", "texte": r"\(100\,\sigma^2\)", "explication": r"C'est \(\mathbb{E}[\lVert \mathbf{U} \rVert^2]\), la somme des carrés des vraies erreurs. Les résidus sont la projection de \(\mathbf{U}\) sur un sous-espace de dimension \(100 - 4\) : l'ajustement absorbe 4 dimensions."},
            {"id": "c", "texte": r"\(@Q06_d_p@\,\sigma^2\)", "explication": r"C'est \(\sigma^2\,\mathrm{tr}(\mathbf{H})\), l'espérance de \(\lVert \mathbf{H}\mathbf{U} \rVert^2\), la part de bruit absorbée par l'ajustement, pas celle qui reste dans les résidus."},
            {"id": "d", "texte": r"\(@Q06_d_double@\,\sigma^2\)", "explication": r"La constante a été retirée deux fois (\(100 - 4 - 1\)) : elle fait déjà partie des 4 colonnes de \(\mathbf{A}\). Le nombre de degrés de liberté est \(T - p\), \(p\) comptant toutes les colonnes."},
        ],
        "solution": r"""Réponse : \(@Q06_exact@\,\sigma^2\).

Notations : \(\mathbf{H} = \mathbf{A}(\mathbf{A}^\top \mathbf{A})^{-1}\mathbf{A}^\top\) (matrice chapeau), \(T = 100\), \(p = 4\).

Étape 1 : \(\hat{\mathbf{U}} = (\mathbf{I} - \mathbf{H})\mathbf{Y}\), et \((\mathbf{I} - \mathbf{H})\mathbf{A} = \mathbf{A} - \mathbf{A} = \mathbf{0}\), donc \(\hat{\mathbf{U}} = (\mathbf{I} - \mathbf{H})\mathbf{U}\).

Étape 2 : \(\mathbf{I} - \mathbf{H}\) est symétrique et idempotente (projection orthogonale, leçon, section 3), donc \(\lVert \hat{\mathbf{U}} \rVert^2 = \mathbf{U}^\top (\mathbf{I} - \mathbf{H})^\top (\mathbf{I} - \mathbf{H})\mathbf{U} = \mathbf{U}^\top (\mathbf{I} - \mathbf{H})\mathbf{U}\).

Étape 3 : un nombre est égal à sa trace, et \(\mathrm{tr}(\mathbf{M}\mathbf{N}) = \mathrm{tr}(\mathbf{N}\mathbf{M})\), donc \(\mathbf{U}^\top (\mathbf{I} - \mathbf{H})\mathbf{U} = \mathrm{tr}\big((\mathbf{I} - \mathbf{H})\mathbf{U}\mathbf{U}^\top\big)\). Par linéarité de l'espérance et de la trace, \(\mathbb{E}[\cdot] = \mathrm{tr}\big((\mathbf{I} - \mathbf{H})\,\sigma^2 \mathbf{I}\big) = \sigma^2\,\mathrm{tr}(\mathbf{I} - \mathbf{H})\).

Étape 4 : \(\mathrm{tr}(\mathbf{H}) = \mathrm{tr}\big((\mathbf{A}^\top \mathbf{A})^{-1}\mathbf{A}^\top \mathbf{A}\big) = \mathrm{tr}(\mathbf{I}_4) = 4\), donc \(\mathrm{tr}(\mathbf{I} - \mathbf{H}) = 100 - 4 = @Q06_exact@\).

Conséquence : \(\lVert \hat{\mathbf{U}} \rVert^2/(T - p)\) est un estimateur sans biais de \(\sigma^2\). Simulation (script, 50 000 tirages gaussiens) : moyenne de \(\lVert \hat{\mathbf{U}} \rVert^2/\sigma^2\) égale à \(@Q06_rss_sur_sigma2_mc:2m@\), pour une erreur standard \(@Q06_rss_sur_sigma2_se:2m@\).""",
    },
    {
        "id": "AL-07", "section": "math", "theme": "Conditionnement et sensibilité d'un système linéaire", "difficulte": 3, "temps_cible_s": 120,
        "enonce": r"""Soit \(\mathbf{S}\) une matrice symétrique définie positive \(3 \times 3\) de valeurs propres 4, 1 et 0,04. Pour un vecteur \(\mathbf{m} \ne \mathbf{0}\), on note \(\mathbf{w}\) la solution de \(\mathbf{S}\mathbf{w} = \mathbf{m}\). On perturbe le second membre en \(\mathbf{m} + \boldsymbol{\delta}\), avec \(\lVert \boldsymbol{\delta} \rVert / \lVert \mathbf{m} \rVert = 0{,}001\) (norme euclidienne), et l'on note \(\mathbf{w} + \boldsymbol{\Delta}\) la nouvelle solution.

Quelle est la plus grande valeur possible de \(\lVert \boldsymbol{\Delta} \rVert / \lVert \mathbf{w} \rVert\), sur tous les choix de \(\mathbf{m}\) et de \(\boldsymbol{\delta}\) ?""",
        "options": [
            {"id": "a", "texte": "@Q07_exact:1@", "explication": r"Oui : le conditionnement vaut \(4/0{,}04 = @Q07_kappa@\), et la borne \(\kappa \times 0{,}001\) est atteinte pour \(\mathbf{m}\) le long du vecteur propre de valeur propre 4 et \(\boldsymbol{\delta}\) le long de celui de valeur propre 0,04."},
            {"id": "b", "texte": "@Q07_d_inv:3@", "explication": r"C'est \(0{,}001/0{,}04\) : seule la norme de \(\mathbf{S}^{-1}\) a été prise en compte. Elle borne l'erreur absolue \(\lVert \boldsymbol{\Delta} \rVert \le \lVert \boldsymbol{\delta} \rVert/0{,}04\) ; pour l'erreur relative, il faut aussi \(\lVert \mathbf{m} \rVert \le 4\,\lVert \mathbf{w} \rVert\)."},
            {"id": "c", "texte": "@Q07_d_racine:2@", "explication": r"C'est \(0{,}001 \times \sqrt{@Q07_kappa@}\). La racine du conditionnement n'intervient pas : l'amplification maximale de l'erreur relative est le conditionnement lui-même."},
            {"id": "d", "texte": "@Q07_d_max:3@", "explication": r"C'est \(0{,}001 \times 4\), avec la plus grande valeur propre. Ce sont les petites valeurs propres qui amplifient : \(\mathbf{S}^{-1}\) multiplie par \(1/0{,}04\) les composantes le long du vecteur propre de valeur propre 0,04."},
        ],
        "solution": r"""Réponse : \(@Q07_kappa@ \times 0{,}001 = @Q07_exact:1m@\).

Notations : \(\mathbf{q}_1\), \(\mathbf{q}_2\), \(\mathbf{q}_3\) sont des vecteurs propres unitaires orthogonaux de valeurs propres 4, 1 et 0,04 ; \(\kappa = 4/0{,}04 = @Q07_kappa@\) est le conditionnement (pour une matrice symétrique définie positive, valeurs singulières = valeurs propres).

Étape 1, borne (leçon, section 6) : \(\boldsymbol{\Delta} = \mathbf{S}^{-1}\boldsymbol{\delta}\), donc \(\lVert \boldsymbol{\Delta} \rVert \le \lVert \boldsymbol{\delta} \rVert/0{,}04\) ; et \(\lVert \mathbf{m} \rVert = \lVert \mathbf{S}\mathbf{w} \rVert \le 4\,\lVert \mathbf{w} \rVert\). En multipliant : \(\lVert \boldsymbol{\Delta} \rVert/\lVert \mathbf{w} \rVert \le \kappa\,\lVert \boldsymbol{\delta} \rVert/\lVert \mathbf{m} \rVert = @Q07_exact:1m@\).

Étape 2, la borne est atteinte : \(\mathbf{m} = \mathbf{q}_1\) donne \(\mathbf{w} = \mathbf{q}_1/4\) ; \(\boldsymbol{\delta} = 0{,}001\,\mathbf{q}_3\) donne \(\boldsymbol{\Delta} = 0{,}001\,\mathbf{q}_3/0{,}04\). Le rapport vaut \((0{,}001/0{,}04)/(1/4) = @Q07_exact:1m@\).

Vérification (script) : l'exemple diagonal atteint la borne, et 2000 couples tirés au hasard ne la dépassent jamais. Lecture pour un portefeuille : des poids \(\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}\) calculés avec une covariance mal conditionnée amplifient d'autant les erreurs sur les rendements espérés.""",
    },
    {
        "id": "AL-08", "section": "math", "theme": "Calcul matriciel, gradient d'une forme quadratique", "difficulte": 2, "temps_cible_s": 75,
        "enonce": r"""Soit \(\mathbf{A}\) la matrice \(2 \times 2\) de lignes \((1, 2)\) et \((0, 3)\), et \(f(\mathbf{x}) = \mathbf{x}^\top \mathbf{A}\,\mathbf{x}\) pour \(\mathbf{x} \in \mathbb{R}^2\).

Que vaut le gradient de \(f\) au point \(\mathbf{x} = (1, -1)\) ?""",
        "options": [
            {"id": "a", "texte": r"\(@Q08_exact@\)", "explication": r"Oui : \(\nabla f(\mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\,\mathbf{x}\), et \(\mathbf{A} + \mathbf{A}^\top\) a pour lignes \((@Q08_s11@, @Q08_s12@)\) et \((@Q08_s12@, @Q08_s22@)\)."},
            {"id": "b", "texte": r"\(@Q08_d_2ax@\)", "explication": r"C'est \(2\mathbf{A}\mathbf{x}\), la formule valable seulement pour \(\mathbf{A}\) symétrique. Ici \(\mathbf{A}\) ne l'est pas (\(2 \ne 0\) hors de la diagonale)."},
            {"id": "c", "texte": r"\(@Q08_d_2atx@\)", "explication": r"C'est \(2\mathbf{A}^\top \mathbf{x}\) : même erreur que \(2\mathbf{A}\mathbf{x}\), avec l'autre moitié de la formule. Le gradient fait intervenir les deux : \(\mathbf{A}\mathbf{x} + \mathbf{A}^\top \mathbf{x}\)."},
            {"id": "d", "texte": r"\(@Q08_d_ax@\)", "explication": r"C'est \(\mathbf{A}\mathbf{x}\), sans le second terme \(\mathbf{A}^\top \mathbf{x}\). La dérivée d'un terme \(a_{ij} x_i x_j\) touche à la fois \(x_i\) et \(x_j\)."},
        ],
        "solution": r"""Réponse : \(\nabla f(1, -1) = @Q08_exact@\).

Étape 1, formule (leçon, section 7) : \(f(\mathbf{x}) = \sum_{i,j} a_{ij} x_i x_j\), et \(\partial f/\partial x_k = \sum_j a_{kj} x_j + \sum_i a_{ik} x_i = (\mathbf{A}\mathbf{x})_k + (\mathbf{A}^\top \mathbf{x})_k\).

Étape 2 : \(\mathbf{A}\mathbf{x} = @Q08_ax@\) et \(\mathbf{A}^\top \mathbf{x} = @Q08_atx@\), de somme \(@Q08_exact@\).

Contrôle direct : \(f(x_1, x_2) = x_1^2 + (2 + 0)\,x_1 x_2 + 3x_2^2\), de dérivées partielles \(@Q08_s11@x_1 + @Q08_s12@x_2\) et \(@Q08_s12@x_1 + @Q08_s22@x_2\), qui valent bien \(@Q08_exact@\) en \((1, -1)\). Seule la partie symétrique \((\mathbf{A} + \mathbf{A}^\top)/2\), de lignes \((@Q08_h11@, @Q08_h12@)\) et \((@Q08_h12@, @Q08_h22@)\), compte pour \(f\).""",
    },
    {
        "id": "AL-09", "section": "math", "theme": "SVD, meilleure approximation de rang k", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""Une matrice réelle \(\mathbf{X}\) de taille \(6 \times 4\) a pour valeurs singulières 4, 3, 2 et 1. On note \(\lVert \mathbf{M} \rVert_F = \sqrt{\sum_{i,j} m_{ij}^2}\) la norme de Frobenius d'une matrice \(\mathbf{M}\) de coefficients \(m_{ij}\).

Quelle est la plus petite valeur de \(\lVert \mathbf{X} - \mathbf{B} \rVert_F\), sur toutes les matrices \(\mathbf{B}\) de taille \(6 \times 4\) et de rang au plus 2 ?""",
        "options": [
            {"id": "a", "texte": r"\(\sqrt{@Q09_somme_carres@} \approx @Q09_exact:3m@\)", "explication": r"Oui : par Eckart-Young, la meilleure approximation est la SVD tronquée aux deux plus grandes valeurs singulières, et l'erreur est \(\sqrt{2^2 + 1^2}\)."},
            {"id": "b", "texte": "@Q09_d_spectrale@", "explication": r"C'est l'erreur minimale en norme d'**opérateur** (la troisième valeur singulière). En norme de Frobenius, toutes les valeurs singulières omises contribuent, au carré."},
            {"id": "c", "texte": "@Q09_d_somme@", "explication": r"C'est \(2 + 1\), la somme des valeurs singulières omises. La norme de Frobenius additionne les **carrés** : \(\lVert \mathbf{M} \rVert_F^2 = \sum_k s_k(\mathbf{M})^2\)."},
            {"id": "d", "texte": "@Q09_somme_carres@", "explication": r"C'est le **carré** de l'erreur, \(2^2 + 1^2\). L'erreur elle-même en est la racine."},
        ],
        "solution": r"""Réponse : \(\sqrt{2^2 + 1^2} = \sqrt{@Q09_somme_carres@} \approx @Q09_exact:3m@\).

Notations : \(\mathbf{X} = \sum_{k=1}^{4} s_k \mathbf{u}_k \mathbf{v}_k^\top\) est la SVD, avec \(s = (4, 3, 2, 1)\).

Étape 1 (leçon, section 5, Eckart-Young démontré) : pour toute \(\mathbf{B}\) de rang au plus 2, \(\lVert \mathbf{X} - \mathbf{B} \rVert_F^2 \ge s_3^2 + s_4^2\), avec égalité pour \(\mathbf{B} = s_1 \mathbf{u}_1 \mathbf{v}_1^\top + s_2 \mathbf{u}_2 \mathbf{v}_2^\top\).

Étape 2 : \(\mathbf{X} - \mathbf{B} = s_3 \mathbf{u}_3 \mathbf{v}_3^\top + s_4 \mathbf{u}_4 \mathbf{v}_4^\top\), de norme de Frobenius au carré \(s_3^2 + s_4^2 = @Q09_somme_carres@\).

Vérification (script) : matrice construite avec ces valeurs singulières, erreur de la SVD tronquée égale à \(@Q09_exact:3m@\), et aucune des matrices de rang 2 tirées au hasard ne fait mieux.""",
    },
    # ------------------------------------------------- ACP
    {
        "id": "AL-10", "section": "math", "theme": "ACP de rendements standardisés, part de variance", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""On dispose des rendements de 5 actifs sur \(T\) dates, rangés dans une matrice \(T \times 5\) dont aucune colonne n'est constante. Chaque colonne est centrée, puis divisée par son écart-type empirique calculé avec le diviseur \(T\) ; on note \(\mathbf{Z}\) la matrice obtenue, et l'on fait l'ACP de \(\mathbf{Z}^\top \mathbf{Z}/T\). Les quatre plus grandes valeurs propres de cette matrice valent 2,6 ; 1,1 ; 0,7 et 0,4.

Quelle part de la variance totale de \(\mathbf{Z}\) expliquent les deux premières composantes principales ?""",
        "options": [
            {"id": "a", "texte": "@Q10_exact:2@", "explication": r"Oui : \(\mathbf{Z}^\top \mathbf{Z}/T\) est la matrice de corrélation, de trace 5 ; la part vaut \((2{,}6 + 1{,}1)/5\)."},
            {"id": "b", "texte": "@Q10_d_quatre:3@", "explication": r"C'est \((2{,}6 + 1{,}1)/@Q10_d_somme4:1@\) : on a divisé par la somme des quatre valeurs propres données, en oubliant la cinquième, \(5 - @Q10_d_somme4:1@ = @Q10_lam5:1@\)."},
            {"id": "c", "texte": "@Q10_d_premiere:2@", "explication": r"C'est la part de la seule première composante, \(2{,}6/5\). La question porte sur les deux premières."},
            {"id": "d", "texte": "Impossible à déterminer sans la cinquième valeur propre.",
             "explication": r"La cinquième se déduit des autres : la somme des valeurs propres est la trace, et la trace d'une matrice de corrélation \(5 \times 5\) vaut 5. Donc \(\lambda_5 = @Q10_lam5:1@\)."},
        ],
        "solution": r"""Réponse : \((2{,}6 + 1{,}1)/5 = @Q10_somme2:1m@/5 = @Q10_exact:2m@\).

Étape 1 (leçon, section 9, point 4) : avec des écarts-types de diviseur \(T\) et la matrice \(\mathbf{Z}^\top \mathbf{Z}/T\), de même diviseur, le coefficient \((i, j)\) est la covariance empirique des actifs \(i\) et \(j\) divisée par le produit de leurs écarts-types : c'est la **matrice de corrélation empirique**. Son coefficient diagonal vaut 1.

Étape 2 : la variance totale de \(\mathbf{Z}\) est la trace, \(5\), égale à la somme des cinq valeurs propres (leçon, section 1). Donc \(\lambda_5 = 5 - @Q10_d_somme4:1m@ = @Q10_lam5:1m@\), cohérent avec \(\lambda_5 \le \lambda_4 = 0{,}4\).

Étape 3 : part des deux premières composantes, \((2{,}6 + 1{,}1)/5 = @Q10_somme2:1m@/5 = @Q10_exact:2m@\).

Remarques : avec un diviseur \(T - 1\) pour les écarts-types et \(T\) pour la matrice, toutes les valeurs propres seraient multipliées par la même constante : la réponse, un rapport, ne changerait pas. Vérification (script) : une matrice de corrélation \(5 \times 5\) ayant exactement ces cinq valeurs propres existe (construction explicite par rotations), ce qui garantit que l'énoncé est réalisable.""",
    },
    {
        "id": "AL-11", "section": "math", "theme": "ACP de la corrélation dans un modèle à un facteur", "difficulte": 4, "temps_cible_s": 150,
        "enonce": r"""Les rendements de 20 actifs s'écrivent \(R_i = F + E_i\) pour \(i = 1, \dots, 20\), où \(F\) est une variable aléatoire de variance \(0{,}01^2\), et où les \(E_i\) sont des variables aléatoires de variance \(0{,}02^2\) chacune, non corrélées entre elles et non corrélées avec \(F\).

On fait l'ACP de la matrice de corrélation (théorique) du vecteur \((R_1, \dots, R_{20})\), ce qui revient à une ACP de rendements standardisés. Quelle part de la variance totale explique la première composante principale ?""",
        "options": [
            {"id": "a", "texte": "@Q11_exact:2@", "explication": r"Oui : la corrélation commune vaut \(\rho = @Q11_rho:1m@\), la plus grande valeur propre \(1 + (20 - 1)\rho = @Q11_lam1:1m@\), et la trace vaut 20."},
            {"id": "b", "texte": "@Q11_rho:2@", "explication": r"C'est la corrélation \(\rho\) entre deux actifs, qui est la limite de la part quand le nombre d'actifs tend vers l'infini. Avec 20 actifs, la part vaut \((1 + (20 - 1)\rho)/20\), un peu plus grande."},
            {"id": "c", "texte": "@Q11_d_n:2@", "explication": r"La plus grande valeur propre a été prise égale à \(1 + 20\rho\) au lieu de \(1 + (20 - 1)\rho\) : la diagonale contribue 1, et chacun des 19 autres coefficients de la ligne contribue \(\rho\)."},
            {"id": "d", "texte": "@Q11_d_inv:2@", "explication": r"C'est \(1/20\), la part de chaque composante si les actifs étaient non corrélés. Ici le facteur commun crée une corrélation positive, qui concentre la variance sur la première composante."},
        ],
        "solution": r"""Réponse : \((1 + (20 - 1)\rho)/20 = @Q11_exact:2m@\), avec \(\rho = @Q11_rho:1m@\).

Étape 1, corrélation : \(\mathrm{Var}(R_i) = 0{,}01^2 + 0{,}02^2 = @Q11_var_tot:4m@\) (non-corrélation de \(F\) et \(E_i\)) et, pour \(i \ne j\), \(\mathrm{Cov}(R_i, R_j) = \mathrm{Var}(F) = 0{,}01^2\) (bilinéarité, les autres termes étant nuls). Donc \(\rho = 0{,}01^2/@Q11_var_tot:4m@ = @Q11_rho:1m@\) pour toute paire.

Étape 2 : la matrice de corrélation est l'équicorrélation \(20 \times 20\) de paramètre \(\rho\) (leçon, section 2), de valeurs propres \(1 + (20 - 1)\rho = @Q11_lam1:1m@\) (vecteur \(\mathbf{1}\), portefeuille équipondéré) et \(1 - \rho\) (multiplicité \(20 - 1\)).

Étape 3 : la variance totale est la trace, 20 ; la part de la première composante vaut \(@Q11_lam1:1m@/20 = @Q11_exact:2m@\).

Vérification (script) : valeurs propres de la matrice de corrélation exacte, et part de la première composante de la corrélation empirique d'une simulation de 20 000 dates : \(@Q11_part_simulee:4m@\).""",
    },
    # ------------------------------------------------- Ledoit-Wolf
    {
        "id": "AL-12", "section": "math", "theme": "Rétrécissement de Ledoit-Wolf, conditionnement", "difficulte": 4, "temps_cible_s": 150,
        "enonce": r"""Une covariance empirique \(\mathbf{S}\) de taille \(4 \times 4\) a pour valeurs propres 6, 3, 1 et 0. On la rétrécit vers la cible \(\mu\,\mathbf{I}\), où \(\mu = \mathrm{tr}(\mathbf{S})/4\) et \(\mathbf{I}\) est l'identité \(4 \times 4\), avec l'intensité \(\delta = 0{,}2\) : \(\hat{\boldsymbol{\Sigma}} = (1 - \delta)\,\mathbf{S} + \delta\,\mu\,\mathbf{I}\).

Que vaut le conditionnement de \(\hat{\boldsymbol{\Sigma}}\), rapport de sa plus grande à sa plus petite valeur propre ?""",
        "options": [
            {"id": "a", "texte": "@Q12_exact:1@", "explication": r"Oui : \(\mu = @Q12_mu:1m@\), valeurs propres extrêmes \(0{,}8 \times 6 + 0{,}2\mu = @Q12_nouv_max:1m@\) et \(0{,}8 \times 0 + 0{,}2\mu = @Q12_nouv_min:1m@\)."},
            {"id": "b", "texte": "@Q12_d_inverse:1@", "explication": r"Les poids ont été échangés : \(0{,}2\,\mathbf{S} + 0{,}8\,\mu\,\mathbf{I}\). L'intensité \(\delta\) est le poids de la **cible**."},
            {"id": "c", "texte": "@Q12_d_identite:1@", "explication": r"C'est le rétrécissement vers \(\mathbf{I}\) au lieu de \(\mu\,\mathbf{I}\) : valeurs propres extrêmes \(@Q12_d_id_max:1m@\) et \(@Q12_d_id_min:1m@\). La cible de Ledoit-Wolf est remise à l'échelle par \(\mu\), la moyenne des valeurs propres."},
            {"id": "d", "texte": r"Infini : \(\mathbf{S}\) est singulière, et le rétrécissement conserve son noyau.",
             "explication": r"Le rétrécissement conserve les vecteurs propres, pas les valeurs propres : la valeur propre nulle devient \(\delta\mu > 0\), et \(\hat{\boldsymbol{\Sigma}}\) est inversible."},
        ],
        "solution": r"""Réponse : \(@Q12_nouv_max:1m@/@Q12_nouv_min:1m@ = @Q12_exact:1m@\).

Étape 1 : \(\mu = (6 + 3 + 1 + 0)/4 = @Q12_mu:1m@\).

Étape 2 (leçon, section 10) : \(\hat{\boldsymbol{\Sigma}}\) a les mêmes vecteurs propres que \(\mathbf{S}\), de valeurs propres \((1 - \delta)\lambda + \delta\mu\), soit respectivement @Q12_nouv@ pour \(\lambda\) égal à 6, 3, 1 et 0.

Étape 3 : conditionnement \(@Q12_nouv_max:1m@/@Q12_nouv_min:1m@ = @Q12_exact:1m@\), alors que celui de \(\mathbf{S}\) est infini. Il décroît vers 1 quand \(\delta\) croît vers 1 (leçon, section 10, démontré).

Vérification (script) : matrice construite avec une base de vecteurs propres aléatoire, valeurs propres et conditionnement recalculés numériquement.""",
    },
]
