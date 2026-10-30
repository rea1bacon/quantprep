"""Séance de mathématiques 1 : leçon de probabilités (quatre outils), puis 12 questions à choix multiple.

Les questions sont équivalentes, jamais identiques, à quatre questions de probabilités du test de
positionnement : DIAG-M2 (ruine du joueur), DIAG-M3 (indicatrices), DIAG-M6 (statistiques
d'ordre), DIAG-M7 (tests multiples).
Toutes les valeurs : verif/seance_math_1.py. Dans chaque question, l'option "a" est la bonne.
"""
import lecons_probas
import questions_diagnostic as diag

ID = "seance-math-1"
TITRE = "Séance de mathématiques 1 : probabilités"
VERIF = "verif/seance_math_1.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [{"id": "lecon", "nom": "Leçon", "budget_min": 40}, {"id": "math", "nom": "Mathématiques", "budget_min": 25}]
PRESENTATION = (
    "D'abord une leçon d'environ 40 minutes sur quatre outils de calcul rapide en probabilités, avec six points de contrôle "
    "corrigés immédiatement. Ensuite 12 questions à choix multiple, trois ou quatre par outil, équivalentes à quatre "
    "questions de probabilités du test de positionnement, avec d'autres nombres et d'autres habillages.\n\n"
    "Objectif de vitesse : chaque question se résout en moins de 2 minutes avec les outils de la leçon. Commence par "
    "reconnaître l'outil (tableau « Réflexes en 30 secondes »), puis calcule. Au-delà du double du temps cible "
    "affiché, choisis « Je ne sais pas » et passe à la suivante : c'est la règle d'abandon à suivre dans un test chronométré."
)
AVERTISSEMENTS = [
    "Les questions de mathématiques sont présentées ici en choix multiple à une seule bonne réponse. Cherche la réponse avant de "
    "lire les options, pour t'entraîner aussi au format à réponse numérique.",
    "Ici le retour en arrière est permis et il n'y a pas de pénalité : ce sont les règles de cette plateforme, pas "
    "celles de tout test en ligne.",
]

LECONS = [lecons_probas.LECON_PROBAS]
CODES = []

QUESTIONS = [
    # ------------------------------------------------- Outil 1 : linéarité, indicatrices
    {
        "id": "S1M-01", "section": "math", "theme": "Linéarité de l'espérance, indicatrices", "difficulte": 2, "temps_cible_s": 75,
        "enonce": r"""On lance 12 fois une pièce équilibrée ; les lancers sont indépendants et chacun donne P (pile) ou F (face) avec probabilité 1/2. Une occurrence du motif PFP est un numéro \(j\) tel que les lancers numéros \(j\), \(j+1\) et \(j+2\) existent et donnent P, F, P dans cet ordre. Les occurrences peuvent se chevaucher : la suite PFPFP contient @Q01_ex@ occurrences.

Quelle est l'espérance du nombre d'occurrences du motif PFP, arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q01_exact:3@", "explication": r"Oui : @Q01_pos@ positions possibles, chacune occupée par le motif avec probabilité \(1/@Q01_p_den@\)."},
            {"id": "b", "texte": "@Q01_d_12pos:3@", "explication": r"On a compté 12 positions de départ. Le motif occupe 3 lancers : il ne peut commencer qu'aux positions 1 à @Q01_pos@."},
            {"id": "c", "texte": "@Q01_d_au_moins_un:3@", "explication": r"C'est \(1 - (1 - 1/@Q01_p_den@)^{@Q01_pos@}\), un calcul de la probabilité d'au moins une occurrence qui suppose de plus les positions indépendantes, ce qu'elles ne sont pas. La question demande l'espérance du nombre d'occurrences, qui peut dépasser 1."},
            {"id": "d", "texte": "@Q01_d_proba1:3@", "explication": r"C'est la probabilité d'une occurrence à une position donnée, \(\mathbb{E}[I_j] = 1/@Q01_p_den@\). Il faut encore sommer sur les @Q01_pos@ positions."},
        ],
        "solution": r"""Réponse : \(@Q01_pos@ \times 1/@Q01_p_den@ = @Q01_exact:3m@\).

Notations : pour \(j = 1, \dots, @Q01_pos@\), \(I_j\) est l'indicatrice de l'événement « les lancers \(j\), \(j+1\), \(j+2\) donnent P, F, P », et \(N = I_1 + \dots + I_{@Q01_pos@}\) est le nombre d'occurrences.

Étape 1 : le motif occupe 3 lancers consécutifs ; il peut commencer aux positions 1 à \(12 - 2 = @Q01_pos@\).

Étape 2 : les trois lancers concernés sont indépendants, donc \(\mathbb{P}(I_j = 1) = (1/2)^3 = 1/@Q01_p_den@\).

Étape 3 : par linéarité de l'espérance, qui ne demande aucune indépendance entre les \(I_j\) (deux positions voisines ne sont pas indépendantes), \(\mathbb{E}[N] = @Q01_pos@/@Q01_p_den@ = @Q01_exact:3m@\).

Vérification (script) : énumération des \(2^{12}\) suites, et simulation : @Q01_mc:3@.""",
    },
    {
        "id": "S1M-02", "section": "math", "theme": "Linéarité de l'espérance, indicatrices", "difficulte": 3, "temps_cible_s": 120,
        "enonce": r"""Une urne contient 20 boules numérotées de 1 à 20. On effectue 10 tirages avec remise : chaque tirage choisit une boule uniformément au hasard, indépendamment des autres tirages, puis la remet dans l'urne. Par exemple, les tirages 4, 7, 4, 4, 12, 7, 1, 1, 9, 20 donnent @Q02_ex_distincts@ numéros distincts.

Quelle est l'espérance du nombre de numéros distincts obtenus, arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q02_exact:3@", "explication": r"Oui : une indicatrice par numéro possible, de probabilité \(1 - (19/20)^{10}\), et 20 numéros."},
            {"id": "b", "texte": "@Q02_d_absents:3@", "explication": r"C'est \(20 \times (19/20)^{10}\), l'espérance du nombre de numéros **jamais** obtenus. Le nombre de numéros distincts est 20 moins cette quantité."},
            {"id": "c", "texte": "@Q02_d_paires:3@", "explication": r"C'est 10 moins l'espérance du nombre de paires de tirages égaux, \(\binom{10}{2}/20 = @Q02_nb_paires@/20\). Ce décompte est faux dès que trois tirages coïncident : ils forment 3 paires pour seulement 2 tirages en trop."},
            {"id": "d", "texte": "@Q02_d_uniques:3@", "explication": r"C'est \(10 \times (19/20)^{9}\), l'espérance du nombre de tirages dont le numéro ne sort qu'une fois. Un numéro sorti plusieurs fois compte aussi comme un numéro distinct."},
        ],
        "solution": r"""Réponse : \(20\,\big(1 - (19/20)^{10}\big) \approx @Q02_exact:3m@\).

Notations : pour chaque numéro \(k \in \{1, \dots, 20\}\), \(I_k\) est l'indicatrice de l'événement « le numéro \(k\) sort au moins une fois », et \(N = I_1 + \dots + I_{20}\) est le nombre de numéros distincts.

Étape 1 : à un tirage donné, \(k\) ne sort pas avec probabilité \(19/20 = @Q02_ratio:2m@\). Les 10 tirages étant indépendants, \(k\) ne sort jamais avec probabilité \((19/20)^{10} \approx @Q02_p_absent:4m@\).

Étape 2 : \(\mathbb{E}[I_k] = \mathbb{P}(I_k = 1) = 1 - (19/20)^{10}\).

Étape 3 : par linéarité de l'espérance (les \(I_k\) ne sont pas indépendants, et c'est sans importance), \(\mathbb{E}[N] = 20\,\big(1 - (19/20)^{10}\big) \approx @Q02_exact:3m@\).

Réflexe : une indicatrice par **valeur possible** (20 numéros), pas par tirage.""",
    },
    {
        "id": "S1M-03", "section": "math", "theme": "Linéarité de l'espérance, variance d'un comptage", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""On tire une permutation \(S\) de \(\{1, \dots, 10\}\) uniformément au hasard parmi les \(10!\) permutations. Un point fixe de \(S\) est un entier \(i\) tel que \(S(i) = i\). Soit \(N\) le nombre de points fixes de \(S\).

Que vaut \(\mathrm{Var}(N)\) ?""",
        "options": [
            {"id": "a", "texte": "@Q03_exact@", "explication": r"Oui : \(\mathbb{E}[N] = 1\) et \(\mathbb{E}[N^2] = 2\)."},
            {"id": "b", "texte": "@Q03_d_indep:1@", "explication": r"C'est \(\sum_i \mathrm{Var}(I_i) = 10 \times \tfrac{1}{10} \times \tfrac{9}{10}\), qui suppose les indicatrices non corrélées. Elles sont positivement corrélées : \(\mathbb{P}(I_1 = 1, I_2 = 1) = 1/@Q03_nb_couples@\), plus grand que \(\mathbb{P}(I_1 = 1)\,\mathbb{P}(I_2 = 1) = 1/@Q03_n2@\)."},
            {"id": "c", "texte": "@Q03_d_e2@", "explication": r"C'est \(\mathbb{E}[N^2]\). La variance est \(\mathbb{E}[N^2] - \mathbb{E}[N]^2\), avec \(\mathbb{E}[N] = 1\)."},
            {"id": "d", "texte": "@Q03_d_paires:1@", "explication": r"Les couples d'indices distincts ont été comptés une seule fois (\(\binom{10}{2} = @Q03_nb_paires@\) paires) au lieu de deux : dans \(N^2 = \sum_i \sum_j I_i I_j\), les termes \(I_1 I_2\) et \(I_2 I_1\) apparaissent tous les deux, soit @Q03_nb_couples@ couples ordonnés."},
        ],
        "solution": r"""Réponse : \(\mathrm{Var}(N) = @Q03_exact@\).

Notations : \(I_i\) est l'indicatrice de l'événement \(\{S(i) = i\}\), et \(N = I_1 + \dots + I_{10}\).

Étape 1, espérance : les permutations qui fixent \(i\) sont les \(9!\) permutations des autres éléments, donc \(\mathbb{P}(S(i) = i) = 9!/10! = 1/10\) et, par linéarité, \(\mathbb{E}[N] = 10 \times 1/10 = 1\).

Étape 2, développement : \(N^2 = \sum_i I_i^2 + \sum_{i \ne j} I_i I_j\), où la seconde somme porte sur les \(10 \times 9 = @Q03_nb_couples@\) couples **ordonnés** d'indices distincts. Comme \(I_i^2 = I_i\), la première somme a pour espérance \(\mathbb{E}[N] = 1\).

Étape 3 : \(I_i I_j\) est l'indicatrice de « \(i\) et \(j\) sont fixes », de probabilité \(8!/10! = 1/@Q03_nb_couples@\). Par linéarité, la seconde somme a pour espérance \(@Q03_nb_couples@ \times 1/@Q03_nb_couples@ = 1\).

Étape 4 : \(\mathbb{E}[N^2] = @Q03_e2@\) et \(\mathrm{Var}(N) = @Q03_e2@ - 1^2 = @Q03_exact@\). Le résultat vaut pour tout \(n \ge 2\) (leçon, outil 1, exemple 1).

Vérification (script) : énumération exacte pour \(n = 2\) à 7, et simulation pour \(n = 10\) : @Q03_mc:3@.""",
    },
    # ------------------------------------------------- Outil 2 : premier pas
    {
        "id": "S1M-04", "section": "math", "theme": "Marche aléatoire, durée moyenne", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""Une particule part de la position 2 sur les entiers relatifs. À chaque pas, indépendamment des pas précédents, elle se déplace de +1 avec probabilité 1/2 et de −1 avec probabilité 1/2. Elle s'arrête dès qu'elle atteint la position −3 ou la position 6.

Quelle est l'espérance du nombre de pas effectués jusqu'à l'arrêt ?""",
        "options": [
            {"id": "a", "texte": "@Q04_exact@", "explication": r"Oui : produit des distances du départ aux deux barrières, \(@Q04_dist_bas@ \times @Q04_dist_haut@\)."},
            {"id": "b", "texte": "@Q04_d_produit@", "explication": r"C'est \(2 \times 6\), la position de départ multipliée par la barrière haute. La formule multiplie les **distances** du départ aux deux barrières : \(@Q04_dist_bas@ \times @Q04_dist_haut@\)."},
            {"id": "c", "texte": "@Q04_d_sans_translation@", "explication": r"C'est \(2 \times (6 - 2)\) : la formule \(i\,(n - i)\) appliquée sans translation, comme si la barrière basse était en 0. En translatant de 3, les barrières deviennent 0 et @Q04_largeur@, et le départ @Q04_i_translate@."},
            {"id": "d", "texte": "@Q04_d_min@", "explication": r"C'est le nombre **minimal** de pas (@Q04_d_min@ pas vers le haut), pas une espérance : la particule fait des allers-retours et peut aussi finir en −3."},
        ],
        "solution": r"""Réponse : @Q04_exact@ pas.

Étape 1, translation : ajouter 3 à toutes les positions ne change ni les pas ni le moment de l'arrêt. La marche part alors de \(i = @Q04_i_translate@\) entre les barrières 0 et \(n = @Q04_largeur@\).

Étape 2 : pour la marche symétrique entre 0 et \(n\), l'espérance du temps d'arrêt partant de \(i\) est \(d(i) = i\,(n - i)\) (leçon, outil 2 : équation du premier pas \(d(i) = 1 + \tfrac12 d(i+1) + \tfrac12 d(i-1)\), \(d(0) = d(n) = 0\), finitude par blocs et unicité de la solution).

Étape 3 : \(d(@Q04_i_translate@) = @Q04_i_translate@ \times (@Q04_largeur@ - @Q04_i_translate@) = @Q04_exact@\). Forme directe sans translation : (départ − barrière basse) × (barrière haute − départ) \(= @Q04_dist_bas@ \times @Q04_dist_haut@\).

Vérification (script) : système linéaire résolu en rationnels, et simulation : @Q04_mc:2@.""",
    },
    {
        "id": "S1M-05", "section": "math", "theme": "Ruine du joueur, jeu biaisé", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Un joueur dispose de 3 euros. À chaque partie, indépendamment des parties précédentes, il gagne 1 euro avec probabilité 0,4 et perd 1 euro avec probabilité 0,6. Il s'arrête dès que sa fortune atteint 0 euro ou 6 euros.

Quelle est la probabilité qu'il atteigne 6 euros, arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q05_exact:3@", "explication": r"Oui : \(r = q/p = @Q05_r:1m@\) et, comme le départ est au milieu, \(h = 1/(1 + r^3)\)."},
            {"id": "b", "texte": "@Q05_d_sym:3@", "explication": r"C'est \(i/n = 3/6\), la formule du jeu équitable. Ici le jeu est défavorable (\(p < 1/2\)), donc la probabilité d'atteindre 6 doit être inférieure à 1/2."},
            {"id": "c", "texte": "@Q05_d_inverse:3@", "explication": r"C'est le résultat avec \(r = p/q\) au lieu de \(q/p\), c'est-à-dire la probabilité de finir à 0. Contrôle en 5 secondes : avec \(p < 1/2\), la réponse doit être sous \(i/n = 1/2\)."},
            {"id": "d", "texte": "@Q05_d_trois:3@", "explication": r"C'est \(0{,}4^3\), la probabilité de gagner les trois premières parties : un seul des chemins qui mènent à 6. Les chemins avec des allers-retours comptent aussi."},
        ],
        "solution": r"""Réponse : \(\tfrac{@Q05_exact_num@}{@Q05_exact_den@} \approx @Q05_exact:3m@\).

Notations : \(p = 0{,}4\) (gain), \(q = 0{,}6\) (perte), départ \(i = 3\), barrières 0 et \(n = 6\), \(r = q/p = @Q05_r:1m@\).

Étape 1 : comme \(p \ne 1/2\), la probabilité de finir en haut est \(h(i) = (1 - r^i)/(1 - r^n)\) (leçon, outil 2 : équation du premier pas \(h(i) = p\,h(i+1) + q\,h(i-1)\), \(h(0) = 0\), \(h(n) = 1\), solution unique).

Étape 2 : \(r^3 = \tfrac{@Q05_r3_num@}{@Q05_r3_den@}\) et \(r^6 = \tfrac{@Q05_r6_num@}{@Q05_r6_den@}\), donc \(h(3) = \dfrac{@Q05_un_moins_r3_num@/@Q05_un_moins_r3_den@}{@Q05_un_moins_r6_num@/@Q05_un_moins_r6_den@} = \tfrac{@Q05_exact_num@}{@Q05_exact_den@}\).

Raccourci quand le départ est au milieu (\(i = n/2\)) : \(1 - r^n = (1 - r^i)(1 + r^i)\), donc \(h = 1/(1 + r^3) = 1/(@Q05_un_plus_r3_num@/@Q05_un_plus_r3_den@) = \tfrac{@Q05_exact_num@}{@Q05_exact_den@}\).

Contrôle : \(p < 1/2\), et la réponse est bien sous \(i/n = 1/2\). Simulation (script) : @Q05_mc:3@.""",
    },
    {
        "id": "S1M-06", "section": "math", "theme": "Ruine du joueur, durée moyenne d'un jeu biaisé", "difficulte": 5, "temps_cible_s": 150,
        "enonce": r"""Un joueur dispose de 3 euros. À chaque partie, indépendamment des parties précédentes, il gagne 1 euro avec probabilité 0,4 et perd 1 euro avec probabilité 0,6. Il s'arrête dès que sa fortune atteint 0 euro ou 6 euros.

Quelle est l'espérance du nombre de parties jouées, arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q06_exact:3@", "explication": r"Oui : \(d = (n\,h - i)/(p - q)\) avec \(h = \tfrac{@Q05_exact_num@}{@Q05_exact_den@}\)."},
            {"id": "b", "texte": "@Q06_d_sym:3@", "explication": r"C'est \(i\,(n - i) = 3 \times 3\), la formule du jeu équitable. Elle n'est valable que pour \(p = 1/2\)."},
            {"id": "c", "texte": "@Q06_d_une_barriere:3@", "explication": r"C'est \(i/(q - p) = 3/@Q06_q_moins_p:1m@\), la durée moyenne jusqu'à la ruine **sans** barrière haute. La barrière en 6 arrête certaines parties plus tôt, ce qui raccourcit la durée moyenne (leçon, outil 2, cas limites)."},
            {"id": "d", "texte": "@Q06_d_signe:3@", "explication": r"Erreur de signe sur le terme de correction : \((i + n\,h)/(q - p)\) au lieu de \((i - n\,h)/(q - p)\). Ce résultat dépasse \(i/(q - p) = @Q06_d_une_barriere@\), la durée sans barrière haute, ce qui est impossible."},
        ],
        "solution": r"""Réponse : \(\tfrac{@Q06_exact_num@}{@Q06_exact_den@} \approx @Q06_exact:3m@\) parties.

Notations : \(p = 0{,}4\), \(q = 0{,}6\), \(i = 3\), \(n = 6\), \(r = q/p = @Q05_r:1m@\) ; \(h\) est la probabilité de finir à 6, \(d\) l'espérance du nombre de parties.

Étape 1 : \(h = 1/(1 + r^3) = \tfrac{@Q05_exact_num@}{@Q05_exact_den@}\) (départ au milieu, voir la question précédente ou la leçon, outil 2).

Étape 2, identité de Wald : la fortune finale \(Z_T\) vérifie \(\mathbb{E}[Z_T] - i = (p - q)\,\mathbb{E}[T]\), car chaque partie jouée rapporte \(p - q\) en moyenne et la décision de jouer la \(k\)-ième partie ne dépend que des parties précédentes. Comme \(\mathbb{E}[Z_T] = n\,h\) : \(d = (n\,h - i)/(p - q)\).

Étape 3 : \(n\,h = \tfrac{@Q06_nh_num@}{@Q06_nh_den@}\), \(n\,h - i = \tfrac{@Q06_nh_moins_i_num@}{@Q06_nh_moins_i_den@}\), \(p - q = @Q06_p_moins_q:1m@\), donc \(d = \tfrac{@Q06_exact_num@}{@Q06_exact_den@} \approx @Q06_exact:3m@\).

Contrôles : \(d\) est positif, inférieur à \(i/(q - p) = @Q06_d_une_barriere@\) (durée sans barrière haute) et différent de \(3 \times 3\) puisque le jeu n'est pas équitable. Simulation (script) : @Q06_mc:3@.""",
    },
    {
        "id": "S1M-07", "section": "math", "theme": "Premier pas, temps d'attente d'un motif", "difficulte": 3, "temps_cible_s": 120,
        "enonce": r"""On lance une pièce équilibrée de façon répétée, les lancers étant indépendants, jusqu'à obtenir pour la première fois deux piles consécutifs, c'est-à-dire deux lancers successifs qui donnent tous deux pile. On compte tous les lancers effectués, y compris les deux derniers.

Quelle est l'espérance du nombre de lancers ?""",
        "options": [
            {"id": "a", "texte": "@Q07_exact@", "explication": r"Oui : \(e_1 = @Q07_e1@\) depuis l'état « dernier lancer pile », \(e_0 = @Q07_exact@\) depuis le départ."},
            {"id": "b", "texte": "@Q07_d_inverse@", "explication": r"C'est \(1/\mathbb{P}(\text{PP}) = 1/(1/4)\), le temps d'attente moyen d'un succès de probabilité 1/4 si les essais étaient indépendants et disjoints. Ici, après un pile suivi d'un face, il faut tout recommencer, et les paires de lancers successifs se chevauchent."},
            {"id": "c", "texte": "@Q07_d_blocs@", "explication": r"C'est l'attente d'un bloc PP quand on découpe les lancers en blocs disjoints de deux : 4 blocs en moyenne, soit 8 lancers. Deux piles consécutifs peuvent aussi chevaucher deux blocs (F P puis P F), ce qui les fait arriver plus tôt : 8 n'est qu'un majorant."},
            {"id": "d", "texte": "@Q07_d_naif@", "explication": r"C'est 2 + 1 : attendre un pile (2 lancers en moyenne), puis un lancer de plus. Cela oublie qu'un face au lancer suivant oblige à tout recommencer."},
        ],
        "solution": r"""Réponse : @Q07_exact@ lancers.

Notations : deux états. État 0 : « rien d'utile » (au départ, ou dernier lancer face). État 1 : « dernier lancer pile ». \(e_0\) et \(e_1\) sont les espérances du nombre de lancers restants depuis chaque état.

Étape 1, finitude : en découpant les lancers en blocs disjoints de deux, chaque bloc vaut PP avec probabilité 1/4, indépendamment des autres ; l'argument des blocs de la leçon (outil 2, étape 1 de la preuve) majore l'espérance par \(2 \times 4 = @Q07_borne@\).

Étape 2, premier lancer depuis l'état 0 : pile mène à l'état 1, face laisse en état 0 : \(e_0 = 1 + \tfrac12 e_1 + \tfrac12 e_0\), donc \(e_0 = 2 + e_1\).

Étape 3, premier lancer depuis l'état 1 : pile termine, face ramène à l'état 0 : \(e_1 = 1 + \tfrac12 \times 0 + \tfrac12 e_0\).

Étape 4 : en substituant, \(e_1 = 1 + \tfrac12 (2 + e_1)\), donc \(e_1 = @Q07_e1@\) et \(e_0 = @Q07_exact@\). Le système a une solution unique, et les vraies espérances, finies, le vérifient : ce sont donc ces valeurs.

Comparaison utile : pour le motif pile puis face, l'attente moyenne n'est que de @Lb_pf_e0@ lancers (leçon, outil 2), car un pile raté n'oblige pas à recommencer. Simulation (script) : @Q07_mc:3@.""",
    },
    # ------------------------------------------------- Outil 3 : statistiques d'ordre
    {
        "id": "S1M-08", "section": "math", "theme": "Statistiques d'ordre", "difficulte": 2, "temps_cible_s": 60,
        "enonce": r"""Soient \(U_1, \dots, U_5\) cinq variables aléatoires indépendantes de loi uniforme sur \([0, 1]\). Quelle est l'espérance de la deuxième plus grande des cinq valeurs ?""",
        "options": [
            {"id": "a", "texte": "@Q08_exact_num@/@Q08_exact_den@", "explication": r"Oui : la deuxième plus grande est \(U_{(4)}\), d'espérance \(4/(5+1)\)."},
            {"id": "b", "texte": "@Q08_d_petit_num@/@Q08_d_petit_den@", "explication": r"C'est \(\mathbb{E}[U_{(2)}] = 2/6\), la deuxième plus **petite**. La deuxième plus grande est \(U_{(n-1)} = U_{(4)}\)."},
            {"id": "c", "texte": "@Q08_d_n_num@/@Q08_d_n_den@", "explication": r"Le dénominateur est \(n = 5\) au lieu de \(n + 1 = 6\) : 5 points découpent le segment en 6 morceaux de même longueur moyenne."},
            {"id": "d", "texte": "@Q08_d_max_num@/@Q08_d_max_den@", "explication": r"C'est l'espérance du maximum, \(5/6\). La deuxième plus grande se trouve en moyenne un morceau plus bas."},
        ],
        "solution": r"""Réponse : \(\tfrac{@Q08_exact_num@}{@Q08_exact_den@}\).

Étape 1 : en rangeant les valeurs, \(U_{(1)} < \dots < U_{(5)}\), la deuxième plus grande est \(U_{(4)}\).

Étape 2 : pour \(n\) uniformes indépendantes sur \([0, 1]\), \(\mathbb{E}[U_{(k)}] = k/(n+1)\) (leçon, outil 3 : formule de la queue, loi binomiale et intégrale bêta).

Étape 3 : \(\mathbb{E}[U_{(4)}] = 4/6 = \tfrac{@Q08_exact_num@}{@Q08_exact_den@}\). Contrôle de symétrie : la deuxième plus petite a pour espérance \(2/6\), et \(4/6 + 2/6 = 1\).

Simulation (script) : @Q08_mc:4@.""",
    },
    {
        "id": "S1M-09", "section": "math", "theme": "Statistiques d'ordre, étendue", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Soient \(V_1, \dots, V_6\) six variables aléatoires indépendantes de loi uniforme sur \([0, 2]\). Quelle est l'espérance de \(\max(V_1, \dots, V_6) - \min(V_1, \dots, V_6)\) ?""",
        "options": [
            {"id": "a", "texte": "@Q09_exact_num@/@Q09_exact_den@", "explication": r"Oui : \(2 \times (n - 1)/(n + 1)\) avec \(n = 6\)."},
            {"id": "b", "texte": "@Q09_d_echelle_num@/@Q09_d_echelle_den@", "explication": r"C'est l'étendue moyenne de 6 uniformes sur \([0, 1]\), \((n - 1)/(n + 1)\). Sur \([0, 2]\), toutes les longueurs sont multipliées par 2."},
            {"id": "c", "texte": "@Q09_d_max_num@/@Q09_d_max_den@", "explication": r"C'est l'espérance du maximum seul, \(2 \times 6/7\). Il faut encore soustraire celle du minimum, \(2 \times 1/7\)."},
            {"id": "d", "texte": "@Q09_d_n_num@/@Q09_d_n_den@", "explication": r"C'est \(2 \times 5/6\), avec le dénominateur \(n\) au lieu de \(n + 1\) : 6 points découpent l'intervalle en 7 morceaux."},
        ],
        "solution": r"""Réponse : \(\tfrac{@Q09_exact_num@}{@Q09_exact_den@}\).

Notations : \(U_j = V_j/2\), qui sont indépendantes et uniformes sur \([0, 1]\) ; \(n = 6\).

Étape 1, mise à l'échelle : \(V_j = 2\,U_j\), transformation croissante, donc \(\max_j V_j - \min_j V_j = 2\,(U_{(6)} - U_{(1)})\).

Étape 2 : \(\mathbb{E}[U_{(6)}] = 6/7\) et \(\mathbb{E}[U_{(1)}] = 1/7\) (leçon, outil 3), donc par linéarité \(\mathbb{E}[U_{(6)} - U_{(1)}] = 5/7\).

Étape 3 : \(\mathbb{E}[\max - \min] = 2 \times 5/7 = \tfrac{@Q09_exact_num@}{@Q09_exact_den@} \approx @Q09_exact:3m@\).

Simulation (script) : @Q09_mc:3@.""",
    },
    {
        "id": "S1M-10", "section": "math", "theme": "Statistiques d'ordre, loi jointe du minimum et du maximum", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""Soient \(U_1, \dots, U_4\) quatre variables aléatoires indépendantes de loi uniforme sur \([0, 1]\). Quelle est la probabilité que leur minimum soit supérieur ou égal à 0,2 **et** que leur maximum soit inférieur ou égal à 0,9, arrondie à quatre décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q10_exact:4@", "explication": r"Oui : les quatre valeurs doivent toutes tomber dans \([0{,}2 ; 0{,}9]\), de longueur \(@Q10_largeur:1m@\)."},
            {"id": "b", "texte": "@Q10_d_indep:4@", "explication": r"C'est \(\mathbb{P}(\max \le 0{,}9) \times \mathbb{P}(\min \ge 0{,}2) = 0{,}9^4 \times @Q10_q_min:1m@^4\) : ce produit suppose le minimum et le maximum indépendants, ce qu'ils ne sont pas. Le bon calcul porte sur l'événement joint."},
            {"id": "c", "texte": "@Q10_d_diff:4@", "explication": r"C'est \(0{,}9^4 - 0{,}2^4 = \mathbb{P}(0{,}2 < \max \le 0{,}9)\) : cet événement ne contraint que le maximum, et laisse le minimum descendre sous 0,2."},
            {"id": "d", "texte": "@Q10_d_un:4@", "explication": r"C'est la probabilité qu'**une** variable tombe dans \([0{,}2 ; 0{,}9]\). Les quatre doivent y tomber, indépendamment : il faut la puissance 4."},
        ],
        "solution": r"""Réponse : \(0{,}7^4 = @Q10_exact:4m@\).

Étape 1 : « minimum au moins 0,2 et maximum au plus 0,9 » équivaut à « chacune des quatre valeurs est dans \([0{,}2 ; 0{,}9]\) ».

Étape 2 : chaque \(U_j\) y tombe avec probabilité \(0{,}9 - 0{,}2 = @Q10_largeur:1m@\).

Étape 3 : par indépendance, \(\mathbb{P} = @Q10_largeur:1m@^4 = @Q10_exact:4m@\).

Piège visé : le minimum et le maximum ne sont pas indépendants ; on ne multiplie pas leurs probabilités, on traduit l'événement joint en une condition sur chaque variable (leçon, outil 3, piège 3). Simulation (script) : @Q10_mc:4@.""",
    },
    # ------------------------------------------------- Outil 4 : complémentaire, Boole, Bonferroni
    {
        "id": "S1M-11", "section": "math", "theme": "Tests multiples, complémentaire, approximation", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""Un chercheur teste 200 signaux, dont aucun n'a d'effet réel. Les 200 tests sont indépendants et chacun est réalisé au niveau 0,5 % : un test dont l'hypothèse nulle est vraie la rejette avec probabilité exactement 0,005.

Quelle est la probabilité qu'au moins un des 200 tests rejette son hypothèse nulle, arrondie à deux décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q11_exact:2@", "explication": r"Oui : \(1 - @Q11_q:3m@^{200}\), que l'on approche sans calculatrice par \(1 - e^{-1}\)."},
            {"id": "b", "texte": "@Q11_d_boole:2@", "explication": r"C'est \(200 \times 0{,}005\), l'espérance du nombre de rejets et la borne de Boole : un majorant, ici sans information, pas la probabilité. Aucun rejet reste possible, avec probabilité \(@Q11_q:3m@^{200} \approx @Q11_aucun:2m@\)."},
            {"id": "c", "texte": "@Q11_aucun:2@", "explication": r"C'est \(@Q11_q:3m@^{200}\), la probabilité qu'aucun test ne rejette : le complémentaire de l'événement demandé."},
            {"id": "d", "texte": "@Q11_d_un:3@", "explication": r"C'est le niveau d'un seul test. Avec 200 tests, les occasions de rejeter à tort se multiplient."},
        ],
        "solution": r"""Réponse : \(1 - @Q11_q:3m@^{200} \approx @Q11_exact:3m@\), soit @Q11_exact:2@ à deux décimales.

Étape 1, complémentaire : « au moins un rejet » est le contraire de « aucun rejet ».

Étape 2, indépendance : chaque test ne rejette pas avec probabilité @Q11_q:3@, donc aucun ne rejette avec probabilité \(@Q11_q:3m@^{200}\).

Étape 3, calcul rapide : \(m = 200\), \(p = 0{,}005\), \(mp = @Q11_mp@\). L'encadrement de la leçon (outil 4, valable pour \(p \le 1/2\)) donne \(1 - e^{-1} \le 1 - @Q11_q:3m@^{200} \le 1 - e^{-1} + m p^2 e^{-1}\), avec \(e^{-1} \approx @Q11_e_moins_un:3m@\) et \(m p^2 = @Q11_mp2:3m@\). La vraie valeur est donc entre \(@Q11_poisson:4m@\) et \(@Q11_haut:4m@\), et s'arrondit à @Q11_exact:2@ dans tous les cas.

Simulation (script) : @Q11_mc:3@.""",
    },
    {
        "id": "S1M-12", "section": "math", "theme": "Tests multiples, Bonferroni et dépendance", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""On teste 25 hypothèses nulles, toutes vraies. Chaque test \(j\) fournit une p-value \(P_j\) de loi uniforme sur \([0, 1]\), et le test \(j\) au niveau \(t\) rejette son hypothèse nulle si \(P_j \le t\) ; il la rejette donc à tort avec probabilité exactement \(t\). La dépendance entre les 25 p-values est inconnue et peut être quelconque.

On veut un niveau commun \(t\), le même pour les 25 tests, qui garantisse, **quelle que soit cette dépendance**, que la probabilité d'au moins un rejet est au plus 0,05. Quelle affirmation est vraie ?""",
        "options": [
            {"id": "a", "texte": r"\(t = @Q12_bonf:3m@\) le garantit ; \(t = 1 - @Q12_q:2m@^{1/25} \approx @Q12_sidak:5m@\) ne le garantit pas.", "explication": r"Oui : Boole garantit le premier niveau ; des rejets disjoints mettent le second en défaut."},
            {"id": "b", "texte": r"Les deux niveaux \(@Q12_bonf:3m@\) et \(1 - @Q12_q:2m@^{1/25} \approx @Q12_sidak:5m@\) le garantissent ; le second est préférable car moins strict.", "explication": r"Le niveau de Šidák donne exactement 0,05 sous indépendance, mais pas pour toute dépendance. Contre-exemple : \(U\) uniforme sur \([0, 1[\) et \(P_j\) la partie fractionnaire de \(U - (j - 1)\,t\). Chaque \(P_j\) est uniforme, le test \(j\) rejette exactement quand \(U\) tombe dans \([(j-1)\,t, j\,t]\), et ces intervalles ne se chevauchent qu'en des points, de probabilité nulle. La probabilité d'au moins un rejet vaut alors \(25\,t \approx @Q12_fwer_sidak_disjoint:4m@\), plus que 0,05."},
            {"id": "c", "texte": "Aucun niveau commun ne peut le garantir sans connaître la dépendance.", "explication": r"La borne de Boole donne une garantie sans aucune hypothèse de dépendance : la probabilité d'au moins un rejet est au plus \(25\,t\), qui vaut 0,05 pour \(t = @Q12_bonf:3m@\)."},
            {"id": "d", "texte": r"\(t = 0{,}05\) le garantit, puisque chaque test pris seul ne rejette à tort qu'avec probabilité 0,05.", "explication": r"Le niveau de chaque test ne borne pas la probabilité d'au moins un rejet : par exemple, sous indépendance, elle vaut \(1 - @Q12_q:2m@^{25} \approx @Q12_fwer_005_indep:3m@\)."},
        ],
        "solution": r"""Réponse : le niveau de Bonferroni, \(0{,}05/25 = @Q12_bonf:3m@\), le garantit ; le niveau de Šidák, \(1 - @Q12_q:2m@^{1/25} \approx @Q12_sidak:5m@\), ne le garantit pas.

Étape 1, Bonferroni : par la borne de Boole, qui ne demande aucune indépendance, \(\mathbb{P}(\text{au moins un rejet}) \le \sum_{j=1}^{25} \mathbb{P}(P_j \le t) = 25\,t\), égal à 0,05 pour \(t = @Q12_bonf:3m@\).

Étape 2, une dépendance défavorable : soit \(U\) uniforme sur \([0, 1[\) et \(P_j\) la partie fractionnaire de \(U - (j-1)\,t\). Un décalage modulo 1 d'une uniforme est uniforme, donc chaque test est valide. Pour \(25\,t \le 1\), le test \(j\) rejette exactement quand \(U\) est dans \([(j-1)\,t, j\,t]\) : à des points près, de probabilité nulle, les rejets sont disjoints, et \(\mathbb{P}(\text{au moins un rejet}) = 25\,t\).

Étape 3 : avec le niveau de Šidák, cela donne \(25\,t \approx @Q12_fwer_sidak_disjoint:4m@ > 0{,}05\). Plus généralement, tout niveau commun \(t > @Q12_bonf:3m@\) échoue pour cette dépendance : \(@Q12_bonf:3m@\) est le plus grand niveau commun garanti.

Lien avec la leçon (outil 4, cas limites) : le niveau de Šidák rend la probabilité exactement égale à 0,05 sous indépendance, mais il dépasse \(\alpha/m\) et n'est garanti que sous cette hypothèse. Simulation de la construction (script) : @Q12_disjoint_sidak_mc:4@.""",
    },
]
