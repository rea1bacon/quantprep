"""Séance de mathématiques 2 : leçon express sur le conditionnement et les lois usuelles, puis 8 questions à choix multiple.

Thèmes : Bayes, variance totale et sommes aléatoires, minimum d'exponentielles et absence de mémoire, processus de
Poisson, collectionneur de coupons, inclusion-exclusion et dérangements. La séance 1 (indicatrices, premier pas,
statistiques d'ordre, Boole) n'est pas reprise.
Toutes les valeurs : verif/seance_math_2.py. Dans chaque question, l'option "a" est la bonne.
"""
import lecons_math_2
import questions_diagnostic as diag

ID = "seance-math-2"
TITRE = "Séance de mathématiques 2 : conditionnement et lois usuelles"
VERIF = "verif/seance_math_2.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [{"id": "lecon", "nom": "Leçon", "budget_min": 20}, {"id": "math", "nom": "Mathématiques", "budget_min": 20}]
PRESENTATION = (
    "D'abord une leçon express d'environ 20 minutes : conditionnement (probabilités totales, Bayes, espérance et "
    "variance totales), lois usuelles et leurs moments, absence de mémoire et minimum d'exponentielles, processus de "
    "Poisson, collectionneur de coupons, dénombrement et inclusion-exclusion, avec cinq points de contrôle corrigés "
    "immédiatement. Les outils de la séance 1 sont supposés connus et seulement cités.\n\n"
    "Ensuite 8 questions à choix multiple, dont la moitié de difficulté 4 ou 5. Objectif de vitesse : chaque question "
    "se résout dans son temps cible avec les résultats de la leçon (tableau « Réflexes en 30 secondes »). Au-delà du "
    "double du temps cible affiché, choisis « Je ne sais pas » et passe à la suivante : c'est la règle d'abandon "
    "à suivre dans un test chronométré."
)
AVERTISSEMENTS = [
    "Les questions de mathématiques sont présentées ici en choix multiple à une seule bonne réponse. Cherche la réponse avant de "
    "lire les options, pour t'entraîner aussi au format à réponse numérique.",
    "Ici le retour en arrière est permis et il n'y a pas de pénalité : ce sont les règles de cette plateforme, pas "
    "celles de tout test en ligne.",
]

LECONS = [lecons_math_2.LECON_MATH_2]
CODES = []

QUESTIONS = [
    # ------------------------------------------------- Partie 1 : conditionnement
    {
        "id": "M2-01", "section": "math", "theme": "Formule de Bayes", "difficulte": 2, "temps_cible_s": 75,
        "enonce": r"""Un chercheur teste un grand nombre de signaux. Avant tout test, chaque signal a un effet réel avec probabilité 0,05. Le test utilisé rejette l'hypothèse « pas d'effet » avec probabilité 0,8 quand le signal a un effet réel, et avec probabilité 0,05 quand il n'en a pas.

On tire un signal au hasard, on le teste, et le test rejette. Quelle est la probabilité que ce signal ait un effet réel, arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q01_exact:3@", "explication": r"Oui : \(@Q01_joint:2m@ / (@Q01_joint:2m@ + @Q01_faux:4m@) = \tfrac{@Q01_exact_num@}{@Q01_exact_den@}\)."},
            {"id": "b", "texte": "@Q01_d_inverse:3@", "explication": r"C'est \(\mathbb{P}(\text{rejet} \mid \text{effet})\), la puissance du test : le conditionnement a été inversé. La question demande \(\mathbb{P}(\text{effet} \mid \text{rejet})\)."},
            {"id": "c", "texte": "@Q01_d_joint:3@", "explication": r"C'est \(\mathbb{P}(\text{effet et rejet}) = 0{,}05 \times 0{,}8\), une probabilité jointe. Il faut encore la diviser par \(\mathbb{P}(\text{rejet}) = @Q01_total:4m@\)."},
            {"id": "d", "texte": "@Q01_d_sans_prior:3@", "explication": r"C'est \(0{,}8/(0{,}8 + 0{,}05)\) : la formule de Bayes avec des a priori égaux, comme si un signal sur deux avait un effet. L'a priori 0,05 a été oublié ; or les signaux sans effet sont bien plus nombreux, et leurs faux rejets pèsent lourd."},
        ],
        "solution": r"""Réponse : \(\tfrac{@Q01_exact_num@}{@Q01_exact_den@} \approx @Q01_exact:3m@\).

Notations : \(E\) est l'événement « le signal a un effet réel », \(R\) l'événement « le test rejette ».

Étape 1, vrais rejets : \(\mathbb{P}(E \cap R) = \mathbb{P}(R \mid E)\,\mathbb{P}(E) = 0{,}8 \times 0{,}05 = @Q01_joint:2m@\).

Étape 2, faux rejets : \(\mathbb{P}(E^c \cap R) = 0{,}05 \times 0{,}95 = @Q01_faux:4m@\).

Étape 3, probabilités totales (leçon, (C1)) : \(\mathbb{P}(R) = @Q01_joint:2m@ + @Q01_faux:4m@ = @Q01_total:4m@\).

Étape 4, Bayes (leçon, (C2)) : \(\mathbb{P}(E \mid R) = @Q01_joint:2m@ / @Q01_total:4m@ = \tfrac{@Q01_exact_num@}{@Q01_exact_den@} \approx @Q01_exact:3m@\). Moins d'un rejet sur deux correspond à un effet réel, malgré une puissance de 0,8 et un niveau de 0,05.

Contrôle de vraisemblance : le rejet est plus probable avec effet (0,8) que sans (0,05), donc l'a posteriori doit dépasser l'a priori 0,05 ; c'est le cas. Simulation (script) : @Q01_mc:3@.""",
    },
    {
        "id": "M2-02", "section": "math", "theme": "Variance totale, somme d'un nombre aléatoire de termes", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""Le nombre \(N\) de transactions d'une journée suit la loi binomiale de paramètres 10 et 0,4. Les gains des transactions, \(X_1, X_2, \dots\) (en euros), ont tous la même loi, d'espérance 2 et de variance 9, et sont indépendants entre eux et indépendants de \(N\). Le gain de la journée est \(S = X_1 + \dots + X_N\), avec \(S = 0\) si \(N = 0\).

Que vaut \(\mathrm{Var}(S)\), en euros carrés ?""",
        "options": [
            {"id": "a", "texte": "@Q02_exact:1@", "explication": r"Oui : \(\mathbb{E}[N]\,\sigma^2 + \mathrm{Var}(N)\,\mu^2 = @Q02_intra@ + @Q02_inter:1m@\)."},
            {"id": "b", "texte": "@Q02_d_intra@", "explication": r"C'est \(\mathbb{E}[N]\,\sigma^2 = \mathbb{E}[\mathrm{Var}(S \mid N)]\) seul. Le nombre de transactions varie lui aussi : le terme \(\mathrm{Var}(\mathbb{E}[S \mid N]) = \mathrm{Var}(N)\,\mu^2\) manque."},
            {"id": "c", "texte": "@Q02_d_poisson@", "explication": r"C'est \(\mathbb{E}[N]\,\mathbb{E}[X^2] = \mathbb{E}[N]\,(\sigma^2 + \mu^2)\), la formule de la somme à nombre de termes de Poisson. Elle suppose \(\mathrm{Var}(N) = \mathbb{E}[N]\) ; pour une binomiale, \(\mathrm{Var}(N) = @Q02_vn:1@\), moins que \(\mathbb{E}[N] = @Q02_en@\)."},
            {"id": "d", "texte": "@Q02_d_mu:1@", "explication": r"C'est \(\mathbb{E}[N]\,\sigma^2 + \mathrm{Var}(N)\,\mu\) : l'espérance \(\mu\) n'a pas été mise au carré. Or \(\mathrm{Var}(N\mu) = \mu^2\,\mathrm{Var}(N)\)."},
        ],
        "solution": r"""Réponse : \(\mathrm{Var}(S) = @Q02_exact:1m@\) euros carrés.

Notations : \(\mu = 2\) et \(\sigma^2 = 9\) sont l'espérance et la variance de chaque \(X_i\).

Étape 1, moments de \(N\) (leçon, tableau) : \(\mathbb{E}[N] = 10 \times 0{,}4 = @Q02_en@\) et \(\mathrm{Var}(N) = 10 \times 0{,}4 \times 0{,}6 = @Q02_vn:1m@\).

Étape 2, conditionnement : les \(X_i\) étant indépendants de \(N\), sachant \(N = n\), \(S\) est une somme de \(n\) variables indépendantes : \(\mathbb{E}[S \mid N] = N\mu\) et \(\mathrm{Var}(S \mid N) = N\sigma^2\).

Étape 3, variance totale (leçon, (C4)) : \(\mathrm{Var}(S) = \mathbb{E}[N\sigma^2] + \mathrm{Var}(N\mu) = @Q02_en@ \times 9 + @Q02_vn:1m@ \times 2^2 = @Q02_intra@ + @Q02_inter:1m@ = @Q02_exact:1m@\).

Contrôle exact indépendant (script) : somme sur les 11 valeurs de \(N\) de \(\mathbb{P}(N = n)\,(n\sigma^2 + n^2\mu^2)\), moins \(\mathbb{E}[S]^2 = @Q02_es@^2\). Simulation (script) : @Q02_mc:2@.""",
    },
    # ------------------------------------------------- Partie 3 : exponentielles
    {
        "id": "M2-03", "section": "math", "theme": "Minimum d'exponentielles, absence de mémoire", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Trois composants A, B et C fonctionnent à partir de l'instant 0 et ne sont jamais remplacés. Leurs durées de vie sont indépendantes et suivent des lois exponentielles d'espérances respectives 1 an, 2 ans et 4 ans.

Quelle est la probabilité que A soit le dernier des trois à tomber en panne ?""",
        "options": [
            {"id": "a", "texte": "@Q03_exact_num@/@Q03_exact_den@", "explication": r"Oui : \(\tfrac{@Q03_pb_num@}{@Q03_pb_den@} \times \tfrac{@Q03_c_avant_a_num@}{@Q03_c_avant_a_den@} + \tfrac{@Q03_pc_num@}{@Q03_pc_den@} \times \tfrac{@Q03_b_avant_a_num@}{@Q03_b_avant_a_den@}\), en relançant la course après la première panne."},
            {"id": "b", "texte": "@Q03_d_premier_num@/@Q03_d_premier_den@", "explication": r"C'est \(\mathbb{P}(\text{A tombe en panne le premier}) = 1/(1 + \tfrac12 + \tfrac14)\). A a l'intensité la plus forte : il est le plus souvent le **premier** à tomber en panne, et rarement le dernier."},
            {"id": "c", "texte": "@Q03_d_moy_num@/@Q03_d_moy_den@", "explication": r"C'est \(1/(1 + 2 + 4)\) : la part de l'espérance de A dans la somme des espérances. Les chances d'être premier sont proportionnelles aux intensités (leçon, (A4)), et aucune règle de proportionnalité ne donne directement la chance d'être dernier."},
            {"id": "d", "texte": "@Q03_d_prod_num@/@Q03_d_prod_den@", "explication": r"C'est \(\mathbb{P}(A > B) \times \mathbb{P}(A > C) = \tfrac{@Q03_b_avant_a_num@}{@Q03_b_avant_a_den@} \times \tfrac{@Q03_c_avant_a_num@}{@Q03_c_avant_a_den@}\) : ce produit suppose les deux événements indépendants. Ils sont positivement liés (tous deux favorisés quand A dure longtemps), et la vraie probabilité de l'intersection est plus grande."},
        ],
        "solution": r"""Réponse : \(\tfrac{@Q03_exact_num@}{@Q03_exact_den@} \approx @Q03_exact:3m@\).

Notations : les intensités (inverses des espérances) sont 1, 1/2 et 1/4 par an pour A, B et C ; leur somme est \(\lambda = \tfrac{@Q03_tot_num@}{@Q03_tot_den@}\).

Étape 1, découpage : A est le dernier si et seulement si la première panne est celle de B ou de C, puis l'autre de ces deux tombe en panne avant A.

Étape 2, première panne (leçon, (A4)) : B est le premier avec probabilité \(\tfrac{1/2}{\lambda} = \tfrac{@Q03_pb_num@}{@Q03_pb_den@}\), C avec probabilité \(\tfrac{1/4}{\lambda} = \tfrac{@Q03_pc_num@}{@Q03_pc_den@}\).

Étape 3, la course recommence (leçon, (A5)) : après la panne de B, les durées restantes de A et C sont des exponentielles indépendantes d'intensités 1 et 1/4, et C tombe avant A avec probabilité \(\tfrac{1/4}{1 + 1/4} = \tfrac{@Q03_c_avant_a_num@}{@Q03_c_avant_a_den@}\). Après la panne de C, B tombe avant A avec probabilité \(\tfrac{1/2}{1 + 1/2} = \tfrac{@Q03_b_avant_a_num@}{@Q03_b_avant_a_den@}\).

Étape 4 : \(\tfrac{@Q03_pb_num@}{@Q03_pb_den@} \times \tfrac{@Q03_c_avant_a_num@}{@Q03_c_avant_a_den@} + \tfrac{@Q03_pc_num@}{@Q03_pc_den@} \times \tfrac{@Q03_b_avant_a_num@}{@Q03_b_avant_a_den@} = \tfrac{@Q03_t1_num@}{@Q03_t1_den@} + \tfrac{@Q03_t2_num@}{@Q03_t2_den@} = \tfrac{@Q03_exact_num@}{@Q03_exact_den@}\).

Seconde méthode, par inclusion-exclusion (leçon, partie 6) : \(\mathbb{P}(A > B, A > C) = 1 - \mathbb{P}(A < B) - \mathbb{P}(A < C) + \mathbb{P}(A < B, A < C) = 1 - \tfrac{@Q03_ab_num@}{@Q03_ab_den@} - \tfrac{@Q03_ac_num@}{@Q03_ac_den@} + \tfrac{@Q03_a_premier_num@}{@Q03_a_premier_den@}\), le dernier terme étant la probabilité que A soit le premier. On retrouve \(\tfrac{@Q03_exact_num@}{@Q03_exact_den@}\).

Vérifications (script) : intégrale exacte avec sympy, et simulation : @Q03_mc:4@.""",
    },
    # ------------------------------------------------- Partie 4 : processus de Poisson
    {
        "id": "M2-04", "section": "math", "theme": "Processus de Poisson, superposition", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""Les ordres d'achat arrivent selon un processus de Poisson d'intensité 2 par minute, et les ordres de vente selon un processus de Poisson d'intensité 1 par minute ; les deux processus sont indépendants.

Quelle est la probabilité qu'au moins 3 ordres, achats et ventes confondus, arrivent pendant une fenêtre donnée de 90 secondes ? Arrondir à trois décimales.""",
        "options": [
            {"id": "a", "texte": "@Q04_exact:3@", "explication": r"Oui : le total suit \(\mathcal{P}(@Q04_mu:1m@)\), et \(1 - e^{-@Q04_mu:1m@}(1 + @Q04_mu:1m@ + @Q04_mu:1m@^2/2)\)."},
            {"id": "b", "texte": "@Q04_d_achats:3@", "explication": r"C'est le calcul avec les seuls achats, de paramètre \(2 \times 1{,}5 = @Q04_mu_achats@\). Les ventes s'ajoutent : par superposition, l'intensité totale est \(2 + 1\) par minute."},
            {"id": "c", "texte": "@Q04_d_egal:3@", "explication": r"C'est \(\mathbb{P}(N = 3)\), la probabilité d'exactement 3 ordres. « Au moins 3 » ajoute 4, 5, 6 ordres et plus : on passe par le complémentaire \(1 - \mathbb{P}(N \le 2)\)."},
            {"id": "d", "texte": "@Q04_d_strict:3@", "explication": r"C'est \(1 - \mathbb{P}(N \le 3) = \mathbb{P}(N \ge 4)\) : le terme \(k = 3\) a été retiré à tort. Le complémentaire de « au moins 3 » est « au plus 2 »."},
        ],
        "solution": r"""Réponse : \(1 - e^{-@Q04_mu:1m@}\big(1 + @Q04_mu:1m@ + @Q04_mu:1m@^2/2\big) \approx @Q04_exact:3m@\).

Étape 1, superposition (leçon, (P3)) : achats et ventes forment ensemble un processus de Poisson d'intensité \(2 + 1 = 3\) par minute.

Étape 2, unités : 90 secondes font 1,5 minute ; le nombre total \(N\) d'ordres dans la fenêtre suit \(\mathcal{P}(3 \times 1{,}5) = \mathcal{P}(@Q04_mu:1m@)\) (définition, propriété 2).

Étape 3, complémentaire : \(\mathbb{P}(N \ge 3) = 1 - \mathbb{P}(N \le 2) = 1 - e^{-@Q04_mu:1m@}\big(1 + @Q04_mu:1m@ + @Q04_mu:1m@^2/2\big)\), avec \(1 + @Q04_mu:1m@ + @Q04_mu:1m@^2/2 = @Q04_somme:3m@\) et \(e^{-@Q04_mu:1m@} \approx @Q04_e:5m@\), d'où \(@Q04_exact:3m@\).

Simulation (script), deux flux simulés séparément puis additionnés : @Q04_mc:3@.""",
    },
    # ------------------------------------------------- Partie 5 : coupons
    {
        "id": "M2-05", "section": "math", "theme": "Collectionneur de coupons, collection partielle", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""Un indice contient 10 actifs. À chaque tirage, on choisit un actif uniformément au hasard parmi les 10, indépendamment des tirages précédents ; un même actif peut donc être tiré plusieurs fois.

Quelle est l'espérance du nombre de tirages nécessaires pour avoir vu 5 actifs distincts, arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q05_exact:3@", "explication": r"Oui : \(10/10 + 10/9 + 10/8 + 10/7 + 10/6\), les cinq premiers termes du collectionneur de coupons."},
            {"id": "b", "texte": "@Q05_d_inverse:3@", "explication": r"C'est \(10/1 + 10/2 + \dots + 10/5\), les cinq **derniers** termes, ceux des types les plus difficiles à obtenir. Pour les 5 premiers types, la probabilité de nouveauté part de 10/10, pas de 1/10."},
            {"id": "c", "texte": "@Q05_d_min@", "explication": r"C'est le nombre **minimal** de tirages, atteint seulement si les 5 premiers tirages sont tous distincts. L'espérance compte aussi les répétitions."},
            {"id": "d", "texte": "@Q05_d_moitie:3@", "explication": r"C'est la moitié de \(10\,H_{10}\), l'espérance pour voir les 10 actifs. Les termes ne sont pas égaux : les derniers types coûtent beaucoup plus cher que les premiers, donc la moitié des types coûte bien moins que la moitié du total."},
        ],
        "solution": r"""Réponse : \(\tfrac{10}{10} + \tfrac{10}{9} + \tfrac{10}{8} + \tfrac{10}{7} + \tfrac{10}{6} = \tfrac{@Q05_exact_num@}{@Q05_exact_den@} \approx @Q05_exact:3m@\).

Étape 1, découpage (leçon, partie 5) : \(C_5 = G_1 + \dots + G_5\), où \(G_k\) est le nombre de tirages entre l'apparition du \((k-1)\)-ième actif nouveau (exclu) et celle du \(k\)-ième (inclus).

Étape 2 : après avoir vu \(k - 1\) actifs, chaque tirage est nouveau avec probabilité \((10 - k + 1)/10\), indépendamment du passé, donc \(G_k\) est géométrique d'espérance \(10/(11 - k)\).

Étape 3, linéarité : \(\mathbb{E}[C_5] = \sum_{k=1}^{5} 10/(11 - k)\) ; le terme \(k = 5\) vaut \(10/6 \approx @Q05_g5:3m@\), le plus grand des cinq.

Simulation (script) : @Q05_mc:3@.""",
    },
    # ------------------------------------------------- Partie 6 : inclusion-exclusion
    {
        "id": "M2-06", "section": "math", "theme": "Inclusion-exclusion, surjections", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""On lance 8 fois un dé équilibré à six faces ; les lancers sont indépendants.

Quelle est la probabilité que chacune des six faces apparaisse au moins une fois, arrondie à quatre décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q06_exact:4@", "explication": r"Oui : \(\sum_{j=0}^{6} (-1)^j \binom{6}{j} (6 - j)^8 / 6^8 = @Q06_fav@/@Q06_tot@\)."},
            {"id": "b", "texte": "@Q06_d_six:4@", "explication": r"C'est \(6!/6^6\), la probabilité de voir les six faces en **6** lancers (tous distincts). Avec 8 lancers, il y a deux lancers de marge, ce qui rend l'événement bien plus probable."},
            {"id": "c", "texte": "@Q06_d_tronque:4@", "explication": r"C'est l'inclusion-exclusion arrêtée après les paires de faces absentes : \(1 - 6\,(5/6)^8 + 15\,(4/6)^8\). Les termes suivants ne sont pas négligeables : le seul terme \(20\,(3/6)^8 \approx @Q06_s3:4m@\) pèse plus de la moitié du résultat."},
            {"id": "d", "texte": "@Q06_d_surcompte:4@", "explication": r"C'est \(\binom{8}{6}\,6!\,6^2/6^8\) : choisir 6 lancers qui montrent les six faces, puis laisser les 2 autres libres. Une suite favorable est alors comptée plusieurs fois (3 ou 4 fois selon la répartition des répétitions) : c'est un surcompte."},
        ],
        "solution": r"""Réponse : \(\tfrac{@Q06_fav@}{@Q06_tot@} \approx @Q06_exact:4m@\).

Notations : pour chaque face \(c\), \(A_c\) est l'événement « la face \(c\) n'apparaît dans aucun des 8 lancers ». On cherche \(1 - \mathbb{P}(A_1 \cup \dots \cup A_6)\).

Étape 1 : une intersection de \(j\) événements \(A_c\) signifie que les 8 lancers évitent \(j\) faces données : par indépendance, probabilité \(((6 - j)/6)^8\). Elle ne dépend que de \(j\), et il y a \(\binom{6}{j}\) telles intersections.

Étape 2, inclusion-exclusion (leçon, partie 6) : \(\mathbb{P}(\text{toutes les faces}) = \sum_{j=0}^{6} (-1)^j \binom{6}{j} \big(\tfrac{6 - j}{6}\big)^8\).

Étape 3, calcul : les termes d'ordre 1, 2 et 3 valent \(6\,(5/6)^8 \approx @Q06_s1:4m@\), \(15\,(4/6)^8 \approx @Q06_s2:4m@\) et \(20\,(3/6)^8 \approx @Q06_s3:4m@\) : le premier dépasse 1 à lui seul, et le troisième pèse encore plus de la moitié du résultat. Il faut aller jusqu'au bout. En nombres de suites : \(6^8 - 6 \times 5^8 + 15 \times 4^8 - 20 \times 3^8 + 15 \times 2^8 - 6 \times 1^8 + 0 = @Q06_fav@\), sur \(6^8 = @Q06_tot@\) suites équiprobables.

Vérifications (script) : énumération des \(6^8\) suites, nombre de Stirling \(6!\,S(8, 6)\) avec sympy, et simulation : @Q06_mc:4@.""",
    },
    {
        "id": "M2-07", "section": "math", "theme": "Dérangements, exactement j points fixes", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""On place au hasard 6 lettres adressées dans 6 enveloppes, une lettre par enveloppe, les \(6!\) répartitions étant équiprobables. Chaque lettre a exactement une bonne enveloppe, et chaque enveloppe exactement une bonne lettre.

Quelle est la probabilité qu'exactement 2 lettres soient dans leur bonne enveloppe, arrondie à quatre décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q07_exact:4@", "explication": r"Oui : \(\binom{6}{2}\,d_4/6! = @Q07_c62@ \times @Q07_d4@/@Q07_tot@\)."},
            {"id": "b", "texte": "@Q07_d_binom:4@", "explication": r"C'est \(\binom{6}{2}\,(1/6)^2\,(5/6)^4\), une loi binomiale : elle traite les 6 lettres comme indépendantes. Or une permutation place les lettres sans remise : savoir que 4 lettres sont mal placées change les chances des autres."},
            {"id": "c", "texte": "@Q07_d_poisson:4@", "explication": r"C'est \(e^{-1}/2!\), la limite de Poisson de paramètre 1 quand le nombre de lettres tend vers l'infini. Elle approche la réponse, mais la question demande la valeur exacte pour 6 lettres."},
            {"id": "d", "texte": "@Q07_d_paires:4@", "explication": r"C'est \(\binom{6}{2}\,4!/6!\) : on a fixé 2 lettres bien placées et laissé les 4 autres libres. Cela compte « au moins ces 2 », et chaque répartition avec \(j\) lettres bien placées est comptée \(\binom{j}{2}\) fois : on obtient l'espérance du nombre de paires de lettres bien placées, pas une probabilité."},
        ],
        "solution": r"""Réponse : \(\binom{6}{2}\,d_4/6! = @Q07_fav@/@Q07_tot@ = @Q07_exact:4m@\).

Étape 1, choisir les lettres bien placées : \(\binom{6}{2} = @Q07_c62@\) façons.

Étape 2, les 4 autres lettres doivent toutes être mal placées : c'est un dérangement de 4 éléments, \(d_4 = 4!\,(1 - 1 + 1/2! - 1/3! + 1/4!) = @Q07_d4@\) façons (leçon, partie 6, exemple 2). Sans cette condition, une troisième lettre pourrait être bien placée, et l'on compterait « au moins 2 ».

Étape 3 : \(@Q07_c62@ \times @Q07_d4@ = @Q07_fav@\) répartitions favorables sur \(6! = @Q07_tot@\), soit \(\tfrac{@Q07_exact_f_num@}{@Q07_exact_f_den@} = @Q07_exact:4m@\).

Vérifications (script) : énumération des 720 permutations, et simulation : @Q07_mc:4@.""",
    },
    # ------------------------------------------------- Synthèse : loi conditionnelle et variance totale
    {
        "id": "M2-08", "section": "math", "theme": "Variance totale avec un nombre de termes dépendant des valeurs", "difficulte": 5, "temps_cible_s": 150,
        "enonce": r"""On lance un dé équilibré à six faces de façon répétée, les lancers étant indépendants, jusqu'à obtenir un 6 pour la première fois. Soit \(S\) la somme de toutes les faces obtenues, y compris le 6 final. Par exemple, les lancers 2, 5, 6 donnent \(S = @Q08_ex@\).

Que vaut \(\mathrm{Var}(S)\) ?""",
        "options": [
            {"id": "a", "texte": "@Q08_exact@", "explication": r"Oui : \(\mathbb{E}[\mathrm{Var}(S \mid N)] + \mathrm{Var}(\mathbb{E}[S \mid N]) = @Q08_intra@ + @Q08_inter@\), avec la loi des faces sachant \(N\)."},
            {"id": "b", "texte": "@Q08_d_inter@", "explication": r"C'est \(\mathrm{Var}(\mathbb{E}[S \mid N]) = 9\,\mathrm{Var}(N)\) seul : il manque \(\mathbb{E}[\mathrm{Var}(S \mid N)]\), la dispersion des faces 1 à 5 quand le nombre de lancers est fixé."},
            {"id": "c", "texte": "@Q08_d_indep@", "explication": r"C'est la formule de la somme aléatoire \(\mathbb{E}[N]\,\mathrm{Var}(X) + \mathrm{Var}(N)\,\mathbb{E}[X]^2\) avec les moments d'un dé, \(\tfrac{@Q08_var_d6_num@}{@Q08_var_d6_den@}\) et \(@Q08_moy_d6:1m@\). Elle suppose les faces indépendantes de \(N\), ce qui est faux ici : \(N\) est le rang du premier 6, donc les faces qui le précèdent valent 1 à 5 et la dernière vaut 6."},
            {"id": "d", "texte": "@Q08_d_carre@", "explication": r"C'est \(@Q08_intra@ + 3\,\mathrm{Var}(N)\) : la variance de \(3N + 3\) a été prise égale à \(3\,\mathrm{Var}(N)\). Un facteur 3 multiplie la variance par \(3^2\)."},
        ],
        "solution": r"""Réponse : \(\mathrm{Var}(S) = @Q08_exact@\).

Notations : \(N\) est le rang du premier 6 ; il suit la loi géométrique de paramètre 1/6, d'où \(\mathbb{E}[N] = @Lc_geo_e@\) et \(\mathrm{Var}(N) = @Lc_geo_var@\) (leçon, partie 2, exemple). \(D_1, D_2, \dots\) sont les faces obtenues.

Étape 1, loi des faces sachant \(N = n\) : pour des faces \(x_1, \dots, x_{n-1}\) entre 1 et 5, \(\mathbb{P}(D_1 = x_1, \dots, D_{n-1} = x_{n-1}, D_n = 6) = (1/6)^n\), et \(\mathbb{P}(N = n) = (5/6)^{n-1}(1/6)\). Le quotient vaut \((1/5)^{n-1}\) : sachant \(N = n\), les \(n - 1\) premières faces sont indépendantes et uniformes sur \(\{1, \dots, 5\}\), et la dernière vaut 6. Les faces ne sont donc **pas** indépendantes de \(N\), et la formule de la somme aléatoire ne s'applique pas (leçon, partie 1, piège 4).

Étape 2, moments conditionnels : une face uniforme sur \(\{1, \dots, 5\}\) a pour espérance \(@Q08_moy_u5@\) et pour variance \(@Q08_e2_u5@ - @Q08_moy_u5@^2 = @Q08_var_u5@\). Donc \(\mathbb{E}[S \mid N] = 3(N - 1) + 6 = 3N + 3\) et \(\mathrm{Var}(S \mid N) = @Q08_var_u5@\,(N - 1)\).

Étape 3, variance totale (leçon, (C4)) : \(\mathbb{E}[@Q08_var_u5@\,(N - 1)] = @Q08_var_u5@ \times (@Lc_geo_e@ - 1) = @Q08_intra@\) et \(\mathrm{Var}(3N + 3) = 9 \times @Lc_geo_var@ = @Q08_inter@\). Total : \(@Q08_exact@\).

Contrôle par une seconde méthode exacte (script) : les équations du premier lancer, \(\mathbb{E}[S] = \tfrac16 \times 6 + \tfrac16 \sum_{x=1}^{5} (x + \mathbb{E}[S])\) et la même chose pour \(\mathbb{E}[S^2]\), donnent \(\mathbb{E}[S] = @Q08_es@\) et \(\mathbb{E}[S^2] = @Q08_e2@\), d'où \(@Q08_e2@ - @Q08_es@^2 = @Q08_exact@\). Simulations (script) : @Q08_mc:1@ par la loi conditionnelle, @Q08_direct_mc:1@ en lançant directement le dé.""",
    },
]
