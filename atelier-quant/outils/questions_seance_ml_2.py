"""Séance de ML 2 : leçon sur les arbres, forêts aléatoires, boosting et XGBoost, puis 12 questions à choix multiple.

Les questions couvrent toute la leçon. Elles comprennent des équivalents, jamais identiques, des deux questions
d'ensembles d'arbres du test de positionnement : DIAG-F4 (variance d'une moyenne
d'arbres corrélés : M1-04, M1-05, M1-06) et DIAG-A1 (poids optimal d'une feuille XGBoost : M1-08), ainsi qu'une
question « comparer forêt aléatoire et XGBoost » (M1-11).
Toutes les valeurs : verif/seance_ml_2.py. Dans chaque question, l'option "a" est la bonne.
"""
import lecons_arbres
import questions_diagnostic as diag

ID = "seance-ml-2"
TITRE = "Séance de ML 2 : arbres, forêts aléatoires, boosting et XGBoost"
VERIF = "verif/seance_ml_2.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [{"id": "lecon", "nom": "Leçon", "budget_min": 45}, {"id": "mla", "nom": "Machine Learning avancé", "budget_min": 25}]
PRESENTATION = (
    "D'abord une leçon d'environ 45 minutes sur les arbres de décision, le bagging, les forêts aléatoires, le boosting "
    "de gradient et XGBoost, avec six points de contrôle corrigés immédiatement et des expériences dont la sortie est "
    "réelle. Ensuite 12 questions à choix multiple qui couvrent toute la leçon, dont des équivalents des questions "
    "DIAG-F4 et DIAG-A1 du test de positionnement, avec d'autres nombres et d'autres habillages.\n\n"
    "Objectif de vitesse : chaque question se résout dans son temps cible avec les résultats de la leçon (tableau "
    "« Réflexes en 30 secondes »). Au-delà du double du temps cible affiché, choisis « Je ne sais pas » et passe à la "
    "suivante : c'est la règle d'abandon à suivre dans un test chronométré."
)
AVERTISSEMENTS = [
    "Les questions de ML sont présentées ici en choix multiple à une seule bonne réponse. Cherche la réponse avant de "
    "lire les options, pour t'entraîner aussi au format à réponse numérique.",
    "Ici le retour en arrière est permis et il n'y a pas de pénalité : ce sont les règles de cette plateforme, pas "
    "celles de tout test en ligne.",
]

LECONS = [lecons_arbres.LECON_ARBRES]
CODES = []

QUESTIONS = [
    # ------------------------------------------------- Partie 1 : arbre de décision
    {
        "id": "M1-01", "section": "mla", "theme": "Arbre de décision, gain de Gini", "difficulte": 2, "temps_cible_s": 90,
        "enonce": r"""Un nœud d'un arbre de classification contient 10 observations : 6 de classe 1 et 4 de classe 0. Une coupure envoie à gauche 4 observations, dont 1 de classe 1, et à droite les 6 autres, dont 5 de classe 1.

L'impureté d'un nœud est l'indice de Gini, \(1 - \sum_c \hat p_c^2\), où \(\hat p_c\) est la proportion de la classe \(c\) parmi les observations du nœud et la somme porte sur les deux classes. Le gain de la coupure est l'impureté du parent moins la moyenne des impuretés des deux enfants pondérée par leurs effectifs.

Que vaut le gain, arrondi à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q01_exact:3@", "explication": r"Oui : \(\tfrac{@Q01_parent_num@}{@Q01_parent_den@} - \tfrac{@Q01_enfants_num@}{@Q01_enfants_den@} = \tfrac{@Q01_exact_num@}{@Q01_exact_den@}\)."},
            {"id": "b", "texte": "@Q01_d_non_pondere:3@", "explication": r"C'est le parent moins la moyenne **simple** des impuretés des enfants. Le gain pondère chaque enfant par sa part des observations, \(4/10\) et \(6/10\)."},
            {"id": "c", "texte": "@Q01_d_demi:3@", "explication": r"C'est la moitié de la bonne réponse : l'indice de Gini à deux classes a été pris égal à \(\hat p\,(1 - \hat p)\) au lieu de \(1 - \hat p^2 - (1 - \hat p)^2 = 2\hat p\,(1 - \hat p)\)."},
            {"id": "d", "texte": "@Q01_enfants:3@", "explication": r"C'est l'impureté moyenne pondérée des enfants, \(\tfrac{@Q01_enfants_num@}{@Q01_enfants_den@}\). Le gain est la **baisse** d'impureté : il faut la retrancher de l'impureté du parent."},
        ],
        "solution": r"""Réponse : \(\tfrac{@Q01_exact_num@}{@Q01_exact_den@} \approx @Q01_exact:3m@\).

Étape 1, parent : proportion de la classe 1 égale à \(6/10\), donc Gini \(= 2 \times 0{,}6 \times 0{,}4 = \tfrac{@Q01_parent_num@}{@Q01_parent_den@}\) (à deux classes, \(1 - \hat p^2 - (1 - \hat p)^2 = 2\hat p\,(1 - \hat p)\)).

Étape 2, enfants : à gauche, proportion \(1/4\), Gini \(\tfrac{@Q01_gauche_num@}{@Q01_gauche_den@}\) ; à droite, proportion \(5/6\), Gini \(\tfrac{@Q01_droite_num@}{@Q01_droite_den@}\).

Étape 3, moyenne pondérée : \(\tfrac{4}{10} \times \tfrac{@Q01_gauche_num@}{@Q01_gauche_den@} + \tfrac{6}{10} \times \tfrac{@Q01_droite_num@}{@Q01_droite_den@} = \tfrac{@Q01_enfants_num@}{@Q01_enfants_den@}\).

Étape 4, gain : \(\tfrac{@Q01_parent_num@}{@Q01_parent_den@} - \tfrac{@Q01_enfants_num@}{@Q01_enfants_den@} = \tfrac{@Q01_exact_num@}{@Q01_exact_den@} \approx @Q01_exact:3m@\), positif comme tout gain de Gini (leçon, partie 1, R3).""",
    },
    {
        "id": "M1-02", "section": "mla", "theme": "Arbre de décision, invariance aux transformations monotones", "difficulte": 3, "temps_cible_s": 75,
        "enonce": r"""Un arbre de régression est entraîné avec une seule variable \(x\), dont les quatre observations d'entraînement valent −3, −1, 2 et 4, et des cibles deux à deux distinctes. On le réentraîne, avec les mêmes réglages et les mêmes cibles, après avoir remplacé \(x\) par l'une des transformations ci-dessous.

Laquelle peut changer la partition des observations d'entraînement produite par l'arbre ?""",
        "options": [
            {"id": "a", "texte": r"\(x^2\)", "explication": r"Oui : \(x^2\) n'est pas monotone sur des valeurs de signes différents. Les carrés valent @Q02_carres@, dans un autre ordre que les \(x\) ; sur les @Q02_nb_coupures@ partages candidats, @Q02_communes@ seul est commun aux deux variables."},
            {"id": "b", "texte": r"\(2x + 5\)", "explication": r"Fonction affine strictement croissante : même ordre, mêmes partages candidats, mêmes gains, donc même partition ; les seuils se correspondent même exactement, et les prédictions sur de nouveaux points sont identiques."},
            {"id": "c", "texte": r"\(e^{x}\)", "explication": r"Fonction strictement croissante : l'ordre des observations est conservé, donc la partition d'entraînement aussi (R6). Seule la prédiction d'un nouveau point situé entre deux valeurs d'entraînement qui encadrent un seuil peut changer."},
            {"id": "d", "texte": r"\(x^3\)", "explication": r"Le cube est strictement croissant sur tous les réels, valeurs négatives comprises : l'ordre, donc la partition, est conservé."},
        ],
        "solution": r"""Réponse : \(x^2\).

Étape 1 : la partition d'un nœud ne dépend que de l'ordre des valeurs de la variable ; les partages candidats sont « les \(k\) plus petites valeurs d'un côté, les autres de l'autre » (leçon, partie 1, R6). Une transformation strictement croissante (ou strictement décroissante) conserve ces partages : \(2x + 5\), \(e^x\) et \(x^3\) ne peuvent rien changer.

Étape 2 : avec \(x^2\), les valeurs deviennent @Q02_carres@. Rangées, elles placent −1 en premier, puis 2, puis −3, puis 4. Les @Q02_nb_coupures@ partages candidats de \(x\) isolent \(\{-3\}\), \(\{-3, -1\}\), \(\{-3, -1, 2\}\) ; ceux de \(x^2\) isolent \(\{-1\}\), \(\{-1, 2\}\), \(\{-1, 2, -3\}\). Un seul partage est commun (@Q02_communes@ sur @Q02_nb_coupures@) : si le meilleur partage pour \(x\) était l'un des deux autres, il n'est plus disponible, et la partition change.""",
    },
    # ------------------------------------------------- Partie 2 : bagging
    {
        "id": "M1-03", "section": "mla", "theme": "Bagging, échantillon bootstrap", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""On tire un échantillon bootstrap à partir de 8 observations distinctes : 8 tirages indépendants, chacun uniforme parmi les 8 observations, avec remise.

Quelle est l'espérance du nombre d'observations distinctes de l'échantillon, arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q03_exact:3@", "explication": r"Oui : une indicatrice par observation, présente avec probabilité \(1 - (7/8)^8\), et 8 observations."},
            {"id": "b", "texte": "@Q03_d_lim:3@", "explication": r"C'est \(8\,(1 - 1/e)\), avec la limite \(1/e\) à la place de la vraie probabilité d'absence \((7/8)^8\). Pour \(n = 8\), \((1 - 1/n)^n\) est strictement plus petit que \(1/e\), donc il y a un peu plus d'observations distinctes."},
            {"id": "c", "texte": "@Q03_d_oob:3@", "explication": r"C'est \(8\,(7/8)^8\), le nombre moyen d'observations **absentes** (hors sac). Le nombre d'observations distinctes est 8 moins cette quantité."},
            {"id": "d", "texte": "@Q03_d_un:3@", "explication": r"C'est la probabilité qu'**une** observation donnée soit présente. Il faut sommer sur les 8 observations (linéarité de l'espérance)."},
        ],
        "solution": r"""Réponse : \(8\,\big(1 - (7/8)^8\big) \approx @Q03_exact:3m@\).

Notations : pour \(i = 1, \dots, 8\), \(I_i\) est l'indicatrice de l'événement « l'observation \(i\) est tirée au moins une fois », et \(N = I_1 + \dots + I_8\) le nombre d'observations distinctes.

Étape 1 : un tirage manque l'observation \(i\) avec probabilité \(7/8\) ; les 8 tirages sont indépendants, donc \(i\) est absente avec probabilité \((7/8)^8 \approx @Q03_p_abs:4m@\).

Étape 2 : par linéarité de l'espérance, qui ne demande aucune indépendance entre les \(I_i\), \(\mathbb{E}[N] = 8\,\big(1 - (7/8)^8\big) \approx @Q03_exact:3m@\).

Simulation (script) : @Q03_mc:3@.""",
    },
    {
        "id": "M1-04", "section": "mla", "theme": "Variance d'une moyenne d'arbres corrélés", "difficulte": 5, "temps_cible_s": 150,
        "enonce": r"""Une forêt aléatoire est formée d'arbres dont les prédictions en un point fixé, \(T_1, T_2, \dots\), sont des variables aléatoires de même variance \(\sigma^2\) et de même corrélation \(\rho\) deux à deux ; le hasard porte sur l'échantillon d'entraînement et sur les tirages de la forêt.

La variance de la moyenne des \(B\) premiers arbres vaut exactement 0,5 pour \(B = 5\) et exactement 0,35 pour \(B = 20\).

Que vaut \(\rho\), arrondi à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q04_rho:3@", "explication": r"Oui : le plancher \(\rho\,\sigma^2\) vaut \(@Q04_a:1m@\), la variance d'un arbre \(\sigma^2 = @Q04_sigma2:1m@\), et leur rapport est \(\tfrac{@Q04_rho_num@}{@Q04_rho_den@}\)."},
            {"id": "b", "texte": "@Q04_a:3@", "explication": r"C'est le plancher \(\rho\,\sigma^2\), pas \(\rho\) : il faut encore le diviser par \(\sigma^2 = @Q04_sigma2:1m@\)."},
            {"id": "c", "texte": "0,350", "explication": r"C'est la variance mesurée pour \(B = 20\), prise pour le plancher : elle contient encore le terme \((1 - \rho)\,\sigma^2/20\). Et ce serait \(\rho\,\sigma^2\), pas \(\rho\)."},
            {"id": "d", "texte": "@Q04_d_un_moins:3@", "explication": r"C'est \(1 - \rho\) : la part \((1 - \rho)\,\sigma^2 = @Q04_b@\) a été divisée par \(\sigma^2\) à la place du plancher \(\rho\,\sigma^2\)."},
        ],
        "solution": r"""Réponse : \(\rho = \tfrac{@Q04_rho_num@}{@Q04_rho_den@} \approx @Q04_rho:3m@\).

Notations : \(\bar T_B\) est la moyenne des \(B\) premiers arbres ; \(a = \rho\,\sigma^2\) et \(b = (1 - \rho)\,\sigma^2\).

Étape 1 : \(\mathrm{Var}(\bar T_B) = \rho\,\sigma^2 + (1 - \rho)\,\sigma^2/B = a + b/B\) (leçon, partie 2, B2 : double somme des covariances, \(B\) variances et \(B(B-1)\) covariances égales à \(\rho\,\sigma^2\), divisée par \(B^2\)).

Étape 2 : \(a + b/5 = 0{,}5\) et \(a + b/20 = 0{,}35\). Par différence, \(b\,(1/5 - 1/20) = 0{,}5 - 0{,}35 = @Q04_diff:2m@\), d'où \(b = @Q04_b@\), puis \(a = 0{,}5 - b/5 = @Q04_a:1m@\).

Étape 3 : \(\sigma^2 = a + b = @Q04_sigma2:1m@\) et \(\rho = a/\sigma^2 = \tfrac{@Q04_rho_num@}{@Q04_rho_den@} \approx @Q04_rho:3m@\).

Lecture : quel que soit le nombre d'arbres, la variance ne descendra pas sous \(@Q04_a:1m@\) ; pour la baisser, il faut décorréler les arbres, pas en ajouter.""",
    },
    # ------------------------------------------------- Partie 3 : forêt aléatoire
    {
        "id": "M1-05", "section": "mla", "theme": "Forêt aléatoire, erreur quadratique et nombre d'arbres", "difficulte": 4, "temps_cible_s": 100,
        "enonce": r"""En un point fixé \(\mathbf{x}\), les arbres d'une forêt ont des prédictions \(T_1, T_2, \dots\) de même espérance, de variance \(\sigma^2 = 1\) et de corrélation \(\rho = 0{,}2\) deux à deux. La cible en ce point est une variable aléatoire \(Y\) d'espérance \(\mu\) et de variance 0,5, indépendante de l'échantillon d'entraînement et des tirages de la forêt. Le biais d'un arbre vérifie \((\mathbb{E}[T_1] - \mu)^2 = 0{,}1\).

On note \(\bar T_B\) la moyenne des \(B\) premiers arbres. Vers quelle valeur tend \(\mathbb{E}\big[(\bar T_B - Y)^2\big]\) quand \(B\) tend vers l'infini ?""",
        "options": [
            {"id": "a", "texte": "@Q05_lim:1@", "explication": r"Oui : biais au carré + plancher de variance + bruit, \(0{,}1 + 0{,}2 \times 1 + 0{,}5\)."},
            {"id": "b", "texte": "@Q05_sans_rho:1@", "explication": r"Cela suppose que la moyenne supprime toute la variance des arbres. Elle ne supprime que la part \((1 - \rho)\,\sigma^2/B\) ; le plancher \(\rho\,\sigma^2 = 0{,}2\) reste."},
            {"id": "c", "texte": "@Q05_B1:1@", "explication": r"C'est l'erreur d'un arbre seul (\(B = 1\)), comme si ajouter des arbres ne changeait rien. L'erreur espérée décroît avec \(B\) ; avec 100 arbres elle vaut déjà @Q05_B100:3@."},
            {"id": "d", "texte": "@Q05_sans_bruit:1@", "explication": r"Il manque la variance de la cible, 0,5 : aucun modèle ne prédit le bruit de \(Y\), qui s'ajoute à l'erreur par indépendance."},
        ],
        "solution": r"""Réponse : @Q05_lim:1@.

Étape 1, décomposition : on écrit \(\bar T_B - Y = (\bar T_B - \mathbb{E}\bar T_B) + (\mathbb{E}\bar T_B - \mu) + (\mu - Y)\). Les doubles produits ont une espérance nulle (termes centrés, constants, ou indépendants et centrés), donc \(\mathbb{E}[(\bar T_B - Y)^2] = (\mathbb{E}[T_1] - \mu)^2 + \mathrm{Var}(\bar T_B) + \mathrm{Var}(Y)\), avec \(\mathbb{E}\bar T_B = \mathbb{E}[T_1]\) par linéarité (leçon, partie 3).

Étape 2 : \(\mathrm{Var}(\bar T_B) = \rho\,\sigma^2 + (1 - \rho)\,\sigma^2/B\), qui tend vers \(\rho\,\sigma^2 = 0{,}2\).

Étape 3 : la limite vaut \(0{,}1 + 0{,}2 + 0{,}5 = @Q05_lim:1m@\). Pour \(B = 1\), on trouve \(@Q05_B1:1m@\), et pour \(B = 100\), \(@Q05_B100:3m@\) : l'erreur espérée décroît avec \(B\), mais vers un plancher qui contient le biais et \(\rho\,\sigma^2\).""",
    },
    {
        "id": "M1-06", "section": "mla", "theme": "Forêt aléatoire contre bagging, compromis entre sigma2 et rho", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""En un point fixé, on compare deux ensembles d'arbres dont les prédictions ont la même espérance.

- Bagging : variance d'un arbre \(\sigma^2 = 1\), corrélation entre deux arbres \(\rho = 0{,}6\).
- Forêt aléatoire (tirage de variables à chaque coupure) : \(\sigma^2 = 1{,}3\), \(\rho = 0{,}3\).

Les deux ensembles comptent le même nombre d'arbres \(B\). Pour quelles valeurs de \(B\) la moyenne de la forêt a-t-elle une variance strictement plus faible que celle du bagging ?""",
        "options": [
            {"id": "a", "texte": r"Pour \(B \ge @Q06_B_min@\).", "explication": r"Oui : \(@Q06_rf_plancher:2m@ + @Q06_rf_b:2m@/B < @Q06_bag_plancher:1m@ + @Q06_bag_b:1m@/B\) équivaut à \(B > @Q06_b_seuil:3m@\)."},
            {"id": "b", "texte": r"Pour tout \(B \ge 1\).", "explication": r"Pour \(B = 1\), la forêt a la variance d'un de ses arbres, @Q06_rf_1:1@, plus grande que @Q06_bag_1@ ; pour \(B = 2\), \(@Q06_rf_2:3m@\) contre \(@Q06_bag_2:1m@\). Le tirage de variables ne paie qu'une fois le terme en \(1/B\) assez petit."},
            {"id": "c", "texte": r"Pour aucun \(B\) : chaque arbre de la forêt a une variance plus grande.", "explication": r"Quand \(B\) grandit, seule compte \(\rho\,\sigma^2\) : \(@Q06_rf_plancher:2m@\) pour la forêt contre \(@Q06_bag_plancher:1m@\) pour le bagging. La baisse de \(\rho\) l'emporte sur la hausse de \(\sigma^2\)."},
            {"id": "d", "texte": r"Seulement à la limite, quand \(B \to \infty\).", "explication": r"L'inégalité est vraie dès \(B = @Q06_B_min@\) : \(@Q06_rf_3:3m@\) pour la forêt contre \(@Q06_bag_3:3m@\) pour le bagging."},
        ],
        "solution": r"""Réponse : pour \(B \ge @Q06_B_min@\).

Étape 1 : la variance de la moyenne de \(B\) arbres est \(\rho\,\sigma^2 + (1 - \rho)\,\sigma^2/B\) (leçon, partie 2, B2). Bagging : \(@Q06_bag_plancher:1m@ + @Q06_bag_b:1m@/B\). Forêt : \(@Q06_rf_plancher:2m@ + @Q06_rf_b:2m@/B\).

Étape 2 : la forêt est strictement meilleure si et seulement si \(@Q06_rf_plancher:2m@ + @Q06_rf_b:2m@/B < @Q06_bag_plancher:1m@ + @Q06_bag_b:1m@/B\), soit \(@Q06_num:2m@/B < @Q06_den:2m@\), soit \(B > @Q06_num:2m@/@Q06_den:2m@ \approx @Q06_b_seuil:3m@\).

Étape 3 : \(B\) est entier, donc \(B \ge @Q06_B_min@\). Contrôle : \(B = 2\) donne \(@Q06_rf_2:3m@\) contre \(@Q06_bag_2:1m@\), \(B = 3\) donne \(@Q06_rf_3:3m@\) contre \(@Q06_bag_3:3m@\).

Lecture : décorréler les arbres coûte en variance individuelle et rapporte sur le plancher ; la variance n'est qu'une partie de l'erreur, et le biais peut aussi monter (leçon, partie 3, expérience).""",
    },
    # ------------------------------------------------- Partie 4 : boosting de gradient
    {
        "id": "M1-07", "section": "mla", "theme": "Boosting de gradient, pseudo-résidu logistique", "difficulte": 3, "temps_cible_s": 75,
        "enonce": r"""Boosting de gradient avec la perte logistique \(\ell(y, f) = -\big[y \ln \pi + (1 - y)\ln(1 - \pi)\big]\), où \(y \in \{0, 1\}\) est l'étiquette, \(f\) le score du modèle (un réel) et \(\pi = 1/(1 + e^{-f})\) la probabilité prédite.

Une observation d'étiquette \(y = 0\) a un score courant \(f = \ln 3\). Quel est son pseudo-résidu, c'est-à-dire \(-\partial \ell/\partial f\) évalué au score courant ?""",
        "options": [
            {"id": "a", "texte": "@Q07_r:2@", "explication": r"Oui : \(\pi = 1/(1 + 1/3) = @Q07_pi:2m@\) et le pseudo-résidu vaut \(y - \pi\)."},
            {"id": "b", "texte": "@Q07_pi:2@", "explication": r"C'est \(\pi - y\), la dérivée \(\partial \ell/\partial f\) elle-même. Le pseudo-résidu est son opposé, la direction qui fait baisser la perte."},
            {"id": "c", "texte": "@Q07_d_score:3@", "explication": r"C'est \(y - f\), le résidu sur l'échelle des scores, qui serait le pseudo-résidu de la perte quadratique appliquée au score. Pour la perte logistique, la différence se prend sur l'échelle des probabilités : \(y - \pi\)."},
            {"id": "d", "texte": "@Q07_h:4@", "explication": r"C'est \(\pi\,(1 - \pi)\), la dérivée **seconde** de la perte, celle que XGBoost note \(h\)."},
        ],
        "solution": r"""Réponse : \(y - \pi = @Q07_r:2m@\).

Étape 1 : \(\pi = 1/(1 + e^{-\ln 3}) = 1/(1 + 1/3) = @Q07_pi:2m@\).

Étape 2, dérivée : \(d\pi/df = \pi\,(1 - \pi)\), donc \(\partial \ell/\partial f = -y\,(1 - \pi) + (1 - y)\,\pi = \pi - y\) (leçon, partie 4, D1).

Étape 3 : pseudo-résidu \(= -(\pi - y) = 0 - @Q07_pi:2m@ = @Q07_r:2m@\). Il est négatif : le prochain arbre doit faire baisser le score de cette observation d'étiquette 0.""",
    },
    # ------------------------------------------------- Partie 5 : XGBoost
    {
        "id": "M1-08", "section": "mla", "theme": "XGBoost, poids optimal d'une feuille, perte logistique", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Notations. Pour une observation \(i\) : \(y_i \in \{0, 1\}\) est l'étiquette et \(\hat y_i\) le score courant du modèle, somme des arbres déjà construits. La perte logistique, vue comme une fonction du score \(f\), est \(\ell(y, f) = -\big[y \ln \pi + (1 - y)\ln(1 - \pi)\big]\) avec \(\pi = 1/(1 + e^{-f})\). On note \(g_i\) et \(h_i\) ses dérivées première et seconde par rapport au score, évaluées au score courant : \(g_i = \frac{\partial \ell}{\partial f}(y_i, \hat y_i)\) et \(h_i = \frac{\partial^2 \ell}{\partial f^2}(y_i, \hat y_i)\).

Un nouvel arbre ajoute au score de chaque observation d'une même feuille une même constante \(w\), le poids de la feuille. XGBoost choisit \(w\) en minimisant, sur les observations \(i\) de la feuille, \(\sum_i \big[g_i\,w + \tfrac12 h_i\,w^2\big] + \tfrac12 \lambda\,w^2\), où \(\lambda \ge 0\) est le coefficient de régularisation.

Une feuille contient 5 observations, d'étiquettes 1, 1, 1, 1 et 0, toutes de score courant \(\hat y_i = 0\). On prend \(\lambda = 0{,}5\). Que vaut le \(w\) optimal, arrondi à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q08_exact:3@", "explication": r"Oui : \(G = @Q08_G:1m@\), \(H = @Q08_H:2m@\) et \(w = -G/(H + \lambda)\)."},
            {"id": "b", "texte": "@Q08_d_sans_lambda:3@", "explication": r"C'est \(-G/H\), sans la régularisation : \(\lambda = 0{,}5\) s'ajoute au dénominateur et rétrécit le poids vers 0."},
            {"id": "c", "texte": "@Q08_d_h_un:3@", "explication": r"C'est \(-G/(5 + \lambda)\), avec \(h_i = 1\) comme pour la perte quadratique. Pour la perte logistique, \(h_i = \pi_i\,(1 - \pi_i) = 1/4\) ici."},
            {"id": "d", "texte": "@Q08_d_moyenne:3@", "explication": r"C'est la moyenne des pseudo-résidus \(y_i - \pi_i\), le poids d'un boosting qui ajusterait les probabilités par moindres carrés. XGBoost divise la somme par \(H + \lambda\), pas par le nombre d'observations."},
        ],
        "solution": r"""Réponse : \(w = \tfrac{@Q08_exact_num@}{@Q08_exact_den@} \approx @Q08_exact:3m@\).

Notations : \(G\) et \(H\) sont les sommes des \(g_i\) et des \(h_i\) sur les observations de la feuille.

Étape 1, forme du poids optimal : le critère s'écrit \(G\,w + \tfrac12 (H + \lambda)\,w^2\), polynôme du second degré strictement convexe dès que \(H + \lambda > 0\) ; sa dérivée \(G + (H + \lambda)\,w\) s'annule en \(w = -G/(H + \lambda)\) (leçon, partie 5, X1).

Étape 2, dérivées de la perte logistique : \(g_i = \pi_i - y_i\) et \(h_i = \pi_i\,(1 - \pi_i)\), car \(d\pi/df = \pi\,(1 - \pi)\) (leçon, partie 4, D1).

Étape 3 : score courant nul, donc \(\pi_i = 1/2\) pour toutes les observations. \(g_i = -1/2\) pour les quatre étiquettes 1 et \(+1/2\) pour l'étiquette 0 : \(G = @Q08_G:1m@\). \(h_i = 1/4\) pour toutes : \(H = @Q08_H:2m@\).

Étape 4 : \(w = -(@Q08_G:1m@)/(@Q08_H:2m@ + 0{,}5) = \tfrac{@Q08_exact_num@}{@Q08_exact_den@} \approx @Q08_exact:3m@\). Le poids est positif : une feuille dominée par des étiquettes 1 voit son score augmenter.

Remarque : pour la perte logistique, ce poids est un pas de Newton sur la perte pénalisée de la feuille, pas son minimum exact (leçon, partie 5, X5).""",
    },
    {
        "id": "M1-09", "section": "mla", "theme": "XGBoost, gain d'une coupure avec lambda et gamma", "difficulte": 4, "temps_cible_s": 150,
        "enonce": r"""XGBoost avec la perte quadratique \(\ell(y, f) = \tfrac12 (y - f)^2\). Pour chaque observation \(i\), de cible \(y_i\) et de prédiction courante \(\hat y_i\), on note \(g_i = \hat y_i - y_i\) et \(h_i = 1\) les dérivées première et seconde de la perte par rapport à la prédiction ; pour un ensemble \(S\) d'observations, \(G_S\) et \(H_S\) sont les sommes des \(g_i\) et des \(h_i\) sur \(S\).

Le critère d'un arbre est \(\sum_{\text{feuilles } S} \big[G_S\,w_S + \tfrac12 (H_S + \lambda)\,w_S^2\big] + \gamma \times (\text{nombre de feuilles})\), où \(w_S\) est le poids de la feuille \(S\) ; pour une structure d'arbre donnée, on le minimise en les poids. Le gain d'une coupure est la baisse de ce minimum quand on coupe une feuille en deux.

Une feuille contient 5 observations dont les résidus \(y_i - \hat y_i\) valent 2, 2, 3, −1 et −2. On envisage la coupure qui sépare les trois premières des deux dernières. Avec \(\lambda = 1\) et \(\gamma = 1\), que vaut le gain, arrondi à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q09_exact:3@", "explication": r"Oui : \(\tfrac12\big[@Q09_sL:2m@ + @Q09_sR@ - @Q09_sP:4m@\big] - 1\)."},
            {"id": "b", "texte": "@Q09_d_sans_demi:3@", "explication": r"Le facteur \(\tfrac12\) a été oublié : la valeur minimale d'une feuille est \(-\tfrac12 G^2/(H + \lambda)\), pas \(-G^2/(H + \lambda)\)."},
            {"id": "c", "texte": "@Q09_d_sans_lambda:3@", "explication": r"C'est le gain calculé avec \(\lambda = 0\) dans les dénominateurs. Ici \(\lambda = 1\) s'ajoute à chaque \(H\)."},
            {"id": "d", "texte": "@Q09_d_sans_gamma:3@", "explication": r"Il manque \(-\gamma\) : la coupure crée une feuille de plus, qui coûte \(\gamma = 1\)."},
        ],
        "solution": r"""Réponse : \(@Q09_exact:3m@\), à trois décimales ; la coupure améliore le critère.

Étape 1, valeur minimale d'une feuille : \(G\,w + \tfrac12 (H + \lambda)\,w^2\) est minimal en \(w = -G/(H + \lambda)\) et vaut alors \(-\tfrac12 G^2/(H + \lambda)\) (leçon, partie 5, X1).

Étape 2, gain : avant la coupure, la feuille contribue \(-\tfrac12 G^2/(H + \lambda) + \gamma\) ; après, \(-\tfrac12 G_L^2/(H_L + \lambda) - \tfrac12 G_R^2/(H_R + \lambda) + 2\gamma\). La baisse vaut \(\tfrac12\big[\tfrac{G_L^2}{H_L + \lambda} + \tfrac{G_R^2}{H_R + \lambda} - \tfrac{G^2}{H + \lambda}\big] - \gamma\), avec \(G = G_L + G_R\) et \(H = H_L + H_R\).

Étape 3, sommes : \(g_i\) est l'opposé du résidu. Gauche : \(G_L = @Q09_GL@\), \(H_L = @Q09_HL@\). Droite : \(G_R = @Q09_GR@\), \(H_R = @Q09_HR@\). Feuille entière : \(G = @Q09_G@\), \(H = 5\).

Étape 4 : \(\tfrac{G_L^2}{H_L + 1} = @Q09_sL:2m@\), \(\tfrac{G_R^2}{H_R + 1} = @Q09_sR@\), \(\tfrac{G^2}{H + 1} = \tfrac{@Q09_sP_num@}{@Q09_sP_den@} \approx @Q09_sP:4m@\) ; gain \(= \tfrac12\big[@Q09_sL:2m@ + @Q09_sR@ - @Q09_sP:4m@\big] - 1 \approx @Q09_exact:4m@\).

Pour aller plus loin : les poids des deux nouvelles feuilles valent \(w_L = @Q09_wL:2m@\) et \(w_R = @Q09_wR@\).""",
    },
    {
        "id": "M1-10", "section": "mla", "theme": "XGBoost, min_child_weight et perte logistique", "difficulte": 3, "temps_cible_s": 75,
        "enonce": r"""XGBoost avec la perte logistique et le réglage `min_child_weight` égal à 1 : une coupure est refusée si, dans l'un des deux enfants, la somme des \(h_i\) est strictement inférieure à 1. Ici \(h_i = \pi_i\,(1 - \pi_i)\) est la dérivée seconde de la perte logistique par rapport au score, \(\pi_i\) étant la probabilité prédite pour l'observation \(i\).

Toutes les observations d'un enfant envisagé ont la même probabilité prédite, 0,8. Combien d'observations doit-il contenir au minimum pour que la coupure ne soit pas refusée pour cette raison ?""",
        "options": [
            {"id": "a", "texte": "@Q10_exact@", "explication": r"Oui : \(h = 0{,}8 \times 0{,}2 = @Q10_h:2m@\), et \(n \times @Q10_h:2m@ \ge 1\) exige \(n \ge @Q10_inv:2m@\)."},
            {"id": "b", "texte": "@Q10_d_max@", "explication": r"C'est le minimum quand \(\pi = 1/2\), où \(h\) atteint son maximum \(1/4\). Avec \(\pi = 0{,}8\), \(h\) est plus petit et il faut plus d'observations."},
            {"id": "c", "texte": "@Q10_d_inv_un_moins@", "explication": r"C'est \(1/(1 - \pi)\) : \(h\) a été pris égal à \(1 - \pi\). La dérivée seconde de la perte logistique est \(\pi\,(1 - \pi)\)."},
            {"id": "d", "texte": "1", "explication": r"C'est la lecture de `min_child_weight` comme un nombre d'observations, valable seulement pour la perte quadratique, où \(h_i = 1\). Ici chaque observation ne pèse que \(h = @Q10_h:2m@\)."},
        ],
        "solution": r"""Réponse : @Q10_exact@ observations.

Étape 1 : \(h = \pi\,(1 - \pi) = 0{,}8 \times 0{,}2 = @Q10_h:2m@\) pour chaque observation (leçon, partie 4, D1 : dérivée seconde de la perte logistique).

Étape 2 : avec \(n\) observations, la somme vaut \(n \times @Q10_h:2m@\), au moins 1 si et seulement si \(n \ge 1/@Q10_h:2m@ = @Q10_inv:2m@\).

Étape 3 : \(n\) est entier, donc \(n \ge @Q10_exact@\). Plus le modèle est confiant (\(\pi\) proche de 0 ou de 1), plus il faut d'observations pour couper.""",
    },
    # ------------------------------------------------- Partie 6 : comparaison
    {
        "id": "M1-11", "section": "mla", "theme": "Comparer forêt aléatoire et XGBoost", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""On hésite entre une forêt aléatoire et XGBoost pour une régression évaluée par l'erreur quadratique. Laquelle de ces affirmations est vraie ?""",
        "options": [
            {"id": "a", "texte": "Ajouter des arbres à la forêt ne peut pas augmenter son erreur quadratique de test espérée, alors que l'erreur de test de XGBoost peut augmenter avec le nombre d'arbres, qui se règle par arrêt précoce sur un jeu de validation.",
             "explication": r"Oui : pour la forêt, seul le terme \((1 - \rho)\,\sigma^2/B\) de l'erreur espérée dépend de \(B\), et il décroît. Pour le boosting, l'erreur d'entraînement tend vers 0 quand les arbres isolent chaque observation, bruit compris : avec un taux d'apprentissage de 0,1, elle passe sous 1 % de sa valeur initiale après @D_m_1pct@ arbres."},
            {"id": "b", "texte": "La forêt réduit surtout le biais d'arbres peu profonds, et XGBoost surtout la variance d'arbres profonds.",
             "explication": r"C'est l'inverse. La moyenne d'une forêt garde l'espérance d'un arbre, donc son biais, et réduit la variance d'arbres profonds ; le boosting corrige pas à pas le biais d'arbres peu profonds."},
            {"id": "c", "texte": "XGBoost construit ses arbres indépendamment les uns des autres, en parallèle, alors que la forêt les construit l'un après l'autre.",
             "explication": r"C'est l'inverse. Les arbres d'une forêt sont indépendants sachant les données ; l'arbre \(m\) d'un boosting est ajusté aux pseudo-résidus du modèle \(f_{m-1}\), donc après lui. XGBoost parallélise la recherche des coupures à l'intérieur d'un arbre, pas la suite des arbres."},
            {"id": "d", "texte": "Les deux méthodes moyennent des arbres : pour chacune, l'erreur de test espérée décroît avec le nombre d'arbres.",
             "explication": r"Le boosting ne moyenne pas : il additionne des corrections successives, et l'erreur de test peut remonter après un minimum (leçon, partie 4, expérience)."},
        ],
        "solution": r"""Réponse : la forêt ne peut pas voir son erreur quadratique de test espérée augmenter avec le nombre d'arbres ; XGBoost, si.

Étape 1, forêt : en un point, \(\mathbb{E}[(\bar T_B - Y)^2] = (\mathbb{E}[T_1] - \mu)^2 + \rho\,\sigma^2 + (1 - \rho)\,\sigma^2/B + \mathrm{Var}(Y)\), dont seul le terme en \(1/B\) dépend de \(B\) (leçon, partie 3). C'est une affirmation sur l'espérance : une courbe d'erreur réalisée peut fluctuer.

Étape 2, boosting : avec la perte quadratique, chaque arbre multiplie la somme des carrés des résidus d'entraînement par \((1 - \eta)^2\) quand il isole chaque observation (leçon, partie 4, D3) ; le modèle finit par reproduire le bruit, et l'erreur de test, après un minimum, remonte (expérience de la leçon). D'où l'arrêt précoce.

Étape 3, les autres options inversent les rôles : forêt = moyenne, variance, parallèle ; boosting = somme séquentielle, biais.""",
    },
    # ------------------------------------------------- Partie 7 : importance des variables
    {
        "id": "M1-12", "section": "mla", "theme": "Importance des variables, impureté contre permutation", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""Une forêt aléatoire de classification, aux feuilles d'une observation, est entraînée sur 1000 observations avec trois variables : une variable binaire très informative, un identifiant client tiré au hasard (une valeur différente par observation, indépendante de la cible et des autres variables) et une variable binaire de bruit. L'importance par impureté (`feature_importances_` de scikit-learn) vaut 0,75 pour l'identifiant et 0,23 pour la variable informative.

Que faut-il en conclure ?""",
        "options": [
            {"id": "a", "texte": "Rien sur l'utilité de l'identifiant : l'importance par impureté est mesurée sur les données d'entraînement et favorise les variables qui offrent de nombreuses coupures ; il faut mesurer l'importance par permutation sur un jeu de validation.",
             "explication": r"Oui : sous l'hypothèse nulle, avec 100 observations, le meilleur gain de Gini d'une variable continue vaut en moyenne @F_rapport:1@ fois celui d'une variable binaire (leçon, partie 7). Sur la validation, l'importance par permutation de l'identifiant est d'espérance nulle (I2)."},
            {"id": "b", "texte": "L'identifiant est la variable la plus utile : on peut retirer la variable informative sans perte.",
             "explication": r"L'identifiant n'a aucun lien avec la cible : ses coupures n'apprennent que le bruit de l'échantillon d'entraînement, et ne servent à rien sur de nouvelles données."},
            {"id": "c", "texte": "L'importance par permutation aurait le même biais, puisqu'elle est calculée avec le même modèle.",
             "explication": r"Le modèle est le même, mais la permutation se mesure sur des données que le modèle n'a pas vues. Pour une variable indépendante de la cible et des autres variables, permuter ne change pas la loi des données de validation : l'importance est d'espérance nulle (leçon, partie 7, I2)."},
            {"id": "d", "texte": "Augmenter le nombre d'arbres ferait disparaître le biais.",
             "explication": r"Le biais est présent dans chaque arbre, dont les gains sont mesurés sur ses propres données d'entraînement ; la moyenne sur les arbres conserve l'espérance, donc le biais."},
        ],
        "solution": r"""Réponse : on ne peut rien conclure de l'importance par impureté sur l'utilité de l'identifiant ; il faut l'importance par permutation sur un jeu de validation.

Étape 1 : l'importance par impureté additionne des gains de coupures mesurés sur les données qui ont servi à choisir ces coupures. Tout gain est positif ou nul (leçon, partie 1, R2 à R4) : une variable de bruit utilisée pour couper accumule une importance positive.

Étape 2 : une variable aux valeurs toutes distinctes offre une coupure candidate par valeur, contre une seule pour une variable binaire. Sous l'hypothèse nulle, son meilleur gain est en espérance au moins celui de la variable binaire (leçon, partie 7, I1) ; avec 100 observations, @F_cont_moy:4@ contre @F_bin_moy:4@ en moyenne (script). Avec des feuilles d'une observation, l'arbre coupe sur l'identifiant jusqu'à isoler chaque observation.

Étape 3 : l'importance par permutation compare le score de validation avant et après permutation de la variable. Pour une variable indépendante de la cible et des autres variables, les données permutées ont la même loi que les données d'origine : importance d'espérance nulle. L'expérience de la leçon montre ce renversement.""",
    },
]
