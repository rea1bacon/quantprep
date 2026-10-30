"""OA QRT Quant Researcher, examen : 21 questions reformulées et 3 questions de remplacement, au format « examen blanc »
(pas de pause, pas de « Je ne sais pas »).

Numérotation : T9-Qnn est la question nn (2 à 11 : mathématiques ; 12 à 16 : ML fondamental ; 17 à 24 : ML avancé). Les
énoncés des questions 14, 15 et 18 ne sont pas disponibles : ce sont des questions de remplacement, signalées comme
telles. Chaque autre question est une reformulation en français (mêmes données, mêmes options, même type de réponse), et
sa réponse est établie par preuve ou calcul.

Format (voir outils/CONSIGNES_SEANCES.md) : réponse unique, l'option "a" est la bonne ; plusieurs réponses, la question
porte "correctes". Le champ "ordre_origine" donne l'ordre d'origine des options (la page mélange l'ordre à l'affichage ;
le corrigé, outils/questions_debrief_test_9_octobre.py, réutilise l'ordre d'origine).
Toutes les valeurs : verif/test_9_octobre.py. Les identifiants restent ceux de la première version, pour le suivi.
"""
import questions_diagnostic as diag
from code_test_9_octobre import CODE

ID = "test-9-octobre"
TITRE = "OA QRT Quant Researcher : examen"
VERIF = "verif/test_9_octobre.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0, "pause": False, "je_ne_sais_pas": False}
SECTIONS = [
    {"id": "code", "nom": "Code, science des données", "budget_min": 30},
    {"id": "math", "nom": "Mathématiques", "budget_min": 28},
    {"id": "mlf", "nom": "Machine Learning fondamental", "budget_min": 9},
    {"id": "mla", "nom": "Machine Learning avancé", "budget_min": 15},
]
PRESENTATION = (
    "L'évaluation en ligne Quant Researcher de QRT, session du 9 octobre 2026, dans les mêmes conditions : 82 minutes d'une traite, "
    "sans notes, sans recherche, sans pause. Quatre sections : 1 question de code (budget 30 minutes), 10 de mathématiques "
    "(28 minutes), 5 de ML fondamental (9 minutes), 8 de ML avancé (15 minutes).\n\n"
    "Ce que contient cette page, exactement :\n\n"
    "- **21 questions reformulées** en français : mêmes données, mêmes options, même type de réponse (unique ou multiple), mais pas "
    "le texte d'origine. Cette page n'est pas un document officiel de QRT.\n"
    "- **3 questions de remplacement** (numéros 14, 15 et 18), parce que les énoncés d'origine ne sont pas disponibles. Elles "
    "portent sur un thème de la même section et ne prétendent pas reproduire ces questions.\n"
    "- Quand une question d'origine est ambiguë ou contient une incohérence, la reformulation la conserve : la correction donne la "
    "réponse la plus défendable et dit quelle autre lecture mènerait à une autre réponse.\n\n"
    "Barème : une question à plusieurs réponses est notée ici **tout ou rien** (il faut cocher exactement les affirmations justes). "
    "C'est un choix de cette plateforme : le barème réel et la grille du correcteur ne sont pas connus. Les réponses données ici sont "
    "établies par preuve ou par calcul vérifié (`verif/test_9_octobre.py`).\n\n"
    "Règle d'abandon : au-delà du double du temps cible affiché, choisir l'option la plus plausible après élimination et passer à la "
    "suite ; il n'y a pas de pénalité ici. Le corrigé question par question est sur une page séparée, à ouvrir après l'examen."
)
AVERTISSEMENTS = [
    "Les réponses attendues sont celles que cette plateforme a établies et justifiées ; la grille de correction d'origine n'est pas "
    "connue. Pour les questions signalées comme ambiguës, une autre réponse a pu être attendue.",
    "Sur la page, l'ordre des options est mélangé à chaque passage.",
]

# La question de code est dans outils/code_test_9_octobre.py (variable CODE).
CODES = [CODE]

QUESTIONS = [
    # =============================================================== MATHÉMATIQUES (questions 2 à 11)
    {
        "id": "T9-Q02", "section": "math", "theme": "Tests d'adéquation", "difficulte": 2, "temps_cible_s": 90,
        "correctes": ["a", "b", "c", "d"], "ordre_origine": ["a", "b", "c", "d"],
        "enonce": r"""Question 2. Un test d'adéquation confronte un échantillon à une loi théorique. Parmi les associations suivantes entre un test d'adéquation et le type de loi auquel on l'applique, lesquelles sont correctes ?""",
        "options": [
            {"id": "a", "texte": "Test du khi-deux d'adéquation : loi binomiale",
             "explication": r"Juste. Le khi-deux compare, classe par classe, un effectif observé à un effectif attendu ; une loi binomiale a un nombre fini de valeurs, qui servent de classes."},
            {"id": "b", "texte": "Test de Kolmogorov-Smirnov : loi continue",
             "explication": r"Juste. La loi de sa statistique sous l'hypothèse nulle ne dépend pas de la loi testée précisément quand celle-ci est continue (étape 2 de la correction)."},
            {"id": "c", "texte": "Test du khi-deux d'adéquation : loi de Poisson",
             "explication": r"Juste. La loi de Poisson est discrète ; ses valeurs sont en nombre infini, on regroupe donc la queue en une classe « au moins tant » pour revenir à un nombre fini de classes."},
            {"id": "d", "texte": "Test d'Anderson-Darling : loi continue",
             "explication": r"Juste. Comme Kolmogorov-Smirnov, il compare la fonction de répartition empirique à celle de la loi continue testée, avec un poids plus fort dans les queues."},
        ],
        "solution": r"""Réponse : les quatre associations sont correctes.

Notations : \(X_1, \dots, X_n\) sont les \(n\) observations, supposées indépendantes et de même loi ; \(F\) est la fonction de répartition de la loi testée, \(F(x) = \mathbb{P}(X \le x)\) ; \(F_n\) est la fonction de répartition empirique, \(F_n(x)\) étant la proportion d'observations inférieures ou égales à \(x\).

Étape 1, khi-deux. On découpe les valeurs possibles en \(m\) classes. Pour la classe \(j\), \(O_j\) est le nombre d'observations qui y tombent et \(E_j = n\,p_j\) le nombre attendu, où \(p_j\) est la probabilité de la classe sous la loi testée. La statistique est \(\sum_{j=1}^{m} (O_j - E_j)^2 / E_j\). Elle ne demande qu'un nombre fini de classes : elle convient à toute loi discrète (binomiale ; Poisson après regroupement de la queue), et aussi à une loi continue découpée en intervalles. Sa loi limite est un khi-deux à \(m - 1 - r\) degrés de liberté, où \(r\) est le nombre de paramètres estimés sur les classes (théorème de Pearson et Fisher, admis ici).

Étape 2, Kolmogorov-Smirnov. La statistique est \(D_n = \sup_x \lvert F_n(x) - F(x) \rvert\). Si \(F\) est continue, les variables \(U_i = F(X_i)\) sont uniformes sur \([0, 1]\) sous l'hypothèse nulle. Preuve : pour \(u\) dans \(]0, 1[\), soit \(x_u\) le plus grand réel tel que \(F(x_u) = u\) (il existe car \(F\) est continue et croissante de limites 0 et 1) ; alors \(F(X_i) \le u\) équivaut à \(X_i \le x_u\), de probabilité \(F(x_u) = u\). Comme \(D_n\) ne dépend des données qu'à travers les \(U_i\) (c'est le plus grand écart entre la fonction de répartition empirique des \(U_i\) et celle de la loi uniforme), sa loi est la même pour toute loi continue : une seule table suffit. Pour une loi discrète, \(F(X_i)\) n'est plus uniforme et cette table rend le test trop prudent.

Étape 3, Anderson-Darling. La statistique est une distance quadratique entre \(F_n\) et \(F\), pondérée par \(1 / \big(F(x)\,(1 - F(x))\big)\), poids qui grandit dans les queues. Le même argument (\(F(X_i)\) uniforme quand \(F\) est continue) justifie son emploi pour une loi continue entièrement spécifiée.

Piège commun aux trois, constaté par simulation (script) : les tables supposent la loi testée **entièrement connue**. Sur @Q02_rep@ échantillons gaussiens de taille @Q02_n@, le test de Kolmogorov-Smirnov au niveau 5 % rejette à tort dans @Q02_ks_vrais:3@ des cas quand la moyenne et l'écart-type sont les vrais, et dans @Q02_ks_estimes:3@ des cas quand ils sont estimés sur l'échantillon : le test ne rejette alors presque plus jamais (il faut la correction de Lilliefors). Pour le khi-deux sur une loi de Poisson à paramètre estimé et @Q02_chi_classes@ classes, le taux de rejet vaut @Q02_chi_bon:3@ avec @Q02_chi_ddl@ degrés de liberté, et @Q02_chi_naif:3@ si l'on oublie de retirer un degré pour le paramètre estimé.

Remarque sur l'énoncé : il ne dit pas « s'applique uniquement à ». Le khi-deux s'applique aussi à une loi continue découpée en classes ; cela ne rend fausse aucune des quatre associations.""",
    },
    {
        "id": "T9-Q03", "section": "math", "theme": "Hypothèses de la régression linéaire", "difficulte": 1, "temps_cible_s": 60,
        "conceptuelle": True, "ordre_origine": ["b", "c", "d", "a"],
        "enonce": r"""Question 3. Laquelle des propositions suivantes **n'est pas** une hypothèse valide de la régression linéaire ?""",
        "options": [
            {"id": "a", "texte": "La variance autour de la droite de régression varie avec les valeurs de la variable explicative.",
             "explication": "Oui : c'est l'hétéroscédasticité, c'est-à-dire la violation de l'hypothèse de variance constante des erreurs."},
            {"id": "b", "texte": "Il existe une relation linéaire entre la variable dépendante et les régresseurs.",
             "explication": "C'est l'hypothèse de linéarité, la première du modèle : l'espérance de la variable expliquée est une fonction linéaire des régresseurs."},
            {"id": "c", "texte": "Les erreurs (ou résidus) sont distribuées normalement et indépendantes les unes des autres.",
             "explication": "Ce sont deux hypothèses classiques : l'indépendance des erreurs, et leur normalité, cette dernière n'étant nécessaire que pour les tests exacts à petit échantillon."},
            {"id": "d", "texte": "La multicolinéarité entre variables explicatives est minimale.",
             "explication": "C'est l'hypothèse d'absence de colinéarité : avec des régresseurs presque liés, les coefficients restent calculables mais leur variance devient très grande."},
        ],
        "solution": r"""Réponse : « la variance autour de la droite de régression varie avec les valeurs de la variable explicative ». C'est l'énoncé de l'hétéroscédasticité, le contraire de l'hypothèse d'homoscédasticité.

Le modèle. Pour l'observation \(i\), \(Y_i\) est la variable expliquée, \(\mathbf{x}_i\) le vecteur de ses régresseurs, \(\boldsymbol{\beta}\) le vecteur des coefficients inconnus et \(\varepsilon_i\) l'erreur aléatoire : \(Y_i = \mathbf{x}_i^{\top} \boldsymbol{\beta} + \varepsilon_i\). Les hypothèses classiques sont : (1) linéarité, c'est la forme du modèle ; (2) erreurs centrées sachant les régresseurs ; (3) erreurs de même variance \(\sigma^2\) quel que soit \(\mathbf{x}_i\) (homoscédasticité) ; (4) erreurs non corrélées entre elles ; (5) régresseurs non colinéaires ; et, pour les tests exacts à petit échantillon seulement, (6) erreurs gaussiennes.

Les options « relation linéaire », « erreurs normales et indépendantes » et « multicolinéarité minimale » reprennent (1), (4) et (6), puis (5). L'option restante affirme le contraire de (3).

Ce que coûte la violation de (3), sous (1), (2), (4), (5) : l'estimateur des moindres carrés reste sans biais, mais il n'est plus celui de variance minimale parmi les estimateurs linéaires sans biais, et les erreurs-types usuelles sont fausses, donc les tests et intervalles aussi. Remède : erreurs-types robustes, ou moindres carrés pondérés.

Remarque sur l'énoncé : « multicolinéarité minimale » est une formulation lâche. L'hypothèse exacte est l'absence de colinéarité parfaite ; une forte colinéarité n'invalide pas le modèle, elle dégrade la précision.""",
    },
    {
        "id": "T9-Q04", "section": "math", "theme": "Corrélation et covariance", "difficulte": 2, "temps_cible_s": 75,
        "correctes": ["a", "d"], "ordre_origine": ["a", "b", "c", "d"],
        "enonce": r"""Question 4. Quelles affirmations sont correctes ?""",
        "options": [
            {"id": "a", "texte": "La corrélation n'est pas modifiée par un changement d'échelle, alors que la covariance l'est.",
             "explication": r"Juste : multiplier \(X\) par \(a > 0\) multiplie la covariance par \(a\) et laisse la corrélation inchangée (étapes 1 et 2)."},
            {"id": "b", "texte": "La corrélation et la covariance sont toutes deux indépendantes de l'échelle.",
             "explication": r"Faux pour la covariance : exprimer \(X\) en centimètres plutôt qu'en mètres la multiplie par 100 (de \(@Q04_cov:2m@\) à \(@Q04_cov_cm:0m@\) dans l'exemple)."},
            {"id": "c", "texte": "La corrélation et la covariance sont sans dimension, c'est-à-dire sans unité.",
             "explication": r"Faux pour la covariance : son unité est le produit des unités de \(X\) et de \(Y\). Seule la corrélation, quotient de la covariance par le produit des écarts-types, est sans unité."},
            {"id": "d", "texte": "La corrélation et la covariance ne mesurent toutes deux qu'une relation linéaire entre les variables.",
             "explication": r"Juste : elles peuvent être nulles alors que \(Y\) est une fonction de \(X\) ; exemple \(Y = X^2\) avec \(X\) symétrique (étape 3)."},
        ],
        "solution": r"""Réponse : « la corrélation n'est pas modifiée par un changement d'échelle, alors que la covariance l'est » et « toutes deux ne mesurent qu'une relation linéaire ».

Notations : \(X\) et \(Y\) sont deux variables aléatoires de variances finies et non nulles ; \(\mathrm{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]\) ; la corrélation est \(\rho(X, Y) = \mathrm{Cov}(X, Y) / (\sigma_X\,\sigma_Y)\), où \(\sigma_X\) et \(\sigma_Y\) sont les écarts-types ; \(a\) et \(b\) sont des constantes.

Étape 1, covariance. \(aX + b\) a pour espérance \(a\,\mathbb{E}[X] + b\), donc \(aX + b - \mathbb{E}[aX + b] = a\,(X - \mathbb{E}[X])\) et \(\mathrm{Cov}(aX + b, Y) = a\,\mathrm{Cov}(X, Y)\) : la covariance change avec l'échelle, et son unité est le produit des unités de \(X\) et de \(Y\).

Étape 2, corrélation. L'écart-type de \(aX + b\) est \(\lvert a \rvert\,\sigma_X\), donc \(\rho(aX + b, Y) = \dfrac{a}{\lvert a \rvert}\,\rho(X, Y)\) : inchangée si \(a > 0\), de signe opposé si \(a < 0\). Un changement d'échelle (changement d'unité, \(a > 0\)) ne la modifie pas, et le quotient est sans unité.

Exemple (script, loi uniforme sur quatre couples) : \(X\) vaut 1, 2, 3, 4 et \(Y\) vaut 2, 1, 4, 3. Covariance \(@Q04_cov:2m@\), corrélation \(@Q04_corr:1m@\). Avec \(100\,X\) : covariance \(@Q04_cov_cm:0m@\), corrélation \(@Q04_corr_cm:1m@\). Avec \(-X\) : corrélation \(@Q04_corr_oppose:1m@\).

Étape 3, relation linéaire seulement. Par l'inégalité de Cauchy-Schwarz, \(\lvert \rho \rvert \le 1\), avec égalité si et seulement si \(Y\) est une fonction affine de \(X\) : \(\rho\) mesure la proximité à une droite. Contre-exemple à toute lecture plus large : \(X\) uniforme sur \(\{-1, 0, 1\}\) et \(Y = X^2\). Alors \(\mathbb{E}[X] = 0\) et \(\mathbb{E}[XY] = \mathbb{E}[X^3] = 0\), donc \(\mathrm{Cov}(X, Y) = @Q04_cov_carre@\) et la corrélation est nulle, alors que \(Y\) est entièrement déterminée par \(X\). Simulation de cette covariance : @Q04_cov_carre_mc:4@ (erreur standard @Q04_cov_carre_se:4@).

Remarque sur l'énoncé : « changement d'échelle » est pris au sens d'un facteur strictement positif. Avec un facteur négatif, la corrélation change de signe.""",
    },
    {
        "id": "T9-Q05", "section": "math", "theme": "Loi de Poisson et probabilité a priori", "difficulte": 4, "temps_cible_s": 240,
        "ordre_origine": ["b", "c", "a", "d"],
        "enonce": r"""Question 5. Des crises financières surviennent à un taux moyen constant, et une nouvelle crise survient indépendamment du temps écoulé depuis la précédente. Pour laquelle des valeurs proposées la probabilité que les crises surviennent à un taux moyen égal à cette valeur **et** que 6 crises surviennent pendant les 10 prochaines années est-elle maximale ?

On sait que la probabilité a priori de \(N\) crises en 10 ans est proportionnelle à \(1/N\) (normalisée pour que les probabilités somment à 1 sur toutes les valeurs possibles de \(N\)).""",
        "options": [
            {"id": "a", "texte": "6", "explication": r"Oui : la probabilité jointe est proportionnelle à \(N^5 e^{-N}\), qui vaut \(@Q05_g_6:2m@\) en 6, la plus grande des quatre valeurs."},
            {"id": "b", "texte": "2", "explication": r"\(N^5 e^{-N}\) vaut \(@Q05_g_2:2m@\) en 2, la plus petite des quatre valeurs. La valeur 2 ne l'emporte que si l'on lit les options comme des taux par an (moyenne de @Q05_alt_moy_2@ crises en 10 ans pour la valeur 2) : voir la fin de la correction."},
            {"id": "c", "texte": "4", "explication": r"Le distracteur le plus proche : \(N^5 e^{-N}\) vaut \(@Q05_g_4:2m@\) en 4 contre \(@Q05_g_6:2m@\) en 6. Le maximum sur les réels est en 5, à égale distance de 4 et de 6, mais la fonction n'est pas symétrique autour de 5 : il faut comparer les deux valeurs."},
            {"id": "d", "texte": "8", "explication": r"\(N^5 e^{-N}\) vaut \(@Q05_g_8:2m@\) en 8 : au-delà de 5 l'exponentielle l'emporte sur la puissance, et l'a priori en \(1/N\) pénalise encore les grandes valeurs."},
        ],
        "solution": r"""Réponse : 6.

Lecture retenue. Taux constant et absence de mémoire : les crises forment un processus de Poisson, et le nombre \(K\) de crises en 10 ans suit une loi de Poisson. Son paramètre, le nombre moyen de crises en 10 ans, est inconnu : on le note \(\Lambda\) (variable aléatoire, puisqu'on lui donne une loi a priori), et \(n\) désigne une valeur possible. L'a priori est \(\mathbb{P}(\Lambda = n) = \dfrac{1}{h\,n}\), où \(h\) est la constante de normalisation, somme des \(1/n\) sur les valeurs possibles.

Étape 1, la probabilité jointe. Par définition d'une probabilité conditionnelle, \(\mathbb{P}(\Lambda = n,\ K = 6) = \mathbb{P}(\Lambda = n)\ \mathbb{P}(K = 6 \mid \Lambda = n) = \dfrac{1}{h\,n} \cdot \dfrac{e^{-n}\,n^6}{6!} = \dfrac{n^5 e^{-n}}{h \cdot 6!}\).

Étape 2. Le dénominateur \(h \cdot 6!\) ne dépend pas de \(n\) : il suffit de comparer \(g(n) = n^5 e^{-n}\) sur les quatre valeurs.

| \(n\) | 2 | 4 | 6 | 8 |
|---|---|---|---|---|
| \(g(n) = n^5 e^{-n}\) | @Q05_g_2:2@ | @Q05_g_4:2@ | @Q05_g_6:2@ | @Q05_g_8:2@ |

Le maximum est en \(n = 6\).

Étape 3, contrôle sans calculatrice. \(g'(n) = n^4 e^{-n}\,(5 - n)\) : \(g\) croît jusqu'à 5 puis décroît, donc seuls 4 et 6 sont candidats. Leur rapport est \(g(6)/g(4) = (3/2)^5\,e^{-2} = @Q05_rapport_6_4:3@\), supérieur à 1 : 6 l'emporte, de peu.

Vérification (script). Si les valeurs possibles sont exactement 2, 4, 6, 8, alors \(h = @Q05_H_num@/@Q05_H_den@\) et les probabilités jointes valent @Q05_jointe_2:5@, @Q05_jointe_4:5@, @Q05_jointe_6:5@ et @Q05_jointe_8:5@ ; simulation : @Q05_jointe_2_mc:5@, @Q05_jointe_4_mc:5@, @Q05_jointe_6_mc:5@ et @Q05_jointe_8_mc:5@ (erreur standard @Q05_jointe_6_se:5@ pour la troisième).

Ce qui est ambigu dans la question d'origine, et ce que cela change :

1. « Toutes les valeurs possibles de \(N\) » n'est pas précisé. Si c'étaient tous les entiers strictement positifs, la somme des \(1/N\) serait infinie (série harmonique) et la normalisation impossible : l'ensemble doit être fini. La réponse n'en dépend pas, puisque \(h\) se simplifie.
2. L'unité du taux n'est pas dite. La lecture retenue (les options sont des nombres moyens de crises **en 10 ans**) est celle que suggère la phrase sur l'a priori. Si les options étaient des taux **par an**, la moyenne sur 10 ans serait dix fois l'option, et la probabilité jointe, proportionnelle à \((10\lambda)^6 e^{-10\lambda}/\lambda\) pour un taux annuel \(\lambda\), serait maximale pour l'option @Q05_alt@.
3. L'a priori est attribué au « nombre de crises », alors qu'il doit porter sur le taux pour que la question ait un sens : c'est la lecture faite ici.

Sans a priori (maximum de vraisemblance seul), la réponse serait aussi @Q05_vrais_max@ : la vraisemblance \(e^{-n} n^6/6!\) est maximale en \(n = 6\).""",
    },
    {
        "id": "T9-Q06", "section": "math", "theme": "Asymétrie d'une distribution", "difficulte": 2, "temps_cible_s": 60,
        "correctes": ["a", "c"], "ordre_origine": ["a", "b", "c", "d"],
        "enonce": r"""Question 6. Pour une distribution asymétrique à droite (asymétrie positive), quelles affirmations sont correctes dans la plupart des cas ?""",
        "options": [
            {"id": "a", "texte": "La moyenne est supérieure au mode.",
             "explication": r"Juste dans le cas type (une seule bosse, queue longue à droite) : la queue tire la moyenne vers la droite du sommet. Loi log-normale : mode \(@Q06_ln_mode:3m@\), moyenne \(@Q06_ln_moyenne:3m@\)."},
            {"id": "b", "texte": "La moyenne est inférieure au mode.",
             "explication": "C'est la situation type d'une asymétrie à gauche (queue longue vers les petites valeurs), pas à droite."},
            {"id": "c", "texte": "La moyenne est supérieure à la médiane.",
             "explication": r"Juste dans le cas type : la médiane ne dépend que des rangs, la moyenne est tirée par les grandes valeurs. Loi log-normale : médiane \(@Q06_ln_mediane:0m@\), moyenne \(@Q06_ln_moyenne:3m@\)."},
            {"id": "d", "texte": "La moyenne est inférieure à la médiane.",
             "explication": "C'est la situation type d'une asymétrie à gauche. Elle peut se produire avec une asymétrie positive (contre-exemple de la correction), mais ce n'est pas le cas type que vise l'énoncé."},
        ],
        "solution": r"""Réponse : « la moyenne est supérieure au mode » et « la moyenne est supérieure à la médiane ».

Définitions. Pour une variable aléatoire \(X\) de moyenne \(\mu\) et d'écart-type \(\sigma\), le coefficient d'asymétrie est \(\mathbb{E}[(X - \mu)^3] / \sigma^3\). Il est positif quand les grands écarts au-dessus de la moyenne pèsent plus que ceux d'en dessous : queue longue à droite. Le mode est la valeur la plus probable (le sommet de la densité), la médiane la valeur qui partage la probabilité en deux moitiés.

Le cas type : une densité à une seule bosse, avec une queue longue à droite. L'ordre est alors mode, médiane, moyenne. Deux lois où il se calcule exactement (script) :

- loi log-normale de paramètres 0 et 1 : mode \(e^{-1} = @Q06_ln_mode:3m@\), médiane \(@Q06_ln_mediane:0m@\), moyenne \(e^{1/2} = @Q06_ln_moyenne:3m@\), asymétrie \(@Q06_ln_asym:2m@\) ; simulation de la moyenne : @Q06_ln_moyenne_mc:3@ ;
- loi exponentielle de paramètre 1 : mode 0, médiane \(\ln 2 = @Q06_exp_mediane:3m@\), moyenne 1.

Pourquoi « dans la plupart des cas » et pas « toujours ». Ce n'est pas un théorème : le signe de l'asymétrie ne fixe pas l'ordre de la moyenne et de la médiane. Contre-exemple (script) : \(X\) vaut 0, 1 ou 5 avec probabilités @Q06_ce_p0:2@, @Q06_ce_p1:2@ et @Q06_ce_p5:2@. Son asymétrie vaut \(@Q06_ce_asym:2m@\), positive à cause de la valeur 5, mais sa moyenne \(@Q06_ce_moyenne:2m@\) est **inférieure** à sa médiane et à son mode, tous deux égaux à @Q06_ce_mediane@.

La réponse retenue suit donc la formulation de l'énoncé : ce qui est vrai dans le cas type.""",
    },
    {
        "id": "T9-Q07", "section": "math", "theme": "Statistiques d'ordre, corde coupée en trois", "difficulte": 4, "temps_cible_s": 210,
        "ordre_origine": ["b", "c", "d", "a"],
        "enonce": r"""Question 7. Une corde de longueur 1 est coupée au hasard en trois morceaux. Quelle est l'espérance de la longueur du morceau le plus **court** ?""",
        "options": [
            {"id": "a", "texte": "@Q07_exact_num@/@Q07_exact_den@", "explication": r"Oui : \(\mathbb{P}(\text{le plus court} > t) = (1 - 3t)^2\), d'intégrale \(@Q07_exact_num@/@Q07_exact_den@\) sur \([0, 1/3]\)."},
            {"id": "b", "texte": "@Q07_d_moyen_num@/@Q07_d_moyen_den@", "explication": "C'est l'espérance de la longueur d'un morceau désigné à l'avance (les trois ont la même loi et leur somme vaut 1). C'est aussi la plus grande valeur possible du morceau le plus court : son espérance est forcément plus petite."},
            {"id": "c", "texte": "@Q07_d_sixieme_num@/@Q07_d_sixieme_den@", "explication": r"C'est la longueur moyenne 1/3 divisée par 2. Pour trois morceaux, l'espérance du plus court est la longueur moyenne 1/3 divisée par 3, pas par 2 (pour \(m\) morceaux : \(1/m^2\), voir la correction)."},
            {"id": "d", "texte": "@Q07_d_huitieme_num@/@Q07_d_huitieme_den@", "explication": r"Valeur proche de la bonne, mais fausse : la simulation donne @Q07_exact_mc:4@ avec une erreur standard de @Q07_exact_se:5@, incompatible avec @Q07_d_huitieme:3@. Pour comparaison, avec une seule coupe (deux morceaux) l'espérance du plus court est \(@Q07_deux_morceaux_num@/@Q07_deux_morceaux_den@\)."},
        ],
        "solution": r"""Réponse : \(@Q07_exact_num@/@Q07_exact_den@\).

Modèle retenu (l'énoncé d'origine dit seulement « au hasard ») : deux points de coupe \(U_1\) et \(U_2\), indépendants, de loi uniforme sur \([0, 1]\). On note \(A = \min(U_1, U_2)\) et \(B = \max(U_1, U_2)\) ; les morceaux ont pour longueurs \(A\), \(B - A\) et \(1 - B\) ; \(L\) est la plus petite des trois.

Étape 1, loi du couple \((A, B)\). Le couple \((U_1, U_2)\) est uniforme sur le carré unité, d'aire 1. Le couple \((A, B)\) est donc uniforme sur le triangle \(\{0 \le a \le b \le 1\}\), d'aire 1/2 : chaque point du triangle provient de deux points du carré, symétriques par rapport à la diagonale.

Étape 2, \(\mathbb{P}(L > t)\) pour \(0 \le t \le 1/3\). L'événement \(L > t\) s'écrit \(a > t\), \(b - a > t\), \(1 - b > t\). Le changement de variables \(a' = a - t\), \(b' = b - 2t\) est une translation, qui conserve les aires ; il transforme ces trois conditions en \(0 < a' < b' < 1 - 3t\), triangle d'aire \((1 - 3t)^2/2\). En divisant par l'aire 1/2 du triangle total : \(\mathbb{P}(L > t) = (1 - 3t)^2\). Pour \(t > 1/3\), la probabilité est nulle, car trois morceaux plus longs que 1/3 auraient une somme supérieure à 1.

Étape 3, espérance. Pour une variable positive, \(\mathbb{E}[L] = \int_0^{\infty} \mathbb{P}(L > t)\,dt\) (formule de la queue : on écrit \(L = \int_0^{\infty} I_{\{L > t\}}\,dt\) et on échange espérance et intégrale, par le théorème de Fubini, admis). Donc \(\mathbb{E}[L] = \int_0^{1/3} (1 - 3t)^2\,dt = \Big[-\dfrac{(1 - 3t)^3}{9}\Big]_0^{1/3} = \dfrac{1}{9}\).

Vérifications (script) : intégrale double exacte pour \(\mathbb{P}(L > t)\), et simulation : @Q07_exact_mc:4@ (erreur standard @Q07_exact_se:5@). Pour \(m\) morceaux (\(m - 1\) coupes), le même argument de translation donne \(\mathbb{P}(L > t) = (1 - m\,t)^{m-1}\) et \(\mathbb{E}[L] = 1/m^2\) ; ce cas général n'est pas rédigé ici, il est contrôlé par simulation pour \(m\) de 2 à @Q07_general_max@ (script). Les deux autres morceaux ont pour espérances \(@Q07_milieu_num@/@Q07_milieu_den@\) (le moyen) et \(@Q07_max_num@/@Q07_max_den@\) (le plus long) ; la somme des trois vaut bien 1.

Ce qui est ambigu dans la question d'origine : « coupée au hasard » ne dit pas comment. Le modèle des deux points uniformes indépendants est la convention usuelle, et il donne une des quatre options. Un autre procédé donne une autre valeur : par exemple, couper une première fois uniformément, puis recouper uniformément le morceau de droite, donne par simulation @Q07_sequentiel_mc:4@ (erreur standard @Q07_sequentiel_se:5@), qui n'est aucune des options.""",
    },
    {
        "id": "T9-Q08", "section": "math", "theme": "Densité conditionnelle", "difficulte": 3, "temps_cible_s": 180,
        "ordre_origine": ["b", "c", "d", "a"],
        "enonce": r"""Question 8. La densité de probabilité d'un vecteur aléatoire \((X, Y)\) de dimension 2 est

\(\displaystyle f_{XY}(x, y) = \frac{e^{-y/x}\,e^{-x}}{x}\) pour \(0 < x < +\infty\) et \(0 < y < +\infty\), et 0 ailleurs.

Que vaut \(\mathbb{P}(Y > 1 \mid X = x)\), pour \(x > 0\) ?""",
        "options": [
            {"id": "a", "texte": r"\(e^{-1/x}\)", "explication": r"Oui : sachant \(X = x\), \(Y\) suit une loi exponentielle de moyenne \(x\), dont la queue en 1 est \(e^{-1/x}\)."},
            {"id": "b", "texte": r"\(e^{-x - 1/x}\)", "explication": r"C'est \(\int_1^{\infty} f_{XY}(x, y)\,dy\) : on a intégré la densité **jointe** sans la diviser par la densité marginale \(f_X(x) = e^{-x}\). Pour \(x = 2\) : \(@Q08_d_joint_x2:3m@\) au lieu de \(@Q08_val_x2:3m@\)."},
            {"id": "c", "texte": r"\(e^{1/x}\)", "explication": r"Erreur de signe dans l'exposant : cette quantité dépasse 1 pour tout \(x > 0\) (\(@Q08_d_plus_x2:3m@\) pour \(x = 2\)), ce ne peut pas être une probabilité."},
            {"id": "d", "texte": r"\(e^{1/y}\)", "explication": r"La réponse ne peut pas dépendre de \(y\) : \(y\) est la variable d'intégration, elle disparaît quand on calcule la probabilité de l'événement \(Y > 1\). Seul \(x\), la valeur sur laquelle on conditionne, peut rester."},
        ],
        "solution": r"""Réponse : \(e^{-1/x}\).

Notations : \(f_X\) est la densité marginale de \(X\) ; \(f_{Y \mid X}(y \mid x) = f_{XY}(x, y) / f_X(x)\) est la densité conditionnelle de \(Y\) sachant \(X = x\), définie quand \(f_X(x) > 0\).

Étape 1, densité marginale. Pour \(x > 0\), \(f_X(x) = \int_0^{\infty} f_{XY}(x, y)\,dy = \dfrac{e^{-x}}{x} \int_0^{\infty} e^{-y/x}\,dy = \dfrac{e^{-x}}{x} \cdot x = e^{-x}\), car \(\int_0^{\infty} e^{-y/x}\,dy = x\). \(X\) suit donc la loi exponentielle de paramètre 1, et l'intégrale totale de \(f_{XY}\) vaut bien 1.

Étape 2, densité conditionnelle. \(f_{Y \mid X}(y \mid x) = \dfrac{e^{-y/x}\,e^{-x}/x}{e^{-x}} = \dfrac{1}{x}\,e^{-y/x}\) pour \(y > 0\) : c'est la densité de la loi exponentielle de moyenne \(x\).

Étape 3. \(\mathbb{P}(Y > 1 \mid X = x) = \int_1^{\infty} \dfrac{1}{x}\,e^{-y/x}\,dy = \big[-e^{-y/x}\big]_1^{\infty} = e^{-1/x}\).

Contrôles de bon sens : quand \(x\) tend vers 0, la moyenne de \(Y\) tend vers 0 et la probabilité tend vers 0 ; quand \(x\) grandit, elle tend vers 1.

Vérifications (script) : calcul symbolique des trois étapes ; pour \(x = @Q08_x_ex@\), \(e^{-1/2} = @Q08_val_x2:4m@\), et l'intégration numérique de la densité jointe sur la bande \(1{,}9 < x < 2{,}1\) donne @Q08_bande:4@ (simulation : @Q08_bande_mc:4@, erreur standard @Q08_bande_se:4@) ; sans conditionner, \(\mathbb{P}(Y > 1) = @Q08_p_y1:4m@\) (simulation : @Q08_p_y1_mc:4@).

Remarque sur l'énoncé : la fin de la ligne qui donne le domaine de \(y\) manque dans la version disponible. Le domaine \(0 < y < +\infty\) est la lecture retenue ; c'est avec lui que la densité s'intègre à 1.""",
    },
    {
        "id": "T9-Q09", "section": "math", "theme": "Densité jointe, constante de normalisation", "difficulte": 2, "temps_cible_s": 120,
        "ordre_origine": ["b", "a", "c", "d"],
        "enonce": r"""Question 9. On considère la densité de probabilité jointe \(f(x, y) = c\,x^2\,(1 + y)\) pour \(0 \le x \le 3\) et \(0 \le y \le 3\) (et 0 ailleurs), où \(c\) est une constante inconnue. Que vaut \(c\) ?""",
        "options": [
            {"id": "a", "texte": "@Q09_exact_num@/@Q09_exact_den@", "explication": r"Oui : l'intégrale de \(x^2 (1 + y)\) sur le carré vaut \(@Q09_ix@ \times @Q09_iy_num@/@Q09_iy_den@ = @Q09_total_num@/@Q09_total_den@\), et \(c\) en est l'inverse."},
            {"id": "b", "texte": "@Q09_d_97_num@/@Q09_d_97_den@", "explication": r"Avec cette valeur, l'intégrale de la densité sur le carré vaudrait \(@Q09_d_97_masse_num@/@Q09_d_97_masse_den@ \approx @Q09_d_97_masse:3m@\) au lieu de 1."},
            {"id": "c", "texte": "@Q09_d_65_num@/@Q09_d_65_den@", "explication": r"Avec cette valeur, l'intégrale de la densité vaudrait \(@Q09_d_65_masse_num@/@Q09_d_65_masse_den@ \approx @Q09_d_65_masse:3m@\) au lieu de 1. L'écriture ressemble à la bonne (même numérateur 2) : il faut finir le produit \(@Q09_ix@ \times @Q09_iy_num@/@Q09_iy_den@\) avant de choisir."},
            {"id": "d", "texte": "@Q09_d_10_num@/@Q09_d_10_den@", "explication": r"Avec cette valeur, l'intégrale de la densité vaudrait \(@Q09_d_10_masse_num@/@Q09_d_10_masse_den@ = @Q09_d_10_masse:2m@\) : la masse totale serait près de 7 fois trop grande."},
        ],
        "solution": r"""Réponse : \(c = @Q09_exact_num@/@Q09_exact_den@\).

Une densité est positive et d'intégrale 1 : \(c\) est l'inverse de l'intégrale de \(x^2 (1 + y)\) sur le carré \([0, 3]^2\).

Étape 1. La fonction est un produit d'une fonction de \(x\) par une fonction de \(y\) et le domaine est un rectangle : l'intégrale double est le produit des deux intégrales simples (théorème de Fubini, admis).

Étape 2. \(\int_0^3 x^2\,dx = \Big[\dfrac{x^3}{3}\Big]_0^3 = @Q09_ix@\) et \(\int_0^3 (1 + y)\,dy = \Big[y + \dfrac{y^2}{2}\Big]_0^3 = 3 + \dfrac{9}{2} = \dfrac{@Q09_iy_num@}{@Q09_iy_den@}\).

Étape 3. Produit : \(@Q09_ix@ \times \dfrac{@Q09_iy_num@}{@Q09_iy_den@} = \dfrac{@Q09_total_num@}{@Q09_total_den@} = @Q09_total:1m@\), donc \(c = \dfrac{@Q09_exact_num@}{@Q09_exact_den@}\).

Vérification (script) : intégrales exactes, et estimation de l'intégrale par tirages uniformes dans le carré : @Q09_total_mc:2@ (erreur standard @Q09_total_se:2@), compatible avec @Q09_total:1@.

Contrôle rapide à faire en test : multiplier l'option choisie par \(@Q09_total_num@/@Q09_total_den@\) doit donner 1.""",
    },
    {
        "id": "T9-Q10", "section": "math", "theme": "Probabilités conditionnelles, deux enfants", "difficulte": 2, "temps_cible_s": 90,
        "ordre_origine": ["b", "a", "c", "d"],
        "enonce": r"""Question 10. Affirmation 1 : A a deux enfants. L'aîné est une fille. La probabilité que les deux enfants soient des filles est \(p_1\).

Affirmation 2 : B a deux enfants. Au moins l'un d'eux est un garçon. La probabilité que les deux enfants soient des garçons est \(p_2\).

Quelle relation lie \(p_1\) et \(p_2\) ?""",
        "options": [
            {"id": "a", "texte": r"\(p_1 > p_2\)", "explication": r"Oui : \(p_1 = @Q10_p1_num@/@Q10_p1_den@\) et \(p_2 = @Q10_p2_num@/@Q10_p2_den@\)."},
            {"id": "b", "texte": r"\(p_1 = p_2\)", "explication": r"C'est traiter « au moins un garçon » comme « tel enfant précis est un garçon ». Désigner un enfant laisse @Q10_cas1@ cas équiprobables ; « au moins un » en laisse @Q10_cas2@, dont un seul favorable."},
            {"id": "c", "texte": r"\(p_2 > p_1\)", "explication": r"L'ordre est inversé : l'information « au moins un garçon » est moins précise que « l'aîné est une fille », elle laisse plus de cas possibles (@Q10_cas2@ contre @Q10_cas1@) pour un seul cas favorable, donc une probabilité plus petite."},
            {"id": "d", "texte": "On ne peut pas le déterminer.", "explication": r"Avec les conventions usuelles (sexes indépendants et équiprobables, conditionnement par l'événement énoncé), les deux probabilités se calculent : \(@Q10_p1_num@/@Q10_p1_den@\) et \(@Q10_p2_num@/@Q10_p2_den@\). Cette option ne se défend que si l'on refuse de fixer la façon dont l'information a été obtenue (fin de la correction)."},
        ],
        "solution": r"""Réponse : \(p_1 > p_2\), avec \(p_1 = @Q10_p1_num@/@Q10_p1_den@\) et \(p_2 = @Q10_p2_num@/@Q10_p2_den@\).

Hypothèses, non écrites dans la question d'origine mais nécessaires : chaque enfant est une fille (F) ou un garçon (G) avec probabilité 1/2, indépendamment de l'autre ; et l'on conditionne par l'événement énoncé. Les couples (aîné, cadet) FF, FG, GF, GG sont alors équiprobables, de probabilité 1/4.

Étape 1. « L'aîné est une fille » : restent FF et FG, soit @Q10_cas1@ cas équiprobables, dont FF. Donc \(p_1 = 1/2\).

Étape 2. « Au moins un garçon » : restent FG, GF et GG, soit @Q10_cas2@ cas équiprobables, dont GG. Donc \(p_2 = \dfrac{1/4}{3/4} = \dfrac{1}{3}\).

Vérification (script) : énumération exacte et simulation, @Q10_p1_mc:4@ et @Q10_p2_mc:4@.

Ce qui est ambigu dans la question d'origine : la façon dont on apprend « au moins l'un d'eux est un garçon ». Si quelqu'un qui connaît les deux enfants répond « oui » à la question « y a-t-il au moins un garçon ? », on conditionne par l'événement et \(p_2 = 1/3\). Si l'on a aperçu un enfant pris au hasard parmi les deux et que c'est un garçon, la formule de Bayes donne \(p_2 = @Q10_p2_alt_num@/@Q10_p2_alt_den@\) (simulation : @Q10_p2_alt_mc:4@), et l'on aurait \(p_1 = p_2\). La lecture retenue est la première, celle du conditionnement par l'événement tel qu'il est écrit.""",
    },
    {
        "id": "T9-Q11", "section": "math", "theme": "Écart réduit, comparaison de notes", "difficulte": 1, "temps_cible_s": 75,
        "ordre_origine": ["b", "c", "a"],
        "enonce": r"""Question 11. Un étudiant a obtenu les notes suivantes à deux examens. On suppose que, dans chaque matière, les notes de l'ensemble des candidats suivent une loi normale, de moyenne et d'écart-type donnés dans le tableau.

| Matière | Note obtenue (sur 100) | Moyenne | Écart-type |
|---|---|---|---|
| Mathématiques | 70 | 60 | 15 |
| Sciences | 72 | 68 | 6 |

Dans quelle matière sa performance relative est-elle la meilleure ?""",
        "options": [
            {"id": "a", "texte": "La performance est la même dans les deux matières.", "explication": r"Oui : les deux écarts réduits valent \(@Q11_z_maths_num@/@Q11_z_maths_den@\)."},
            {"id": "b", "texte": "Mathématiques", "explication": r"C'est comparer les écarts bruts à la moyenne, @Q11_ecart_maths@ points contre @Q11_ecart_sciences@. Mais les notes de mathématiques sont plus dispersées (écart-type 15 contre 6) : rapportés à l'écart-type, les deux écarts sont égaux."},
            {"id": "c", "texte": "Sciences", "explication": "C'est comparer les notes brutes, 72 contre 70. Une performance relative se mesure par la position dans la distribution de chaque matière, pas par la note elle-même."},
        ],
        "solution": r"""Réponse : la performance est la même dans les deux matières.

Notations : pour une note \(x\), dans une matière de moyenne \(\mu\) et d'écart-type \(\sigma\), l'écart réduit (ou cote z) est \(z = (x - \mu)/\sigma\) : le nombre d'écarts-types qui séparent la note de la moyenne.

Étape 1. Mathématiques : \(z = (70 - 60)/15 = @Q11_z_maths_num@/@Q11_z_maths_den@\). Sciences : \(z = (72 - 68)/6 = @Q11_z_sciences_num@/@Q11_z_sciences_den@\).

Étape 2. Si la note \(X\) d'un candidat suit la loi normale de moyenne \(\mu\) et d'écart-type \(\sigma\), alors \((X - \mu)/\sigma\) suit la loi normale centrée réduite, quelle que soit la matière. La proportion de candidats moins bien notés que l'étudiant est donc \(\Phi(z)\) dans les deux cas, où \(\Phi\) est la fonction de répartition de la loi normale centrée réduite : \(\Phi(2/3) = @Q11_rang:4m@\), soit le même rang dans les deux matières (environ @Q11_rang_pct:1@ % des candidats derrière lui).

Vérification (script) : simulation de la proportion de notes inférieures, @Q11_rang_maths_mc:4@ en mathématiques et @Q11_rang_sciences_mc:4@ en sciences.

L'hypothèse de normalité sert à l'étape 2 : sans elle, deux écarts réduits égaux ne garantissent pas le même rang.""",
    },
    # =============================================================== ML FONDAMENTAL (questions 12 à 16)
    {
        "id": "T9-Q12", "section": "mlf", "theme": "Réseau convolutif : formes des tenseurs, concaténation", "difficulte": 3, "temps_cible_s": 150,
        "ordre_origine": ["b", "a", "c", "d"],
        "enonce": r"""Question 12. Un réseau de neurones convolutif enchaîne les couches décrites par une figure et par une liste. Les deux sont reproduites telles qu'elles apparaissent dans la question d'origine.

La figure, de gauche à droite : une entrée (256, 256, 3) ; un bloc « Conv 2D, 64, (3 x 3) » ; un bloc « Conv 2D, 128, (3 x 3) » ; un bloc de concaténation, qui reçoit la sortie du deuxième bloc de convolution et, par une flèche de contournement, la sortie du premier ; un bloc « Max. Pool / 2 » ; la sortie.

La liste :

- l'entrée, de taille 256 x 256 x 3 ;
- convolution 2D, 64 filtres 3 x 3 ;
- convolution 2D, 64 filtres 3 x 3 ;
- concaténation selon l'axe de la profondeur ;
- max pooling, filtre 2 x 2, pas (stride) de 2.

Quelle est la forme de la sortie (largeur x hauteur x profondeur) ?""",
        "options": [
            {"id": "a", "texte": "@Q12_cote@ x @Q12_cote@ x @Q12_prof_figure@", "explication": r"Oui, en suivant la figure et en supposant que les convolutions conservent la hauteur et la largeur : concaténation de 64 et 128 canaux, puis division par 2 de chaque côté."},
            {"id": "b", "texte": "Aucune sortie, à cause d'une incompatibilité de formes entre tenseurs dans le réseau.", "explication": r"La concaténation en profondeur exige la même hauteur et la même largeur, pas le même nombre de canaux : 64 et 128 canaux se concatènent sans difficulté. Cette option ne devient juste que si les convolutions n'ont aucune marge (tailles @Q12_valid_1@ puis @Q12_valid_2@), ce que la question d'origine ne précise pas."},
            {"id": "c", "texte": "@Q12_avant_pool@ x @Q12_avant_pool@ x 128", "explication": "Deux oublis : la concaténation (seuls les 128 canaux de la deuxième convolution sont comptés) et le pooling, qui divise la hauteur et la largeur par 2."},
            {"id": "d", "texte": "@Q12_cote@ x @Q12_cote@ x 128", "explication": r"En suivant la figure, c'est oublier la concaténation : le pooling reçoit \(64 + 128\) canaux, pas 128. C'est en revanche la réponse si l'on suit la liste écrite (deux convolutions à 64 filtres, \(64 + 64\)) plutôt que la figure."},
        ],
        "solution": r"""Réponse : @Q12_cote@ x @Q12_cote@ x @Q12_prof_figure@, en suivant la figure et en supposant que les convolutions conservent la hauteur et la largeur. La question d'origine contient une incohérence et une omission, détaillées à la fin.

Règles de calcul. On note \(n\) la taille d'entrée le long d'un axe spatial (hauteur ou largeur), \(k\) la taille du filtre, \(p\) le nombre de zéros ajoutés de chaque côté (la marge, padding) et \(s\) le pas (stride). La taille de sortie le long de cet axe est \(\lfloor (n + 2p - k)/s \rfloor + 1\) : le filtre occupe \(k\) positions, il reste \(n + 2p - k\) décalages possibles, dont on garde un sur \(s\), plus la position de départ. La profondeur de sortie d'une convolution est son nombre de filtres, quelle que soit la profondeur d'entrée. La concaténation en profondeur exige des hauteurs et des largeurs égales, et additionne les profondeurs. Le pooling agit canal par canal : il ne change pas la profondeur.

Étape 1, première convolution, 64 filtres 3 x 3. Avec une marge de 1 et un pas de 1 (convolution dite « same ») : \((256 + 2 - 3)/1 + 1 = 256\). Sortie 256 x 256 x 64.

Étape 2, deuxième convolution, 128 filtres selon la figure : 256 x 256 x 128.

Étape 3, concaténation des sorties des étapes 1 et 2 : mêmes hauteur et largeur, profondeur \(64 + 128 = @Q12_prof_figure@\). Sortie 256 x 256 x @Q12_prof_figure@.

Étape 4, max pooling 2 x 2 de pas 2, sans marge : \((256 - 2)/2 + 1 = @Q12_cote@\). Sortie @Q12_cote@ x @Q12_cote@ x @Q12_prof_figure@.

Vérification (script) : la formule, puis une vraie convolution numpy sur un modèle réduit (entrée @Q12_petit_n@ x @Q12_petit_n@ x 3, @Q12_petit_f1@ puis @Q12_petit_f2@ filtres) : sortie @Q12_petit_cote@ x @Q12_petit_cote@ x @Q12_petit_prof@, comme le prévoit la formule.

Ce qui ne va pas dans la question d'origine, et ce que cela change :

1. **La figure et la liste se contredisent.** La figure donne 128 filtres à la deuxième convolution, la liste 64. Avec la liste : \(64 + 64 = @Q12_prof_liste@\) canaux, et la sortie serait @Q12_cote@ x @Q12_cote@ x @Q12_prof_liste@, qui figure aussi parmi les options.
2. **La marge des convolutions n'est pas donnée.** Sans marge (\(p = 0\)), la première convolution sort en \(256 - 3 + 1 = @Q12_valid_1@\) de côté et la deuxième en @Q12_valid_2@ : les deux tenseurs à concaténer n'ont plus la même taille et la concaténation échoue (constaté sur le modèle réduit : @Q12_petit_valid_1@ contre @Q12_petit_valid_2@, erreur de numpy). La réponse serait alors « aucune sortie ».

Pourquoi retenir @Q12_cote@ x @Q12_cote@ x @Q12_prof_figure@ : toutes les options chiffrées ont des côtés de 256 ou de 128, ce qui suppose des convolutions qui conservent la taille ; et les trois options chiffrées se lisent comme la bonne réponse et deux oublis à partir de la figure (oublier la concaténation, oublier aussi le pooling). C'est un raisonnement sur l'intention de l'auteur, pas une preuve : la réponse attendue par le correcteur n'est pas connue.""",
    },
    {
        "id": "T9-Q13", "section": "mlf", "theme": "Réglages d'entraînement d'une segmentation", "difficulte": 2, "temps_cible_s": 120,
        "ordre_origine": ["a", "b", "c", "d"],
        "enonce": r"""Question 13. On prépare un modèle de segmentation d'images, qui classe chaque pixel dans l'une de 2 catégories (objet ou fond). On dispose de 2000 paires formées d'une image d'entrée et de son masque (l'étiquette de chaque pixel).

Quel réglage d'entraînement relève de la bonne pratique dans cette situation ?""",
        "options": [
            {"id": "a", "texte": "Entraînement : 1600 paires, validation : 200, test : 200 ; 100 pas par époque ; lots de 16 ; pas d'apprentissage 0,001 ; perte : entropie croisée binaire.",
             "explication": r"Oui : découpage @Q13_part_train_A:0@ % / @Q13_part_val_A:0@ % / @Q13_part_test:0@ %, \(100 \times 16 = @Q13_vus_A@\) paires par époque, soit exactement l'ensemble d'entraînement, et une perte adaptée à deux classes."},
            {"id": "b", "texte": "Entraînement : 1600 paires, validation : 200, test : 200 ; 150 pas par époque ; lots de 10 ; pas d'apprentissage 0,01 ; perte : entropie croisée catégorielle.",
             "explication": r"Le découpage est bon, mais \(150 \times 10 = @Q13_vus_B@\) : chaque époque laisse de côté @Q13_manque_B@ paires d'entraînement, et la perte catégorielle suppose un masque codé sur deux canaux, ce qui n'est pas le codage naturel d'un masque objet ou fond."},
            {"id": "c", "texte": "Entraînement : 1000 paires, validation : 800, test : 200 ; 100 pas par époque ; lots de 8 ; pas d'apprentissage 0,001 ; perte : entropie croisée binaire.",
             "explication": r"La perte est adaptée, mais @Q13_part_val_C:0@ % des données partent en validation au lieu de servir à l'apprentissage, et \(100 \times 8 = @Q13_vus_C@\) : une époque ne parcourt même pas les 1000 paires d'entraînement (il en manque @Q13_manque_C@)."},
            {"id": "d", "texte": "Entraînement : 1000 paires, validation : 800, test : 200 ; 50 pas par époque ; lots de 20 ; pas d'apprentissage 0,01 ; perte : entropie croisée catégorielle.",
             "explication": r"Ici \(50 \times 20 = @Q13_vus_D@\) couvre bien l'ensemble d'entraînement, mais celui-ci ne contient que @Q13_part_train_D:0@ % des données, et la perte catégorielle n'est pas le choix direct pour deux classes."},
        ],
        "solution": r"""Réponse : 1600 / 200 / 200, 100 pas par époque, lots de 16, pas d'apprentissage 0,001, entropie croisée binaire.

Vocabulaire. Un lot (batch) est le groupe d'exemples utilisé pour un pas de gradient. Une époque est un passage sur l'ensemble d'entraînement : le nombre de pas par époque doit donc valoir (taille de l'ensemble d'entraînement) / (taille du lot). L'ensemble de validation sert à choisir les réglages et à arrêter l'entraînement, l'ensemble de test à estimer la performance finale, une seule fois.

Critère 1, le découpage. Les quatre réglages gardent 200 paires de test (@Q13_part_test:0@ %). Deux n'entraînent que sur 1000 paires et en consacrent 800 à la validation : la validation n'a pas besoin de @Q13_part_val_C:0@ % des données pour comparer des réglages, et le modèle perd 600 exemples d'apprentissage par rapport au découpage 1600 / 200 / 200.

Critère 2, la cohérence pas par époque fois taille de lot (script) :

| Réglage | Entraînement | Pas par époque x lot | Écart |
|---|---|---|---|
| 1600, 100 pas, lots de 16 | 1600 | @Q13_vus_A@ | 0 |
| 1600, 150 pas, lots de 10 | 1600 | @Q13_vus_B@ | @Q13_manque_B@ paires jamais vues dans l'époque |
| 1000, 100 pas, lots de 8 | 1000 | @Q13_vus_C@ | @Q13_manque_C@ paires jamais vues dans l'époque |
| 1000, 50 pas, lots de 20 | 1000 | @Q13_vus_D@ | 0 |

Critère 3, la perte. Avec deux classes, le masque est un seul canal de 0 et de 1, la sortie du réseau une probabilité par pixel (sigmoïde), et la perte adaptée est l'entropie croisée binaire. L'entropie croisée catégorielle attend un masque codé sur autant de canaux que de classes.

Seul le premier réglage satisfait les trois critères.

Deux nuances, pour ne pas affirmer plus que ce qui est vrai :

- L'entropie croisée catégorielle n'est pas « fausse » pour deux classes : avec un masque codé sur deux canaux et une sortie softmax, elle est **égale** à l'entropie croisée binaire appliquée à la sigmoïde de la différence des deux scores (\(e^{s_1}/(e^{s_0} + e^{s_1}) = 1/(1 + e^{-(s_1 - s_0)})\), vérifié numériquement par le script). Elle demande seulement un codage que l'énoncé ne suggère pas.
- Le pas d'apprentissage ne départage pas les réglages à lui seul : 0,001 ou 0,01 peuvent convenir selon l'optimiseur. La question d'origine juge plusieurs critères à la fois sans dire lesquels comptent ; les trois critères ci-dessus désignent le même réglage, ce qui rend la réponse sûre.""",
    },
    {
        "id": "T9-Q14", "section": "mlf", "theme": "Matrice de confusion, rappel et précision", "difficulte": 2, "temps_cible_s": 90,
        "enonce": r"""Question de remplacement (la question d'origine n'a pas été capturée). Question 14. Un détecteur de fraude est évalué sur 1000 transactions, dont 50 sont réellement frauduleuses. Il en signale 80 comme frauduleuses ; parmi ces 80, 30 le sont réellement.

Que vaut le rappel (recall) du détecteur pour la classe « fraude » ?""",
        "options": [
            {"id": "a", "texte": "@Q14_rappel:1@", "explication": "Oui : 30 fraudes détectées sur les 50 fraudes réelles."},
            {"id": "b", "texte": "@Q14_precision:3@", "explication": "C'est la précision, 30 / 80 : la proportion de signalements qui sont de vraies fraudes. Le rappel se rapporte aux fraudes réelles, pas aux signalements."},
            {"id": "c", "texte": "@Q14_exactitude:2@", "explication": r"C'est l'exactitude (accuracy), \((30 + @Q14_vn@)/1000\), dominée par les @Q14_vn@ transactions honnêtes non signalées. Un détecteur qui ne signale jamais rien aurait déjà une exactitude de @Q14_tout_negatif:2@."},
            {"id": "d", "texte": "@Q14_f1:3@", "explication": "C'est le score F1, moyenne harmonique de la précision et du rappel. Il résume les deux, mais ce n'est pas le rappel."},
        ],
        "solution": r"""Réponse : @Q14_rappel:1@.

Notations : VP (vrais positifs) est le nombre de fraudes signalées, FP (faux positifs) le nombre de transactions honnêtes signalées, FN (faux négatifs) le nombre de fraudes non signalées, VN (vrais négatifs) le nombre de transactions honnêtes non signalées.

Étape 1, la matrice de confusion. VP = 30. FP = \(80 - 30 = @Q14_fp@\). FN = \(50 - 30 = @Q14_fn@\). VN = \(1000 - 30 - @Q14_fp@ - @Q14_fn@ = @Q14_vn@\).

Étape 2. Le rappel est la proportion des fraudes réelles qui sont détectées : VP / (VP + FN) = \(30/50 = @Q14_rappel:1m@\).

Pour comparaison : précision VP / (VP + FP) = \(30/80 = @Q14_precision:3m@\) ; exactitude (VP + VN) / 1000 = \(@Q14_exactitude:2m@\) ; score F1 = \(@Q14_f1_num@/@Q14_f1_den@ \approx @Q14_f1:3m@\).

Lecture probabiliste, vérifiée par simulation (script) : le rappel est la probabilité qu'une fraude tirée au hasard parmi les fraudes ait été signalée, @Q14_rappel_mc:3@ sur un million de tirages.

Avec 5 % de fraudes, l'exactitude est trompeuse : elle récompense surtout la classe majoritaire. C'est le rappel et la précision de la classe rare qu'il faut regarder.""",
    },
    {
        "id": "T9-Q15", "section": "mlf", "theme": "k plus proches voisins : rôle de k", "difficulte": 3, "temps_cible_s": 105,
        "enonce": r"""Question de remplacement (la question d'origine n'a pas été capturée). Question 15. Un classifieur des \(k\) plus proches voisins (distance euclidienne, vote à la majorité parmi les \(k\) voisins) est entraîné sur @Q15_n@ points deux à deux distincts : @Q15_n0@ de la classe 0 et @Q15_n1@ de la classe 1. On mesure son exactitude (proportion de prédictions correctes) **sur ces mêmes @Q15_n@ points** ; chaque point fait partie de ses propres voisins, à distance nulle.

Que valent l'exactitude pour \(k = 1\), puis pour \(k = @Q15_n@\) ?""",
        "options": [
            {"id": "a", "texte": "@Q15_k1:0@ pour k = 1, puis @Q15_kn:1@ pour k = @Q15_n@", "explication": r"Oui : avec \(k = 1\) chaque point se retrouve lui-même ; avec \(k = @Q15_n@\) tous les points votent, et la classe 0 gagne partout."},
            {"id": "b", "texte": "@Q15_k1:0@ pour k = 1, puis @Q15_d_minoritaire:1@ pour k = @Q15_n@", "explication": r"Le cas \(k = 1\) est juste, mais avec \(k = @Q15_n@\) le vote donne la classe **majoritaire**, qui est correcte pour @Q15_n0@ points sur @Q15_n@ ; @Q15_d_minoritaire:1@ est la part de la classe minoritaire."},
            {"id": "c", "texte": "@Q15_kn:1@ pour k = 1, puis @Q15_k1:0@ pour k = @Q15_n@", "explication": r"Les deux cas sont inversés : c'est le petit \(k\) qui colle parfaitement aux données d'entraînement, et le grand \(k\) qui les ignore au profit de la classe majoritaire."},
            {"id": "d", "texte": "@Q15_k1:0@ pour k = 1, puis @Q15_d_hasard:1@ pour k = @Q15_n@", "explication": r"@Q15_d_hasard:1@ serait l'exactitude d'un tirage à pile ou face entre deux classes équilibrées. Ici le vote est déterministe (@Q15_n0@ voix contre @Q15_n1@) et les classes ne sont pas équilibrées."},
        ],
        "solution": r"""Réponse : @Q15_k1:0@ pour \(k = 1\), puis @Q15_kn:1@ pour \(k = @Q15_n@\).

Étape 1, \(k = 1\). Le plus proche voisin d'un point d'entraînement est ce point lui-même, à distance 0 ; aucun autre point n'est à distance 0 puisque les points sont distincts. La prédiction est donc l'étiquette du point : exactitude 1 sur l'ensemble d'entraînement, quelle que soit la qualité du modèle.

Étape 2, \(k = @Q15_n@\). Les \(k\) voisins de n'importe quel point sont les @Q15_n@ points : le vote est toujours @Q15_n0@ voix pour la classe 0 contre @Q15_n1@, et le classifieur prédit 0 partout. Il a raison sur les @Q15_n0@ points de classe 0 : exactitude \(@Q15_n0@/@Q15_n@ = @Q15_kn:1m@\).

Vérification (script) : implémentation numpy sur @Q15_n@ points gaussiens, exactitude @Q15_k1:0@ pour \(k = 1\), @Q15_k15:3@ pour \(k = 15\), @Q15_kn:1@ pour \(k = @Q15_n@\).

Ce que cela montre. \(k\) règle le compromis biais-variance : \(k = 1\) donne un modèle très flexible (biais faible, variance forte, erreur d'entraînement nulle qui ne dit rien de l'erreur sur de nouvelles données) ; \(k\) égal à la taille de l'échantillon donne un modèle constant (variance minimale, biais maximal). On choisit \(k\) par validation, jamais sur l'erreur d'entraînement.""",
    },
    {
        "id": "T9-Q16", "section": "mlf", "theme": "Biais et variance : les quatre cibles", "difficulte": 1, "temps_cible_s": 60,
        "conceptuelle": True, "correctes": ["b", "c"], "ordre_origine": ["a", "b", "c", "d"],
        "enonce": r"""Question 16. Quatre cibles de tir, notées A, B, C et D, illustrent le biais et la variance d'un modèle ; le centre de la cible représente la vraie valeur, chaque impact une prédiction.

- Cible A : les impacts sont très groupés, sur le centre.
- Cible B : les impacts sont dispersés tout autour du centre, leur nuage est centré sur lui.
- Cible C : les impacts sont très groupés, mais loin du centre (en haut de la cible).
- Cible D : les impacts sont dispersés, et leur nuage est loin du centre (en haut à gauche).

Quelles affirmations sont correctes ?""",
        "options": [
            {"id": "a", "texte": "A a un biais élevé et une variance élevée.",
             "explication": "Faux : A est le cas idéal, impacts groupés (variance faible) sur le centre (biais faible). C'est D qui cumule biais élevé et variance élevée."},
            {"id": "b", "texte": "B a un biais faible et une variance élevée.",
             "explication": "Juste : le nuage est centré sur la cible (en moyenne, on vise juste), mais les impacts sont dispersés."},
            {"id": "c", "texte": "C a un biais élevé et une variance faible.",
             "explication": "Juste : les impacts sont groupés (peu de dispersion), mais tous décalés du centre de la même façon (erreur systématique)."},
            {"id": "d", "texte": "D a un biais faible et une variance faible.",
             "explication": "Faux : D est dispersé et décalé, donc biais élevé et variance élevée. La description « biais faible, variance faible » est celle de A."},
        ],
        "solution": r"""Réponse : « B a un biais faible et une variance élevée » et « C a un biais élevé et une variance faible ».

Lecture de l'image. Chaque impact représente la prédiction d'un modèle entraîné sur un échantillon différent ; le centre est la valeur à prédire. On note \(\mathbf{P}\) la position aléatoire d'un impact, \(\mathbf{c}\) le centre (fixe) et \(\mathbf{m} = \mathbb{E}[\mathbf{P}]\) la position moyenne des impacts.

- Le **biais** est l'écart \(\mathbf{m} - \mathbf{c}\) entre la position moyenne des impacts et le centre : une erreur systématique.
- La **variance** est \(\mathbb{E}[\lVert \mathbf{P} - \mathbf{m} \rVert^2]\), la dispersion des impacts autour de leur propre moyenne.

| Cible | Nuage centré sur la cible ? | Impacts groupés ? | Biais | Variance |
|---|---|---|---|---|
| A | oui | oui | faible | faible |
| B | oui | non | faible | élevée |
| C | non | oui | élevé | faible |
| D | non | non | élevé | élevée |

Les deux se combinent dans l'erreur quadratique moyenne : \(\mathbb{E}[\lVert \mathbf{P} - \mathbf{c} \rVert^2] = \lVert \mathbf{m} - \mathbf{c} \rVert^2 + \mathbb{E}[\lVert \mathbf{P} - \mathbf{m} \rVert^2]\). Preuve : on écrit \(\mathbf{P} - \mathbf{c} = (\mathbf{P} - \mathbf{m}) + (\mathbf{m} - \mathbf{c})\) et on développe le carré de la norme ; le terme croisé \(2\,(\mathbf{m} - \mathbf{c})^{\top}\,\mathbb{E}[\mathbf{P} - \mathbf{m}]\) est nul, car \(\mathbb{E}[\mathbf{P} - \mathbf{m}] = \mathbf{0}\) par définition de \(\mathbf{m}\).""",
    },
    # =============================================================== ML AVANCÉ (questions 17 à 24)
    {
        "id": "T9-Q17", "section": "mla", "theme": "Apprentissage par renforcement", "difficulte": 2, "temps_cible_s": 90,
        "conceptuelle": True, "correctes": ["a", "b", "c"], "ordre_origine": ["a", "b", "c", "d"],
        "enonce": r"""Question 17. Quelles affirmations sur l'apprentissage par renforcement sont correctes ? (Choisis toutes celles qui conviennent.)""",
        "options": [
            {"id": "a", "texte": "En apprentissage par renforcement, l'agent apprend en recevant un retour sous forme de récompenses ou de pénalités, selon les actions qu'il effectue dans un environnement.",
             "explication": "Juste : c'est la définition. Il n'y a pas d'étiquette donnant la bonne action, seulement un signal de récompense qui suit les actions."},
            {"id": "b", "texte": "L'objectif d'un agent est de maximiser la récompense cumulée, qu'on appelle le « retour » (return).",
             "explication": "Juste : le retour est la somme (actualisée) des récompenses futures, et l'agent cherche la politique qui en maximise l'espérance."},
            {"id": "c", "texte": "Dans les méthodes de gradient de politique, la politique est paramétrée et optimisée directement, plutôt que déduite d'une estimation de la fonction de valeur.",
             "explication": "Juste : c'est ce qui les oppose aux méthodes fondées sur la valeur (Q-learning), où la politique se déduit d'une fonction de valeur estimée."},
            {"id": "d", "texte": "Le compromis exploration-exploitation consiste à choisir les actions au hasard pour que toutes les actions possibles soient explorées également.",
             "explication": "Faux : choisir au hasard uniformément, c'est de l'exploration pure, sans exploitation. Le compromis consiste à doser les deux : exploiter l'action estimée la meilleure, tout en essayant parfois les autres."},
        ],
        "solution": r"""Réponse : les trois premières affirmations (récompenses et pénalités ; maximisation du retour ; gradient de politique). La quatrième est fausse.

Le cadre. À chaque instant \(t\), l'agent observe un état \(S_t\), choisit une action \(A_t\), reçoit une récompense \(R_{t+1}\) (un nombre, négatif pour une pénalité) et passe à l'état \(S_{t+1}\). Une politique \(\pi\) est la règle qui donne la probabilité de chaque action dans chaque état. Le retour depuis l'instant \(t\) est \(G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots\), où \(\gamma\), entre 0 et 1, est le facteur d'actualisation. L'objectif est de trouver la politique qui maximise \(\mathbb{E}[G_t]\).

Affirmation 1 : c'est la définition de l'apprentissage par renforcement ; il n'y a pas d'étiquette donnant la bonne action.

Affirmation 2 : juste, avec une précision. C'est l'**espérance** du retour qui est maximisée (les récompenses sont aléatoires), et le retour est actualisé par \(\gamma\) quand l'horizon est infini.

Affirmation 3 : juste. Deux familles de méthodes. Fondées sur la valeur : on estime \(q(s, a)\), l'espérance du retour quand on fait l'action \(a\) dans l'état \(s\), et la politique s'en déduit (prendre l'action de plus grande valeur). Gradient de politique : la politique \(\pi_{\boldsymbol{\theta}}\) dépend d'un vecteur de paramètres \(\boldsymbol{\theta}\), ajusté par montée de gradient sur l'espérance du retour. Nuance : les méthodes acteur-critique estiment **aussi** une fonction de valeur, pour réduire la variance du gradient ; mais la politique y reste paramétrée et optimisée directement, ce qui est ce qu'affirme la phrase.

Affirmation 4 : fausse. Exploiter, c'est choisir l'action que l'on estime la meilleure ; explorer, c'est en essayer d'autres pour améliorer ses estimations. Choisir toujours au hasard uniformément est de l'exploration pure : l'agent n'utilise jamais ce qu'il a appris. La stratégie epsilon-gloutonne illustre le compromis : avec probabilité \(1 - \epsilon\) on exploite, avec probabilité \(\epsilon\) on tire une action au hasard. Rien n'exige que toutes les actions soient explorées **également**.""",
    },
    {
        "id": "T9-Q18", "section": "mla", "theme": "Dropout inversé", "difficulte": 3, "temps_cible_s": 120,
        "enonce": r"""Question de remplacement (la question d'origine n'a pas été capturée). Question 18. Une couche de dropout « inversé » de taux \(p = @Q18_p:1m@\) fonctionne ainsi pendant l'entraînement : chaque activation est, indépendamment des autres, mise à 0 avec probabilité \(p\), et sinon divisée par \(1 - p\). On considère une activation dont la valeur avant dropout est fixée à \(a = 3\).

Que valent l'espérance et la variance de sa valeur après dropout, pendant l'entraînement ?""",
        "options": [
            {"id": "a", "texte": "Espérance @Q18_esp:0@, variance @Q18_var:2@", "explication": r"Oui : la valeur est \(3/0{,}8 = @Q18_valeur_gardee:2m@\) avec probabilité @Q18_garde:1@ et 0 sinon."},
            {"id": "b", "texte": "Espérance @Q18_d_esp_sans:1@, variance @Q18_d_var_sans:2@", "explication": r"C'est le dropout **sans** remise à l'échelle : la valeur est 3 avec probabilité @Q18_garde:1@ et 0 sinon. La division par \(1 - p\) sert justement à ramener l'espérance à 3."},
            {"id": "c", "texte": "Espérance @Q18_esp:0@, variance @Q18_d_var_p:2@", "explication": r"L'espérance est juste, mais la variance est celle de \(3M\), où \(M\) vaut 1 avec probabilité @Q18_garde:1@ : il manque le carré du facteur \(1/(1 - p)\). La variance exacte est \(a^2\,p/(1 - p)\), pas \(a^2\,p\,(1 - p)\)."},
            {"id": "d", "texte": "Espérance @Q18_valeur_gardee:2@, variance 0", "explication": r"C'est supposer que l'activation est toujours conservée et divisée par \(1 - p\). Elle est mise à 0 avec probabilité @Q18_p:1@ : la sortie est aléatoire, donc de variance non nulle."},
        ],
        "solution": r"""Réponse : espérance @Q18_esp:0@, variance @Q18_var:2@.

Notations : \(M\) est le masque de l'activation, variable aléatoire qui vaut 1 avec probabilité \(1 - p = @Q18_garde:1m@\) (activation conservée) et 0 avec probabilité \(p = @Q18_p:1m@\). La valeur après dropout est \(Z = \dfrac{M\,a}{1 - p}\) : elle vaut \(3/0{,}8 = @Q18_valeur_gardee:2m@\) ou 0.

Étape 1, espérance. \(\mathbb{E}[Z] = \dfrac{a}{1 - p}\,\mathbb{E}[M] = \dfrac{a}{1 - p}\,(1 - p) = a = @Q18_esp:0m@\). C'est le rôle de la division par \(1 - p\) : l'espérance de l'activation est la même à l'entraînement (avec dropout) et à l'inférence (sans dropout, sans facteur).

Étape 2, variance. \(M\) est une variable de Bernoulli, de variance \(p\,(1 - p)\). Donc \(\mathrm{Var}(Z) = \dfrac{a^2}{(1 - p)^2}\,p\,(1 - p) = \dfrac{a^2\,p}{1 - p} = \dfrac{9 \times 0{,}2}{0{,}8} = @Q18_var:2m@\).

Vérification (script) : calcul exact en fractions, et simulation d'un million de masques : moyenne @Q18_esp_mc:3@, variance @Q18_var_mc:3@.

À l'inférence, le dropout est désactivé : la sortie vaut \(a\), sans bruit. Le dropout ajoute donc du bruit pendant l'entraînement seulement, ce qui agit comme une régularisation.""",
    },
    {
        "id": "T9-Q19", "section": "mla", "theme": "Classes déséquilibrées et variance", "difficulte": 3, "temps_cible_s": 90,
        "ordre_origine": ["b", "c", "d", "a"],
        "enonce": r"""Question 19. On dispose d'un jeu de données dans lequel le nombre d'observations diffère nettement d'une classe à l'autre. Quelle technique est la plus appropriée pour améliorer la performance d'un modèle en traitant d'éventuels problèmes de variance ?""",
        "options": [
            {"id": "a", "texte": "Appliquer des techniques de régularisation pour réduire la complexité du modèle et éviter le surapprentissage.",
             "explication": "Oui, dans la lecture retenue : un « problème de variance » d'un modèle est un surapprentissage, et la régularisation est la technique qui réduit la variance d'un modèle."},
            {"id": "b", "texte": "Augmenter la complexité du modèle pour capter plus précisément les structures sous-jacentes des données.",
             "explication": "Augmenter la complexité réduit le biais et augmente la variance : c'est le remède au sous-apprentissage, l'inverse de ce qui est demandé."},
            {"id": "c", "texte": "Recourir à des tests d'hypothèses pour repérer les différences significatives entre classes et ajuster le modèle en conséquence.",
             "explication": "Un test d'hypothèse dit si une différence observée est compatible avec le hasard ; il ne modifie ni la flexibilité du modèle ni sa variance."},
            {"id": "d", "texte": "Utiliser un échantillonnage stratifié pour que les ensembles d'entraînement et de test aient une répartition des classes similaire.",
             "explication": "C'est l'autre réponse défendable : la stratification est la précaution propre au déséquilibre des classes, et elle réduit la variance de l'**évaluation** (composition des ensembles). Mais elle ne change pas le modèle et ne réduit pas sa variance ; voir la discussion dans la correction."},
        ],
        "solution": r"""Réponse : appliquer des techniques de régularisation. C'est la réponse la plus défendable, pas une réponse certaine : la question d'origine est ambiguë, et l'échantillonnage stratifié se défend aussi.

Pourquoi la question est ambiguë. Elle réunit deux éléments qui ne pointent pas vers la même option :

- la situation décrite, des classes déséquilibrées, appelle la précaution spécifique à ce cas : l'échantillonnage stratifié ;
- le but annoncé, « améliorer la performance en traitant des problèmes de variance », et le titre de la question d'origine, qui nomme la régularisation, appellent la régularisation.

Ce que fait chaque technique, précisément.

- **Régularisation.** La variance d'un modèle est la sensibilité de ses prédictions à l'échantillon d'entraînement ; un modèle de forte variance surapprend. Pénaliser la taille des coefficients (ou limiter la profondeur d'un arbre, etc.) réduit cette variance au prix d'un peu de biais. Expérience (script, celle de la question 22) : sur @Q22_n@ points, un polynôme de degré @Q22_degre@ sans pénalité a une erreur de test de @Q22_test_sans:3@ ; avec une pénalité ridge, @Q22_test_ridge:3@.
- **Échantillonnage stratifié.** Découper au hasard un jeu déséquilibré peut laisser très peu d'exemples de la classe rare dans l'un des ensembles ; stratifier impose la même proportion de chaque classe dans chaque ensemble. Cela réduit la variance de l'**estimation de la performance** d'un découpage à l'autre. Cela ne modifie pas le modèle, donc pas sa variance au sens du compromis biais-variance, et n'« améliore » sa performance que dans la mesure où l'entraînement dispose d'une part garantie de la classe rare.

Pourquoi retenir la régularisation : c'est la seule option qui traite la variance **du modèle**, et qui améliore sa performance sur de nouvelles données ; le titre de la question d'origine va dans le même sens. Pourquoi l'autre lecture se défend : si « problèmes de variance » désigne l'instabilité due à la composition aléatoire des ensembles quand une classe est rare, la stratification est la réponse, et c'est la seule option qui utilise la donnée de l'énoncé (le déséquilibre). La réponse attendue par le correcteur n'est pas connue.

Les deux autres options s'écartent sans hésiter : augmenter la complexité augmente la variance, et un test d'hypothèse ne la traite pas.""",
    },
    {
        "id": "T9-Q20", "section": "mla", "theme": "Arbres de décision : mécanique", "difficulte": 2, "temps_cible_s": 90,
        "correctes": ["a", "c"], "ordre_origine": ["a", "b", "c", "d"],
        "enonce": r"""Question 20. Quelles affirmations sur le fonctionnement des arbres de décision sont correctes ? (Choisis toutes celles qui conviennent.)""",
        "options": [
            {"id": "a", "texte": "Dans un arbre de décision, le nœud racine est l'endroit où est choisie la variable la plus significative pour découper les données.",
             "explication": "Juste, au sens suivant : la racine porte la coupure qui réduit le plus l'impureté sur l'ensemble des données. « La plus significative » est une formulation vague, discutée dans la correction."},
            {"id": "b", "texte": "Les arbres de décision ne traitent que des données catégorielles et exigent que toutes les données numériques soient discrétisées avant l'entraînement.",
             "explication": r"Faux : sur une variable numérique, l'arbre cherche lui-même un seuil \(s\) et coupe selon « valeur inférieure ou égale à \(s\) ». Aucune discrétisation préalable n'est nécessaire (l'arbre de l'expérience est entraîné sur deux variables numériques brutes)."},
            {"id": "c", "texte": "À chaque nœud, un arbre de décision découpe les données en choisissant la variable qui produit la plus grande réduction d'impureté (par exemple l'impureté de Gini ou l'entropie).",
             "explication": "Juste : c'est le critère glouton de construction. Plus précisément, il choisit le couple (variable, seuil) qui maximise cette réduction."},
            {"id": "d", "texte": "La profondeur d'un arbre de décision est déterminée par le nombre de variables du jeu de données.",
             "explication": r"Faux : une même variable peut servir à plusieurs niveaux. Expérience (script) : avec @Q20_nb_variables@ variables et @Q20_n@ points, un arbre non limité atteint la profondeur @Q20_profondeur@. La profondeur dépend des données et des critères d'arrêt."},
        ],
        "solution": r"""Réponse : « le nœud racine est l'endroit où est choisie la variable la plus significative » et « à chaque nœud, l'arbre choisit la variable qui produit la plus grande réduction d'impureté ».

Construction d'un arbre (algorithme CART). À un nœud contenant un ensemble d'observations, on note \(i\) son impureté : pour la classification, l'impureté de Gini \(1 - \sum_c p_c^2\) ou l'entropie \(-\sum_c p_c \ln p_c\), où \(p_c\) est la proportion de la classe \(c\) dans le nœud. Pour chaque variable et chaque seuil possibles, on calcule la réduction d'impureté : l'impureté du nœud moins la moyenne des impuretés des deux nœuds fils, pondérée par leurs effectifs. On retient la coupure qui la maximise, puis on recommence dans chaque fils, jusqu'à un critère d'arrêt (profondeur maximale, effectif minimal, nœud pur).

Affirmation 3 : c'est exactement cette règle. Juste.

Affirmation 1 : la racine applique la règle à l'ensemble des données ; la variable qui y est choisie est celle qui, seule et en une coupure, sépare le mieux les classes. C'est le sens dans lequel elle est « la plus significative », et l'affirmation est retenue comme juste. Réserve : ce n'est pas un test statistique, et ce n'est pas forcément la variable la plus importante de l'arbre entier (une variable peut ne servir qu'en interaction avec une autre, plus bas dans l'arbre).

Affirmation 2 : fausse. Une variable numérique est traitée par seuil, sans discrétisation préalable.

Affirmation 4 : fausse. Expérience (script) : sur @Q20_n@ points décrits par @Q20_nb_variables@ variables numériques, avec des étiquettes tirées au hasard, un arbre sans limite de profondeur atteint la profondeur @Q20_profondeur@ et classe parfaitement ses données d'entraînement (exactitude @Q20_exactitude_train:0@) : la profondeur n'est pas bornée par le nombre de variables. Elle est bornée par le nombre d'observations et fixée par les critères d'arrêt.

Remarque sur la question d'origine : l'affirmation 1 est formulée de façon vague. Si le correcteur l'a jugée fausse (au motif que « significative » n'est pas le critère), la seule affirmation juste serait la troisième. La lecture retenue ici est la plus courante.""",
    },
    {
        "id": "T9-Q21", "section": "mla", "theme": "Importance des variables", "difficulte": 3, "temps_cible_s": 120,
        "correctes": ["a", "c", "d"], "ordre_origine": ["a", "b", "c", "d"],
        "enonce": r"""Question 21. Quelles affirmations sur l'importance des variables dans les modèles d'apprentissage automatique sont correctes ? (Choisis toutes celles qui conviennent.)""",
        "options": [
            {"id": "a", "texte": "Dans les arbres de décision, l'importance d'une variable est calculée à partir de la réduction totale d'impureté (impureté de Gini ou entropie, par exemple) sur tous les nœuds où la variable est utilisée.",
             "explication": "Juste : c'est la définition de l'importance par réduction d'impureté, chaque réduction étant pondérée par la part des observations qui passent par le nœud."},
            {"id": "b", "texte": "Les importances issues de modèles à base d'arbres, comme les forêts aléatoires, sont toujours fiables et doivent être interprétées directement comme l'importance des variables dans le monde réel.",
             "explication": r"Faux. Expérience (script) : dans une forêt, une variable de pur bruit reçoit une importance par impureté de @Q21_mdi_bruit_continu:2@, alors que la mélanger ne dégrade pas la performance sur des données de test."},
            {"id": "c", "texte": "L'importance par permutation mesure l'importance d'une variable par la baisse de performance du modèle quand les valeurs de cette variable sont mélangées au hasard.",
             "explication": "Juste : c'est sa définition. Mélanger la colonne casse son lien avec la cible en conservant sa loi ; la baisse de performance mesure ce que le modèle tirait de cette variable."},
            {"id": "d", "texte": "Une limite des importances tirées de modèles linéaires, comme la régression logistique, est qu'elles supposent des relations linéaires entre les variables et la cible.",
             "explication": "Juste : un coefficient résume l'effet d'une variable par une seule pente. Si l'effet réel n'est pas linéaire (ou passe par une interaction), le coefficient le décrit mal. Précision : pour la régression logistique, la linéarité porte sur le logarithme des cotes."},
        ],
        "solution": r"""Réponse : les affirmations sur la réduction d'impureté, sur l'importance par permutation et sur la limite des modèles linéaires. Celle qui déclare les importances des forêts « toujours fiables » est fausse.

Affirmation 1, juste. L'importance par réduction d'impureté d'une variable est la somme, sur tous les nœuds où elle sert à couper, de la réduction d'impureté obtenue, pondérée par la proportion des observations d'entraînement qui atteignent le nœud ; dans une forêt, on fait la moyenne sur les arbres, puis on normalise pour que les importances somment à 1.

Affirmation 3, juste. Importance par permutation : on mesure la performance du modèle sur un jeu de données, on mélange au hasard les valeurs d'une colonne, on mesure à nouveau ; l'importance est la baisse de performance.

Affirmation 4, juste. Dans un modèle linéaire, l'importance d'une variable est lue sur son coefficient (après mise à la même échelle des variables). Le modèle suppose que la cible, ou le logarithme des cotes pour la régression logistique, est une fonction linéaire des variables : un effet en U, par exemple, donne un coefficient proche de 0 pour une variable pourtant déterminante.

Affirmation 2, fausse : le mot « toujours » suffit à la réfuter, par un contre-exemple. Expérience (script) : une cible binaire dépend d'une seule variable utile ; on ajoute une variable de bruit continue et une variable de bruit binaire, sans lien avec la cible. Une forêt de 200 arbres non limités donne :

| Variable | Importance par impureté | Importance par permutation, sur données de test |
|---|---|---|
| utile | @Q21_mdi_utile:3@ | @Q21_perm_utile:3@ |
| bruit continu | @Q21_mdi_bruit_continu:3@ | @Q21_perm_bruit_continu:3@ |
| bruit binaire | @Q21_mdi_bruit_binaire:3@ | @Q21_perm_bruit_binaire:3@ |

Le bruit continu reçoit une importance par impureté du même ordre que la variable utile : les arbres profonds s'en servent pour mémoriser le bruit des données d'entraînement, et une variable continue offre beaucoup plus de seuils qu'une variable binaire. Son importance par permutation sur des données de test est négative ou nulle : elle n'aide pas à prédire. C'est une constatation sur cette expérience, qui suffit comme contre-exemple. Autres causes connues de mauvaise lecture : variables corrélées qui se partagent l'importance, et confusion entre importance pour le modèle et effet causal.

Remarque : cocher les affirmations 1 et 4 seulement laisse de côté la 3, qui est une définition exacte.""",
    },
    {
        "id": "T9-Q22", "section": "mla", "theme": "Surapprentissage et régularisation", "difficulte": 2, "temps_cible_s": 75,
        "ordre_origine": ["b", "a", "c", "d"],
        "enonce": r"""Question 22. On dispose d'un jeu de données présentant une variance importante, et l'on veut construire un modèle prédictif de grande exactitude. Quelle technique serait la plus efficace pour traiter le surapprentissage tout en préservant la capacité du modèle à généraliser ?""",
        "options": [
            {"id": "a", "texte": "Appliquer des techniques de régularisation pour réduire la complexité du modèle.",
             "explication": "Oui : la régularisation réduit la variance du modèle, donc le surapprentissage, en conservant toutes les données et toutes les variables."},
            {"id": "b", "texte": "Augmenter la complexité du modèle pour capter davantage de variance.",
             "explication": r"C'est le remède au **sous**-apprentissage. Un modèle plus complexe colle encore mieux aux données d'entraînement, bruit compris : dans l'expérience, passer du degré @Q22_degre@ au degré @Q22_degre_plus@ fait passer l'erreur de test de @Q22_test_sans:2@ à @Q22_test_plus:2@."},
            {"id": "c", "texte": "Retirer des variables au hasard pour réduire la dimension.",
             "explication": "Réduire la dimension peut réduire la variance, mais retirer des variables au hasard risque d'écarter les plus utiles. Une sélection doit être guidée par un critère (pénalité L1, validation)."},
            {"id": "d", "texte": "S'en remettre uniquement à des tests d'hypothèses pour sélectionner les variables pertinentes.",
             "explication": "Les tests variable par variable ignorent les interactions et, répétés sur beaucoup de variables, laissent passer des faux positifs ; « uniquement » exclut la validation sur des données non vues, qui est le vrai garde-fou."},
        ],
        "solution": r"""Réponse : appliquer des techniques de régularisation.

Définitions. Un modèle surapprend quand il reproduit le bruit de son échantillon d'entraînement : son erreur d'entraînement est faible, son erreur sur de nouvelles données est grande. Dans la décomposition de l'erreur de prédiction en biais au carré, variance et bruit irréductible, le surapprentissage est un excès de **variance** (sensibilité des prédictions à l'échantillon tiré). Régulariser, c'est ajouter à la perte une pénalité sur la taille des paramètres, ce qui réduit la variance au prix d'un biais.

Expérience (script). Données : @Q22_n@ points, \(y = \sin(3x)\) plus un bruit de variance @Q22_bruit_var:2@ (aucun modèle ne peut descendre sous cette erreur de test). Modèle : polynôme en \(x\), ajusté par moindres carrés.

| Modèle | Erreur quadratique d'entraînement | Erreur quadratique de test |
|---|---|---|
| degré @Q22_degre@, sans pénalité | @Q22_train_sans:4@ | @Q22_test_sans:3@ |
| degré @Q22_degre@, pénalité ridge de coefficient @Q22_lambda:1@ | @Q22_train_ridge:4@ | @Q22_test_ridge:3@ |
| degré @Q22_degre_plus@, sans pénalité (plus complexe) | @Q22_train_plus:4@ | @Q22_test_plus:2@ |

La pénalité augmente un peu l'erreur d'entraînement et réduit nettement l'erreur de test. Augmenter la complexité fait l'inverse : l'erreur d'entraînement baisse, l'erreur de test explose. Ce sont des constatations sur un tirage ; le sens de l'effet est celui que prévoit la décomposition biais-variance.

Le piège de formulation. L'option « augmenter la complexité pour capter davantage de variance » joue sur le mot variance : « capter la variance des données » sonne comme une qualité, mais un modèle plus complexe augmente la variance **du modèle**, donc le surapprentissage que la question demande de traiter.

Remarque sur la question d'origine : « un jeu de données présentant une variance importante » ne dit pas de quelle variance il s'agit (dispersion de la cible, bruit, ou variance du modèle). Cela ne change pas la réponse, puisque la question demande explicitement de traiter le surapprentissage.""",
    },
    {
        "id": "T9-Q23", "section": "mla", "theme": "Analyse en composantes principales", "difficulte": 3, "temps_cible_s": 120,
        "correctes": ["a", "b", "d"], "ordre_origine": ["a", "b", "c", "d"],
        "enonce": r"""Question 23. Quelles affirmations sur l'usage de l'analyse en composantes principales (ACP) sont correctes ? (Choisis toutes celles qui conviennent.)""",
        "options": [
            {"id": "a", "texte": "L'ACP sert principalement à réduire la dimension, en transformant les variables d'origine en un ensemble de composantes orthogonales qui captent le maximum de variance des données.",
             "explication": "Juste : les composantes sont les projections sur les vecteurs propres de la matrice de covariance, orthogonaux entre eux, rangés par variance décroissante."},
            {"id": "b", "texte": "Avant d'appliquer l'ACP, il est essentiel de standardiser les variables (moyenne nulle, variance 1), car l'ACP est sensible à l'échelle des données.",
             "explication": r"Retenue comme juste : l'ACP est sensible à l'échelle (multiplier une variable par @Q23_facteur@ fait passer la part de la première composante de @Q23_part_egales:1@ à @Q23_part_echelle:4@ dans l'exemple). Réserve : « essentiel » est trop fort quand toutes les variables ont la même unité ; voir la correction."},
            {"id": "c", "texte": "La première composante principale capte toujours plus de 90 % de la variance, quelle que soit la complexité ou la structure des données.",
             "explication": r"Faux : avec @Q23_dim_iso@ variables non corrélées de même variance, la première composante porte \(1/@Q23_dim_iso@\) de la variance, soit @Q23_part_iso_pct:0@ %."},
            {"id": "d", "texte": "Les composantes principales obtenues par ACP sont des combinaisons linéaires des variables d'origine.",
             "explication": "Juste : chaque composante est le produit scalaire du vecteur des variables centrées par un vecteur propre de la matrice de covariance."},
        ],
        "solution": r"""Réponse : les affirmations sur la réduction de dimension, sur la standardisation et sur les combinaisons linéaires. Celle des « plus de 90 % » est fausse.

Rappel. On note \(\mathbf{X}\) le vecteur aléatoire des \(d\) variables, \(\boldsymbol{\Sigma}\) sa matrice de covariance, de valeurs propres \(\lambda_1 \ge \dots \ge \lambda_d \ge 0\) et de vecteurs propres orthonormés \(\mathbf{w}_1, \dots, \mathbf{w}_d\) (théorème spectral). La \(j\)-ième composante principale est \(\mathbf{w}_j^{\top}(\mathbf{X} - \mathbb{E}[\mathbf{X}])\) ; sa variance vaut \(\lambda_j\) et la part de variance qu'elle porte est \(\lambda_j / (\lambda_1 + \dots + \lambda_d)\).

Affirmation 1, juste : les directions \(\mathbf{w}_j\) sont orthogonales, les composantes sont non corrélées, et \(\mathbf{w}_1\) est la direction de variance maximale. Garder les premières composantes réduit la dimension.

Affirmation 4, juste : \(\mathbf{w}_j^{\top}(\mathbf{X} - \mathbb{E}[\mathbf{X}])\) est une combinaison linéaire des variables centrées, de coefficients les coordonnées de \(\mathbf{w}_j\). Vérifié sur un échantillon par le script (vecteurs propres orthonormés, composantes non corrélées).

Affirmation 3, fausse, par un contre-exemple exact : si \(\boldsymbol{\Sigma}\) est la matrice identité de taille @Q23_dim_iso@, toutes les valeurs propres valent 1 et la première composante porte @Q23_part_iso:1@ de la variance.

Affirmation 2, retenue comme juste, avec une réserve.

- Ce qui est vrai sans réserve : l'ACP dépend de l'échelle. Exemple exact : deux variables non corrélées de variance 1 ; la première composante porte @Q23_part_egales:1@ de la variance. On exprime la première variable dans une unité @Q23_facteur@ fois plus petite : sa variance devient @Q23_var_apres@, et la première composante, qui est alors cette variable seule, porte \(@Q23_var_apres@/(@Q23_var_apres@ + 1) = @Q23_part_echelle:4m@\) de la variance. Le résultat reflète le choix d'unité, pas la structure des données. Standardiser revient à faire l'ACP sur la matrice de corrélation.
- La réserve : « essentiel » est trop général. Le centrage est toujours nécessaire. La réduction à variance 1 l'est quand les variables ont des unités ou des ordres de grandeur différents ; quand elles ont la même unité (des rendements d'actifs, par exemple), l'ACP sur la covariance est légitime et garde l'information de volatilité que la standardisation efface.

Conséquence : si le correcteur a jugé l'affirmation 2 fausse à cause du mot « essentiel », la réponse attendue serait réduite aux affirmations 1 et 4. La lecture retenue ici (affirmation 2 juste) est celle de la justification donnée dans la phrase elle-même, « car l'ACP est sensible à l'échelle ».

Dans tous les cas, l'affirmation 4 est juste et devait être cochée.""",
    },
    {
        "id": "T9-Q24", "section": "mla", "theme": "Régularisation L1, L2, Elastic Net", "difficulte": 2, "temps_cible_s": 75,
        "ordre_origine": ["a", "b", "c", "d"],
        "enonce": r"""Question 24. On dispose d'un jeu de données dont l'espace des variables est de grande dimension. Quelle technique de régularisation appliquer pour éviter le surapprentissage tout en conservant l'interprétabilité du modèle ?""",
        "options": [
            {"id": "a", "texte": "Régularisation L1 (Lasso)",
             "explication": "Oui : la pénalité L1 annule exactement une partie des coefficients. Le modèle ne garde que quelques variables, ce qui le rend lisible."},
            {"id": "b", "texte": "Régularisation L2 (Ridge)",
             "explication": r"Ridge évite le surapprentissage, mais il rétrécit les coefficients sans en annuler aucun : dans l'expérience, les @Q24_nz_ridge@ coefficients restent non nuls. Un modèle à @Q24_nz_ridge@ variables n'est pas plus interprétable qu'avant."},
            {"id": "c", "texte": "Régularisation Elastic Net (combinaison de L1 et L2)",
             "explication": r"Elle annule aussi des coefficients, mais moins que le Lasso à pénalité égale (@Q24_nz_enet@ coefficients non nuls contre @Q24_nz_lasso@ dans l'expérience). C'est le second choix défendable, utile quand des variables sont très corrélées ; pour l'interprétabilité seule, le Lasso est plus direct."},
            {"id": "d", "texte": "Aucune régularisation n'est nécessaire",
             "explication": r"En grande dimension, c'est l'inverse : avec plus de variables que d'observations (@Q24_p@ contre @Q24_n@ dans l'expérience), les moindres carrés sans pénalité reproduisent exactement les données d'entraînement, bruit compris."},
        ],
        "solution": r"""Réponse : la régularisation L1 (Lasso).

Notations : \(\boldsymbol{\beta}\) est le vecteur des coefficients du modèle linéaire, \(\beta_j\) son \(j\)-ième coefficient, \(\lambda > 0\) le coefficient de pénalité. Ridge ajoute à la somme des carrés des résidus la pénalité \(\lambda \sum_j \beta_j^2\) (norme L2 au carré) ; le Lasso ajoute \(\lambda \sum_j \lvert \beta_j \rvert\) (norme L1) ; Elastic Net ajoute une combinaison des deux.

Les deux mots de l'énoncé. « Éviter le surapprentissage » : les trois pénalités le font, en réduisant la variance. « Conserver l'interprétabilité » en grande dimension : il faut un modèle qui n'utilise que peu de variables, donc des coefficients **exactement nuls**. C'est ce qui distingue L1 de L2.

Pourquoi L1 annule et pas L2, dans le cas où les colonnes de la matrice des variables sont orthonormées (cas démontré dans la séance de ML 1, partie Lasso). On note \(b_j\) le coefficient des moindres carrés sans pénalité. La solution Ridge est \(b_j / (1 + \lambda)\) : nulle seulement si \(b_j\) l'est. La solution du Lasso, pour la perte \(\tfrac12 \lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert^2 + \lambda \sum_j \lvert \beta_j \rvert\), est le seuillage doux : \(b_j - \lambda\) si \(b_j > \lambda\), \(b_j + \lambda\) si \(b_j < -\lambda\), et **0** si \(\lvert b_j \rvert \le \lambda\). Tout coefficient plus petit que le seuil est annulé.

Expérience (script) : @Q24_n@ observations, @Q24_p@ variables, dont @Q24_utiles@ seulement agissent sur la cible.

| Méthode | Coefficients non nuls sur @Q24_p@ |
|---|---|
| Lasso | @Q24_nz_lasso@ (dont les @Q24_lasso_retrouve@ variables utiles) |
| Elastic Net, même coefficient de pénalité, moitié L1 | @Q24_nz_enet@ |
| Ridge | @Q24_nz_ridge@ |

C'est une constatation sur un tirage et un réglage ; le fait que Ridge n'annule aucun coefficient, lui, est général (presque sûrement, pour des données continues).

Nuance : Elastic Net est aussi parcimonieux et se défend quand des groupes de variables sont très corrélés (le Lasso en garde alors une par groupe, de façon instable). La question demande une seule technique et met en avant l'interprétabilité : le Lasso est la réponse la plus directe.""",
    },
]
