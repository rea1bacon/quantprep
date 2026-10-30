"""Séance de mathématiques 4 : leçon express (Markov, martingales, AR(1), mouvement brownien), puis 8 questions.

Pondération des thèmes : marches aléatoires, chaînes de Markov et martingales d'abord, AR(1) ensuite, calcul
stochastique en dernier. D'où 4 questions sur Markov et martingales, 2 sur l'AR(1), 2 sur le mouvement brownien.
Toutes les valeurs : verif/seance_math_4.py. Dans chaque question, l'option "a" est la bonne.
"""
import lecons_math_4
import questions_diagnostic as diag

ID = "seance-math-4"
TITRE = "Séance de mathématiques 4 : Markov, martingales et séries temporelles"
VERIF = "verif/seance_math_4.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [{"id": "lecon", "nom": "Leçon", "budget_min": 20}, {"id": "math", "nom": "Mathématiques", "budget_min": 16}]
PRESENTATION = (
    "D'abord une leçon express d'environ 20 minutes : chaînes de Markov (loi stationnaire, absorption par le "
    "système linéaire), martingales et théorème d'arrêt de Doob, processus AR(1), mouvement brownien et lemme "
    "d'Itô, avec quatre points de contrôle corrigés immédiatement. Elle prolonge l'outil 2 de la séance de "
    "mathématiques 1 (premier pas, ruine du joueur, identité de Wald). Ensuite 8 questions à choix multiple : "
    "quatre sur Markov et les martingales, deux sur l'AR(1), deux sur le mouvement brownien, dont la moitié de "
    "difficulté 4 ou 5.\n\n"
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

LECONS = [lecons_math_4.LECON_MATH_4]
CODES = []

QUESTIONS = [
    # ------------------------------------------------- Chaînes de Markov
    {
        "id": "M4-01", "section": "math", "theme": "Chaîne de Markov à deux états, loi stationnaire", "difficulte": 2, "temps_cible_s": 75,
        "enonce": r"""Un modèle de marché a deux régimes, calme et agité. Chaque jour, le régime du lendemain ne dépend que du régime du jour (propriété de Markov) : un marché calme devient agité le lendemain avec probabilité 0,15 et reste calme sinon ; un marché agité redevient calme le lendemain avec probabilité 0,35 et reste agité sinon. Le marché est calme au jour 0.

Vers quelle valeur tend la probabilité que le marché soit agité au jour \(t\), quand \(t\) tend vers l'infini ?""",
        "options": [
            {"id": "a", "texte": "@Q01_exact:2@", "explication": r"Oui : \(\pi_{\text{agité}} = a/(a+b) = 0{,}15/@Q01_somme:1m@\), et la convergence est géométrique de raison \(1 - a - b = @Q01_lambda:1m@\)."},
            {"id": "b", "texte": "@Q01_d_inverse:2@", "explication": r"C'est \(b/(a+b)\), la probabilité limite du régime **calme**. L'équilibre des flux \(\pi_{\text{calme}}\,a = \pi_{\text{agité}}\,b\) donne au régime agité le poids \(a\), la probabilité d'y entrer."},
            {"id": "c", "texte": "@Q01_d_ratio:3@", "explication": r"C'est \(a/b = \pi_{\text{agité}}/\pi_{\text{calme}}\), le **rapport** des deux probabilités limites, pas une probabilité : il faut encore normaliser par \(\pi_{\text{calme}} + \pi_{\text{agité}} = 1\)."},
            {"id": "d", "texte": "@Q01_d_un_pas:2@", "explication": r"C'est la probabilité d'être agité au jour 1, après un seul pas depuis le calme. À long terme, le marché passe aussi des jours agités consécutifs, et l'état initial est oublié."},
        ],
        "solution": r"""Réponse : \(\dfrac{0{,}15}{0{,}15 + 0{,}35} = @Q01_exact:1m@\).

Notations : \(a = 0{,}15\) (calme vers agité), \(b = 0{,}35\) (agité vers calme), \(u_t\) la probabilité d'être agité au jour \(t\), \(\boldsymbol{\pi} = (\pi_{\text{calme}} ; \pi_{\text{agité}})\) la loi stationnaire.

Étape 1, loi stationnaire : \(\boldsymbol{\pi}\mathbf{P} = \boldsymbol{\pi}\) se réduit à \(\pi_{\text{calme}}\,a = \pi_{\text{agité}}\,b\) ; avec la somme égale à 1, \(\pi_{\text{agité}} = a/(a+b) = 0{,}15/@Q01_somme:1m@ = @Q01_exact:1m@\).

Étape 2, convergence (leçon, partie 1, exemple à deux états) : \(u_{t+1} = a + (1 - a - b)\,u_t\), donc \(u_t - \pi_{\text{agité}} = (1 - a - b)^t\,(u_0 - \pi_{\text{agité}})\), avec \(1 - a - b = @Q01_lambda:1m@\). L'écart est divisé par 2 chaque jour : \(u_t \to @Q01_exact:1m@\), quel que soit le régime du jour 0.

Simulation (script), \(u_{60}\) : @Q01_mc:3@.""",
    },
    {
        "id": "M4-02", "section": "math", "theme": "Chaîne absorbante, temps d'attente d'un motif", "difficulte": 3, "temps_cible_s": 120,
        "enonce": r"""On lance un dé équilibré à six faces de façon répétée, les lancers étant indépendants, jusqu'à obtenir pour la première fois deux 6 consécutifs, c'est-à-dire deux lancers successifs qui donnent tous deux 6. On compte tous les lancers effectués, y compris les deux derniers.

Quelle est l'espérance du nombre de lancers ?""",
        "options": [
            {"id": "a", "texte": "@Q02_exact@", "explication": r"Oui : \(e_1 = @Q02_e1@\) depuis l'état « dernier lancer 6 », \(e_0 = 6 + e_1 = @Q02_exact@\) depuis le départ."},
            {"id": "b", "texte": "@Q02_d_inverse@", "explication": r"C'est \(1/\mathbb{P}(66) = 1/(1/36)\), l'attente d'un succès de probabilité 1/36 si les essais étaient indépendants et sans chevauchement. Ici, un 6 suivi d'un autre chiffre oblige à recommencer, et il faut d'abord atteindre un premier 6."},
            {"id": "c", "texte": "@Q02_d_somme@", "explication": r"C'est 6 + 6 : attendre un 6, puis encore un 6. Mais le second 6 doit tomber **immédiatement** après le premier ; sinon on repart de zéro, ce qui allonge beaucoup l'attente."},
            {"id": "d", "texte": "@Q02_d_blocs@", "explication": r"C'est l'attente d'un bloc 66 quand on découpe les lancers en blocs disjoints de deux : 36 blocs en moyenne. Deux 6 consécutifs peuvent chevaucher deux blocs : ce nombre n'est qu'un majorant."},
        ],
        "solution": r"""Réponse : @Q02_exact@ lancers.

Notations : état 0, « dernier lancer différent de 6, ou aucun lancer » ; état 1, « dernier lancer égal à 6 ». \(e_0\) et \(e_1\) sont les espérances du nombre de lancers restants depuis chaque état.

Étape 1, finitude : en blocs disjoints de deux lancers, chaque bloc vaut 66 avec probabilité 1/36, indépendamment des autres ; l'argument des blocs (outil 2 de la séance 1, étape 1) majore l'espérance par \(2 \times 36 = @Q02_borne@\).

Étape 2, équations du premier pas, c'est-à-dire \((\mathbf{I} - \mathbf{Q})\,\mathbf{d} = \mathbf{1}\) (leçon, partie 1) : depuis l'état 1, un 6 termine et tout autre chiffre ramène à l'état 0 : \(e_1 = 1 + \tfrac56 e_0\). Depuis l'état 0, un 6 mène à l'état 1, sinon on reste en 0 : \(e_0 = 1 + \tfrac56 e_0 + \tfrac16 e_1\), soit \(e_0 = 6 + e_1\).

Étape 3 : \(e_1 = 1 + \tfrac56 (6 + e_1) = 6 + \tfrac56 e_1\), donc \(e_1 = @Q02_e1@\) et \(e_0 = @Q02_exact@\). La solution est unique (\(\mathbf{I} - \mathbf{Q}\) inversible) et les vraies espérances, finies, la vérifient.

Simulation des lancers (script) : @Q02_mc:2@.""",
    },
    {
        "id": "M4-03", "section": "math", "theme": "Chaîne absorbante, probabilité d'absorption", "difficulte": 4, "temps_cible_s": 150,
        "enonce": r"""Au tennis, un jeu arrivé à « égalité » se poursuit ainsi : le joueur qui gagne le point suivant prend l'« avantage » ; s'il gagne aussi le point d'après, il remporte le jeu ; s'il le perd, on revient à égalité. Le serveur gagne chaque point avec probabilité 0,6, indépendamment des autres points.

Partant de l'égalité, quelle est la probabilité que le serveur remporte le jeu, arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q03_exact:3@", "explication": r"Oui : \(h = p^2/(p^2 + q^2) = @Q03_p2:2m@/@Q03_den:2m@ = @Q03_exact_num@/@Q03_exact_den@\)."},
            {"id": "b", "texte": "@Q03_d_p:3@", "explication": r"C'est la probabilité de gagner **un** point. Il faut en gagner deux de plus que l'adversaire, ce qui avantage encore le meilleur joueur : la réponse doit dépasser 0,6."},
            {"id": "c", "texte": "@Q03_d_deux:3@", "explication": r"C'est \(p^2\), gagner les deux points suivants. Les retours à égalité (probabilité \(2pq = @Q03_2pq:2m@\) à chaque cycle de deux points) laissent au serveur de nouvelles chances."},
            {"id": "d", "texte": "@Q03_d_un_retour:3@", "explication": r"C'est \(p^2 + 2pq\,p^2\) : un seul retour à égalité est compté. Il peut y en avoir un nombre quelconque ; la série géométrique complète, ou le système linéaire, donne \(p^2/(1 - 2pq)\)."},
        ],
        "solution": r"""Réponse : \(\dfrac{p^2}{p^2 + q^2} = \dfrac{@Q03_p2:2m@}{@Q03_den:2m@} = \tfrac{@Q03_exact_num@}{@Q03_exact_den@} \approx @Q03_exact:3m@\).

Notations : \(p = 0{,}6\), \(q = 0{,}4\). États transitoires : E (égalité), A (avantage serveur), B (avantage receveur) ; états absorbants : jeu gagné, jeu perdu. \(h_E\), \(h_A\), \(h_B\) sont les probabilités que le serveur gagne le jeu depuis chaque état.

Étape 1, système \((\mathbf{I} - \mathbf{Q})\,\mathbf{H} = \mathbf{R}\) (leçon, partie 1) : \(h_A = p + q\,h_E\) (un point gagné termine, un point perdu ramène à E) ; \(h_B = p\,h_E\) ; \(h_E = p\,h_A + q\,h_B\).

Étape 2 : \(h_E = p^2 + pq\,h_E + qp\,h_E = @Q03_p2:2m@ + @Q03_2pq:2m@\,h_E\), donc \(h_E = @Q03_p2:2m@/@Q03_den:2m@ = \tfrac{@Q03_exact_num@}{@Q03_exact_den@}\), avec \(1 - 2pq = p^2 + q^2 = @Q03_den:2m@\).

Autre route : l'écart de points depuis l'égalité est une marche de pas \(\pm 1\) arrêtée en \(-2\) ou \(+2\) ; c'est la ruine du joueur avec \(i = 2\), \(n = 4\), et \(h = 1/(1 + r^2)\) avec \(r = q/p\) (outil 2 de la séance 1).

Pour aller plus loin : la durée moyenne depuis l'égalité vaut \(2/(p^2 + q^2) = \tfrac{@Q03_duree_num@}{@Q03_duree_den@} \approx @Q03_duree:2m@\) points (simulation : @Q03_duree_mc:2@). Simulation de la probabilité (script) : @Q03_mc:3@.""",
    },
    # ------------------------------------------------- Martingales
    {
        "id": "M4-04", "section": "math", "theme": "Martingale exponentielle, atteinte d'un niveau", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Le gain cumulé d'une stratégie, en euros, part de 0. Chaque jour, indépendamment des autres jours, il augmente de 1 euro avec probabilité 0,45 et diminue de 1 euro avec probabilité 0,55. Il n'a pas de borne inférieure : il peut devenir aussi négatif que l'on veut, et la stratégie n'est jamais arrêtée.

Quelle est la probabilité que le gain cumulé atteigne un jour la valeur +3 euros, arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q04_exact:3@", "explication": r"Oui : \((p/q)^3 = (@Q04_p_sur_q_num@/@Q04_p_sur_q_den@)^3 = @Q04_exact_num@/@Q04_exact_den@\), par la martingale \(r^{Z_t}\) arrêtée, puis en éloignant la barrière basse."},
            {"id": "b", "texte": "@Q04_d_trois:3@", "explication": r"C'est \(0{,}45^3\), monter trois jours de suite dès le départ : un seul des chemins qui mènent à +3. Les chemins qui descendent d'abord puis remontent comptent aussi."},
            {"id": "c", "texte": "@Q04_d_un@", "explication": r"La marche **symétrique** atteint tout niveau avec probabilité 1. Ici elle dérive vers le bas (\(p < 1/2\)) : avec une probabilité non nulle, elle s'éloigne vers \(-\infty\) sans jamais atteindre +3."},
            {"id": "d", "texte": "@Q04_d_exposant:3@", "explication": r"C'est \(p/q\), la probabilité de monter d'**un** euro. Monter de 3 exige de monter de 1 trois fois, chaque montée partant du niveau atteint : par la propriété de Markov, les trois probabilités se multiplient."},
        ],
        "solution": r"""Réponse : \((p/q)^3 = \tfrac{@Q04_exact_num@}{@Q04_exact_den@} \approx @Q04_exact:3m@\).

Notations : \(p = 0{,}45\), \(q = 0{,}55\), \(Z_t\) le gain au jour \(t\), \(r = q/p = @Q04_r_num@/@Q04_r_den@\). Pour un entier \(m \ge 1\), \(T_m\) est le premier jour où \(Z_t\) vaut \(3\) ou \(-m\), et \(h_m\) la probabilité que ce soit 3.

Étape 1, martingale : \(r^{Z_t}\) est une martingale, car \(\mathbb{E}[r^{X}] = p\,r + q/r = q + p = 1\) (leçon, partie 2).

Étape 2, arrêt : \(T_m\) est fini avec probabilité 1 (outil 2 de la séance 1, argument des blocs), et avant l'arrêt \(r^{-m} \le r^{Z_{t \wedge T_m}} \le r^3\) : cas (b) du théorème de Doob, \(\mathbb{E}[r^{Z_{T_m}}] = r^0 = 1\). Donc \(h_m\,r^3 + (1 - h_m)\,r^{-m} = 1\), soit \(h_m = \dfrac{1 - r^{-m}}{r^3 - r^{-m}}\).

Étape 3, limite : atteindre 3 est la réunion croissante en \(m\) des événements « atteindre 3 avant \(-m\) », car une trajectoire qui atteint 3 en un temps fini reste au-dessus d'un niveau fini avant. Par continuité croissante, la probabilité cherchée est \(\lim_m h_m = r^{-3} = (p/q)^3\), puisque \(r > 1\) donne \(r^{-m} \to 0\).

Vérification (script) : avec la barrière basse en \(-@Q04_m@\), \(h_{@Q04_m@} = @Q04_h60:6m@\), à moins de \(10^{-5}\) de la limite ; simulation : @Q04_mc:3@.""",
    },
    # ------------------------------------------------- AR(1)
    {
        "id": "M4-05", "section": "math", "theme": "AR(1), autocovariance", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""Le processus \(Y_t = 1 + 0{,}6\,Y_{t-1} + \varepsilon_t\) est stationnaire. Les chocs \(\varepsilon_t\) sont indépendants, de même loi, d'espérance 0 et d'écart-type 2, et chaque \(\varepsilon_t\) est indépendant de \(Y_{t-1}, Y_{t-2}, \dots\).

Que vaut \(\mathrm{Cov}(Y_t, Y_{t+2})\) ?""",
        "options": [
            {"id": "a", "texte": "@Q05_exact:2@", "explication": r"Oui : \(\gamma_2 = \phi^2 \gamma_0 = @Q05_phi2:2m@ \times @Q05_g0:2m@\)."},
            {"id": "b", "texte": "@Q05_d_var:2@", "explication": r"C'est \(\gamma_0 = \sigma^2/(1 - \phi^2)\), la variance de \(Y_t\), c'est-à-dire l'autocovariance au retard 0. Au retard 2, elle est multipliée par \(\phi^2\)."},
            {"id": "c", "texte": "@Q05_d_rho:2@", "explication": r"C'est l'autocorrélation \(\rho_2 = \phi^2\), sans dimension. La covariance demandée vaut \(\rho_2\) multiplié par la variance \(\gamma_0\)."},
            {"id": "d", "texte": "@Q05_d_mauvais_g0:2@", "explication": r"La variance a été prise égale à \(\sigma^2/(1 - \phi)\) au lieu de \(\sigma^2/(1 - \phi^2)\) : le dénominateur \(1 - \phi\) est celui de la moyenne \(c/(1 - \phi)\)."},
        ],
        "solution": r"""Réponse : \(\mathrm{Cov}(Y_t, Y_{t+2}) = @Q05_exact:2m@\).

Notations : \(c = 1\), \(\phi = 0{,}6\), \(\sigma = 2\) ; \(\gamma_h\) est l'autocovariance au retard \(h\).

Étape 1, variance : \(\gamma_0 = \sigma^2/(1 - \phi^2) = 4/@Q05_un_moins_phi2:2m@ = @Q05_g0:2m@\) (leçon, partie 3 : \(\mathrm{Var}(Y_t) = \phi^2\,\mathrm{Var}(Y_{t-1}) + \sigma^2\) par indépendance du choc, et stationnarité).

Étape 2, récurrence : \(\gamma_h = \phi\,\gamma_{h-1}\), car \(\varepsilon_{t+h}\) est indépendant de \(Y_t\) et une constante a une covariance nulle ; donc \(\gamma_2 = \phi^2 \gamma_0 = @Q05_phi2:2m@ \times @Q05_g0:2m@ = @Q05_exact:2m@\).

La constante \(c = 1\) ne change que la moyenne, \(\mu = c/(1 - \phi) = @Q05_mu:1m@\), pas les covariances. Simulation (script) : @Q05_mc:3@.""",
    },
    {
        "id": "M4-06", "section": "math", "theme": "AR(1), variance d'une somme", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Le processus \(Y_t = 0{,}5\,Y_{t-1} + \varepsilon_t\) est stationnaire. Les chocs \(\varepsilon_t\) sont indépendants, de même loi, d'espérance 0 et de variance 1, et chaque \(\varepsilon_t\) est indépendant de \(Y_{t-1}, Y_{t-2}, \dots\). On cumule trois valeurs consécutives, par exemple un signal sommé sur trois jours.

Que vaut \(\mathrm{Var}(Y_1 + Y_2 + Y_3)\), arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q06_exact:3@", "explication": r"Oui : \(3\gamma_0 + 2\,(2\gamma_1 + \gamma_2) = @Q06_exact_num@/@Q06_exact_den@\)."},
            {"id": "b", "texte": "@Q06_d_indep:3@", "explication": r"C'est \(3\gamma_0\) : les covariances entre dates ont été oubliées, comme si les \(Y_t\) étaient indépendants. Avec \(\phi = 0{,}5 > 0\), elles sont positives et augmentent la variance."},
            {"id": "c", "texte": "@Q06_d_une_fois:3@", "explication": r"Chaque paire de dates distinctes a été comptée une fois : \(3\gamma_0 + 2\gamma_1 + \gamma_2\). Dans \(\mathrm{Var}(\sum_u Y_u) = \sum_u \sum_v \mathrm{Cov}(Y_u, Y_v)\), les couples \((1, 2)\) et \((2, 1)\) apparaissent tous deux."},
            {"id": "d", "texte": "@Q06_d_sigma:3@", "explication": r"C'est trois fois la variance d'un **choc**, \(\sigma^2 = 1\). La variance du processus est \(\gamma_0 = \sigma^2/(1 - \phi^2)\), plus grande, et les dates sont corrélées."},
        ],
        "solution": r"""Réponse : \(\tfrac{@Q06_exact_num@}{@Q06_exact_den@} \approx @Q06_exact:3m@\).

Notations : \(\phi = 0{,}5\), \(\sigma^2 = 1\), \(\gamma_h\) l'autocovariance au retard \(h\).

Étape 1, autocovariances (leçon, partie 3) : \(\gamma_0 = 1/(1 - @Q06_phi2:2m@) = \tfrac{@Q06_g0_num@}{@Q06_g0_den@}\), \(\gamma_1 = \phi\,\gamma_0 = \tfrac{@Q06_g1_num@}{@Q06_g1_den@}\), \(\gamma_2 = \phi^2 \gamma_0 = \tfrac{@Q06_g2_num@}{@Q06_g2_den@}\).

Étape 2, variance d'une somme : \(\mathrm{Var}(Y_1 + Y_2 + Y_3) = \sum_{u=1}^{3} \sum_{v=1}^{3} \gamma_{\lvert u - v \rvert}\). Les 3 termes diagonaux donnent \(3\gamma_0\) ; les couples à distance 1, \((1, 2)\), \((2, 1)\), \((2, 3)\), \((3, 2)\), donnent \(4\gamma_1\) ; les couples à distance 2, \((1, 3)\), \((3, 1)\), donnent \(2\gamma_2\).

Étape 3 : \(3 \times \tfrac{@Q06_g0_num@}{@Q06_g0_den@} + \tfrac{@Q06_croises_num@}{@Q06_croises_den@} = \tfrac{@Q06_exact_num@}{@Q06_exact_den@}\), soit \(\gamma_0\,(3 + 4\phi + 2\phi^2)\).

Lecture : la somme de trois valeurs persistantes varie bien plus que trois valeurs indépendantes de même variance (\(@Q06_d_indep:1m@\)). Simulation (script) : @Q06_mc:3@.""",
    },
    # ------------------------------------------------- Mouvement brownien
    {
        "id": "M4-07", "section": "math", "theme": "Mouvement brownien, covariance", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""\(W\) est un mouvement brownien standard : \(W_0 = 0\), accroissements indépendants sur des intervalles disjoints, \(W_t - W_s\) de loi normale d'espérance 0 et de variance \(t - s\) pour \(s < t\).

Que vaut \(\mathrm{Var}(W_1 + W_2 + W_3)\) ?""",
        "options": [
            {"id": "a", "texte": "@Q07_exact@", "explication": r"Oui : \(\sum_{s, t} \min(s, t) = @Q07_diag@ + @Q07_croises@\), ou \(3W_1 + 2(W_2 - W_1) + (W_3 - W_2)\) de variance \(9 + 4 + 1\)."},
            {"id": "b", "texte": "@Q07_d_indep@", "explication": r"C'est \(1 + 2 + 3\), la somme des variances, qui suppose \(W_1\), \(W_2\), \(W_3\) indépendants. Ils ne le sont pas : \(\mathrm{Cov}(W_s, W_t) = \min(s, t) > 0\)."},
            {"id": "c", "texte": "@Q07_d_trois_w2@", "explication": r"C'est \(\mathrm{Var}(3W_2) = 9 \times 2\) : remplacer la somme par trois fois sa valeur du milieu suppose les trois valeurs égales, ce qui surestime la variance."},
            {"id": "d", "texte": "@Q07_d_produit@", "explication": r"Les covariances ont été prises égales à \(s\,t\) au lieu de \(\min(s, t)\) : \(\mathrm{Cov}(W_s, W_t) = \mathrm{Var}(W_s) = s\) pour \(s \le t\), car \(W_t - W_s\) est indépendant de \(W_s\)."},
        ],
        "solution": r"""Réponse : @Q07_exact@.

Étape 1 : \(\mathrm{Cov}(W_s, W_t) = \min(s, t)\) (leçon, partie 4 : pour \(s \le t\), \(W_t = W_s + (W_t - W_s)\), deux termes indépendants).

Étape 2 : \(\mathrm{Var}(W_1 + W_2 + W_3) = \sum_{s=1}^{3} \sum_{t=1}^{3} \min(s, t)\). Termes diagonaux : \(1 + 2 + 3 = @Q07_diag@\). Termes croisés : \(2\,(\min(1,2) + \min(1,3) + \min(2,3)) = 2\,(1 + 1 + 2) = @Q07_croises@\). Total : @Q07_exact@.

Contrôle par accroissements indépendants : \(W_1 + W_2 + W_3 = 3W_1 + 2(W_2 - W_1) + (W_3 - W_2)\), de variance \(9 \times 1 + 4 \times 1 + 1 \times 1 = @Q07_exact@\). Simulation (script) : @Q07_mc:2@.""",
    },
    {
        "id": "M4-08", "section": "math", "theme": "Mouvement brownien géométrique, moments", "difficulte": 5, "temps_cible_s": 150,
        "enonce": r"""Un prix vérifie \(dS_t = 0{,}05\,S_t\,dt + 0{,}2\,S_t\,dW_t\) avec \(S_0 = 100\), où \(W\) est un mouvement brownien standard et le temps est mesuré en années.

Que vaut \(\mathrm{Var}(S_1)\), arrondie à l'unité ?""",
        "options": [
            {"id": "a", "texte": "@Q08_exact:0@", "explication": r"Oui : \(s_0^2\,e^{2\mu}\,(e^{\sigma^2} - 1)\), avec \(\mathbb{E}[S_1^2] = s_0^2\,e^{2\mu + \sigma^2}\) et \(\mathbb{E}[S_1] = s_0\,e^{\mu}\)."},
            {"id": "b", "texte": "@Q08_d_naif:0@", "explication": r"C'est \(s_0^2\,\sigma^2\), la variance d'un modèle où \(S_1 = s_0\,(1 + \mu + \sigma W_1)\), linéaire en \(W_1\). Le brownien géométrique est exponentiel en \(W_1\) : la convexité de l'exponentielle et la dérive augmentent la variance."},
            {"id": "c", "texte": "@Q08_d_sans_derive:0@", "explication": r"C'est \(s_0^2\,(e^{\sigma^2} - 1)\) : la dérive a été oubliée. Elle multiplie l'espérance par \(e^{\mu}\), donc la variance par \(e^{2\mu}\)."},
            {"id": "d", "texte": "@Q08_d_mediane:0@", "explication": r"C'est \(s_0^2\,e^{2\mu - \sigma^2}(e^{\sigma^2} - 1)\) : la médiane \(s_0\,e^{\mu - \sigma^2/2}\) a été utilisée à la place de l'espérance \(s_0\,e^{\mu}\). Le terme \(-\sigma^2/2\) est compensé par \(\mathbb{E}[e^{\sigma W_1}] = e^{\sigma^2/2}\)."},
        ],
        "solution": r"""Réponse : \(\mathrm{Var}(S_1) = 100^2\,e^{@Q08_expo_moy:1m@}\,(e^{@Q08_s2:2m@} - 1) \approx @Q08_exact:2m@\), soit @Q08_exact:0@ à l'unité.

Notations : \(s_0 = 100\), \(\mu = 0{,}05\), \(\sigma = 0{,}2\).

Étape 1, solution (leçon, partie 4, par le lemme d'Itô) : \(S_1 = s_0 \exp\big(\mu - \sigma^2/2 + \sigma W_1\big) = 100 \exp\big(@Q08_drift_log:2m@ + 0{,}2\,W_1\big)\).

Étape 2, premier moment : \(\mathbb{E}[e^{\theta W_1}] = e^{\theta^2/2}\) (fonction génératrice de la loi normale) ; avec \(\theta = 0{,}2\), \(\mathbb{E}[S_1] = 100\,e^{@Q08_drift_log:2m@ + @Q08_s2_demi:2m@} = 100\,e^{0{,}05} \approx @Q08_es:3m@\).

Étape 3, second moment : \(S_1^2 = 100^2 \exp\big(@Q08_expo1:2m@ + @Q08_deux_sigma:1m@\,W_1\big)\), et \(\mathbb{E}[e^{@Q08_deux_sigma:1m@ W_1}] = e^{@Q08_expo2:2m@}\), donc \(\mathbb{E}[S_1^2] = 100^2\,e^{@Q08_expo_tot:2m@} \approx @Q08_es2:2m@\).

Étape 4 : \(\mathrm{Var}(S_1) = 100^2\,(e^{@Q08_expo_tot:2m@} - e^{@Q08_expo_moy:1m@}) = 100^2\,e^{@Q08_expo_moy:1m@}\,(e^{@Q08_s2:2m@} - 1) \approx @Q08_exact:2m@\).

Simulation (script) : @Q08_mc:1@.""",
    },
]
