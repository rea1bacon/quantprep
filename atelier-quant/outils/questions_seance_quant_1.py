"""Séance quantitative 1 : leçon express (sélection de modèles, finance quantitative, optimisation), puis 8 questions.

La leçon L-QUANT-1 (outils/lecons_quant_1.py) couvre les outils de la recherche quantitative : AIC et BIC, la double
descente, l'importance par permutation, les valeurs de Shapley, le coefficient d'information, la loi fondamentale
de la gestion active, les étiquettes qui se chevauchent, la non-stationnarité, le biais de sélection, l'erreur-type
du ratio de Sharpe, Taylor, la convexité, Lagrange et le portefeuille de variance minimale.
Toutes les valeurs : verif/seance_quant_1.py. Dans chaque question, l'option "a" est la bonne.
"""
import lecons_quant_1
import questions_diagnostic as diag

ID = "seance-quant-1"
TITRE = "Séance quantitative : sélection de modèles, finance quantitative et optimisation"
VERIF = "verif/seance_quant_1.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [{"id": "lecon", "nom": "Leçon", "budget_min": 30}, {"id": "mla", "nom": "Machine Learning avancé", "budget_min": 16}]
PRESENTATION = (
    "D'abord une leçon express d'environ 30 minutes sur les outils de la recherche quantitative : choisir un modèle "
    "(AIC, BIC, double descente), expliquer un modèle (importance par permutation, valeurs de Shapley), juger un signal "
    "(coefficient d'information, étiquettes qui se chevauchent, non-stationnarité, biais de sélection, erreur-type du "
    "ratio de Sharpe) et optimiser (Taylor, convexité, Lagrange, portefeuille de variance minimale), avec cinq points "
    "de contrôle corrigés immédiatement. Ensuite 8 questions à choix multiple graduées, dont la moitié de difficulté 4.\n\n"
    "Objectif de vitesse : chaque question se résout dans son temps cible avec les résultats de la leçon (tableau "
    "« Réflexes en 30 secondes »). Au-delà du double du temps cible affiché, choisis « Je ne sais pas » et passe à la "
    "suivante : c'est la règle d'abandon à suivre dans un test chronométré."
)
AVERTISSEMENTS = [
    "Les questions sont présentées ici en choix multiple à une seule bonne réponse. Cherche la réponse avant de "
    "lire les options, pour t'entraîner aussi au format à réponse numérique.",
    "Ici le retour en arrière est permis et il n'y a pas de pénalité : ce sont les règles de cette plateforme, pas "
    "celles de tout test en ligne.",
]

LECONS = [lecons_quant_1.LECON_QUANT_1]
CODES = []

QUESTIONS = [
    # ------------------------------------------------- Partie 1 : AIC et BIC
    {
        "id": "Q1-01", "section": "mla", "theme": "Sélection de modèles, AIC contre BIC", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""Deux modèles A et B sont ajustés par maximum de vraisemblance sur les mêmes \(n = 200\) observations de la même variable cible. B contient tous les paramètres de A et 3 paramètres de plus. La log-vraisemblance maximale de B dépasse celle de A de 5.

On note \(k\) le nombre de paramètres estimés d'un modèle et \(\hat\ell\) sa log-vraisemblance maximale ; \(\mathrm{AIC} = -2\hat\ell + 2k\) et \(\mathrm{BIC} = -2\hat\ell + k \ln n\), et chaque critère retient le modèle de plus petite valeur.

Quel modèle chaque critère retient-il ?""",
        "options": [
            {"id": "a", "texte": "L'AIC retient B, le BIC retient A.",
             "explication": r"Oui : le gain \(2 \times 5 = @Q01_gain@\) dépasse le coût AIC \(2 \times 3 = @Q01_pen_aic@\), mais pas le coût BIC \(3 \ln 200 \approx @Q01_pen_bic:2m@\)."},
            {"id": "b", "texte": "Les deux critères retiennent B.",
             "explication": r"Ce serait vrai si le BIC coûtait aussi 2 par paramètre. Il coûte \(\ln 200 \approx @Q01_ln_n:2m@\) par paramètre, soit \(@Q01_pen_bic:2m@\) pour 3 paramètres, plus que le gain \(@Q01_gain@\)."},
            {"id": "c", "texte": "Les deux critères retiennent A.",
             "explication": r"L'AIC compare \(2\Delta\ell = @Q01_gain@\) à \(2\Delta k = @Q01_pen_aic@\) : le gain l'emporte, l'AIC retient B. Seul le BIC, plus sévère dès \(n \ge @L_n_seuil@\), retient A."},
            {"id": "d", "texte": "L'AIC retient A, le BIC retient B.",
             "explication": r"Rôles inversés : pour \(n \ge @L_n_seuil@\), \(\ln n > 2\), donc le BIC est toujours le plus sévère des deux. Il ne peut pas retenir le grand modèle quand l'AIC retient le petit."},
        ],
        "solution": r"""Réponse : l'AIC retient B, le BIC retient A.

Notations : \(\Delta k = 3\) paramètres de plus, \(\Delta\ell = 5\) de log-vraisemblance de plus pour B.

Étape 1, AIC : \(\mathrm{AIC}_B - \mathrm{AIC}_A = -2\Delta\ell + 2\Delta k = -@Q01_gain@ + @Q01_pen_aic@ = @Q01_daic@ < 0\) : B a le plus petit AIC.

Étape 2, BIC : \(\mathrm{BIC}_B - \mathrm{BIC}_A = -2\Delta\ell + \Delta k \ln n = -@Q01_gain@ + 3 \ln 200 \approx @Q01_dbic:3m@ > 0\) : A a le plus petit BIC.

Lecture : les critères divergent exactement quand \(2\Delta k < 2\Delta\ell < \Delta k \ln n\) (leçon, partie 1). Ici \(@Q01_pen_aic@ < @Q01_gain@ < @Q01_pen_bic:2m@\).""",
    },
    # ------------------------------------------------- Partie 3 : importance par permutation
    {
        "id": "Q1-02", "section": "mla", "theme": "Importance par permutation, variables redondantes", "difficulte": 3, "temps_cible_s": 100,
        "enonce": r"""Les données ont deux colonnes identiques : \(X_2 = X_1\), où \(X_1\) est de variance 1. La cible vaut \(Y = X_1 + \varepsilon\), où \(\varepsilon\) est centré, de variance 1, indépendant de \(X_1\). Le modèle retenu est \(f(\mathbf{x}) = 0{,}5\,x_1 + 0{,}5\,x_2\), qui prédit exactement \(X_1\).

L'importance de \(x_1\) est définie comme la hausse de l'erreur quadratique espérée quand on remplace \(X_1\), et elle seule, par une copie indépendante de même loi (idéalisation d'une permutation sur un très grand jeu de validation), \(X_2\) restant inchangée. Que vaut-elle ?""",
        "options": [
            {"id": "a", "texte": "@Q02_exact:1@",
             "explication": r"Oui : \(2\,b_1^2\,\mathrm{Var}(X_1) = 2 \times 0{,}25 \times 1\). Simulation (script) : @Q02_exact_mc:3@."},
            {"id": "b", "texte": "@Q02_A:0@",
             "explication": r"C'est l'importance de \(x_1\) pour le modèle \(f(\mathbf{x}) = x_1\), qui fait pourtant les mêmes prédictions. Ici \(b_1 = 0{,}5\), et l'importance est proportionnelle à \(b_1^2\)."},
            {"id": "c", "texte": "@Q02_d_moitie:0@",
             "explication": r"C'est la moitié de l'importance du modèle \(x_1\), comme si l'importance se partageait linéairement entre les deux copies. Elle dépend de \(b_1^2\) : diviser le coefficient par 2 divise l'importance par 4."},
            {"id": "d", "texte": "@Q02_d_sans2:2@",
             "explication": r"C'est \(b_1^2\,\mathrm{Var}(X_1)\) : il manque le facteur 2, car \(\mathbb{E}[(X_1 - \tilde X_1)^2] = 2\,\mathrm{Var}(X_1)\) pour une copie indépendante \(\tilde X_1\)."},
        ],
        "solution": r"""Réponse : @Q02_exact:1@.

Notations : \(\tilde X_1\) est la copie indépendante de \(X_1\), de même loi.

Étape 1 : après remplacement, \(Y - f = \varepsilon + X_1 - 0{,}5\,\tilde X_1 - 0{,}5\,X_1 = \varepsilon + 0{,}5\,(X_1 - \tilde X_1)\).

Étape 2 : \(\varepsilon\) est centré et indépendant de \(X_1\) et de \(\tilde X_1\), donc le double produit est d'espérance nulle ; \(\mathbb{E}[(X_1 - \tilde X_1)^2] = 2\,\mathrm{Var}(X_1) = 2\).

Étape 3 : l'erreur passe de 1 à \(1 + 0{,}25 \times 2\) : hausse \(@Q02_exact:1m@\) (leçon, partie 3, résultat P1). Simulation (script) : @Q02_exact_mc:3@.

Lecture : le modèle \(x_1\) seul a les mêmes prédictions et une importance \(@Q02_A:0m@\) pour \(x_1\). L'importance par permutation décrit le modèle choisi, pas l'information contenue dans les données (dilution entre variables redondantes, `outils/lecons_arbres.py`, partie 7, I3).""",
    },
    # ------------------------------------------------- Partie 3 : Shapley
    {
        "id": "Q1-03", "section": "mla", "theme": "Valeurs de Shapley, interaction", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Le modèle est \(f(\mathbf{x}) = x_1 x_2\). Les variables \(X_1\) et \(X_2\) sont indépendantes, d'espérances respectives 1 et 2. Pour une coalition \(S \subset \{1, 2\}\), la valeur est \(v(S) = \mathbb{E}[f(\mathbf{X}) \mid \mathbf{X}_S = \mathbf{x}_S]\), et la valeur de Shapley de la variable \(j\) est la moyenne, sur les deux ordres d'arrivée équiprobables des variables, de sa contribution \(v(S \cup \{j\}) - v(S)\), où \(S\) est l'ensemble des variables arrivées avant elle.

Que vaut la valeur de Shapley de \(x_1\) au point \(\mathbf{x} = (3, 4)\) ?""",
        "options": [
            {"id": "a", "texte": "@Q03_exact@",
             "explication": r"Oui : moyenne de la contribution en premier, \(@Q03_c1@\), et en second, \(@Q03_c2@\)."},
            {"id": "b", "texte": "@Q03_d_dernier@",
             "explication": r"C'est la contribution de \(x_1\) quand elle arrive en second, \(v(\{1, 2\}) - v(\{2\}) = @Q03_fx@ - @Q03_v2@\). Il faut la moyenner avec sa contribution quand elle arrive en premier, \(@Q03_c1@\)."},
            {"id": "c", "texte": "@Q03_d_premier@",
             "explication": r"C'est la contribution de \(x_1\) quand elle arrive en premier, \(v(\{1\}) - v(\emptyset) = @Q03_v1@ - @Q03_ef@\). C'est aussi, par coïncidence, la valeur de Shapley de \(x_2\) ; il manque la moyenne avec la contribution en second, \(@Q03_c2@\)."},
            {"id": "d", "texte": "@Q03_d_moitie@",
             "explication": r"C'est la moitié de l'écart total \(f(\mathbf{x}) - \mathbb{E}f(\mathbf{X}) = @Q03_diff@\), comme si les deux variables jouaient des rôles symétriques. Elles ne le font pas : chacune multiplie l'autre, et \(x_2\) comme son espérance sont plus grands que \(x_1\) et la sienne, donc l'écart de \(x_1\) à sa moyenne pèse davantage."},
        ],
        "solution": r"""Réponse : @Q03_exact@.

Étape 1, valeurs des coalitions, par indépendance : \(v(\emptyset) = \mathbb{E}[X_1]\,\mathbb{E}[X_2] = @Q03_ef@\) ; \(v(\{1\}) = 3 \times \mathbb{E}[X_2] = @Q03_v1@\) ; \(v(\{2\}) = \mathbb{E}[X_1] \times 4 = @Q03_v2@\) ; \(v(\{1, 2\}) = 3 \times 4 = @Q03_fx@\).

Étape 2, contributions de \(x_1\) : en premier, \(@Q03_v1@ - @Q03_ef@ = @Q03_c1@\) ; en second, \(@Q03_fx@ - @Q03_v2@ = @Q03_c2@\).

Étape 3 : \(\phi_1 = \tfrac12(@Q03_c1@ + @Q03_c2@) = @Q03_exact@\) (leçon, partie 3, formule à deux variables).

Contrôle par l'efficacité : \(\phi_2 = @Q03_phi2@\), et \(\phi_1 + \phi_2 = @Q03_diff@ = f(\mathbf{x}) - \mathbb{E}f(\mathbf{X})\). Simulation sur des ordres tirés au hasard (script) : @Q03_exact_mc:3@.""",
    },
    # ------------------------------------------------- Partie 4 : coefficient d'information
    {
        "id": "Q1-04", "section": "mla", "theme": "Coefficient d'information, corrélation de rang", "difficulte": 2, "temps_cible_s": 120,
        "enonce": r"""À une date donnée, six actifs ont les valeurs de signal suivantes et, sur la période suivante, les rendements suivants (en %) :

| Actif | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| signal | 0,8 | −0,3 | 1,5 | 0,1 | −1,2 | 0,6 |
| rendement futur | 1,2 | −0,5 | 0,9 | 0,4 | −0,8 | −0,2 |

Le coefficient d'information de la date est la corrélation de rang de Spearman entre le signal et le rendement futur. Que vaut-il, arrondi à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q04_exact:3@",
             "explication": r"Oui : \(\sum d_i^2 = @Q04_d2@\), et \(1 - @Q04_6d2@/@Q04_N3@ = \tfrac{@Q04_exact_num@}{@Q04_exact_den@}\)."},
            {"id": "b", "texte": "@Q04_d_pearson:3@",
             "explication": r"C'est la corrélation de Pearson des valeurs brutes. Le coefficient d'information se calcule sur les rangs, ce qui le rend insensible à l'échelle et aux valeurs extrêmes."},
            {"id": "c", "texte": "@Q04_d_sans6:3@",
             "explication": r"Le facteur 6 du numérateur a été oublié : \(1 - @Q04_d2@/@Q04_N3@\) au lieu de \(1 - 6 \times @Q04_d2@/@Q04_N3@\)."},
            {"id": "d", "texte": "@Q04_d_cube:3@",
             "explication": r"Le dénominateur \(N(N^2 - 1) = @Q04_N3@\) a été remplacé par \(N^3 = @Q04_N_cube@\) : la variance des rangs vaut \((N^2 - 1)/12\), pas \(N^2/12\)."},
        ],
        "solution": r"""Réponse : \(\tfrac{@Q04_exact_num@}{@Q04_exact_den@} \approx @Q04_exact:3m@\).

Étape 1, rangs (1 pour la plus petite valeur, aucune égalité) : signal \(@Q04_rangs_signal@\), rendement \(@Q04_rangs_rend@\), dans l'ordre A à F.

Étape 2, écarts \(d_i\) : \(@Q04_d@\), d'où \(\sum d_i^2 = @Q04_d2@\).

Étape 3 : \(1 - \dfrac{6 \times @Q04_d2@}{6 \times (6^2 - 1)} = 1 - \tfrac{@Q04_6d2@}{@Q04_N3@} = \tfrac{@Q04_exact_num@}{@Q04_exact_den@} \approx @Q04_exact:3m@\) (leçon, partie 4 : formule démontrée sans égalités, contrôlée par `scipy.stats.spearmanr` dans le script).

Lecture : sous l'hypothèse d'absence de lien, l'écart-type d'un IC sur 6 actifs vaut \(1/\sqrt{5} \approx @Q04_se_nulle:2m@\) : un IC élevé sur une seule date de 6 actifs ne prouve presque rien.""",
    },
    # ------------------------------------------------- Partie 6 : biais de sélection
    {
        "id": "Q1-05", "section": "mla", "theme": "Biais de sélection, meilleur de N stratégies sans effet", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Un chercheur teste 50 stratégies sur 4 années de rendements quotidiens (1008 jours). Les rendements de chaque stratégie sont i.i.d. normaux d'espérance **nulle**, et les 50 stratégies sont indépendantes. Pour chacune, le ratio de Sharpe annualisé estimé est \(\sqrt{252}\,m/s\), où \(m\) et \(s\) sont la moyenne et l'écart-type empiriques des rendements quotidiens. Il retient la stratégie de plus grand ratio estimé.

On donne : l'espérance du maximum de 50 variables indépendantes de loi \(\mathcal{N}(0, 1)\) vaut environ \(@Q05_emax:3m@\). Quelle est, à deux décimales, l'espérance du ratio de Sharpe annualisé estimé de la stratégie retenue ?""",
        "options": [
            {"id": "a", "texte": "@Q05_exact:2@",
             "explication": r"Oui : chaque ratio annualisé vaut \(t/\sqrt{4}\), où \(t\) est une statistique de Student très proche de \(\mathcal{N}(0, 1)\) : \(@Q05_emax:3m@/2\)."},
            {"id": "b", "texte": "@Q05_d_sans_T:2@",
             "explication": r"C'est l'espérance du maximum de 50 normales centrées réduites, sans la division par \(\sqrt{T} = 2\). C'est \(\sqrt{T}\) fois le ratio annualisé qui est approximativement \(\mathcal{N}(0, 1)\), pas le ratio lui-même."},
            {"id": "c", "texte": "@Q05_d_borne:2@",
             "explication": r"C'est \(\sqrt{2 \ln 50}/2\) : le majorant \(\sqrt{2\ln m}\) a été pris pour la valeur. Il majore l'espérance du maximum, \(@Q05_borne_z:3m@\) contre \(@Q05_emax:3m@\)."},
            {"id": "d", "texte": "0,00",
             "explication": r"C'est l'espérance du ratio d'une stratégie prise au hasard. La sélection du maximum crée un biais positif : la meilleure de 50 stratégies sans effet a un ratio espéré nettement positif."},
        ],
        "solution": r"""Réponse : environ @Q05_exact:2@.

Étape 1 : pour chaque stratégie, \(\sqrt{T}\,\hat S = \sqrt{n/252}\,\sqrt{252}\,m/s = \sqrt{n}\,m/s\) est la statistique de Student du test « espérance nulle » (question DIAG-A5 du test de positionnement), avec \(T = 4\) années et \(n = 1008\) jours. Sous l'hypothèse d'espérance nulle et de normalité, elle suit une loi de Student à 1007 degrés de liberté, presque \(\mathcal{N}(0, 1)\).

Étape 2 : les 50 statistiques sont indépendantes ; l'espérance de leur maximum vaut environ \(@Q05_emax:3m@\) (leçon, partie 6 : entre le minorant \(@L_max_bas_50:3m@\) et le majorant \(\sqrt{2\ln 50} \approx @Q05_borne_z:3m@\)).

Étape 3 : \(\mathbb{E}[\max \hat S] \approx @Q05_emax:3m@/\sqrt{4} \approx @Q05_exact:3m@\).

Contrôles (script) : avec la loi de Student exacte, \(@Q05_student:3m@\) ; simulation directe de 2000 répétitions des 50 stratégies : @Q05_student_mc:3@. Un ratio de Sharpe supérieur à 1 sur 4 ans ne prouve rien si l'on a essayé 50 stratégies.""",
    },
    # ------------------------------------------------- Partie 6 : erreur-type du Sharpe
    {
        "id": "Q1-06", "section": "mla", "theme": "Erreur-type du ratio de Sharpe annualisé", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Une stratégie a des rendements mensuels i.i.d. normaux ; on dispose de 10 années, soit \(n = 120\) mois. Son vrai ratio de Sharpe mensuel est \(\theta\), et son vrai ratio annualisé \(\theta_a = \sqrt{12}\,\theta = 1{,}2\).

On admet le résultat de Lo (2002) : l'estimateur mensuel \(\hat\theta\) (moyenne empirique divisée par l'écart-type empirique) vérifie approximativement \(\mathrm{Var}(\hat\theta) \approx (1 + \theta^2/2)/n\). On estime le ratio annualisé par \(\sqrt{12}\,\hat\theta\). Quelle est son erreur-type approchée, à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q06_exact:3@",
             "explication": r"Oui : \(\sqrt{12\,(1 + \theta^2/2)/120}\) avec \(\theta^2 = 1{,}44/12\), soit \(\sqrt{(1 + 1{,}44/24)/10}\)."},
            {"id": "b", "texte": "@Q06_d_sans_sr:3@",
             "explication": r"C'est \(1/\sqrt{10}\), l'erreur-type sous l'hypothèse \(\theta = 0\) : le terme \(\theta^2/2\) a été négligé. Il vaut ici \(1{,}44/24 = @Q06_terme:2m@\), pas négligeable à trois décimales."},
            {"id": "c", "texte": "@Q06_d_annuel:3@",
             "explication": r"Le ratio **annualisé** 1,2 a été mis dans le terme \(\theta^2/2\), qui demande le ratio **mensuel** \(\theta = 1{,}2/\sqrt{12}\)."},
            {"id": "d", "texte": "@Q06_d_periode:3@",
             "explication": r"C'est l'erreur-type du ratio **mensuel** : il faut la multiplier par \(\sqrt{12}\) pour l'estimateur annualisé, dont la variance est 12 fois plus grande."},
        ],
        "solution": r"""Réponse : environ \(@Q06_exact:3m@\).

Étape 1 : \(\theta = 1{,}2/\sqrt{12}\), donc \(\theta^2 = 1{,}44/12 = @Q06_theta2:2m@\) et \(\mathrm{Var}(\hat\theta) \approx (1 + @Q06_terme:2m@)/120\).

Étape 2 : \(\mathrm{Var}(\sqrt{12}\,\hat\theta) = 12\,\mathrm{Var}(\hat\theta) \approx @Q06_var_an:3m@\), d'où une erreur-type \(\sqrt{@Q06_var_an:3m@} \approx @Q06_exact:3m@\). Formule directe de la leçon (partie 6) : \(\sqrt{(1 + \theta_a^2/(2q))/T}\) avec \(q = 12\), \(T = 10\).

Contrôle (script) : écart-type de l'estimateur sur 40 000 simulations de 120 mois, @Q06_mc:3@ ; l'écart vient de ce que la formule est asymptotique.

Lecture : un Sharpe annualisé estimé à 1,2 sur 10 ans de données mensuelles a un intervalle approché à 95 % d'environ \(\pm 2 \times @Q06_exact:2m@\) : l'incertitude reste grande.""",
    },
    # ------------------------------------------------- Partie 5 : étiquettes qui se chevauchent
    {
        "id": "Q1-07", "section": "mla", "theme": "Étiquettes qui se chevauchent, erreur-type de la moyenne", "difficulte": 4, "temps_cible_s": 150,
        "enonce": r"""Les rendements quotidiens \(X_1, X_2, \dots\) d'un actif sont non corrélés, de même variance \(v\). Chaque jour \(t = 1, \dots, 1000\), on calcule l'étiquette \(Y_t = X_t + X_{t+1} + \dots + X_{t+19}\), le rendement des 20 jours suivants, et l'on estime l'espérance des étiquettes par leur moyenne \(\bar Y\).

L'erreur-type naïve traite les 1000 étiquettes comme indépendantes, chacune de variance \(20v\) : elle vaut \(\sqrt{20v/1000}\). Par quel facteur faut-il la multiplier pour obtenir l'écart-type exact de \(\bar Y\), arrondi à deux décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q07_rapport_se:2@",
             "explication": r"Oui : le rapport des variances vaut \(@Q07_somme_c2@/@Q07_Th@ = @Q07_rapport_var:3m@\), et l'on prend sa racine."},
            {"id": "b", "texte": "@Q07_rapport_var:2@",
             "explication": r"C'est le rapport des **variances**. La question porte sur l'erreur-type, un écart-type : il faut prendre la racine carrée."},
            {"id": "c", "texte": "1,00",
             "explication": r"Ce serait vrai pour des étiquettes non chevauchantes. Deux étiquettes voisines partagent 19 rendements quotidiens : elles sont fortement corrélées."},
            {"id": "d", "texte": "@Q07_d_corr:2@",
             "explication": r"C'est la corrélation \((h - 1)/h\) entre deux étiquettes voisines, pas le facteur de correction : la variance de la moyenne additionne les covariances de toutes les paires d'étiquettes qui se chevauchent."},
        ],
        "solution": r"""Réponse : environ @Q07_rapport_se:2@.

Notations : \(h = 20\), \(T = 1000\) ; \(c_s\) est le nombre d'étiquettes qui contiennent le rendement du jour \(s\).

Étape 1 : \(\bar Y = \tfrac{1}{T}\sum_s c_s X_s\), donc \(\mathrm{Var}(\bar Y) = \tfrac{v}{T^2}\sum_s c_s^2\), les \(X_s\) étant non corrélés (leçon, partie 5).

Étape 2 : \(c_s = 20\) pour les \(T - h + 1 = @Q07_nb_plein@\) jours \(s = 20, \dots, 1000\) ; aux deux bords, \(c_s\) prend les valeurs 1 à 19, une fois de chaque côté. \(\sum_s c_s^2 = @Q07_nb_plein@ \times @Q07_h2@ + 2\sum_{k=1}^{19} k^2 = @Q07_cle@ + @Q07_bords@ = @Q07_somme_c2@\).

Étape 3 : rapport des variances \(\dfrac{v \sum_s c_s^2/T^2}{20v/T} = \dfrac{@Q07_somme_c2@}{@Q07_Th@} = @Q07_rapport_var:3m@\), et facteur sur l'erreur-type \(\sqrt{@Q07_rapport_var:3m@} \approx @Q07_rapport_se:3m@\). Approximation rapide : le rapport tend vers \(h\), donc le facteur vers \(\sqrt{20} \approx @Q07_racine_h:3m@\).

Contrôle (script) : variance de \(\bar Y\) sur 20 000 simulations avec \(v = 1\), @Q07_var_moyenne_mc:4@, pour une valeur exacte \(@Q07_var_moyenne_exact:4m@\).""",
    },
    # ------------------------------------------------- Partie 7 : Lagrange, variance minimale
    {
        "id": "Q1-08", "section": "mla", "theme": "Portefeuille de variance minimale, corrélation forte", "difficulte": 3, "temps_cible_s": 100,
        "enonce": r"""Deux actifs ont des rendements de volatilités \(\sigma_1 = 0{,}1\) et \(\sigma_2 = 0{,}2\) et de corrélation \(\rho = 0{,}8\). On cherche les poids \((w_1, w_2)\) qui minimisent la variance du portefeuille sous la contrainte \(w_1 + w_2 = 1\), les poids négatifs (ventes à découvert) étant permis.

Que vaut \(w_2\) ?""",
        "options": [
            {"id": "a", "texte": "@Q08_exact:3@",
             "explication": r"Oui : \(w_2 = (\sigma_1^2 - \rho\sigma_1\sigma_2)/(\sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2)\), négatif car \(\rho = 0{,}8 > \sigma_1/\sigma_2 = @Q08_ratio:1m@\)."},
            {"id": "b", "texte": "@Q08_d_invvar:1@",
             "explication": r"Poids proportionnels aux inverses des variances : optimal seulement pour une corrélation nulle (DIAG-M8). Ici, sa variance \(@Q08_var_invvar:5m@\) dépasse le minimum \(@Q08_var:4m@\)."},
            {"id": "c", "texte": "@Q08_d_invvol:3@",
             "explication": r"Poids proportionnels aux inverses des volatilités : ce n'est l'optimum d'aucun critère de variance ici, et la corrélation est ignorée."},
            {"id": "d", "texte": "0,5",
             "explication": r"L'équipondération ne minimise la variance que si les deux actifs ont la même volatilité."},
        ],
        "solution": r"""Réponse : \(w_2 = \tfrac{@Q08_exact_num@}{@Q08_exact_den@} \approx @Q08_exact:3m@\).

Étape 1 : variance \(f(\mathbf{w}) = \mathbf{w}^\top\boldsymbol{\Sigma}\mathbf{w}\), avec \(\boldsymbol{\Sigma}\) de coefficients \(\sigma_1^2 = @Q08_s1c:2m@\), \(\sigma_2^2 = @Q08_s2c:2m@\) et \(\rho\sigma_1\sigma_2 = @Q08_cov:3m@\) ; \(\boldsymbol{\Sigma}\) est définie positive car \(\lvert \rho \rvert < 1\), donc \(f\) est strictement convexe (leçon, partie 7, C3).

Étape 2, Lagrange : \(2\boldsymbol{\Sigma}\mathbf{w} = \lambda\,\mathbf{1}\), donc \(\mathbf{w}\) est proportionnel à \(\boldsymbol{\Sigma}^{-1}\mathbf{1}\), et la contrainte normalise : à deux actifs, \(w_2 = \dfrac{\sigma_1^2 - \rho\sigma_1\sigma_2}{\sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2}\).

Étape 3 : numérateur \(@Q08_s1c:2m@ - @Q08_cov:3m@ = @Q08_num:3m@\), dénominateur \(@Q08_somme:2m@ - @Q08_2cov:3m@ = @Q08_den:3m@\), d'où \(w_2 = \tfrac{@Q08_exact_num@}{@Q08_exact_den@}\) et \(w_1 = \tfrac{@Q08_w1_num@}{@Q08_w1_den@}\).

Lecture : avec une forte corrélation, vendre à découvert l'actif le plus volatil couvre une partie du risque du premier ; la volatilité minimale vaut \(@Q08_vol:4m@\), sous celle du premier actif. Contrôle (script) : recherche sur une grille fine de \(w_2\).""",
    },
]
