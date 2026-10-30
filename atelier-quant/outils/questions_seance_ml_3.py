"""Séance de ML 3 : leçon express sur la classification, les métriques et la validation, puis 8 questions à choix multiple.

Programme : régression logistique, métriques, plus proches voisins, mise à l'échelle, malédiction de la
dimension, validation croisée temporelle avec purge et embargo.
Toutes les valeurs : verif/seance_ml_3.py. Dans chaque question, l'option "a" est la bonne.
"""
import lecons_ml_3
import questions_diagnostic as diag

ID = "seance-ml-3"
TITRE = "Séance de ML 3 : classification, métriques et validation"
VERIF = "verif/seance_ml_3.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [{"id": "lecon", "nom": "Leçon", "budget_min": 30}, {"id": "mlf", "nom": "Machine Learning fondamental", "budget_min": 20}]
PRESENTATION = (
    "D'abord une leçon express d'environ 20 minutes : régression logistique (gradient, convexité, rapport de cotes, "
    "séparation), métriques de classification (précision, rappel, F1, ROC et AUC, log-loss, Brier, calibration, classes "
    "déséquilibrées), plus proches voisins, mise à l'échelle et malédiction de la dimension, puis validation (fuites, "
    "validation croisée temporelle, purge et embargo). Cinq points de contrôle sont corrigés immédiatement. Ensuite "
    "8 questions à choix multiple, dont 5 de difficulté 4 ou 5.\n\n"
    "Objectif de vitesse : chaque question se résout en moins de 2 minutes et demie avec la leçon. Au-delà du double "
    "du temps cible affiché, choisis « Je ne sais pas » et passe à la suivante : c'est la règle d'abandon à suivre dans un test chronométré."
)
AVERTISSEMENTS = [
    "Les questions de ML sont présentées ici en choix multiple à une seule bonne réponse. Cherche la réponse avant de "
    "lire les options, pour t'entraîner aussi au format à réponse numérique.",
    "Ici le retour en arrière est permis et il n'y a pas de pénalité : ce sont les règles de cette plateforme, pas "
    "celles de tout test en ligne.",
]

LECONS = [lecons_ml_3.LECON_ML_3]
CODES = []

QUESTIONS = [
    # ------------------------------------------------- Régression logistique
    {
        "id": "ML3-01", "section": "mlf", "theme": "Régression logistique : rapport de cotes", "difficulte": 2, "temps_cible_s": 75,
        "enonce": r"""Une régression logistique \(\mathbb{P}(Y = 1 \mid \mathbf{x}) = \sigma(\mathbf{x}^\top \boldsymbol{\beta})\), avec \(\sigma(t) = 1/(1 + e^{-t})\), a pour coefficient de la variable \(x_j\) la valeur \(\beta_j = 0{,}4\). Pour un individu donné, la probabilité prédite vaut 0,2.

On augmente \(x_j\) de 1 pour cet individu, toutes les autres variables restant fixées. Que vaut la nouvelle probabilité prédite, arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q01_exact:3@", "explication": r"Oui : la cote \(0{,}2/0{,}8\) est multipliée par \(e^{0{,}4}\), puis on revient à la probabilité par \(c/(1 + c)\)."},
            {"id": "b", "texte": "@Q01_d_prob:3@", "explication": r"C'est \(0{,}2 \times e^{0{,}4}\) : le facteur \(e^{\beta_j}\) a été appliqué à la probabilité. Il multiplie la **cote** \(\pi/(1 - \pi)\), pas la probabilité."},
            {"id": "c", "texte": "@Q01_d_lin:3@", "explication": r"C'est l'approximation linéaire \(0{,}2 + \beta_j \pi(1 - \pi) = 0{,}2 + 0{,}4 \times @Q01_derivee:2m@\), qui utilise la dérivée au point de départ. La probabilité n'est pas linéaire en \(x_j\) : pour un pas de 1, il faut le calcul exact par la cote."},
            {"id": "d", "texte": "@Q01_d_sig:3@", "explication": r"C'est \(\sigma(0{,}4)\) : le point de départ a été oublié, comme si le score linéaire valait 0 avant l'augmentation. Le nouveau score vaut \(\ln(0{,}2/0{,}8) + 0{,}4\)."},
        ],
        "solution": r"""Réponse : \(@Q01_exact:3m@\).

Étape 1 : la cote de départ vaut \(0{,}2/0{,}8 = @Q01_cote0:2m@\).

Étape 2 : le logit est linéaire en \(x_j\), de pente \(\beta_j\) ; augmenter \(x_j\) de 1 multiplie la cote par \(e^{0{,}4} \approx @Q01_or:4m@\) (leçon, partie 1, énoncé 5). Nouvelle cote : \(@Q01_cote1:4m@\).

Étape 3 : retour à la probabilité, \(c/(1 + c) = @Q01_cote1:4m@/(1 + @Q01_cote1:4m@) \approx @Q01_exact:4m@\).

Contrôle : \(\sigma\big(\ln(0{,}25) + 0{,}4\big)\) donne la même valeur (script).""",
    },
    {
        "id": "ML3-02", "section": "mlf", "theme": "Régression logistique : gradient de la log-loss", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""Régression logistique avec constante sur trois observations : \(x = (-1, 0, 2)\), étiquettes \(\mathbf{y} = (0, 1, 1)\). La matrice \(\mathbf{X}\) a pour lignes \((1, x_i)\). Pour \(\mathbf{b} = (b_0, b_1)\), la probabilité prédite est \(\pi_i = \sigma(b_0 + b_1 x_i)\), avec \(\sigma(t) = 1/(1 + e^{-t})\), et la log-loss moyenne est \(\ell(\mathbf{b}) = -\tfrac13 \sum_{i=1}^3 \big[y_i \ln \pi_i + (1 - y_i)\ln(1 - \pi_i)\big]\).

Que vaut le gradient \(\nabla \ell\) au point \(\mathbf{b} = (0, 1)\), coordonnées arrondies à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q02_exact:3@", "explication": r"Oui : \(\tfrac13 \mathbf{X}^\top (\boldsymbol{\pi} - \mathbf{y})\), avec \(\boldsymbol{\pi} = (\sigma(-1), \sigma(0), \sigma(2))\)."},
            {"id": "b", "texte": "@Q02_d_signe:3m@", "explication": r"C'est \(\tfrac13 \mathbf{X}^\top (\mathbf{y} - \boldsymbol{\pi})\), le gradient de la log-vraisemblance moyenne, que l'on **maximise**. La log-loss en est l'opposé : son gradient est \(\tfrac13 \mathbf{X}^\top (\boldsymbol{\pi} - \mathbf{y})\)."},
            {"id": "c", "texte": "@Q02_d_sans_n:3m@", "explication": r"C'est \(\mathbf{X}^\top (\boldsymbol{\pi} - \mathbf{y})\) sans le facteur \(\tfrac13\) : le gradient de la somme des pertes, pas de leur moyenne telle que l'énoncé la définit."},
            {"id": "d", "texte": "@Q02_d_poids:3m@", "explication": r"Les résidus \(\pi_i - y_i\) ont été multipliés par les poids \(\pi_i(1 - \pi_i)\). Ces poids apparaissent dans la hessienne, pas dans le gradient : la dérivée de \(\ln(1 + e^{t}) - y t\) est \(\sigma(t) - y\), sans autre facteur."},
        ],
        "solution": r"""Réponse : \(\nabla \ell(0, 1) \approx @Q02_exact:3m@\).

Étape 1 (leçon, partie 1, énoncé 1) : \(\nabla \ell(\mathbf{b}) = \tfrac1n \mathbf{X}^\top (\boldsymbol{\pi} - \mathbf{y})\), car la dérivée de \(\ln(1 + e^{t}) - y t\) par rapport à \(t\) vaut \(\sigma(t) - y\) et \(\partial t_i/\partial b_j = x_{ij}\).

Étape 2 : scores \(t = (-1, 0, 2)\), probabilités \(\boldsymbol{\pi} \approx @Q02_pi:4m@\), résidus \(\boldsymbol{\pi} - \mathbf{y} \approx @Q02_res:4m@\).

Étape 3 : coordonnée de la constante : somme des résidus \(\approx @Q02_somme:4m@\) ; coordonnée de la pente : \(\sum_i x_i (\pi_i - y_i) \approx @Q02_somme_x:4m@\). On divise par 3.

Lecture : les deux coordonnées sont négatives, donc augmenter \(b_0\) et \(b_1\) fait baisser la perte ; un pas de descente de longueur 0,1 la fait passer de \(@Q02_loss_b:4m@\) à \(@Q02_loss_pas:4m@\) (script, qui contrôle aussi le gradient par différences finies).""",
    },
    {
        "id": "ML3-03", "section": "mlf", "theme": "Régression logistique : existence du maximum de vraisemblance, séparation", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Régression logistique **sans pénalité**, avec constante et une variable \(x\), qui prend les valeurs \(x = (1, 2, 3, 4)\) pour quatre observations. On considère quatre vecteurs d'étiquettes possibles pour ces quatre observations, dans cet ordre :

- J1 : \((0, 1, 0, 1)\) ;
- J2 : \((0, 0, 1, 1)\) ;
- J3 : \((0, 0, 0, 1)\) ;
- J4 : \((1, 1, 1, 1)\).

Pour lequel de ces jeux la log-loss moyenne, vue comme fonction des deux coefficients \((b_0, b_1)\) réels, **atteint-elle** son minimum ?""",
        "options": [
            {"id": "a", "texte": "J1 seulement.",
             "explication": r"Oui : J1 n'est pas séparable (une étiquette 0 en \(x = 3\) entre deux étiquettes 1), la log-loss est strictement convexe et atteint son minimum en un point unique ; J2, J3 et J4 sont séparables par la direction indiquée dans la correction."},
            {"id": "b", "texte": "J1 et J4.",
             "explication": r"J4 a toutes ses étiquettes égales à 1 : la direction de la constante \((1, 0)\) le sépare. Avec \(\mathbf{b} = (c, 0)\), chaque terme vaut \(\ln(1 + e^{-c})\), qui tend vers 0 sans l'atteindre : la constante part à \(+\infty\)."},
            {"id": "c", "texte": "J1, J3 et J4 : seul J2 pose problème, car il est parfaitement séparé au milieu.",
             "explication": r"J3 est aussi séparé, par le seuil \(x = 3{,}5\) : direction \((-3{,}5 ; 1)\), positive seulement en \(x = 4\). Une seule observation d'un côté suffit pour une séparation."},
            {"id": "d", "texte": "Les quatre : la log-loss est convexe, donc elle atteint toujours son minimum.",
             "explication": r"La convexité donne l'unicité éventuelle et l'absence de minimum local non global, pas l'existence. Sous séparation, la log-loss tend vers 0 sans l'atteindre, puisqu'elle est strictement positive en tout point."},
        ],
        "solution": r"""Réponse : J1 seulement.

Étape 1, critère (leçon, partie 1, énoncé 6) : s'il existe \(\mathbf{b}^\star\) avec \(b_0^\star + b_1^\star x_i > 0\) pour les étiquettes 1 et \(< 0\) pour les étiquettes 0, la log-loss tend vers 0 le long de \(c\,\mathbf{b}^\star\) quand \(c \to \infty\), et n'atteint jamais 0, car chaque terme \(\ln(1 + e^{\pm t_i})\) est strictement positif.

Étape 2, J2, J3, J4 sont séparés : J2 par \(\mathbf{b}^\star = (-2{,}5 ; 1)\) (seuil 2,5) ; J3 par \((-3{,}5 ; 1)\) (seuil 3,5) ; J4 par \((1 ; 0)\) (constante seule). Log-loss le long de \(c\,\mathbf{b}^\star\), pour \(c = 1\) puis \(c = 10\) : J2 \(@Q03_b_c1:4m@\) puis \(@Q03_b_c10:4m@\) ; J3 \(@Q03_c_c1:4m@\) puis \(@Q03_c_c10:4m@\) ; J4 \(@Q03_d_c1:4m@\) puis \(@Q03_d_c10:6m@\).

Étape 3, J1 : une fonction affine de \(x\) positive en \(x = 2\) et \(x = 4\) et négative en \(x = 1\) et \(x = 3\) changerait de signe trois fois, impossible : pas de séparation. Le minimiseur, calculé par la méthode de Newton (script), vaut \(\hat{\mathbf{b}} \approx @Q03_bhat:3m@\), de log-loss \(@Q03_loss_hat:4m@\), avec une hessienne définie positive en ce point. Remarque : ce sont les données de l'exemple de la leçon décalées de 2 en \(x\) ; la pente est la même, la constante absorbe le décalage.

Avec la pénalité L2 par défaut de scikit-learn, un minimiseur existe pour J1, J2 et J3 ; pour J4, une seule classe, scikit-learn refuse l'ajustement (`ValueError`, vérifié sous scikit-learn 1.1.1).""",
    },
    # ------------------------------------------------- Métriques
    {
        "id": "ML3-04", "section": "mlf", "theme": "Métriques sous déséquilibre : précision, exactitude", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""Un détecteur de fraude est évalué sur \(n = 10\,000\) transactions, dont 200 frauduleuses (positives). Au seuil choisi, il détecte 80 % des fraudes (taux de vrais positifs 0,8) et signale à tort 5 % des transactions honnêtes (taux de faux positifs 0,05). On compare à un classifieur constant qui déclare toutes les transactions honnêtes.

Quelle affirmation est vraie ?""",
        "options": [
            {"id": "a", "texte": r"La précision vaut environ @Q04_prec:3@, et l'exactitude du détecteur, @Q04_exact:3@, est inférieure à celle du classifieur constant, @Q04_const:2@.",
             "explication": r"Oui : VP = @Q04_vp@, FP = @Q04_fp@, précision \(@Q04_vp@/@Q04_pp@\) ; exactitude \(@Q04_vp_vn@/10\,000\) contre \(@Q04_neg@/10\,000\) pour le constant."},
            {"id": "b", "texte": r"La précision vaut @Q04_rap:1@, puisque le détecteur trouve 80 % des fraudes.",
             "explication": r"0,8 est le **rappel**, \(\mathrm{VP}/(\mathrm{VP} + \mathrm{FN})\). La précision divise par les prédictions positives, \(\mathrm{VP} + \mathrm{FP}\), dominées ici par les @Q04_fp@ fausses alertes."},
            {"id": "c", "texte": r"La précision vaut @Q04_d_spec:2m@, puisque seulement 5 % des transactions honnêtes sont signalées.",
             "explication": r"0,95 est la spécificité \(1 - \mathrm{TFP}\), la part des honnêtes reconnues comme telles. Un petit taux de faux positifs appliqué à @Q04_neg@ transactions honnêtes donne pourtant @Q04_fp@ fausses alertes, contre @Q04_vp@ vraies."},
            {"id": "d", "texte": r"La précision vaut environ @Q04_prec:3@, et l'exactitude du détecteur, @Q04_exact:3@, dépasse celle du classifieur constant, ce qui confirme son utilité.",
             "explication": r"Le classifieur constant a une exactitude de \(1 - 200/10\,000 = @Q04_const:2m@\), supérieure à \(@Q04_exact:3m@\). C'est précisément pourquoi l'exactitude est trompeuse sous déséquilibre : elle préfère le classifieur qui ne détecte rien."},
        ],
        "solution": r"""Réponse : précision \(\tfrac{@Q04_prec_num@}{@Q04_prec_den@} \approx @Q04_prec:3m@\), exactitude \(@Q04_exact:3m@ < @Q04_const:2m@\).

Étape 1, matrice de confusion : VP \(= 0{,}8 \times 200 = @Q04_vp@\), FN \(= @Q04_fn@\) ; FP \(= 0{,}05 \times @Q04_neg@ = @Q04_fp@\), VN \(= @Q04_vn@\).

Étape 2 : précision \(= @Q04_vp@/(@Q04_vp@ + @Q04_fp@) \approx @Q04_prec:3m@\) ; c'est aussi \(q\,\mathrm{TVP}/(q\,\mathrm{TVP} + (1 - q)\mathrm{TFP})\) avec \(q = 0{,}02\) (leçon, partie 2).

Étape 3 : exactitude \(= (@Q04_vp@ + @Q04_vn@)/10\,000 = @Q04_exact:3m@\) ; classifieur constant : \(@Q04_neg@/10\,000 = @Q04_const:2m@\).

Complément : F1 \(= 2\mathrm{VP}/(2\mathrm{VP} + \mathrm{FP} + \mathrm{FN}) \approx @Q04_f1:3m@\). Pour juger ce détecteur : courbe précision-rappel, ou coût des deux types d'erreur.""",
    },
    {
        "id": "ML3-05", "section": "mlf", "theme": "AUC comme proportion de paires, ex aequo", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Sept observations de test ont reçu les scores suivants :

- positifs (étiquette 1) : 0,9 ; 0,6 ; 0,4 ;
- négatifs (étiquette 0) : 0,7 ; 0,4 ; 0,3 ; 0,1.

L'AUC est l'aire sous la courbe ROC empirique, obtenue en faisant varier le seuil et en joignant les points successifs par des segments ; un positif et un négatif de même score franchissent le seuil ensemble (segment diagonal). C'est la convention de `roc_auc_score` de scikit-learn.

Que vaut l'AUC, arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q05_exact:3@", "explication": r"Oui : @Q05_bien@ paires bien ordonnées et @Q05_egal@ ex aequo compté \(\tfrac12\), sur @Q05_paires@ paires : \(@Q05_demi:1m@/@Q05_paires@\)."},
            {"id": "b", "texte": "@Q05_d_zero:3m@", "explication": r"C'est \(@Q05_bien@/@Q05_paires@\) : la paire ex aequo (0,4 contre 0,4) a été comptée 0. Le segment diagonal correspondant porte la moitié d'un rectangle : elle compte \(\tfrac12\)."},
            {"id": "c", "texte": "@Q05_d_un:3@", "explication": r"C'est \(@Q05_bien_egal@/@Q05_paires@\) : la paire ex aequo a été comptée comme bien ordonnée. Un ex aequo n'est ni bien ni mal ordonné : il compte \(\tfrac12\)."},
            {"id": "d", "texte": "@Q05_d_exactitude:3m@", "explication": r"C'est l'exactitude au seuil 0,5 (prédire 1 si le score dépasse 0,5) : @Q05_ex_vp@ positifs et @Q05_ex_vn@ négatifs bien classés sur 7. L'AUC ne dépend d'aucun seuil : elle résume tous les seuils à la fois."},
        ],
        "solution": r"""Réponse : \(\mathrm{AUC} = \tfrac{@Q05_exact_num@}{@Q05_exact_den@} \approx @Q05_exact:3m@\).

Étape 1 (leçon, partie 2) : l'AUC est la proportion de paires (positif, négatif) où le positif a le score le plus élevé, une paire ex aequo comptant \(\tfrac12\) (segment diagonal, dont l'aire est la moitié du rectangle).

Étape 2 : il y a \(3 \times 4 = @Q05_paires@\) paires. Le positif 0,9 bat @Q05_p1@ négatifs (tous) ; 0,6 bat 0,4, 0,3 et 0,1, soit @Q05_p2@ ; 0,4 bat 0,3 et 0,1, soit @Q05_p3@, et fait ex aequo avec le négatif 0,4. Bien ordonnées : \(@Q05_bien@\), ex aequo : \(@Q05_egal@\).

Étape 3 : \((@Q05_bien@ + \tfrac12)/@Q05_paires@ = \tfrac{@Q05_exact_num@}{@Q05_exact_den@}\).

Vérification (script) : l'aire calculée directement sous l'escalier ROC, trapèzes compris, donne la même fraction.""",
    },
    {
        "id": "ML3-06", "section": "mlf", "theme": "Score de Brier, calibration et AUC", "difficulte": 5, "temps_cible_s": 150,
        "enonce": r"""Une population se compose de deux groupes de même taille. Dans le groupe 1, chaque individu a la probabilité 0,1 d'être positif ; dans le groupe 2, la probabilité 0,5 ; les étiquettes sont indépendantes. Le modèle A prédit pour chaque individu la vraie probabilité de son groupe (0,1 ou 0,5). Le modèle B prédit 0,3 pour tout le monde.

On note, pour chaque modèle, son score de Brier espéré \(\mathbb{E}[(R - Y)^2]\) (\(R\) : prévision, \(Y\) : étiquette, individu tiré au hasard dans la population), son AUC en population, \(\mathbb{P}(S^+ > S^-) + \tfrac12 \mathbb{P}(S^+ = S^-)\), où \(S^+\) est la prévision d'un positif tiré au hasard et \(S^-\), indépendamment, celle d'un négatif tiré au hasard, et on dit qu'un modèle est calibré si \(\mathbb{P}(Y = 1 \mid R = r) = r\) pour chaque valeur \(r\) qu'il prédit.

Quelle affirmation est vraie ?""",
        "options": [
            {"id": "a", "texte": r"Les deux modèles sont calibrés ; Brier : A @Q06_brier_A:2@, B @Q06_brier_B:2@ ; AUC : A @Q06_auc_A:3@, B 0,5.",
             "explication": r"Oui : B est calibré, car parmi les individus à qui il prédit 0,3 (tous), la proportion de positifs est \((0{,}1 + 0{,}5)/2\). Mais il ne discrimine pas : AUC \(\tfrac12\), et un Brier plus élevé de la moyenne des \((0{,}3 - q)^2\)."},
            {"id": "b", "texte": r"B n'est pas calibré, puisqu'il se trompe sur chaque groupe ; Brier : A @Q06_brier_A:2m@, B @Q06_brier_B:2m@.",
             "explication": r"La calibration porte sur les valeurs prédites, pas sur les groupes : pour la seule valeur prédite 0,3, la fréquence de positifs est \(@Q06_calib_B:1m@\). B est calibré mais peu informatif ; c'est la discrimination (AUC) qui lui manque."},
            {"id": "c", "texte": r"Les deux modèles sont calibrés, donc ils ont le même score de Brier, @Q06_d_brier_moy:2m@.",
             "explication": r"Le score de Brier mesure la calibration **et** la discrimination : son espérance vaut la moyenne de \((r - q)^2 + q(1 - q)\), et B paie en plus la moyenne des \((0{,}3 - q)^2\). Être calibré ne suffit pas pour minimiser le Brier."},
            {"id": "d", "texte": r"A a une AUC de 1, puisqu'il prédit les vraies probabilités.",
             "explication": r"Connaître les vraies probabilités ne sépare pas les classes : dans le groupe 2, la moitié des individus sont négatifs et reçoivent la même prévision que les positifs. L'AUC de A vaut \(\tfrac{@Q06_auc_A_num@}{@Q06_auc_A_den@}\)."},
        ],
        "solution": r"""Réponse : les deux modèles sont calibrés ; Brier A \(@Q06_brier_A:2m@\), B \(@Q06_brier_B:2m@\) ; AUC A \(\tfrac{@Q06_auc_A_num@}{@Q06_auc_A_den@} \approx @Q06_auc_A:3m@\), B \(\tfrac12\).

Notations : \(q_1 = 0{,}1\), \(q_2 = 0{,}5\) les probabilités des groupes.

Étape 1, calibration : A prédit \(q_g\) dans le groupe \(g\), où la fréquence de positifs est \(q_g\). B ne prédit que 0,3, et la fréquence de positifs dans la population entière est \((q_1 + q_2)/2 = @Q06_calib_B:1m@\).

Étape 2, Brier (leçon, partie 2) : pour une prévision \(r\) d'une Bernoulli de paramètre \(q\), \(\mathbb{E}[(r - Y)^2] = (r - q)^2 + q(1 - q)\). A : \(\tfrac12 \big(q_1(1 - q_1) + q_2(1 - q_2)\big) = @Q06_brier_A:2m@\). B : \(@Q06_brier_A:2m@ + \tfrac12 \big((0{,}3 - q_1)^2 + (0{,}3 - q_2)^2\big) = @Q06_brier_B:2m@\).

Étape 3, AUC de A : par la formule de Bayes, un positif vient du groupe 2 avec probabilité \(\tfrac{q_2}{q_1 + q_2} = \tfrac{@Q06_pp2_num@}{@Q06_pp2_den@}\) (du groupe 1 avec probabilité \(\tfrac{@Q06_pp1_num@}{@Q06_pp1_den@}\)), un négatif du groupe 1 avec probabilité \(\tfrac{1 - q_1}{(1 - q_1) + (1 - q_2)} = \tfrac{@Q06_pn1_num@}{@Q06_pn1_den@}\) (du groupe 2 avec probabilité \(\tfrac{@Q06_pn2_num@}{@Q06_pn2_den@}\)). \(S^+ > S^-\) exactement quand le positif est du groupe 2 et le négatif du groupe 1 ; ex aequo quand ils sont du même groupe. AUC \(= \tfrac{@Q06_pp2_num@}{@Q06_pp2_den@} \cdot \tfrac{@Q06_pn1_num@}{@Q06_pn1_den@} + \tfrac12\big(\tfrac{@Q06_pp2_num@}{@Q06_pp2_den@} \cdot \tfrac{@Q06_pn2_num@}{@Q06_pn2_den@} + \tfrac{@Q06_pp1_num@}{@Q06_pp1_den@} \cdot \tfrac{@Q06_pn1_num@}{@Q06_pn1_den@}\big) = \tfrac{@Q06_auc_A_num@}{@Q06_auc_A_den@}\). AUC de B : toujours ex aequo, \(\tfrac12\).

Simulation (script, 400 000 individus) : Brier A \(@Q06_brier_A_mc:4m@\), B \(@Q06_brier_B_mc:4m@\), AUC de A \(@Q06_auc_A_mc:4m@\).""",
    },
    # ------------------------------------------------- Dimension
    {
        "id": "ML3-07", "section": "mlf", "theme": "Malédiction de la dimension : concentration des distances, Tchebychev", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Soient \(\mathbf{X}\) et \(\mathbf{X}'\) deux vecteurs aléatoires de \(\mathbb{R}^{50}\) dont les 100 coordonnées sont indépendantes et de loi normale \(\mathcal{N}(0, 1)\). On note \(D^2 = \lVert \mathbf{X} - \mathbf{X}' \rVert^2\) le carré de leur distance euclidienne.

Que valent le rapport \(\sqrt{\mathrm{Var}(D^2)}/\mathbb{E}[D^2]\), et la borne que l'inégalité de Tchebychev donne pour \(\mathbb{P}\big(\lvert D^2/\mathbb{E}[D^2] - 1 \rvert \ge 0{,}5\big)\) ?""",
        "options": [
            {"id": "a", "texte": r"Rapport @Q07_ratio:2@, borne @Q07_borne:2@.",
             "explication": r"Oui : \(X_j - X_j'\) suit \(\mathcal{N}(0, 2)\), donc \((X_j - X_j')^2\) a pour espérance @Q07_m2@ et pour variance \(@Q07_m4@ - @Q07_m2sq@ = @Q07_var1@\) ; rapport \(\sqrt{50 \times @Q07_var1@}/(50 \times @Q07_m2@)\), puis borne égale au carré du rapport divisé par \(0{,}5^2\)."},
            {"id": "b", "texte": r"Rapport @Q07_d_var2_ratio:2@, borne @Q07_d_var2_borne:2@.",
             "explication": r"La variance de \((X_j - X_j')^2\) a été prise égale à 2, celle d'un khi-deux à un degré de liberté, en oubliant que \(X_j - X_j'\) a pour variance 2 et non 1 : \((X_j - X_j')^2 = 2 Z^2\) avec \(Z\) normale centrée réduite, de variance \(2^2\,\mathrm{Var}(Z^2) = @Q07_var1@\)."},
            {"id": "c", "texte": r"Rapport @Q07_d_carre_ratio:2@, borne @Q07_d_carre_borne:4@.",
             "explication": r"@Q07_ratio2:2@ est \(\mathrm{Var}(D^2)/\mathbb{E}[D^2]^2\), le **carré** du rapport demandé ; la borne a ensuite été calculée avec ce carré, élevé de nouveau au carré."},
            {"id": "d", "texte": r"Rapport @Q07_ratio:2@, borne @Q07_d_lin_borne:1@.",
             "explication": r"C'est le rapport divisé par 0,5, sans carré : Tchebychev borne la probabilité par \(\mathrm{Var}/a^2\), donc par le carré du rapport divisé par \(\varepsilon^2\), soit \(@Q07_ratio2:2m@/0{,}25\)."},
        ],
        "solution": r"""Réponse : rapport \(@Q07_ratio:2m@\), borne \(\tfrac{@Q07_borne_num@}{@Q07_borne_den@} = @Q07_borne:2m@\).

Étape 1 : \(X_j - X_j'\), différence de deux normales indépendantes centrées réduites, suit \(\mathcal{N}(0, 2)\). Ses moments : \(\mathbb{E}[(X_j - X_j')^2] = @Q07_m2@\), \(\mathbb{E}[(X_j - X_j')^4] = 3 \times 2^2 = @Q07_m4@\) (moment d'ordre 4 d'une normale centrée de variance \(s^2\) : \(3s^4\) ; intégrales calculées par sympy dans le script). Donc \(v = \mathrm{Var}\big((X_j - X_j')^2\big) = @Q07_var1@\).

Étape 2 (leçon, partie 3) : \(D^2\) est une somme de 50 termes indépendants de même loi : \(\mathbb{E}[D^2] = 50 \times 2 = @Q07_esp@\), \(\mathrm{Var}(D^2) = 50 \times 8 = @Q07_var@\). Rapport \(\sqrt{@Q07_var@}/@Q07_esp@ = @Q07_ratio:2m@\), égal à \(\sqrt{2/p}\) avec \(p = 50\).

Étape 3, Tchebychev : \(\mathbb{P}(\lvert D^2 - \mathbb{E}D^2 \rvert \ge 0{,}5\,\mathbb{E}D^2) \le \mathrm{Var}(D^2)/(0{,}5\,\mathbb{E}D^2)^2 = @Q07_ratio2:2m@/0{,}25 = @Q07_borne:2m@\).

Simulation (script, 100 000 paires) : espérance \(@Q07_esp_mc:2m@\), variance \(@Q07_var_mc:1m@\), et fréquence observée de l'événement \(@Q07_freq:4m@\), bien sous la borne : Tchebychev est valable pour toute loi, donc rarement serrée.""",
    },
    # ------------------------------------------------- Validation temporelle
    {
        "id": "ML3-08", "section": "mlf", "theme": "Validation croisée temporelle : purge et embargo", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Données quotidiennes, dates numérotées de 0 à 1499 ; les prix sont connus assez longtemps avant et après pour que toutes les variables et étiquettes ci-dessous soient définies. On note \(R_s\) le rendement du jour \(s\).

- L'étiquette de la date \(t\) est \(R_{t+1} + \dots + R_{t+10}\) (horizon 10 jours).
- Les variables de la date \(t\) sont calculées à partir de \(R_{t-59}, \dots, R_t\) (60 jours).
- Validation croisée par 5 blocs de 300 dates contiguës ; on considère le pli dont le bloc de test est formé des dates 600 à 899.
- **Purge** : on retire de l'apprentissage toute date (hors bloc de test) dont l'étiquette contient au moins un rendement qui entre aussi dans l'étiquette d'au moins une date de test.
- **Embargo** : on retire en plus toute date postérieure au bloc de test, non purgée, dont les variables utilisent au moins un rendement qui entre dans l'étiquette d'au moins une date de test.

Combien de dates restent dans l'ensemble d'apprentissage de ce pli ?""",
        "options": [
            {"id": "a", "texte": "@Q08_exact@", "explication": r"Oui : @Q08_sans@ dates hors test, moins @Q08_n_pav@ purgées avant, @Q08_n_pap@ purgées après, et @Q08_n_emb@ sous embargo."},
            {"id": "b", "texte": "@Q08_d_purge_seule@", "explication": r"C'est la purge seule (@Q08_n_purge@ dates). Les dates @Q08_emb1@ à @Q08_emb2@ ont des variables qui contiennent des rendements \(R_{@Q08_r1@}, \dots, R_{@Q08_r2@}\) utilisés par les étiquettes de test : l'embargo les retire."},
            {"id": "c", "texte": "@Q08_d_purge_h@", "explication": r"La purge a retiré \(h\) dates de chaque côté au lieu de \(h - 1\). La date @Q08_avant_purge@ a pour étiquette \(R_{@Q08_pav1@} + \dots + R_{@Q08_t0@}\), qui ne contient aucun rendement des étiquettes de test (elles commencent à \(R_{@Q08_r1@}\)) : elle reste. Une étiquette à horizon \(h\) impose \(h - 1\) dates de purge de chaque côté."},
            {"id": "d", "texte": "@Q08_d_emb_deux@", "explication": r"L'embargo a été appliqué des deux côtés. Avant le bloc, les variables n'utilisent que des rendements antérieurs à la date, donc antérieurs à \(R_{@Q08_r1@}\) une fois la purge faite : rien à retirer. L'embargo ne concerne que l'après-bloc."},
        ],
        "solution": r"""Réponse : \(@Q08_exact@\) dates.

Étape 1, purge (leçon, partie 4) : les étiquettes de test utilisent les rendements \(R_{@Q08_r1@}, \dots, R_{@Q08_r2@}\). L'étiquette de \(t\), \(R_{t+1}, \dots, R_{t+10}\), en contient un si et seulement si \(t + 10 \ge @Q08_r1@\) et \(t + 1 \le @Q08_r2@\), soit \(@Q08_pav1@ \le t \le @Q08_pap2@\). Hors test : dates @Q08_pav1@ à @Q08_pav2@ et @Q08_pap1@ à @Q08_pap2@, soit \(@Q08_n_purge@\) dates.

Étape 2, embargo : les variables de \(t\) utilisent \(R_{t-59}, \dots, R_t\) ; elles contiennent un rendement de \(R_{@Q08_r1@}, \dots, R_{@Q08_r2@}\) si et seulement si \(t - 59 \le @Q08_r2@\) (pour \(t\) après le bloc), soit \(t \le @Q08_emb2@\). Après la purge : dates @Q08_emb1@ à @Q08_emb2@, soit \(@Q08_n_emb@\) dates, la longueur de la fenêtre des variables.

Étape 3 : \(1500 - 300 - @Q08_n_purge@ - @Q08_n_emb@ = @Q08_exact@\).

Vérification (script) : énumération directe des 1500 dates avec les deux règles, et contrôle des formules générales (purge de \(h - 1\) dates de chaque côté, embargo de \(L\) dates après) sur 200 configurations tirées au hasard.""",
    },
]
