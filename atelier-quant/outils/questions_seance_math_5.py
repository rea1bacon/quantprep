"""Séance de mathématiques 5 : leçon (forme d'une loi, corrélation, tests d'adéquation, densités conjointes,
intensité de Poisson a posteriori, bâton coupé, deux enfants), puis 12 questions à choix multiple.


séances 1 à 4 ne couvraient pas ; les questions sont nouvelles (autres données, autres angles).
Cinq questions sont à plusieurs réponses (champ "correctes") : notation tout ou rien, une explication par option.
Dans les questions à réponse unique, l'option "a" est la bonne.
Toutes les valeurs : verif/seance_math_5.py.
"""
import lecons_math_5
import questions_diagnostic as diag

ID = "seance-math-5"
TITRE = "Séance de mathématiques 5 : statistique descriptive, tests d'adéquation et densités jointes"
VERIF = "verif/seance_math_5.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [{"id": "lecon", "nom": "Leçon", "budget_min": 50}, {"id": "math", "nom": "Mathématiques", "budget_min": 24}]
PRESENTATION = (
    "D'abord une leçon d'environ 50 minutes en sept parties indépendantes (20 minutes en ne lisant que les énoncés, "
    "les exemples et les réflexes) : forme d'une loi (moyenne, médiane, mode, asymétrie, kurtosis), covariance, "
    "corrélation et cote Z, tests d'adéquation (khi-deux, Kolmogorov-Smirnov, Anderson-Darling), densités "
    "conjointes, intensité d'une loi de Poisson a priori et a posteriori, bâton coupé au hasard, problème des deux "
    "enfants. Cinq points de contrôle sont corrigés immédiatement, et les simulations affichent leur vraie sortie. "
    "Ensuite 12 questions à choix multiple, dont 5 à plusieurs réponses et 5 de difficulté 4 ou 5.\n\n"
    ""
    "plutôt que de longs calculs. Objectif de vitesse : chaque question se résout dans son temps cible avec le "
    "tableau « Réflexes en 30 secondes ». Au-delà du double du temps cible affiché, choisis « Je ne sais pas » et "
    "passe à la suivante : c'est la règle d'abandon à suivre dans un test chronométré."
)
AVERTISSEMENTS = [
    "Deux formats de question : à réponse unique, ou à plusieurs réponses. Dans le second cas, il faut cocher "
    "toutes les affirmations justes et seulement elles : la notation est en tout ou rien.",
    "Ici le retour en arrière est permis et il n'y a pas de pénalité : ce sont les règles de cette plateforme, pas "
    "celles d'un test en ligne en général.",
    "Les hypothèses de la régression linéaire ne sont pas reprises dans cette séance : elles sont traitées dans la "
    "« Séance de ML 1 : régression linéaire, Ridge et Lasso », partie 2.",
]

LECONS = [lecons_math_5.LECON_MATH_5]
CODES = []

QUESTIONS = [
    # ------------------------------------------------- Tests d'adéquation
    {
        "id": "M5-01", "section": "math", "theme": "Tests d'adéquation, quel test pour quelle loi", "difficulte": 3, "temps_cible_s": 90,
        "conceptuelle": True, "correctes": ["a", "c", "d"],
        "enonce": r"""On dispose de \(n\) observations indépendantes et de même loi, et l'on veut tester l'adéquation de cette loi à un modèle. Dans les affirmations ci-dessous, \(k\) désigne le nombre de classes d'un test du khi-deux, et « la table usuelle » du test de Kolmogorov-Smirnov est celle qui vaut pour une loi continue entièrement spécifiée à l'avance.

Quelles affirmations sont justes ?""",
        "options": [
            {"id": "a", "texte": r"Pour tester « les comptages suivent une loi de Poisson », de paramètre inconnu estimé sur les données, on peut utiliser le test du khi-deux, en comparant la statistique à une loi du khi-deux à \(k - 2\) degrés de liberté.",
             "explication": r"Juste : le khi-deux est le test naturel d'une loi discrète, et l'on retire un degré de liberté pour la contrainte sur le total et un autre pour le paramètre estimé, soit \(k - 1 - 1\)."},
            {"id": "b", "texte": "Pour cette même hypothèse (loi de Poisson de paramètre estimé), le test de Kolmogorov-Smirnov lu dans sa table usuelle a exactement le niveau affiché.",
             "explication": r"Faux, pour deux raisons : la loi de Poisson est discrète, et son paramètre est estimé sur l'échantillon. Dans les deux cas la statistique est plus petite que sous la loi de la table, et le niveau réel est inférieur au niveau affiché."},
            {"id": "c", "texte": "Si la loi testée est continue et entièrement spécifiée à l'avance, la loi de la statistique de Kolmogorov-Smirnov sous l'hypothèse nulle est la même quelle que soit cette loi.",
             "explication": r"Juste : en écrivant \(X_i = Q(U_i)\) avec des \(U_i\) uniformes, la statistique devient \(\sup_u \lvert G_n(u) - u \rvert\), qui ne dépend que des \(U_i\) (leçon, partie 3)."},
            {"id": "d", "texte": "La statistique d'Anderson-Darling donne plus de poids que celle de Kolmogorov-Smirnov aux écarts situés dans les queues de la loi.",
             "explication": r"Juste : elle divise chaque écart au carré par sa variance \(u(1-u)/n\), petite près de 0 et de 1, alors que Kolmogorov-Smirnov prend l'écart brut maximal, dominé par le centre."},
        ],
        "solution": r"""Réponse : les affirmations sur le khi-deux à \(k - 2\) degrés de liberté, sur la loi libre de la statistique de Kolmogorov-Smirnov et sur le poids des queues dans Anderson-Darling sont justes ; celle sur le test de Kolmogorov-Smirnov appliqué à une loi de Poisson de paramètre estimé est fausse.

Le classement à retenir (leçon, partie 3, tableau « Quel test pour quelle loi ») :

- loi discrète (binomiale, Poisson) : khi-deux, avec \(k - 1 - s\) degrés de liberté quand \(s\) paramètres sont estimés, et des effectifs attendus d'au moins 5 (règle d'usage) ;
- loi continue entièrement spécifiée : Kolmogorov-Smirnov, dont la loi sous l'hypothèse nulle ne dépend pas de la loi testée, ou Anderson-Darling, plus sensible aux queues ;
- paramètres estimés sur l'échantillon : la table de Kolmogorov ne s'applique plus ; il faut des seuils propres (Lilliefors pour la loi normale ou exponentielle) ou un test conçu pour ce cas (Shapiro-Wilk pour la normalité).

Pourquoi le test de Kolmogorov-Smirnov échoue pour une loi discrète : la statistique s'écrit \(\sup_{u \in \mathcal{A}} \lvert G_n(u) - u \rvert\), où \(\mathcal{A}\) est l'ensemble des valeurs de la fonction de répartition testée ; pour une loi discrète, \(\mathcal{A}\) n'est qu'une partie de \([0, 1]\) et le sup est plus petit.""",
    },
    {
        "id": "M5-02", "section": "math", "theme": "Khi-deux d'adéquation, statistique et décision", "difficulte": 2, "temps_cible_s": 90,
        "enonce": r"""Un générateur est censé tirer, à chaque tirage et indépendamment des autres, l'un des symboles A, B, C, D avec la même probabilité 1/4. Sur 200 tirages, on observe 38 fois A, 62 fois B, 55 fois C et 45 fois D.

On teste l'hypothèse nulle « les quatre symboles sont équiprobables » par le test du khi-deux d'adéquation, au niveau 5 %. On donne les quantiles d'ordre 0,95 de la loi du khi-deux : @Q02_q2:2@ pour 2 degrés de liberté, @Q02_q3:2@ pour 3, @Q02_q4:2@ pour 4.

Que vaut la statistique du test, et quelle est la décision ?""",
        "options": [
            {"id": "a", "texte": "@Q02_exact:2@ ; on ne rejette pas l'hypothèse nulle.", "explication": r"Oui : \((144 + 144 + 25 + 25)/50 = @Q02_exact:2m@\), sous le quantile \(@Q02_q3:2m@\) de la loi à \(4 - 1 = 3\) degrés de liberté."},
            {"id": "b", "texte": "@Q02_exact:2@ ; on rejette l'hypothèse nulle.", "explication": r"La statistique est juste, mais elle a été comparée au quantile à 2 degrés de liberté (\(@Q02_q2:2m@\)). Aucun paramètre n'est estimé ici : il y a \(k - 1 = 3\) degrés de liberté, et \(@Q02_exact:2m@ < @Q02_q3:2m@\)."},
            {"id": "c", "texte": "@Q02_d_sur_n:2@ ; on ne rejette pas l'hypothèse nulle.", "explication": r"Les carrés des écarts ont été divisés par le nombre de tirages, 200, au lieu de l'effectif attendu de chaque classe, 50. Chaque terme est \((O_j - e_j)^2/e_j\)."},
            {"id": "d", "texte": "@Q02_d_brut@ ; on rejette l'hypothèse nulle.", "explication": r"C'est la somme des carrés des écarts, sans la division par les effectifs attendus. Sans cette division, la statistique dépendrait de la taille de l'échantillon et n'aurait pas la loi du khi-deux."},
        ],
        "solution": r"""Réponse : la statistique vaut \(@Q02_exact:2m@\) et l'on ne rejette pas l'hypothèse nulle au niveau 5 %.

Notations : \(O_j\) est l'effectif observé du symbole \(j\), \(e_j = 200 \times 1/4 = 50\) l'effectif attendu sous l'hypothèse nulle, \(C = \sum_j (O_j - e_j)^2/e_j\) la statistique.

Étape 1, écarts : \(38 - 50 = -12\), \(62 - 50 = 12\), \(55 - 50 = 5\), \(45 - 50 = -5\). Leur somme est nulle, comme toujours.

Étape 2, statistique : \(C = (144 + 144 + 25 + 25)/50 = @Q02_somme_carres@/50 = @Q02_exact:2m@\).

Étape 3, degrés de liberté : \(k = 4\) classes, aucun paramètre estimé, donc \(k - 1 = @Q02_ddl@\) (leçon, partie 3). Le quantile d'ordre 0,95 est \(@Q02_q3:2m@\), et \(@Q02_exact:2m@ < @Q02_q3:2m@\) : pas de rejet. Tous les effectifs attendus valent 50, bien au-dessus de 5 : l'approximation par la loi du khi-deux est acceptable.

La p-value approchée vaut \(@Q02_p:3m@\) ; la simulation de la loi exacte de la statistique sous l'hypothèse nulle (script) donne @Q02_p_sim:3@, et une moyenne de la statistique de @Q02_esp_mc:3@, pour une valeur exacte \(k - 1 = 3\).""",
    },
    {
        "id": "M5-03", "section": "math", "theme": "Khi-deux d'adéquation à une loi de Poisson, paramètre estimé", "difficulte": 4, "temps_cible_s": 150,
        "enonce": r"""On compte les ordres reçus pendant 200 intervalles d'une minute : 300 ordres au total. On veut tester l'hypothèse « le nombre d'ordres par intervalle suit une loi de Poisson », de paramètre inconnu, par un test du khi-deux d'adéquation sur les cinq classes « 0 », « 1 », « 2 », « 3 » et « 4 ou plus ». Le paramètre de la loi de Poisson est estimé par le nombre moyen d'ordres par intervalle. On donne \(e^{-1{,}5} \approx @Q03_exp:4m@\).

Quel est l'effectif attendu de la classe « 4 ou plus », arrondi à une décimale, et à combien de degrés de liberté compare-t-on la statistique ?""",
        "options": [
            {"id": "a", "texte": "@Q03_exact:1@ intervalles ; @Q03_ddl@ degrés de liberté", "explication": r"Oui : \(200 \times (1 - @Q03_p_bas:4m@) = @Q03_exact:1m@\), et \(5 - 1 - 1 = @Q03_ddl@\) degrés de liberté, le dernier retrait venant du paramètre estimé."},
            {"id": "b", "texte": "@Q03_d_egal4:1@ intervalles ; @Q03_ddl@ degrés de liberté", "explication": r"C'est \(200\,\mathbb{P}(N = 4)\) : la classe est « 4 ou plus », elle doit recevoir tout le complément \(1 - \mathbb{P}(N \le 3)\). Sinon les effectifs attendus n'ont pas pour somme 200."},
            {"id": "c", "texte": "@Q03_exact:1@ intervalles ; @Q03_d_ddl_sans_estim@ degrés de liberté", "explication": r"\(k - 1 = 4\) vaut quand la loi est entièrement spécifiée. Ici le paramètre est estimé sur les données, ce qui retire un degré de liberté de plus."},
            {"id": "d", "texte": "@Q03_exact:1@ intervalles ; @Q03_d_ddl_classes@ degrés de liberté", "explication": r"Le nombre de classes n'est pas le nombre de degrés de liberté : les cinq écarts ont une somme nulle (un degré en moins), et un paramètre est estimé (un autre en moins)."},
        ],
        "solution": r"""Réponse : \(@Q03_exact:1m@\) intervalles attendus, et @Q03_ddl@ degrés de liberté.

Notations : \(N\) est le nombre d'ordres d'un intervalle, \(\hat{\lambda}\) le paramètre estimé.

Étape 1, estimation : \(\hat{\lambda} = 300/200 = @Q03_lambda:1m@\) ordre par intervalle.

Étape 2, probabilité de la dernière classe : \(\mathbb{P}(N \le 3) = e^{-1{,}5}\big(1 + 1{,}5 + \tfrac{1{,}5^2}{2} + \tfrac{1{,}5^3}{6}\big) = @Q03_exp:4m@ \times @Q03_somme:4m@ = @Q03_p_bas:4m@\), donc \(\mathbb{P}(N \ge 4) = @Q03_p_queue:4m@\) (simulation : @Q03_p_queue_mc:4@).

Étape 3, effectif attendu : \(200 \times @Q03_p_queue:4m@ = @Q03_exact:1m@\). Les quatre autres valent \(@Q03_att:1m@\) ; tous dépassent 5, la règle d'usage est respectée.

Étape 4, degrés de liberté : \(k - 1 - s\) avec \(k = 5\) classes et \(s = @Q03_n_param@\) paramètre estimé, soit @Q03_ddl@ (leçon, partie 3).

Remarque : regrouper les valeurs rares dans « 4 ou plus » est ce qui permet de respecter la règle sur les effectifs attendus ; sans regroupement, la classe « 6 », par exemple, aurait un effectif attendu inférieur à 1.""",
    },
    # ------------------------------------------------- Covariance, corrélation, cote Z
    {
        "id": "M5-04", "section": "math", "theme": "Covariance et corrélation, changement d'unités", "difficulte": 3, "temps_cible_s": 120,
        "correctes": ["a", "b", "d"],
        "enonce": r"""Pour un jour tiré au hasard dans l'année, \(X\) est la température à midi, en degrés Celsius, et \(Y\) le montant des ventes d'un glacier, en euros. L'écart-type de \(X\) vaut 5 °C, celui de \(Y\) vaut 40 euros, et leur covariance vaut 120 (en °C fois euro). On note \(X' = 1{,}8\,X + 32\) la même température en degrés Fahrenheit.

Quelles affirmations sont justes ?""",
        "options": [
            {"id": "a", "texte": r"La corrélation entre \(X\) et \(Y\) vaut @Q04_rho:1@, et la corrélation entre \(X'\) et \(Y\) aussi.",
             "explication": r"Juste : \(120/(5 \times 40) = @Q04_rho:1m@\), et un changement affine de pente positive ne change pas la corrélation, qui n'a pas d'unité."},
            {"id": "b", "texte": r"La covariance entre \(X'\) et \(Y\) vaut @Q04_cov_f@ (en °F fois euro).",
             "explication": r"Juste : \(\mathrm{Cov}(1{,}8\,X + 32, Y) = 1{,}8 \times 120 = @Q04_cov_f@\). La constante 32 disparaît avec le centrage ; l'ajouter donnerait @Q04_d_cov_affine@, ce qui est faux."},
            {"id": "c", "texte": r"La droite de régression de \(Y\) sur \(X\) a pour pente @Q04_rho:1@ euro par degré Celsius.",
             "explication": r"Faux : la pente est \(\mathrm{Cov}(X, Y)/\mathrm{Var}(X) = 120/25 = @Q04_pente:1m@\) euros par degré Celsius. La corrélation est sans unité et ne se lit pas comme une pente."},
            {"id": "d", "texte": r"Si l'on remplace \(Y\) par \(-Y\), la covariance et la corrélation avec \(X\) changent de signe sans changer de valeur absolue.",
             "explication": r"Juste : \(\mathrm{Cov}(X, -Y) = -\mathrm{Cov}(X, Y)\) et l'écart-type de \(-Y\) est celui de \(Y\), donc la corrélation devient \(-@Q04_rho:1m@\)."},
        ],
        "solution": r"""Réponse : sont justes les affirmations sur la corrélation inchangée (\(@Q04_rho:1m@\)), sur la covariance \(@Q04_cov_f@\) en degrés Fahrenheit et sur le changement de signe ; l'affirmation sur la pente est fausse.

Règle (leçon, partie 2, (K1)) : \(\mathrm{Cov}(aX + b, cY + d) = ac\,\mathrm{Cov}(X, Y)\), et la corrélation est multipliée par le signe de \(ac\).

Étape 1, corrélation : \(\rho = 120/(5 \times 40) = @Q04_rho:1m@\).

Étape 2, degrés Fahrenheit (\(a = 1{,}8\), \(b = 32\)) : covariance \(1{,}8 \times 120 = @Q04_cov_f@\), écart-type de \(X'\) égal à \(1{,}8 \times 5 = @Q04_sx_f@\) °F, corrélation \(@Q04_cov_f@/(@Q04_sx_f@ \times 40) = @Q04_rho:1m@\) (simulation de la covariance : @Q04_cov_f_mc:1@).

Étape 3, pente : \(\mathrm{Cov}(X, Y)/\mathrm{Var}(X) = 120/25 = @Q04_pente:1m@\) euros par degré Celsius, soit aussi \(\rho\,\sigma_Y/\sigma_X = 0{,}6 \times 40/5\). En degrés Fahrenheit, elle devient \(@Q04_pente:1m@/1{,}8 \approx @Q04_pente_f:2m@\) euros par degré : la pente dépend des unités, la corrélation non.

Étape 4, signe : \(a = 1\), \(c = -1\), donc covariance \(-120\) et corrélation \(-@Q04_rho:1m@\).""",
    },
    {
        "id": "M5-05", "section": "math", "theme": "Cote Z, comparaison de notes", "difficulte": 2, "temps_cible_s": 60,
        "enonce": r"""Trois candidats passent trois examens différents. Pour chaque examen, on donne la note du candidat, puis la moyenne et l'écart-type des notes de tous les candidats de cet examen.

- Examen A : note 78 sur 100 ; moyenne 66, écart-type 8.
- Examen B : note 1340 sur 1600 ; moyenne 1100, écart-type 200.
- Examen C : note 15 sur 20 ; moyenne 11,4, écart-type 2.

Quelle note est la meilleure en position relative, mesurée par la cote Z, et que vaut cette cote ?""",
        "options": [
            {"id": "a", "texte": "Celle de l'examen C, de cote Z @Q05_zc:1@.", "explication": r"Oui : \((15 - 11{,}4)/2 = @Q05_zc:1m@\), contre \(@Q05_za:1m@\) pour A et \(@Q05_zb:1m@\) pour B."},
            {"id": "b", "texte": "Celle de l'examen B, la plus éloignée de sa moyenne : @Q05_d_ecart_b@ points.", "explication": r"Un écart en points dépend de l'échelle de l'examen. Rapporté à l'écart-type de 200 points, il ne vaut que \(@Q05_zb:1m@\) écart-type, la plus petite des trois cotes."},
            {"id": "c", "texte": "Celle de l'examen B, qui atteint @Q05_d_pct_b:2@ % du barème.", "explication": r"Le pourcentage du barème ne dit rien de la position parmi les candidats : il ignore la moyenne et la dispersion des notes de chaque examen."},
            {"id": "d", "texte": "Celle de l'examen C, de cote Z @Q05_d_ecart_c:1@.", "explication": r"L'examen est le bon, mais \(@Q05_d_ecart_c:1m@\) est l'écart à la moyenne en points, \(15 - 11{,}4\) : il reste à le diviser par l'écart-type 2."},
        ],
        "solution": r"""Réponse : la note de l'examen C, de cote Z \(@Q05_zc:1m@\).

La cote Z d'une note \(x\) est \(z = (x - \mu)/\sigma\), où \(\mu\) et \(\sigma\) sont la moyenne et l'écart-type des notes de l'examen : le nombre d'écarts-types entre la note et la moyenne (leçon, partie 2). Elle n'a pas d'unité, ce qui permet de comparer des barèmes différents.

- Examen A : \((78 - 66)/8 = @Q05_za:1m@\).
- Examen B : \((1340 - 1100)/200 = @Q05_zb:1m@\).
- Examen C : \((15 - 11{,}4)/2 = @Q05_zc:1m@\).

Lecture en rang, sous l'hypothèse supplémentaire que les notes de chaque examen suivent une loi normale : la proportion de candidats dépassés serait \(@Q05_phi_a:3m@\) pour A, \(@Q05_phi_b:3m@\) pour B et \(@Q05_phi_c:3m@\) pour C (simulation pour C : @Q05_phi_c_mc:3@). Sans hypothèse sur la forme des lois, les cotes comparent des positions en écarts-types, pas des rangs.""",
    },
    # ------------------------------------------------- Forme d'une loi
    {
        "id": "M5-06", "section": "math", "theme": "Asymétrie : moyenne, médiane et mode d'une loi log-normale", "difficulte": 3, "temps_cible_s": 90,
        "correctes": ["a", "b"],
        "enonce": r"""La variable aléatoire \(X\) est strictement positive, et \(\ln X\) suit la loi normale d'espérance 0 et de variance 1 : \(X\) suit une loi log-normale.

Quelles affirmations sont justes ?""",
        "options": [
            {"id": "a", "texte": r"La médiane de \(X\) vaut @Q06_med@.",
             "explication": r"Juste : l'exponentielle est croissante, donc \(\mathbb{P}(X \le 1) = \mathbb{P}(\ln X \le 0) = 1/2\). Une fonction croissante transporte la médiane."},
            {"id": "b", "texte": r"L'espérance de \(X\) vaut \(e^{1/2} \approx @Q06_moy:3m@\).",
             "explication": r"Juste : \(\mathbb{E}[e^{Y}] = e^{1/2}\) pour \(Y\) de loi \(\mathcal{N}(0, 1)\). L'espérance, elle, n'est pas transportée par l'exponentielle : elle dépasse \(e^{\mathbb{E}[\ln X]} = 1\) (inégalité de Jensen)."},
            {"id": "c", "texte": r"Le mode de \(X\) vaut \(e^{-1/2} \approx @Q06_d_mode:3m@\).",
             "explication": r"Faux : le mode vaut \(e^{-1} \approx @Q06_mode:3m@\). L'exposant est \(a - b^2\) avec \(a = 0\) et \(b^2 = 1\) ; \(1/2\) est le terme \(b^2/2\) de la moyenne, pas celui du mode."},
            {"id": "d", "texte": "Pour toute loi de variance finie dont le coefficient d'asymétrie est strictement positif, la moyenne est supérieure à la médiane.",
             "explication": r"Faux : c'est une règle empirique, vraie ici, mais pas un théorème. La loi qui vaut \(-2\), 1, 4 avec probabilités 0,4, 0,5, 0,1 a une asymétrie positive (\(@La_B_skew:3m@\)), une moyenne de \(@La_B_moy:1m@\) et une médiane de @La_B_med@ (leçon, partie 1)."},
        ],
        "solution": r"""Réponse : seules les affirmations sur la médiane (@Q06_med@) et sur l'espérance (\(e^{1/2} \approx @Q06_moy:3m@\)) sont justes.

Notations : \(X = e^{Y}\) avec \(Y\) de loi \(\mathcal{N}(0, 1)\) ; dans les formules de la leçon (partie 1), \(a = 0\) et \(b = 1\).

Étape 1, médiane : \(\mathbb{P}(X \le e^0) = \mathbb{P}(Y \le 0) = 1/2\), donc la médiane vaut @Q06_med@ (simulation de \(\mathbb{P}(X \le 1)\) : @Q06_med_p_mc:3@).

Étape 2, moyenne : \(\mathbb{E}[e^{Y}] = e^{1/2} \approx @Q06_moy:3m@\) (fonction génératrice de la loi normale ; simulation : @Q06_moy_mc:3@).

Étape 3, mode : la densité de \(X\) en \(x\) est \(\varphi(\ln x)/x\) ; avec \(t = \ln x\), son logarithme vaut \(-t^2/2 - t\) à une constante près, maximal en \(t = -1\). Le mode est \(e^{-1} \approx @Q06_mode:3m@\).

Étape 4 : mode < médiane < moyenne (\(@Q06_mode:3m@ < 1 < @Q06_moy:3m@\)), avec un coefficient d'asymétrie \((e + 2)\sqrt{e - 1} \approx @Q06_skew:2m@\). La loi log-normale suit la règle empirique, mais la règle n'est pas générale : ce qui vaut pour toute loi de variance finie, c'est que l'écart entre moyenne et médiane ne dépasse pas l'écart-type (ici \(@Q06_sigma:3m@\)).""",
    },
    # ------------------------------------------------- Intensité de Poisson a posteriori
    {
        "id": "M5-07", "section": "math", "theme": "Processus de Poisson, intensité la plus probable a posteriori dans une liste", "difficulte": 4, "temps_cible_s": 150,
        "enonce": r"""Des appels arrivent selon un processus de Poisson dont l'intensité \(\Lambda\), en appels par heure, est inconnue. On sait que \(\Lambda\) vaut 0,5, 1, 2 ou 4, et la loi a priori donne à chaque valeur \(\lambda\) de cette liste un poids proportionnel à \(1/\lambda\). On observe exactement 3 appels pendant 2 heures.

On donne \(e^{-1} \approx @Q07_e1:4m@\), \(e^{-2} \approx @Q07_e2:4m@\), \(e^{-4} \approx @Q07_e4:4m@\), \(e^{-8} \approx @Q07_e8:5m@\).

Quelle est la valeur de \(\Lambda\) la plus probable a posteriori, et quelle est sa probabilité a posteriori ?""",
        "options": [
            {"id": "a", "texte": "1 appel par heure, avec la probabilité @Q07_exact:3@", "explication": r"Oui : les poids a posteriori sont proportionnels à \(\lambda^{2} e^{-2\lambda}\), soit \(@Q07_noyau:4m@\), et \(@Q07_e2:4m@/@Q07_somme_noyau:4m@ = @Q07_exact:3m@\)."},
            {"id": "b", "texte": "2 appels par heure, avec la probabilité @Q07_d_unif_2:3@", "explication": r"C'est le résultat pour un a priori uniforme sur la liste, c'est-à-dire la vraisemblance \(\lambda^3 e^{-2\lambda}\) seule. L'a priori en \(1/\lambda\) retire une puissance de \(\lambda\) et favorise les petites intensités."},
            {"id": "c", "texte": "1,5 appel par heure, le nombre d'appels divisé par la durée", "explication": r"\(3/2\) est le maximum de vraisemblance sur tous les réels positifs. Il n'appartient pas à la liste des valeurs possibles de \(\Lambda\), et il ignore la loi a priori."},
            {"id": "d", "texte": "0,5 appel par heure, avec la probabilité @Q07_d_prior:3@", "explication": r"C'est la valeur la plus probable **a priori**, avec son poids a priori \(2/3{,}75\). Les 3 appels observés en 2 heures rendent cette petite intensité moins plausible."},
        ],
        "solution": r"""Réponse : 1 appel par heure, de probabilité a posteriori \(@Q07_exact:3m@\).

Notations : \(N\) est le nombre d'appels en \(t = 2\) heures ; sachant \(\Lambda = \lambda\), \(N\) suit la loi de Poisson de paramètre \(2\lambda\).

Étape 1, formule (leçon, partie 5, (B1)) : le poids a posteriori de \(\lambda\) est proportionnel à l'a priori fois la vraisemblance, \(\tfrac{1}{\lambda} \times \lambda^3 e^{-2\lambda} = \lambda^2 e^{-2\lambda}\) ; les facteurs communs \(2^3/3!\) et la constante de l'a priori se simplifient.

Étape 2, calcul pour \(\lambda = 0{,}5\), 1, 2, 4 : \(0{,}25\,e^{-1}\), \(e^{-2}\), \(4\,e^{-4}\), \(16\,e^{-8}\), soit \(@Q07_noyau:4m@\), de somme \(@Q07_somme_noyau:4m@\).

Étape 3, normalisation : probabilités a posteriori \(@Q07_post:3m@\). La plus grande est celle de \(\lambda = 1\) : \(@Q07_exact:3m@\) (simulation : @Q07_mc:3@).

Comparaison : avec un a priori uniforme sur la liste, les poids seraient \(\lambda^3 e^{-2\lambda}\), les probabilités a posteriori \(@Q07_post_unif:3m@\), et la valeur la plus probable serait 2. Le maximum de vraisemblance sans contrainte vaut \(@Q07_emv:1m@\), entre les deux : l'a priori décide.""",
    },
    # ------------------------------------------------- Bâton coupé
    {
        "id": "M5-08", "section": "math", "theme": "Bâton coupé en quatre, deuxième plus petit morceau", "difficulte": 5, "temps_cible_s": 150,
        "enonce": r"""Un bâton de longueur 1 est coupé en quatre morceaux par trois points tirés indépendamment et uniformément sur le bâton. On range les quatre longueurs par ordre croissant.

Quelle est l'espérance de la deuxième plus petite longueur ?""",
        "options": [
            {"id": "a", "texte": r"\(\tfrac{@Q08_exact_num@}{@Q08_exact_den@}\)", "explication": r"Oui : \(\tfrac14\big(\tfrac14 + \tfrac13\big) = \tfrac{@Q08_exact_num@}{@Q08_exact_den@} \approx @Q08_exact:4m@\)."},
            {"id": "b", "texte": r"\(\tfrac{@Q08_d_min_num@}{@Q08_d_min_den@}\)", "explication": r"C'est \(1/n^2\) avec \(n = 4\), l'espérance du **plus petit** morceau. Le deuxième ajoute le terme \(\tfrac14 \times \tfrac13\)."},
            {"id": "c", "texte": r"\(\tfrac{@Q08_d_double_num@}{@Q08_d_double_den@}\)", "explication": r"C'est deux fois l'espérance du plus petit morceau. Les écarts entre longueurs triées successives ne sont pas égaux : le deuxième écart vaut \(\tfrac14 \times \tfrac13\), plus grand que le premier, \(\tfrac14 \times \tfrac14\)."},
            {"id": "d", "texte": r"\(\tfrac{@Q08_d_rang_num@}{@Q08_d_rang_den@}\)", "explication": r"C'est \(2/(1 + 2 + 3 + 4)\) : des longueurs triées supposées proportionnelles à leur rang. La formule \(k/(n+1)\) vaut pour les **positions** de points uniformes triés, pas pour les longueurs triées des morceaux."},
        ],
        "solution": r"""Réponse : \(\tfrac14\big(\tfrac14 + \tfrac13\big) = \tfrac{@Q08_exact_num@}{@Q08_exact_den@} \approx @Q08_exact:4m@\).

Notations : \(n = 4\) morceaux, de longueurs triées \(L_{(1)} \le L_{(2)} \le L_{(3)} \le L_{(4)}\).

Étape 1, représentation (leçon, partie 6, (E5)) : les quatre morceaux ont la loi de \((E_1/T, \dots, E_4/T)\), où les \(E_i\) sont des exponentielles indépendantes de paramètre 1 et \(T\) leur somme, et ce vecteur est indépendant de \(T\).

Étape 2, exponentielles triées : la plus petite est le minimum de 4 exponentielles, d'espérance \(1/4\) ; par absence de mémoire, l'attente supplémentaire jusqu'à la deuxième est le minimum de 3 exponentielles, d'espérance \(1/3\). Donc \(\mathbb{E}[E_{(2)}] = \tfrac14 + \tfrac13\).

Étape 3, retour au bâton : \(E_{(2)} = (E_{(2)}/T) \times T\), produit de deux variables indépendantes, avec \(\mathbb{E}[T] = 4\). Donc \(\mathbb{E}[L_{(2)}] = \tfrac14\big(\tfrac14 + \tfrac13\big) = \tfrac{@Q08_exact_num@}{@Q08_exact_den@}\).

Contrôle : les quatre espérances \(\tfrac{@Q08_d_min_num@}{@Q08_d_min_den@}\), \(\tfrac{@Q08_exact_num@}{@Q08_exact_den@}\), \(\tfrac{@Q08_3e_num@}{@Q08_3e_den@}\), \(\tfrac{@Q08_max_num@}{@Q08_max_den@}\) ont pour somme 1. Simulation du bâton (script) : @Q08_mc:4@.""",
    },
    {
        "id": "M5-09", "section": "math", "theme": "Bâton coupé en trois, loi des morceaux", "difficulte": 4, "temps_cible_s": 150,
        "correctes": ["a", "b", "d"],
        "enonce": r"""Un bâton de longueur 1 est coupé en trois morceaux par deux points tirés indépendamment et uniformément sur le bâton. Le « morceau de gauche » va de l'extrémité gauche au premier point de coupe, le « morceau du milieu » est compris entre les deux points de coupe.

Quelles affirmations sont justes ?""",
        "options": [
            {"id": "a", "texte": r"L'espérance de la longueur du plus grand des trois morceaux vaut \(\tfrac{@Q09_max_num@}{@Q09_max_den@}\).",
             "explication": r"Juste : \(\tfrac13\big(\tfrac13 + \tfrac12 + 1\big) = \tfrac{@Q09_max_num@}{@Q09_max_den@}\), par la formule du \(k\)-ième plus petit morceau avec \(k = n = 3\)."},
            {"id": "b", "texte": r"La probabilité que le plus petit morceau mesure plus de 0,1 vaut @Q09_p_min:2@.",
             "explication": r"Juste : \(\mathbb{P}(L_{(1)} > x) = (1 - 3x)^2\), soit \(0{,}7^2 = @Q09_p_min:2m@\) pour \(x = 0{,}1\) : on réserve 0,1 à chaque morceau, il reste un bâton de longueur 0,7."},
            {"id": "c", "texte": r"La probabilité que les trois morceaux puissent former un triangle vaut \(\tfrac{@Q09_d_triangle_num@}{@Q09_d_triangle_den@}\).",
             "explication": r"Faux : elle vaut \(\tfrac{@Q09_triangle_num@}{@Q09_triangle_den@}\). Il faut que le plus grand morceau soit sous \(1/2\), et \(\mathbb{P}(L_{(3)} \ge 1/2) = 3 \times (1/2)^2 = 3/4\). La valeur \(1/2\) vient d'un argument de symétrie sans fondement : rien ne dit que le plus grand morceau est au-dessus ou au-dessous de \(1/2\) avec la même probabilité."},
            {"id": "d", "texte": "Le morceau de gauche et le morceau du milieu ont la même loi.",
             "explication": r"Juste : chaque morceau vérifie \(\mathbb{P}(L_j > x) = (1 - x)^2\), quelle que soit sa place. En revanche ils ne sont pas indépendants : leur corrélation vaut \(@Q09_corr:1m@\)."},
        ],
        "solution": r"""Réponse : sont justes l'espérance \(\tfrac{@Q09_max_num@}{@Q09_max_den@}\) du plus grand morceau, la probabilité \(@Q09_p_min:2m@\) et l'égalité des lois du morceau de gauche et du morceau du milieu ; la probabilité de former un triangle vaut \(\tfrac{@Q09_triangle_num@}{@Q09_triangle_den@}\), pas \(\tfrac{@Q09_d_triangle_num@}{@Q09_d_triangle_den@}\).

Tout vient de la loi conjointe (leçon, partie 6, (E1)) : pour des réels positifs \(c_1, c_2, c_3\) de somme \(c \le 1\), \(\mathbb{P}(L_1 > c_1, L_2 > c_2, L_3 > c_3) = (1 - c)^2\).

Étape 1, même loi : avec un seul \(c_j = x\) non nul, \(\mathbb{P}(L_j > x) = (1 - x)^2\), pour le morceau de gauche comme pour celui du milieu. Par exemple \(\mathbb{P}(L_j > 1/3) = \tfrac{@Q09_p_tiers_num@}{@Q09_p_tiers_den@}\) (simulations : @Q09_gauche_mc:3@ à gauche, @Q09_milieu_mc:3@ au milieu).

Étape 2, plus petit morceau : avec \(c_1 = c_2 = c_3 = 0{,}1\), \(\mathbb{P}(L_{(1)} > 0{,}1) = (1 - 0{,}3)^2 = @Q09_p_min:2m@\) (simulation : @Q09_p_min_mc:3@).

Étape 3, triangle : il faut \(L_{(3)} < 1/2\). Deux morceaux ne peuvent pas dépasser \(1/2\) tous les deux, donc \(\mathbb{P}(L_{(3)} \ge 1/2) = 3 \times (1 - 1/2)^2 = 3/4\), et la probabilité cherchée vaut \(\tfrac{@Q09_triangle_num@}{@Q09_triangle_den@}\) (simulation : @Q09_triangle_mc:3@).

Étape 4, plus grand morceau : \(\mathbb{E}[L_{(3)}] = \tfrac13\big(\tfrac13 + \tfrac12 + 1\big) = \tfrac{@Q09_max_num@}{@Q09_max_den@} \approx @Q09_max:3m@\) (leçon, (E3) ; simulation : @Q09_max_mc:3@).""",
    },
    # ------------------------------------------------- Densités conjointes
    {
        "id": "M5-10", "section": "math", "theme": "Densité conjointe, probabilité sachant X = x", "difficulte": 3, "temps_cible_s": 120,
        "enonce": r"""Le couple \((X, Y)\) a pour densité conjointe \(f(x, y) = c\,(x + 2y)\) pour \(0 \le x \le 1\) et \(0 \le y \le 1\), et \(f(x, y) = 0\) ailleurs, où \(c\) est la constante de normalisation.

Que vaut \(\mathbb{P}(Y > 1/2 \mid X = 1/4)\) ?""",
        "options": [
            {"id": "a", "texte": r"\(\tfrac{@Q10_exact_num@}{@Q10_exact_den@}\)", "explication": r"Oui : \(\int_{1/2}^{1} f(1/4, y)\,dy = \tfrac{@Q10_d_joint_num@}{@Q10_d_joint_den@}\), divisé par \(f_X(1/4) = \tfrac{@Q10_fx_num@}{@Q10_fx_den@}\)."},
            {"id": "b", "texte": r"\(\tfrac{@Q10_d_marg_num@}{@Q10_d_marg_den@}\)", "explication": r"C'est \(\mathbb{P}(Y > 1/2)\) sans condition. \(X\) et \(Y\) ne sont pas indépendantes (\(x + 2y\) n'est pas un produit) : savoir \(X = 1/4\) change la loi de \(Y\)."},
            {"id": "c", "texte": r"\(\tfrac{@Q10_d_joint_num@}{@Q10_d_joint_den@}\)", "explication": r"C'est \(\int_{1/2}^{1} f(1/4, y)\,dy\), le numérateur seul. Il faut le diviser par la densité marginale \(f_X(1/4)\) pour que la densité conditionnelle ait une intégrale égale à 1."},
            {"id": "d", "texte": r"\(\tfrac{@Q10_d_unif_num@}{@Q10_d_unif_den@}\)", "explication": r"Ce serait vrai si, sachant \(X = 1/4\), \(Y\) était uniforme sur \([0, 1]\). La densité conditionnelle est proportionnelle à \(1/4 + 2y\), croissante en \(y\) : elle favorise les grandes valeurs."},
        ],
        "solution": r"""Réponse : \(\tfrac{@Q10_exact_num@}{@Q10_exact_den@}\).

Étape 1, normalisation : \(\int_0^1 \int_0^1 (x + 2y)\,dx\,dy = \tfrac12 + 1 = \tfrac32\), donc \(c = \tfrac{@Q10_c_num@}{@Q10_c_den@}\).

Étape 2, marginale : \(f_X(x) = \tfrac23 \int_0^1 (x + 2y)\,dy = \tfrac23\,(x + 1)\), donc \(f_X(1/4) = \tfrac{@Q10_fx_num@}{@Q10_fx_den@}\).

Étape 3, conditionnelle (leçon, partie 4) : \(f_{Y \mid X}(y \mid x) = \dfrac{f(x, y)}{f_X(x)} = \dfrac{x + 2y}{x + 1}\) pour \(0 \le y \le 1\) ; la constante \(c\) se simplifie.

Étape 4 : \(\mathbb{P}(Y > 1/2 \mid X = 1/4) = \dfrac{\int_{1/2}^{1} (1/4 + 2y)\,dy}{1/4 + 1} = \dfrac{1/8 + 3/4}{5/4} = \tfrac{@Q10_exact_num@}{@Q10_exact_den@}\).

Contrôles : sans condition, \(\mathbb{P}(Y > 1/2) = \tfrac{@Q10_d_marg_num@}{@Q10_d_marg_den@}\) (simulation : @Q10_marg_mc:3@) ; simulation de la probabilité conditionnelle, dans une bande étroite autour de \(x = 1/4\) : @Q10_mc:3@.""",
    },
    {
        "id": "M5-11", "section": "math", "theme": "Densité conjointe sur un domaine non rectangulaire, lois marginales et conditionnelles", "difficulte": 4, "temps_cible_s": 150,
        "correctes": ["a", "b"],
        "enonce": r"""Le couple \((X, Y)\) a pour densité conjointe \(f(x, y) = 2\,e^{-x-y}\) pour \(0 < x < y\), et \(f(x, y) = 0\) ailleurs.

Quelles affirmations sont justes ?""",
        "options": [
            {"id": "a", "texte": r"\(X\) suit la loi exponentielle de paramètre 2.",
             "explication": r"Juste : \(f_X(x) = 2e^{-x} \int_x^\infty e^{-y}\,dy = 2e^{-2x}\). Son espérance vaut \(\tfrac{@Q11_ex_num@}{@Q11_ex_den@}\)."},
            {"id": "b", "texte": r"Sachant \(X = x\), la variable \(Y - x\) suit la loi exponentielle de paramètre 1.",
             "explication": r"Juste : \(f_{Y \mid X}(y \mid x) = 2e^{-x-y}/(2e^{-2x}) = e^{-(y - x)}\) pour \(y > x\), densité de \(x\) plus une exponentielle de paramètre 1."},
            {"id": "c", "texte": r"\(X\) et \(Y\) sont indépendantes, car \(f(x, y) = 2e^{-x} \times e^{-y}\) est un produit d'une fonction de \(x\) par une fonction de \(y\).",
             "explication": r"Faux : le domaine \(0 < x < y\) n'est pas un pavé. Savoir \(X = x\) impose \(Y > x\) ; d'ailleurs \(\mathrm{Cov}(X, Y) = \tfrac{@Q11_cov_num@}{@Q11_cov_den@}\), non nulle."},
            {"id": "d", "texte": r"\(Y\) suit la loi exponentielle de paramètre 1.",
             "explication": r"Faux : \(f_Y(y) = 2e^{-y} \int_0^y e^{-x}\,dx = 2e^{-y}(1 - e^{-y})\), d'espérance \(\tfrac{@Q11_ey_num@}{@Q11_ey_den@}\) et non 1. C'est la loi conditionnelle de \(Y - x\) qui est exponentielle, pas la loi de \(Y\)."},
        ],
        "solution": r"""Réponse : seules les affirmations « \(X\) suit la loi exponentielle de paramètre 2 » et « sachant \(X = x\), \(Y - x\) suit la loi exponentielle de paramètre 1 » sont justes.

Étape 1, marginale de \(X\) : à \(x\) fixé, \(y\) va de \(x\) à l'infini. \(f_X(x) = \int_x^\infty 2e^{-x-y}\,dy = 2e^{-x} \cdot e^{-x} = 2e^{-2x}\) pour \(x > 0\) : loi exponentielle de paramètre 2, d'espérance \(\tfrac{@Q11_ex_num@}{@Q11_ex_den@}\) (simulation : @Q11_ex_mc:3@).

Étape 2, conditionnelle : \(f_{Y \mid X}(y \mid x) = \dfrac{2e^{-x-y}}{2e^{-2x}} = e^{-(y - x)}\) pour \(y > x\). Ainsi \(\mathbb{P}(Y > x + 1 \mid X = x) = e^{-1} \approx @Q11_p_cond:3m@\) pour tout \(x\) (simulation autour de \(x = 0{,}5\) : @Q11_p_cond_mc:3@).

Étape 3, marginale de \(Y\) : à \(y\) fixé, \(x\) va de 0 à \(y\). \(f_Y(y) = 2e^{-y}(1 - e^{-y})\), qui n'est pas une densité exponentielle : \(\mathbb{E}[Y] = \tfrac{@Q11_ey_num@}{@Q11_ey_den@}\) (simulation : @Q11_ey_mc:3@) et \(\mathbb{P}(Y > 1) = 2e^{-1} - e^{-2} \approx @Q11_p_y1:3m@\), contre \(@Q11_d_p_y1:3m@\) pour une exponentielle de paramètre 1.

Étape 4, indépendance : la formule est un produit, mais sur le domaine \(0 < x < y\), qui n'est pas un pavé (leçon, partie 4, (D4)). \(f_X(x)\,f_Y(y) > 0\) pour \(x > y > 0\), où \(f = 0\) : pas d'indépendance.

Lecture : \(f\) est la densité du couple (minimum, maximum) de deux durées exponentielles indépendantes de paramètre 1. Le minimum est exponentiel de paramètre 2, puis la course recommence pour la durée restante (séance de mathématiques 2, résultats (A3) et (A5)).""",
    },
    # ------------------------------------------------- Conditionner sur la bonne information
    {
        "id": "M5-12", "section": "math", "theme": "Conditionnement, famille de trois enfants", "difficulte": 3, "temps_cible_s": 75,
        "enonce": r"""Une famille a trois enfants. Chaque enfant est une fille ou un garçon avec probabilité 1/2, indépendamment des autres. À la question « avez-vous au moins une fille ? », le parent répond oui, et il dit vrai.

Quelle est la probabilité que les trois enfants soient des filles ?""",
        "options": [
            {"id": "a", "texte": r"\(\tfrac{@Q12_exact_num@}{@Q12_exact_den@}\)", "explication": r"Oui : @Q12_nb_issues@ des 8 familles équiprobables contiennent au moins une fille, et une seule est composée de trois filles."},
            {"id": "b", "texte": r"\(\tfrac{@Q12_d_aine_num@}{@Q12_d_aine_den@}\)", "explication": r"C'est la réponse quand un enfant **désigné** est une fille (l'aîné, ou un enfant rencontré au hasard) : les deux autres sont alors des filles avec probabilité \(1/2 \times 1/2\). Ici aucun enfant n'est désigné."},
            {"id": "c", "texte": r"\(\tfrac{@Q12_d_sans_num@}{@Q12_d_sans_den@}\)", "explication": r"C'est la probabilité de trois filles avant toute information. La réponse du parent élimine la famille de trois garçons : la probabilité doit augmenter."},
            {"id": "d", "texte": r"\(\tfrac{@Q12_d_deux_num@}{@Q12_d_deux_den@}\)", "explication": r"C'est la réponse du problème à deux enfants (trois familles possibles, une favorable). Avec trois enfants, l'information « au moins une fille » laisse sept familles possibles."},
        ],
        "solution": r"""Réponse : \(\tfrac{@Q12_exact_num@}{@Q12_exact_den@}\).

Univers : les 8 triplets (aîné, deuxième, benjamin) formés de F et de G, chacun de probabilité \(1/8\).

Étape 1, événement observé : le parent répond oui exactement quand la famille n'est pas GGG. L'événement contient @Q12_nb_issues@ triplets, de probabilité totale \(7/8\).

Étape 2 : un seul de ces triplets est FFF, de probabilité \(1/8\). Donc \(\mathbb{P}(\text{FFF} \mid \text{oui}) = \dfrac{1/8}{7/8} = \tfrac{@Q12_exact_num@}{@Q12_exact_den@}\) (simulation : @Q12_mc:3@).

Comparaison (leçon, partie 7) : si l'on avait rencontré un enfant choisi au hasard et constaté que c'est une fille, la probabilité de trois filles serait \(\tfrac{@Q12_d_aine_num@}{@Q12_d_aine_den@}\) (simulation : @Q12_hasard_mc:3@), car la famille FFF montre une fille trois fois plus souvent qu'une famille à une seule fille. La règle d'observation, et pas seulement la phrase « au moins une fille », fixe la réponse.""",
    },
]
