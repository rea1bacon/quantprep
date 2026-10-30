"""Séance de mathématiques 3 : leçon express de statistique et d'inégalités, puis 8 questions à choix multiple.

Thèmes : estimateurs, biais, variance, erreur
quadratique moyenne ; maximum de vraisemblance ; intervalles de confiance ; tests et p-values ; tests multiples ;
inégalités de Markov, Tchebychev, Jensen, Cauchy-Schwarz.
Toutes les valeurs : verif/seance_math_3.py. Dans chaque question, l'option "a" est la bonne.
"""
import lecons_math_3
import questions_diagnostic as diag

ID = "seance-math-3"
TITRE = "Séance de mathématiques 3 : statistique et inégalités"
VERIF = "verif/seance_math_3.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [{"id": "lecon", "nom": "Leçon", "budget_min": 25}, {"id": "math", "nom": "Mathématiques", "budget_min": 20}]
PRESENTATION = (
    "D'abord une leçon express de 20 à 25 minutes : biais et erreur quadratique moyenne, maximum de "
    "vraisemblance, intervalles de confiance, tests et p-values, tests multiples (Bonferroni, Benjamini-Hochberg), "
    "inégalités de Markov, Tchebychev, Jensen et Cauchy-Schwarz. Cinq points de contrôle sont corrigés "
    "immédiatement, et le notebook permet de relancer les simulations de couverture et de p-values. Ensuite 8 "
    "questions à choix multiple, dont la moitié de difficulté 4 ou 5.\n\n"
    "Objectif de vitesse : chaque question se résout dans son temps cible (1 à 2 minutes et demie) avec la leçon. "
    "Au-delà du double du temps cible affiché, choisis « Je ne sais pas » et passe à la suivante : c'est la règle "
    "d'abandon à suivre dans un test chronométré."
)
AVERTISSEMENTS = [
    "Les questions de mathématiques sont présentées ici en choix multiple à une seule bonne réponse. Cherche la réponse avant de "
    "lire les options, pour t'entraîner aussi au format à réponse numérique.",
    "Ici le retour en arrière est permis et il n'y a pas de pénalité : ce sont les règles de cette plateforme, pas "
    "celles de tout test en ligne.",
]

LECONS = [lecons_math_3.LECON_MATH_3]
CODES = []

QUESTIONS = [
    # ------------------------------------------------- Estimateurs, EQM
    {
        "id": "M3-01", "section": "math", "theme": "Estimateurs, décomposition de l'erreur quadratique moyenne", "difficulte": 2, "temps_cible_s": 75,
        "enonce": r"""Soient \(X_1, \dots, X_9\) neuf variables aléatoires indépendantes et de même loi, d'espérance \(\mu = 2\) et de variance \(\sigma^2 = 4\). On estime \(\mu\) par \(\tilde{\mu} = \tfrac{1}{10}(X_1 + \dots + X_9)\) : la somme est divisée par 10, et non par 9.

Que vaut l'erreur quadratique moyenne \(\mathbb{E}[(\tilde{\mu} - \mu)^2]\), arrondie à deux décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q01_exact:2@", "explication": r"Oui : variance \(@Q01_var:2m@\) plus biais au carré \((@Q01_biais:1m@)^2 = @Q01_biais_carre:2m@\)."},
            {"id": "b", "texte": "@Q01_d_xbar:2@", "explication": r"C'est \(\sigma^2/9\), l'erreur quadratique de la moyenne empirique \(\bar{X}\), sans biais. L'estimateur demandé divise par 10 : sa variance et son biais changent tous les deux."},
            {"id": "c", "texte": "@Q01_var:2@", "explication": r"C'est la variance seule, \(9\sigma^2/100\). L'estimateur est biaisé (\(\mathbb{E}[\tilde{\mu}] = 0{,}9\,\mu\)) : il faut ajouter le carré du biais."},
            {"id": "d", "texte": "@Q01_d_moins:2@", "explication": r"Le carré du biais a été soustrait au lieu d'être ajouté. Les deux termes de la décomposition sont positifs : l'erreur quadratique est au moins égale à la variance."},
        ],
        "solution": r"""Réponse : \(@Q01_exact:2m@\).

Étape 1, biais : \(\mathbb{E}[\tilde{\mu}] = \tfrac{9}{10}\mu = 1{,}8\), donc le biais vaut \(1{,}8 - 2 = @Q01_biais:1m@\).

Étape 2, variance : les \(X_i\) étant indépendantes, \(\mathrm{Var}(X_1 + \dots + X_9) = 9\sigma^2 = 36\), donc \(\mathrm{Var}(\tilde{\mu}) = 36/100 = @Q01_var:2m@\).

Étape 3 : erreur quadratique moyenne = variance + biais au carré (leçon, partie 1) \(= @Q01_var:2m@ + @Q01_biais_carre:2m@ = @Q01_exact:2m@\).

Remarque : \(\bar{X}\) a une erreur quadratique de \(4/9 \approx @Q01_d_xbar:3m@\), plus grande. Mais l'erreur de \(\tilde{\mu}\), égale à \(0{,}36 + \mu^2/100\), dépend de \(\mu\) inconnu : \(\tilde{\mu}\) ne bat \(\bar{X}\) que si \(\mu^2 < \tfrac{@Q01_mu2_seuil_num@}{@Q01_mu2_seuil_den@}\). Simulation (script, loi normale) : @Q01_mc:3@.""",
    },
    # ------------------------------------------------- Maximum de vraisemblance
    {
        "id": "M3-02", "section": "math", "theme": "Maximum de vraisemblance, loi exponentielle", "difficulte": 2, "temps_cible_s": 60,
        "enonce": r"""On observe six durées entre événements, en minutes, modélisées comme des réalisations de variables indépendantes de loi exponentielle de paramètre \(\lambda > 0\), de densité \(\lambda e^{-\lambda x}\) pour \(x \ge 0\) : 0,2 ; 0,5 ; 0,6 ; 0,9 ; 1,1 ; 1,5.

Quelle est l'estimation du maximum de vraisemblance de \(\lambda\), en événements par minute, arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q02_exact:3@", "explication": r"Oui : \(\hat{\lambda} = n/\sum_i x_i = 6/@Q02_somme:1m@\)."},
            {"id": "b", "texte": "@Q02_moy:3@", "explication": r"C'est la moyenne \(\bar{x}\), estimation du maximum de vraisemblance de l'espérance \(1/\lambda\), pas de \(\lambda\). Par invariance, l'estimation de \(\lambda\) est son inverse."},
            {"id": "c", "texte": "@Q02_d_sans_biais:3@", "explication": r"C'est \((n-1)/\sum_i x_i = 5/@Q02_somme:1m@\), l'estimateur **sans biais** de \(\lambda\) (leçon, partie 2 : \(\mathbb{E}[1/T] = \lambda/(n-1)\)). Ce n'est pas le maximum de la vraisemblance, atteint en \(n/\sum_i x_i\)."},
            {"id": "d", "texte": "@Q02_d_mediane:3@", "explication": r"C'est \(\ln 2\) divisé par la médiane empirique \(@Q02_mediane:2m@\) : un estimateur fondé sur la médiane de la loi exponentielle, \(\ln 2/\lambda\). Il est raisonnable, mais ce n'est pas le maximum de vraisemblance."},
        ],
        "solution": r"""Réponse : \(\hat{\lambda} = 6/@Q02_somme:1m@ = @Q02_exact:3m@\) par minute.

Étape 1 : la log-vraisemblance vaut \(\ell(\lambda) = \sum_i \ln(\lambda e^{-\lambda x_i}) = 6\ln\lambda - \lambda \sum_i x_i\), avec \(\sum_i x_i = @Q02_somme:1m@\).

Étape 2 : \(\ell'(\lambda) = 6/\lambda - @Q02_somme:1m@\), positive pour \(\lambda < 6/@Q02_somme:1m@\) et négative après : le maximum est en \(\hat{\lambda} = 6/@Q02_somme:1m@ = @Q02_exact:3m@\) (et \(\ell''(\lambda) = -6/\lambda^2 < 0\)).

Contrôle : la durée moyenne observée est \(@Q02_moy:1m@\) minute, et \(1/@Q02_moy:1m@ = @Q02_exact:2m@\) événement par minute.""",
    },
    {
        "id": "M3-03", "section": "math", "theme": "Biais de l'estimateur du maximum de vraisemblance, Jensen", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Soient \(X_1, \dots, X_5\) cinq variables aléatoires indépendantes de loi exponentielle de paramètre \(\lambda = 2\) (densité \(2e^{-2x}\) pour \(x \ge 0\), espérance \(1/2\)). On note \(\bar{X} = \tfrac15 (X_1 + \dots + X_5)\) et \(\hat{\lambda} = 1/\bar{X}\), l'estimateur du maximum de vraisemblance de \(\lambda\).

Que vaut \(\mathbb{E}[\hat{\lambda}]\) ?""",
        "options": [
            {"id": "a", "texte": "@Q03_exact:1@", "explication": r"Oui : \(\mathbb{E}[1/\bar{X}] = \tfrac{n}{n-1}\lambda = \tfrac54 \times 2\)."},
            {"id": "b", "texte": "2", "explication": r"Ce serait vrai si \(\mathbb{E}[1/\bar{X}] = 1/\mathbb{E}[\bar{X}]\). Or \(x \mapsto 1/x\) est strictement convexe sur \(]0, \infty[\) et \(\bar{X}\) n'est pas constante : par Jensen, \(\mathbb{E}[1/\bar{X}] > 1/\mathbb{E}[\bar{X}] = 2\)."},
            {"id": "c", "texte": "@Q03_d_inverse:1@", "explication": r"C'est \(\tfrac{n-1}{n}\lambda\) : le facteur est inversé. Jensen impose une espérance **supérieure** à 2, ce qui élimine cette option sans calcul."},
            {"id": "d", "texte": "@Q03_d_moyenne:1@", "explication": r"C'est \(\mathbb{E}[\bar{X}] = 1/\lambda\), l'espérance de la moyenne des durées, pas celle de son inverse."},
        ],
        "solution": r"""Réponse : \(\mathbb{E}[\hat{\lambda}] = \tfrac54 \times 2 = @Q03_exact:1m@\).

Notations : \(T = X_1 + \dots + X_5\), de sorte que \(\hat{\lambda} = 5/T\).

Étape 1 : pour \(s > 0\), \(1/s = \int_0^\infty e^{-us}\,du\) ; par Fubini-Tonelli (intégrande positive), \(\mathbb{E}[1/T] = \int_0^\infty \mathbb{E}[e^{-uT}]\,du\).

Étape 2 : \(\mathbb{E}[e^{-uX_1}] = \lambda/(\lambda + u)\), et par indépendance \(\mathbb{E}[e^{-uT}] = (\lambda/(\lambda + u))^5\).

Étape 3 : \(\int_0^\infty \lambda^5 (\lambda + u)^{-5}\,du = \lambda^5 \cdot \lambda^{-4}/4 = \lambda/4\). Donc \(\mathbb{E}[\hat{\lambda}] = 5\lambda/4 = @Q03_exact:1m@\) (leçon, partie 2).

Contrôle de sens : Jensen avec \(1/x\), strictement convexe, impose \(\mathbb{E}[\hat{\lambda}] > 2\). Simulation (script) : @Q03_mc:3@.""",
    },
    # ------------------------------------------------- Intervalle de confiance
    {
        "id": "M3-04", "section": "math", "theme": "Intervalle de confiance de Student", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""On dispose de 9 mesures modélisées comme indépendantes de loi normale \(\mathcal{N}(\mu, \sigma^2)\), avec \(\mu\) et \(\sigma\) inconnus. La moyenne observée est 10,4 et l'écart-type empirique, calculé avec le diviseur \(n - 1 = 8\), vaut \(s = 1{,}5\).

Quantiles utiles, arrondis à trois décimales : \(z_{0{,}975} = @L3_z975:3m@\), \(z_{0{,}95} = @L4_z95:3m@\) (loi normale centrée réduite) ; \(t_{8; 0{,}975} = @Q04_t8:3m@\), \(t_{9; 0{,}975} = @Q04_t9:3m@\), \(t_{8; 0{,}95} = @Q04_t8_unilat:3m@\) (loi de Student, le premier indice étant le nombre de degrés de liberté).

Quelle est la demi-largeur de l'intervalle de confiance bilatéral exact de niveau 95 % pour \(\mu\), arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q04_exact:3@", "explication": r"Oui : \(t_{8; 0{,}975} \times s/\sqrt{9} = @Q04_t8:3m@ \times @Q04_erreur_type:1m@\)."},
            {"id": "b", "texte": "@Q04_d_z:3@", "explication": r"C'est le quantile normal \(z_{0{,}975}\), valable si \(\sigma\) est connu. Avec \(\sigma\) estimé, le pivot suit une loi de Student ; avec \(z\), la couverture réelle n'est que \(@Q04_couv_z:3m@\) pour \(n = 9\) (script)."},
            {"id": "c", "texte": "@Q04_d_t9:3@", "explication": r"Mauvais nombre de degrés de liberté : \(n = 9\) mesures donnent \(n - 1 = 8\) degrés de liberté, car les 9 écarts \(x_i - \bar{x}\) sont liés par une contrainte : leur somme est nulle."},
            {"id": "d", "texte": "@Q04_d_unilat:3@", "explication": r"C'est le quantile d'ordre 0,95, celui d'un intervalle **unilatéral**. Un intervalle bilatéral à 95 % laisse 2,5 % de chaque côté : quantile d'ordre 0,975."},
        ],
        "solution": r"""Réponse : \(@Q04_t8:3m@ \times 1{,}5/3 = @Q04_exact:3m@\).

Étape 1 : sous normalité, \(T = \sqrt{n}(\bar{X} - \mu)/S\) suit la loi de Student à \(n - 1 = 8\) degrés de liberté (leçon, partie 3).

Étape 2 : l'intervalle exact est \(\bar{x} \pm t_{8; 0{,}975}\,s/\sqrt{n}\), et l'erreur-type vaut \(s/\sqrt{9} = 1{,}5/3 = @Q04_erreur_type:1m@\).

Étape 3 : demi-largeur \(@Q04_t8:3m@ \times @Q04_erreur_type:1m@ = @Q04_exact:3m@\) ; intervalle \([@Q04_bas:3m@ ; @Q04_haut:3m@]\).

Vérification (script) : couverture simulée de cet intervalle, @Q04_couv_mc:3@ pour 0,95.""",
    },
    # ------------------------------------------------- p-value et Bayes
    {
        "id": "M3-05", "section": "math", "theme": "Tests : lecture d'un rejet, formule de Bayes", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Une équipe teste un grand nombre de signaux. Avant tout test, chaque signal a un effet réel avec probabilité 0,05. Chaque test rejette l'hypothèse « pas d'effet » avec probabilité exactement 0,05 quand le signal n'a pas d'effet (niveau), et avec probabilité 0,5 quand il en a un (puissance).

On choisit un signal au hasard ; son test vient de rejeter l'hypothèse « pas d'effet ». Quelle est la probabilité que ce signal n'ait en réalité aucun effet, arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q05_exact:3@", "explication": r"Oui : \(\dfrac{0{,}95 \times 0{,}05}{0{,}95 \times 0{,}05 + 0{,}05 \times 0{,}5}\)."},
            {"id": "b", "texte": "0,050", "explication": r"C'est le niveau, \(\mathbb{P}(\text{rejet} \mid \text{pas d'effet})\). La question demande la probabilité inverse, \(\mathbb{P}(\text{pas d'effet} \mid \text{rejet})\), qui dépend aussi de la proportion de vrais effets et de la puissance."},
            {"id": "c", "texte": "@Q05_d_complement:3@", "explication": r"C'est \(\mathbb{P}(\text{effet réel} \mid \text{rejet})\), le complémentaire de la quantité demandée."},
            {"id": "d", "texte": "@Q05_num:4@", "explication": r"C'est la probabilité jointe \(\mathbb{P}(\text{pas d'effet et rejet}) = 0{,}95 \times 0{,}05\). Il faut encore diviser par la probabilité de rejet, \(@Q05_den:4m@\)."},
        ],
        "solution": r"""Réponse : \(\tfrac{@Q05_exact_num@}{@Q05_exact_den@} \approx @Q05_exact:3m@\).

Notations : \(N\) est l'événement « le signal n'a pas d'effet », \(R\) l'événement « le test rejette ».

Étape 1 : \(\mathbb{P}(N \cap R) = 0{,}95 \times 0{,}05 = @Q05_num:4m@\) et \(\mathbb{P}(N^c \cap R) = 0{,}05 \times 0{,}5 = @Q05_vrais:3m@\).

Étape 2, probabilités totales : \(\mathbb{P}(R) = @Q05_num:4m@ + @Q05_vrais:3m@ = @Q05_den:4m@\).

Étape 3, Bayes : \(\mathbb{P}(N \mid R) = @Q05_num:4m@/@Q05_den:4m@ = @Q05_exact:3m@\).

Lecture : avec peu de vrais effets et une puissance moyenne, près de deux rejets sur trois sont des fausses découvertes, bien que chaque test soit de niveau 5 % (leçon, partie 4). Simulation (script) : @Q05_mc:3@.""",
    },
    # ------------------------------------------------- Benjamini-Hochberg
    {
        "id": "M3-06", "section": "math", "theme": "Tests multiples, procédure de Benjamini-Hochberg", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""On teste 8 hypothèses nulles. Les p-values, dans l'ordre des tests, sont :

0,20 ; 0,034 ; 0,80 ; 0,001 ; 0,070 ; 0,45 ; 0,048 ; 0,030.

On applique la procédure de Benjamini-Hochberg au niveau \(q = 0{,}10\) : on trie les p-values par ordre croissant, \(p_{(1)} \le \dots \le p_{(8)}\), on cherche le plus grand \(i\) tel que \(p_{(i)} \le i\,q/8\), et l'on rejette les \(i\) hypothèses de plus petites p-values (aucune si un tel \(i\) n'existe pas).

Combien d'hypothèses sont rejetées ?""",
        "options": [
            {"id": "a", "texte": "@Q06_exact@", "explication": r"Oui : \(p_{(4)} = 0{,}048 \le 4 \times 0{,}0125 = 0{,}05\), et aucun \(i\) plus grand ne passe."},
            {"id": "b", "texte": "@Q06_d_descendant@", "explication": r"C'est le résultat si l'on s'arrête au premier échec, \(p_{(2)} = 0{,}030 > 0{,}025\) : procédure descendante, pas celle de Benjamini-Hochberg. C'est aussi le nombre de rejets de Bonferroni au seuil \(0{,}10/8 = @Q06_bonf_seuil:4m@\)."},
            {"id": "c", "texte": "@Q06_d_individuel@", "explication": r"On a rejeté seulement les p-values inférieures à leur propre seuil (\(p_{(1)}\), \(p_{(3)}\), \(p_{(4)}\)). La procédure est montante : une fois \(k = 4\) trouvé, on rejette les 4 plus petites, \(p_{(2)}\) comprise."},
            {"id": "d", "texte": "@Q06_d_brut@", "explication": r"C'est le nombre de p-values au plus égales à 0,10, sans aucune correction. \(p_{(5)} = 0{,}070\) dépasse son seuil \(5 \times 0{,}0125 = 0{,}0625\)."},
        ],
        "solution": r"""Réponse : @Q06_exact@ hypothèses rejetées.

Étape 1, tri : \(@Q06_tries:3m@\).

Étape 2, seuils \(i\,q/8 = 0{,}0125\,i\) : \(@Q06_seuils:4m@\).

Étape 3, comparaison : \(i = 1\) passe (\(0{,}001 \le 0{,}0125\)) ; \(i = 2\) échoue (\(0{,}030 > 0{,}025\)) ; \(i = 3\) passe (\(0{,}034 \le 0{,}0375\)) ; \(i = 4\) passe (\(0{,}048 \le 0{,}05\)) ; \(i = 5\) à 8 échouent (\(0{,}070 > 0{,}0625\), puis \(0{,}20\), \(0{,}45\), \(0{,}80\) au-dessus de leurs seuils, tous au plus 0,1).

Étape 4 : le plus grand \(i\) qui passe est \(k = 4\) ; on rejette les tests de p-values 0,001 ; 0,030 ; 0,034 ; 0,048, y compris celui dont la p-value dépasse son propre seuil (leçon, partie 5, piège 1).""",
    },
    # ------------------------------------------------- EQM sous normalité
    {
        "id": "M3-07", "section": "math", "theme": "Erreur quadratique des estimateurs de la variance", "difficulte": 5, "temps_cible_s": 150,
        "enonce": r"""Soient \(X_1, \dots, X_{10}\) dix variables aléatoires indépendantes de loi normale \(\mathcal{N}(\mu, \sigma^2)\), avec \(\mu\) et \(\sigma^2 > 0\) inconnus, \(\bar{X}\) leur moyenne et \(Q = \sum_{i=1}^{10} (X_i - \bar{X})^2\). On estime \(\sigma^2\) par \(Q/d\), où \(d > 0\) est un réel fixé.

Pour quelle valeur de \(d\) l'erreur quadratique moyenne \(\mathbb{E}[(Q/d - \sigma^2)^2]\) est-elle minimale ? (La réponse ne dépend ni de \(\mu\) ni de \(\sigma^2\).)""",
        "options": [
            {"id": "a", "texte": "@Q07_exact@", "explication": r"Oui : \(d = n + 1\), d'erreur \(\tfrac{@Q07_eqm_11_num@}{@Q07_eqm_11_den@}\sigma^4\)."},
            {"id": "b", "texte": "@Q07_d_sans_biais@", "explication": r"\(d = n - 1\) donne l'estimateur **sans biais**, d'erreur \(\tfrac{@Q07_eqm_9_num@}{@Q07_eqm_9_den@}\sigma^4\). L'absence de biais n'est pas le minimum de l'erreur : un peu de biais achète une baisse de variance plus grande."},
            {"id": "c", "texte": "@Q07_d_emv@", "explication": r"\(d = n\) donne l'estimateur du maximum de vraisemblance, d'erreur \(\tfrac{@Q07_eqm_10_num@}{@Q07_eqm_10_den@}\sigma^4\) : meilleur que \(n - 1\), mais pas optimal."},
            {"id": "d", "texte": "@Q07_d_khi_n@", "explication": r"C'est \(n + 2\), l'optimum si \(Q/\sigma^2\) suivait \(\chi^2_n\), c'est-à-dire si l'on centrait sur \(\mu\) connu. Centrer sur \(\bar{X}\) fait perdre un degré de liberté : \(\chi^2_{n-1}\), et l'optimum devient \((n - 1) + 2 = n + 1\). L'erreur en \(d = 12\) vaut \(\tfrac{@Q07_eqm_12_num@}{@Q07_eqm_12_den@}\sigma^4\)."},
        ],
        "solution": r"""Réponse : \(d = @Q07_exact@\), soit \(n + 1\).

Notations : \(n = 10\), \(k = n - 1 = @Q07_k@\), \(W = Q/\sigma^2\), \(c = 1/d\).

Étape 1 : par le théorème de Fisher (leçon, partie 1, étape 5), \(W\) suit \(\chi^2_k\), donc \(\mathbb{E}[W] = k\) et \(\mathbb{E}[W^2] = \mathrm{Var}(W) + k^2 = 2k + k^2\).

Étape 2 : \(\mathbb{E}[(cQ - \sigma^2)^2] = \sigma^4\,\mathbb{E}[(cW - 1)^2] = \sigma^4 \big(c^2 (2k + k^2) - 2ck + 1\big)\), un trinôme en \(c\) de coefficient dominant positif.

Étape 3 : il est minimal où sa dérivée s'annule : \(2c\,(2k + k^2) = 2k\), soit \(c = \dfrac{1}{k + 2} = \dfrac{1}{n + 1}\). Donc \(d = n + 1 = @Q07_exact@\).

Valeurs, en unités de \(\sigma^4\) : \(d = 9\) : \(@Q07_eqm_9:4m@\) ; \(d = 10\) : \(@Q07_eqm_10:4m@\) ; \(d = 11\) : \(@Q07_eqm_11:4m@\) ; \(d = 12\) : \(@Q07_eqm_12:4m@\). Simulation (script) pour \(d = 11\) : @Q07_eqm_11_mc:4@.

Condition de validité : ce résultat utilise la normalité (variance \(2k\) du khi-deux). Pour une autre loi, l'optimum dépend du coefficient d'aplatissement.""",
    },
    # ------------------------------------------------- Tchebychev
    {
        "id": "M3-08", "section": "math", "theme": "Inégalité de Tchebychev, taille d'échantillon", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""On estime la probabilité inconnue \(\theta \in [0, 1]\) qu'un ordre soit exécuté par la proportion \(\hat{\theta}\) d'ordres exécutés parmi \(n\), les \(n\) ordres étant modélisés comme des tirages de Bernoulli indépendants de paramètre \(\theta\).

Quel est le plus petit entier \(n\) pour lequel l'inégalité de Tchebychev, appliquée à \(\hat{\theta}\), garantit \(\mathbb{P}(\lvert \hat{\theta} - \theta \rvert \ge 0{,}05) \le 0{,}05\) pour toute valeur de \(\theta\) ?""",
        "options": [
            {"id": "a", "texte": "@Q08_exact@", "explication": r"Oui : \(\mathrm{Var}(\hat{\theta}) \le 1/(4n)\), et \(\tfrac{1}{4n \times 0{,}05^2} \le 0{,}05\) équivaut à \(n \ge @Q08_exact@\)."},
            {"id": "b", "texte": "@Q08_d_tcl@", "explication": r"C'est l'approximation du théorème central limite, \(n \ge (z_{0{,}975} \times 0{,}5/0{,}05)^2 \approx @Q08_tcl_brut:2m@\). Ce n'est pas une garantie, ni une application de Tchebychev : pour \(\theta = 1/2\) et \(n = @Q08_d_tcl@\), la probabilité exacte (loi binomiale) vaut \(@Q08_p_tcl:4m@\), au-dessus de 0,05."},
            {"id": "c", "texte": "@Q08_d_sigma1@", "explication": r"La variance d'une Bernoulli a été majorée par 1. Elle vaut \(\theta(1 - \theta) \le 1/4\), car \(1/4 - \theta(1 - \theta) = (\theta - 1/2)^2 \ge 0\)."},
            {"id": "d", "texte": "@Q08_d_eps@", "explication": r"L'écart 0,05 n'a pas été mis au carré : Tchebychev divise la variance par \(a^2\), pas par \(a\)."},
        ],
        "solution": r"""Réponse : \(n = @Q08_exact@\).

Étape 1 : \(\hat{\theta}\) est la moyenne de \(n\) Bernoulli indépendantes, donc \(\mathbb{E}[\hat{\theta}] = \theta\) et \(\mathrm{Var}(\hat{\theta}) = \theta(1 - \theta)/n\) (leçon, partie 1, étape 2).

Étape 2 : \(\theta(1 - \theta) \le 1/4\) pour tout \(\theta\), avec égalité en \(\theta = 1/2\), car \(1/4 - \theta(1 - \theta) = (\theta - 1/2)^2\).

Étape 3, Tchebychev : \(\mathbb{P}(\lvert \hat{\theta} - \theta \rvert \ge 0{,}05) \le \dfrac{\theta(1 - \theta)}{n \times 0{,}05^2}\), dont le maximum en \(\theta\) vaut \(\dfrac{1}{4n \times 0{,}0025} = \dfrac{100}{n}\).

Étape 4 : \(100/n \le 0{,}05\) équivaut à \(n \ge @Q08_exact@\) ; pour \(n = 1999\), la borne dépasse 0,05 en \(\theta = 1/2\), donc Tchebychev ne garantit plus rien.

Remarque : la borne est très prudente ; la probabilité exacte pour \(n = @Q08_exact@\) et \(\theta = 1/2\) est de l'ordre de \(10^{-5}\) (script, loi binomiale).""",
    },
]
