"""Séance de ML 4 : leçon express (optimisation, réseaux, architectures, noyaux, EM), puis 8 questions à choix multiple.

Adam (DIAG-A4), la rétropropagation (DIAG-A3) et les convolutions (DIAG-A2) sont déjà traités dans le test de
positionnement : la leçon L-ML-4 (outils/lecons_ml_4.py) ne fait que les rappeler et se concentre sur le
reste du ML avancé : conditionnement, SGD, momentum, initialisation, normalisations, dropout,
connexions résiduelles, champ réceptif dilaté, LSTM, attention, SVM, noyaux, processus gaussiens et EM.
Toutes les valeurs : verif/seance_ml_4.py. Dans chaque question, l'option "a" est la bonne.
"""
import lecons_ml_4
import questions_diagnostic as diag

ID = "seance-ml-4"
TITRE = "Séance de ML 4 : optimisation, réseaux de neurones, noyaux et EM"
VERIF = "verif/seance_ml_4.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [{"id": "lecon", "nom": "Leçon", "budget_min": 35}, {"id": "mla", "nom": "Machine Learning avancé", "budget_min": 16}]
PRESENTATION = (
    "D'abord une leçon d'environ 35 minutes (15 en ne lisant que les énoncés, exemples et réflexes) : descente de gradient et conditionnement, momentum, gradient "
    "stochastique et calendriers de pas, rappel d'Adam, rétropropagation matricielle, initialisation, disparition du "
    "gradient, normalisations, dropout, connexions résiduelles, convolutions causales dilatées, RNN et LSTM, attention, "
    "SVM, astuce du noyau, ridge à noyau, processus gaussiens et EM, avec cinq points de contrôle corrigés "
    "immédiatement et des expériences dont la sortie est réelle. Adam, la rétropropagation et les convolutions, "
    "déjà traités dans le test de positionnement, ne sont que rappelés. Ensuite 8 questions à choix multiple, dont 5 de difficulté 4 ou 5.\n\n"
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

LECONS = [lecons_ml_4.LECON_ML_4]
CODES = []

QUESTIONS = [
    # ------------------------------------------------- Optimisation
    {
        "id": "ML4-01", "section": "mla", "theme": "Descente de gradient, pas optimal et conditionnement", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""On minimise \(f(\boldsymbol{\theta}) = \tfrac12 \boldsymbol{\theta}^\top \mathbf{A}\,\boldsymbol{\theta} - \mathbf{b}^\top \boldsymbol{\theta}\), où \(\mathbf{A}\) est une matrice symétrique \(2 \times 2\) de valeurs propres 2 et 18 et \(\mathbf{b} \in \mathbb{R}^2\) est fixé, par descente de gradient à pas fixe \(\eta > 0\) : \(\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta\,\nabla f(\boldsymbol{\theta}_{t-1})\). On note \(\boldsymbol{\theta}^*\) le minimum et \(q(\eta)\) le plus petit nombre tel que \(\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert \le q(\eta)^t\,\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert\) pour tout point de départ et tout \(t \ge 1\).

Quel pas \(\eta\) minimise \(q(\eta)\), et que vaut alors \(q\) ?""",
        "options": [
            {"id": "a", "texte": r"\(\eta = \tfrac{@Q01_eta_num@}{@Q01_eta_den@}\) et \(q = \tfrac{@Q01_q_num@}{@Q01_q_den@}\)",
             "explication": r"Oui : \(\eta^* = 2/(\lambda_{\min} + L) = 2/20\), et les deux directions sont contractées du même facteur \(\lvert 1 - 2/10 \rvert = \lvert 1 - 18/10 \rvert = (\kappa - 1)/(\kappa + 1)\) avec \(\kappa = @Q01_kappa@\)."},
            {"id": "b", "texte": r"\(\eta = \tfrac{@Q01_d_invL_eta_num@}{@Q01_d_invL_eta_den@}\) et \(q = \tfrac{@Q01_d_invL_q_num@}{@Q01_d_invL_q_den@}\)",
             "explication": r"Le pas \(1/L\) converge, avec \(q = 1 - 1/\kappa\), mais il n'est pas optimal : il annule l'erreur dans la direction de courbure 18 et laisse la direction plate trop lente. Augmenter \(\eta\) jusqu'à \(2/(\lambda_{\min} + L)\) équilibre les deux directions."},
            {"id": "c", "texte": r"\(\eta = \tfrac{@Q01_d_max_eta_num@}{@Q01_d_max_eta_den@}\) et \(q = \tfrac{@Q01_d_max_qlent_num@}{@Q01_d_max_qlent_den@}\)",
             "explication": r"C'est le pas limite \(2/L\), et \(\tfrac{@Q01_d_max_qlent_num@}{@Q01_d_max_qlent_den@}\) n'est que le facteur de la direction plate. Dans la direction de courbure 18, \(\lvert 1 - 18 \times 2/18 \rvert = 1\) : l'erreur ne décroît pas, donc \(q = @Q01_d_max_q@\)."},
            {"id": "d", "texte": r"\(\eta = \tfrac{@Q01_d_lmin_eta_num@}{@Q01_d_lmin_eta_den@}\) et \(q = 0\)",
             "explication": r"Le pas \(1/\lambda_{\min}\) annule l'erreur de la direction plate en un pas, mais multiplie celle de la direction de courbure 18 par \(\lvert 1 - 9 \rvert\) : \(q = @Q01_d_lmin_q@\), la descente diverge. Le pas doit rester sous \(2/L\)."},
        ],
        "solution": r"""Réponse : \(\eta = \tfrac{@Q01_eta_num@}{@Q01_eta_den@}\), et \(q = \tfrac{@Q01_q_num@}{@Q01_q_den@}\).

Étape 1 : l'erreur \(\mathbf{e}_t = \boldsymbol{\theta}_t - \boldsymbol{\theta}^*\) vérifie \(\mathbf{e}_t = (\mathbf{I} - \eta\mathbf{A})\,\mathbf{e}_{t-1}\) ; dans la base propre, chaque coordonnée est multipliée par \(1 - \eta\lambda_k\) à chaque pas, donc \(q(\eta) = \max(\lvert 1 - 2\eta \rvert, \lvert 1 - 18\eta \rvert)\) (leçon, partie 1.1, O1).

Étape 2 : le maximum de deux fonctions, l'une décroissante tant que \(\eta < 1/2\), l'autre croissante dès que \(\eta > 1/18\), est minimal quand elles sont égales avec des signes opposés : \(1 - 2\eta = 18\eta - 1\), soit \(\eta = 2/20 = \tfrac{@Q01_eta_num@}{@Q01_eta_den@}\) (O3).

Étape 3 : \(q = 1 - 2/10 = \tfrac{@Q01_q_num@}{@Q01_q_den@}\), soit \((\kappa - 1)/(\kappa + 1)\) avec \(\kappa = 18/2 = @Q01_kappa@\). Le script vérifie qu'aucun pas d'une grille de 999 valeurs ne fait mieux.""",
    },
    {
        "id": "ML4-02", "section": "mla", "theme": "Gradient stochastique à pas constant, variance stationnaire", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""On minimise \(f(\theta) = \tfrac{a}{2}\theta^2\) avec \(a = 2\), dont le minimum est en 0, par gradient stochastique à pas constant \(\eta = 0{,}1\). Au pas \(t \ge 1\), le gradient estimé est \(G_t = a\,\Theta_{t-1} + Z_t\), où les \(Z_t\) sont des variables indépendantes, de moyenne 0 et de variance \(\sigma^2 = 0{,}5\), chacune indépendante de \(\Theta_0, \dots, \Theta_{t-1}\) ; puis \(\Theta_t = \Theta_{t-1} - \eta\,G_t\), avec \(\Theta_0 = 3\).

Vers quelle valeur tend \(\mathrm{Var}(\Theta_t)\) quand \(t \to \infty\), arrondie à quatre décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q02_exact:4@", "explication": r"Oui : \(\eta^2\sigma^2/(1 - (1 - \eta a)^2) = 0{,}005/(1 - 0{,}64) = \tfrac{@Q02_exact_num@}{@Q02_exact_den@}\)."},
            {"id": "b", "texte": "@Q02_d_approx:4@", "explication": r"C'est l'équivalent pour \(\eta\) petit, \(\eta\sigma^2/(2a)\), qui néglige le terme \(\eta a\) de \(\eta\sigma^2/(a(2 - \eta a))\). Ici \(\eta a = 0{,}2\) n'est pas négligeable devant 2 ; la question demande la valeur exacte."},
            {"id": "c", "texte": "@Q02_d_sans_carre:4@", "explication": r"Le facteur \(1 - \eta a\) n'a pas été élevé au carré dans la récurrence des variances. Or \(\mathrm{Var}(c\,\Theta) = c^2\,\mathrm{Var}(\Theta)\) : \(v_t = (1 - \eta a)^2 v_{t-1} + \eta^2\sigma^2\)."},
            {"id": "d", "texte": "@Q02_d_un_pas:4@", "explication": r"C'est \(\eta^2\sigma^2\), la variance ajoutée par un seul pas. Le bruit s'accumule d'un pas à l'autre, amorti par le facteur \((1 - \eta a)^2\) : la limite est la somme géométrique \(\eta^2\sigma^2 \sum_{j \ge 0} (1 - \eta a)^{2j}\)."},
        ],
        "solution": r"""Réponse : \(\tfrac{@Q02_exact_num@}{@Q02_exact_den@} \approx @Q02_exact:4m@\).

Étape 1 : \(\Theta_t = (1 - \eta a)\,\Theta_{t-1} - \eta Z_t\), avec \(1 - \eta a = 0{,}8\).

Étape 2 : \(Z_t\) est indépendante de \(\Theta_{t-1}\), donc \(v_t = \mathrm{Var}(\Theta_t)\) vérifie \(v_t = 0{,}8^2\,v_{t-1} + \eta^2\sigma^2\), avec \(v_0 = 0\) (\(\Theta_0\) est fixe) et \(\eta^2\sigma^2 = 0{,}005\) (leçon, partie 1.3, O5).

Étape 3 : \(v_t - v_\infty = @Q02_facteur:2m@\,(v_{t-1} - v_\infty)\) tend vers 0, avec \(v_\infty = 0{,}005/(1 - @Q02_facteur:2m@) = \tfrac{@Q02_exact_num@}{@Q02_exact_den@}\). Le départ \(\Theta_0 = 3\) n'agit que sur l'espérance, \(3 \times 0{,}8^t \to 0\).

Simulation de 40000 chaînes sur 200 pas (script) : @Q02_mc:4@, erreur standard @Q02_se:4@.""",
    },
    # ------------------------------------------------- Réseaux
    {
        "id": "ML4-03", "section": "mla", "theme": "Initialisation, propagation de la variance avec ReLU", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Un réseau sans biais a toutes ses couches de largeur \(n\). Le vecteur \(\mathbf{a}^{(1)} \in \mathbb{R}^n\) a des coordonnées indépendantes de loi normale centrée réduite. Pour \(l = 1, \dots, 10\) : \(\mathbf{a}^{(l+1)} = \mathbf{W}^{(l+1)}\,\mathrm{ReLU}(\mathbf{a}^{(l)})\), où ReLU s'applique coordonnée par coordonnée et où les matrices aléatoires \(\mathbf{W}^{(2)}, \dots, \mathbf{W}^{(11)}\), de taille \(n \times n\), ont des coefficients indépendants de loi \(\mathcal{N}(0, 1/n)\) (initialisation de Glorot, \(2/(n + n)\)), indépendants de \(\mathbf{a}^{(1)}\).

Que vaut \(\mathrm{Var}(a^{(11)}_1)\), la variance de la première coordonnée de \(\mathbf{a}^{(11)}\) ?""",
        "options": [
            {"id": "a", "texte": r"\(\tfrac{@Q03_exact_num@}{@Q03_exact_den@}\)",
             "explication": r"Oui : chaque couche multiplie le moment d'ordre 2 par \(n \times \tfrac1n \times \tfrac12\), et il y a 10 couches entre \(\mathbf{a}^{(1)}\) et \(\mathbf{a}^{(11)}\)."},
            {"id": "b", "texte": "1",
             "explication": r"La règle \(1/n\) conserve la variance d'une couche **linéaire**. ReLU met à zéro la moitié du moment d'ordre 2 (leçon, partie 2.2, I2) : il faudrait \(2/n\) (He) pour le conserver."},
            {"id": "c", "texte": r"\(\tfrac{@Q03_d_onze_num@}{@Q03_d_onze_den@}\)",
             "explication": r"C'est \(2^{-11}\) : une couche de trop. De \(\mathbf{a}^{(1)}\) à \(\mathbf{a}^{(11)}\), il y a 10 multiplications par une matrice de poids, donc 10 divisions par 2."},
            {"id": "d", "texte": r"\(\big(\tfrac12 - \tfrac{1}{2\pi}\big)^{10} \approx @Q03_d_var_relu:7m@\)",
             "explication": r"C'est la **variance** de \(\mathrm{ReLU}(A)\) pour \(A\) normale centrée réduite, utilisée à la place du moment d'ordre 2. Or \(\mathbb{E}[a_i^2] = s^2\sum_j \mathbb{E}[h_j^2]\) fait intervenir \(\mathbb{E}[h_j^2] = \tfrac12\mathbb{E}[a_j^2]\), pas \(\mathrm{Var}(h_j)\) : les \(h_j\) ne sont pas centrés, et c'est le moment d'ordre 2 qui compte. De plus, après la première couche, les \(a\) ne sont plus gaussiens."},
        ],
        "solution": r"""Réponse : \(2^{-10} = \tfrac{@Q03_exact_num@}{@Q03_exact_den@}\).

Étape 1 (leçon, partie 2.2, I1) : \(a^{(l+1)}_i = \sum_j W_{ij}h_j\) avec \(\mathbf{h} = \mathrm{ReLU}(\mathbf{a}^{(l)})\) indépendant de \(\mathbf{W}^{(l+1)}\) ; les termes croisés ont une espérance nulle, donc \(\mathbb{E}[(a^{(l+1)}_i)^2] = \tfrac1n\sum_{j=1}^{n}\mathbb{E}[h_j^2]\). Par symétrie des poids, \(a^{(l+1)}_i\) est centrée : son moment d'ordre 2 est sa variance.

Étape 2 (I2) : \(\mathbf{a}^{(1)}\) est gaussienne centrée, et chaque \(\mathbf{a}^{(l)}\) suivant a une loi symétrique ; donc \(\mathbb{E}[h_j^2] = \tfrac12\mathbb{E}[(a^{(l)}_j)^2]\), et \(\mathbb{E}[(a^{(l+1)}_i)^2] = \tfrac12\,\mathbb{E}[(a^{(l)}_j)^2]\) (toutes les coordonnées d'une couche ont la même loi).

Étape 3 : 10 couches, \(\mathrm{Var}(a^{(11)}_1) = 2^{-10} \times 1 = \tfrac{@Q03_exact_num@}{@Q03_exact_den@}\). Le résultat ne dépend pas de \(n\).

Simulation (script, réseaux de largeur 10, un réseau neuf par tirage) : @Q03_mc:6@, erreur standard @Q03_se:6@.""",
    },
    {
        "id": "ML4-04", "section": "mla", "theme": "Dropout, remise à l'échelle et variance", "difficulte": 3, "temps_cible_s": 75,
        "enonce": r"""Pendant l'entraînement, une couche de dropout reçoit une activation de valeur \(h = 2\). Avec probabilité \(p = 0{,}2\), elle renvoie 0 ; sinon, elle renvoie \(h/(1 - p)\). On note \(\tilde H\) la valeur renvoyée.

Que vaut \(\mathrm{Var}(\tilde H)\) ?""",
        "options": [
            {"id": "a", "texte": "@Q04_exact@", "explication": r"Oui : \(h^2\,p/(1 - p) = 4 \times 0{,}2/0{,}8\)."},
            {"id": "b", "texte": "@Q04_d_sans_echelle:2@", "explication": r"C'est \(h^2 p(1 - p)\), la variance de \(D\,h\) **sans** la division par \(1 - p\). Diviser une variable par \(0{,}8\) divise sa variance par \(0{,}8^2\)."},
            {"id": "c", "texte": "@Q04_d_moment2@", "explication": r"C'est \(\mathbb{E}[\tilde H^2] = h^2/(1 - p)\), le moment d'ordre 2. Il faut retrancher le carré de l'espérance, \(h^2 = 4\)."},
            {"id": "d", "texte": "@Q04_d_inverse@", "explication": r"C'est \(h^2(1 - p)/p\) : \(p\) a été pris pour la probabilité de **garder** l'activation. Ici \(p = 0{,}2\) est la probabilité de l'éteindre."},
        ],
        "solution": r"""Réponse : @Q04_exact@.

Étape 1 : \(\tilde H\) vaut 0 avec probabilité 0,2 et \(2/0{,}8 = @Q04_haut:1m@\) avec probabilité 0,8.

Étape 2 : \(\mathbb{E}[\tilde H] = 0{,}8 \times @Q04_haut:1m@ = 2 = h\) : la remise à l'échelle conserve l'espérance (leçon, partie 2.5, D1).

Étape 3 : \(\mathbb{E}[\tilde H^2] = 0{,}8 \times @Q04_haut:1m@^2 = @Q04_d_moment2@\), donc \(\mathrm{Var}(\tilde H) = @Q04_d_moment2@ - 4 = @Q04_exact@\), comme \(h^2 p/(1 - p)\).

Simulation (script, 400000 tirages) : @Q04_mc:3@, erreur standard @Q04_se:4@.""",
    },
    # ------------------------------------------------- Architectures
    {
        "id": "ML4-05", "section": "mla", "theme": "Convolutions causales dilatées, champ réceptif", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""On empile 5 couches de convolution causales à une dimension, de pas 1. Une couche de noyau de taille \(k\) et de dilatation \(d\) calcule \(y_t = \sum_{i=0}^{k-1} w_i\,u_{t - di}\) à partir de la suite \(u\) qu'elle reçoit (complétée par des zéros avant son début). Les quatre premières couches ont un noyau de taille 3 et des dilatations 1, 2, 4 et 8 ; la cinquième a un noyau de taille 2 et une dilatation 16. Les poids sont quelconques, tous non nuls.

Loin du début de la série, de combien d'entrées \(x_s\) distinctes la sortie de la cinquième couche au temps \(t\) peut-elle dépendre ?""",
        "options": [
            {"id": "a", "texte": "@Q05_exact@", "explication": r"Oui : \(1 + 2(1 + 2 + 4 + 8) + 1 \times 16\), et il n'y a aucun trou, car chaque dilatation est au plus le champ réceptif des couches précédentes."},
            {"id": "b", "texte": "@Q05_d_sans_un@", "explication": r"Il manque le 1 de la formule : la somme \(\sum_l d_l(k_l - 1)\) est le plus grand **retard**, et les retards vont de 0 à cette valeur, ce qui fait une entrée de plus."},
            {"id": "c", "texte": "@Q05_d_k@", "explication": r"C'est \(\sum_l d_l k_l\) : chaque couche n'ajoute que \(d_l(k_l - 1)\) au plus grand retard, car son premier coefficient lit le retard 0."},
            {"id": "d", "texte": "@Q05_d_sans_dil@", "explication": r"C'est le champ réceptif sans dilatation, \(1 + \sum_l (k_l - 1)\). Une dilatation \(d\) écarte les coefficients du noyau de \(d\) positions."},
        ],
        "solution": r"""Réponse : @Q05_exact@ entrées.

Étape 1 (leçon, partie 3.1, C1) : les retards possibles sont les sommes \(\sum_l d_l i_l\) avec \(0 \le i_l \le k_l - 1\) ; le plus grand vaut \(2 \times 1 + 2 \times 2 + 2 \times 4 + 2 \times 8 + 1 \times 16 = @Q05_d_sans_un@\), donc la fenêtre compte \(R = @Q05_exact@\) positions.

Étape 2 (C2), absence de trou : après les couches 1 à 4, le champ vaut \(R_4 = @Q05_avant@\), sans trou car \(1 \le 1\), \(2 \le 3\), \(4 \le 7\), \(8 \le 15\). La cinquième couche réunit l'intervalle des retards \(0\) à \(@Q05_avant@ - 1\) et son translaté de 16 ; comme \(16 \le @Q05_avant@\), les deux se chevauchent et la réunion est l'intervalle de 0 à \(@Q05_d_sans_un@\).

Étape 3 : toutes les @Q05_exact@ entrées de la fenêtre interviennent. Le script le vérifie par énumération des retards et par convolution d'une impulsion.""",
    },
    {
        "id": "ML4-06", "section": "mla", "theme": "LSTM, nombre de paramètres", "difficulte": 3, "temps_cible_s": 75,
        "enonce": r"""Une couche LSTM reçoit des entrées \(\mathbf{x}_t \in \mathbb{R}^{32}\) et a un état caché \(\mathbf{h}_t \in \mathbb{R}^{64}\) et une cellule \(\mathbf{c}_t \in \mathbb{R}^{64}\). Avec \(\mathbf{z}_t = (\mathbf{h}_{t-1}, \mathbf{x}_t)\), elle calcule \(\mathbf{f}_t = \mathrm{sig}(\mathbf{W}_f\mathbf{z}_t + \mathbf{b}_f)\), \(\mathbf{i}_t = \mathrm{sig}(\mathbf{W}_i\mathbf{z}_t + \mathbf{b}_i)\), \(\mathbf{o}_t = \mathrm{sig}(\mathbf{W}_o\mathbf{z}_t + \mathbf{b}_o)\), \(\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c\mathbf{z}_t + \mathbf{b}_c)\), \(\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t\) et \(\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)\), où \(\mathrm{sig}\) est la sigmoïde et \(\odot\) le produit coordonnée par coordonnée.

Combien de paramètres apprenables a cette couche (coefficients des quatre matrices et des quatre vecteurs de biais) ?""",
        "options": [
            {"id": "a", "texte": "@Q06_exact@", "explication": r"Oui : \(4 \times (64 \times 96 + 64)\), chaque matrice étant de taille \(64 \times (64 + 32)\)."},
            {"id": "b", "texte": "@Q06_d_sans_biais@", "explication": r"Il manque les 4 vecteurs de biais de 64 coefficients chacun."},
            {"id": "c", "texte": "@Q06_d_rnn@", "explication": r"C'est le nombre de paramètres d'un seul bloc, celui d'un RNN simple de même taille. Le LSTM en a quatre : trois portes et le candidat."},
            {"id": "d", "texte": "@Q06_d_trois@", "explication": r"Trois blocs seulement : le candidat \(\tilde{\mathbf{c}}_t\) a lui aussi sa matrice et son biais. (Trois blocs, c'est le compte d'un GRU, une autre architecture.)"},
        ],
        "solution": r"""Réponse : @Q06_exact@.

Étape 1 : \(\mathbf{z}_t\) a \(64 + 32 = 96\) coordonnées, donc chaque matrice \(\mathbf{W}_\bullet\) a \(64 \times 96\) coefficients, et chaque biais 64.

Étape 2 : un bloc compte \(64 \times 96 + 64 = @Q06_bloc@\) paramètres ; il y a 4 blocs (\(\mathbf{f}\), \(\mathbf{i}\), \(\mathbf{o}\), \(\tilde{\mathbf{c}}\)) : @Q06_exact@ (leçon, partie 3.3).

Piège : `nn.LSTM` de PyTorch utilise deux vecteurs de biais par bloc ; avec cette convention, on compterait @Q06_deux_biais@ paramètres. L'énoncé précise un biais par bloc.""",
    },
    # ------------------------------------------------- Noyaux
    {
        "id": "ML4-07", "section": "mla", "theme": "Processus gaussien et ridge à noyau, moyenne a posteriori", "difficulte": 5, "temps_cible_s": 150,
        "enonce": r"""\(F\) est un processus gaussien centré sur \(\mathbb{R}\) de covariance \(k(x, x') = \exp\big(-(x - x')^2/2\big)\) : pour tous points \(x_1, \dots, x_m\), le vecteur \((F(x_1), \dots, F(x_m))\) est gaussien centré de covariance \(\big(k(x_i, x_j)\big)_{i,j}\). On observe \(Y_1 = F(0) + E_1\) et \(Y_2 = F(1) + E_2\), où \(E_1\) et \(E_2\) sont indépendantes, de loi normale centrée de variance 0,5, et indépendantes de \(F\). Les valeurs observées sont \(y_1 = 1\) et \(y_2 = 0\).

Que vaut \(\mathbb{E}[F(0) \mid Y_1 = 1, Y_2 = 0]\), arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q07_exact:3@", "explication": r"Oui : \(\mathbf{s}^\top(\mathbf{K} + 0{,}5\,\mathbf{I})^{-1}\mathbf{y} = (1{,}5 - e^{-1})/(2{,}25 - e^{-1})\)."},
            {"id": "b", "texte": "1", "explication": r"C'est l'interpolation, obtenue sans bruit (\(\sigma^2 = 0\)) : la moyenne a posteriori passerait par l'observation. Avec un bruit de variance 0,5, elle est tirée vers 0, la moyenne a priori."},
            {"id": "c", "texte": "@Q07_d_sigma:3@", "explication": r"C'est le calcul avec l'**écart-type** du bruit, \(\sqrt{0{,}5}\), ajouté à la diagonale. C'est la variance \(\sigma^2 = 0{,}5\) qui joue le rôle de \(\lambda\)."},
            {"id": "d", "texte": "@Q07_d_seul:3@", "explication": r"C'est \(1/(1 + 0{,}5)\), le calcul qui ignore la seconde observation. \(Y_2\) est corrélée à \(F(0)\) (covariance \(e^{-1/2}\)) : l'observation \(y_2 = 0\) abaisse encore l'estimation."},
        ],
        "solution": r"""Réponse : \(\dfrac{1{,}5 - e^{-1}}{2{,}25 - e^{-1}} \approx @Q07_exact:4m@\), soit @Q07_exact:3@.

Notations : \(a = k(0, 1) = e^{-1/2} \approx @Q07_a:4m@\) ; \(\mathbf{K}\), la matrice de Gram des points 0 et 1, a ses coefficients diagonaux égaux à 1 et ses deux coefficients hors diagonale égaux à \(a\) ; \(\mathbf{s} = (k(0, 0), k(1, 0)) = (1, a)\) ; \(\mathbf{y} = (1, 0)\).

Étape 1 (leçon, partie 4.4, P1) : la moyenne a posteriori est \(\mathbf{s}^\top(\mathbf{K} + \sigma^2\mathbf{I})^{-1}\mathbf{y}\), la prédiction de la ridge à noyau avec \(\lambda = \sigma^2 = 0{,}5\).

Étape 2 : \(\mathbf{K} + 0{,}5\,\mathbf{I}\) a pour coefficients diagonaux 1,5 et hors diagonale \(a\) ; son déterminant vaut \(2{,}25 - a^2\), et son inverse a pour coefficients diagonaux \(1{,}5/(2{,}25 - a^2)\) et hors diagonale \(-a/(2{,}25 - a^2)\) (inverse d'une matrice \(2 \times 2\) : on échange les coefficients diagonaux, on change le signe des autres, on divise par le déterminant). Donc \((\mathbf{K} + 0{,}5\,\mathbf{I})^{-1}\mathbf{y}\), sa première colonne, vaut \((1{,}5, -a)/(2{,}25 - a^2)\).

Étape 3 : produit scalaire avec \(\mathbf{s} = (1, a)\) : \(\dfrac{1{,}5 - a^2}{2{,}25 - a^2}\), avec \(a^2 = e^{-1} \approx @Q07_a2:4m@\) : \(@Q07_num:4m@/@Q07_den:4m@ \approx @Q07_exact:4m@\).

Pour aller plus loin : la variance a posteriori vaut \(1 - \mathbf{s}^\top(\mathbf{K} + 0{,}5\,\mathbf{I})^{-1}\mathbf{s} \approx @Q07_var:4m@\) (simulation du modèle par le script : @Q07_var_mc:4@).""",
    },
    # ------------------------------------------------- EM
    {
        "id": "ML4-08", "section": "mla", "theme": "EM, mélange de deux gaussiennes, une itération", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Mélange de deux gaussiennes sur \(\mathbb{R}\) : une étiquette cachée \(Z \in \{1, 2\}\) avec \(\mathbb{P}(Z = c) = \omega_c\), puis \(X\) de loi \(\mathcal{N}(\mu_c, \sigma_c^2)\) sachant \(Z = c\). On observe trois valeurs, \(0\), \(1\) et \(5\), indépendantes. On part de \(\omega_1 = \omega_2 = 1/2\), \(\mu_1 = 0\), \(\mu_2 = 4\), \(\sigma_1^2 = \sigma_2^2 = 1\), et l'on fait une itération de EM : étape E (responsabilités \(r_{i1} = \mathbb{P}(Z_i = 1 \mid X_i = x_i)\) sous les paramètres de départ), puis étape M.

Que vaut le nouveau \(\mu_1\), arrondi à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q08_exact:3@", "explication": r"Oui : \(\mu_1 = \sum_i r_{i1}x_i/\sum_i r_{i1}\), moyenne pondérée par les responsabilités, à peu près @Q08_r@ pour les trois observations."},
            {"id": "b", "texte": "@Q08_d_dur:3@", "explication": r"C'est l'affectation dure (comme k-means) : 0 et 1 entièrement à la composante 1. EM pondère : l'observation 1 n'appartient à la composante 1 qu'avec probabilité \(@Q08_r1:3m@\), et 5 avec une probabilité infime mais non nulle."},
            {"id": "c", "texte": "@Q08_d_sur_n:3@", "explication": r"La somme pondérée a été divisée par le nombre d'observations, 3, au lieu de la somme des responsabilités, \(n_1 \approx @Q08_n1:3m@\)."},
            {"id": "d", "texte": "@Q08_d_moyenne:3@", "explication": r"C'est la moyenne simple des trois observations : elle ignore les responsabilités. C'est l'estimation d'une seule gaussienne, pas d'une composante du mélange."},
        ],
        "solution": r"""Réponse : \(\mu_1 \approx @Q08_exact:3m@\).

Étape 1, étape E : à poids et variances égaux, \(r_{i1} = \dfrac{\mathcal{N}(x_i; 0, 1)}{\mathcal{N}(x_i; 0, 1) + \mathcal{N}(x_i; 4, 1)} = \dfrac{1}{1 + \exp\big((x_i^2 - (x_i - 4)^2)/2\big)}\). Pour \(x = 0\) : \(1/(1 + e^{-8})\) ; pour \(x = 1\) : \(1/(1 + e^{-4}) \approx @Q08_r1:3m@\) ; pour \(x = 5\) : \(1/(1 + e^{12}) \approx @Q08_r3:7m@\). Arrondies : @Q08_r@.

Étape 2, étape M (leçon, partie 5, E1) : \(n_1 = \sum_i r_{i1} \approx @Q08_n1:4m@\) et \(\mu_1 = (0 \times r_{11} + 1 \times r_{21} + 5 \times r_{31})/n_1 \approx @Q08_exact:4m@\).

Contrôle (E2) : la log-vraisemblance passe de \(@Q08_ell0:3m@\) à \(@Q08_ell1:3m@\) après l'itération complète (script). Le nouveau \(\mu_2\) vaut \(@Q08_mu2:3m@\).""",
    },
]
