"""Séance de ML 5 : leçon (réseaux convolutifs, mise en place d'un entraînement, biais et variance, apprentissage par
renforcement, réflexes pour les affirmations à cocher), puis 14 questions à choix multiple, dont 8 à plusieurs réponses.

La leçon L-ML-5 (outils/lecons_ml_5.py) comble ce que les séances de ML 1 à 4, d'algèbre linéaire et quantitative ne
traitent pas ou seulement en partie : formes des tenseurs, époques et lots, pertes de segmentation, Dice et IoU,
stratification et pondération, équations de Bellman, Q-learning, gradient de politique, bandits, elastic net.
Toutes les valeurs : verif/seance_ml_5.py. Dans une question à réponse unique, l'option "a" est la bonne ; une question
à plusieurs réponses déclare "correctes" ; une question sans aucun nombre déclare "conceptuelle".
"""
import lecons_ml_5
import questions_diagnostic as diag

ID = "seance-ml-5"
TITRE = "Séance de ML 5 : réseaux convolutifs, entraînement, apprentissage par renforcement et réflexes de choix multiple"
VERIF = "verif/seance_ml_5.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [{"id": "lecon", "nom": "Leçon", "budget_min": 65},
            {"id": "mlf", "nom": "Machine Learning fondamental", "budget_min": 13},
            {"id": "mla", "nom": "Machine Learning avancé", "budget_min": 14}]
PRESENTATION = (
    "D'abord une leçon d'environ 65 minutes en cinq parties indépendantes (30 minutes en ne lisant que les énoncés, les "
    "exemples et les pièges) : arithmétique des réseaux convolutifs (formes de sortie, paramètres, pooling, "
    "concaténation et addition, champ réceptif, convolution transposée) ; mise en place d'un entraînement (découpages, "
    "époques et lots, pas d'apprentissage, choix de la perte, Dice et IoU, classes déséquilibrées) ; biais et variance "
    "en image de cible ; apprentissage par renforcement (équations de Bellman, itération sur la valeur, Q-learning, "
    "gradient de politique, exploration) ; réflexes pour les affirmations à cocher sur les arbres, les importances de "
    "variables, l'ACP et les pénalités L1, L2 et elastic net. Cinq points de contrôle sont corrigés immédiatement, et "
    "la sortie de chaque expérience est réelle. Ensuite 14 questions à choix multiple, dont 8 à plusieurs réponses et "
    "6 de difficulté 4 ou 5.\n\n"
    "Objectif de vitesse : chaque question se résout dans son temps cible avec le tableau « Réflexes en 30 secondes » "
    "de la leçon. Règle d'abandon : au-delà du double du temps cible affiché, choisis « Je ne sais pas » et passe à la "
    "question suivante."
)
AVERTISSEMENTS = [
    "Questions à plusieurs réponses : coche toutes les affirmations justes et seulement elles ; leur nombre varie d'une "
    "question à l'autre, et il y en a toujours au moins une. La notation est tout ou "
    "rien : une affirmation fausse cochée, ou une juste oubliée, et la question est perdue. Juge chaque affirmation "
    "séparément, en cherchant un contre-exemple à celles qui disent « toujours », « jamais » ou « garantit ».",
    "Ici le retour en arrière est permis et il n'y a pas de pénalité pour une réponse fausse.",
]

LECONS = [lecons_ml_5.LECON_ML_5]
CODES = []

QUESTIONS = [
    # ------------------------------------------------- Réseaux convolutifs
    {
        "id": "ML5-01", "section": "mla", "theme": "Réseau convolutif : forme après deux branches, concaténation et pooling", "difficulte": 2, "temps_cible_s": 90,
        "enonce": r"""Une image de forme \(128 \times 128 \times 3\) (hauteur, largeur, canaux) entre dans deux branches, qui lisent toutes deux l'image :

- branche A : convolution \(5 \times 5\) à 12 filtres, pas 2, remplissage de 2 zéros de chaque côté ;
- branche B : max-pooling \(3 \times 3\) de pas 2 avec un remplissage de 1 de chaque côté, suivi d'une convolution \(1 \times 1\) à 20 filtres, de pas 1, sans remplissage.

Les deux sorties sont concaténées selon l'axe des canaux, puis on applique un max-pooling \(2 \times 2\) de pas 2 sans remplissage.

Quelle est la forme du tenseur final ?""",
        "options": [
            {"id": "a", "texte": r"\(@Q01_sortie@ \times @Q01_sortie@ \times @Q01_canaux@\)",
             "explication": r"Oui : chaque branche sort en \(@Q01_brancheA@ \times @Q01_brancheA@\), avec 12 et 20 canaux ; la concaténation additionne les canaux, et le dernier pooling divise la taille spatiale par 2 sans toucher aux canaux."},
            {"id": "b", "texte": r"\(@Q01_brancheA@ \times @Q01_brancheA@ \times @Q01_canaux@\)",
             "explication": r"C'est la forme après la concaténation : le dernier max-pooling \(2 \times 2\) de pas 2 a été oublié. Il ramène la taille spatiale à \(\lfloor @Q01_brancheA@/2 \rfloor\)."},
            {"id": "c", "texte": r"\(@Q01_sortie@ \times @Q01_sortie@ \times @Q01_d_produit@\)",
             "explication": r"Les nombres de canaux ont été multipliés (\(12 \times 20\)). Concaténer met les canaux bout à bout : \(12 + 20\)."},
            {"id": "d", "texte": r"\(@Q01_sortie@ \times @Q01_sortie@ \times @Q01_d_entree@\)",
             "explication": r"Les 3 canaux de l'image ont été ajoutés aux 12 et aux 20. Or la sortie d'une convolution a autant de canaux que de filtres, quels que soient les canaux d'entrée, et l'image elle-même n'est pas concaténée."},
        ],
        "solution": r"""Réponse : \(@Q01_sortie@ \times @Q01_sortie@ \times @Q01_canaux@\).

Formule par axe (leçon, partie 1.1, C1) : \(\lfloor (n + 2p - k)/s \rfloor + 1\) pour une taille \(n\), un noyau \(k\), un remplissage \(p\), un pas \(s\) (dilatation 1).

Étape 1, branche A : \(\lfloor (128 + 4 - 5)/2 \rfloor + 1 = 63 + 1 = @Q01_brancheA@\) ; 12 filtres, donc \(@Q01_brancheA@ \times @Q01_brancheA@ \times 12\).

Étape 2, branche B : pooling \(\lfloor (128 + 2 - 3)/2 \rfloor + 1 = @Q01_brancheB@\), canaux inchangés (3) ; la convolution \(1 \times 1\) garde la taille et sort 20 canaux : \(@Q01_brancheB@ \times @Q01_brancheB@ \times 20\).

Étape 3 : mêmes dimensions spatiales, la concaténation est possible : \(@Q01_brancheA@ \times @Q01_brancheA@ \times @Q01_canaux@\) (C3).

Étape 4 : max-pooling \(2 \times 2\) de pas 2 : \(\lfloor @Q01_brancheA@/2 \rfloor = @Q01_sortie@\), canaux inchangés.

Le script refait l'enchaînement par une vraie convolution numpy sur une image huit fois plus petite.""",
    },
    {
        "id": "ML5-02", "section": "mla", "theme": "Réseau convolutif : quand peut-on concaténer ou additionner", "difficulte": 4, "temps_cible_s": 120,
        "correctes": ["a", "c"],
        "enonce": r"""Une carte d'activations de forme \(64 \times 64 \times 16\) (hauteur, largeur, canaux) entre dans plusieurs couches, chacune lisant directement cette carte. Toutes les convolutions ont une dilatation de 1 ; « remplissage \(p\) » signifie \(p\) zéros ajoutés de chaque côté. Aucun rognage ni complément n'est appliqué aux sorties.

Lesquelles de ces affirmations sont justes ?""",
        "options": [
            {"id": "a", "texte": r"Les sorties d'une convolution \(3 \times 3\) à 32 filtres (pas 1, remplissage 1) et d'une convolution \(5 \times 5\) à 8 filtres (pas 1, remplissage 2) peuvent être concaténées selon les canaux, et le résultat a la forme \(@Q02_a1@ \times @Q02_a1@ \times @Q02_a_canaux@\).",
             "explication": r"Vrai : \(64 + 2 - 3 + 1 = @Q02_a1@\) et \(64 + 4 - 5 + 1 = @Q02_a2@\), mêmes dimensions spatiales ; les canaux s'additionnent, \(32 + 8\)."},
            {"id": "b", "texte": r"Les deux sorties de l'affirmation précédente peuvent aussi être additionnées terme à terme, comme dans une connexion résiduelle.",
             "explication": r"Faux : l'addition terme à terme exige la même forme, canaux compris ; ici 32 canaux d'un côté et 8 de l'autre. C'est la différence entre concaténer et additionner."},
            {"id": "c", "texte": r"Les sorties d'une convolution \(3 \times 3\) à 32 filtres (pas 2, remplissage 1) et d'un max-pooling \(2 \times 2\) de pas 2 sans remplissage peuvent être concaténées selon les canaux, et le résultat a la forme \(@Q02_c1@ \times @Q02_c1@ \times @Q02_c_canaux@\).",
             "explication": r"Vrai : \(\lfloor (64 + 2 - 3)/2 \rfloor + 1 = @Q02_c1@\) pour la convolution, \(\lfloor 64/2 \rfloor = @Q02_c2@\) pour le pooling, qui garde ses 16 canaux : \(32 + 16\) canaux."},
            {"id": "d", "texte": r"Les sorties d'une convolution \(3 \times 3\) à 16 filtres et d'une convolution \(5 \times 5\) à 16 filtres, toutes deux de pas 1 et sans remplissage, peuvent être concaténées selon les canaux, puisqu'elles ont le même nombre de filtres.",
             "explication": r"Faux : sans remplissage, les tailles spatiales valent \(@Q02_d1@\) et \(@Q02_d2@\). La concaténation selon les canaux demande les mêmes dimensions spatiales ; le nombre de filtres, lui, est libre."},
        ],
        "solution": r"""Réponse : les affirmations sur les convolutions remplies \(3 \times 3\) et \(5 \times 5\) (forme \(@Q02_a1@ \times @Q02_a1@ \times @Q02_a_canaux@\)) et sur la convolution de pas 2 avec le max-pooling (forme \(@Q02_c1@ \times @Q02_c1@ \times @Q02_c_canaux@\)) sont justes ; les deux autres sont fausses.

Règle (leçon, partie 1.3, C3) : concaténer selon les canaux exige les mêmes dimensions spatiales, et les canaux s'additionnent ; additionner exige la même forme complète.

Étape 1 : tailles par (C1). Remplissage 1 et noyau 3, pas 1 : \(@Q02_a1@\). Remplissage 2 et noyau 5, pas 1 : \(@Q02_a2@\). Noyau 3, pas 2, remplissage 1 : \(@Q02_c1@\). Pooling 2 de pas 2 : \(@Q02_c2@\). Sans remplissage, noyaux 3 et 5 : \(@Q02_d1@\) et \(@Q02_d2@\).

Étape 2 : les deux premières paires ont des dimensions spatiales égales, donc se concatènent ; la dernière non.

Étape 3 : l'addition des sorties à 32 et 8 canaux est impossible, puisque \(32 \ne 8\).""",
    },
    # ------------------------------------------------- Mise en place d'un entraînement
    {
        "id": "ML5-03", "section": "mlf", "theme": "Entraînement : époques, lots et nombre de mises à jour", "difficulte": 2, "temps_cible_s": 75,
        "enonce": r"""Un jeu de 2400 paires (image, masque) est découpé au hasard en 75 % pour l'apprentissage, 15 % pour la validation et 10 % pour le test. On entraîne pendant 25 époques avec des lots de 48 paires ; à chaque époque, toutes les paires d'apprentissage sont utilisées une fois, et le dernier lot, s'il est incomplet, est conservé.

Combien de mises à jour des poids (itérations) l'entraînement compte-t-il au total ?""",
        "options": [
            {"id": "a", "texte": "@Q03_exact@",
             "explication": r"Oui : \(@Q03_app@\) paires d'apprentissage, \(\lceil @Q03_app@/48 \rceil = @Q03_pas@\) itérations par époque (\(@Q03_quotient@\) lots complets et un lot de \(@Q03_reste@\)), fois 25 époques."},
            {"id": "b", "texte": "@Q03_d_plancher@",
             "explication": r"C'est \(25 \times \lfloor @Q03_app@/48 \rfloor\) : le dernier lot, incomplet, a été écarté, alors que l'énoncé dit qu'il est conservé."},
            {"id": "c", "texte": "@Q03_d_total@",
             "explication": r"C'est \(25 \times 2400/48\) : la taille totale du jeu a été utilisée. Seules les paires d'apprentissage servent aux mises à jour ; la validation et le test ne modifient pas les poids."},
            {"id": "d", "texte": "@Q03_d_epoque@",
             "explication": r"C'est le nombre d'itérations d'**une** époque. La question porte sur les 25 époques."},
        ],
        "solution": r"""Réponse : \(@Q03_exact@\) mises à jour.

Étape 1 : tailles des ensembles : \(@Q03_app@\) paires d'apprentissage, \(@Q03_val@\) de validation, \(@Q03_test@\) de test (leur somme vaut bien 2400).

Étape 2 : itérations par époque (leçon, partie 2.2, E1) : \(@Q03_app@ = @Q03_quotient@ \times 48 + @Q03_reste@\), donc \(@Q03_quotient@\) lots complets et un lot incomplet : \(\lceil @Q03_app@/48 \rceil = @Q03_pas@\).

Étape 3 : \(25 \times @Q03_pas@ = @Q03_exact@\).""",
    },
    {
        "id": "ML5-04", "section": "mla", "theme": "Segmentation à deux classes : sortie, perte et métrique", "difficulte": 3, "temps_cible_s": 100,
        "correctes": ["a", "b", "d"],
        "enonce": r"""On entraîne un réseau de segmentation à deux classes (objet, fond) sur des paires (image, masque). Dans les masques, 4 % des pixels sont de la classe « objet ». L'**exactitude par pixel** est la proportion de pixels bien classés. Le **coefficient de Dice** de la classe « objet » est le double du nombre de pixels « objet » à la fois dans la prédiction et dans le vrai masque, divisé par la somme des nombres de pixels « objet » de la prédiction et du vrai masque.

Lesquelles de ces affirmations sont justes ?""",
        "options": [
            {"id": "a", "texte": "Une carte de sortie à un canal suivie d'une sigmoïde, entraînée avec l'entropie croisée binaire moyennée sur les pixels, est un choix cohérent.",
             "explication": r"Vrai : chaque pixel est une classification binaire, modélisée par une loi de Bernoulli de paramètre \(\mathrm{sig}(z)\), dont l'opposé de la log-vraisemblance est l'entropie croisée binaire."},
            {"id": "b", "texte": "Une carte de sortie à deux canaux suivie d'un softmax, entraînée avec l'entropie croisée catégorielle par pixel, décrit la même famille de modèles que le choix précédent.",
             "explication": r"Vrai : avec deux logits \(z_0\) et \(z_1\), le softmax donne \(\mathrm{sig}(z_1 - z_0)\) pour la classe 1 ; les deux écritures donnent les mêmes probabilités et la même perte."},
            {"id": "c", "texte": r"Un modèle dont l'exactitude par pixel atteint \(@Q04_fond:2m@\) segmente nécessairement bien l'objet.",
             "explication": r"Faux : le masque « tout en fond » classe bien les 96 % de pixels de fond et obtient exactement cette exactitude, avec un rappel et un Dice nuls sur l'objet."},
            {"id": "d", "texte": "Le coefficient de Dice de la classe « objet » ne dépend pas du nombre de pixels de fond correctement classés.",
             "explication": r"Vrai : il vaut \(2\,\mathrm{VP}/(2\,\mathrm{VP} + \mathrm{FP} + \mathrm{FN})\) ; les vrais négatifs, c'est-à-dire les pixels de fond bien classés, n'y figurent pas. C'est ce qui le rend utile quand l'objet est rare."},
        ],
        "solution": r"""Réponse : trois affirmations sont justes : les deux sur la sortie et la perte (un canal avec sigmoïde et entropie croisée binaire ; deux canaux avec softmax et entropie croisée catégorielle, équivalents) et celle sur le coefficient de Dice, qui ignore les pixels de fond bien classés. Celle sur l'exactitude par pixel est fausse.

Étape 1 : la segmentation est une classification par pixel ; la perte est l'opposé de la log-vraisemblance du modèle de sortie (leçon, partie 2.4, V1).

Étape 2 : le softmax à deux classes vaut \(e^{z_1}/(e^{z_0} + e^{z_1}) = \mathrm{sig}(z_1 - z_0)\) (V3) : même famille de modèles.

Étape 3 : un masque entièrement « fond » a une exactitude de \(1 - 0{,}04 = @Q04_fond:2m@\) : l'exactitude par pixel ne suffit pas quand l'objet est rare.

Étape 4 : avec VP, FP et FN les nombres de vrais positifs, faux positifs et faux négatifs de la classe « objet », le Dice vaut \(2\,\mathrm{VP}/(2\,\mathrm{VP} + \mathrm{FP} + \mathrm{FN})\) : les vrais négatifs n'y entrent pas (partie 2.5).""",
    },
    {
        "id": "ML5-05", "section": "mla", "theme": "Segmentation : coefficient de Dice et IoU", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""Sur une image, un modèle de segmentation prédit « objet » pour 50 pixels. Le vrai masque contient 70 pixels « objet », et 40 pixels sont « objet » à la fois dans la prédiction et dans le vrai masque.

Que valent le coefficient de Dice et l'IoU (intersection sur union) de cette prédiction ?""",
        "options": [
            {"id": "a", "texte": r"Dice \(= \tfrac{@Q05_dice_num@}{@Q05_dice_den@}\) et IoU \(= \tfrac{@Q05_iou_num@}{@Q05_iou_den@}\)",
             "explication": r"Oui : Dice \(= 2 \times 40/(50 + 70)\) et IoU \(= 40/(50 + 70 - 40)\) ; on retrouve Dice \(= 2\,\mathrm{IoU}/(1 + \mathrm{IoU})\)."},
            {"id": "b", "texte": r"Dice \(= \tfrac{@Q05_iou_num@}{@Q05_iou_den@}\) et IoU \(= \tfrac{@Q05_dice_num@}{@Q05_dice_den@}\)",
             "explication": r"Les deux mesures ont été échangées. Un contrôle rapide : le Dice est toujours supérieur ou égal à l'IoU, avec égalité seulement en 0 et en 1."},
            {"id": "c", "texte": r"Dice \(= \tfrac{@Q05_d_sans2_num@}{@Q05_d_sans2_den@}\) et IoU \(= \tfrac{@Q05_iou_num@}{@Q05_iou_den@}\)",
             "explication": r"Le facteur 2 du Dice a été oublié : \(40/(50 + 70)\). Sans lui, une prédiction parfaite aurait un Dice de \(\tfrac12\) au lieu de 1."},
            {"id": "d", "texte": r"Dice \(= \tfrac{@Q05_dice_num@}{@Q05_dice_den@}\) et IoU \(= \tfrac{@Q05_d_union_num@}{@Q05_d_union_den@}\)",
             "explication": r"L'union a été prise égale à \(50 + 70\), ce qui compte deux fois les 40 pixels communs. L'union contient \(50 + 70 - 40 = @Q05_union@\) pixels."},
        ],
        "solution": r"""Réponse : Dice \(= \tfrac{@Q05_dice_num@}{@Q05_dice_den@}\) et IoU \(= \tfrac{@Q05_iou_num@}{@Q05_iou_den@}\).

Notations : \(A\) est l'ensemble des pixels prédits « objet », \(B_0\) celui des pixels réellement « objet », \(\lvert \cdot \rvert\) le nombre de pixels.

Étape 1 : \(\lvert A \cup B_0 \rvert = \lvert A \rvert + \lvert B_0 \rvert - \lvert A \cap B_0 \rvert = 50 + 70 - 40 = @Q05_union@\), donc IoU \(= 40/@Q05_union@ = \tfrac{@Q05_iou_num@}{@Q05_iou_den@}\).

Étape 2 : Dice \(= 2\,\lvert A \cap B_0 \rvert/(\lvert A \rvert + \lvert B_0 \rvert) = 80/120 = \tfrac{@Q05_dice_num@}{@Q05_dice_den@}\).

Contrôle (leçon, partie 2.5, S1) : \(2\,\mathrm{IoU}/(1 + \mathrm{IoU}) = 1/1{,}5 = \tfrac{@Q05_dice_num@}{@Q05_dice_den@}\). Le Dice est aussi le score F1 de la classe « objet » : précision \(\tfrac{@Q05_precision_num@}{@Q05_precision_den@}\), rappel \(\tfrac{@Q05_rappel_num@}{@Q05_rappel_den@}\).""",
    },
    {
        "id": "ML5-06", "section": "mlf", "theme": "Classes déséquilibrées : stratification, pondération, fuite", "difficulte": 4, "temps_cible_s": 150,
        "correctes": ["a", "c"],
        "enonce": r"""Un jeu de 500 observations indépendantes contient 20 positifs et 480 négatifs. On l'évalue par validation croisée à 5 plis de 100 observations.

Lesquelles de ces affirmations sont justes ?""",
        "options": [
            {"id": "a", "texte": r"Si les plis sont formés au hasard sans stratification, le nombre de positifs d'un pli donné est aléatoire, de moyenne \(@Q06_moy@\) et d'écart-type environ \(@Q06_sd:2m@\) ; avec des plis stratifiés, il vaut exactement \(@Q06_moy@\).",
             "explication": r"Vrai : tirage sans remise de 100 observations parmi 500 dont 20 positives, de variance \(100 \times 0{,}04 \times 0{,}96 \times \tfrac{400}{499} \approx @Q06_var:3m@\) ; la stratification tire séparément 4 positifs parmi les 20."},
            {"id": "b", "texte": "La stratification réduit le biais du modèle ajusté : c'est une forme de régularisation.",
             "explication": r"Faux : la stratification fixe la composition des plis. Elle stabilise l'évaluation (variance de la métrique d'un découpage à l'autre) et ne change ni la famille de modèles ni sa pénalité."},
            {"id": "c", "texte": r"Si l'on multiplie par 3 le poids des positifs dans l'entropie croisée, la prévision qui minimise la perte espérée en un point où la probabilité vraie d'être positif vaut \(\tfrac14\) est \(\tfrac{@Q06_cote_num@}{@Q06_cote_den@}\), et non \(\tfrac14\).",
             "explication": r"Vrai : le minimiseur est \(wq/(wq + 1 - q) = 0{,}75/(0{,}75 + 0{,}75)\) ; la cote est multipliée par le poids, donc les probabilités prédites ne sont plus calibrées."},
            {"id": "d", "texte": "Pour que les plis de validation soient équilibrés, il faut dupliquer les positifs dans tout le jeu avant de former les plis.",
             "explication": r"Faux : des copies d'une même observation se retrouvent alors dans l'apprentissage et dans la validation d'un même tour (fuite), et la validation ne mesure plus la performance à la vraie proportion de positifs."},
        ],
        "solution": r"""Réponse : les affirmations sur le nombre de positifs par pli (moyenne \(@Q06_moy@\), écart-type \(@Q06_sd:2m@\) sans stratification, exactement \(@Q06_moy@\) avec) et sur l'effet de la pondération (prévision \(\tfrac{@Q06_cote_num@}{@Q06_cote_den@}\)) sont justes ; les deux autres sont fausses.

Étape 1 (leçon, partie 2.6, St1) : le nombre \(X\) de positifs d'un pli de \(n = 100\) tiré sans remise dans \(N = 500\) observations de proportion \(q = 0{,}04\) vérifie \(\mathbb{E}[X] = nq = @Q06_moy@\) et \(\mathrm{Var}(X) = nq(1 - q)\tfrac{N - n}{N - 1} = @Q06_var:3m@\), d'écart-type \(@Q06_sd:2m@\). La probabilité d'un pli à 1 positif ou moins vaut \(@Q06_p1:3m@\) (calcul exact ; simulation @Q06_p1_mc:3@, erreur standard @Q06_p1_se:3@).

Étape 2 : la stratification agit sur la variance de l'évaluation, pas sur le modèle.

Étape 3 (W1) : \(h(r) = -3q \ln r - (1 - q)\ln(1 - r)\) est minimale en \(r^* = 3q/(3q + 1 - q)\) ; pour \(q = \tfrac14\), \(r^* = \tfrac{@Q06_cote_num@}{@Q06_cote_den@}\).

Étape 4 : tout rééquilibrage se fait sur l'apprentissage de chaque tour, après le découpage.""",
    },
    # ------------------------------------------------- Biais et variance
    {
        "id": "ML5-07", "section": "mlf", "theme": "Biais et variance : lecture d'une cible et remèdes", "difficulte": 3, "temps_cible_s": 90,
        "conceptuelle": True, "correctes": ["a", "b"],
        "enonce": r"""Sur une cible de tir, le centre représente la vraie valeur à prédire en un point fixé, et chaque impact la prédiction, en ce point, d'un modèle entraîné sur un jeu d'apprentissage différent, tous les jeux étant tirés de la même loi et de même taille.

Lesquelles de ces affirmations sont justes ?""",
        "options": [
            {"id": "a", "texte": "Des impacts groupés mais loin du centre correspondent à un fort biais et à une faible variance ; ajouter des données d'apprentissage de la même loi resserre encore le groupe sans le ramener au centre.",
             "explication": "Vrai : le biais est l'écart entre le centre du nuage et le centre de la cible, la variance la dispersion du nuage ; plus de données réduit la dispersion, pas l'écart dû à un modèle trop rigide."},
            {"id": "b", "texte": "Des impacts dispersés dont le centre de gravité est le centre de la cible correspondent à un faible biais et à une forte variance ; une régularisation plus forte réduit la dispersion, au prix d'un biais qui peut augmenter.",
             "explication": "Vrai : c'est l'arbitrage démontré pour Ridge, dont chaque variance décroît et chaque biais croît en valeur absolue quand la pénalité augmente."},
            {"id": "c", "texte": "Une forte variance se reconnaît à une erreur d'apprentissage et une erreur de validation toutes deux élevées et proches l'une de l'autre.",
             "explication": "Faux : deux erreurs élevées et proches signalent un fort biais (le modèle n'ajuste même pas ses données). Une forte variance donne une erreur d'apprentissage faible et une erreur de validation nettement plus élevée."},
            {"id": "d", "texte": "Augmenter la complexité du modèle est un remède à une forte variance.",
             "explication": "Faux : dans le régime classique, plus de complexité réduit le biais et augmente la variance. C'est le remède du sous-apprentissage ; contre la variance, on régularise, on simplifie ou on ajoute des données."},
        ],
        "solution": r"""Réponse : les deux affirmations qui décrivent les cibles (impacts groupés loin du centre : fort biais, faible variance ; impacts dispersés autour du centre : faible biais, forte variance) avec leur remède sont justes ; les deux autres sont fausses.

Étape 1 : biais \(= \mathbb{E}[\hat Y_0] - m_0\), écart entre le centre du nuage et le centre de la cible ; variance \(= \mathrm{Var}(\hat Y_0)\), dispersion du nuage (leçon, partie 3.1).

Étape 2 : plus de données réduit la variance et laisse le biais ; plus de régularisation réduit la variance et augmente le biais ; plus de complexité fait l'inverse (partie 3.2, tableau des remèdes et expérience des polynômes).

Étape 3 : la lecture des deux erreurs : élevées et proches pour le biais, écart entre apprentissage et validation pour la variance.""",
    },
    # ------------------------------------------------- Apprentissage par renforcement
    {
        "id": "ML5-08", "section": "mla", "theme": "Apprentissage par renforcement : équation de Bellman d'optimalité", "difficulte": 4, "temps_cible_s": 150,
        "enonce": r"""Un processus de décision markovien a deux états, 1 et 2, et un facteur d'actualisation \(\gamma = 0{,}8\). Le retour est la somme des récompenses futures, la récompense reçue \(i\) pas après la première étant multipliée par \(\gamma^i\).

- Dans l'état 1, deux actions : « garder » rapporte 2 et laisse dans l'état 1 ; « changer » rapporte 0 et mène dans l'état 2.
- Dans l'état 2, une seule action : elle rapporte 4, puis l'état suivant est 1 avec probabilité \(\tfrac12\) et 2 avec probabilité \(\tfrac12\).

Quelle est la valeur optimale de l'état 1, c'est-à-dire le plus grand retour espéré que l'on puisse obtenir en partant de l'état 1 ?""",
        "options": [
            {"id": "a", "texte": r"\(\tfrac{@Q08_exact_num@}{@Q08_exact_den@} \approx @Q08_exact:2m@\)",
             "explication": r"Oui : avec « changer », \(v(1) = 0{,}8\,v(2)\) et \(v(2) = 4 + 0{,}4\,v(1) + 0{,}4\,v(2)\), d'où \(v(1) = \tfrac{@Q08_exact_num@}{@Q08_exact_den@}\), ce qui dépasse la valeur de « garder »."},
            {"id": "b", "texte": r"\(@Q08_d_garder_toujours@\)",
             "explication": r"C'est \(2/(1 - 0{,}8)\), la valeur de la politique qui garde toujours. Elle prend la meilleure récompense immédiate, mais l'état 2 rapporte 4 par pas : accepter une récompense nulle pour y aller vaut plus."},
            {"id": "c", "texte": r"\(\tfrac{@Q08_v2_num@}{@Q08_v2_den@} \approx @Q08_v2:2m@\)",
             "explication": r"C'est la valeur optimale de l'état **2**. Celle de l'état 1 s'en déduit par \(v(1) = 0 + 0{,}8\,v(2)\) : il faut un pas, à récompense nulle, pour y arriver."},
            {"id": "d", "texte": r"\(\tfrac{@Q08_d_q_garder_num@}{@Q08_d_q_garder_den@} \approx @Q08_d_q_garder:2m@\)",
             "explication": r"C'est \(2 + 0{,}8 \times \tfrac{@Q08_exact_num@}{@Q08_exact_den@}\) : la valeur de « garder » une fois puis d'agir au mieux. La valeur optimale est le **maximum** sur les actions, atteint par « changer »."},
        ],
        "solution": r"""Réponse : \(\tfrac{@Q08_exact_num@}{@Q08_exact_den@} \approx @Q08_exact:2m@\), obtenue en choisissant « changer » dans l'état 1.

Étape 1 : état 2 (une seule action), équation de Bellman : \(v(2) = 4 + 0{,}8\,\big(\tfrac12 v(1) + \tfrac12 v(2)\big)\), soit \(0{,}6\,v(2) = 4 + 0{,}4\,v(1)\).

Étape 2 : on essaie « changer » en 1 : \(v(1) = 0 + 0{,}8\,v(2)\). En reportant : \(0{,}6\,v(2) = 4 + 0{,}32\,v(2)\), donc \(v(2) = 4/0{,}28 = \tfrac{@Q08_v2_num@}{@Q08_v2_den@}\) et \(v(1) = \tfrac{@Q08_exact_num@}{@Q08_exact_den@}\).

Étape 3 : on vérifie l'équation d'optimalité en 1 (leçon, partie 4.2, Be2) : « garder » vaudrait \(2 + 0{,}8 \times \tfrac{@Q08_exact_num@}{@Q08_exact_den@} = \tfrac{@Q08_d_q_garder_num@}{@Q08_d_q_garder_den@} < \tfrac{@Q08_exact_num@}{@Q08_exact_den@}\). Le maximum est bien atteint par « changer » : ces valeurs sont le point fixe, donc les valeurs optimales.

À comparer : garder toujours donne \(2/(1 - 0{,}8) = @Q08_d_garder_toujours@\). Le script résout les deux politiques en fractions, retrouve la valeur par itération sur la valeur et par simulation (@Q08_exact_mc:3@, erreur standard @Q08_exact_se:3@).""",
    },
    {
        "id": "ML5-09", "section": "mla", "theme": "Apprentissage par renforcement : Q-learning, gradient de politique, exploration", "difficulte": 3, "temps_cible_s": 100,
        "conceptuelle": True, "correctes": ["a", "b", "c"],
        "enonce": r"""On considère un agent qui apprend par renforcement dans un processus de décision markovien à états et actions en nombre fini, dont il ne connaît ni les probabilités de transition ni la fonction de récompense.

Lesquelles de ces affirmations sont justes ?""",
        "options": [
            {"id": "a", "texte": "Le Q-learning est une méthode hors politique : sa cible utilise le maximum des valeurs estimées dans l'état suivant, quelle que soit l'action que l'agent y choisira réellement.",
             "explication": "Vrai : la cible est la récompense plus la valeur actualisée de la meilleure action estimée dans l'état suivant ; l'agent peut explorer autrement tout en apprenant les valeurs optimales. SARSA, qui utilise l'action réellement choisie, est sur politique."},
            {"id": "b", "texte": "Une méthode de gradient de politique peut estimer le gradient du retour espéré sans connaître les probabilités de transition : seul le logarithme de la politique est dérivé par rapport aux paramètres.",
             "explication": "Vrai : le logarithme de la probabilité d'une trajectoire est une somme, et seuls les termes de la politique dépendent des paramètres ; les transitions disparaissent du gradient, qui s'estime sur des trajectoires jouées."},
            {"id": "c", "texte": "Dans REINFORCE, retrancher du retour une ligne de base qui ne dépend que de l'état ne change pas l'espérance du gradient estimé, et peut réduire sa variance.",
             "explication": "Vrai : l'espérance du gradient du logarithme de la politique, sachant l'état, est nulle (les probabilités des actions somment à un), donc le terme retranché est d'espérance nulle."},
            {"id": "d", "texte": "Un agent qui choisit toujours l'action de meilleure valeur estimée, sans jamais explorer, finit nécessairement par identifier la meilleure action, puisque ses estimations s'améliorent à chaque essai.",
             "explication": "Faux : seule l'estimation de l'action essayée s'améliore. Une bonne action sous-estimée au départ n'est plus jamais choisie, et son estimation n'est jamais corrigée."},
        ],
        "solution": r"""Réponse : trois affirmations sont justes : celle sur le Q-learning (hors politique, cible avec le maximum), celle sur le gradient de politique (aucun besoin des probabilités de transition) et celle sur la ligne de base de REINFORCE (même espérance, variance réduite). Celle sur l'agent qui n'explore jamais est fausse.

Étape 1 : cible du Q-learning, \(r + \gamma \max_{a'} Q(s', a')\), indépendante de l'action suivie (leçon, partie 4.3).

Étape 2 : \(\nabla J(\boldsymbol{\theta}) = \mathbb{E}\big[g(\tau)\sum_t \nabla \ln \pi_{\boldsymbol{\theta}}(A_t \mid S_t)\big]\) : ni les transitions ni la récompense ne sont dérivées (partie 4.4, PG1).

Étape 3 : \(\sum_a \pi_{\boldsymbol{\theta}}(a \mid s)\,\nabla \ln \pi_{\boldsymbol{\theta}}(a \mid s) = \nabla \sum_a \pi_{\boldsymbol{\theta}}(a \mid s) = \mathbf{0}\) : une ligne de base \(b(s)\) ne biaise pas (PG2, PG3).

Étape 4 : la stratégie sans exploration peut rester bloquée sur un bras moins bon, avec une probabilité strictement positive (partie 4.5, X1).""",
    },
    {
        "id": "ML5-10", "section": "mla", "theme": "Gradient de politique : espérance et variance de l'estimateur REINFORCE", "difficulte": 5, "temps_cible_s": 150,
        "enonce": r"""Un agent choisit une seule fois entre deux actions. Il choisit l'action 1 avec probabilité \(\pi = \mathrm{sig}(\theta)\), où \(\mathrm{sig}(\theta) = 1/(1 + e^{-\theta})\), et l'action 0 sinon ; le paramètre \(\theta\) est tel que \(\pi = \tfrac15\). L'action 1 rapporte la récompense 6, l'action 0 la récompense 1 (sans aléa). On note \(A\) l'action choisie, \(R\) la récompense reçue, et l'on estime la dérivée du gain espéré par rapport à \(\theta\) par

\(\displaystyle \hat g_b = (R - b)\,\frac{d}{d\theta}\ln \mathbb{P}_\theta(A),\)

où \(\mathbb{P}_\theta(A)\) est la probabilité que la politique donnait à l'action choisie et \(b\) une constante (ligne de base).

Que valent l'espérance et la variance de \(\hat g_b\), pour \(b = 0\) puis pour \(b = 2\) ?""",
        "options": [
            {"id": "a", "texte": r"Espérance \(@Q10_esp:1m@\) dans les deux cas ; variance \(@Q10_var0@\) pour \(b = 0\) et \(@Q10_var_b:2m@\) pour \(b = 2\).",
             "explication": r"Oui : \(\hat g_b\) vaut \((6 - b) \times 0{,}8\) avec probabilité \(\tfrac15\) et \(-(1 - b) \times 0{,}2\) sinon ; l'espérance ne dépend pas de \(b\), la variance \(\pi(1 - \pi)\) fois le carré de l'écart entre les deux valeurs, si."},
            {"id": "b", "texte": r"Espérance \(@Q10_esp:1m@\) pour \(b = 0\) et 0 pour \(b = 2\) ; variance \(@Q10_var0@\) puis \(@Q10_var_b:2m@\).",
             "explication": r"L'espérance ne s'annule pas : \(b = 2\) est bien la récompense moyenne, donc \(\mathbb{E}[R - b] = 0\), mais \(R - b\) est multiplié par un facteur qui dépend de l'action. Le terme en \(b\) est d'espérance nulle à lui seul."},
            {"id": "c", "texte": r"Espérance \(@Q10_esp:1m@\) et variance \(@Q10_var0@\) dans les deux cas.",
             "explication": r"Retrancher \(b\) à \(R\) ne translate pas \(\hat g_b\) d'une constante : le terme retranché, \(b\) fois la dérivée du logarithme, est aléatoire. La variance change donc, ici de \(@Q10_var0@\) à \(@Q10_var_b:2m@\)."},
            {"id": "d", "texte": r"Espérance \(@Q10_esp:1m@\) dans les deux cas ; variance \(@Q10_d_m2_0:2m@\) pour \(b = 0\) et \(@Q10_d_m2_b:2m@\) pour \(b = 2\).",
             "explication": r"Ce sont les moments d'ordre 2, \(\mathbb{E}[\hat g_b^2]\) : il reste à retrancher le carré de l'espérance, \(0{,}8^2\)."},
        ],
        "solution": r"""Réponse : espérance \(@Q10_esp:1m@\) dans les deux cas ; variance \(@Q10_var0@\) pour \(b = 0\), \(@Q10_var_b:2m@\) pour \(b = 2\).

Étape 1 : dérivées. \(\tfrac{d}{d\theta}\ln \pi = 1 - \pi = 0{,}8\) (action 1) et \(\tfrac{d}{d\theta}\ln(1 - \pi) = -\pi = -0{,}2\) (action 0), car \(\mathrm{sig}' = \mathrm{sig}(1 - \mathrm{sig})\).

Étape 2, \(b = 0\) : \(\hat g_0\) vaut \(6 \times 0{,}8 = @Q10_g1:1m@\) avec probabilité \(\tfrac15\) et \(1 \times (-0{,}2) = @Q10_g0:1m@\) avec probabilité \(\tfrac45\). Espérance \(0{,}96 - 0{,}16 = @Q10_esp:1m@\), qui est bien \(\pi(1 - \pi)(6 - 1)\), la dérivée du gain espéré \(6\pi + (1 - \pi)\).

Étape 3 : une variable qui prend deux valeurs \(u_1\) et \(u_0\) avec probabilités \(\pi\) et \(1 - \pi\) a pour variance \(\pi(1 - \pi)(u_1 - u_0)^2\). Ici \(0{,}16 \times 5^2 = @Q10_var0@\).

Étape 4, \(b = 2\) : valeurs \(4 \times 0{,}8 = @Q10_gb1:1m@\) et \(-(1 - 2) \times 0{,}2 = @Q10_gb0:1m@\) ; espérance \(0{,}64 + 0{,}16 = @Q10_esp:1m@\), inchangée (leçon, partie 4.4, PG3) ; variance \(0{,}16 \times 3^2 = @Q10_var_b:2m@\).

Complément : la variance s'annule pour \(b = 6 \times 0{,}8 + 1 \times 0{,}2 = @Q10_b_opt@\), qui n'est pas la récompense moyenne \(@Q10_moyR@\). Simulation du script : variances @Q10_var0_mc:3@ et @Q10_var_b_mc:3@.""",
    },
    # ------------------------------------------------- Réflexes : arbres, importances, ACP, pénalités
    {
        "id": "ML5-11", "section": "mlf", "theme": "Arbres de décision et importances de variables : affirmations", "difficulte": 4, "temps_cible_s": 120,
        "conceptuelle": True, "correctes": ["a", "c"],
        "enonce": r"""On ajuste un arbre de décision de classification par l'algorithme CART (coupures binaires de la forme « variable inférieure ou égale à un seuil », critère de Gini), puis l'on s'intéresse à l'importance des variables.

Lesquelles de ces affirmations sont justes ?""",
        "options": [
            {"id": "a", "texte": "À la racine, l'algorithme retient le couple (variable, seuil) qui réduit le plus l'impureté sur l'ensemble des observations d'apprentissage ; une variable qui n'est utile qu'en interaction avec une autre peut donc y avoir un gain nul.",
             "explication": "Vrai : le critère évalue une coupure sur une variable à la fois. Dans un ou exclusif de deux variables binaires équiprobables, chacune prise seule laisse les proportions de classes inchangées : gain nul, alors qu'à deux elles déterminent la cible."},
            {"id": "b", "texte": "Une variable numérique continue doit être découpée en classes avant l'ajustement, car un nœud ne sait tester que l'égalité à une modalité.",
             "explication": "Faux : un nœud compare la variable à un seuil, choisi parmi les milieux de valeurs observées consécutives. Aucune mise en classes préalable n'est nécessaire, et la même variable peut resservir plus bas avec un autre seuil."},
            {"id": "c", "texte": "L'importance par réduction d'impureté, calculée sur les données d'apprentissage, peut classer une variable continue de pur bruit devant une variable binaire réellement informative.",
             "explication": "Vrai : une variable aux nombreuses valeurs offre beaucoup de coupures candidates, donc plus d'occasions d'un gain dû au hasard, et tout gain est positif ou nul. C'est le biais démontré dans la séance de ML 2."},
            {"id": "d", "texte": "Dans une régression linéaire sur des variables non standardisées, la variable dont le coefficient est le plus grand en valeur absolue est celle qui pèse le plus dans les prédictions.",
             "explication": "Faux : changer l'unité d'une variable divise son coefficient par le facteur de conversion sans changer aucune prédiction. Il faut comparer les coefficients multipliés par l'écart-type de leur variable."},
        ],
        "solution": r"""Réponse : les affirmations sur le choix de la racine (meilleure réduction d'impureté d'une coupure, une variable à la fois) et sur le biais de l'importance par impureté sont justes ; les deux autres sont fausses.

Étape 1 : le gain d'une coupure ne regarde qu'une variable ; dans l'ou exclusif, \(\mathbb{P}(Y = 1 \mid X_1) = \tfrac12\) quelle que soit la valeur de \(X_1\), donc les enfants ont l'impureté du parent (leçon, partie 5.2, T1).

Étape 2 : les coupures d'une variable numérique sont des seuils entre valeurs observées consécutives (séance de ML 2, partie 1) : pas de mise en classes.

Étape 3 : l'importance par impureté est biaisée vers les variables offrant beaucoup de coupures (séance de ML 2, partie 7, I1).

Étape 4 : si \(x_j\) est remplacée par \(c\,x_j\), le coefficient devient \(\hat b_j/c\) et les prédictions ne changent pas (leçon, partie 5.3, Co1).""",
    },
    {
        "id": "ML5-12", "section": "mlf", "theme": "ACP : affirmations et part de variance", "difficulte": 3, "temps_cible_s": 100,
        "correctes": ["a"],
        "enonce": r"""On réalise une analyse en composantes principales sur quatre variables centrées et réduites (chacune de variance 1). Leur matrice de corrélation a pour valeurs propres 2,2 ; 1 ; 0,5 et 0,3.

Lesquelles de ces affirmations sont justes ?""",
        "options": [
            {"id": "a", "texte": r"La première composante principale explique \(@Q12_part1_pct@\) % de la variance totale, et les deux premières ensemble \(@Q12_part2_pct@\) %.",
             "explication": r"Vrai : la variance totale est la somme des valeurs propres, 4, égale au nombre de variables réduites ; les parts sont \(2{,}2/4\) et \((2{,}2 + 1)/4\)."},
            {"id": "b", "texte": "Quelle que soit la loi des données, les deux premières composantes principales sont des variables indépendantes l'une de l'autre.",
             "explication": "Faux : elles sont toujours non corrélées, et leurs directions orthogonales, mais non corrélé n'implique indépendant que dans des cas particuliers, par exemple pour un vecteur gaussien."},
            {"id": "c", "texte": "Chaque composante principale est l'une des quatre variables d'origine : garder deux composantes revient à garder les deux variables de plus grande variance.",
             "explication": "Faux : une composante est une combinaison linéaire de toutes les variables. L'ACP ne sélectionne pas de variables ; ici elles ont d'ailleurs toutes la même variance."},
            {"id": "d", "texte": "Garder les deux premières composantes garantit de conserver les deux directions les plus utiles pour prédire une variable cible.",
             "explication": "Faux : l'ACP ne regarde pas la cible. Une cible peut dépendre surtout d'une direction de petite variance, que l'on écarterait en ne gardant que les premières composantes."},
        ],
        "solution": r"""Réponse : une seule affirmation est juste, celle sur les parts de variance (\(@Q12_part1_pct@\) % puis \(@Q12_part2_pct@\) %) ; les trois autres sont fausses.

Étape 1 : la part de la composante \(k\) est \(\lambda_k/\sum_i \lambda_i\), et la somme des valeurs propres d'une matrice de corrélation de 4 variables vaut 4 : \(2{,}2/4 = @Q12_part1:2m@\) et \(3{,}2/4 = @Q12_part2:1m@\) (séance d'algèbre linéaire, section 9).

Étape 2 : les composantes sont non corrélées et leurs directions orthogonales, toujours ; « indépendantes » demande une hypothèse de plus sur la loi (leçon, partie 5.4, fait 2).

Étape 3 : une composante est une combinaison linéaire de toutes les variables (fait 3) ; l'ACP est non supervisée (fait 6, et l'expérience où la cible est portée par la dernière composante).""",
    },
    {
        "id": "ML5-13", "section": "mlf", "theme": "Elastic net : solution à colonnes orthonormées", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""On ajuste un modèle linéaire sans constante par elastic net : on minimise sur \(\mathbf{b} \in \mathbb{R}^3\)

\(\displaystyle \tfrac12 \lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2 + \lambda\Big[\alpha\,\lVert \mathbf{b} \rVert_1 + \tfrac{1 - \alpha}{2}\,\lVert \mathbf{b} \rVert^2\Big],\)

où \(\lVert \mathbf{b} \rVert_1\) est la somme des valeurs absolues des coordonnées et \(\lVert \mathbf{b} \rVert\) la norme euclidienne, avec \(\lambda = 2\) et \(\alpha = \tfrac12\). Les trois colonnes de \(\mathbf{X}\) sont orthonormées (\(\mathbf{X}^\top \mathbf{X} = \mathbf{I}\)) et \(\mathbf{X}^\top \mathbf{y} = (4 ; -1{,}5 ; 0{,}6)\).

Quelle est la solution \(\hat{\mathbf{b}}\) ?""",
        "options": [
            {"id": "a", "texte": r"\(@Q13_exact:2m@\)",
             "explication": r"Oui : seuillage doux au niveau \(\lambda\alpha = @Q13_seuil@\), puis division par \(1 + \lambda(1 - \alpha) = @Q13_div@\) ; la troisième coordonnée, sous le seuil, est exactement nulle."},
            {"id": "b", "texte": r"\(@Q13_d_sans_div:2m@\)",
             "explication": r"C'est le seuillage doux seul, la solution du Lasso de pénalité \(\lambda\alpha\). La part L2 de la pénalité divise encore chaque coordonnée par \(1 + \lambda(1 - \alpha)\)."},
            {"id": "c", "texte": r"\(@Q13_d_sans_seuil:2m@\)",
             "explication": r"C'est la division seule, la solution Ridge de pénalité \(\lambda(1 - \alpha)\) : aucun zéro. La part L1 retire d'abord \(\lambda\alpha\) à chaque valeur absolue, et annule la troisième coordonnée."},
            {"id": "d", "texte": r"\(@Q13_d_seuil_lambda:2m@\)",
             "explication": r"Le seuil a été pris égal à \(\lambda = 2\) au lieu de \(\lambda\alpha = 1\) : la deuxième coordonnée, de valeur absolue 1,5, a été annulée à tort."},
        ],
        "solution": r"""Réponse : \(\hat{\mathbf{b}} = @Q13_exact:2m@\).

Étape 1 : avec des colonnes orthonormées et \(\mathbf{z} = \mathbf{X}^\top \mathbf{y}\), le critère vaut une constante plus \(\sum_j \big[\tfrac12(z_j - b_j)^2 + \lambda\alpha\,\lvert b_j \rvert + \tfrac{\lambda(1 - \alpha)}{2}\,b_j^2\big]\) : il se découple coordonnée par coordonnée (leçon, partie 5.5, EN3).

Étape 2 : pour une coordonnée, \(\hat b_j = 0\) si \(\lvert z_j \rvert \le \lambda\alpha\) ; sinon la dérivée nulle donne \(\hat b_j = (z_j - \lambda\alpha\,\mathrm{signe}(z_j))/(1 + \lambda(1 - \alpha))\).

Étape 3 : \(\lambda\alpha = @Q13_seuil@\) et \(1 + \lambda(1 - \alpha) = @Q13_div@\). Donc \((4 - 1)/2\), \((-1{,}5 + 1)/2\) et 0 (car \(0{,}6 \le 1\)) : \(@Q13_exact:2m@\).

Le script retrouve cette solution par minimisation numérique du critère.""",
    },
    {
        "id": "ML5-14", "section": "mlf", "theme": "Pénalités en grande dimension et remèdes au surapprentissage : affirmations", "difficulte": 3, "temps_cible_s": 100,
        "conceptuelle": True, "correctes": ["a", "b", "c"],
        "enonce": r"""On ajuste une régression linéaire pénalisée sur des données qui comptent beaucoup plus de variables que d'observations ; les variables sont standardisées. On souhaite un modèle que l'on puisse lire, c'est-à-dire qui retienne peu de variables.

Lesquelles de ces affirmations sont justes ?""",
        "options": [
            {"id": "a", "texte": "Avec une pénalité L1 assez forte, des coefficients sont exactement nuls : le modèle retient un sous-ensemble de variables.",
             "explication": "Vrai : un coefficient du Lasso reste nul tant que la corrélation de sa variable avec le résidu ne dépasse pas la pénalité en valeur absolue ; au-delà d'un seuil de pénalité, tous sont nuls."},
            {"id": "b", "texte": "Avec une pénalité L2 seule, la solution est unique même quand il y a plus de variables que d'observations, mais elle ne met aucun coefficient exactement à zéro, sauf cas de probabilité nulle.",
             "explication": "Vrai : la matrice du système Ridge est définie positive pour toute pénalité strictement positive, d'où l'unicité ; et un coefficient Ridge ne s'annule que sur un ensemble de données de probabilité nulle."},
            {"id": "c", "texte": "Si deux variables sont exactement identiques, l'elastic net, dont la pénalité contient une part L2 non nulle, leur donne le même coefficient, alors que le problème L1 seul n'a pas toujours une solution unique.",
             "explication": "Vrai : la part L2 rend le critère strictement convexe, donc la solution unique ; échanger les deux coefficients donne une solution, qui est donc la même. Avec L1 seule, tout partage d'un même total entre les deux copies, à signes égaux, a le même critère."},
            {"id": "d", "texte": "Si l'erreur d'apprentissage est très inférieure à l'erreur de validation, il faut diminuer la pénalité ou ajouter des variables construites à partir des premières.",
             "explication": "Faux : cet écart signale une forte variance. Diminuer la pénalité ou ajouter des variables augmente la complexité, donc l'aggrave ; il faut au contraire régulariser davantage ou simplifier."},
        ],
        "solution": r"""Réponse : trois affirmations sont justes : celle sur la pénalité L1 (coefficients exactement nuls, sélection de variables), celle sur la pénalité L2 (solution unique, aucun zéro exact) et celle sur l'elastic net (même coefficient pour deux variables identiques). Celle qui propose de diminuer la pénalité ou d'ajouter des variables est fausse.

Étape 1 : conditions d'optimalité du Lasso, avec \(g_j\) le produit scalaire de la colonne \(j\) et du résidu, et \(\lambda\) la pénalité : un coefficient non nul exige \(\lvert g_j \rvert = \lambda\), un coefficient nul vérifie \(\lvert g_j \rvert \le \lambda\) ; au-delà d'un seuil de \(\lambda\), tous sont nuls (séance de ML 1, partie 5).

Étape 2 : Ridge : \(\mathbf{X}^\top \mathbf{X} + \lambda\mathbf{I}\) est définie positive, la solution est unique quel que soit le rang, sans zéro exact hors cas de probabilité nulle (séance de ML 1, partie 4).

Étape 3 : entre variables presque identiques, le Lasso choisit de façon instable ; l'elastic net, strictement convexe, donne le même coefficient à deux colonnes identiques (leçon, partie 5.5, EN1).

Étape 4 : un écart entre erreur d'apprentissage et erreur de validation se traite en réduisant la variance : plus de pénalité, moins de variables, plus de données (partie 5.6).""",
    },
]
