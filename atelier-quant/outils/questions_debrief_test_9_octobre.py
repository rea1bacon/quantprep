"""Corrigé de l'OA QRT Quant Researcher, question par question : une page de type leçon, sans question notée.

Pour ne rien écrire deux fois, les énoncés, options, corrections et explications d'options viennent du module
outils/questions_test_9_octobre.py (la page d'examen) ; ce module-ci ajoute, pour chaque question, ce qu'il faut retenir
et la leçon à revoir. Les lettres A, B, C... désignent les options dans l'ordre d'origine (champ "ordre_origine").

Découpage : une partie par titre de niveau 2 ; les deux dernières rubriques ont la forme attendue par
outils/questions_fiche_derniere_heure.py.
"""
import re

import questions_diagnostic as diag
import questions_test_9_octobre as t9

ID = "debrief-test-9-octobre"
TITRE = "OA QRT Quant Researcher : corrigé"
VERIF = "verif/test_9_octobre.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [{"id": "lecon", "nom": "Corrigé", "budget_min": 90}]
PRESENTATION = (
    "Le corrigé de l'évaluation en ligne Quant Researcher de QRT, session du 9 octobre 2026 : pour chaque question, l'énoncé reformulé, "
    "la réponse attendue, la justification complète, le sort de chaque option, ce qu'il faut retenir et la leçon à revoir. Aucune "
    "question notée sur cette page.\n\n"
    "Limites : cette page n'est pas un document officiel de QRT ; les réponses attendues sont celles que cette plateforme a établies par "
    "preuve ou par calcul vérifié, la grille du correcteur n'étant pas connue ; les énoncés des questions 14, 15 et 18 ne sont pas "
    "disponibles.\n\n"
    "Pour passer l'examen dans ses conditions, le faire **avant** de lire ce corrigé."
)
AVERTISSEMENTS = [
    "Les énoncés sont des reformulations en français ; les lettres A, B, C, D suivent l'ordre d'origine des options.",
    "Plusieurs questions d'origine sont ambiguës ou mal posées. Le corrigé le dit à chaque fois, et indique quelle autre lecture "
    "mènerait à une autre réponse.",
]

M5 = "la séance de mathématiques 5 (statistique descriptive, tests d'adéquation et densités jointes)"
ML5 = "la séance de ML 5 (réseaux convolutifs, entraînement, apprentissage par renforcement et réflexes de choix multiple)"
C3 = "la séance de code 3 (une chaîne complète de classification avec scikit-learn)"

# Pour chaque question : titre court, à retenir, leçon à revoir.
FICHES = {
    2: {
        "titre": "tests d'adéquation",
        "retenir": r"Khi-deux : des classes, donc toute loi discrète (et une loi continue découpée). Kolmogorov-Smirnov et Anderson-Darling : fonction de répartition d'une loi **continue entièrement spécifiée**. Dès qu'un paramètre est estimé, les degrés de liberté ou les tables changent.",
        "lecon": f"{M5}. Pour le vocabulaire des tests (p-value, niveau) : « Séance de mathématiques 3 : statistique et inégalités », partie 4.",
    },
    3: {
        "titre": "hypothèses de la régression linéaire",
        "retenir": "Les hypothèses sont linéarité, erreurs centrées, variance constante, erreurs non corrélées, pas de colinéarité parfaite, et normalité pour les tests exacts seulement. Une phrase qui décrit une variance qui dépend du régresseur décrit une violation, pas une hypothèse.",
        "lecon": "« Séance de ML 1 : régression linéaire, Ridge et Lasso », partie 2 (le modèle statistique et ce que garantit chaque hypothèse).",
    },
    4: {
        "titre": "corrélation et covariance",
        "retenir": r"\(\mathrm{Cov}(aX + b, Y) = a\,\mathrm{Cov}(X, Y)\) : la covariance a une unité et suit l'échelle ; la corrélation n'en a pas et ne change que de signe si \(a < 0\). Les deux sont aveugles à une dépendance non linéaire (\(Y = X^2\)).",
        "lecon": f"{M5} ; la matrice de corrélation est aussi traitée dans la « Séance d'algèbre linéaire : spectre, ACP et covariance », partie 2.",
    },
    5: {
        "titre": "crises, loi de Poisson et probabilité a priori",
        "retenir": r"Probabilité jointe = a priori fois vraisemblance ; les constantes qui ne dépendent pas du paramètre se simplifient. Ici \(n^5 e^{-n}\), maximal en 5 sur les réels : entre deux candidats entiers de part et d'autre, on compare leur rapport.",
        "lecon": "« Séance de mathématiques 2 : conditionnement et lois usuelles », parties 1 (Bayes) et 4 (processus de Poisson).",
    },
    6: {
        "titre": "asymétrie d'une distribution",
        "retenir": "Queue longue à droite : mode, puis médiane, puis moyenne, dans le cas type à une bosse. C'est une règle empirique, pas un théorème : un contre-exemple discret existe.",
        "lecon": f"{M5}.",
    },
    7: {
        "titre": "corde coupée en trois, morceau le plus court",
        "retenir": r"Loi du minimum par sa queue : \(\mathbb{P}(L > t) = (1 - 3t)^2\), puis \(\mathbb{E}[L] = \int \mathbb{P}(L > t)\,dt = @Q07_exact_num@/@Q07_exact_den@\). Pour \(m\) morceaux : \(1/m^2\) (contrôlé par simulation, démontré ici pour 3).",
        "lecon": "« Séance de mathématiques 1 : probabilités », outil 3 (statistiques d'ordre de lois uniformes), pour la formule de la queue et les espacements.",
    },
    8: {
        "titre": "densité conditionnelle",
        "retenir": r"Trois gestes dans l'ordre : densité marginale \(f_X\) en intégrant sur \(y\) ; densité conditionnelle \(f_{XY}/f_X\) ; reconnaître la loi (ici exponentielle de moyenne \(x\)) et lire sa queue. Une réponse qui dépend de \(y\) ou qui dépasse 1 s'élimine sans calcul.",
        "lecon": f"{M5} ; l'espérance conditionnelle est introduite dans la « Séance de mathématiques 2 : conditionnement et lois usuelles », partie 1.",
    },
    9: {
        "titre": "densité jointe, constante de normalisation",
        "retenir": r"Densité à variables séparées sur un rectangle : produit de deux intégrales simples, \(@Q09_ix@ \times @Q09_iy_num@/@Q09_iy_den@\). Finir le produit, puis contrôler en multipliant l'option choisie par l'intégrale : on doit trouver 1.",
        "lecon": f"{M5}.",
    },
    10: {
        "titre": "deux enfants, probabilités conditionnelles",
        "retenir": "« Tel enfant précis est... » laisse 2 cas ; « au moins un est... » en laisse 3. Écrire les quatre couples équiprobables et barrer ceux que l'information exclut.",
        "lecon": "« Séance de mathématiques 2 : conditionnement et lois usuelles », partie 1 (probabilités totales et Bayes).",
    },
    11: {
        "titre": "comparer deux notes par l'écart réduit",
        "retenir": r"Performance relative = écart réduit \((x - \mu)/\sigma\), pas l'écart brut ni la note brute. Sous normalité, même écart réduit signifie même rang.",
        "lecon": f"{M5}.",
    },
    12: {
        "titre": "réseau convolutif, forme de la sortie",
        "retenir": r"Taille de sortie \(\lfloor (n + 2p - k)/s \rfloor + 1\) ; profondeur = nombre de filtres ; la concaténation en profondeur additionne les canaux et exige la même hauteur et la même largeur ; le pooling ne touche pas à la profondeur. Si la marge n'est pas donnée, regarder les options pour savoir quelle convention l'auteur suppose.",
        "lecon": f"{ML5}. Les convolutions à une dimension (marge, pas, dilatation) sont dans la « Séance de ML 4 : optimisation, réseaux de neurones, noyaux et EM », partie 3.",
    },
    13: {
        "titre": "réglages d'entraînement d'une segmentation",
        "retenir": "Trois contrôles : la plus grande part des données à l'entraînement ; pas par époque fois taille de lot égal à la taille de l'ensemble d'entraînement ; perte accordée au codage de la cible (binaire pour un masque à un canal).",
        "lecon": f"{ML5}. Les rôles des ensembles d'entraînement, de validation et de test sont dans la « Séance de ML 3 : classification, métriques et validation », partie 4.",
    },
    16: {
        "titre": "biais et variance, les quatre cibles",
        "retenir": "Biais = le nuage est décalé par rapport au centre ; variance = le nuage est étalé. Deux questions indépendantes à poser à chaque cible.",
        "lecon": "« Séance de ML 1 : régression linéaire, Ridge et Lasso », partie 6 (biais, variance et rôle de lambda) ; « Séance de mathématiques 3 : statistique et inégalités », partie 1.",
    },
    17: {
        "titre": "apprentissage par renforcement",
        "retenir": "Agent, état, action, récompense ; objectif : l'espérance du retour (somme actualisée des récompenses). Deux familles : valeur (on estime, la politique s'en déduit) et gradient de politique (on optimise la politique paramétrée). Explorer n'est pas tirer au hasard uniformément.",
        "lecon": f"{ML5}.",
    },
    19: {
        "titre": "classes déséquilibrées et problèmes de variance",
        "retenir": "Régularisation : réduit la variance du modèle. Stratification : stabilise la composition des ensembles, donc la variance de l'évaluation. Quand la situation et le but de l'énoncé désignent deux options différentes, s'en tenir au verbe de la question (ici : traiter la variance pour améliorer la performance).",
        "lecon": f"{ML5} pour les classes déséquilibrées ; « Séance de ML 1 : régression linéaire, Ridge et Lasso », parties 4 à 6, pour la régularisation.",
    },
    20: {
        "titre": "arbres de décision, mécanique",
        "retenir": "À chaque nœud, la coupure (variable et seuil) qui réduit le plus l'impureté ; les variables numériques se coupent par seuil, sans discrétisation ; la profondeur dépend des données et des critères d'arrêt, pas du nombre de variables.",
        "lecon": "« Séance de ML 2 : arbres, forêts aléatoires, boosting et XGBoost », partie 1 (l'arbre de décision).",
    },
    21: {
        "titre": "importance des variables",
        "retenir": "Deux définitions à connaître mot pour mot : réduction d'impureté cumulée sur les nœuds où la variable coupe ; baisse de performance quand on mélange la colonne. Et un réflexe : « toujours fiable » est faux (bruit continu, variables corrélées).",
        "lecon": "« Séance de ML 2 : arbres, forêts aléatoires, boosting et XGBoost », partie 7 (importance des variables) ; « Séance quantitative : sélection de modèles, finance quantitative et optimisation », partie 3.",
    },
    22: {
        "titre": "surapprentissage, la technique à choisir",
        "retenir": "Surapprentissage = excès de variance du modèle. Remèdes : régulariser, simplifier, ajouter des données. Augmenter la complexité est le remède au sous-apprentissage. Se méfier des tournures comme « capter plus de variance » : la variance des données n'est pas la variance du modèle.",
        "lecon": "« Séance de ML 1 : régression linéaire, Ridge et Lasso », parties 4 (Ridge) et 6 (biais, variance et rôle de lambda).",
    },
    23: {
        "titre": "analyse en composantes principales",
        "retenir": r"Une composante principale est \(\mathbf{w}_j^{\top}(\mathbf{X} - \mathbb{E}[\mathbf{X}])\) : une combinaison linéaire des variables centrées. Directions orthogonales, variances décroissantes \(\lambda_j\), part \(\lambda_j / \sum \lambda\). Sensible à l'échelle : standardiser quand les unités diffèrent.",
        "lecon": "« Séance d'algèbre linéaire : spectre, ACP et covariance », partie 9 (analyse en composantes principales).",
    },
    24: {
        "titre": "régularisation en grande dimension",
        "retenir": "L1 annule des coefficients (seuillage doux), L2 les rétrécit tous sans en annuler. « Interprétable » ou « sélection de variables » en grande dimension appelle L1 ; « variables corrélées, stabilité » appelle L2 ou Elastic Net.",
        "lecon": "« Séance de ML 1 : régression linéaire, Ridge et Lasso », parties 4 (Ridge) et 5 (Lasso).",
    },
}

MANQUANTES = {14: "la matrice de confusion (rappel et précision)", 15: "les k plus proches voisins", 18: "le dropout"}
QUESTIONS_T9 = {int(q["id"][-2:]): q for q in t9.QUESTIONS}
LETTRES = "ABCDE"


def _ordre(q):
    return q.get("ordre_origine", [o["id"] for o in q["options"]])


def _attendues(q):
    """Lettres des options justes, dans l'ordre d'origine."""
    justes = set(q.get("correctes", ["a"]))
    return "".join(LETTRES[k] for k, oid in enumerate(_ordre(q)) if oid in justes)


def _liste_lettres(lettres):
    lettres = list(lettres)
    return lettres[0] if len(lettres) == 1 else ", ".join(lettres[:-1]) + " et " + lettres[-1]


def _bloc_question(num):
    q, fiche = QUESTIONS_T9[num], FICHES[num]
    options = {o["id"]: o for o in q["options"]}
    ordre = _ordre(q)
    justes = set(q.get("correctes", ["a"]))
    multiple = "correctes" in q
    enonce = re.sub(r"^Question \d+\. ", "", q["enonce"])
    assert q["solution"].startswith("Réponse : ") and all("\n" not in o["explication"] for o in q["options"])
    tete, _, suite = q["solution"][len("Réponse : "):].partition("\n\n")
    lignes = [
        f"### Question {num} : {fiche['titre']}",
        enonce,
        "Type : " + ("une ou plusieurs réponses." if multiple else "une seule réponse."),
        "\n".join(f"- **{LETTRES[k]}.** {options[oid]['texte']}" for k, oid in enumerate(ordre)),
        f"**Réponse attendue :** {_liste_lettres(_attendues(q))}. {tete}",
        "**Justification.**",
        suite,
        "**Option par option.**",
        "\n".join(f"- **{LETTRES[k]}** ({'juste' if oid in justes else 'fausse'}). {options[oid]['explication']}" for k, oid in enumerate(ordre)),
        f"**À retenir.** {fiche['retenir']}",
        f"**Leçon à revoir :** {fiche['lecon']}",
    ]
    return "\n\n".join(lignes)


def _bloc_manquante(num):
    return (f"### Question {num} : énoncé non disponible\n\n"
            "L'énoncé de cette question n'est pas disponible, et rien n'en est reconstitué ici. Sur la page d'examen, cette place est "
            f"occupée par une question de remplacement sur {MANQUANTES[num]}, qui a sa propre correction ; elle ne dit rien de la "
            "question d'origine.")


def _blocs(numeros):
    return "\n\n".join(_bloc_manquante(n) if n in MANQUANTES else _bloc_question(n) for n in numeros)


def _tableau():
    rangs = ["| N° | Thème | Type | Réponse |", "|---|---|---|---|",
             "| 1 | code : classification binaire dans un notebook | code | fichier de prédictions |"]
    for num in range(2, 25):
        if num in MANQUANTES:
            rangs.append(f"| {num} | énoncé non disponible | inconnu | inconnue |")
            continue
        q = QUESTIONS_T9[num]
        rangs.append(f"| {num} | {FICHES[num]['titre']} | {'multiple' if 'correctes' in q else 'unique'} | {_liste_lettres(_attendues(q))} |")
    return "\n".join(rangs)


assert sorted(list(FICHES) + list(MANQUANTES)) == list(range(2, 25))

INTRO = r"""## Présentation et méthodes

Cette page est le corrigé de l'évaluation en ligne Quant Researcher de QRT, session du 9 octobre 2026. Les énoncés sont reformulés en français ; les lettres A, B, C, D désignent les options dans leur ordre d'origine. Ce n'est pas un document officiel de QRT : les réponses sont établies ici par preuve ou par calcul, et la grille du correcteur n'est pas connue. Les nombres viennent de `verif/test_9_octobre.py`, qui les calcule de deux façons quand il s'agit d'une probabilité ou d'une espérance.

**Conventions.** Majuscule : variable aléatoire ; minuscule : valeur ou paramètre fixe ; gras : vecteur ou matrice. \(\mathbb{P}\) est une probabilité, \(\mathbb{E}\) une espérance, \(\mathrm{Var}\) une variance, \(\mathrm{Cov}\) une covariance. Chaque autre symbole est défini dans la question où il sert.

### Le tableau des 24 questions

""" + _tableau() + r"""

Les énoncés des questions 14, 15 et 18 ne sont pas disponibles.

### Méthodes pour ce format

1. **Questions à plusieurs réponses : traiter chaque affirmation comme une question vrai ou faux à part entière.** Une définition exacte se coche, même quand deux autres cases sont déjà trouvées. Il faut lire chaque affirmation seule, décider, et cocher tout ce qui est vrai.
2. **Biais, variance, régularisation : revenir à la définition avant de lire les options.** Surapprendre, c'est trop de variance ; on la réduit en régularisant, pas en augmentant la complexité ; L1 annule des coefficients, L2 non. Les tournures comme « capter plus de variance » confondent la variance des données et celle du modèle.
3. **Contrôler un calcul court.** Une constante de normalisation se vérifie en dix secondes : multiplier l'option choisie par l'intégrale doit donner 1.
4. **Une question mal posée se gère.** Les questions 5, 7, 10, 12, 19, 20 et 23 laissent une convention non dite, ou se contredisent (figure et liste de la question 12). Méthode : chercher la lecture pour laquelle une option est exactement juste, se servir des options pour deviner la convention de l'auteur, et ne pas y passer plus que le temps prévu.
"""

CODE = r"""## Section 1 : code (question 1)

### Question 1 : prédire une charge de climatisation

**La tâche.** Un notebook Jupyter (Python 3.9) et des fichiers : un fichier d'entraînement, un fichier de test, un fichier modèle de soumission, et un fichier de soumission à produire. Chaque ligne décrit une maison ; la cible, nommée `outcome`, vaut 0 (charge de climatisation faible) ou 1 (charge élevée). Il faut construire un classifieur binaire, prédire la cible des maisons du fichier de test et écrire ces prédictions dans le fichier de soumission. Les livrables annoncés sont le notebook, documenté, et ce fichier.

Colonnes de l'aperçu des cinq premières lignes : un identifiant ; une colonne de texte qui réunit **deux** informations, la compacité relative et l'orientation ; la surface des murs ; la surface du toit ; la hauteur totale ; la surface vitrée ; la cible.

Consignes connues (la suite du texte n'est pas disponible) : préparer les données ; supprimer toute ligne où une valeur manque ; ajouter au moins une nouvelle variable pour améliorer le modèle ; encoder les variables catégorielles là où c'est nécessaire.

**Ce qui est sale dans les données, d'après l'aperçu.**

- La colonne double n'a pas de format stable : la compacité est écrite tantôt en chiffres, tantôt en toutes lettres en anglais, et le séparateur entre les deux informations change d'une ligne à l'autre (point-virgule, tiret bas, virgule).
- La hauteur totale vaut 3,5 sur quatre des cinq lignes de l'aperçu et -6,0 sur la cinquième : une hauteur négative est une valeur aberrante.
- La cible est affichée en nombre à virgule (0.0) : à convertir en entier.

**Réponse attendue :** un fichier de prédictions au format du fichier modèle, produit par une chaîne de traitement appliquée à l'identique aux données d'entraînement et de test. La métrique d'évaluation n'est pas connue.

**La démarche attendue, étape par étape.**

1. Lire les trois fichiers, afficher les types, compter les valeurs manquantes, regarder les valeurs distinctes de la colonne de texte. Ouvrir le fichier modèle de soumission : il fixe les colonnes, leur ordre et le nombre de lignes à rendre.
2. Écrire **une fonction** de nettoyage qui prend un tableau et rend un tableau, et l'appliquer aux deux fichiers. Dedans : séparer la colonne double en deux colonnes avec une expression régulière qui accepte les trois séparateurs ; convertir la compacité en nombre, avec une petite table de correspondance pour les nombres écrits en lettres ; convertir l'orientation en entier.
3. Traiter les valeurs aberrantes (la hauteur négative) par une règle écrite et commentée : valeur absolue si l'on juge que c'est une erreur de signe, ou valeur manquante sinon. L'énoncé disponible ne dit pas laquelle était attendue.
4. Supprimer les lignes d'entraînement où une valeur manque, comme demandé. Pour le fichier de test, vérifier d'abord le nombre de lignes du fichier modèle : s'il faut une prédiction par ligne de test, on impute au lieu de supprimer.
5. Ajouter au moins une variable : par exemple un rapport de deux surfaces. Peu importe laquelle, pourvu qu'elle soit calculée de la même façon sur les deux fichiers.
6. Encoder l'orientation, qui est une catégorie et non une grandeur, par des indicatrices, en ajustant l'encodeur sur les données d'entraînement seulement.
7. Entraîner un premier classifieur simple (régression logistique avec standardisation, ou forêt aléatoire), l'évaluer par validation croisée stratifiée, puis seulement ensuite comparer à un second modèle.
8. Réentraîner sur toutes les données d'entraînement, prédire le fichier de test, écrire le fichier de soumission, puis le **relire** : colonnes, nombre de lignes, valeurs 0 ou 1 en entiers, aucun manquant.

**Les pièges.**

- Nettoyer le fichier d'entraînement cellule par cellule, puis ne pas pouvoir refaire la même chose sur le fichier de test : d'où la fonction unique de l'étape 2.
- Ajuster un encodeur, une standardisation ou une imputation sur l'ensemble des données, test compris : c'est une fuite d'information.
- Supprimer des lignes du fichier de test et rendre un fichier de soumission trop court.
- Laisser passer une catégorie présente dans le test mais absente de l'entraînement : l'encodeur doit l'ignorer sans erreur.
- Viser le meilleur modèle avant d'avoir un fichier de soumission valide. Avec un budget de 30 minutes sur 82, l'ordre est : un fichier valide avec un modèle simple d'abord, les améliorations ensuite.

**À retenir.** Une chaîne complète tient en une fonction de nettoyage, un encodage ajusté sur l'entraînement seul, un modèle simple validé, et un fichier de sortie relu. C'est l'ordre qui protège : d'abord un résultat valide, ensuite la qualité.

**Leçon à revoir :** """ + C3 + r""", qui contient le corrigé complet sur un jeu de données de même forme. Pour les manipulations pandas : « Séance de code 1 : groupby, décalage, rangs ».
"""

FIN = r"""## Réflexes en 30 secondes

| Indice dans l'énoncé | Premier geste |
|---|---|
| « cochez toutes les affirmations justes » | une décision vrai ou faux par affirmation ; une définition exacte se coche ; « toujours », « uniquement », « quelle que soit » se réfutent par un contre-exemple |
| test d'adéquation | khi-deux : classes, loi discrète ou découpée ; Kolmogorov-Smirnov, Anderson-Darling : loi continue entièrement spécifiée |
| corrélation ou covariance, changement d'unité | \(\mathrm{Cov}(aX + b, Y) = a\,\mathrm{Cov}(X, Y)\) ; corrélation inchangée si \(a > 0\) ; les deux ne voient que le linéaire |
| asymétrie à droite | cas type : mode, médiane, moyenne dans cet ordre |
| « a priori proportionnel à », « le plus probable » | a priori fois vraisemblance, simplifier les constantes, comparer les candidats |
| bâton ou corde coupés au hasard, plus petit morceau | \(\mathbb{P}(\text{min} > t) = (1 - m\,t)^{m-1}\) pour \(m\) morceaux, espérance \(1/m^2\) |
| densité jointe, « sachant \(X = x\) » | marginale, puis quotient, puis reconnaître la loi ; la réponse ne dépend pas de la variable intégrée |
| constante de normalisation | intégrale égale à 1 ; variables séparées : produit de deux intégrales ; contrôler l'option choisie |
| « au moins un des deux » | écrire les quatre couples, barrer, compter |
| comparer des notes de deux épreuves | écart réduit \((x - \mu)/\sigma\) |
| forme de sortie d'une convolution | \(\lfloor (n + 2p - k)/s \rfloor + 1\) ; profondeur = nombre de filtres ; concaténation : mêmes hauteur et largeur, profondeurs additionnées |
| réglages d'entraînement | part d'entraînement majoritaire ; pas par époque fois lot = taille d'entraînement ; perte accordée au codage de la cible |
| surapprentissage, variance | régulariser ou simplifier ; jamais « augmenter la complexité » |
| grande dimension et interprétabilité | L1 (Lasso) : coefficients exactement nuls |
| classes déséquilibrées | découpage stratifié ; rappel et précision de la classe rare, pas l'exactitude |
| importance des variables | impureté : cumulée sur les nœuds, biaisée vers les variables à nombreux seuils ; permutation : baisse de performance, sur données non vues |
| ACP | centrer toujours, réduire si les unités diffèrent ; composantes = combinaisons linéaires orthogonales ; part \(\lambda_j / \sum \lambda\) |
| question de code avec fichiers | fichier modèle de soumission d'abord ; une fonction de nettoyage pour entraînement et test ; soumettre un modèle simple, améliorer ensuite |

## À savoir refaire sans notes

- La loi du plus petit des trois morceaux d'une corde : \(\mathbb{P}(L > t) = (1 - 3t)^2\) par translation du triangle, puis l'espérance par la formule de la queue.
- Densité marginale, densité conditionnelle et probabilité conditionnelle à partir de \(e^{-y/x}\,e^{-x}/x\).
- La constante de normalisation d'une densité à variables séparées sur un rectangle, avec son contrôle.
- Le problème des deux enfants, avec les quatre couples, et la raison pour laquelle le protocole d'observation change la réponse.
- \(\mathrm{Cov}(aX + b, Y)\) et \(\rho(aX + b, Y)\), et le contre-exemple \(Y = X^2\).
- La probabilité jointe « a priori fois vraisemblance » pour une loi de Poisson, et la comparaison de deux candidats par leur rapport.
- La forme de sortie d'un enchaînement convolution, convolution, concaténation, pooling, avec et sans marge.
- La décomposition de l'erreur quadratique en biais au carré plus variance, avec la preuve du terme croisé nul.
- Les définitions de l'importance par réduction d'impureté et de l'importance par permutation.
- Pourquoi L1 annule des coefficients et pas L2, dans le cas orthonormé (seuillage doux contre division par \(1 + \lambda\)).
- Les quatre affirmations sur l'ACP : orthogonalité, variance maximale, combinaisons linéaires, sensibilité à l'échelle, chacune avec sa justification.
- L'espérance et la variance d'une activation après dropout inversé.
"""

_CONTENU = "\n\n".join([
    INTRO.rstrip("\n"),
    CODE.rstrip("\n"),
    "## Section 2 : mathématiques (questions 2 à 11)\n\n" + _blocs(range(2, 12)),
    "## Section 3 : ML fondamental (questions 12 à 16)\n\n" + _blocs(range(12, 17)),
    "## Section 4 : ML avancé (questions 17 à 24)\n\n" + _blocs(range(17, 25)),
    FIN,
])

LECONS = [{
    "id": "L-DEBRIEF-T9", "type": "lecon", "section": "lecon", "theme": "Corrigé de l'OA QRT Quant Researcher",
    "difficulte": 3, "temps_cible_s": 5400, "contenu": _CONTENU, "controles": {},
    "verif": {"script": "verif/test_9_octobre.py"},
}]
CODES = []
QUESTIONS = []
