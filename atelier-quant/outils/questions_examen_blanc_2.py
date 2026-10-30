"""Examen blanc 2 : format d'un test en ligne de recherche quantitative (1 question de code, 10 de mathématiques, 5 de ML fondamental,
8 de ML avancé), sans leçon, en conditions réelles.

Questions toutes nouvelles : aucune ne reprend une question des banques existantes (test de positionnement, séances),
même avec d'autres nombres. Toutes les valeurs : verif/examen_blanc_2.py. Dans chaque question, l'option "a"
est la bonne ; l'ordre d'affichage est mélangé par la page.
"""
import questions_diagnostic as diag

ID = "examen-blanc-2"
TITRE = "Examen blanc 2"
VERIF = "verif/examen_blanc_2.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0, "pause": False, "je_ne_sais_pas": False}
SECTIONS = [
    {"id": "code", "nom": "Coding, data science", "budget_min": 24},
    {"id": "math", "nom": "Mathématiques", "budget_min": 25},
    {"id": "mlf", "nom": "Machine Learning fondamental", "budget_min": 7},
    {"id": "mla", "nom": "Machine Learning avancé", "budget_min": 16},
]
PRESENTATION = (
    "Deuxième examen blanc au format d'un test en ligne de recherche quantitative : 1 question de code, 10 de "
    "mathématiques, 5 de ML fondamental, 8 de ML avancé. À passer **en conditions réelles** : d'une traite, sans notes, "
    "sans recherche, sans pause (le bouton Pause est désactivé), seul. Toutes les questions sont nouvelles.\n\n"
    "Le chronomètre affiche la somme des budgets de section : "
    "24 minutes pour le code, 25 pour les mathématiques, 7 pour le ML fondamental, 16 pour le ML avancé. Cette somme "
    "laisse 10 minutes de marge sur un format de référence de 82 minutes ; l'objectif est de finir avant la fin du "
    "chronomètre.\n\n"
    "Règles à appliquer :\n\n"
    "- Code : écris une solution simple, lance les tests visibles, **soumets tout de suite**, puis améliore et soumets "
    "à nouveau. Un code jamais soumis ne rapporte rien.\n"
    "- Règle d'abandon : au-delà du double du temps cible affiché sur une question, choisis l'option la plus plausible "
    "après élimination, retiens le numéro de la question et passe à la suivante ; tu y reviendras avec la marge.\n"
    "- Il n'y a pas de « Je ne sais pas » : sans pénalité, une question laissée vide ne rapporte jamais "
    "plus qu'une réponse devinée."
)
AVERTISSEMENTS = [
    "Les questions de mathématiques et de ML sont présentées en choix multiple à une seule bonne réponse. Cherche la "
    "réponse avant de lire les options, pour t'entraîner aussi au format à réponse numérique.",
    "Ici le retour en arrière est permis entre toutes les questions et il n'y a pas de pénalité : ce sont les règles de "
    "cette plateforme, pas celles de tout test en ligne. Si un test interdit le retour vers une section quittée, il "
    "faut finir chaque section avant de passer à la suivante.",
    "La question de code est un notebook avec des fichiers de données, exécuté avec Python 3.9, pandas 1.4.2 et "
    "numpy 1.21.6, l'environnement des notebooks de la plateforme.",
]

CODES = [
    {
        "id": "EB2-C1", "type": "code", "section": "code", "theme": "pandas : transactions et cotations, jointure temporelle",
        "difficulte": 4, "temps_cible_s": 1200,
        "enonce": """## Déséquilibre des ordres par la règle du point milieu

Pour savoir si une transaction a été déclenchée par un acheteur ou par un vendeur, on la compare à la dernière cotation connue juste avant elle. Une cotation donne le meilleur prix d'achat proposé (`achat` : le prix le plus haut auquel un acheteur est prêt à acheter) et le meilleur prix de vente proposé (`vente` : le prix le plus bas auquel un vendeur est prêt à vendre). Une transaction au-dessus du milieu de ces deux prix a été déclenchée par un acheteur (transaction à l'achat) ; en dessous, par un vendeur (transaction à la vente).

Dans le notebook, définis la fonction `desequilibre(transactions, cotations)` qui renvoie une `Series` pandas.

### Entrées

`transactions` est un DataFrame dont les colonnes sont exactement :

| Colonne | Type | Contenu |
|---|---|---|
| `temps` | entier (`int64`) | instant de la transaction, en millisecondes depuis l'ouverture |
| `actif` | chaîne | identifiant de l'actif |
| `prix` | flottant, strictement positif | prix de la transaction |
| `quantite` | entier (`int64`), au moins 1 | nombre de titres échangés |

`cotations` est un DataFrame dont les colonnes sont exactement :

| Colonne | Type | Contenu |
|---|---|---|
| `temps` | entier (`int64`) | instant de la cotation, en millisecondes depuis l'ouverture |
| `actif` | chaîne | identifiant de l'actif |
| `achat` | flottant ou `NaN` | meilleur prix d'achat proposé |
| `vente` | flottant ou `NaN` | meilleur prix de vente proposé |

Garanties : lignes dans un ordre **quelconque** et index quelconque, dans les deux DataFrames ; au plus une cotation par couple (`actif`, `temps`) ; plusieurs transactions peuvent avoir le même `temps` ; chacun des deux DataFrames peut être vide (zéro ligne, avec les colonnes et les types ci-dessus).

### Calcul demandé

1. **Cotations valides.** Une cotation est valide si `achat` et `vente` sont tous deux non manquants et si `achat < vente` (inégalité stricte). Les cotations non valides sont ignorées, comme si elles n'existaient pas.
2. **Cotation de référence.** Pour une transaction d'instant `t` sur l'actif `a`, c'est, parmi les cotations valides du même actif `a` dont l'instant est **strictement inférieur** à `t`, celle dont l'instant `tc` est le plus grand. Il n'y a pas de cotation de référence s'il n'existe aucune telle cotation, ou si `t - tc > 5000` (cotation de plus de 5 secondes). Une cotation de même instant que la transaction n'est pas antérieure à elle.
3. **Signe.** Avec `m = (achat + vente) / 2`, le milieu de la cotation de référence : le signe vaut +1 si `prix > m`, -1 si `prix < m`, et 0 si `prix == m` ou s'il n'y a pas de cotation de référence.
4. **Déséquilibre d'un actif.** Pour chaque actif présent dans `transactions`, `V+` est la somme des quantités de ses transactions de signe +1, et `V-` la somme des quantités de ses transactions de signe -1. Son déséquilibre vaut `(V+ - V-) / (V+ + V-)`, ou `NaN` si `V+ + V-` vaut 0.
5. **Résultat.** Une `Series` de flottants dont l'index est la liste **triée** (ordre croissant des chaînes) des actifs distincts de `transactions`, et dont les valeurs sont leurs déséquilibres. Un actif présent seulement dans `cotations` n'y figure pas. Si `transactions` est vide, le résultat est une `Series` vide.

Tous les prix des tests sont des multiples de 0,25, représentés exactement en flottant : les comparaisons avec le milieu sont exactes. Tolérance de comparaison des résultats : `1e-9`. Limite de temps : 10 secondes par test.

### Exemple

`transactions` :

| temps | actif | prix | quantite |
|---|---|---|---|
| 3000 | A | 100.5 | 10 |
| 3000 | B | 50.0 | 5 |
| 4000 | C | 20.0 | 1 |
| 5000 | A | 99.75 | 4 |
| 6500 | A | 100.25 | 7 |
| 8000 | B | 50.25 | 6 |
| 9200 | B | 51.0 | 2 |
| 9500 | B | 51.5 | 3 |

`cotations` :

| temps | actif | achat | vente |
|---|---|---|---|
| 1000 | A | 99.5 | 100.5 |
| 2500 | A | 101.0 | 100.0 |
| 3000 | B | 50.0 | 50.5 |
| 9000 | B | 51.0 | 51.5 |

- La cotation de A à 2500 est croisée (`achat > vente`) : elle n'est pas valide.
- A à 3000 : référence A à 1000 (milieu 100), prix 100,5 : signe +1. A à 5000 : même référence, 4000 ms avant, prix 99,75 : signe -1. A à 6500 : la cotation de 1000 date de 5500 ms, trop ancienne : signe 0. Déséquilibre de A : (10 - 4) / (10 + 4), soit @C1_A:4@ à 4 décimales.
- B à 3000 : la seule cotation de B à un instant inférieur ou égal est au même instant : pas de référence, signe 0. B à 8000 : référence B à 3000, exactement 5000 ms avant, donc retenue ; milieu 50,25, égal au prix : signe 0. B à 9200 : référence B à 9000 (milieu 51,25), prix 51 : signe -1. B à 9500 : prix 51,5 : signe +1. Déséquilibre de B : (3 - 2) / (3 + 2) = @C1_B:1@.
- C n'a aucune cotation : `NaN`.

La fonction renvoie la `Series` d'index `A`, `B`, `C` et de valeurs @C1_A:4@ (à 4 décimales), @C1_B:1@ et `NaN`.

### Fonctionnement

- Le notebook s'exécute avec Python 3.9 et pandas 1.4.2, l'environnement des notebooks de la plateforme.
- Les fichiers `transactions.csv` et `cotations.csv` (mêmes colonnes) sont dans le répertoire courant pour explorer : `pd.read_csv("transactions.csv")`.
- **Tests visibles** exécute 3 tests sur ta fonction, dont l'exemple ci-dessus. **Soumettre** exécute les @C1_nb_caches@ tests cachés : seul le nombre de réussites s'affiche, et la dernière soumission compte. Un code jamais soumis ne rapporte rien.
- Raccourcis : Maj+Entrée exécute la cellule et passe à la suivante, Ctrl+Entrée l'exécute sans changer de cellule.
""",
        "dossier": "coding/examen2",
        "module_tests": "coding/examen2/tests.py",
        "fonction": "desequilibre",
        "cellules_initiales": [
            "import numpy as np\nimport pandas as pd\n\ntransactions = pd.read_csv(\"transactions.csv\")\n"
            "cotations = pd.read_csv(\"cotations.csv\")\ntransactions.head()",
            "def desequilibre(transactions: pd.DataFrame, cotations: pd.DataFrame) -> pd.Series:\n"
            "    # À compléter\n    return pd.Series(dtype=float)",
            "desequilibre(transactions, cotations)",
        ],
        "tests_visibles": 3,
        "tests_caches": 15,
        "fichier_correction": "coding/examen2/_correction.py",
        "verif": {"script": "coding/examen2/tests.py"},
    },
]

QUESTIONS = [
    # =============================================================== MATHÉMATIQUES
    {
        "id": "EB2-M01", "section": "math", "theme": "Chaîne de Markov, temps de retour", "difficulte": 4, "temps_cible_s": 180,
        "enonce": r"""Un modèle de régimes de marché décrit chaque jour de bourse par un état parmi trois : C (calme), N (normal), A (agité). La suite \((X_t)_{t \ge 0}\), où \(X_t\) est l'état du jour \(t\), est une chaîne de Markov homogène : sachant tout le passé jusqu'au jour \(t\), l'état du jour \(t + 1\) ne dépend que de \(X_t\), avec les probabilités de transition du tableau (ligne : état du jour \(t\) ; colonne : état du jour \(t + 1\)).

| depuis | vers C | vers N | vers A |
|---|---|---|---|
| C | 0,8 | 0,2 | 0 |
| N | 0,1 | 0,7 | 0,2 |
| A | 0 | 0,5 | 0,5 |

Le jour 0 est agité (\(X_0 = \mathrm{A}\)). Soit \(T = \min\{t \ge 1 : X_t = \mathrm{A}\}\) le premier jour agité après le jour 0 (\(T = 1\) si le jour 1 est agité). Que vaut \(\mathbb{E}[T]\) ?""",
        "options": [
            {"id": "a", "texte": "@M01_exact@", "explication": r"Oui : \(1 + 0{,}5\,h_{\mathrm{N}}\), où \(h_{\mathrm{N}}\) est le temps moyen pour atteindre A depuis N ; c'est aussi l'inverse de la probabilité stationnaire de A."},
            {"id": "b", "texte": "@M01_d_sejour@", "explication": r"C'est \(1/(1 - 0{,}5)\), la durée moyenne d'un séjour en A (nombre de jours consécutifs passés en A). La question demande le temps de retour : quand la chaîne quitte A, elle passe par N et met en moyenne @M01_hN:1@ jours à revenir."},
            {"id": "c", "texte": "@M01_d_un_plus_hN:1@", "explication": r"C'est \(1 + h_{\mathrm{N}}\), où \(h_{\mathrm{N}} = @M01_hN:1m@\) est le temps moyen pour atteindre A depuis N : on a supposé que la chaîne quitte toujours A au premier pas. Avec probabilité 0,5, elle reste en A et \(T = 1\) ; le terme \(h_{\mathrm{N}}\) ne compte qu'avec le poids 0,5."},
            {"id": "d", "texte": "@M01_d_uniforme@", "explication": r"C'est l'inverse de 1/3 : le lemme de Kac appliqué à une loi stationnaire supposée uniforme sur les trois états. La loi stationnaire de cette chaîne n'est pas uniforme : la probabilité stationnaire de A vaut \(@M01_piA_num@/@M01_piA_den@\)."},
        ],
        "solution": r"""Réponse : \(\mathbb{E}[T] = @M01_exact_num@/@M01_exact_den@ = @M01_exact:2m@\) jours.

Notations : pour un état \(x\) parmi C et N, \(h_x\) est l'espérance du nombre de jours nécessaires pour atteindre A en partant de \(x\), c'est-à-dire \(\mathbb{E}[\min\{t \ge 1 : X_t = \mathrm{A}\} \mid X_0 = x]\).

Étape 0, ces espérances sont finies. Depuis n'importe quel état, la chaîne atteint A dans les deux jours suivants avec probabilité au moins \(0{,}2 \times 0{,}2 = @M01_p_deux_pas:2m@\) (le pire cas est C, par le chemin C, N, A). Par la propriété de Markov, la probabilité de ne pas avoir atteint A après \(2k\) jours est au plus \((1 - @M01_p_deux_pas:2m@)^k\), et l'espérance, somme des probabilités de queue, est finie.

Étape 1, premier pas depuis N. Le premier jour compte pour 1 ; avec probabilité 0,2 la chaîne est en A et c'est fini ; avec probabilité 0,7 elle est en N, avec probabilité 0,1 en C, et, par la propriété de Markov, elle repart comme une chaîne neuve depuis ce nouvel état. Donc \(h_{\mathrm{N}} = 1 + 0{,}1\,h_{\mathrm{C}} + 0{,}7\,h_{\mathrm{N}}\). De même depuis C : \(h_{\mathrm{C}} = 1 + 0{,}8\,h_{\mathrm{C}} + 0{,}2\,h_{\mathrm{N}}\).

Étape 2, résolution. La seconde équation donne \(0{,}2\,h_{\mathrm{C}} = 1 + 0{,}2\,h_{\mathrm{N}}\), soit \(h_{\mathrm{C}} = 5 + h_{\mathrm{N}}\). En reportant dans la première : \(0{,}3\,h_{\mathrm{N}} = 1 + 0{,}1\,(5 + h_{\mathrm{N}})\), donc \(0{,}2\,h_{\mathrm{N}} = 1{,}5\), \(h_{\mathrm{N}} = @M01_hN:1m@\) et \(h_{\mathrm{C}} = @M01_hC:1m@\).

Étape 3, premier pas depuis A. Avec probabilité 0,5 le jour 1 est agité et \(T = 1\) ; avec probabilité 0,5 il est normal et \(T\) vaut 1 plus le temps d'atteinte de A depuis N. Donc \(\mathbb{E}[T] = 1 + 0{,}5 \times 0 + 0{,}5 \times h_{\mathrm{N}} = 1 + 0{,}5 \times @M01_hN:1m@ = @M01_exact:2m@\).

Contrôle par le lemme de Kac (théorème admis ici : pour une chaîne irréductible sur un ensemble fini d'états, l'espérance du temps de retour en un état est l'inverse de sa probabilité stationnaire). La loi stationnaire \((\pi_{\mathrm{C}}, \pi_{\mathrm{N}}, \pi_{\mathrm{A}})\) vérifie \(\pi_{\mathrm{C}} = 0{,}8\,\pi_{\mathrm{C}} + 0{,}1\,\pi_{\mathrm{N}}\), donc \(\pi_{\mathrm{N}} = 2\,\pi_{\mathrm{C}}\), et \(\pi_{\mathrm{A}} = 0{,}2\,\pi_{\mathrm{N}} + 0{,}5\,\pi_{\mathrm{A}}\), donc \(\pi_{\mathrm{A}} = 0{,}4\,\pi_{\mathrm{N}}\). Avec une somme égale à 1 : \(\pi = (@M01_piC_num@/@M01_piC_den@, @M01_piN_num@/@M01_piN_den@, @M01_piA_num@/@M01_piA_den@)\), et \(1/\pi_{\mathrm{A}} = @M01_exact_num@/@M01_exact_den@\). Les deux méthodes concordent.

Simulation (script) : temps de retour moyen @M01_mc:3@.""",
    },
    {
        "id": "EB2-M02", "section": "math", "theme": "Urne de Pólya, échangeabilité, martingale", "difficulte": 4, "temps_cible_s": 150,
        "enonce": r"""Une urne contient au départ 1 boule rouge et 1 boule bleue. À chaque tirage, on tire une boule uniformément au hasard parmi celles de l'urne, puis on la remet **avec** une boule supplémentaire de la même couleur (urne de Pólya). On effectue 8 tirages.

Quelle est la probabilité qu'exactement 3 des 8 boules tirées soient rouges ?""",
        "options": [
            {"id": "a", "texte": "@M02_exact_num@/@M02_exact_den@ ≈ @M02_exact:3@", "explication": r"""Oui : le nombre de rouges tirées est uniforme sur \(\{0, 1, \dots, 8\}\)."""},
            {"id": "b", "texte": "@M02_d_binomiale_num@/@M02_d_binomiale_den@ ≈ @M02_d_binomiale:3@", "explication": r"""C'est \(\binom{8}{3}/2^8\), la loi binomiale de 8 tirages indépendants de probabilité 1/2. Les tirages ne sont pas indépendants : chaque rouge tirée rend la rouge suivante plus probable."""},
            {"id": "c", "texte": "@M02_d_huit_num@/@M02_d_huit_den@ = @M02_d_huit:3@", "explication": "C'est une loi uniforme sur 8 valeurs. Le nombre de rouges tirées peut valoir 0, 1, ..., 8 : il y a 9 valeurs possibles."},
            {"id": "d", "texte": "@M02_d_suite_num@/@M02_d_suite_den@ ≈ @M02_d_suite:3@", "explication": r"""C'est la probabilité d'une suite particulière de tirages contenant 3 rouges, par exemple trois rouges puis cinq bleues. Il faut la multiplier par le nombre \(\binom{8}{3}\) de positions possibles des rouges."""},
        ],
        "solution": r"""Réponse : \(@M02_exact_num@/@M02_exact_den@ \approx @M02_exact:3m@\).

Notations : \(n = 8\) tirages, \(k = 3\) rouges voulues.

Étape 1, probabilité d'une suite donnée. Considérons une suite de tirages contenant \(k\) rouges et \(n - k\) bleues, dans un ordre fixé. Au \(m\)-ième tirage, l'urne contient \(m + 1\) boules (2 au départ, une de plus par tirage) : les dénominateurs successifs font le produit \(2 \times 3 \times \dots \times (n + 1) = (n + 1)!\). Quand on tire la \(j\)-ième rouge, l'urne contient \(j\) rouges (1 au départ, plus les \(j - 1\) déjà ajoutées) : les numérateurs des tirages rouges font \(1 \times 2 \times \dots \times k = k!\), ceux des tirages bleus \((n - k)!\). La probabilité de la suite est donc \(k!\,(n - k)!/(n + 1)!\), quel que soit l'ordre des couleurs (échangeabilité).

Étape 2 : il y a \(\binom{n}{k}\) suites contenant \(k\) rouges, toutes de même probabilité, donc \(\mathbb{P}(k \text{ rouges}) = \binom{n}{k}\,\dfrac{k!\,(n - k)!}{(n + 1)!} = \dfrac{1}{n + 1}\), soit \(1/9\) ici.

Lien avec les martingales : la proportion de rouges dans l'urne après chaque tirage est une martingale (son espérance conditionnelle au tirage suivant est égale à sa valeur actuelle) ; elle converge presque sûrement, et sa limite suit la loi uniforme sur \([0, 1]\), cohérente avec la loi uniforme du nombre de rouges. Ce dernier point est cité, pas démontré ici.

Vérifications (script) : énumération exacte des \(2^8\) suites, et simulation : @M02_mc:4@.""",
    },
    {
        "id": "EB2-M03", "section": "math", "theme": "Dénombrement, arrangements avec contrainte", "difficulte": 3, "temps_cible_s": 150,
        "enonce": r"""On appelle anagramme du mot ARBITRAGE toute suite de ses 9 lettres (A, R, B, I, T, R, A, G, E), chaque lettre étant utilisée exactement une fois, que la suite ait un sens ou non ; deux anagrammes sont identiques si elles s'écrivent de la même façon (échanger les deux A, ou les deux R, ne donne pas une nouvelle anagramme).

Combien d'anagrammes de ARBITRAGE ne contiennent pas deux R côte à côte ?""",
        "options": [
            {"id": "a", "texte": "@M03_exact@", "explication": "Oui : @M03_total@ anagrammes en tout, moins les @M03_adjacents@ où les deux R se touchent."},
            {"id": "b", "texte": "@M03_total@", "explication": r"""C'est le nombre total d'anagrammes, \(9!/(2!\,2!)\) : la contrainte sur les R n'a pas été appliquée."""},
            {"id": "c", "texte": "@M03_adjacents@", "explication": r"""C'est le nombre d'anagrammes où les deux R sont **côte à côte** (bloc RR traité comme une seule lettre, \(8!/2!\)). La question demande le complémentaire."""},
            {"id": "d", "texte": "@M03_d_a_distincts@", "explication": r"""C'est \(9!/2! - 8!\) : la répétition des deux A a été oubliée. Échanger les deux A ne crée pas une nouvelle anagramme, il faut diviser chaque décompte par \(2!\)."""},
        ],
        "solution": r"""Réponse : @M03_exact@.

Étape 1, total. Les 9 lettres comptent deux A et deux R. Parmi les \(9!\) ordres des 9 lettres rendues distinctes, chaque anagramme apparaît \(2!\,2!\) fois (on peut permuter les A entre eux et les R entre eux sans changer l'écriture). Total : \(9!/(2!\,2!) = @M03_total@\).

Étape 2, R côte à côte. On colle les deux R en un bloc RR. Il reste 8 objets (RR, deux A, B, I, T, G, E), dont deux A identiques : \(8!/2! = @M03_adjacents@\) suites. Chaque anagramme avec les R adjacents correspond à exactement une de ces suites (le bloc est à la place des deux R).

Étape 3, complémentaire : \(@M03_total@ - @M03_adjacents@ = @M03_exact@\).

Contrôle (script) : l'énumération des permutations des 9 lettres, sans doublon, donne les mêmes trois nombres.

Méthode alternative : placer d'abord les 7 autres lettres (\(7!/2!\) façons), puis choisir 2 des 8 intervalles qu'elles délimitent pour les R (\(\binom{8}{2}\)) : même résultat. Piège : oublier une répétition (A ou R), ou compter le bloc RR comme deux objets.""",
    },
    {
        "id": "EB2-M04", "section": "math", "theme": "Espérance et variance conditionnelles", "difficulte": 3, "temps_cible_s": 150,
        "enonce": r"""Le nombre \(N\) d'ordres reçus par un courtier en une minute est modélisé ainsi : l'intensité \(\Lambda\) de la minute est une variable aléatoire de loi exponentielle d'espérance 4 (densité \(\tfrac14 e^{-\lambda/4}\) pour \(\lambda > 0\)) et, sachant \(\Lambda = \lambda\), \(N\) suit la loi de Poisson de paramètre \(\lambda\).

Que vaut \(\mathrm{Var}(N)\) ?""",
        "options": [
            {"id": "a", "texte": "@M04_exact@", "explication": r"Oui : \(\mathbb{E}[\Lambda] + \mathrm{Var}(\Lambda) = @M04_E@ + @M04_var_lambda@\)."},
            {"id": "b", "texte": "@M04_E@", "explication": r"C'est \(\mathbb{E}[N]\), qui serait aussi la variance si l'intensité était fixe et égale à 4 (loi de Poisson). Une intensité aléatoire ajoute sa propre variance, \(\mathrm{Var}(\Lambda)\)."},
            {"id": "c", "texte": "@M04_var_lambda@", "explication": r"C'est \(\mathrm{Var}(\Lambda)\) seule : il manque la dispersion de Poisson autour de l'intensité, \(\mathbb{E}[\mathrm{Var}(N \mid \Lambda)] = \mathbb{E}[\Lambda] = @M04_E@\)."},
            {"id": "d", "texte": "@M04_d_somme_moyennes@", "explication": r"C'est \(4 + 4\) : la variance d'une loi exponentielle a été prise égale à son espérance, comme pour une loi de Poisson. Pour une exponentielle d'espérance \(m\), la variance vaut \(m^2\)."},
        ],
        "solution": r"""Réponse : \(\mathrm{Var}(N) = @M04_exact@\).

Étape 1, formule de la variance totale : \(\mathrm{Var}(N) = \mathbb{E}[\mathrm{Var}(N \mid \Lambda)] + \mathrm{Var}(\mathbb{E}[N \mid \Lambda])\). Preuve : \(\mathbb{E}[N^2] = \mathbb{E}\big[\mathbb{E}[N^2 \mid \Lambda]\big] = \mathbb{E}\big[\mathrm{Var}(N \mid \Lambda)\big] + \mathbb{E}\big[\mathbb{E}[N \mid \Lambda]^2\big]\) et \(\mathbb{E}[N] = \mathbb{E}\big[\mathbb{E}[N \mid \Lambda]\big]\) ; en retranchant \(\mathbb{E}[N]^2\), les deux derniers termes forment la variance de \(\mathbb{E}[N \mid \Lambda]\).

Étape 2, moments conditionnels : une loi de Poisson de paramètre \(\lambda\) a pour espérance et pour variance \(\lambda\). Donc \(\mathbb{E}[N \mid \Lambda] = \mathrm{Var}(N \mid \Lambda) = \Lambda\).

Étape 3, moments de \(\Lambda\) : pour une exponentielle d'espérance \(m = 4\), \(\mathbb{E}[\Lambda^2] = \int_0^\infty \lambda^2\,\tfrac1m e^{-\lambda/m}\,d\lambda = 2m^2\) (deux intégrations par parties), donc \(\mathrm{Var}(\Lambda) = 2m^2 - m^2 = m^2 = @M04_var_lambda@\).

Étape 4 : \(\mathrm{Var}(N) = \mathbb{E}[\Lambda] + \mathrm{Var}(\Lambda) = @M04_E@ + @M04_var_lambda@ = @M04_exact@\).

Contrôle : en intégrant sur \(\lambda\) (script), \(\mathbb{P}(N = k) = \tfrac15 \big(\tfrac45\big)^k\) pour \(k \ge 0\) : \(N\) suit une loi géométrique sur \(\{0, 1, 2, \dots\}\), d'espérance @M04_E@ et de variance \((4/5)/(1/5)^2 = @M04_exact@\). Simulation : \(\mathbb{E}[N] \approx @M04_E_mc:3m@\) et \(\mathbb{E}[N^2] \approx @M04_E2_mc:2m@\), pour des valeurs exactes @M04_E@ et @M04_E2@.

Lecture : une variance de comptage supérieure à l'espérance (surdispersion) signale une intensité qui varie d'une minute à l'autre ; un modèle de Poisson à intensité fixe la sous-estimerait.""",
    },
    {
        "id": "EB2-M05", "section": "math", "theme": "Processus de Poisson, maximum d'exponentielles", "difficulte": 3, "temps_cible_s": 120,
        "enonce": r"""Les ordres d'achat arrivent selon un processus de Poisson d'intensité 3 par minute, et les ordres de vente selon un processus de Poisson d'intensité 2 par minute, indépendant du premier. On observe à partir de l'instant 0.

Quelle est l'espérance, en minutes, du temps d'attente jusqu'à avoir reçu au moins un ordre d'achat **et** au moins un ordre de vente ?""",
        "options": [
            {"id": "a", "texte": "@M05_exact_num@/@M05_exact_den@ ≈ @M05_exact:3@", "explication": r"Oui : \(1/3 + 1/2 - 1/5\), l'espérance du maximum de deux exponentielles indépendantes."},
            {"id": "b", "texte": "@M05_d_max_moyennes_num@/@M05_d_max_moyennes_den@ = @M05_d_max_moyennes:3@", "explication": r"C'est le plus grand des deux temps moyens, \(\max(1/3, 1/2)\). L'espérance d'un maximum dépasse le maximum des espérances (la fonction max est convexe) : avec une probabilité non nulle, c'est l'ordre d'achat qui arrive le dernier."},
            {"id": "c", "texte": "@M05_d_somme_num@/@M05_d_somme_den@ ≈ @M05_d_somme:3@", "explication": r"C'est \(1/3 + 1/2\), qui revient à attendre le premier achat, puis seulement ensuite le premier vente. Les deux attentes se déroulent en même temps : il faut retirer leur partie commune, l'espérance du minimum."},
            {"id": "d", "texte": "@M05_d_min_num@/@M05_d_min_den@ = @M05_d_min:3@", "explication": r"C'est l'espérance du minimum, le temps jusqu'au premier ordre quel que soit son type. La question demande d'avoir reçu les deux types."},
        ],
        "solution": r"""Réponse : \(@M05_exact_num@/@M05_exact_den@ \approx @M05_exact:3m@\) minute.

Notations : \(A\) est l'instant du premier ordre d'achat, \(B\) celui du premier ordre de vente ; le temps demandé est \(\max(A, B)\).

Étape 1, lois. \(A > t\) signifie qu'aucun ordre d'achat n'arrive dans \([0, t]\), dont le nombre suit une loi de Poisson de paramètre \(3t\) : \(\mathbb{P}(A > t) = e^{-3t}\). Donc \(A\) suit la loi exponentielle de paramètre 3, d'espérance 1/3 ; de même \(B\) est exponentielle de paramètre 2, d'espérance 1/2, et \(A\), \(B\) sont indépendants.

Étape 2, minimum. Par indépendance, \(\mathbb{P}(\min(A, B) > t) = \mathbb{P}(A > t)\,\mathbb{P}(B > t) = e^{-5t}\) : le minimum est exponentiel de paramètre 5, d'espérance 1/5.

Étape 3, l'identité \(\max(a, b) = a + b - \min(a, b)\), vraie pour tous réels \(a\), \(b\), et la linéarité de l'espérance donnent \(\mathbb{E}[\max(A, B)] = 1/3 + 1/2 - 1/5 = @M05_exact_num@/@M05_exact_den@\).

Contrôle direct (script) : \(\mathbb{E}[\max(A, B)] = \int_0^\infty \big(1 - (1 - e^{-3t})(1 - e^{-2t})\big)\,dt = @M05_exact_num@/@M05_exact_den@\), par la formule \(\mathbb{E}[Y] = \int_0^\infty \mathbb{P}(Y > t)\,dt\) pour \(Y \ge 0\). Simulation : @M05_mc:4@.""",
    },
    {
        "id": "EB2-M06", "section": "math", "theme": "Conditionnement, valeurs singulières d'une matrice non symétrique", "difficulte": 4, "temps_cible_s": 150,
        "enonce": r"""Soit \(\mathbf{A}\) la matrice \(2 \times 2\) de lignes \((1, 1)\) et \((0, 1)\). Pour une matrice inversible \(\mathbf{M}\), on note \(\lVert \mathbf{M} \rVert_2 = \max_{\mathbf{x} \ne \mathbf{0}} \lVert \mathbf{M}\mathbf{x} \rVert/\lVert \mathbf{x} \rVert\) sa norme d'opérateur associée à la norme euclidienne, et \(\kappa(\mathbf{M}) = \lVert \mathbf{M} \rVert_2\,\lVert \mathbf{M}^{-1} \rVert_2\) son conditionnement.

Que vaut \(\kappa(\mathbf{A})\), arrondi à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@M06_exact:3@", "explication": r"""Oui : \((3 + \sqrt5)/2\), rapport des valeurs singulières extrêmes."""},
            {"id": "b", "texte": "@M06_d_valeurs_propres:3@", "explication": r"""C'est le rapport des valeurs propres de \(\mathbf{A}\), toutes deux égales à 1. Pour une matrice non symétrique, le conditionnement se lit sur les valeurs singulières (racines des valeurs propres de \(\mathbf{A}^\top\mathbf{A}\)), pas sur les valeurs propres."""},
            {"id": "c", "texte": "@M06_d_carre:3@", "explication": r"""C'est le rapport des valeurs propres de \(\mathbf{A}^\top\mathbf{A}\), c'est-à-dire \(\kappa(\mathbf{A})^2\) : il manque la racine carrée qui transforme ces valeurs propres en valeurs singulières de \(\mathbf{A}\)."""},
            {"id": "d", "texte": "@M06_d_infini:3@", "explication": r"""C'est le conditionnement pour la norme d'opérateur associée à la norme du maximum (plus grande somme des valeurs absolues d'une ligne : 2 pour \(\mathbf{A}\) et 2 pour \(\mathbf{A}^{-1}\)). L'énoncé fixe la norme euclidienne."""},
        ],
        "solution": r"""Réponse : \(\kappa(\mathbf{A}) = (3 + \sqrt5)/2 \approx @M06_exact:3m@\).

Étape 1, norme d'opérateur et valeurs singulières. \(\lVert \mathbf{A}\mathbf{x} \rVert^2 = \mathbf{x}^\top \mathbf{A}^\top\mathbf{A}\,\mathbf{x}\), et \(\mathbf{A}^\top\mathbf{A}\) est symétrique semi-définie positive, de valeurs propres \(\lambda_{\max} \ge \lambda_{\min} \ge 0\). En décomposant \(\mathbf{x}\) dans une base orthonormée de vecteurs propres (théorème spectral), \(\mathbf{x}^\top \mathbf{A}^\top\mathbf{A}\,\mathbf{x}/\lVert \mathbf{x} \rVert^2\) est une moyenne pondérée des valeurs propres, maximale en un vecteur propre de \(\lambda_{\max}\). Donc \(\lVert \mathbf{A} \rVert_2 = \sqrt{\lambda_{\max}}\), la plus grande valeur singulière.

Étape 2, inverse. \(\mathbf{A}^{-1}\) a pour valeurs singulières les inverses de celles de \(\mathbf{A}\) (si \(\mathbf{A} = \mathbf{U}\mathbf{D}\mathbf{V}^\top\), alors \(\mathbf{A}^{-1} = \mathbf{V}\mathbf{D}^{-1}\mathbf{U}^\top\)), donc \(\lVert \mathbf{A}^{-1} \rVert_2 = 1/\sqrt{\lambda_{\min}}\) et \(\kappa(\mathbf{A}) = \sqrt{\lambda_{\max}/\lambda_{\min}}\).

Étape 3, calcul. \(\mathbf{A}^\top\mathbf{A}\) a pour lignes \((1, 1)\) et \((1, 2)\) : trace 3, déterminant 1, donc ses valeurs propres sont les racines de \(\lambda^2 - 3\lambda + 1\), soit \((3 \pm \sqrt5)/2\), environ @M06_vp_max:3@ et @M06_vp_min:3@. Comme leur produit vaut 1, \(\lambda_{\max}/\lambda_{\min} = \lambda_{\max}^2\) et \(\kappa(\mathbf{A}) = \lambda_{\max} = (3 + \sqrt5)/2 \approx @M06_exact:3m@\).

Piège : les valeurs propres de \(\mathbf{A}\) valent toutes deux 1, et pourtant \(\mathbf{A}\) amplifie certaines erreurs relatives d'un facteur @M06_exact:3@. Pour une matrice symétrique définie positive, valeurs propres et valeurs singulières coïncident, et le conditionnement est le rapport des valeurs propres extrêmes ; pas pour une matrice quelconque. Contrôle (script) : sympy et numpy donnent la même valeur.""",
    },
    {
        "id": "EB2-M07", "section": "math", "theme": "Méthode delta, approximation d'un écart-type", "difficulte": 3, "temps_cible_s": 150,
        "enonce": r"""Soient \(X_1, \dots, X_{100}\) des variables aléatoires indépendantes, de même loi, strictement positives, d'espérance \(\mu = 2\) et de variance \(\sigma^2 = 4\), et \(\bar X = (X_1 + \dots + X_{100})/100\) leur moyenne.

Par la méthode delta (développement de Taylor à l'ordre 1 de la fonction \(\ln\) autour de \(\mu\)), quelle est l'approximation de l'écart-type de \(\ln \bar X\) ?""",
        "options": [
            {"id": "a", "texte": "@M07_exact:2@", "explication": r"""Oui : \(\sigma/(\mu\sqrt n)\), la dérivée de \(\ln\) en 2 valant 1/2."""},
            {"id": "b", "texte": "@M07_d_sans_derivee:2@", "explication": r"""C'est \(\sigma/\sqrt n\), l'écart-type de \(\bar X\) lui-même : la dérivée \(1/\mu\) de \(\ln\) en \(\mu\), qui convertit une variation de \(\bar X\) en variation de \(\ln \bar X\), a été oubliée."""},
            {"id": "c", "texte": "@M07_d_variance:2@", "explication": r"""C'est l'approximation de la **variance** de \(\ln \bar X\), \(\sigma^2/(\mu^2 n)\). L'écart-type en est la racine carrée."""},
            {"id": "d", "texte": "@M07_d_derivee_seconde:2@", "explication": r"""C'est \(\sigma/(\mu^2\sqrt n)\) : la dérivée de \(\ln\) a été prise égale à \(1/\mu^2\), qui est (au signe près) sa dérivée seconde. La dérivée de \(\ln x\) est \(1/x\)."""},
        ],
        "solution": r"""Réponse : \(\sigma/(\mu\sqrt n) = 2/(2 \times 10) = @M07_exact:2m@\).

Notations : \(n = 100\), \(g = \ln\), \(g'(x) = 1/x\).

Étape 1 : \(\bar X\) a pour espérance \(\mu\) et pour écart-type \(\sigma/\sqrt n = @M07_sd_moyenne:1m@\), petit devant \(\mu = 2\) : \(\bar X\) reste proche de \(\mu\), ce qui justifie un développement local.

Étape 2, Taylor à l'ordre 1 : \(\ln \bar X \approx \ln \mu + (\bar X - \mu)/\mu\). Le second membre est une fonction affine de \(\bar X\), de variance \(\mathrm{Var}(\bar X)/\mu^2 = \sigma^2/(\mu^2 n)\).

Étape 3 : écart-type approché \(\sigma/(\mu\sqrt n) = @M07_exact:2m@\).

Statut : c'est une approximation, pas une égalité. Elle devient exacte au premier ordre quand \(n \to \infty\) : \(\sqrt n\,(\ln \bar X - \ln \mu)\) converge en loi vers une normale de variance \(\sigma^2/\mu^2\) (théorème central limite et méthode delta, admis). Illustration (script) : pour une loi exponentielle d'espérance 2 (donc de variance 4), l'écart-type simulé de \(\ln \bar X\) vaut @M07_sd_sim:4@. Lecture : l'écart-type de \(\ln \bar X\) est l'écart-type relatif de \(\bar X\), \((\sigma/\sqrt n)/\mu\).""",
    },
    {
        "id": "EB2-M08", "section": "math", "theme": "Test d'une proportion, p-value", "difficulte": 2, "temps_cible_s": 120,
        "enonce": r"""Un signal prédit chaque jour le sens (hausse ou baisse) du rendement du lendemain. Sur \(n = 400\) jours, il a raison 220 fois. On modélise les 400 jours comme indépendants, chacun avec la même probabilité de succès \(p\), et l'on teste \(H_0 : p = 1/2\) contre \(H_1 : p \ne 1/2\).

On note \(\hat P\) la proportion de succès, \(\Phi\) la fonction de répartition de la loi normale centrée réduite, et l'on utilise la statistique \(Z = (\hat P - 1/2)/\sqrt{(1/2)(1/2)/n}\), dont on approche la loi sous \(H_0\) par la loi normale centrée réduite (théorème central limite). Quelle est la p-value bilatérale, arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@M08_exact:3@", "explication": r"Oui : \(Z = @M08_z@\) et \(2\,(1 - \Phi(2))\)."},
            {"id": "b", "texte": "@M08_d_unilaterale:3@", "explication": r"C'est la p-value unilatérale \(\mathbb{P}(Z \ge 2)\). L'alternative \(p \ne 1/2\) est bilatérale : un écart aussi grand vers le bas serait aussi un signe contre \(H_0\), et l'on additionne les deux queues."},
            {"id": "c", "texte": "@M08_d_complement:3@", "explication": r"C'est \(\mathbb{P}(\lvert Z \rvert < 2)\), la probabilité de l'événement contraire. La p-value est la probabilité, sous \(H_0\), d'un écart au moins aussi extrême que celui observé."},
            {"id": "d", "texte": "@M08_d_div_n:3@", "explication": r"L'écart-type de \(\hat P\) a été pris égal à \((1/2)(1/2)/n\) au lieu de sa racine carrée : \(Z\) vaut alors @M08_d_div_n_z@ et la p-value est presque nulle. La variance d'une moyenne de \(n\) variables indépendantes est divisée par \(n\), son écart-type par \(\sqrt n\)."},
        ],
        "solution": r"""Réponse : \(2\,(1 - \Phi(2)) \approx @M08_exact:3m@\).

Étape 1 : \(\hat P = 220/400 = @M08_phat:2m@\).

Étape 2 : sous \(H_0\), le nombre de succès suit la loi binomiale de paramètres \(n\) et 1/2, donc \(\hat P\) a pour espérance 1/2 et pour écart-type \(\sqrt{(1/2)(1/2)/n} = @M08_se:3m@\). Ainsi \(Z = (@M08_phat:2m@ - 0{,}5)/@M08_se:3m@ = @M08_z@\).

Étape 3 : la p-value bilatérale est \(\mathbb{P}(\lvert Z \rvert \ge 2)\) sous \(H_0\), approchée par \(2\,(1 - \Phi(2)) \approx @M08_exact:4m@\).

Lecture : au niveau 5 %, on rejette \(H_0\), de justesse. Mais la p-value binomiale exacte du même test, \(\mathbb{P}(\lvert K - 200 \rvert \ge 20)\) avec \(K\) de loi binomiale de paramètres 400 et 1/2, vaut @M08_binom:4@ (script) : au-dessus de 5 %. Près du seuil, la conclusion dépend de l'approximation ; et si ce signal a été retenu parmi plusieurs, il faut de plus corriger pour les tests multiples.

Simulations (script) : @M08_mc:4@ pour la loi normale, @M08_binom_mc:4@ pour la loi binomiale.""",
    },
    {
        "id": "EB2-M09", "section": "math", "theme": "Intégrale gaussienne, espérance tronquée", "difficulte": 3, "temps_cible_s": 150,
        "enonce": r"""Soit \(Z\) une variable aléatoire de loi normale centrée réduite. On note \(\varphi\) sa densité et \(\Phi\) sa fonction de répartition ; on donne \(\varphi(1) \approx @M09_phi1:4@\) et \(\Phi(1) \approx @M09_Phi1:4@\).

Que vaut \(\mathbb{E}[\max(Z - 1, 0)]\), arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@M09_exact:3@", "explication": r"Oui : \(\varphi(1) - (1 - \Phi(1))\)."},
            {"id": "b", "texte": "@M09_phi1:3@", "explication": r"C'est \(\int_1^\infty z\,\varphi(z)\,dz = \varphi(1)\) seul : le terme \(-1 \times \mathbb{P}(Z > 1)\), qui retranche le seuil, a été oublié."},
            {"id": "c", "texte": "@M09_queue:3@", "explication": r"C'est \(\mathbb{P}(Z > 1)\), la probabilité que le maximum soit non nul, et non l'espérance du montant qui dépasse 1."},
            {"id": "d", "texte": "@M09_d_jensen:3@", "explication": r"C'est \(\max(\mathbb{E}[Z] - 1, 0)\) : l'espérance a été passée à l'intérieur de la fonction. Comme \(x \mapsto \max(x - 1, 0)\) est convexe, l'inégalité de Jensen donne \(\mathbb{E}[\max(Z - 1, 0)] \ge \max(\mathbb{E}[Z] - 1, 0)\), et l'inégalité est stricte puisque \(Z\) dépasse 1 avec une probabilité non nulle."},
        ],
        "solution": r"""Réponse : \(\varphi(1) - (1 - \Phi(1)) \approx @M09_exact:3m@\).

Étape 1 : \(\max(Z - 1, 0) = (Z - 1)\,\mathbf{1}_{\{Z > 1\}}\), donc \(\mathbb{E}[\max(Z - 1, 0)] = \int_1^\infty (z - 1)\,\varphi(z)\,dz = \int_1^\infty z\,\varphi(z)\,dz - \mathbb{P}(Z > 1)\).

Étape 2 : \(\varphi(z) = e^{-z^2/2}/\sqrt{2\pi}\) vérifie \(\varphi'(z) = -z\,\varphi(z)\). Donc \(\int_1^\infty z\,\varphi(z)\,dz = \big[-\varphi(z)\big]_1^\infty = \varphi(1)\), puisque \(\varphi(z) \to 0\) quand \(z \to +\infty\).

Étape 3 : \(\varphi(1) - (1 - \Phi(1)) \approx @M09_phi1:4m@ - @M09_queue:4m@ = @M09_exact:4m@\).

Généralisation : pour un seuil \(k\) réel, le même calcul donne \(\mathbb{E}[\max(Z - k, 0)] = \varphi(k) - k\,(1 - \Phi(k))\). C'est le prix, sans actualisation, d'une option d'achat de prix d'exercice \(k\) sur un sous-jacent de loi normale centrée réduite (modèle de Bachelier). Cas limite \(k = 0\) : \(\mathbb{E}[\max(Z, 0)] = \varphi(0) \approx @M09_E_zplus:4m@\).

Simulation (script) : @M09_mc:4@.""",
    },
    {
        "id": "EB2-M10", "section": "math", "theme": "Mouvement brownien, intégrale en temps", "difficulte": 4, "temps_cible_s": 180,
        "enonce": r"""Soit \((W_t)_{t \ge 0}\) un mouvement brownien standard : \(W_0 = 0\), trajectoires continues, accroissements indépendants, et \(W_t - W_s\) de loi normale centrée de variance \(t - s\) pour \(0 \le s < t\). On pose \(I = \int_0^1 W_s\,ds\), l'aire algébrique sous la trajectoire entre les instants 0 et 1, qui est une variable aléatoire.

Que vaut \(\mathrm{Var}(I)\) ?""",
        "options": [
            {"id": "a", "texte": "@M10_exact_num@/@M10_exact_den@", "explication": r"Oui : \(\int_0^1\!\int_0^1 \min(s, t)\,ds\,dt\)."},
            {"id": "b", "texte": "@M10_d_second_moment_num@/@M10_d_second_moment_den@", "explication": r"C'est \(\mathbb{E}\big[\int_0^1 W_s^2\,ds\big] = \int_0^1 s\,ds\) : on a calculé l'espérance de l'intégrale du carré au lieu de celle du carré de l'intégrale. \(I^2 = \int_0^1\!\int_0^1 W_s W_t\,ds\,dt\) fait intervenir les covariances entre instants différents."},
            {"id": "c", "texte": "@M10_d_un@", "explication": r"C'est \(\mathrm{Var}(W_1)\), la variance du point final. L'intégrale moyenne la trajectoire sur \([0, 1]\), dont les valeurs proches de l'instant 0 sont petites : sa variance est plus faible."},
            {"id": "d", "texte": "@M10_d_moitie_num@/@M10_d_moitie_den@", "explication": r"C'est \(\int_0^1\!\int_0^t s\,ds\,dt\), l'intégrale de \(\min(s, t)\) sur le seul triangle \(s \le t\). Le carré \([0, 1]^2\) contient deux triangles symétriques : il faut doubler."},
        ],
        "solution": r"""Réponse : \(\mathrm{Var}(I) = @M10_exact_num@/@M10_exact_den@\).

Étape 1, covariance. Pour \(0 \le s \le t\), \(W_t = W_s + (W_t - W_s)\), où l'accroissement est centré et indépendant de \(W_s\). Donc \(\mathbb{E}[W_s W_t] = \mathbb{E}[W_s^2] + \mathbb{E}[W_s]\,\mathbb{E}[W_t - W_s] = s\). Ainsi \(\mathbb{E}[W_s W_t] = \min(s, t)\).

Étape 2, espérance : \(\mathbb{E}[I] = \int_0^1 \mathbb{E}[W_s]\,ds = 0\), l'échange étant permis par le théorème de Fubini puisque \(\int_0^1 \mathbb{E}\lvert W_s \rvert\,ds < \infty\).

Étape 3 : \(I^2 = \int_0^1\!\int_0^1 W_s W_t\,ds\,dt\). Le théorème de Fubini s'applique car \(\mathbb{E}\lvert W_s W_t \rvert \le \sqrt{st} \le 1\) (inégalité de Cauchy-Schwarz). Donc \(\mathbb{E}[I^2] = \int_0^1\!\int_0^1 \min(s, t)\,ds\,dt = 2\int_0^1\!\int_0^t s\,ds\,dt = 2\int_0^1 \tfrac{t^2}{2}\,dt = \tfrac13\), la symétrie en \((s, t)\) permettant de ne calculer que le triangle \(s \le t\).

Seconde méthode, par le calcul d'Itô (formule d'Itô et isométrie d'Itô admises). La formule d'Itô pour le produit \(t\,W_t\), où \(t\) est déterministe, donne \(d(t\,W_t) = W_t\,dt + t\,dW_t\), donc \(W_1 = \int_0^1 W_s\,ds + \int_0^1 s\,dW_s\) et \(I = \int_0^1 (1 - s)\,dW_s\). Par l'isométrie d'Itô, \(\mathrm{Var}(I) = \int_0^1 (1 - s)^2\,ds = \tfrac13\). Cette écriture montre aussi que \(I\) est gaussienne : \(I\) suit la loi normale centrée de variance 1/3.

Vérification numérique (script) : la somme de Riemann à 200 pas, \(\tfrac{1}{200}\sum_{k=1}^{200} W_{k/200}\), a pour variance exacte @M10_var_riemann_200:4@, qui tend vers 1/3 quand le pas diminue ; sa simulation donne @M10_riemann_200_mc:4@.""",
    },
    # =============================================================== ML FONDAMENTAL
    {
        "id": "EB2-F01", "section": "mlf", "theme": "Log-loss, meilleur prédicteur constant", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""Un jeu de test contient 20 % d'exemples positifs (étiquette 1) et 80 % de négatifs (étiquette 0). Un prédicteur constant attribue la même probabilité \(p \in \,]0, 1[\) d'être positif à tous les exemples. La log-loss moyenne est \(-\tfrac{1}{n}\sum_{i=1}^{n} \big[y_i \ln p + (1 - y_i)\ln(1 - p)\big]\), où \(n\) est le nombre d'exemples et \(y_i\) l'étiquette de l'exemple \(i\), avec le logarithme népérien.

Quelle est la plus petite log-loss moyenne que peut atteindre un prédicteur constant, arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@F01_exact:3@", "explication": r"""Oui : \(p = 0{,}2\) et \(-(0{,}2 \ln 0{,}2 + 0{,}8 \ln 0{,}8)\)."""},
            {"id": "b", "texte": "@F01_d_demi:3@", "explication": r"""C'est \(\ln 2\), la log-loss du prédicteur constant \(p = 1/2\). Ce n'est pas le meilleur constant quand les classes sont déséquilibrées : la meilleure constante est la proportion de positifs."""},
            {"id": "c", "texte": "@F01_d_negatifs:3@", "explication": r"""C'est \(-\ln 0{,}8\), la perte sur un exemple négatif seulement. La moyenne pondère aussi la perte \(-\ln 0{,}2\) des 20 % de positifs."""},
            {"id": "d", "texte": "@F01_d_brier:3@", "explication": r"""C'est \(0{,}2 \times 0{,}8\), le score de Brier (erreur quadratique moyenne des probabilités) du meilleur prédicteur constant. La question porte sur la log-loss."""},
        ],
        "solution": r"""Réponse : \(-(0{,}2 \ln 0{,}2 + 0{,}8 \ln 0{,}8) \approx @F01_exact:3m@\), atteinte en \(p = 0{,}2\).

Étape 1 : avec une proportion \(q = 0{,}2\) de positifs, la log-loss moyenne du prédicteur constant vaut \(L(p) = -\big(q \ln p + (1 - q)\ln(1 - p)\big)\).

Étape 2 : \(L'(p) = -q/p + (1 - q)/(1 - p)\), qui s'annule en \(p = q\) ; \(L''(p) = q/p^2 + (1 - q)/(1 - p)^2 > 0\), donc \(L\) est strictement convexe et \(p = q\) est son unique minimum.

Étape 3 : \(L(0{,}2) = -(0{,}2 \ln 0{,}2 + 0{,}8 \ln 0{,}8) \approx @F01_exact:3m@\) : c'est l'entropie de la loi de Bernoulli de paramètre 0,2.

Lecture : c'est le niveau de référence. Un modèle dont la log-loss de test dépasse @F01_exact:3@ fait moins bien que la simple fréquence de base ; la comparer à \(\ln 2\) serait trop indulgent quand les classes sont déséquilibrées. Contrôle (script) : minimisation sur une grille, minimum en 0,2.""",
    },
    {
        "id": "EB2-F02", "section": "mlf", "theme": "R² hors échantillon", "difficulte": 2, "temps_cible_s": 60,
        "enonce": r"""Un modèle prédit la valeur constante 4 pour toutes les observations (c'est la moyenne de la cible sur l'échantillon d'apprentissage). Sur un échantillon de test, les cibles valent 1, 2, 3, 4, 5. On définit le \(R^2\) de test par \(1 - \mathrm{SCR}/\mathrm{SCT}\), où \(\mathrm{SCR} = \sum_i (y_i - \hat y_i)^2\) est la somme des carrés des erreurs de prédiction sur le test et \(\mathrm{SCT} = \sum_i (y_i - \bar y)^2\), \(\bar y\) étant la moyenne des cibles de test.

Que vaut ce \(R^2\) de test ?""",
        "options": [
            {"id": "a", "texte": "@F02_exact:1@", "explication": r"""Oui : \(1 - @F02_scr@/@F02_sct@\)."""},
            {"id": "b", "texte": "@F02_d_zero@", "explication": r"""Un prédicteur constant n'a un \(R^2\) nul que si sa constante est la moyenne **du test**. Ici la constante, 4, vient de l'apprentissage et diffère de la moyenne de test, 3."""},
            {"id": "c", "texte": "@F02_d_signe:1@", "explication": r"""Le signe est perdu. Un \(R^2\) de test n'est pas un carré : il peut être négatif, ce qui arrive dès que le modèle fait pire que la moyenne du test."""},
            {"id": "d", "texte": "@F02_d_sans_un:1@", "explication": r"""C'est \(-\mathrm{SCR}/\mathrm{SCT}\) : le 1 de la définition a été oublié."""},
        ],
        "solution": r"""Réponse : \(R^2 = 1 - @F02_scr@/@F02_sct@ = @F02_exact:1m@\).

Étape 1 : \(\mathrm{SCR} = (1 - 4)^2 + (2 - 4)^2 + (3 - 4)^2 + 0 + (5 - 4)^2 = @F02_scr@\).

Étape 2 : la moyenne de test vaut 3, \(\mathrm{SCT} = 4 + 1 + 0 + 1 + 4 = @F02_sct@\).

Étape 3 : \(R^2 = 1 - @F02_scr@/@F02_sct@ = @F02_exact:1m@\).

Lecture : sur l'échantillon d'apprentissage, avec constante, le \(R^2\) des moindres carrés est compris entre 0 et 1, parce que la moyenne d'apprentissage fait partie des modèles possibles. Hors échantillon, rien ne le garantit : un \(R^2\) négatif signifie que le modèle prédit moins bien que la moyenne du test. En prévision de rendements, où le signal est faible, une petite erreur sur le niveau moyen suffit à rendre le \(R^2\) de test négatif ; il faut donc préciser la référence utilisée (moyenne du test, moyenne d'apprentissage ou zéro).""",
    },
    {
        "id": "EB2-F03", "section": "mlf", "theme": "Seuil de décision, coûts asymétriques", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""Un classifieur fournit, pour chaque transaction, une probabilité \(p\) d'être frauduleuse, supposée bien calibrée : parmi les transactions de probabilité \(p\), une proportion \(p\) est frauduleuse. Laisser passer une fraude (faux négatif) coûte 9 ; bloquer une transaction honnête (faux positif) coûte 1 ; une bonne décision ne coûte rien. On bloque une transaction si et seulement si \(p > s\).

Quel seuil \(s\) minimise l'espérance du coût de chaque décision ?""",
        "options": [
            {"id": "a", "texte": "@F03_exact:2@", "explication": r"""Oui : bloquer coûte en moyenne \(1 - p\), laisser passer \(9p\) ; on bloque quand \(1 - p < 9p\)."""},
            {"id": "b", "texte": "@F03_d_defaut:2@", "explication": "C'est le seuil qui minimise le nombre d'erreurs quand les deux erreurs coûtent autant. Ici un faux négatif coûte 9 fois plus : on bloque dès une probabilité bien plus faible."},
            {"id": "c", "texte": "@F03_d_inverse:2@", "explication": "Les coûts ont été inversés : avec ce seuil, on ne bloquerait que les transactions presque sûrement frauduleuses, ce qui convient si c'est le faux positif qui coûte cher."},
            {"id": "d", "texte": "@F03_d_rapport_num@/@F03_d_rapport_den@ ≈ @F03_d_rapport:3@", "explication": r"""C'est le rapport des coûts, 1/9. La condition \(1 - p < 9p\) donne \(p > 1/(1 + 9)\) : le seuil est \(c_{\mathrm{FP}}/(c_{\mathrm{FP}} + c_{\mathrm{FN}})\), pas \(c_{\mathrm{FP}}/c_{\mathrm{FN}}\)."""},
        ],
        "solution": r"""Réponse : \(s = 1/(1 + 9) = @F03_exact:1m@\).

Notations : \(c_{\mathrm{FN}} = 9\) et \(c_{\mathrm{FP}} = 1\) sont les coûts d'un faux négatif et d'un faux positif.

Étape 1 : pour une transaction de probabilité \(p\), bloquer coûte \(c_{\mathrm{FP}}\) si elle est honnête, ce qui arrive avec probabilité \(1 - p\) (calibration) : coût moyen \((1 - p)\,c_{\mathrm{FP}}\). Laisser passer coûte \(c_{\mathrm{FN}}\) si elle est frauduleuse : coût moyen \(p\,c_{\mathrm{FN}}\).

Étape 2 : bloquer est préférable si et seulement si \((1 - p)\,c_{\mathrm{FP}} < p\,c_{\mathrm{FN}}\), c'est-à-dire \(p > c_{\mathrm{FP}}/(c_{\mathrm{FP}} + c_{\mathrm{FN}}) = 1/10\). La règle « bloquer si \(p > 0{,}1\) » prend la meilleure décision pour chaque transaction, donc minimise le coût moyen.

Condition de validité : le raisonnement utilise la calibration. Avec des scores non calibrés, le seuil 0,1 n'a pas ce sens ; on le choisit alors sur un échantillon de validation. Expérience (script) : probabilités uniformes et étiquettes tirées selon elles ; le coût moyen simulé est minimal au seuil @F03_exp_meilleur_seuil:2@ et vaut @F03_exp_cout_01:3@ contre @F03_exp_cout_05:3@ au seuil 0,5.""",
    },
    {
        "id": "EB2-F04", "section": "mlf", "theme": "Malédiction de la dimension, voisinages", "difficulte": 2, "temps_cible_s": 75,
        "enonce": r"""Des points sont tirés indépendamment et uniformément dans le cube \([0, 1]^{10}\). Autour d'un point, on prend comme voisinage un sous-cube de côté \(e\), à faces parallèles aux axes, entièrement contenu dans \([0, 1]^{10}\).

Quelle valeur de \(e\) donne un voisinage qui contient en moyenne 1 % des points, arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@F04_exact:3@", "explication": r"Oui : \(e^{10} = 0{,}01\), soit \(e = 0{,}01^{1/10}\)."},
            {"id": "b", "texte": "@F04_d_fraction:3@", "explication": r"C'est la fraction de points voulue, prise pour une longueur de côté. En dimension 10, un sous-cube de côté 0,01 a pour volume \(0{,}01^{10}\) : il ne contient pratiquement aucun point."},
            {"id": "c", "texte": "@F04_d_div:3@", "explication": r"C'est \(0{,}01/10\) : la dimension a divisé la fraction au lieu d'en prendre la racine dixième. Le volume d'un cube est le produit de ses côtés, \(e^{10}\)."},
            {"id": "d", "texte": "@F04_d_dim2:3@", "explication": r"C'est \(0{,}01^{1/2}\), la bonne réponse en dimension 2. En dimension 10, il faut la racine dixième."},
        ],
        "solution": r"""Réponse : \(e = 0{,}01^{1/10} \approx @F04_exact:3m@\).

Étape 1 : pour un point uniforme dans \([0, 1]^{10}\), la probabilité de tomber dans une partie du cube est son volume. Un sous-cube de côté \(e\) a pour volume \(e^{10}\), produit des longueurs de ses côtés.

Étape 2 : la proportion moyenne de points dans le voisinage est cette probabilité (linéarité de l'espérance appliquée aux indicatrices « le point \(i\) est dans le voisinage »). On veut \(e^{10} = 0{,}01\), soit \(e = 0{,}01^{1/10} = 10^{-0{,}2} \approx @F04_exact:3m@\).

Lecture : pour capturer 1 % des données, le voisinage doit couvrir une proportion @F04_exact:2@ de l'étendue de chaque variable ; il n'a plus rien de local, et la prédiction des k plus proches voisins mélange des points éloignés. En dimension 100, il faudrait un côté de @F04_arete_d100:3@. C'est la malédiction de la dimension. Simulation (script) : une proportion @F04_mc:4@ des points tirés tombe dans un tel cube.""",
    },
    {
        "id": "EB2-F05", "section": "mlf", "theme": "Validation croisée un-contre-tous, levier", "difficulte": 4, "temps_cible_s": 105,
        "enonce": r"""On ajuste par moindres carrés ordinaires, avec constante, une régression \(y \approx b_0 + b_1 x\) sur 5 observations dont les valeurs de \(x\) sont 0, 1, 2, 3 et 9. Le résidu d'ajustement de la cinquième observation (\(x = 9\)), c'est-à-dire \(y_5 - \hat y_5\), vaut 0,4.

On réajuste le même modèle sans la cinquième observation, et l'on note \(\tilde y_5\) sa prédiction en \(x = 9\). Que vaut le résidu de validation croisée un-contre-tous \(y_5 - \tilde y_5\) ?""",
        "options": [
            {"id": "a", "texte": "@F05_exact@", "explication": r"""Oui : \(0{,}4/(1 - h_{55})\) avec un levier \(h_{55} = @F05_h_num@/@F05_h_den@\)."""},
            {"id": "b", "texte": "@F05_d_residu:1@", "explication": r"""C'est le résidu d'ajustement. Retirer l'observation change la droite : sans elle, rien ne tire la droite vers \(y_5\), et l'erreur de prédiction est bien plus grande."""},
            {"id": "c", "texte": "@F05_d_sans_1n:2@", "explication": r"""Le levier a été calculé sans le terme \(1/n\) dû à la constante : \(h_{55} = 1/n + (x_5 - \bar x)^2/\sum_j (x_j - \bar x)^2\)."""},
            {"id": "d", "texte": "@F05_d_1n:1@", "explication": r"""Le levier a été pris égal à \(1/n\), sa valeur pour une observation située à la moyenne des \(x\). Le point \(x = 9\) est loin de la moyenne 3 : son levier est proche de 1."""},
        ],
        "solution": r"""Réponse : \(0{,}4/(1 - @F05_h_num@/@F05_h_den@) = @F05_exact@\).

Notations : \(\mathbf{H}\) est la matrice de projection (matrice chapeau) des moindres carrés, \(\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}\), et \(h_{ii}\) son \(i\)-ème coefficient diagonal, le levier de l'observation \(i\) ; \(e_i = y_i - \hat y_i\).

Étape 1, formule du résidu un-contre-tous : \(y_i - \tilde y_i = e_i/(1 - h_{ii})\). Preuve. Soit \(\tilde{\mathbf{b}}\) l'estimateur obtenu sans l'observation \(i\), et \(\mathbf{z}\) le vecteur égal à \(\mathbf{y}\) sauf \(z_i = \tilde y_i\). Pour tout \(\mathbf{b}\), la somme des carrés des résidus sur \(\mathbf{z}\) est la somme sur les observations \(j \ne i\) plus \((\tilde y_i - \mathbf{x}_i^\top \mathbf{b})^2 \ge 0\) ; en \(\mathbf{b} = \tilde{\mathbf{b}}\), le premier morceau est minimal et le second nul. Donc \(\tilde{\mathbf{b}}\) est l'estimateur des moindres carrés pour \(\mathbf{z}\), et \(\tilde y_i = (\mathbf{H}\mathbf{z})_i = \hat y_i - h_{ii}\,(y_i - \tilde y_i)\), car \(\mathbf{z} = \mathbf{y} - (y_i - \tilde y_i)\,\mathbf{e}_i\). En réarrangeant : \((1 - h_{ii})(y_i - \tilde y_i) = y_i - \hat y_i\).

Étape 2, levier en régression simple avec constante : \(h_{ii} = 1/n + (x_i - \bar x)^2/\sum_j (x_j - \bar x)^2\). Preuve : \(\mathbf{H}\) projette sur l'espace engendré par \(\mathbf{1}\) et \(\mathbf{x}\), qui est aussi engendré par les vecteurs orthogonaux \(\mathbf{1}\) et \(\mathbf{x} - \bar x\,\mathbf{1}\) ; la projection sur une base orthogonale est la somme des projections sur chaque vecteur, d'où les deux termes.

Étape 3 : \(\bar x = @F05_moy_x@\), \(\sum_j (x_j - \bar x)^2 = 9 + 4 + 1 + 0 + 36 = @F05_sxx@\), donc \(h_{55} = 1/5 + 36/@F05_sxx@ = @F05_h_num@/@F05_h_den@\) et \(y_5 - \tilde y_5 = 0{,}4/(1 - @F05_h:2m@) = @F05_exact@\).

Lecture : un point à fort levier attire la droite à lui ; son petit résidu d'ajustement cache une grande erreur de prédiction. La formule donne toute la validation croisée un-contre-tous d'une régression linéaire en un seul ajustement. Contrôles (script) : réajustement explicite sans l'observation, et formule vérifiée sur des données aléatoires pour chaque observation.""",
    },
    # =============================================================== ML AVANCÉ
    {
        "id": "EB2-A01", "section": "mla", "theme": "Boosting de gradient, perte absolue", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Boosting de gradient avec la perte absolue \(\ell(y, f) = \lvert y - f \rvert\), où \(y\) est la cible et \(f\) la prédiction. Une feuille du nouvel arbre contient 5 observations dont les résidus courants \(r_i = y_i - \hat y_i\) (cible moins prédiction courante \(\hat y_i\)) valent −2 ; 0,5 ; 1 ; 4 ; 10.

La valeur \(w\) de la feuille, ajoutée à la prédiction de chacune de ces 5 observations, est choisie pour minimiser exactement \(\sum_i \lvert y_i - (\hat y_i + w) \rvert\) : pas de développement de Taylor, pas de régularisation, pas de taux d'apprentissage. Que vaut \(w\) ?""",
        "options": [
            {"id": "a", "texte": "@A01_exact@", "explication": r"Oui : la médiane des résidus de la feuille."},
            {"id": "b", "texte": "@A01_d_moyenne:1@", "explication": r"C'est la moyenne des résidus, qui minimise la somme des carrés \(\sum_i (r_i - w)^2\), pas la somme des valeurs absolues. Elle est tirée vers le haut par le résidu 10 ; la perte absolue y vaut @A01_perte_moyenne:1@ contre @A01_perte_mediane:1@ à la médiane."},
            {"id": "c", "texte": "@A01_d_signes:1@", "explication": r"C'est la moyenne des pseudo-résidus \(\mathrm{signe}(r_i)\), c'est-à-dire la valeur qu'un arbre de régression ajusté aux pseudo-résidus mettrait dans cette feuille. Cette valeur sert à choisir les coupures ; la valeur de la feuille est ensuite recalculée pour minimiser la vraie perte."},
            {"id": "d", "texte": r"Aucune : la dérivée seconde de la perte est nulle, le poids \(-G/(H + \lambda)\) n'est pas défini", "explication": r"La formule \(-G/(H + \lambda)\) vient d'un développement de Taylor à l'ordre 2, que l'énoncé exclut. Le minimum exact de \(\sum_i \lvert r_i - w \rvert\) existe toujours : c'est une fonction convexe, affine par morceaux, qui tend vers l'infini aux deux bouts."},
        ],
        "solution": r"""Réponse : \(w = @A01_exact@\), la médiane des résidus.

Étape 1 : \(y_i - (\hat y_i + w) = r_i - w\), donc on minimise \(g(w) = \sum_i \lvert r_i - w \rvert\).

Étape 2 : chaque terme \(\lvert r_i - w \rvert\) a pour pente +1 quand \(w > r_i\) et −1 quand \(w < r_i\). Entre deux résidus consécutifs, \(g\) est donc affine, de pente \(\#\{i : r_i < w\} - \#\{i : r_i > w\}\).

Étape 3 : avec les 5 résidus triés, −2 < 0,5 < 1 < 4 < 10, la pente vaut \(2 - 3 = -1\) juste avant 1 et \(3 - 2 = +1\) juste après : \(g\) décroît puis croît, son minimum est en 1, la médiane, où elle vaut @A01_perte_mediane:1@.

Mécanisme : c'est la robustesse de la perte absolue ; le résidu 10, peut-être aberrant, n'agit sur la feuille que par son signe. Dans le boosting de gradient de Friedman avec cette perte (Friedman, « Greedy function approximation: a gradient boosting machine », Annals of Statistics, 2001, algorithme LAD_TreeBoost), l'arbre est d'abord ajusté aux pseudo-résidus \(\mathrm{signe}(r_i)\) (l'opposé du gradient de la perte), puis chaque feuille reçoit la médiane des résidus qu'elle contient. Cas limite : avec un nombre pair de résidus, tout point entre les deux résidus centraux est un minimum.""",
    },
    {
        "id": "EB2-A02", "section": "mla", "theme": "LSTM, disparition du gradient", "difficulte": 3, "temps_cible_s": 105,
        "enonce": r"""Dans une cellule LSTM à une dimension, la mémoire évolue selon \(c_t = f_t\,c_{t-1} + i_t\,g_t\), où \(f_t\) est la porte d'oubli, \(i_t\) la porte d'entrée et \(g_t\) la valeur candidate. On suit seulement le chemin direct de la mémoire : les portes et la valeur candidate sont traitées comme des constantes qui ne dépendent pas de \(c_{t-1}\), avec \(f_t = 0{,}99\) à chaque pas.

Que vaut \(\partial c_{50}/\partial c_0\), arrondi à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@A02_exact:3@", "explication": r"""Oui : \(0{,}99^{50}\), produit des portes d'oubli le long du chemin."""},
            {"id": "b", "texte": "@A02_rnn:3@", "explication": r"""C'est \(0{,}9^{50}\), la valeur pour un réseau récurrent simple \(h_t = \tanh(w\,h_{t-1} + \dots)\) de poids \(w = 0{,}9\) près de 0. La mémoire de la LSTM est multipliée par la porte d'oubli, 0,99, pas par 0,9."""},
            {"id": "c", "texte": "@A02_d_lineaire:3@", "explication": r"""C'est \(1 - 50 \times 0{,}01\), le développement au premier ordre de \(0{,}99^{50}\). Avec 50 facteurs, l'approximation linéaire n'est plus valable : \((1 - x)^n \approx 1 - nx\) suppose \(nx\) petit."""},
            {"id": "d", "texte": "@A02_d_somme:3@", "explication": r"""C'est \(50 \times 0{,}99\) : les dérivées des pas successifs ont été additionnées. Par la règle de dérivation des fonctions composées, elles se multiplient."""},
        ],
        "solution": r"""Réponse : \(0{,}99^{50} \approx @A02_exact:3m@\).

Étape 1 : sous l'hypothèse de l'énoncé, \(\partial c_t/\partial c_{t-1} = f_t = 0{,}99\).

Étape 2, règle de dérivation des fonctions composées : \(\partial c_{50}/\partial c_0 = \prod_{t=1}^{50} \partial c_t/\partial c_{t-1} = 0{,}99^{50} \approx @A02_exact:3m@\).

Comparaison : dans un réseau récurrent simple \(h_t = \tanh(w\,h_{t-1} + u\,x_t)\), le facteur par pas est \(w \tanh'(\cdot)\), au plus \(\lvert w \rvert\) ; avec \(w = 0{,}9\), il reste au mieux \(0{,}9^{50} \approx @A02_rnn:3m@\) après 50 pas. C'est la disparition du gradient : le réseau ne peut pas apprendre de dépendances longues. La LSTM la contourne parce que sa mémoire se propage additivement, avec un facteur \(f_t\) que le réseau apprend et peut garder proche de 1.

Condition de validité : dans une vraie LSTM, les portes dépendent de l'état caché précédent, donc indirectement de \(c_{t-1}\) ; le gradient complet ajoute d'autres chemins. Le chemin direct calculé ici est celui qui ne s'éteint pas. Contrôle (script) : dérivée numérique de \(c_{50}\) par rapport à \(c_0\).""",
    },
    {
        "id": "EB2-A03", "section": "mla", "theme": "Normalisation par lot (BatchNorm)", "difficulte": 2, "temps_cible_s": 105,
        "enonce": r"""En entraînement, une couche de normalisation par lot (BatchNorm) reçoit, pour un neurone donné, les 4 valeurs du lot : 1, 2, 3, 6. Elle calcule la moyenne \(m\) et la variance \(v\) de ces 4 valeurs, la variance étant calculée avec le diviseur 4 (taille du lot), puis renvoie pour chaque valeur \(h\) la sortie \(\gamma\,(h - m)/\sqrt{v + \epsilon} + \beta\), avec \(\gamma = 2\), \(\beta = 1\) et \(\epsilon = 0\).

Quelle est la sortie pour la valeur \(h = 6\), arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@A03_exact:3@", "explication": r"""Oui : \(2 \times 3/\sqrt{@A03_var:1m@} + 1\)."""},
            {"id": "b", "texte": "@A03_d_n_moins_1:3@", "explication": "La variance a été calculée avec le diviseur 3 au lieu de 4. L'énoncé fixe le diviseur égal à la taille du lot, ce que fait la normalisation par lot en entraînement."},
            {"id": "c", "texte": "@A03_d_sans_affine:3@", "explication": r"""C'est la valeur normalisée \((h - m)/\sqrt v\), avant la transformation affine apprise \(\gamma\,(\cdot) + \beta\)."""},
            {"id": "d", "texte": "@A03_d_variance:3@", "explication": r"""L'écart à la moyenne a été divisé par la variance au lieu de l'écart-type \(\sqrt v\)."""},
        ],
        "solution": r"""Réponse : \(2 \times 3/\sqrt{3{,}5} + 1 \approx @A03_exact:3m@\).

Étape 1 : \(m = (1 + 2 + 3 + 6)/4 = @A03_moyenne@\).

Étape 2 : \(v = \big((1 - 3)^2 + (2 - 3)^2 + 0 + (6 - 3)^2\big)/4 = 14/4 = @A03_var:1m@\).

Étape 3 : \((6 - 3)/\sqrt{3{,}5} \approx @A03_d_sans_affine:4m@\), puis \(2 \times @A03_d_sans_affine:4m@ + 1 \approx @A03_exact:3m@\).

Mécanisme : la normalisation rend la distribution des pré-activations de chaque neurone stable (moyenne 0, variance 1 sur le lot), et \(\gamma\), \(\beta\) rendent au réseau la possibilité de choisir une autre échelle et un autre centre. Pièges : la sortie d'un exemple dépend des autres exemples du lot ; en évaluation, \(m\) et \(v\) sont remplacées par des moyennes mobiles calculées pendant l'entraînement ; \(\epsilon > 0\) évite une division par zéro quand toutes les valeurs du lot sont égales.""",
    },
    {
        "id": "EB2-A04", "section": "mla", "theme": "Momentum, pas effectif", "difficulte": 3, "temps_cible_s": 120,
        "enonce": r"""Descente de gradient avec momentum sur un paramètre réel \(\theta\) : \(v_t = \beta\,v_{t-1} + g_t\), puis \(\theta_t = \theta_{t-1} - \eta\,v_t\), avec \(v_0 = 0\), \(\beta = 0{,}9\) et \(\eta = 0{,}01\). Le gradient vaut \(g_t = 1\) à chaque pas (pente constante).

De combien \(\theta\) a-t-il diminué après 3 pas, c'est-à-dire que vaut \(\theta_0 - \theta_3\) ?""",
        "options": [
            {"id": "a", "texte": "@A04_exact:4@", "explication": r"""Oui : \(0{,}01 \times (1 + @A04_v2:1m@ + @A04_v3:2m@)\)."""},
            {"id": "b", "texte": "@A04_d_sans_moment:4@", "explication": r"""C'est le déplacement sans momentum, \(3 \times \eta \times 1\). Le momentum accumule les gradients passés : la vitesse \(v_t\) grandit tant que la pente reste la même."""},
            {"id": "c", "texte": "@A04_d_dernier:4@", "explication": r"""C'est le seul troisième pas, \(\eta\,v_3\). La question demande le déplacement total, somme des trois pas."""},
            {"id": "d", "texte": "@A04_d_limite:4@", "explication": r"""C'est 3 pas à la vitesse limite \(\eta/(1 - \beta) = @A04_pas_limite:1m@\). La vitesse part de 0 et n'atteint cette limite qu'après un nombre de pas de l'ordre de \(1/(1 - \beta) = 10\)."""},
        ],
        "solution": r"""Réponse : \(\theta_0 - \theta_3 = 0{,}01 \times (1 + @A04_v2:1m@ + @A04_v3:2m@) = @A04_exact:4m@\).

Étape 1, vitesses : \(v_1 = 1\), \(v_2 = 0{,}9 \times 1 + 1 = @A04_v2:1m@\), \(v_3 = 0{,}9 \times @A04_v2:1m@ + 1 = @A04_v3:2m@\).

Étape 2 : chaque pas diminue \(\theta\) de \(\eta\,v_t\) ; au total \(\eta\,(v_1 + v_2 + v_3) = @A04_exact:4m@\).

Mécanisme : avec un gradient constant \(g\), \(v_t = g\,(1 - \beta^t)/(1 - \beta)\) (somme géométrique), qui tend vers \(g/(1 - \beta)\). Le pas effectif devient \(\eta/(1 - \beta) = @A04_pas_limite:1m@\), dix fois le pas nominal : dans une direction de pente constante, le momentum accélère ; dans une direction où le gradient change de signe, les contributions successives se compensent et amortissent les oscillations. Conséquence pratique : en ajoutant un momentum \(\beta = 0{,}9\) à un réglage qui fonctionnait sans, il faut diviser \(\eta\) par 10 pour garder le même pas effectif ; à défaut, l'itération peut diverger.""",
    },
    {
        "id": "EB2-A05", "section": "mla", "theme": "Astuce du noyau, noyau polynomial", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Le noyau polynomial \(k(\mathbf{x}, \mathbf{z}) = (1 + \mathbf{x}^\top \mathbf{z})^2\) est défini pour \(\mathbf{x}, \mathbf{z} \in \mathbb{R}^{10}\).

Quel est le plus petit entier \(D\) tel qu'il existe une application \(\boldsymbol{\phi} : \mathbb{R}^{10} \to \mathbb{R}^D\) vérifiant \(k(\mathbf{x}, \mathbf{z}) = \boldsymbol{\phi}(\mathbf{x})^\top \boldsymbol{\phi}(\mathbf{z})\) pour tous \(\mathbf{x}, \mathbf{z} \in \mathbb{R}^{10}\) ?""",
        "options": [
            {"id": "a", "texte": "@A05_exact@", "explication": r"Oui : la constante, 10 termes linéaires, 10 carrés et 45 produits croisés."},
            {"id": "b", "texte": "@A05_d_ordonne@", "explication": r"C'est \(1 + 10 + 10 \times 10\) : l'application explicite qui contient \(x_i x_j\) pour tous les couples ordonnés \((i, j)\). Elle convient, mais \(x_i x_j\) et \(x_j x_i\) sont la même fonction : on les fusionne en une coordonnée \(\sqrt2\,x_i x_j\), ce qui réduit la dimension."},
            {"id": "c", "texte": "@A05_d_degre2@", "explication": r"C'est \(\binom{11}{2}\), le nombre de monômes de degré exactement 2. Le développement \(1 + 2\,\mathbf{x}^\top\mathbf{z} + (\mathbf{x}^\top\mathbf{z})^2\) contient aussi la constante et les termes de degré 1."},
            {"id": "d", "texte": "@A05_d_carres@", "explication": r"C'est \(1 + 10 + 10\) : la constante, les termes linéaires et les carrés \(x_i^2\), sans les produits croisés \(x_i x_j\) (\(i \ne j\)) qui apparaissent dans \((\mathbf{x}^\top\mathbf{z})^2 = \sum_{i,j} x_i x_j z_i z_j\)."},
        ],
        "solution": r"""Réponse : \(D = @A05_exact@\).

Étape 1, une application de dimension @A05_exact@. En développant, \(k(\mathbf{x}, \mathbf{z}) = 1 + 2\sum_i x_i z_i + \sum_i x_i^2 z_i^2 + 2\sum_{i < j} x_i x_j z_i z_j\). Avec \(\boldsymbol{\phi}(\mathbf{x}) = \big(1, (\sqrt2\,x_i)_{i}, (x_i^2)_{i}, (\sqrt2\,x_i x_j)_{i < j}\big)\), on a \(\boldsymbol{\phi}(\mathbf{x})^\top \boldsymbol{\phi}(\mathbf{z}) = k(\mathbf{x}, \mathbf{z})\), en dimension \(1 + @A05_lineaires@ + @A05_carres@ + @A05_croises@ = @A05_exact@\).

Étape 2, on ne peut pas faire moins. Si \(k(\mathbf{x}, \mathbf{z}) = \boldsymbol{\phi}(\mathbf{x})^\top \boldsymbol{\phi}(\mathbf{z})\) avec \(\boldsymbol{\phi}\) à valeurs dans \(\mathbb{R}^D\), alors pour chaque \(\mathbf{z}\) fixé, la fonction \(\mathbf{x} \mapsto k(\mathbf{x}, \mathbf{z})\) est une combinaison linéaire des \(D\) fonctions coordonnées de \(\boldsymbol{\phi}\). L'espace \(E\) engendré par ces fonctions de \(\mathbf{x}\), quand \(\mathbf{z}\) parcourt \(\mathbb{R}^{10}\), est donc de dimension au plus \(D\). Il suffit de montrer que \(E\) contient les @A05_exact@ monômes \(1\), \(x_i\), \(x_i x_j\) (\(i \le j\)), linéairement indépendants (un polynôme nul en tout point a tous ses coefficients nuls).

2a. Pour \(\mathbf{z} = t\,\mathbf{a}\), avec \(t\) réel et \(\mathbf{a}\) fixé : \(k(\mathbf{x}, t\mathbf{a}) = 1 + 2t\,(\mathbf{a}^\top\mathbf{x}) + t^2\,(\mathbf{a}^\top\mathbf{x})^2\). Avec trois valeurs distinctes de \(t\), le système qui donne ces trois fonctions de \(\mathbf{x}\) a une matrice de Vandermonde inversible : \(1\), \(\mathbf{a}^\top\mathbf{x}\) et \((\mathbf{a}^\top\mathbf{x})^2\) sont dans \(E\).

2b. Avec \(\mathbf{a} = \mathbf{e}_i\), \(i\)-ème vecteur de la base canonique, \(x_i\) et \(x_i^2\) sont dans \(E\). Avec \(\mathbf{a} = \mathbf{e}_i + \mathbf{e}_j\), \((x_i + x_j)^2\) est dans \(E\), donc aussi \(2\,x_i x_j = (x_i + x_j)^2 - x_i^2 - x_j^2\). D'où \(D \ge @A05_exact@\).

Vérification numérique (script) : la matrice \(300 \times 300\) des \(k(\mathbf{x}_a, \mathbf{x}_b)\) pour 300 points aléatoires est de rang @A05_rang_gram@.

Intérêt de l'astuce du noyau : évaluer \(k\) coûte un produit scalaire en dimension 10, au lieu de construire @A05_exact@ coordonnées ; pour un degré \(p\), la dimension \(\binom{10 + p}{p}\) explose, le coût du noyau non.""",
    },
    {
        "id": "EB2-A06", "section": "mla", "theme": "SVM à marge dure, marge", "difficulte": 4, "temps_cible_s": 150,
        "enonce": r"""Trois points du plan : \((2, 0)\) et \((0, 2)\) sont de classe \(+1\), \((0, 0)\) est de classe \(-1\). Le SVM à marge dure cherche \(\mathbf{w} \in \mathbb{R}^2\) et \(b \in \mathbb{R}\) qui minimisent \(\tfrac12 \lVert \mathbf{w} \rVert^2\) sous les contraintes \(y_i\,(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1\) pour chaque point \(\mathbf{x}_i\) de classe \(y_i\). La largeur de la marge est la distance entre les droites \(\mathbf{w}^\top \mathbf{x} + b = 1\) et \(\mathbf{w}^\top \mathbf{x} + b = -1\), soit \(2/\lVert \mathbf{w} \rVert\).

Que vaut la largeur de la marge de la solution, arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@A06_exact:3@", "explication": r"""Oui : \(\mathbf{w} = (1, 1)\), \(b = -1\), largeur \(2/\sqrt2\)."""},
            {"id": "b", "texte": "@A06_d_demi:3@", "explication": r"""C'est la demi-largeur \(1/\lVert \mathbf{w} \rVert\), distance de la frontière \(\mathbf{w}^\top\mathbf{x} + b = 0\) à chaque classe. L'énoncé définit la largeur entre les deux droites \(\pm 1\)."""},
            {"id": "c", "texte": "@A06_d_points:3@", "explication": r"""C'est la distance du point négatif à chacun des positifs. La marge sépare le négatif de l'ensemble convexe des positifs, le segment de \((2, 0)\) à \((0, 2)\), dont le point le plus proche de l'origine est \((1, 1)\), à distance \(\sqrt2\)."""},
            {"id": "d", "texte": "@A06_d_positifs:3@", "explication": "C'est la distance entre les deux points positifs. La marge mesure la séparation entre les classes, pas l'écart entre les points d'une même classe."},
        ],
        "solution": r"""Réponse : \(2/\sqrt2 = \sqrt2 \approx @A06_exact:3m@\).

Étape 1, une solution réalisable. Avec \(\mathbf{w} = (1, 1)\) et \(b = -1\) : \((2, 0)\) et \((0, 2)\) donnent \(\mathbf{w}^\top\mathbf{x} + b = 1\), et \((0, 0)\) donne \(-1\), soit \(y_i\,(\mathbf{w}^\top\mathbf{x}_i + b) = 1\) pour les trois points. Ici \(\lVert \mathbf{w} \rVert = \sqrt2\).

Étape 2, on ne peut pas faire mieux. Pour tout \((\mathbf{w}, b)\) réalisable, la contrainte du point \((0, 0)\) donne \(-b \ge 1\) ; celles des points positifs donnent \(2w_1 + b \ge 1\) et \(2w_2 + b \ge 1\). En additionnant ces deux dernières, \(w_1 + w_2 \ge 1 - b \ge 2\). Par l'inégalité de Cauchy-Schwarz, \(w_1 + w_2 \le \sqrt2\,\lVert \mathbf{w} \rVert\), donc \(\lVert \mathbf{w} \rVert \ge \sqrt2\), avec égalité pour \(\mathbf{w} = (1, 1)\), \(b = -1\).

Étape 3 : largeur \(2/\lVert \mathbf{w} \rVert = \sqrt2 \approx @A06_exact:3m@\).

Lecture : les trois points saturent leur contrainte, ce sont les trois vecteurs de support ; la frontière \(x_1 + x_2 = 1\) passe à égale distance, \(1/\sqrt2\), de l'origine et du segment des positifs. Retirer un point qui n'est pas vecteur de support ne changerait pas la solution. Contrôle (script) : recherche sur une grille de \((w_1, w_2, b)\) ; aucune solution réalisable n'a une norme plus petite que \(\sqrt2\).""",
    },
    {
        "id": "EB2-A07", "section": "mla", "theme": "Non-stationnarité, moyenne mobile exponentielle", "difficulte": 3, "temps_cible_s": 105,
        "enonce": r"""Pour suivre une volatilité qui change dans le temps, on estime chaque jour la variance des rendements par une moyenne mobile exponentielle : \(\hat s_t = \lambda\,\hat s_{t-1} + (1 - \lambda)\,r_t^2\), où \(r_t\) est le rendement du jour \(t\) et \(\lambda = 0{,}94\). En déroulant la récurrence, le carré \(r_{t-k}\) d'il y a \(k\) jours reçoit le poids \((1 - \lambda)\,\lambda^k\).

La demi-vie est le nombre \(h\) (réel) de jours tel que \(\lambda^h = 1/2\) : le poids d'une observation est divisé par deux tous les \(h\) jours. Que vaut \(h\), arrondi à deux décimales ?""",
        "options": [
            {"id": "a", "texte": "@A07_exact:2@", "explication": r"""Oui : \(\ln(1/2)/\ln 0{,}94\)."""},
            {"id": "b", "texte": "@A07_d_inverse:2@", "explication": r"""C'est \(1/(1 - \lambda)\), qu'on présente parfois comme une « longueur de fenêtre équivalente ». Ce n'est pas la demi-vie : il faut résoudre \(\lambda^h = 1/2\)."""},
            {"id": "c", "texte": "@A07_d_age:2@", "explication": r"""C'est \(\lambda/(1 - \lambda)\), l'âge moyen des observations pondérées par leurs poids \((1 - \lambda)\lambda^k\). La moyenne d'une loi géométrique dépasse sa médiane, proche de la demi-vie."""},
            {"id": "d", "texte": "@A07_d_lineaire:2@", "explication": r"""C'est \(0{,}5/(1 - \lambda)\) : on a supposé que le poids perd \(1 - \lambda = 6\) % par jour de façon additive. La décroissance est multiplicative : \(\lambda^h\), pas \(1 - h\,(1 - \lambda)\)."""},
        ],
        "solution": r"""Réponse : \(h = \ln(1/2)/\ln(0{,}94) \approx @A07_exact:2m@\) jours.

Étape 1 : \(\lambda^h = 1/2 \iff h \ln \lambda = -\ln 2 \iff h = \ln 2/(-\ln \lambda)\), car \(\ln \lambda < 0\).

Étape 2 : \(-\ln 0{,}94 \approx @A07_moins_ln:4m@\), d'où \(h \approx @A07_exact:2m@\). Approximation utile : \(-\ln \lambda \approx 1 - \lambda\) quand \(\lambda\) est proche de 1, donc \(h \approx \ln 2/(1 - \lambda) \approx @A07_approx:2m@\).

Contrôle des poids : \(\sum_{k \ge 0} (1 - \lambda)\lambda^k = 1\) (série géométrique), et l'âge moyen \(\sum_k k\,(1 - \lambda)\lambda^k = \lambda/(1 - \lambda) \approx @A07_d_age:2m@\) jours (script, avec sympy).

Mécanisme : la moyenne exponentielle répond à la non-stationnarité en oubliant progressivement le passé. Une demi-vie courte suit vite les changements de régime mais donne une estimation bruitée (peu d'observations effectives) ; une demi-vie longue est stable mais en retard. Le choix de \(\lambda\) se valide hors échantillon, comme un hyperparamètre.""",
    },
    {
        "id": "EB2-A08", "section": "mla", "theme": "Ratio de Sharpe d'une combinaison de stratégies", "difficulte": 3, "temps_cible_s": 135,
        "enonce": r"""Deux stratégies ont des rendements annuels \(R_1\) et \(R_2\) de même écart-type \(\sigma > 0\), de même espérance \(\mu > 0\) et de corrélation 0,5 ; chacune a donc le ratio de Sharpe \(\mu/\sigma = 1\) (taux sans risque nul). On investit la moitié du capital dans chacune : rendement \((R_1 + R_2)/2\).

Quel est le ratio de Sharpe de cette combinaison, arrondi à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@A08_exact:3@", "explication": r"""Oui : \(1/\sqrt{(1 + 0{,}5)/2}\)."""},
            {"id": "b", "texte": "@A08_d_moyenne:3@", "explication": "C'est la moyenne des deux Sharpe. L'espérance de la combinaison est la moyenne des espérances, mais son écart-type est plus petit que la moyenne des écarts-types dès que la corrélation est inférieure à 1 : c'est la diversification."},
            {"id": "c", "texte": "@A08_d_independance:3@", "explication": r"""C'est \(\sqrt2\), le résultat pour deux stratégies non corrélées. Avec une corrélation 0,5, la variance de la combinaison vaut \(0{,}75\,\sigma^2\), pas \(0{,}5\,\sigma^2\)."""},
            {"id": "d", "texte": "@A08_d_somme:3@", "explication": "C'est la somme des deux Sharpe, comme si les espérances s'additionnaient sans que le risque augmente. Le Sharpe ne dépend pas de l'échelle des positions : additionner deux stratégies identiques donnerait encore 1."},
        ],
        "solution": r"""Réponse : \(1/\sqrt{0{,}75} \approx @A08_exact:3m@\).

Étape 1, espérance : \(\mathbb{E}[(R_1 + R_2)/2] = \mu\).

Étape 2, variance : \(\mathrm{Var}\big((R_1 + R_2)/2\big) = \tfrac14\big(\sigma^2 + \sigma^2 + 2 \times 0{,}5\,\sigma^2\big) = @A08_var_comb:2m@\,\sigma^2\).

Étape 3 : Sharpe \(= \mu/(\sigma\sqrt{@A08_var_comb:2m@}) = 1/\sqrt{@A08_var_comb:2m@} \approx @A08_exact:3m@\).

Généralisation : \(N\) stratégies de même Sharpe \(S\), même volatilité et corrélation \(\rho\) deux à deux, à poids égaux, ont un Sharpe \(S\sqrt{N/(1 + (N - 1)\rho)}\) (même calcul : la variance de la moyenne vaut \(\sigma^2(1 + (N - 1)\rho)/N\)). Quand \(N \to \infty\), il tend vers \(S/\sqrt\rho\), soit @A08_limite:3@ ici : la corrélation plafonne le gain de diversification, comme pour les arbres d'une forêt aléatoire. Simulation (script) : espérance @A08_moyenne_mc:3@ et variance @A08_var_mc:3@ du rendement combiné, pour 1 et 0,75.""",
    },
]
