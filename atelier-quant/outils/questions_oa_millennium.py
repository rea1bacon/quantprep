"""Examen OA Millennium x AmplifyME : six problèmes, 2 heures, sans pause.

Reconstitution de mémoire, non officielle. Tests et corrections : coding/millennium/. Aucun nombre calculé par le site :
les exemples d'énoncé sont des données d'entrée et les résultats attendus sont dans les tests.
"""
import commun_mm as cm
import questions_diagnostic as diag

ID = "oa-millennium"
TITRE = "OA Millennium x AmplifyME Market Making : examen"
VERIF = None
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0, "pause": False, "je_ne_sais_pas": False}
SECTIONS = [{"id": "code", "nom": "Six utilitaires de desk", "budget_min": 120}]
PRESENTATION = (
    "Un entraînement à l'évaluation en ligne « Market Making Dataset Assessment » de Millennium, proposée par AmplifyME : "
    "six problèmes indépendants à traiter en 2 heures, sans pause, dans un notebook Python, dans l'ordre choisi.\n\n"
    "**Statut de cette page.** Ces énoncés sont une **reconstitution de mémoire, non officielle** : ils ont été réécrits après "
    "l'épreuve à partir de souvenirs, et ne viennent ni de Millennium ni d'AmplifyME, qui ne les ont pas validés. Les formules, "
    "les noms et les cas limites peuvent différer de l'épreuve réelle, et une version future peut être différente. La durée de "
    "2 heures vient de la même source non officielle. Le barème réel est inconnu : la note affichée est la proportion de tests "
    "réussis, un choix de cette plateforme.\n\n"
    "Lire toutes les règles avant de coder : la formule centrale de chaque problème tient en une ligne, et la note dépend surtout "
    "des cas limites. Les leçons utiles sont dans le bloc « Market making »."
)
AVERTISSEMENTS = [
    "Reconstitution de mémoire, non officielle, sans lien avec Millennium ni AmplifyME.", cm.AVERTISSEMENT_VERSIONS,
    "Temps indicatif : 20 minutes par problème. Toutes les données sont synthétiques.",
]
INTRO = "Correction commentée des six problèmes, en pandas"
LARGE = r"""**Relevé de prix.** `prices` est un DataFrame large : une colonne `timestamp` (date, ou chaîne lisible par pandas) et une colonne de flottants par actif, de nom quelconque (`Asset_1`, `Asset_2`...). Il n'est pas forcément trié."""

P1 = r"""## Problème 1 : réparer le relevé

Écris `clean_prices(prices)`, qui rend une copie réparée. L'entrée n'est pas modifiée.

""" + LARGE + r""" Il peut contenir des horodatages répétés, et des valeurs manquantes, nulles ou négatives.

1. Convertir `timestamp` en date, trier par temps et garder **une** ligne par horodatage : la première dans l'ordre du fichier d'entrée.
2. Réindexer de 0 à \(n-1\).
3. Dans chaque colonne d'actif, une valeur inférieure ou égale à 0 devient manquante.
4. Propager vers le bas la dernière valeur valide, colonne par colonne. Ne **jamais** propager vers le haut : ce serait utiliser un prix futur.
5. Un trou de tête, sans valeur antérieure, reste manquant.

Code vectorisé attendu : pandas et numpy, sans boucle ligne par ligne.
"""

P2 = r"""## Problème 2 : mesurer la liquidité

Écris `liquidity_scores(prices, requests)`, qui rend un dictionnaire `{actif: score}`, chaque score arrondi à 6 décimales.

""" + LARGE + r""" Ici il est propre. Les actifs sont toutes ses colonnes sauf `timestamp`. `requests` a une ligne par demande client et au moins une colonne `asset` ; il peut avoir d'autres colonnes, à ignorer, et aucune ligne.

Pour chaque actif : \(v\) est l'écart-type (`ddof=1`) de `prices[actif].pct_change()`, \(f\) est sa part dans le nombre total de demandes (0 s'il n'y a aucune demande). \(v\) et \(f\) sont **chacun** normalisés sur l'ensemble des actifs : \((x - \min)/(\max - \min)\). Si un signal a la même valeur pour tous les actifs, sa version normalisée vaut 0,5 pour tous.

Le score vaut \(0{,}1 + 0{,}9\,\bigl(0{,}5\,(1 - v_n) + 0{,}5\,f_n\bigr)\), où \(v_n\) et \(f_n\) sont les valeurs normalisées.
"""

P3 = r"""## Problème 3 : valeur de référence à un instant donné

Écris `reference_price(prices, asset, ts, max_staleness_minutes)`, qui rend `{"price": valeur}`, où `valeur` est un flottant ou `None`. L'entrée n'est pas modifiée.

""" + LARGE + r""" Il peut contenir des valeurs manquantes, nulles ou négatives, toutes invalides. `asset` nomme une colonne. `ts` est une chaîne lisible par pandas ou un `pd.Timestamp`. `max_staleness_minutes` est un nombre.

1. Parmi les lignes d'horodatage inférieur ou égal à `ts` dont le prix de **cet actif** est valide, prendre la plus récente. Un autre actif invalide sur la ligne n'a aucun effet.
2. S'il n'en existe aucune, rendre `{"price": None}`.
3. Si son âge dépasse strictement `max_staleness_minutes`, rendre `{"price": None}` : un âge exactement égal à la limite est accepté. L'âge compte toutes les minutes écoulées, jours compris.
4. Sinon, rendre son prix.
"""

P4 = r"""## Problème 4 : coter une demande

Écris `quote_request(reference, half_spread, side, quantity, size_params, tick_size)`, qui rend le prix coté, en flottant.

`reference`, `half_spread` et `tick_size` sont des flottants. `side` vaut exactement `"BUY"` ou `"SELL"`, du point de vue du client : un client qui achète fait **vendre** le teneur, qui cote au-dessus de la référence. `quantity` est un nombre, `size_params` un dictionnaire de clé `"k"`, la pénalité par unité.

\(\displaystyle \text{BUY} : \left\lceil \frac{\text{réf} + \text{demi-écart} + k\,q}{\text{tick}} \right\rceil \text{tick}, \qquad \text{SELL} : \left\lfloor \frac{\text{réf} - \text{demi-écart} - k\,q}{\text{tick}} \right\rfloor \text{tick}.\)

Avant la partie entière, arrondir le rapport à 9 décimales : sinon le bruit de l'arithmétique flottante envoie au tick suivant une valeur tombant exactement sur la grille.

**Exemple.** `quote_request(100.0, 0.5, "BUY", 13, {"k": 0.01}, 0.05)` vaut 100,65, à l'arrondi flottant près.
"""

P5 = r"""## Problème 5 : le flux de cotations et l'impact cumulé

Écris `quote_impact_stream(reference, impact_params, liquidity_score, requests_for_asset)`, qui rend une liste de flottants, une cote par demande, dans le même ordre.

`impact_params` est un dictionnaire de clés `"k"`, `"decay"` (dans \(]0, 1]\)) et `"half_spread"`. `liquidity_score` est dans \([0, 1]\). `requests_for_asset` est une liste, éventuellement vide, de dictionnaires `{"side": "BUY" ou "SELL", "quantity": nombre}`.

Un état unique, `impact`, part de 0. Pour chaque demande, dans l'ordre, avec \(e = +1\) pour `BUY` et \(-1\) pour `SELL`, et \(q\) la quantité :

1. `cote = reference + e * half_spread + impact` : la cote utilise l'état **avant** l'effet de cette demande.
2. `impact = (impact + e * k * q * (1 - liquidity_score)) * decay` : le déclin multiplie **tout** l'état.

**Exemple.** Référence 100, `k=0.02`, `decay=0.9`, `half_spread=0.1`, score 0,5, demandes BUY 10, BUY 10, SELL 5 : cotes 100,1 ; 100,19 ; 100,071.
"""

P6 = r"""## Problème 6 : tenir le livre

Écris `track_book(filled_requests, prices)`, qui rend `{"inventory": {actif: quantité}, "cash": flottant, "pnl": flottant}`.

`filled_requests` a une ligne par ordre client exécuté, de colonnes `asset`, `side` (`"BUY"` ou `"SELL"`), `quantity`, `fill_price` ; il peut être vide. """ + LARGE + r""" Ici il est propre, et seule sa dernière ligne sert à valoriser le stock final.

Le teneur prend le côté **opposé** du client, en partant de zéro pour chaque actif de `prices`, avec une caisse nulle :

- client `BUY` : l'inventaire de l'actif baisse de la quantité, la caisse augmente de prix fois quantité ;
- client `SELL` : l'inventaire augmente de la quantité, la caisse baisse de prix fois quantité.

`pnl = cash + somme(inventaire[a] * dernier_prix[a])`. Arrondir `cash` et `pnl` à 6 décimales ; ne pas arrondir l'inventaire.
"""

CODES = [
    cm.code("MIL-P1", "code", "Réparer un relevé de prix", 3, 1200, P1, "recall_p1",
            [cm.IMPORTS, "def clean_prices(prices):\n    pass  # À compléter"], "recall.py", INTRO),
    cm.code("MIL-P2", "code", "Score de liquidité par normalisation", 3, 1200, P2, "recall_p2",
            [cm.IMPORTS, "def liquidity_scores(prices, requests):\n    pass  # À compléter"], "recall.py", INTRO),
    cm.code("MIL-P3", "code", "Recherche au plus tard à un instant, avec limite d'âge", 4, 1200, P3, "recall_p3",
            [cm.IMPORTS, "def reference_price(prices, asset, ts, max_staleness_minutes):\n    pass  # À compléter"], "recall.py", INTRO),
    cm.code("MIL-P4", "code", "Cotation arrondie à la grille de prix", 3, 1200, P4, "recall_p4",
            [cm.IMPORTS, "def quote_request(reference, half_spread, side, quantity, size_params, tick_size):\n    pass  # À compléter"], "recall.py", INTRO),
    cm.code("MIL-P5", "code", "Impact cumulé à déclin, état unique", 4, 1200, P5, "recall_p5",
            [cm.IMPORTS, "def quote_impact_stream(reference, impact_params, liquidity_score, requests_for_asset):\n    pass  # À compléter"], "recall.py", INTRO),
    cm.code("MIL-P6", "code", "Inventaire, caisse et résultat d'un teneur", 3, 1200, P6, "recall_p6",
            [cm.IMPORTS, "def track_book(filled_requests, prices):\n    pass  # À compléter"], "recall.py", INTRO),
]
QUESTIONS = []
