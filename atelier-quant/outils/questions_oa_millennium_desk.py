"""Entraînement de la famille de l'OA Millennium : sept utilitaires de desk en pandas (hors reconstitution)."""
import commun_mm as cm
import questions_diagnostic as diag

ID = "oa-millennium-desk"
TITRE = "OA Millennium : entraînement, utilitaires de desk"
VERIF = None
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0, "pause": False, "je_ne_sais_pas": False}
SECTIONS = [{"id": "code", "nom": "Sept problèmes pandas", "budget_min": 120}]
PRESENTATION = (
    "Sept problèmes indépendants de la même famille que l'examen, mais avec d'autres formules et d'autres formes de données : retirer "
    "plutôt que réparer, jointure sur un référentiel, barres, résultat réalisé et latent, passage du format long au format large. "
    "Ces énoncés sont des **exercices d'entraînement écrits pour la préparation** : ils ne prétendent pas reproduire une épreuve réelle."
)
AVERTISSEMENTS = ["Exercices d'entraînement, sans lien avec Millennium ni AmplifyME.", cm.AVERTISSEMENT_VERSIONS, "Toutes les données sont synthétiques."]
INTRO = "Correction commentée des sept problèmes, en pandas"

P1 = r"""## Problème 1 : rapprocher les confirmations

Écris `reconcile_fills(confirmations)`. L'entrée a les colonnes `trade_id`, `timestamp` (date ou chaîne lisible), `asset`, `price`, `quantity`, `status` (`"FILLED"` ou `"CANCELLED"`), n'est pas forcément triée, et peut répéter un `trade_id`.

Rendre une copie : garder les lignes `FILLED` ; retirer celles de prix ou de quantité inférieurs ou égaux à 0 (une confirmation corrompue se retire, elle ne se répare pas) ; trier par temps ; pour un `trade_id` répété, garder la ligne la plus ancienne ; réindexer de 0 à \(n-1\).
"""
P2 = r"""## Problème 2 : noter la qualité d'exécution

Écris `execution_quality_scores(fills)`, qui rend `{actif: score}` arrondi à 6 décimales. `fills` a une ligne par exécution, de colonnes `asset`, `slippage_bps` (signé, positif signifie pire) et `fill_time_ms` (latence, strictement positive).

Pour chaque actif, calculer la **moyenne** de chaque signal, puis normaliser chaque moyenne sur les actifs par \((x - \min)/(\max - \min)\), avec 0,5 pour tous si un signal a la même valeur partout. Le score vaut \(0{,}1 + 0{,}9\,(0{,}5\,(1 - g_n) + 0{,}5\,(1 - l_n))\), où \(g_n\) et \(l_n\) sont les moyennes normalisées de glissement et de latence. Plus bas est meilleur pour les deux signaux.
"""
P3 = r"""## Problème 3 : valeurs de référence par lot

Écris `batch_reference_prices(prices_long, queries, max_staleness_minutes)`, qui rend une liste de flottants ou de `None`, une entrée par ligne de `queries`, dans le même ordre.

`prices_long` est au format long : colonnes `timestamp`, `asset`, `price`, non triées ; un prix manquant, nul ou négatif est invalide. `queries` a les colonnes `asset` et `ts` (chaîne ou `pd.Timestamp`).

Pour chaque requête : prendre le prix valide le plus récent de cet actif, au plus tard à `ts`. Rendre `None` s'il n'en existe pas, ou si son âge dépasse strictement `max_staleness_minutes` ; un âge égal à la limite est accepté.
"""
P4 = r"""## Problème 4 : joindre le référentiel

Écris `enrich_fills(fills, instruments)`. `fills` a les colonnes `trade_id`, `asset`, `quantity`, `fill_price`. `instruments` a une ligne par actif **et par publication**, de colonnes `asset`, `multiplier`, `currency`, `as_of` (date ou chaîne), non triée.

Rendre une copie de `fills` avec quatre colonnes ajoutées : `multiplier`, `currency`, `notional` (`quantity * fill_price * multiplier`, arrondi à 6 décimales), `unmapped`.

- Garder **chaque** exécution, dans son ordre d'origine : le résultat a exactement `len(fills)` lignes, réindexées de 0 à \(n-1\).
- N'utiliser, pour chaque actif, que sa ligne de référentiel de `as_of` le plus récent.
- Un actif absent du référentiel donne `multiplier`, `currency` et `notional` manquants, et `unmapped` vrai ; sinon `unmapped` est faux.
- Si `fills` est vide, les quatre colonnes existent quand même.
"""
P5 = r"""## Problème 5 : agréger en barres

Écris `bar_summary(tape, freq)`. `tape` a une ligne par transaction, de colonnes `timestamp`, `asset`, `price`, `size` (strictement positifs), non triée. `freq` est un alias de fréquence pandas (`"5min"`, `"1h"`).

Rendre une ligne par couple (barre, actif) ayant au moins une transaction, de colonnes exactement `bar`, `asset`, `vwap`, `volume`, `share`, `prev_vwap` :

- `bar` : début de la barre, sur la grille de `freq`, intervalle fermé à gauche, ouvert à droite ;
- `volume` : somme de `size` ;
- `vwap` : somme de prix fois taille, divisée par `volume` ;
- `share` : `volume` divisé par le volume total de tous les actifs dans la **même barre** ;
- `prev_vwap` : le `vwap` du **même actif** dans sa barre précédente existante, manquant pour sa première barre.

Trier par `bar` puis `asset`, réindexer, arrondir `vwap`, `volume` et `share` à 6 décimales. Ne pas créer de ligne pour une barre vide.
"""
P6 = r"""## Problème 6 : résultat réalisé et latent

Écris `track_book_with_realized(filled_requests, prices)`. Les entrées ont la forme du problème 6 de l'examen. Hypothèse : une exécution n'inverse jamais le signe d'une position.

Pour chaque actif, tenir `position`, `avg_cost` et `realized_pnl`, tous nuls au départ. Pour chaque exécution, `s = -quantity` si `side == "BUY"`, `quantity` sinon, car le teneur prend le côté opposé.

- Si `position == 0` ou si `s` a le signe de `position` (ouverture ou renforcement) : `avg_cost = (|position| * avg_cost + |s| * fill_price) / |position + s|`, puis `position += s`.
- Sinon (réduction) : `realized_pnl += min(|s|, |position|) * (fill_price - avg_cost) * signe(position)`, puis `position += s` ; `avg_cost` ne change pas.

À la fin, `unrealized_pnl[a] = (dernier_prix[a] - avg_cost[a]) * position[a]`, et `total_pnl` est la somme de tous les résultats réalisés et latents. Rendre `{"position", "avg_cost", "realized_pnl", "unrealized_pnl", "total_pnl"}`, les trois derniers arrondis à 6 décimales. Les quatre premiers sont des dictionnaires par actif.
"""
P7 = r"""## Problème 7 : le tableau de bord quotidien

Écris `daily_scorecard(fills)`. `fills` a les colonnes `timestamp`, `asset`, `quantity`, `fill_price` et peut être vide. Rendre `{"stats": ..., "daily": ..., "cumulative": ...}`, trois DataFrame arrondis à 6 décimales (`trades` reste entier).

- `stats` : indexé par actif trié, colonnes exactement `trades` (nombre d'exécutions), `volume` (somme des quantités), `avg_price` (moyenne de `fill_price`), `price_std` (écart-type à `ddof=1`, manquant pour une seule exécution).
- `daily` : lignes = dates à minuit, triées ; colonnes = actifs triés ; valeur = somme de `quantity * fill_price`. Une case sans transaction vaut 0.0. Un jour sans transaction est absent.
- `cumulative` : `daily` cumulé le long des dates, mêmes étiquettes.
"""
CODES = [
    cm.code(f"DESK-P{i}", "code", t, d, s, e, f"desk_p{i}", [cm.IMPORTS, c], "desk.py", INTRO)
    for i, (t, d, s, e, c) in enumerate([
        ("Rapprocher des confirmations", 2, 720, P1, "def reconcile_fills(confirmations):\n    pass  # À compléter"),
        ("Score d'exécution par groupe", 3, 900, P2, "def execution_quality_scores(fills):\n    pass  # À compléter"),
        ("Recherches par lot au format long", 3, 1080, P3, "def batch_reference_prices(prices_long, queries, max_staleness_minutes):\n    pass  # À compléter"),
        ("Jointure sur un référentiel à clés répétées", 4, 1080, P4, "def enrich_fills(fills, instruments):\n    pass  # À compléter"),
        ("Barres, transform et décalage par groupe", 4, 1200, P5, "def bar_summary(tape, freq):\n    pass  # À compléter"),
        ("Résultat réalisé et latent, coût moyen", 4, 1200, P6, "def track_book_with_realized(filled_requests, prices):\n    pass  # À compléter"),
        ("Agrégation unique et passage au format large", 3, 1020, P7, "def daily_scorecard(fills):\n    pass  # À compléter"),
    ], 1)
]
QUESTIONS = []
