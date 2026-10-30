"""Entraînement de la famille de l'OA Millennium : neuf problèmes de vectorisation numpy (hors reconstitution)."""
import commun_mm as cm
import questions_diagnostic as diag

ID = "oa-millennium-numpy"
TITRE = "OA Millennium : entraînement, vectorisation numpy"
VERIF = None
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0, "pause": False, "je_ne_sais_pas": False}
SECTIONS = [{"id": "code", "nom": "Neuf problèmes numpy", "budget_min": 120}]
PRESENTATION = (
    "Neuf problèmes indépendants, tous résolubles sans boucle Python sur les données et sans pandas : fenêtres glissantes, recherche "
    "triée, comptage pondéré, cumul, moindres carrés, tri stable, réductions tolérantes aux valeurs manquantes, diffusion de formes. "
    "Les tests ne contrôlent pas l'absence de boucle : c'est une consigne à tenir soi-même. Ces énoncés sont des **exercices "
    "d'entraînement écrits pour la préparation** : ils ne prétendent pas reproduire une épreuve réelle."
)
AVERTISSEMENTS = ["Exercices d'entraînement, sans lien avec Millennium ni AmplifyME.", cm.AVERTISSEMENT_VERSIONS]
INTRO = "Correction commentée des neuf problèmes"

P = [
    ("Z-score glissant", 3, "rolling_zscore(prices, window)", r"""`prices` est un tableau de flottants à une dimension, sans valeur manquante ; `window` un entier au moins égal à 1. Rendre un tableau de flottants de même longueur. L'élément \(i\) compare `prices[i]` à la fenêtre de `window` points finissant en \(i\), inclus : (valeur moins moyenne) divisé par l'écart-type **d'échantillon** (`ddof=1`).

Les `window - 1` premiers éléments valent `nan`. Si la longueur est inférieure à `window`, tout vaut `nan`. Si l'écart-type vaut 0, ou si `window` vaut 1, l'élément vaut 0.0. Outil : `np.lib.stride_tricks.sliding_window_view`."""),
    ("Recherche au plus tard, vectorisée", 3, "asof_indices(tape_ts, query_ts, max_staleness_minutes)", r"""`tape_ts` est un tableau `datetime64[ns]` trié, sans doublon, non vide ; `query_ts` un tableau `datetime64[ns]` quelconque, éventuellement vide. Rendre un tableau d'entiers de la longueur de `query_ts` : l'indice dans `tape_ts` de la dernière date inférieure ou égale à la requête, ou -1 si elle n'existe pas ou si son âge dépasse strictement `max_staleness_minutes`. Un âge égal à la limite est accepté. Outil : `np.searchsorted(..., side="right") - 1`."""),
    ("Prix moyen pondéré par étiquette", 3, "vwap_by_label(prices, sizes, labels)", r"""Trois tableaux de même longueur : prix, tailles positives ou nulles, étiquettes (chaînes, répétées, non triées). Rendre `(etiquettes, vwap)` : les étiquettes distinctes triées et, pour chacune, la somme de prix fois taille divisée par la somme des tailles. Une étiquette de taille totale nulle donne `nan`. Une entrée vide rend deux tableaux vides. Outils : `np.unique(..., return_inverse=True)` et `np.bincount(..., weights=...)`."""),
    ("Perte maximale depuis un sommet", 3, "max_drawdown(equity)", r"""`equity` est un tableau de valeurs strictement positives, dans l'ordre du temps. Avec `s[i]` le maximum de `equity[:i+1]`, on pose `d[i] = equity[i] / s[i] - 1`. Rendre `{"max_drawdown": ..., "peak_idx": ..., "trough_idx": ...}` : la valeur la plus négative de `d`, arrondie à 6 décimales, le premier indice où elle est atteinte, et le premier indice inférieur ou égal à ce creux où `equity` vaut `s[creux]`. Une courbe croissante donne 0.0, 0, 0. Un tableau vide donne 0.0, -1, -1."""),
    ("Bêtas multifactoriels", 3, "factor_betas(asset_returns, factor_returns)", r"""`asset_returns` a la forme \((n,)\), `factor_returns` la forme \((n, k)\), \(k \geq 1\). Ajuster par moindres carrés, **avec constante**, \(y_t = \alpha + \sum_j \beta_j f_{t,j} + e_t\), avec `np.linalg.lstsq(X, y, rcond=None)`, sans inverser \(X^\top X\) : cette fonction tolère des facteurs colinéaires.

Rendre `{"alpha": flottant, "betas": tableau de forme (k,), "r_squared": flottant}`, tout arrondi à 6 décimales. \(R^2 = 1 - \text{SS}_{\text{rés}}/\text{SS}_{\text{tot}}\), avec \(\text{SS}_{\text{tot}}\) calculée autour de la moyenne de \(y\) ; si \(\text{SS}_{\text{tot}}\) est nulle, \(R^2\) vaut 0.0."""),
    ("Les k meilleurs par ligne", 3, "top_k_indices(scores, k)", r"""`scores` a la forme \((n, m)\) et peut contenir `nan` ; `k` est un entier positif ou nul. Rendre un tableau d'entiers de forme \((n, \min(k, m))\) : pour chaque ligne, les indices de colonne des plus grandes valeurs, de la plus grande à la plus petite. Égalité : la plus petite colonne d'abord. Un `nan` est classé après tout réel. `k = 0` rend la forme \((n, 0)\). Outil : `np.argsort(..., kind="stable")` sur l'opposé."""),
    ("Winsoriser puis standardiser", 3, "winsorize_zscore(signal, lower_pct, upper_pct)", r"""`signal` est un tableau de flottants pouvant contenir `nan` ; `lower_pct` et `upper_pct` sont dans \([0, 100]\), avec `lower_pct <= upper_pct`.

1. Calculer les deux percentiles sur les valeurs non manquantes.
2. Ramener chaque valeur dans cet intervalle, les `nan` restant `nan`.
3. Standardiser les valeurs bornées par leur moyenne et leur écart-type d'échantillon (`ddof=1`), sans compter les `nan`.

Si l'écart-type borné est nul, rendre 0.0 partout où une valeur existe. Si tout est `nan`, rendre une copie. L'ordre importe : borner d'abord."""),
    ("Propagation vers le bas à deux dimensions", 4, "ffill_2d(tape)", r"""`tape` a la forme \((n, m)\) et peut être entièrement `nan`. Rendre un **nouveau** tableau où chaque `nan` prend la dernière valeur non manquante au-dessus de lui, dans sa colonne. Un `nan` de tête reste `nan`, aucune valeur ne remonte, les colonnes sont indépendantes. Outils : une matrice d'indices de lignes, `np.maximum.accumulate(..., axis=0)`, `np.take_along_axis`."""),
    ("Valoriser le livre", 4, "book_exposure(positions, prices, fx_rates)", r"""`positions` et `prices` ont la forme \((j, a)\) pour \(j\) jours et \(a\) actifs, `positions` étant signé ; `fx_rates` a la forme \((a,)\), un taux par actif, constant dans le temps. Soit \(u_{t,i}\) le produit position, prix, taux.

Rendre, sans boucle ni `np.tile`, `{"value": ..., "weights": ..., "gross": ...}`, arrondis à 6 décimales : `value`, de forme \((j,)\), est la somme de \(u_{t,i}\) sur les actifs ; `weights`, de forme \((j, a)\), est \(u_{t,i}\) divisé par la valeur du jour, ou 0 pour tous les actifs d'un jour de valeur nulle ; `gross`, de forme \((j,)\), est la somme des valeurs absolues. La diffusion de `fx_rates` se fait seule ; celle de la valeur du jour exige une seconde dimension de longueur 1."""),
]


def _question(i, t, d, signature, texte):
    entete = f"## Problème {i} : {t.lower()}\n\nÉcris `{signature}`.\n\n"
    cellules = [cm.IMPORTS, f"def {signature}:\n    pass  # À compléter"]
    return cm.code(f"NUM-P{i}", "code", t, d, 800, entete + texte + "\n", f"numpy_p{i}", cellules, "vectorisation.py", INTRO)


CODES = [_question(i, *p) for i, p in enumerate(P, 1)]
QUESTIONS = []
