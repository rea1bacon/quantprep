"""Séance de market making 3 : couverture et bêta.

Leçon reprise du cours de préparation Millennium (jour 3). Corrige deux erreurs de la source : le mélange de ddof dans le
bêta de référence, et l'idée qu'une efficacité de 1 signalerait une sur-couverture.
Exercices et tests : coding/millennium/ ; valeurs : verif/seance_mm_3.py.
"""
import commun_mm as cm
import questions_diagnostic as diag

ID = "seance-mm-3"
TITRE = "Market making 3 : couverture et bêta"
VERIF = "verif/seance_mm_3.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [{"id": "lecon", "nom": "Leçon", "budget_min": 25}, {"id": "code", "nom": "Exercices de code", "budget_min": 40}]
PRESENTATION = (
    "Une leçon de 25 minutes sur la couverture d'une position par un ETF : le bêta comme solution d'un problème de minimisation, "
    "l'erreur d'estimation, la bande de non-intervention et la mesure de l'efficacité ; puis deux exercices de code évalués.\n\n" + cm.ABANDON
)
AVERTISSEMENTS = [cm.AVERTISSEMENT_VERSIONS, "Toutes les données sont synthétiques."]

CONTENU = r"""## Leçon : retirer le risque qui n'est pas rémunéré

Durée visée : 25 minutes. Exception aux conventions : \(\hat{\beta}\), calculé sur des rendements aléatoires, est une variable aléatoire malgré sa minuscule grecque.

### 1. Le ratio de couverture

#### Le problème

Un fonds détient une action parce qu'il pense qu'elle battra le marché. Son rendement contient pourtant une part qui vient du marché entier, sur laquelle il ne prétend à aucun avantage. Vendre l'action retirerait aussi le pari voulu. Il cherche plutôt à vendre à découvert un ETF (fonds coté qui suit un indice), juste assez pour annuler la part de marché.

#### Cadre

\(R_a\) est le rendement de l'actif sur une période, \(R_c\) celui de l'instrument de couverture, deux variables aléatoires de variance finie, avec \(\mathrm{Var}(R_c) > 0\). Une couverture de ratio \(b\) (nombre réel) donne le rendement \(R_a - b\,R_c\) par euro d'actif.

#### Énoncé

La variance de \(R_a - b R_c\) est minimale pour un unique \(b\), le **bêta** :

\(\displaystyle \beta = \frac{\mathrm{Cov}(R_a, R_c)}{\mathrm{Var}(R_c)}.\)

Le minimum vaut \((1 - \rho^2)\mathrm{Var}(R_a)\), où \(\rho\) est la corrélation entre \(R_a\) et \(R_c\).

#### Preuve

1. \(\mathrm{Var}(R_a - bR_c) = \mathrm{Var}(R_a) - 2b\,\mathrm{Cov}(R_a, R_c) + b^2\,\mathrm{Var}(R_c)\), par bilinéarité de la covariance.
2. C'est un trinôme en \(b\), de coefficient dominant strictement positif : son minimum unique est au point où sa dérivée s'annule, soit \(b = \beta\).
3. En remplaçant : \(\mathrm{Var}(R_a) - \mathrm{Cov}(R_a, R_c)^2/\mathrm{Var}(R_c) = \mathrm{Var}(R_a)(1 - \rho^2)\), par définition de \(\rho\).

Ce même \(\beta\) est la pente de la droite des moindres carrés de \(R_a\) sur \(R_c\) : son estimateur empirique est donc une régression.

#### Exemple

@H_E@ euros d'une action de bêta \(@H_b:2m@\) : on vend à découvert \(@H_not@\) euros de l'ETF. Si l'ETF baisse de 1 %, l'action baisse en moyenne de \(@H_pct:2m@\) %, soit \(@H_perte@\) euros, et la position courte gagne \(@H_perte@\) euros. Il reste le rendement propre à l'action.

#### Cas limites

- \(\rho = 0\) : la couverture ne retire rien. \(\lvert\rho\rvert = 1\) : elle retire toute la variance.
- Une action dont le risque est surtout spécifique reste peu couvrable : c'est une propriété de l'actif, pas une erreur de l'estimateur.

### 2. Mesurer l'efficacité

#### Définition

\(\displaystyle \text{efficacité} = 1 - \frac{\mathrm{Var}(\text{rendements couverts})}{\mathrm{Var}(\text{rendements non couverts})}.\)

D'après la partie 1, cette quantité vaut \(\rho^2\) au bêta exact. Pour \(R_a = \beta R_c + E\) avec \(E\) indépendant de \(R_c\), de variance \(\sigma_i^2\), et \(\sigma_m^2 = \mathrm{Var}(R_c)\) : \(\rho^2 = \beta^2\sigma_m^2/(\beta^2\sigma_m^2 + \sigma_i^2)\).

#### Exemple

\(\beta = @R2_b:1m@\), volatilités annualisées \(\sigma_m = @R2_sm:2m@\) et \(\sigma_i = @R2_si:2m@\) : l'efficacité maximale vaut \(@R2_th:4m@\). Sur une simulation de 1500 jours, une couverture par bêta glissant de 60 jours, estimé sur le passé seulement, obtient \(@R2_mes:4m@\). L'écart vient de l'erreur d'estimation et de l'échantillon.

#### Cas limites

- 1 : toute la variance a disparu. Cela demande \(\lvert\rho\rvert = 1\) et un bêta exact ; une sur-couverture, au contraire, **diminue** l'efficacité, puisque \(b \neq \beta\) n'est pas le minimum.
- 0 : aucun effet. Valeur négative : la couverture a ajouté du risque, ce qui arrive avec un signe faux ou un \(b\) trop grand.

[[controle:eff]]

### 3. L'erreur d'estimation

#### Le problème

Le bêta n'est jamais observé. Une fenêtre courte donne une estimation bruitée ; une fenêtre longue, une estimation stable mais lente à réagir si la relation change.

#### Énoncé

Supposons \(n\) périodes indépendantes, \(R_a = \alpha + \beta R_c + E\) avec \(E\) normale de variance \(\sigma_i^2\) indépendante de \(R_c\), et \(R_c\) normale de variance \(\sigma_m^2\), \(n > 3\). Alors l'estimateur des moindres carrés \(\hat\beta\) est sans biais et \(\mathrm{Var}(\hat\beta) = \dfrac{\sigma_i^2}{\sigma_m^2\,(n-3)}\).

#### Preuve

1. Conditionnellement aux \(R_c\), \(\hat\beta - \beta = \sum_t (R_{c,t} - \bar{R}_c) E_t / \sum_t (R_{c,t} - \bar{R}_c)^2\), d'espérance nulle et de variance \(\sigma_i^2 / \sum_t (R_{c,t} - \bar{R}_c)^2\).
2. L'espérance conditionnelle nulle donne l'absence de biais. Par la variance totale, \(\mathrm{Var}(\hat\beta) = \sigma_i^2\,\mathbb{E}\bigl[1/\sum_t (R_{c,t} - \bar{R}_c)^2\bigr]\).
3. \(\sum_t (R_{c,t} - \bar{R}_c)^2 / \sigma_m^2\) suit une loi du khi-deux à \(n - 1\) degrés de liberté, dont l'inverse a pour espérance \(1/(n-3)\) (intégrale directe de la densité, admise ici).

#### Exemple

Avec les paramètres de la partie 2, l'écart-type de \(\hat\beta\) vaut :

| Fenêtre (jours) | Exact | Monte-Carlo |
|---|---|---|
| 10 | \(@SE_ex_10:4m@\) | \(@SE_mc_10:4m@\) |
| 60 | \(@SE_ex_60:4m@\) | \(@SE_mc_60:4m@\) |
| 250 | \(@SE_ex_250:4m@\) | \(@SE_mc_250:4m@\) |

Une fenêtre de 10 jours laisse un écart-type de \(@SE_ex_10:2m@\) pour un bêta de \(@R2_b:1m@\) : le ratio de couverture y contient beaucoup de bruit. La longueur optimale dépend de la vitesse à laquelle le vrai bêta change, qui n'est pas observable : elle n'a pas de valeur universelle.

#### Pièges

**Le même `ddof` en haut et en bas.** `np.cov(x, y)` calcule en `ddof=1`, alors que `np.var(y)` calcule en `ddof=0`. Les mélanger multiplie le bêta par \(n/(n-1)\), soit \(@DD_rapport:4m@\) pour \(n = 60\) : sur un exemple, \(@DD_faux:4m@\) au lieu de \(@DD_bon:4m@\). Rien ne plante, et un test à tolérance large ne le voit pas. Cette erreur figurait dans le code de référence d'origine de cet exercice. Le contrôle sûr est de comparer à la pente de `np.polyfit`.

### 4. La bande de non-intervention

#### Le problème

Réajuster à chaque variation de l'estimation coûte une commission pour corriger surtout du bruit. Ne jamais réajuster laisse dériver l'exposition.

#### Règle

Avec \(c\) la valeur de la position (euros), la cible est \(-c\,b\) et l'ajustement est \(\Delta = -c\,b - \text{couverture actuelle}\). On ne traite que si \(\lvert\Delta\rvert\) atteint une fraction \(f\) de \(\lvert c\rvert\). \(\Delta < 0\) : la couverture doit devenir plus courte, on **vend**. \(\Delta > 0\) : on **rachète**. Le volume est \(\lvert\Delta\rvert\) divisé par le prix de l'instrument.

#### Exemple

\(c = 100\,000\), \(b = 0{,}70\), \(f = 3\) %, soit une bande de 3000 euros et une cible de \(-70\,000\).

- Couverture à \(-69\,000\) : \(\Delta = -1000\), rien à faire.
- Couverture à 0, prix 50 : \(\Delta = -70\,000\), vente de \(@BA_v1@\) unités.
- Couverture à \(-95\,000\), prix 25 : \(\Delta = +25\,000\), rachat de \(@BA_v2@\) unités.

#### Pièges

- Le signe. Se tromper double l'exposition au lieu de l'annuler.
- Le regard vers le futur : le bêta appliqué au rendement du jour \(t\) doit être estimé sur des rendements strictement antérieurs à \(t\).

[[controle:signe]]

### 5. Réflexes et acquis

#### Réflexes en 30 secondes

- Couverture à la variance minimale : \(\beta = \mathrm{Cov}/\mathrm{Var}\).
- Efficacité maximale : \(\rho^2\).
- Notionnel vendu : valeur de la position fois \(\beta\).
- Plus court : vente. Moins court : achat.
- Même `ddof` au numérateur et au dénominateur.

#### À savoir refaire

- La minimisation du trinôme et la valeur \((1-\rho^2)\mathrm{Var}(R_a)\) du minimum.
- La variance de \(\hat\beta\) conditionnellement aux rendements de la couverture.
- Les trois cas de la bande de non-intervention.

[[controle:fenetre]]
"""

CONTROLES = {
    "eff": {
        "question": r"Corrélation de 0,6 entre l'action et l'ETF. Quelle part de la variance une couverture au vrai bêta retire-t-elle ?",
        "options": [
            {"id": "a", "texte": "36 %", "explication": r"Oui : l'efficacité maximale est \(\rho^2 = 0{,}36\)."},
            {"id": "b", "texte": "60 %", "explication": r"La corrélation n'est pas la part de variance : c'est son carré qui l'est, comme le montre le calcul du minimum."},
            {"id": "c", "texte": "64 %", "explication": r"Cette valeur, \(1 - \rho^2\), est la part de variance qui reste après couverture, pas celle qui est retirée."},
            {"id": "d", "texte": "100 %", "explication": r"Retirer toute la variance demande une corrélation de 1 ou de -1 : ici un risque spécifique subsiste."},
        ],
    },
    "signe": {
        "question": r"Position de 200 000 euros, bêta cible 0,5, couverture actuelle de -60 000 euros, bande de 2 %. Quel ordre passer ?",
        "options": [
            {"id": "a", "texte": "Vendre pour 40 000 euros", "explication": r"Oui : la cible est -100 000, donc \(\Delta = -40\,000\), hors de la bande de 4000 euros, et il faut devenir plus court."},
            {"id": "b", "texte": "Acheter pour 40 000 euros", "explication": r"Le signe est inversé : acheter ramènerait la couverture à -20 000, soit plus loin encore de la cible de -100 000."},
            {"id": "c", "texte": "Ne rien faire", "explication": r"L'ajustement de 40 000 euros dépasse la bande de 4000 euros : il faut traiter."},
            {"id": "d", "texte": "Vendre pour 100 000 euros", "explication": r"Ce montant est la cible et non l'ajustement : il ignore les 60 000 euros déjà vendus à découvert."},
        ],
    },
    "fenetre": {
        "question": r"Dans le modèle de la partie 3, par combien faut-il environ multiplier la fenêtre pour diviser par deux l'écart-type de \(\hat\beta\) ?",
        "options": [
            {"id": "a", "texte": "Environ 4", "explication": r"Oui : l'écart-type décroît en \(1/\sqrt{n-3}\), donc il faut quatre fois plus de données pour une grande fenêtre."},
            {"id": "b", "texte": "Environ 2", "explication": r"Doubler la fenêtre divise la variance par deux, donc l'écart-type par seulement \(\sqrt{2}\)."},
            {"id": "c", "texte": "Environ 8", "explication": r"Aucune puissance trois n'intervient : la variance est inversement proportionnelle à \(n - 3\)."},
            {"id": "d", "texte": "Rien ne change", "explication": r"La formule dépend bien de \(n\) : seule la vitesse de changement du vrai bêta n'y figure pas."},
        ],
    },
}
LECONS = [{"id": "L-MM-3", "type": "lecon", "section": "lecon", "theme": "Leçon : bêta, efficacité, erreur d'estimation et bande de non-intervention",
           "difficulte": 3, "temps_cible_s": 1500, "contenu": CONTENU, "controles": CONTROLES, "verif": {"script": "verif/seance_mm_3.py"}}]

E1 = r"""## Exercice 1 : un couvreur à bande de non-intervention

Complète `RollingHedger`, sous-classe de `HedgerBase`, de constructeur `(window=60, no_trade_band_pct=0.03)`.

**`hedge_ratio(asset_returns, hedge_returns)`** rend, en flottant, le bêta de la partie 1 : covariance empirique des deux séries divisée par la variance empirique de `hedge_returns`, **avec le même `ddof`**. Les deux entrées sont des `Series` ou des tableaux de même longueur, sans valeur manquante. Le résultat doit égaler la pente de `np.polyfit(hedge_returns, asset_returns, 1)`.

**`rebalance(equity_value, current_hedge_notional, target_ratio, hedge_price, timestamp=0)`** rend l'ordre à passer, ou `None` :

1. Si `hedge_price` n'est pas strictement positif, lever `ValueError`.
2. `ajustement = -equity_value * target_ratio - current_hedge_notional`.
3. Si `ajustement` est nul, ou si `abs(ajustement) < no_trade_band_pct * abs(equity_value)`, rendre `None`. Un ajustement exactement égal à la bande est traité.
4. Sinon, rendre `ExchangeTrade(ticker="HEDGE", trade_volume=abs(ajustement) / hedge_price, ref_price=hedge_price, action=..., timestamp=timestamp)`, avec `action="sell"` si `ajustement < 0`, `"buy"` sinon.

**Exemple.** Avec `no_trade_band_pct=0.03`, `rebalance(100000, 0.0, 0.70, 50.0)` rend un ordre `sell` de volume 1400.0.
"""

E2 = r"""## Exercice 2 : des rendements couverts sans regard vers le futur

Écris `rendements_couverts(actif, couverture, fenetre)`. Les deux premières entrées sont des `Series` de rendements de même longueur \(n\), d'index 0 à \(n-1\), sans valeur manquante ; `fenetre` est un entier au moins égal à 2.

**Sortie.** Une `Series` de longueur \(n\). Pour \(t\) < `fenetre`, la valeur est `NaN`. Pour \(t\) \(\geq\) `fenetre`, on estime \(\hat\beta_t\) sur les `fenetre` rendements des indices \(t - \text{fenetre}\) à \(t - 1\), bornes comprises, et la valeur est `actif[t] - beta * couverture[t]`.

**Ce que vérifient les tests.** La forme, la première valeur, et l'absence de regard vers le futur : modifier une valeur d'indice 100 ou plus ne change aucune sortie d'indice inférieur à 100.
"""

CODES = [
    cm.code("MM3-E1", "code", "Bêta et bande de non-intervention", 3, 1320, E1, "mm3_e1",
            [cm.IMPORTS + "\nfrom amplify_quant_trading import ExchangeTrade, HedgerBase",
             "class RollingHedger(HedgerBase):\n    def __init__(self, window=60, no_trade_band_pct=0.03):\n        pass  # À compléter\n\n    def hedge_ratio(self, asset_returns, hedge_returns):\n        pass  # À compléter\n\n    def rebalance(self, equity_value, current_hedge_notional, target_ratio, hedge_price, timestamp=0):\n        pass  # À compléter",
             "RollingHedger().rebalance(100000, 0.0, 0.70, 50.0)"],
            "mm_3.py", "Correction commentée des deux exercices"),
    cm.code("MM3-E2", "code", "Couverture glissante sans regard vers le futur", 4, 1080, E2, "mm3_e2",
            [cm.IMPORTS, "def rendements_couverts(actif, couverture, fenetre):\n    pass  # À compléter"],
            "mm_3.py", "Correction commentée des deux exercices"),
]
QUESTIONS = []
