"""Séance de market making 1 : cotation, écart et sélection adverse.

Leçon reprise du cours de préparation Millennium (jour 1), complétée par la démonstration de Glosten-Milgrom.
Exercices et tests : coding/millennium/ ; valeurs : verif/seance_mm_1.py.
"""
import commun_mm as cm
import questions_diagnostic as diag

ID = "seance-mm-1"
TITRE = "Market making 1 : cotation, écart et sélection adverse"
VERIF = "verif/seance_mm_1.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [{"id": "lecon", "nom": "Leçon", "budget_min": 30}, {"id": "code", "nom": "Exercices de code", "budget_min": 45}]
PRESENTATION = (
    "Une leçon de 30 minutes sur ce que fait un teneur de marché et sur la raison d'être de l'écart entre ses prix, avec la "
    "démonstration du modèle de Glosten-Milgrom ; puis deux exercices de code évalués : un teneur de marché dont l'écart suit la "
    "volatilité, et le calcul des prix de Glosten-Milgrom.\n\n" + cm.ABANDON
)
AVERTISSEMENTS = [cm.AVERTISSEMENT_VERSIONS, "Toutes les données sont synthétiques."]

CONTENU = r"""## Leçon : pourquoi un écart existe

Durée visée : 30 minutes. Exception aux conventions : \(s\) est le prix observé à l'instant de la décision, une valeur fixe ; les prix futurs, aléatoires, sont en majuscules.

### 1. Le métier et son vocabulaire

#### Le problème

Une personne qui veut acheter ou vendre maintenant a besoin d'une contrepartie disponible maintenant. Le **teneur de marché** (market maker) rend ce service : il affiche en permanence deux prix et accepte de traiter à l'un ou à l'autre.

#### Définitions

- **Bid** : le prix auquel le teneur achète. **Ask** : le prix auquel il vend. L'ask est strictement supérieur au bid.
- **Écart** (spread) : ask moins bid, en euros ; c'est le revenu brut d'un aller-retour d'une unité. **Prix médian** (mid) : la moyenne du bid et de l'ask.
- **Point de base** (pb, en anglais bps) : un dix-millième. Un écart de 100 pb sur un prix de 100 euros vaut 1 euro.
- **Exécution** (fill) : une contrepartie traite contre l'un des deux prix. **Inventaire** : le nombre d'unités détenues, net des achats et des ventes.
- **Valeur de référence** (`ref_price` dans le code) : l'estimation de la juste valeur autour de laquelle on cote.

#### Intuition

Un revendeur qui achète une paire de chaussures à 90 euros et la revend à 92 euros gagne 2 euros sans parier sur le prix : il est payé pour être disponible. Si ce service était sans risque, la concurrence ramènerait l'écart à zéro. Un écart positif et durable signale donc un risque rémunéré. Il y en a deux, indépendants : la **sélection adverse** (cette séance) et le **risque d'inventaire** (séance suivante).

### 2. La sélection adverse, démontrée

#### Le problème

Le teneur ne sait pas à qui il parle. Certaines contreparties, dites **informées**, connaissent mieux que lui la valeur future. Chaque exécution, prise isolément, a l'air d'acheter bas et de vendre haut, et pourtant elle peut perdre en moyenne. Quel écart rend l'activité viable ?

#### Le modèle de Glosten et Milgrom (1985)

Hypothèses, toutes nécessaires à ce qui suit :

1. La valeur future \(V\), variable aléatoire, vaut \(v_h\) avec probabilité \(p\) et \(v_b\) sinon, avec \(v_b < v_h\) (euros).
2. Une seule contrepartie se présente, pour une unité. Elle est informée avec probabilité \(a\), indépendamment de \(V\).
3. Une contrepartie informée connaît \(V\) : elle achète si \(V = v_h\), vend si \(V = v_b\).
4. Une contrepartie non informée achète ou vend avec probabilité \(1/2\), indépendamment de \(V\).
5. La concurrence entre teneurs annule leur gain espéré, et ils sont neutres au risque : leur prix est l'espérance de la valeur **sachant l'ordre reçu**.

#### Intuition

Un ordre d'achat est une information : il est plus probable quand la valeur est haute. Le teneur doit coter en supposant que l'ordre qu'il reçoit est celui qu'il aurait préféré ne pas recevoir. L'ask est sa valeur estimée après avoir vu un achat, le bid après avoir vu une vente.

#### Énoncé

Pour \(p = 1/2\) :

\(\displaystyle \text{ask} = \frac{v_h + v_b}{2} + a\,\frac{v_h - v_b}{2}, \qquad \text{bid} = \frac{v_h + v_b}{2} - a\,\frac{v_h - v_b}{2}, \qquad \text{ask} - \text{bid} = a\,(v_h - v_b).\)

L'écart est proportionnel à la part de flux informé et à l'amplitude de l'incertitude. Pour \(p\) quelconque, les prix sont donnés par l'étape 3 de la preuve.

#### Preuve

1. Par les hypothèses 3 et 4, \(\mathbb{P}(\text{achat} \mid V = v_h) = a + (1-a)/2 = (1+a)/2\) : soit la contrepartie est informée, et elle achète, soit elle ne l'est pas, et elle achète une fois sur deux. De même, \(\mathbb{P}(\text{achat} \mid V = v_b) = (1-a)/2\).
2. Par la formule de Bayes, \(\mathbb{P}(V = v_h \mid \text{achat}) = \dfrac{p\,(1+a)/2}{p\,(1+a)/2 + (1-p)\,(1-a)/2}\).
3. L'hypothèse 5 donne \(\text{ask} = \mathbb{E}[V \mid \text{achat}] = v_h\,\mathbb{P}(V = v_h \mid \text{achat}) + v_b\,\mathbb{P}(V = v_b \mid \text{achat})\), et de même pour le bid en échangeant les rôles de \((1+a)/2\) et \((1-a)/2\).
4. Pour \(p = 1/2\), le dénominateur de l'étape 2 vaut \(1/2\), donc \(\mathbb{P}(V = v_h \mid \text{achat}) = (1+a)/2\).
5. Alors \(\text{ask} = v_h(1+a)/2 + v_b(1-a)/2 = (v_h+v_b)/2 + a(v_h-v_b)/2\). Le calcul du bid est symétrique, et la différence donne l'écart.

#### Exemple entièrement calculé

\(v_b = @GM_vb@\), \(v_h = @GM_vh@\), \(p = 1/2\), \(a = @GM_a:2m@\). Alors \(\mathbb{P}(\text{achat} \mid V = v_h) = @GM_pa_h:3m@\) et \(\mathbb{P}(\text{achat} \mid V = v_b) = @GM_pa_b:3m@\). L'ask vaut \(@GM_ask:2m@\), le bid \(@GM_bid:2m@\), l'écart \(@GM_ecart:2m@\).

Contrôle du gain nul sur un achat. La contrepartie est informée avec probabilité \(@GM_p_inf:2m@\) sachant l'achat : le teneur vend alors à \(@GM_ask:2m@\) ce qui vaut \(@GM_vh@\), et perd \(@GM_perte:2m@\). Sinon, avec probabilité \(@GM_p_noninf:2m@\), il gagne \(@GM_gain:2m@\) contre une valeur moyenne de 100. L'espérance est \(0{,}75 \times 0{,}5 - 0{,}25 \times 1{,}5 = 0\). Une simulation de deux millions d'ordres donne \(\mathbb{E}[V \mid \text{achat}] \approx @GM_mc:4m@\), avec une erreur standard de \(@GM_mc_se:4m@\), cohérente avec la valeur exacte.

#### Cas limites

- \(a = 0\) : personne n'est informé et l'écart est nul dans ce modèle. Un écart observé s'expliquerait alors par d'autres coûts (inventaire, frais).
- \(a = 1\) : l'ask vaut \(v_h\) et le bid \(v_b\), et le marché n'apporte plus aucun gain à quiconque.
- \(p = 0\) ou \(p = 1\) : la valeur est certaine et les deux prix se confondent avec elle.

#### Pièges

- Le modèle ne dit **pas** que l'écart suit toujours la volatilité : il dit qu'il suit \(a(v_h - v_b)\). L'amplitude \(v_h - v_b\) joue le rôle d'une incertitude sur la valeur. Si la volatilité augmente sans que la part de flux informé \(a\) change, l'écart de ce modèle augmente proportionnellement ; si \(a\) diminue en même temps, le résultat peut s'inverser.
- Le prix d'un ordre intègre l'information de cet ordre : le prix se déplace après chaque exécution. C'est ce déplacement qui fait perdre à qui cote trop serré.

[[controle:gm]]

### 3. Faire suivre la volatilité à l'écart

#### Le problème

Le teneur n'observe ni \(a\) ni \(v_h - v_b\). Il observe en revanche la **volatilité réalisée** : l'écart-type des rendements logarithmiques récents, annualisé. Elle sert d'approximation de l'incertitude. C'est une règle de décision, pas un théorème : le lien entre volatilité réalisée et danger du prochain ordre n'est pas démontré ici.

#### La règle de l'exercice

\(\displaystyle \text{écart}_{\text{pb}} = \min\bigl(\max(e_0 + c\,\hat{\sigma},\ e_{\min}),\ e_{\max}\bigr)\)

où \(e_0\) est l'écart de base en points de base, \(c\) la sensibilité (points de base par unité de volatilité annualisée), \(\hat{\sigma}\) la volatilité réalisée annualisée, sans unité (0,20 signifie 20 % par an), \(e_{\min}\) et \(e_{\max}\) le plancher et le plafond.

Avec \(e_0 = 300\) et \(c = 800\) : 20 % de volatilité ajoute \(@EC_ajout@\) pb ; l'écart vaut \(@EC_calme@\) pb à 15 % et \(@EC_tension@\) pb à 55 %. Sur une journée simulée de 4000 pas, l'écart moyen est de @B_calme@ pb hors de la période de tension et de @B_tension@ pb pendant celle-ci.

Le plafond a une raison d'être : au-delà d'une certaine largeur, un prix n'est plus une offre réelle de traiter, et un desk préfère cesser de coter.

#### Ce que l'estimateur ne voit pas

La volatilité réalisée est calculée sur une fenêtre passée : elle est en retard. Sur l'exemple, l'écart met @B_retard@ pas à franchir la moitié du chemin entre ses niveaux moyens après le début de la tension. Ce retard est une caractéristique de l'estimateur et de la fenêtre de 30 pas du moteur, pas une anomalie à corriger.

### 4. Le moteur de simulation et ses limites

#### Le modèle

La fonction `simulate_market_making` déroule un chemin de prix. À chaque pas, elle appelle `quote`, puis l'ask est levé avec probabilité \(b\,e^{-d\,h}\), où \(b = @FB:2m@\) est l'intensité de base, \(d = @FD@\) la décroissance et \(h\) la distance relative entre le prix coté et la juste valeur. Le bid suit la même règle.

#### Exemple

À 420 pb d'écart, \(h = 0{,}021\) et la probabilité par pas et par côté vaut \(@F_p420:4m@\). Le gain espéré par pas et par côté, en fraction du prix, est \(h\,b\,e^{-d h}\) : sa dérivée s'annule en \(h = 1/d\), soit un demi-écart de \(@F_hopt_bps:1m@\) pb et un écart de \(@F_ecart_opt:1m@\) pb. Les valeurs par défaut de l'exercice cotent bien plus large : elles ne maximisent pas ce gain, elles répondent à un cahier des charges.

#### Pièges

**Ce moteur ne contient aucun flux informé.** Les exécutions ne dépendent pas du mouvement suivant : sur @S_jours@ journées, la corrélation entre le sens net des exécutions et le rendement des 20 pas suivants vaut \(@S_corr:4m@\), avec une erreur standard de \(@S_corr_se:4m@\). La sélection adverse de la partie 2 n'y existe donc pas, et l'exercice entraîne la mécanique de la cotation, pas la mesure de ce risque. Le backtest ne prouve pas qu'élargir l'écart protège d'un flux informé.

[[controle:moteur]]

### 5. Réflexes et acquis

#### Réflexes en 30 secondes

- Un client achète : le teneur vend, à l'ask.
- Écart en euros : prix multiplié par les points de base, divisé par 10 000.
- Prix de Glosten-Milgrom : espérance de la valeur sachant l'ordre, donc une formule de Bayes.
- \(p = 1/2\) : écart égal à \(a(v_h - v_b)\).
- Une volatilité estimée sur le passé est en retard sur un changement de régime.

#### À savoir refaire

- La démonstration de l'écart de Glosten-Milgrom, et le contrôle du gain nul sur un achat.
- Le passage des points de base à un bid et un ask autour d'un prix de référence.
- La dérivation du demi-écart \(1/d\) qui maximise le gain espéré du moteur.

[[controle:unites]]
"""

CONTROLES = {
    "gm": {
        "question": r"Dans le modèle de la partie 2, \(v_b = 90\), \(v_h = 110\), \(p = 1/2\), \(a = 0{,}1\). Quel est l'écart ?",
        "options": [
            {"id": "a", "texte": "2", "explication": r"Oui : \(a(v_h - v_b) = 0{,}1 \times 20 = 2\), soit un bid de 99 et un ask de 101."},
            {"id": "b", "texte": "1", "explication": r"Cette valeur est le demi-écart : elle sépare chaque prix de la valeur moyenne 100, pas le bid de l'ask."},
            {"id": "c", "texte": "20", "explication": r"C'est l'amplitude de la valeur, atteinte seulement si \(a = 1\), c'est-à-dire si tout le flux est informé."},
            {"id": "d", "texte": "0", "explication": r"Un écart nul suppose \(a = 0\). Ici un ordre sur dix est informé et il faut couvrir la perte correspondante."},
        ],
    },
    "moteur": {
        "question": r"Un backtest sur `simulate_market_making` montre un gain positif avec un écart serré. Que peut-on en conclure sur la sélection adverse ?",
        "options": [
            {"id": "a", "texte": "Rien : le moteur n'en contient pas", "explication": r"Oui : les exécutions y sont indépendantes du mouvement suivant, comme le confirme la corrélation mesurée, nulle aux erreurs près."},
            {"id": "b", "texte": "Qu'elle est faible sur cet actif", "explication": r"Les données sont simulées et aucun flux informé n'y est généré : on ne peut rien apprendre sur un risque absent du modèle."},
            {"id": "c", "texte": "Que l'écart la couvre exactement", "explication": r"Cette propriété est celle de l'équilibre de Glosten-Milgrom, un modèle différent, où le gain espéré est nul par construction."},
            {"id": "d", "texte": "Que la volatilité la mesure bien", "explication": r"Le moteur n'établit aucun lien entre volatilité et information : il n'y a pas d'information du tout dans ses exécutions."},
        ],
    },
    "unites": {
        "question": r"Prix de référence 250 euros, écart de 300 pb. Quels sont le bid et l'ask ?",
        "options": [
            {"id": "a", "texte": "246,25 et 253,75", "explication": r"Oui : l'écart vaut \(250 \times 0{,}03 = 7{,}5\) euros, soit 3,75 de chaque côté."},
            {"id": "b", "texte": "242,50 et 257,50", "explication": r"Ce calcul place l'écart entier de chaque côté : le total serait de 600 pb, le double de ce qui est demandé."},
            {"id": "c", "texte": "247,00 et 253,00", "explication": r"Ce calcul prend 3 euros au lieu de 3 % : il confond une distance en euros et une distance relative au prix."},
            {"id": "d", "texte": "249,25 et 250,75", "explication": r"Ce calcul prend 300 pb pour 0,3 % : 1 pb vaut un dix-millième, donc 300 pb valent 3 %, pas 0,3 %."},
        ],
    },
}

LECONS = [{"id": "L-MM-1", "type": "lecon", "section": "lecon", "theme": "Leçon : le métier de teneur de marché, l'écart et la sélection adverse",
           "difficulte": 3, "temps_cible_s": 1800, "contenu": CONTENU, "controles": CONTROLES, "verif": {"script": "verif/seance_mm_1.py"}}]

E1 = r"""## Exercice 1 : un teneur de marché sensible à la volatilité

Complète la classe `VolAwareMarketMaker`, sous-classe de `MarketMakerBase`.

**Objets fournis.** `MarketState` porte les attributs `ref_price` (prix de référence, en euros, au moins égal à 1), `realized_vol` (volatilité réalisée annualisée, positive ou nulle, 0.20 pour 20 %), `time_remaining` et `inventory`, ces deux derniers étant inutiles ici. `Quote(bid, ask)` est un prix double : il lève `ValueError` si un prix n'est pas strictement positif ou si `bid >= ask` ; il a les propriétés `mid`, `spread` et `spread_bps`.

**Constructeur.** `VolAwareMarketMaker(base_spread_bps=300.0, vol_sensitivity=800.0, min_spread_bps=100.0, max_spread_bps=900.0)` conserve ses quatre paramètres.

**Méthode `quote(state)`.**

1. Si `state.ref_price` n'est pas strictement positif, lever `ValueError`.
2. Calculer `s = base_spread_bps + vol_sensitivity * state.realized_vol`, puis le ramener dans l'intervalle `[min_spread_bps, max_spread_bps]`.
3. Le demi-écart en euros vaut `ref_price * s / 10000 / 2`.
4. Rendre `Quote(bid=round(ref_price - demi_ecart, 2), ask=round(ref_price + demi_ecart, 2))`.

`quote` est une fonction pure de `state` et de la configuration : deux appels identiques rendent le même résultat, sans hasard ni modification d'état.

**Exemple.** Avec les valeurs par défaut, `ref_price=100.0` et `realized_vol=0.20`, l'écart vaut 460 pb, donc `Quote(bid=97.7, ask=102.3)`.

**Ce que vérifient les tests.** Validité du prix double, écart strictement positif à volatilité nulle, écart croissant avec la volatilité, plafond, prix de référence compris entre le bid et l'ask, et trois simulations : l'écart moyen pendant la période de tension dépasse 1,15 fois l'écart moyen hors tension, les exécutions dépassent 1 % des pas, et le résultat reste fini.
"""

E2 = r"""## Exercice 2 : les prix de Glosten-Milgrom

Écris `glosten_milgrom(v_bas, v_haut, proba_haut, part_informes)`, qui rend le couple `(bid, ask)` en flottants, pour le modèle de la partie 2 de la leçon, avec une probabilité \(p\) quelconque.

**Entrées.** `v_bas` et `v_haut` sont les deux valeurs possibles, `proba_haut` est \(p\), `part_informes` est \(a\). Si `v_bas >= v_haut`, ou si `proba_haut` ou `part_informes` sort de l'intervalle fermé \([0, 1]\), lever `ValueError`.

**Sortie.** `ask` est \(\mathbb{E}[V \mid \text{achat}]\) et `bid` est \(\mathbb{E}[V \mid \text{vente}]\). Si un ordre a une probabilité nulle (par exemple \(p = 1\) et \(a = 1\) pour une vente), le prix correspondant est la valeur certaine : `v_haut` si \(p = 1\), `v_bas` sinon.

**Exemple.** `glosten_milgrom(98.0, 102.0, 0.5, 0.25)` rend `(99.5, 100.5)`.
"""

CODES = [
    cm.code("MM1-E1", "code", "Coter un écart qui suit la volatilité", 3, 1500, E1, "mm1_e1",
            [cm.IMPORTS + "\nfrom amplify_quant_trading import MarketMakerBase, MarketState, Quote",
             "class VolAwareMarketMaker(MarketMakerBase):\n    def __init__(self, base_spread_bps=300.0, vol_sensitivity=800.0, min_spread_bps=100.0, max_spread_bps=900.0):\n        pass  # À compléter\n\n    def quote(self, state):\n        pass  # À compléter",
             "VolAwareMarketMaker().quote(MarketState(timestamp=0, ref_price=100.0, realized_vol=0.20, time_remaining=1.0))"],
            "mm_1.py", "Correction commentée des deux exercices"),
    cm.code("MM1-E2", "code", "Prix de Glosten-Milgrom par la formule de Bayes", 4, 1200, E2, "mm1_e2",
            [cm.IMPORTS, "def glosten_milgrom(v_bas, v_haut, proba_haut, part_informes):\n    pass  # À compléter",
             "glosten_milgrom(98.0, 102.0, 0.5, 0.25)"],
            "mm_1.py", "Correction commentée des deux exercices"),
]
QUESTIONS = []
