"""Séance de market making 4 : arbitrage statistique entre un ETF et son panier.

Leçon reprise du cours de préparation Millennium (jour 4). Corrige la source sur un point : la suite de tests ne modélise
pas la latence d'exécution, et les quatre jambes d'un aller-retour sont comptées explicitement.
Exercices et tests : coding/millennium/ ; valeurs : verif/seance_mm_4.py.
"""
import commun_mm as cm
import questions_diagnostic as diag

ID = "seance-mm-4"
TITRE = "Market making 4 : arbitrage statistique entre un ETF et son panier"
VERIF = "verif/seance_mm_4.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [{"id": "lecon", "nom": "Leçon", "budget_min": 30}, {"id": "code", "nom": "Exercices de code", "budget_min": 40}]
PRESENTATION = (
    "Une leçon de 30 minutes sur l'écart entre un ETF et ses composants : cointégration, z-score et sa borne, demi-vie et biais de "
    "son estimateur, seuil de rentabilité ; puis deux exercices de code évalués.\n\n" + cm.ABANDON
)
AVERTISSEMENTS = [cm.AVERTISSEMENT_VERSIONS, "Toutes les données sont synthétiques."]

CONTENU = r"""## Leçon : parier sur la fermeture d'un écart

Durée visée : 30 minutes. Exception aux conventions : \(\hat\phi\), calculé sur des données aléatoires, est aléatoire malgré sa minuscule grecque.

### 1. Le prix synthétique

#### Le problème

Un ETF représente un panier d'actions. Il se négocie pourtant sur son propre carnet, et son prix peut s'écarter un moment de la valeur du panier. Si l'écart se referme, il y a un gain presque indépendant du marché : vendre le côté cher, acheter le côté bon marché.

#### Définitions

Le **prix synthétique** est \(\sum_i w_i p_i\), où \(p_i\) est le prix du composant \(i\) (euros) et \(w_i\) sa quantité par part. L'**écart relatif** (mispricing) vaut \(x_t = \text{prix de l'ETF}/\text{prix synthétique} - 1\), sans unité. \(x_t > 0\) : l'ETF est cher, on vend l'ETF et on achète le panier.

#### Exemple

\(w = (0{,}4 ; 0{,}6)\), prix 10 et 20 euros : le synthétique vaut 16 euros.

### 2. La cointégration

#### Le problème

Tout écart ne se referme pas. Parier sur la fermeture n'a de sens que s'il existe une force de rappel.

#### Définition

Deux séries de prix sont **cointégrées** si chacune est non stationnaire, mais qu'une combinaison linéaire d'elles est stationnaire : individuellement elles errent, leur écart reste autour d'un niveau fixe.

#### Intuition

Pour un ETF, la force de rappel a un mécanisme concret : un participant habilité peut échanger des parts contre le panier, et inversement, ce qui pousse les deux prix à converger quand l'écart dépasse les frais. Pour une paire quelconque d'actions, il n'existe pas de tel mécanisme : la stationnarité passée n'est qu'une constatation, qui peut cesser.

#### Le modèle des données

Le générateur `generate_etf_basket` fabrique \(x_t\) par \(x_t = x_{t-1} - \theta x_{t-1}\Delta + \eta\sqrt{\Delta}\,Z_t\), discrétisation d'un processus d'Ornstein-Uhlenbeck, avec \(\theta = @HL_theta:0m@\) par an, \(\eta = 0{,}004\) par racine d'année, \(\Delta = 1/252\) an, et \(Z_t\) normales centrées réduites indépendantes. C'est un AR(1) de coefficient \(\phi = 1 - \theta\Delta = @HL_phi:5m@\). Son écart-type stationnaire vaut \(\eta\sqrt{\Delta}/\sqrt{1-\phi^2}\), soit \(@AR_sd_pb:1m@\) pb. La relation est donc vraie par construction : sur des données réelles, elle serait à tester.

### 3. Le z-score

#### Le problème

Un écart de 10 centimes ne dit rien sans son échelle habituelle.

#### Définition

Pour une fenêtre \(x_1, \dots, x_n\) dont le dernier point est l'observation courante, de moyenne \(m\) et d'écart-type de population \(e\) (`ddof=0`) : \(z = (x_n - m)/e\).

#### Exemple

\((0, 0, 0, 0, 0{,}02)\) : \(m = @Z_m:3m@\), \(e = @Z_s:3m@\), \(z = @Z_ex:1m@\).

#### Énoncé

Si la fenêtre contient l'observation courante, alors \(\lvert z\rvert \leq \sqrt{n-1}\), quelles que soient les données.

#### Preuve

1. Posons \(d_i = x_i - m\), de somme nulle. Alors \(d_n = -\sum_{i<n} d_i\).
2. Par l'inégalité de Cauchy-Schwarz, \(d_n^2 \leq (n-1)\sum_{i<n} d_i^2\).
3. Donc \(n e^2 = \sum_{i \leq n} d_i^2 \geq d_n^2 + d_n^2/(n-1) = d_n^2\,n/(n-1)\).
4. Soit \(z^2 = d_n^2/e^2 \leq n-1\).

#### Pièges

L'exemple ci-dessus atteint la borne : avec \(n = 5\), aucun z ne peut dépasser 2, donc un seuil de 2 ne se déclenche que sur ce cas extrême. Pour \(n = 61\), la borne vaut \(@Z_borne_61:2m@\) et ne gêne plus. Un z-score glissant se compare aussi à une référence qui contient déjà le choc : le choc gonfle \(e\) et son propre z.

[[controle:borne]]

### 4. La demi-vie

#### Le problème

Une position immobilise du capital tant que l'écart ne s'est pas refermé. Combien de temps attendre ?

#### Énoncé

Pour \(X_t = \phi X_{t-1} + E_t\), avec \(0 < \phi < 1\) et \(E_t\) centrée indépendante du passé, l'espérance de \(X_{t+h}\) sachant \(X_t\) vaut \(\phi^h X_t\), et elle est divisée par deux en \(h = \ln 2 / (-\ln\phi)\) pas.

#### Preuve

1. Par récurrence, \(\mathbb{E}[X_{t+h} \mid X_t] = \phi\,\mathbb{E}[X_{t+h-1} \mid X_t]\), car \(E_{t+h}\) est centrée et indépendante du passé.
2. Donc \(\mathbb{E}[X_{t+h} \mid X_t] = \phi^h X_t\).
3. \(\phi^h = 1/2\) donne la formule par passage au logarithme.

#### Exemple

\(\phi = @HL_phi:5m@\) donne \(@HL_ar:2m@\) jours. Le générateur déclare \(\ln 2/\theta \times 252 = @HL_ou:2m@\) jours : la valeur du processus continu, légèrement différente parce que le pas de temps n'est pas infinitésimal.

#### L'estimation et son biais

On estime \(\phi\) par la pente de la régression de \(x_t\) sur \(x_{t-1}\), constante comprise. Cet estimateur est **biaisé vers le bas** quand \(\phi\) est proche de 1 (biais de Kendall, approximativement \(-(1+3\phi)/n\) pour \(n\) observations, résultat asymptotique admis). Le biais se traduit fortement sur la demi-vie :

| Observations | \(\phi\) moyen simulé | Approximation de Kendall | Demi-vie médiane (vraie : \(@HL_ar:1m@\)) |
|---|---|---|---|
| 252 | \(@BI_phi_252:4m@\) | \(@BI_k_252:4m@\) | \(@BI_hl_252:1m@\) |
| 3000 | \(@BI_phi_3000:4m@\) | \(@BI_k_3000:4m@\) | \(@BI_hl_3000:1m@\) |

Sur un an de données, la demi-vie estimée est environ la moitié de la vraie : le retour à la moyenne paraît deux fois plus rapide qu'il ne l'est. L'approximation est bonne à 3000 observations, moins à 252, où \(n\) n'est que quelques fois la demi-vie. Sur la série de 3000 jours du test, l'estimation vaut \(@ST_hl:1m@\).

#### Cas limites

- \(\hat\phi \geq 1\) : aucun retour à la moyenne n'est détecté, la demi-vie n'est pas finie.
- \(\hat\phi \leq 0\) : aucune persistance, l'écart disparaît dès le pas suivant ; on rend 0.

[[controle:demi]]

### 5. Deux barres et un compte de coûts

#### Le problème

Un écart inhabituel peut être trop petit pour payer. Un écart grand peut être normal pour cette paire. Il faut franchir **deux** barres.

#### Règle

Avec \(c\) la commission par jambe, en fraction du notionnel, et \(m\) un multiple de sécurité :

1. Barre statistique : \(\lvert z\rvert \geq z_e\).
2. Barre économique : \(\lvert x\rvert \geq m \times 2c\).

#### Le compte exact

Un aller-retour compte **quatre** jambes : deux à l'entrée, deux à la sortie. Son coût vaut donc \(4c\), et un multiple \(m = 2\) ne couvre le coût que tout juste si l'écart se referme entièrement. Dans le test, \(c = @CO_c_pb:0m@\) pb et \(m = @CO_m:0m@\) : le seuil vaut \(@CO_seuil_pb:0m@\) pb, le coût d'un aller-retour \(@CO_ar_pb:0m@\) pb, et l'écart-type de l'écart \(@AR_sd_pb:1m@\) pb, de sorte qu'il faut un écart d'au moins \(@CO_sigmas:2m@\) écart-type. Cette comptabilité reste optimiste : un panier réel a une jambe par composant, et non une seule.

#### Exemple

Dans la stratégie du test, sur 3000 jours, avec une entrée sur les deux barres, une sortie aux 70 % de fermeture ou après 40 jours : @ST_n@ opérations, pour un gain cumulé de \(@ST_total_pb:1m@\) pb de notionnel, après coûts. Ce nombre décrit des données construites pour être cointégrées, avec le vrai prix synthétique connu. Il ne dit rien d'un marché réel.

#### Pièges

**La latence n'est pas modélisée dans cette évaluation.** Un ordre à plusieurs jambes s'exécute jambe après jambe ; le marché peut bouger entre-temps, c'est le risque de jambe (legging risk). La méthode `Exchange.execute_with_latency` du paquet permet de le simuler, mais le test ci-dessus suppose toutes les jambes exécutées au prix de décision. Le gain affiché est donc un plafond du résultat exécutable, et non une estimation de celui-ci.

### 6. Réflexes et acquis

#### Réflexes en 30 secondes

- ETF cher : vendre l'ETF, acheter le panier.
- Demi-vie : \(\ln 2 / (-\ln\phi)\) pas.
- Fenêtre de \(n\) points incluant le présent : \(\lvert z\rvert \leq \sqrt{n-1}\).
- Aller-retour : quatre jambes au minimum.
- Un backtest sans latence est un plafond.

#### À savoir refaire

- La borne du z-score par Cauchy-Schwarz.
- La demi-vie d'un AR(1), en partant de l'espérance conditionnelle.
- Le compte des coûts d'un aller-retour et le lien entre le seuil d'entrée et l'écart-type de l'écart.

[[controle:cout]]
"""

CONTROLES = {
    "borne": {
        "question": r"Quel est le plus grand \(\lvert z\rvert\) possible sur une fenêtre de 10 points incluant l'observation courante, en écart-type de population ?",
        "options": [
            {"id": "a", "texte": "3", "explication": r"Oui : \(\sqrt{10 - 1} = 3\), atteint quand les neuf premiers points sont égaux."},
            {"id": "b", "texte": "10", "explication": r"Cette valeur oublie que l'observation courante entre dans la moyenne et dans l'écart-type, ce qui limite z."},
            {"id": "c", "texte": "Infini", "explication": r"Ce serait vrai si la fenêtre excluait l'observation courante : ici, un choc gonfle aussi le dénominateur."},
            {"id": "d", "texte": "1", "explication": r"Une série peut largement dépasser un écart-type : la borne ne vaut 1 que pour une fenêtre de deux points."},
        ],
    },
    "demi": {
        "question": r"Un AR(1) a pour coefficient \(\phi = 0{,}5\) par jour. Quelle est sa demi-vie ?",
        "options": [
            {"id": "a", "texte": "1 jour", "explication": r"Oui : \(\ln 2/(-\ln 0{,}5) = 1\), l'espérance est divisée par deux à chaque pas."},
            {"id": "b", "texte": "2 jours", "explication": r"Cette valeur confond demi-vie et durée moyenne : en un pas, l'espérance a déjà été multipliée par \(\phi = 1/2\)."},
            {"id": "c", "texte": "0,5 jour", "explication": r"Le coefficient n'est pas une durée : \(\phi\) est un facteur de rappel par pas, sans unité de temps."},
            {"id": "d", "texte": "0,69 jour", "explication": r"Cette valeur est \(\ln 2\) seul : il manque la division par \(-\ln\phi\)."},
        ],
    },
    "cout": {
        "question": r"Commission de 5 pb par jambe, ETF contre un seul instrument synthétique. Un écart de 15 pb se referme entièrement. Quel est le gain net ?",
        "options": [
            {"id": "a", "texte": "-5 pb", "explication": r"Oui : quatre jambes coûtent 20 pb, pour un gain brut de 15 pb. L'opération perd."},
            {"id": "b", "texte": "5 pb", "explication": r"Ce calcul ne compte que deux jambes : il oublie la sortie de la position."},
            {"id": "c", "texte": "10 pb", "explication": r"Ce calcul ne compte qu'une jambe : l'ETF et le panier paient tous deux, à l'entrée et à la sortie."},
            {"id": "d", "texte": "15 pb", "explication": r"Ce calcul oublie les commissions, soit justement le risque visé par la barre économique."},
        ],
    },
}

LECONS = [{"id": "L-MM-4", "type": "lecon", "section": "lecon", "theme": "Leçon : cointégration, z-score, demi-vie et coûts d'un arbitrage",
           "difficulte": 4, "temps_cible_s": 1800, "contenu": CONTENU, "controles": CONTROLES, "verif": {"script": "verif/seance_mm_4.py"}}]

E1 = r"""## Exercice 1 : estimer une demi-vie

Écris `estimate_half_life(mispricing)`. L'entrée est une `Series` d'au moins trois flottants sans valeur manquante, d'index quelconque.

1. Ajuste \(x_t = \phi x_{t-1} + c\) par moindres carrés, avec constante : `np.polyfit(x[:-1], x[1:], 1)` sur les valeurs, sans passer par l'index.
2. Si \(\hat\phi \geq 1\), rends `float("inf")`. Si \(\hat\phi \leq 0\), rends `0.0`.
3. Sinon, rends `np.log(2) / -np.log(phi)` en flottant, exprimé en pas de la série.

**Exemple.** Pour \(x_t = 0{,}5^t\), le résultat vaut 1.

**Ce que vérifient les tests.** Le cas exact, les deux cas dégénérés, et la série du générateur sur 3000 jours, où l'estimation doit tomber à 35 % près de la demi-vie déclarée.
"""

E2 = r"""## Exercice 2 : la décision d'arbitrage

Complète `CointegrationArbBot`, sous-classe de `ArbitrageBotBase`, de constructeur `(entry_z=2.0, commission_pct=0.001, min_edge_multiple=2.0)`.

- **`synthetic_price(weights, prices)`** : somme des `weights[t] * prices[t]` sur les clés `t` de `weights`, deux dictionnaires.
- **`zscore(mispricing_history)`** : pour une `Series` non vide, (dernière valeur moins moyenne) divisé par l'écart-type de **population** (`ddof=0`), calculés sur toute la série fournie. Si cet écart-type est nul, rendre `0.0`.
- **`signal(mispricing_pct, z_score)`** : `mispricing_pct` est l'écart relatif \(x\) (0.01 pour 1 %). Rendre `"no_trade"` si `abs(z_score) < entry_z`, ou si `abs(mispricing_pct) < min_edge_multiple * 2 * commission_pct`. Sinon, rendre `"sell_etf_buy_basket"` si l'écart est positif et `"buy_etf_sell_basket"` s'il est négatif. Les deux seuils sont donc inclusifs : une valeur égale au seuil passe.

**Exemple.** `zscore(pd.Series([0.0, 0.0, 0.0, 0.0, 0.02]))` vaut 2.

**Ce que vérifient les tests.** Les trois méthodes, puis la stratégie de la partie 5 sur 3000 jours, qui doit produire au moins 3 opérations et un gain cumulé positif après coûts.
"""

CODES = [
    cm.code("MM4-E1", "code", "Demi-vie d'un écart par une régression AR(1)", 3, 900, E1, "mm4_e1",
            [cm.IMPORTS + "\nfrom amplify_quant_trading import generate_etf_basket",
             "def estimate_half_life(mispricing):\n    pass  # À compléter",
             "estimate_half_life(pd.Series(0.5 ** np.arange(30)))"],
            "mm_4.py", "Correction commentée des deux exercices"),
    cm.code("MM4-E2", "code", "Décision d'arbitrage à deux barres", 4, 1500, E2, "mm4_e2",
            [cm.IMPORTS + "\nfrom amplify_quant_trading import ArbitrageBotBase",
             "class CointegrationArbBot(ArbitrageBotBase):\n    def __init__(self, entry_z=2.0, commission_pct=0.001, min_edge_multiple=2.0):\n        pass  # À compléter\n\n    def synthetic_price(self, weights, prices):\n        pass  # À compléter\n\n    def zscore(self, mispricing_history):\n        pass  # À compléter\n\n    def signal(self, mispricing_pct, z_score):\n        pass  # À compléter"],
            "mm_4.py", "Correction commentée des deux exercices"),
]
QUESTIONS = []
