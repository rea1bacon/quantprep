"""Séance de market making 2 : risque d'inventaire et modèle d'Avellaneda-Stoikov.

Leçon reprise du cours de préparation Millennium (jour 2). La dérivation du terme de liquidité, présentée à la source
comme hors de portée sans équation de Hamilton-Jacobi-Bellman, est donnée ici, avec son approximation déclarée.
Exercices et tests : coding/millennium/ ; valeurs : verif/seance_mm_2.py.
"""
import commun_mm as cm
import questions_diagnostic as diag

ID = "seance-mm-2"
TITRE = "Market making 2 : inventaire et modèle d'Avellaneda-Stoikov"
VERIF = "verif/seance_mm_2.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [{"id": "lecon", "nom": "Leçon", "budget_min": 35}, {"id": "code", "nom": "Exercices de code", "budget_min": 40}]
PRESENTATION = (
    "Une leçon de 35 minutes sur le risque d'inventaire : équivalent certain, prix d'indifférence, formules d'Avellaneda et Stoikov et "
    "conversions d'unités ; puis deux exercices de code évalués.\n\n" + cm.ABANDON
)
AVERTISSEMENTS = [cm.AVERTISSEMENT_VERSIONS, "Toutes les données sont synthétiques."]

CONTENU = r"""## Leçon : gérer un inventaire subi

Durée visée : 35 minutes. Exception aux conventions : \(W\), \(S_T\) sont aléatoires ; \(s\), \(q\) sont le prix et l'inventaire observés à l'instant de la décision, donc fixes.

### 1. Le risque d'inventaire

#### Le problème

Sans aucun flux informé, un flux d'ordres déséquilibré laisse le teneur avec une position qu'il n'a pas choisie. Exemple : cotant 99,50 et 100,50, il subit 200 achats de clients et se retrouve à 200 unités vendues à découvert. Si le prix passe à 103 euros, racheter coûte 600 euros de plus, sans qu'aucune exécution n'ait été mauvaise isolément.

#### Intuition

Le remède est de faire travailler ses propres prix contre l'accumulation. Détenteur de stock, le teneur baisse ses deux prix : son ask plus bas attire les acheteurs, son bid plus bas décourage les vendeurs. En position courte, il les monte. On déplace le **centre** de la cotation ; sa **largeur** relève d'une autre décision.

Reste à savoir de combien déplacer. La réponse vient d'un calcul d'indifférence.

### 2. L'équivalent certain

#### Cadre

\(\gamma > 0\) est l'**aversion au risque** (inverse d'euros). La satisfaction d'une richesse \(w\) (euros) est \(u(w) = -e^{-\gamma w}\), l'utilité dite CARA. L'**équivalent certain** d'une richesse aléatoire \(W\) est le montant sûr qui donne la même utilité espérée.

#### Énoncé

Si \(W\) suit une loi normale de moyenne \(m\) et de variance \(v\), alors son équivalent certain vaut \(m - \gamma v/2\).

#### Preuve

1. La transformée de Laplace de la loi normale donne \(\mathbb{E}[e^{-\gamma W}] = e^{-\gamma m + \gamma^2 v/2}\) : on complète le carré dans l'intégrale de la densité (vérifié symboliquement dans `verif/seance_mm_2.py`).
2. L'équivalent certain \(c\) vérifie \(-e^{-\gamma c} = -e^{-\gamma m + \gamma^2 v/2}\).
3. Le logarithme donne \(c = m - \gamma v/2\).

#### Exemple

Une position de 5 unités à 100 euros, \(\gamma = 0{,}1\), une variance du prix de \(@EX_sig2tau:4m@\) euros carrés sur l'horizon : \(m = 500\), \(v = 25 \times @EX_sig2tau:4m@\) et l'équivalent certain vaut \(@CE_ex:4m@\). Une simulation de cinq millions de tirages donne \(@CE_mc:4m@\), erreur standard \(@CE_se:4m@\).

### 3. Le prix d'indifférence

#### Hypothèses

Le prix suit une marche sans dérive, à volatilité constante : \(S_T\) suit une loi normale de moyenne \(s\) et de variance \(\sigma^2\tau\), où \(\sigma\) est en euros par racine d'année et \(\tau\) le temps restant en années. Le teneur ne traite plus d'ici l'horizon : c'est l'approximation du calcul, qui néglige les exécutions futures.

#### Énoncé

Pour un inventaire \(q\), le prix maximal d'achat d'une unité et le prix minimal de vente valent

\(\displaystyle r_a = s - \tfrac{1}{2}(2q+1)\gamma\sigma^2\tau, \qquad r_v = s - \tfrac{1}{2}(2q-1)\gamma\sigma^2\tau.\)

Leur moyenne est le **prix de réservation** \(r = s - q\gamma\sigma^2\tau\), et leur différence vaut \(r_v - r_a = \gamma\sigma^2\tau\).

#### Preuve

1. La position vaut \(q S_T\), normale de moyenne \(qs\) et de variance \(q^2\sigma^2\tau\). Par la partie 2, sa valeur certaine est \(c(q) = qs - \tfrac{\gamma}{2} q^2 \sigma^2\tau\).
2. Acheter une unité à \(r_a\) laisse indifférent si \(c(q+1) - r_a = c(q)\). L'utilité CARA permet cette soustraction, car l'argent certain s'ajoute à l'équivalent certain.
3. Donc \(r_a = c(q+1) - c(q) = s - \tfrac{\gamma}{2}\sigma^2\tau\,((q+1)^2 - q^2) = s - \tfrac{\gamma}{2}(2q+1)\sigma^2\tau\).
4. De même, \(r_v = c(q) - c(q-1) = s - \tfrac{\gamma}{2}(2q-1)\sigma^2\tau\).
5. La moyenne donne \(r\), et la différence donne \(\gamma\sigma^2\tau\).

#### Cas limites

- \(q = 0\) : \(r = s\) et aucun décalage. \(\tau = 0\) : plus de risque, plus de décalage.
- \(\gamma\) tendant vers 0 : neutralité au risque, \(r = s\) quel que soit le stock.

### 4. Le terme de liquidité

#### Le problème

Le prix d'indifférence donne le centre, pas la distance à laquelle coter. Coter loin rapporte plus par exécution et moins d'exécutions. Il faut un modèle de ce compromis.

#### Cadre

Les exécutions à la distance \(\delta\) (euros) d'un prix d'indifférence arrivent selon un processus de Poisson d'intensité \(A e^{-k\delta}\), où \(k\) (inverse d'euros) mesure la sensibilité des contreparties au prix. **Un marché concurrentiel a un \(k\) grand** : la probabilité d'exécution chute vite avec la distance. Un marché captif a un \(k\) petit. Le moteur des exercices a la même forme : `fill_decay` vaut 140 par unité de prix relatif, soit \(k = @K_moteur:1m@\) par euro à un prix de 100 euros.

#### Énoncé

Le demi-écart qui maximise le gain d'utilité espéré par unité de temps est \(\delta^* = \tfrac{1}{\gamma}\ln(1 + \gamma/k)\).

#### Preuve

1. Une exécution à la distance \(\delta\) d'un prix d'indifférence donne au teneur un surplus certain de \(\delta\) euros par rapport à sa situation de référence.
2. L'utilité vaut \(-e^{-\gamma w}\) : ajouter \(\delta\) multiplie l'utilité (négative) par \(e^{-\gamma\delta}\). Le gain d'utilité vaut donc \(U(1 - e^{-\gamma\delta})\), où \(U > 0\) est la valeur absolue de l'utilité courante, indépendante de \(\delta\).
3. Le gain d'utilité espéré par unité de temps est proportionnel à \(g(\delta) = e^{-k\delta}(1 - e^{-\gamma\delta})\).
4. \(g'(\delta) = e^{-k\delta}\bigl(-k + (k + \gamma)e^{-\gamma\delta}\bigr)\), qui s'annule si et seulement si \(e^{\gamma\delta} = 1 + \gamma/k\).
5. \(g'\) passe du positif au négatif en ce point : c'est le maximum global sur \(\delta > 0\), et \(\delta^*\) est la valeur annoncée.

**Statut.** C'est le calcul d'Avellaneda et Stoikov (2008) pour le premier ordre : il utilise la valeur « sans nouvelle exécution » de la partie 3 et ne prétend pas résoudre exactement le problème complet, qui s'exprime par une équation de Hamilton-Jacobi-Bellman. Les formules finales sont donc des approximations.

#### Les formules d'Avellaneda et Stoikov

Le bid est à \(\delta^*\) sous \(r_a\), l'ask à \(\delta^*\) au-dessus de \(r_v\). L'écart total vaut

\(\displaystyle e = (r_v + \delta^*) - (r_a - \delta^*) = \gamma\sigma^2\tau + \tfrac{2}{\gamma}\ln\bigl(1 + \tfrac{\gamma}{k}\bigr), \qquad \text{bid} = r - e/2, \quad \text{ask} = r + e/2.\)

Le second terme ne dépend pas du niveau de prix : \(k\) et \(\delta\) sont des grandeurs absolues, en euros. Il décroît avec \(k\) : plus le marché est concurrentiel, plus il faut coter serré.

#### Exemple entièrement calculé

\(s = 100\), \(q = 5\), \(\gamma = 0{,}1\), \(k = 1{,}5\), volatilité annualisée de 30 %, soit \(\sigma = 30\) euros, et une journée restante, \(\tau = 1/252\).

1. \(\sigma^2\tau = @EX_sig2tau:4m@\) et \(\gamma\sigma^2\tau = @EX_risque:4m@\).
2. Décalage \(q\gamma\sigma^2\tau = @EX_decal:4m@\), donc \(r = @EX_r:4m@\) (avec \(r_a = @EX_ra:4m@\) et \(r_v = @EX_rv:4m@\)).
3. Terme de liquidité \(20 \ln(1 + 0{,}1/1{,}5) = @EX_liq:4m@\), à comparer à sa limite \(2/k = @EX_lim:4m@\) quand \(\gamma\) tend vers 0.
4. Écart \(e = @EX_ecart:4m@\), bid \(@EX_bid:2m@\), ask \(@EX_ask:2m@\).

Le maximum de \(g\) trouvé sur une grille vaut \(@D_num:4m@\), égal à \(\delta^* = @D_ex:4m@\).

[[controle:signe]]

### 5. Le piège des unités

#### Pièges

`state.realized_vol` est **relative et annualisée**. `state.time_remaining` est une **fraction de journée**. La formule attend une volatilité en euros et un temps en années. Deux conversions sont donc nécessaires :

- `sigma = realized_vol * ref_price` ;
- `tau = time_remaining / 252`, avec 252 jours de bourse par an.

Sur l'exemple, oublier la seconde donne un décalage de \(@ER_sans252:1m@\) euros au lieu de \(@EX_decal:4m@\). Oublier la première donne \(@ER_sans_euros:7m@\) euro. Aucune des deux erreurs ne provoque de message d'erreur : le code tourne et les nombres sont faux.

La formule du terme de liquidité vaut pour \(k\) en inverse d'euros. Dans l'exercice, \(k\) reste fixe alors que le moteur a une décroissance relative : l'équivalence \(k = d/\text{prix}\) n'est exacte qu'au prix de 100 euros.

#### Ce que la simulation montre

Sur @B_n@ journées simulées de 5000 pas, avec un inventaire plafonné à 60, le rapport des écarts-types d'inventaire entre ce teneur et un écart fixe de 1,5 % de chaque côté vaut \(@B_rapport:3m@\), erreur standard \(@B_rapport_se:3m@\). L'inventaire est mieux tenu ; ce résultat ne dit rien du gain, et le moteur ignore la sélection adverse, comme expliqué dans la séance précédente.

[[controle:unites]]

### 6. Réflexes et acquis

#### Réflexes en 30 secondes

- Stock positif : les deux prix baissent. Position courte : ils montent.
- Décalage : \(q\gamma\sigma^2\tau\), en euros.
- Écart : risque \(\gamma\sigma^2\tau\) plus liquidité \((2/\gamma)\ln(1+\gamma/k)\).
- Convertir \(\sigma\) en euros et \(\tau\) en années avant tout calcul.
- \(k\) grand : marché concurrentiel, cotation serrée.

#### À savoir refaire

- L'équivalent certain d'une richesse normale.
- Les prix \(r_a\) et \(r_v\), leur moyenne et leur différence.
- La maximisation de \(e^{-k\delta}(1 - e^{-\gamma\delta})\).
- L'exemple numérique complet, sans notes.

[[controle:k]]
"""

CONTROLES = {
    "signe": {
        "question": r"Un teneur est en position courte de 10 unités. Par rapport au prix \(s\), son prix de réservation est :",
        "options": [
            {"id": "a", "texte": "supérieur à s", "explication": r"Oui : \(q = -10\) donne \(r = s + 10\gamma\sigma^2\tau\). Il veut racheter, donc il monte ses prix."},
            {"id": "b", "texte": "inférieur à s", "explication": r"C'est le cas d'un stock positif. Ici \(q < 0\) inverse le signe de la pénalité : il faut encourager les ventes des clients."},
            {"id": "c", "texte": "égal à s", "explication": r"L'égalité demande \(q = 0\), ou \(\tau = 0\), ou une neutralité au risque. Aucune de ces conditions n'est donnée."},
            {"id": "d", "texte": "indéterminé sans k", "explication": r"Le paramètre \(k\) n'intervient que dans la largeur de l'écart : le centre \(r\) ne dépend pas de lui."},
        ],
    },
    "unites": {
        "question": r"Dans l'exercice, `realized_vol=0.30`, `ref_price=200`, `time_remaining=0.5`. Que vaut \(\sigma^2\tau\) en euros carrés ?",
        "options": [
            {"id": "a", "texte": "1800/252", "explication": r"Oui : \(\sigma = 60\) euros, \(\sigma^2 = 3600\), \(\tau = 0{,}5/252\), soit \(1800/252\)."},
            {"id": "b", "texte": "1800", "explication": r"La fraction de journée a été utilisée comme une fraction d'année : le facteur 252 manque, et la pénalité est 252 fois trop grande."},
            {"id": "c", "texte": "0,045/252", "explication": r"La volatilité est restée relative : il manque le facteur \(200^2\) qui la convertit en euros carrés."},
            {"id": "d", "texte": "3600/252", "explication": r"Ce calcul prend une journée entière : `time_remaining` vaut 0,5, donc il reste une demi-journée."},
        ],
    },
    "k": {
        "question": r"À \(\gamma\) fixé, que devient le terme de liquidité \((2/\gamma)\ln(1 + \gamma/k)\) quand \(k\) augmente ?",
        "options": [
            {"id": "a", "texte": "Il diminue", "explication": r"Oui : \(\gamma/k\) diminue, donc le logarithme aussi. Un marché concurrentiel impose une cotation plus serrée."},
            {"id": "b", "texte": "Il augmente", "explication": r"C'est l'erreur d'une lecture qui prend un grand \(k\) pour un marché étroit. Le logarithme décroît pourtant avec \(k\)."},
            {"id": "c", "texte": "Il ne change pas", "explication": r"Ce terme ne dépend ni de \(\sigma\) ni de \(q\), mais il dépend bien de \(k\), puisqu'il vient de l'intensité des exécutions."},
            {"id": "d", "texte": "Il dépend du prix", "explication": r"Ni \(k\) ni \(\gamma\) ne font intervenir le niveau de prix : ce terme est une distance en euros."},
        ],
    },
}

LECONS = [{"id": "L-MM-2", "type": "lecon", "section": "lecon", "theme": "Leçon : équivalent certain, prix d'indifférence, Avellaneda-Stoikov",
           "difficulte": 4, "temps_cible_s": 2100, "contenu": CONTENU, "controles": CONTROLES, "verif": {"script": "verif/seance_mm_2.py"}}]

E1 = r"""## Exercice 1 : les prix d'indifférence

Écris `prix_indifference(s, q, gamma, sigma, tau)`, qui rend le triplet `(r_achat, r_vente, r)` défini dans la partie 3 de la leçon : \(r_a\), \(r_v\) et leur moyenne. `s` est le prix en euros, `q` l'inventaire (signé), `gamma` l'aversion au risque, `sigma` la volatilité en euros par racine d'année, `tau` le temps restant en années.

**Exemple.** `prix_indifference(100.0, 2, 0.5, 2.0, 0.25)` rend `(98.75, 99.25, 99.0)`.
"""

E2 = r"""## Exercice 2 : le teneur d'Avellaneda-Stoikov

Complète la classe `AvellanedaStoikovMarketMaker`, sous-classe de `MarketMakerBase`, de constructeur `(gamma=0.1, k=1.5, max_skew_bps=400.0)`. Voir la séance précédente pour `MarketState` et `Quote` ; `ref_price` est au moins égal à 1.

**Méthode `quote(state)`.** Si `ref_price` n'est pas strictement positif, lever `ValueError`. Sinon, dans l'ordre :

1. `tau = time_remaining / 252` et `sigma = realized_vol * ref_price`.
2. `risque = gamma * sigma**2 * tau`.
3. Le décalage vaut `inventory * risque`, ramené dans l'intervalle `[-m, m]` où `m = ref_price * max_skew_bps / 10000` : ce plafond garde le prix valide pour un inventaire extrême.
4. `r = ref_price - decalage`.
5. `ecart = max(risque + (2 / gamma) * ln(1 + gamma / k), 0.001 * ref_price)`.
6. `bid = round(r - ecart / 2, 2)` et `ask = round(r + ecart / 2, 2)`.
7. Garde-fous, dans cet ordre : si `bid <= 0`, remplacer `bid` par `round(r * 0.5, 2)` ; puis, si `bid >= ask`, remplacer `ask` par `round(bid + max(ecart, 0.01), 2)`.

**Exemple.** `ref_price=100.0`, `realized_vol=0.30`, `time_remaining=1.0`, `inventory=5.0` donne `Quote(bid=97.39, ask=99.04)`.

**Ce que vérifient les tests.** La formule exacte, le plafond du décalage, l'absence de décalage à stock nul ou à la clôture, les signes, un décalage croissant avec le stock, la validité à inventaire extrême, et une simulation où l'écart-type de l'inventaire est inférieur à 0,85 fois celui d'un écart fixe.
"""

CODES = [
    cm.code("MM2-E1", "code", "Prix d'indifférence d'un teneur de marché", 2, 480, E1, "mm2_e1",
            [cm.IMPORTS, "def prix_indifference(s, q, gamma, sigma, tau):\n    pass  # À compléter", "prix_indifference(100.0, 2, 0.5, 2.0, 0.25)"],
            "mm_2.py", "Correction commentée des deux exercices"),
    cm.code("MM2-E2", "code", "Cotation d'Avellaneda-Stoikov et conversions d'unités", 4, 1920, E2, "mm2_e2",
            [cm.IMPORTS + "\nfrom amplify_quant_trading import MarketMakerBase, MarketState, Quote",
             "class AvellanedaStoikovMarketMaker(MarketMakerBase):\n    def __init__(self, gamma=0.1, k=1.5, max_skew_bps=400.0):\n        pass  # À compléter\n\n    def quote(self, state):\n        pass  # À compléter",
             "AvellanedaStoikovMarketMaker().quote(MarketState(timestamp=0, ref_price=100.0, realized_vol=0.30, time_remaining=1.0, inventory=5.0))"],
            "mm_2.py", "Correction commentée des deux exercices"),
]
QUESTIONS = []
