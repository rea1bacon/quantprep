"""Leçon express de la séance de mathématiques 2 : conditionnement et lois usuelles.

Format : voir outils/lecons.py (élément de type "lecon", contenu en Markdown réduit, points de contrôle
[[controle:IDENT]] sur une ligne seule, option "a" correcte, une explication par option).
Aucun exemple exécuté : la page affiche la leçon seule.
Aucun nombre calculé n'est écrit à la main : chaque jeton @CLE@, @CLE:3@ ou @CLE:3m@ est remplacé par la valeur
calculée par verif/seance_math_2.py (fichier verif/seance_math_2.out.json).
Ne reprend pas la séance 1 (outils/lecons_probas.py : indicatrices, premier pas, statistiques d'ordre, Boole) : y renvoie.
"""

LECON_MATH_2 = {
    "id": "L-MATH-2", "type": "lecon", "section": "lecon",
    "theme": "Leçon express : conditionnement, lois usuelles, Poisson, coupons, inclusion-exclusion",
    "difficulte": 3, "temps_cible_s": 1200,
    "contenu": r"""## Leçon express : conditionnement et lois usuelles

Durée visée : 20 minutes. Six résultats qui servent dans les questions courtes : conditionner, citer les moments des lois usuelles, exploiter l'absence de mémoire, le processus de Poisson, le collectionneur de coupons, l'inclusion-exclusion. Pour chacun : l'énoncé, une preuve courte mais complète, un exemple calculé, les pièges.

Les outils de la séance 1 (leçon L-PROBAS-1 : linéarité et indicatrices, premier pas, statistiques d'ordre, borne de Boole) sont supposés connus et seulement cités (« séance 1, outil 2 »). Tous les nombres viennent du script `verif/seance_math_2.py`, qui les calcule exactement et les vérifie par simulation.

**Conventions de notation** :

- une majuscule désigne une variable aléatoire, une minuscule une valeur ou un paramètre fixe ;
- les événements sont aussi notés par des majuscules (\(A\), \(B\), \(A_j\)) ; \(A^c\) est le complémentaire de \(A\) et \(I_A\) son indicatrice (1 si \(A\) se produit, 0 sinon) ;
- exception d'usage déclarée : \(H_n = 1 + \tfrac12 + \dots + \tfrac1n\), le \(n\)-ième nombre harmonique, est un nombre fixe malgré sa majuscule ;
- \(\mathbb{P}\) désigne une probabilité, \(\mathbb{E}\) une espérance, \(\mathrm{Var}\) une variance ; \(\binom{n}{k}\) est le nombre de parties à \(k\) éléments d'un ensemble à \(n\) éléments ; \(\lvert \mathcal{J} \rvert\) est le nombre d'éléments d'un ensemble fini \(\mathcal{J}\).

## 1. Conditionnement

Le problème : une quantité dépend d'une cause inconnue (le signal a-t-il un effet, combien de transactions, que donne le premier lancer). On calcule cause par cause, là où c'est simple, puis on recombine.

### Probabilités totales et Bayes

Cadre : \(B_1, \dots, B_m\) est une **partition** : événements deux à deux disjoints, de réunion certaine, chacun de probabilité strictement positive. Pour \(\mathbb{P}(B) > 0\), \(\mathbb{P}(A \mid B) = \mathbb{P}(A \cap B)/\mathbb{P}(B)\).

- **(C1) Probabilités totales** : \(\mathbb{P}(A) = \sum_{j=1}^{m} \mathbb{P}(A \mid B_j)\,\mathbb{P}(B_j)\).
- **(C2) Bayes** : si \(\mathbb{P}(A) > 0\), \(\mathbb{P}(B_i \mid A) = \dfrac{\mathbb{P}(A \mid B_i)\,\mathbb{P}(B_i)}{\sum_{j=1}^{m} \mathbb{P}(A \mid B_j)\,\mathbb{P}(B_j)}\).

Preuve :

1. Chaque issue de \(A\) est dans exactement un \(B_j\) : \(A\) est la réunion disjointe des \(A \cap B_j\), donc \(\mathbb{P}(A) = \sum_j \mathbb{P}(A \cap B_j)\), et \(\mathbb{P}(A \cap B_j) = \mathbb{P}(A \mid B_j)\,\mathbb{P}(B_j)\) par définition. Même preuve pour une partition dénombrable.
2. \(\mathbb{P}(B_i \mid A) = \mathbb{P}(A \cap B_i)/\mathbb{P}(A)\), numérateur comme à l'étape 1, dénominateur par (C1).

Exemple : une stratégie candidate est rentable avec probabilité 0,1 a priori ; son backtest est « concluant » avec probabilité 0,9 si elle est rentable, 0,2 sinon. Rentable et concluant : \(0{,}1 \times 0{,}9 = @La_joint:2m@\) ; non rentable et concluant : \(0{,}9 \times 0{,}2 = @La_faux:2m@\) ; donc \(\mathbb{P}(\text{concluant}) = @La_total:2m@\) et \(\mathbb{P}(\text{rentable} \mid \text{concluant}) = @La_joint:2m@/@La_total:2m@ = \tfrac{@La_post_num@}{@La_post_den@}\) (simulation du script : @La_post_mc:3@). Les faux positifs sont rares par stratégie non rentable, mais ces stratégies sont bien plus nombreuses.

### Espérance et variance conditionnelles

Cadre : \(Y\) est discrète (valeurs \(y\) avec \(\mathbb{P}(Y = y) > 0\)), \(X\) est réelle d'espérance finie. Le nombre \(g(y) = \mathbb{E}[X\,I_{\{Y = y\}}]/\mathbb{P}(Y = y)\) est la moyenne de \(X\) sur les issues où \(Y = y\). L'**espérance conditionnelle** \(\mathbb{E}[X \mid Y]\) est la variable aléatoire \(g(Y)\). Si \(\mathbb{E}[X^2] < \infty\), \(v(y) = \mathbb{E}[X^2 \mid Y = y] - g(y)^2\) est la variance de \(X\) sachant \(Y = y\), et \(\mathrm{Var}(X \mid Y) = v(Y)\).

- **(C3) Espérance totale** : \(\mathbb{E}\big[\mathbb{E}[X \mid Y]\big] = \mathbb{E}[X]\), soit \(\mathbb{E}[X] = \sum_y \mathbb{E}[X \mid Y = y]\,\mathbb{P}(Y = y)\).
- **(C4) Variance totale** : si \(\mathbb{E}[X^2] < \infty\), \(\mathrm{Var}(X) = \mathbb{E}\big[\mathrm{Var}(X \mid Y)\big] + \mathrm{Var}\big(\mathbb{E}[X \mid Y]\big)\) : dispersion moyenne à l'intérieur des groupes \(\{Y = y\}\), plus dispersion des moyennes de groupe.

Preuve :

1. (C3) : \(\sum_y g(y)\,\mathbb{P}(Y = y) = \sum_y \mathbb{E}[X\,I_{\{Y = y\}}] = \mathbb{E}\big[X \sum_y I_{\{Y = y\}}\big] = \mathbb{E}[X]\), car exactement un événement \(\{Y = y\}\) se produit. L'échange somme-espérance est la linéarité (séance 1, outil 1) ; pour une infinité de valeurs, il vient du théorème de convergence dominée (admis), les sommes partielles étant majorées par \(\lvert X \rvert\).
2. \(v(y) \ge 0\) (c'est une variance pour la probabilité sachant \(Y = y\)), donc \(g(y)^2 \le \mathbb{E}[X^2 \mid Y = y]\) et tout est fini. L'étape 1 appliquée à \(X^2\) donne \(\mathbb{E}[v(Y)] = \mathbb{E}[X^2] - \mathbb{E}[g(Y)^2]\).
3. L'étape 1 donne \(\mathbb{E}[g(Y)] = \mathbb{E}[X]\), donc \(\mathrm{Var}(g(Y)) = \mathbb{E}[g(Y)^2] - \mathbb{E}[X]^2\).
4. En additionnant, \(\mathbb{E}[g(Y)^2]\) s'annule : \(\mathbb{E}[v(Y)] + \mathrm{Var}(g(Y)) = \mathbb{E}[X^2] - \mathbb{E}[X]^2 = \mathrm{Var}(X)\).

Pour \(Y\) à densité, (C3) et (C4) restent vrais avec la définition générale de l'espérance conditionnelle (admise ici).

### Somme d'un nombre aléatoire de termes

Cadre : \(N\) est à valeurs entières positives, d'espérance et de variance finies ; \(X_1, X_2, \dots\) ont la même loi, d'espérance \(\mu\) et de variance \(\sigma^2\), et sont indépendantes entre elles **et de \(N\)** ; \(S = X_1 + \dots + X_N\), avec \(S = 0\) si \(N = 0\). Alors \(\mathbb{E}[S] = \mathbb{E}[N]\,\mu\) et \(\mathrm{Var}(S) = \mathbb{E}[N]\,\sigma^2 + \mathrm{Var}(N)\,\mu^2\).

Preuve : les \(X_i\) étant indépendantes de \(N\), sachant \(N = n\) elles gardent leur loi et leur indépendance, d'où \(\mathbb{E}[S \mid N = n] = n\mu\) et \(\mathrm{Var}(S \mid N = n) = n\sigma^2\). Donc \(\mathbb{E}[S \mid N] = N\mu\), \(\mathrm{Var}(S \mid N) = N\sigma^2\), et (C3), (C4) donnent \(\mathbb{E}[N]\,\mu\) et \(\mathbb{E}[N\sigma^2] + \mathrm{Var}(N\mu)\).

Exemple : nombre de clients d'une heure de loi de Poisson de paramètre 3 (espérance et variance 3, partie 2), montants d'espérance 10 euros et d'écart-type 5 euros : \(\mathbb{E}[S] = @Lb_es@\) euros, \(\mathrm{Var}(S) = 3 \times 5^2 + 3 \times 10^2 = @Lb_intra@ + @Lb_inter@ = @Lb_var@\) euros carrés (simulation : @Lb_es_mc:2@ et @Lb_var_mc:1@).

### Pièges

1. Confondre \(\mathbb{P}(A \mid B)\) et \(\mathbb{P}(B \mid A)\), ou oublier l'a priori : sans lui, Bayes revient à supposer toutes les causes également probables.
2. Oublier \(\mathrm{Var}(\mathbb{E}[X \mid Y])\) : la variance d'un mélange n'est pas la moyenne des variances.
3. \(\mathbb{E}[N]\,\mathbb{E}[X^2]\) n'est la variance de \(S\) que si \(\mathrm{Var}(N) = \mathbb{E}[N]\), par exemple pour \(N\) de Poisson.
4. Si l'on s'arrête selon les valeurs observées (au premier 6, au premier gain), les \(X_i\) ne sont plus indépendantes de \(N\) : calculer d'abord leur loi sachant \(N = n\).

[[controle:vartot]]

## 2. Lois usuelles et leurs moments

Paramètres : \(0 < p < 1\), \(n \ge 1\) entier, \(\lambda > 0\) (une intensité : nombre moyen d'événements par unité de temps), \(a < b\), \(\mu\) réel, \(\sigma > 0\).

| Loi | Situation type | Valeurs | \(\mathbb{P}(X = k)\) ou densité en \(x\) | Espérance | Variance |
|---|---|---|---|---|---|
| Bernoulli \(\mathcal{B}(p)\) | un essai réussi avec probabilité \(p\) | \(\{0, 1\}\) | \(\mathbb{P}(X = 1) = p\) | \(p\) | \(p(1-p)\) |
| binomiale \(\mathcal{B}(n, p)\) | succès en \(n\) essais indépendants | \(\{0, \dots, n\}\) | \(\binom{n}{k} p^k (1-p)^{n-k}\) | \(np\) | \(np(1-p)\) |
| géométrique \(\mathcal{G}(p)\) | rang du premier succès | \(\{1, 2, \dots\}\) | \((1-p)^{k-1} p\) | \(1/p\) | \((1-p)/p^2\) |
| Poisson \(\mathcal{P}(\lambda)\) | événements d'un processus de Poisson (partie 4) | \(\{0, 1, \dots\}\) | \(e^{-\lambda} \lambda^k / k!\) | \(\lambda\) | \(\lambda\) |
| uniforme \(\mathcal{U}([a, b])\) | point au hasard dans \([a, b]\) | \([a, b]\) | \(1/(b - a)\) | \((a+b)/2\) | \((b-a)^2/12\) |
| exponentielle \(\mathcal{E}(\lambda)\) | attente sans mémoire (partie 3) | \([0, +\infty[\) | \(\lambda e^{-\lambda x}\) | \(1/\lambda\) | \(1/\lambda^2\) |
| normale \(\mathcal{N}(\mu, \sigma^2)\) | limite du théorème central limite | \(\mathbb{R}\) | \(e^{-(x-\mu)^2/(2\sigma^2)}/(\sigma\sqrt{2\pi})\) | \(\mu\) | \(\sigma^2\) |

Preuves des moments les plus demandés :

1. Binomiale : \(X = I_1 + \dots + I_n\), indicatrices indépendantes des succès, de variance \(p - p^2\) car \(I_j^2 = I_j\). Linéarité, puis additivité des variances par indépendance.
2. Géométrique, espérance : \(\mathbb{P}(G > k) = (1-p)^k\) (les \(k\) premiers essais échouent), et la formule de la queue pour une variable entière positive (séance 1, outil 2, étape 2 de la preuve) donne \(\mathbb{E}[G] = \sum_{k \ge 0} (1-p)^k = 1/p\).
3. Géométrique, variance, par le premier essai : avec probabilité \(p\), \(G = 1\) ; sinon tout recommence et \(G = 1 + G'\), où \(G'\), fonction des essais suivants, a la loi de \(G\). Donc \(\mathbb{E}[G^2] = p + (1-p)\big(1 + 2/p + \mathbb{E}[G^2]\big)\) ; \(\mathbb{E}[G^2]\) étant fini (la série \(\sum k^2 (1-p)^{k-1}\) converge), \(p\,\mathbb{E}[G^2] = 1 + 2(1-p)/p\), soit \(\mathbb{E}[G^2] = (2-p)/p^2\) et \(\mathrm{Var}(G) = (1-p)/p^2\).
4. Poisson : \(\mathbb{E}[X] = \lambda \sum_{k \ge 1} e^{-\lambda}\lambda^{k-1}/(k-1)! = \lambda\) (somme des probabilités de la loi) ; de même \(\mathbb{E}[X(X-1)] = \lambda^2\), donc \(\mathrm{Var}(X) = \lambda^2 + \lambda - \lambda^2 = \lambda\).
5. Exponentielle : \(\mathbb{P}(X > x) = e^{-\lambda x}\) pour \(x \ge 0\). Formule de la queue (séance 1, outil 3, (F)) : \(\mathbb{E}[X] = \int_0^\infty e^{-\lambda x}\,dx = 1/\lambda\). Comme \(X^2 = \int_0^\infty 2x\,I_{\{x < X\}}\,dx\), Fubini-Tonelli (admis) donne \(\mathbb{E}[X^2] = \int_0^\infty 2x\,e^{-\lambda x}\,dx = 2/\lambda^2\) (intégration par parties), d'où \(\mathrm{Var}(X) = 1/\lambda^2\).
6. Uniforme et normale : pour \(U\) uniforme sur \([0, 1]\), \(\mathbb{E}[U^2] - \mathbb{E}[U]^2 = 1/3 - 1/4\), puis \(a + (b-a)\,U\) multiplie la variance par \((b-a)^2\). Pour \(Z\) de loi \(\mathcal{N}(0, 1)\), de densité \(\varphi\), \(\mathbb{E}[Z] = 0\) par imparité et, comme \(\varphi'(z) = -z\,\varphi(z)\), une intégration par parties donne \(\mathbb{E}[Z^2] = \big[-z\varphi(z)\big]_{-\infty}^{+\infty} + \int \varphi = 1\) ; enfin \(\mu + \sigma Z\) suit \(\mathcal{N}(\mu, \sigma^2)\).

Exemple : on lance un dé jusqu'au premier 6 ; \(G\) suit \(\mathcal{G}(1/6)\) : \(\mathbb{E}[G] = @Lc_geo_e@\), \(\mathbb{E}[G^2] = @Lc_geo_e2@\), \(\mathrm{Var}(G) = @Lc_geo_e2@ - @Lc_geo_e@^2 = @Lc_geo_var@\) (simulation : @Lc_geo_e_mc:2@ et @Lc_geo_var_mc:1@).

Pièges :

1. Géométrique : nombre d'essais (valeurs \(1, 2, \dots\), espérance \(1/p\)) ou nombre d'échecs avant le succès, \(G - 1\) (valeurs \(0, 1, \dots\), espérance \((1-p)/p\), même variance). Lire l'énoncé.
2. Exponentielle : le paramètre écrit est l'intensité ou l'espérance selon les sources ; regarder l'unité (« 2 par minute » est une intensité, « 2 minutes en moyenne » une espérance).
3. \(np(1-p)\) exige des essais indépendants : faux pour des tirages sans remise.

## 3. Absence de mémoire, minimum d'exponentielles

Le problème : « on a déjà attendu \(s\), combien reste-t-il ? », « parmi plusieurs attentes indépendantes, laquelle finit la première, et quand ? ».

- **(A1)** Si \(G\) suit \(\mathcal{G}(p)\), pour des entiers \(j, k \ge 0\) : \(\mathbb{P}(G > j + k \mid G > j) = \mathbb{P}(G > k)\).
- **(A2)** Si \(X\) suit \(\mathcal{E}(\lambda)\), pour des réels \(s, t \ge 0\) : \(\mathbb{P}(X > s + t \mid X > s) = \mathbb{P}(X > t)\). Sachant \(X > s\), le temps restant \(X - s\) suit encore \(\mathcal{E}(\lambda)\), et \(\mathbb{E}[X \mid X > s] = s + 1/\lambda\).

Preuve : \(\{G > j + k\}\) est inclus dans \(\{G > j\}\), donc la probabilité conditionnelle vaut \((1-p)^{j+k}/(1-p)^j = (1-p)^k\). Même calcul avec \(e^{-\lambda(s+t)}/e^{-\lambda s} = e^{-\lambda t}\) ; la loi conditionnelle de \(X - s\) a donc la queue de \(\mathcal{E}(\lambda)\), d'espérance \(1/\lambda\).

Exemple : \(X\) exponentielle d'espérance 2 minutes. Après 3 minutes sans événement, \(\mathbb{P}(\text{encore plus de 2 minutes}) = e^{-1} \approx @Ld_cond:4m@\), et l'attente totale moyenne est \(3 + 2 = @Ld_esp_cond@\) minutes (simulation : @Ld_cond_mc:4@ et @Ld_esp_cond_mc:2@).

[[controle:memoire]]

### Minimum de variables exponentielles indépendantes

Cadre : \(X_1, \dots, X_n\) indépendantes, \(X_i\) de loi \(\mathcal{E}(\lambda_i)\) ; \(\lambda = \lambda_1 + \dots + \lambda_n\) ; \(M = \min_i X_i\) ; \(K\) est l'indice de la plus petite (deux \(X_i\) sont égales avec probabilité nulle, car indépendantes et à densité).

- **(A3)** \(M\) suit \(\mathcal{E}(\lambda)\), donc \(\mathbb{E}[M] = 1/\lambda\).
- **(A4)** \(\mathbb{P}(K = i) = \lambda_i/\lambda\), et \(K\) est indépendante de \(M\).
- **(A5) La course recommence** : sachant \(K = i\) et la valeur de \(M\), les temps restants \(X_j - M\), \(j \ne i\), sont indépendants, de lois \(\mathcal{E}(\lambda_j)\).

Preuve :

1. (A3) : \(M > t\) si et seulement si toutes les \(X_i\) dépassent \(t\) ; par indépendance, \(\mathbb{P}(M > t) = \prod_i e^{-\lambda_i t} = e^{-\lambda t}\).
2. On fixe \(i\), \(t \ge 0\) et des \(u_j \ge 0\) pour \(j \ne i\). L'événement « \(K = i\), \(M > t\), et \(X_j - M > u_j\) pour tout \(j \ne i\) » s'écrit « \(X_i > t\) et \(X_j > X_i + u_j\) pour tout \(j \ne i\) ». En conditionnant par \(X_i = x\), de densité \(\lambda_i e^{-\lambda_i x}\), les autres variables étant indépendantes de \(X_i\) (Fubini, admis), sa probabilité vaut \(\int_t^\infty \lambda_i e^{-\lambda_i x} \prod_{j \ne i} e^{-\lambda_j (x + u_j)}\,dx = \dfrac{\lambda_i}{\lambda}\,e^{-\lambda t} \prod_{j \ne i} e^{-\lambda_j u_j}\).
3. Avec tous les \(u_j = 0\) et \(t = 0\) : \(\mathbb{P}(K = i) = \lambda_i/\lambda\). La probabilité est un produit de facteurs séparés : \(K\), \(M\) et les temps restants sont indépendants, chaque \(X_j - M\) ayant la queue \(e^{-\lambda_j u_j}\) de \(\mathcal{E}(\lambda_j)\). D'où (A4) et (A5).

Exemple : intensités 1, 2 et 3 par heure. \(\lambda = @Le_lambda@\) ; le premier événement arrive en moyenne au bout de \(\tfrac{@Le_emin_num@}{@Le_emin_den@}\) heure ; c'est celui d'intensité 1 avec probabilité \(\tfrac{@Le_p1_num@}{@Le_p1_den@}\), celui d'intensité 3 avec probabilité \(\tfrac{@Le_p3_num@}{@Le_p3_den@}\). Indépendance : \(\mathbb{P}(K = 1, M > 0{,}2) = \tfrac{1}{6}\,e^{-1{,}2} \approx @Le_joint:4m@\) (simulation : @Le_joint_mc:4@).

Pièges :

1. La chance d'arriver le premier est proportionnelle à l'**intensité** \(\lambda_i\), pas à l'espérance \(1/\lambda_i\).
2. L'absence de mémoire n'est pas « l'événement est dû » : après une longue attente, le temps restant ne raccourcit pas.
3. Le maximum n'est pas exponentiel (séance 1, outil 3, cas limites) : pour une question de « dernier », enchaîner les courses avec (A5).
4. « \(X_1 < X_2\) » et « \(X_1 < X_3\) » ne sont pas indépendants (tous deux favorisés quand \(X_1\) est petite) : ne pas multiplier leurs probabilités.

[[controle:course]]

## 4. Processus de Poisson

Le problème : des arrivées (ordres, clients, pannes) au hasard, à un rythme moyen constant. Combien dans une fenêtre, quand arrive la \(k\)-ième, que donne la fusion de deux flux ?

Définition : un processus de Poisson d'intensité \(\lambda > 0\) est une famille de variables \(N(t)\), \(t \ge 0\), où \(N(t)\) compte les événements de \([0, t]\), telle que :

1. \(N(0) = 0\), et \(t \mapsto N(t)\) est croissante, à valeurs entières, par sauts de 1 ;
2. pour \(0 \le s < t\), le nombre d'événements dans \(]s, t]\), \(N(t) - N(s)\), suit \(\mathcal{P}(\lambda\,(t - s))\) ;
3. les nombres d'événements dans des intervalles disjoints sont indépendants.

On note \(S_k\) l'instant du \(k\)-ième événement.

- **(P1) Attente** : \(S_1\) suit \(\mathcal{E}(\lambda)\) ; depuis tout instant fixé \(s\), l'attente du prochain événement suit aussi \(\mathcal{E}(\lambda)\).
- **(P2) \(k\)-ième événement** : \(\mathbb{P}(S_k > t) = \sum_{j=0}^{k-1} e^{-\lambda t}(\lambda t)^j/j!\) et \(\mathbb{E}[S_k] = k/\lambda\).
- **(P3) Superposition** : si \(N_1\), \(N_2\) sont des processus de Poisson indépendants d'intensités \(\lambda_1\), \(\lambda_2\), alors \(N_1 + N_2\) est un processus de Poisson d'intensité \(\lambda_1 + \lambda_2\), dont le premier événement vient de \(N_1\) avec probabilité \(\lambda_1/(\lambda_1 + \lambda_2)\).

Preuve :

1. (P1) : \(S_1 > t\) si et seulement si \(N(t) = 0\), de probabilité \(e^{-\lambda t}\) (propriété 2) ; depuis \(s\), aucun événement dans \(]s, s + u]\) avec probabilité \(e^{-\lambda u}\).
2. (P2) : par sauts de 1, \(S_k > t\) si et seulement si \(N(t) \le k - 1\). Formule de la queue (séance 1, outil 3, (F)) et changement de variable \(u = \lambda t\) : chaque terme donne \(\tfrac{1}{\lambda\,j!} \int_0^\infty u^j e^{-u}\,du = \tfrac{1}{\lambda}\), car cette intégrale \(I_j\) vérifie \(I_j = j\,I_{j-1}\) (intégration par parties) et \(I_0 = 1\). Les \(k\) termes donnent \(k/\lambda\).
3. (P3) : l'accroissement de \(N_1 + N_2\) sur \(]s, t]\) est la somme de deux variables indépendantes de lois \(\mathcal{P}(a)\) et \(\mathcal{P}(b)\), avec \(a = \lambda_1 (t-s)\), \(b = \lambda_2 (t-s)\) : \(\sum_{j=0}^{k} e^{-a}\tfrac{a^j}{j!}\,e^{-b}\tfrac{b^{k-j}}{(k-j)!} = \tfrac{e^{-(a+b)}}{k!} \sum_{j=0}^{k} \binom{k}{j} a^j b^{k-j} = e^{-(a+b)}\tfrac{(a+b)^k}{k!}\) (binôme). Sur des intervalles disjoints, ces accroissements sont des fonctions de familles disjointes de variables indépendantes, donc indépendants. Pas de saut simultané : en coupant \([0, t]\) en \(r\) morceaux, un saut commun exige un événement des deux processus dans un même morceau, de probabilité au plus \(r \times (\lambda_1 t/r)(\lambda_2 t/r) \to 0\), puisque \(1 - e^{-y} \le y\). Enfin les premiers instants des deux processus sont des exponentielles indépendantes (P1), et (A4) donne \(\lambda_1/(\lambda_1 + \lambda_2)\).

Réciproque admise (équivalence des deux définitions, résultat général) : les durées entre événements successifs sont indépendantes, de loi \(\mathcal{E}(\lambda)\). Le script simule le processus ainsi et retrouve les probabilités ci-dessous : c'est un constat numérique, pas une preuve.

Exemple : ordres d'intensité 2 par minute, fenêtre de 1,5 minute, donc \(N(1{,}5)\) suit \(\mathcal{P}(@Lf_mu@)\).

- \(\mathbb{P}(N(1{,}5) = 0) = e^{-3} \approx @Lf_p0:4m@\) et \(\mathbb{P}(N(1{,}5) = 2) = e^{-3}\,3^2/2! \approx @Lf_p2:4m@\) (simulation : @Lf_p0_mc:4@ et @Lf_p2_mc:4@) ; le 3e ordre arrive en moyenne à \(3/2 = @Lf_es3:1m@\) minute.
- Avec un second flux indépendant de 3 ordres par minute, le flux fusionné a l'intensité @Lf_sup@ : en une minute, \(\mathbb{P}(\text{exactement 5 ordres}) = e^{-5}\,5^5/5! \approx @Lf_sup_p5:4m@\), et le premier ordre vient du premier flux avec probabilité \(\tfrac{@Lf_premier_num@}{@Lf_premier_den@}\).

Pièges :

1. Le paramètre de la loi de Poisson est l'intensité **multipliée par la durée**, en unités cohérentes.
2. Paradoxe de l'autobus : depuis un instant fixé, l'attente moyenne est \(1/\lambda\) (P1), pas la moitié de l'écart moyen entre deux événements.
3. « Au moins \(k\) » vaut \(1 - \mathbb{P}(N \le k - 1)\).

## 5. Collectionneur de coupons

Le problème : on tire avec remise, uniformément, parmi \(n\) types. Combien de tirages en moyenne pour voir \(r\) types distincts, ou tous ?

Énoncé : soit \(C_r\) le nombre de tirages nécessaires pour voir \(r\) types distincts, \(1 \le r \le n\). Alors

\(\displaystyle \mathbb{E}[C_r] = \sum_{k=1}^{r} \frac{n}{n - k + 1} = \frac{n}{n} + \frac{n}{n-1} + \dots + \frac{n}{n - r + 1},\)

en particulier \(\mathbb{E}[C_n] = n\,H_n\), avec \(\ln(n + 1) \le H_n \le 1 + \ln n\).

Preuve :

1. \(C_r = G_1 + \dots + G_r\), où \(G_k\) compte les tirages après l'apparition du \((k-1)\)-ième type nouveau, jusqu'au \(k\)-ième inclus.
2. Fixons tout l'historique jusqu'à l'apparition du \((k-1)\)-ième type nouveau. Les tirages suivants sont indépendants de cet historique, et chacun retombe sur un type déjà vu avec probabilité \((k-1)/n\) : \(\mathbb{P}(G_k > j \mid \text{historique}) = ((k-1)/n)^j\). Donc \(G_k\) suit \(\mathcal{G}\big((n - k + 1)/n\big)\), d'espérance \(n/(n - k + 1)\) (partie 2).
3. Linéarité (séance 1, outil 1), sans besoin d'indépendance entre les \(G_k\) : la formule. Pour \(r = n\), la somme est \(n\,(1/n + \dots + 1/1) = n\,H_n\).
4. Encadrement : \(1/j \ge \int_j^{j+1} dx/x\) (car \(1/x \le 1/j\) sur \([j, j+1]\)) donne \(H_n \ge \ln(n+1)\) ; \(1/j \le \int_{j-1}^{j} dx/x\) pour \(j \ge 2\) donne \(H_n \le 1 + \ln n\).

Exemple : un dé. \(\mathbb{E}[C_6] = 6/6 + 6/5 + \dots + 6/1 = 6 \times \tfrac{@Lg_h6_num@}{@Lg_h6_den@} = @Lg_e:1m@\) lancers, dont \(6/1\) pour la seule dernière face ; \(\ln 7 \approx @Lg_ln7:3m@ \le H_6 = @Lg_h6:2m@ \le 1 + \ln 6 \approx @Lg_1ln6:3m@\) (simulation : @Lg_e_mc:2@).

Pièges :

1. \(n \ln n\) sous-estime toujours, car \(H_n > \ln n\) : \(6 \ln 6 \approx @Lg_6ln6:2m@\).
2. Pour \(r < n\) types, sommer les \(r\) **premiers** termes \(n/n, n/(n-1), \dots\), les plus petits.
3. La probabilité d'avoir tout vu en \(k\) tirages se calcule par inclusion-exclusion (partie 6), pas avec l'espérance.

[[controle:coupons]]

## 6. Dénombrement et inclusion-exclusion

Le problème : avec des issues équiprobables, une probabilité est un rapport de cardinaux. Il faut compter vite, et compter une réunion sans compter deux fois.

Tirer \(k\) objets parmi \(n\) objets distincts :

| Tirage | l'ordre compte | l'ordre ne compte pas |
|---|---|---|
| sans remise (\(k \le n\)) | \(n!/(n-k)!\) | \(\binom{n}{k} = \tfrac{n!}{k!\,(n-k)!}\) |
| avec remise | \(n^k\) | \(\binom{n+k-1}{k}\) |

1. Ordonné sans remise : \(n\), puis \(n - 1\), ..., puis \(n - k + 1\) choix (principe multiplicatif). Avec remise : \(n\) choix à chaque tirage.
2. Non ordonné sans remise : chaque partie à \(k\) éléments correspond à \(k!\) tirages ordonnés.
3. Non ordonné avec remise (étoiles et barres) : un tirage est déterminé par les effectifs \(k_1 + \dots + k_n = k\), qu'on écrit comme \(k\) étoiles séparées par \(n - 1\) barres (\(k_1\) étoiles, une barre, \(k_2\) étoiles...) ; c'est une bijection, et il suffit de placer les \(k\) étoiles parmi \(n + k - 1\) places. Exemple : 5 ordres identiques répartis entre 3 courtiers, \(\binom{7}{2} = @Lh_eb@\) façons.

Piège capital : les issues non ordonnées d'un tirage avec remise ne sont **pas** équiprobables. Avec deux dés, \(\{1, 2\}\) a pour probabilité \(\tfrac{@Lh_mixte_num@}{@Lh_mixte_den@}\) et \(\{1, 1\}\) a pour probabilité \(\tfrac{@Lh_double_num@}{@Lh_double_den@}\). Pour une probabilité, compter des suites ordonnées, jamais les @Lh_nb_non_ord@ issues non ordonnées.

### Inclusion-exclusion

Énoncé : pour des événements \(A_1, \dots, A_m\),

\(\displaystyle \mathbb{P}\Big(\bigcup_{j=1}^{m} A_j\Big) = \sum_{k=1}^{m} (-1)^{k+1} \sum_{\lvert \mathcal{J} \rvert = k} \mathbb{P}\Big(\bigcap_{j \in \mathcal{J}} A_j\Big),\)

la somme intérieure portant sur les parties \(\mathcal{J}\) de \(\{1, \dots, m\}\) à \(k\) éléments. Même formule pour des cardinaux (probabilité uniforme). Quand la probabilité d'une intersection ne dépend que de \(k\), la somme intérieure vaut \(\binom{m}{k}\) fois cette probabilité.

Preuve par les indicatrices :

1. Une issue n'est dans aucun \(A_j\) si et seulement si tous les facteurs \(1 - I_{A_j}\) valent 1 : \(1 - I_{A_1 \cup \dots \cup A_m} = \prod_{j=1}^{m} (1 - I_{A_j})\).
2. On développe en choisissant dans chaque facteur 1 ou \(-I_{A_j}\) : choisir \(-I_{A_j}\) exactement pour \(j \in \mathcal{J}\) donne \((-1)^{\lvert \mathcal{J} \rvert} \prod_{j \in \mathcal{J}} I_{A_j}\), et ce produit est l'indicatrice de l'intersection des \(A_j\), \(j \in \mathcal{J}\) ; la partie vide donne 1.
3. On retranche 1, on change les signes, puis on prend l'espérance : linéarité et \(\mathbb{E}[I_A] = \mathbb{P}(A)\) (séance 1, outil 1).

Exemple 1, tout voir : 5 tirages avec remise parmi 3 couleurs équiprobables. Avec \(A_c\) = « la couleur \(c\) n'apparaît pas » : il y a \(2^5\) suites sans une couleur donnée, \(1^5\) sans deux couleurs données, aucune sans les trois. Cas favorables : \(3^5 - 3 \times 2^5 + 3 \times 1^5 = @Lh_c_tot@ - @Lh_c_un@ + @Lh_c_deux@ = @Lh_c_fav@\), probabilité \(\tfrac{@Lh_c_p_num@}{@Lh_c_p_den@} \approx @Lh_c_p:3m@\) (simulation : @Lh_c_p_mc:3@). En général, \(\sum_{j=0}^{n} (-1)^j \binom{n}{j} (n - j)^k\) suites de longueur \(k\) utilisent les \(n\) symboles.

Exemple 2, dérangements : \(S\) est une permutation uniforme de \(\{1, \dots, n\}\), et \(A_i = \{S(i) = i\}\). Une intersection de \(k\) de ces événements a pour probabilité \((n-k)!/n!\), et la somme intérieure vaut \(\binom{n}{k}(n-k)!/n! = 1/k!\). Donc

\(\displaystyle \mathbb{P}(\text{aucun point fixe}) = 1 - \sum_{k=1}^{n} \frac{(-1)^{k+1}}{k!} = \sum_{k=0}^{n} \frac{(-1)^k}{k!},\)

et le nombre de dérangements (permutations sans point fixe) est \(d_n = n! \sum_{k=0}^{n} (-1)^k/k!\). Pour \(n = 4\) : \(d_4 = 4!\,(1 - 1 + 1/2 - 1/6 + 1/24) = @Lh_d4@\), probabilité \(\tfrac{@Lh_p4_num@}{@Lh_p4_den@}\). C'est la série de \(e^{-1}\) tronquée ; par le critère des séries alternées, l'écart est au plus \(1/(n+1)!\). Pour \(n = 10\) : \(@Lh_p10:7m@\) contre \(e^{-1} \approx @Lh_e_moins_un:7m@\). Le nombre de permutations à **exactement** \(j\) points fixes est \(\binom{n}{j}\,d_{n-j}\) : choisir les points fixes, puis déranger les autres.

Pièges :

1. Signe \((-1)^{k+1}\) : plus pour les événements seuls, moins pour les paires, plus pour les triplets.
2. Le premier terme seul ne donne que la borne de Boole (séance 1, outil 4).
3. « Exactement \(j\) » n'est pas « au moins \(j\) ».
4. Imposer d'abord ce qui est exigé, puis remplir le reste librement, compte plusieurs fois certaines issues.

[[controle:derangement]]

## Réflexes en 30 secondes

| Indice dans l'énoncé | Outil | Premier geste |
|---|---|---|
| « sachant que », test, signal, a priori | Bayes | a priori fois vraisemblance pour chaque cause, puis diviser par la somme |
| mélange, somme d'un nombre aléatoire de termes | variance totale | \(\mathbb{E}[\mathrm{Var}(X \mid Y)] + \mathrm{Var}(\mathbb{E}[X \mid Y])\) ; si \(N\) est indépendant des termes : \(\mathbb{E}[N]\sigma^2 + \mathrm{Var}(N)\mu^2\) |
| « jusqu'au premier succès » | géométrique | \(1/p\) et \((1-p)/p^2\) ; convention essais ou échecs |
| « déjà attendu », durée exponentielle | absence de mémoire | le temps restant a la loi de départ |
| « lequel en premier », plusieurs exponentielles | minimum | \(\mathcal{E}(\lambda)\) et \(\lambda_i/\lambda\) ; ensuite la course recommence |
| arrivées au hasard à rythme constant | Poisson | \(\mathcal{P}(\lambda t)\) ; \(k\)-ième arrivée en \(k/\lambda\) en moyenne ; flux indépendants : intensités additionnées |
| « tous les types », « \(r\) types distincts » | coupons | somme des \(n/(n - k + 1)\), \(k = 1, \dots, r\) |
| « au moins un de chaque », « personne à sa place » | inclusion-exclusion | événements « type absent » ou « point fixe », intersections selon leur taille |

Contrôles de vraisemblance en 5 secondes :

- avec deux causes \(B\) et \(B^c\), si \(A\) est plus probable sous \(B\) que sous \(B^c\), alors \(\mathbb{P}(B \mid A) > \mathbb{P}(B)\) : \(\mathbb{P}(A)\) est une moyenne pondérée de \(\mathbb{P}(A \mid B)\) et \(\mathbb{P}(A \mid B^c)\), donc inférieure à \(\mathbb{P}(A \mid B)\) quand \(0 < \mathbb{P}(B) < 1\) ;
- \(\mathrm{Var}(X)\) dépasse chacun des deux termes de la variance totale ;
- \(n\,H_n > n \ln n\), et le dernier type coûte à lui seul \(n\) tirages en moyenne.

## À savoir refaire sans notes

- Bayes ; l'espérance totale (cas discret) ; la variance totale en quatre lignes ; la variance d'une somme d'un nombre aléatoire de termes indépendants de ce nombre.
- Espérance et variance de la géométrique (queue, premier essai), de la loi de Poisson (moment factoriel), de l'exponentielle (queue).
- L'absence de mémoire ; l'intégrale qui donne d'un coup \(\mathbb{P}(K = i) = \lambda_i/\lambda\), l'indépendance de \(K\) et \(M\), et la relance de la course.
- \(\mathbb{P}(S_k > t)\), \(\mathbb{E}[S_k] = k/\lambda\), la superposition par le binôme.
- Le collectionneur de coupons par une somme de géométriques, et l'encadrement de \(H_n\).
- L'inclusion-exclusion par les indicatrices, et les dérangements.
""",
    "controles": {
        "vartot": {
            "question": r"La variable \(Y\) vaut 0 ou 1 avec probabilité 1/2 chacun. Sachant \(Y = 0\), \(X\) suit \(\mathcal{N}(0, 1)\) ; sachant \(Y = 1\), \(X\) suit \(\mathcal{N}(4, 1)\). Que vaut \(\mathrm{Var}(X)\) ?",
            "options": [
                {"id": "a", "texte": "@Cv_exact@",
                 "explication": r"Oui : \(\mathbb{E}[\mathrm{Var}(X \mid Y)] = 1\), et \(\mathbb{E}[X \mid Y] = 4Y\) a pour variance \(16 \times \tfrac14 = @Cv_inter@\) ; la somme vaut @Cv_exact@."},
                {"id": "b", "texte": "@Cv_intra@",
                 "explication": r"C'est seulement \(\mathbb{E}[\mathrm{Var}(X \mid Y)]\), la dispersion à l'intérieur de chaque groupe. Les deux groupes ont des moyennes différentes, 0 et 4 : leur écart ajoute le terme \(\mathrm{Var}(\mathbb{E}[X \mid Y])\)."},
                {"id": "c", "texte": "@Cv_inter@",
                 "explication": r"C'est seulement \(\mathrm{Var}(\mathbb{E}[X \mid Y]) = \mathrm{Var}(4Y)\), la dispersion des moyennes de groupe. Il manque la dispersion à l'intérieur de chaque groupe, de variance 1."},
                {"id": "d", "texte": "@Cv_d_carre@",
                 "explication": r"C'est \(\mathbb{E}[(4Y)^2] + 1 = 16 \times \tfrac12 + 1\) : le second moment de \(\mathbb{E}[X \mid Y]\) a été pris pour sa variance, sans retrancher le carré de sa moyenne, \((4 \times \tfrac12)^2\)."},
            ],
        },
        "memoire": {
            "question": r"On lance un dé équilibré, les lancers étant indépendants, et l'on note \(G\) le nombre total de lancers jusqu'au premier 6 inclus. Les 4 premiers lancers n'ont donné aucun 6. Sachant cela, quelle est l'espérance de \(G\), qui compte aussi les 4 lancers déjà faits ?",
            "options": [
                {"id": "a", "texte": "@Cm_total@",
                 "explication": r"Oui : sachant \(G > 4\), le nombre de lancers restants \(G - 4\) a la loi de \(G\) (absence de mémoire, (A1)), d'espérance \(1/p = @Cm_d_sans@\) ; total \(4 + @Cm_d_sans@\)."},
                {"id": "b", "texte": "@Cm_d_sans@",
                 "explication": r"C'est \(\mathbb{E}[G]\) sans information. C'est bien l'espérance du nombre de lancers **restants**, mais la question demande le total, qui inclut les 4 lancers déjà faits."},
                {"id": "c", "texte": "@Cm_d_du@",
                 "explication": r"C'est \(@Cm_d_sans@ - 4\) : l'idée que le 6 serait « dû » après 4 échecs. Les lancers futurs sont indépendants des précédents : il reste \(@Cm_d_sans@\) lancers en moyenne, pas \(@Cm_d_du@\)."},
                {"id": "d", "texte": "@Cm_d_echecs@",
                 "explication": r"C'est 4 plus l'espérance du nombre d'**échecs** avant le premier 6, \((1-p)/p\) avec \(p = 1/6\) : on a oublié le lancer qui donne le 6."},
            ],
        },
        "course": {
            "question": r"Les variables \(X_1\), \(X_2\), \(X_3\) sont indépendantes, de lois exponentielles d'intensités respectives 1, 2 et 5 par heure. Quelle est la probabilité que \(X_2\) soit la plus petite des trois ?",
            "options": [
                {"id": "a", "texte": "@Cc_exact_num@/@Cc_exact_den@",
                 "explication": r"Oui : \(\lambda_2/\lambda = 2/(1 + 2 + 5)\), par (A4)."},
                {"id": "b", "texte": "@Cc_d_moy_num@/@Cc_d_moy_den@",
                 "explication": r"C'est le calcul fait avec les espérances 1, 1/2 et 1/5 au lieu des intensités : \((1/2)/(1 + 1/2 + 1/5)\). La chance d'arriver le premier est proportionnelle à l'intensité, pas à la durée moyenne."},
                {"id": "c", "texte": "1/3",
                 "explication": r"Les trois variables n'ont pas la même loi : \(X_3\), d'intensité 5, est en moyenne la plus courte et gagne le plus souvent. L'égalité des chances ne vaut que pour des intensités égales."},
                {"id": "d", "texte": "@Cc_d_prod_num@/@Cc_d_prod_den@",
                 "explication": r"C'est \(\mathbb{P}(X_2 < X_1) \times \mathbb{P}(X_2 < X_3) = \tfrac{2}{2+1} \times \tfrac{2}{2+5}\) : ce produit suppose les deux événements indépendants. Ils ne le sont pas : tous deux sont favorisés quand \(X_2\) est petite."},
            ],
        },
        "coupons": {
            "question": r"On lance un dé équilibré, les lancers étant indépendants, jusqu'à avoir vu chacune des six faces au moins une fois. Quelle est l'espérance du nombre de lancers ?",
            "options": [
                {"id": "a", "texte": "@Lg_e:1@",
                 "explication": r"Oui : \(6\,H_6 = 6/6 + 6/5 + 6/4 + 6/3 + 6/2 + 6/1\)."},
                {"id": "b", "texte": "6",
                 "explication": r"C'est le nombre **minimal** de lancers, atteint seulement si les six premiers lancers sont tous différents. L'espérance compte aussi les répétitions."},
                {"id": "c", "texte": "@Cg_d_carre@",
                 "explication": r"C'est \(6 \times 6\) : comme si chaque face nouvelle coûtait 6 lancers en moyenne. Au début, presque tout lancer donne une face nouvelle ; seule la dernière face coûte 6 lancers en moyenne."},
                {"id": "d", "texte": "@Lg_6ln6:2@",
                 "explication": r"C'est \(6 \ln 6\), l'équivalent asymptotique \(n \ln n\). Il sous-estime toujours \(n\,H_n\), car \(H_n \ge \ln(n + 1) > \ln n\)."},
            ],
        },
        "derangement": {
            "question": r"Quatre personnes déposent leur chapeau ; les chapeaux leur sont rendus au hasard, les \(4!\) attributions étant équiprobables. Quelle est la probabilité que personne ne récupère le sien ?",
            "options": [
                {"id": "a", "texte": "@Cd_exact_num@/@Cd_exact_den@",
                 "explication": r"Oui : \(d_4/4! = @Lh_d4@/4!\), avec \(d_4 = 4!\,(1 - 1 + 1/2 - 1/6 + 1/24)\) par inclusion-exclusion."},
                {"id": "b", "texte": "@Cd_d_indep_num@/@Cd_d_indep_den@",
                 "explication": r"C'est \((3/4)^4\) : chaque personne rate son chapeau avec probabilité 3/4, et l'on a multiplié comme si ces événements étaient indépendants. Ils ne le sont pas : les chapeaux sont distribués sans remise."},
                {"id": "c", "texte": r"\(e^{-1} \approx @Cd_d_lim:3m@\)",
                 "explication": r"C'est la limite quand le nombre de personnes tend vers l'infini. Pour 4 personnes, la valeur exacte en diffère d'au plus \(1/5!\), mais ne lui est pas égale."},
                {"id": "d", "texte": "@Cd_d_boole@",
                 "explication": r"C'est \(1 - \sum_i \mathbb{P}(A_i) = 1 - 4 \times \tfrac14\), où \(A_i\) est l'événement « la personne \(i\) récupère son chapeau » : seul le premier terme de l'inclusion-exclusion a été gardé. Par la borne de Boole, c'est seulement un minorant, ici sans information."},
            ],
        },
    },
    "verif": {"script": "verif/seance_math_2.py"},
}
