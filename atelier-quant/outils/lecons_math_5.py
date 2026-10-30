"""Leçon de la séance de mathématiques 5 : forme d'une loi, covariance et corrélation, tests d'adéquation,
densités conjointes, intensité d'une loi de Poisson a posteriori, bâton coupé au hasard, problème des deux enfants.

Format : voir outils/lecons.py (élément de type "lecon", contenu en Markdown réduit, points de contrôle
[[controle:IDENT]] sur une ligne seule, option "a" correcte, une explication par option).
Aucun nombre calculé n'est écrit à la main : chaque jeton @CLE@, @CLE:3@ ou @CLE:3m@ est remplacé par la valeur
calculée par verif/seance_math_5.py (fichier verif/seance_math_5.out.json), et les blocs ```python-exec sont
exécutés à la construction par le noyau de l'environnement du notebook (numpy 1.21.6, scipy 1.8.1,
statsmodels 0.13.2), qui insère leur vraie sortie. Graine aléatoire fixée dans le premier bloc.
Ne redémontre pas ce que les séances de mathématiques 1 à 4 démontrent déjà : y renvoie par leur titre.
"""

LECON_MATH_5 = {
    "id": "L-MATH-5", "type": "lecon", "section": "lecon",
    "theme": "Leçon : forme d'une loi, corrélation, tests d'adéquation, densités conjointes, a posteriori, bâton coupé, deux enfants",
    "difficulte": 4, "temps_cible_s": 3000, "dossier": "coding/lecons",
    "contenu": r"""## Leçon : statistique descriptive, tests d'adéquation et densités jointes

Durée visée : 50 minutes, en sept parties indépendantes que l'on peut lire séparément.Cette leçon traite ces thèmes de façon générale, avec les cas voisins. Pour chaque notion : le problème, l'intuition, l'énoncé précis, la preuve en étapes numérotées, un exemple entièrement calculé, les cas limites, les pièges.

Ce que la leçon ne redémontre pas, et où le trouver :

- formule de Bayes, lois usuelles, absence de mémoire, minimum d'exponentielles (résultats (A3) à (A5)), processus de Poisson (résultats (P1) à (P3)) : « Séance de mathématiques 2 : conditionnement et lois usuelles » ;
- statistiques d'ordre de lois uniformes et formule de la queue (F) : « Séance de mathématiques 1 : probabilités », outil 3 ;
- vocabulaire des tests, p-value, maximum de vraisemblance, inégalités de Jensen et de Cauchy-Schwarz : « Séance de mathématiques 3 : statistique et inégalités » ;
- fonction génératrice de la loi normale : « Séance de mathématiques 4 : Markov, martingales et séries temporelles », partie 4 ;
- hypothèses de la régression linéaire et conséquence de chaque violation : « Séance de ML 1 : régression linéaire, Ridge et Lasso », partie 2. Elles ne sont pas reprises ici.

Conventions de notation. Une majuscule désigne une variable aléatoire, une minuscule une valeur observée ou un paramètre fixe, le gras un vecteur ou une matrice. Exceptions d'usage, déclarées ici : les fonctions de répartition \(F\), \(F_0\), \(G\) et la fonction quantile \(Q\) sont des fonctions fixes, alors que les fonctions de répartition empiriques \(F_n\) et \(G_n\) sont aléatoires ; \(\Phi\) et \(\varphi\) sont la fonction de répartition et la densité de la loi normale \(\mathcal{N}(0, 1)\) ; \(\Gamma\) est la fonction gamma, \(\Gamma(a) = \int_0^\infty u^{a-1} e^{-u}\,du\) pour \(a > 0\), qui vérifie \(\Gamma(a + 1) = a\,\Gamma(a)\) (intégration par parties) et \(\Gamma(n) = (n-1)!\) pour un entier \(n \ge 1\) ; \(H_0\) désigne une hypothèse nulle ; les paramètres grecs \(\mu\), \(\sigma\), \(\lambda\), \(\rho\), \(\gamma_1\), \(\kappa\) sont des nombres fixes, sauf en partie 5 où l'intensité inconnue est une variable aléatoire notée \(\Lambda\), de valeurs \(\lambda\). \(\mathbb{P}\), \(\mathbb{E}\), \(\mathrm{Var}\), \(\mathrm{Cov}\) désignent probabilité, espérance, variance, covariance ; \(I_A\) est l'indicatrice de l'événement \(A\) (1 si \(A\) se produit, 0 sinon). Tous les nombres hors des sorties de code viennent de `verif/seance_math_5.py`, qui contrôle chaque probabilité et chaque espérance par un calcul exact et par une simulation.

```python-exec
import numpy as np
from scipy import stats
rng = np.random.default_rng(5)
print("prêt")
```

## Partie 1 : forme d'une loi

### Le problème

On résume une loi par quelques nombres : où elle se situe (moyenne, médiane, mode), à quel point elle est dispersée (écart-type), si elle est dissymétrique (coefficient d'asymétrie), quel poids ont ses valeurs extrêmes (kurtosis). Les questions types : « la loi est étalée à droite, dans quel ordre sont la moyenne, la médiane et le mode ? », « que vaut l'asymétrie de telle loi ? », « qu'est-ce qui est vrai pour toute loi ? ».

### Définitions

\(X\) est une variable aléatoire réelle, de fonction de répartition \(F(x) = \mathbb{P}(X \le x)\).

- **Moyenne** : \(\mu = \mathbb{E}[X]\), si \(\mathbb{E}\lvert X \rvert\) est finie. **Écart-type** : \(\sigma = \sqrt{\mathrm{Var}(X)}\), supposé fini et strictement positif. \(\mu\) et \(\sigma\) ont l'unité de \(X\).
- **Médiane** : tout réel \(m\) tel que \(\mathbb{P}(X \le m) \ge 1/2\) et \(\mathbb{P}(X \ge m) \ge 1/2\). Si \(F\) est continue et strictement croissante, \(m\) est l'unique solution de \(F(m) = 1/2\).
- **Mode** : un point où la densité de \(X\) (loi à densité) ou la probabilité \(\mathbb{P}(X = x)\) (loi discrète) est maximale. La loi est **unimodale** si cette fonction croît puis décroît.
- **Coefficient d'asymétrie** (skewness) : \(\gamma_1 = \mathbb{E}[(X - \mu)^3]/\sigma^3\), si \(\mathbb{E}\lvert X \rvert^3\) est finie.
- **Kurtosis** (aplatissement) : \(\kappa = \mathbb{E}[(X - \mu)^4]/\sigma^4\), si \(\mathbb{E}[X^4]\) est finie ; l'**excès de kurtosis** est \(\kappa - 3\), nul pour une loi normale.

Avec la variable centrée réduite \(Z = (X - \mu)/\sigma\), d'espérance 0 et de variance 1 : \(\gamma_1 = \mathbb{E}[Z^3]\) et \(\kappa = \mathbb{E}[Z^4]\).

### L'intuition

Le cube conserve le signe : une longue queue à droite produit de grands cubes positifs, donc \(\gamma_1 > 0\). Cette queue tire la moyenne vers la droite, déplace moins la médiane (qui ne compte que le nombre de valeurs de chaque côté) et pas du tout le mode. D'où la règle « étalée à droite : mode, puis médiane, puis moyenne ». La puissance 4 donne un poids énorme aux valeurs éloignées : \(\kappa\) mesure le poids des queues, pas la forme du sommet.

### L'énoncé

1. **Invariance.** Pour des réels \(a \ne 0\) et \(b\), la variable \(aX + b\) a pour coefficient d'asymétrie \(\gamma_1\) si \(a > 0\), \(-\gamma_1\) si \(a < 0\), et pour kurtosis \(\kappa\). Ces deux nombres n'ont pas d'unité.
2. **Toujours vrai.** Si \(\mathrm{Var}(X)\) est finie, toute médiane \(m\) vérifie \(\lvert \mu - m \rvert \le \sigma\).
3. **Toujours vrai.** \(\kappa \ge 1\). Si la loi est symétrique autour d'un réel \(c\) (\(X - c\) et \(c - X\) ont la même loi) et si \(\mathbb{E}\lvert X \rvert^3\) est finie, alors \(\mu = c\), \(c\) est une médiane et \(\gamma_1 = 0\).
4. **Somme.** Si \(X_1, \dots, X_n\) sont indépendantes, de même loi d'asymétrie \(\gamma_1\) et de kurtosis \(\kappa\), leur somme a pour asymétrie \(\gamma_1/\sqrt{n}\) et pour excès de kurtosis \((\kappa - 3)/n\).
5. **Règle empirique, pas théorème.** « \(\gamma_1 > 0\) entraîne mode < médiane < moyenne » est vrai pour la loi exponentielle et la loi log-normale (démontré plus bas), pour la loi gamma de forme 2 (démontré plus bas) et de forme \(a\) entre @La_gam_grille_min@ et @La_gam_grille_max@ (constaté numériquement par le script sur une grille de pas 0,1), et **faux en général** : deux contre-exemples calculés plus bas.

### La preuve

1. Invariance : \(aX + b - \mathbb{E}[aX + b] = a\,(X - \mu)\), et l'écart-type de \(aX + b\) est \(\lvert a \rvert \sigma\). La variable centrée réduite de \(aX + b\) est donc \(Z\) si \(a > 0\) et \(-Z\) si \(a < 0\) ; \(\mathbb{E}[(-Z)^3] = -\gamma_1\) et \(\mathbb{E}[(-Z)^4] = \kappa\).
2. La médiane minimise l'écart absolu moyen : pour tout réel \(c\), \(\mathbb{E}\lvert X - m \rvert \le \mathbb{E}\lvert X - c \rvert\). Soit \(c > m\). Si \(x \le m\), \(\lvert x - c \rvert - \lvert x - m \rvert = c - m\). Si \(x > m\), \(\lvert x - c \rvert - \lvert x - m \rvert \ge -(c - m)\) par l'inégalité triangulaire. En prenant l'espérance, \(\mathbb{E}\lvert X - c \rvert - \mathbb{E}\lvert X - m \rvert \ge (c - m)\,\big(\mathbb{P}(X \le m) - \mathbb{P}(X > m)\big) \ge 0\), car \(\mathbb{P}(X \le m) \ge 1/2 \ge \mathbb{P}(X > m)\). Le cas \(c < m\) est identique avec \(\mathbb{P}(X \ge m) \ge 1/2\).
3. Énoncé 2, trois inégalités : \(\lvert \mu - m \rvert = \lvert \mathbb{E}[X - m] \rvert \le \mathbb{E}\lvert X - m \rvert\), car \(-\lvert Y \rvert \le Y \le \lvert Y \rvert\) pour toute variable \(Y\) ; \(\mathbb{E}\lvert X - m \rvert \le \mathbb{E}\lvert X - \mu \rvert\) par l'étape 2 avec \(c = \mu\) ; \(\mathbb{E}\lvert X - \mu \rvert \le \sqrt{\mathbb{E}[(X - \mu)^2]} = \sigma\) par l'inégalité de Cauchy-Schwarz appliquée à \(\lvert X - \mu \rvert\) et à la constante 1 (séance de mathématiques 3, partie 6).
4. Énoncé 3 : \(\kappa = \mathbb{E}[Z^4] = \mathrm{Var}(Z^2) + \mathbb{E}[Z^2]^2 \ge 1\), avec égalité si et seulement si \(Z^2 = 1\) avec probabilité 1, c'est-à-dire si \(X\) vaut \(\mu - \sigma\) ou \(\mu + \sigma\) avec probabilité 1/2 chacun. Symétrie : \(X - c\) et \(c - X\) ont la même espérance, donc \(\mathbb{E}[X - c] = -\mathbb{E}[X - c] = 0\) ; de même \(\mathbb{E}[(X - c)^3] = 0\) ; enfin \(\mathbb{P}(X \le c) = \mathbb{P}(X - c \le 0) = \mathbb{P}(c - X \le 0) = \mathbb{P}(X \ge c)\), et ces deux probabilités, de somme au moins 1, valent chacune au moins 1/2.
5. Énoncé 4. On pose \(A_i = X_i - \mu\), d'espérance 0, de variance \(\sigma^2\), avec \(\mathbb{E}[A_i^3] = \gamma_1 \sigma^3\) et \(\mathbb{E}[A_i^4] = \kappa\,\sigma^4\). En développant \((A_1 + \dots + A_n)^3\), tout produit où un indice apparaît une seule fois a une espérance nulle (indépendance, puis \(\mathbb{E}[A_i] = 0\)) : il reste les \(n\) termes \(A_i^3\), donc \(\mathbb{E}[(\sum_i A_i)^3] = n\,\gamma_1 \sigma^3\). La variance de la somme vaut \(n\sigma^2\), d'où l'asymétrie \(n\,\gamma_1 \sigma^3/(n\sigma^2)^{3/2} = \gamma_1/\sqrt{n}\).
6. Puissance 4 : les produits d'espérance non nulle sont les \(n\) termes \(A_i^4\) et les termes \(A_i^2 A_j^2\) avec \(i \ne j\) ; il y a \(n(n-1)/2\) paires \(\{i, j\}\) et \(\binom{4}{2} = 6\) façons de placer les deux \(A_i\) parmi les quatre facteurs, soit \(3n(n-1)\) termes d'espérance \(\sigma^4\). Donc \(\mathbb{E}[(\sum_i A_i)^4] = n\kappa\,\sigma^4 + 3n(n-1)\,\sigma^4\), et le kurtosis de la somme vaut \(\big(n\kappa + 3n(n-1)\big)/n^2 = 3 + (\kappa - 3)/n\).

### Exemples calculés

**Loi exponentielle de paramètre \(\lambda\).** \(\mathbb{P}(X > x) = e^{-\lambda x}\), moyenne et écart-type \(1/\lambda\) (séance de mathématiques 2, partie 2). La densité \(\lambda e^{-\lambda x}\) décroît : le mode est 0. La médiane résout \(e^{-\lambda m} = 1/2\) : \(m = \ln 2/\lambda\). Pour \(\gamma_1\) et \(\kappa\), on prend \(\lambda = 1\) (énoncé 1 : \(X/\lambda\) a la même asymétrie et le même kurtosis), où \(\mathbb{E}[X^r] = \Gamma(r + 1) = r!\) : \(\mathbb{E}[(X - 1)^3] = 3! - 3 \times 2! + 3 \times 1 - 1 = @La_exp_skew@\) et \(\mathbb{E}[(X - 1)^4] = 4! - 4 \times 3! + 6 \times 2! - 4 \times 1 + 1 = @La_exp_kurt@\). Donc \(\gamma_1 = @La_exp_skew@\) et \(\kappa = @La_exp_kurt@\) (excès @La_exp_exces@), quel que soit \(\lambda\). Avec une moyenne de 2 : mode 0, médiane \(2\ln 2 \approx @La_exp_med:3m@\), moyenne @La_exp_moy@ ; l'écart \(\lvert \mu - m \rvert = @La_exp_ecart:3m@\) est bien sous \(\sigma = @La_exp_sigma@\).

**Loi log-normale.** \(X = e^{a + bY}\), où \(Y\) suit \(\mathcal{N}(0, 1)\), \(a\) est un réel et \(b > 0\).

1. Moments : \(\mathbb{E}[X^r] = e^{ra}\,\mathbb{E}[e^{rbY}] = e^{ra + r^2 b^2/2}\) (fonction génératrice de la loi normale, séance de mathématiques 4, partie 4). Moyenne : \(e^{a + b^2/2}\).
2. Médiane : l'exponentielle est croissante, donc \(\mathbb{P}(X \le e^a) = \mathbb{P}(Y \le 0) = 1/2\) : la médiane est \(e^a\).
3. Mode : \(\mathbb{P}(X \le x) = \Phi\big((\ln x - a)/b\big)\), de dérivée \(\varphi\big((\ln x - a)/b\big)/(bx)\). Avec \(t = \ln x\), le logarithme de cette densité vaut \(-(t - a)^2/(2b^2) - t\) à une constante près ; sa dérivée en \(t\), \(-(t - a)/b^2 - 1\), est positive avant \(t = a - b^2\) et négative après. Le mode est \(e^{a - b^2}\).
4. Donc \(e^{a - b^2} < e^a < e^{a + b^2/2}\) : mode < médiane < moyenne pour tout \(b > 0\).
5. Asymétrie. On prend \(a = 0\) (\(e^a\) est un facteur d'échelle, énoncé 1) et \(w = e^{b^2}\). Les trois premiers moments sont \(w^{1/2}\), \(w^2\), \(w^{9/2}\) ; la variance vaut \(w^2 - w = w(w - 1)\) ; le moment centré d'ordre 3 vaut \(w^{9/2} - 3w^{5/2} + 2w^{3/2} = w^{3/2}(w^3 - 3w + 2) = w^{3/2}(w - 1)^2 (w + 2)\). D'où \(\gamma_1 = (w + 2)\sqrt{w - 1} > 0\).

Avec \(a = 0\) et \(b = 0{,}5\) : mode \(@La_ln_mode:3m@\), médiane @La_ln_med@, moyenne \(@La_ln_moy:3m@\) (simulation : @La_ln_moy_mc:3@), écart-type \(@La_ln_sigma:3m@\), \(\gamma_1 = @La_ln_skew:3m@\).

**Loi gamma de forme \(a > 0\) et de paramètre \(\lambda > 0\)** : densité \(\lambda^a x^{a-1} e^{-\lambda x}/\Gamma(a)\) pour \(x > 0\) ; \(a = 1\) redonne la loi exponentielle. Avec \(\lambda = 1\), \(\mathbb{E}[X^r] = \Gamma(a + r)/\Gamma(a) = a(a + 1)\cdots(a + r - 1)\). Moyenne \(a\), variance \(a(a+1) - a^2 = a\), moment centré d'ordre 3 : \(a(a+1)(a+2) - 3a \cdot a(a+1) + 2a^3 = 2a\). Donc, pour tout \(\lambda\) : moyenne \(a/\lambda\), variance \(a/\lambda^2\), \(\gamma_1 = 2a/a^{3/2} = 2/\sqrt{a}\), et le même calcul à l'ordre 4 (fait par le script) donne l'excès \(6/a\). Le mode est \((a - 1)/\lambda\) si \(a \ge 1\) : le logarithme de la densité, \((a - 1)\ln x - \lambda x\) à une constante près, a pour dérivée \((a - 1)/x - \lambda\).

La médiane d'une loi gamma n'a pas de formule fermée en général : \(F(m) = 1/2\) est une équation transcendante en \(m\) (pour \(a = 1\), elle se résout : \(\ln 2/\lambda\)). Pour \(a = 2\), \(\lambda = 1\) : une intégration par parties donne \(F(x) = 1 - e^{-x}(1 + x)\), donc \(F(1) = 1 - 2/e \approx @La_gam_F1:3m@ < 1/2\) et \(F(2) = 1 - 3/e^2 \approx @La_gam_F2:3m@ > 1/2\). La médiane est entre le mode 1 et la moyenne 2, ce qui démontre la règle pour cette loi ; numériquement \(m \approx @La_gam_med:3m@\), et \(\gamma_1 = 2/\sqrt{2} \approx @La_gam_skew:3m@\).

**Somme.** La somme de \(@La_somme_n@\) variables exponentielles indépendantes de même paramètre a pour asymétrie \(2/\sqrt{25} = @La_somme_skew:1m@\) et pour excès \(6/25 = @La_somme_exces:2m@\) : l'asymétrie s'efface en \(1/\sqrt{n}\), l'excès de kurtosis en \(1/n\).

**Contre-exemple discret à la règle.** \(X\) vaut \(-2\) avec probabilité 0,4, 1 avec probabilité 0,5 et 4 avec probabilité 0,1. Les probabilités croissent puis décroissent : la loi est unimodale, de mode 1. \(\mathbb{P}(X \le 1) = 0{,}9\) et \(\mathbb{P}(X \ge 1) = 0{,}6\) : la médiane est @La_B_med@. Moyenne : \(-0{,}8 + 0{,}5 + 0{,}4 = @La_B_moy:1m@\). Variance \(@La_B_var:2m@\), moment centré d'ordre 3 \(@La_B_c3:3m@\), donc \(\gamma_1 = @La_B_skew:3m@ > 0\). L'asymétrie est positive et pourtant la moyenne (\(@La_B_moy:1m@\)) est **sous** la médiane (@La_B_med@).

**Contre-exemple continu unimodal : loi de Weibull.** Soit \(k > 1\) et \(X = Y^{1/k}\), où \(Y\) suit la loi exponentielle de paramètre 1. Alors \(\mathbb{P}(X > x) = \mathbb{P}(Y > x^k) = e^{-x^k}\), \(\mathbb{E}[X^r] = \mathbb{E}[Y^{r/k}] = \Gamma(1 + r/k)\), la médiane vaut \((\ln 2)^{1/k}\), et la densité \(k x^{k-1} e^{-x^k}\), dont le logarithme a pour dérivée \((k - 1)/x - k x^{k-1}\), a pour mode \(\big((k-1)/k\big)^{1/k}\). Pour \(k = @La_w_k:1m@\) : moyenne \(@La_w_moy:4m@\), médiane \(@La_w_med:4m@\), mode \(@La_w_mode:4m@\), et \(\gamma_1 = @La_w_skew:4m@ > 0\). L'ordre est entièrement inversé, moyenne < médiane < mode, avec une asymétrie positive.

**Kurtosis de référence** : @La_norm_kurt@ pour la loi normale (\(\mathbb{E}[Y^4] = 3\), séance de mathématiques 3, partie 1), \(@La_unif_kurt:1m@\) pour la loi uniforme (excès \(@La_unif_exces:1m@\) : queues inexistantes), @La_exp_kurt@ pour l'exponentielle, excès @La_t5_exces@ pour la loi de Student à 5 degrés de liberté (queues lourdes), 1 pour la variable qui vaut \(-1\) ou \(+1\) avec probabilité 1/2 (le minimum).

Sur un échantillon, on remplace les espérances par des moyennes. Convention de scipy 1.8.1, vérifiée par le code suivant : `stats.kurtosis` rend par défaut l'**excès**.

```python-exec
x = rng.exponential(2.0, 400000)
print("moyenne, médiane             :", round(x.mean(), 3), round(float(np.median(x)), 3))
print("stats.skew                   :", round(float(stats.skew(x)), 2))
print("stats.kurtosis (par défaut)  :", round(float(stats.kurtosis(x)), 2))
print("stats.kurtosis(fisher=False) :", round(float(stats.kurtosis(x, fisher=False)), 2))
```

### Cas limites

- **Moments absents.** La densité de la loi de Student à \(\nu\) degrés de liberté décroît comme \(\lvert x \rvert^{-(\nu + 1)}\) : \(\mathbb{E}\lvert X \rvert^r\) est finie si et seulement si \(r < \nu\). L'asymétrie n'existe que pour \(\nu > 3\), le kurtosis pour \(\nu > 4\) ; pour \(\nu = 1\) (loi de Cauchy), même la moyenne n'existe pas, alors que la médiane vaut 0.
- **Médiane non unique.** Pour un dé équilibré, tout réel de \([3, 4]\) est une médiane.
- **Loi symétrique unimodale** : mode, médiane et moyenne coïncident au centre de symétrie.

### Pièges

1. Le signe de \(\gamma_1\) ne donne pas l'ordre de la moyenne et de la médiane (contre-exemples ci-dessus). Ce qui est garanti : \(\lvert \mu - m \rvert \le \sigma\).
2. \(\gamma_1 = 0\) n'implique pas la symétrie : la loi qui vaut \(-2\), 1, 3 avec probabilités 0,4, 0,5, 0,1 a une moyenne nulle et un moment centré d'ordre 3 nul (\(-3{,}2 + 0{,}5 + 2{,}7\)), sans être symétrique.
3. Kurtosis ou excès : toujours vérifier la convention. Une loi normale a un kurtosis de 3 et un excès de 0.
4. Un kurtosis élevé vient des queues, pas d'un sommet pointu : il est dominé par les valeurs éloignées de plusieurs écarts-types.
5. Asymétrie et kurtosis empiriques sont instables : la variance de la moyenne empirique des \(Z_i^4\) fait intervenir \(\mathbb{E}[Z^8]\), qui peut être immense ou infinie.

[[controle:forme]]

## Partie 2 : covariance, corrélation, cote Z

### Le problème

Deux quantités varient ensemble : dans quel sens, avec quelle force, et la réponse dépend-elle des unités ? Et comment comparer deux mesures faites sur des échelles différentes (deux notes à deux examens) ?

### Définitions

\(X\) et \(Y\) sont deux variables aléatoires de variances finies, de moyennes \(\mu_X\), \(\mu_Y\) et d'écarts-types \(\sigma_X\), \(\sigma_Y\).

- **Covariance** : \(\mathrm{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)] = \mathbb{E}[XY] - \mu_X \mu_Y\). Son unité est le produit des unités de \(X\) et de \(Y\).
- **Corrélation** (de Pearson) : \(\rho = \mathrm{Cov}(X, Y)/(\sigma_X \sigma_Y)\), définie seulement si \(\sigma_X > 0\) et \(\sigma_Y > 0\). Sans unité.
- **Cote Z** d'une valeur \(x\) : \(z = (x - \mu_X)/\sigma_X\), le nombre d'écarts-types entre \(x\) et la moyenne. Sans unité. La variable centrée réduite \(Z_X = (X - \mu_X)/\sigma_X\) a pour espérance 0 et pour variance 1, et \(\rho = \mathbb{E}[Z_X Z_Y]\) : la corrélation est la covariance des cotes Z.
- **Corrélation de rang** (de Spearman) d'un échantillon de \(n\) couples \((x_i, y_i)\) : la corrélation de Pearson calculée après avoir remplacé chaque \(x_i\) par son rang parmi les \(x\) et chaque \(y_i\) par son rang parmi les \(y\). C'est le coefficient d'information de la « Séance quantitative », partie 4.

### L'énoncé

1. **(K1) Changement affine d'unité.** Pour des réels \(a\), \(b\), \(c\), \(d\) : \(\mathrm{Cov}(aX + b, cY + d) = ac\,\mathrm{Cov}(X, Y)\). Si \(a \ne 0\) et \(c \ne 0\), la corrélation de \(aX + b\) et \(cY + d\) vaut \(\rho\) si \(ac > 0\) et \(-\rho\) si \(ac < 0\).
2. **(K2) Bornes.** \(-1 \le \rho \le 1\), et \(\lvert \rho \rvert = 1\) si et seulement si \(Y = \alpha + \beta X\) avec probabilité 1, pour des réels \(\alpha\) et \(\beta \ne 0\).
3. **(K3) Dépendance linéaire.** Le minimum de \(\mathbb{E}[(Y - \alpha - \beta X)^2]\) sur les réels \(\alpha\), \(\beta\) vaut \(\sigma_Y^2\,(1 - \rho^2)\) ; il est atteint pour la pente \(\beta = \mathrm{Cov}(X, Y)/\sigma_X^2 = \rho\,\sigma_Y/\sigma_X\). \(\rho^2\) est la part de la variance de \(Y\) expliquée par la meilleure droite.
4. **(K4)** Si \(X\) et \(Y\) sont indépendantes, \(\mathrm{Cov}(X, Y) = 0\). La réciproque est fausse.
5. **(K5)** La corrélation de rang ne change pas si l'on applique à tous les \(x_i\) une fonction strictement croissante, ou à tous les \(y_i\) ; elle vaut 1 dès que \(y_i = g(x_i)\) avec \(g\) strictement croissante et des \(x_i\) distincts. La corrélation de Pearson n'a aucune de ces deux propriétés quand la fonction appliquée, ou \(g\), n'est pas affine.

### La preuve

1. (K1) : \(aX + b - \mathbb{E}[aX + b] = a(X - \mu_X)\) et \(cY + d - \mathbb{E}[cY + d] = c(Y - \mu_Y)\) ; le produit est \(ac\,(X - \mu_X)(Y - \mu_Y)\). Les écarts-types deviennent \(\lvert a \rvert \sigma_X\) et \(\lvert c \rvert \sigma_Y\), donc la corrélation est multipliée par \(ac/\lvert ac \rvert\), qui vaut 1 ou \(-1\).
2. (K2) : c'est l'inégalité de Cauchy-Schwarz appliquée à \(X - \mu_X\) et \(Y - \mu_Y\), avec son cas d'égalité (séance de mathématiques 3, partie 6).
3. (K3) : pour \(\beta\) fixé, \(W = Y - \beta X\) vérifie \(\mathbb{E}[(W - \alpha)^2] = \mathrm{Var}(W) + (\mathbb{E}[W] - \alpha)^2\), minimal pour \(\alpha = \mathbb{E}[W]\). Reste \(\mathrm{Var}(Y - \beta X) = \sigma_Y^2 - 2\beta\,\mathrm{Cov}(X, Y) + \beta^2 \sigma_X^2\), trinôme en \(\beta\) minimal en \(\beta = \mathrm{Cov}(X, Y)/\sigma_X^2\), où il vaut \(\sigma_Y^2 - \mathrm{Cov}(X, Y)^2/\sigma_X^2 = \sigma_Y^2 (1 - \rho^2)\).
4. (K4) : si \(X\) et \(Y\) sont indépendantes et intégrables, \(\mathbb{E}[XY] = \mathbb{E}[X]\,\mathbb{E}[Y]\) (théorème de Fubini, admis). Le contre-exemple de la réciproque est calculé plus bas.
5. (K5) : si \(g\) est strictement croissante, \(x_i < x_j\) équivaut à \(g(x_i) < g(x_j)\) : les rangs ne changent pas. Si \(y_i = g(x_i)\), les rangs des \(y\) sont ceux des \(x\), et la corrélation de Pearson d'une suite non constante avec elle-même vaut 1.

### Exemples calculés

**Unités.** Deux rendements journaliers exprimés en nombres décimaux : \(\sigma_X = 0{,}01\), \(\sigma_Y = 0{,}02\), \(\mathrm{Cov}(X, Y) = 0{,}00012\), donc \(\rho = 0{,}00012/(0{,}01 \times 0{,}02) = @Lb_rho:1m@\). En pourcentage (\(a = c = 100\)) : écarts-types \(@Lb_sx_pct@\) et \(@Lb_sy_pct@\), covariance \(100 \times 100 \times 0{,}00012 = @Lb_cov_pct:1m@\) (en pour cent au carré), et \(\rho = @Lb_cov_pct:1m@/(1 \times 2) = @Lb_rho:1m@\), inchangée. La covariance a été multipliée par 10 000 ; la corrélation non. La pente de la meilleure droite de \(Y\) en \(X\) vaut \(\rho\,\sigma_Y/\sigma_X = @Lb_pente:1m@\), et la part de variance non expliquée \(1 - \rho^2 = @Lb_var_res:2m@\).

**Corrélation nulle sans indépendance.** \(X\) vaut \(-1\), 0 ou 1 avec probabilité 1/3 chacun, et \(Y = X^2\). \(\mathbb{E}[X] = 0\) et \(\mathbb{E}[XY] = \mathbb{E}[X^3] = 0\), donc \(\mathrm{Cov}(X, Y) = 0\). Pourtant \(Y\) est une fonction de \(X\) : \(\mathbb{P}(X = 0, Y = 0) = \tfrac{@Lb_dep_joint_num@}{@Lb_dep_joint_den@}\), alors que \(\mathbb{P}(X = 0)\,\mathbb{P}(Y = 0) = \tfrac{@Lb_dep_prod_num@}{@Lb_dep_prod_den@}\). La corrélation ne voit que la partie linéaire de la dépendance.

**Deux lois normales non corrélées et dépendantes.** \(X\) suit \(\mathcal{N}(0, 1)\), \(S\) vaut \(-1\) ou 1 avec probabilité 1/2, indépendamment de \(X\), et \(Y = SX\). Par symétrie de la loi normale, \(Y\) suit \(\mathcal{N}(0, 1)\) ; \(\mathrm{Cov}(X, Y) = \mathbb{E}[S]\,\mathbb{E}[X^2] = 0\) ; mais \(\lvert Y \rvert = \lvert X \rvert\). Le résultat « non corrélées donc indépendantes » exige que le **couple** soit gaussien, c'est-à-dire que toute combinaison linéaire soit normale (résultat général sur les vecteurs gaussiens, admis) ; ici \(X + Y\) vaut 0 avec probabilité 1/2 et n'est pas normale.

**Pearson contre Spearman.** \(X\) suit \(\mathcal{N}(0, 1)\) et \(Y = e^X\), fonction strictement croissante de \(X\) : la corrélation de rang de tout échantillon vaut @Lb_spearman_exp@. Corrélation de Pearson : comme \(x - x^2/2 = -(x - 1)^2/2 + 1/2\), on a \(e^x \varphi(x) = e^{1/2}\,\varphi(x - 1)\), donc \(\mathbb{E}[X e^X] = e^{1/2} \int x\,\varphi(x - 1)\,dx = e^{1/2}\) (moyenne de \(\mathcal{N}(1, 1)\)) ; \(\mathbb{E}[e^X] = e^{1/2}\) et \(\mathbb{E}[e^{2X}] = e^2\) (séance de mathématiques 4, partie 4). Donc \(\mathrm{Cov}(X, Y) = e^{1/2}\), \(\mathrm{Var}(Y) = e^2 - e\), et \(\rho = e^{1/2}/\sqrt{e(e - 1)} = 1/\sqrt{e - 1} \approx @Lb_pearson_exp:3m@\) (simulation : @Lb_pearson_exp_mc:3@). Une dépendance parfaite mais non linéaire donne une corrélation de Pearson nettement inférieure à 1.

**Cote Z : comparer deux notes.** Examen 1 : note 14 sur 20, moyenne des candidats 11, écart-type 2, cote \(z = (14 - 11)/2 = @Lb_z1:1m@\). Examen 2 : note 620, moyenne 500, écart-type 100, cote \(z = @Lb_z2:1m@\). La première note est plus loin au-dessus de sa moyenne, en écarts-types : c'est la meilleure en position relative. Traduire une cote en rang demande la forme de la loi. Si les deux lois de notes sont normales, les proportions de candidats dépassés sont \(\Phi(1{,}5) \approx @Lb_phi_15:3m@\) et \(\Phi(1{,}2) \approx @Lb_phi_12:3m@\), dans le même ordre que les cotes. Avec des formes différentes, l'ordre peut s'inverser : pour des notes de loi exponentielle de moyenne 1 (écart-type 1), la cote 1,5 correspond à la note \(@Lb_exp_x:1m@\), dépassée par une proportion \(e^{-2{,}5}\) des candidats seulement : la proportion de candidats dépassés est \(@Lb_exp_15:3m@\). Une cote plus faible, 1,4, sur une loi normale donne \(\Phi(1{,}4) \approx @Lb_phi_14:3m@\), une proportion plus élevée.

### Cas limites

- Si \(X\) est constante, la covariance est nulle et la corrélation n'est pas définie (division par \(\sigma_X = 0\)).
- Avec deux points distincts en abscisse et en ordonnée, la corrélation empirique vaut toujours 1 ou \(-1\) : une droite passe par deux points.
- Si seul \(X\) change d'unité (\(a \ne 1\), \(c = 1\)), la pente de \(Y\) en \(X\) est divisée par \(a\), la covariance multipliée par \(a\), la corrélation inchangée si \(a > 0\).

### Pièges

1. Une covariance « grande » ne veut rien dire sans les unités : on compare des corrélations.
2. \(\rho\) n'est pas une pente : la pente est \(\rho\,\sigma_Y/\sigma_X\). Une corrélation de 0,6 ne dit pas que \(Y\) monte de 0,6 quand \(X\) monte de 1.
3. Corrélation nulle n'est pas indépendance, sauf pour un couple gaussien.
4. Pearson mesure une liaison **linéaire**, Spearman une liaison **monotone** ; aucune des deux ne détecte \(Y = X^2\) quand la loi de \(X\) est symétrique autour de 0.
5. Deux cotes Z ne se traduisent en rangs comparables que si les deux lois ont la même forme.

[[controle:unites]]

## Partie 3 : tests d'adéquation

### Le problème

On observe \(x_1, \dots, x_n\), réalisations de variables \(X_1, \dots, X_n\) indépendantes et de même fonction de répartition \(F\) inconnue. Les données sont-elles compatibles avec une loi donnée ? L'hypothèse nulle \(H_0\) est soit **simple**, \(F = F_0\) entièrement spécifiée (loi exponentielle de moyenne 2), soit **composite**, \(F\) appartient à une famille dont les paramètres sont estimés (une loi de Poisson, de paramètre inconnu). Le vocabulaire (niveau, p-value, puissance) est celui de la séance de mathématiques 3, partie 4. Trois idées : comparer des effectifs par classe (khi-deux), comparer les fonctions de répartition par leur plus grand écart (Kolmogorov-Smirnov) ou par un écart quadratique pondéré (Anderson-Darling). Le bon test dépend du type de loi : discrète ou continue, paramètres fixés ou estimés.

### Le khi-deux d'adéquation : définition et énoncé

On découpe les valeurs possibles en \(k \ge 2\) classes disjointes (pour une loi discrète : ses valeurs, les plus rares regroupées). Sous \(H_0\), la classe \(j\) a la probabilité \(p_j > 0\), avec \(p_1 + \dots + p_k = 1\). \(O_j\) est le nombre (aléatoire) d'observations dans la classe \(j\), et \(e_j = n p_j\) l'effectif attendu (un nombre fixe). La statistique du test est

\(\displaystyle C = \sum_{j=1}^{k} \frac{(O_j - e_j)^2}{e_j}.\)

1. **Exact, pour tout \(n\).** Sous \(H_0\) simple, \(\mathbb{E}[C] = k - 1\).
2. **Asymptotique.** Sous \(H_0\) simple, quand \(n \to \infty\), \(C\) converge en loi vers la loi du khi-deux à \(k - 1\) degrés de liberté, notée \(\chi^2_{k-1}\) : la loi de la somme de \(k - 1\) carrés de variables \(\mathcal{N}(0, 1)\) indépendantes. On rejette \(H_0\) quand \(C\) dépasse le quantile d'ordre \(1 - \alpha\) de cette loi ; la p-value de la valeur observée \(c\) est la probabilité qu'une variable de loi \(\chi^2_{k-1}\) dépasse \(c\).
3. **Paramètres estimés** (théorème de Fisher, 1924, admis). Si les \(p_j\) dépendent de \(s\) paramètres estimés par maximum de vraisemblance à partir des effectifs des classes, la loi limite est \(\chi^2_{k-1-s}\) : un degré de liberté en moins par paramètre estimé.
4. **Condition d'usage.** L'énoncé 2 est une limite. La règle « tous les \(e_j\) valent au moins 5 » est une **règle d'usage**, pas un théorème : elle dit quand l'approximation est jugée acceptable. Si elle n'est pas remplie, on regroupe des classes voisines (ce qui diminue \(k\)).

### La preuve du khi-deux

1. Énoncé 1 : \(O_j\) compte les succès de \(n\) essais indépendants de probabilité \(p_j\), donc suit la loi binomiale de paramètres \(n\) et \(p_j\) : \(\mathbb{E}[(O_j - e_j)^2] = \mathrm{Var}(O_j) = n p_j (1 - p_j)\). Donc \(\mathbb{E}[C] = \sum_j (1 - p_j) = k - 1\).
2. Énoncé 2 pour \(k = 2\). Comme \(O_1 + O_2 = n = e_1 + e_2\), on a \(O_2 - e_2 = -(O_1 - e_1)\), donc \(C = (O_1 - np_1)^2 \big(\tfrac{1}{np_1} + \tfrac{1}{np_2}\big) = \dfrac{(O_1 - np_1)^2}{n p_1 p_2}\), puisque \(\tfrac{1}{p_1} + \tfrac{1}{p_2} = \tfrac{1}{p_1 p_2}\). C'est le carré de la variable binomiale centrée réduite, qui converge en loi vers \(\mathcal{N}(0, 1)\) par le théorème central limite (admis) ; son carré converge donc en loi vers \(\chi^2_1\) (une fonction continue conserve la convergence en loi, admis).
3. Énoncé 2 pour \(k\) quelconque. On pose \(R_j = (O_j - e_j)/\sqrt{e_j}\), de sorte que \(C = R_1^2 + \dots + R_k^2\). Pour une seule observation, les indicatrices \(I_i\), \(I_j\) de deux classes distinctes vérifient \(I_i I_j = 0\), donc \(\mathrm{Cov}(I_i, I_j) = -p_i p_j\), et \(\mathrm{Var}(I_j) = p_j(1 - p_j)\). En sommant \(n\) observations indépendantes : \(\mathrm{Cov}(R_i, R_j) = -\sqrt{p_i p_j}\) pour \(i \ne j\) et \(\mathrm{Var}(R_j) = 1 - p_j\). La matrice de covariance du vecteur \(\mathbf{R}\) est \(\mathbf{I} - \mathbf{u}\mathbf{u}^\top\), où \(\mathbf{u} = (\sqrt{p_1}, \dots, \sqrt{p_k})\) est de norme 1 : c'est le projecteur orthogonal sur l'hyperplan orthogonal à \(\mathbf{u}\), de rang \(k - 1\).
4. Le théorème central limite vectoriel (admis) donne la convergence en loi de \(\mathbf{R}\) vers un vecteur gaussien centré de covariance \(\mathbf{I} - \mathbf{u}\mathbf{u}^\top\), qui est la projection sur cet hyperplan d'un vecteur de \(k\) variables \(\mathcal{N}(0, 1)\) indépendantes. Dans une base orthonormée de l'hyperplan, cette projection a \(k - 1\) coordonnées \(\mathcal{N}(0, 1)\) indépendantes (invariance par rotation des vecteurs gaussiens standards, admise) : le carré de sa norme suit \(\chi^2_{k-1}\).
5. D'où vient le « \(- 1\) » : \(\sum_j \sqrt{p_j}\,R_j = \sum_j (O_j - e_j)/\sqrt{n} = 0\). Les \(k\) écarts sont liés par une relation linéaire.

### Exemples calculés du khi-deux

**Un dé.** 120 lancers ; effectifs observés des faces 1 à 6 : 15, 27, 18, 22, 14, 24. Sous \(H_0\) (dé équilibré), \(e_j = 20\). \(C = (25 + 49 + 4 + 4 + 36 + 16)/20 = @Lc_de_stat:1m@\). Avec \(k - 1 = @Lc_de_ddl@\) degrés de liberté, le quantile d'ordre 0,95 est \(@Lc_de_seuil:2m@\) : on ne rejette pas \(H_0\) au niveau 5 %. La p-value approchée vaut \(@Lc_de_p:3m@\) ; la simulation de la loi exacte de \(C\) sous \(H_0\) donne @Lc_de_p_sim:3@.

**Une loi de Poisson de paramètre inconnu.** On compte des événements sur 100 intervalles de même durée ; nombre d'intervalles avec 0, 1, 2, 3 événements et avec 4 ou plus : 20, 36, 24, 14, 6 ; @Lc_po_total@ événements au total. On estime \(\lambda\) par la moyenne, \(\hat{\lambda} = @Lc_po_lambda:2m@\). Effectifs attendus \(100\,\mathbb{P}(N = j)\) pour \(N\) de loi de Poisson de paramètre \(@Lc_po_lambda:2m@\), la dernière classe recevant le complément : \(@Lc_po_att:2m@\), tous au moins égaux à 5. \(C = @Lc_po_stat:3m@\), à comparer à \(\chi^2\) à \(5 - 1 - 1 = @Lc_po_ddl@\) degrés de liberté, de quantile d'ordre 0,95 \(@Lc_po_seuil:2m@\) : pas de rejet (p-value \(@Lc_po_p:3m@\)).

Une précision : ici \(\hat{\lambda}\) est le maximum de vraisemblance des données **brutes**, pas des effectifs regroupés que demande l'énoncé 3. Dans ce cas, la loi limite est comprise entre \(\chi^2_{k-1-s}\) et \(\chi^2_{k-1}\) (Chernoff et Lehmann, 1954, « The use of maximum likelihood estimates in chi-square tests for goodness of fit », Annals of Mathematical Statistics, 25(3), 579-586 ; résultat cité, non démontré ici). Constat par simulation dans le script, pour une intensité vraie de 1,5 et 100 intervalles : le test au niveau nominal 5 % rejette avec une fréquence de @Lc_po_niveau_ddl3_sim:3@ avec 3 degrés de liberté, et de @Lc_po_niveau_ddl4_sim:3@ avec 4.

L'expérience suivante mesure le niveau réel du test, au niveau nominal 5 %, selon la taille des effectifs attendus ; elle affiche aussi la moyenne de \(C\), qui doit valoir \(k - 1\) (énoncé 1).

```python-exec
def niveau_khi2(n, p, nb=40000):
    p = np.asarray(p, dtype=float)
    o = rng.multinomial(n, p, size=nb)
    c = ((o - n * p) ** 2 / (n * p)).sum(axis=1)
    return round(float(np.mean(c >= stats.chi2.ppf(0.95, len(p) - 1))), 4), round(float(c.mean()), 3)

print("erreur standard d'un niveau simulé proche de 0,05 :", round(float(np.sqrt(0.05 * 0.95 / 40000)), 4))
print("6 classes égales, n = 120, e_j = 20            :", niveau_khi2(120, [1 / 6] * 6))
print("6 classes égales, n = 30,  e_j = 5             :", niveau_khi2(30, [1 / 6] * 6))
print("p = (0,9 ; 0,05 ; 0,03 ; 0,02), n = 50, min e_j = 1 :", niveau_khi2(50, [0.9, 0.05, 0.03, 0.02]))
print("2 classes, p = (0,9 ; 0,1), n = 10, min e_j = 1        :", niveau_khi2(10, [0.9, 0.1]))
print("2 classes, p = (0,99 ; 0,01), n = 20, min e_j = 0,2    :", niveau_khi2(20, [0.99, 0.01]))
```

Lecture : chaque ligne donne la fréquence de rejet, puis la moyenne de \(C\). La moyenne vaut \(k - 1\) dans tous les cas, comme le dit l'énoncé 1, exact. Avec des effectifs attendus de 20 ou de 5, la fréquence de rejet est à moins de trois erreurs standard de 0,05. Avec des effectifs attendus de 1 ou moins, elle s'en écarte, vers le haut ou vers le bas selon la configuration : \(C\) ne prend alors que peu de valeurs distinctes et sa loi n'est plus proche d'une loi du khi-deux. La règle des 5 est une marge de prudence, pas une frontière.

### Kolmogorov-Smirnov : définition et énoncé

La **fonction de répartition empirique** est \(F_n(x) = \tfrac{1}{n} \sum_{i=1}^{n} I_{\{X_i \le x\}}\), la proportion d'observations au plus égales à \(x\) : une fonction en escalier, aléatoire. La statistique de Kolmogorov-Smirnov (KS) est le plus grand écart vertical entre \(F_n\) et la loi testée :

\(\displaystyle D_n = \sup_{x} \lvert F_n(x) - F_0(x) \rvert.\)

1. **Calcul.** Si \(F_0\) est continue, en notant \(x_{(1)} \le \dots \le x_{(n)}\) les observations triées et \(u_{(i)} = F_0(x_{(i)})\) : \(D_n = \max_{1 \le i \le n} \max\big(\tfrac{i}{n} - u_{(i)},\ u_{(i)} - \tfrac{i-1}{n}\big)\).
2. **Loi libre.** Si \(H_0\) est vraie et si \(F_0\) est **continue**, la loi de \(D_n\) ne dépend que de \(n\), pas de \(F_0\) : une seule table sert pour toutes les lois continues.
3. **Asymptotique** (Kolmogorov, 1933, admis). Sous les mêmes hypothèses, \(\sqrt{n}\,D_n\) converge en loi vers une variable \(K\) telle que \(\mathbb{P}(K \le x) = 1 - 2\sum_{j \ge 1} (-1)^{j-1} e^{-2 j^2 x^2}\) ; son quantile d'ordre 0,95 est \(@Lc_ks_c95:3m@\). Pour \(n\) grand, on rejette au niveau 5 % si \(D_n > @Lc_ks_c95:3m@/\sqrt{n}\) ; pour \(n\) petit, on lit la loi exacte.
4. **Ce qui casse, cas discret.** Si \(F_0\) est discrète, \(D_n\) est plus petite (au sens précis de la preuve, étape 6) que dans le cas continu : le test lu dans la table a un niveau réel inférieur au niveau affiché, donc perd de la puissance.
5. **Ce qui casse, paramètres estimés.** Si les paramètres de \(F_0\) sont estimés sur le même échantillon, la statistique \(\hat{D}_n\) obtenue n'a plus la loi de l'énoncé 2 : la loi ajustée suit les données, et l'expérience ci-dessous montre un niveau réel très inférieur au niveau affiché. Pour une famille de position et d'échelle (loi normale de moyenne et d'écart-type estimés), la loi de \(\hat{D}_n\) sous \(H_0\) ne dépend pas des vrais paramètres : on peut la tabuler par simulation. C'est la **correction de Lilliefors** (1967).

### La preuve de Kolmogorov-Smirnov

1. Énoncé 1. Sur chaque intervalle \([x_{(i)}, x_{(i+1)}[\), \(F_n\) vaut \(i/n\) et \(F_0\) croît de \(u_{(i)}\) vers \(u_{(i+1)}\). L'écart \(F_n - F_0\) y est donc maximal à gauche, où il vaut \(\tfrac{i}{n} - u_{(i)}\), et l'écart \(F_0 - F_n\) y approche \(u_{(i+1)} - \tfrac{i}{n}\) à droite (continuité de \(F_0\)). Avant \(x_{(1)}\), \(F_n = 0\) et \(F_0 \le u_{(1)}\) ; après \(x_{(n)}\), \(F_n = 1\) et \(1 - F_0 \le 1 - u_{(n)}\). Le plus grand de tous ces écarts est la formule.
2. Fonction quantile. Pour \(0 < u < 1\), soit \(Q(u) = \inf\{x : F_0(x) \ge u\}\). \(F_0\) étant croissante et continue à droite, l'ensemble \(\{x : F_0(x) \ge u\}\) est l'intervalle \([Q(u), +\infty[\). D'où l'équivalence (E) : \(Q(u) \le x\) si et seulement si \(u \le F_0(x)\).
3. Soit \(U\) de loi uniforme sur \([0, 1]\). Par (E), \(\mathbb{P}(Q(U) \le x) = \mathbb{P}(U \le F_0(x)) = F_0(x)\) : \(Q(U)\) a pour fonction de répartition \(F_0\). La loi de \(D_n\) ne dépendant que de la loi de l'échantillon, on peut la calculer en prenant \(X_i = Q(U_i)\), avec \(U_1, \dots, U_n\) indépendantes et uniformes.
4. Par (E) encore, \(I_{\{X_i \le x\}} = I_{\{U_i \le F_0(x)\}}\) pour tout \(x\). Donc \(F_n(x) = G_n(F_0(x))\), où \(G_n(u) = \tfrac{1}{n} \sum_i I_{\{U_i \le u\}}\) est la fonction de répartition empirique des \(U_i\). Ainsi \(D_n = \sup_x \lvert G_n(F_0(x)) - F_0(x) \rvert = \sup_{u \in \mathcal{A}} \lvert G_n(u) - u \rvert\), où \(\mathcal{A}\) est l'ensemble des valeurs prises par \(F_0\).
5. Si \(F_0\) est continue, de limites 0 et 1, elle prend toutes les valeurs de \(]0, 1[\) (théorème des valeurs intermédiaires). En \(u = 0\) et \(u = 1\), l'écart \(\lvert G_n(u) - u \rvert\) est nul avec probabilité 1. Donc \(D_n = \sup_{0 < u < 1} \lvert G_n(u) - u \rvert\), fonction des seules \(U_i\) : sa loi ne dépend pas de \(F_0\). C'est l'énoncé 2.
6. Énoncé 4. Si \(F_0\) est discrète, \(\mathcal{A}\) est un ensemble au plus dénombrable (les sommes cumulées des probabilités) : le sup de l'étape 4 porte sur une partie seulement de \([0, 1]\) et il est, pour les mêmes \(U_i\), inférieur ou égal au sup sur \(]0, 1[\). Donc \(\mathbb{P}(D_n > d)\) est au plus la probabilité lue dans la table du cas continu.
7. Énoncé 5, invariance. Soit \(X_i = \mu + \sigma Y_i\), avec \(Y_i\) de loi \(\mathcal{N}(0, 1)\). La moyenne et l'écart-type empiriques vérifient \(\bar{X} = \mu + \sigma \bar{Y}\) et \(S_X = \sigma S_Y\). En posant \(x = \mu + \sigma y\) dans \(\hat{D}_n = \sup_x \lvert F_n(x) - \Phi((x - \bar{X})/S_X) \rvert\) : la proportion des \(X_i \le x\) est celle des \(Y_i \le y\), et \((x - \bar{X})/S_X = (y - \bar{Y})/S_Y\). \(\hat{D}_n\) est donc une fonction des seuls \(Y_i\). Que sa loi soit concentrée sur des valeurs plus petites que celle de \(D_n\) est constaté par simulation ci-dessous, pas démontré ici.

### Exemple calculé de Kolmogorov-Smirnov

Cinq durées, en heures : 0,3 ; 0,9 ; 1,5 ; 2,4 ; 5,1. \(H_0\) : loi exponentielle de moyenne 2, \(F_0(x) = 1 - e^{-x/2}\).

1. \(u_{(i)} = F_0(x_{(i)})\) : \(@Lc_ks_u:3m@\).
2. \(\tfrac{i}{5} - u_{(i)}\) : \(@Lc_ks_haut:3m@\).
3. \(u_{(i)} - \tfrac{i-1}{5}\) : \(@Lc_ks_bas:3m@\).
4. \(D_5 = @Lc_ks_d:3m@\), atteint juste avant le deuxième saut de \(F_n\) : oublier la ligne 3 donnerait une valeur fausse. Le quantile exact d'ordre 0,95 de \(D_5\) est \(@Lc_ks_seuil5:3m@\) : pas de rejet (p-value \(@Lc_ks_p:3m@\)).

L'expérience suivante mesure, pour \(n = 50\) et un niveau nominal de 5 %, la fréquence de rejet sous \(H_0\) dans quatre situations, puis le seuil de Lilliefors obtenu par simulation.

```python-exec
def d_ks(u):
    # statistique de KS de chaque ligne, à partir de u = F0(x) (formule de l'énoncé 1)
    u = np.sort(u, axis=1)
    n = u.shape[1]
    i = np.arange(1, n + 1)
    return np.maximum((i / n - u).max(axis=1), (u - (i - 1) / n).max(axis=1))

n, nb = 50, 20000
seuil = stats.kstwo.ppf(0.95, n)            # quantile exact de D_n, loi continue entièrement spécifiée
x = rng.normal(10.0, 3.0, (nb, n))
d_normale = d_ks(stats.norm.cdf(x, 10.0, 3.0))
d_expo = d_ks(stats.expon.cdf(rng.exponential(2.0, (nb, n)), scale=2.0))
z = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, ddof=1, keepdims=True)
d_estime = d_ks(stats.norm.cdf(z))          # moyenne et écart-type estimés sur l'échantillon
k = rng.poisson(3.0, (nb, n))               # loi discrète : sup de |F_n - F_0| sur le support
support = np.arange(0, 40)
d_discret = np.abs((k[:, :, None] <= support).mean(axis=1) - stats.poisson.cdf(support, 3.0)).max(axis=1)

print("seuil de la table pour n = 50 :", round(float(seuil), 4))
for nom, d in [("normale, paramètres vrais   ", d_normale), ("exponentielle, paramètre vrai", d_expo),
               ("normale, paramètres estimés ", d_estime), ("Poisson (loi discrète)      ", d_discret)]:
    print(nom, ": fréquence de rejet", round(float(np.mean(d > seuil)), 4), "; moyenne de la statistique", round(float(d.mean()), 4))
seuil_lilliefors = np.quantile(d_estime, 0.95)
x2 = rng.normal(-2.0, 0.5, (nb, n))         # autres paramètres vrais : même loi de la statistique
z2 = (x2 - x2.mean(axis=1, keepdims=True)) / x2.std(axis=1, ddof=1, keepdims=True)
print("seuil simulé avec paramètres estimés :", round(float(seuil_lilliefors), 4))
print("fréquence de rejet avec ce seuil, autres paramètres vrais :", round(float(np.mean(d_ks(stats.norm.cdf(z2)) > seuil_lilliefors)), 4))
```

Lecture : les deux premières lignes illustrent l'énoncé 2 (même loi de \(D_n\), même fréquence de rejet proche de 0,05, pour deux lois continues différentes) ; les deux suivantes illustrent les énoncés 5 et 4 (statistique plus petite en moyenne, test qui ne rejette presque plus) ; les deux dernières montrent le principe de la correction de Lilliefors : un seuil propre, plus petit, valable quels que soient les vrais paramètres.

### Anderson-Darling, Shapiro-Wilk, Jarque-Bera

**Anderson-Darling (AD).** Avec les notations de la preuve précédente (\(U_i = F_0(X_i)\), \(G_n\) leur fonction de répartition empirique), la statistique est

\(\displaystyle A^2 = n \int_0^1 \frac{(G_n(u) - u)^2}{u\,(1 - u)}\,du.\)

Pourquoi ce poids : sous \(H_0\), \(n\,G_n(u)\) suit la loi binomiale de paramètres \(n\) et \(u\), donc \(\mathrm{Var}(G_n(u)) = u(1 - u)/n\), petite près de 0 et de 1. AD divise chaque écart au carré par sa variance : un écart dans une **queue** compte autant qu'un écart au centre. KS prend l'écart brut maximal, dominé par le centre, où la variance \(u(1-u)/n\) est la plus grande. AD est donc plus sensible que KS aux désaccords dans les queues. Formule de calcul (vérifiée numériquement par le script contre l'intégrale, non démontrée ici) : \(A^2 = -n - \tfrac{1}{n} \sum_{i=1}^{n} (2i - 1)\big(\ln u_{(i)} + \ln(1 - u_{(n+1-i)})\big)\) ; pour les cinq durées de l'exemple précédent, \(A^2 = @Lc_ad_stat:3m@\). Comme \(A^2\) ne dépend que des \(U_i\), sa loi sous \(H_0\) est libre pour une loi continue entièrement spécifiée (même preuve que pour KS). Avec des paramètres estimés, les seuils dépendent de la famille testée. Sans le poids \(1/(u(1-u))\), la même intégrale est la statistique de Cramér-von Mises.

**Shapiro-Wilk**, en une phrase : test réservé à la normalité (paramètres inconnus), dont la statistique \(W \le 1\) compare les observations triées aux valeurs attendues des statistiques d'ordre d'un échantillon normal, et qui rejette quand \(W\) est petit.

**Jarque-Bera** : test de normalité fondé sur la partie 1. Avec l'asymétrie empirique \(\hat{\gamma}_1\) et le kurtosis empirique \(\hat{\kappa}\), la statistique \(\tfrac{n}{6}\big(\hat{\gamma}_1^2 + \tfrac{(\hat{\kappa} - 3)^2}{4}\big)\) converge en loi vers \(\chi^2_2\) sous l'hypothèse de normalité (admis) ; il ne voit que les écarts d'asymétrie et de kurtosis.

L'expérience suivante compare, pour \(n = 50\) et un niveau nominal de 5 %, les fréquences de rejet de quatre tests de normalité à paramètres estimés : sous \(H_0\) (données normales), puis pour des données de Student à 4 degrés de liberté (symétriques, queues lourdes) et exponentielles (asymétriques). Le code montre aussi ce que rend `stats.anderson` : la statistique et des seuils propres à la famille testée.

```python-exec
from statsmodels.stats.diagnostic import lilliefors, normal_ad

def frequences_de_rejet(tirer, n=50, nb=2000):
    rejets = np.zeros(4)
    for _ in range(nb):
        x = tirer(n)
        rejets += [lilliefors(x, dist="norm")[1] <= 0.05, normal_ad(x)[1] <= 0.05,
                   stats.shapiro(x)[1] <= 0.05, stats.jarque_bera(x)[1] <= 0.05]
    return (rejets / nb).round(3)

print("erreur standard maximale d'une fréquence sur 2000 échantillons :", round(float(np.sqrt(0.25 / 2000)), 4))
print("ordre des colonnes : Lilliefors, Anderson-Darling, Shapiro-Wilk, Jarque-Bera")
print("données normales (niveau réel) :", frequences_de_rejet(lambda n: rng.normal(0.0, 1.0, n)))
print("Student à 4 degrés de liberté  :", frequences_de_rejet(lambda n: rng.standard_t(4, n)))
print("exponentielle                  :", frequences_de_rejet(lambda n: rng.exponential(1.0, n)))
r = stats.anderson(np.random.default_rng(11).normal(0.0, 1.0, 50), dist="norm")   # un échantillon normal
print("stats.anderson :", round(float(r.statistic), 3), "; seuils", r.critical_values, "aux niveaux (%)", r.significance_level)
```

Lecture : la première ligne donne les niveaux réels, qui doivent être proches de 0,05 ou en dessous ; les deux suivantes des puissances. Sur la dernière ligne, la statistique de l'échantillon normal est sous tous les seuils : pas de rejet. Face aux queues lourdes de la loi de Student, le test de Lilliefors, fondé sur l'écart maximal de KS, rejette moins souvent que celui d'Anderson-Darling, qui pondère les queues. Ces fréquences sont des constats sur cette expérience (2000 échantillons par ligne), pas des résultats généraux : la puissance d'un test dépend de l'alternative et de \(n\).

### Quel test pour quelle loi

| Test | Type de loi sous \(H_0\) | Paramètres | Conditions et limites |
|---|---|---|---|
| khi-deux | discrète (binomiale, Poisson, dé) ; continue seulement après découpage en classes | fixés : \(k - 1\) degrés de liberté ; estimés : \(k - 1 - s\) | loi asymptotique ; effectifs attendus d'au moins 5 (règle d'usage) ; ignore l'ordre des classes |
| Kolmogorov-Smirnov | continue | entièrement fixés | loi exacte et libre pour tout \(n\) ; surtout sensible au centre ; niveau réel trop bas si la loi est discrète ou si les paramètres sont estimés |
| Lilliefors | normale ou exponentielle | estimés | statistique de KS avec des seuils propres à la famille |
| Anderson-Darling | continue | fixés ; estimés avec des seuils propres à chaque famille | pondère les queues |
| Shapiro-Wilk | normale seulement | estimés | réservé à la normalité |
| Jarque-Bera | normale seulement | estimés | loi asymptotique ; ne voit que l'asymétrie et le kurtosis |

### Pièges

1. Ne pas rejeter n'est pas prouver \(H_0\) : avec peu de données, la puissance est faible (séance de mathématiques 3, partie 4).
2. Degrés de liberté du khi-deux : \(k - 1 - s\). Oublier le \(- s\) donne des p-values trop grandes.
3. Lire la table de Kolmogorov après avoir estimé les paramètres sur l'échantillon, ou pour une loi discrète.
4. Avec \(n\) très grand, tout test rejette un écart minuscule : une p-value ne mesure pas la taille de l'écart.
5. Le khi-deux appliqué à une loi continue dépend du découpage en classes choisi.

[[controle:adequation]]

## Partie 4 : densités conjointes

### Le problème

Un couple \((X, Y)\) est décrit par une fonction de deux variables. Il faut savoir en extraire, vite et sans erreur de bornes : la constante de normalisation, la loi de chaque variable, la loi de l'une sachant la valeur de l'autre, une probabilité « sachant \(X = x\) », et dire si les deux variables sont indépendantes.

### Définitions

Le couple \((X, Y)\) admet la **densité conjointe** \(f\), fonction positive de deux variables, si \(\mathbb{P}((X, Y) \in A) = \iint_A f(x, y)\,dx\,dy\) pour toute partie \(A\) du plan (mesurable) ; en particulier \(\iint f = 1\).

- **Densité marginale** de \(X\) : \(f_X(x) = \int f(x, y)\,dy\). C'est la densité de \(X\), car \(\mathbb{P}(X \le t) = \int_{-\infty}^{t} \big(\int f(x, y)\,dy\big)\,dx\) (théorème de Fubini, admis). De même \(f_Y(y) = \int f(x, y)\,dx\).
- **Densité conditionnelle** de \(Y\) sachant \(X = x\), pour \(f_X(x) > 0\) : \(f_{Y \mid X}(y \mid x) = f(x, y)/f_X(x)\). C'est une densité en \(y\) : son intégrale en \(y\) vaut 1.
- \(\mathbb{P}(Y \in B \mid X = x) = \int_B f_{Y \mid X}(y \mid x)\,dy\) et \(\mathbb{E}[Y \mid X = x] = \int y\,f_{Y \mid X}(y \mid x)\,dy\).
- \(X\) et \(Y\) sont **indépendantes** si et seulement si \(f(x, y) = f_X(x)\,f_Y(y)\) (sauf sur un ensemble d'aire nulle).

### L'intuition

\(\{X = x\}\) est de probabilité nulle : la définition \(\mathbb{P}(A \cap B)/\mathbb{P}(B)\) ne s'applique pas. La densité conditionnelle est la limite du conditionnement par une bande étroite : pour \(h > 0\) petit, \(\mathbb{P}(Y \in B \mid x \le X \le x + h) = \dfrac{\int_x^{x+h} \int_B f(s, y)\,dy\,ds}{\int_x^{x+h} f_X(s)\,ds}\), et si \(f\) est continue, numérateur et dénominateur valent \(h\) fois leur valeur en \(s = x\), à un terme négligeable devant \(h\) près : la limite est \(\int_B f(x, y)\,dy / f_X(x)\). Géométriquement : on coupe la surface \(f\) par la droite verticale d'abscisse \(x\), et l'on renormalise la tranche pour que son aire vaille 1.

### L'énoncé

1. **(D1) Normalisation.** Si \(f = c\,g\) avec \(g \ge 0\) connue, \(c = 1/\iint g\).
2. **(D2) Probabilités totales.** \(\mathbb{P}(Y \in B) = \int \mathbb{P}(Y \in B \mid X = x)\,f_X(x)\,dx\) et \(\mathbb{E}[Y] = \int \mathbb{E}[Y \mid X = x]\,f_X(x)\,dx\).
3. **(D3) Bayes pour les densités.** \(f_{X \mid Y}(x \mid y) = \dfrac{f_{Y \mid X}(y \mid x)\,f_X(x)}{\int f_{Y \mid X}(y \mid s)\,f_X(s)\,ds}\).
4. **(D4) Factorisation sur un pavé.** Si \(f(x, y) = g(x)\,h(y)\) sur un pavé \(\mathcal{I} \times \mathcal{J}\) (produit de deux intervalles) et \(f = 0\) ailleurs, alors \(X\) et \(Y\) sont indépendantes, de densités \(g/\int_{\mathcal{I}} g\) et \(h/\int_{\mathcal{J}} h\). Si l'ensemble où \(f > 0\) n'est pas un produit de deux parties de la droite, à un ensemble d'aire nulle près (un triangle, par exemple), \(X\) et \(Y\) ne sont pas indépendantes, même si la formule de \(f\) est un produit.
5. **(D5) Reconnaître une loi.** À \(x\) fixé, \(y \mapsto f(x, y)\) est proportionnelle à \(f_{Y \mid X}(\cdot \mid x)\) : on identifie la loi conditionnelle en ne gardant que les facteurs qui contiennent \(y\).

### La preuve

1. (D2) : \(\int \mathbb{P}(Y \in B \mid X = x)\,f_X(x)\,dx = \int \big(\int_B f(x, y)\,dy\big)\,dx = \mathbb{P}(Y \in B)\) (Fubini) ; les \(x\) où \(f_X(x) = 0\) ne contribuent pas. Même calcul avec \(y\,f(x, y)\) pour l'espérance.
2. (D3) : \(f(x, y) = f_{Y \mid X}(y \mid x)\,f_X(x)\) par définition, et aussi \(f(x, y) = f_{X \mid Y}(x \mid y)\,f_Y(y)\), avec \(f_Y(y) = \int f(s, y)\,ds\).
3. (D4), sens direct : pour \(x \in \mathcal{I}\), \(f_X(x) = g(x) \int_{\mathcal{J}} h\) ; pour \(y \in \mathcal{J}\), \(f_Y(y) = h(y) \int_{\mathcal{I}} g\) ; et \(1 = \iint f = \int_{\mathcal{I}} g \times \int_{\mathcal{J}} h\). Donc \(f_X(x)\,f_Y(y) = g(x)\,h(y) = f(x, y)\) sur le pavé, et les deux membres sont nuls ailleurs.
4. (D4), second point : si \(X\) et \(Y\) sont indépendantes, \(f = f_X f_Y\) est strictement positive exactement sur le produit de \(\{f_X > 0\}\) par \(\{f_Y > 0\}\). Si l'on trouve une zone d'aire non nulle où \(f = 0\) alors que \(f_X(x) > 0\) et \(f_Y(y) > 0\), il n'y a pas indépendance.
5. (D5) : dans \(f_{Y \mid X}(y \mid x) = f(x, y)/f_X(x)\), le dénominateur ne dépend pas de \(y\).

### Exemple A, calcul complet

\(f(x, y) = c\,xy\) pour \(0 < x < y < 1\), et 0 ailleurs.

1. Normalisation : \(\int_0^1 \int_0^y xy\,dx\,dy = \int_0^1 \tfrac{y^3}{2}\,dy = \tfrac18\), donc \(c = @Ld_A_c@\).
2. Marginales : \(f_X(x) = \int_x^1 8xy\,dy = 4x(1 - x^2)\) pour \(0 < x < 1\) ; \(f_Y(y) = \int_0^y 8xy\,dx = 4y^3\) pour \(0 < y < 1\). Les bornes viennent du triangle : à \(x\) fixé, \(y\) va de \(x\) à 1.
3. Conditionnelle : \(f_{Y \mid X}(y \mid x) = \dfrac{8xy}{4x(1 - x^2)} = \dfrac{2y}{1 - x^2}\) pour \(x < y < 1\).
4. Probabilité sachant \(X = 1/2\) : \(f_X(1/2) = \tfrac{@Ld_A_fx_demi_num@}{@Ld_A_fx_demi_den@}\), et \(\mathbb{P}(Y > 3/4 \mid X = 1/2) = \int_{3/4}^{1} \tfrac{2y}{3/4}\,dy = \tfrac{1 - 9/16}{3/4} = \tfrac{@Ld_A_p_num@}{@Ld_A_p_den@}\) (simulation dans une bande étroite autour de \(x = 1/2\) : @Ld_A_p_mc:3@). Sans condition, \(\mathbb{P}(Y > 3/4) = 1 - (3/4)^4 = \tfrac{@Ld_A_p_marg_num@}{@Ld_A_p_marg_den@} \approx @Ld_A_p_marg:3m@\). Enfin \(\mathbb{E}[Y \mid X = 1/2] = \tfrac{@Ld_A_e_num@}{@Ld_A_e_den@}\).
5. Indépendance : non. La formule \(8xy\) est un produit, mais le domaine est un triangle : au point \((3/4, 1/4)\), \(f = 0\) alors que \(f_X(3/4)\,f_Y(1/4) = \tfrac{@Ld_A_prod_num@}{@Ld_A_prod_den@} > 0\).

### Exemple B, reconnaître une loi usuelle

\(f(x, y) = x\,e^{-x(1 + y)}\) pour \(x > 0\) et \(y > 0\).

1. À \(x\) fixé, le seul facteur qui contient \(y\) est \(e^{-xy}\) : c'est le noyau de la loi exponentielle de paramètre \(x\). Donc \(f_{Y \mid X}(y \mid x) = x\,e^{-xy}\), et \(f_X(x) = f(x, y)/f_{Y \mid X}(y \mid x) = e^{-x}\) : \(X\) suit la loi exponentielle de paramètre 1.
2. \(\mathbb{P}(Y > 1 \mid X = 2) = e^{-2} \approx @Ld_B_p:3m@\) (simulation : @Ld_B_p_mc:3@), alors que \(\mathbb{P}(Y > 1) = @Ld_B_p_marg:1m@\).
3. À \(y\) fixé, les facteurs qui contiennent \(x\) sont \(x\,e^{-(1+y)x}\) : noyau de la loi gamma de forme 2 et de paramètre \(1 + y\) (partie 1). Donc \(f_{X \mid Y}(x \mid y) = (1 + y)^2\,x\,e^{-(1+y)x}\), \(f_Y(y) = 1/(1 + y)^2\), et \(\mathbb{E}[X \mid Y = y] = 2/(1 + y)\), soit @Ld_B_e@ pour \(y = 1\).

Lecture bayésienne : \(X\) est un paramètre inconnu (l'intensité), de loi a priori exponentielle ; sachant \(X = x\), on observe une durée \(Y\) exponentielle de paramètre \(x\) ; la loi de \(X\) sachant l'observation est une loi gamma. La partie 5 fait le même calcul pour un comptage.

### Cas limites

**Conditionner par un événement de probabilité nulle dépend de la variable qui le décrit.** \(X\) et \(Y\) sont indépendantes, uniformes sur \([0, 1]\). « Sachant \(X = Y\), quelle est l'espérance de \(X\) ? » n'a pas de réponse sans préciser la variable conditionnante.

- Sachant \(X - Y = 0\) : la densité du couple \((X, X - Y)\) au point \((x, 0)\) vaut 1 pour \(0 < x < 1\) ; la loi conditionnelle de \(X\) est uniforme, d'espérance \(\tfrac{@Ld_bk_diff_num@}{@Ld_bk_diff_den@}\).
- Sachant \(X/Y = 1\) : avec \(r = x/y\), soit \(y = x/r\), la dérivée de \(y\) par rapport à \(r\) vaut \(-x/r^2\) ; la densité du couple \((X, X/Y)\) au point \((x, 1)\) vaut \(x\) pour \(0 < x < 1\) (changement de variables, admis). La loi conditionnelle de \(X\) a la densité \(2x\), d'espérance \(\tfrac{@Ld_bk_ratio_num@}{@Ld_bk_ratio_den@}\).

La bande \(\lvert X - Y \rvert < h\) a une largeur constante ; la bande \(\lvert X/Y - 1 \rvert < h\) a une largeur proportionnelle à \(x\), donc favorise les grands \(x\). Simulation avec \(h = 0{,}002\) : @Ld_bk_diff_mc:3@ et @Ld_bk_ratio_mc:3@.

**Couple mixte.** Si \(X\) est à densité et \(N\) discrète, les mêmes formules valent en remplaçant l'intégrale en \(n\) par une somme ; et \(\{N = k\}\) ayant une probabilité non nulle, la loi de \(X\) sachant \(N = k\) relève de la définition élémentaire (partie 5).

### Pièges

1. Bornes : à \(x\) fixé, les bornes de \(y\) dépendent de \(x\) dès que le domaine n'est pas un pavé.
2. Conclure à l'indépendance parce que la formule est un produit, sans regarder le domaine.
3. Oublier de diviser par \(f_X(x)\) : \(\int_B f(x, y)\,dy\) n'est pas une probabilité conditionnelle.
4. Une densité conditionnelle est une densité **en \(y\)**, de paramètre \(x\) : vérifier que son intégrale en \(y\) vaut 1.
5. « Sachant \(X = Y\) » sans dire comment l'événement est approché.

## Partie 5 : intensité d'une loi de Poisson, a priori et a posteriori

### Le problème

Des événements arrivent selon un processus de Poisson (séance de mathématiques 2, partie 4) dont l'intensité est inconnue. On observe \(k\) événements pendant une durée \(t\). Quelle intensité est la plus probable, parmi une liste de valeurs candidates ou parmi tous les réels positifs, compte tenu de ce que l'on savait avant d'observer ?

### Définitions et modèle

- \(\Lambda\) est l'intensité, **aléatoire** parce qu'inconnue : un nombre d'événements par unité de temps. Ses valeurs possibles sont notées \(\lambda > 0\).
- \(N\) est le nombre d'événements observés pendant la durée \(t > 0\). Modèle : sachant \(\Lambda = \lambda\), \(N\) suit la loi de Poisson de paramètre \(\lambda t\). La **vraisemblance** de \(\lambda\), une fois \(k\) observé, est \(L(\lambda) = \mathbb{P}(N = k \mid \Lambda = \lambda) = e^{-\lambda t} (\lambda t)^k/k!\).
- La **loi a priori** est la loi de \(\Lambda\) avant l'observation : des poids \(\pi_1, \dots, \pi_m\) de somme 1 sur une liste \(\lambda_1, \dots, \lambda_m\), ou une densité \(\pi(\lambda)\) sur \(]0, +\infty[\).
- La **loi a posteriori** est la loi de \(\Lambda\) sachant \(N = k\).
- Le **maximum a posteriori** (MAP) est la valeur de \(\lambda\) qui maximise la loi a posteriori (le poids, ou la densité). Le **maximum de vraisemblance** (EMV, séance de mathématiques 3, partie 2) maximise \(L\) : comme \(\ln L(\lambda) = k \ln \lambda - \lambda t\) à une constante près, de dérivée \(k/\lambda - t\), c'est \(\hat{\lambda} = k/t\).

### L'énoncé

1. **(B1) Liste finie.** \(\mathbb{P}(\Lambda = \lambda_i \mid N = k) = \dfrac{\pi_i\,\lambda_i^k\,e^{-\lambda_i t}}{\sum_{j=1}^{m} \pi_j\,\lambda_j^k\,e^{-\lambda_j t}}\).
2. **(B2) Densité.** La loi de \(\Lambda\) sachant \(N = k\) a la densité \(\pi(\lambda \mid k) = \dfrac{\pi(\lambda)\,\lambda^k e^{-\lambda t}}{\int_0^\infty \pi(s)\,s^k e^{-s t}\,ds}\).
3. **(B3) Conjugaison gamma-Poisson.** Si \(\Lambda\) suit a priori la loi gamma de forme \(a > 0\) et de paramètre \(b > 0\) (densité \(b^a \lambda^{a-1} e^{-b\lambda}/\Gamma(a)\), moyenne \(a/b\), partie 1), alors sachant \(N = k\) elle suit la loi gamma de forme \(a + k\) et de paramètre \(b + t\). Sa moyenne a posteriori est \(\dfrac{a + k}{b + t} = \dfrac{b}{b + t} \cdot \dfrac{a}{b} + \dfrac{t}{b + t} \cdot \dfrac{k}{t}\), et son MAP est \(\dfrac{a + k - 1}{b + t}\) si \(a + k \ge 1\).
4. **(B4) Deux candidates.** Avec des poids a priori égaux, \(\lambda_j > \lambda_i\) est plus probable a posteriori que \(\lambda_i\) si et seulement si \(\dfrac{k}{t} > \dfrac{\lambda_j - \lambda_i}{\ln \lambda_j - \ln \lambda_i}\). Ce seuil, la moyenne logarithmique des deux candidates, n'est pas leur milieu.
5. **(B5) A priori impropre en \(1/\lambda\).** Avec \(\pi(\lambda) = 1/\lambda\), la formule (B2) donne la loi gamma de forme \(k\) et de paramètre \(t\) si \(k \ge 1\), et rien si \(k = 0\).

### La preuve

1. (B1) : formule de Bayes pour la partition des événements \(\{\Lambda = \lambda_i\}\) (séance de mathématiques 2, résultat (C2)) ; le facteur \(t^k/k!\), commun au numérateur et à tous les termes du dénominateur, se simplifie.
2. (B2) : par les probabilités totales, \(\mathbb{P}(\Lambda \in A, N = k) = \int_A \pi(\lambda)\,L(\lambda)\,d\lambda\). Comme \(\mathbb{P}(N = k) > 0\), la définition élémentaire donne \(\mathbb{P}(\Lambda \in A \mid N = k) = \int_A \pi L \big/ \int_0^\infty \pi L\), et \(t^k/k!\) se simplifie.
3. (B3) : \(\pi(\lambda)\,\lambda^k e^{-\lambda t}\) est proportionnel à \(\lambda^{a-1} e^{-b\lambda} \cdot \lambda^k e^{-\lambda t} = \lambda^{a + k - 1} e^{-(b + t)\lambda}\), noyau de la loi gamma de forme \(a + k\) et de paramètre \(b + t\) ; la constante est imposée par l'intégrale égale à 1. Moyenne et mode : partie 1. Enfin \(\tfrac{b}{b+t} \cdot \tfrac{a}{b} + \tfrac{t}{b+t} \cdot \tfrac{k}{t} = \tfrac{a + k}{b + t}\).
4. (B4) : le rapport des deux poids a posteriori vaut \((\lambda_j/\lambda_i)^k e^{-(\lambda_j - \lambda_i)t}\) ; il dépasse 1 si et seulement si \(k\,(\ln \lambda_j - \ln \lambda_i) > (\lambda_j - \lambda_i)\,t\).
5. (B5) : \((1/\lambda) \cdot \lambda^k e^{-\lambda t} = \lambda^{k-1} e^{-\lambda t}\), noyau de la loi gamma de forme \(k\) et de paramètre \(t\) si \(k \ge 1\). Si \(k = 0\), \(e^{-\lambda t}/\lambda\) n'est pas intégrable près de 0 : il n'y a pas de loi a posteriori.

### Ce que veut dire un a priori impropre

\(\int_0^\infty d\lambda/\lambda\) diverge, en 0 comme à l'infini : \(1/\lambda\) n'est la densité d'aucune loi. C'est une convention, qui n'a de sens que par la loi a posteriori qu'elle produit, quand celle-ci est une vraie loi (ici dès que \(k \ge 1\)). Trois lectures :

- c'est la limite de la loi gamma de forme \(a\) et de paramètre \(b\) quand \(a\) et \(b\) tendent vers 0 : (B3) se lit « l'a priori vaut \(a\) événements fictifs observés pendant une durée fictive \(b\) », et ici il n'apporte aucun événement fictif ;
- c'est la densité constante sur \(\ln \lambda\), puisque \(d\lambda/\lambda = d(\ln \lambda)\) : aucun ordre de grandeur de \(\lambda\) n'est favorisé, et la forme \(1/\lambda\) est conservée si l'on change d'unité de temps (\(\lambda\) multiplié par une constante) ;
- sur une **liste finie**, des poids proportionnels à \(1/\lambda_i\) forment une loi a priori ordinaire une fois divisés par leur somme.

| A priori | A posteriori, \(k\) événements en une durée \(t\) | Moyenne a posteriori | MAP |
|---|---|---|---|
| gamma de forme \(a\), de paramètre \(b\) | gamma de forme \(a + k\), de paramètre \(b + t\) | \((a + k)/(b + t)\) | \((a + k - 1)/(b + t)\) |
| \(1/\lambda\) (impropre), \(k \ge 1\) | gamma de forme \(k\), de paramètre \(t\) | \(k/t\) | \((k - 1)/t\) |
| constant (impropre) | gamma de forme \(k + 1\), de paramètre \(t\) | \((k + 1)/t\) | \(k/t\) |

L'EMV \(k/t\) coïncide avec le MAP pour l'a priori constant, et avec la moyenne a posteriori pour l'a priori en \(1/\lambda\).

### Exemples calculés

**Liste, deux a priori.** Intensités candidates 1, 2, 3 et 5 événements par heure ; on observe 5 événements en 2 heures (EMV \(@Le_emv:1m@\)).

- A priori uniforme sur la liste : poids \(\lambda^5 e^{-2\lambda}\), soit \(@Le_noyau_unif:3m@\) ; probabilités a posteriori \(@Le_post_unif:3m@\). Le MAP est @Le_map_unif@. Les candidates 2 et 3 sont à la même distance de 2,5, mais leur moyenne logarithmique \(1/\ln(3/2) \approx @Le_logmoy_23:3m@\) est sous 2,5 : (B4) donne l'avantage à 3.
- A priori proportionnel à \(1/\lambda\) (poids \(@Le_prior_inv:3m@\)) : poids \(\lambda^4 e^{-2\lambda}\), soit \(@Le_noyau_inv:3m@\) ; probabilités a posteriori \(@Le_post_inv:3m@\). Le MAP est @Le_map_inv@ : l'a priori a changé la réponse.

**Le seuil n'est pas le milieu.** Candidates 2 et 6, poids égaux, 19 événements en 5 heures : le taux observé \(@Le_lm_taux:1m@\) est sous le milieu @Le_lm_milieu@, donc plus proche de 2. Mais la moyenne logarithmique vaut \(4/\ln 3 \approx @Le_lm_seuil:3m@ < @Le_lm_taux:1m@\) : 6 est la plus probable, avec un rapport des poids a posteriori de \(3^{19} e^{-20} \approx @Le_lm_rapport:2m@\).

**Conjugaison.** A priori gamma de forme 2 et de paramètre 1 par heure (moyenne @Le_g_prior_moy@, mode @Le_g_prior_mode@) ; on observe 7 événements en 3 heures. A posteriori : gamma de forme @Le_g_alpha@ et de paramètre @Le_g_beta@. Moyenne \(\tfrac{@Le_g_moy_num@}{@Le_g_moy_den@} = \tfrac14 \times 2 + \tfrac34 \times \tfrac73\) (simulation : @Le_g_moy_mc:3@) ; MAP \(8/4 = @Le_g_map@\) ; EMV \(7/3 \approx @Le_g_emv:3m@\). Trois estimations, trois nombres différents. Avec l'a priori en \(1/\lambda\) : gamma de forme 7 et de paramètre 3, moyenne \(\tfrac{@Le_imp_moy_num@}{@Le_imp_moy_den@}\), MAP @Le_imp_map@. Avec l'a priori constant : gamma de forme 8 et de paramètre 3, MAP \(\tfrac{@Le_plat_map_num@}{@Le_plat_map_den@}\), moyenne \(\tfrac{@Le_plat_moy_num@}{@Le_plat_moy_den@}\).

### Cas limites

- **Beaucoup de données.** Si \(t \to \infty\) avec \(k/t\) fixé, le poids \(b/(b + t)\) de l'a priori dans (B3) tend vers 0 et la variance a posteriori \((a + k)/(b + t)^2\) aussi : la loi a posteriori se concentre en \(k/t\), quel que soit l'a priori gamma.
- **Plusieurs fenêtres.** Pour des comptages \(k_1, \dots, k_r\) sur des durées disjointes \(t_1, \dots, t_r\), indépendants sachant \(\Lambda\), la vraisemblance est le produit des vraisemblances, proportionnel à \(\lambda^{k_1 + \dots + k_r} e^{-\lambda (t_1 + \dots + t_r)}\) : seuls comptent le nombre total d'événements et la durée totale.
- **Observer une durée.** Si l'on observe l'instant \(s\) du \(k\)-ième événement, la vraisemblance est la densité de cet instant, \(\lambda^k s^{k-1} e^{-\lambda s}/(k-1)!\) (dérivée en \(s\) de \(1 - \sum_{j < k} e^{-\lambda s}(\lambda s)^j/j!\), résultat (P2) de la séance de mathématiques 2 : la somme se télescope). Même noyau \(\lambda^k e^{-\lambda s}\) : même loi a posteriori que pour \(k\) événements comptés en une durée \(s\).
- **Le MAP dépend de la paramétrisation.** Si \(\Lambda\) a la densité \(p\), la durée moyenne entre événements \(1/\Lambda\) a la densité \(p(1/\theta)/\theta^2\) en \(\theta\). Pour la loi gamma de forme @Le_g_alpha@ et de paramètre @Le_g_beta@, le mode de \(1/\Lambda\) est \(4/10 = @Le_theta_mode:1m@\), alors que l'inverse du MAP de \(\Lambda\) est \(@Le_theta_inv_map:1m@\). Sur une liste finie, ce problème n'existe pas : les poids ne changent pas quand on renomme les valeurs.

### Pièges

1. Oublier l'a priori : choisir la candidate de plus grande vraisemblance, c'est supposer des poids égaux.
2. Choisir la candidate la plus proche de \(k/t\) en distance : le seuil entre deux candidates est leur moyenne logarithmique.
3. Le paramètre de la loi de Poisson est \(\lambda t\), intensité fois durée, en unités cohérentes.
4. MAP, moyenne a posteriori et EMV sont trois nombres différents ; lire lequel est demandé.
5. Avec l'a priori en \(1/\lambda\) et \(k = 0\), il n'y a pas de loi a posteriori.

[[controle:posterieur]]

## Partie 6 : bâton coupé au hasard

### Le problème

Un bâton de longueur 1 est coupé en \(n\) morceaux par \(n - 1\) points tirés indépendamment et uniformément. L'outil 3 de la séance de mathématiques 1 donne l'espérance de chaque morceau, \(1/n\). Ici : la loi des morceaux, l'espérance du plus petit, du \(k\)-ième plus petit, et la probabilité de pouvoir former un triangle.

### Notations

\(n \ge 2\) est le nombre de morceaux. \(U_1, \dots, U_{n-1}\) sont indépendantes, uniformes sur \([0, 1]\) ; \(U_{(1)} < \dots < U_{(n-1)}\) sont ces points triés, avec les conventions \(U_{(0)} = 0\) et \(U_{(n)} = 1\). Le \(j\)-ième morceau en partant de la gauche a pour longueur \(L_j = U_{(j)} - U_{(j-1)}\), et \(L_1 + \dots + L_n = 1\). Les longueurs triées sont \(L_{(1)} \le \dots \le L_{(n)}\) : \(L_{(1)}\) est le plus petit morceau. Pour un réel \(x\), \((x)_+ = \max(x, 0)\).

### L'intuition

Les \(n\) morceaux se comportent comme \(n\) durées exponentielles indépendantes divisées par leur somme (énoncé (E5)). Or le minimum de \(n\) exponentielles de même paramètre est \(n\) fois plus petit en moyenne que chacune (séance de mathématiques 2, résultat (A3)). Le plus petit de \(n\) morceaux de moyenne \(1/n\) mesure donc \(1/n^2\) en moyenne, et non \(1/(2n)\) ou \(1/(n+1)\).

### L'énoncé

1. **(E1) Loi conjointe.** Pour des réels \(c_1, \dots, c_n \ge 0\) de somme \(c \le 1\) : \(\mathbb{P}(L_1 > c_1, \dots, L_n > c_n) = (1 - c)^{n-1}\).
2. **(E2) Chaque morceau, et le plus petit.** Chaque \(L_j\) a la même loi : \(\mathbb{P}(L_j > x) = (1 - x)^{n-1}\) pour \(0 \le x \le 1\). Les morceaux ne sont pas indépendants. \(\mathbb{P}(L_{(1)} > x) = (1 - nx)_+^{\,n-1}\) et \(\mathbb{E}[L_{(1)}] = \dfrac{1}{n^2}\).
3. **(E3) Le \(k\)-ième plus petit.** \(\mathbb{E}[L_{(k)}] = \dfrac{1}{n}\Big(\dfrac{1}{n} + \dfrac{1}{n-1} + \dots + \dfrac{1}{n-k+1}\Big)\) ; pour le plus grand, \(\dfrac{1}{n}\Big(1 + \dfrac12 + \dots + \dfrac1n\Big)\).
4. **(E4) Le plus grand, et le triangle.** Pour \(x \ge 1/2\), \(\mathbb{P}(L_{(n)} > x) = n\,(1 - x)^{n-1}\). Trois morceaux forment un triangle avec probabilité \(1/4\).
5. **(E5) Représentation.** Si \(E_1, \dots, E_n\) sont indépendantes de loi exponentielle de paramètre 1 et si \(T = E_1 + \dots + E_n\), le vecteur \((E_1/T, \dots, E_n/T)\) a la loi de \((L_1, \dots, L_n)\) et il est indépendant de \(T\).

### La preuve

1. Loi des points triés. Le vecteur \((U_{(1)}, \dots, U_{(n-1)})\) a la densité \((n-1)!\) sur \(\{0 < u_1 < \dots < u_{n-1} < 1\}\) : pour une région \(\mathcal{B}\) de cet ensemble, « les points triés sont dans \(\mathcal{B}\) » est la réunion disjointe, sur les \((n-1)!\) ordres possibles des \(U_i\), d'événements de probabilité égale au volume de \(\mathcal{B}\) (les \(U_i\) non triés ont la densité 1 sur le cube, et permuter les coordonnées conserve le volume).
2. Loi des morceaux. Le passage \(l_j = u_j - u_{j-1}\) (avec \(u_0 = 0\)) est linéaire, de matrice triangulaire à diagonale de 1, donc de déterminant 1 : \((L_1, \dots, L_{n-1})\) a la densité \((n-1)!\) sur le simplexe \(\mathcal{S} = \{l_j > 0,\ l_1 + \dots + l_{n-1} < 1\}\) (changement de variables, admis), et \(L_n = 1 - L_1 - \dots - L_{n-1}\). La loi est uniforme sur \(\mathcal{S}\), dont le volume est donc \(1/(n-1)!\).
3. (E1), cas \(c < 1\). En termes des \(n - 1\) premières longueurs, l'événement s'écrit \(l_j > c_j\) pour \(j < n\) et \(l_1 + \dots + l_{n-1} < 1 - c_n\). En posant \(l_j = c_j + (1 - c)\,l'_j\), ces conditions deviennent \(l'_j > 0\) et \(l'_1 + \dots + l'_{n-1} < 1\) : la région de l'événement est l'image de \(\mathcal{S}\) par une translation suivie d'une homothétie de rapport \(1 - c\) en dimension \(n - 1\), de volume \((1 - c)^{n-1}/(n-1)!\). Multiplié par la densité \((n-1)!\) : \((1 - c)^{n-1}\). Si \(c = 1\), l'événement exige \(L_j = c_j\) pour tout \(j\), de probabilité 0.
4. (E2). Avec \(c_j = x\) et les autres nuls : \(\mathbb{P}(L_j > x) = (1 - x)^{n-1}\), quel que soit \(j\). Avec tous les \(c_j\) égaux à \(x \le 1/n\) : \(\mathbb{P}(L_{(1)} > x) = (1 - nx)^{n-1}\), et cette probabilité est nulle si \(x > 1/n\) (la somme des longueurs vaut 1). Par la formule de la queue (séance de mathématiques 1, outil 3, (F)) : \(\mathbb{E}[L_{(1)}] = \int_0^{1/n} (1 - nx)^{n-1}\,dx = \Big[-\dfrac{(1 - nx)^n}{n^2}\Big]_0^{1/n} = \dfrac{1}{n^2}\). Non-indépendance : \(\mathbb{P}(L_1 > 0{,}6, L_2 > 0{,}6) = 0\), alors que chaque probabilité est non nulle.
5. (E4). Si \(x \ge 1/2\), deux morceaux ne peuvent pas dépasser \(x\) tous les deux (leur somme dépasserait 1) : les événements \(\{L_j > x\}\) sont deux à deux disjoints, donc \(\mathbb{P}(L_{(n)} > x) = n\,(1 - x)^{n-1}\). Trois longueurs forment un triangle (non aplati) si et seulement si chacune est strictement inférieure à la somme des deux autres ; leur somme valant 1, cela équivaut à : chacune est inférieure à \(1/2\). Probabilité : \(1 - \mathbb{P}(L_{(3)} \ge 1/2) = 1 - 3 \times (1/2)^2 = \tfrac14\).
6. (E5). La densité de \((E_1, \dots, E_n)\) est \(e^{-(e_1 + \dots + e_n)}\) sur les réels positifs. Les sommes partielles \(s_j = e_1 + \dots + e_j\) (déterminant 1) ont la densité \(e^{-s_n}\) sur \(\{0 < s_1 < \dots < s_n\}\). On pose \(v_j = s_j/s_n\) pour \(j < n\), en gardant \(s_n\) : \(s_j = v_j s_n\), de jacobien \(s_n^{\,n-1}\). La densité de \((V_1, \dots, V_{n-1}, T)\) est donc \(t^{n-1} e^{-t} = (n-1)! \times \dfrac{t^{n-1} e^{-t}}{(n-1)!}\) sur le produit de \(\{0 < v_1 < \dots < v_{n-1} < 1\}\) par \(\{t > 0\}\). C'est un produit sur un domaine produit : par (D4) de la partie 4 (même preuve avec un vecteur à la place de \(x\)), \((V_1, \dots, V_{n-1})\) a la densité \((n-1)!\), celle des points triés de l'étape 1, et il est indépendant de \(T\), de loi gamma de forme \(n\). Les écarts successifs des \(V_j\) sont les \(E_j/T\).
7. (E3). Soient \(E_{(1)} \le \dots \le E_{(n)}\) les exponentielles triées. \(E_{(1)}\) est le minimum de \(n\) exponentielles de paramètre 1, d'espérance \(1/n\) (résultat (A3) de la séance de mathématiques 2). Par (A5) (« la course recommence »), les \(n - 1\) durées restantes \(E_j - E_{(1)}\) sont indépendantes, de loi exponentielle de paramètre 1 : \(E_{(2)} - E_{(1)}\) est le minimum de \(n - 1\) exponentielles, d'espérance \(1/(n-1)\), et ainsi de suite. Donc \(\mathbb{E}[E_{(k)}] = \tfrac1n + \tfrac{1}{n-1} + \dots + \tfrac{1}{n-k+1}\).
8. Fin de (E3). Diviser par \(T > 0\) conserve l'ordre : \(L_{(k)}\) a la loi de \(E_{(k)}/T\). Et \(E_{(k)} = (E_{(k)}/T) \times T\) est le produit de deux variables indépendantes par (E5), donc \(\mathbb{E}[E_{(k)}] = \mathbb{E}[E_{(k)}/T]\,\mathbb{E}[T] = n\,\mathbb{E}[L_{(k)}]\). On divise l'étape 7 par \(n\).

### Exemple calculé

Trois morceaux (deux points de coupe) : le plus petit mesure en moyenne \(\tfrac13 \times \tfrac13 = \tfrac{@Lf_3_min_num@}{@Lf_3_min_den@}\), celui du milieu \(\tfrac13\big(\tfrac13 + \tfrac12\big) = \tfrac{@Lf_3_mil_num@}{@Lf_3_mil_den@}\), le plus grand \(\tfrac13\big(\tfrac13 + \tfrac12 + 1\big) = \tfrac{@Lf_3_max_num@}{@Lf_3_max_den@}\) ; la somme vaut bien 1. Probabilité de former un triangle : \(\tfrac{@Lf_triangle_num@}{@Lf_triangle_den@}\). Deux morceaux (un point de coupe) : \(\tfrac{@Lf_2_min_num@}{@Lf_2_min_den@}\) et \(\tfrac{@Lf_2_max_num@}{@Lf_2_max_den@}\). La corrélation entre le morceau de gauche et celui du milieu, pour trois morceaux, vaut \(@Lf_3_corr:1m@\) (calcul d'intégrales du script).

```python-exec
nb = 400000
coupes = np.sort(rng.random((nb, 2)), axis=1)
morceaux = np.diff(np.concatenate([np.zeros((nb, 1)), coupes, np.ones((nb, 1))], axis=1), axis=1)
tries = np.sort(morceaux, axis=1)
print("moyennes des morceaux triés   :", tries.mean(axis=0).round(4), "; exact :", np.round([1 / 9, 5 / 18, 11 / 18], 4))
print("moyennes gauche, milieu, droite :", morceaux.mean(axis=0).round(4))
print("triangle possible             :", round(float(np.mean(tries[:, 2] < 0.5)), 4))
e = rng.exponential(1.0, (nb, 3))
print("exponentielles normalisées, triées :", np.sort(e / e.sum(axis=1, keepdims=True), axis=1).mean(axis=0).round(4))
```

### Cas limites

- **Bâton de longueur \(\ell\)** : toutes les longueurs sont multipliées par \(\ell\), les probabilités d'événements comme « former un triangle » sont inchangées.
- **Cercle.** \(n\) points uniformes indépendants sur un cercle de circonférence 1 le découpent en \(n\) arcs qui ont la loi des \(n\) morceaux du bâton : on coupe le cercle au premier point, et les \(n - 1\) autres, repérés à partir de lui, sont indépendants et uniformes.
- **Coupes successives : une autre loi.** On coupe en un point uniforme, puis on recoupe le **plus grand** des deux morceaux en un point uniforme de ce morceau. Le plus grand morceau \(M\) est uniforme sur \([1/2, 1]\) ; la seconde coupe le partage en \(WM\) et \((1 - W)M\), avec \(W\) uniforme sur \([0, 1]\), indépendante de \(M\). Le morceau \(1 - M\) est sous \(1/2\) ; le triangle exige \(WM < 1/2\) et \((1 - W)M < 1/2\), soit \(1 - \tfrac{1}{2M} < W < \tfrac{1}{2M}\), de probabilité \(\tfrac1M - 1\). En intégrant contre la densité 2 de \(M\) : \(\int_{1/2}^{1} 2\big(\tfrac1m - 1\big)\,dm = 2\ln 2 - 1 \approx @Lf_seq:3m@\) (simulation : @Lf_seq_mc:3@), et non \(1/4\).
- **Plus de trois morceaux** : pour \(n = 4\), \(\mathbb{P}(L_{(4)} < 1/2) = 1 - 4 \times (1/2)^3 = \tfrac{@Lf_poly4_num@}{@Lf_poly4_den@}\) par (E4).

### Pièges

1. Le plus petit de trois morceaux mesure \(1/9\) en moyenne, pas \(1/6\) ni \(1/4\) : \(1/n^2\), avec \(n\) le nombre de **morceaux**.
2. \(n\) morceaux viennent de \(n - 1\) coupes ; l'outil 3 de la séance de mathématiques 1 compte \(n\) points, donc \(n + 1\) morceaux.
3. Le morceau de gauche a la même loi que celui du milieu (E2), mais les morceaux ne sont pas indépendants.
4. Lire comment les coupes sont faites : deux points tirés ensemble, ou coupes successives.

## Partie 7 : conditionner sur la bonne information

### Le problème

La phrase « il y a au moins un garçon » peut conduire à \(1/3\) ou à \(1/2\) selon la façon dont l'information a été obtenue. La règle : on ne conditionne pas par une phrase vraie, mais par l'événement « j'ai observé ceci », décrit dans un univers écrit explicitement, avec la règle d'observation.

### Le cadre

Une famille a deux enfants. Hypothèses du modèle : chaque enfant est un garçon (G) ou une fille (F) avec probabilité \(1/2\), indépendamment de l'autre. L'univers est l'ensemble des couples (aîné, cadet) : GG, GF, FG, FF, chacun de probabilité \(1/4\).

### L'énoncé

1. **Version 1.** On sait que l'aîné est une fille. Alors \(\mathbb{P}(\text{deux filles}) = \tfrac{@Lg_v1_num@}{@Lg_v1_den@}\).
2. **Version 2.** À la question « avez-vous au moins un garçon ? », le parent répond oui, et il dit vrai. Alors \(\mathbb{P}(\text{deux garçons}) = \tfrac{@Lg_v2_num@}{@Lg_v2_den@}\).
3. **Version 3.** On rencontre l'un des deux enfants, choisi au hasard (aîné ou cadet avec probabilité \(1/2\), indépendamment des sexes) : c'est un garçon. Alors \(\mathbb{P}(\text{deux garçons}) = \tfrac{@Lg_v3_num@}{@Lg_v3_den@}\).

### La preuve

1. Version 1 : l'événement observé est \(\{\text{FG}, \text{FF}\}\), de probabilité \(1/2\) ; FF a la probabilité \(1/4\) ; le rapport vaut \(1/2\). Le sexe du cadet est indépendant de celui de l'aîné.
2. Version 2 : l'événement observé est \(\{\text{GG}, \text{GF}, \text{FG}\}\), de probabilité \(3/4\) ; GG a la probabilité \(1/4\) ; le rapport vaut \(1/3\).
3. Version 3 : l'univers doit contenir l'enfant rencontré. Il a @Lg_v3_nb_issues@ issues (famille, enfant rencontré), chacune de probabilité \(1/8\). « L'enfant rencontré est un garçon » contient @Lg_v3_nb_vu@ issues : (GG, aîné), (GG, cadet), (GF, aîné), (FG, cadet), dont @Lg_v3_nb_gg@ avec deux garçons. Le rapport vaut \(2/4 = 1/2\).
4. Pourquoi les versions 2 et 3 diffèrent, alors que « au moins un garçon » est vrai dans les deux : par la formule de Bayes, la probabilité d'**observer** un garçon vaut 1 pour GG et seulement \(1/2\) pour GF et pour FG dans la version 3, alors que la réponse « oui » a la probabilité 1 pour ces trois familles dans la version 2. La version 3 favorise GG d'un facteur 2.

### Exemple calculé : le garçon né un mardi

Le parent répond oui à « avez-vous au moins un garçon né un mardi ? », chaque enfant étant né un jour de la semaine uniforme et indépendant du reste. Cas général : soit une propriété que chaque garçon possède avec probabilité \(q\), indépendamment (\(q = 1/7\) pour le mardi).

1. Un enfant donné est « un garçon ayant la propriété » avec probabilité \(q/2\). Donc \(\mathbb{P}(\text{au moins un tel enfant}) = 1 - (1 - q/2)^2 = q(4 - q)/4\).
2. \(\mathbb{P}(\text{deux garçons, dont au moins un a la propriété}) = \tfrac14\big(1 - (1 - q)^2\big) = q(2 - q)/4\).
3. Rapport : \(\mathbb{P}(\text{deux garçons} \mid \text{oui}) = \dfrac{2 - q}{4 - q}\).

Pour \(q = 1\) (aucune précision), on retrouve \(1/3\), la version 2 ; pour \(q = 1/7\), \(\tfrac{@Lg_mardi_num@}{@Lg_mardi_den@} \approx @Lg_mardi:3m@\) (énumération des @Lg_mardi_nb@ couples de types d'enfant, un type étant un sexe et un jour, et simulation du script : @Lg_mardi_mc:3@) ; quand \(q \to 0\), la limite est \(1/2\). Plus la propriété est rare, plus elle désigne un enfant précis, et plus on se rapproche de la version 1.

### Cas limites

- **Énoncé sans règle d'observation.** « M. Durand a deux enfants, dont un garçon. Probabilité qu'il ait deux garçons ? » est ambigu : \(1/3\) si l'information vient d'un filtre sur la famille (version 2), \(1/2\) si elle vient de l'observation d'un enfant désigné (versions 1 et 3). Un énoncé correct dit comment l'information a été obtenue.
- **Même mécanisme ailleurs** : dans le jeu des trois portes, la réponse dépend de la règle suivie par l'animateur quand il ouvre une porte, pas seulement de la porte ouverte.
- Les hypothèses (probabilité \(1/2\), indépendance des sexes) sont les conventions du problème ; avec d'autres probabilités, la méthode est la même : écrire l'univers, les probabilités des issues, l'événement observé.

### Pièges

1. Conditionner par la phrase « au moins un garçon » quand on a en réalité **vu** un enfant.
2. Parler de « l'autre enfant » dans la version 2 : aucun enfant n'y est désigné.
3. Oublier que GF et FG sont deux issues distinctes, de probabilité totale \(1/2\).

[[controle:enfants]]

## Réflexes en 30 secondes

| Indice dans l'énoncé | Premier geste |
|---|---|
| « étalée à droite », ordre de la moyenne, de la médiane et du mode | règle empirique mode < médiane < moyenne (exponentielle, log-normale, gamma) ; seule garantie générale : \(\lvert \mu - m \rvert \le \sigma\) |
| asymétrie, kurtosis d'une loi usuelle | exponentielle : 2 et 9 ; gamma de forme \(a\) : \(2/\sqrt{a}\) et \(3 + 6/a\) ; normale : 0 et 3 ; somme de \(n\) : divisés par \(\sqrt{n}\) et \(n\) (pour l'excès) |
| changement d'unité | covariance multipliée par \(ac\) ; corrélation inchangée, au signe de \(ac\) près ; pente \(\rho\,\sigma_Y/\sigma_X\) |
| « corrélation nulle », « mesure quoi » | liaison linéaire seulement ; \(Y = X^2\) avec \(X\) symétrique : corrélation nulle, dépendance totale |
| comparer deux notes | cote Z \((x - \mu)/\sigma\) ; rangs comparables seulement si les lois ont la même forme |
| adéquation à une loi discrète (binomiale, Poisson) | khi-deux, \(k - 1 - s\) degrés de liberté, effectifs attendus d'au moins 5 |
| adéquation à une loi continue spécifiée | Kolmogorov-Smirnov (écart maximal), Anderson-Darling (queues) |
| normalité, paramètres estimés | Lilliefors, Shapiro-Wilk, Jarque-Bera ; jamais la table de Kolmogorov |
| densité conjointe | constante par \(\iint = 1\) ; marginale en intégrant l'autre variable ; conditionnelle = conjointe divisée par marginale ; indépendance = produit **sur un pavé** |
| intensité de Poisson, a priori | poids a posteriori proportionnel à \(\pi_i\,\lambda_i^k e^{-\lambda_i t}\) ; gamma de forme \(a\), paramètre \(b\) devient forme \(a + k\), paramètre \(b + t\) ; a priori \(1/\lambda\) : exposant \(k - 1\) |
| bâton coupé en \(n\) morceaux | plus petit \(1/n^2\) ; \(k\)-ième plus petit \(\tfrac1n\big(\tfrac1n + \dots + \tfrac{1}{n-k+1}\big)\) ; triangle \(1/4\) |
| deux enfants | écrire GG, GF, FG, FF ; « aîné » ou enfant vu : \(1/2\) ; « au moins un » par filtre : \(1/3\) |

## À savoir refaire sans notes

- \(\lvert \mu - m \rvert \le \sigma\) : la médiane minimise l'écart absolu moyen, puis trois inégalités.
- Moyenne, médiane, mode et asymétrie de la loi exponentielle et de la loi log-normale ; un contre-exemple à la règle de l'ordre.
- L'effet d'un changement affine sur la covariance et la corrélation ; la variance résiduelle \(\sigma_Y^2 (1 - \rho^2)\) ; un exemple de corrélation nulle sans indépendance.
- Le khi-deux : \(\mathbb{E}[C] = k - 1\), le cas \(k = 2\) par la binomiale centrée réduite, le compte des degrés de liberté.
- Kolmogorov-Smirnov : la formule de calcul, et la preuve que la loi de \(D_n\) est libre par \(X_i = Q(U_i)\) ; ce qui casse pour une loi discrète et avec des paramètres estimés.
- Sur une densité conjointe : normalisation, marginales, conditionnelle, probabilité sachant \(X = x\), critère d'indépendance.
- La loi a posteriori sur une liste, la conjugaison gamma-Poisson, l'effet de l'a priori en \(1/\lambda\).
- \(\mathbb{P}(L_1 > c_1, \dots, L_n > c_n) = (1 - c)^{n-1}\) par le volume du simplexe, \(\mathbb{E}[L_{(1)}] = 1/n^2\), le triangle, et \(\mathbb{E}[L_{(k)}]\) par les exponentielles.
- Les trois versions du problème des deux enfants, univers écrit.
""",
    "controles": {
        "forme": {
            "question": r"On sait seulement qu'une loi a une variance finie et un coefficient d'asymétrie strictement positif. Qu'est-ce qui est garanti ?",
            "options": [
                {"id": "a", "texte": r"Rien sur l'ordre de la moyenne et de la médiane ; seulement que leur écart est au plus égal à l'écart-type.",
                 "explication": r"Oui : \(\lvert \mu - m \rvert \le \sigma\) vaut pour toute loi de variance finie. Dans le contre-exemple discret de la leçon, l'asymétrie vaut \(@La_B_skew:3m@\), l'écart entre moyenne et médiane \(@Ca_ecart:1m@\) et l'écart-type \(@Ca_sigma:3m@\)."},
                {"id": "b", "texte": "La moyenne est supérieure à la médiane.",
                 "explication": r"C'est la règle empirique, vraie pour les lois exponentielle, log-normale et gamma, mais pas un théorème : la loi qui vaut \(-2\), 1, 4 avec probabilités 0,4, 0,5, 0,1 a une asymétrie positive, une moyenne de \(@La_B_moy:1m@\) et une médiane de @La_B_med@."},
                {"id": "c", "texte": "Le mode est inférieur à la médiane.",
                 "explication": r"Faux en général : pour la loi de Weibull de forme \(@La_w_k:1m@\), l'asymétrie est positive (\(@La_w_skew:3m@\)) et le mode \(@La_w_mode:3m@\) dépasse la médiane \(@La_w_med:3m@\)."},
                {"id": "d", "texte": "Le kurtosis est supérieur à 3.",
                 "explication": r"Asymétrie et kurtosis sont deux moments distincts : le contre-exemple discret de la leçon a une asymétrie positive et un kurtosis de \(@La_B_kurt:2m@\), sous celui de la loi normale. Seul \(\kappa \ge 1\) est garanti."},
            ],
        },
        "unites": {
            "question": r"Deux rendements exprimés en nombres décimaux ont une covariance de 0,00012 et une corrélation de 0,6. On les exprime tous deux en pourcentage (multiplication par 100). Que deviennent la covariance et la corrélation ?",
            "options": [
                {"id": "a", "texte": r"Covariance \(@Lb_cov_pct:1m@\), corrélation \(@Lb_rho:1m@\).",
                 "explication": r"Oui : \(\mathrm{Cov}(aX, cY) = ac\,\mathrm{Cov}(X, Y)\) avec \(a = c = 100\), soit un facteur 10 000 ; la corrélation, sans unité, ne change pas puisque \(ac > 0\)."},
                {"id": "b", "texte": r"Covariance \(@Cb_cov_une_fois:3m@\), corrélation \(@Lb_rho:1m@\).",
                 "explication": r"Le facteur 100 n'a été appliqué qu'une fois. La covariance est un produit de deux écarts : chacun des deux est multiplié par 100."},
                {"id": "c", "texte": r"Covariance \(@Lb_cov_pct:1m@\), corrélation \(@Cb_rho_cent:0m@\).",
                 "explication": r"La corrélation a été multipliée par 100 comme une quantité qui aurait une unité. Elle est le rapport de la covariance au produit des écarts-types, tous deux multipliés par 100 : elle reste entre \(-1\) et 1."},
                {"id": "d", "texte": r"Covariance \(@Lb_cov_dec:5m@\), corrélation \(@Lb_rho:1m@\).",
                 "explication": r"C'est la corrélation qui est insensible aux unités, pas la covariance : celle-ci a l'unité du produit des deux variables et change avec elle."},
            ],
        },
        "adequation": {
            "question": r"On veut tester la normalité de 80 rendements. On estime la moyenne et l'écart-type sur ces 80 valeurs, on calcule la statistique de Kolmogorov-Smirnov contre la loi normale ainsi ajustée, et on lit la p-value dans la table de Kolmogorov. Que vaut cette procédure ?",
            "options": [
                {"id": "a", "texte": "Son niveau réel est très inférieur au niveau affiché et elle perd de la puissance ; il faut des seuils propres au cas estimé (Lilliefors), ou un autre test de normalité.",
                 "explication": r"Oui : la loi ajustée suit les données, la statistique est plus petite que sous la loi de la table. Dans la simulation du script (\(n = 50\)), la fréquence de rejet au niveau nominal 5 % tombe à \(@Lc_lil_niv_table_sim:4m@\), et le seuil correct est \(@Lc_lil_seuil_sim:3m@\) au lieu de \(@Lc_ks_seuil50:3m@\)."},
                {"id": "b", "texte": "Elle est exacte : la loi de la statistique ne dépend pas de la loi continue testée.",
                 "explication": r"Ce résultat suppose une loi testée **entièrement spécifiée** avant de voir les données. Ici, la loi testée dépend de l'échantillon par la moyenne et l'écart-type estimés."},
                {"id": "c", "texte": "Elle rejette trop souvent une hypothèse de normalité vraie.",
                 "explication": r"C'est le sens inverse : estimer les paramètres rapproche la loi ajustée des données, donc diminue la statistique. Le test rejette trop rarement, pas trop souvent."},
                {"id": "d", "texte": "Elle devient exacte si l'on retire deux degrés de liberté, un par paramètre estimé.",
                 "explication": r"Les degrés de liberté appartiennent au test du khi-deux. La loi de la statistique de Kolmogorov-Smirnov n'a pas de degrés de liberté ; avec des paramètres estimés, il faut une autre table."},
            ],
        },
        "posterieur": {
            "question": r"L'intensité \(\Lambda\) d'un processus de Poisson suit a priori la loi gamma de forme 2 et de paramètre 1 (par heure). On observe 7 événements en 3 heures. Quelle est la loi a posteriori, et quel est le maximum a posteriori ?",
            "options": [
                {"id": "a", "texte": r"Loi gamma de forme @Le_g_alpha@ et de paramètre @Le_g_beta@ ; maximum a posteriori @Le_g_map@.",
                 "explication": r"Oui : la forme reçoit le nombre d'événements (\(2 + 7\)), le paramètre reçoit la durée (\(1 + 3\)), et le mode d'une loi gamma de forme \(\alpha\) et de paramètre \(\beta\) est \((\alpha - 1)/\beta = 8/4\)."},
                {"id": "b", "texte": r"Loi gamma de forme @Le_g_alpha@ et de paramètre @Le_g_beta@ ; maximum a posteriori \(@Le_g_moy:2m@\).",
                 "explication": r"La loi est juste, mais \(9/4\) est sa **moyenne**. Le maximum a posteriori est le mode de la densité, \((\alpha - 1)/\beta\), plus petit que la moyenne pour une loi gamma."},
                {"id": "c", "texte": r"Loi gamma de forme @Le_g_alpha@ et de paramètre 3 ; maximum a posteriori \(@Ce_d_sans_duree:2m@\).",
                 "explication": r"La durée observée remplace ici le paramètre a priori au lieu de s'y ajouter. Le noyau a posteriori est \(\lambda^{2 + 7 - 1} e^{-(1 + 3)\lambda}\) : les exposants de \(e^{-\lambda}\) s'additionnent."},
                {"id": "d", "texte": r"Le maximum a posteriori vaut \(7/3 \approx @Le_g_emv:2m@\), quelle que soit la loi a priori.",
                 "explication": r"\(7/3\) est le maximum de vraisemblance. Il ne coïncide avec le maximum a posteriori que pour un a priori constant ; un a priori gamma de forme 2 et de paramètre 1 tire l'estimation vers sa propre zone de forte densité."},
            ],
        },
        "enfants": {
            "question": r"M. Martin a deux enfants (chaque enfant est un garçon ou une fille avec probabilité 1/2, indépendamment). Vous le croisez accompagné de l'un des deux, choisi au hasard avec probabilité 1/2 indépendamment des sexes : c'est un garçon. Quelle est la probabilité que M. Martin ait deux garçons ?",
            "options": [
                {"id": "a", "texte": r"\(\tfrac{@Lg_v3_num@}{@Lg_v3_den@}\)",
                 "explication": r"Oui : parmi les 8 issues (famille, enfant vu), 4 montrent un garçon, dont 2 viennent d'une famille de deux garçons. De façon équivalente, le sexe de l'enfant resté à la maison est indépendant de celui de l'enfant vu."},
                {"id": "b", "texte": r"\(\tfrac{@Lg_v2_num@}{@Lg_v2_den@}\)",
                 "explication": r"C'est la réponse quand on sait seulement, par un filtre sur la famille, qu'il y a au moins un garçon. Ici un enfant précis a été vu, et une famille de deux garçons montre un garçon deux fois plus souvent qu'une famille mixte."},
                {"id": "c", "texte": r"\(\tfrac14\)",
                 "explication": r"C'est la probabilité de deux garçons avant toute observation. Avoir vu un garçon élimine la famille de deux filles et rend les familles avec garçons plus probables."},
                {"id": "d", "texte": r"\(\tfrac23\)",
                 "explication": r"C'est la probabilité que la famille soit mixte quand on sait par un filtre qu'il y a au moins un garçon : deux erreurs à la fois, le mauvais conditionnement et le mauvais événement."},
            ],
        },
    },
    "verif": {"script": "verif/seance_math_5.py"},
}
