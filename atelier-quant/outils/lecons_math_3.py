"""Leçon express de la séance de mathématiques 3 : statistique et inégalités.

Format : voir outils/lecons.py (élément de type "lecon", contenu en Markdown réduit, points de contrôle
[[controle:IDENT]] sur une ligne seule, option "a" correcte, une explication par option).
Aucun nombre calculé n'est écrit à la main : chaque jeton @CLE@, @CLE:3@ ou @CLE:3m@ est remplacé par la valeur
calculée par verif/seance_math_3.py (fichier verif/seance_math_3.out.json), et les blocs ```python-exec sont
exécutés à la construction par le noyau de l'environnement des notebooks (numpy 1.21.6, scipy 1.8.1), qui insère
leur vraie sortie. Graine aléatoire fixée dans le premier bloc.
"""

LECON_MATH_3 = {
    "id": "L-MATH-3", "type": "lecon", "section": "lecon",
    "theme": "Leçon statistique : estimateurs, vraisemblance, intervalles, tests, tests multiples, inégalités",
    "difficulte": 3, "temps_cible_s": 1500, "dossier": "coding/lecons",
    "contenu": r"""## Leçon express : statistique et inégalités

Durée visée : 20 à 25 minutes. Six parties courtes : biais et erreur quadratique, maximum de vraisemblance, intervalles de confiance, tests et p-values, tests multiples, quatre inégalités. Chaque partie donne l'énoncé précis, une preuve courte et complète, un exemple calculé et les pièges. Les parties « Réflexes en 30 secondes » et « À savoir refaire » sont à relire juste avant un test ou un entretien.

Conventions de notation. Une majuscule désigne une variable aléatoire, une minuscule une valeur observée ou un paramètre fixe : \(X_i\) est aléatoire, \(x_i\) sa valeur observée. Exceptions d'usage, déclarées ici : les paramètres grecs \(\theta\), \(\mu\), \(\sigma^2\), \(\lambda\) sont fixes et inconnus ; un estimateur, noté avec un chapeau (\(\hat{\theta}\)) ou un tilde (\(\tilde{\theta}\)), est une fonction des \(X_i\), donc **aléatoire** ; la moyenne empirique \(\bar{X}\) est aléatoire et \(\bar{x}\) est sa valeur observée. \(\mathbb{P}\), \(\mathbb{E}\), \(\mathrm{Var}\), \(\mathrm{Cov}\) désignent probabilité, espérance, variance, covariance. Pour un événement \(A\), \(I_A\) est son indicatrice : la variable aléatoire qui vaut 1 si \(A\) se produit et 0 sinon. \(\Phi\) est la fonction de répartition de la loi normale \(\mathcal{N}(0, 1)\), et \(z_u\) son quantile d'ordre \(u\) : \(\Phi(z_u) = u\). Tous les nombres hors des sorties de code sont calculés par `verif/seance_math_3.py`, qui contrôle chaque probabilité par un calcul exact et une simulation.

Cadre commun : \(n \ge 2\) est un entier, \(X_1, \dots, X_n\) sont des variables aléatoires réelles indépendantes et de même loi (on dit i.i.d.), d'espérance \(\mu\) et de variance \(\sigma^2\) finies. \(\bar{X} = \tfrac1n \sum_{i=1}^n X_i\) et \(Q = \sum_{i=1}^n (X_i - \bar{X})^2\), la somme des carrés des écarts à la moyenne empirique, sont aléatoires.

```python-exec
import numpy as np
from scipy import stats
rng = np.random.default_rng(3)
print("prêt")
```

## Partie 1 : biais, variance, erreur quadratique moyenne

### Le problème et les définitions

Un estimateur \(\hat{\theta}\) d'un paramètre \(\theta\) se trompe d'un échantillon à l'autre. Pour le juger, on sépare l'erreur systématique de la dispersion. Pour \(\hat{\theta}\) de carré intégrable (\(\mathbb{E}[\hat{\theta}^2] < \infty\)) :

- le **biais** est le nombre \(b = \mathbb{E}[\hat{\theta}] - \theta\) ; l'estimateur est sans biais si \(b = 0\) pour toute valeur possible de \(\theta\) ;
- l'**erreur quadratique moyenne** (EQM ; MSE, mean squared error) est \(\mathrm{EQM}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \theta)^2]\).

### L'énoncé

1. \(\mathrm{EQM}(\hat{\theta}) = \mathrm{Var}(\hat{\theta}) + b^2\).
2. \(\mathrm{Var}(\bar{X}) = \sigma^2/n\), et \(\mathbb{E}[Q] = (n - 1)\,\sigma^2\). Donc \(S^2 = Q/(n-1)\) est sans biais pour \(\sigma^2\), et \(Q/n\) a le biais \(-\sigma^2/n\). Aucune normalité n'est requise.
3. Si de plus les \(X_i\) sont de loi normale, pour tout réel \(c > 0\), avec \(k = n - 1\) : \(\mathrm{EQM}(c\,Q) = \sigma^4 \big(c^2 (2k + k^2) - 2ck + 1\big)\). En particulier \(\mathrm{EQM}(Q/(n-1)) = 2\sigma^4/(n-1)\) et \(\mathrm{EQM}(Q/n) = (2n - 1)\,\sigma^4/n^2\), plus petite.

### La preuve

1. **Décomposition.** On écrit \(\hat{\theta} - \theta = (\hat{\theta} - \mathbb{E}\hat{\theta}) + b\). Le carré vaut \((\hat{\theta} - \mathbb{E}\hat{\theta})^2 + 2b\,(\hat{\theta} - \mathbb{E}\hat{\theta}) + b^2\). Le terme croisé a une espérance nulle, car \(b\) est un nombre et \(\mathbb{E}[\hat{\theta} - \mathbb{E}\hat{\theta}] = 0\). Il reste \(\mathrm{Var}(\hat{\theta}) + b^2\).
2. **Variance d'une somme.** \(\mathrm{Var}(\sum_i X_i) = \sum_i \sum_j \mathrm{Cov}(X_i, X_j)\) par bilinéarité de la covariance ; pour \(i \ne j\), \(\mathrm{Cov}(X_i, X_j) = 0\) par indépendance. Donc \(\mathrm{Var}(\sum_i X_i) = n\sigma^2\) et \(\mathrm{Var}(\bar{X}) = n\sigma^2/n^2 = \sigma^2/n\).
3. **Identité de Huygens.** En écrivant \(X_i - \mu = (X_i - \bar{X}) + (\bar{X} - \mu)\) et en développant, le terme croisé \(2(\bar{X} - \mu) \sum_i (X_i - \bar{X})\) est nul car \(\sum_i (X_i - \bar{X}) = 0\). D'où \(\sum_i (X_i - \mu)^2 = Q + n\,(\bar{X} - \mu)^2\).
4. **Espérance de \(Q\).** En prenant l'espérance de l'identité : \(n\sigma^2 = \mathbb{E}[Q] + n\,\mathrm{Var}(\bar{X}) = \mathbb{E}[Q] + \sigma^2\), d'où \(\mathbb{E}[Q] = (n-1)\,\sigma^2\). Le biais de \(Q/n\) vaut \(\tfrac{n-1}{n}\sigma^2 - \sigma^2 = -\sigma^2/n\).
5. **Cas normal.** On admet le théorème de Fisher (théorème général sur les vecteurs gaussiens) : si les \(X_i\) sont de loi \(\mathcal{N}(\mu, \sigma^2)\), \(W = Q/\sigma^2\) suit la loi du khi-deux à \(k = n - 1\) degrés de liberté, notée \(\chi^2_k\), c'est-à-dire la loi de \(Z_1^2 + \dots + Z_k^2\) pour \(Z_1, \dots, Z_k\) indépendantes de loi \(\mathcal{N}(0, 1)\) ; et \(W\) est indépendante de \(\bar{X}\).
6. **Moments du khi-deux.** Soit \(Z\) de loi \(\mathcal{N}(0, 1)\), de densité \(\varphi\), qui vérifie \(\varphi'(z) = -z\,\varphi(z)\). Par intégration par parties, \(\mathbb{E}[Z^4] = \int z^3 \cdot z\varphi(z)\,dz = \int z^3 (-\varphi'(z))\,dz = \int 3z^2 \varphi(z)\,dz = 3\) (le terme de bord \(z^3 \varphi(z)\) tend vers 0 aux deux infinis). Donc \(\mathrm{Var}(Z^2) = 3 - 1 = 2\), et par l'étape 2, \(\mathbb{E}[W] = k\), \(\mathrm{Var}(W) = 2k\), \(\mathbb{E}[W^2] = 2k + k^2\).
7. **EQM.** \(\mathrm{EQM}(c\,Q) = \mathbb{E}[(c\sigma^2 W - \sigma^2)^2] = \sigma^4\,\mathbb{E}[c^2 W^2 - 2cW + 1] = \sigma^4 \big(c^2 (2k + k^2) - 2ck + 1\big)\). Avec \(c = 1/k\) : \(\sigma^4 (2/k + 1 - 2 + 1) = 2\sigma^4/k\). Avec \(c = 1/(k+1)\), en réduisant au dénominateur \((k+1)^2\) : \(\sigma^4 (2k + 1)/(k + 1)^2 = \sigma^4 (2n - 1)/n^2\). Enfin \(\tfrac{2}{n-1} - \tfrac{2n-1}{n^2} = \tfrac{3n - 1}{n^2 (n-1)} > 0\).

### Exemple calculé

\(n = @L1_n@\) observations normales, \(\sigma^2 = 1\). Biais de \(Q/n\) : \(@L1_biais_n:1m@\). EQM de \(S^2 = Q/4\) : \(@L1_eqm_nm1:2m@\) ; EQM de \(Q/5\) : \(\tfrac{@L1_eqm_n_num@}{@L1_eqm_n_den@} = @L1_eqm_n:2m@\) ; EQM de \(Q/6\) : \(\tfrac{@L1_eqm_np1_num@}{@L1_eqm_np1_den@}\). Simulation du script : \(@L1_eqm_nm1_mc:3m@\), \(@L1_eqm_n_mc:3m@\), \(@L1_eqm_np1_mc:3m@\).

### Pièges

1. Sans biais ne veut pas dire meilleur : sous normalité, l'estimateur biaisé \(Q/n\) a une EQM plus petite que \(S^2\). La décomposition du point 1 dit qu'on peut échanger un peu de biais contre beaucoup de variance ; c'est le principe de la régularisation Ridge.
2. Le point 2 ne demande ni normalité ni connaissance de \(\mu\) ; le point 3 demande la normalité (avec une autre loi, \(\mathrm{Var}(W)\) dépend du coefficient d'aplatissement).
3. \(S^2\) sans biais n'implique pas \(S = \sqrt{S^2}\) sans biais pour \(\sigma\) (partie 6, Jensen).

[[controle:biais]]

## Partie 2 : maximum de vraisemblance

### Le principe

Le modèle est une famille de densités (ou de probabilités, pour une loi discrète) \(f(x; \theta)\). Après observation de \(x_1, \dots, x_n\), la **vraisemblance** est la fonction \(L(\theta) = \prod_i f(x_i; \theta)\) du paramètre, et la log-vraisemblance \(\ell(\theta) = \sum_i \ln f(x_i; \theta)\). L'estimateur du maximum de vraisemblance (EMV ; MLE en anglais) est la valeur \(\hat{\theta}\) qui maximise \(L\), donc \(\ell\) (le logarithme est croissant) : le paramètre sous lequel les données observées étaient les plus probables.

### Trois exemples

1. **Bernoulli.** \(f(x; \theta) = \theta^x (1 - \theta)^{1-x}\) pour \(x \in \{0, 1\}\). Avec \(k\) succès sur \(n\) : \(\ell(\theta) = k \ln\theta + (n - k)\ln(1 - \theta)\), \(\ell'(\theta) = k/\theta - (n-k)/(1-\theta)\), qui s'annule en \(\theta = k/n\), est positive avant et négative après : \(\hat{\theta} = k/n\). Si \(k = 0\) (ou \(k = n\)), \(\ell\) est décroissante (croissante) et le maximum sur \([0, 1]\) est au bord, \(\hat{\theta} = 0\) (ou 1). Exemple : \(@L2_bern_k@\) succès sur \(@L2_bern_n@\), \(\hat{\theta} = @L2_bern_hat:2m@\).
2. **Normale.** \(f(x; \mu, v) = (2\pi v)^{-1/2} e^{-(x - \mu)^2/(2v)}\), où \(v = \sigma^2\). \(\ell(\mu, v) = -\tfrac{n}{2}\ln(2\pi v) - \tfrac{1}{2v}\sum_i (x_i - \mu)^2\). Par l'identité de Huygens (partie 1, étape 3, valable aussi pour des nombres), \(\sum_i (x_i - \mu)^2 = q + n(\bar{x} - \mu)^2\), où \(q\) est la valeur observée de \(Q\) : à \(v\) fixé, le maximum est en \(\mu = \bar{x}\). Ensuite \(\partial \ell/\partial v = -\tfrac{n}{2v} + \tfrac{q}{2v^2}\), positive pour \(v < q/n\) et négative après (si \(q > 0\)) : \(\hat{v} = q/n\). L'EMV de la variance est donc l'estimateur **biaisé** \(Q/n\). Exemple : données 2, 4, 4, 5, 10 ; \(\bar{x} = @L2_norm_moy@\), \(q = @L2_norm_sc@\), \(\hat{v} = @L2_norm_emv:1m@\), alors que \(s^2 = q/4 = @L2_norm_s2@\).
3. **Exponentielle.** \(f(x; \lambda) = \lambda e^{-\lambda x}\) pour \(x \ge 0\). \(\ell(\lambda) = n\ln\lambda - \lambda \sum_i x_i\), \(\ell'(\lambda) = n/\lambda - \sum_i x_i\), nulle en \(\hat{\lambda} = n/\sum_i x_i = 1/\bar{x}\), maximum car \(\ell'' = -n/\lambda^2 < 0\).

### L'EMV peut être biaisé : calcul exact pour l'exponentielle

Énoncé : pour \(n \ge 2\), \(\mathbb{E}[1/\bar{X}] = \tfrac{n}{n-1}\lambda\). Preuve, avec \(T = \sum_i X_i\) :

1. Pour \(s > 0\), \(1/s = \int_0^\infty e^{-us}\,du\). Donc \(\mathbb{E}[1/T] = \int_0^\infty \mathbb{E}[e^{-uT}]\,du\) (échange permis pour une fonction positive : Fubini-Tonelli, admis).
2. \(\mathbb{E}[e^{-uX_1}] = \int_0^\infty \lambda e^{-(\lambda + u)x}\,dx = \lambda/(\lambda + u)\), et par indépendance \(\mathbb{E}[e^{-uT}] = (\lambda/(\lambda + u))^n\).
3. \(\int_0^\infty \lambda^n (\lambda + u)^{-n}\,du = \lambda^n \lambda^{1-n}/(n-1) = \lambda/(n-1)\). Donc \(\mathbb{E}[1/\bar{X}] = n\,\mathbb{E}[1/T] = \tfrac{n}{n-1}\lambda\).

Pour \(n = @L2_exp_n@\) : \(\mathbb{E}[\hat{\lambda}] = @L2_exp_facteur:1m@\,\lambda\) ; simulation avec \(\lambda = 1\) : \(@L2_exp_esp_mc:3m@\). La partie 6 (Jensen) explique le sens du biais sans calcul.

**Invariance.** Si \(g\) est une bijection, l'EMV de \(\eta = g(\theta)\) est \(g(\hat{\theta})\) : la vraisemblance exprimée en \(\eta\) est \(L(g^{-1}(\eta))\), maximale quand \(g^{-1}(\eta) = \hat{\theta}\). Ainsi l'EMV de la moyenne \(1/\lambda\) de l'exponentielle est \(\bar{X}\), sans biais, alors que \(1/\bar{X}\) est biaisé : l'invariance ne conserve pas l'absence de biais.

### Propriétés asymptotiques, énoncées et admises

Sous des conditions de régularité (modèle identifiable, support de \(f\) indépendant de \(\theta\), \(\theta\) intérieur à l'ensemble des paramètres, \(\ln f\) deux fois dérivable en \(\theta\) avec des conditions de domination), quand \(n \to \infty\) :

- **convergence** : \(\hat{\theta}_n\) converge en probabilité vers \(\theta\) ;
- **normalité asymptotique** : \(\sqrt{n}\,(\hat{\theta}_n - \theta)\) converge en loi vers \(\mathcal{N}(0, 1/\mathcal{I}(\theta))\), où l'**information de Fisher** est \(\mathcal{I}(\theta) = \mathbb{E}\big[(\partial_\theta \ln f(X_1; \theta))^2\big] = -\mathbb{E}\big[\partial_\theta^2 \ln f(X_1; \theta)\big]\) ;
- **efficacité asymptotique** : \(1/\mathcal{I}(\theta)\) est la plus petite variance asymptotique possible parmi les estimateurs réguliers (borne de Cramér-Rao).

Ce sont des énoncés **limites** : ils ne disent rien de garanti à \(n\) fixé. Exemples : Bernoulli, \(\mathcal{I}(\theta) = 1/(\theta(1-\theta))\), d'où l'erreur-type approchée \(\sqrt{\hat{\theta}(1 - \hat{\theta})/n} = @L2_bern_se:3m@\) dans l'exemple ; exponentielle, \(\mathcal{I}(\lambda) = 1/\lambda^2\).

Contre-exemple de régularité : loi uniforme sur \([0, \theta]\). Le support dépend de \(\theta\) ; l'EMV est le maximum des observations (\(L(\theta) = \theta^{-n}\) si \(\theta \ge \max_i x_i\), 0 sinon), d'espérance \(\tfrac{n}{n+1}\theta\) (`outils/lecons_probas.py`, outil 3, après mise à l'échelle), et son erreur est d'ordre \(1/n\), pas \(1/\sqrt{n}\).

[[controle:asymptotique]]

## Partie 3 : intervalles de confiance

### Définition et interprétation

Un intervalle de confiance de niveau \(1 - \alpha\) est un intervalle \([A, B]\) dont les bornes sont des **fonctions des données**, donc aléatoires, tel que \(\mathbb{P}(A \le \theta \le B) \ge 1 - \alpha\) pour toute valeur de \(\theta\) (avec égalité pour un intervalle exact). La probabilité porte sur la procédure : sur un grand nombre d'échantillons, environ \(100(1-\alpha)\) % des intervalles produits contiennent \(\theta\). Une fois les données observées, \([a, b]\) et \(\theta\) sont des nombres fixes : \(\theta\) est dedans ou non, et aucune probabilité fréquentiste n'y est attachée.

### Construction exacte sous normalité

La **loi de Student** à \(k\) degrés de liberté est la loi de \(Z/\sqrt{W/k}\), où \(Z\) suit \(\mathcal{N}(0, 1)\) et \(W\) suit \(\chi^2_k\), indépendantes. Elle est symétrique, et \(t_{k; u}\) désigne son quantile d'ordre \(u\).

Énoncé : si les \(X_i\) sont de loi \(\mathcal{N}(\mu, \sigma^2)\) et \(S = \sqrt{Q/(n-1)}\), alors \(\bar{X} \pm t_{n-1; 1-\alpha/2}\,S/\sqrt{n}\) contient \(\mu\) avec probabilité exactement \(1 - \alpha\). Si \(\sigma\) est connu, \(\bar{X} \pm z_{1-\alpha/2}\,\sigma/\sqrt{n}\) aussi.

1. \(\bar{X}\) suit \(\mathcal{N}(\mu, \sigma^2/n)\) (une combinaison linéaire de normales indépendantes est normale, admis), donc \(Z = \sqrt{n}(\bar{X} - \mu)/\sigma\) suit \(\mathcal{N}(0, 1)\).
2. \(T = \sqrt{n}(\bar{X} - \mu)/S = Z / \sqrt{W/(n-1)}\) avec \(W = Q/\sigma^2\) ; par le théorème de Fisher (partie 1, étape 5), \(Z\) et \(W\) sont indépendantes et \(W\) suit \(\chi^2_{n-1}\) : \(T\) suit la loi de Student à \(n - 1\) degrés de liberté. Sa loi ne dépend d'aucun paramètre inconnu (c'est un **pivot**).
3. Avec \(t = t_{n-1; 1-\alpha/2}\) et par symétrie, \(\mathbb{P}(-t \le T \le t) = 1 - \alpha\). Et \(-t \le \sqrt{n}(\bar{X} - \mu)/S \le t\) équivaut à \(\bar{X} - tS/\sqrt{n} \le \mu \le \bar{X} + tS/\sqrt{n}\). Même raisonnement avec \(Z\) si \(\sigma\) est connu.

Exemple : \(n = @L3_n@\), \(\bar{x} = @L3_xbar:1m@\), \(s = @L3_s:1m@\). \(t_{9; 0{,}975} = @L3_t9:3m@\) (contre \(z_{0{,}975} = @L3_z975:3m@\)), demi-largeur \(@L3_t9:3m@ \times @L3_s:1m@/\sqrt{10} = @L3_demi_t:3m@\), intervalle \([@L3_bas:3m@ ; @L3_haut:3m@]\). Le quantile de Student est plus grand que celui de la normale : il paie l'estimation de \(\sigma\).

### Intervalle asymptotique

Théorème central limite (TCL, admis) : si les \(X_i\) sont i.i.d. de variance \(\sigma^2\) finie et non nulle, \(\sqrt{n}(\bar{X} - \mu)/\sigma\) converge en loi vers \(\mathcal{N}(0, 1)\). Le lemme de Slutsky (admis) permet d'y remplacer \(\sigma\) par \(S\), qui converge en probabilité vers \(\sigma\). Donc \(\mathbb{P}(\bar{X} - z_{1-\alpha/2} S/\sqrt{n} \le \mu \le \bar{X} + z_{1-\alpha/2} S/\sqrt{n}) \to 1 - \alpha\), pour **toute** loi de variance finie : la couverture n'est garantie qu'à la limite.

Exemple d'échec à \(n\) fini : proportion \(\hat{\theta}\) de succès, intervalle de Wald \(\hat{\theta} \pm z_{0{,}975}\sqrt{\hat{\theta}(1 - \hat{\theta})/n}\). Couverture exacte (somme sur la loi binomiale) pour \(n = @L3_wald_n@\) : \(@L3_wald_05:3m@\) si \(\theta = 0{,}5\), mais \(@L3_wald_01:3m@\) si \(\theta = 0{,}1\). Une cause visible : avec probabilité \(0{,}9^{20} \approx @L3_wald_k0:3m@\), aucun succès, et l'intervalle se réduit au point 0.

L'expérience suivante mesure la couverture de l'intervalle de Student à 95 % sur des données normales, où il est exact, et exponentielles, où il n'est qu'asymptotique.

```python-exec
def couverture_t(tirer, mu, n, nb=20000):
    x = tirer((nb, n))
    m = x.mean(axis=1)
    s = x.std(axis=1, ddof=1)
    demi = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    return np.mean(np.abs(m - mu) <= demi)

print("erreur standard d'une couverture simulée proche de 0,95 :", round(np.sqrt(0.95 * 0.05 / 20000), 4))
for n in [5, 10, 30, 100]:
    c_norm = couverture_t(lambda t: rng.normal(1.0, 2.0, t), 1.0, n)
    c_expo = couverture_t(lambda t: rng.exponential(1.0, t), 1.0, n)
    print("n =", n, "; normale :", round(c_norm, 4), "; exponentielle :", round(c_expo, 4))
```

### Pièges

1. Lire « \(\theta\) a 95 % de chances d'être dans \([a, b]\) ».
2. Utiliser \(z\) au lieu de \(t\) pour un petit échantillon : intervalle trop étroit.
3. La demi-largeur est proportionnelle à \(1/\sqrt{n}\) : la diviser par 2 demande 4 fois plus de données.

[[controle:ic]]

## Partie 4 : tests, p-values, puissance

### Vocabulaire

On veut décider entre une hypothèse nulle \(H_0\) (par exemple \(\mu = \mu_0\)) et une alternative \(H_1\) (\(\mu \ne \mu_0\)). Un test est une règle de rejet de \(H_0\), fonction des données. L'**erreur de première espèce** est de rejeter \(H_0\) vraie ; le test est **de niveau** \(\alpha\) si sa probabilité est au plus \(\alpha\) pour toute loi de \(H_0\). L'**erreur de seconde espèce** est de ne pas rejeter \(H_0\) fausse ; la **puissance** en un point de \(H_1\) est la probabilité de rejeter, soit 1 moins la probabilité d'erreur de seconde espèce. On fixe le niveau, puis on cherche la puissance la plus grande.

### La p-value

Définition : soit \(D\) une statistique de test dont les grandes valeurs témoignent contre \(H_0\) (par exemple \(D = \lvert T \rvert\)), de valeur observée \(d\). La p-value est \(p = \mathbb{P}_{H_0}(D \ge d)\) : la probabilité, **calculée en supposant \(H_0\) vraie**, d'obtenir une statistique au moins aussi extrême que celle observée. Rejeter quand \(p \le \alpha\) donne un test de niveau \(\alpha\) ; l'énoncé suivant le démontre, avec égalité, dans le cas d'une statistique de loi continue.

Énoncé : si \(H_0\) fixe entièrement la loi de \(D\) et que sa fonction de répartition \(G\) est continue, la p-value aléatoire \(P = 1 - G(D)\) suit la loi uniforme sur \([0, 1]\) sous \(H_0\).

1. Soit \(u \in ]0, 1[\) et \(c = \inf\{x : G(x) \ge 1 - u\}\). \(G\) est continue à droite, donc \(G(c) \ge 1 - u\) ; et \(G(x) < 1 - u\) pour \(x < c\), donc par continuité \(G(c) \le 1 - u\). Ainsi \(G(c) = 1 - u\).
2. \(G\) est croissante : si \(D \ge c\), \(G(D) \ge 1 - u\) ; si \(D < c\), \(G(D) < 1 - u\) par définition de \(c\). Donc \(\{P \le u\} = \{G(D) \ge 1 - u\} = \{D \ge c\}\).
3. \(\mathbb{P}(D \ge c) = 1 - G(c) = u\), la loi de \(D\) n'ayant pas d'atome. Donc \(\mathbb{P}(P \le u) = u\).

Une statistique discrète (un comptage) ne donne pas une p-value uniforme : \(P\) n'y prend qu'un nombre dénombrable de valeurs.

```python-exec
def pvalues(effet, n=20, nb=20000):
    x = rng.normal(effet, 1.0, (nb, n))
    t = x.mean(axis=1) / (x.std(axis=1, ddof=1) / np.sqrt(n))
    return 2 * stats.t.sf(np.abs(t), n - 1)

bornes = np.linspace(0, 1, 11)
for effet in [0.0, 0.5]:
    p = pvalues(effet)
    print("effet", effet, ": proportion de p-values <= 0,05 :", round(np.mean(p <= 0.05), 4))
    print("   effectifs par dixième de [0, 1] :", np.histogram(p, bins=bornes)[0])
```

Lecture : sous \(H_0\) (effet 0), chacun des dix effectifs a pour espérance 2000, un dixième des 20 000 tirages ; sous l'alternative, les p-values s'entassent près de 0, et la proportion sous 0,05 estime la puissance du test.

### Lecture correcte et contresens

La p-value n'est **pas** la probabilité que \(H_0\) soit vraie : elle est calculée en supposant \(H_0\) vraie. Exemple : 10 % des signaux testés ont un effet réel, puissance 0,8, niveau 0,05. Par la formule de Bayes, \(\mathbb{P}(H_0 \mid \text{rejet}) = \dfrac{0{,}9 \times 0{,}05}{0{,}9 \times 0{,}05 + 0{,}1 \times 0{,}8} = \dfrac{@L4_bayes_num:3m@}{@L4_bayes_den:3m@} = @L4_bayes:2m@\) : plus d'un rejet sur trois est une fausse découverte, bien que chaque test soit au niveau 5 %. Autres contresens : une petite p-value ne mesure pas la taille de l'effet (elle diminue avec \(n\) à effet fixé) ; \(p > 0{,}05\) ne prouve pas \(H_0\) (manque de puissance possible).

### Le test t

Sous normalité, pour \(H_0 : \mu = \mu_0\), \(T = \sqrt{n}(\bar{X} - \mu_0)/S\) suit sous \(H_0\) la loi de Student à \(n - 1\) degrés de liberté (partie 3). Test bilatéral de niveau \(\alpha\) : rejet si \(\lvert T \rvert > t_{n-1; 1-\alpha/2}\) ; p-value \(2\,\mathbb{P}(T_{n-1} \ge \lvert t \rvert)\), où \(T_{n-1}\) suit la loi de Student.

**Dualité.** \(\lvert T \rvert \le t_{n-1; 1-\alpha/2}\) équivaut à \(\mu_0 \in [\bar{X} \pm t_{n-1; 1-\alpha/2} S/\sqrt{n}]\) (même manipulation qu'en partie 3) : le test rejette exactement les \(\mu_0\) hors de l'intervalle de confiance.

Exemple : \(n = @L4_n@\), \(\bar{x} = 0{,}5\), \(s = 1{,}2\), \(\mu_0 = 0\). \(t = 0{,}5/(1{,}2/4) = \tfrac{@L4_t_num@}{@L4_t_den@} \approx @L4_t:3m@\), sous \(t_{15; 0{,}975} = @L4_t15:3m@\) : pas de rejet à 5 %, p-value \(@L4_p:3m@\). Avec la loi normale au lieu de Student, on trouverait \(@L4_p_z:3m@\), trop petit.

### Puissance d'un test z unilatéral

\(\sigma\) connu, \(H_0 : \mu = 0\) contre \(H_1 : \mu > 0\), rejet si \(\sqrt{n}\,\bar{X}/\sigma \ge z_{1-\alpha}\). Si \(\mu = \delta > 0\), \(\sqrt{n}\,\bar{X}/\sigma = \sqrt{n}\,\delta/\sigma + Z\) avec \(Z\) de loi \(\mathcal{N}(0, 1)\), donc la puissance vaut \(\mathbb{P}(Z \ge z_{1-\alpha} - \sqrt{n}\,\delta/\sigma) = \Phi(\sqrt{n}\,\delta/\sigma - z_{1-\alpha})\). Exemple : \(\delta/\sigma = 0{,}5\), \(n = @L4_puiss_n@\), \(\alpha = 0{,}05\) : \(\Phi(5 \times 0{,}5 - @L4_z95:3m@) = \Phi(@L4_puiss_arg:3m@) \approx @L4_puiss:3m@\).

[[controle:pvalue]]

## Partie 5 : tests multiples

### Taux d'erreur familial

On réalise \(m\) tests ; \(m_0\) hypothèses nulles sont vraies. \(V\) est le nombre (aléatoire) de rejets à tort, \(R\) le nombre total de rejets. Le taux d'erreur familial est \(\mathrm{FWER} = \mathbb{P}(V \ge 1)\). Bonferroni (chaque test au niveau \(\alpha/m\)) garantit \(\mathrm{FWER} \le \alpha\) pour toute dépendance, par la borne de Boole : énoncé, preuve et optimalité dans `outils/lecons_probas.py`, outil 4.

### Proportion de fausses découvertes et Benjamini-Hochberg

Avec beaucoup de tests, exiger zéro fausse découverte coûte trop de puissance. On contrôle plutôt la proportion de fausses découvertes \(\mathrm{FDP} = V/\max(R, 1)\) (nulle s'il n'y a aucun rejet), en moyenne : \(\mathrm{FDR} = \mathbb{E}[\mathrm{FDP}]\) (false discovery rate).

**Procédure de Benjamini-Hochberg (BH)** au niveau \(q\) : trier les p-values \(p_{(1)} \le \dots \le p_{(m)}\) ; poser \(k = \max\{i : p_{(i)} \le i\,q/m\}\) ; rejeter les \(k\) hypothèses de plus petites p-values (aucune si l'ensemble est vide). C'est une procédure **montante** : on cherche le **plus grand** \(i\) qui passe son seuil, et l'on rejette tout ce qui est en dessous, même des p-values qui dépassent leur propre seuil.

**Garantie** (Benjamini et Hochberg, 1995, « Controlling the false discovery rate: a practical and powerful approach to multiple testing », Journal of the Royal Statistical Society, série B, 57(1), 289-300) : si les p-values des hypothèses nulles vraies sont indépendantes entre elles et des autres, et uniformes, alors \(\mathrm{FDR} \le m_0\,q/m \le q\). La preuve n'est pas donnée ici, faute de place : elle est **admise**. Benjamini et Yekutieli (2001, Annals of Statistics, 29(4), 1165-1188) étendent la garantie à une dépendance positive particulière, et donnent une variante valable pour toute dépendance, au prix d'un seuil divisé par \(\sum_{j=1}^m 1/j\).

Constaté par simulation dans le script (non démontré ici) : \(m = @L5_fdr_m@\), \(m_0 = @L5_fdr_m0@\), \(q = @L5_fdr_q:1m@\), p-values indépendantes : FDR simulée \(@L5_fdr_mc:4m@\), pour \(m_0 q/m = @L5_fdr_theo:3m@\). Dans la même expérience, le FWER de BH vaut \(@L5_fwer_bh:3m@\), bien au-dessus de \(q\) : BH ne garantit pas le FWER au niveau \(q\).

**Lien entre les deux**, démontré : \(\mathrm{FDP} \le I_{\{V \ge 1\}}\), car \(\mathrm{FDP} = 0\) si \(V = 0\) et \(\mathrm{FDP} \le 1\) sinon. En espérance, \(\mathrm{FDR} \le \mathrm{FWER}\) : contrôler le FWER contrôle la FDR. Si toutes les nulles sont vraies (\(m_0 = m\)), \(V = R\), donc \(\mathrm{FDP} = I_{\{V \ge 1\}}\) et \(\mathrm{FDR} = \mathrm{FWER}\).

Exemple : \(m = @L5_m@\), \(q = 0{,}05\), p-values triées 0,004 ; 0,021 ; 0,028 ; 0,035 ; 0,5 ; seuils \(i\,q/m\) : \(@L5_seuils:2m@\). \(p_{(2)} = 0{,}021 > 0{,}02\) échoue, mais \(p_{(4)} = 0{,}035 \le 0{,}04\) passe : \(k = @L5_k@\), et l'on rejette les quatre premières, \(p_{(2)}\) comprise. Bonferroni au seuil \(0{,}05/5 = @L5_bonf_seuil:2m@\) n'en rejette que \(@L5_bonf@\).

```python-exec
def benjamini_hochberg(p, q):
    p = np.asarray(p, dtype=float)
    m = len(p)
    ordre = np.argsort(p)
    ok = p[ordre] <= q * np.arange(1, m + 1) / m
    k = np.nonzero(ok)[0].max() + 1 if ok.any() else 0
    rejet = np.zeros(m, dtype=bool)
    rejet[ordre[:k]] = True
    return rejet

p = [0.004, 0.021, 0.028, 0.035, 0.5]
print("BH, q = 0,05       :", benjamini_hochberg(p, 0.05))
print("Bonferroni, 0,01   :", np.asarray(p) <= 0.05 / 5)
```

### Pièges

1. S'arrêter au premier échec (procédure descendante) au lieu de chercher le plus grand \(i\) qui passe.
2. Comparer chaque p-value à \(q\), ou à \(q/m\) : ce sont le non-ajustement et Bonferroni.
3. Croire que BH borne la probabilité d'au moins une fausse découverte.

## Partie 6 : quatre inégalités

### Markov

Énoncé : si \(X \ge 0\) et \(a > 0\), \(\mathbb{P}(X \ge a) \le \mathbb{E}[X]/a\). Preuve : \(a\,I_{\{X \ge a\}} \le X\) en tout point (si \(X \ge a\), le membre de gauche vaut \(a \le X\) ; sinon il vaut \(0 \le X\)) ; on prend l'espérance. Égalité si \(X\) ne prend que les valeurs 0 et \(a\). Exemple : \(X \ge 0\) de moyenne 2 : \(\mathbb{P}(X \ge 10) \le @L6_markov:1m@\) ; pour une exponentielle de moyenne 2, la vraie valeur est \(e^{-5} \approx @L6_exp:4m@\) : la borne est valable pour toute loi positive, donc grossière pour une loi donnée.

### Tchebychev

Énoncé : si \(\mathrm{Var}(X)\) est finie et \(a > 0\), \(\mathbb{P}(\lvert X - \mathbb{E}X \rvert \ge a) \le \mathrm{Var}(X)/a^2\). Preuve : Markov appliquée à \((X - \mathbb{E}X)^2 \ge 0\) et \(a^2\), car \(\{\lvert X - \mathbb{E}X \rvert \ge a\} = \{(X - \mathbb{E}X)^2 \ge a^2\}\). Usage : avec \(\bar{X}\), \(\mathbb{P}(\lvert \bar{X} - \mu \rvert \ge \varepsilon) \le \sigma^2/(n\varepsilon^2)\), qui tend vers 0 (loi faible des grands nombres). Exemple : proportion de Bernoulli, \(\sigma^2 = \theta(1 - \theta) \le 1/4\) ; pour garantir \(\mathbb{P}(\lvert \bar{X} - \theta \rvert \ge 0{,}1) \le 0{,}05\) quel que soit \(\theta\), il suffit de \(n \ge \tfrac{1/4}{0{,}1^2 \times 0{,}05} = @L6_cheb_n@\). L'approximation du TCL donnerait \(n \ge (z_{0{,}975} \times 0{,}5/0{,}1)^2 \approx @L6_tcl_brut:2m@\), soit \(@L6_tcl_n@\), mais ce n'est qu'une approximation, pas une garantie.

### Jensen

Énoncé : si \(\varphi\) est convexe sur un intervalle contenant les valeurs de \(X\), et si \(X\) et \(\varphi(X)\) sont intégrables, \(\varphi(\mathbb{E}X) \le \mathbb{E}[\varphi(X)]\). Si \(\varphi\) est strictement convexe, l'égalité impose \(X\) constante avec probabilité 1.

1. Soit \(m = \mathbb{E}X\), supposé intérieur à l'intervalle, et \(\varphi\) dérivable en \(m\) (cas de \(x^2\), \(e^x\), \(1/x\) et \(-\ln x\) sur \(]0, \infty[\) ; le cas général remplace \(\varphi'(m)\) par la dérivée à droite, admis). Par convexité, pour \(t \in ]0, 1]\), \(\varphi(m + t(x - m)) \le \varphi(m) + t\,(\varphi(x) - \varphi(m))\). On divise par \(t\) et on fait tendre \(t\) vers 0 : \(\varphi'(m)(x - m) \le \varphi(x) - \varphi(m)\). La courbe est au-dessus de sa tangente en \(m\).
2. Avec \(x = X\) et en prenant l'espérance : \(\mathbb{E}[\varphi(X)] \ge \varphi(m) + \varphi'(m)\,\mathbb{E}[X - m] = \varphi(m)\).
3. Égalité : la variable \(\varphi(X) - \varphi(m) - \varphi'(m)(X - m)\) est positive d'espérance nulle, donc nulle avec probabilité 1. Si \(\varphi\) est strictement convexe, elle ne s'annule qu'en \(x = m\) : si elle s'annulait en \(x \ne m\), au milieu \(y = (x + m)/2\) la stricte convexité donnerait \(\varphi(y) < \tfrac12(\varphi(x) + \varphi(m)) = \varphi(m) + \varphi'(m)(y - m)\), contraire à l'étape 1.

Usages : \(\varphi(x) = x^2\) donne \(\mathrm{Var}(X) \ge 0\) ; \(\varphi(x) = 1/x\) donne \(\mathbb{E}[1/\bar{X}] > 1/\mathbb{E}[\bar{X}] = \lambda\) pour l'exponentielle (partie 2) : l'EMV \(\hat{\lambda}\) surestime \(\lambda\) en moyenne ; \(\varphi = -\sqrt{\cdot}\), convexe, donne \(\mathbb{E}[S] \le \sqrt{\mathbb{E}[S^2]} = \sigma\), strictement si \(S\) n'est pas constante. Pour \(n = 2\) et des observations normales de variance 1, \(\mathbb{E}[S] = \sqrt{2/\pi} \approx @L6_es:3m@\) (simulation \(@L6_es_mc:3m@\)). De même \(\mathbb{E}[\ln Y] \le \ln \mathbb{E}[Y]\) pour \(Y > 0\) : la croissance logarithmique moyenne est inférieure au logarithme de la croissance moyenne.

### Cauchy-Schwarz

Énoncé : si \(\mathbb{E}[X^2]\) et \(\mathbb{E}[Y^2]\) sont finies, \(\mathbb{E}[XY]^2 \le \mathbb{E}[X^2]\,\mathbb{E}[Y^2]\), avec égalité si et seulement si \(X\) et \(Y\) sont proportionnelles avec probabilité 1.

1. Si \(\mathbb{E}[Y^2] = 0\), \(Y = 0\) avec probabilité 1 (Markov appliquée à \(Y^2\) : \(\mathbb{P}(Y^2 \ge a) = 0\) pour tout \(a > 0\)), et les deux membres sont nuls.
2. Sinon, pour tout réel \(t\), \(0 \le \mathbb{E}[(X - tY)^2] = \mathbb{E}[X^2] - 2t\,\mathbb{E}[XY] + t^2\,\mathbb{E}[Y^2]\). Ce trinôme en \(t\) est positif, donc son discriminant est négatif ou nul : \(4\,\mathbb{E}[XY]^2 - 4\,\mathbb{E}[X^2]\,\mathbb{E}[Y^2] \le 0\).
3. Égalité : le discriminant est nul, le trinôme s'annule en un \(t_0\), donc \(\mathbb{E}[(X - t_0 Y)^2] = 0\) et \(X = t_0 Y\) avec probabilité 1.

Usage : appliquée à \(X - \mathbb{E}X\) et \(Y - \mathbb{E}Y\), elle donne \(\lvert \mathrm{Cov}(X, Y) \rvert \le \sqrt{\mathrm{Var}(X)\,\mathrm{Var}(Y)}\), donc une corrélation entre -1 et 1. Exemple : \(\mathrm{Var}(X) = 4\), \(\mathrm{Var}(Y) = 9\) imposent \(\lvert \mathrm{Cov}(X, Y) \rvert \le @L6_cs@\), atteint pour \(Y = 1{,}5\,X\) plus une constante.

[[controle:jensen]]

## Réflexes en 30 secondes

| Indice dans l'énoncé | Premier geste |
|---|---|
| « biais », « erreur quadratique », « quel estimateur est meilleur » | EQM = variance + biais au carré ; calculer les deux séparément |
| variance empirique, « n ou n − 1 » | \(\mathbb{E}[Q] = (n-1)\sigma^2\) ; sous normalité \(Q/\sigma^2\) suit \(\chi^2_{n-1}\), de variance \(2(n-1)\) |
| « maximum de vraisemblance » | écrire \(\ell\), dériver, vérifier le signe ; regarder les bords ; Bernoulli \(k/n\), normale \((\bar{x}, q/n)\), exponentielle \(1/\bar{x}\) |
| « intervalle de confiance », petit \(n\), \(\sigma\) inconnu | \(\bar{x} \pm t_{n-1; 1-\alpha/2}\,s/\sqrt{n}\), avec \(n - 1\) degrés de liberté |
| « p-value », « probabilité que l'hypothèse soit vraie » | p-value calculée sous \(H_0\) ; pour \(\mathbb{P}(H_0 \mid \text{rejet})\), formule de Bayes avec la proportion de vrais effets et la puissance |
| « puissance », « taille d'échantillon » | \(\Phi(\sqrt{n}\,\delta/\sigma - z_{1-\alpha})\) en unilatéral |
| plusieurs tests, « fausses découvertes » | FWER : Bonferroni \(\alpha/m\) ; FDR : BH, trier, seuils \(i\,q/m\), plus grand \(i\) qui passe |
| « borne », « au plus », loi inconnue | Markov (positive, moyenne), Tchebychev (variance), Jensen (fonction convexe d'une espérance), Cauchy-Schwarz (produit, covariance) |

Contrôles de vraisemblance en 5 secondes : une EQM est au moins égale à la variance ; un quantile de Student dépasse celui de la normale au même ordre (au-dessus de 1/2) ; une p-value est uniforme sous \(H_0\) continue ; BH rejette au moins autant que Bonferroni au même niveau ; \(\mathbb{E}[1/X] \ge 1/\mathbb{E}[X]\) pour \(X > 0\).

## À savoir refaire sans notes

- La décomposition EQM = variance + biais au carré, et \(\mathbb{E}[Q] = (n-1)\sigma^2\) par l'identité de Huygens.
- L'EMV de Bernoulli, de la normale (avec le biais de \(Q/n\)) et de l'exponentielle, et \(\mathbb{E}[1/\bar{X}] = \tfrac{n}{n-1}\lambda\) par la transformée de Laplace.
- L'intervalle de Student par le pivot \(T\), et sa dualité avec le test t.
- L'uniformité de la p-value sous \(H_0\), et le calcul de \(\mathbb{P}(H_0 \mid \text{rejet})\) par Bayes.
- La procédure de Benjamini-Hochberg sur une liste, et \(\mathrm{FDR} \le \mathrm{FWER}\).
- Les preuves de Markov, Tchebychev, Jensen (tangente) et Cauchy-Schwarz (discriminant).
""",
    "controles": {
        "biais": {
            "question": r"Soient \(X_1, \dots, X_n\) i.i.d. de variance \(\sigma^2\), avec \(n \ge 2\), et \(Q = \sum_i (X_i - \bar{X})^2\). Quel est le biais de \(Q/n\) comme estimateur de \(\sigma^2\) ?",
            "options": [
                {"id": "a", "texte": r"\(-\sigma^2/n\)",
                 "explication": r"Oui : \(\mathbb{E}[Q] = (n-1)\sigma^2\), donc \(\mathbb{E}[Q/n] - \sigma^2 = \tfrac{n-1}{n}\sigma^2 - \sigma^2 = -\sigma^2/n\)."},
                {"id": "b", "texte": r"0, car \(\bar{X}\) est sans biais pour \(\mu\).",
                 "explication": r"Centrer sur \(\bar{X}\) au lieu de \(\mu\) fait perdre le terme \(n(\bar{X} - \mu)^2\) de l'identité de Huygens, d'espérance \(\sigma^2\) : \(Q\) est en moyenne plus petit que \(\sum_i (X_i - \mu)^2\)."},
                {"id": "c", "texte": r"\(+\sigma^2/n\)",
                 "explication": r"Erreur de signe : \(Q \le \sum_i (X_i - \mu)^2\) en tout point, car \(\bar{X}\) minimise \(\sum_i (X_i - c)^2\) en \(c\). \(Q/n\) sous-estime donc \(\sigma^2\) en moyenne."},
                {"id": "d", "texte": r"\(-\sigma^2/(n-1)\)",
                 "explication": r"Le biais se calcule à partir de \(\mathbb{E}[Q/n] = \tfrac{n-1}{n}\sigma^2\) : la différence avec \(\sigma^2\) a pour dénominateur \(n\), pas \(n - 1\)."},
            ],
        },
        "asymptotique": {
            "question": r"On observe \(n = 3\) durées i.i.d. de loi exponentielle de paramètre \(\lambda\), et l'on estime \(\lambda\) par l'EMV \(\hat{\lambda} = 1/\bar{X}\). Que garantit le théorème de normalité asymptotique de l'EMV pour cet échantillon ?",
            "options": [
                {"id": "a", "texte": r"Rien à \(n = 3\) : c'est un énoncé sur la limite quand \(n \to \infty\). Ici \(\hat{\lambda}\) a même un biais de 50 %.",
                 "explication": r"Oui : \(\mathbb{E}[\hat{\lambda}] = \tfrac{n}{n-1}\lambda = @L2_exp_facteur:1m@\,\lambda\) pour \(n = 3\). Le théorème dit seulement que \(\sqrt{n}(\hat{\lambda}_n - \lambda)\) se rapproche en loi de \(\mathcal{N}(0, \lambda^2)\) quand \(n\) grandit."},
                {"id": "b", "texte": r"Que \(\hat{\lambda}\) est sans biais.",
                 "explication": r"Aucun énoncé asymptotique ne donne l'absence de biais à \(n\) fixé, et ici le calcul exact donne \(\mathbb{E}[\hat{\lambda}] = \tfrac{3}{2}\lambda\)."},
                {"id": "c", "texte": r"Que \(\hat{\lambda}\) suit exactement la loi \(\mathcal{N}(\lambda, \lambda^2/3)\).",
                 "explication": r"C'est l'approximation suggérée par le théorème avec \(\mathcal{I}(\lambda) = 1/\lambda^2\), pas une loi exacte : \(\hat{\lambda}\) est positif et sa loi est asymétrique, alors qu'une normale charge les négatifs."},
                {"id": "d", "texte": r"Que \(\hat{\lambda}\) a la plus petite variance parmi tous les estimateurs de \(\lambda\).",
                 "explication": r"L'efficacité est asymptotique et concerne les estimateurs réguliers ; à \(n\) fixé, l'estimateur constant \(\tilde{\lambda} = 1\) a une variance nulle, plus petite, sans être un bon estimateur."},
            ],
        },
        "ic": {
            "question": r"Sur un échantillon normal de taille 10, on obtient l'intervalle de Student à 95 % \([@Cic_bas:3m@ ; @Cic_haut:3m@]\) pour \(\mu\). Quelle phrase est juste ?",
            "options": [
                {"id": "a", "texte": r"La méthode qui produit cet intervalle capture \(\mu\) pour 95 % des échantillons ; cet intervalle-ci contient \(\mu\) ou non, sans probabilité attachée.",
                 "explication": r"Oui : la probabilité 0,95 porte sur les bornes aléatoires \(A\) et \(B\), avant observation. Après, tout est fixé."},
                {"id": "b", "texte": r"\(\mathbb{P}(@Cic_bas:3m@ \le \mu \le @Cic_haut:3m@) = 0{,}95\).",
                 "explication": r"\(\mu\) est un paramètre fixe et les bornes sont des nombres : cette probabilité vaut 0 ou 1. Lui donner un sens demanderait un modèle bayésien, avec une loi a priori sur \(\mu\)."},
                {"id": "c", "texte": "95 % des observations de l'échantillon sont dans l'intervalle.",
                 "explication": r"L'intervalle porte sur la moyenne \(\mu\) ; sa largeur décroît en \(1/\sqrt{n}\), alors que la dispersion des observations ne décroît pas avec \(n\)."},
                {"id": "d", "texte": "Un nouvel échantillon de même taille aura sa moyenne dans cet intervalle avec probabilité 0,95.",
                 "explication": r"Cette probabilité dépend de la position de \(\mu\), inconnue, par rapport à cet intervalle-ci. Avant observation, elle est même inférieure à 0,95 : la différence des deux moyennes a pour variance \(2\sigma^2/n\), le double de celle d'une seule moyenne."},
            ],
        },
        "pvalue": {
            "question": r"Un test t bilatéral de \(H_0 : \mu = 0\) donne une p-value de 0,03. Quelle lecture est juste ?",
            "options": [
                {"id": "a", "texte": r"Si \(H_0\) était vraie, une statistique au moins aussi extrême que celle observée aurait une probabilité de 0,03.",
                 "explication": r"Oui : c'est la définition, \(p = \mathbb{P}_{H_0}(\lvert T \rvert \ge \lvert t \rvert)\). On rejette \(H_0\) au niveau 5 %, pas au niveau 1 %."},
                {"id": "b", "texte": r"La probabilité que \(H_0\) soit vraie vaut 0,03.",
                 "explication": r"La p-value est calculée en supposant \(H_0\) vraie. \(\mathbb{P}(H_0 \mid \text{données})\) demande une loi a priori : dans l'exemple de la leçon, \(\mathbb{P}(H_0 \mid \text{rejet}) = @L4_bayes:2m@\) pour des tests au niveau 0,05."},
                {"id": "c", "texte": "L'effet est grand.",
                 "explication": r"À effet fixé, la statistique \(T\) croît comme \(\sqrt{n}\) : un effet minuscule donne une p-value aussi petite qu'on veut si \(n\) est assez grand. La taille de l'effet se lit sur \(\bar{x} - \mu_0\) et son intervalle."},
                {"id": "d", "texte": "L'effet est réel avec probabilité 0,97.",
                 "explication": r"C'est le même contresens que \(\mathbb{P}(H_0) = 0{,}03\), pris au complémentaire : la p-value n'est pas une probabilité sur les hypothèses."},
            ],
        },
        "jensen": {
            "question": r"Soient \(X_1, \dots, X_n\) i.i.d. normales de variance \(\sigma^2 > 0\), \(n \ge 2\), et \(S = \sqrt{Q/(n-1)}\), où \(S^2 = Q/(n-1)\) est sans biais pour \(\sigma^2\). Que vaut \(\mathbb{E}[S]\) ?",
            "options": [
                {"id": "a", "texte": r"Strictement moins que \(\sigma\).",
                 "explication": r"Oui : \(-\sqrt{\cdot}\) est strictement convexe et \(S^2\) n'est pas constante, donc Jensen donne \(\mathbb{E}[S] < \sqrt{\mathbb{E}[S^2]} = \sigma\). Pour \(n = 2\), \(\mathbb{E}[S] = \sqrt{2/\pi}\,\sigma \approx @L6_es:3m@\,\sigma\)."},
                {"id": "b", "texte": r"Exactement \(\sigma\), puisque \(S^2\) est sans biais.",
                 "explication": r"L'espérance ne commute pas avec une fonction non affine : \(\mathbb{E}[\sqrt{S^2}] \ne \sqrt{\mathbb{E}[S^2]}\) dès que \(S^2\) varie."},
                {"id": "c", "texte": r"Strictement plus que \(\sigma\).",
                 "explication": r"Sens inversé : la racine est concave, donc la moyenne des racines est sous la racine de la moyenne. Le sens « plus grand » vaut pour une fonction convexe, comme \(1/x\)."},
                {"id": "d", "texte": r"Cela dépend de \(\mu\).",
                 "explication": r"\(Q\) est inchangé si l'on ajoute une constante à tous les \(X_i\) : la loi de \(S\) ne dépend pas de \(\mu\)."},
            ],
        },
    },
    "verif": {"script": "verif/seance_math_3.py"},
}
