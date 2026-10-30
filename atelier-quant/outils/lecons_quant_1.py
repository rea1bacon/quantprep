"""Leçon express de la séance quantitative 1 : sélection de modèles, finance quantitative et optimisation.

Format : voir outils/lecons.py (élément de type "lecon", contenu en Markdown réduit, points de contrôle
[[controle:IDENT]] sur une ligne seule, option "a" correcte, une explication par option).
Le bloc python-exec est exécuté à la construction par le noyau de l'environnement des notebooks (numpy 1.21.6),
dans le dossier coding/lecons, et sa vraie sortie est insérée ; graine fixée.
Aucun autre nombre calculé n'est écrit à la main : chaque jeton @CLE@, @CLE:3@ ou @CLE:3m@ est remplacé par la
valeur calculée par verif/seance_quant_1.py (fichier verif/seance_quant_1.out.json). Seules les données des
exemples sont écrites en clair.
"""

LECON_QUANT_1 = {
    "id": "L-QUANT-1", "type": "lecon", "section": "lecon",
    "theme": "Leçon quantitative : sélection de modèles, finance quantitative, optimisation",
    "difficulte": 4, "temps_cible_s": 1800, "dossier": "coding/lecons",
    "contenu": r"""## Leçon express : sélection de modèles, finance quantitative et optimisation

Durée visée : 30 minutes (environ 5 000 mots, dont une dizaine de preuves) ; les preuves de la partie 7 (Taylor, convexité, Lagrange) sont des rappels de classe préparatoire, à survoler si elles sont acquises. Cette leçon réunit les outils de la recherche quantitative qui servent à décider si un modèle ou un signal est réel. Sept parties courtes, chacune dans l'ordre : le problème, l'énoncé, la preuve, un exemple calculé, les pièges. Les nombres sont calculés, et contrôlés par simulation, par le script `verif/seance_quant_1.py` ; la seule expérience affichée (double descente) est exécutée à la construction de cette page.

**Conventions de notation**, valables pour toute la leçon :

- une majuscule désigne une variable aléatoire, une minuscule une valeur ou un paramètre fixe, le gras un vecteur ou une matrice ; une matrice en gras est fixe sauf mention contraire ;
- \(n\) est un nombre d'observations, \(k\) un nombre de paramètres, \(p\) un nombre de variables explicatives ;
- exceptions d'usage, toutes des entiers fixes : \(N\) est un nombre d'actifs ou de paris, \(T\) un nombre de dates ou d'années, \(P\) le nombre total de variables disponibles (partie 2) ; les sigles AIC, BIC, IC (coefficient d'information) et IR (ratio d'information) désignent des nombres calculés sur des données observées ;
- un estimateur à chapeau calculé à partir de variables aléatoires est aléatoire, même écrit avec une lettre grecque : \(\hat\theta\) ;
- \(\mathbb{P}\), \(\mathbb{E}\), \(\mathrm{Var}\), \(\mathrm{Cov}\) désignent probabilité, espérance, variance et covariance ; \(Z\) désigne toujours une variable de loi normale centrée réduite \(\mathcal{N}(0, 1)\), \(\Phi\) sa fonction de répartition et \(\varphi\) sa densité ; \(\mathbf{1}\) est le vecteur dont toutes les coordonnées valent 1, \(^\top\) la transposée, \(\mathrm{tr}\) la trace.

## Partie 1 : AIC et BIC

### Le problème

Un modèle à plus de paramètres s'ajuste toujours au moins aussi bien aux données d'entraînement : sa log-vraisemblance maximale ne peut que monter. Pour comparer des modèles de tailles différentes, il faut pénaliser le nombre de paramètres. AIC et BIC le font différemment, et ne répondent pas à la même question.

### Les définitions

Pour un modèle à \(k\) paramètres estimés sur \(n\) observations, on note \(\hat\ell\) sa log-vraisemblance maximale, un nombre calculé sur les données observées. Le critère d'Akaike (Akaike, 1974, IEEE Transactions on Automatic Control) et le critère bayésien de Schwarz (Schwarz, 1978, Annals of Statistics) valent :

\(\displaystyle \mathrm{AIC} = -2\hat\ell + 2k, \qquad \mathrm{BIC} = -2\hat\ell + k \ln n.\)

On retient le modèle de critère le plus **petit**. Chaque paramètre coûte 2 pour l'AIC et \(\ln n\) pour le BIC : le BIC pénalise davantage dès que \(\ln n > 2\), c'est-à-dire \(n > e^2 \approx @L_e2:3m@\), soit dès \(n = @L_n_seuil@\).

### Ce que chacun estime

1. (A1) **AIC, cas démontré.** Modèle linéaire gaussien à plan fixe : \(\mathbf{Y} = \boldsymbol{\mu} + \boldsymbol{\varepsilon}\), où \(\boldsymbol{\mu} \in \mathbb{R}^n\) est fixe, \(\boldsymbol{\varepsilon}\) suit \(\mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})\) avec \(\sigma^2\) connu ; le modèle candidat utilise les \(k\) colonnes d'une matrice fixe \(\mathbf{X}\) de rang \(k\), et il est juste : \(\boldsymbol{\mu}\) est dans l'image de \(\mathbf{X}\). On note \(\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\) la matrice chapeau, et \(\mathbf{Y}'\) de nouvelles données de même loi, indépendantes de \(\mathbf{Y}\). Alors la déviance hors échantillon du modèle ajusté, \(-2\) fois la log-vraisemblance de \(\mathbf{Y}'\) évaluée en l'estimateur, a pour espérance \(\mathbb{E}[\mathrm{AIC}]\) : l'AIC estime sans biais l'erreur de prédiction sur de nouvelles données.
2. Hors de ce cas, le résultat d'Akaike n'est qu'asymptotique, sous des conditions de régularité (admis).
3. (A2) **BIC.** Il vient d'une approximation de la vraisemblance marginale d'un modèle en statistique bayésienne (admis). Sa propriété : si le vrai modèle est l'un des candidats, de dimension finie, le BIC le choisit avec une probabilité qui tend vers 1 quand \(n\) grandit ; l'AIC, non. Cas démontré ci-dessous.

### La preuve de (A1)

1. La log-vraisemblance gaussienne d'un vecteur \(\mathbf{y}\) de moyenne \(\mathbf{m}\) vaut \(-\tfrac{n}{2}\ln(2\pi\sigma^2) - \lVert \mathbf{y} - \mathbf{m} \rVert^2/(2\sigma^2)\). Son maximum sur les \(\mathbf{m}\) de l'image de \(\mathbf{X}\) est atteint en \(\mathbf{H}\mathbf{y}\) (projection orthogonale, leçon de régression linéaire, `outils/lecons_reglin.py`, partie 1). Donc \(-2\hat\ell = n\ln(2\pi\sigma^2) + \lVert \mathbf{Y} - \mathbf{H}\mathbf{Y} \rVert^2/\sigma^2\), et la déviance hors échantillon vaut \(n\ln(2\pi\sigma^2) + \lVert \mathbf{Y}' - \mathbf{H}\mathbf{Y} \rVert^2/\sigma^2\).
2. Entraînement : \(\mathbf{Y} - \mathbf{H}\mathbf{Y} = (\mathbf{I} - \mathbf{H})\boldsymbol{\varepsilon}\), car \(\mathbf{H}\boldsymbol{\mu} = \boldsymbol{\mu}\). Son carré moyen vaut \(\sigma^2(n - k)\) (même leçon, partie 2, preuve, étape 3 : trace de \(\mathbf{I} - \mathbf{H}\)).
3. Hors échantillon : \(\mathbf{Y}' - \mathbf{H}\mathbf{Y} = \boldsymbol{\varepsilon}' - \mathbf{H}\boldsymbol{\varepsilon}\). En développant, \(\mathbb{E}\lVert \boldsymbol{\varepsilon}' \rVert^2 = n\sigma^2\) ; le double produit est d'espérance nulle, car \(\boldsymbol{\varepsilon}'\) est indépendant de \(\boldsymbol{\varepsilon}\) et centré ; et \(\mathbb{E}\lVert \mathbf{H}\boldsymbol{\varepsilon} \rVert^2 = \mathbb{E}[\boldsymbol{\varepsilon}^\top \mathbf{H}\boldsymbol{\varepsilon}] = \sigma^2\,\mathrm{tr}(\mathbf{H}) = \sigma^2 k\), car \(\mathbf{H}\) est symétrique et idempotente, de trace \(k\). Total : \(\sigma^2(n + k)\).
4. La différence des espérances vaut \(2k\sigma^2\), soit \(2k\) une fois divisée par \(\sigma^2\) : \(\mathbb{E}[\text{déviance hors échantillon}] = \mathbb{E}[-2\hat\ell] + 2k = \mathbb{E}[\mathrm{AIC}]\).

Contrôle par simulation (script, \(n = @L_opt_n@\), \(k = @L_opt_k@\), \(\sigma^2 = 1\)) : écart moyen entre erreur hors échantillon et erreur d'entraînement \(@L_opt_mc:2m@\), pour une valeur exacte \(2k\sigma^2 = @L_opt_exact@\).

### Le cas où ils divergent, démontré

Deux modèles emboîtés, M0 à \(k\) colonnes et M1 à \(k + 1\) colonnes (celles de M0 plus une), même cadre gaussien, et le vrai \(\boldsymbol{\mu}\) est dans l'image de M0 : la colonne ajoutée est inutile. On note \(\mathbf{H}_0\) et \(\mathbf{H}_1\) leurs matrices chapeau.

1. L'image de M0 est incluse dans celle de M1, donc \(\mathbf{H}_1\mathbf{H}_0 = \mathbf{H}_0\) (projeter un vecteur déjà dans l'image ne le change pas), et, en transposant, \(\mathbf{H}_0\mathbf{H}_1 = \mathbf{H}_0\).
2. \(\mathbf{I} - \mathbf{H}_0 = (\mathbf{I} - \mathbf{H}_1) + (\mathbf{H}_1 - \mathbf{H}_0)\), et le produit des deux termes est nul par l'étape 1 ; par Pythagore, la baisse de la somme des carrés des résidus vaut \(\lVert (\mathbf{H}_1 - \mathbf{H}_0)\mathbf{Y} \rVert^2\), et \(2(\hat\ell_1 - \hat\ell_0) = \lVert (\mathbf{H}_1 - \mathbf{H}_0)\mathbf{Y} \rVert^2/\sigma^2\).
3. \(\mathbf{H}_1 - \mathbf{H}_0\) est symétrique, idempotente (développer avec l'étape 1) et de trace 1 : ses valeurs propres valent 0 ou 1 (\(\lambda^2 = \lambda\)), une seule vaut 1, donc par le théorème spectral \(\mathbf{H}_1 - \mathbf{H}_0 = \mathbf{u}\mathbf{u}^\top\) pour un vecteur unitaire \(\mathbf{u}\). Comme \((\mathbf{H}_1 - \mathbf{H}_0)\boldsymbol{\mu} = \boldsymbol{\mu} - \boldsymbol{\mu} = \mathbf{0}\), on obtient \(2(\hat\ell_1 - \hat\ell_0) = (\mathbf{u}^\top\boldsymbol{\varepsilon}/\sigma)^2 = Z^2\), car \(\mathbf{u}^\top\boldsymbol{\varepsilon}\) est gaussienne centrée de variance \(\sigma^2\lVert \mathbf{u} \rVert^2 = \sigma^2\).
4. L'AIC choisit M1 si et seulement si \(Z^2 > 2\) : probabilité \(\mathbb{P}(\lvert Z \rvert > \sqrt2) \approx @L_p_aic:4m@\), **quel que soit** \(n\). Le BIC choisit M1 si et seulement si \(Z^2 > \ln n\) : @L_p_bic_100:4@ pour \(n = 100\), @L_p_bic_1000:4@ pour \(n = 1000\), et cette probabilité tend vers 0.

Simulation de la régression elle-même (script, \(n = 100\)) : l'AIC ajoute la variable inutile dans une proportion @L_p_aic_mc:4@ des cas, le BIC dans @L_p_bic_100_mc:4@.

Règle générale : si le grand modèle a \(\Delta k\) paramètres de plus et une log-vraisemblance maximale plus grande de \(\Delta\ell\), l'AIC le préfère si \(2\Delta\ell > 2\Delta k\), le BIC si \(2\Delta\ell > \Delta k \ln n\). Ils divergent quand \(2\Delta k < 2\Delta\ell < \Delta k \ln n\).

### Pièges

1. On ne compare des AIC que pour des modèles ajustés sur les **mêmes** observations de la **même** variable cible (pas \(y\) contre \(\ln y\)). Seules les différences comptent.
2. \(k\) compte tous les paramètres estimés (constante, variance du bruit si elle est estimée), de la même façon pour tous les modèles comparés.
3. AIC : vise la prédiction, sur-ajuste avec une probabilité qui ne s'annule pas. BIC : vise à retrouver le vrai modèle s'il est parmi les candidats, et choisit des modèles au plus aussi grands que l'AIC dès \(n \ge @L_n_seuil@\).

[[controle:aic_bic]]

## Partie 2 : la double descente

### Le problème

La courbe en U classique dit qu'au-delà d'une certaine complexité l'erreur de test remonte, et qu'un modèle qui interpole les données d'entraînement (erreur nulle) est mauvais. Or des modèles avec bien plus de paramètres que d'observations prédisent bien. Belkin, Hsu, Ma et Mandal (PNAS, 2019) ont nommé « double descente » la courbe qui réconcilie les deux constats.

### L'expérience

\(n = 40\) observations d'entraînement, \(P = 200\) variables gaussiennes indépendantes de variance 1, cible \(y = \mathbf{x}^\top\boldsymbol{\beta} + \) bruit d'écart-type 0,5, avec \(\lVert \boldsymbol{\beta} \rVert = 1\). On ajuste les moindres carrés sur les \(p\) premières variables seulement ; pour \(p > n\), il existe une infinité de solutions, et `np.linalg.lstsq` renvoie celle de **norme minimale**. Deux profils : signal concentré sur les premières variables (\(\beta_j\) proportionnel à \(1/j\)) ou réparti (\(\beta_j\) constant). Pour chaque \(p\) : erreur de test médiane sur 100 échantillons d'entraînement, plus grande erreur d'entraînement, et plus petite valeur singulière médiane de la matrice des données.

```python-exec
import numpy as np

def double_descente(profil):
    g = np.random.default_rng(0)
    n, P, sigma, R = 40, 200, 0.5, 100
    j = np.arange(1, P + 1)
    beta = 1.0 / j if profil == "concentre" else np.ones(P)
    beta = beta / np.linalg.norm(beta)
    ps = [2, 5, 10, 20, 30, 38, 40, 42, 50, 80, 120, 200]
    Xt = g.standard_normal((5000, P))                      # points de test
    yt = Xt.dot(beta) + sigma * g.standard_normal(5000)
    test, entr, smin = {p: [] for p in ps}, {p: [] for p in ps}, {p: [] for p in ps}
    for _ in range(R):
        X = g.standard_normal((n, P))
        y = X.dot(beta) + sigma * g.standard_normal(n)
        for p in ps:
            b = np.linalg.lstsq(X[:, :p], y, rcond=None)[0]  # norme minimale si p > n
            test[p].append(np.mean((Xt[:, :p].dot(b) - yt) ** 2))
            entr[p].append(np.mean((X[:, :p].dot(b) - y) ** 2))
            smin[p].append(np.linalg.svd(X[:, :p], compute_uv=False).min())
    print("profil", profil)
    for p in ps:
        print(f"  p = {p:3d}: test (médiane) {np.median(test[p]):8.3f}   entraînement (max) {np.max(entr[p]):.1e}   "
              f"plus petite valeur singulière {np.median(smin[p]):.3f}")

double_descente("concentre")
double_descente("reparti")
```

Lecture, contrôlée par les assertions du script sur la même expérience : l'erreur d'entraînement est nulle (aux arrondis près) dès \(p \ge n = @L_dd_n@\) ; l'erreur de test culmine exactement au seuil d'interpolation \(p = n\), là où la plus petite valeur singulière médiane est minimale ; au-delà, elle redescend. Profil concentré : une première descente puis une remontée avant le pic, et le meilleur modèle reste un modèle sous-paramétré. Profil réparti : le meilleur modèle est le plus sur-paramétré, \(p = 200\).

### Ce qui est démontré

1. (DD1) Si \(p \ge n\) et si la matrice \(\mathbf{X}\) (\(n \times p\)) est de rang \(n\), les interpolateurs (\(\mathbf{X}\mathbf{b} = \mathbf{y}\)) existent, et celui de norme minimale est \(\mathbf{b}^\star = \mathbf{X}^\top(\mathbf{X}\mathbf{X}^\top)^{-1}\mathbf{y}\). Preuve : \(\mathbf{X}\mathbf{b}^\star = \mathbf{y}\) ; tout autre interpolateur s'écrit \(\mathbf{b}^\star + \mathbf{h}\) avec \(\mathbf{X}\mathbf{h} = \mathbf{0}\) ; \(\mathbf{b}^\star = \mathbf{X}^\top \mathbf{c}\) avec \(\mathbf{c} = (\mathbf{X}\mathbf{X}^\top)^{-1}\mathbf{y}\), donc \(\mathbf{h}^\top\mathbf{b}^\star = (\mathbf{X}\mathbf{h})^\top\mathbf{c} = 0\), et par Pythagore \(\lVert \mathbf{b}^\star + \mathbf{h} \rVert^2 = \lVert \mathbf{b}^\star \rVert^2 + \lVert \mathbf{h} \rVert^2\).
2. (DD2) Amplification du bruit, à \(\mathbf{X}\) fixé (l'espérance porte sur le bruit seul). Soient \(s_1, \dots, s_r\) les valeurs singulières non nulles de \(\mathbf{X}\), \(r = \min(n, p)\), rang plein. Si \(\boldsymbol{\varepsilon}\) est un bruit centré de covariance \(\sigma^2\mathbf{I}\), la part de l'estimateur due au bruit, \(\mathbf{X}^{+}\boldsymbol{\varepsilon}\) (où \(\mathbf{X}^{+}\) vaut \((\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\) si \(p \le n\) et \(\mathbf{X}^\top(\mathbf{X}\mathbf{X}^\top)^{-1}\) si \(p \ge n\)), vérifie \(\mathbb{E}\lVert \mathbf{X}^{+}\boldsymbol{\varepsilon} \rVert^2 = \sigma^2 \sum_{j=1}^{r} 1/s_j^2\). Preuve : \(\mathbb{E}\lVert \mathbf{M}\boldsymbol{\varepsilon} \rVert^2 = \sigma^2\,\mathrm{tr}(\mathbf{M}\mathbf{M}^\top)\) ; \(\mathbf{X}^{+}\mathbf{X}^{+\top}\) vaut \((\mathbf{X}^\top\mathbf{X})^{-1}\) ou \((\mathbf{X}\mathbf{X}^\top)^{-1}\), dont les valeurs propres sont les \(1/s_j^2\). Pour un nouveau point \(\mathbf{x}\) centré, de covariance \(\mathbf{I}\), indépendant des données, l'erreur quadratique en excès d'un estimateur \(\hat{\mathbf{b}}\) vaut \(\lVert \hat{\mathbf{b}} - \boldsymbol{\beta} \rVert^2\) ; en moyenne sur le bruit, la part \(\sigma^2\sum_j 1/s_j^2\) s'ajoute exactement au reste (le terme croisé est d'espérance nulle). Une seule petite valeur singulière suffit à faire exploser l'erreur. Dans l'expérience, les variables non utilisées, indépendantes des autres, jouent le rôle d'un bruit supplémentaire de variance \(\sum_{j > p}\beta_j^2\), qui s'ajoute à \(\sigma^2\).

### Ce qui n'est pas démontré ici

- Que la plus petite valeur singulière d'une matrice gaussienne \(n \times p\) soit proche de 0 quand \(p\) est proche de \(n\) : c'est **constaté** dans la dernière colonne. Avec (DD2), cela explique le pic.
- La forme exacte de la courbe, et quand la seconde descente bat la première : établies asymptotiquement par Hastie, Montanari, Rosset et Tibshirani (Annals of Statistics, 2022) pour la régression sans pénalité à norme minimale ; ici, c'est un constat sur deux profils, et les deux conclusions diffèrent.

### Pièges

1. « L'interpolation est toujours à proscrire » : faux sur le profil réparti. « Plus de paramètres est toujours mieux » : faux sur le profil concentré.
2. Le pic est un artefact de l'estimateur sans régularisation au seuil d'interpolation, pas une propriété de la complexité en général.
3. Lien quantitatif : 40 dates et 200 signaux placent une régression exactement dans ce régime.

[[controle:double_descente]]

## Partie 3 : importance par permutation et valeurs de Shapley

### Importance par permutation et variables corrélées

Définitions, biais de l'importance par impureté, espérance nulle pour une variable indépendante (I2) et dilution entre variables redondantes (I3) : leçon sur les arbres, `outils/lecons_arbres.py`, partie 7. On ajoute ici un résultat exact qui montre d'où vient la dilution.

(P1) Modèle linéaire \(f(\mathbf{x}) = b_0 + \sum_j b_j x_j\), données \(Y = f(\mathbf{X}) + \varepsilon\) avec \(\varepsilon\) centré et indépendant de \(\mathbf{X}\) (le modèle est juste). On idéalise la permutation de la variable \(j\) sur un grand jeu de validation par son remplacement par une copie \(\tilde X_j\), indépendante de tout le reste et de même loi que \(X_j\). Alors la hausse d'erreur quadratique espérée vaut \(2\,b_j^2\,\mathrm{Var}(X_j)\).

Preuve : \(Y - f(\tilde{\mathbf{X}}) = \varepsilon + b_j(X_j - \tilde X_j)\). Le double produit est d'espérance nulle (\(\varepsilon\) centré, indépendant de \(X_j\) et de \(\tilde X_j\)) ; \(\mathbb{E}[(X_j - \tilde X_j)^2] = 2\,\mathrm{Var}(X_j)\), car les deux sont indépendantes de même loi. L'erreur passe de \(\mathrm{Var}(\varepsilon)\) à \(\mathrm{Var}(\varepsilon) + 2b_j^2\,\mathrm{Var}(X_j)\).

Conséquence, variables corrélées : si \(X_2 = X_1\) exactement, les modèles \(x_1\) et \(\tfrac12 x_1 + \tfrac12 x_2\) font les **mêmes** prédictions, mais (P1) leur donne des importances différentes, \((@Q02_A:0m@ ; 0)\) contre \((@Q02_exact:1m@ ; @Q02_exact:1m@)\) pour \(\mathrm{Var}(X_1) = 1\) (simulation du script : @Q02_A_mc:3@ et @Q02_exact_mc:3@). L'importance mesure le modèle choisi, pas l'information des données. Quand la corrélation est forte sans être parfaite, permuter \(X_1\) fabrique en plus des couples \((\tilde X_1, X_2)\) absents des données : le modèle est évalué là où il n'a jamais été entraîné. Remèdes possibles, non traités ici : permuter ensemble un groupe de variables corrélées.

### Valeurs de Shapley : définition

**Le problème** : répartir entre les \(p\) variables, de façon équitable, l'écart entre la prédiction en un point \(\mathbf{x}\) et la prédiction moyenne.

On note \(F = \{1, \dots, p\}\) l'ensemble des variables, et, pour toute coalition \(S \subset F\), \(v(S)\) la prédiction attendue quand on ne connaît que les coordonnées de \(\mathbf{x}\) indicées par \(S\). Deux choix usuels : la valeur **conditionnelle** \(v(S) = \mathbb{E}[f(\mathbf{X}) \mid \mathbf{X}_S = \mathbf{x}_S]\), et la valeur **interventionnelle**, où les variables hors de \(S\) sont tirées selon leur loi marginale, indépendamment de \(\mathbf{x}_S\). La valeur de Shapley de la variable \(j\) (Shapley, 1953 ; appliquée aux modèles par Lundberg et Lee, NeurIPS 2017, méthode SHAP) est :

\(\displaystyle \phi_j = \sum_{S \subset F \setminus \{j\}} \frac{\lvert S \rvert!\,(p - \lvert S \rvert - 1)!}{p!}\,\big[v(S \cup \{j\}) - v(S)\big].\)

Lecture : on fait entrer les variables une à une dans un ordre tiré uniformément parmi les \(p!\) ordres ; \(\phi_j\) est l'espérance de la contribution marginale de \(j\) à son arrivée. Le poids est exactement la probabilité que l'ensemble des variables arrivées avant \(j\) soit \(S\) : \(\lvert S \rvert!\) façons de ranger \(S\) avant \(j\), puis \((p - \lvert S \rvert - 1)!\) façons de ranger le reste après, sur \(p!\) ordres. Ces poids somment donc à 1 (contrôlé par le script pour \(p = 1\) à 8).

### Les quatre axiomes

1. **Efficacité** : \(\sum_j \phi_j = v(F) - v(\emptyset)\). Preuve : pour un ordre fixé, les contributions marginales successives se télescopent en \(v(F) - v(\emptyset)\) ; on prend l'espérance sur l'ordre.
2. **Symétrie** : deux variables interchangeables dans \(v\) reçoivent la même valeur.
3. **Joueur nul** : si \(v(S \cup \{j\}) = v(S)\) pour tout \(S\), alors \(\phi_j = 0\) (chaque crochet est nul).
4. **Additivité** : \(\phi\) est linéaire en \(v\) (la formule l'est).

Théorème de Shapley (1953), admis : \(\phi\) est la **seule** répartition qui vérifie les quatre axiomes.

### Cas linéaire à variables indépendantes, démontré

Énoncé : si \(f(\mathbf{x}) = b_0 + \sum_k b_k x_k\) et si les coordonnées de \(\mathbf{X}\) sont indépendantes, d'espérances \(\mu_k\), alors, avec la valeur conditionnelle, \(\phi_j = b_j(x_j - \mu_j)\).

Preuve :

1. Par indépendance, \(\mathbb{E}[X_k \mid \mathbf{X}_S = \mathbf{x}_S] = \mu_k\) pour \(k \notin S\) ; par linéarité, \(v(S) = b_0 + \sum_{k \in S} b_k x_k + \sum_{k \notin S} b_k \mu_k\).
2. Pour tout \(S\) qui ne contient pas \(j\), \(v(S \cup \{j\}) - v(S) = b_j(x_j - \mu_j)\) : la contribution ne dépend pas de \(S\).
3. Les poids somment à 1, donc \(\phi_j = b_j(x_j - \mu_j)\).

Exemple calculé (script, par la formule des coalitions) : \(f(\mathbf{x}) = 1 + 2x_1 - 3x_2 + 0{,}5\,x_3\), \(\boldsymbol{\mu} = (1, 0, 2)\), \(\mathbf{x} = (2, 1, 2)\). Valeurs : \(@L_sh_phi1@\), \(@L_sh_phi2@\), \(@L_sh_phi3@\). Efficacité : leur somme \(@L_sh_somme@\) égale \(f(\mathbf{x}) - \mathbb{E}f(\mathbf{X}) = @L_sh_fx@ - @L_sh_ef@\). La troisième variable a un coefficient non nul mais vaut sa moyenne : contribution nulle.

Avec deux variables, la formule se réduit à \(\phi_1 = \tfrac12\big[v(\{1\}) - v(\emptyset)\big] + \tfrac12\big[v(\{1, 2\}) - v(\{2\})\big]\) : la moyenne de la contribution de la variable 1 arrivée première et arrivée seconde. Pour un modèle avec interaction, ces deux contributions diffèrent, et l'interaction est partagée.

### Pièges

1. Variables corrélées, valeur conditionnelle : soit \(X_2 = X_1\), \(\mathbb{E}[X_1] = 1\), \(f(\mathbf{x}) = x_1\) et le point \(\mathbf{x} = (3, 3)\). Connaître \(x_2\) revient à connaître \(x_1\), donc \(v(\{2\}) = 3\) : on obtient \(\phi = @L_sh_cond@\), et la variable 2, absente du modèle, reçoit une part. La valeur interventionnelle donne \(@L_sh_interv@\). Les deux répondent à des questions différentes : « que dit cette information ? » contre « que fait le modèle de cette entrée ? ».
2. \(\phi_j = b_j\,x_j\) est faux : il faut retrancher la moyenne.
3. Le calcul exact demande \(2^{p}\) coalitions : on l'estime en tirant des ordres au hasard.

[[controle:shapley]]

## Partie 4 : coefficient d'information et loi fondamentale

### Définition

À une date donnée, \(N\) actifs ; pour l'actif \(i\), \(s_i\) est la valeur du signal, connue à cette date, et \(r_i\) le rendement sur la période suivante, observé plus tard. Le **coefficient d'information** (IC) de la date est la corrélation de rang de Spearman entre \((s_i)\) et \((r_i)\) : la corrélation de Pearson entre les rangs \(a_i\) des \(s_i\) et les rangs \(b_i\) des \(r_i\) (rang 1 pour la plus petite valeur, rang moyen en cas d'égalité, comme `rank()` de pandas). Il mesure si le signal **classe** bien les actifs, sans dépendre de l'échelle du signal ni des rendements extrêmes.

### Formule sans égalités, démontrée

Énoncé : sans égalités, avec \(d_i = a_i - b_i\), \(\mathrm{IC} = 1 - \dfrac{6\sum_i d_i^2}{N(N^2 - 1)}\).

1. Les \(a_i\) sont une permutation de \(1, \dots, N\) : moyenne \(\bar a = (N+1)/2\), et \(\sum_i a_i^2 = N(N+1)(2N+1)/6\), d'où une variance (division par \(N\)) \(\kappa = \sum_i a_i^2/N - \bar a^2 = (N^2 - 1)/12\). Idem pour les \(b_i\).
2. \(\sum_i d_i^2 = \sum_i a_i^2 + \sum_i b_i^2 - 2\sum_i a_i b_i\), donc \(\sum_i a_i b_i/N = \sum_i a_i^2/N - \sum_i d_i^2/(2N)\).
3. La corrélation vaut \(\big(\sum_i a_i b_i/N - \bar a^2\big)/\kappa = \big(\kappa - \sum_i d_i^2/(2N)\big)/\kappa = 1 - \sum_i d_i^2/(2N\kappa)\), et \(2N\kappa = N(N^2 - 1)/6\).

Exemple : signal \((0{,}3 ; -1{,}0 ; 2{,}1 ; 0{,}7 ; -0{,}2)\), rendements \((0{,}5 ; -1{,}1 ; 0{,}2 ; 1{,}4 ; -0{,}3)\). Rangs \(@L_ic_rangs_s@\) et \(@L_ic_rangs_r@\), écarts \(@L_ic_d@\), \(\sum d_i^2 = @L_ic_d2@\), \(\mathrm{IC} = 1 - 6 \times @L_ic_d2@/@L_ic_N3@ = @L_ic_ex:1m@\).

### Loi sous l'hypothèse nulle, démontrée

Énoncé : si le signal n'a aucun lien avec les rendements (les rangs \(b\) sont une permutation uniforme, indépendante des \(a\)), alors \(\mathbb{E}[\mathrm{IC}] = 0\) et \(\mathrm{Var}(\mathrm{IC}) = 1/(N - 1)\) exactement. L'IC est ici vu comme une variable aléatoire.

1. On centre : \(a'_i = a_i - \bar a\), \(b'_i = b_i - \bar a\), et \(\mathrm{IC} = \sum_i a'_i B'_i/(N\kappa)\), où \(B'_i\) est le rang centré aléatoirement attribué à l'actif \(i\).
2. \(\mathbb{E}[B'_i] = 0\) par symétrie ; \(\mathbb{E}[B'^2_i] = \sum_k b'^2_k/N\) ; pour \(i \ne i'\), \(\mathbb{E}[B'_i B'_{i'}] = \sum_{k \ne l} b'_k b'_l/(N(N-1)) = -\sum_k b'^2_k/(N(N-1))\), car \(\sum_{k \ne l} b'_k b'_l = (\sum_k b'_k)^2 - \sum_k b'^2_k\) et \(\sum_k b'_k = 0\).
3. \(\mathrm{Var}\big(\sum_i a'_i B'_i\big) = \tfrac{\sum_k b'^2_k}{N}\big[\sum_i a'^2_i - \tfrac{1}{N-1}\sum_{i \ne i'} a'_i a'_{i'}\big] = \tfrac{\sum_k b'^2_k}{N}\,\sum_i a'^2_i\,\big[1 + \tfrac{1}{N-1}\big] = \tfrac{(N\kappa)^2}{N-1}\), en utilisant de même \(\sum_{i \ne i'} a'_i a'_{i'} = -\sum_i a'^2_i\) et \(\sum_i a'^2_i = \sum_k b'^2_k = N\kappa\). On divise par \((N\kappa)^2\).

Ordres de grandeur : avec \(N = 500\) actifs, l'écart-type d'un IC sans effet vaut \(1/\sqrt{499} \approx @L_ic_se500:4m@\) ; la moyenne de 250 IC de dates **indépendantes** a un écart-type \(@L_ic_se500_250:4m@\). Un IC moyen de 0,02 est donc très significatif sur 250 dates indépendantes, et rien du tout sur une seule date. L'indépendance entre dates tombe avec des étiquettes qui se chevauchent (partie 5).

### Loi fondamentale de la gestion active

Énoncé de Grinold (« The Fundamental Law of Active Management », Journal of Portfolio Management, vol. 15, n° 3, 1989) : \(\mathrm{IR} \approx \mathrm{IC} \times \sqrt{\mathrm{BR}}\), où IR est le ratio d'information (rendement actif moyen divisé par son écart-type), BR (breadth) le nombre de paris **indépendants** par an, et IC la corrélation entre prévisions et rendements, la même pour tous les paris. C'est une approximation, qui suppose en outre une utilisation optimale des prévisions et un IC petit.

Version démontrée ici, sous hypothèses explicites. Une période, \(N\) paris indépendants. Pour le pari \(i\) : \(G_i\) est le signal standardisé, de loi \(\mathcal{N}(0, 1)\) ; le rendement est \(R_i = \omega\,\big(c\,G_i + \sqrt{1 - c^2}\,E_i\big)\), où \(E_i\) est un bruit \(\mathcal{N}(0, 1)\) indépendant, \(\omega > 0\) la volatilité et \(c \in [0, 1[\) la corrélation entre signal et rendement ; tous les \(G_i, E_i\) sont indépendants. Position proportionnelle au signal : gain \(\Pi = \sum_i G_i R_i\). Alors \(\mathrm{IR} = \mathbb{E}[\Pi]/\sqrt{\mathrm{Var}(\Pi)} = c\sqrt{N}/\sqrt{1 + c^2}\).

1. \(\mathbb{E}[G_i R_i] = \omega c\,\mathbb{E}[G_i^2] = \omega c\).
2. \(\mathrm{Var}(G_i R_i) = \omega^2\big[c^2\,\mathrm{Var}(G_i^2) + (1 - c^2)\,\mathrm{Var}(G_i E_i) + 2c\sqrt{1 - c^2}\,\mathrm{Cov}(G_i^2, G_i E_i)\big]\), avec \(\mathrm{Var}(G_i^2) = \mathbb{E}[G_i^4] - 1 = 2\), \(\mathrm{Var}(G_i E_i) = \mathbb{E}[G_i^2]\,\mathbb{E}[E_i^2] = 1\), et \(\mathrm{Cov}(G_i^2, G_i E_i) = \mathbb{E}[G_i^3]\,\mathbb{E}[E_i] = 0\). Total : \(\omega^2(1 + c^2)\).
3. Les \(N\) paris sont indépendants : espérance \(N\omega c\), variance \(N\omega^2(1 + c^2)\), d'où le ratio.

Exemple : \(c = @L_lf_c:2m@\), \(N = @L_lf_N@\) : \(\mathrm{IR} = @L_lf_ir:4m@\) par période (approximation \(c\sqrt{N} = @L_lf_approx:1m@\) ; simulation @L_lf_ir_mc:4@). Sur 12 périodes indépendantes par an, l'espérance et la variance sont multipliées par 12, l'IR par \(\sqrt{12}\) : \(@L_lf_ir_an:2m@\).

Pièges : des paris corrélés (même secteur, même facteur) comptent pour moins de \(N\) paris indépendants ; l'IC réel varie dans le temps ; les coûts de transaction ne sont pas dans la formule.

## Partie 5 : étiquettes qui se chevauchent, non-stationnarité

### Étiquettes qui se chevauchent

**Le problème** : on prédit le rendement des \(h\) prochains jours, et on calcule cette étiquette chaque jour. Deux étiquettes voisines partagent \(h - 1\) rendements quotidiens : leur corrélation vaut \((h - 1)/h\) si les rendements quotidiens sont non corrélés et de même variance (leçon de régression linéaire, `outils/lecons_reglin.py`, partie 2, autocorrélation, point 4).

Énoncé. Rendements quotidiens \(X_1, X_2, \dots\) non corrélés, de variance \(v\) ; étiquettes \(Y_t = X_t + \dots + X_{t+h-1}\) pour \(t = 1, \dots, T\), avec \(T \ge h\) ; \(\bar Y\) leur moyenne. Alors

\(\displaystyle \mathrm{Var}(\bar Y) = \frac{v}{T^2}\Big[(T - h + 1)\,h^2 + 2\sum_{k=1}^{h-1} k^2\Big],\)

alors que la formule naïve, qui traite les \(T\) étiquettes (chacune de variance \(hv\)) comme indépendantes, donne \(hv/T\). Le rapport des deux tend vers \(h\) quand \(T \to \infty\).

1. \(\bar Y = \tfrac1T \sum_s c_s X_s\), où \(c_s\) est le nombre d'étiquettes qui contiennent le jour \(s\) : \(c_s = h\) pour \(h \le s \le T\) (\(T - h + 1\) jours), \(c_s = s\) pour \(s < h\), et \(c_s\) décroît de \(h - 1\) à 1 pour \(s > T\).
2. Les \(X_s\) étant non corrélés, \(\mathrm{Var}(\bar Y) = \tfrac{v}{T^2}\sum_s c_s^2\), d'où la formule.
3. Le rapport à \(hv/T\) vaut \(\big[(T - h + 1)h^2 + 2\sum_{k<h} k^2\big]/(Th)\), qui tend vers \(h\).

Exemple : \(h = @L_ch_h@\), \(T = @L_ch_T@\) : rapport des variances \(@L_ch_rapport_var:3m@\), donc l'erreur-type naïve est trop petite d'un facteur \(@L_ch_rapport_se:3m@\), et une statistique t calculée naïvement est gonflée d'autant. Simulation (script) : variance de la moyenne @L_ch_var_moyenne_mc:5@ pour une valeur exacte @L_ch_var_moyenne_exact:5@, avec \(v = 1\).

Conséquences :

1. Les tests sur des étiquettes chevauchantes demandent une erreur-type corrigée (Newey-West, même leçon, même partie) ou des étiquettes non chevauchantes, une tous les \(h\) jours.
2. Validation croisée : une observation de test dont la fenêtre d'étiquette recoupe celle d'une observation d'entraînement fuit de l'information. On **purge** l'entraînement des observations dont l'étiquette recoupe la période de test, et on ajoute un **embargo**, une marge d'observations retirées après la période de test (López de Prado, Advances in Financial Machine Learning, Wiley, 2018).

### Non-stationnarité

**Définition** : une série \((X_t)\) est faiblement stationnaire si \(\mathbb{E}[X_t]\) et \(\mathrm{Cov}(X_t, X_{t+j})\), pour chaque décalage entier \(j\), ne dépendent pas de \(t\). Un prix n'en est pas une (sa variance croît avec le temps pour une marche aléatoire) ; on modélise les rendements, et même eux changent de régime.

Énoncé démontré : \(X_i = \mu_i + e_i\), \(i = 1, \dots, n\), avec \(e_i\) indépendantes, centrées, de variance \(\sigma^2\), et des moyennes \(\mu_i\) fixes. Avec \(\bar\mu\) la moyenne des \(\mu_i\) et \(S^2\) la variance empirique (division par \(n - 1\)) : \(\mathbb{E}[S^2] = \sigma^2 + \sum_i (\mu_i - \bar\mu)^2/(n - 1)\).

Preuve : \(\sum_i (X_i - \bar X)^2 = \sum_i (\mu_i - \bar\mu)^2 + 2\sum_i (\mu_i - \bar\mu)(e_i - \bar e) + \sum_i (e_i - \bar e)^2\), où \(\bar X\), \(\bar e\) sont les moyennes empiriques ; le terme croisé est d'espérance nulle (\(e\) centrées), le dernier a pour espérance \((n - 1)\sigma^2\) (variance empirique sans biais).

Exemple : une rupture de moyenne d'amplitude \(\Delta\) au milieu de l'échantillon (\(n\) pair) donne \(\sum_i (\mu_i - \bar\mu)^2 = n\Delta^2/4\). Avec \(\sigma = 1\), \(\Delta = @L_ns_delta:0m@\), \(n = @L_ns_n@\) : \(\mathbb{E}[S^2] = @L_ns_es2:4m@\) (simulation @L_ns_es2_mc:4@). La volatilité estimée double presque alors qu'aucun jour n'est plus risqué ; et la moyenne de l'échantillon, \((\mu_1 + \mu_2)/2\), n'est la moyenne d'aucun des deux régimes. D'où les validations glissantes (walk-forward), qui n'estiment qu'avec le passé de chaque date, et le compromis de longueur de fenêtre : longue, moins de variance ; courte, moins de retard sur un changement de régime.

[[controle:non_stationnarite]]

## Partie 6 : biais de sélection, ratio de Sharpe

### Le meilleur de \(m\) stratégies sans effet

**Le problème** : on teste \(m\) stratégies, aucune n'a d'effet réel, et l'on retient la meilleure. Son ratio de Sharpe estimé est positif par construction.

Notations : \(T\) années de rendements quotidiens, \(n = 252\,T\) jours ; pour chaque stratégie, rendements i.i.d. normaux d'espérance nulle ; \(\hat S_i\) est son ratio de Sharpe annualisé estimé, \(\sqrt{252}\) fois la moyenne divisée par l'écart-type empiriques. D'après la question DIAG-A5 du test de positionnement, \(\sqrt{T}\,\hat S_i\) est exactement la statistique de Student du test « espérance nulle », de loi de Student à \(n - 1\) degrés de liberté, très proche de \(\mathcal{N}(0, 1)\) pour \(n\) grand. Si les \(m\) stratégies sont indépendantes, \(\mathbb{E}[\max_i \hat S_i] \approx \mathbb{E}[M_m]/\sqrt{T}\), où \(M_m\) est le maximum de \(m\) variables \(\mathcal{N}(0, 1)\) indépendantes \(Z_1, \dots, Z_m\).

Énoncé : pour tout réel \(a\),

\(\displaystyle a\big(1 - \Phi(a)^m\big) - \varphi(a)\,\Phi(a)^{m-1} \;\le\; \mathbb{E}[M_m] \;\le\; \sqrt{2 \ln m}.\)

Preuve du majorant :

1. Pour \(t > 0\), la fonction exponentielle est convexe, donc par l'inégalité de Jensen \(e^{t\,\mathbb{E}[M_m]} \le \mathbb{E}[e^{t M_m}]\).
2. \(e^{t M_m} = \max_i e^{t Z_i} \le \sum_i e^{t Z_i}\), car tous les termes sont positifs.
3. \(\mathbb{E}[e^{t Z_i}] = e^{t^2/2}\) (on complète le carré : \(tx - x^2/2 = t^2/2 - (x - t)^2/2\), et la densité translatée intègre à 1).
4. Donc \(\mathbb{E}[M_m] \le \ln(m)/t + t/2\) ; le choix \(t = \sqrt{2\ln m}\) donne \(\sqrt{2\ln m}\).

Preuve du minorant :

1. \(\mathbb{E}[M_m] = \mathbb{E}[M_m I_{\{M_m \ge a\}}] + \mathbb{E}[M_m I_{\{M_m < a\}}]\), où \(I_A\) est l'indicatrice de l'événement \(A\).
2. Sur \(\{M_m \ge a\}\), \(M_m \ge a\) : le premier terme est au moins \(a\,\mathbb{P}(M_m \ge a) = a(1 - \Phi(a)^m)\), par indépendance.
3. Partout, \(M_m \ge Z_1\) : le second terme est au moins \(\mathbb{E}[Z_1 I_{\{M_m < a\}}]\). L'événement \(\{M_m < a\}\) est l'intersection des \(\{Z_i < a\}\) ; par indépendance, ce terme vaut \(\mathbb{E}[Z_1 I_{\{Z_1 < a\}}]\,\Phi(a)^{m-1}\), et \(\mathbb{E}[Z_1 I_{\{Z_1 < a\}}] = \int_{-\infty}^{a} x\varphi(x)\,dx = -\varphi(a)\), car \(\varphi'(x) = -x\varphi(x)\).

Valeurs (script : intégration numérique de \(\mathbb{E}[M_m] = \int x\,m\,\varphi(x)\Phi(x)^{m-1}dx\), simulation, et minorant optimisé en \(a\)) :

| \(m\) | minorant | \(\mathbb{E}[M_m]\) | simulation | majorant \(\sqrt{2\ln m}\) |
|---|---|---|---|---|
| 10 | @L_max_bas_10:3@ | @L_max_10:4@ | @L_max_10_mc:4@ | @L_max_haut_10:3@ |
| 100 | @L_max_bas_100:3@ | @L_max_100:4@ | @L_max_100_mc:4@ | @L_max_haut_100:3@ |
| 1000 | @L_max_bas_1000:3@ | @L_max_1000:4@ | @L_max_1000_mc:4@ | @L_max_haut_1000:3@ |

Exemple : 100 stratégies sans effet sur 10 ans donnent un meilleur Sharpe annualisé espéré de \(@L_max_100:4m@/\sqrt{10} \approx @L_sel_ex:2m@\). Remèdes : seuil corrigé pour \(m\) essais (Bonferroni, leçon de probabilités, `outils/lecons_probas.py`, outil 4), Sharpe « dégonflé » (deflated Sharpe ratio, Bailey et López de Prado, Journal of Portfolio Management, 2014), et une période de test jamais utilisée pour choisir. Piège : des stratégies corrélées comptent pour moins de \(m\) essais indépendants, et la formule surestime alors le biais ; ne pas compter les essais abandonnés le sous-estime.

### Erreur-type du ratio de Sharpe

Énoncé (Lo, « The Statistics of Sharpe Ratios », Financial Analysts Journal, vol. 58, n° 4, 2002). Rendements par période \(R_1, \dots, R_n\) i.i.d. normaux, d'espérance \(\mu\) et d'écart-type \(\sigma\) ; \(\theta = \mu/\sigma\) est le vrai ratio de Sharpe par période, \(\hat\theta = \bar R/S\) son estimateur (moyenne et écart-type empiriques). Alors \(\sqrt{n}(\hat\theta - \theta)\) converge en loi vers \(\mathcal{N}(0, 1 + \theta^2/2)\). Avec \(q\) périodes par an et \(T\) années (\(n = qT\)), le ratio annualisé \(\theta_a = \sqrt{q}\,\theta\) a pour erreur-type approchée

\(\displaystyle \mathrm{se}(\hat\theta_a) \approx \sqrt{\frac{1 + \theta_a^2/(2q)}{T}}.\)

Preuve par la méthode delta (admise : si un vecteur d'estimateurs est asymptotiquement normal, une fonction dérivable de ce vecteur l'est aussi, de variance \(\nabla g^\top \mathbf{C}\,\nabla g\), où \(\mathbf{C}\) est la matrice de covariance asymptotique et \(\nabla g\) le gradient de la fonction) :

1. \(\hat\theta = g(\bar R, S^2)\) avec \(g(u, w) = u/\sqrt{w}\). Au point \((\mu, \sigma^2)\), \(\partial g/\partial u = 1/\sigma\) et \(\partial g/\partial w = -\mu/(2\sigma^3)\).
2. À un terme en \(1/n\) près, \(S^2\) est la moyenne des \((R_i - \mu)^2\) ; par le théorème central limite (admis), \(\sqrt{n}\big(\bar R - \mu, S^2 - \sigma^2\big)\) a pour covariance asymptotique la covariance de \(\big(R, (R - \mu)^2\big)\) : \(\mathrm{Var}(R) = \sigma^2\), \(\mathrm{Var}((R - \mu)^2) = \mathbb{E}[(R - \mu)^4] - \sigma^4 = 2\sigma^4\) (moment d'ordre 4 d'une normale : \(3\sigma^4\)), et \(\mathrm{Cov}(R, (R - \mu)^2) = \mathbb{E}[(R - \mu)^3] = 0\) (symétrie).
3. Variance asymptotique : \(\tfrac{1}{\sigma^2}\sigma^2 + \tfrac{\mu^2}{4\sigma^6}\,2\sigma^4 = 1 + \theta^2/2\).
4. Annualisation : \(\mathrm{Var}(\hat\theta_a) = q\,\mathrm{Var}(\hat\theta) \approx q(1 + \theta^2/2)/(qT)\), et \(\theta^2 = \theta_a^2/q\).

Exemple : rendements quotidiens, 4 ans, Sharpe annualisé 1 : erreur-type \(@L_sr_se_4ans:4m@\) (simulation @L_sr_se_4ans_mc:4@), presque \(1/\sqrt{T} = @L_sr_se_4ans_simple:1m@\). Le lien avec DIAG-A5 : \(t = \hat\theta_a\sqrt{T}\), soit le Sharpe divisé par l'erreur-type \(1/\sqrt{T}\) qu'il aurait sous l'hypothèse nulle.

Cas limites et pièges :

1. Hors normalité, la même preuve, avec l'asymétrie \(\gamma_3\) et le kurtosis \(\gamma_4\) (moments centrés d'ordre 3 et 4 divisés par \(\sigma^3\) et \(\sigma^4\)), donne \(1 - \gamma_3\theta + (\gamma_4 - 1)\theta^2/4\) ; contrôlé par simulation sur des rendements de loi exponentielle (écart relatif @L_sr_asym_expo_rel:3@).
2. Avec des rendements autocorrélés, le facteur \(\sqrt{q}\) d'annualisation est faux (Lo, 2002).
3. Le terme \(\theta_a^2/(2q)\) est négligeable en quotidien ; il compte en mensuel avec un Sharpe élevé.

## Partie 7 : optimisation

### Développement de Taylor à l'ordre 2

Énoncé : si \(f : \mathbb{R}^d \to \mathbb{R}\) est deux fois continûment dérivable, de gradient \(\nabla f\) et de matrice hessienne \(\nabla^2 f\) (matrice des dérivées secondes), alors \(f(\mathbf{x} + \mathbf{h}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^\top\mathbf{h} + \tfrac12\,\mathbf{h}^\top\nabla^2 f(\mathbf{x})\,\mathbf{h} + o(\lVert \mathbf{h} \rVert^2)\).

Preuve : \(u(t) = f(\mathbf{x} + t\mathbf{h})\) a pour dérivées \(u'(t) = \nabla f(\mathbf{x} + t\mathbf{h})^\top\mathbf{h}\) et \(u''(t) = \mathbf{h}^\top\nabla^2 f(\mathbf{x} + t\mathbf{h})\,\mathbf{h}\). La formule de Taylor-Lagrange en dimension 1 (admise) donne \(u(1) = u(0) + u'(0) + \tfrac12 u''(\tau)\) pour un \(\tau \in\, ]0, 1[\). L'écart \(\tfrac12\lvert \mathbf{h}^\top(\nabla^2 f(\mathbf{x} + \tau\mathbf{h}) - \nabla^2 f(\mathbf{x}))\mathbf{h} \rvert\) est au plus \(\tfrac12\lVert \mathbf{h} \rVert^2\) fois la norme d'opérateur de la différence des hessiennes, qui tend vers 0 par continuité.

Exemple, la croissance logarithmique (le critère utilisé, sous contrainte de drawdown, pour dimensionner ses positions) : on investit une fraction \(w\) de sa richesse dans un pari de rendement \(R\) ; la croissance espérée est \(\psi(w) = \mathbb{E}[\ln(1 + wR)]\). Taylor à l'ordre 2 de \(\ln(1 + u)\) en 0 donne \(\psi(w) \approx w\,\mathbb{E}[R] - \tfrac{w^2}{2}\mathbb{E}[R^2]\), maximal en \(w \approx \mathbb{E}[R]/\mathbb{E}[R^2]\). Pour \(R = 0{,}2\) avec probabilité 0,4 et \(R = -0{,}1\) avec probabilité 0,6 : \(\mathbb{E}[R] = @L_k_mu:2m@\), \(\mathbb{E}[R^2] = @L_k_m2:3m@\), approximation \(w \approx \tfrac{@L_k_approx_num@}{@L_k_approx_den@} \approx @L_k_approx:3m@\) ; l'optimum exact (dérivée de \(\psi\) nulle) est \(w = @L_k_exact@\), et la croissance perdue est faible : \(@L_k_g_approx:5m@\) contre \(@L_k_g_exact:5m@\), car \(\psi\) est plate près de son maximum.

### Convexité

Définition : \(f\), définie sur un ensemble convexe, est convexe si \(f(t\mathbf{x} + (1 - t)\mathbf{y}) \le t f(\mathbf{x}) + (1 - t) f(\mathbf{y})\) pour tous \(\mathbf{x}, \mathbf{y}\) et \(t \in [0, 1]\) ; strictement convexe si l'inégalité est stricte pour \(\mathbf{x} \ne \mathbf{y}\) et \(t \in\, ]0, 1[\).

Énoncé, pour \(f\) deux fois continûment dérivable sur un intervalle ouvert (dimension 1) ou un convexe ouvert (dimension \(d\)) :

1. (C1) Dimension 1 : \(f\) convexe si et seulement si \(f'' \ge 0\).
2. (C2) Dimension \(d\) : \(f\) convexe si et seulement si \(\nabla^2 f(\mathbf{x})\) est semi-définie positive en tout point.
3. (C3) Hessienne définie positive partout : strictement convexe. La réciproque est fausse : \(x^4\) est strictement convexe et sa dérivée seconde est nulle en 0.
4. (C4) Si \(f\) est convexe et dérivable, \(\nabla f(\mathbf{x}^\star) = \mathbf{0}\) si et seulement si \(\mathbf{x}^\star\) est un minimum global ; si \(f\) est strictement convexe, ce minimum est unique.

Preuve :

1. (C1), sens direct : si \(f''(x_0) < 0\), Taylor-Young donne \(\tfrac12[f(x_0 + h) + f(x_0 - h)] - f(x_0) = \tfrac12 f''(x_0)h^2 + o(h^2) < 0\) pour \(h\) petit, ce qui contredit la convexité avec \(t = \tfrac12\).
2. (C1), réciproque : Taylor-Lagrange donne \(f(y) = f(z) + f'(z)(y - z) + \tfrac12 f''(\tau)(y - z)^2 \ge f(z) + f'(z)(y - z)\) (la courbe est au-dessus de ses tangentes). Avec \(z = tx + (1 - t)y\), on écrit cette inégalité en \(x\) et en \(y\), on les multiplie par \(t\) et \(1 - t\) et on additionne : les termes en \(f'(z)\) se compensent car \(t(x - z) + (1 - t)(y - z) = 0\).
3. (C2) : la définition ne fait intervenir que des points d'un segment, donc \(f\) est convexe si et seulement si chaque restriction \(u(t) = f(\mathbf{x} + t\mathbf{h})\) l'est, soit par (C1) \(u''(t) = \mathbf{h}^\top\nabla^2 f(\mathbf{x} + t\mathbf{h})\mathbf{h} \ge 0\) pour tous \(\mathbf{x}, \mathbf{h}, t\), ce qui est la semi-définie positivité en tout point.
4. (C3) : même argument avec des inégalités strictes, l'inégalité de Taylor-Lagrange devenant stricte pour \(y \ne z\).
5. (C4) : l'inégalité des tangentes de l'étape 2, appliquée sur chaque segment, donne \(f(\mathbf{y}) \ge f(\mathbf{x}^\star) + \nabla f(\mathbf{x}^\star)^\top(\mathbf{y} - \mathbf{x}^\star) = f(\mathbf{x}^\star)\) ; réciproquement, en un minimum, chaque dérivée directionnelle est nulle. Unicité : deux minimums distincts auraient un milieu de valeur strictement plus petite.

Exemples : \(\lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2\) a pour hessienne \(2\mathbf{X}^\top\mathbf{X}\), semi-définie positive ; \(\mathbf{w}^\top\boldsymbol{\Sigma}\mathbf{w}\) a pour hessienne \(2\boldsymbol{\Sigma}\) ; \(\psi\) ci-dessus a pour dérivée seconde \(-\mathbb{E}[R^2/(1 + wR)^2] < 0\) sur l'intervalle \(-5 < w < 10\), où \(1 + wR > 0\) pour les deux valeurs de \(R\) : elle y est concave, et sa dérivée nulle donne bien son maximum.

[[controle:convexite]]

### Multiplicateur de Lagrange, une contrainte d'égalité linéaire

Énoncé : soient \(f\) dérivable sur \(\mathbb{R}^d\), \(\mathbf{a} \in \mathbb{R}^d\) non nul et \(b\) réel. Si \(\mathbf{x}^\star\) minimise \(f\) (même localement) sur l'hyperplan \(\{\mathbf{x} : \mathbf{a}^\top\mathbf{x} = b\}\), il existe un réel \(\lambda\), le multiplicateur, tel que \(\nabla f(\mathbf{x}^\star) = \lambda\,\mathbf{a}\). Si \(f\) est convexe, cette condition est aussi suffisante.

1. Les points de l'hyperplan sont les \(\mathbf{x}^\star + t\mathbf{h}\) avec \(\mathbf{a}^\top\mathbf{h} = 0\). Pour un tel \(\mathbf{h}\), \(t \mapsto f(\mathbf{x}^\star + t\mathbf{h})\) a un minimum local en \(t = 0\), donc sa dérivée y est nulle : \(\nabla f(\mathbf{x}^\star)^\top\mathbf{h} = 0\).
2. On décompose \(\mathbf{g} = \nabla f(\mathbf{x}^\star) = \lambda\,\mathbf{a} + \mathbf{r}\) avec \(\lambda = \mathbf{g}^\top\mathbf{a}/\lVert \mathbf{a} \rVert^2\), de sorte que \(\mathbf{a}^\top\mathbf{r} = 0\). L'étape 1 avec \(\mathbf{h} = \mathbf{r}\) donne \(0 = \mathbf{g}^\top\mathbf{r} = \lambda\,\mathbf{a}^\top\mathbf{r} + \lVert \mathbf{r} \rVert^2 = \lVert \mathbf{r} \rVert^2\), donc \(\mathbf{r} = \mathbf{0}\).
3. Suffisance si \(f\) est convexe : pour \(\mathbf{y}\) de l'hyperplan, \(f(\mathbf{y}) \ge f(\mathbf{x}^\star) + \lambda\,\mathbf{a}^\top(\mathbf{y} - \mathbf{x}^\star) = f(\mathbf{x}^\star)\), car \(\mathbf{a}^\top\mathbf{y} = \mathbf{a}^\top\mathbf{x}^\star = b\).

### Application : portefeuille de variance minimale

Énoncé : \(d\) actifs de matrice de covariance \(\boldsymbol{\Sigma}\) définie positive ; \(\mathbf{w}\) est le vecteur des poids, avec la contrainte budgétaire \(\mathbf{1}^\top\mathbf{w} = 1\) (poids négatifs permis : ventes à découvert). Le portefeuille de variance minimale est unique :

\(\displaystyle \mathbf{w}^\star = \frac{\boldsymbol{\Sigma}^{-1}\mathbf{1}}{\mathbf{1}^\top\boldsymbol{\Sigma}^{-1}\mathbf{1}}, \qquad \mathbf{w}^{\star\top}\boldsymbol{\Sigma}\,\mathbf{w}^\star = \frac{1}{\mathbf{1}^\top\boldsymbol{\Sigma}^{-1}\mathbf{1}}.\)

1. \(f(\mathbf{w}) = \mathbf{w}^\top\boldsymbol{\Sigma}\mathbf{w}\) a pour hessienne \(2\boldsymbol{\Sigma}\), définie positive : \(f\) est strictement convexe (C3).
2. Lagrange avec \(\mathbf{a} = \mathbf{1}\) : \(2\boldsymbol{\Sigma}\mathbf{w}^\star = \lambda\,\mathbf{1}\), donc \(\mathbf{w}^\star = \tfrac{\lambda}{2}\boldsymbol{\Sigma}^{-1}\mathbf{1}\).
3. La contrainte donne \(\tfrac{\lambda}{2}\,\mathbf{1}^\top\boldsymbol{\Sigma}^{-1}\mathbf{1} = 1\) ; le dénominateur est strictement positif car \(\boldsymbol{\Sigma}^{-1}\) est définie positive (ses valeurs propres sont les inverses de celles de \(\boldsymbol{\Sigma}\)) et \(\mathbf{1} \ne \mathbf{0}\).
4. Par la suffisance (convexité), c'est un minimum global, unique par stricte convexité. Variance : \(\mathbf{w}^{\star\top}\boldsymbol{\Sigma}\mathbf{w}^\star = \mathbf{1}^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{-1}\mathbf{1}/(\mathbf{1}^\top\boldsymbol{\Sigma}^{-1}\mathbf{1})^2\). Le multiplicateur vaut \(\lambda = 2/(\mathbf{1}^\top\boldsymbol{\Sigma}^{-1}\mathbf{1})\), le double de la variance minimale.

Exemple (script, en fractions exactes) : volatilités 0,10, 0,15 et 0,20 ; corrélations 0,3 entre les actifs 1 et 2, 0 entre 1 et 3, 0,5 entre 2 et 3. Poids \(\tfrac{1}{@L_pf_w1_den@}(@L_pf_w1_num@ ; @L_pf_w2_num@ ; @L_pf_w3_num@) \approx (@L_pf_w1:3m@ ; @L_pf_w2:3m@ ; @L_pf_w3:3m@)\), variance \(\tfrac{@L_pf_var_num@}{@L_pf_var_den@}\), volatilité \(@L_pf_vol:4m@\), sous celle du moins volatil des actifs. Contrôles du script : minimisation numérique sous contrainte, et \(2\boldsymbol{\Sigma}\mathbf{w}^\star\) de coordonnées toutes égales à \(\lambda = @L_pf_lambda:5m@\).

Deux actifs, de volatilités \(\sigma_1, \sigma_2\) et de corrélation \(\rho\) : la formule donne \(w_1 = \dfrac{\sigma_2^2 - \rho\sigma_1\sigma_2}{\sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2}\), et le poids \(w_2 = 1 - w_1\) de l'actif le plus volatil (\(\sigma_2 > \sigma_1\)) est négatif si et seulement si \(\rho > \sigma_1/\sigma_2\). Pour \(\rho = 0\), on retrouve les poids proportionnels aux inverses des variances (DIAG-M8).

Pièges :

1. Les poids proportionnels à l'inverse des variances ne sont optimaux que pour des actifs non corrélés.
2. \(\boldsymbol{\Sigma}\) estimée avec moins de dates que d'actifs n'est pas inversible ; même inversible, son inverse amplifie l'erreur d'estimation : rétrécissement de Ledoit-Wolf (leçon d'algèbre linéaire, `outils/lecons_algebre.py`, parties 8 et 10).
3. Avec une contrainte d'inégalité (poids positifs), Lagrange ne suffit plus : il faut les conditions de Karush-Kuhn-Tucker, non traitées ici.

## Réflexes en 30 secondes

| Indice dans l'énoncé | Premier geste |
|---|---|
| AIC contre BIC | coût par paramètre 2 contre \(\ln n\) ; comparer \(2\Delta\ell\) à \(2\Delta k\) et à \(\Delta k \ln n\) |
| AIC, variable inutile ajoutée | \(2\Delta\ell = Z^2\) ; AIC l'ajoute avec probabilité \(\approx @L_p_aic:3m@\) pour tout \(n\) |
| interpolation, \(p \ge n\) | erreur d'entraînement nulle ; norme minimale \(\mathbf{X}^\top(\mathbf{X}\mathbf{X}^\top)^{-1}\mathbf{y}\) ; bruit \(\sigma^2\sum 1/s_j^2\), pic en \(p = n\) |
| importance par permutation, modèle linéaire | \(2b_j^2\,\mathrm{Var}(X_j)\) ; variables redondantes : dépend des coefficients choisis |
| Shapley, linéaire, indépendantes | \(b_j(x_j - \mu_j)\) ; deux variables : moyenne des contributions en premier et en second |
| IC | rangs, \(1 - 6\sum d^2/(N(N^2 - 1))\) ; sans effet, écart-type \(1/\sqrt{N - 1}\) |
| étiquettes de \(h\) jours chevauchantes | corrélation \((h - 1)/h\) ; variance de la moyenne environ \(h\) fois la valeur naïve |
| meilleur de \(m\) essais sans effet | \(\mathbb{E}[\max] \le \sqrt{2\ln m}\) ; Sharpe annualisé espéré \(\mathbb{E}[M_m]/\sqrt{T}\) |
| erreur-type du Sharpe | \(\sqrt{(1 + \theta_a^2/(2q))/T}\), environ \(1/\sqrt{T}\) en quotidien |
| variance minimale | \(\boldsymbol{\Sigma}^{-1}\mathbf{1}/(\mathbf{1}^\top\boldsymbol{\Sigma}^{-1}\mathbf{1})\) ; deux actifs : \(w_2 < 0\) si \(\rho > \sigma_1/\sigma_2\) |

Contrôles de vraisemblance en 5 secondes : un IC est entre \(-1\) et 1 ; le meilleur de \(m\) essais sans effet a une espérance positive et sous \(\sqrt{2\ln m}\) ; la volatilité du portefeuille de variance minimale est au plus celle du moins volatil des actifs ; les valeurs de Shapley somment à \(f(\mathbf{x}) - \mathbb{E}f(\mathbf{X})\).

## À savoir refaire sans notes

- L'optimisme \(2k\sigma^2\) de l'erreur d'entraînement, et \(2\Delta\ell = Z^2\) pour deux modèles emboîtés.
- L'interpolateur de norme minimale et l'amplification du bruit \(\sigma^2\sum 1/s_j^2\).
- (P1), et les valeurs de Shapley du modèle linéaire à variables indépendantes, avec le lemme des poids.
- La formule de Spearman et \(\mathrm{Var}(\mathrm{IC}) = 1/(N - 1)\) sous l'hypothèse nulle ; la loi fondamentale dans le modèle gaussien.
- La variance de la moyenne d'étiquettes chevauchantes ; \(\mathbb{E}[S^2]\) avec une rupture de moyenne.
- Le majorant \(\sqrt{2\ln m}\) par Jensen ; l'erreur-type du Sharpe par la méthode delta.
- (C1) et (C2) ; Lagrange avec une contrainte linéaire ; le portefeuille de variance minimale.
""",
    "controles": {
        "aic_bic": {
            "question": r"Deux modèles emboîtés sont comparés sur \(n\) observations. À partir de quelle valeur de \(n\) le BIC fait-il payer chaque paramètre supplémentaire plus cher que l'AIC ?",
            "options": [
                {"id": "a", "texte": r"\(n \ge @L_n_seuil@\)",
                 "explication": r"Oui : le coût par paramètre est 2 pour l'AIC et \(\ln n\) pour le BIC ; \(\ln n > 2\) équivaut à \(n > e^2 \approx @L_e2:3m@\)."},
                {"id": "b", "texte": r"\(n \ge @C_n_ln1@\)",
                 "explication": r"C'est la condition \(\ln n > 1\) : on a comparé \(\ln n\) à 1 au lieu de 2. L'AIC compte \(2k\), pas \(k\)."},
                {"id": "c", "texte": "Jamais : les deux critères pénalisent de la même façon, seule la vraisemblance diffère.",
                 "explication": r"Les deux utilisent la même log-vraisemblance maximale ; c'est la pénalité qui diffère, \(2k\) contre \(k \ln n\)."},
                {"id": "d", "texte": r"\(n \ge 100\)",
                 "explication": r"Aucun seuil n'est lié à 100 : la comparaison porte sur \(\ln n\) et 2, et \(\ln 100 \approx @L_ln100:2m@\) dépasse 2 depuis longtemps."},
            ],
        },
        "double_descente": {
            "question": "Dans l'expérience de double descente de la leçon, laquelle de ces affirmations est démontrée, et non seulement constatée ?",
            "options": [
                {"id": "a", "texte": r"La part de l'erreur due au bruit vaut \(\sigma^2\sum_j 1/s_j^2\) : elle est grande dès qu'une valeur singulière de la matrice des données est petite.",
                 "explication": r"Oui : c'est (DD2), une identité de trace. Que la plus petite valeur singulière soit proche de 0 en \(p = n\) est, en revanche, constaté dans la sortie."},
                {"id": "b", "texte": "Le meilleur modèle sur-paramétré bat toujours le meilleur modèle sous-paramétré.",
                 "explication": r"Faux en général : sur le profil concentré, le meilleur modèle est sous-paramétré. Sur le profil réparti, c'est l'inverse : deux constats, aucune loi générale."},
                {"id": "c", "texte": "Le pic de l'erreur de test se trouve en p = n pour toute loi des données.",
                 "explication": r"C'est constaté sur cette expérience, pas démontré pour toute loi ; l'explication passe par la plus petite valeur singulière, dont le comportement dépend de la loi de \(\mathbf{X}\)."},
                {"id": "d", "texte": "Un modèle qui interpole les données d'entraînement a forcément une mauvaise erreur de test.",
                 "explication": r"Contredit par l'expérience : au profil réparti, le modèle à \(p = 200\), qui interpole, est le meilleur."},
            ],
        },
        "shapley": {
            "question": r"Modèle \(f(\mathbf{x}) = 3 + 2x_1 - x_2\), variables \(X_1\) et \(X_2\) indépendantes d'espérances 0 et 1, valeur conditionnelle. Que vaut la valeur de Shapley de \(x_2\) au point \(\mathbf{x} = (1, 3)\) ?",
            "options": [
                {"id": "a", "texte": "@C_sh_phi2@",
                 "explication": r"Oui : \(b_2(x_2 - \mu_2) = -1 \times (3 - 1)\). Contrôle : \(@C_sh_phi1@ + (@C_sh_phi2@) = f(\mathbf{x}) - \mathbb{E}f(\mathbf{X})\)."},
                {"id": "b", "texte": "@C_sh_d_sans_mu@",
                 "explication": r"C'est \(b_2 x_2\) : la moyenne \(\mu_2 = 1\) n'a pas été retranchée. Shapley répartit l'écart à la prédiction moyenne."},
                {"id": "c", "texte": "@C_sh_d_coef@",
                 "explication": r"C'est le coefficient \(b_2\) seul. La contribution dépend aussi de l'écart de \(x_2\) à sa moyenne."},
                {"id": "d", "texte": "0, car l'effet de la constante 3 absorbe tout.",
                 "explication": r"La constante est dans \(v(\emptyset)\) comme dans \(v(F)\) : elle disparaît des contributions marginales."},
            ],
        },
        "non_stationnarite": {
            "question": r"50 rendements indépendants d'écart-type 1 : les 25 premiers d'espérance 0, les 25 suivants d'espérance 1. Que vaut l'espérance de leur variance empirique (division par 49) ?",
            "options": [
                {"id": "a", "texte": "@C_ns:4@",
                 "explication": r"Oui : \(1 + n\Delta^2/(4(n - 1)) = 1 + 50/196\). Simulation (script) : @C_ns_mc:4@."},
                {"id": "b", "texte": "1",
                 "explication": r"C'est la variance de chaque rendement autour de sa propre espérance ; la variance empirique mesure les écarts à la moyenne commune, qui ajoutent \(\sum_i (\mu_i - \bar\mu)^2/(n - 1)\)."},
                {"id": "c", "texte": "@C_ns_sans_n:4@",
                 "explication": r"C'est \(1 + \Delta^2/4\), la variance de la population mélangée : avec la division par \(n - 1\), le terme de rupture vaut \(n\Delta^2/(4(n - 1))\)."},
                {"id": "d", "texte": "2",
                 "explication": r"C'est \(\sigma^2 + \Delta^2\) : chaque rendement n'est qu'à \(\Delta/2\) de la moyenne commune, d'où \(\Delta^2/4\) et non \(\Delta^2\)."},
            ],
        },
        "convexite": {
            "question": "Laquelle de ces affirmations est vraie pour une fonction deux fois continûment dérivable ?",
            "options": [
                {"id": "a", "texte": r"Une hessienne définie positive partout suffit à la stricte convexité, mais n'est pas nécessaire : \(x^4\) est strictement convexe avec une dérivée seconde nulle en 0.",
                 "explication": r"Oui : (C3) et son contre-exemple. Le bon critère d'équivalence est la semi-définie positivité, pour la convexité simple (C2)."},
                {"id": "b", "texte": "Une fonction convexe a une hessienne définie positive en tout point.",
                 "explication": r"Une fonction affine est convexe et sa hessienne est nulle. (C2) donne seulement la semi-définie positivité."},
                {"id": "c", "texte": "Un point où le gradient s'annule est toujours un minimum global.",
                 "explication": r"Seulement pour une fonction convexe (C4). Pour \(x^3\), la dérivée s'annule en 0, qui n'est même pas un minimum local."},
                {"id": "d", "texte": "Si la hessienne est semi-définie positive en un point, la fonction est convexe.",
                 "explication": r"Il la faut semi-définie positive en **tout** point du domaine convexe ; en un seul point, cela ne dit rien de la forme globale."},
            ],
        },
    },
    "verif": {"script": "verif/seance_quant_1.py"},
}
