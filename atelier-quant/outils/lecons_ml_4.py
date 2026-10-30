"""Leçon express de la séance de ML 4 : optimisation, réseaux de neurones, architectures, noyaux et EM.

Format : voir outils/lecons.py (élément de type "lecon", contenu en Markdown réduit, points de contrôle
[[controle:IDENT]] sur une ligne seule, option "a" correcte, une explication par option).
Les blocs python-exec sont exécutés à la construction par le noyau de l'environnement .venv-hackerrank
(numpy 1.21.6, sans PyTorch), dans le dossier coding/lecons ; leur sortie réelle est insérée. Graines fixées.
Les autres nombres sont des jetons @CLE@ remplacés par les valeurs de verif/seance_ml_4.out.json.
"""

LECON_ML_4 = {
    "id": "L-ML-4", "type": "lecon", "section": "lecon",
    "theme": "Leçon express : optimisation, réseaux, architectures, noyaux et EM",
    "difficulte": 4, "temps_cible_s": 2100, "dossier": "coding/lecons",
    "contenu": r"""## Leçon express : optimisation, réseaux, architectures, noyaux et EM

Durée visée : 35 minutes (la leçon couvre toute la section ML avancé hors arbres ; environ 15 minutes si tu ne lis que les énoncés, les exemples et les réflexes, à compléter plus tard par les preuves). Adam (DIAG-A4), la rétropropagation (DIAG-A3) et les convolutions (DIAG-A2), déjà traités dans le test de positionnement, ne sont que rappelés. Chaque résultat a son énoncé, une preuve courte mais complète, un exemple calculé et ses pièges. Expériences exécutées avec numpy 1.21.6 (environnement des notebooks de la plateforme, sans PyTorch) ; autres nombres : script `verif/seance_ml_4.py`, qui les contrôle aussi par simulation.

**Conventions de notation**, valables pour toute la leçon :

- une majuscule désigne une variable aléatoire, une minuscule une valeur ou un paramètre fixe, le gras un vecteur ou une matrice ; une matrice est toujours en majuscule grasse, et le texte dit quand elle est aléatoire (poids tirés à l'initialisation) ;
- exceptions assumées, parce que ce sont les notations universelles : \(L\) (plus grande valeur propre, constante de régularité du gradient), \(C\) (coefficient du SVM), \(T\) (longueur d'une séquence), \(B\) (taille d'un lot) et \(M\) (nombre de couches) sont des nombres fixes ;
- \(\lambda_k\), avec un indice, est une valeur propre ; \(\lambda\), sans indice, un coefficient de régularisation ;
- \(\odot\) est le produit coordonnée par coordonnée ; \(\lVert \mathbf{v} \rVert\) la norme euclidienne ; pour une matrice, \(\lVert \mathbf{M} \rVert = \max_{\mathbf{v} \ne \mathbf{0}} \lVert \mathbf{M}\mathbf{v} \rVert / \lVert \mathbf{v} \rVert\) (norme d'opérateur) ;
- \(\mathrm{sig}(z) = 1/(1 + e^{-z})\) est la sigmoïde, \(\mathrm{ReLU}(z) = \max(0, z)\), et \(\mathcal{N}(x; m, s^2)\) la densité gaussienne de moyenne \(m\) et de variance \(s^2\) au point \(x\).

## Partie 1 : optimisation

### 1.1 Descente de gradient sur une quadratique : conditionnement et pas maximal

**Le problème.** Pourquoi la descente de gradient est-elle lente sur certains problèmes, et quel pas peut-on prendre sans diverger ? Une quadratique répond exactement, et décrit tout problème lisse près de son minimum (développement de Taylor à l'ordre 2).

**Le cadre.** \(\mathbf{A}\) est une matrice symétrique définie positive \(p \times p\), de valeurs propres \(L = \lambda_1 \ge \dots \ge \lambda_p = \lambda_{\min} > 0\) et de vecteurs propres orthonormés \(\mathbf{u}_1, \dots, \mathbf{u}_p\) (théorème spectral, leçon d'algèbre, section 1). \(\mathbf{b} \in \mathbb{R}^p\) est fixe et \(f(\boldsymbol{\theta}) = \tfrac12 \boldsymbol{\theta}^\top \mathbf{A}\,\boldsymbol{\theta} - \mathbf{b}^\top \boldsymbol{\theta}\). Son gradient \(\nabla f(\boldsymbol{\theta}) = \mathbf{A}\boldsymbol{\theta} - \mathbf{b}\) s'annule en l'unique minimum \(\boldsymbol{\theta}^* = \mathbf{A}^{-1}\mathbf{b}\). Le gradient est \(L\)-lipschitzien : \(\lVert \nabla f(\boldsymbol{\theta}) - \nabla f(\boldsymbol{\theta}') \rVert = \lVert \mathbf{A}(\boldsymbol{\theta} - \boldsymbol{\theta}') \rVert \le L\,\lVert \boldsymbol{\theta} - \boldsymbol{\theta}' \rVert\) : c'est le \(L\) de la règle « pas inférieur à \(2/L\) ». Le **conditionnement** est \(\kappa = L/\lambda_{\min} \ge 1\). La descente à pas fixe \(\eta > 0\) part de \(\boldsymbol{\theta}_0\) et itère \(\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta\,\nabla f(\boldsymbol{\theta}_{t-1})\). On note \(q(\eta) = \max_k \lvert 1 - \eta\lambda_k \rvert\), le **facteur de contraction**.

**L'intuition.** Dans la direction propre \(\mathbf{u}_k\), la fonction est une parabole de courbure \(\lambda_k\), et un pas multiplie l'erreur par \(1 - \eta\lambda_k\). Un même pas doit convenir aux directions très courbées (qui imposent \(\eta\) petit) et aux directions plates (qui avancent alors lentement).

**L'énoncé.**

1. (O1) Pour tout \(t\) : \(\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert \le q(\eta)^t\,\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert\), avec égalité si \(\boldsymbol{\theta}_0 - \boldsymbol{\theta}^*\) est un vecteur propre \(\mathbf{u}_k\) qui réalise le maximum.
2. (O2) La suite converge vers \(\boldsymbol{\theta}^*\) pour tout point de départ si et seulement si \(0 < \eta < 2/L\).
3. (O3) \(q(\eta) = \max(\lvert 1 - \eta\lambda_{\min} \rvert, \lvert 1 - \eta L \rvert)\) est minimal pour \(\eta^* = 2/(\lambda_{\min} + L)\), où il vaut \(q^* = (\kappa - 1)/(\kappa + 1)\). Avec le pas \(1/L\), il vaut \(1 - 1/\kappa\).

**La preuve.**

1. L'erreur \(\mathbf{e}_t = \boldsymbol{\theta}_t - \boldsymbol{\theta}^*\) vérifie \(\mathbf{e}_t = \mathbf{e}_{t-1} - \eta\,\mathbf{A}\,\mathbf{e}_{t-1} = (\mathbf{I} - \eta\mathbf{A})\,\mathbf{e}_{t-1}\), car \(\nabla f(\boldsymbol{\theta}) = \mathbf{A}(\boldsymbol{\theta} - \boldsymbol{\theta}^*)\).
2. On décompose \(\mathbf{e}_0 = \sum_k c_k \mathbf{u}_k\). Comme \((\mathbf{I} - \eta\mathbf{A})\mathbf{u}_k = (1 - \eta\lambda_k)\mathbf{u}_k\), \(\mathbf{e}_t = \sum_k c_k (1 - \eta\lambda_k)^t \mathbf{u}_k\), et par orthonormalité \(\lVert \mathbf{e}_t \rVert^2 = \sum_k c_k^2 (1 - \eta\lambda_k)^{2t} \le q(\eta)^{2t} \sum_k c_k^2 = q(\eta)^{2t}\lVert \mathbf{e}_0 \rVert^2\) : c'est (O1), et l'égalité pour \(\mathbf{e}_0 = \mathbf{u}_k\) se lit sur la même ligne.
3. (O2) Si \(q(\eta) < 1\), (O1) donne la convergence. Si \(q(\eta) \ge 1\), le départ \(\mathbf{e}_0 = \mathbf{u}_k\) qui réalise le maximum garde une erreur de norme \(q(\eta)^t \ge 1\). Enfin \(q(\eta) < 1\) équivaut à \(-1 < 1 - \eta\lambda_k < 1\) pour tout \(k\), soit \(0 < \eta < 2/\lambda_k\) pour tout \(k\), soit \(0 < \eta < 2/L\).
4. (O3) \(\lambda \mapsto \lvert 1 - \eta\lambda \rvert\) est convexe, donc son maximum sur \([\lambda_{\min}, L]\) est atteint en une extrémité. En \(\eta^*\) : \(1 - \eta^*\lambda_{\min} = (L - \lambda_{\min})/(L + \lambda_{\min})\) et \(\eta^* L - 1\) vaut la même chose, donc \(q(\eta^*) = (\kappa - 1)/(\kappa + 1)\). Pour \(\eta < \eta^*\), \(q(\eta) \ge 1 - \eta\lambda_{\min} > 1 - \eta^*\lambda_{\min} = q^*\) ; pour \(\eta > \eta^*\), \(q(\eta) \ge \eta L - 1 > \eta^* L - 1 = q^*\). Avec \(\eta = 1/L\) : \(q = \max(1 - 1/\kappa, 0) = 1 - 1/\kappa\).

**Exemple.** Valeurs propres 1 et 10, donc \(\kappa = 10\). Pas maximal \(2/L = @O_pas_max:1m@\). Pas optimal \(\eta^* = \tfrac{@O_eta_opt_num@}{@O_eta_opt_den@}\), facteur \(q^* = \tfrac{@O_q_opt_num@}{@O_q_opt_den@}\) ; avec \(\eta = 1/L\), \(q = @O_q_invL:1m@\) ; avec \(\eta = 0{,}21\), au-delà de \(2/L\), \(q = @O_q_021:1m@ > 1\) et la suite diverge le long de \(\mathbf{u}_1\). Pour diviser l'erreur par \(10^6\), il faut @O_it_opt10@ pas au pas optimal. Avec \(\kappa = 100\), \(q^* = \tfrac{@O_q100_num@}{@O_q100_den@}\) et il en faut @O_it_gd100@ : à pas optimal, le nombre de pas est de l'ordre de \(\tfrac{\kappa}{2}\ln(1/\text{précision})\), puisque \(-\ln q^* \approx 2/\kappa\) pour \(\kappa\) grand.

**Pièges.** Le pas maximal est fixé par la **plus grande** courbure (\(2/L\), pas \(2/\lambda_{\min}\)). Seul le rapport \(\kappa\) compte : multiplier \(f\) par 10 change le pas, pas la vitesse. Hors des quadratiques, des résultats du même type valent quand la hessienne a ses valeurs propres dans \([\lambda_{\min}, L]\) en tout point (admis ici).

[[controle:pas]]

### 1.2 Momentum

**La définition** (momentum de la boule pesante). Avec \(0 \le \beta < 1\) et \(\mathbf{m}_0 = \mathbf{0}\) : \(\mathbf{m}_t = \beta\,\mathbf{m}_{t-1} + \nabla f(\boldsymbol{\theta}_{t-1})\), puis \(\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta\,\mathbf{m}_t\). \(\mathbf{m}_t\) est une somme pondérée des gradients passés, l'analogue du \(m_t\) d'Adam. L'intuition : elle cumule les gradients récents ; les composantes qui changent de signe d'un pas à l'autre (oscillations dans les directions courbées) se compensent, celles qui gardent leur signe (direction plate) s'accumulent.

**L'énoncé** (O4). Sur la quadratique, avec \(\beta = \big((\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)\big)^2\) et \(\eta = 4/(\sqrt{L} + \sqrt{\lambda_{\min}})^2\), l'erreur décroît dans chaque direction propre au taux asymptotique \(\sqrt{\beta} = (\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)\) : \(\kappa\) est remplacé par \(\sqrt{\kappa}\).

**La preuve.**

1. Comme \(\boldsymbol{\theta}_t - \boldsymbol{\theta}_{t-1} = -\eta\,\mathbf{m}_t = \beta(\boldsymbol{\theta}_{t-1} - \boldsymbol{\theta}_{t-2}) - \eta\,\nabla f(\boldsymbol{\theta}_{t-1})\), la coordonnée \(e_t\) de l'erreur sur \(\mathbf{u}_k\) vérifie \(e_t = (1 + \beta - \eta\lambda_k)\,e_{t-1} - \beta\,e_{t-2}\).
2. Cette récurrence linéaire d'ordre 2 a pour polynôme caractéristique \(z^2 - (1 + \beta - \eta\lambda_k)\,z + \beta\). Le produit de ses racines vaut \(\beta\) ; si elles sont complexes conjuguées (ou double), elles ont donc le même module \(\sqrt{\beta}\), et \(e_t\) est de l'ordre de \(\sqrt{\beta}^{\,t}\) (fois \(t\) en cas de racine double).
3. Les racines sont complexes ou double si et seulement si le discriminant est négatif ou nul : \(\lvert 1 + \beta - \eta\lambda_k \rvert \le 2\sqrt{\beta}\), soit \((1 - \sqrt{\beta})^2 \le \eta\lambda_k \le (1 + \sqrt{\beta})^2\).
4. Avec les réglages de l'énoncé, \(1 - \sqrt{\beta} = 2\sqrt{\lambda_{\min}}/(\sqrt{L} + \sqrt{\lambda_{\min}})\), donc \((1 - \sqrt{\beta})^2 = \eta\lambda_{\min}\), et de même \((1 + \sqrt{\beta})^2 = \eta L\). Toute valeur propre de \([\lambda_{\min}, L]\) vérifie donc la condition 3 : toutes les directions ont le taux \(\sqrt{\beta}\).

**Exemple.** \(\kappa = 100\) : taux \(\tfrac{@O_mom100_num@}{@O_mom100_den@}\) au lieu de \(\tfrac{@O_q100_num@}{@O_q100_den@}\), soit @O_it_mom100@ pas au lieu de @O_it_gd100@ pour gagner un facteur \(10^6\) (calcul sur les taux asymptotiques). Expérience sur \(\mathbf{A} = \mathrm{diag}(1, 100)\), départ \((1, 1)\) :

```python-exec
import numpy as np
np.set_printoptions(precision=4, suppress=True)
lam = np.array([1.0, 100.0])          # A = diag(1, 100), minimum en 0, gradient A theta

def nb_pas(eta, beta, tol=1e-6):
    theta, m = np.ones(2), np.zeros(2)
    for t in range(1, 100001):
        m = beta * m + lam * theta
        theta = theta - eta * m
        if np.linalg.norm(theta) <= tol * np.sqrt(2):
            return t

L, lmin = lam.max(), lam.min()
r = (np.sqrt(L / lmin) - 1) / (np.sqrt(L / lmin) + 1)
print("gradient, pas 2/(lmin + L) :", nb_pas(2 / (lmin + L), 0.0), "pas")
print("momentum, réglages de (O4) :", nb_pas(4 / (np.sqrt(L) + np.sqrt(lmin)) ** 2, r ** 2), "pas")
```

Le momentum demande plus de pas que le compte asymptotique, à cause du facteur \(t\) des racines doubles aux deux extrémités du spectre, mais reste bien plus rapide.

**Pièges.** (O4) est un résultat sur les quadratiques, avec \(\lambda_{\min}\) et \(L\) connus ; sur un réseau de neurones, \(\beta = 0{,}9\) est un réglage empirique, sans garantie de ce type.

### 1.3 Gradient stochastique et calendriers de pas

**Le problème.** Le gradient d'une perte moyenne sur \(n\) observations coûte \(n\) calculs. Le gradient stochastique (SGD) utilise le gradient d'une observation, ou d'un lot de \(B\) observations, tirées au hasard : une estimation sans biais du gradient, mais bruitée.

**Le modèle.** En dimension 1, \(f(\theta) = \tfrac{a}{2}\theta^2\) avec \(a > 0\), minimum en 0. Au pas \(t\), le gradient estimé est \(G_t = a\,\Theta_{t-1} + Z_t\), où les \(Z_t\) sont indépendantes, de moyenne 0 et de variance \(\sigma^2\), et indépendantes de \(\Theta_0, \dots, \Theta_{t-1}\) ; \(\Theta_t = \Theta_{t-1} - \eta\,G_t\), avec \(\Theta_0 = \theta_0\) fixé.

**L'énoncé** (O5). Si \(0 < \eta a < 2\) : \(\mathbb{E}[\Theta_t] = (1 - \eta a)^t\theta_0 \to 0\), et \(\mathrm{Var}(\Theta_t) \to \dfrac{\eta^2\sigma^2}{1 - (1 - \eta a)^2} = \dfrac{\eta\,\sigma^2}{a\,(2 - \eta a)}\). À pas constant, le SGD ne converge pas vers le minimum : il fluctue autour, avec une variance proportionnelle à \(\eta\) quand \(\eta\) est petit (équivalent \(\eta\sigma^2/(2a)\)).

**La preuve.**

1. \(\Theta_t = (1 - \eta a)\,\Theta_{t-1} - \eta Z_t\). En espérance, \(\mathbb{E}[Z_t] = 0\) donne \(\mathbb{E}[\Theta_t] = (1 - \eta a)\,\mathbb{E}[\Theta_{t-1}]\).
2. \(Z_t\) est indépendante de \(\Theta_{t-1}\), donc les variances s'ajoutent : \(v_t = (1 - \eta a)^2 v_{t-1} + \eta^2\sigma^2\), où \(v_t = \mathrm{Var}(\Theta_t)\) et \(v_0 = 0\).
3. Le point fixe est \(v_\infty = \eta^2\sigma^2/(1 - (1 - \eta a)^2)\), et \(v_t - v_\infty = (1 - \eta a)^2 (v_{t-1} - v_\infty)\) tend vers 0 car \((1 - \eta a)^2 < 1\). Enfin \(1 - (1 - \eta a)^2 = \eta a\,(2 - \eta a)\).

**Exemple.** \(a = 1\), \(\sigma^2 = 1\), \(\eta = 0{,}1\) : \(v_\infty = \tfrac{@O_sgd_var_num@}{@O_sgd_var_den@} \approx @O_sgd_var:4m@\) (simulation de 40000 chaînes : @O_sgd_var_mc:4@), proche de l'équivalent \(\eta\sigma^2/(2a) = @O_sgd_approx:2m@\). Un lot de \(B = 10\) observations indépendantes divise \(\sigma^2\) par 10, donc \(v_\infty\) aussi : \(\tfrac{@O_sgd_lot10_num@}{@O_sgd_lot10_den@}\).

**Calendriers de pas.** D'où l'idée de faire décroître \(\eta_t\). Les conditions de Robbins et Monro, \(\sum_t \eta_t = \infty\) et \(\sum_t \eta_t^2 < \infty\) (par exemple \(\eta_t = \eta_0/(1 + t/t_0)\)), assurent la convergence du SGD sous des hypothèses de convexité et de bruit borné (théorème admis). La première est nécessaire dès que les gradients sont bornés : (O6) si chaque gradient utilisé \(\mathbf{g}_s\) (exact ou estimé) est de norme au plus \(g_{\max}\), alors \(\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}_0 \rVert = \lVert \sum_{s \le t} \eta_s\mathbf{g}_s \rVert \le g_{\max}\sum_s \eta_s\) (inégalité triangulaire sur la somme des pas) ; avec \(\eta_t = 0{,}1 \times 0{,}9^t\) pour \(t \ge 0\), la somme vaut @O_geo_somme@ et l'on ne s'éloigne jamais de plus de \(g_{\max}\) du départ, même si le minimum est plus loin. Calendriers usuels en réseaux de neurones : paliers (pas divisé par une constante à des époques fixées) ; cosinus, \(\eta_t = \eta_{\min} + \tfrac12(\eta_{\max} - \eta_{\min})\big(1 + \cos(\pi t/T)\big)\) pour \(t\) de 0 au nombre total de pas \(T\) ; échauffement (warmup), montée linéaire de 0 à \(\eta_{\max}\) sur les premiers pas, pratique empirique dont la justification théorique reste discutée.

### 1.4 Adam, rappel

Notations du test de positionnement : \(m_t = \beta_1 m_{t-1} + (1 - \beta_1)g_t\), \(v_t = \beta_2 v_{t-1} + (1 - \beta_2)g_t^2\), \(\hat m_t = m_t/(1 - \beta_1^t)\), \(\hat v_t = v_t/(1 - \beta_2^t)\), \(\theta_t = \theta_{t-1} - \eta\,\hat m_t/(\sqrt{\hat v_t} + \epsilon)\), coordonnée par coordonnée. Trois faits, chacun en une ligne :

1. Au premier pas, \(\hat m_1 = g_1\) et \(\hat v_1 = g_1^2\) : le pas vaut \(-\eta\,g_1/(\lvert g_1 \rvert + \epsilon)\), environ \(\eta\) en valeur absolue (DIAG-A4).
2. Multiplier tous les gradients d'une coordonnée par \(c > 0\) multiplie \(m\) par \(c\) et \(v\) par \(c^2\) : le pas ne change pas (au \(\epsilon\) près). Adam est insensible à l'échelle de chaque coordonnée, ce qui compense en partie un mauvais conditionnement diagonal.
3. Une pénalité \(\tfrac{\lambda}{2}\theta^2\) ajoutée à la perte ajoute \(\lambda\theta\) au gradient, qui est ensuite divisé par \(\sqrt{\hat v_t}\) : les coordonnées à grands gradients sont moins régularisées. La décroissance des poids découplée (AdamW : Loshchilov et Hutter, « Decoupled Weight Decay Regularization », ICLR 2019) retranche \(\eta\lambda\theta\) directement, hors de la division.

## Partie 2 : réseaux de neurones

### 2.1 Rétropropagation matricielle, une couche cachée

**Le réseau.** Entrée \(\mathbf{x} \in \mathbb{R}^{n_0}\) ; couche cachée de \(n_1\) neurones : \(\mathbf{a} = \mathbf{W}_1\mathbf{x} + \mathbf{b}_1\) (pré-activations, \(\mathbf{W}_1\) de taille \(n_1 \times n_0\)), \(\mathbf{h} = \phi(\mathbf{a})\) avec \(\phi\) une fonction dérivable appliquée coordonnée par coordonnée ; sortie \(\hat{\mathbf{y}} = \mathbf{W}_2\mathbf{h} + \mathbf{b}_2 \in \mathbb{R}^{n_2}\) ; perte \(\ell = \tfrac12 \lVert \hat{\mathbf{y}} - \mathbf{y} \rVert^2\) pour une cible \(\mathbf{y}\).

**L'énoncé** (B1). Avec \(\boldsymbol{\delta}_2 = \hat{\mathbf{y}} - \mathbf{y}\) et \(\boldsymbol{\delta}_1 = (\mathbf{W}_2^\top \boldsymbol{\delta}_2) \odot \phi'(\mathbf{a})\) : \(\partial\ell/\partial\mathbf{W}_2 = \boldsymbol{\delta}_2\mathbf{h}^\top\), \(\partial\ell/\partial\mathbf{b}_2 = \boldsymbol{\delta}_2\), \(\partial\ell/\partial\mathbf{W}_1 = \boldsymbol{\delta}_1\mathbf{x}^\top\), \(\partial\ell/\partial\mathbf{b}_1 = \boldsymbol{\delta}_1\) (chaque gradient a la taille du paramètre). Pour la perte moyenne d'un lot de \(B\) exemples rangés en lignes dans \(\mathbf{X}\) (\(B \times n_0\)), avec \(\boldsymbol{\Delta}_1\) la matrice \(B \times n_1\) dont la ligne \(i\) est \(\boldsymbol{\delta}_1^\top\) de l'exemple \(i\) : \(\partial\ell/\partial\mathbf{W}_1 = \boldsymbol{\Delta}_1^\top \mathbf{X}/B\).

**La preuve**, coordonnée par coordonnée (règle de la chaîne) :

1. \(\hat y_i = \sum_j (\mathbf{W}_2)_{ij} h_j + (\mathbf{b}_2)_i\), donc \(\partial\ell/\partial(\mathbf{W}_2)_{ij} = (\hat y_i - y_i)\,h_j\), le coefficient \((i, j)\) de \(\boldsymbol{\delta}_2\mathbf{h}^\top\).
2. \(h_j\) intervient dans toutes les sorties : \(\partial\ell/\partial h_j = \sum_i (\hat y_i - y_i)(\mathbf{W}_2)_{ij} = (\mathbf{W}_2^\top \boldsymbol{\delta}_2)_j\) ; puis \(\partial\ell/\partial a_j = (\mathbf{W}_2^\top \boldsymbol{\delta}_2)_j\,\phi'(a_j) = (\boldsymbol{\delta}_1)_j\), car \(h_j\) ne dépend que de \(a_j\).
3. \(a_j = \sum_k (\mathbf{W}_1)_{jk} x_k + (\mathbf{b}_1)_j\), donc \(\partial\ell/\partial(\mathbf{W}_1)_{jk} = (\boldsymbol{\delta}_1)_j\,x_k\). Pour un lot, la perte moyenne se dérive en moyennant les gradients, et \(\sum_i \boldsymbol{\delta}_1^{(i)}\mathbf{x}^{(i)\top} = \boldsymbol{\Delta}_1^\top \mathbf{X}\).

**Contrôle** par différences finies, avec \(\phi = \tanh\) et \(\tanh' = 1 - \tanh^2\) :

```python-exec
rng = np.random.default_rng(0)
n0, n1, n2 = 3, 4, 2
W1, b1, W2, b2 = rng.normal(size=(n1, n0)), rng.normal(size=n1), rng.normal(size=(n2, n1)), rng.normal(size=n2)
x, y = rng.normal(size=n0), rng.normal(size=n2)

def perte(W1):
    return 0.5 * np.sum((W2.dot(np.tanh(W1.dot(x) + b1)) + b2 - y) ** 2)

h = np.tanh(W1.dot(x) + b1)
d2 = W2.dot(h) + b2 - y
d1 = W2.T.dot(d2) * (1 - h ** 2)
gW1 = np.outer(d1, x)                      # formule (B1)
num = np.zeros_like(W1)
for j in range(n1):
    for k in range(n0):
        E = np.zeros_like(W1)
        E[j, k] = 1e-6
        num[j, k] = (perte(W1 + E) - perte(W1 - E)) / 2e-6
print("écart maximal entre (B1) et les différences finies :", np.abs(num - gW1).max())
```

**Pièges.** \(\phi'\) s'évalue en la pré-activation \(\mathbf{a}\), pas en \(\mathbf{h}\) (sauf quand on écrit \(\tanh'\) avec \(\mathbf{h}\), comme ci-dessus) ; c'est \(\mathbf{W}_2^\top\), pas \(\mathbf{W}_2\), qui renvoie l'erreur vers l'arrière ; un gradient de poids est un produit extérieur « erreur en sortie de la couche » fois « entrée de la couche ».

### 2.2 Initialisation : pourquoi une variance 1/n ou 2/n

**Le cadre.** Couche \(l \ge 1\) : \(\mathbf{a}^{(l)} = \mathbf{W}^{(l)}\mathbf{h}^{(l-1)}\), \(\mathbf{h}^{(l)} = \phi(\mathbf{a}^{(l)})\), \(\mathbf{h}^{(0)}\) l'entrée, biais nuls à l'initialisation. La matrice \(\mathbf{W}^{(l)}\) est **aléatoire**, de taille \(n_l \times n_{l-1}\) : ses coefficients sont indépendants, de même loi symétrique (par exemple gaussienne) de moyenne 0 et de variance \(s_l^2\), et indépendants des autres couches et de l'entrée. \(n_{l-1}\) est le nombre d'entrées de la couche (fan-in).

**L'énoncé.**

1. (I1) Pour tout neurone \(i\) de la couche \(l\) : \(\mathbb{E}\big[(a^{(l)}_i)^2\big] = s_l^2 \sum_{j=1}^{n_{l-1}} \mathbb{E}\big[(h^{(l-1)}_j)^2\big]\), et pour \(l \ge 2\), \(a^{(l)}_i\) a une loi symétrique (même loi que \(-a^{(l)}_i\)).
2. (I2) Si \(A\) a une loi symétrique et un moment d'ordre 2 fini, \(\mathbb{E}[\mathrm{ReLU}(A)^2] = \tfrac12\mathbb{E}[A^2]\).
3. Conséquences, si les \(n_{l-1}\) coordonnées de \(\mathbf{h}^{(l-1)}\) ont le même moment d'ordre 2 : avec ReLU, \(\mathbb{E}[(a^{(l)}_i)^2] = \tfrac12\,n_{l-1}s_l^2\,\mathbb{E}[(a^{(l-1)}_j)^2]\), conservé si \(s_l^2 = 2/n_{l-1}\) (initialisation de He : He, Zhang, Ren et Sun, « Delving Deep into Rectifiers », ICCV 2015) ; avec une activation linéaire, ou \(\tanh\) près de 0 où \(\tanh(z) \approx z\), conservé si \(s_l^2 = 1/n_{l-1}\) (LeCun). Glorot et Bengio (« Understanding the difficulty of training deep feedforward neural networks », AISTATS 2010) proposent \(2/(n_{l-1} + n_l)\), compromis entre la condition de l'aller \(n_{l-1}s^2 = 1\) et celle du retour \(n_l s^2 = 1\) (même calcul sur \(\boldsymbol{\delta}^{(l-1)} = (\mathbf{W}^{(l)\top}\boldsymbol{\delta}^{(l)}) \odot \phi'(\mathbf{a}^{(l-1)})\), en supposant de plus \(\boldsymbol{\delta}^{(l)}\) indépendant de \(\mathbf{W}^{(l)}\), ce qui n'est qu'une approximation).

**La preuve.**

1. \(a_i = \sum_j W_{ij}h_j\) (indice \(l\) omis). En développant le carré : \(\mathbb{E}[a_i^2] = \sum_j \mathbb{E}[W_{ij}^2]\,\mathbb{E}[h_j^2] + \sum_{j \ne j'} \mathbb{E}[W_{ij}]\,\mathbb{E}[W_{ij'}]\,\mathbb{E}[h_j h_{j'}]\), car \(\mathbf{h}^{(l-1)}\) ne dépend que des couches précédentes, donc est indépendant de \(\mathbf{W}^{(l)}\), et les \(W_{ij}\) sont indépendants entre eux. Les termes croisés sont nuls (\(\mathbb{E}[W_{ij}] = 0\)), d'où (I1).
2. Symétrie : sachant \(\mathbf{h}^{(l-1)}\), changer \(\mathbf{W}^{(l)}\) en \(-\mathbf{W}^{(l)}\) ne change pas la loi de \(\mathbf{W}^{(l)}\) (coefficients de loi symétrique), et change \(a_i\) en \(-a_i\).
3. (I2) \(\mathrm{ReLU}(A)^2 = A^2\,I_{\{A > 0\}}\). Par symétrie, \(\mathbb{E}[A^2 I_{\{A > 0\}}] = \mathbb{E}[A^2 I_{\{A < 0\}}]\), et la somme des deux vaut \(\mathbb{E}[A^2]\) (le terme \(A = 0\) ne contribue pas, \(A^2\) y étant nul) : chacun vaut la moitié.

**Exemple.** Avec ReLU et \(s^2 = 1/n\), le moment d'ordre 2 est divisé par 2 à chaque couche : après @R_profondeur@ couches, facteur \(2^{-50} \approx @R_lecun50_m:1m@ \times 10^{@R_lecun50_e@}\) ; avec \(s^2 = 3/n\), facteur \(1{,}5^{50} \approx @R_trois50_m:1m@ \times 10^{@R_trois50_e@}\). Expérience, largeur 256, entrée de variance 1 : logarithme en base 2 du moment d'ordre 2 des pré-activations aux couches 10, 20, ..., 50, observé (moyenne sur 8 réseaux tirés au hasard), puis prédit par (I1) et (I2), soit \(\log_2 c + (l - 1)\log_2(c/2)\) pour \(s^2 = c/n\) :

```python-exec
rng = np.random.default_rng(1)
n, M, R = 256, 50, 8
for c in [1, 2, 3]:
    moments = np.zeros((R, M // 10))
    for r in range(R):
        a = rng.normal(size=(n, 50))            # 50 entrées de variance 1, en colonnes
        for l in range(1, M + 1):
            W = rng.normal(scale=np.sqrt(c / n), size=(n, n))
            a = W.dot(a if l == 1 else np.maximum(a, 0))
            if l % 10 == 0:
                moments[r, l // 10 - 1] = np.mean(a ** 2)
    pred = [np.log2(c) + (l - 1) * np.log2(c / 2) for l in range(10, M + 1, 10)]
    print("s2 =", c, "/ n ; observé :", np.round(np.log2(moments.mean(axis=0)), 1), "; prédit :", np.round(pred, 1))
```

Chaque réseau fluctue autour de la prédiction, qui porte sur l'espérance ; la pente, elle, est celle annoncée.

**Pièges.** (I2) porte sur le moment d'ordre 2, pas sur la variance : pour \(A\) gaussienne centrée réduite, \(\mathrm{Var}(\mathrm{ReLU}(A)) = \tfrac12 - \tfrac{1}{2\pi} \approx @R_var_relu:4m@\) (intégrale calculée par sympy, simulation @R_var_relu_mc:4@), et c'est \(\mathbb{E}[h^2]\), non \(\mathrm{Var}(h)\), qui entre dans (I1). L'argument ne vaut qu'à l'initialisation : l'entraînement crée des dépendances entre poids et activations.

### 2.3 Disparition et explosion du gradient

**L'énoncé** (G1). En rétropropagation, \(\boldsymbol{\delta}^{(l-1)} = \mathbf{D}_{l-1}\mathbf{W}^{(l)\top}\boldsymbol{\delta}^{(l)}\), où \(\mathbf{D}_{l-1}\) est diagonale de coefficients \(\phi'(a^{(l-1)}_j)\) ; donc \(\lVert \boldsymbol{\delta}^{(0)} \rVert \le \prod_{l=1}^{M}\big(\sup\lvert \phi' \rvert\,\lVert \mathbf{W}^{(l)} \rVert\big)\,\lVert \boldsymbol{\delta}^{(M)} \rVert\).

**La preuve.** Par (B1) appliqué couche par couche. Une matrice diagonale vérifie \(\lVert \mathbf{D}\mathbf{v} \rVert \le \max_j \lvert d_j \rvert\,\lVert \mathbf{v} \rVert\) ; et \(\lVert \mathbf{W}^\top \mathbf{v} \rVert \le \lVert \mathbf{W} \rVert\,\lVert \mathbf{v} \rVert\), car \(\lVert \mathbf{W}^\top \mathbf{v} \rVert = \max_{\lVert \mathbf{z} \rVert = 1} \mathbf{z}^\top \mathbf{W}^\top \mathbf{v} = \max_{\lVert \mathbf{z} \rVert = 1} (\mathbf{W}\mathbf{z})^\top \mathbf{v} \le \lVert \mathbf{W} \rVert\,\lVert \mathbf{v} \rVert\) (Cauchy-Schwarz). On enchaîne les \(M\) couches.

**Exemple.** La sigmoïde a pour dérivée \(\mathrm{sig}(1 - \mathrm{sig})\), maximale en 0 où elle vaut \(\tfrac{@R_sig_max_num@}{@R_sig_max_den@}\). Avec 10 couches sigmoïdes et \(\lVert \mathbf{W}^{(l)} \rVert \le 2\), le gradient de la première couche est au plus \((2/4)^{10} = \tfrac{@R_sig_borne10_num@}{@R_sig_borne10_den@}\) fois celui de la dernière. La borne est une majoration : elle prouve la **disparition** quand le produit est petit. L'**explosion** se voit sur un réseau linéaire avec \(\mathbf{W}^{(l)} = c\,\mathbf{O}_l\), \(\mathbf{O}_l\) orthogonale : le gradient est multiplié exactement par \(c^M\). Remèdes : initialisation (2.2), normalisation (2.4), connexions résiduelles (2.6), LSTM (3.3) ; contre l'explosion, l'écrêtage du gradient \(\mathbf{g} \leftarrow \mathbf{g}\,\min(1, \tau/\lVert \mathbf{g} \rVert)\) pour un seuil \(\tau > 0\), qui garde la direction et borne la norme.

### 2.4 Normalisation par lot et par couche

**Les définitions.** Les pré-activations d'un lot forment une matrice \(\mathbf{H}\) de taille \(B \times n\) : une ligne par exemple, une colonne par neurone. On note \(\epsilon > 0\) une petite constante de stabilité.

- **Par lot** (BatchNorm) : pour chaque **colonne** \(j\), moyenne \(m_j\) et variance \(v_j\) calculées sur les \(B\) exemples (axe 0, celui du lot), puis \(\hat H_{ij} = (H_{ij} - m_j)/\sqrt{v_j + \epsilon}\) et sortie \(\gamma_j\hat H_{ij} + \nu_j\), avec une échelle \(\gamma_j\) et un décalage \(\nu_j\) appris par neurone. En évaluation, \(m_j\) et \(v_j\) sont remplacées par des moyennes mobiles estimées pendant l'entraînement.
- **Par couche** (LayerNorm) : pour chaque **ligne** \(i\), moyenne et variance sur les \(n\) neurones de l'exemple \(i\) (axe 1), la même formule ensuite, et le même calcul en entraînement et en évaluation.

**L'énoncé.**

1. (N1) En entraînement, la sortie BatchNorm d'un exemple dépend des autres exemples du lot ; avec \(B = 1\), \(v_j = 0\) et \(\hat H_{1j} = 0\) : la sortie vaut \(\nu_j\) quelle que soit l'entrée.
2. (N2) Pour \(\epsilon = 0\) : BatchNorm est inchangée si l'on remplace une colonne \(\mathbf{H}_{\cdot j}\) par \(c\,\mathbf{H}_{\cdot j} + d\) (\(c > 0\)) ; LayerNorm est inchangée si l'on remplace une ligne par \(c\) fois cette ligne plus \(d\). En particulier, multiplier les poids d'un neurone par \(c > 0\) ne change pas la sortie BatchNorm, et multiplier toute la matrice de poids de la couche par \(c\) ne change pas la sortie LayerNorm.

**La preuve.** (N1) : avec un seul exemple, la moyenne est cet exemple. (N2) : la moyenne devient \(c\,m + d\), l'écart-type \(c\,\sqrt{v}\), et \((cH + d - cm - d)/(c\sqrt{v}) = (H - m)/\sqrt{v}\).

```python-exec
H = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 300.0], [3.0, 60.0, 200.0], [6.0, 30.0, 400.0]])  # B = 4, n = 3
eps = 1e-5
bn = (H - H.mean(axis=0)) / np.sqrt(H.var(axis=0) + eps)
ln = (H - H.mean(axis=1, keepdims=True)) / np.sqrt(H.var(axis=1, keepdims=True) + eps)
print("BatchNorm, moyennes et variances par colonne :", bn.mean(axis=0).round(6) + 0.0, bn.var(axis=0).round(4))
print("LayerNorm, moyennes et variances par ligne :", ln.mean(axis=1).round(6) + 0.0, ln.var(axis=1).round(4))
H1 = H[:1]
print("BatchNorm sur un lot d'un exemple :", (H1 - H1.mean(axis=0)) / np.sqrt(H1.var(axis=0) + eps))
```

**Pièges.** BatchNorm normalise chaque neurone sur le lot, pas chaque exemple ; LayerNorm est préférée pour les séquences et les Transformers, car elle ne dépend ni du lot ni des autres positions ; oublier de passer en mode évaluation fait dépendre la prédiction d'un exemple des autres exemples du lot.

[[controle:norme]]

### 2.5 Dropout et sa remise à l'échelle

**La définition.** En entraînement, chaque activation \(h_j\) est multipliée par \(D_j/(1 - p)\), où les \(D_j\) sont indépendantes, de loi de Bernoulli : \(D_j = 0\) avec probabilité \(p\) (neurone éteint), 1 sinon. En évaluation, rien n'est fait. C'est le comportement de `nn.Dropout(p)` de PyTorch, contrôlé par le script de vérification quand PyTorch est installé.

**L'énoncé** (D1). Sachant \(h_j\), la sortie \(\tilde H_j = D_j h_j/(1 - p)\) vérifie \(\mathbb{E}[\tilde H_j] = h_j\) et \(\mathrm{Var}(\tilde H_j) = h_j^2\,p/(1 - p)\).

**La preuve.** \(\mathbb{E}[D_j] = 1 - p\), donc \(\mathbb{E}[\tilde H_j] = h_j\). \(D_j^2 = D_j\), donc \(\mathbb{E}[\tilde H_j^2] = h_j^2/(1 - p)\) et \(\mathrm{Var}(\tilde H_j) = h_j^2\big(1/(1 - p) - 1\big) = h_j^2\,p/(1 - p)\).

**Exemple.** \(p = 1/2\), \(h = 3\) : la sortie vaut 0 ou 6 (facteur \(1/(1 - p) = @R_do_garde@\)), d'espérance 3 (simulation @R_do_esp_mc:3@) et de variance @R_do_var@. La remise à l'échelle en entraînement fait que l'espérance vue par la couche suivante est la même en entraînement et en évaluation. La version d'origine, sans division en entraînement, multiplie par \(1 - p\) en évaluation : même espérance, autre implémentation.

**Pièges.** \(p\) est la probabilité d'**éteindre** dans PyTorch ; le dropout conserve l'espérance mais ajoute de la variance ; il doit être désactivé en évaluation.

### 2.6 Connexions résiduelles

**La définition.** Un bloc résiduel calcule \(\mathbf{h}^{(l)} = \mathbf{h}^{(l-1)} + F_l(\mathbf{h}^{(l-1)})\), où \(F_l\) est un petit sous-réseau ; \(\mathbf{J}_l\) est la jacobienne de \(F_l\) au point considéré.

**L'énoncé** (Res). La jacobienne de \(\mathbf{h}^{(M)}\) par rapport à \(\mathbf{h}^{(0)}\) est le produit des \(\mathbf{I} + \mathbf{J}_l\), et le gradient rétropropagé est \(\boldsymbol{\delta}^{(0)} = (\mathbf{I} + \mathbf{J}_1)^\top \cdots (\mathbf{I} + \mathbf{J}_M)^\top \boldsymbol{\delta}^{(M)}\). Si \(\lVert \mathbf{J}_l \rVert \le \varepsilon < 1\) pour tout \(l\) : \(\lVert \boldsymbol{\delta}^{(0)} \rVert \ge (1 - \varepsilon)^M\,\lVert \boldsymbol{\delta}^{(M)} \rVert \ge (1 - M\varepsilon)\,\lVert \boldsymbol{\delta}^{(M)} \rVert\). Sans connexion résiduelle, la jacobienne est le produit des \(\mathbf{J}_l\), de norme au plus \(\varepsilon^M\).

**La preuve.** Règle de la chaîne pour la jacobienne d'une composée. Pour tout \(\mathbf{v}\) : \(\lVert (\mathbf{I} + \mathbf{J})^\top \mathbf{v} \rVert \ge \lVert \mathbf{v} \rVert - \lVert \mathbf{J}^\top \mathbf{v} \rVert \ge (1 - \varepsilon)\lVert \mathbf{v} \rVert\), par l'inégalité triangulaire et \(\lVert \mathbf{J}^\top \rVert \le \lVert \mathbf{J} \rVert\) (démontré en 2.3). On itère \(M\) fois. Enfin \((1 - \varepsilon)^M \ge 1 - M\varepsilon\) est l'inégalité de Bernoulli (démontrée dans la leçon de probabilités, outil 4, cas limites).

**Exemple.** \(M = 50\), \(\varepsilon = 0{,}01\) : le gradient garde au moins \(0{,}99^{50} \approx @R_res_borne:3m@\) de sa norme (Bernoulli donne \(@R_res_bernoulli:1m@\)), alors que le produit des seules \(\mathbf{J}_l\) est au plus \(@R_plain_m@ \times 10^{@R_plain_e@}\). L'identité offre au gradient un chemin direct, que les blocs ne peuvent qu'ajuster.

**Piège.** Le résultat demande \(\lVert \mathbf{J}_l \rVert < 1\) ; il ne dit rien de blocs qui amplifient. Pour s'en approcher à l'initialisation, on peut initialiser la dernière couche de chaque bloc près de zéro (alors \(\mathbf{J}_l \approx \mathbf{0}\)), ou normaliser dans le bloc.

## Partie 3 : architectures

### 3.1 Convolution causale et champ réceptif dilaté

**La définition.** Une couche de noyau de taille \(k\), de dilatation \(d\) et de pas 1 calcule \(y_t = \sum_{i=0}^{k-1} w_i\,x_{t - d i}\) (plus un biais), avec \(x_s = 0\) avant le début de la série : on ajoute \(d(k-1)\) zéros **à gauche seulement**. Elle est **causale** : \(y_t\) ne dépend que de \(x_s\) pour \(s \le t\). On empile \(M\) couches de tailles \(k_l\) et dilatations \(d_l\). Le **champ réceptif** \(R\) est la longueur de la fenêtre des entrées dont dépend une sortie : de \(x_{t-R+1}\) à \(x_t\).

**L'énoncé.**

1. (C1) L'ensemble des retards \(s\) tels que \(x_{t-s}\) intervient dans la sortie est \(S = \{\sum_l d_l i_l : 0 \le i_l \le k_l - 1\}\), donc \(R = 1 + \sum_{l=1}^{M} d_l(k_l - 1)\).
2. (C2) Avec \(R_0 = 1\) et \(R_l = R_{l-1} + d_l(k_l - 1)\) : si \(d_l \le R_{l-1}\) pour tout \(l\), \(S\) contient tous les retards de 0 à \(R - 1\) (aucun trou).

**La preuve.** Par récurrence sur les couches : la couche \(l\) lit les sorties de la couche \(l-1\) aux retards \(0, d_l, \dots, d_l(k_l - 1)\), donc \(S_l = \{s + d_l i : s \in S_{l-1}, 0 \le i \le k_l - 1\}\), avec \(S_0 = \{0\}\). Le plus grand retard augmente de \(d_l(k_l - 1)\) : c'est (C1). Pour (C2), si \(S_{l-1} = \{0, \dots, R_{l-1} - 1\}\), les translatés de cet intervalle par \(0, d_l, 2d_l, \dots\) se chevauchent ou se touchent dès que \(d_l \le R_{l-1}\) : leur réunion est l'intervalle \(\{0, \dots, R_{l-1} - 1 + d_l(k_l - 1)\}\).

**Exemple.** Noyau 3, dilatations 1, 2, 4, 8 : \(R = 1 + 2(1 + 2 + 4 + 8) = @A_champ_ex@\), sans trou. Noyau 2, dilatations \(1, 2, \dots, 512\) (10 couches, type WaveNet) : \(R = @A_wavenet@\) : le champ double à chaque couche, donc croît exponentiellement avec la profondeur, pour un coût linéaire. Contre-exemple de (C2) : noyau 2, dilatations 1 puis 4 : \(R = @A_trou_span@\) mais \(S\) ne contient que les retards @A_trou_ens@, soit @A_trou_nb@ entrées.

**Pièges.** Avec un pas (stride) supérieur à 1, la formule change (les dilatations effectives se multiplient par les pas des couches précédentes). Une convolution centrée (zéros des deux côtés) fait dépendre \(y_t\) d'entrées futures : c'est une fuite d'information si \(y_t\) sert à prédire au temps \(t\), ou si l'on convolue toute la série avant de la découper en échantillons d'entraînement et de test ; ce n'est pas un problème pour une séquence entièrement observée avant la prédiction.

### 3.2 RNN

\(\mathbf{h}_t = \tanh(\mathbf{W}\mathbf{h}_{t-1} + \mathbf{U}\mathbf{x}_t + \mathbf{b})\). La jacobienne d'un pas est \(\partial\mathbf{h}_t/\partial\mathbf{h}_{t-1} = \mathrm{diag}(1 - \mathbf{h}_t \odot \mathbf{h}_t)\,\mathbf{W}\), et celle de \(T - t\) pas en est le produit. Comme \(0 < \tanh' \le 1\), (G1) donne \(\lVert \partial\mathbf{h}_T/\partial\mathbf{h}_t \rVert \le \lVert \mathbf{W} \rVert^{T-t}\) : décroissance exponentielle avec la distance si \(\lVert \mathbf{W} \rVert < 1\). Si chaque jacobienne d'un pas a une norme 0,5, 100 pas donnent \(0{,}5^{100} \approx @A_rnn100_m:1m@ \times 10^{@A_rnn100_e@}\) : le RNN n'apprend pas les dépendances lointaines.

### 3.3 LSTM, équation par équation

Entrée \(\mathbf{x}_t \in \mathbb{R}^{n_x}\), état caché \(\mathbf{h}_t \in \mathbb{R}^{n_h}\), cellule \(\mathbf{c}_t \in \mathbb{R}^{n_h}\), et \(\mathbf{z}_t = (\mathbf{h}_{t-1}, \mathbf{x}_t)\) leur concaténation, de dimension \(n_h + n_x\). Chaque matrice \(\mathbf{W}_\bullet\) est de taille \(n_h \times (n_h + n_x)\).

1. Porte d'oubli \(\mathbf{f}_t = \mathrm{sig}(\mathbf{W}_f\mathbf{z}_t + \mathbf{b}_f)\) : quelle part de l'ancienne cellule garder (coordonnées entre 0 et 1).
2. Porte d'entrée \(\mathbf{i}_t = \mathrm{sig}(\mathbf{W}_i\mathbf{z}_t + \mathbf{b}_i)\) : quelle part du candidat écrire.
3. Candidat \(\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c\mathbf{z}_t + \mathbf{b}_c)\) : la nouvelle information, entre −1 et 1.
4. Cellule \(\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t\) : mémoire mise à jour **additivement**.
5. Porte de sortie \(\mathbf{o}_t = \mathrm{sig}(\mathbf{W}_o\mathbf{z}_t + \mathbf{b}_o)\) et état \(\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)\) : quelle part de la mémoire montrer.

**L'énoncé** (L1). Le long du chemin direct de cellule en cellule (portes considérées comme fixées), \(\partial\mathbf{c}_t/\partial\mathbf{c}_{t-1} = \mathrm{diag}(\mathbf{f}_t)\), donc \(\partial\mathbf{c}_T/\partial\mathbf{c}_t = \mathrm{diag}(\mathbf{f}_T \odot \dots \odot \mathbf{f}_{t+1})\) : ni \(\mathbf{W}\), ni dérivée de \(\tanh\). Preuve : l'équation 4 est linéaire en \(\mathbf{c}_{t-1}\), de coefficient \(\mathbf{f}_t\). Le gradient total contient aussi les chemins qui passent par \(\mathbf{h}_{t-1}\) et les portes ; (L1) dit seulement qu'il existe un chemin que le réseau peut garder ouvert. Avec des portes d'oubli à 0,99, 100 pas conservent \(0{,}99^{100} \approx @A_oubli100:3m@\), contre \(@A_rnn100_m:1m@ \times 10^{@A_rnn100_e@}\) pour le RNN ci-dessus.

**Nombre de paramètres** : 4 blocs (trois portes et le candidat), chacun de \(n_h(n_h + n_x)\) poids et \(n_h\) biais, soit \(4\big(n_h(n_h + n_x) + n_h\big)\) ; pour \(n_x = 10\), \(n_h = 20\) : @A_lstm_ex@. Piège : `nn.LSTM` de PyTorch a deux vecteurs de biais par bloc, d'où \(4n_h\) paramètres de plus (contrôlé par le script quand PyTorch est installé).

### 3.4 Attention par produit scalaire mis à l'échelle

**La définition.** Pour une séquence de \(T\) positions, chaque position \(i\) a une requête \(\mathbf{q}_i \in \mathbb{R}^d\), une clé \(\mathbf{k}_i \in \mathbb{R}^d\) et une valeur \(\mathbf{v}_i\) (projections linéaires apprises de son entrée). Les scores sont \(s_{ij} = \mathbf{q}_i^\top \mathbf{k}_j/\sqrt{d}\), les poids \(p_{ij} = e^{s_{ij}}/\sum_{j'} e^{s_{ij'}}\) (softmax sur les clés \(j\), ligne par ligne), et la sortie \(\mathbf{o}_i = \sum_j p_{ij}\mathbf{v}_j\). Masque causal : \(s_{ij} = -\infty\) pour \(j > i\), d'où \(p_{ij} = 0\).

**L'énoncé.**

1. (A1) On regarde une requête et une clé comme aléatoires (à l'initialisation) : leurs coordonnées \(Q_1, \dots, Q_d\) et \(K_1, \dots, K_d\) sont \(2d\) variables indépendantes, de moyenne 0 et de variance 1. Alors le produit scalaire \(P = \sum_{m=1}^d Q_m K_m\) vérifie \(\mathbb{E}[P] = 0\) et \(\mathrm{Var}(P) = d\) : sans division, l'écart-type des scores est \(\sqrt{d}\) ; divisé par \(\sqrt{d}\), il vaut 1.
2. (A2) \(\partial p_{ij}/\partial s_{ij'} = p_{ij}\,(I_{\{j = j'\}} - p_{ij'})\) : si un poids est proche de 1 et les autres de 0, toutes ces dérivées sont proches de 0 (softmax saturé, gradient presque nul).
3. (A3) Le calcul des \(T^2\) scores coûte de l'ordre de \(T^2 d\) opérations et \(T^2\) cases mémoire par tête : doubler \(T\) multiplie ce coût par @A_cout_x@.
4. (A4) Sans information de position, permuter les positions de l'entrée permute de la même façon les sorties : l'ordre est perdu, d'où les encodages de position.

**La preuve.**

1. (A1) Par indépendance, \(\mathbb{E}[Q_mK_m] = \mathbb{E}[Q_m]\,\mathbb{E}[K_m] = 0\) pour chaque \(m\), et \(\mathbb{E}[Q_m^2K_m^2] = \mathbb{E}[Q_m^2]\,\mathbb{E}[K_m^2] = 1\). Pour \(m \ne m'\), \(\mathbb{E}[Q_mK_mQ_{m'}K_{m'}]\) se factorise en quatre espérances nulles. Donc \(\mathrm{Var}(P) = \mathbb{E}[P^2] = d\).
2. (A2) Dérivée du quotient \(e^{s_{ij}}/S\) avec \(S = \sum_{j'} e^{s_{ij'}}\) : \(\big(I_{\{j = j'\}}e^{s_{ij}}S - e^{s_{ij}}e^{s_{ij'}}\big)/S^2 = p_{ij}(I_{\{j = j'\}} - p_{ij'})\). Si \(p_{ij}\) est proche de 0 ou de 1, \(p_{ij}(1 - p_{ij})\) et les produits \(p_{ij}p_{ij'}\) sont petits.
3. (A3) Un produit scalaire de dimension \(d\) par couple \((i, j)\). (A4) : les scores, les poids et les sorties de chaque position ne dépendent que des contenus, pas des indices.

**Exemple.** \(d = @A_d@\) : écart-type des scores bruts \(@A_std_brut@\) (simulation de la variance : @A_var_score_mc:2@ pour 64). Des scores d'écart-type 8 donnent des \(e^{s}\) qui diffèrent de plusieurs ordres de grandeur, et le softmax devient presque un maximum.

**Pièges.** Diviser par \(d\) au lieu de \(\sqrt{d}\) ; (A1) suppose des coordonnées de variance 1, ce qui vaut à l'initialisation, pas forcément après entraînement ; le softmax se prend sur les clés, pour chaque requête.

[[controle:attention]]

## Partie 4 : noyaux

### 4.1 SVM à marge souple

**Le cadre.** Observations \((\mathbf{x}_i, y_i)\), \(i = 1, \dots, n\), avec \(y_i \in \{-1, +1\}\) ; classifieur \(\mathrm{signe}(f(\mathbf{x}))\) avec \(f(\mathbf{x}) = \mathbf{w}^\top\mathbf{x} + b\). Formulation primale, pour \(C > 0\) fixé : minimiser \(\tfrac12\lVert \mathbf{w} \rVert^2 + C\sum_i \xi_i\) en \((\mathbf{w}, b, \boldsymbol{\xi})\) sous les contraintes \(y_i f(\mathbf{x}_i) \ge 1 - \xi_i\) et \(\xi_i \ge 0\). \(\xi_i\) mesure de combien l'observation \(i\) viole la marge.

**L'énoncé.**

1. (S1) Le problème équivaut à minimiser \(\tfrac12\lVert \mathbf{w} \rVert^2 + C\sum_i \max\big(0, 1 - y_i f(\mathbf{x}_i)\big)\) (perte charnière, hinge loss).
2. (S2) La distance entre les hyperplans \(f = 1\) et \(f = -1\), la largeur de la marge, vaut \(2/\lVert \mathbf{w} \rVert\).
3. (S3) En divisant le critère par \(Cn\) : perte charnière moyenne plus \(\tfrac{1}{2Cn}\lVert \mathbf{w} \rVert^2\). \(C\) joue le rôle de l'inverse d'un coefficient de régularisation : \(C\) grand, marge étroite et peu de violations (faible biais, forte variance) ; \(C\) petit, marge large et violations tolérées. Si les données sont séparables, \(C \to \infty\) donne la marge dure.
4. (S4) Dual (dualité de Lagrange de l'optimisation convexe, admise) : la solution s'écrit \(\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i\) avec \(0 \le \alpha_i \le C\), et les \(\alpha_i\) maximisent \(\sum_i \alpha_i - \tfrac12\sum_{i,j}\alpha_i\alpha_j y_i y_j\,\mathbf{x}_i^\top\mathbf{x}_j\) sous \(\sum_i \alpha_i y_i = 0\). Les observations telles que \(\alpha_i > 0\) sont les vecteurs de support ; une observation strictement au-delà de la marge a \(\alpha_i = 0\). Les données n'interviennent que par les produits scalaires \(\mathbf{x}_i^\top\mathbf{x}_j\).

**La preuve.** (S1) : pour \((\mathbf{w}, b)\) fixés, les contraintes imposent \(\xi_i \ge \max(0, 1 - y_i f(\mathbf{x}_i))\), et le critère croît avec chaque \(\xi_i\) : l'optimum prend l'égalité. (S2) : deux points \(\mathbf{x}_+\) et \(\mathbf{x}_-\) des deux hyperplans vérifient \(\mathbf{w}^\top(\mathbf{x}_+ - \mathbf{x}_-) = 2\) ; la distance entre les hyperplans parallèles est la projection de \(\mathbf{x}_+ - \mathbf{x}_-\) sur la normale unitaire \(\mathbf{w}/\lVert \mathbf{w} \rVert\), soit \(2/\lVert \mathbf{w} \rVert\). (S4), la formule de \(\mathbf{w}\) : le lagrangien \(\tfrac12\lVert \mathbf{w} \rVert^2 + C\sum\xi_i - \sum\alpha_i(y_i f(\mathbf{x}_i) - 1 + \xi_i) - \sum\mu_i\xi_i\) a pour gradient en \(\mathbf{w}\) le vecteur \(\mathbf{w} - \sum_i\alpha_i y_i\mathbf{x}_i\), nul à l'optimum ; la dérivée en \(\xi_i\), \(C - \alpha_i - \mu_i = 0\) avec \(\mu_i \ge 0\), donne \(\alpha_i \le C\).

**Exemple.** Une variable, cinq observations \((x, y)\) : \((-2, -1)\), \((-1, -1)\), \((0{,}5, -1)\), \((1, +1)\), \((2, +1)\). Elles sont séparables, avec une marge dure étroite entre 0,5 et 1. Optimum calculé par le script (le \(w\) optimal est unique, car le critère est strictement convexe en \(w\)) : \(C = 0{,}1\), \(w = @N_svm_c01_w:2m@\), largeur \(@N_svm_c01_marge:2m@\), somme des charnières \(@N_svm_c01_charniere:3m@\) ; \(C = 1\), \(w = @N_svm_c1_w:3m@\), largeur \(@N_svm_c1_marge:1m@\), somme \(@N_svm_c1_charniere:3m@\) ; \(C = 100\), \(w = @N_svm_c100_w@\), largeur \(@N_svm_c100_marge:1m@\), aucune violation : c'est la marge dure.

**Pièges.** \(C\) varie en sens inverse de \(\lambda\) ; le SVM n'est pas invariant par changement d'échelle des variables (standardiser) ; il ne donne pas de probabilité.

[[controle:svm]]

### 4.2 L'astuce du noyau

**Le principe.** Un algorithme qui n'utilise les données qu'à travers des produits scalaires peut travailler dans un espace de caractéristiques \(\boldsymbol{\psi}(\mathbf{x})\) sans jamais le calculer, si l'on sait évaluer le noyau \(k(\mathbf{x}, \mathbf{x}') = \boldsymbol{\psi}(\mathbf{x})^\top\boldsymbol{\psi}(\mathbf{x}')\). La matrice de Gram \(\mathbf{K}\), \(n \times n\), a pour coefficients \(K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)\).

**L'énoncé.**

1. (K0) Si \(k\) s'écrit \(\boldsymbol{\psi}(\mathbf{x})^\top\boldsymbol{\psi}(\mathbf{x}')\), toute matrice de Gram est semi-définie positive. Réciproquement, une fonction symétrique dont toutes les matrices de Gram sont semi-définies positives s'écrit ainsi, avec un espace de caractéristiques éventuellement de dimension infinie (théorème de Moore-Aronszajn, admis).
2. Exemple : dans \(\mathbb{R}^2\), \((\mathbf{x}^\top\mathbf{x}' + 1)^2 = \boldsymbol{\psi}(\mathbf{x})^\top\boldsymbol{\psi}(\mathbf{x}')\) avec \(\boldsymbol{\psi}(\mathbf{x}) = (1, \sqrt2 x_1, \sqrt2 x_2, x_1^2, x_2^2, \sqrt2 x_1x_2)\), @N_dim_poly@ caractéristiques (identité vérifiée par le script). Le noyau gaussien \(k(\mathbf{x}, \mathbf{x}') = \exp(-\lVert \mathbf{x} - \mathbf{x}' \rVert^2/(2\ell^2))\), de longueur d'échelle \(\ell > 0\), correspond à un espace de dimension infinie.

**La preuve** de (K0), sens direct : avec \(\boldsymbol{\Psi}\) la matrice \(n \times D\) des \(\boldsymbol{\psi}(\mathbf{x}_i)^\top\) en lignes, \(\mathbf{K} = \boldsymbol{\Psi}\boldsymbol{\Psi}^\top\) et \(\mathbf{v}^\top\mathbf{K}\mathbf{v} = \lVert \boldsymbol{\Psi}^\top\mathbf{v} \rVert^2 \ge 0\). L'exemple : développer \((x_1x_1' + x_2x_2' + 1)^2\) donne les six produits.

### 4.3 Ridge à noyau, forme fermée

**L'énoncé** (K1). Critère ridge sur les caractéristiques, avec les notations de la leçon de régression (partie 4) : \(r_\lambda(\mathbf{b}) = \tfrac12\lVert \mathbf{y} - \boldsymbol{\Psi}\mathbf{b} \rVert^2 + \tfrac{\lambda}{2}\lVert \mathbf{b} \rVert^2\), \(\lambda > 0\). Sa prédiction en \(\mathbf{x}\) vaut \(\boldsymbol{\psi}(\mathbf{x})^\top\hat{\boldsymbol{\beta}}_\lambda = \mathbf{s}(\mathbf{x})^\top(\mathbf{K} + \lambda\mathbf{I}_n)^{-1}\mathbf{y}\), où \(\mathbf{s}(\mathbf{x}) \in \mathbb{R}^n\) a pour coordonnées \(s_i(\mathbf{x}) = k(\mathbf{x}_i, \mathbf{x})\). Autrement dit \(f(\mathbf{x}) = \sum_i \alpha_i\,k(\mathbf{x}_i, \mathbf{x})\) avec \(\boldsymbol{\alpha} = (\mathbf{K} + \lambda\mathbf{I}_n)^{-1}\mathbf{y}\) : seul le noyau intervient.

**La preuve.**

1. Forme fermée de la ridge (leçon de régression, partie 4, énoncé 1) : \(\hat{\boldsymbol{\beta}}_\lambda = (\boldsymbol{\Psi}^\top\boldsymbol{\Psi} + \lambda\mathbf{I}_D)^{-1}\boldsymbol{\Psi}^\top\mathbf{y}\).
2. Identité de poussée : \(\boldsymbol{\Psi}^\top(\boldsymbol{\Psi}\boldsymbol{\Psi}^\top + \lambda\mathbf{I}_n) = \boldsymbol{\Psi}^\top\boldsymbol{\Psi}\boldsymbol{\Psi}^\top + \lambda\boldsymbol{\Psi}^\top = (\boldsymbol{\Psi}^\top\boldsymbol{\Psi} + \lambda\mathbf{I}_D)\boldsymbol{\Psi}^\top\). Les deux matrices entre parenthèses sont semi-définies positives plus \(\lambda\) fois l'identité, donc définies positives et inversibles ; en multipliant à gauche par \((\boldsymbol{\Psi}^\top\boldsymbol{\Psi} + \lambda\mathbf{I}_D)^{-1}\) et à droite par \((\mathbf{K} + \lambda\mathbf{I}_n)^{-1}\) : \((\boldsymbol{\Psi}^\top\boldsymbol{\Psi} + \lambda\mathbf{I}_D)^{-1}\boldsymbol{\Psi}^\top = \boldsymbol{\Psi}^\top(\mathbf{K} + \lambda\mathbf{I}_n)^{-1}\).
3. Donc \(\boldsymbol{\psi}(\mathbf{x})^\top\hat{\boldsymbol{\beta}}_\lambda = \boldsymbol{\psi}(\mathbf{x})^\top\boldsymbol{\Psi}^\top(\mathbf{K} + \lambda\mathbf{I}_n)^{-1}\mathbf{y}\), et \(\boldsymbol{\Psi}\boldsymbol{\psi}(\mathbf{x}) = \mathbf{s}(\mathbf{x})\).

La preuve suppose \(D\) fini ; pour un espace de dimension infinie (noyau gaussien), la même formule découle du théorème du représentant (admis). Le script vérifie (K1) sur 50 cas avec le noyau polynomial et ses 6 caractéristiques explicites. Coût : résoudre un système \(n \times n\), de l'ordre de \(n^3\) opérations, contre \(D^3\) pour la ridge directe. Pièges : \(\lambda\) s'ajoute à la diagonale seulement ; quand \(\lambda \to 0\) et \(\mathbf{K}\) est inversible, la prédiction interpole les données d'entraînement.

### 4.4 Processus gaussiens

**La définition.** \(F\) est une fonction aléatoire telle que, pour tous points \(\mathbf{x}_1, \dots, \mathbf{x}_m\), le vecteur \((F(\mathbf{x}_1), \dots, F(\mathbf{x}_m))\) est gaussien centré de covariance \(\big(k(\mathbf{x}_i, \mathbf{x}_j)\big)_{i,j}\). On observe \(Y_i = F(\mathbf{x}_i) + E_i\), où les \(E_i\) sont indépendantes de loi \(\mathcal{N}(0, \sigma^2)\), indépendantes de \(F\).

**L'énoncé** (P1). Sachant \(\mathbf{Y} = \mathbf{y}\), \(F(\mathbf{x})\) suit une loi normale de moyenne \(\mathbf{s}(\mathbf{x})^\top(\mathbf{K} + \sigma^2\mathbf{I})^{-1}\mathbf{y}\), **la prédiction de la ridge à noyau avec \(\lambda = \sigma^2\)**, et de variance \(k(\mathbf{x}, \mathbf{x}) - \mathbf{s}(\mathbf{x})^\top(\mathbf{K} + \sigma^2\mathbf{I})^{-1}\mathbf{s}(\mathbf{x})\).

**La preuve.**

1. \((F(\mathbf{x}), \mathbf{Y})\) est un vecteur gaussien (transformation linéaire du vecteur gaussien \((F(\mathbf{x}), F(\mathbf{x}_1), \dots, E_1, \dots)\)), avec \(\mathrm{Cov}(\mathbf{Y}) = \mathbf{K} + \sigma^2\mathbf{I}\) et \(\mathrm{Cov}(\mathbf{Y}, F(\mathbf{x})) = \mathbf{s}(\mathbf{x})\).
2. Soit \(\mathbf{c} = (\mathbf{K} + \sigma^2\mathbf{I})^{-1}\mathbf{s}(\mathbf{x})\) et \(R = F(\mathbf{x}) - \mathbf{c}^\top\mathbf{Y}\). Alors \(\mathrm{Cov}(\mathbf{Y}, R) = \mathbf{s}(\mathbf{x}) - (\mathbf{K} + \sigma^2\mathbf{I})\mathbf{c} = \mathbf{0}\).
3. Dans un vecteur gaussien, des composantes non corrélées sont indépendantes (résultat général sur les vecteurs gaussiens, admis) : \(R\) est indépendant de \(\mathbf{Y}\). Donc, sachant \(\mathbf{Y} = \mathbf{y}\), \(F(\mathbf{x}) = \mathbf{c}^\top\mathbf{y} + R\) suit la loi normale de moyenne \(\mathbf{c}^\top\mathbf{y}\) et de variance \(\mathrm{Var}(R)\).
4. \(\mathrm{Var}(R) = k(\mathbf{x}, \mathbf{x}) - 2\mathbf{c}^\top\mathbf{s}(\mathbf{x}) + \mathbf{c}^\top(\mathbf{K} + \sigma^2\mathbf{I})\mathbf{c} = k(\mathbf{x}, \mathbf{x}) - \mathbf{s}(\mathbf{x})^\top(\mathbf{K} + \sigma^2\mathbf{I})^{-1}\mathbf{s}(\mathbf{x})\).

C'est la version « fonction » de l'interprétation bayésienne de la ridge (leçon de régression, partie 4, énoncé 4).

**Exemple.** Noyau gaussien de longueur d'échelle 1, points \(-1, 0, 1\), observations \(0, 1, 0\), \(\sigma^2 = 0{,}1\), prédiction en 0,5 : moyenne \(@N_gp_moy:3m@\), variance \(@N_gp_var:4m@\) (simulation du modèle : @N_gp_var_mc:4@).

**Pièges.** \(\lambda = \sigma^2\), la variance du bruit, pas l'écart-type ; la variance a posteriori ne dépend pas des observations \(\mathbf{y}\), seulement des positions.

## Partie 5 : EM pour un mélange de deux gaussiennes

**Le modèle.** Pour \(i = 1, \dots, n\), indépendamment : une étiquette cachée \(Z_i \in \{1, 2\}\), avec \(\mathbb{P}(Z_i = c) = \omega_c\) (\(\omega_1 + \omega_2 = 1\)), puis \(X_i\) de loi \(\mathcal{N}(\mu_c, \sigma_c^2)\) sachant \(Z_i = c\). On observe \(x_1, \dots, x_n\), pas les \(Z_i\). Paramètre \(\boldsymbol{\vartheta} = (\omega_1, \omega_2, \mu_1, \mu_2, \sigma_1^2, \sigma_2^2)\). Log-vraisemblance : \(\ell(\boldsymbol{\vartheta}) = \sum_i \ln\big(\omega_1\mathcal{N}(x_i; \mu_1, \sigma_1^2) + \omega_2\mathcal{N}(x_i; \mu_2, \sigma_2^2)\big)\). Le logarithme d'une somme empêche toute forme fermée du maximum. Si l'on connaissait les \(Z_i\), on estimerait chaque gaussienne séparément : EM remplace les \(Z_i\) inconnues par leurs probabilités.

**L'algorithme**, à partir de \(\boldsymbol{\vartheta}\) courant :

1. Étape E : \(r_{ic} = \dfrac{\omega_c\mathcal{N}(x_i; \mu_c, \sigma_c^2)}{\omega_1\mathcal{N}(x_i; \mu_1, \sigma_1^2) + \omega_2\mathcal{N}(x_i; \mu_2, \sigma_2^2)}\), qui est \(\mathbb{P}(Z_i = c \mid X_i = x_i)\) sous \(\boldsymbol{\vartheta}\) (formule de Bayes).
2. Étape M : \(n_c = \sum_i r_{ic}\), puis \(\omega_c = n_c/n\), \(\mu_c = \sum_i r_{ic}x_i/n_c\), \(\sigma_c^2 = \sum_i r_{ic}(x_i - \mu_c)^2/n_c\) avec le nouveau \(\mu_c\) : moyenne et variance pondérées par les responsabilités.

**L'énoncé.**

1. (E1) L'étape M maximise \(Q(\boldsymbol{\vartheta}') = \sum_i\sum_c r_{ic}\big[\ln\omega_c' + \ln\mathcal{N}(x_i; \mu_c', \sigma_c'^2)\big]\), les \(r_{ic}\) étant fixés.
2. (E2) Monotonie : la nouvelle valeur \(\boldsymbol{\vartheta}'\) vérifie \(\ell(\boldsymbol{\vartheta}') \ge \ell(\boldsymbol{\vartheta})\).

**La preuve.**

1. (E1) \(Q\) se sépare. Les poids : \(n_1\ln\omega_1' + n_2\ln(1 - \omega_1')\), concave, de dérivée nulle en \(\omega_1' = n_1/n\). Les moyennes : \(-\sum_i r_{ic}(x_i - \mu_c')^2/(2\sigma_c'^2)\), quadratique concave en \(\mu_c'\), maximale en la moyenne pondérée. Les variances : avec \(u = \sigma_c'^2\) et \(S = \sum_i r_{ic}(x_i - \mu_c')^2\), \(-\tfrac{n_c}{2}\ln u - S/(2u)\) a pour dérivée \((S - n_c u)/(2u^2)\), positive avant \(S/n_c\) et négative après.
2. (E2), outil : \(\ln\) est concave (dérivée seconde \(-1/u^2 < 0\)), donc pour \(a, b > 0\) et des poids \(r, 1 - r\) avec \(0 \le r \le 1\) : \(\ln(ra + (1 - r)b) \ge r\ln a + (1 - r)\ln b\). C'est l'inégalité de Jensen à deux points, qui suffit pour deux composantes.
3. On note \(p_c(x_i; \boldsymbol{\vartheta}') = \omega_c'\mathcal{N}(x_i; \mu_c', \sigma_c'^2)\). Pour chaque \(i\) (cas \(0 < r_{i1} < 1\) ; si \(r_{ic} = 0\), le terme correspondant disparaît des deux côtés) : \(\ln\sum_c p_c(x_i; \boldsymbol{\vartheta}') = \ln\sum_c r_{ic}\,\dfrac{p_c(x_i; \boldsymbol{\vartheta}')}{r_{ic}} \ge \sum_c r_{ic}\ln\dfrac{p_c(x_i; \boldsymbol{\vartheta}')}{r_{ic}}\), par l'étape 2. On somme sur \(i\) : \(\ell(\boldsymbol{\vartheta}') \ge Q(\boldsymbol{\vartheta}') - \sum_{i,c} r_{ic}\ln r_{ic}\).
4. En \(\boldsymbol{\vartheta}' = \boldsymbol{\vartheta}\), il y a égalité : \(p_c(x_i; \boldsymbol{\vartheta})/r_{ic} = \sum_{c'} p_{c'}(x_i; \boldsymbol{\vartheta})\) ne dépend pas de \(c\), et la moyenne pondérée d'une constante est cette constante.
5. Donc \(\ell(\boldsymbol{\vartheta}') \ge Q(\boldsymbol{\vartheta}') - \sum r\ln r \ge Q(\boldsymbol{\vartheta}) - \sum r\ln r = \ell(\boldsymbol{\vartheta})\), la deuxième inégalité venant de ce que \(\boldsymbol{\vartheta}'\) maximise \(Q\) (E1).

**Exemple.** Données \(-1, 0, 4, 5\) ; départ \(\omega = (1/2, 1/2)\), \(\mu = (0, 3)\), variances 1. Responsabilités de la composante 1, arrondies : @E_r@. Après une itération : \(\omega_1 = @E_om1:3m@\), \(\mu_1 = @E_mu1:3m@\), \(\mu_2 = @E_mu2:3m@\), \(\sigma_1^2 = @E_s21:3m@\), \(\sigma_2^2 = @E_s22:3m@\) ; la log-vraisemblance passe de \(@E_ell0:3m@\) à \(@E_ell1:3m@\). Trente itérations :

```python-exec
x = np.array([-1.0, 0.0, 4.0, 5.0])

def dens(x, mu, s2):
    return np.exp(-(x - mu) ** 2 / (2 * s2)) / np.sqrt(2 * np.pi * s2)

om, mu, s2 = np.array([0.5, 0.5]), np.array([0.0, 3.0]), np.array([1.0, 1.0])
ells = []
for it in range(30):
    p = om * dens(x[:, None], mu, s2)            # p[i, c] = omega_c N(x_i ; mu_c, s2_c)
    ells.append(np.log(p.sum(axis=1)).sum())
    r = p / p.sum(axis=1, keepdims=True)         # étape E
    nc = r.sum(axis=0)                           # étape M
    om, mu = nc / len(x), (r * x[:, None]).sum(axis=0) / nc
    s2 = (r * (x[:, None] - mu) ** 2).sum(axis=0) / nc
print("log-vraisemblance, 5 premières itérations :", np.round(ells[:5], 4))
print("plus petite variation d'une itération à la suivante :", np.diff(ells).min())
print("paramètres finaux :", om.round(4), mu.round(4), s2.round(4))
```

**Pièges.**

1. Le maximum de vraisemblance n'existe pas : avec \(\mu_1 = x_1\) et \(\sigma_1 \to 0\), le terme de \(x_1\) tend vers \(+\infty\) et les autres restent minorés par \(\ln(\omega_2\mathcal{N}(x_i; \mu_2, \sigma_2^2))\). Sur l'exemple, \(\ell\) vaut @E_deg@ pour \(\sigma_1 = 0{,}1\), \(10^{-3}\), \(10^{-6}\). EM peut glisser vers une telle dégénérescence : on borne les variances par en dessous.
2. La monotonie ne garantit qu'une convergence vers un point stationnaire ou un maximum local : on relance depuis plusieurs départs.
3. Les étiquettes des composantes sont interchangeables : échanger les composantes 1 et 2 donne la même vraisemblance.

[[controle:em]]

## Réflexes en 30 secondes

| Indice dans l'énoncé | Premier geste |
|---|---|
| pas maximal, divergence de la descente | \(\eta < 2/L\), \(L\) plus grande valeur propre de la hessienne |
| vitesse, conditionnement \(\kappa\) | facteur \(\max_k \lvert 1 - \eta\lambda_k \rvert\) ; optimum \((\kappa - 1)/(\kappa + 1)\) en \(\eta = 2/(\lambda_{\min} + L)\) ; momentum \((\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)\) |
| SGD à pas constant | variance stationnaire \(\eta^2\sigma^2/(1 - (1 - \eta a)^2)\), de l'ordre de \(\eta\) ; lot de \(B\) : bruit divisé par \(B\) |
| variance des poids à l'initialisation | aller : \(n_{\text{entrée}}s^2 \times \mathbb{E}[h^2]\) ; ReLU : \(\mathbb{E}[h^2] = \tfrac12\mathbb{E}[a^2]\), d'où \(2/n\) ; linéaire ou tanh : \(1/n\) |
| BatchNorm ou LayerNorm | BatchNorm : par neurone, sur le lot (axe 0) ; LayerNorm : par exemple, sur les neurones (axe 1) |
| dropout de probabilité \(p\) | garder et diviser par \(1 - p\) ; espérance conservée, variance \(h^2 p/(1 - p)\) |
| gradient à travers \(M\) couches | produit des jacobiennes ; résiduel : au moins \((1 - \varepsilon)^M\) |
| champ réceptif, convolutions causales | \(1 + \sum d_l(k_l - 1)\) ; sans trou si \(d_l \le R_{l-1}\) |
| LSTM | cellule additive, \(\partial\mathbf{c}_t/\partial\mathbf{c}_{t-1} = \mathrm{diag}(\mathbf{f}_t)\) ; paramètres \(4(n_h(n_h + n_x) + n_h)\) |
| attention | scores de variance \(d\), division par \(\sqrt{d}\) ; coût en \(T^2\) |
| SVM, rôle de \(C\) | \(C\) grand : peu de violations, marge étroite ; largeur \(2/\lVert \mathbf{w} \rVert\) |
| ridge à noyau, processus gaussien | \(\mathbf{s}(\mathbf{x})^\top(\mathbf{K} + \lambda\mathbf{I})^{-1}\mathbf{y}\), \(\lambda = \sigma^2\) |
| EM, mélange | E : responsabilités par Bayes ; M : moyennes et variances pondérées ; \(\ell\) ne décroît jamais |

Contrôles de vraisemblance en 5 secondes :

- un facteur de contraction est dans \([0, 1[\) pour un pas qui converge ; \((\kappa - 1)/(\kappa + 1) < 1 - 1/\kappa\) dès que \(\kappa > 1\) ;
- le champ réceptif de \(M\) couches causales de noyau au moins 2 vaut au moins \(M + 1\), et il croît avec chaque dilatation ;
- une variance a posteriori de processus gaussien est entre 0 et \(k(\mathbf{x}, \mathbf{x})\) ;
- une responsabilité est entre 0 et 1, et les deux responsabilités d'une observation somment à 1.

## À savoir refaire sans notes

- (O1) à (O3) par la base propre, avec le calcul du pas optimal ; la récurrence d'ordre 2 du momentum et la condition sur le discriminant.
- La variance stationnaire du SGD sur la quadratique de dimension 1.
- (B1) coordonnée par coordonnée ; (I1) et (I2), d'où 1/n et 2/n.
- (G1) et la borne résiduelle \((1 - \varepsilon)^M\).
- Les deux normalisations, avec leurs axes ; l'espérance et la variance du dropout.
- (C1) et (C2) par récurrence ; les cinq équations du LSTM et (L1) ; (A1) et la jacobienne du softmax.
- (S1) et (S2) ; (K1) par l'identité de poussée ; (P1) par le résidu non corrélé.
- Les étapes E et M, et la monotonie de EM par Jensen à deux points.
""",
    "controles": {
        "pas": {
            "question": r"La hessienne d'une quadratique a pour valeurs propres 0,5 et 4. Pour quels pas fixes \(\eta > 0\) la descente de gradient converge-t-elle quel que soit le point de départ ?",
            "options": [
                {"id": "a", "texte": r"Pour \(\eta < @C_pas_max:1m@\).",
                 "explication": r"Oui : il faut \(\lvert 1 - \eta\lambda_k \rvert < 1\) pour chaque valeur propre, la plus contraignante étant \(L = 4\) : \(\eta < 2/L\)."},
                {"id": "b", "texte": r"Pour \(\eta < @C_pas_lmin@\).",
                 "explication": r"C'est \(2/\lambda_{\min}\). Avec \(\eta = 1\), par exemple, la direction de courbure 4 est multipliée par \(\lvert 1 - 4 \rvert = 3\) à chaque pas : divergence."},
                {"id": "c", "texte": r"Pour \(\eta \le \tfrac{@C_pas_opt_num@}{@C_pas_opt_den@}\) seulement.",
                 "explication": r"\(2/(\lambda_{\min} + L)\) est le pas **optimal**, celui qui minimise le facteur de contraction. Les pas un peu plus grands, jusqu'à \(2/L\) exclu, convergent aussi, plus lentement."},
                {"id": "d", "texte": r"Pour \(\eta \le @C_pas_invL:2m@\) seulement.",
                 "explication": r"\(1/L\) est un pas sûr, qui converge avec le facteur \(1 - 1/\kappa\), mais ce n'est pas la limite : tout \(\eta < 2/L\) converge."},
            ],
        },
        "norme": {
            "question": r"Les pré-activations d'un lot forment une matrice de 32 lignes (exemples) et 128 colonnes (neurones). Que calcule LayerNorm ?",
            "options": [
                {"id": "a", "texte": "32 moyennes et 32 variances, une par exemple, sur ses 128 neurones.",
                 "explication": r"Oui : LayerNorm normalise chaque ligne (axe 1). Le résultat d'un exemple ne dépend pas des autres exemples."},
                {"id": "b", "texte": "128 moyennes et 128 variances, une par neurone, sur les 32 exemples.",
                 "explication": r"C'est BatchNorm, qui normalise chaque colonne sur le lot (axe 0)."},
                {"id": "c", "texte": "Une seule moyenne et une seule variance sur les 4096 nombres.",
                 "explication": r"Aucune des deux normalisations ne mélange ainsi tous les exemples et tous les neurones : la moyenne est prise ligne par ligne (LayerNorm) ou colonne par colonne (BatchNorm)."},
                {"id": "d", "texte": "Des moyennes mobiles accumulées pendant l'entraînement, utilisées en évaluation.",
                 "explication": r"Les moyennes mobiles sont propres à BatchNorm en évaluation. LayerNorm fait le même calcul en entraînement et en évaluation."},
            ],
        },
        "attention": {
            "question": r"Une requête et une clé ont chacune \(d = @C_att_d@\) coordonnées ; ces \(2d\) coordonnées sont des variables indépendantes, de moyenne 0 et de variance 1. Quel est l'écart-type de leur produit scalaire, avant division par \(\sqrt{d}\) ?",
            "options": [
                {"id": "a", "texte": "@C_att_std@",
                 "explication": r"Oui : la variance vaut \(d = @C_att_d@\) (A1), donc l'écart-type \(\sqrt{d} = @C_att_std@\) ; la division par \(\sqrt{d}\) le ramène à 1."},
                {"id": "b", "texte": "@C_att_d@",
                 "explication": r"C'est la **variance**, pas l'écart-type. Diviser les scores par \(d\) ramènerait leur écart-type à \(1/\sqrt{d}\), trop petit."},
                {"id": "c", "texte": "1",
                 "explication": r"C'est l'écart-type **après** division par \(\sqrt{d}\). Avant, chaque coordonnée apporte une variance 1 et les \(d\) termes s'additionnent."},
                {"id": "d", "texte": "0, car les coordonnées sont centrées.",
                 "explication": r"Centrées, elles donnent un produit scalaire d'**espérance** nulle, pas de dispersion nulle."},
            ],
        },
        "svm": {
            "question": r"Sur des données presque séparables, on multiplie le coefficient \(C\) d'un SVM à marge souple par 100. Qu'attendre ?",
            "options": [
                {"id": "a", "texte": "Une marge plus étroite et moins de violations : le modèle suit davantage les données d'entraînement.",
                 "explication": r"Oui : \(C\) pèse les violations contre \(\tfrac12\lVert \mathbf{w} \rVert^2\) ; le relever autorise un \(\lVert \mathbf{w} \rVert\) plus grand, donc une largeur \(2/\lVert \mathbf{w} \rVert\) plus petite (exemple de la leçon : de \(@N_svm_c1_marge:1m@\) à \(@N_svm_c100_marge:1m@\))."},
                {"id": "b", "texte": "Une marge plus large, car le modèle est plus régularisé.",
                 "explication": r"C'est l'effet inverse : \(C\) est l'inverse d'un coefficient de régularisation (\(\lambda = 1/(Cn)\)). Plus \(C\) est grand, moins le modèle est régularisé."},
                {"id": "c", "texte": "Aucun changement : la solution d'un SVM ne dépend que des vecteurs de support.",
                 "explication": r"Les vecteurs de support eux-mêmes changent avec \(C\), et la borne \(\alpha_i \le C\) aussi : la solution dépend de \(C\)."},
                {"id": "d", "texte": "Toutes les observations deviennent des vecteurs de support.",
                 "explication": r"Au contraire, avec \(C\) grand, les observations bien classées au-delà de la marge ont \(\alpha_i = 0\) ; c'est avec \(C\) petit et une marge très large que beaucoup d'observations se retrouvent dans la marge."},
            ],
        },
        "em": {
            "question": r"Après une itération de EM sur un mélange de deux gaussiennes, la log-vraisemblance a baissé. Que faut-il en conclure ?",
            "options": [
                {"id": "a", "texte": "Il y a une erreur d'implémentation : une itération exacte ne peut pas faire baisser la log-vraisemblance.",
                 "explication": r"Oui : (E2) garantit \(\ell(\boldsymbol{\vartheta}') \ge \ell(\boldsymbol{\vartheta})\) pour une étape E et une étape M exactes. Une baisse au-delà des erreurs d'arrondi signale une faute, par exemple une variance calculée avec l'ancienne moyenne ou des responsabilités non normalisées."},
                {"id": "b", "texte": "L'algorithme est passé par un maximum local, ce qui arrive normalement.",
                 "explication": r"EM peut s'arrêter sur un maximum local, mais il n'y descend jamais : la suite des \(\ell\) est croissante."},
                {"id": "c", "texte": "La vraisemblance est non bornée, donc elle peut aussi baisser.",
                 "explication": r"La non-bornitude permet à \(\ell\) de tendre vers \(+\infty\) (variance qui s'effondre), pas de baisser."},
                {"id": "d", "texte": "C'est normal à la première itération, à cause de l'initialisation.",
                 "explication": r"La monotonie vaut dès la première itération, quel que soit le départ : la preuve n'utilise aucune propriété du point courant."},
            ],
        },
    },
    "cellules_initiales": [
        "import numpy as np\nnp.set_printoptions(precision=4, suppress=True)\n\n# Essaie ici les exemples de la leçon, en les modifiant.\nlam = np.array([1.0, 100.0])\nfor eta in [0.01, 2 / 101, 0.0199, 0.0201]:\n    print(eta, np.abs(1 - eta * lam).max())",
    ],
    "verif": {"script": "verif/seance_ml_4.py"},
}
