"""Leçon express de la séance de mathématiques 4 : chaînes de Markov, martingales, AR(1), mouvement brownien.

Format : voir outils/lecons.py (élément de type "lecon", contenu en Markdown réduit, points de contrôle
[[controle:IDENT]] sur une ligne seule, option "a" correcte, une explication par option).
Aucun nombre calculé n'est écrit à la main : chaque jeton @CLE@, @CLE:3@ ou @CLE:3m@ est remplacé par la valeur
calculée par verif/seance_math_4.py (fichier verif/seance_math_4.out.json).
Renvois : « outil 2 » désigne l'outil 2 de la leçon L-PROBAS-1 (outils/lecons_probas.py, séance de mathématiques 1).
"""

LECON_MATH_4 = {
    "id": "L-MATH-4", "type": "lecon", "section": "lecon",
    "theme": "Leçon express : Markov, martingales, AR(1), mouvement brownien",
    "difficulte": 4, "temps_cible_s": 1200,
    # Aucun notebook ni exemple exécuté : la page affiche la leçon seule.
    "contenu": r"""## Leçon express : Markov, martingales et séries temporelles

Durée visée : 20 minutes. Pour chaque résultat : l'énoncé, une preuve courte mais complète, un exemple calculé, les pièges. La leçon prolonge l'**outil 2** de la séance de mathématiques 1 (premier pas, ruine du joueur, finitude par blocs, identité de Wald), auquel elle renvoie.

**Poids des thèmes.** Marches aléatoires, chaînes de Markov et martingales d'abord ; AR(1) ensuite ; calcul stochastique en dernier et en bref.

**Conventions.** Majuscule : variable aléatoire ; minuscule : valeur ou paramètre fixe ; gras : vecteur ou matrice. Exceptions d'usage : les matrices \(\mathbf{P}\), \(\mathbf{Q}\), \(\mathbf{R}\), \(\mathbf{N}\), \(\mathbf{H}\) de la partie 1 sont fixes ; les chocs \(\varepsilon_t\) de la partie 3 sont aléatoires. \(\mathbb{P}\) (probabilité) n'est pas la matrice \(\mathbf{P}\) ; \(\mathbb{E}\) est une espérance, \(\mathrm{Var}\) une variance, \(\mathrm{Cov}\) une covariance, \(I_A\) l'indicatrice de \(A\), \(a \wedge b = \min(a, b)\). Les lettres \(c\), \(\mu\), \(\sigma\), \(n\) sont redéfinies dans chaque partie. Les nombres viennent de `verif/seance_math_4.py`, qui les vérifie aussi par simulation.

## 1. Chaînes de Markov à états finis

Le problème : un système change d'état au hasard parmi un nombre fini d'états (régime de marché, note de crédit, dernier motif vu), avec des probabilités qui ne dépendent que de l'état présent. Loi dans longtemps, probabilité de finir dans tel état, temps moyen avant l'absorption : chaque fois, un système linéaire.

### Définitions

États \(1, \dots, k\), avec \(k \ge 2\) ; \(X_t\) est l'état à la date entière \(t \ge 0\). \((X_t)\) est une **chaîne de Markov** de matrice de transition \(\mathbf{P}\) (\(k \times k\), coefficients \(p_{ij}\)) si \(\mathbb{P}(X_{t+1} = j \mid X_t = i, X_{t-1} = x_{t-1}, \dots, X_0 = x_0) = p_{ij}\) pour toutes valeurs dont l'événement conditionnant est de probabilité non nulle. La ligne \(i\) de \(\mathbf{P}\) est la loi de l'état suivant depuis \(i\).

La loi de \(X_t\) est le vecteur **ligne** \(\boldsymbol{\nu}_t\), \(\nu_t(j) = \mathbb{P}(X_t = j)\). Une **loi stationnaire** est un vecteur ligne \(\boldsymbol{\pi}\), positif de somme 1, tel que \(\boldsymbol{\pi}\mathbf{P} = \boldsymbol{\pi}\). La chaîne est **irréductible** si de tout état \(i\) on atteint tout état \(j\) par des états \(i = s_0, \dots, s_l = j\) avec \(p_{s_0 s_1} > 0, \dots, p_{s_{l-1} s_l} > 0\).

### Loi stationnaire

Énoncé.

1. \(\boldsymbol{\nu}_{t+1} = \boldsymbol{\nu}_t \mathbf{P}\), donc \(\boldsymbol{\nu}_t = \boldsymbol{\nu}_0 \mathbf{P}^t\) ; si \(\boldsymbol{\nu}_0 = \boldsymbol{\pi}\), alors \(\boldsymbol{\nu}_t = \boldsymbol{\pi}\) pour tout \(t\).
2. Toute chaîne finie a au moins une loi stationnaire ; si elle est irréductible, cette loi est unique.
3. Admis (théorème de convergence des chaînes finies) : si la chaîne est irréductible et apériodique (pour chaque état, le pgcd des longueurs des chemins qui en partent et y reviennent vaut 1), \(\boldsymbol{\nu}_t \to \boldsymbol{\pi}\) quelle que soit \(\boldsymbol{\nu}_0\). Démontré ci-dessous pour deux états.

Preuve.

1. En sommant la définition sur les valeurs passées, \(\mathbb{P}(X_{t+1} = j \mid X_t = i) = p_{ij}\) ; par les probabilités totales selon \(X_t\), \(\nu_{t+1}(j) = \sum_i \nu_t(i)\,p_{ij}\).
2. Existence. Pour une loi \(\boldsymbol{\nu}_0\), la moyenne \(\bar{\boldsymbol{\nu}}_s = \tfrac1s(\boldsymbol{\nu}_0 + \dots + \boldsymbol{\nu}_{s-1})\) est une loi, et par télescopage \(\bar{\boldsymbol{\nu}}_s\mathbf{P} - \bar{\boldsymbol{\nu}}_s = \tfrac1s(\boldsymbol{\nu}_s - \boldsymbol{\nu}_0)\), de coordonnées au plus \(1/s\) en valeur absolue. Les lois forment un fermé borné de \(\mathbb{R}^k\) : une sous-suite converge (Bolzano-Weierstrass) vers une loi \(\boldsymbol{\pi}\), qui vérifie \(\boldsymbol{\pi}\mathbf{P} = \boldsymbol{\pi}\).
3. Unicité. Si \(\mathbf{P}\mathbf{f} = \mathbf{f}\) (vecteur colonne), chaque \(f_i = \sum_j p_{ij} f_j\) est une moyenne pondérée. En un \(i\) où \(f\) est maximal, tous les \(f_j\) de poids \(p_{ij} > 0\) valent ce maximum ; par irréductibilité, de proche en proche, \(\mathbf{f}\) est constant (principe du maximum, outil 2, étape 5). Le noyau de \(\mathbf{P} - \mathbf{I}\) est la droite de \(\mathbf{1}\) (vecteur de 1), donc \(\mathbf{P} - \mathbf{I}\) est de rang \(k - 1\), comme sa transposée : les solutions de \(\mathbf{x}(\mathbf{P} - \mathbf{I}) = 0\) forment une droite, qui contient une seule loi.

Exemple, deux régimes : calme (1) devient agité (2) avec probabilité \(a\), agité redevient calme avec probabilité \(b\), \(a\) et \(b\) dans \([0, 1]\) non tous deux nuls ; \(\mathbf{P} = \begin{pmatrix} 1 - a & a \\ b & 1 - b \end{pmatrix}\).

1. La coordonnée 1 de \(\boldsymbol{\pi}\mathbf{P} = \boldsymbol{\pi}\) s'écrit \(\pi_1(1 - a) + \pi_2 b = \pi_1\), soit \(\pi_1 a = \pi_2 b\) (flux sortant égal au flux entrant) ; la coordonnée 2 donne la même équation. Avec \(\pi_1 + \pi_2 = 1\) : \(\pi_1 = b/(a+b)\), \(\pi_2 = a/(a+b)\).
2. Convergence : \(u_t = \nu_t(2)\) vérifie \(u_{t+1} = a\,(1 - u_t) + (1 - b)\,u_t = a + \lambda u_t\) avec \(\lambda = 1 - a - b\) ; comme \(\pi_2 = a + \lambda \pi_2\), \(u_t - \pi_2 = \lambda^t (u_0 - \pi_2)\). Si \(0 < a + b < 2\), \(\lvert \lambda \rvert < 1\) et la loi converge vers \(\boldsymbol{\pi}\), quel que soit le départ.
3. \(a = 0{,}1\), \(b = 0{,}4\) : \(\boldsymbol{\pi} = (@La_pi1:1m@ ; @La_pi2:1m@)\), \(\lambda = @La_lambda:1m@\) ; départ calme, \(u_3 = \pi_2(1 - \lambda^3) = @La_u3:3m@\) (simulation : @La_u3_mc:3@).

Cas limites : \(a = b = 1\) donne \(\lambda = -1\), la chaîne alterne et ne converge pas, bien que \(\boldsymbol{\pi} = (1/2 ; 1/2)\) soit unique (chaîne périodique) ; \(a = b = 0\) : toute loi est stationnaire (chaîne non irréductible).

### Absorption : le système linéaire

Cadre. Un état \(e\) est **absorbant** si \(p_{ee} = 1\), les autres sont **transitoires**. Hypothèse (A) : depuis chaque état transitoire, un chemin de probabilité non nulle mène à un état absorbant. \(m\) est le nombre d'états transitoires, \(\mathbf{Q}\) (\(m \times m\), coefficients \(\mathbf{Q}_{ij}\)) la matrice des transitions entre états transitoires, \(\mathbf{R}\) (coefficients \(\mathbf{R}_{ie}\), une colonne par état absorbant) celle des transitions vers les états absorbants. \(T\) est la première date d'absorption ; pour \(i\) transitoire, \(d_i = \mathbb{E}[T \mid X_0 = i]\) (vecteur colonne \(\mathbf{d}\)) et \(h_{ie} = \mathbb{P}(X_T = e \mid X_0 = i)\) (matrice \(\mathbf{H}\)).

Énoncé. Sous (A), \(\mathbf{I} - \mathbf{Q}\) est inversible, \(\mathbf{N} = (\mathbf{I} - \mathbf{Q})^{-1} = \sum_{t \ge 0} \mathbf{Q}^t\), et

\(\displaystyle (\mathbf{I} - \mathbf{Q})\,\mathbf{d} = \mathbf{1}, \qquad (\mathbf{I} - \mathbf{Q})\,\mathbf{H} = \mathbf{R}.\)

Ligne par ligne, \(d_i = 1 + \sum_j \mathbf{Q}_{ij} d_j\) et \(h_{ie} = \mathbf{R}_{ie} + \sum_j \mathbf{Q}_{ij} h_{je}\) : les équations du premier pas de l'outil 2, pour une chaîne quelconque.

Preuve.

1. Pour \(i\), \(j\) transitoires, \((\mathbf{Q}^t)_{ij} = \mathbb{P}(X_t = j, T > t \mid X_0 = i)\), par récurrence sur \(t\) : être en \(j\) à la date \(t + 1\) sans absorption, c'est être en un état transitoire \(l\) à la date \(t\) sans absorption puis passer de \(l\) à \(j\) (probabilité \(\mathbf{Q}_{lj}\), propriété de Markov) ; on somme sur \(l\).
2. Blocs (outil 2, étape 1) : un plus court chemin vers l'absorption ne repasse par aucun état, donc a au plus \(m\) pas ; soit \(\delta > 0\) la plus petite des probabilités de ces chemins. Depuis tout état transitoire, l'absorption dans les \(m\) pas suivants a probabilité au moins \(\delta\), d'où \(\mathbb{P}(T > lm \mid X_0 = i) \le (1 - \delta)^l\).
3. Par 1, \(\sum_j (\mathbf{Q}^t)_{ij} = \mathbb{P}(T > t \mid X_0 = i)\), de somme sur \(t\) au plus \(m/\delta\) par 2 : la série positive \(\sum_t \mathbf{Q}^t\) converge vers une matrice \(\mathbf{N}\), et \(\mathbf{Q}^t \to 0\). Comme \((\mathbf{I} - \mathbf{Q})(\mathbf{I} + \dots + \mathbf{Q}^{s-1}) = \mathbf{I} - \mathbf{Q}^s\), à la limite \((\mathbf{I} - \mathbf{Q})\,\mathbf{N} = \mathbf{I}\).
4. \(d_i = \sum_{t \ge 0} \mathbb{P}(T > t \mid X_0 = i)\) (outil 2, étape 2) \(= (\mathbf{N}\mathbf{1})_i\).
5. Être absorbé en \(e\) à la date \(t + 1\), c'est être en \(j\) transitoire à la date \(t\) sans absorption puis sauter en \(e\) : probabilité \(\sum_j (\mathbf{Q}^t)_{ij}\,\mathbf{R}_{je}\). En sommant sur \(t\), \(\mathbf{H} = \mathbf{N}\mathbf{R}\).

Exemple 1, attendre pile, face, pile (PFP) avec une pièce équilibrée. L'état est le plus long suffixe des lancers faits qui commence PFP : « vide », « P » ou « PF ». Depuis « vide », P mène à « P », F laisse « vide » ; depuis « P », P laisse « P », F mène à « PF » ; depuis « PF », P termine, F ramène à « vide ». Le système \((\mathbf{I} - \mathbf{Q})\,\mathbf{d} = \mathbf{1}\) :

1. \(d_{\text{PF}} = 1 + \tfrac12 d_{\text{vide}}\) ;
2. \(d_{\text{P}} = 1 + \tfrac12 d_{\text{P}} + \tfrac12 d_{\text{PF}}\), soit \(d_{\text{P}} = 2 + d_{\text{PF}}\) ;
3. \(d_{\text{vide}} = 1 + \tfrac12 d_{\text{vide}} + \tfrac12 d_{\text{P}}\), soit \(d_{\text{vide}} = 2 + d_{\text{P}} = 4 + d_{\text{PF}} = 5 + \tfrac12 d_{\text{vide}}\).

D'où \(d_{\text{vide}} = @La_pfp_d0@\), \(d_{\text{P}} = @La_pfp_d1@\), \(d_{\text{PF}} = @La_pfp_d2@\) (simulation : @La_pfp_mc:2@), plus que \(1/\mathbb{P}(\text{PFP}) = @La_pfp_inv_p@\), car un F après « PF » ramène à « vide ».

Exemple 2, un prêt sain (S) ou fragile (F) finit en défaut (D) ou remboursé (R), absorbants. Chaque mois, depuis S : S avec probabilité 0,7, F avec 0,2, R avec 0,1 ; depuis F : S avec 0,3, F avec 0,4, D avec 0,2, R avec 0,1. Donc \(\mathbf{Q} = \begin{pmatrix} 0{,}7 & 0{,}2 \\ 0{,}3 & 0{,}4 \end{pmatrix}\), \(\mathbf{R} = \begin{pmatrix} 0 & 0{,}1 \\ 0{,}2 & 0{,}1 \end{pmatrix}\) (colonnes D puis R), et la résolution donne \(\mathbf{H} = \begin{pmatrix} @La_cr_h_sd_num@/@La_cr_h_sd_den@ & @La_cr_h_sr_num@/@La_cr_h_sr_den@ \\ @La_cr_h_fd_num@/@La_cr_h_fd_den@ & @La_cr_h_fr_num@/@La_cr_h_fr_den@ \end{pmatrix}\) : un prêt sain fait défaut avec probabilité \(@La_cr_h_sd_num@/@La_cr_h_sd_den@\) (simulation : @La_cr_h_sd_mc:3@). Chaque ligne de \(\mathbf{H}\) somme à 1.

Pièges :

1. \(\boldsymbol{\pi}\) est un vecteur ligne à gauche, \(\boldsymbol{\pi}\mathbf{P} = \boldsymbol{\pi}\) ; le système \(\boldsymbol{\pi}(\mathbf{P} - \mathbf{I}) = 0\) est de rang \(k - 1\) : remplacer une équation par \(\sum_j \pi_j = 1\).
2. Le « 1 + » appartient à l'équation de durée, pas à celle de probabilité.
3. Pour un motif, après PP on est en « P », pas en « vide » ; loi stationnaire n'est pas loi limite (chaîne périodique).

[[controle:absorption]]

## 2. Martingales et théorème d'arrêt

Le problème : le premier pas demande un système par question. Une martingale (quantité d'espérance constante), évaluée au moment de l'arrêt, donne la réponse en une ligne, à condition que le théorème d'arrêt de Doob s'applique.

### Cadre et définitions

Cadre de l'outil 2 : pas \(X_1, X_2, \dots\) indépendants, +1 avec probabilité \(p\), −1 avec probabilité \(q = 1 - p\) ; \(Z_t = i + X_1 + \dots + X_t\) ; \(\mathbf{X} = (X_1, \dots, X_t)\). Pour \(Y\) d'espérance finie, \(\mathbb{E}[Y \mid \mathbf{X}]\) est la variable \(g(\mathbf{X})\) où \(g(\mathbf{x}) = \mathbb{E}[Y \mid \mathbf{X} = \mathbf{x}]\). Deux règles :

- (R1) pour \(u\) bornée, \(\mathbb{E}[u(\mathbf{X})\,Y] = \mathbb{E}\big[u(\mathbf{X})\,\mathbb{E}[Y \mid \mathbf{X}]\big]\) : les deux membres valent \(\sum_{\mathbf{x}} u(\mathbf{x})\,\mathbb{E}[Y\,I_{\{\mathbf{X} = \mathbf{x}\}}]\), série absolument convergente. Avec \(u = 1\) : \(\mathbb{E}\big[\mathbb{E}[Y \mid \mathbf{X}]\big] = \mathbb{E}[Y]\).
- (R2) \(\mathbb{E}[w(Z_t, X_{t+1}) \mid \mathbf{X}] = \psi(Z_t)\) avec \(\psi(z) = \mathbb{E}[w(z, X_{t+1})]\) (si ces espérances sont finies) : sachant \(\mathbf{X} = \mathbf{x}\), \(Z_t\) est fixé et \(X_{t+1}\), indépendant de \(\mathbf{X}\), garde sa loi.

Une **martingale** est une suite \(M_0, M_1, \dots\), \(M_0\) constante, \(M_t\) fonction de \(X_1, \dots, X_t\) d'espérance finie, avec \(\mathbb{E}[M_{t+1} \mid X_1, \dots, X_t] = M_t\) ; par (R1) et récurrence, \(\mathbb{E}[M_t] = M_0\) pour tout \(t\) **fixé**. Un **temps d'arrêt** est une variable \(T\) à valeurs dans \(\{0, 1, \dots\} \cup \{+\infty\}\) telle que l'indicatrice de \(\{T \le t\}\) soit une fonction de \(X_1, \dots, X_t\) pour tout \(t\) : on s'arrête sans regarder le futur (premier passage en 0 ou \(n\) : oui ; instant du maximum sur \([0, 100]\) : non).

### Trois martingales de la marche

\(\lvert Z_t \rvert \le \lvert i \rvert + t\) : toutes les espérances sont finies. Par (R2) :

1. \(Z_t - (p - q)\,t\), et \(Z_t\) si \(p = 1/2\) : \(\mathbb{E}[Z_t + X_{t+1} \mid \mathbf{X}] = Z_t + (p - q)\).
2. \(Z_t^2 - t\), si \(p = 1/2\) : \(\mathbb{E}[(Z_t + X_{t+1})^2 \mid \mathbf{X}] = Z_t^2 + 2 Z_t\,\mathbb{E}[X_{t+1}] + \mathbb{E}[X_{t+1}^2] = Z_t^2 + 1\).
3. \(r^{Z_t}\), \(r = q/p\), pour tout \(p\) dans \(]0, 1[\) : \(\mathbb{E}[r^{Z_t + X_{t+1}} \mid \mathbf{X}] = r^{Z_t}\,(p\,r + q/r) = r^{Z_t}\,(q + p) = r^{Z_t}\).

### Le théorème d'arrêt de Doob

Énoncé. Si \(M\) est une martingale et \(T\) un temps d'arrêt, \(\mathbb{E}[M_T] = M_0\) dans chacun des cas suivants (\(c\) constante) :

- (a) \(T \le c\) avec probabilité 1, pour un entier \(c\) ;
- (b) \(T\) est fini avec probabilité 1 et \(\lvert M_{t \wedge T} \rvert \le c\) pour tout \(t\) ;
- (c) \(\mathbb{E}[T] < \infty\) et \(\lvert M_{t+1} - M_t \rvert \le c\) pour tout \(t\).

Preuve de (a).

1. Télescopage : \(M_T = M_0 + \sum_{k=1}^{c} (M_k - M_{k-1})\,I_{\{T \ge k\}}\).
2. \(I_{\{T \ge k\}} = 1 - I_{\{T \le k-1\}}\) est une fonction bornée de \(X_1, \dots, X_{k-1}\) (temps d'arrêt).
3. Par (R1) : \(\mathbb{E}[(M_k - M_{k-1})\,I_{\{T \ge k\}}] = \mathbb{E}\big[I_{\{T \ge k\}}\,(\mathbb{E}[M_k \mid X_1, \dots, X_{k-1}] - M_{k-1})\big] = 0\).
4. Somme finie : \(\mathbb{E}[M_T] = M_0\).

Passage à la limite, (b) et (c).

5. \(T \wedge t\) est un temps d'arrêt borné par \(t\) (\(\{T \wedge t \le s\}\) vaut \(\{T \le s\}\) si \(s < t\), l'événement certain sinon) : par (a), \(\mathbb{E}[M_{T \wedge t}] = M_0\) pour tout \(t\).
6. Si \(T < \infty\), \(M_{T \wedge t} = M_T\) dès que \(t \ge T\).
7. En (b), \(\lvert M_{T \wedge t} \rvert \le c\) ; en (c), \(T\) est fini avec probabilité 1 et \(\lvert M_{T \wedge t} \rvert \le \lvert M_0 \rvert + c\,T\), d'espérance finie. La convergence dominée (admise) donne \(\mathbb{E}[M_T] = \lim_t \mathbb{E}[M_{T \wedge t}] = M_0\).

### Applications : ruine et durée en une ligne

Barrières 0 et \(n\), départ \(0 < i < n\), \(T\) le premier passage en 0 ou \(n\), fini avec probabilité 1 (outil 2, étape 1), \(h\) la probabilité de finir en \(n\).

- Ruine, \(p \ne 1/2\) : \(0 \le Z_{t \wedge T} \le n\), donc \(r^{Z_{t \wedge T}} \le \max(1, r^n)\) : cas (b), \(h\,r^n + (1 - h) = r^i\), soit \(h = \dfrac{1 - r^i}{1 - r^n}\). Avec \(p = 2/3\), \(i = 2\), \(n = 5\) : \(r = @Lb_r:1m@\), \(h = \tfrac{@Lb_h_num@}{@Lb_h_den@}\), comme à l'outil 2 (simulation : @Lb_h_mc:3@).
- Ruine, \(p = 1/2\) : \(Z_t\), arrêtée bornée par \(n\) : \(n\,h = i\), \(h = i/n\).
- Durée, \(p = 1/2\), avec \(Z_t^2 - t\), qui n'est ni dans (b) ni, faute de savoir \(\mathbb{E}[T]\) fini, dans (c). Étape 5 : \(\mathbb{E}[Z_{T \wedge t}^2] - \mathbb{E}[T \wedge t] = i^2\). Comme \(Z_{T \wedge t}^2 \le n^2\), \(\mathbb{E}[T \wedge t] \le n^2 - i^2\), et la convergence monotone (admise) donne \(\mathbb{E}[T] < \infty\). À la limite (dominée pour \(Z^2\), monotone pour \(T\)) : \(\mathbb{E}[T] = \mathbb{E}[Z_T^2] - i^2 = n^2 \cdot i/n - i^2 = i\,(n - i)\) ; pour \(i = 2\), \(n = 5\) : @Lb_d_sym@ (simulation : @Lb_d_sym_mc:2@).
- Durée, \(p \ne 1/2\) : \(Z_t - (p - q)\,t\) a des accroissements bornés par 2 et \(\mathbb{E}[T] < \infty\) (outil 2, étape 2) ; le cas (c) donne \(\mathbb{E}[Z_T] - (p - q)\,\mathbb{E}[T] = i\) : l'**identité de Wald**, démontrée à l'outil 2, d'où \(\mathbb{E}[T] = (n\,h - i)/(p - q)\).

### Le contre-exemple : une seule barrière

Marche symétrique partie de 0, \(T\) le premier passage en 1. La marche \(1 - Z_t\) est symétrique, partie de 1, et \(T\) est son premier passage en 0 : par l'outil 2 (cas limites), \(T\) est fini avec probabilité 1 et \(\mathbb{E}[T] = +\infty\). Or \(Z_T = 1\), donc \(\mathbb{E}[Z_T] = 1 \ne 0 = Z_0\) : \(T\) n'est pas borné (a), \(Z_{t \wedge T}\) n'est pas minoré (b), \(\mathbb{E}[T]\) est infini (c). Pour tout \(t\) pourtant \(\mathbb{E}[Z_{T \wedge t}] = 0\) ; calcul exact à \(t = @Lb_ce_t@\) : \(\mathbb{P}(T > @Lb_ce_t@) = @Lb_ce_pT:4m@\) et \(\mathbb{E}[Z_{@Lb_ce_t@} \mid T > @Lb_ce_t@] = @Lb_ce_cond:2m@\). Une perte de plus en plus grande sur un événement de plus en plus rare compense le gain, puis disparaît de l'espérance à la limite.

Pièges :

1. Écrire \(\mathbb{E}[M_T] = M_0\) sans vérifier (a), (b) ou (c).
2. \(r = p/q\) au lieu de \(q/p\) : \(\mathbb{E}[(p/q)^X] = p^2/q + q^2/p\), égal à 1 seulement si \(p = 1/2\).
3. \(Z_t^2 - t\) n'est une martingale que si \(p = 1/2\) ; sinon, \(\big(Z_t - (p - q)\,t\big)^2 - \big(1 - (p - q)^2\big)\,t\) l'est (même calcul).

[[controle:arret]]

## 3. Le processus AR(1)

Le problème : un spread ou un signal s'écarte de sa moyenne puis y revient. Modèle le plus simple : une fraction de la valeur précédente, plus un choc.

### Définitions et énoncé

\(c\), \(\phi\) et \(\sigma > 0\) sont des réels fixés. Les chocs \(\varepsilon_1, \varepsilon_2, \dots\) sont indépendants, de même loi, d'espérance 0 et de variance \(\sigma^2\) ; \(Y_t = c + \phi\,Y_{t-1} + \varepsilon_t\) pour \(t \ge 1\), avec \(Y_0\) indépendant des chocs, donc \(\varepsilon_t\) indépendant de \(Y_0, \dots, Y_{t-1}\). Le processus est **stationnaire** (au sens faible) si \(\mathbb{E}[Y_t]\) et \(\mathrm{Var}(Y_t) < \infty\) ne dépendent pas de \(t\) et \(\mathrm{Cov}(Y_t, Y_{t+h})\) ne dépend que de \(h\) ; on note alors \(\mu\) la moyenne, \(\gamma_h = \mathrm{Cov}(Y_t, Y_{t+h})\) l'autocovariance au retard \(h \ge 0\), \(\rho_h = \gamma_h/\gamma_0\) l'autocorrélation.

1. Si \(\lvert \phi \rvert < 1\) et si \(Y_0\) a pour espérance \(\mu = c/(1 - \phi)\) et pour variance \(\sigma^2/(1 - \phi^2)\), le processus est stationnaire, avec \(\gamma_0 = \dfrac{\sigma^2}{1 - \phi^2}\), \(\gamma_h = \phi^h \gamma_0\), \(\rho_h = \phi^h\).
2. Si \(\lvert \phi \rvert \ge 1\), aucun choix de \(Y_0\) ne le rend stationnaire.

Preuve.

1. \(\mathbb{E}[Y_t] = c + \phi\,\mathbb{E}[Y_{t-1}]\), et \(c + \phi c/(1 - \phi) = c/(1 - \phi)\) : récurrence.
2. \(\varepsilon_t\) indépendant de \(Y_{t-1}\) : \(\mathrm{Var}(Y_t) = \phi^2\,\mathrm{Var}(Y_{t-1}) + \sigma^2\), et \(\phi^2\sigma^2/(1 - \phi^2) + \sigma^2 = \sigma^2/(1 - \phi^2)\) : récurrence.
3. Pour \(h \ge 1\), \(\mathrm{Cov}(Y_t, Y_{t+h}) = \mathrm{Cov}(Y_t, c + \phi\,Y_{t+h-1} + \varepsilon_{t+h}) = \phi\,\mathrm{Cov}(Y_t, Y_{t+h-1})\) (constante de covariance nulle, \(\varepsilon_{t+h}\) indépendant de \(Y_t\)) ; par récurrence, \(\phi^h \gamma_0\), indépendant de \(t\).
4. Énoncé 2 : la stationnarité et l'étape 2 donneraient \(\gamma_0\,(1 - \phi^2) = \sigma^2 > 0\), impossible si \(\phi^2 \ge 1\) puisque \(\gamma_0 \ge 0\).

Exemple : \(c = 1\), \(\phi = 0{,}8\), \(\sigma = 1\) : \(\mu = 1/@Lc_un_moins_phi:1m@ = @Lc_mu@\), \(\gamma_0 = 1/@Lc_un_moins_phi2:2m@ = \tfrac{@Lc_g0_num@}{@Lc_g0_den@} \approx @Lc_g0:3m@\), \(\rho_3 = 0{,}8^3 = @Lc_rho3:3m@\), \(\gamma_3 \approx @Lc_g3:3m@\) (simulation avec des chocs uniformes, seules leur espérance et leur variance comptant : @Lc_g3_mc:2@).

Cas limite, la marche aléatoire : \(\phi = 1\), \(c = 0\), \(Y_0 = y_0\) fixe. \(Y_t = y_0 + \varepsilon_1 + \dots + \varepsilon_t\), \(\mathrm{Var}(Y_t) = t\,\sigma^2\) croît sans borne ; pour \(h \ge 0\), \(\mathrm{Cov}(Y_t, Y_{t+h}) = t\,\sigma^2\) (les \(t\) premiers chocs sont communs) dépend de \(t\), et \(\mathrm{Corr}(Y_t, Y_{t+h}) = \sqrt{t/(t+h)}\), soit \(@Lc_rw_corr:3m@\) pour \(t = 100\), \(h = 10\). Des corrélations proches de 1 ne prouvent donc aucun retour à la moyenne ; les accroissements \(\varepsilon_t\), eux, sont indépendants.

Pièges :

1. \(\rho_h = \phi^h\), ni \(\phi\) ni \(h\phi\) ; le signe alterne si \(\phi < 0\). Stationnarité : \(\lvert \phi \rvert < 1\), pas \(\phi < 1\).
2. \(\gamma_0 = \sigma^2/(1 - \phi^2)\), pas \(\sigma^2/(1 - \phi)\), et pas \(\sigma^2\).
3. \(\mathrm{Var}(Y_1 + \dots + Y_s) = \sum_{u=1}^{s} \sum_{v=1}^{s} \gamma_{\lvert u - v \rvert}\) : chaque paire de dates distinctes compte deux fois.

[[controle:ar1]]

## 4. Mouvement brownien et lemme d'Itô

Thème jugé le moins probable (introduction) : l'essentiel seulement.

### Définition et propriétés

Un **mouvement brownien standard** est une famille \(W_t\), \(t \ge 0\) réel, avec \(W_0 = 0\), \(W_t - W_s\) de loi normale \(\mathcal{N}(0, t - s)\) pour \(s < t\), des accroissements indépendants sur des intervalles disjoints et des trajectoires continues. C'est la limite de la marche symétrique renormalisée : \(Z_{\lfloor jt \rfloor}/\sqrt{j}\) a une loi proche de \(\mathcal{N}(0, t)\) pour \(j\) grand (théorème central limite).

1. \(\mathrm{Cov}(W_s, W_t) = \min(s, t)\) : pour \(s \le t\), \(W_t = W_s + (W_t - W_s)\), termes indépendants, donc \(\mathrm{Cov}(W_s, W_t) = \mathrm{Var}(W_s) = s\).
2. Si \(G\) suit \(\mathcal{N}(0, v)\), \(\mathbb{E}[e^{\theta G}] = e^{\theta^2 v/2}\) pour tout réel \(\theta\) : \(\theta x - x^2/(2v) = -(x - \theta v)^2/(2v) + \theta^2 v/2\), et la densité de \(\mathcal{N}(\theta v, v)\) intègre à 1. Ainsi \(\mathbb{E}[e^{W_1}] = e^{1/2} \approx @Ld_ew1:4m@\) (simulation : @Ld_ew1_mc:4@), et non 1.
3. Variation quadratique (énoncé) : pour une subdivision \(0 = t_0 < \dots < t_n = t\) de pas maximal \(\Delta\), \(V_n = \sum_l (W_{t_{l+1}} - W_{t_l})^2\) vérifie \(\mathbb{E}[V_n] = t\) et \(\mathrm{Var}(V_n) = 2\sum_l (t_{l+1} - t_l)^2 \le 2\,t\,\Delta\) (carrés indépendants, et \(\mathbb{E}[G^4] = 3v^2\), coefficient de \(\theta^4\) dans \(e^{\theta^2 v/2}\) multiplié par \(4!\)) : \(V_n \to t\). Pour une fonction de dérivée bornée, la même somme tend vers 0. Lecture abrégée : \((dW_t)^2 = dt\).

### Le lemme d'Itô (énoncé admis)

Pour \(G_s\) ne dépendant que de \((W_u)_{u \le s}\), \(\int_0^t G_s\,dW_s\) est la limite en moyenne quadratique de \(\sum_l G_{t_l}(W_{t_{l+1}} - W_{t_l})\), \(G\) pris au point **gauche** ; admis : si \(\mathbb{E}\big[\int_0^t G_s^2\,ds\big] < \infty\), son espérance est nulle. Pour \(f(t, x)\) déterministe, une fois continûment dérivable en \(t\) et deux fois en \(x\) :

\(\displaystyle f(t, W_t) = f(0, 0) + \int_0^t \Big(f_t + \tfrac12 f_{xx}\Big)(s, W_s)\,ds + \int_0^t f_x(s, W_s)\,dW_s,\)

noté \(df = \big(f_t + \tfrac12 f_{xx}\big)\,dt + f_x\,dW_t\). Intuition (pas une preuve) : Taylor à l'ordre 2 avec \((dW_t)^2 = dt\) ; le terme \(\tfrac12 f_{xx}\,dt\) est la seule différence avec le calcul ordinaire.

1. \(f(x) = x^2\) : \(W_t^2 = t + 2\int_0^t W_s\,dW_s\), donc \(\int_0^t W_s\,dW_s = (W_t^2 - t)/2\), et non \(W_t^2/2\). Contrôle direct : la somme à gauche vaut exactement \((W_t^2 - V_n)/2\) (sommer \(W_{t_{l+1}}^2 - W_{t_l}^2 = 2W_{t_l}(W_{t_{l+1}} - W_{t_l}) + (W_{t_{l+1}} - W_{t_l})^2\)), et \(V_n \to t\).
2. \(f(x) = e^x\) : \(d\,e^{W_t} = \tfrac12 e^{W_t}\,dt + e^{W_t}\,dW_t\) ; en espérance, \(m(t) = \mathbb{E}[e^{W_t}]\) vérifie \(m(t) = 1 + \tfrac12\int_0^t m(s)\,ds\), donc \(m(t) = e^{t/2}\) : la propriété 2 par une autre route.

### Le mouvement brownien géométrique

\(\mu\) (dérive) et \(\sigma > 0\) (volatilité), par unité de temps, et \(s_0 > 0\) sont fixés ; \(dS_t = \mu\,S_t\,dt + \sigma\,S_t\,dW_t\), \(S_0 = s_0\).

Énoncé : \(S_t = s_0 \exp\big((\mu - \sigma^2/2)\,t + \sigma W_t\big)\) est solution (unicité admise), \(\mathbb{E}[S_t] = s_0\,e^{\mu t}\), et la médiane vaut \(s_0\,e^{(\mu - \sigma^2/2)\,t}\). Preuve : avec \(f(t, x) = s_0 \exp\big((\mu - \sigma^2/2)\,t + \sigma x\big)\), \(f_t = (\mu - \sigma^2/2) f\), \(f_x = \sigma f\), \(f_{xx} = \sigma^2 f\), donc \(df = \mu f\,dt + \sigma f\,dW_t\) ; la propriété 2 avec \(\theta = \sigma\), \(v = t\) donne \(\mathbb{E}[e^{\sigma W_t}] = e^{\sigma^2 t/2}\), qui compense le \(-\sigma^2 t/2\) ; \(\ln S_t\) est normale, et l'exponentielle croissante conserve sa médiane, égale à sa moyenne.

Exemple : \(s_0 = 100\), \(\mu = 0{,}08\), \(\sigma = 0{,}2\), un an : \(\mathbb{E}[S_1] = 100\,e^{0{,}08} \approx @Ld_gbm_e:2m@\) (simulation : @Ld_gbm_e_mc:2@), médiane \(100\,e^{@Ld_gbm_drift_log:2m@} \approx @Ld_gbm_med:2m@\), et \(\mathbb{P}(S_1 > 100) = \Phi(@Ld_gbm_z:1m@) \approx @Ld_gbm_p:4m@\), où \(\Phi\) est la fonction de répartition de \(\mathcal{N}(0, 1)\).

Pièges :

1. \(\mathbb{E}[S_t] = s_0\,e^{\mu t}\) ; \(s_0\,e^{(\mu - \sigma^2/2)\,t}\) est la médiane.
2. Le terme \(\tfrac12 f_{xx}\,dt\) ne s'oublie jamais : \(\int_0^t W_s\,dW_s \ne W_t^2/2\), \(\mathbb{E}[e^{W_t}] \ne 1\).
3. \(\mathrm{Cov}(W_s, W_t) = \min(s, t)\), ni \(s\,t\) ni 0.

[[controle:ito]]

## Réflexes en 30 secondes

| Indice dans l'énoncé | Premier geste |
|---|---|
| « à long terme », matrice de transition | \(\boldsymbol{\pi}\mathbf{P} = \boldsymbol{\pi}\), somme 1 ; deux états : \(\pi_2 = a/(a+b)\) |
| « temps moyen », « jusqu'au motif » | états = plus long suffixe utile ; \(d_i = 1 + \sum_j \mathbf{Q}_{ij} d_j\) |
| « probabilité de finir en ... » | \(h_{ie} = \mathbf{R}_{ie} + \sum_j \mathbf{Q}_{ij} h_{je}\) |
| marche biaisée, atteindre un niveau | \(r^{Z_t}\), \(r = q/p\) ; sans barrière basse et \(p < 1/2\) : \((p/q)^k\) pour monter de \(k\) |
| « théorème d'arrêt applicable ? » | (a) \(T\) borné ; (b) arrêtée bornée ; (c) \(\mathbb{E}[T]\) fini et accroissements bornés |
| AR(1) | \(\lvert \phi \rvert < 1\) ; \(\mu = c/(1 - \phi)\), \(\gamma_0 = \sigma^2/(1 - \phi^2)\), \(\rho_h = \phi^h\) |
| brownien | \(\min(s, t)\) ; \(\mathbb{E}[e^{\theta W_t}] = e^{\theta^2 t/2}\) ; Itô : \(\big(f_t + \tfrac12 f_{xx}\big)\,dt + f_x\,dW_t\) |
| brownien géométrique | \(\mathbb{E}[S_t] = s_0 e^{\mu t}\) ; \(\ln S_t\) normale, moyenne \(\ln s_0 + (\mu - \sigma^2/2)\,t\), variance \(\sigma^2 t\) |

## À savoir refaire sans notes

- La loi stationnaire à deux états et la convergence en \(\lambda^t\) ; l'unicité par le principe du maximum.
- \((\mathbf{I} - \mathbf{Q})\,\mathbf{d} = \mathbf{1}\) et \((\mathbf{I} - \mathbf{Q})\,\mathbf{H} = \mathbf{R}\) par \((\mathbf{Q}^t)_{ij} = \mathbb{P}(X_t = j, T > t)\) ; l'attente de PFP.
- Les trois martingales de la marche, le théorème d'arrêt borné par télescopage et le passage à la limite ; ruine, durée, et le contre-exemple du premier passage en 1.
- Moyenne, variance et autocorrélation de l'AR(1) ; la non-stationnarité de la marche aléatoire.
- \(\min(s, t)\), \(\mathbb{E}[e^{\theta G}] = e^{\theta^2 v/2}\), Itô sur \(W_t^2\) et \(e^{W_t}\), \(\mathbb{E}[S_t] = s_0 e^{\mu t}\).
""",
    "controles": {
        "absorption": {
            "question": r"Dans l'exemple du prêt (S sain, F fragile ; D défaut et R remboursé absorbants), on note \(h_S\) et \(h_F\) les probabilités de finir en défaut depuis S et depuis F. Quelle équation est juste ?",
            "options": [
                {"id": "a", "texte": r"\(h_F = 0{,}2 + 0{,}3\,h_S + 0{,}4\,h_F\)",
                 "explication": r"Oui : c'est la ligne F de \((\mathbf{I} - \mathbf{Q})\,\mathbf{H} = \mathbf{R}\), colonne D. Défaut immédiat avec probabilité 0,2, ou passage par S ou F puis défaut plus tard ; le remboursement (0,1) contribue 0."},
                {"id": "b", "texte": r"\(h_F = 1 + 0{,}3\,h_S + 0{,}4\,h_F\)",
                 "explication": r"C'est la forme de l'équation de durée : le « 1 + » compte le pas effectué. Une probabilité d'absorption n'a pas de terme 1, mais la probabilité d'être absorbé au premier pas dans l'état visé, ici 0,2."},
                {"id": "c", "texte": r"\(h_F = 0{,}2\)",
                 "explication": r"C'est la probabilité de faire défaut **au premier mois**. Un prêt fragile qui reste fragile ou redevient sain peut encore faire défaut ensuite : la vraie valeur, \(@La_cr_h_fd_num@/@La_cr_h_fd_den@\), est plus grande."},
                {"id": "d", "texte": r"\(h_F\) est la coordonnée D de la loi stationnaire, solution de \(\boldsymbol{\pi}\mathbf{P} = \boldsymbol{\pi}\).",
                 "explication": r"Avec deux états absorbants, toute loi portée par D et R est stationnaire : il n'y a pas d'unicité, et la loi stationnaire ne dépend pas de l'état de départ, alors que \(h_S \ne h_F\)."},
            ],
        },
        "arret": {
            "question": r"Marche symétrique partie de 0, \(T\) le premier instant où \(Z_t = 1\). On a \(\mathbb{E}[Z_T] = 1 \ne 0 = Z_0\). Que dit-on du théorème d'arrêt ?",
            "options": [
                {"id": "a", "texte": r"\(T\) est un temps d'arrêt fini avec probabilité 1, mais aucune des trois hypothèses (a), (b), (c) n'est vérifiée.",
                 "explication": r"Oui : \(T\) n'est pas borné, \(Z_{t \wedge T}\) peut descendre aussi bas que l'on veut, et \(\mathbb{E}[T] = +\infty\). Le théorème ne s'applique pas, et la conclusion est fausse."},
                {"id": "b", "texte": r"\(T\) n'est pas un temps d'arrêt.",
                 "explication": r"C'en est un : savoir si \(Z\) a déjà touché 1 avant la date \(t\) ne demande que \(X_1, \dots, X_t\)."},
                {"id": "c", "texte": r"\(Z_t\) n'est pas une martingale quand \(p = 1/2\).",
                 "explication": r"C'en est une : \(\mathbb{E}[Z_t + X_{t+1} \mid X_1, \dots, X_t] = Z_t + \mathbb{E}[X_{t+1}] = Z_t\). Le problème vient du temps d'arrêt, pas de la martingale."},
                {"id": "d", "texte": r"\(T\) est infini avec une probabilité non nulle, d'où l'écart.",
                 "explication": r"\(T\) est fini avec probabilité 1 (outil 2 : la ruine est certaine pour la marche symétrique avec une seule barrière). C'est \(\mathbb{E}[T]\) qui est infini, pas \(T\)."},
            ],
        },
        "ar1": {
            "question": r"\(Y_t = -0{,}5\,Y_{t-1} + \varepsilon_t\) est stationnaire, avec des chocs indépendants, centrés, de variance 1, \(\varepsilon_t\) indépendant du passé. Que vaut l'autocorrélation au retard 3, \(\rho_3\) ?",
            "options": [
                {"id": "a", "texte": r"\(@Cc_rho3:3m@\)",
                 "explication": r"Oui : \(\rho_3 = \phi^3 = (-0{,}5)^3\). Le signe alterne avec le retard quand \(\phi < 0\)."},
                {"id": "b", "texte": r"\(@Cc_d_signe:3m@\)",
                 "explication": r"C'est \(\lvert \phi \rvert^3\) : le signe a été perdu. Avec \(\phi < 0\), une valeur haute est suivie en moyenne d'une valeur basse, et les retards impairs ont une autocorrélation négative."},
                {"id": "c", "texte": r"\(@Cc_d_produit:1m@\)",
                 "explication": r"C'est \(3\,\phi\) : la récurrence multiplie par \(\phi\) à chaque retard, elle n'additionne pas. Une autocorrélation est d'ailleurs comprise entre −1 et 1."},
                {"id": "d", "texte": r"\(@Cc_d_rho1:1m@\)",
                 "explication": r"C'est \(\rho_1 = \phi\), l'autocorrélation au retard 1. Au retard 3, on applique trois fois la récurrence \(\gamma_h = \phi\,\gamma_{h-1}\)."},
            ],
        },
        "ito": {
            "question": r"Pour un mouvement brownien standard \(W\), que vaut \(\int_0^t W_s\,dW_s\) ?",
            "options": [
                {"id": "a", "texte": r"\((W_t^2 - t)/2\)",
                 "explication": r"Oui : Itô avec \(f(x) = x^2\) donne \(d(W_t^2) = dt + 2W_t\,dW_t\). Contrôle : l'espérance vaut \((t - t)/2 = 0\), comme pour toute intégrale stochastique de ce type."},
                {"id": "b", "texte": r"\(W_t^2/2\)",
                 "explication": r"C'est le calcul ordinaire, qui oublie \(\tfrac12 f_{xx}\,dt = dt\). Son espérance serait \(t/2\), alors qu'une intégrale stochastique de ce type est d'espérance nulle."},
                {"id": "c", "texte": r"\(W_t^2 - t\)",
                 "explication": r"Le terme \(dt\) est bien là, mais le facteur 2 de \(f_x = 2x\) a été oublié : \(W_t^2 - t = 2\int_0^t W_s\,dW_s\)."},
                {"id": "d", "texte": r"0",
                 "explication": r"C'est son **espérance**, pas sa valeur : l'intégrale est une variable aléatoire, \((W_t^2 - t)/2\), qui varie d'une trajectoire à l'autre."},
            ],
        },
    },
    "verif": {"script": "verif/seance_math_4.py"},
}
