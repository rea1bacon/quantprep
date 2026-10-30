"""Leçon de probabilités de la séance de mathématiques 1 : quatre outils de calcul rapide.

Format : voir outils/lecons.py (élément de type "lecon", contenu en Markdown réduit, points de contrôle
[[controle:IDENT]] sur une ligne seule, option "a" correcte, une explication par option).
Aucun nombre calculé n'est écrit à la main : chaque jeton @CLE@, @CLE:3@ ou @CLE:3m@ est remplacé par la valeur
calculée par verif/seance_math_1.py (fichier verif/seance_math_1.out.json).
"""

LECON_PROBAS = {
    "id": "L-PROBAS-1", "type": "lecon", "section": "lecon",
    "theme": "Leçon probabilités : quatre outils de calcul rapide",
    "difficulte": 3, "temps_cible_s": 2400,
    # Aucun notebook ni exemple exécuté : la page affiche la leçon seule.
    "contenu": r"""## Leçon : quatre outils de calcul rapide en probabilités

Durée visée : 40 minutes. Quatre questions de probabilités du test de positionnement relèvent chacune d'un outil précis : les indicatrices (nombre de faces distinctes), le conditionnement sur le premier pas (ruine du joueur), les statistiques d'ordre (étendue de variables uniformes) et le passage au complémentaire (tests multiples).

La leçon a deux objectifs : savoir **refaire les preuves**, et surtout **reconnaître l'outil en 30 secondes**, puis finir le calcul en une minute. Chaque outil suit le même ordre : le problème, l'intuition, l'énoncé, la preuve, un exemple chiffré, les cas limites, les pièges. La dernière partie, « Réflexes en 30 secondes », est celle à relire juste avant un test ou un entretien.

Conventions de notation, valables pour toute la leçon : une majuscule désigne une variable aléatoire, une minuscule une valeur ou un paramètre fixe. \(\mathbb{P}\) désigne une probabilité, \(\mathbb{E}\) une espérance, \(\mathrm{Var}\) une variance. Pour un événement \(A\), \(A^c\) est son complémentaire (« \(A\) ne se produit pas ») et \(I_A\) son **indicatrice** : la variable aléatoire qui vaut 1 si \(A\) se produit et 0 sinon. Tous les nombres de la leçon sont calculés par le script `verif/seance_math_1.py`, qui les vérifie aussi par simulation.

## Outil 1 : linéarité de l'espérance et indicatrices

### Le problème

On demande l'espérance d'un **comptage** : nombre de points fixes d'une permutation, nombre d'occurrences d'un motif dans une suite de lancers, nombre de valeurs distinctes dans un échantillon. La loi du comptage est compliquée : il faudrait calculer la probabilité de chaque valeur possible. Son espérance, elle, se calcule en deux lignes.

### L'intuition

Un comptage est une somme de 0 et de 1, un terme par objet susceptible d'être compté. L'espérance d'un terme 0 ou 1 est la probabilité qu'il vaille 1. L'espérance d'une somme est la somme des espérances, et cela ne demande **aucune indépendance** entre les termes : il suffit de connaître la loi de chaque terme pris seul.

### L'énoncé

1. Pour tout événement \(A\) : \(\mathbb{E}[I_A] = \mathbb{P}(A)\).
2. Soient \(m \ge 1\) un entier et \(X_1, \dots, X_m\) des variables aléatoires réelles définies sur le même espace, d'espérances finies (\(\mathbb{E}[\lvert X_j \rvert] < \infty\) pour chaque \(j\)). Alors \(\mathbb{E}[X_1 + \dots + X_m] = \mathbb{E}[X_1] + \dots + \mathbb{E}[X_m]\), quelle que soit la dépendance entre les \(X_j\).
3. Conséquence : si \(N = I_{A_1} + \dots + I_{A_m}\) compte combien des événements \(A_1, \dots, A_m\) se produisent, alors \(\mathbb{E}[N] = \mathbb{P}(A_1) + \dots + \mathbb{P}(A_m)\).

### La preuve

Elle est donnée pour des variables discrètes, c'est-à-dire à valeurs dans un ensemble fini ou dénombrable, ce qui couvre tous les comptages. Pour des variables quelconques, l'énoncé est la linéarité de l'intégrale, admise ici.

1. \(I_A\) vaut 1 avec probabilité \(\mathbb{P}(A)\) et 0 avec probabilité \(1 - \mathbb{P}(A)\), donc \(\mathbb{E}[I_A] = 1 \times \mathbb{P}(A) + 0 \times (1 - \mathbb{P}(A)) = \mathbb{P}(A)\).
2. Deux variables \(X\) et \(Y\), de valeurs possibles notées \(x\) et \(y\). Par le théorème de transfert (l'espérance d'une fonction du couple se calcule avec la loi du couple) : \(\mathbb{E}[X + Y] = \sum_{x, y} (x + y)\,\mathbb{P}(X = x, Y = y)\).
3. Cette série converge absolument : \(\lvert x + y \rvert \le \lvert x \rvert + \lvert y \rvert\), et \(\sum_{x, y} \lvert x \rvert\,\mathbb{P}(X = x, Y = y) = \sum_x \lvert x \rvert\,\mathbb{P}(X = x) = \mathbb{E}[\lvert X \rvert] < \infty\), de même pour \(y\). On peut donc la couper en deux séries et sommer dans l'ordre voulu.
4. \(\sum_{x, y} x\,\mathbb{P}(X = x, Y = y) = \sum_x x \sum_y \mathbb{P}(X = x, Y = y) = \sum_x x\,\mathbb{P}(X = x) = \mathbb{E}[X]\), car les événements \(\{Y = y\}\) forment une partition (formule des probabilités totales). De même, le terme en \(y\) vaut \(\mathbb{E}[Y]\).
5. Pourquoi l'indépendance est inutile : à l'étape 4, la loi jointe du couple est sommée sur l'une des deux variables et disparaît. Seules les lois de \(X\) et de \(Y\) prises séparément restent.
6. Pour \(m\) variables, récurrence sur \(m\) : on applique le cas de deux variables à \(X_1 + \dots + X_{m-1}\) et \(X_m\). La première est d'espérance finie, car \(\lvert X_1 + \dots + X_{m-1} \rvert \le \lvert X_1 \rvert + \dots + \lvert X_{m-1} \rvert\), dont l'espérance est finie par l'hypothèse de récurrence appliquée aux \(\lvert X_j \rvert\).

### Exemple 1 : points fixes d'une permutation

Soit \(n \ge 2\) un entier et \(S\) une permutation de \(\{1, \dots, n\}\) tirée uniformément parmi les \(n!\) permutations. Un point fixe est un \(i\) tel que \(S(i) = i\). On note \(N\) le nombre de points fixes et \(I_i\) l'indicatrice de l'événement \(\{S(i) = i\}\), de sorte que \(N = I_1 + \dots + I_n\).

1. Les permutations qui fixent \(i\) sont les permutations des \(n - 1\) autres éléments : \(\mathbb{P}(S(i) = i) = (n-1)!/n! = 1/n\).
2. Par linéarité : \(\mathbb{E}[N] = n \times 1/n = 1\), pour tout \(n\).
3. Question suivante probable, la variance. On développe \(N^2 = \sum_i I_i^2 + \sum_{i \ne j} I_i I_j\), la seconde somme portant sur les \(n(n-1)\) couples **ordonnés** d'indices distincts. Comme \(I_i^2 = I_i\), la première somme a pour espérance 1.
4. \(I_i I_j\) est l'indicatrice de « \(i\) et \(j\) sont fixes », de probabilité \((n-2)!/n! = 1/(n(n-1))\). Par linéarité, la seconde somme a pour espérance \(n(n-1) \times 1/(n(n-1)) = 1\).
5. Donc \(\mathbb{E}[N^2] = 2\) et \(\mathrm{Var}(N) = 2 - 1^2 = 1\), pour tout \(n \ge 2\).

Contrôle par énumération des @La_pf_nb_perm@ permutations de \(\{1, 2, 3, 4\}\) : moyenne @La_pf_moy_enum@, second moment @La_pf_e2_enum@, variance @La_pf_var_enum@. Pour \(n = 4\), \(\mathbb{P}(I_1 = 1, I_2 = 1) = \tfrac{@La_pf_p12_num@}{@La_pf_p12_den@}\) alors que \(\mathbb{P}(I_1 = 1)\,\mathbb{P}(I_2 = 1) = \tfrac{@La_pf_p1_carre_num@}{@La_pf_p1_carre_den@}\) : les indicatrices sont dépendantes, et la linéarité s'applique quand même.

[[controle:indep]]

### Exemple 2 : un motif qui peut se chevaucher

On lance 10 fois une pièce équilibrée, lancers indépendants, chacun donnant P (pile) ou F (face) avec probabilité 1/2. Soit \(N\) le nombre de positions \(j\) telles que les lancers \(j\) et \(j+1\) donnent tous deux P ; les occurrences peuvent se chevaucher : PPP en contient @La_motif_ex@.

1. Il y a @La_motif_pos@ positions possibles, \(j = 1, \dots, @La_motif_pos@\). Soit \(I_j\) l'indicatrice de « lancers \(j\) et \(j+1\) égaux à P » : \(\mathbb{P}(I_j = 1) = \tfrac{@La_motif_p_num@}{@La_motif_p_den@}\) par indépendance des deux lancers.
2. Par linéarité : \(\mathbb{E}[N] = @La_motif_pos@ \times \tfrac{@La_motif_p_num@}{@La_motif_p_den@} = @La_motif_esp:2m@\).
3. Les \(I_j\) voisins sont dépendants : \(\mathbb{P}(I_1 = 1, I_2 = 1) = \mathbb{P}(\text{PPP au début}) = \tfrac{@La_motif_joint_num@}{@La_motif_joint_den@}\), différent de \(\tfrac{@La_motif_prod_num@}{@La_motif_prod_den@}\). Cela ne change rien à l'espérance.

Énumération des \(2^{10}\) suites et simulation : voir `verif/seance_math_1.py` (simulation : @La_motif_esp_mc:3@).

### Cas limites

- Somme infinie : pour des variables **positives**, en particulier des indicatrices, \(\mathbb{E}\big[\sum_{j \ge 1} X_j\big] = \sum_{j \ge 1} \mathbb{E}[X_j]\) reste vraie, les deux membres pouvant valoir \(+\infty\) (théorème de convergence monotone, admis). Cet énoncé sert à l'outil 2.
- Variables sans espérance finie : l'énoncé ne s'applique pas. Si \(X\) suit une loi de Cauchy, \(X + (-X) = 0\) a pour espérance 0, mais ni \(X\) ni \(-X\) n'a d'espérance.
- La linéarité donne l'espérance, pas la loi : \(\mathbb{P}(N = 0)\), la probabilité de n'avoir aucun point fixe, demande un autre outil (l'inclusion-exclusion).

### Pièges

1. La variance n'est pas additive sans hypothèse : \(\mathrm{Var}(I_1 + \dots + I_n) = \sum_i \mathrm{Var}(I_i)\) exige des termes non corrélés deux à deux. Pour les points fixes, \(\sum_i \mathrm{Var}(I_i) = n \times \tfrac1n (1 - \tfrac1n) = 1 - \tfrac1n\), alors que \(\mathrm{Var}(N) = 1\).
2. Une indicatrice par **objet compté**, pas par paire d'objets : écrire « nombre de valeurs distinctes = nombre de tirages moins nombre de paires égales » est faux dès que trois tirages coïncident (ils forment 3 paires pour 2 tirages en trop).
3. Le nombre de positions : un motif de longueur \(\ell\) dans une suite de longueur \(L\) a \(L - \ell + 1\) positions de départ.
4. Confondre \(\mathbb{E}[N]\) et \(\mathbb{P}(N \ge 1)\) : on a seulement \(\mathbb{P}(N \ge 1) \le \mathbb{E}[N]\) (borne de Boole, outil 4).

[[controle:urne]]

## Outil 2 : conditionnement sur le premier pas

### Le problème

Une marche aléatoire, ou la fortune d'un joueur, avance par pas de +1 ou −1 jusqu'à toucher l'une de deux barrières. Deux questions reviennent : la probabilité de toucher la barrière haute avant la basse, et l'espérance du nombre de pas avant l'arrêt. La loi du temps d'arrêt est compliquée ; ces deux nombres ne le sont pas.

### Le cadre

Il sert à toute cette partie. On fixe des entiers \(n \ge 2\) et \(i\) avec \(0 \le i \le n\), un réel \(p\) avec \(0 < p < 1\), et \(q = 1 - p\). Les pas \(X_1, X_2, \dots\) sont des variables aléatoires indépendantes, chacune valant +1 avec probabilité \(p\) et −1 avec probabilité \(q\). La position après \(t\) pas est \(Z_t = i + X_1 + \dots + X_t\), avec \(Z_0 = i\). Le temps d'arrêt \(T\) est le plus petit \(t \ge 0\) tel que \(Z_t = 0\) ou \(Z_t = n\) (\(T = +\infty\) si cela n'arrive jamais).

On note \(h(i)\) la probabilité que la marche partie de \(i\) s'arrête en \(n\), et \(d(i)\) l'espérance de \(T\) pour la marche partie de \(i\). Ce sont des fonctions déterministes de la position de départ, d'où les minuscules.

Barrières quelconques \(a < b\), départ \(i\) entre les deux : on se ramène au cadre en remplaçant \(i\) par \(i - a\) et \(n\) par \(b - a\).

### L'intuition

Après le premier pas, la marche repart depuis \(i + 1\) ou \(i - 1\) avec exactement les mêmes règles, sans mémoire du pas effectué. Le problème posé depuis \(i\) se ramène donc au même problème posé depuis les deux voisins : une équation par position, et des valeurs connues sur les barrières.

### L'énoncé

1. \(d(i) \le n/p^n\) ; en particulier \(T\) est fini avec probabilité 1 et son espérance est finie.
2. Pour \(0 < i < n\) : \(h(i) = p\,h(i+1) + q\,h(i-1)\), avec \(h(0) = 0\) et \(h(n) = 1\).
3. Pour \(0 < i < n\) : \(d(i) = 1 + p\,d(i+1) + q\,d(i-1)\), avec \(d(0) = d(n) = 0\).
4. Chacun de ces deux systèmes a une **unique** solution. Donc toute suite qui vérifie l'équation et les deux conditions aux bords est la réponse : on peut la deviner, la vérifier, et conclure.
5. Cas symétrique \(p = 1/2\) : \(h(i) = i/n\) et \(d(i) = i\,(n - i)\).
6. Cas \(p \ne 1/2\), avec \(r = q/p\) : \(h(i) = \dfrac{1 - r^i}{1 - r^n}\) et \(d(i) = \dfrac{n\,h(i) - i}{p - q}\).

### La preuve

1. **Finitude.** On découpe les pas en blocs consécutifs de \(n\) pas. Si la marche est encore strictement entre les barrières au début d'un bloc et que les \(n\) pas du bloc valent tous +1, elle atteint \(n\) pendant ce bloc, car il lui faut au plus \(n - 1\) pas montants. Chaque bloc est entièrement montant avec probabilité \(p^n\), indépendamment des autres blocs, puisque les blocs portent sur des pas disjoints. Rester entre les barrières pendant les \(k\) premiers blocs exige donc qu'aucun d'eux ne soit entièrement montant : \(\mathbb{P}(T > kn) \le (1 - p^n)^k\).
2. Pour une variable à valeurs entières positives, \(T = \sum_{t \ge 0} I_{\{T > t\}}\) (il y a exactement \(T\) entiers \(t\) avec \(0 \le t < T\)), donc \(\mathbb{E}[T] = \sum_{t \ge 0} \mathbb{P}(T > t)\) par l'outil 1 (somme infinie de variables positives). Pour \(kn \le t < (k+1)n\), \(\mathbb{P}(T > t) \le \mathbb{P}(T > kn)\). En regroupant les \(t\) par blocs de \(n\) : \(\mathbb{E}[T] \le n \sum_{k \ge 0} (1 - p^n)^k = n/p^n\).
3. **Équation de \(h\).** Formule des probabilités totales selon la valeur de \(X_1\) : \(h(i) = p\,\mathbb{P}(\text{arrêt en } n \mid X_1 = +1) + q\,\mathbb{P}(\text{arrêt en } n \mid X_1 = -1)\). Comme \(0 < i < n\), la marche ne s'est pas arrêtée à l'instant 0. Sachant \(X_1 = +1\), la suite \(Z_1, Z_2, \dots\) s'écrit \(i + 1 + X_2 + \dots\) ; les pas \(X_2, X_3, \dots\) sont indépendants de \(X_1\) et ont la même loi que \(X_1, X_2, \dots\) : c'est une marche de même règle partie de \(i + 1\), et la probabilité conditionnelle vaut \(h(i+1)\). De même \(h(i-1)\) pour un premier pas −1. Aux bords, la marche est arrêtée dès l'instant 0 : \(h(0) = 0\) et \(h(n) = 1\).
4. **Équation de \(d\).** Même découpage : sachant \(X_1 = +1\), \(T = 1 + T'\), où \(T'\) est le temps d'arrêt de la marche partie de \(i + 1\) ; donc \(\mathbb{E}[T \mid X_1 = +1] = 1 + d(i+1)\), et de même avec −1. La moyenne pondérée donne \(d(i) = 1 + p\,d(i+1) + q\,d(i-1)\). Toutes ces valeurs sont finies par l'étape 2, ce qui autorise les soustractions de l'étape suivante.
5. **Unicité** (principe du maximum). Si \(u\) et \(v\) sont deux solutions du même système, leur différence \(w = u - v\) vérifie \(w(i) = p\,w(i+1) + q\,w(i-1)\) pour \(0 < i < n\) (le terme 1 disparaît par différence) et \(w(0) = w(n) = 0\). Soit \(\mu\) le maximum de \(w\) sur \(\{0, \dots, n\}\), atteint en une position \(j\). Si \(0 < j < n\), \(w(j)\) est une moyenne à poids strictement positifs \(p\) et \(q\) de deux nombres au plus égaux à \(\mu\) ; l'égalité \(w(j) = \mu\) force \(w(j+1) = w(j-1) = \mu\). De proche en proche, le maximum est atteint en 0, donc \(\mu = w(0) = 0\). Le même argument appliqué à \(-w\) montre que le minimum vaut 0. Donc \(w = 0\) : \(u = v\).
6. **Cas symétrique.** Avec \(p = q = 1/2\), l'équation de \(h\) s'écrit \(h(i+1) - h(i) = h(i) - h(i-1)\) : accroissements constants, \(h\) est affine, et les bords donnent \(h(i) = i/n\). Pour \(d\), on vérifie que \(i\,(n-i)\) convient : elle s'annule en 0 et en \(n\), et \(\tfrac12 (i+1)(n-i-1) + \tfrac12 (i-1)(n-i+1) = i\,(n-i) - 1\), donc \(1 + \tfrac12 d(i+1) + \tfrac12 d(i-1) = i\,(n-i)\). Par unicité, c'est \(d\).
7. **Cas \(p \ne 1/2\), probabilité.** La suite constante 1 vérifie l'équation de \(h\), et la suite \(r^i\) aussi : \(p\,r^{i+1} + q\,r^{i-1} = r^{i-1}(p\,r^2 + q)\), et \(p\,r^2 + q = q^2/p + q = q\,(q + p)/p = r\). Toute combinaison \(c_1 + c_2 r^i\) la vérifie donc, et \((1 - r^i)/(1 - r^n)\) vaut 0 en 0 et 1 en \(n\) (le dénominateur est non nul car \(r \ne 1\)). Par unicité, c'est \(h\).
8. **Cas \(p \ne 1/2\), durée.** La suite \(i/(q - p)\) vérifie l'équation de \(d\) : \(1 + p\,\tfrac{i+1}{q-p} + q\,\tfrac{i-1}{q-p} = 1 + \tfrac{i + p - q}{q - p} = \tfrac{i}{q-p}\). La suite \(h\) vérifie l'équation sans le terme 1 (étape 7). Par linéarité, \(\dfrac{i - n\,h(i)}{q - p}\) vérifie donc l'équation de \(d\) ; elle vaut 0 en 0 (car \(h(0) = 0\)) et 0 en \(n\) (car \(h(n) = 1\)). Par unicité c'est \(d\) : \(d(i) = (n\,h(i) - i)/(p - q)\).

### Seconde méthode pour la durée : l'identité de Wald

C'est la route la plus rapide quand \(p \ne 1/2\) : calculer \(h(i)\), puis en déduire \(d(i)\).

1. La position finale s'écrit \(Z_T = i + \sum_{k \ge 1} X_k\,I_{\{T \ge k\}}\) : on additionne les pas effectués avant l'arrêt.
2. L'événement \(\{T \ge k\}\) est le contraire de \(\{T \le k - 1\}\), qui ne dépend que de \(X_1, \dots, X_{k-1}\). Il est donc indépendant de \(X_k\), et \(\mathbb{E}[X_k\,I_{\{T \ge k\}}] = \mathbb{E}[X_k]\,\mathbb{P}(T \ge k) = (p - q)\,\mathbb{P}(T \ge k)\).
3. On somme sur \(k\) ; l'échange de la somme et de l'espérance est permis car \(\sum_k \mathbb{E}\big[\lvert X_k \rvert\,I_{\{T \ge k\}}\big] = \sum_{k \ge 1} \mathbb{P}(T \ge k) = \mathbb{E}[T] < \infty\). D'où \(\mathbb{E}[Z_T] - i = (p - q)\,\mathbb{E}[T]\).
4. \(Z_T\) vaut \(n\) avec probabilité \(h(i)\) et 0 sinon : \(\mathbb{E}[Z_T] = n\,h(i)\), d'où \(d(i) = (n\,h(i) - i)/(p - q)\), la même formule.
5. Si \(p = 1/2\), l'identité donne \(\mathbb{E}[Z_T] = i\), donc \(h(i) = i/n\), mais elle ne dit rien de \(\mathbb{E}[T]\) (elle se réduit à \(0 = 0\)).

### Exemple chiffré

Départ \(i = 2\), barrières 0 et \(n = 5\).

- Cas symétrique : \(h(2) = 2/5 = @Lb_sym_h:1m@\) et \(d(2) = 2 \times 3 = @Lb_sym_d@\).
- Cas \(p = 2/3\), \(q = 1/3\), donc \(r = 1/2\), \(r^2 = 1/@Lb_r2_inv@\) et \(r^5 = 1/@Lb_r5_inv@\). Probabilité : \(h(2) = \dfrac{@Lb_un_moins_r2_num@/@Lb_un_moins_r2_den@}{@Lb_un_moins_r5_num@/@Lb_un_moins_r5_den@} = \tfrac{@Lb_h_num@}{@Lb_h_den@} \approx @Lb_h:3m@\), plus grand que \(2/5\) puisque la marche dérive vers le haut. Durée : \(n\,h(2) - 2 = \tfrac{@Lb_5h_num@}{@Lb_5h_den@} - 2 = \tfrac{@Lb_5h_moins_2_num@}{@Lb_5h_moins_2_den@}\) et \(p - q = \tfrac{@Lb_p_moins_q_num@}{@Lb_p_moins_q_den@}\), donc \(d(2) = \tfrac{@Lb_d_num@}{@Lb_d_den@} \approx @Lb_d:3m@\).

Simulation (script de vérification) : \(h(2) \approx @Lb_h_mc:3m@\), \(d(2) \approx @Lb_d_mc:3m@\).

### Au-delà des marches : attendre un motif

La méthode s'applique à toute chaîne dont on décrit l'état. Exemple : on lance une pièce équilibrée jusqu'à obtenir P suivi immédiatement de F ; combien de lancers en moyenne ? États : « rien d'utile » (départ, ou dernier lancer F) et « dernier lancer P ». On note \(e_0\) et \(e_1\) les espérances du nombre de lancers restants depuis chaque état.

1. Depuis « rien d'utile », un lancer, puis l'état « dernier lancer P » ou le même état : \(e_0 = 1 + \tfrac12 e_1 + \tfrac12 e_0\).
2. Depuis « dernier lancer P », un lancer : F termine, P laisse dans le même état : \(e_1 = 1 + \tfrac12 \times 0 + \tfrac12 e_1\).
3. Les espérances sont finies : en découpant les lancers en blocs disjoints de deux, chaque bloc vaut PF avec probabilité \(1/4\), d'où un majorant de \(2 \times 4 = @Lb_pf_borne@\) lancers par l'argument de l'étape 1 de la preuve.
4. Le système \(2 \times 2\) a une unique solution : \(e_1 = @Lb_pf_e1@\), \(e_0 = @Lb_pf_e0@\).

### Cas limites

- \(i = 0\) ou \(i = n\) : \(h\) vaut 0 ou 1, et \(d = 0\).
- \(p \to 1/2\) : \(r \to 1\) et \(\dfrac{1 - r^i}{1 - r^n} = \dfrac{1 + r + \dots + r^{i-1}}{1 + r + \dots + r^{n-1}} \to \dfrac{i}{n}\) : les deux formules se raccordent.
- Barrière haute infinie. L'événement « atteindre 0 » est la réunion croissante, en \(n\), des événements « atteindre 0 avant \(n\) » : une trajectoire qui atteint 0 en un temps fini reste sous une hauteur finie avant. Par continuité croissante de la probabilité, la probabilité de ruine vaut \(\lim_{n \to \infty} (1 - h(i))\) : 1 si \(p \le 1/2\), et \(r^i\) si \(p > 1/2\). Exemple : \(p = 2/3\), départ 2 : ruine avec probabilité \(r^2 = \tfrac{@Lb_ruine_inf_num@}{@Lb_ruine_inf_den@}\).
- Une seule barrière en 0 et \(p < 1/2\) : la durée moyenne jusqu'à la ruine vaut \(i/(q - p)\). Notons \(T_n\) le temps d'arrêt avec la barrière haute \(n\). Pour sortir de \(]0, n+1[\), il faut d'abord sortir de \(]0, n[\), donc \(T_n\) croît avec \(n\) ; et sur une trajectoire ruinée, \(T_n\) est égal au temps de ruine dès que \(n\) dépasse la hauteur maximale atteinte avant la ruine. Comme \(r > 1\), \(n\,h(i) \le n\,r^i/(r^n - 1) \to 0\), donc \(\mathbb{E}[T_n] = (n\,h(i) - i)/(p - q) \to i/(q - p)\), et le théorème de convergence monotone (admis) donne la durée moyenne jusqu'à la ruine. En particulier, ajouter une barrière haute ne peut que raccourcir la partie : \(d(i) \le i/(q - p)\).
- Cas symétrique avec une seule barrière en 0 : la ruine est certaine, mais son temps moyen est infini. Pour tout \(n > i\), atteindre 0 exige de quitter d'abord l'intervalle \(]0, n[\), donc le temps d'atteinte de 0 est au moins \(T\), d'espérance \(i\,(n - i)\), qui tend vers l'infini avec \(n\).

### Pièges

1. Échanger \(p\) et \(q\) dans \(r\) : \(r = q/p\) est le rapport « probabilité de descendre sur probabilité de monter » quand \(h\) est la probabilité de finir **en haut**. Contrôle en 5 secondes : si \(p > 1/2\), \(h(i)\) doit dépasser \(i/n\).
2. Appliquer \(i/n\) ou \(i\,(n - i)\) quand \(p \ne 1/2\).
3. Oublier le « 1 + » de l'équation de durée : sans lui, la solution est \(d = 0\).
4. Multiplier le départ par la barrière haute, \(i \times n\), au lieu des deux distances aux barrières, \(i \times (n - i)\).
5. Utiliser \(i/(q - p)\), la durée moyenne vers une seule barrière, quand il y en a deux.

[[controle:unicite]]

[[controle:derive]]

## Outil 3 : statistiques d'ordre de lois uniformes

### Le problème

On tire \(n\) points uniformément et indépendamment dans \([0, 1]\). Où tombe le plus grand ? le plus petit ? le \(k\)-ième ? Quelle est la longueur moyenne entre deux points voisins ? Ces questions reviennent sous des habillages variés : meilleure de \(n\) offres, étendue d'un échantillon, écart entre deux points.

### Notations

\(U_1, \dots, U_n\) sont indépendantes, de loi uniforme sur \([0, 1]\), avec \(n \ge 1\) fixé. Avec probabilité 1 elles sont deux à deux distinctes : pour \(i \ne j\), sachant \(U_j\), la probabilité que \(U_i\) tombe exactement sur cette valeur est nulle. On les range : \(U_{(1)} < U_{(2)} < \dots < U_{(n)}\). \(U_{(k)}\) est le \(k\)-ième plus petit, \(U_{(1)}\) le minimum, \(U_{(n)}\) le maximum. Ce sont des variables aléatoires, **non indépendantes** entre elles.

### L'intuition

Les \(n\) points découpent \([0, 1]\) en \(n + 1\) morceaux, et aucun morceau n'est favorisé : chacun mesure \(1/(n+1)\) en moyenne. Le \(k\)-ième point se trouve au bout des \(k\) premiers morceaux, donc en \(k/(n+1)\) en moyenne. Ce raisonnement n'est qu'une intuition (pourquoi aucun morceau ne serait-il favorisé ?) ; la preuve ci-dessous l'établit.

### L'énoncé

1. Pour \(x \in [0, 1]\) : \(\mathbb{P}(U_{(n)} \le x) = x^n\) et \(\mathbb{P}(U_{(1)} > x) = (1 - x)^n\).
2. \(\mathbb{E}[U_{(n)}] = n/(n+1)\) et \(\mathbb{E}[U_{(1)}] = 1/(n+1)\).
3. Pour \(1 \le k \le n\) : \(\mathbb{E}[U_{(k)}] = k/(n+1)\).
4. Avec les conventions \(U_{(0)} = 0\) et \(U_{(n+1)} = 1\), chacun des \(n + 1\) écarts \(U_{(k)} - U_{(k-1)}\) a pour espérance \(1/(n+1)\). L'étendue \(U_{(n)} - U_{(1)}\) a pour espérance \((n-1)/(n+1)\).

### Deux résultats utilisés par la preuve

**(F) Formule de la queue.** Pour une variable \(Y\) positive : \(\mathbb{E}[Y] = \int_0^\infty \mathbb{P}(Y > x)\,dx\) ; si \(Y\) est à valeurs dans \([0, 1]\), l'intégrande est nulle au-delà de 1 et \(\mathbb{E}[Y] = \int_0^1 \mathbb{P}(Y > x)\,dx\). Preuve : pour chaque issue, \(Y = \int_0^\infty I_{\{x < Y\}}\,dx\), car l'intégrande vaut 1 pour \(x < Y\) et 0 ensuite. On prend l'espérance et on échange espérance et intégrale, ce qui est permis pour une fonction positive (théorème de Fubini-Tonelli, admis).

**(B) Intégrale bêta.** Pour des entiers \(a, b \ge 0\) : \(\int_0^1 x^a (1 - x)^b\,dx = \dfrac{a!\,b!}{(a + b + 1)!}\). Preuve probabiliste : soient \(V_0, V_1, \dots, V_{a+b}\) indépendantes de loi uniforme sur \([0, 1]\), et \(D\) l'événement « \(V_1, \dots, V_a\) sont sous \(V_0\), et \(V_{a+1}, \dots, V_{a+b}\) au-dessus ».

1. Sachant \(V_0 = x\), les autres variables, indépendantes de \(V_0\), sont chacune sous \(x\) avec probabilité \(x\) : \(\mathbb{P}(D \mid V_0 = x) = x^a (1 - x)^b\). En intégrant contre la densité 1 de \(V_0\) : \(\mathbb{P}(D) = \int_0^1 x^a (1 - x)^b\,dx\).
2. Les \(a + b + 1\) variables sont distinctes avec probabilité 1 et jouent des rôles symétriques (indépendantes et de même loi) : leurs \((a + b + 1)!\) ordres relatifs sont équiprobables. \(D\) regroupe les ordres où \(V_0\) est au rang \(a + 1\), les \(a\) premières variables étant rangées entre elles de l'une des \(a!\) façons, et les \(b\) dernières de l'une des \(b!\) façons : \(\mathbb{P}(D) = a!\,b!/(a + b + 1)!\).

### La preuve

1. \(U_{(n)} \le x\) si et seulement si tous les \(U_j\) sont au plus \(x\) ; par indépendance, cela a pour probabilité \(x^n\). De même, \(U_{(1)} > x\) si et seulement si tous dépassent \(x\) : probabilité \((1 - x)^n\).
2. Par (F) : \(\mathbb{E}[U_{(n)}] = \int_0^1 (1 - x^n)\,dx = 1 - \tfrac{1}{n+1}\) et \(\mathbb{E}[U_{(1)}] = \int_0^1 (1 - x)^n\,dx = \tfrac{1}{n+1}\).
3. Cas général. \(U_{(k)} > x\) si et seulement si au plus \(k - 1\) des \(U_j\) sont au plus \(x\). Le nombre \(K_x\) de \(U_j\) au plus égaux à \(x\) suit la loi binomiale de paramètres \(n\) et \(x\) (\(n\) essais indépendants, chacun réussi avec probabilité \(x\)). Donc \(\mathbb{P}(U_{(k)} > x) = \sum_{j=0}^{k-1} \binom{n}{j} x^j (1 - x)^{n-j}\).
4. Par (F), puis (B) avec \(a = j\) et \(b = n - j\) : \(\mathbb{E}[U_{(k)}] = \sum_{j=0}^{k-1} \binom{n}{j} \dfrac{j!\,(n-j)!}{(n+1)!} = \sum_{j=0}^{k-1} \dfrac{n!}{(n+1)!} = \dfrac{k}{n+1}\).
5. Écarts, par linéarité (outil 1) : \(\mathbb{E}[U_{(k)} - U_{(k-1)}] = \tfrac{k}{n+1} - \tfrac{k-1}{n+1} = \tfrac{1}{n+1}\), y compris pour \(k = 1\) et \(k = n + 1\) avec les conventions de l'énoncé. L'intuition est ainsi justifiée après coup.

### Exemple chiffré

\(n = 5\) points uniformes sur \([0, 1]\).

- Deuxième plus petit : \(\mathbb{E}[U_{(2)}] = 2/6 = \tfrac{@Lc_k2_num@}{@Lc_k2_den@}\).
- Maximum et minimum : \(\tfrac{@Lc_max_num@}{@Lc_max_den@}\) et \(\tfrac{@Lc_min_num@}{@Lc_min_den@}\) ; étendue moyenne \(4/6 = \tfrac{@Lc_etendue_num@}{@Lc_etendue_den@}\) ; écart moyen entre deux points voisins \(\tfrac{@Lc_ecart_num@}{@Lc_ecart_den@}\).
- \(\mathbb{P}(U_{(5)} \le 0{,}9) = 0{,}9^5 = @Lc_p_max:5m@\).
- Deux points (\(n = 2\)) : l'écart moyen \(\mathbb{E}[\lvert U_1 - U_2 \rvert] = \mathbb{E}[U_{(2)} - U_{(1)}] = \tfrac{@Lc_deux_num@}{@Lc_deux_den@}\).

### Cas limites

- \(n = 1\) : \(\mathbb{E}[U_{(1)}] = 1/2\), l'espérance d'une uniforme.
- Uniforme sur \([a, b]\) avec \(a < b\) : \(a + (b - a)\,U\) est uniforme sur \([a, b]\), et cette transformation croissante conserve l'ordre. Donc le \(k\)-ième plus petit a pour espérance \(a + (b - a)\,k/(n+1)\), et l'étendue \((b - a)(n-1)/(n+1)\).
- Le \(k\)-ième plus **grand** est \(U_{(n-k+1)}\), d'espérance \((n - k + 1)/(n + 1)\).
- Autres lois : la formule est propre à l'uniforme. Soient \(G_1, G_2\) indépendantes de loi exponentielle de paramètre 1 (fonction de répartition \(1 - e^{-x}\) pour \(x \ge 0\), espérance 1). Par indépendance, \(\mathbb{P}(\min(G_1, G_2) > x) = e^{-x} e^{-x} = e^{-2x}\), donc \(\mathbb{E}[\min(G_1, G_2)] = \int_0^\infty e^{-2x}\,dx = @Lc_exp_min:1m@\) (formule de la queue sur \([0, \infty[\)). Comme \(\max + \min = G_1 + G_2\), \(\mathbb{E}[\max(G_1, G_2)] = 2 - @Lc_exp_min:1m@ = @Lc_exp_max:1m@\). Transposer « \(k/(n+1)\) » en prenant le quantile d'ordre \(2/3\), \(\ln 3 \approx @Lc_exp_quantile:3m@\), donne une valeur fausse.

### Pièges

1. Dénominateur \(n\) au lieu de \(n + 1\).
2. Confondre \(\mathbb{P}(\max \le x) = x^n\) et \(\mathbb{P}(\min \le x) = 1 - (1 - x)^n\).
3. Traiter le maximum et le minimum comme indépendants : pour \(0 \le a < b \le 1\), \(\mathbb{P}(U_{(1)} > a \text{ et } U_{(n)} \le b) = (b - a)^n\) (tous les points dans \(]a, b]\)), et non le produit \(\mathbb{P}(U_{(1)} > a)\,\mathbb{P}(U_{(n)} \le b)\).
4. Oublier la mise à l'échelle quand l'intervalle n'est pas \([0, 1]\).

[[controle:ordre]]

## Outil 4 : passer au complémentaire, borne de Boole, Bonferroni

### Le problème

« Probabilité qu'au moins un des \(m\) événements se produise » : au moins un test qui rejette à tort, au moins un signal faussement significatif, au moins une panne. La réunion se calcule mal directement : l'inclusion-exclusion compte \(2^m - 1\) termes. Son complémentaire, « aucun ne se produit », est une intersection, qui se calcule bien sous indépendance.

### Notations

\(m \ge 1\) est un entier, \(A_1, \dots, A_m\) des événements, \(p_j = \mathbb{P}(A_j)\), et \(\Omega\) l'ensemble des issues. Les événements sont **mutuellement indépendants** si, pour toute sous-famille d'indices \(\mathcal{J} \subset \{1, \dots, m\}\), \(\mathbb{P}\big(\bigcap_{j \in \mathcal{J}} A_j\big) = \prod_{j \in \mathcal{J}} p_j\).

### L'énoncé

1. Toujours : \(\mathbb{P}(A_1 \cup \dots \cup A_m) = 1 - \mathbb{P}(A_1^c \cap \dots \cap A_m^c)\).
2. Sous indépendance mutuelle : \(\mathbb{P}(A_1 \cup \dots \cup A_m) = 1 - (1 - p_1) \cdots (1 - p_m)\), et \(1 - (1 - p)^m\) si tous les \(p_j\) valent \(p\).
3. Borne de Boole, sans aucune hypothèse de dépendance : \(\mathbb{P}(A_1 \cup \dots \cup A_m) \le p_1 + \dots + p_m\), avec égalité si et seulement si, avec probabilité 1, deux des \(A_j\) ne se produisent jamais ensemble.
4. Correction de Bonferroni : si chacun des \(m\) tests est réalisé au niveau \(\alpha/m\), la probabilité de rejeter à tort au moins une hypothèse nulle vraie est au plus \(\alpha\), quelle que soit la dépendance entre les tests.
5. Approximation : pour \(0 \le p \le 1/2\), \(1 - e^{-mp} \le 1 - (1 - p)^m \le 1 - e^{-mp} + m\,p^2 e^{-mp}\).

### La preuve

1. **Complémentaire.** Une issue n'est dans aucun \(A_j\) si et seulement si elle est dans chaque \(A_j^c\) : \((A_1 \cup \dots \cup A_m)^c = A_1^c \cap \dots \cap A_m^c\) (loi de De Morgan). On conclut avec \(\mathbb{P}(B^c) = 1 - \mathbb{P}(B)\) pour tout événement \(B\).
2. **Les complémentaires restent indépendants.** On remplace \(A_1\) par \(A_1^c\). Les sous-familles sans l'indice 1 ne changent pas. Pour une sous-famille \(\mathcal{J}\) contenant 1, soit \(C\) l'intersection des \(A_j\) pour \(j \in \mathcal{J}\) autre que 1 (\(C = \Omega\) si \(\mathcal{J} = \{1\}\)). Comme \(A_1^c \cap C\) est \(C\) privé de \(A_1 \cap C\) : \(\mathbb{P}(A_1^c \cap C) = \mathbb{P}(C) - \mathbb{P}(A_1 \cap C) = \mathbb{P}(C) - p_1\,\mathbb{P}(C) = (1 - p_1)\,\mathbb{P}(C)\), où \(\mathbb{P}(A_1 \cap C) = p_1\,\mathbb{P}(C)\) et \(\mathbb{P}(C)\) se factorisent par l'indépendance de la famille. On remplace ensuite \(A_2\), et ainsi de suite, un indice à la fois. Donc \(\mathbb{P}(A_1^c \cap \dots \cap A_m^c) = (1 - p_1) \cdots (1 - p_m)\), et l'énoncé 2 découle de l'énoncé 1.
3. **Boole.** Pour chaque issue, \(I_{A_1 \cup \dots \cup A_m} \le I_{A_1} + \dots + I_{A_m}\) : si l'issue n'est dans aucun \(A_j\), les deux membres sont nuls ; sinon le membre de gauche vaut 1 et celui de droite au moins 1. L'espérance conserve l'inégalité, et l'outil 1 (linéarité, \(\mathbb{E}[I_A] = \mathbb{P}(A)\)) donne la borne. Cas d'égalité : la différence \(W\) des deux membres est une variable à valeurs entières positives, et \(W \ge 1\) exactement quand au moins deux \(A_j\) se produisent ; comme \(\mathbb{E}[W] \ge \mathbb{P}(W \ge 1)\), l'égalité \(\mathbb{E}[W] = 0\) équivaut à \(\mathbb{P}(W \ge 1) = 0\).
4. **Bonferroni.** Vocabulaire : chaque test \(j\) porte sur une hypothèse nulle ; le rejeter alors qu'elle est vraie est un **rejet à tort** (erreur de première espèce) ; un test au niveau \(t\) rejette à tort avec probabilité au plus \(t\) quand son hypothèse nulle est vraie. Le **taux d'erreur familial** (FWER, family-wise error rate) est la probabilité d'au moins un rejet à tort. Soit \(\mathcal{J}_0\) l'ensemble des tests dont l'hypothèse nulle est vraie, et \(m_0 \le m\) leur nombre. Par Boole : FWER \(\le \sum_{j \in \mathcal{J}_0} \mathbb{P}(\text{le test } j \text{ rejette}) \le m_0\,\alpha/m \le \alpha\). Boole ne demande aucune indépendance : la garantie vaut pour toute dépendance.
5. **Approximation, minoration.** Pour tout réel \(y\), \(e^{-y} \ge 1 - y\) : l'exponentielle est convexe, donc au-dessus de sa tangente en 0. Avec \(y = p\) et en élevant à la puissance \(m\) (les deux membres sont positifs car \(p \le 1\)) : \((1 - p)^m \le e^{-mp}\), d'où \(1 - (1 - p)^m \ge 1 - e^{-mp}\).
6. **Approximation, majoration.** La fonction \(f(p) = \ln(1 - p) + p + p^2\) vaut 0 en 0 et a pour dérivée \(f'(p) = -\tfrac{1}{1-p} + 1 + 2p = \tfrac{p\,(1 - 2p)}{1 - p} \ge 0\) sur \([0, 1/2]\), donc \(\ln(1 - p) \ge -p - p^2\) sur cet intervalle. Ainsi \((1 - p)^m \ge e^{-mp} e^{-mp^2} \ge e^{-mp}(1 - m p^2)\), en utilisant de nouveau \(e^{-y} \ge 1 - y\). D'où \(1 - (1 - p)^m \le 1 - e^{-mp} + m\,p^2 e^{-mp}\).

Lecture, pour \(m\) événements indépendants de même probabilité \(p\) : \(1 - e^{-mp}\) sous-estime la vraie valeur \(1 - (1 - p)^m\) d'au plus \(m p^2 e^{-mp} \le m p^2\). L'approximation est bonne quand \(m p^2\) est petit, c'est-à-dire \(p\) petit devant \(1/\sqrt{m}\). La borne de Boole \(mp\) n'est proche de la vraie valeur que si \(mp\) lui-même est petit : la fonction \(g(y) = e^{-y} - 1 + y - y^2/2\) vaut 0 en 0 et a pour dérivée \(-e^{-y} + 1 - y \le 0\), donc \(1 - e^{-y} \ge y - y^2/2\) pour \(y \ge 0\), et Boole surestime la vraie valeur d'au plus \((mp)^2/2\).

### Exemple chiffré

On teste 50 signaux sans effet réel, indépendamment, chacun au niveau 5 % : \(m = 50\), \(p = 0{,}05\).

- Exact : \(1 - @Ld_q:2m@^{50} \approx 1 - @Ld_aucun:4m@ = @Ld_exact:4m@\).
- Approximation : \(mp = @Ld_mp:1m@\) et \(1 - e^{-2{,}5} \approx @Ld_poisson:4m@\). L'écart réel, \(@Ld_ecart:4m@\), est bien sous la borne \(m p^2 e^{-mp} \approx @Ld_borne:4m@\).
- Boole : \(mp = @Ld_mp:1m@\), plus grand que 1, n'apprend rien.
- Bonferroni : chaque test au niveau \(0{,}05/50 = @Ld_seuil_bonf:3m@\). Sous indépendance, le taux d'erreur familial vaut alors exactement \(1 - (1 - @Ld_seuil_bonf:3m@)^{50} \approx @Ld_fwer_bonf:4m@\), sous 0,05 comme garanti.

### Cas limites

- Événements identiques (\(A_1 = \dots = A_m\), chacun de probabilité \(p\)) : la réunion a pour probabilité \(p\). Événements disjoints (possible si \(mp \le 1\), par exemple \(A_j = \{(j-1)p \le U < jp\}\) avec \(U\) uniforme sur \([0, 1]\)) : la réunion a pour probabilité \(mp\), et Boole est une égalité. Sans hypothèse de dépendance, on sait donc seulement que la probabilité de la réunion est entre \(p\) et \(\min(1, mp)\), et la borne supérieure est atteinte quand \(mp \le 1\).
- La même construction disjointe montre que, pour une dépendance quelconque, aucun niveau commun \(t > \alpha/m\) ne garantit un taux d'erreur familial au plus \(\alpha\) : avec des rejets à tort disjoints, il vaut \(mt > \alpha\) (si \(mt \le 1\)). Bonferroni est donc le meilleur niveau commun garanti pour une dépendance quelconque. Le niveau de Šidák, \(t_S = 1 - (1 - \alpha)^{1/m}\), donne un taux exactement égal à \(\alpha\) sous indépendance, mais il est strictement supérieur à \(\alpha/m\) pour \(m \ge 2\) et \(0 < \alpha < 1\), et n'est donc pas garanti sans elle. Preuve : l'inégalité de Bernoulli \((1 - x)^m \ge 1 - mx\) pour \(0 \le x \le 1\), stricte si \(m \ge 2\) et \(x > 0\), s'obtient par récurrence, car \((1 - x)^{m+1} \ge (1 - mx)(1 - x) = 1 - (m+1)x + m x^2\) ; avec \(x = \alpha/m\), elle donne \((1 - \alpha/m)^m > 1 - \alpha\), donc \(1 - \alpha/m > (1 - \alpha)^{1/m}\), c'est-à-dire \(t_S > \alpha/m\).
- Coût de Bonferroni : quand les tests sont très corrélés (à l'extrême, identiques), le taux réel tombe à \(\alpha/m\), bien sous \(\alpha\), et l'on perd de la puissance.

### Pièges

1. Utiliser \(1 - (1 - p)^m\) sans indépendance.
2. Prendre \(mp\) pour une probabilité : c'est l'espérance du nombre de rejets (outil 1), qui peut dépasser 1, et seulement un majorant de la probabilité cherchée.
3. Le complémentaire de « au moins un » est « aucun », pas « tous ».
4. Bonferroni contrôle le taux d'erreur familial, pas la proportion de fausses découvertes parmi les rejets (FDR, false discovery rate), ni le risque de manquer un vrai effet.

[[controle:bonferroni]]

## Réflexes en 30 secondes

Lire l'énoncé en cherchant ces indices, puis faire immédiatement le premier geste :

| Indice dans l'énoncé | Outil | Premier geste |
|---|---|---|
| « espérance du nombre de ... », « combien en moyenne » | 1, indicatrices | écrire \(N\) comme somme d'indicatrices, une par objet compté ; calculer la probabilité pour un seul objet ; multiplier |
| « probabilité d'atteindre ... avant ... » | 2, premier pas | \(p = 1/2\) : \(i/n\) ; sinon \(r = q/p\) et \((1 - r^i)/(1 - r^n)\) |
| « durée moyenne », « jusqu'à obtenir ... » | 2, premier pas | \(p = 1/2\) : produit des distances aux barrières ; sinon \((n\,h - i)/(p - q)\) ; pour un motif : états, une équation par état |
| maximum, minimum, \(k\)-ième, étendue, écart, de variables uniformes | 3, statistiques d'ordre | \(k/(n+1)\) pour les espérances ; \(x^n\) et \((1 - x)^n\) pour les probabilités ; mise à l'échelle si l'intervalle n'est pas \([0, 1]\) |
| « au moins un », « aucun », « plusieurs tests », « fausse découverte » | 4, complémentaire | passer à « aucun » ; indépendance : produit ; sinon : Boole ; seuil \(\alpha/m\) pour Bonferroni |

Contrôles de vraisemblance en 5 secondes :

- une probabilité est entre 0 et 1, et la vraie probabilité d'une réunion est entre la plus grande des \(p_j\) et leur somme ;
- si la marche dérive vers le haut (\(p > 1/2\)), \(h(i) > i/n\) ;
- \(\mathbb{E}[U_{(k)}]\) croît avec \(k\), et \(\mathbb{E}[U_{(k)}] + \mathbb{E}[U_{(n+1-k)}] = 1\) par symétrie ;
- une espérance de comptage ne dépasse pas le nombre d'objets que l'on peut compter.

## À savoir refaire sans notes

- La linéarité de l'espérance pour deux variables discrètes, et l'étape où la dépendance disparaît.
- L'espérance et la variance du nombre de points fixes d'une permutation uniforme.
- Les équations du premier pas pour \(h\) et \(d\), la finitude de \(\mathbb{E}[T]\) par blocs, l'unicité par le principe du maximum, les formules des deux cas et l'identité de Wald.
- \(\mathbb{P}(\max \le x) = x^n\), la formule de la queue, et \(\mathbb{E}[U_{(k)}] = k/(n+1)\) par la binomiale et l'intégrale bêta.
- La borne de Boole par les indicatrices, la garantie de Bonferroni, et l'encadrement de \(1 - (1 - p)^m\) par \(1 - e^{-mp}\).
""",
    "controles": {
        "indep": {
            "question": r"Soit \(S\) une permutation uniforme de \(\{1, \dots, n\}\), avec \(n \ge 3\), et \(I_1\), \(I_2\) les indicatrices de \(\{S(1) = 1\}\) et \(\{S(2) = 2\}\). Que peut-on dire ?",
            "options": [
                {"id": "a", "texte": r"Elles sont dépendantes, car \(\mathbb{P}(I_1 = 1, I_2 = 1) = \tfrac{1}{n(n-1)} \ne \tfrac{1}{n^2}\) ; et pourtant \(\mathbb{E}[I_1 + I_2] = 2/n\).",
                 "explication": r"Oui : savoir que 1 est fixe laisse \(n - 1\) images possibles pour 2, ce qui augmente la chance que 2 soit fixe. La linéarité n'a besoin que de \(\mathbb{E}[I_1] = \mathbb{E}[I_2] = 1/n\)."},
                {"id": "b", "texte": r"Elles sont indépendantes, car la permutation est uniforme.",
                 "explication": r"L'uniformité de \(S\) ne rend pas les événements indépendants : \(\mathbb{P}(S(2) = 2 \mid S(1) = 1) = 1/(n-1)\), différent de \(\mathbb{P}(S(2) = 2) = 1/n\)."},
                {"id": "c", "texte": r"Elles sont dépendantes, donc \(\mathbb{E}[I_1 + I_2]\) ne peut se calculer qu'avec leur loi jointe.",
                 "explication": r"C'est l'erreur que l'outil 1 corrige : l'espérance d'une somme ne fait intervenir que les lois des termes pris séparément, quelle que soit leur dépendance."},
                {"id": "d", "texte": r"Elles sont dépendantes, et \(\mathbb{E}[I_1 + I_2] = \tfrac{2}{n} - \tfrac{1}{n(n-1)}\), en retirant le cas où les deux sont fixes.",
                 "explication": r"Retirer l'intersection donne \(\mathbb{P}(I_1 = 1 \text{ ou } I_2 = 1)\), la probabilité qu'au moins un des deux soit fixe. La somme \(I_1 + I_2\) vaut 2 quand les deux sont fixes : ce cas doit compter double, pas être retiré."},
            ],
        },
        "urne": {
            "question": r"Une urne contient 10 boules numérotées de 1 à 10. On en tire 4 sans remise, les @Ca_nb_sous_ens@ sous-ensembles de 4 boules étant équiprobables. Quelle est l'espérance de la somme des 4 numéros tirés ?",
            "options": [
                {"id": "a", "texte": "@Ca_exact@",
                 "explication": r"Oui : avec \(I_k\) l'indicatrice de « la boule \(k\) est tirée », de probabilité \(\tfrac{@Ca_p_tiree_num@}{@Ca_p_tiree_den@}\), la somme vaut \(\sum_k k\,I_k\), d'espérance \(\tfrac{@Ca_p_tiree_num@}{@Ca_p_tiree_den@} \times @Ca_somme_tot@ = @Ca_exact@\). Autre voie : chacun des 4 tirages, pris seul, est uniforme sur 1 à 10, d'espérance @Ca_d_un:1@."},
                {"id": "b", "texte": "On ne peut pas conclure sans la loi jointe, car les tirages sans remise sont dépendants.",
                 "explication": r"La linéarité ne demande aucune indépendance : les tirages sont dépendants, mais chacun, pris seul, a la loi uniforme sur 1 à 10, et cela suffit."},
                {"id": "c", "texte": "@Ca_d_un:1@",
                 "explication": r"C'est l'espérance d'un seul numéro. La somme de 4 numéros a une espérance 4 fois plus grande."},
                {"id": "d", "texte": "@Ca_d_moy_fausse@",
                 "explication": r"La moyenne de 1 à 10 a été prise égale à \(10/2\). Elle vaut \((1 + 10)/2\) : la moyenne d'une suite d'entiers consécutifs est la demi-somme de ses extrémités."},
            ],
        },
        "unicite": {
            "question": r"Pour la marche symétrique entre 0 et \(n\), on a vérifié que \(i\,(n - i)\) satisfait \(d(i) = 1 + \tfrac12 d(i+1) + \tfrac12 d(i-1)\) pour \(0 < i < n\) et s'annule en 0 et en \(n\). Qu'est-ce qui permet de conclure que c'est bien l'espérance du temps d'arrêt ?",
            "options": [
                {"id": "a", "texte": "L'espérance du temps d'arrêt est finie et vérifie ce même système, qui n'a qu'une solution (principe du maximum).",
                 "explication": r"Oui : il faut les trois ingrédients. Finitude (argument des blocs), équation vérifiée par la vraie espérance (premier pas), unicité de la solution. La suite devinée est alors forcément la bonne."},
                {"id": "b", "texte": "Une suite qui vérifie l'équation et la condition en 0 est unique.",
                 "explication": r"Faux : l'équation est une récurrence d'ordre 2, et une seule condition laisse une constante libre. Par exemple \(-i^2 + c\,i\) vérifie l'équation et s'annule en 0 pour tout réel \(c\)."},
                {"id": "c", "texte": "La marche s'arrête avec probabilité 1, cela suffit.",
                 "explication": r"Un arrêt presque sûr ne donne ni la valeur de l'espérance, ni même sa finitude : avec une seule barrière en 0, la marche symétrique s'arrête avec probabilité 1 mais son temps moyen est infini."},
                {"id": "d", "texte": "Toute fonction du second degré en \(i\) vérifie cette équation.",
                 "explication": r"Faux : seules les fonctions \(-i^2 + c_1 i + c_2\) la vérifient, car le second membre impose une différence seconde égale à \(-2\). Et cela ne dirait rien du lien avec l'espérance."},
            ],
        },
        "derive": {
            "question": r"Une marche part de 5, monte de 1 avec probabilité 0,6 et descend de 1 avec probabilité 0,4, pas indépendants, et s'arrête en 0 ou en 10. Quelle est la probabilité qu'elle s'arrête en 10, à trois décimales ?",
            "options": [
                {"id": "a", "texte": "@Cb_h:3@",
                 "explication": r"Oui : \(r = q/p = 2/3\), et pour \(i = n/2\), \(h = \dfrac{1 - r^5}{1 - r^{10}} = \dfrac{1}{1 + r^5}\) (factoriser \(1 - r^{10} = (1 - r^5)(1 + r^5)\)), avec \(r^5 = \tfrac{@Cb_r5_num@}{@Cb_r5_den@}\) : \(h = \tfrac{@Cb_h_num@}{@Cb_h_den@}\)."},
                {"id": "b", "texte": "0,500",
                 "explication": r"C'est \(i/n\), la formule du cas symétrique. Ici la marche dérive vers le haut, donc \(h\) doit dépasser 1/2."},
                {"id": "c", "texte": "@Cb_d_inverse:3@",
                 "explication": r"C'est le résultat avec \(r = p/q\) au lieu de \(q/p\) : la probabilité de finir en bas. Le contrôle « \(p > 1/2\) implique \(h > i/n\) » détecte l'inversion en 5 secondes."},
                {"id": "d", "texte": "@Cb_d_cinq:3@",
                 "explication": r"C'est \(0{,}6^5\), la probabilité de monter 5 fois de suite : un seul des chemins qui mènent à 10. Les chemins avec des allers-retours comptent aussi."},
            ],
        },
        "ordre": {
            "question": r"On tire 7 variables indépendantes de loi uniforme sur \([0, 1]\). Quelle est l'espérance de la deuxième plus grande ?",
            "options": [
                {"id": "a", "texte": "@Cc_exact_num@/@Cc_exact_den@",
                 "explication": r"Oui : la deuxième plus grande est \(U_{(6)}\), d'espérance \(6/(7+1)\)."},
                {"id": "b", "texte": "@Cc_d_petit_num@/@Cc_d_petit_den@",
                 "explication": r"C'est \(\mathbb{E}[U_{(2)}] = 2/8\), la deuxième plus petite. La deuxième plus grande est \(U_{(n-1)}\)."},
                {"id": "c", "texte": "@Cc_d_n_num@/@Cc_d_n_den@",
                 "explication": r"Le dénominateur est \(n = 7\) au lieu de \(n + 1 = 8\) : 7 points découpent le segment en 8 morceaux."},
                {"id": "d", "texte": "@Cc_d_max_num@/@Cc_d_max_den@",
                 "explication": r"C'est l'espérance du maximum, \(7/8\). La deuxième plus grande se trouve en moyenne un morceau plus bas."},
            ],
        },
        "bonferroni": {
            "question": r"On réalise 10 tests et l'on veut que la probabilité de rejeter à tort au moins une hypothèse nulle vraie soit au plus 0,05, sans rien savoir de la dépendance entre les tests. Que fait la correction de Bonferroni, et que garantit-elle ?",
            "options": [
                {"id": "a", "texte": "Chaque test au niveau @Cd_seuil:3@ ; le taux d'erreur familial est au plus 0,05 quelle que soit la dépendance.",
                 "explication": r"Oui : par la borne de Boole, la probabilité d'au moins un rejet à tort est au plus la somme des 10 probabilités, chacune au plus @Cd_seuil:3@."},
                {"id": "b", "texte": "Chaque test au niveau @Cd_seuil:3@ ; la garantie suppose les tests indépendants.",
                 "explication": r"La borne de Boole ne demande aucune indépendance. C'est le calcul exact \(1 - (1 - t)^m\) qui la demanderait."},
                {"id": "c", "texte": "Chaque test au niveau @Cd_seuil:3@ ; au plus 5 % des hypothèses rejetées sont des fausses découvertes.",
                 "explication": r"Cela décrit le contrôle de la proportion de fausses découvertes (FDR), une autre garantie. Bonferroni borne la probabilité d'au moins une fausse découverte."},
                {"id": "d", "texte": "Chaque test au niveau 0,05 ; chaque test ne se trompe qu'avec probabilité 0,05, donc l'ensemble aussi.",
                 "explication": r"Le niveau de chaque test ne borne pas le taux familial : sous indépendance, 10 tests au niveau 0,05 donnent au moins un rejet à tort avec probabilité \(1 - @Cd_q:2m@^{10} \approx @Cd_fwer_sans:3m@\), et Boole ne garantit que \(@Cd_borne_sans:1m@\)."},
            ],
        },
    },
    "verif": {"script": "verif/seance_math_1.py"},
}
