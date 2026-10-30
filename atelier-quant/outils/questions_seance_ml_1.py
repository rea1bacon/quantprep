"""Séance de ML 1 : leçon sur la régression linéaire, Ridge et Lasso, puis 12 questions à choix multiple.

Thèmes : moindres carrés, Ridge et Lasso au niveau démonstration.
Toutes les valeurs : verif/seance_ml_1.py. Dans chaque question, l'option "a" est la bonne.
"""
import lecons_reglin
import questions_diagnostic as diag

ID = "seance-ml-1"
TITRE = "Séance de ML 1 : régression linéaire, Ridge et Lasso"
VERIF = "verif/seance_ml_1.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [{"id": "lecon", "nom": "Leçon", "budget_min": 50}, {"id": "mlf", "nom": "Machine Learning fondamental", "budget_min": 25}]
PRESENTATION = (
    "D'abord une leçon d'environ 50 minutes sur la régression linéaire au niveau démonstration : moindres carrés, "
    "hypothèses du modèle et conséquence de chaque violation, R², Ridge, Lasso, biais et variance. Sept points de "
    "contrôle sont corrigés immédiatement, et le notebook permet de relancer chaque expérience. Ensuite 12 questions "
    "à choix multiple couvrant toute la leçon, dont la moitié de difficulté 4 ou 5.\n\n"
    "Objectif de vitesse : chaque question se résout en moins de 2 minutes et demie avec la leçon. Au-delà du double "
    "du temps cible affiché, choisis « Je ne sais pas » et passe à la suivante : c'est la règle d'abandon à suivre dans un test chronométré."
)
AVERTISSEMENTS = [
    "Les questions de ML sont présentées ici en choix multiple à une seule bonne réponse. Cherche la réponse avant de "
    "lire les options, pour t'entraîner aussi au format à réponse numérique.",
    "Ici le retour en arrière est permis et il n'y a pas de pénalité : ce sont les règles de cette plateforme, pas "
    "celles de tout test en ligne.",
]

LECONS = [lecons_reglin.LECON_REGLIN]
CODES = []

QUESTIONS = [
    # ------------------------------------------------- Moindres carrés
    {
        "id": "F1-01", "section": "mlf", "theme": "Moindres carrés, régression simple", "difficulte": 2, "temps_cible_s": 75,
        "enonce": r"""On observe cinq couples \((x_i, y_i)\) : (1, 2), (2, 4), (3, 5), (4, 4), (5, 5). On ajuste par moindres carrés ordinaires le modèle \(y_i \approx b_1 + b_2 x_i\), **avec constante**.

Que vaut la pente estimée \(\hat{b}_2\), arrondie à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q01_exact:3@", "explication": r"Oui : \(S_{xy}/S_{xx} = @Q01_sxy@/@Q01_sxx@\), avec les sommes centrées autour de \(\bar{x} = @Q01_xbar@\) et \(\bar{y} = @Q01_ybar@\)."},
            {"id": "b", "texte": "@Q01_d_origine:3@", "explication": r"C'est \(\sum_i x_i y_i / \sum_i x_i^2 = @Q01_xy@/@Q01_xx@\), la pente de la régression **sans constante**, forcée de passer par l'origine. Avec constante, on centre \(x\) et \(y\)."},
            {"id": "c", "texte": "@Q01_d_inverse:3@", "explication": r"C'est \(S_{xy}/S_{yy} = @Q01_sxy@/@Q01_syy@\), la pente de la régression de \(x\) sur \(y\) : les rôles sont échangés. La régression de \(y\) sur \(x\) divise par la dispersion de \(x\)."},
            {"id": "d", "texte": "@Q01_d_corr:3@", "explication": r"C'est la corrélation \(S_{xy}/\sqrt{S_{xx} S_{yy}}\). Elle n'égale la pente que si \(x\) et \(y\) ont la même dispersion (\(S_{xx} = S_{yy}\))."},
        ],
        "solution": r"""Réponse : \(\hat{b}_2 = @Q01_sxy@/@Q01_sxx@ = @Q01_exact:3m@\).

Notations : \(\bar{x}\) et \(\bar{y}\) sont les moyennes des \(x_i\) et des \(y_i\) ; \(S_{xx} = \sum_i (x_i - \bar{x})^2\), \(S_{xy} = \sum_i (x_i - \bar{x})(y_i - \bar{y})\), \(S_{yy} = \sum_i (y_i - \bar{y})^2\).

Étape 1 : les équations normales pour les colonnes \(\mathbf{1}\) et \(\mathbf{x}\) disent que les résidus sont de somme nulle et orthogonaux à \(\mathbf{x}\). La première donne \(\hat{b}_1 = \bar{y} - \hat{b}_2 \bar{x}\) ; en la reportant dans la seconde, \(\hat{b}_2 = S_{xy}/S_{xx}\) (leçon, partie 2, hétéroscédasticité, étape 3).

Étape 2 : \(\bar{x} = @Q01_xbar@\), \(\bar{y} = @Q01_ybar@\) ; écarts \(x_i - \bar{x}\) : \(@Q01_dx@\) ; écarts \(y_i - \bar{y}\) : \(@Q01_dy@\).

Étape 3 : \(S_{xx} = @Q01_sxx@\), \(S_{xy} = (-2)(-2) + 0 + 0 + 0 + 2 \times 1 = @Q01_sxy@\), donc \(\hat{b}_2 = @Q01_exact:3m@\) et \(\hat{b}_1 = @Q01_ybar@ - @Q01_exact:1m@ \times @Q01_xbar@ = @Q01_const:1m@\).

Vérification (script) : `np.polyfit` donne la même pente.""",
    },
    {
        "id": "F1-02", "section": "mlf", "theme": "Matrice chapeau, variance résiduelle", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""Modèle \(\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}\) avec \(n = 50\) observations et une matrice \(\mathbf{X}\) fixe à 4 colonnes, la constante comprise, de rang 4. Le bruit vérifie \(\mathbb{E}[\boldsymbol{\varepsilon}] = \mathbf{0}\) et \(\mathrm{Var}(\boldsymbol{\varepsilon}) = 2\,\mathbf{I}\), avec \(\mathbf{I}\) la matrice identité \(50 \times 50\). Soit \(\hat{\boldsymbol{\varepsilon}}\) le vecteur des résidus des moindres carrés ordinaires.

Que vaut \(\mathbb{E}\big[\lVert \hat{\boldsymbol{\varepsilon}} \rVert^2\big]\), l'espérance de la somme des carrés des résidus ?""",
        "options": [
            {"id": "a", "texte": "@Q02_exact@", "explication": r"Oui : \(\sigma^2\,\mathrm{tr}(\mathbf{I} - \mathbf{H}) = 2 \times (50 - 4)\)."},
            {"id": "b", "texte": "@Q02_d_n@", "explication": r"C'est \(n\sigma^2 = \mathbb{E}\lVert \boldsymbol{\varepsilon} \rVert^2\), l'espérance pour le vrai bruit. Les résidus sont la projection du bruit sur un sous-espace de dimension \(n - p\) : ils perdent \(p\) dimensions."},
            {"id": "c", "texte": "@Q02_d_p@", "explication": r"C'est \(p\sigma^2 = \sigma^2\,\mathrm{tr}(\mathbf{H}) = \mathbb{E}\lVert \mathbf{H}\boldsymbol{\varepsilon} \rVert^2\), la partie du bruit absorbée par l'ajustement, pas les résidus."},
            {"id": "d", "texte": "@Q02_d_np1@", "explication": r"C'est \(\sigma^2 (n - p - 1)\) : la constante a été comptée deux fois. L'énoncé dit que les 4 colonnes comprennent la constante, donc \(p = 4\)."},
        ],
        "solution": r"""Réponse : \(\mathbb{E}\lVert \hat{\boldsymbol{\varepsilon}} \rVert^2 = \sigma^2 (n - p) = 2 \times @Q02_nmp@ = @Q02_exact@\).

Notations : \(\sigma^2 = 2\), \(p = 4\) (constante comprise), \(\mathbf{H} = \mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top\) la matrice chapeau.

Étape 1 : \(\hat{\boldsymbol{\varepsilon}} = (\mathbf{I} - \mathbf{H})\mathbf{Y} = (\mathbf{I} - \mathbf{H})\boldsymbol{\varepsilon}\), car \((\mathbf{I} - \mathbf{H})\mathbf{X} = \mathbf{0}\).

Étape 2 : \(\mathbf{I} - \mathbf{H}\) est symétrique et idempotente, donc \(\lVert \hat{\boldsymbol{\varepsilon}} \rVert^2 = \boldsymbol{\varepsilon}^\top (\mathbf{I} - \mathbf{H})\boldsymbol{\varepsilon} = \mathrm{tr}\big((\mathbf{I} - \mathbf{H})\boldsymbol{\varepsilon}\boldsymbol{\varepsilon}^\top\big)\).

Étape 3 : en espérance, \(\mathbb{E}[\boldsymbol{\varepsilon}\boldsymbol{\varepsilon}^\top] = \sigma^2 \mathbf{I}\), et \(\mathrm{tr}(\mathbf{I} - \mathbf{H}) = n - \mathrm{tr}(\mathbf{H}) = n - p\) (leçon, partie 1, énoncé 4, et partie 2, étape 3).

Conséquence : \(\hat{\sigma}^2 = \lVert \hat{\boldsymbol{\varepsilon}} \rVert^2/(n - p)\) est sans biais pour \(\sigma^2\). Simulation (script) : @Q02_mc:2@.""",
    },
    {
        "id": "F1-03", "section": "mlf", "theme": "Hétéroscédasticité : variance, formule usuelle, moindres carrés pondérés", "difficulte": 5, "temps_cible_s": 150,
        "enonce": r"""Modèle sans constante à deux observations : \(Y_i = \beta x_i + \varepsilon_i\) pour \(i = 1, 2\), avec \(x_1 = 1\), \(x_2 = 2\), \(\beta\) réel inconnu. Les bruits \(\varepsilon_1\), \(\varepsilon_2\) sont indépendants, centrés, de variances respectives 1 et 4.

On note \(\hat{\beta} = (x_1 Y_1 + x_2 Y_2)/(x_1^2 + x_2^2)\) l'estimateur des moindres carrés ordinaires, \(\hat{\sigma}^2 = \sum_i (Y_i - \hat{\beta} x_i)^2/(n - 1)\) avec \(n = 2\), et « formule usuelle » l'estimateur \(\hat{\sigma}^2/(x_1^2 + x_2^2)\) de la variance de \(\hat{\beta}\). Les moindres carrés pondérés (MCP) minimisent \(\sum_i (Y_i - b x_i)^2/\mathrm{Var}(\varepsilon_i)\).

Quelle affirmation est vraie ?""",
        "options": [
            {"id": "a", "texte": r"\(\hat{\beta}\) est sans biais, de variance @Q03_var_mco:2@ ; la formule usuelle vaut @Q03_formule:2@ en espérance ; les MCP ont une variance de @Q03_var_mcp:1@.",
             "explication": r"Oui : sandwich \(\sum_i x_i^2 \sigma_i^2/(\sum_i x_i^2)^2\), espérance de \(\hat{\sigma}^2\) par la trace de \((\mathbf{I} - \mathbf{H})\boldsymbol{\Sigma}\), et variance des MCP \(1/(\sum_i x_i^2/\sigma_i^2)\)."},
            {"id": "b", "texte": r"\(\hat{\beta}\) est biaisé, car les bruits n'ont pas la même variance.",
             "explication": r"Le caractère sans biais n'utilise que \(\mathbb{E}[\varepsilon_i] = 0\) : \(\mathbb{E}[\hat{\beta}] = \beta\,(x_1^2 + x_2^2)/(x_1^2 + x_2^2) = \beta\). L'hétéroscédasticité fausse la variance, pas l'espérance."},
            {"id": "c", "texte": r"\(\hat{\beta}\) est sans biais, de variance @Q03_formule:2@, et c'est le meilleur estimateur linéaire sans biais.",
             "explication": r"@Q03_formule:2@ est l'espérance de la formule usuelle, qui sous-estime la vraie variance @Q03_var_mco:2@. Et Gauss-Markov ne s'applique plus : les MCP font mieux (@Q03_var_mcp:1@)."},
            {"id": "d", "texte": r"\(\hat{\beta}\) est sans biais, de variance @Q03_var_mcp:1@, et c'est le meilleur estimateur linéaire sans biais.",
             "explication": r"@Q03_var_mcp:1@ est la variance des MCP, pas celle des MCO. Les MCO restent sans biais, mais ne sont plus optimaux sous hétéroscédasticité."},
        ],
        "solution": r"""Réponse : \(\hat{\beta}\) est sans biais, \(\mathrm{Var}(\hat{\beta}) = @Q03_var_mco:2m@\), la formule usuelle vaut \(@Q03_formule:2m@\) en espérance (sous-estimation), et les MCP ont une variance de \(@Q03_var_mcp:1m@\).

Notations : \(\sigma_1^2 = 1\), \(\sigma_2^2 = 4\), \(\boldsymbol{\Sigma}\) la matrice diagonale de coefficients \(\sigma_1^2\), \(\sigma_2^2\) ; \(\mathbf{x} = (1, 2)\), \(\mathbf{x}^\top \mathbf{x} = @Q03_xx@\) ; \(\mathbf{H} = \mathbf{x}\mathbf{x}^\top/@Q03_xx@\).

Étape 1, sans biais : \(\hat{\beta} - \beta = (x_1 \varepsilon_1 + x_2 \varepsilon_2)/@Q03_xx@\), d'espérance nulle.

Étape 2, vraie variance : \(\mathrm{Var}(\hat{\beta}) = (x_1^2 \sigma_1^2 + x_2^2 \sigma_2^2)/@Q03_xx@^2 = @Q03_xSx@/@Q03_xx2@ = @Q03_var_mco:2m@\) (indépendance des bruits).

Étape 3, formule usuelle : \(\mathbb{E}\lVert \hat{\boldsymbol{\varepsilon}} \rVert^2 = \mathrm{tr}\big((\mathbf{I} - \mathbf{H})\boldsymbol{\Sigma}\big) = (\sigma_1^2 + \sigma_2^2) - (x_1^2 \sigma_1^2 + x_2^2 \sigma_2^2)/@Q03_xx@ = @Q03_trS@ - @Q03_trHS:1m@ = @Q03_esp_scr:1m@\) (leçon, partie 2, étape 3, avec \(\boldsymbol{\Sigma}\) à la place de \(\sigma^2 \mathbf{I}\)). Avec \(n - 1 = 1\), \(\mathbb{E}[\hat{\sigma}^2] = @Q03_esp_scr:1m@\) et la formule usuelle vaut en espérance \(@Q03_esp_scr:1m@/@Q03_xx@ = @Q03_formule:2m@\), moins de la moitié de la vraie variance : la grande variance est sur l'observation de plus grand levier (\(x_2 = 2\)).

Étape 4, MCP : diviser chaque équation par \(\sigma_i\) donne un modèle homoscédastique ; ses MCO sont le meilleur estimateur linéaire sans biais (Gauss-Markov), de variance \(1/(x_1^2/\sigma_1^2 + x_2^2/\sigma_2^2) = 1/@Q03_info:1m@ = @Q03_var_mcp:1m@\).

Simulation (script) : variance des MCO @Q03_var_mco_mc:3@, espérance de la formule usuelle @Q03_formule_mc:3@, variance des MCP @Q03_var_mcp_mc:3@.""",
    },
    {
        "id": "F1-04", "section": "mlf", "theme": "Biais de variable omise", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""Le vrai modèle est \(Y_i = 1 + 2 x_i + 3 z_i + \varepsilon_i\) pour \(i = 1, \dots, n\), où les \(x_i\) et \(z_i\) sont des nombres fixes et les \(\varepsilon_i\) des bruits centrés. On estime par moindres carrés ordinaires le modèle \(Y_i \approx b_1 + b_2 x_i\), sans la variable \(z\). Dans l'échantillon, la régression par moindres carrés ordinaires de \(z_i\) sur \((1, x_i)\) a une pente égale à 0,5.

Que vaut l'espérance de la pente estimée \(\hat{b}_2\) ?""",
        "options": [
            {"id": "a", "texte": "@Q04_exact:1@", "explication": r"Oui : \(2 + 3 \times 0{,}5\) : la vraie pente plus l'effet de \(z\) multiplié par la pente de \(z\) sur \(x\)."},
            {"id": "b", "texte": "@Q04_d_vrai@", "explication": r"Ce serait vrai si \(z\) n'était pas liée à \(x\) dans l'échantillon (pente nulle) ou sans effet. Ici le bruit du modèle estimé, \(3 z_i + \varepsilon_i\), varie avec \(x_i\)."},
            {"id": "c", "texte": "@Q04_d_somme@", "explication": r"C'est \(2 + 3\) : tout l'effet de \(z\) serait passé dans la pente. Seule la partie de \(z\) alignée sur \(x\) passe, dans la proportion 0,5."},
            {"id": "d", "texte": "@Q04_d_signe:1@", "explication": r"Erreur de signe : le biais \(\gamma\,\delta = 3 \times 0{,}5\) s'ajoute, puisque \(\mathbb{E}[\hat{\mathbf{b}}] = \boldsymbol{\beta} + \gamma\,(\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{z}\)."},
        ],
        "solution": r"""Réponse : \(\mathbb{E}[\hat{b}_2] = 2 + 3 \times 0{,}5 = @Q04_exact:1m@\).

Notations : \(\mathbf{X}\) la matrice de colonnes \(\mathbf{1}\) et \(\mathbf{x}\), \(\boldsymbol{\beta} = (1, 2)\), \(\gamma = 3\), \(\mathbf{z}\) le vecteur des \(z_i\), \(\boldsymbol{\delta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{z}\), dont la seconde coordonnée est la pente 0,5.

Étape 1 : \(\hat{\mathbf{b}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{Y}\) et \(\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \gamma\,\mathbf{z} + \boldsymbol{\varepsilon}\).

Étape 2 : par linéarité de l'espérance, \(\mathbb{E}[\hat{\mathbf{b}}] = \boldsymbol{\beta} + \gamma\,\boldsymbol{\delta}\) (leçon, partie 2, variable omise).

Étape 3 : seconde coordonnée : \(2 + 3 \times 0{,}5 = @Q04_exact:1m@\). La pente est surestimée, parce que l'effet de \(z\) est positif et que \(z\) croît avec \(x\).

Simulation (script, plan fixe construit pour que la pente de \(z\) sur \((1, x)\) soit exactement 0,5) : @Q04_mc:3@.""",
    },
    {
        "id": "F1-05", "section": "mlf", "theme": "Multicolinéarité, facteur d'inflation de la variance", "difficulte": 4, "temps_cible_s": 90,
        "enonce": r"""Dans une régression avec constante sous les hypothèses usuelles (\(\mathbf{X}\) fixe de rang plein, bruits centrés, non corrélés, de même variance \(\sigma^2\)), la régression par moindres carrés de la colonne \(\mathbf{x}_1\) sur toutes les autres colonnes (constante comprise) a un coefficient de détermination \(R_1^2 = 0{,}9\).

On compare à une situation hypothétique où \(\mathbf{x}_1\) aurait la même somme des carrés centrée, mais un \(R_1^2\) nul. Par quel facteur l'écart-type de \(\hat{\beta}_1\) est-il multiplié dans la situation réelle, arrondi à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q05_exact:3@", "explication": r"Oui : la variance est multipliée par \(1/(1 - R_1^2) = @Q05_vif:0@\), l'écart-type par sa racine."},
            {"id": "b", "texte": "@Q05_d_var:3@", "explication": r"C'est le facteur sur la **variance**, \(1/(1 - 0{,}9)\). La question porte sur l'écart-type : il faut la racine carrée."},
            {"id": "c", "texte": "@Q05_d_r:3@", "explication": r"C'est \(1/\sqrt{1 - 0{,}9^2}\) : \(0{,}9\) a été pris pour une corrélation et élevé au carré. L'énoncé donne directement le coefficient de détermination \(R_1^2 = 0{,}9\)."},
            {"id": "d", "texte": "@Q05_d_inv:3@", "explication": r"C'est \(1/\sqrt{R_1^2}\). La formule fait intervenir \(1 - R_1^2\), la part de \(\mathbf{x}_1\) que les autres colonnes n'expliquent pas."},
        ],
        "solution": r"""Réponse : \(\sqrt{1/(1 - 0{,}9)} = \sqrt{@Q05_vif:0@} \approx @Q05_exact:3m@\).

Notations : \(\tilde{\mathbf{x}}_1\) est le résidu de la régression de \(\mathbf{x}_1\) sur les autres colonnes, \(\mathrm{SCT}_1\) la somme des carrés centrée de \(\mathbf{x}_1\).

Étape 1 (Frisch-Waugh, leçon partie 2) : \(\mathrm{Var}(\hat{\beta}_1) = \sigma^2/\lVert \tilde{\mathbf{x}}_1 \rVert^2\).

Étape 2 : comme les autres colonnes contiennent la constante, \(\lVert \tilde{\mathbf{x}}_1 \rVert^2 = \mathrm{SCT}_1 (1 - R_1^2)\), d'où \(\mathrm{Var}(\hat{\beta}_1) = \dfrac{\sigma^2}{\mathrm{SCT}_1} \times \dfrac{1}{1 - R_1^2}\).

Étape 3 : à \(\mathrm{SCT}_1\) égal, la variance est multipliée par \(1/(1 - 0{,}9) = @Q05_vif:0@\), l'écart-type par \(\sqrt{@Q05_vif:0@} \approx @Q05_exact:3m@\).

L'estimateur reste sans biais et le meilleur estimateur linéaire sans biais : la multicolinéarité imparfaite ne viole aucune hypothèse, elle rend seulement l'estimation imprécise. Vérification (script) : plan construit avec \(R_1^2 = 0{,}9\) exactement, rapport des variances égal à \(@Q05_vif:0m@\), et simulation.""",
    },
    # ------------------------------------------------- R²
    {
        "id": "F1-06", "section": "mlf", "theme": "R² ajusté", "difficulte": 2, "temps_cible_s": 60,
        "enonce": r"""Une régression par moindres carrés avec constante et trois variables explicatives est ajustée sur \(n = 30\) observations. Son coefficient de détermination vaut \(R^2 = 0{,}5\). Le R² ajusté est défini par \(R^2_{\mathrm{aj}} = 1 - \dfrac{\mathrm{SCR}/(n - p)}{\mathrm{SCT}/(n - 1)}\), où \(p\) est le nombre de colonnes de la matrice des variables, constante comprise.

Que vaut \(R^2_{\mathrm{aj}}\), arrondi à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q06_exact:3@", "explication": r"Oui : \(p = 4\) et \(1 - 0{,}5 \times @Q06_nm1@/@Q06_nmp@\)."},
            {"id": "b", "texte": "@Q06_d_p5:3@", "explication": r"C'est \(1 - 0{,}5 \times 29/25\) : la constante a été ajoutée une seconde fois, \(p = 5\). La constante et les trois variables font \(p = 4\) colonnes."},
            {"id": "c", "texte": "@Q06_d_n:3@", "explication": r"C'est \(1 - 0{,}5 \times 30/26\) : \(n\) au lieu de \(n - 1\) au dénominateur de la variance totale. L'estimateur sans biais de la variance de \(y\) divise \(\mathrm{SCT}\) par \(n - 1\)."},
            {"id": "d", "texte": "@Q06_d_inv:3@", "explication": r"C'est \(1 - 0{,}5 \times 26/29\) : le rapport des degrés de liberté est inversé. Le R² ajusté est toujours au plus égal au R² (pour \(p \ge 1\)), et cette valeur le dépasse."},
        ],
        "solution": r"""Réponse : \(R^2_{\mathrm{aj}} = 1 - (1 - R^2)\,\dfrac{n - 1}{n - p} = 1 - 0{,}5 \times \dfrac{@Q06_nm1@}{@Q06_nmp@} \approx @Q06_exact:3m@\).

Étape 1 : \(\mathrm{SCR}/\mathrm{SCT} = 1 - R^2 = 0{,}5\), donc \(R^2_{\mathrm{aj}} = 1 - (1 - R^2)(n - 1)/(n - p)\).

Étape 2 : \(p = 4\) (constante et trois variables), \(n - 1 = @Q06_nm1@\), \(n - p = @Q06_nmp@\).

Étape 3 : \(1 - 0{,}5 \times @Q06_rap:4m@ \approx @Q06_exact:3m@\).

Contrôle : \((n - 1)/(n - p) \ge 1\), donc \(R^2_{\mathrm{aj}} \le R^2\).""",
    },
    {
        "id": "F1-07", "section": "mlf", "theme": "Ajout d'une variable : R², R² ajusté et statistique t", "difficulte": 5, "temps_cible_s": 150,
        "enonce": r"""Une régression par moindres carrés avec \(n = 40\) observations et \(p = 3\) colonnes (constante comprise) a une somme des carrés résiduelle \(\mathrm{SCR} = 100\) et une somme des carrés totale \(\mathrm{SCT} = 200\). On ajoute une quatrième colonne, qui n'est pas combinaison linéaire des trois premières ; dans le nouveau modèle, la statistique t de son coefficient vaut \(t = 0{,}8\), avec \(t = \hat{c}/\sqrt{\hat{\sigma}_+^2 [(\mathbf{X}_+^\top \mathbf{X}_+)^{-1}]_{44}}\), où \(\hat{c}\) est le nouveau coefficient, \(\mathbf{X}_+\) la nouvelle matrice et \(\hat{\sigma}_+^2 = \mathrm{SCR}_{\mathrm{nouveau}}/(n - 4)\).

Avec \(R^2 = 1 - \mathrm{SCR}/\mathrm{SCT}\) et \(R^2_{\mathrm{aj}} = 1 - \dfrac{\mathrm{SCR}/(n - p)}{\mathrm{SCT}/(n - 1)}\), \(p\) étant le nombre de colonnes du modèle considéré, que deviennent les deux indicateurs (arrondis à trois décimales) ?""",
        "options": [
            {"id": "a", "texte": r"R² passe de @Q07_r2_a:3@ à @Q07_r2_b:3@, et R² ajusté de @Q07_aj_a:3@ à @Q07_aj_b:3@.",
             "explication": r"Oui : \(\mathrm{SCR}\) baisse d'un facteur \(1 + t^2/(n - p - 1)\), donc R² monte ; mais \(t^2 < 1\), donc le R² ajusté baisse."},
            {"id": "b", "texte": r"R² passe de @Q07_r2_a:3@ à @Q07_r2_b:3@, et R² ajusté de @Q07_aj_a:3@ à @Q07_aj_oubli:3@.",
             "explication": r"Le second R² ajusté divise le nouveau \(\mathrm{SCR}\) par \(n - 3\) au lieu de \(n - 4\) : le nouveau modèle a 4 colonnes, et c'est ce coût qui fait baisser le R² ajusté quand \(t^2 < 1\)."},
            {"id": "c", "texte": r"R² passe de @Q07_r2_a:3@ à @Q07_r2_baisse:3@, car la variable n'est pas significative.",
             "explication": r"R² ne peut pas baisser quand on ajoute une colonne : le coefficient 0 reste disponible, donc le minimum de \(\mathrm{SCR}\) ne peut que diminuer. Cette valeur multiplie \(\mathrm{SCR}\) au lieu de le diviser."},
            {"id": "d", "texte": r"R² et R² ajusté restent à @Q07_r2_a:3@ et @Q07_aj_a:3@, car \(\lvert t \rvert < 1{,}96\).",
             "explication": r"La significativité ne change rien au calcul : \(\mathrm{SCR}\) baisse de \(\hat{c}^2 \lVert \tilde{\mathbf{x}} \rVert^2 > 0\) dès que \(\hat{c} \ne 0\), et la statistique t non nulle le garantit."},
        ],
        "solution": r"""Réponse : R² passe de \(@Q07_r2_a:3m@\) à \(@Q07_r2_b:3m@\) ; le R² ajusté baisse de \(@Q07_aj_a:3m@\) à \(@Q07_aj_b:3m@\), parce que \(t^2 < 1\).

Notations : \(\tilde{\mathbf{x}}\) est le résidu de la nouvelle colonne sur les trois anciennes.

Étape 1 (leçon, partie 3, énoncés 2 et 3) : \(\mathrm{SCR}_{\mathrm{ancien}} - \mathrm{SCR}_{\mathrm{nouveau}} = \hat{c}^2 \lVert \tilde{\mathbf{x}} \rVert^2\), et par Frisch-Waugh \([(\mathbf{X}_+^\top \mathbf{X}_+)^{-1}]_{44} = 1/\lVert \tilde{\mathbf{x}} \rVert^2\), donc \(\hat{c}^2 \lVert \tilde{\mathbf{x}} \rVert^2 = t^2 \hat{\sigma}_+^2\) et \(\mathrm{SCR}_{\mathrm{ancien}} = \mathrm{SCR}_{\mathrm{nouveau}}\big(1 + t^2/(n - 4)\big)\).

Étape 2 : \(t^2 = @Q07_t2:2m@\), \(n - 4 = @Q07_ddl@\), facteur \(@Q07_facteur:5m@\) ; \(\mathrm{SCR}_{\mathrm{nouveau}} = 100/@Q07_facteur:5m@ \approx @Q07_scr_b:2m@\).

Étape 3 : \(R^2\) : \(1 - 100/200 = @Q07_r2_a:3m@\) devient \(1 - @Q07_scr_b:2m@/200 \approx @Q07_r2_b:3m@\).

Étape 4 : \(R^2_{\mathrm{aj}}\) : \(1 - (100/37)/(200/39) \approx @Q07_aj_a:3m@\) devient \(1 - (@Q07_scr_b:2m@/36)/(200/39) \approx @Q07_aj_b:3m@\). Il baisse, comme le prévoit le critère : le R² ajusté monte si et seulement si \(t^2 > 1\).

Vérification (script) : l'identité de l'étape 1 et le critère \(t^2 > 1\) sont contrôlés sur 300 jeux de données aléatoires.""",
    },
    # ------------------------------------------------- Ridge
    {
        "id": "F1-08", "section": "mlf", "theme": "Ridge, lecture par la SVD", "difficulte": 3, "temps_cible_s": 90,
        "enonce": r"""Soit \(\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^\top\) la décomposition en valeurs singulières d'une matrice \(\mathbf{X}\) de rang 3 : \(\mathbf{U}\) a des colonnes orthonormées \(\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3\), \(\mathbf{V}\) des colonnes orthonormées, et les valeurs singulières sont \(d_1 = 3\), \(d_2 = 1\), \(d_3 = 0{,}2\). Ridge minimise \(\tfrac12 \lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2 + \tfrac{\lambda}{2} \lVert \mathbf{b} \rVert^2\), avec \(\lambda = 1\).

Les valeurs ajustées des moindres carrés ordinaires ont pour composante \(\mathbf{u}_3^\top \mathbf{y}\) selon \(\mathbf{u}_3\). Par quel facteur Ridge multiplie-t-il cette composante, arrondi à trois décimales ?""",
        "options": [
            {"id": "a", "texte": "@Q08_exact:3@", "explication": r"Oui : \(d_3^2/(d_3^2 + \lambda) = @Q08_d2:2m@/(@Q08_d2:2m@ + 1)\)."},
            {"id": "b", "texte": "@Q08_d_lin:3@", "explication": r"C'est \(d_3/(d_3 + \lambda)\) : la valeur singulière n'a pas été élevée au carré. Ce sont les valeurs propres de \(\mathbf{X}^\top \mathbf{X}\), les \(d_j^2\), qui sont augmentées de \(\lambda\)."},
            {"id": "c", "texte": "@Q08_d_compl:3@", "explication": r"C'est \(\lambda/(d_3^2 + \lambda)\), la part **retirée** de la composante, pas le facteur conservé. Les deux somment à 1."},
            {"id": "d", "texte": "@Q08_d_orth:3@", "explication": r"C'est \(1/(1 + \lambda)\), le facteur du cas \(\mathbf{X}^\top \mathbf{X} = \mathbf{I}\) (toutes les valeurs singulières égales à 1). Ici le rétrécissement dépend de la direction : c'est le facteur selon \(\mathbf{u}_2\)."},
        ],
        "solution": r"""Réponse : \(d_3^2/(d_3^2 + \lambda) = @Q08_d2:2m@/@Q08_d2_plus_un:2m@ \approx @Q08_exact:3m@\).

Étape 1 (leçon, partie 4, énoncé 2) : \(\mathbf{X}\hat{\boldsymbol{\beta}}_\lambda = \sum_j \dfrac{d_j^2}{d_j^2 + \lambda} (\mathbf{u}_j^\top \mathbf{y})\,\mathbf{u}_j\), alors que les MCO donnent \(\sum_j (\mathbf{u}_j^\top \mathbf{y})\,\mathbf{u}_j\).

Étape 2 : facteurs \(9/10 = @Q08_f1:1m@\) selon \(\mathbf{u}_1\), \(1/2 = @Q08_f2:1m@\) selon \(\mathbf{u}_2\), et \(@Q08_d2:2m@/@Q08_d2_plus_un:2m@ \approx @Q08_exact:3m@\) selon \(\mathbf{u}_3\).

Lecture : la direction de plus petite valeur singulière, celle que les données déterminent le moins bien, est presque entièrement supprimée ; c'est là que les MCO amplifient le bruit, en divisant par \(d_3\).""",
    },
    {
        "id": "F1-09", "section": "mlf", "theme": "Interprétation bayésienne de Ridge et du Lasso", "difficulte": 4, "temps_cible_s": 90,
        "enonce": r"""Soit \(\mathbf{B}\) un vecteur aléatoire de \(\mathbb{R}^p\) de loi a priori \(\mathcal{N}(\mathbf{0}, \tau^2 \mathbf{I})\) avec \(\tau = 4\). Sachant \(\mathbf{B} = \mathbf{b}\), les réponses \(\mathbf{Y}\) suivent la loi \(\mathcal{N}(\mathbf{X}\mathbf{b}, \sigma^2 \mathbf{I})\) avec \(\sigma = 0{,}5\) et \(\mathbf{X}\) fixe. On observe \(\mathbf{Y} = \mathbf{y}\).

Le mode de la loi a posteriori de \(\mathbf{B}\) est le minimiseur de quel critère ? (\(\lVert \cdot \rVert_1\) est la norme L1.)""",
        "options": [
            {"id": "a", "texte": r"Ridge : \(\tfrac12 \lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2 + \tfrac{\lambda}{2} \lVert \mathbf{b} \rVert^2\) avec \(\lambda = @Q09_exact_num@/@Q09_exact_den@\).",
             "explication": r"Oui : \(\lambda = \sigma^2/\tau^2 = @Q09_s2:2m@/@Q09_t2@\)."},
            {"id": "b", "texte": r"Lasso : \(\tfrac12 \lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2 + \lambda \lVert \mathbf{b} \rVert_1\) avec \(\lambda = @Q09_exact_num@/@Q09_exact_den@\).",
             "explication": r"Un a priori gaussien donne une pénalité quadratique, \(\lVert \mathbf{b} \rVert^2/(2\tau^2)\) dans moins le logarithme de la densité. La pénalité L1 correspond à un a priori de Laplace, de densité proportionnelle à \(e^{-\lvert b_j \rvert/\tau}\)."},
            {"id": "c", "texte": r"Ridge avec \(\lambda = @Q09_d_inv@\).",
             "explication": r"C'est \(\tau^2/\sigma^2\), le rapport inversé. Un a priori plus large (\(\tau\) grand) doit pénaliser **moins** : \(\lambda\) doit décroître avec \(\tau\)."},
            {"id": "d", "texte": r"Ridge avec \(\lambda = @Q09_d_tau_num@/@Q09_d_tau_den@\).",
             "explication": r"C'est \(1/\tau^2\) : la vraisemblance n'a pas été remise à l'échelle. Moins le logarithme a posteriori vaut \(\lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2/(2\sigma^2) + \lVert \mathbf{b} \rVert^2/(2\tau^2)\) ; il faut multiplier par \(\sigma^2\) pour obtenir le critère Ridge."},
        ],
        "solution": r"""Réponse : Ridge avec \(\lambda = \sigma^2/\tau^2 = @Q09_s2:2m@/@Q09_t2@ = \tfrac{@Q09_exact_num@}{@Q09_exact_den@}\).

Étape 1 : par la formule de Bayes, la densité a posteriori est proportionnelle à \(\exp\big(-\lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2/(2\sigma^2)\big) \times \exp\big(-\lVert \mathbf{b} \rVert^2/(2\tau^2)\big)\).

Étape 2 : maximiser cette densité revient à minimiser \(\lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2/(2\sigma^2) + \lVert \mathbf{b} \rVert^2/(2\tau^2)\), puis, en multipliant par \(\sigma^2 > 0\), \(\tfrac12 \lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2 + \tfrac{\sigma^2}{2\tau^2} \lVert \mathbf{b} \rVert^2\).

Étape 3 : \(\sigma^2 = @Q09_s2:2m@\), \(\tau^2 = @Q09_t2@\), donc \(\lambda = \tfrac{@Q09_exact_num@}{@Q09_exact_den@}\). La loi a posteriori est gaussienne (leçon, partie 4, énoncé 4) : sa moyenne est aussi la solution Ridge. Avec un a priori de Laplace de paramètre \(\tau\), on obtiendrait le Lasso avec \(\lambda = \sigma^2/\tau\), mais seulement pour le mode.""",
    },
    # ------------------------------------------------- Lasso
    {
        "id": "F1-10", "section": "mlf", "theme": "Lasso, cas orthonormal : seuillage doux", "difficulte": 3, "temps_cible_s": 75,
        "enonce": r"""La matrice \(\mathbf{X}\) a trois colonnes orthonormées (\(\mathbf{X}^\top \mathbf{X} = \mathbf{I}\)) et \(\mathbf{X}^\top \mathbf{y} = (2{,}5 ; -0{,}4 ; -1{,}1)\). On minimise le critère du Lasso \(\tfrac12 \lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2 + \lambda \lVert \mathbf{b} \rVert_1\), où \(\lVert \mathbf{b} \rVert_1 = \lvert b_1 \rvert + \lvert b_2 \rvert + \lvert b_3 \rvert\), avec \(\lambda = 0{,}6\).

Quel est le minimiseur ?""",
        "options": [
            {"id": "a", "texte": "@Q10_exact:1@", "explication": r"Oui : seuillage doux, chaque coordonnée perd \(0{,}6\) en valeur absolue, et celle de valeur absolue \(0{,}4 \le 0{,}6\) devient nulle."},
            {"id": "b", "texte": "@Q10_d_ridge:4@", "explication": r"C'est la solution Ridge \(\mathbf{X}^\top \mathbf{y}/(1 + \lambda)\) : division, aucun zéro. Le Lasso soustrait \(\lambda\) en valeur absolue."},
            {"id": "c", "texte": "@Q10_d_dur:1@", "explication": r"C'est le seuillage dur : on annule les petites coordonnées et on garde les autres intactes. Le Lasso rétrécit aussi les coordonnées actives de \(\lambda\)."},
            {"id": "d", "texte": "@Q10_d_signe:1@", "explication": r"On a soustrait \(\lambda\) sans tenir compte du signe. Le seuillage doux rapproche chaque coordonnée de 0 : \(\mathrm{signe}(z_j)\max(\lvert z_j \rvert - \lambda, 0)\)."},
        ],
        "solution": r"""Réponse : \(\hat{\mathbf{b}} = @Q10_exact:1m@\).

Notations : \(\mathbf{z} = \mathbf{X}^\top \mathbf{y} = @Q10_z:1m@\), l'estimateur des moindres carrés ordinaires puisque \(\mathbf{X}^\top \mathbf{X} = \mathbf{I}\).

Étape 1 (leçon, partie 5, énoncé 5) : le critère vaut une constante plus \(\sum_j \big(\tfrac12 (z_j - b_j)^2 + \lambda \lvert b_j \rvert\big)\) : trois problèmes séparés.

Étape 2 : pour chacun, les conditions d'optimalité donnent \(\hat{b}_j = \mathrm{signe}(z_j)\max(\lvert z_j \rvert - \lambda, 0)\).

Étape 3 : \(2{,}5 - 0{,}6\) ; \(\lvert -0{,}4 \rvert \le 0{,}6\) donne 0 ; \(-(1{,}1 - 0{,}6)\). D'où \(@Q10_exact:1m@\).

Vérification (script) : scikit-learn, `Lasso(alpha = 0,6/n)` sur une matrice à colonnes orthonormées, redonne ce vecteur.""",
    },
    {
        "id": "F1-11", "section": "mlf", "theme": "Lasso : seuil de nullité, convention de scikit-learn", "difficulte": 4, "temps_cible_s": 90,
        "enonce": r"""`Lasso(alpha=a, fit_intercept=False)` de scikit-learn minimise \(\dfrac{1}{2n} \lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2 + a\,\lVert \mathbf{b} \rVert_1\), où \(n\) est le nombre d'observations et \(\lVert \mathbf{b} \rVert_1\) la somme des valeurs absolues des coordonnées. On a \(n = 200\) observations, trois colonnes, et \(\mathbf{X}^\top \mathbf{y} = (4, -6, 2)\).

Quelle est la plus petite valeur de \(a\) pour laquelle la solution est le vecteur nul ?""",
        "options": [
            {"id": "a", "texte": "@Q11_exact:3@", "explication": r"Oui : \(\max_j \lvert \mathbf{x}_j^\top \mathbf{y} \rvert / n = 6/200\)."},
            {"id": "b", "texte": "@Q11_d_signe:3@", "explication": r"C'est \(4/200\) : le maximum a été pris sur les valeurs signées. La condition porte sur \(\lvert \mathbf{x}_j^\top \mathbf{y} \rvert\), et \(\lvert -6 \rvert = 6\) est le plus grand."},
            {"id": "c", "texte": "@Q11_d_sans_n@", "explication": r"C'est le seuil \(\lambda_{\max} = 6\) du critère \(\tfrac12 \lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2 + \lambda \lVert \mathbf{b} \rVert_1\). Le critère de scikit-learn divise la perte par \(n\), d'où \(a = \lambda/n\)."},
            {"id": "d", "texte": "@Q11_d_2n:3@", "explication": r"C'est \(6/(2n)\) : le facteur \(\tfrac12\) a été compté deux fois. En multipliant le critère de scikit-learn par \(n\), on obtient \(\tfrac12 \lVert \cdot \rVert^2 + n a \lVert \mathbf{b} \rVert_1\), donc \(\lambda = n a\)."},
        ],
        "solution": r"""Réponse : \(a = @Q11_max@/200 = @Q11_exact:3m@\).

Étape 1 : multiplier le critère par \(n > 0\) ne change pas son minimiseur : on obtient \(\tfrac12 \lVert \mathbf{y} - \mathbf{X}\mathbf{b} \rVert^2 + \lambda \lVert \mathbf{b} \rVert_1\) avec \(\lambda = n a\).

Étape 2 (leçon, partie 5, énoncé 4) : la solution est nulle si et seulement si \(\lambda \ge \max_j \lvert \mathbf{x}_j^\top \mathbf{y} \rvert = @Q11_max@\). En effet, en \(\mathbf{b} = \mathbf{0}\), \(\mathbf{g} = \mathbf{X}^\top \mathbf{y}\), et les conditions d'optimalité exigent \(\lvert g_j \rvert \le \lambda\) pour tout \(j\).

Étape 3 : \(n a \ge @Q11_max@\), soit \(a \ge @Q11_max@/200 = @Q11_exact:3m@\).

Vérification (script) : avec scikit-learn, sur des données construites pour que \(\mathbf{X}^\top \mathbf{y} = (4, -6, 2)\), la solution est nulle juste au-dessus de ce seuil et non nulle juste en dessous.""",
    },
    # ------------------------------------------------- Biais et variance
    {
        "id": "F1-12", "section": "mlf", "theme": "Biais, variance et choix de lambda pour Ridge", "difficulte": 4, "temps_cible_s": 120,
        "enonce": r"""Modèle \(\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}\), avec \(\mathbf{X}\) fixe à colonnes orthonormées (\(\mathbf{X}^\top \mathbf{X} = \mathbf{I}\)), \(\mathbb{E}[\boldsymbol{\varepsilon}] = \mathbf{0}\) et \(\mathrm{Var}(\boldsymbol{\varepsilon}) = \sigma^2 \mathbf{I}\), \(\sigma^2 = 1\). Le premier vrai coefficient vaut \(\beta_1 = 2\). Ridge minimise \(\tfrac12 \lVert \mathbf{Y} - \mathbf{X}\mathbf{b} \rVert^2 + \tfrac{\lambda}{2} \lVert \mathbf{b} \rVert^2\), \(\lambda \ge 0\) ; on note \(\hat{\beta}_{\lambda, 1}\) sa première coordonnée.

Quelle valeur de \(\lambda\) minimise \(\mathbb{E}\big[(\hat{\beta}_{\lambda, 1} - \beta_1)^2\big]\), et que vaut alors ce minimum ?""",
        "options": [
            {"id": "a", "texte": r"\(\lambda = @Q12_lopt:2m@\), erreur @Q12_mse_opt:1@.", "explication": r"Oui : \(\lambda = \sigma^2/\beta_1^2\), et l'erreur minimale vaut \(\sigma^2 \beta_1^2/(\sigma^2 + \beta_1^2)\), plus petite que l'erreur \(\sigma^2 = 1\) des MCO."},
            {"id": "b", "texte": r"\(\lambda = 0\), erreur @Q12_mse0@, car les MCO sont sans biais.", "explication": r"Sans biais ne veut pas dire erreur minimale : la dérivée de l'erreur en \(\lambda = 0\) vaut \(-2\sigma^2 < 0\), donc un petit \(\lambda > 0\) fait mieux."},
            {"id": "c", "texte": r"\(\lambda = @Q12_d_inv_l@\), erreur @Q12_d_inv_mse:1@.", "explication": r"C'est \(\lambda = \beta_1^2/\sigma^2\), le rapport inversé, qui écrase trop le coefficient : l'erreur dépasse même celle des MCO."},
            {"id": "d", "texte": r"\(\lambda = @Q12_d_rac_l:1m@\), erreur @Q12_d_rac_mse:3@.", "explication": r"C'est \(\lambda = \sigma/\beta_1\) : écarts-types au lieu de variances. Le point où la dérivée s'annule est \(\lambda \beta_1^2 = \sigma^2\)."},
        ],
        "solution": r"""Réponse : \(\lambda = \sigma^2/\beta_1^2 = @Q12_lopt:2m@\), et l'erreur quadratique vaut alors \(\tfrac{@Q12_mse_opt_num@}{@Q12_mse_opt_den@} = @Q12_mse_opt:1m@\) (contre \(@Q12_mse0@\) pour les MCO).

Notations : \(Z_1 = \mathbf{x}_1^\top \mathbf{Y}\), première coordonnée des MCO, d'espérance \(\beta_1\) et de variance \(\sigma^2\) puisque \(\mathbf{X}^\top \mathbf{X} = \mathbf{I}\).

Étape 1 : avec des colonnes orthonormées, \(\hat{\beta}_{\lambda, 1} = Z_1/(1 + \lambda)\) (leçon, partie 4, cas limites).

Étape 2 : biais \(\beta_1/(1 + \lambda) - \beta_1 = -\lambda \beta_1/(1 + \lambda)\), variance \(\sigma^2/(1 + \lambda)^2\), donc \(g(\lambda) = \dfrac{\lambda^2 \beta_1^2 + \sigma^2}{(1 + \lambda)^2}\) (leçon, partie 6, avec \(\mu_1 = 1\) et \(\theta_1 = \beta_1\)).

Étape 3 : \(g'(\lambda) = \dfrac{2(\lambda \beta_1^2 - \sigma^2)}{(1 + \lambda)^3}\), négative puis positive : minimum en \(\lambda = \sigma^2/\beta_1^2 = 1/4\).

Étape 4 : \(g(1/4) = \dfrac{\tfrac{1}{16} \times 4 + 1}{(5/4)^2} = \dfrac{5/4}{25/16} = \tfrac{@Q12_mse_opt_num@}{@Q12_mse_opt_den@}\).

Limite pratique : \(\beta_1\) est inconnu, donc ce \(\lambda\) optimal aussi ; on le choisit par validation sur des données non utilisées pour l'ajustement. Simulation (script) : @Q12_mse_opt_mc:3@.""",
    },
]
