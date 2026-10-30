"""Leçon de la séance de code 2 : séries temporelles en panel, sans fuite entre actifs ni regard vers le futur.

Format : voir outils/lecons.py. Les blocs python-exec sont exécutés à la construction par le noyau du notebook
(.venv-hackerrank : pandas 1.4.2), dans coding/seance2, et leur vraie sortie est insérée.
Les affirmations sur les versions sont vérifiées dans les deux environnements par verif/lecon_pandas_2.py ;
les nombres cités hors des blocs exécutés viennent de verif/seance_code_2.out.json.
"""

LECON_PANDAS_2 = {
    "id": "L-PANDAS-2", "type": "lecon", "section": "lecon",
    "theme": "Leçon pandas 2 : fenêtres glissantes, rééchantillonnage, merge_asof, normalisation, backtest",
    "difficulte": 3, "temps_cible_s": 1200, "dossier": "coding/seance2",
    "contenu": r"""## Leçon : séries temporelles en panel, sans fuite

Durée visée : 20 minutes. Lis dans l'ordre, exécute les exemples dans le notebook de droite si tu veux les modifier, et réponds à chaque point de contrôle avant de continuer. Les sorties affichées sous chaque exemple sont les vraies sorties de pandas 1.4.2, la version de l'environnement des notebooks de la plateforme.

Conventions de cette leçon : une minuscule désigne une valeur observée, donc fixe (un prix, un rendement déjà réalisé) ; une majuscule une variable aléatoire, utilisée seulement à la section 6 pour justifier l'annualisation du ratio de Sharpe.

### 1. Le problème

Un panel a une ligne par couple (date, actif). La séance de code 1 a traité le décalage et le rang par groupe. Celle-ci traite les quatre gestes suivants d'une chaîne de recherche : moyenne glissante par actif, passage à une fréquence mensuelle, association de la dernière information disponible, normalisation par date, puis backtest.

Tous obéissent à une même règle, qui se viole sans aucun message d'erreur. La valeur calculée pour la ligne (date \(t\), actif \(a\)) ne doit utiliser :

1. pour un calcul **par actif** (fenêtre glissante, rendement), que des lignes de l'actif \(a\) : sinon il y a **fuite entre actifs** ;
2. pour un calcul **par date** (rang, z-score, quantile), que des lignes de la date \(t\) : une statistique calculée sur tout l'échantillon contient les dates futures ;
3. dans tous les cas, que de l'information **disponible en \(t\)** : sinon il y a **lookahead** (regard vers le futur), et le backtest mesure un gain impossible à réaliser.

### 2. Fenêtre glissante par actif

**Ce que calcule `rolling`.** Pour une série `s` et des entiers \(n \ge 1\) (la fenêtre) et \(m\) (le minimum, \(1 \le m \le n\)), `s.rolling(n, min_periods=m).mean()` associe à la ligne de position \(i\) les lignes de positions \(i - n + 1\) à \(i\) (moins au début de la série). Il en garde les valeurs **non manquantes** ; si elles sont au moins \(m\), le résultat est leur moyenne, sinon `NaN`. Sans `min_periods`, \(m\) vaut \(n\). Les positions sont celles des lignes **dans l'ordre où elles se trouvent** : comme `shift`, `rolling` exige un cadre trié par date.

```python-exec
import numpy as np
import pandas as pd

s = pd.Series([1.0, np.nan, 3.0, 4.0])
pd.DataFrame({"min_periods=1": s.rolling(3, min_periods=1).mean(),
              "min_periods=2": s.rolling(3, min_periods=2).mean(),
              "sans min_periods": s.rolling(3).mean()})
```

Lecture de la ligne 3 : la fenêtre contient `NaN`, 3 et 4, soit deux valeurs non manquantes ; avec \(m = 1\) ou \(m = 2\), la moyenne est \((3 + 4)/2\) ; sans `min_periods`, il en faudrait 3, d'où `NaN` partout. Par défaut, toute fenêtre qui contient une valeur manquante, ou qui compte moins de \(n\) lignes, donne donc `NaN`.

**La fuite entre actifs.** Sur un panel trié par date, une fenêtre sans `groupby` déborde d'un actif sur l'autre :

```python-exec
r = pd.DataFrame({"date": ["2026-01-05", "2026-01-06", "2026-01-07"] * 2,
                  "actif": ["A"] * 3 + ["B"] * 3,
                  "rendement": [1.0, np.nan, 3.0, 10.0, 20.0, 30.0]})
d = r.sort_values(["date", "actif"])
d["sans_groupby"] = d["rendement"].rolling(2, min_periods=1).mean()
d
```

La ligne de B au 2026-01-05 reçoit la moyenne de A et de B le même jour : c'est faux. Avec `groupby`, chaque actif a sa propre fenêtre, mais le résultat a **deux niveaux d'index**, l'actif puis l'étiquette d'origine :

```python-exec
m = d.groupby("actif")["rendement"].rolling(2, min_periods=1).mean()
m
```

Ce n'est pas l'index de `d`, et l'affectation directe échoue :

```python-exec-erreur
d["glissante"] = m
```

Deux écritures justes, dans les deux versions de pandas : retirer le niveau de l'actif, ou passer par `transform`, qui garde l'index d'origine.

```python-exec
d["glissante"] = m.reset_index(level=0, drop=True)
d["avec_transform"] = d.groupby("actif")["rendement"].transform(lambda x: x.rolling(2, min_periods=1).mean())
d
```

L'affectation range chaque valeur à sa ligne par étiquette (séance de code 1, section 2). La première ligne de chaque actif ne voit plus que lui-même.

[[controle:fuite]]

### 3. Rééchantillonnage : passer à une fréquence mensuelle

**Ce que fait `resample`.** Sur une série indexée par des dates (un `DatetimeIndex` : il faut convertir les chaînes avec `pd.to_datetime`), `s.resample("M")` découpe le temps en mois civils ; `.last()` donne, pour chaque mois, la **dernière valeur non manquante** ; l'étiquette est le **dernier jour civil** du mois.

```python-exec
p = pd.Series([10.0, 11.0, np.nan, 12.0],
              index=pd.to_datetime(["2026-01-29", "2026-01-30", "2026-01-31", "2026-03-02"]))
p.resample("M").last()
```

Trois faits à lire dans cette sortie : janvier vaut 11, pas `NaN`, car `last` saute la valeur manquante du 31 ; février, sans aucune observation, produit quand même une ligne, `NaN` ; l'étiquette de janvier est le 31 même si la dernière observation est du 30.

**La règle de version.** En pandas 1.4.2, le mois se note `"M"`. Le nom `"ME"`, qu'acceptent des versions plus récentes, n'y existe pas encore :

```python-exec-erreur
p.resample("ME").last()
```

En pandas 2.3.3, `"M"` fonctionne encore mais émet un avertissement, et `"ME"` fonctionne. Dans un environnement en pandas 1.4.2, écrire `"M"`.

**Par actif.** On rééchantillonne à l'intérieur de chaque actif :

```python-exec
q = pd.DataFrame({"date": pd.to_datetime(["2026-01-29", "2026-01-30", "2026-03-02", "2026-01-30"]),
                  "actif": ["A", "A", "A", "B"], "prix": [10.0, 11.0, 12.0, 20.0]})
q.set_index("date").groupby("actif")["prix"].resample("M").last()
```

Le mois de février de A, vide, apparaît en `NaN` : `.dropna()` le retire. B n'a pas de ligne de février : pour chaque actif, resample couvre les mois de sa première à sa dernière observation.

**Le piège des colonnes multiples.** `last` saute les `NaN` **colonne par colonne**. Si l'on garde la date d'observation dans une colonne, elle ne correspond plus au prix retenu :

```python-exec
t = pd.DataFrame({"prix": [10.0, 11.0, np.nan],
                  "jour": pd.to_datetime(["2026-01-29", "2026-01-30", "2026-01-31"])},
                 index=pd.to_datetime(["2026-01-29", "2026-01-30", "2026-01-31"]))
pd.concat({"sans dropna": t.resample("M").last(), "avec dropna": t.dropna(subset=["prix"]).resample("M").last()}, axis=1)
```

Sans `dropna`, le prix est celui du 30 mais le jour affiché est le 31 : retirer les lignes sans prix **avant** de rééchantillonner.

[[controle:resample]]

### 4. merge_asof : la dernière information disponible

**Le problème.** Les signaux sont publiés à des dates irrégulières, week-ends compris ; les décisions sont prises à d'autres dates. Pour chaque décision, il faut la dernière valeur publiée **avant** elle. Une jointure exacte (`merge` sur la date) ne trouve que les publications du jour même : c'est à la fois trop peu et, si la publication a lieu après la décision, du lookahead.

**Ce que fait `pd.merge_asof(gauche, droite, on="date", by="actif")`.** Pour chaque ligne de `gauche`, il cherche dans `droite` les lignes du **même actif** (`by`) dont la date est **inférieure ou égale** (direction par défaut, `"backward"`), et prend la plus récente. Avec `allow_exact_matches=False`, l'inégalité devient **stricte**.

```python-exec
obs = pd.DataFrame({"date": pd.to_datetime(["2026-01-30", "2026-01-30"]), "actif": ["A", "B"]})
sig = pd.DataFrame({"date": pd.to_datetime(["2026-01-28", "2026-01-29", "2026-01-30"]),
                    "actif": ["A", "B", "A"], "signal": [0.5, -1.0, 0.7]})
pd.concat({"par défaut": pd.merge_asof(obs, sig, on="date", by="actif"),
           "strict": pd.merge_asof(obs, sig, on="date", by="actif", allow_exact_matches=False)}, axis=1)
```

Par défaut, A reçoit 0,7, le signal **du jour même**. Si ce signal n'est connu qu'après la clôture du 30 et que la décision est prise à cette clôture, c'est du lookahead ; la version stricte donne 0,5. B reçoit le signal de B, jamais celui de A, grâce à `by`.

Deux contraintes, qui lèvent une erreur au lieu de se tromper en silence. La clé `on` doit être une date ou un nombre, pas une chaîne :

```python-exec-erreur
pd.merge_asof(obs.astype({"date": str}), sig.astype({"date": str}), on="date", by="actif")
```

Et les deux cadres doivent être triés par cette clé, **sur tout le cadre** : trier par actif puis par date ne suffit pas.

```python-exec-erreur
gauche = pd.DataFrame({"date": pd.to_datetime(["2026-01-30", "2026-01-29"]), "actif": ["A", "B"]})
pd.merge_asof(gauche, sig, on="date", by="actif")
```

Le cadre `gauche` est trié par actif puis par date, mais pas par date : refusé.

Dernier réflexe : sélectionner les colonnes du résultat dans l'ordre voulu, `resultat[["actif", "date", ...]]`. En pandas 1.4.2, quand le cadre de gauche est vide, `merge_asof` renvoie ses colonnes dans un autre ordre (vérifié par `verif/lecon_pandas_2.py`).

[[controle:asof]]

### 5. Normalisation transversale et winsorisation

**Le problème.** Un signal n'a pas la même échelle d'une date à l'autre ; on le compare donc aux autres actifs **de la même date**. Deux normalisations : le z-score, qui centre et réduit, et le rang, insensible à l'échelle. Le z-score est sensible aux valeurs extrêmes : on les **winsorise** d'abord, c'est-à-dire qu'on ramène sous un quantile haut et au-dessus d'un quantile bas toutes les valeurs qui les dépassent.

**Le quantile, défini sans ambiguïté.** Pour \(n \ge 1\) valeurs non manquantes d'une date, triées \(x_{(1)} \le \dots \le x_{(n)}\) (des nombres), et un niveau \(q \in [0, 1]\), on pose \(h = (n - 1)\,q\), \(i = \lfloor h \rfloor\) la partie entière de \(h\) et \(f = h - i \in [0, 1[\). Le quantile linéaire est

\(\displaystyle x_{(i+1)} + f\,\bigl(x_{(i+2)} - x_{(i+1)}\bigr)\)

avec la convention qu'il vaut \(x_{(i+1)}\) quand \(f = 0\). C'est une interpolation linéaire entre les valeurs triées placées aux positions \(0, 1, \dots, n - 1\) : elle donne le minimum pour \(q = 0\), le maximum pour \(q = 1\), et varie continûment entre les deux. C'est la méthode par défaut de `Series.quantile` et de `np.quantile` dans les deux versions (vérifié sur des tirages aléatoires par `verif/lecon_pandas_2.py`) ; les autres méthodes (`"lower"`, `"higher"`, `"nearest"`, `"midpoint"`) donnent d'autres valeurs, c'est pourquoi un énoncé doit préciser la sienne.

Exemple : les valeurs 3, 1, 10, 2, triées 1, 2, 3, 10, donc \(n = 4\). Pour \(q = 0{,}1\) : \(h = @L_q10_h:1m@\), \(i = 0\), \(f = @L_q10_h:1m@\), quantile \(1 + @L_q10_h:1m@ \times (2 - 1) = @L_q10:1m@\). Pour \(q = 0{,}9\) : \(h = @L_q90_h:1m@\), \(i = 2\), \(f = @L_q90_f:1m@\), quantile \(3 + @L_q90_f:1m@ \times (10 - 3) = @L_q90:1m@\).

```python-exec
x = pd.DataFrame({"date": ["d1"] * 4 + ["d2"] * 3, "signal": [3.0, 1.0, 10.0, 2.0, 5.0, np.nan, -1.0]})
g = x.groupby("date")["signal"]
bas = x["date"].map(g.quantile(0.1))
haut = x["date"].map(g.quantile(0.9))
x["w"] = x["signal"].clip(lower=bas, upper=haut)
x
```

`g.quantile(0.1)` donne une valeur par date ; `map` la répand sur les lignes de la date ; `clip` accepte des bornes ligne par ligne. Le 10 de `d1` devient @L_q90:1@, le 1 devient @L_q10:1@.

**Le z-score par date** se calcule avec `transform`, qui répand la moyenne et l'écart-type de la date sur ses lignes. L'écart-type de pandas divise par \(n - 1\), celui de numpy par \(n\) :

```python-exec
gw = x.groupby("date")["w"]
x["z"] = (x["w"] - gw.transform("mean")) / gw.transform("std")
x["rang"] = g.rank(pct=True)
x, np.std(x.loc[x["date"] == "d1", "w"]), x.loc[x["date"] == "d1", "w"].std()
```

`rank(pct=True)` divise le rang moyen par le nombre de valeurs **non manquantes** de la date : en `d2`, il y en a 2, pas 3.

**Le piège des valeurs égales.** Si toutes les valeurs d'une date sont égales, le z-score n'est pas défini. Mais l'écart-type **calculé** n'est pas toujours exactement nul :

```python-exec
pd.Series([0.1, 0.1, 0.1]).std()
```

La somme de trois fois 0,1 en flottants n'est pas exactement 0,3, donc la moyenne calculée diffère de 0,1 d'une quantité minuscule, et l'écart-type aussi. Diviser par lui donne des z-scores de l'ordre de 1, absurdes. Tester l'égalité des valeurs, par exemple `transform("nunique") > 1`, et non `std != 0`.

[[controle:quantile]]

### 6. Backtest sans lookahead

**La chronologie.** Notations : \(p_{a,t}\) est le prix observé de l'actif \(a\) à la date \(t\), en unité monétaire ; « \(t+1\) » désigne la date **suivante du calendrier**, c'est-à-dire de l'ensemble des dates du panel, pas la ligne suivante de l'actif. À la clôture de \(t\), on connaît les prix jusqu'à \(t\) et les signaux de \(t\) ; on fixe des poids \(w_{a,t}\) (des nombres sans unité, positifs pour un achat, négatifs pour une vente à découvert) ; la position rapporte entre \(t\) et \(t+1\) le rendement \(r_{a,t} = p_{a,t+1}/p_{a,t} - 1\). Le rendement du portefeuille sur cette période est \(\sum_a w_{a,t}\, r_{a,t}\).

Le rendement \(r_{a,t}\) est indexé par la date de **décision** \(t\), mais il n'est connu qu'en \(t+1\) : il sert à **noter** la décision, jamais à la **prendre**.

**Le format large** aligne le calendrier : une ligne par date, une colonne par actif, `NaN` là où l'actif n'a pas de ligne. Alors `shift(-1)` va chercher la date suivante du calendrier :

```python-exec
b = pd.DataFrame({"date": ["d1", "d1", "d2", "d3", "d3"], "actif": ["A", "B", "A", "A", "B"],
                  "prix": [100.0, 50.0, 110.0, 99.0, 55.0]})
prix = b.pivot(index="date", columns="actif", values="prix")
prix, prix.shift(-1) / prix - 1
```

B n'a pas de ligne en `d2` : son rendement de `d1` à `d2` est `NaN`, et non le rendement de `d1` à `d3`, que donnerait un `groupby("actif")["prix"].shift(-1)` sur le format long. Un énoncé doit dire quoi faire de ce `NaN` (ici, dans l'exercice 4 : compter 0).

**Les trois lookaheads classiques**, tous silencieux :

1. ne garder en \(t\) que les actifs dont le prix de \(t+1\) est connu : la sélection utilise le futur (on élimine d'avance les actifs qui vont disparaître) ;
2. normaliser le signal avec une moyenne ou un écart-type calculés sur **toutes** les dates : la normalisation de \(t\) contient les dates postérieures ;
3. associer à la décision de \(t\) une information publiée en \(t\) après la décision (section 4, `allow_exact_matches`).

L'erreur inverse existe aussi : ranger à la ligne \(t\) le rendement de \(t-1\) à \(t\) (`pct_change()` sans décalage) mesure ce qui s'est passé **avant** la décision.

**Classer en départageant les égalités.** `rank(axis=1, method="first")` classe les actifs de chaque date, et départage les égalités par position de colonne ; `pivot` trie les colonnes, donc l'égalité est départagée par nom d'actif croissant :

```python-exec
pd.DataFrame({"A": [0.2, 1.0], "B": [0.2, np.nan], "C": [0.1, 1.0]}).rank(axis=1, method="first")
```

**Le ratio de Sharpe annualisé.** Pour une série de \(m\) rendements quotidiens de moyenne empirique \(\bar r\) et d'écart-type empirique \(s\) (dénominateur \(m - 1\)), on le définit par \(\sqrt{252}\, \bar r / s\), où 252 est un nombre conventionnel de jours de bourse par an. D'où vient la racine :

1. Modèle : les rendements quotidiens \(R_1, \dots, R_{252}\) d'une année sont des variables aléatoires de même espérance \(\mu\) et de même variance \(\sigma^2 > 0\), **deux à deux non corrélées**, et l'on approche le rendement annuel par leur somme (approximation valable pour de petits rendements, qui néglige la composition).
2. Linéarité de l'espérance : \(\mathbb{E}\bigl[\sum_{j=1}^{252} R_j\bigr] = 252\,\mu\).
3. Variance d'une somme : \(\mathrm{Var}\bigl(\sum_j R_j\bigr) = \sum_j \mathrm{Var}(R_j) + \sum_{j \ne l} \mathrm{Cov}(R_j, R_l) = 252\,\sigma^2\), les covariances étant nulles par hypothèse.
4. Sharpe annuel : \(252\,\mu / \sqrt{252\,\sigma^2} = \sqrt{252}\,\mu/\sigma\), estimé par \(\sqrt{252}\,\bar r/s\).

Si les rendements sont autocorrélés, l'étape 3 garde les covariances et le facteur \(\sqrt{252}\) est faux. Multiplier par 252, ou utiliser l'écart-type à dénominateur \(m\), sont deux erreurs que les tests cachés de l'exercice 4 attrapent. Les dates sans position comptent, avec un rendement nul : les retirer gonfle le Sharpe.

Un ordre de grandeur pour se méfier : le fichier `panel.csv` du notebook contient un signal synthétique fabriqué avec une part du rendement du lendemain, c'est-à-dire un lookahead volontaire. Le backtest de l'exercice 4 y donne un Sharpe annualisé de @E4_sharpe_panel:1@, valeur sans rapport avec ce qu'atteint une stratégie réelle : un Sharpe aussi élevé est d'abord le symptôme d'une fuite.

[[controle:lookahead]]

### 7. Pièges, réflexes et ce qu'il faut savoir refaire

Pièges :

1. `rolling`, `shift`, `pct_change` dépendent de l'ordre des lignes : trier par date, et grouper par actif.
2. `groupby(...).rolling(...)` a un index à deux niveaux : `reset_index(level=0, drop=True)` ou `transform`.
3. `min_periods` compte les valeurs non manquantes ; sans lui, une seule valeur manquante efface la fenêtre.
4. `resample("M")` en pandas 1.4.2 ; il crée des mois vides en `NaN` ; `last` saute les `NaN` colonne par colonne.
5. `merge_asof` : clés en dates, cadres triés par la clé, `by` pour l'actif, `allow_exact_matches=False` pour « strictement avant ».
6. Quantiles, z-scores et rangs **par date** ; écart-type de pandas à dénominateur \(n - 1\) ; égalité des valeurs testée par `nunique`.
7. Backtest : décision avec l'information de \(t\), rendement de \(t\) à \(t+1\) du calendrier, aucune sélection sur le futur.

Réflexes en 30 secondes :

| Besoin | Écriture |
|---|---|
| moyenne glissante par actif | `df.sort_values("date").groupby("actif")["x"].transform(lambda s: s.rolling(n, min_periods=m).mean())` |
| dernier prix du mois par actif | `p.dropna(subset=["prix"]).set_index("date").groupby("actif")["prix"].resample("M").last().dropna()` |
| dernière information strictement antérieure | `pd.merge_asof(g.sort_values("date"), d.sort_values("date"), on="date", by="actif", allow_exact_matches=False)` |
| quantile par date sur chaque ligne | `df["date"].map(df.groupby("date")["x"].quantile(q))` |
| z-score par date | `(x - g.transform("mean")) / g.transform("std")` |
| rang par date dans \(]0, 1]\) | `df.groupby("date")["x"].rank(pct=True)` |
| rendement de \(t\) à \(t+1\) du calendrier | `prix_large.shift(-1) / prix_large - 1` |
| Sharpe annualisé | `np.sqrt(252) * r.mean() / r.std()` |

À savoir refaire sans notes, en moins de 5 minutes chacun : la moyenne glissante par actif, le dernier prix du mois avec le dernier signal strictement antérieur, la winsorisation et le z-score par date, et la démonstration du facteur \(\sqrt{252}\) avec son hypothèse.
""",
    "controles": {
        "fuite": {
            "question": "Avec le cadre `r` de la section 2, on écrit `r.sort_values([\"date\", \"actif\"])[\"rendement\"].rolling(2).mean()`. Que vaut le résultat sur la ligne de B au 2026-01-05 ?",
            "options": [
                {"id": "a", "texte": "@L_fuite:1@ : la moyenne de A et de B le 2026-01-05, une fuite entre actifs.",
                 "explication": "Oui : sans groupby, la fenêtre de deux lignes contient la ligne précédente du cadre trié, qui est A le même jour. Les deux valeurs sont non manquantes, donc min_periods = 2 est atteint."},
                {"id": "b", "texte": "`NaN`, car B n'a qu'une valeur à cette date.",
                 "explication": "Ce serait le résultat juste avec `groupby(\"actif\")` : la fenêtre de B ne contiendrait que B. Sans groupby, la fenêtre ne connaît pas les actifs."},
                {"id": "c", "texte": "10, le rendement de B lui-même.",
                 "explication": "Ce serait le cas avec `groupby` et `min_periods=1`. Ici, ni l'un ni l'autre : la fenêtre prend deux lignes du cadre, tous actifs confondus."},
                {"id": "d", "texte": "Une erreur, car le cadre contient deux actifs.",
                 "explication": "`rolling` ne vérifie rien sur le contenu des autres colonnes : il calcule sur la série dans l'ordre des lignes. C'est ce qui rend la fuite silencieuse."},
            ],
        },
        "resample": {
            "question": "Une série a des observations les 2026-01-15, 2026-01-20 et 2026-04-02 seulement. Combien de lignes renvoie `s.resample(\"M\").last()` sous pandas 1.4.2 ?",
            "options": [
                {"id": "a", "texte": "@L_resample_lignes@ : janvier, février, mars et avril, avec `NaN` pour février et mars.",
                 "explication": "Oui : resample crée tous les mois entre le premier et le dernier mois observés, vides compris. Il faut `dropna()` pour ne garder que les mois observés."},
                {"id": "b", "texte": "2 : seuls les mois observés.",
                 "explication": "C'est le résultat après `dropna()`, ou celui d'un `groupby` sur le mois. `resample` remplit le calendrier."},
                {"id": "c", "texte": "3 : une ligne par observation.",
                 "explication": "resample produit une ligne par période, pas par observation : les deux observations de janvier sont réduites en une seule par `last`."},
                {"id": "d", "texte": "Une erreur : le mois se note `\"ME\"`.",
                 "explication": "C'est l'inverse : sous pandas 1.4.2, `\"ME\"` lève une erreur et `\"M\"` fonctionne. `\"ME\"` n'est accepté que par des versions plus récentes, dont la 2.3.3."},
            ],
        },
        "asof": {
            "question": "Décision à la clôture du 2026-01-30 sur l'actif A. Signaux de A datés du 2026-01-28 (0,5) et du 2026-01-30 (0,7), chacun publié après la clôture de son jour. Quelle écriture associe à la décision le dernier signal réellement disponible ?",
            "options": [
                {"id": "a", "texte": "`pd.merge_asof(obs, sig, on=\"date\", by=\"actif\", allow_exact_matches=False)`, qui donne 0,5.",
                 "explication": "Oui : le signal du 30 est publié après la décision, il faut donc une date strictement antérieure. La version stricte écarte l'égalité."},
                {"id": "b", "texte": "`pd.merge_asof(obs, sig, on=\"date\", by=\"actif\")`, qui donne 0,7.",
                 "explication": "Par défaut, merge_asof accepte une date égale : il prend le signal du 30, publié après la décision. C'est du lookahead."},
                {"id": "c", "texte": "`pd.merge_asof(obs, sig, on=\"date\", by=\"actif\", direction=\"forward\")`.",
                 "explication": "`\"forward\"` cherche la première date supérieure ou égale : il regarde vers le futur par construction, et donne ici 0,7."},
                {"id": "d", "texte": "`pd.merge(obs, sig, on=[\"date\", \"actif\"])`.",
                 "explication": "Une jointure exacte ne garde que le signal du jour même (0,7, publié trop tard) et ne trouve rien pour une décision prise un jour sans publication."},
            ],
        },
        "quantile": {
            "question": r"Une date a les signaux 3, 1, 10, 2. Que vaut leur quantile de niveau 0,9 selon la définition de la section 5 ?",
            "options": [
                {"id": "a", "texte": "@L_q90:1@",
                 "explication": r"Oui : \(h = (4 - 1) \times 0{,}9 = @L_q90_h:1m@\), donc \(i = 2\) et \(f = @L_q90_f:1m@\) ; le quantile vaut \(x_{(3)} + @L_q90_f:1m@\,(x_{(4)} - x_{(3)}) = 3 + @L_q90_f:1m@ \times 7\)."},
                {"id": "b", "texte": "10",
                 "explication": r"C'est la méthode `'higher'` (ou `'nearest'` ici) : on prend une valeur observée au lieu d'interpoler. La définition linéaire interpole entre 3 et 10."},
                {"id": "c", "texte": "3",
                 "explication": r"C'est la méthode `'lower'` : la valeur triée de position \(\lfloor h \rfloor\). Elle ignore la partie fractionnaire \(f\)."},
                {"id": "d", "texte": "@L_q90_bis:1@",
                 "explication": r"C'est le résultat avec la position \(h = n\,q - 1 = @L_q90_bis_h:1m@\) au lieu de \((n - 1)\,q\) : une autre convention de quantile empirique, qui n'est pas celle de la définition."},
            ],
        },
        "lookahead": {
            "question": r"Dans un backtest où la décision est prise à la clôture de \(t\), lequel de ces choix utilise une information postérieure à la décision ?",
            "options": [
                {"id": "a", "texte": r"Ne retenir en \(t\) que les actifs dont le prix de \(t+1\) est connu.",
                 "explication": r"Oui : savoir en \(t\) qu'un actif aura un prix en \(t+1\) est une information future. Elle écarte d'avance les actifs qui vont disparaître ou être suspendus."},
                {"id": "b", "texte": r"Noter la position par le rendement `prix.shift(-1) / prix - 1` rangé à la ligne \(t\).",
                 "explication": r"C'est la définition du résultat de la décision : ce rendement n'est connu qu'en \(t+1\), mais il ne sert qu'à évaluer la position, pas à la choisir."},
                {"id": "c", "texte": r"Classer les actifs par le signal daté de \(t\), connu avant la clôture de \(t\).",
                 "explication": r"Le signal est disponible au moment de la décision : c'est exactement l'information autorisée."},
                {"id": "d", "texte": r"Compter un rendement nul pour un actif en position dont le prix de \(t+1\) manque.",
                 "explication": r"C'est une convention sur le résultat, fixée à l'avance, qui ne change pas la décision prise en \(t\). Elle doit seulement être annoncée."},
            ],
        },
    },
    "cellules_initiales": [
        "import numpy as np\nimport pandas as pd\n\nr = pd.DataFrame({\"date\": [\"2026-01-05\", \"2026-01-06\", \"2026-01-07\"] * 2,\n                  \"actif\": [\"A\"] * 3 + [\"B\"] * 3,\n                  \"rendement\": [1.0, np.nan, 3.0, 10.0, 20.0, 30.0]})\nr",
        "# Essaie ici les exemples de la leçon, en les modifiant.\nd = r.sort_values([\"date\", \"actif\"])\nd.groupby(\"actif\")[\"rendement\"].rolling(2, min_periods=1).mean()",
        "# Fichier d'exploration : panel synthétique (date, actif, prix, signal).\npanel = pd.read_csv(\"panel.csv\")\npanel.head()",
    ],
    "verif": {"script": "verif/lecon_pandas_2.py"},
}
