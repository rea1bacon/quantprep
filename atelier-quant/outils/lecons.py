"""Leçons placées avant les exercices des séances.

Une leçon est un élément de type "lecon" :
- "contenu" : Markdown réduit (titres, tableaux, listes, formules entre \\( \\) ou \\[ \\]) ;
- les blocs ```python-exec sont exécutés à la construction, dans l'ordre, par le noyau du notebook
  (même interpréteur que les exercices : .venv-hackerrank), et leur vraie sortie est insérée ;
  un bloc ```python-exec-erreur doit lever une exception, dont le message est inséré ;
- une ligne [[controle:IDENT]] insère le point de contrôle IDENT de "controles" : question corrigée
  immédiatement, non notée ; l'option "a" est la bonne, chaque option a son explication.
"""

LECON_PANDAS = {
    "id": "L-PANDAS-1", "type": "lecon", "section": "lecon", "theme": "Leçon pandas : calculer par groupe dans un panel",
    "difficulte": 2, "temps_cible_s": 1200, "dossier": "coding/diagnostic",
    "contenu": r"""## Leçon : calculer par groupe dans un panel

Durée visée : 20 minutes. Lis dans l'ordre, exécute les exemples dans le notebook de droite si tu veux les modifier, et réponds à chaque point de contrôle avant de continuer.

### 1. Le problème

Les données de marché arrivent presque toujours en **panel** : une ligne par couple (date, actif). Les questions classiques portent sur des calculs **par groupe** : le rendement futur de chaque actif, le rang d'un signal parmi les actifs d'une même date, une corrélation par date, puis une moyenne de ces corrélations.

Deux difficultés rendent ces calculs faux sans message d'erreur :

1. Les lignes arrivent dans un ordre quelconque, et certaines opérations dépendent de l'ordre des lignes.
2. pandas aligne les calculs sur les **étiquettes** d'index, pas sur les positions : c'est une aide quand on le sait, un piège sinon.

Toute la leçon tient en deux modèles mentaux, puis trois gestes.

Le panel d'exemple, utilisé partout dans la leçon :

```python-exec
import numpy as np
import pandas as pd

df = pd.DataFrame(
    [("2026-01-06", "A", 110.0, 0.3), ("2026-01-05", "A", 100.0, 0.2), ("2026-01-05", "B", 50.0, -1.0),
     ("2026-01-07", "A", 99.0, 0.1), ("2026-01-07", "B", 55.0, 0.3), ("2026-01-06", "B", 52.0, 0.3)],
    columns=["date", "actif", "prix", "signal"])
df
```

### 2. Modèle mental 1 : l'index est une étiquette, pas une position

Chaque ligne porte une étiquette d'index (ici 0 à 5). Trier ne change pas les étiquettes : la ligne d'étiquette 1 reste la ligne d'étiquette 1, elle change seulement de place.

```python-exec
trie = df.sort_values(["actif", "date"])
trie
```

Quand on combine deux séries, pandas apparie les valeurs **de même étiquette**, quelle que soit leur position :

```python-exec
trie["prix"] - df["prix"]
```

Toutes les différences sont nulles : la ligne 1 de `trie` a été appariée à la ligne 1 de `df`, même si elles ne sont pas à la même position. Conséquence pratique : une série calculée sur un cadre trié peut être rangée dans le cadre d'origine par simple affectation, `df["nouvelle"] = serie`, et chaque valeur retrouve sa ligne.

Pour désigner une **position**, on utilise `.iloc` ; pour une **étiquette**, `.loc`.

```python-exec
trie["prix"].iloc[0], trie["prix"].loc[0]
```

`.iloc[0]` est la première ligne du cadre trié (étiquette 1, prix 100) ; `.loc[0]` est la ligne d'étiquette 0 (prix 110), où qu'elle soit.

[[controle:alignement]]

### 3. Modèle mental 2 : groupby découpe, applique, recombine

`df.groupby("actif")` découpe le cadre en un sous-cadre par actif. Ce qui compte ensuite est **la forme du résultat**, qui dépend de l'opération appliquée :

| Opération | Exemples | Forme du résultat |
|---|---|---|
| réduction | `mean`, `sum`, `last`, `agg` | une ligne **par groupe**, indexée par les clés du groupe |
| transformation | `shift`, `rank`, `diff`, `cumsum`, `pct_change`, `transform` | une valeur **par ligne**, avec l'index d'origine |
| application libre | `apply(f)` | dépend de ce que renvoie `f` ; si `f` renvoie un nombre, une valeur par groupe |

```python-exec
df.groupby("actif")["prix"].mean()
```

```python-exec
df.groupby("actif")["prix"].rank()
```

La moyenne a 2 lignes (une par actif) ; le rang en a 6, avec les étiquettes de `df`. Une transformation peut donc être rangée directement dans `df` ; une réduction, non.

Pour répandre une réduction sur chaque ligne de son groupe, on utilise `transform` :

```python-exec
df.groupby("actif")["prix"].transform("mean")
```

[[controle:formes]]

### 4. Geste 1 : décaler à l'intérieur d'un groupe, sur un cadre trié

`shift(-1)` dans un groupe donne la valeur de la **ligne suivante du groupe dans l'ordre où les lignes se trouvent**. Ce n'est la ligne suivante dans le temps que si le cadre est trié par date.

Sur le cadre non trié, le résultat est faux sans aucun message :

```python-exec
faux = df.groupby("actif")["prix"].shift(-1)
juste = df.sort_values("date").groupby("actif")["prix"].shift(-1)
pd.DataFrame({"date": df["date"], "actif": df["actif"], "prix": df["prix"], "suivant_faux": faux, "suivant_juste": juste})
```

Lis la ligne 0 (A, 2026-01-06) : le « suivant » faux vaut 100, le prix de A **le 2026-01-05**, parce que dans `df` la ligne de A au 2026-01-05 vient juste après. Le suivant juste vaut 99, le prix de A le 2026-01-07.

Deux remarques sur la version juste :

- trier par date suffit : `groupby` conserve, à l'intérieur de chaque groupe, l'ordre des lignes du cadre ;
- le résultat a les étiquettes de `df`, rangées dans l'ordre du tri. L'affectation `df["suivant"] = juste` remet chaque valeur à sa ligne, par alignement (section 2).

Le rendement futur s'en déduit, toujours par alignement :

```python-exec
d = df.sort_values("date")
df["rendement_futur"] = d.groupby("actif")["prix"].shift(-1) / d["prix"] - 1
df
```

La dernière date de chaque actif n'a pas de suivante : `NaN`, ce qui est voulu.

[[controle:decalage]]

### 5. Geste 2 : rang par groupe

`rank()` classe par ordre croissant à partir de 1 ; par défaut (`method="average"`), des valeurs égales reçoivent la moyenne des rangs qu'elles occupent, et les `NaN` restent `NaN`. Appliqué après `groupby("date")`, il classe les actifs **à l'intérieur de chaque date** :

```python-exec
df.groupby("date")["signal"].rank()
```

Le 2026-01-06, A et B ont le même signal 0,3 : ils occupent les rangs 1 et 2 et reçoivent chacun 1,5.

Le double `argsort` ne fait pas cela : il départage les égalités arbitrairement.

```python-exec
x = np.array([3.0, 1.0, 3.0])
np.argsort(np.argsort(x)) + 1, pd.Series(x).rank().to_numpy()
```

Variantes utiles : `rank(pct=True)` donne des rangs divisés par le nombre de valeurs non manquantes ; `rank(method="first")` départage par ordre d'apparition.

[[controle:rang]]

### 6. Geste 3 : une valeur par groupe calculée sur plusieurs colonnes

Une corrélation par date a besoin de deux colonnes à la fois. On sélectionne les colonnes utiles, puis `apply` reçoit pour chaque date le sous-cadre correspondant :

```python-exec
def correlation(g):
    return g["signal"].corr(g["rendement_futur"])

df.groupby("date")[["signal", "rendement_futur"]].apply(correlation)
```

Lecture : le 2026-01-05, la corrélation porte sur deux actifs seulement ; avec deux points, une corrélation vaut toujours 1 ou -1 (ici 1, à l'arrondi flottant près), ce qui ne prouve rien ; le 2026-01-06, les signaux sont égaux, la corrélation n'est pas définie ; le 2026-01-07 n'a aucun rendement futur.

Ce que fait `Series.corr`, vérifié sous pandas 1.4.2 :

- il ignore les paires où l'une des deux valeurs manque ;
- il renvoie `NaN` si l'une des deux séries est constante ;
- avec `method="spearman"`, il calcule la corrélation des rangs moyens, ce qui est exactement le coefficient d'information.

Pourquoi sélectionner les colonnes avant `apply` : en pandas 1.4.2, `df.groupby("date").apply(f)` passe aussi la colonne `date` à `f` ; en pandas 2.2, la même écriture émet un avertissement. La sélection `[["signal", "rendement_futur"]]` fonctionne sans surprise dans les deux versions.

[[controle:apply]]

### 7. Assemblage : le coefficient d'information moyen

Les trois gestes mis bout à bout, sur le fichier d'exploration du test de positionnement :

```python-exec
p = pd.read_csv("prix_signaux.csv")
d = p.sort_values("date")
p["r"] = d.groupby("actif")["prix"].shift(-1) / d["prix"] - 1

def ic(g):
    g = g.dropna()
    if len(g) < 3 or g["signal"].nunique() == 1 or g["r"].nunique() == 1:
        return np.nan
    return g["signal"].corr(g["r"], method="spearman")

ics = p.groupby("date")[["signal", "r"]].apply(ic)
ics.head(), ics.mean()
```

`ics.mean()` ignore les `NaN` : c'est la moyenne sur les dates non ignorées, comme le demande l'énoncé du test de positionnement.

### 8. Pièges, et ce qu'il faut savoir refaire

Pièges :

1. `shift`, `diff`, `pct_change`, `cumsum`, `rolling` dépendent de l'ordre des lignes : **trier avant**.
2. Une réduction (`mean`) n'a pas l'index d'origine : utiliser `transform` pour la répandre sur les lignes.
3. `dropna()` sans argument supprime une ligne dès qu'**une** colonne manque : préciser `subset=[...]` quand d'autres colonnes peuvent manquer.
4. Les rangs : `rank()`, jamais un double `argsort`, dès qu'il peut y avoir des égalités.
5. `apply` : sélectionner les colonnes avant, et renvoyer un nombre pour obtenir une valeur par groupe.
6. Soumettre dès que les tests visibles passent, améliorer ensuite.

À savoir refaire sans notes, en moins de 5 minutes : le rendement futur par actif, le rang par date, la corrélation de rang par date et sa moyenne.
""",
    "controles": {
        "alignement": {
            "question": r'On exécute `d = df.sort_values("prix")` puis `s = d["prix"] - df["prix"]`. Que contient `s` ?',
            "options": [
                {"id": "a", "texte": "Six zéros, avec les étiquettes 0 à 5.",
                 "explication": "Oui : la soustraction apparie les lignes de même étiquette, et chaque ligne de `d` est la même ligne de `df`, déplacée."},
                {"id": "b", "texte": "Les différences entre prix voisins, car `d` est trié.",
                 "explication": "Ce serait vrai si pandas appariait par position, comme numpy. pandas apparie par étiquette d'index."},
                {"id": "c", "texte": "Une erreur, car les deux séries ne sont pas dans le même ordre.",
                 "explication": "pandas ne demande pas que les séries soient dans le même ordre : il les réaligne par étiquette avant de calculer."},
                {"id": "d", "texte": "Six `NaN`, car les étiquettes ne correspondent plus.",
                 "explication": "Les étiquettes correspondent : trier ne les modifie pas. On obtient des `NaN` pour les étiquettes présentes dans une seule des deux séries, par exemple si l'une est un sous-ensemble filtré de l'autre."},
            ],
        },
        "formes": {
            "question": "Laquelle de ces expressions renvoie une valeur par ligne de `df`, avec les étiquettes de `df` ?",
            "options": [
                {"id": "a", "texte": "`df.groupby(\"date\")[\"signal\"].transform(\"max\")`",
                 "explication": "Oui : `transform` répand le maximum de chaque date sur toutes les lignes de cette date."},
                {"id": "b", "texte": "`df.groupby(\"date\")[\"signal\"].max()`",
                 "explication": "Une réduction : une ligne par date, indexée par la date."},
                {"id": "c", "texte": "`df.groupby(\"date\")[\"signal\"].agg([\"max\", \"min\"])`",
                 "explication": "Une réduction à deux colonnes : toujours une ligne par date."},
                {"id": "d", "texte": "`df.groupby(\"date\")[[\"signal\", \"prix\"]].apply(lambda g: g[\"signal\"].max())`",
                 "explication": "La fonction renvoie un nombre par groupe : le résultat a une valeur par date."},
            ],
        },
        "decalage": {
            "question": "On écrit `trie = df.sort_values([\"date\", \"actif\"])` puis `trie[\"nxt\"] = df.groupby(\"actif\")[\"prix\"].shift(-1)`. Que contient `trie[\"nxt\"]` ?",
            "options": [
                {"id": "a", "texte": "Pour chaque ligne, le prix de la ligne suivante du même actif dans l'ordre d'origine de `df`, qui n'a pas de sens temporel.",
                 "explication": "Oui : le décalage est calculé sur `df`, non trié. L'affectation range ensuite chaque valeur à sa ligne par étiquette, mais la valeur elle-même est déjà fausse."},
                {"id": "b", "texte": "Le prix du même actif à la date suivante, car `trie` est trié par date.",
                 "explication": "Le tri de `trie` ne change rien au calcul, qui est fait sur `df`. Il aurait fallu `trie.groupby(\"actif\")[\"prix\"].shift(-1)`."},
                {"id": "c", "texte": "Des `NaN` partout, car les index de `trie` et `df` sont dans des ordres différents.",
                 "explication": "L'alignement par étiquette fonctionne quel que soit l'ordre : les valeurs sont bien rangées, mais ce sont de mauvaises valeurs."},
                {"id": "d", "texte": "Le prix de la ligne suivante de `trie`, tous actifs confondus.",
                 "explication": "Le `groupby(\"actif\")` limite le décalage à chaque actif ; et il porte sur l'ordre de `df`, pas de `trie`."},
            ],
        },
        "rang": {
            "question": "Pour une date, les signaux valent 2, 5, 2, 2. Quels rangs donne `rank()` ?",
            "options": [
                {"id": "a", "texte": "2, 4, 2, 2",
                 "explication": "Oui : les trois valeurs égales à 2 occupent les rangs 1, 2 et 3, de moyenne 2 ; la valeur 5 a le rang 4."},
                {"id": "b", "texte": "1, 4, 2, 3",
                 "explication": "C'est `method=\"first\"` : les égalités sont départagées par ordre d'apparition. La méthode par défaut est la moyenne."},
                {"id": "c", "texte": "1, 2, 1, 1",
                 "explication": "C'est `method=\"dense\"` : les valeurs distinctes sont numérotées sans trou. Ce n'est pas la méthode par défaut."},
                {"id": "d", "texte": "1, 4, 1, 1",
                 "explication": "C'est `method=\"min\"` : chaque groupe d'égalités reçoit le plus petit rang qu'il occupe."},
            ],
        },
        "apply": {
            "question": "Sous pandas 1.4.2, on écrit `df.groupby(\"date\").apply(f)`, où `f(g)` calcule `g.mean().mean()` sur toutes les colonnes de `g`. Quel est le risque ?",
            "options": [
                {"id": "a", "texte": "`g` contient aussi la colonne `date` (et `actif`), qui ne sont pas numériques : le calcul échoue ou les ignore selon la version.",
                 "explication": "Oui : `apply` sur un groupby sans sélection passe toutes les colonnes, clé de groupe comprise. Sélectionner `[[...]]` les colonnes utiles évite le problème dans toutes les versions."},
                {"id": "b", "texte": "Aucun : `apply` ne passe jamais la colonne de regroupement.",
                 "explication": "C'est le comportement voulu à partir de pandas 2.2 avec `include_groups=False`, argument qui n'existe pas en pandas 1.4.2."},
                {"id": "c", "texte": "`apply` renvoie une ligne par ligne de `df`, pas une valeur par date.",
                 "explication": "Si `f` renvoie un nombre, `apply` renvoie une valeur par groupe."},
                {"id": "d", "texte": "`apply` modifie `df` en place.",
                 "explication": "`apply` ne modifie pas le cadre d'origine, sauf si `f` le modifie explicitement, ce qui est à éviter."},
            ],
        },
    },
    "cellules_initiales": [
        "import numpy as np\nimport pandas as pd\n\ndf = pd.DataFrame(\n    [(\"2026-01-06\", \"A\", 110.0, 0.3), (\"2026-01-05\", \"A\", 100.0, 0.2), (\"2026-01-05\", \"B\", 50.0, -1.0),\n     (\"2026-01-07\", \"A\", 99.0, 0.1), (\"2026-01-07\", \"B\", 55.0, 0.3), (\"2026-01-06\", \"B\", 52.0, 0.3)],\n    columns=[\"date\", \"actif\", \"prix\", \"signal\"])\ndf",
        "# Essaie ici les exemples de la leçon, en les modifiant.\ndf.groupby(\"actif\")[\"prix\"].shift(-1)",
    ],
    "verif": {"script": "verif/lecon_pandas.py"},
}
