"""Séance de market making 5 : nettoyer un export de transactions.

Leçon et exercices repris du matériel bonus de préparation Millennium. Valeurs : verif/seance_mm_5.py.
"""
import commun_mm as cm
import questions_diagnostic as diag

ID = "seance-mm-5"
TITRE = "Market making 5 : nettoyer un export de transactions"
VERIF = "verif/seance_mm_5.out.json"
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [{"id": "lecon", "nom": "Leçon", "budget_min": 15}, {"id": "code", "nom": "Exercices de code", "budget_min": 35}]
PRESENTATION = (
    "Une leçon de 15 minutes sur la décision de réparer ou de retirer une donnée, et deux exercices de code évalués sur un fichier "
    "sale réel du dossier de la question.\n\n" + cm.ABANDON
)
AVERTISSEMENTS = [cm.AVERTISSEMENT_VERSIONS, "Le fichier est synthétique : ses défauts ont été ajoutés à une série propre, ce qui rend le bon résultat vérifiable."]

CONTENU = r"""## Leçon : réparer ou retirer

Durée visée : 15 minutes.

### 1. Le problème

#### Le problème

Un fichier exporté d'un système contient des défauts de forme et des données fausses. Les deux ne se traitent pas de la même façon, et une erreur d'analyse ne provoque pas forcément d'erreur Python : elle produit un nombre faux qui a l'air correct.

#### Intuition

Une question décide de tout : **la valeur vraie est-elle encore lisible ?** Oui pour `1,446.57` précédé du symbole dollar, écrit autrement mais sans perte : on répare. Non pour une cellule vide ou un volume de -120 : on ne peut pas deviner, on retire la ligne.

### 2. Le fichier

#### Exemple

`data_cache/messy_trading_data.csv` contient @N_brut@ lignes pour trois titres. Ses défauts :

- noms de colonnes avec espaces (` Timestamp`, `Ticker `) ;
- symboles en casse variable, avec espaces (`aapl`, ` GOOG `) ;
- prix écrits simplement, avec le symbole dollar, ou avec un séparateur de milliers ;
- prix absents, en cellule vide ou sous la forme `N/A` ;
- volumes écrits simplement, avec un séparateur de milliers, ou absents ;
- volumes négatifs ;
- lignes exactement dupliquées.

Le décompte de ce qui est retiré, dans l'ordre :

| Étape | Lignes concernées |
|---|---|
| Prix absent | @N_sans_prix@ |
| Volume absent | @N_sans_vol@ |
| Volume nul ou négatif | @N_neg@ |
| Il reste après ces filtres (une ligne peut cumuler les défauts) | @N_filtre@ |
| Doublons exacts retirés | @N_dup@ |
| Lignes finales | @N_final@ |

#### L'ordre importe

1. Normaliser les noms de colonnes avant de les utiliser.
2. Normaliser le symbole et les nombres **avant** de chercher les doublons : deux lignes de même contenu peuvent différer par leur écriture.
3. Convertir l'horodatage, trier par titre puis par temps : les rendements en dépendent.

#### Pièges

- `pd.to_numeric(..., errors="coerce")` transforme tout ce qu'il ne lit pas en `NaN`, sans message : il faut retirer les symboles **avant**, sinon tous les prix en milliers disparaissent.
- Lire un prix jusqu'à la première virgule donne une valeur fausse pour les prix en milliers : pour GOOG, toutes les valeurs deviendraient 1, et la volatilité mesurée \(@VOL_GOOG_faux:1m@\).
- `pct_change` sur tout le cadre calcule un rendement entre deux titres différents à chaque changement de symbole : le calcul doit se faire à l'intérieur de chaque titre.
- Une ligne retirée laisse un trou dans la grille de 5 minutes : sur les @N_grille@ lignes de la grille d'origine, @N_final@ restent, et certains rendements portent donc sur 10 minutes ou plus. L'écart-type calculé n'est pas celui d'un pas constant.

[[controle:quoi]]

### 3. Le contrôle de bout en bout

#### Intuition

Un nombre calculé sur toutes les lignes, ici la volatilité par titre, attrape les erreurs d'analyse qu'un contrôle de type ne voit pas. Valeurs attendues de l'écart-type du rendement par ligne : \(@VOL_AAPL:6m@\) pour AAPL, \(@VOL_GOOG:6m@\) pour GOOG, \(@VOL_MSFT:6m@\) pour MSFT.

#### Réflexes en 30 secondes

- Valeur vraie encore lisible : réparer. Sinon : retirer.
- Normaliser avant de dédupliquer.
- Trier avant tout calcul de rendement.
- Tout groupe calculé reste à l'intérieur de son groupe.

#### À savoir refaire

- La chaîne complète, de `read_csv` au tri final, en vingt lignes.
- Le décompte des lignes retirées, étape par étape.

[[controle:ordre]]
"""

CONTROLES = {
    "quoi": {
        "question": r"Une ligne a pour volume -250. Que faire ?",
        "options": [
            {"id": "a", "texte": "Retirer la ligne", "explication": r"Oui : rien ne dit que la vraie valeur était 250 ; la donnée est fausse, pas mal écrite."},
            {"id": "b", "texte": "Prendre la valeur absolue", "explication": r"Cette réparation invente une valeur : elle suppose une erreur de signe que rien dans le fichier n'établit."},
            {"id": "c", "texte": "Propager le volume précédent", "explication": r"Un volume n'est pas un niveau qui persiste : la propagation convient à un prix de relevé, pas à une transaction."},
            {"id": "d", "texte": "Remplacer par zéro", "explication": r"Une ligne de volume nul reste une ligne sans transaction : on crée un enregistrement faux au lieu de le retirer."},
        ],
    },
    "ordre": {
        "question": r"Pourquoi dédupliquer après avoir normalisé le symbole ?",
        "options": [
            {"id": "a", "texte": "Deux lignes égales peuvent s'écrire différemment", "explication": r"Oui : `aapl` et `AAPL` ne sont des doublons qu'une fois normalisés."},
            {"id": "b", "texte": "drop_duplicates exige des majuscules", "explication": r"La méthode compare des valeurs quelconques : le problème vient du contenu du fichier, pas de la méthode."},
            {"id": "c", "texte": "Pour gagner du temps de calcul", "explication": r"L'ordre change le résultat, pas seulement la durée : retirer trop tard ou trop tôt ne garde pas les mêmes lignes."},
            {"id": "d", "texte": "L'ordre n'a aucune importance", "explication": r"Avant normalisation, deux lignes de même transaction écrites autrement resteraient toutes deux dans le résultat."},
        ],
    },
}

LECONS = [{"id": "L-MM-5", "type": "lecon", "section": "lecon", "theme": "Leçon : réparer ou retirer une donnée sale", "difficulte": 2,
           "temps_cible_s": 900, "contenu": CONTENU, "controles": CONTROLES, "verif": {"script": "verif/seance_mm_5.py"}}]

E1 = r"""## Exercice 1 : charger et nettoyer

Écris `load_and_clean(path)`, où `path` est un chemin de fichier CSV ayant la forme décrite dans la leçon. Il rend un DataFrame de colonnes exactement `timestamp`, `ticker`, `price`, `volume`, dans cet ordre.

1. Noms de colonnes : espaces retirés, minuscules.
2. `ticker` : espaces de début et de fin retirés, majuscules.
3. `price` et `volume` : flottants, en retirant le symbole dollar et les virgules de milliers. Une cellule vide ou `N/A` donne une valeur manquante.
4. `timestamp` : type date.
5. Retirer les lignes de prix manquant, de volume manquant, ou de volume inférieur ou égal à 0.
6. Retirer les lignes exactement dupliquées, après ces conversions.
7. Trier par `ticker` puis `timestamp`, et réindexer de 0 à \(n-1\).

Le fichier de travail est `data_cache/messy_trading_data.csv`, relatif au dossier du notebook.

**Ce que vérifient les tests.** Le nombre de lignes, les symboles, l'absence de volume invalide et de doublon, les types.
"""

E2 = r"""## Exercice 2 : la volatilité par titre

Écris `realized_vol_by_ticker(df)`, où `df` a la forme rendue par `load_and_clean`, sans hypothèse sur l'ordre des lignes. Il rend une `Series` indexée par le symbole, triée par symbole : pour chaque titre, l'écart-type (`ddof=1`, le défaut de pandas) des variations relatives de `price` d'une ligne à la suivante, les lignes étant dans l'ordre du temps à l'intérieur de chaque titre.

**Ce que vérifie le test.** Les trois valeurs de la leçon, à 0,001 près : ce test valide aussi l'analyse des prix de l'exercice 1.
"""

CODES = [
    cm.code("MM5-E1", "code", "Nettoyer un export de transactions", 4, 1320, E1, "mm5_e1",
            [cm.IMPORTS, "pd.read_csv(\"data_cache/messy_trading_data.csv\").head(12)", "def load_and_clean(path):\n    pass  # À compléter",
             "load_and_clean(\"data_cache/messy_trading_data.csv\")"],
            "mm_5.py", "Correction commentée des deux exercices"),
    cm.code("MM5-E2", "code", "Volatilité par titre, calculée à l'intérieur de chaque groupe", 2, 780, E2, "mm5_e2",
            [cm.IMPORTS, "def realized_vol_by_ticker(df):\n    pass  # À compléter", "realized_vol_by_ticker(load_and_clean(\"data_cache/messy_trading_data.csv\"))"],
            "mm_5.py", "Correction commentée des deux exercices"),
]
QUESTIONS = []
