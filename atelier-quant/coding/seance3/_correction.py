"""Corrections de la séance de code 3. À n'ouvrir qu'après avoir tenté les exercices.

Exercices 1 et 2 : deux versions indépendantes, pandas et sans bibliothèque externe ; les tests exigent qu'elles
soient d'accord. Exercice 3 : une seule version, avec scikit-learn (entraîner un classifieur n'a pas d'équivalent
raisonnable sans bibliothèque dans le temps d'une épreuve).
"""
import math
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MOTS = {"zéro": 0.0, "un": 1.0, "deux": 2.0, "trois": 3.0, "quatre": 4.0, "cinq": 5.0}


# ------------------------------------------------------------------ exercice 1
def nettoyer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()                                    # ne jamais modifier le cadre reçu
    # astype(str) d'abord : si une colonne a été lue comme numérique (aucune saleté, ou tout manquant), l'accesseur
    # .str n'existe pas et lèverait AttributeError. Une valeur manquante devient le texte "nan", qui ne correspond
    # à aucun motif ci-dessous et redevient donc NaN.
    # age : un nombre, éventuellement suivi de " ans". Le motif est ancré (^ au début, \Z à la fin) : "inconnu" ne donne aucun groupe.
    age = d["age"].astype(str).str.strip().str.extract(r"^(\d+(?:\.\d+)?)(?: ans)?\Z")[0]
    d["age"] = pd.to_numeric(age, errors="coerce").astype(float)
    # revenu : retirer espaces et symbole, virgule décimale -> point ; ce qui reste illisible devient NaN (coerce).
    revenu = (d["revenu"].astype(str).str.replace(" ", "", regex=False).str.replace("€", "", regex=False)
              .str.replace(",", ".", regex=False))
    d["revenu"] = pd.to_numeric(revenu, errors="coerce").astype(float)
    # appels : les mots par dictionnaire, le reste par conversion numérique.
    appels = d["appels"].astype(str).str.strip()
    d["appels"] = appels.map(MOTS).fillna(pd.to_numeric(appels, errors="coerce")).astype(float)
    # anciennete_mois : valeur impossible (strictement négative) -> manquante. where garde là où la condition est vraie.
    anciennete = pd.to_numeric(d["anciennete_mois"], errors="coerce").astype(float)
    d["anciennete_mois"] = anciennete.where(anciennete >= 0)
    # forfait : mêmes modalités écrites différemment ; les valeurs manquantes restent manquantes.
    d["forfait"] = d["forfait"].where(d["forfait"].isna(), d["forfait"].astype(str).str.strip().str.lower())
    return d


def _manquant(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _nombre(texte):
    try:
        return float(texte)
    except (TypeError, ValueError):
        return math.nan


def nettoyer_pur(lignes):
    """lignes : liste de dictionnaires (une ligne chacun). Rend une nouvelle liste de dictionnaires nettoyés."""
    sortie = []
    for ligne in lignes:
        r = dict(ligne)
        v = ligne["age"]
        if isinstance(v, str):
            t = v.strip()
            t = t[:-4] if t.endswith(" ans") else t
            r["age"] = float(t) if re.fullmatch(r"\d+(?:\.\d+)?", t) else math.nan
        else:
            r["age"] = math.nan if _manquant(v) else float(v)
        v = ligne["revenu"]
        if isinstance(v, str):
            r["revenu"] = _nombre(v.replace(" ", "").replace("€", "").replace(",", "."))
        else:
            r["revenu"] = math.nan if _manquant(v) else float(v)
        v = ligne["appels"]
        if isinstance(v, str):
            t = v.strip()
            r["appels"] = MOTS[t] if t in MOTS else _nombre(t)
        else:
            r["appels"] = math.nan if _manquant(v) else float(v)
        v = ligne["anciennete_mois"]
        r["anciennete_mois"] = math.nan if _manquant(v) or float(v) < 0 else float(v)
        v = ligne["forfait"]
        r["forfait"] = math.nan if _manquant(v) else str(v).strip().lower()
        sortie.append(r)
    return sortie


# ------------------------------------------------------------------ exercice 2
def aligner(train: pd.DataFrame, test: pd.DataFrame, colonnes):
    # get_dummies crée une colonne par modalité PRÉSENTE dans le cadre qu'on lui donne : les deux cadres n'ont donc
    # pas les mêmes colonnes dès qu'une modalité manque d'un côté. La référence est l'apprentissage.
    x_train = pd.get_dummies(train, columns=colonnes, dtype=float)
    # reindex(columns=...) fait les trois corrections d'un coup : il retire les colonnes des modalités inconnues,
    # ajoute à 0 celles des modalités absentes du test, et remet les colonnes dans l'ordre de l'apprentissage.
    x_test = pd.get_dummies(test, columns=colonnes, dtype=float).reindex(columns=x_train.columns, fill_value=0.0)
    return x_train, x_test


def aligner_pur(train, test, colonnes):
    """train, test : listes de dictionnaires dont les clés sont dans l'ordre des colonnes. Rend deux listes de dictionnaires."""
    autres = [c for c in (train[0] if train else {}) if c not in colonnes]
    modalites = {c: sorted({l[c] for l in train if not _manquant(l[c])}) for c in colonnes}

    def encoder(lignes):
        sortie = []
        for l in lignes:
            r = {c: l[c] for c in autres}
            for c in colonnes:
                for v in modalites[c]:
                    r[f"{c}_{v}"] = 1.0 if (not _manquant(l[c]) and l[c] == v) else 0.0
            sortie.append(r)
        return sortie

    return encoder(train), encoder(test)


# ------------------------------------------------------------------ exercice 3
NUMERIQUES = ["age", "revenu", "appels", "anciennete_mois"]
CATEGORIELLES = ["forfait"]


def predire(train: pd.DataFrame, test: pd.DataFrame):
    # 1. Le même nettoyage pour les deux cadres (exercice 1). Aucune ligne de test n'est retirée.
    a, b = nettoyer(train), nettoyer(test)
    # 2. Tout ce qui s'apprend (médianes, moyennes, écarts-types, modalités) est dans la chaîne, donc appris par fit
    #    sur l'apprentissage seul, puis appliqué tel quel au test.
    preparation = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("echelle", StandardScaler())]), NUMERIQUES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORIELLES),   # modalité inconnue au test : ligne de zéros
    ])
    modele = Pipeline([("preparation", preparation), ("modele", LogisticRegression(max_iter=1000))])
    # 3. Les lignes d'apprentissage incomplètes sont gardées : l'imputation par la médiane les rend utilisables.
    modele.fit(a[NUMERIQUES + CATEGORIELLES], a["resilie"])
    return modele.predict(b[NUMERIQUES + CATEGORIELLES]).astype(int)


def predire_baclee(train: pd.DataFrame, test: pd.DataFrame):
    """Ce qu'il ne faut pas faire, pour mesurer ce que coûte l'absence de nettoyage : seule la colonne déjà
    numérique est utilisée, les manquants sont mis à 0, la valeur impossible -1 est laissée."""
    modele = LogisticRegression(max_iter=1000).fit(train[["anciennete_mois"]].fillna(0), train["resilie"])
    return modele.predict(test[["anciennete_mois"]].fillna(0)).astype(int)
