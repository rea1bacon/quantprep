"""Mesures qui fixent les seuils de la question de code T9-C1 (coding/test9oct), et preuve que ses contrôles
attrapent les défauts visés.

1. Les fichiers de données du dépôt sont exactement ceux que produit le générateur (graine fixe).
2. Exactitude, sur les étiquettes cachées du jeu de test, de cinq solutions de soin croissant :
   classe majoritaire ; régression logistique bâclée (colonnes déjà numériques, manquants remplacés par 0,
   colonne sale ignorée) ; nettoyage soigné + régression logistique sans variable ajoutée ; nettoyage soigné
   + boosting par histogrammes sans variable ajoutée ; la correction.
3. Les seuils de tests_test9oct.py séparent ces solutions comme annoncé dans l'énoncé.
4. Des fichiers défectueux (absent, colonne d'index, lignes de test supprimées, identifiants faux, probabilités
   au lieu de classes, doublons) échouent aux contrôles de format.

Sortie : verif/code_test_9_octobre.out.json et verif/code_test_9_octobre.out.txt, identiques dans les deux
environnements (python de la machine et .venv-hackerrank).
Usage : python verif/code_test_9_octobre.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RACINE = Path(__file__).resolve().parent.parent
DOSSIER = RACINE / "coding" / "test9oct"
sys.path.insert(0, str(DOSSIER))
import generer_donnees  # noqa: E402
import tests_test9oct as tests  # noqa: E402

V = {}
LIGNES_TEXTE = []


def dire(texte):
    LIGNES_TEXTE.append(texte)
    print(texte)


# ------------------------------------------------------------------ 1. fichiers et description du jeu
tests.verifier_fichiers()
train = pd.read_csv(DOSSIER / "train.csv")
test = pd.read_csv(DOSSIER / "test.csv")
y_test = np.array(generer_donnees.fabriquer()["y_test"])
assert test["id"].tolist() == tests.IDS_TEST and len(y_test) == len(test)
V["n_train"], V["n_test"] = len(train), len(test)
V["taux_panne_train"] = round(float(train["panne"].mean()), 4)
V["manquants_train"] = {c: int(n) for c, n in train.isna().sum().items() if n}
V["manquants_test"] = {c: int(n) for c, n in test.isna().sum().items() if n}
V["anciennetes_negatives_train"] = int((train["anciennete_ans"] < 0).sum())
V["anciennetes_negatives_test"] = int((test["anciennete_ans"] < 0).sum())
V["debits_en_lettres_train"] = int(train["debit_et_ligne"].str.contains("[a-z]").sum())
V["materiaux_sales_train"] = int((train["materiau"] != train["materiau"].str.strip().str.lower()).sum())
dire(f"jeu : {V['n_train']} lignes d'apprentissage, {V['n_test']} de test ; taux de panne en apprentissage {V['taux_panne_train']}")
dire(f"manquants : apprentissage {V['manquants_train']}, test {V['manquants_test']} ; anciennetés négatives : "
     f"{V['anciennetes_negatives_train']} et {V['anciennetes_negatives_test']}")


def noter(predictions, ids=None):
    """Écrit le fichier dans un dossier temporaire et rend (contrôles visibles réussis, cachés réussis, exactitude)."""
    ici = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            pd.DataFrame({"id": test["id"] if ids is None else ids, "panne": predictions}).to_csv("submissions.csv", index=False)
            visibles, caches = tests.evaluer("visibles", {}), tests.evaluer("caches", {})
        finally:
            os.chdir(ici)
    return sum(r["ok"] for r in visibles), sum(r["ok"] for r in caches)


def exactitude(predictions):
    return round(float((np.asarray(predictions) == y_test).mean()), 4)


# ------------------------------------------------------------------ 2. solutions de soin croissant
# a. Classe majoritaire de l'apprentissage.
majoritaire = int(train["panne"].mean() >= 0.5)
p_majoritaire = np.full(len(test), majoritaire)

# b. Solution bâclée : seules les colonnes déjà numériques, manquants à 0, anciennetés négatives laissées.
BRUTES = ["pression_bar", "temperature_c", "anciennete_ans", "heures_semaine"]
baclee = LogisticRegression(max_iter=5000).fit(train[BRUTES].fillna(0), train["panne"])
p_baclee = baclee.predict(test[BRUTES].fillna(0))

# c et d. Nettoyage soigné (celui de la correction, variable ajoutée retirée), deux modèles.
with contextlib.redirect_stdout(io.StringIO()):
    visibles_ref, caches_ref, acc_ref, variables = tests.evaluer_script(DOSSIER / "correction_test9oct.py")
nettoyer = variables["nettoyer"]
NUM = [c for c in variables["NUMERIQUES"] if c != "charge"]
CAT = variables["CATEGORIELLES"]
assert "charge" in variables["NUMERIQUES"]
a, b = nettoyer(train).dropna(), nettoyer(test)


def chaine(modele):
    preparation = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("echelle", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)], sparse_threshold=0)
    return Pipeline([("preparation", preparation), ("modele", modele)])


p_soignee_rl = chaine(LogisticRegression(max_iter=1000)).fit(a[NUM + CAT], a["panne"]).predict(b[NUM + CAT])
p_soignee_hgb = chaine(HistGradientBoostingClassifier(random_state=0)).fit(a[NUM + CAT], a["panne"]).predict(b[NUM + CAT])

V["exactitude_majoritaire"] = exactitude(p_majoritaire)
V["exactitude_baclee"] = exactitude(p_baclee)
V["exactitude_soignee_logistique_sans_variable"] = exactitude(p_soignee_rl)
V["exactitude_soignee_boosting_sans_variable"] = exactitude(p_soignee_hgb)
V["exactitude_correction"] = round(float(acc_ref), 4)
V["correction_modele_retenu"] = variables["meilleur"]
V["correction_exactitude_validation"] = round(variables["exactitude_validation"], 4)
V["correction_exactitude_cv"] = {k: round(v, 4) for k, v in variables["scores_cv"].items()}
V["correction_lignes_retirees"] = int(variables["avant"] - len(variables["train_net"]))
V["seuils"] = tests.SEUILS
for cle in ["exactitude_majoritaire", "exactitude_baclee", "exactitude_soignee_logistique_sans_variable",
            "exactitude_soignee_boosting_sans_variable", "exactitude_correction"]:
    dire(f"{cle} : {V[cle]}")
dire(f"correction : modèle retenu = {V['correction_modele_retenu']}, validation croisée {V['correction_exactitude_cv']}, "
     f"20 % mis de côté {V['correction_exactitude_validation']}, lignes retirées {V['correction_lignes_retirees']}")

# ------------------------------------------------------------------ 3. les seuils séparent les solutions
s1, s2, s3, s4 = tests.SEUILS
assert V["exactitude_majoritaire"] < s1 <= V["exactitude_baclee"] < s2
assert s3 <= V["exactitude_soignee_logistique_sans_variable"] < s4 and s3 <= V["exactitude_soignee_boosting_sans_variable"] < s4
assert V["exactitude_correction"] >= s4
V["marge_correction"] = round(V["exactitude_correction"] - s4, 4)
V["marge_correction_en_lignes"] = int(round(V["marge_correction"] * len(test)))
V["tests_caches_reussis"] = {
    "classe majoritaire": noter(p_majoritaire)[1], "bâclée": noter(p_baclee)[1],
    "soignée, logistique, sans variable ajoutée": noter(p_soignee_rl)[1],
    "soignée, boosting, sans variable ajoutée": noter(p_soignee_hgb)[1],
    "correction": sum(r["ok"] for r in caches_ref)}
assert list(V["tests_caches_reussis"].values()) == [1, 2, 4, 4, 5], V["tests_caches_reussis"]
assert sum(r["ok"] for r in visibles_ref) == tests.N_VISIBLES == 5 and tests.N_CACHES == 5
dire(f"seuils {tests.SEUILS} ; marge de la correction sur le dernier : {V['marge_correction']} "
     f"({V['marge_correction_en_lignes']} lignes de test) ; tests cachés réussis : {V['tests_caches_reussis']}")

# ------------------------------------------------------------------ 4. fichiers défectueux
ici = os.getcwd()
defauts = {}
with tempfile.TemporaryDirectory() as tmp:
    os.chdir(tmp)
    try:
        def bilan():
            return [int(r["ok"]) for r in tests.evaluer("visibles", {})], sum(r["ok"] for r in tests.evaluer("caches", {}))

        bon = pd.DataFrame({"id": test["id"], "panne": p_baclee})
        defauts["fichier absent"] = bilan()
        bon.to_csv("submissions.csv")                                   # index=False oublié : colonne d'index en trop
        defauts["colonne d'index écrite"] = bilan()
        bon.rename(columns={"panne": "outcome"}).to_csv("submissions.csv", index=False)
        defauts["mauvais nom de colonne"] = bilan()
        bon[test.notna().all(axis=1)].to_csv("submissions.csv", index=False)   # dropna appliqué au jeu de test
        defauts["lignes de test supprimées"] = bilan()
        bon.assign(id=range(len(bon))).to_csv("submissions.csv", index=False)  # index de position au lieu de l'identifiant
        defauts["identifiants faux"] = bilan()
        pd.concat([bon.iloc[:-1], bon.iloc[:1]]).to_csv("submissions.csv", index=False)
        defauts["un doublon, un absent"] = bilan()
        bon.assign(panne=baclee.predict_proba(test[BRUTES].fillna(0))[:, 1]).to_csv("submissions.csv", index=False)
        defauts["probabilités au lieu de classes"] = bilan()
        bon.assign(panne=bon["panne"].astype(bool)).to_csv("submissions.csv", index=False)
        defauts["True et False au lieu de 0 et 1"] = bilan()
        bon.sample(frac=1, random_state=0).assign(panne=lambda d: d["panne"].astype(float)).to_csv("submissions.csv", index=False)
        defauts["(valide) lignes mélangées, 0.0 et 1.0"] = bilan()
    finally:
        os.chdir(ici)
for nom, (visibles, caches) in defauts.items():
    dire(f"fichier défectueux, {nom} : contrôles visibles {visibles}, cachés réussis {caches}")
    if nom.startswith("(valide)"):
        assert visibles == [1] * 5 and caches == 2
    else:
        assert sum(visibles) < 5 and caches == 0
assert defauts["fichier absent"][0] == [0] * 5 and defauts["colonne d'index écrite"][0] == [1, 0, 0, 0, 0]
assert defauts["lignes de test supprimées"][0] == [1, 1, 0, 0, 0] and defauts["identifiants faux"][0] == [1, 1, 1, 0, 0]
assert defauts["probabilités au lieu de classes"][0] == [1, 1, 1, 1, 0]
V["fichiers_defectueux"] = {nom: {"visibles": v, "caches_reussis": c} for nom, (v, c) in defauts.items()}

# write_bytes : fins de ligne Unix quelle que soit la plateforme (write_text n'a pas d'argument newline en Python 3.9).
(RACINE / "verif" / "code_test_9_octobre.out.json").write_bytes(
    (json.dumps(V, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8"))
(RACINE / "verif" / "code_test_9_octobre.out.txt").write_bytes(("\n".join(LIGNES_TEXTE) + "\n").encode("utf-8"))
