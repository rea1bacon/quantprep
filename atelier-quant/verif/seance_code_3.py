"""Vérifications de la séance de code 3 (chaîne complète de classification avec scikit-learn).

1. Écrit les fichiers d'exploration de la leçon, coding/seance3/abonnes_train.csv et abonnes_test.csv
   (données synthétiques, graine fixe), et contrôle qu'ils sont reproduits à l'identique.
2. Recalcule les exemples des énoncés des exercices 1 et 2 par la correction pandas, par la version sans
   bibliothèque et à la main.
3. Prouve que les tests cachés attrapent les erreurs visées : des solutions fausses typiques échouent à au moins
   un test caché, des variantes justes écrites autrement les passent tous.
4. Mesure, pour l'exercice 3, l'exactitude de la classe majoritaire, d'une solution bâclée et de la correction sur
   chaque test, et vérifie la marge de la correction sur chaque seuil.
5. Calcule les valeurs des questions à choix (exactement, par fractions, et par scikit-learn).
6. Vérifie chaque fait d'interface (API) cité dans la leçon L-SKLEARN-1, dans la version installée ; les faits
   propres à scikit-learn 1.1.1 et pandas 1.4.2 ne sont affirmés que dans cet environnement.

Sorties :
  verif/seance_code_3.out.json et .out.txt : valeurs citées par la séance, identiques dans les deux environnements ;
  verif/seance_code_3.api.<version de scikit-learn>.out.json : faits d'interface constatés dans cette version.
Usage : python verif/seance_code_3.py   et   .venv-hackerrank/Scripts/python verif/seance_code_3.py
"""
import contextlib
import copy
import io
import json
import math
import sys
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RACINE = Path(__file__).resolve().parent.parent
DOSSIER = RACINE / "coding" / "seance3"
sys.path.insert(0, str(DOSSIER))
import _communs  # noqa: E402
import tests_e1  # noqa: E402
import tests_e2  # noqa: E402
import tests_e3  # noqa: E402
from _correction import MOTS, aligner, nettoyer, predire, predire_baclee  # noqa: E402

SK = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
PD = tuple(int(x) for x in pd.__version__.split(".")[:2])
REFERENCE = sklearn.__version__ == "1.1.1" and pd.__version__ == "1.4.2"   # l'environnement du notebook
V = {}
API = {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}
LIGNES_TEXTE = []


def dire(texte):
    LIGNES_TEXTE.append(texte)
    print(texte)


def ecrire(chemin, texte):
    chemin.write_bytes(texte.encode("utf-8"))   # fins de ligne Unix quelle que soit la plateforme


def leve(f, *types):
    """Le nom de l'exception levée par f(), ou None."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f()
    except types as e:
        return type(e).__name__
    return None


# ============================================================ 1. fichiers d'exploration de la leçon
N_TRAIN_LECON, N_TEST_LECON = 1500, 600
train_l, test_l, y_test_l = _communs.jeu(3001, N_TRAIN_LECON, N_TEST_LECON, part_pro_test=0.1)
for nom, cadre in [("abonnes_train.csv", train_l), ("abonnes_test.csv", test_l)]:
    texte = cadre.to_csv(index=False).replace("\r\n", "\n")
    if (DOSSIER / nom).exists():
        assert (DOSSIER / nom).read_bytes() == texte.encode("utf-8"), f"{nom} n'est pas reproduit à l'identique"
    ecrire(DOSSIER / nom, texte)
relu = pd.read_csv(DOSSIER / "abonnes_train.csv")
assert relu.shape == (N_TRAIN_LECON, 7) and list(relu.columns) == _communs.COLONNES + [_communs.CIBLE]
assert relu["age"].dtype == object and relu["revenu"].dtype == object and relu["appels"].dtype == object
assert relu["anciennete_mois"].dtype == float and (relu["anciennete_mois"] < 0).any()
assert "pro" not in set(nettoyer(relu)["forfait"]) and "pro" in set(nettoyer(test_l)["forfait"])
V["lecon_n_train"], V["lecon_n_test"] = N_TRAIN_LECON, N_TEST_LECON
dire(f"fichiers de la leçon : {N_TRAIN_LECON} et {N_TEST_LECON} lignes ; modalité pro absente de l'apprentissage, présente au test")

# ============================================================ 2. exemples des énoncés
# Exercice 1.
r = nettoyer(tests_e1.cadre(tests_e1.EXEMPLE))
NAN = math.nan
a_la_main = pd.DataFrame(
    [(11, 34.0, 2450.0, 3.0, 12.0, "premium"), (12, NAN, 1980.0, 0.0, NAN, "base"),
     (13, 51.0, NAN, 5.0, NAN, "plus"), (14, 27.0, 3100.5, 2.0, 40.0, "plus")], columns=tests_e1.COLONNES)
assert tests_e1.comparer(r, a_la_main), r
V["E1_revenu_ligne_14"] = float(r.loc[3, "revenu"])
V["E1_exemple_manquants"] = int(r[_communs.NUMERIQUES].isna().sum().sum())
assert V["E1_exemple_manquants"] == 4
# Exercice 2.
x_train, x_test = aligner(tests_e2.TRAIN, tests_e2.TEST, ["forfait"])
colonnes_attendues = ["revenu", "forfait_base", "forfait_plus", "forfait_premium"]
assert list(x_train.columns) == colonnes_attendues == list(x_test.columns)
assert x_train.to_numpy().tolist() == [[2450.0, 0, 1, 0], [1980.0, 1, 0, 0], [3100.0, 0, 1, 0], [2700.0, 0, 0, 1]]
assert x_test.to_numpy().tolist() == [[2000.0, 1, 0, 0], [2600.0, 0, 0, 0], [3300.0, 1, 0, 0]]
naif = pd.get_dummies(tests_e2.TEST, columns=["forfait"])
assert list(naif.columns) == ["revenu", "forfait_base", "forfait_pro"]
V["E2_colonnes_train"] = len(colonnes_attendues)
V["E2_colonnes_test_naif"] = int(naif.shape[1])
dire("exemples des exercices 1 et 2 : correction pandas, version sans bibliothèque et calcul à la main concordent")


# ============================================================ 3. les tests cachés attrapent les erreurs visées
def passe(module, fonction):
    """Résultat de chaque test caché, comme le noyau : entrées copiées, exception = échec."""
    resultats = []
    for c in module.cas_caches():
        try:
            with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                resultats.append(bool(module.comparer(fonction(*copy.deepcopy(c["args"])), c["attendu"])))
        except Exception:  # noqa: BLE001
            resultats.append(False)
    return resultats


def visibles(module, fonction):
    resultats = []
    for c in module.cas_visibles():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resultats.append(bool(module.comparer(fonction(*copy.deepcopy(c["args"])), c["attendu"])))
        except Exception:  # noqa: BLE001
            resultats.append(False)
    return resultats


# --- exercice 1
def e1_sans_astype(df):          # .str directement : échoue dès qu'une colonne n'est pas une colonne de chaînes
    d = df.copy()
    d["age"] = pd.to_numeric(d["age"].str.strip().str.extract(r"^(\d+(?:\.\d+)?)(?: ans)?\Z")[0], errors="coerce")
    d["revenu"] = pd.to_numeric(d["revenu"].str.replace(" ", "", regex=False).str.replace("€", "", regex=False)
                                .str.replace(",", ".", regex=False), errors="coerce")
    s = d["appels"].str.strip()
    d["appels"] = s.map(MOTS).fillna(pd.to_numeric(s, errors="coerce"))
    d["anciennete_mois"] = d["anciennete_mois"].where(d["anciennete_mois"] >= 0)
    d["forfait"] = d["forfait"].str.strip().str.lower()
    return d


def e1_negatifs_gardes(df):      # oublie la règle des valeurs impossibles
    d = nettoyer(df)
    d["anciennete_mois"] = pd.to_numeric(df["anciennete_mois"], errors="coerce").astype(float)
    return d


def e1_motif_non_ancre(df):      # motif sans ^ ni $ : "34ans" donne 34 au lieu de manquant
    d = nettoyer(df)
    d["age"] = pd.to_numeric(df["age"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce").astype(float)
    return d


def e1_virgule_retiree(df):      # la virgule décimale retirée au lieu d'être remplacée par un point
    d = nettoyer(df)
    s = df["revenu"].astype(str).str.replace(" ", "", regex=False).str.replace("€", "", regex=False).str.replace(",", "", regex=False)
    d["revenu"] = pd.to_numeric(s, errors="coerce").astype(float)
    return d


def e1_forfait_brut(df):         # casse et espaces laissés
    d = nettoyer(df)
    d["forfait"] = df["forfait"]
    return d


def e1_ligne_a_ligne(df):        # variante juste, écrite autrement : une fonction par valeur et apply
    import re

    def age(v):
        if isinstance(v, str):
            m = re.fullmatch(r"(\d+(?:\.\d+)?)(?: ans)?", v.strip())
            return float(m.group(1)) if m else np.nan
        return float(v)

    def revenu(v):
        if isinstance(v, str):
            try:
                return float(v.replace(" ", "").replace("€", "").replace(",", "."))
            except ValueError:
                return np.nan
        return float(v)

    def appels(v):
        if isinstance(v, str):
            v = v.strip()
            if v in MOTS:
                return MOTS[v]
            try:
                return float(v)
            except ValueError:
                return np.nan
        return float(v)

    d = df.copy()
    d["age"] = d["age"].apply(age).astype(float)
    d["revenu"] = d["revenu"].apply(revenu).astype(float)
    d["appels"] = d["appels"].apply(appels).astype(float)
    d["anciennete_mois"] = d["anciennete_mois"].astype(float).mask(d["anciennete_mois"].astype(float) < 0)
    d["forfait"] = d["forfait"].apply(lambda v: v.strip().lower() if isinstance(v, str) else v)
    return d


fausses_e1 = {"accesseur .str sans astype(str)": e1_sans_astype, "anciennetés négatives gardées": e1_negatifs_gardes,
              "motif d'âge non ancré": e1_motif_non_ancre, "virgule décimale retirée": e1_virgule_retiree,
              "forfait non uniformisé": e1_forfait_brut}
V["E1_tests_caches"] = len(tests_e1.cas_caches())
V["E1_fausses"] = {}
for nom, f in fausses_e1.items():
    res = passe(tests_e1, f)
    assert not all(res) or not all(visibles(tests_e1, f)), nom
    V["E1_fausses"][nom] = f"{sum(res)}/{len(res)} cachés, {sum(visibles(tests_e1, f))}/2 visibles"
assert all(passe(tests_e1, e1_ligne_a_ligne)) and all(passe(tests_e1, nettoyer))
# La faute la plus sournoise passe les tests visibles et n'est vue que par les tests cachés.
assert all(visibles(tests_e1, e1_sans_astype)) and not all(passe(tests_e1, e1_sans_astype))
dire(f"exercice 1 : solutions fausses attrapées {V['E1_fausses']} ; variante juste ligne à ligne acceptée")


# --- exercice 2
def e2_separes(train, test, colonnes):                 # get_dummies de chaque côté, sans alignement
    return pd.get_dummies(train, columns=colonnes, dtype=float), pd.get_dummies(test, columns=colonnes, dtype=float)


def e2_concatenes(train, test, colonnes):              # encodage du tout : les modalités du test entrent dans les colonnes
    tout = pd.get_dummies(pd.concat([train, test], keys=["a", "b"]), columns=colonnes, dtype=float)
    return tout.loc["a"], tout.loc["b"]


def e2_sans_remplissage(train, test, colonnes):        # reindex sans fill_value : NaN au lieu de 0
    a = pd.get_dummies(train, columns=colonnes, dtype=float)
    return a, pd.get_dummies(test, columns=colonnes, dtype=float).reindex(columns=a.columns)


def e2_align(train, test, colonnes):                   # variante juste : DataFrame.align sur les colonnes de l'apprentissage
    a = pd.get_dummies(train, columns=colonnes, dtype=float)
    b = pd.get_dummies(test, columns=colonnes, dtype=float)
    a, b = a.align(b, join="left", axis=1, fill_value=0.0)
    return a, b


fausses_e2 = {"get_dummies séparés": e2_separes, "get_dummies sur la concaténation": e2_concatenes,
              "reindex sans fill_value": e2_sans_remplissage}
V["E2_tests_caches"] = len(tests_e2.cas_caches())
V["E2_fausses"] = {}
for nom, f in fausses_e2.items():
    res = passe(tests_e2, f)
    assert not all(res), nom
    V["E2_fausses"][nom] = f"{sum(res)}/{len(res)} cachés, {sum(visibles(tests_e2, f))}/2 visibles"
assert all(passe(tests_e2, e2_align)) and all(passe(tests_e2, aligner))
dire(f"exercice 2 : solutions fausses attrapées {V['E2_fausses']} ; variante juste (DataFrame.align) acceptée")


# --- exercice 3
def e3_majoritaire(train, test):
    return np.full(len(test), int(train["resilie"].mean() >= 0.5))


def e3_dropna_test(train, test):                       # retire des lignes du test : le nombre de prédictions est faux
    return predire(train, nettoyer(test).dropna())


def e3_dummies_separes(train, test):                   # colonnes désalignées dès qu'une modalité diffère
    a, b = nettoyer(train), nettoyer(test)
    xa = pd.get_dummies(a[_communs.NUMERIQUES + ["forfait"]]).fillna(0)
    xb = pd.get_dummies(b[_communs.NUMERIQUES + ["forfait"]]).fillna(0)
    return LogisticRegression(max_iter=5000).fit(xa, a["resilie"]).predict(xb)


def e3_foret(train, test):                             # variante juste : autre modèle, encodage aligné à la main
    a, b = nettoyer(train), nettoyer(test)
    xa, xb = aligner(a[_communs.NUMERIQUES + ["forfait"]], b[_communs.NUMERIQUES + ["forfait"]], ["forfait"])
    mediane = xa.median()
    return RandomForestClassifier(200, min_samples_leaf=5, random_state=0).fit(xa.fillna(mediane), a["resilie"]).predict(xb.fillna(mediane))


def e3_boosting(train, test):                          # variante juste : boosting par histogrammes, NaN laissés tels quels
    a, b = nettoyer(train), nettoyer(test)
    xa, xb = aligner(a[_communs.NUMERIQUES + ["forfait"]], b[_communs.NUMERIQUES + ["forfait"]], ["forfait"])
    return HistGradientBoostingClassifier(random_state=0).fit(xa, a["resilie"]).predict(xb)


V["E3_tests_caches"] = len(tests_e3.cas_caches())
mesures = []
for c in tests_e3.VISIBLES + tests_e3.CACHES:
    cas = tests_e3._cas(*c)
    train, test = cas["args"]
    e = cas["attendu"]
    ligne = {"test": c[0], "seuil": c[4], "correction": round(e.mesurer(predire(train, test)), 4),
             "baclee": round(e.mesurer(predire_baclee(train, test)), 4), "majoritaire": round(e.mesurer(e3_majoritaire(train, test)), 4)}
    mesures.append(ligne)
    if c[4] > 0:   # tests d'exactitude (les tests de format ont un seuil nul)
        assert ligne["correction"] >= c[4] + 0.03, ligne            # marge d'au moins 3 points
        assert max(ligne["baclee"], ligne["majoritaire"]) <= c[4] - 0.05, ligne
    dire(f"exercice 3, {c[0]} : seuil {c[4]}, correction {ligne['correction']}, bâclée {ligne['baclee']}, majoritaire {ligne['majoritaire']}")
V["E3_mesures"] = mesures
V["E3_marge_minimale"] = round(min(m["correction"] - m["seuil"] for m in mesures if m["seuil"] > 0), 4)
notes_e3 = [m for m in mesures if m["seuil"] > 0]
V["E3_seuil"] = max(m["seuil"] for m in notes_e3)
V["E3_seuil_petit"] = min(m["seuil"] for m in notes_e3)
assert sorted({m["seuil"] for m in notes_e3}) == [V["E3_seuil_petit"], V["E3_seuil"]]
assert [m["seuil"] for m in notes_e3 if "petit jeu" in m["test"]] == [V["E3_seuil_petit"]]
V["E3_majoritaire_max"] = max(m["majoritaire"] for m in notes_e3)
V["E3_baclee_max"] = max(m["baclee"] for m in notes_e3)
V["E3_correction_min"] = min(m["correction"] for m in notes_e3)
V["E3_correction_max"] = max(m["correction"] for m in notes_e3)
V["E3_tests_exactitude"] = len(notes_e3)
fausses_e3 = {"classe majoritaire": e3_majoritaire, "solution bâclée": predire_baclee, "dropna sur le test": e3_dropna_test,
              "get_dummies séparés": e3_dummies_separes}
V["E3_fausses"] = {}
for nom, f in fausses_e3.items():
    res = passe(tests_e3, f)
    assert not all(res), nom
    V["E3_fausses"][nom] = f"{sum(res)}/{len(res)} cachés"
assert sum(passe(tests_e3, e3_majoritaire)) <= 1 and sum(passe(tests_e3, predire_baclee)) <= 1   # au plus le test de format
assert all(passe(tests_e3, predire)) and all(passe(tests_e3, e3_foret)) and all(passe(tests_e3, e3_boosting))
dire(f"exercice 3 : marge minimale de la correction {V['E3_marge_minimale']} ; solutions fausses attrapées {V['E3_fausses']} ; "
     "variantes justes (forêt, boosting) acceptées")

# ============================================================ 5. questions à choix
# S3-Q1 : standardisation ajustée sur l'apprentissage seul, ou sur tout (fuite).
x_app, x_test = [2.0, 4.0, 6.0, 8.0], 20.0
m_app = Fraction(sum(map(int, x_app)), len(x_app))
var_app = sum((Fraction(int(x)) - m_app) ** 2 for x in x_app) / len(x_app)             # dénominateur n, comme StandardScaler
tout = x_app + [x_test]
m_tout = Fraction(sum(map(int, tout)), len(tout))
var_tout = sum((Fraction(int(x)) - m_tout) ** 2 for x in tout) / len(tout)
var_app_ddof1 = sum((Fraction(int(x)) - m_app) ** 2 for x in x_app) / (len(x_app) - 1)
assert (m_app, var_app, m_tout, var_tout) == (5, 5, 8, 40)
z_juste = float(20 - m_app) / math.sqrt(var_app)
z_fuite = float(20 - m_tout) / math.sqrt(var_tout)
z_ddof1 = float(20 - m_app) / math.sqrt(var_app_ddof1)
sc = StandardScaler().fit(np.array(x_app).reshape(-1, 1))
assert abs(sc.transform([[x_test]])[0, 0] - z_juste) < 1e-12 and sc.mean_[0] == 5 and abs(sc.var_[0] - 5) < 1e-12
assert abs(StandardScaler().fit_transform(np.array(tout).reshape(-1, 1))[-1, 0] - z_fuite) < 1e-12
assert abs((x_test - np.mean(x_app)) / pd.Series(x_app).std() - z_ddof1) < 1e-12        # pandas : dénominateur n - 1
assert StandardScaler().fit_transform([[x_test]])[0, 0] == 0.0                           # ajusté sur la seule ligne de test
V.update({"Q1_moyenne_app": int(m_app), "Q1_variance_app": int(var_app), "Q1_moyenne_tout": int(m_tout), "Q1_variance_tout": int(var_tout),
          "Q1_z_juste": z_juste, "Q1_z_fuite": z_fuite, "Q1_z_ddof1": z_ddof1, "Q1_z_test_seul": 0})
dire(f"S3-Q1 : z juste {z_juste:.4f}, avec fuite {z_fuite:.4f}, avec dénominateur n - 1 {z_ddof1:.4f}, ajusté sur le test seul 0")

# S3-Q2 : modalité inconnue.
q2_train = pd.DataFrame({"forfait": ["base", "plus", "premium", "plus"]})
q2_test = pd.DataFrame({"forfait": ["plus", "pro"]})
enc = OneHotEncoder(handle_unknown="ignore").fit(q2_train)
m = enc.transform(q2_test).toarray()
assert m.tolist() == [[0, 1, 0], [0, 0, 0]] and list(enc.categories_[0]) == ["base", "plus", "premium"]
assert leve(lambda: OneHotEncoder().fit(q2_train).transform(q2_test), ValueError) == "ValueError"
assert list(pd.get_dummies(q2_test).columns) == ["forfait_plus", "forfait_pro"]
assert pd.get_dummies(pd.concat([q2_train, q2_test])).shape[1] == 4
V.update({"Q2_colonnes": int(m.shape[1]), "Q2_somme_ligne_pro": int(m[1].sum()), "Q2_colonnes_dummies_test": 2, "Q2_colonnes_concat": 4})
dire(f"S3-Q2 : {m.shape[1]} colonnes, ligne pro = {m[1].tolist()} ; get_dummies du test seul : 2 colonnes ; de la concaténation : 4")

# S3-Q4 : exactitude et F1 sous déséquilibre.
vp, fn, fp, vn = 30, 20, 60, 890
n = vp + fn + fp + vn
y_vrai = np.array([1] * (vp + fn) + [0] * (fp + vn))
y_b = np.array([1] * vp + [0] * fn + [1] * fp + [0] * vn)
exactitude_b, precision_b, rappel_b = Fraction(vp + vn, n), Fraction(vp, vp + fp), Fraction(vp, vp + fn)
f1_b = 2 * precision_b * rappel_b / (precision_b + rappel_b)
assert f1_b == Fraction(2 * vp, 2 * vp + fp + fn)
assert abs(accuracy_score(y_vrai, y_b) - float(exactitude_b)) < 1e-12 and abs(f1_score(y_vrai, y_b) - float(f1_b)) < 1e-12
assert abs(precision_score(y_vrai, y_b) - float(precision_b)) < 1e-12 and abs(recall_score(y_vrai, y_b) - float(rappel_b)) < 1e-12
exactitude_a = Fraction(fp + vn, n)
assert accuracy_score(y_vrai, np.zeros(n)) == float(exactitude_a) and f1_score(y_vrai, np.zeros(n), zero_division=0) == 0.0
V.update({"Q4_n": n, "Q4_positifs": vp + fn, "Q4_vp": vp, "Q4_fn": fn, "Q4_fp": fp, "Q4_vn": vn,
          "Q4_exactitude_A": float(exactitude_a), "Q4_exactitude_B": float(exactitude_b), "Q4_precision_B": float(precision_b),
          "Q4_rappel_B": float(rappel_b), "Q4_f1_B": float(f1_b), "Q4_f1_num": f1_b.numerator, "Q4_f1_den": f1_b.denominator,
          "Q4_moyenne_arithmetique": float((precision_b + rappel_b) / 2)})
dire(f"S3-Q4 : exactitude A {float(exactitude_a)}, B {float(exactitude_b)} ; précision {precision_b}, rappel {rappel_b}, F1 {f1_b}")

# Points de contrôle de la leçon.
s = pd.to_numeric(pd.Series(["7", "7 %", "sept"]), errors="coerce")
assert s.iloc[0] == 7 and s.iloc[1:].isna().all()
assert leve(lambda: pd.to_numeric(pd.Series(["7", "7 %", "sept"])), ValueError) == "ValueError"
s2 = pd.to_numeric(pd.Series(["12", "1 200", "3,5", "douze", None]), errors="coerce")
assert s2.iloc[0] == 12 and s2.iloc[1:].isna().all()
V["L_coerce_manquants"] = int(s.isna().sum())
V["L_exactitude_majoritaire_90"] = 0.9

# ============================================================ 6. faits d'interface cités dans la leçon
X_num = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
y_bin = np.array([0, 0, 0, 1, 1, 1])

# Lecture et conversion (pandas).
assert pd.to_numeric(pd.Series(["3.5", "x", None]), errors="coerce").isna().tolist() == [False, True, True]
ext = pd.Series(["34 ans", "inconnu"]).str.extract(r"^(\d+)(?: ans)?\Z")
assert isinstance(ext, pd.DataFrame) and ext[0].tolist()[0] == "34" and pd.isna(ext[0].iloc[1])
API["str.extract rend un DataFrame, NaN quand le motif ne correspond pas"] = True
assert pd.Series(["a,b", "c;d"]).str.split(r"[;,]", regex=True, expand=True).shape == (2, 2)
assert pd.Series(["2 450 €"]).str.replace(r"[^\d,.-]", "", regex=True).iloc[0] == "2450"
assert leve(lambda: pd.Series([1.0, 2.0]).str.strip(), AttributeError) == "AttributeError"
API["accesseur .str sur une colonne numérique : AttributeError"] = True
assert pd.Series([1.0, np.nan]).astype(str).tolist() == ["1.0", "nan"]
assert pd.Series([5.0, -1.0, np.nan]).where(lambda x: x >= 0).isna().tolist() == [False, True, True]
d = pd.get_dummies(pd.DataFrame({"f": ["a", "b"]}))
API["dtype des colonnes de get_dummies"] = str(d["f_a"].dtype)
assert API["dtype des colonnes de get_dummies"] == ("uint8" if PD < (2, 0) else "bool")
assert list(pd.get_dummies(pd.DataFrame({"x": [1.0], "f": ["a"]}), columns=["f"], dtype=float).columns) == ["x", "f_a"]
assert pd.get_dummies(pd.DataFrame({"f": ["a", np.nan]})).to_numpy().tolist() in ([[1], [0]], [[True], [False]])
assert pd.read_csv(io.StringIO('a,b\n"1,5",2\n')).iloc[0, 0] == "1,5"
assert pd.DataFrame({"id": [1]}).to_csv(index=False).replace("\r", "") == "id\n1\n"
assert pd.DataFrame({"id": [1]}).to_csv().replace("\r", "").startswith(",id")
API["to_csv sans index=False écrit l'index en première colonne"] = True

# Faits cités dans les parties 2 à 4 de la leçon.
import re  # noqa: E402
mixte = pd.DataFrame({"n": [1.0, -2.0], "t": ["a", "inconnu"]})
assert list(mixte.describe().columns) == ["n"] and mixte.isna().sum().sum() == 0      # describe : colonnes numériques seulement
assert pd.Series(["un", "sept"]).map({"un": 1}).isna().tolist() == [False, True]      # map : NaN pour une clé absente
assert leve(lambda: {"un": 1}["sept"], KeyError) == "KeyError"
AVEC_SAUT = "34" + chr(10)
assert re.search(r"^(\d+)\Z", "34") and not re.search(r"^(\d+)\Z", AVEC_SAUT) and re.search(r"^(\d+)$", AVEC_SAUT)   # \Z : fin stricte
assert pd.Series(["34ans", "x34y"]).str.extract(r"(\d+)")[0].tolist() == ["34", "34"]   # sans ancres : trouvé n'importe où
assert pd.Series(["32.0;2", "trente-deux_3", "28.0,1"]).str.extract(r"^(.*)[;,_](\d)\Z").to_numpy().tolist() == [
    ["32.0", "2"], ["trente-deux", "3"], ["28.0", "1"]]
assert pd.Series(["1,5;2,3"]).str.extract(r"^(.*)[;,_](\d)\Z").to_numpy().tolist() == [["1,5;2", "3"]]   # dernier séparateur
imp = SimpleImputer(strategy="median").fit([[1.0], [2.0], [9.0], [np.nan]])
assert imp.statistics_.tolist() == [2.0] and imp.transform([[np.nan]]).tolist() == [[2.0]]
assert "add_indicator" in SimpleImputer().get_params()
assert SimpleImputer(strategy="most_frequent").fit_transform(np.array([["a"], ["a"], [np.nan]], dtype=object)).ravel().tolist() == ["a", "a", "a"]
assert OneHotEncoder().fit_transform([[1], [4], [1]]).toarray().tolist() == [[1, 0], [0, 1], [1, 0]]   # accepte des entiers
API["faits des parties 2 à 4 (describe, map, ancres, SimpleImputer)"] = True
# scikit-learn place le seuil d'un arbre au milieu de deux valeurs observées consécutives.
arbre = DecisionTreeClassifier(random_state=0).fit([[0.0], [1.0], [2.0], [10.0]], [0, 0, 1, 1])
assert arbre.tree_.threshold[0] == 1.5
API["seuil d'un arbre : milieu de deux valeurs consécutives"] = float(arbre.tree_.threshold[0])
# Pipeline : fit ajuste chaque étape sur les lignes reçues ; predict ne réajuste rien ; cross_val_score travaille sur des copies.
pp = Pipeline([("echelle", StandardScaler()), ("modele", LogisticRegression())])
Xp = np.array([[1.0], [2.0], [3.0], [4.0], [10.0], [20.0]])
yp = np.array([0, 0, 0, 1, 1, 1])
pp.fit(Xp[:4], np.array([0, 0, 1, 1]))
assert pp.named_steps["echelle"].mean_[0] == 2.5
pp.predict(Xp[4:])
assert pp.named_steps["echelle"].mean_[0] == 2.5
neuf = Pipeline([("echelle", StandardScaler()), ("modele", LogisticRegression())])
cross_val_score(neuf, Xp, yp, cv=3)
assert not hasattr(neuf.named_steps["echelle"], "mean_")
API["Pipeline : étapes ajustées par fit seulement ; cross_val_score laisse l'objet reçu non ajusté"] = True
# AUC = probabilité qu'un positif tiré au hasard ait un score plus élevé qu'un négatif, une égalité comptant pour 1/2.
rng_auc = np.random.default_rng(1)
sc_auc = np.round(rng_auc.random(60), 1)            # arrondi : crée des égalités
y_auc = rng_auc.integers(0, 2, 60)
paires = [(1.0 if p > q else 0.5 if p == q else 0.0) for p in sc_auc[y_auc == 1] for q in sc_auc[y_auc == 0]]
assert abs(roc_auc_score(y_auc, sc_auc) - float(np.mean(paires))) < 1e-12
API["AUC = part des paires (positif, négatif) bien ordonnées, égalités comptées 1/2"] = True

# Valeurs par défaut citées.
defauts = {
    "LogisticRegression": {k: LogisticRegression().get_params()[k] for k in ["C", "penalty", "max_iter", "solver"]},
    "SimpleImputer.strategy": SimpleImputer().get_params()["strategy"],
    "OneHotEncoder.handle_unknown": OneHotEncoder().get_params()["handle_unknown"],
    "ColumnTransformer": {k: ColumnTransformer([]).get_params()[k] for k in ["remainder", "sparse_threshold"]},
    "DummyClassifier.strategy": DummyClassifier().get_params()["strategy"],
    "HistGradientBoostingClassifier": {k: HistGradientBoostingClassifier().get_params()[k] for k in ["learning_rate", "max_iter", "early_stopping"]},
    "RandomForestClassifier.n_estimators": RandomForestClassifier().get_params()["n_estimators"],
    "StandardScaler": StandardScaler().get_params(),
}
API["valeurs par défaut"] = defauts
assert defauts["LogisticRegression"] == {"C": 1.0, "penalty": "l2", "max_iter": 100, "solver": "lbfgs"}
assert defauts["SimpleImputer.strategy"] == "mean" and defauts["OneHotEncoder.handle_unknown"] == "error"
assert defauts["ColumnTransformer"] == {"remainder": "drop", "sparse_threshold": 0.3}
assert defauts["DummyClassifier.strategy"] == "prior" and defauts["RandomForestClassifier.n_estimators"] == 100
assert defauts["HistGradientBoostingClassifier"] == {"learning_rate": 0.1, "max_iter": 100, "early_stopping": "auto"}
import inspect  # noqa: E402
assert inspect.signature(train_test_split).parameters["test_size"].default is None   # None : 0,25 quand train_size n'est pas donné
a_, b_ = train_test_split(np.arange(100))
assert len(b_) == 25
API["train_test_split : 25 % en test par défaut, lignes mélangées"] = sorted(b_.tolist()) != b_.tolist()

# Encodeurs.
assert leve(lambda: OneHotEncoder().fit([["a"], ["b"]]).transform([["c"]]), ValueError) == "ValueError"
assert OneHotEncoder(handle_unknown="ignore").fit([["a"], ["b"]]).transform([["c"]]).toarray().tolist() == [[0, 0]]
import scipy.sparse  # noqa: E402
assert scipy.sparse.issparse(OneHotEncoder().fit_transform([["a"], ["b"]]))
API["OneHotEncoder(sparse=False)"] = leve(lambda: OneHotEncoder(sparse=False).fit_transform([["a"], ["b"]]), TypeError) or "accepté"
API["OneHotEncoder(sparse_output=False)"] = leve(lambda: OneHotEncoder(sparse_output=False).fit_transform([["a"], ["b"]]), TypeError) or "accepté"
if REFERENCE:
    assert API["OneHotEncoder(sparse=False)"] == "accepté" and API["OneHotEncoder(sparse_output=False)"] == "TypeError"
assert list(OneHotEncoder().fit(pd.DataFrame({"f": ["b", "a"]})).get_feature_names_out()) == ["f_a", "f_b"]
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit([["bas"], ["haut"], ["moyen"]])
assert oe.transform([["moyen"], ["inconnu"]]).ravel().tolist() == [2.0, -1.0]          # ordre alphabétique par défaut
oe2 = OrdinalEncoder(categories=[["bas", "moyen", "haut"]]).fit([["bas"], ["haut"], ["moyen"]])
assert oe2.transform([["bas"], ["moyen"], ["haut"]]).ravel().tolist() == [0.0, 1.0, 2.0]
API["OrdinalEncoder : ordre alphabétique par défaut, ordre imposé par categories"] = True

# Ce que les modèles refusent.
X_nan = np.array([[1.0], [np.nan], [3.0], [4.0], [5.0], [6.0]])
assert leve(lambda: LogisticRegression().fit(X_nan, y_bin), ValueError) == "ValueError"
assert leve(lambda: LogisticRegression().fit(np.array([["a"], ["b"]] * 3, dtype=object), y_bin), ValueError) == "ValueError"
assert leve(lambda: StandardScaler().fit(np.array([["a"], ["b"]], dtype=object)), ValueError) == "ValueError"
HistGradientBoostingClassifier(random_state=0, min_samples_leaf=1).fit(X_nan, y_bin)      # accepte les NaN
API["RandomForestClassifier accepte les NaN"] = leve(lambda: RandomForestClassifier(5, random_state=0).fit(X_nan, y_bin), ValueError) is None
if REFERENCE:
    assert API["RandomForestClassifier accepte les NaN"] is False
assert leve(lambda: HistGradientBoostingClassifier().fit(scipy.sparse.csr_matrix(np.tile(X_num, (5, 1))), np.tile(y_bin, 5)), TypeError, ValueError) is not None
API["HistGradientBoostingClassifier refuse une matrice creuse"] = True
API["HistGradientBoostingClassifier a un paramètre class_weight"] = "class_weight" in HistGradientBoostingClassifier().get_params()
API["Pipeline a une méthode set_output"] = hasattr(Pipeline([("m", LogisticRegression())]), "set_output")
if REFERENCE:
    assert not API["HistGradientBoostingClassifier a un paramètre class_weight"] and not API["Pipeline a une méthode set_output"]
assert "class_weight" in LogisticRegression().get_params() and "class_weight" in RandomForestClassifier().get_params()

# ColumnTransformer : colonnes non citées retirées ; sortie creuse ou dense selon la densité.
df_ct = pd.DataFrame({"x": [1.0, 2.0, 3.0], "f": ["a", "b", "c"], "id": [7, 8, 9]})
ct = ColumnTransformer([("num", StandardScaler(), ["x"]), ("cat", OneHotEncoder(handle_unknown="ignore"), ["f"])])
assert ct.fit_transform(df_ct).shape == (3, 4)                                         # id retiré
assert list(ct.get_feature_names_out()) == ["num__x", "cat__f_a", "cat__f_b", "cat__f_c"]
large = pd.DataFrame({"x": np.arange(40.0), "f": [f"m{k}" for k in range(40)]})
assert scipy.sparse.issparse(ct.fit_transform(large))                                   # densité 2/41 < 0,3
assert not scipy.sparse.issparse(ColumnTransformer(ct.transformers, sparse_threshold=0).fit_transform(large))
API["ColumnTransformer : sortie creuse sous la densité 0,3, dense avec sparse_threshold=0"] = True

# Validation : cv entier = StratifiedKFold sans mélange pour un classifieur ; stratify conserve les proportions.
yy = np.array([0] * 90 + [1] * 10)
XX = np.arange(100.0).reshape(-1, 1)
_, _, _, y_val = train_test_split(XX, yy, test_size=0.2, stratify=yy, random_state=0)
assert y_val.sum() == 2
plis = list(StratifiedKFold(5).split(XX, yy))
assert all(yy[v].sum() == 2 for _, v in plis)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    s1 = cross_val_score(DummyClassifier(strategy="most_frequent"), XX, yy, cv=5)
    s2 = cross_val_score(DummyClassifier(strategy="most_frequent"), XX, yy, cv=StratifiedKFold(5))
assert s1.tolist() == s2.tolist() == [0.9] * 5
assert DummyClassifier(strategy="most_frequent").fit(XX, yy).score(XX, yy) == 0.9      # score d'un classifieur = exactitude
API["cross_val_score(cv=5) sur un classifieur : plis stratifiés, score = exactitude"] = True

# AUC : calculée sur des scores, pas sur des classes ; invariante par transformation croissante des scores.
scores = np.array([0.1, 0.3, 0.4, 0.9])
vrais = np.array([0, 0, 1, 1])
assert roc_auc_score(vrais, scores) == 1.0 == roc_auc_score(vrais, scores ** 3)        # les deux positifs au-dessus des deux négatifs
assert roc_auc_score(vrais, (scores >= 0.5).astype(int)) == 0.75                         # sur des classes : l'ordre fin est perdu
API["roc_auc_score : invariant par transformation croissante des scores ; plus faible ici sur des classes que sur des scores"] = True

# Un arbre est insensible à une transformation croissante d'une variable ; pas une régression logistique pénalisée.
rng = np.random.default_rng(0)
Xa = rng.standard_normal((300, 3)) * [1.0, 50.0, 0.01]
ya = (Xa[:, 0] + Xa[:, 1] / 50 + 100 * Xa[:, 2] + 0.5 * rng.standard_normal(300) > 0).astype(int)
Xb = Xa * [1000.0, 0.001, 7.0]
pa = DecisionTreeClassifier(random_state=0).fit(Xa[:200], ya[:200]).predict(Xa[200:])
pb = DecisionTreeClassifier(random_state=0).fit(Xb[:200], ya[:200]).predict(Xb[200:])
assert (pa == pb).all()
la = LogisticRegression(max_iter=10000).fit(Xa[:200], ya[:200]).predict_proba(Xa[200:])[:, 1]
lb = LogisticRegression(max_iter=10000).fit(Xb[:200], ya[:200]).predict_proba(Xb[200:])[:, 1]
assert np.abs(la - lb).max() > 0.05
API["arbre invariant au changement d'échelle, régression logistique pénalisée non"] = True

# Expérience de la fuite (leçon, partie 6) : la lecture qu'en fait la leçon (plus de cinq écarts-types de 0,05 au-dessus
# de 1/2 avec fuite, moins de deux sans) est contrôlée dans l'environnement du notebook, où la leçon est exécutée.
rng_f = np.random.default_rng(0)
bruit = rng_f.standard_normal((100, 2000))
pile_ou_face = rng_f.integers(0, 2, 100)
choisies = SelectKBest(f_classif, k=20).fit_transform(bruit, pile_ou_face)
avec_fuite = cross_val_score(LogisticRegression(max_iter=1000), choisies, pile_ou_face, cv=5).mean()
sans_fuite = cross_val_score(Pipeline([("selection", SelectKBest(f_classif, k=20)), ("modele", LogisticRegression(max_iter=1000))]),
                             bruit, pile_ou_face, cv=5).mean()
API["expérience de la fuite : exactitude avec et sans fuite"] = [round(float(avec_fuite), 3), round(float(sans_fuite), 3)]
assert math.sqrt(0.25 / 100) == 0.05
if REFERENCE:
    assert avec_fuite > 0.5 + 5 * 0.05 and abs(sans_fuite - 0.5) < 2 * 0.05

# La chaîne de la leçon fonctionne de bout en bout sur ses fichiers, modalité inconnue comprise.
a = nettoyer(relu)
b = nettoyer(pd.read_csv(DOSSIER / "abonnes_test.csv"))
num, cat = _communs.NUMERIQUES, ["forfait"]
prep = ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("echelle", StandardScaler())]), num),
                          ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
modele = Pipeline([("preparation", prep), ("modele", LogisticRegression(max_iter=1000))]).fit(a[num + cat], a["resilie"])
acc_lecon = float((modele.predict(b[num + cat]) == y_test_l).mean())
maj_lecon = float(max(y_test_l.mean(), 1 - y_test_l.mean()))
assert acc_lecon > maj_lecon + 0.2
assert leve(lambda: Pipeline([("p", ColumnTransformer([("cat", OneHotEncoder(), cat)])), ("m", LogisticRegression())])
            .fit(a[cat], a["resilie"]).predict(b[cat]), ValueError) == "ValueError"
V["lecon_exactitude_test"] = round(acc_lecon, 4)
V["lecon_majoritaire_test"] = round(maj_lecon, 4)
dire(f"chaîne de la leçon sur ses fichiers : exactitude {V['lecon_exactitude_test']} sur le test, classe majoritaire {V['lecon_majoritaire_test']}")
dire(f"faits d'interface vérifiés sous scikit-learn {sklearn.__version__} et pandas {pd.__version__} : {len(API) - 3} groupes")

ecrire(RACINE / "verif" / "seance_code_3.out.json", json.dumps(V, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
ecrire(RACINE / "verif" / "seance_code_3.out.txt", "\n".join(LIGNES_TEXTE[:-1]) + "\n")   # sans la ligne des versions, propre à l'environnement
ecrire(RACINE / "verif" / f"seance_code_3.api.{sklearn.__version__}.out.json", json.dumps(API, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
