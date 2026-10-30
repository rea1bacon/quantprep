"""Correction de la question T9-C1 (pannes de compresseurs). À n'ouvrir qu'après avoir tenté la question.

Le script se lit de haut en bas, comme les cellules d'un notebook, et s'exécute dans le dossier qui contient
train.csv et test.csv. Chaque étape répond à une consigne de l'énoncé. Tous les nombres sont affichés par le
code : aucun n'est écrit à la main dans les commentaires.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------------------------------ 1. Lire et inspecter
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print("dimensions :", train.shape, test.shape)
print(train.dtypes.to_string())           # une colonne "object" censée être numérique est une colonne sale
print(train.isna().sum().to_string())     # valeurs manquantes déclarées
print(test.isna().sum().to_string())      # le jeu de test en a aussi : il faudra les remplacer, pas supprimer les lignes
print(train["panne"].value_counts(normalize=True).to_string())   # équilibre des classes : il fixe le point de comparaison
print(train.describe().loc[["min", "max"]].to_string())          # un minimum négatif pour une ancienneté : valeur impossible
print(train["materiau"].unique())         # casse et espaces parasites

# ------------------------------------------------------------------ 2. Nettoyer : UNE fonction, appliquée aux deux jeux
# 2a. debit_et_ligne contient deux informations : "<débit><séparateur><ligne>", séparateur parmi ; , _
#     L'expression régulière prend tout jusqu'au DERNIER séparateur (.* est gourmand), puis un chiffre final.
#     ^ ancre le début de la chaîne et \Z sa fin : une valeur qui ne suit pas ce format ne donne aucun groupe.
MOTIF = r"^(.*)[;,_](\d)\Z"
print(sorted(train["debit_et_ligne"].str.extract(MOTIF)[0].unique()))   # toutes les écritures du débit : chiffres et lettres

# 2b. Les débits en lettres sont des mots reliés par des traits d'union : la valeur est la somme des mots.
MOTS = {"vingt": 20, "trente": 30, "quarante": 40, "deux": 2, "quatre": 4, "six": 6, "huit": 8}


def en_nombre(texte):
    """ "32.0" -> 32.0 ; "trente-deux" -> 32.0. Un mot inconnu lève KeyError : mieux vaut une erreur visible
    qu'une valeur fausse silencieuse (le jeu de test pourrait contenir une écriture absente de l'apprentissage)."""
    try:
        return float(texte)
    except ValueError:
        return float(sum(MOTS[mot] for mot in texte.split("-")))


def nettoyer(df):
    df = df.copy()
    parties = df["debit_et_ligne"].str.extract(MOTIF)
    assert parties.notna().all().all(), "une valeur de debit_et_ligne ne suit pas le format attendu"
    df["debit"] = parties[0].map(en_nombre)
    df["ligne"] = parties[1]                      # laissée en chaîne : c'est une étiquette, pas une quantité
    df = df.drop(columns="debit_et_ligne")
    # 2c. Valeur impossible : une ancienneté négative est une erreur de saisie, donc une valeur manquante.
    df.loc[df["anciennete_ans"] < 0, "anciennete_ans"] = np.nan
    # 2d. Catégories : mêmes modalités écrites différemment ("ACIER", " acier ").
    df["materiau"] = df["materiau"].str.strip().str.lower()
    # 2e. Variable ajoutée (consigne) : la charge, produit de la pression par la durée de fonctionnement.
    #     Une régression logistique additionne des effets colonne par colonne ; elle ne peut pas représenter
    #     « la pression compte d'autant plus que la machine tourne longtemps » sans cette colonne produit.
    df["charge"] = df["pression_bar"] * df["heures_semaine"]
    return df


train_net = nettoyer(train)
test_net = nettoyer(test)
print(train_net["materiau"].value_counts().to_string())
print("manquants après nettoyage :", train_net.isna().sum().sum(), "en apprentissage,", test_net.isna().sum().sum(), "en test")

# ------------------------------------------------------------------ 3. Valeurs manquantes
# Apprentissage : la consigne demande de retirer les lignes incomplètes. À faire APRÈS le nettoyage,
# sinon les anciennetés négatives, qui ne sont pas encore des NaN, restent dans le jeu.
avant = len(train_net)
train_net = train_net.dropna()
print("lignes d'apprentissage retirées :", avant - len(train_net), "sur", avant)
# Test : on ne retire JAMAIS de ligne, chaque identifiant doit recevoir une prédiction. Ses valeurs manquantes
# seront remplacées par la médiane apprise sur l'apprentissage (SimpleImputer, dans la chaîne ci-dessous).

# ------------------------------------------------------------------ 4. Variables, cible, encodage
NUMERIQUES = ["debit", "pression_bar", "temperature_c", "anciennete_ans", "heures_semaine", "charge"]
CATEGORIELLES = ["ligne", "materiau"]
X = train_net[NUMERIQUES + CATEGORIELLES]     # l'identifiant n'est pas une variable explicative
y = train_net["panne"]
X_test = test_net[NUMERIQUES + CATEGORIELLES]


def chaine(modele):
    """Préparation + modèle dans un seul objet : tout ce qui s'apprend (médianes, moyennes, écarts-types, liste des
    modalités) est appris par fit sur les seules lignes d'apprentissage qu'on lui donne, puis réappliqué tel quel."""
    preparation = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("echelle", StandardScaler())]), NUMERIQUES),
        # Encodage un-parmi-n ; une modalité jamais vue à l'apprentissage donne une ligne de zéros au lieu d'une erreur.
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORIELLES),
    ], sparse_threshold=0)   # sortie dense : HistGradientBoostingClassifier refuse les matrices creuses
    return Pipeline([("preparation", preparation), ("modele", modele)])


CANDIDATS = {
    "régression logistique": chaine(LogisticRegression(max_iter=1000)),
    "boosting par histogrammes": chaine(HistGradientBoostingClassifier(random_state=0)),
}

# ------------------------------------------------------------------ 5. Découpage de validation stratifié
# 20 % des lignes sont mises de côté, avec la même proportion de pannes que dans l'ensemble (stratify=y).
# Elles ne servent ni à ajuster ni à choisir : seulement à estimer l'exactitude hors échantillon du modèle retenu.
X_app, X_val, y_app, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

# Point de comparaison : prédire toujours la classe majoritaire de l'apprentissage.
classe_majoritaire = int(y_app.mean() >= 0.5)
exactitude_majoritaire = float((y_val == classe_majoritaire).mean())
print(f"classe majoritaire ({classe_majoritaire}) : exactitude de validation {exactitude_majoritaire:.4f}")

# ------------------------------------------------------------------ 6. Choisir le modèle par validation croisée
# 5 plis stratifiés sur la partie d'apprentissage : à chaque pli, la chaîne entière est réajustée sur 4/5 des lignes.
plis = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores_cv = {}
for nom, modele in CANDIDATS.items():
    s = cross_val_score(modele, X_app, y_app, cv=plis, scoring="accuracy")
    scores_cv[nom] = float(s.mean())
    print(f"{nom} : exactitude en validation croisée {s.mean():.4f} (écart-type entre plis {s.std():.4f})")
# Lecture : un écart entre deux candidats plus petit que l'écart-type entre plis ne départage rien de façon fiable ;
# dans ce cas l'un ou l'autre convient. On prend ici simplement la meilleure moyenne.
meilleur = max(scores_cv, key=scores_cv.get)
print("modèle retenu :", meilleur)

# ------------------------------------------------------------------ 7. Estimer l'exactitude hors échantillon
modele = CANDIDATS[meilleur].fit(X_app, y_app)
exactitude_validation = float(modele.score(X_val, y_val))
print(f"exactitude sur les 20 % mis de côté : {exactitude_validation:.4f}")

# ------------------------------------------------------------------ 8. Réajuster sur tout l'apprentissage, prédire, écrire
modele.fit(X, y)
predictions = modele.predict(X_test)
soumission = pd.DataFrame({"id": test["id"], "panne": predictions.astype(int)})
soumission.to_csv("submissions.csv", index=False)   # index=False : sinon une colonne d'index s'ajoute au fichier

# ------------------------------------------------------------------ 9. Relire le fichier écrit et le contrôler
relu = pd.read_csv("submissions.csv")
modele_de_fichier = pd.read_csv("sample_submission.csv")
assert list(relu.columns) == list(modele_de_fichier.columns), "colonnes différentes de sample_submission.csv"
assert len(relu) == len(test) and relu["id"].tolist() == test["id"].tolist(), "une ligne par identifiant de test.csv"
assert relu["panne"].isin([0, 1]).all() and relu["panne"].notna().all()
print(relu.head().to_string(index=False))
print("part de pannes prédites :", round(float(relu["panne"].mean()), 4))   # à comparer à la part observée en apprentissage
