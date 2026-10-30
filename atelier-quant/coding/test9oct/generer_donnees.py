"""Générateur du jeu de données synthétique de la question de code T9-C1 (compresseurs d'atelier).

Ce jeu est inventé : ce n'est pas celui du test en ligne du 9 octobre 2026. Il en reprend le genre de tâche
(classification binaire à partir de deux fichiers CSV) et les genres de saletés : deux informations dans une
même colonne, séparateur variable, nombres écrits en lettres, valeurs impossibles, valeurs manquantes,
casse et espaces parasites.

Tout est déterministe (graine fixe, numpy seul, fichiers écrits ligne à ligne sans passer par pandas) :
deux exécutions, dans n'importe lequel des deux environnements, écrivent des fichiers identiques octet pour octet.
Les étiquettes du jeu de test ne sont écrites dans aucun fichier : tests_test9oct.py les régénère par la graine.

Usage : python generer_donnees.py      (écrit train.csv, test.csv et sample_submission.csv à côté de ce script)
"""
import math
from pathlib import Path

import numpy as np

DOSSIER = Path(__file__).resolve().parent
GRAINE = 20261009
N_TRAIN, N_TEST = 2400, 1200
PREMIER_ID_TRAIN, PREMIER_ID_TEST = 1, 5001
COLONNES = ["id", "debit_et_ligne", "pression_bar", "temperature_c", "anciennete_ans", "heures_semaine", "materiau"]
CIBLE = "panne"

# Débit nominal : un entier pair de 20 à 40, écrit en chiffres ("32.0") ou en lettres ("trente-deux").
DEBITS = list(range(20, 41, 2))
DIZAINES = {20: "vingt", 30: "trente", 40: "quarante"}
UNITES = {2: "deux", 4: "quatre", 6: "six", 8: "huit"}
SEPARATEURS = [";", ",", "_"]
MATERIAUX = ["acier", "fonte", "alu"]

# Modèle qui fabrique la cible. z est un score ; la probabilité de panne est 1 / (1 + exp(-PENTE * z)).
ORDONNEE = -0.55
EFFET_DEBIT = 0.9          # par écart-type de débit
EFFET_LIGNE = {1: 0.0, 2: 1.2, 3: -0.9, 4: 0.5}   # non monotone : le numéro de ligne est une catégorie, pas une quantité
EFFET_TEMPERATURE = 0.8    # par tranche de 10 °C au-dessus de 70 °C
EFFET_ANCIENNETE = 0.5     # par tranche de 5 ans au-dessus de 10 ans
EFFET_MATERIAU = {"acier": 0.0, "fonte": 0.7, "alu": -0.7}
EFFET_CHARGE = 1.1         # interaction : (pression - 9) / 1.7 multiplié par (heures - 94) / 43
PENTE = 2.2

P_LETTRES = 0.4            # part des débits écrits en lettres
P_TEMPERATURE_MANQUANTE = 0.04
P_ANCIENNETE_MANQUANTE = 0.02
P_ANCIENNETE_NEGATIVE = 0.03
P_MATERIAU_SALE = 0.15


def en_lettres(v):
    """20 -> "vingt", 32 -> "trente-deux"."""
    dizaine, unite = 10 * (v // 10), v % 10
    return DIZAINES[dizaine] + ("-" + UNITES[unite] if unite else "")


def nombre(x):
    """Un flottant à une décimale, ou une chaîne vide pour une valeur manquante."""
    return "" if math.isnan(x) else f"{x:.1f}"


def fabriquer():
    """Renvoie {"train": lignes, "test": lignes, "ids_test": [...], "y_test": [...], "y_train": [...]}.

    Chaque ligne est la liste des champs, déjà sous forme de chaînes, dans l'ordre de COLONNES
    (suivis de la cible pour le jeu d'apprentissage)."""
    rng = np.random.default_rng(GRAINE)
    n = N_TRAIN + N_TEST
    # 1. Vraies valeurs, toutes indépendantes.
    debit = np.array(DEBITS)[rng.integers(0, len(DEBITS), n)]
    ligne = rng.integers(1, 5, n)
    pression = np.round(6 + 6 * rng.random(n), 1)
    temperature = np.round(70 + 10 * rng.standard_normal(n), 1)
    anciennete = np.round(0.5 + 19.5 * rng.random(n), 1)
    heures = rng.integers(20, 169, n)
    materiau = np.array(MATERIAUX)[rng.integers(0, len(MATERIAUX), n)]
    # 2. Cible.
    z = (ORDONNEE + EFFET_DEBIT * (debit - 30) / 6.3 + np.array([EFFET_LIGNE[int(l)] for l in ligne])
         + EFFET_TEMPERATURE * (temperature - 70) / 10 + EFFET_ANCIENNETE * (anciennete - 10) / 5
         + np.array([EFFET_MATERIAU[m] for m in materiau])
         + EFFET_CHARGE * ((pression - 9) / 1.7) * ((heures - 94) / 43))
    y = (rng.random(n) < 1 / (1 + np.exp(-PENTE * z))).astype(int)
    # 3. Saletés, tirées après la cible : elles ne portent aucune information sur elle.
    en_toutes_lettres = rng.random(n) < P_LETTRES
    separateur = rng.integers(0, len(SEPARATEURS), n)
    temperature_manquante = rng.random(n) < P_TEMPERATURE_MANQUANTE
    tirage_anciennete = rng.random(n)
    anciennete_fausse = -np.round(1 + 8 * rng.random(n), 1)   # sans lien avec la vraie valeur : abs() ne la retrouve pas
    tirage_materiau = rng.random(n)
    forme_materiau = rng.integers(0, 3, n)
    lignes = []
    for i in range(n):
        d = en_lettres(int(debit[i])) if en_toutes_lettres[i] else f"{debit[i]:.1f}"
        t = math.nan if temperature_manquante[i] else float(temperature[i])
        if tirage_anciennete[i] < P_ANCIENNETE_MANQUANTE:
            a = math.nan
        elif tirage_anciennete[i] < P_ANCIENNETE_MANQUANTE + P_ANCIENNETE_NEGATIVE:
            a = float(anciennete_fausse[i])
        else:
            a = float(anciennete[i])
        m = str(materiau[i])
        if tirage_materiau[i] < P_MATERIAU_SALE:
            m = [m.capitalize(), m.upper(), " " + m + " "][forme_materiau[i]]
        ident = (PREMIER_ID_TRAIN + i) if i < N_TRAIN else (PREMIER_ID_TEST + i - N_TRAIN)
        lignes.append([str(ident), d + SEPARATEURS[separateur[i]] + str(int(ligne[i])), nombre(float(pression[i])),
                       nombre(t), nombre(a), str(int(heures[i])), m])
    train = [l + [str(int(c))] for l, c in zip(lignes[:N_TRAIN], y[:N_TRAIN])]
    return {"train": train, "test": lignes[N_TRAIN:], "ids_test": [PREMIER_ID_TEST + k for k in range(N_TEST)],
            "y_test": [int(c) for c in y[N_TRAIN:]], "y_train": [int(c) for c in y[:N_TRAIN]]}


def texte_csv(entete, lignes):
    for l in lignes:
        assert not any('"' in c or "\n" in c for c in l)
    # Un champ qui contient une virgule (séparateur "," de debit_et_ligne) est mis entre guillemets, comme le veut le format CSV.
    return "".join(",".join(f'"{c}"' if "," in c else c for c in l) + "\n" for l in [entete] + lignes)


def fichiers():
    """Contenu exact des trois fichiers, {nom: texte}."""
    d = fabriquer()
    exemple = [[str(i), str(v)] for i, v in zip(d["ids_test"][:5], [0, 1, 0, 0, 1])]   # valeurs arbitraires : seul le format compte
    return {"train.csv": texte_csv(COLONNES + [CIBLE], d["train"]), "test.csv": texte_csv(COLONNES, d["test"]),
            "sample_submission.csv": texte_csv(["id", CIBLE], exemple)}


if __name__ == "__main__":
    for nom, texte in fichiers().items():
        (DOSSIER / nom).write_bytes(texte.encode("utf-8"))
        print(f"{nom} : {texte.count(chr(10)) - 1} lignes")
