"""Question T9-C1 : évaluation du fichier submissions.csv écrit par le notebook.

Le noyau appelle evaluer(mode, espace) (voir outils/noyau.py) ; le répertoire courant est le dossier de la question.
Les étiquettes du jeu de test ne sont dans aucun fichier : elles sont régénérées par la graine de generer_donnees.py.

Contrôles visibles (5) : le format du fichier, sans aucune information sur la qualité des prédictions.
Contrôles cachés (5) : le format complet, puis l'exactitude au-dessus de quatre seuils croissants. Les seuils sont
fixés d'après les mesures de verif/code_test_9_octobre.py (exactitudes de la classe majoritaire, d'une solution
bâclée, d'une solution soignée sans variable ajoutée et de la correction).

Usage autonome : python tests_test9oct.py reference
(exécute la correction dans un dossier temporaire, puis l'évalue : aucun submissions.csv n'est laissé dans le dépôt)
"""
import os
import runpy
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

import generer_donnees

DOSSIER = Path(__file__).resolve().parent
FICHIER = "submissions.csv"
COLONNES = ["id", generer_donnees.CIBLE]
SEUILS = [0.60, 0.72, 0.78, 0.85]
N_VISIBLES, N_CACHES = 5, 1 + len(SEUILS)

_DONNEES = generer_donnees.fabriquer()
IDS_TEST = _DONNEES["ids_test"]
_ETIQUETTES = dict(zip(IDS_TEST, _DONNEES["y_test"]))


def _resultat(nom, attendu, obtenu, ok, debut, erreur=None):
    return {"nom": nom, "attendu": attendu, "obtenu": obtenu, "erreur": erreur, "ok": bool(ok),
            "duree_s": round(time.perf_counter() - debut, 3)}


def _controles_format():
    """Les cinq contrôles de format, dans l'ordre, et le cadre lu (None si le fichier est illisible).

    Chaque contrôle suppose les précédents réussis : après un échec, les suivants échouent avec « non évalué »."""
    debut = time.perf_counter()
    resultats = []
    df = None
    try:
        if not os.path.isfile(FICHIER):
            raise FileNotFoundError("fichier absent du dossier de travail")
        df = pd.read_csv(FICHIER)
        resultats.append(_resultat(f"le fichier {FICHIER} existe et se lit comme un CSV", "fichier lisible", "fichier lisible", True, debut))
    except Exception as e:  # noqa: BLE001 : tout défaut de lecture est un échec du contrôle, jamais une erreur du noyau
        resultats.append(_resultat(f"le fichier {FICHIER} existe et se lit comme un CSV", "fichier lisible", f"{type(e).__name__} : {e}"[:200], False, debut))
    etapes = [
        ("colonnes : exactement id puis panne", repr(COLONNES),
         lambda: (repr(list(df.columns))[:200], list(df.columns) == COLONNES)),
        ("nombre de lignes : une par ligne de test.csv", str(len(IDS_TEST)),
         lambda: (str(len(df)), len(df) == len(IDS_TEST))),
        ("identifiants : ceux de test.csv, chacun une seule fois (ordre libre)", f"les {len(IDS_TEST)} identifiants de test.csv",
         lambda: _identifiants(df)),
        ("valeurs de panne : 0 ou 1, aucune manquante", "0 ou 1",
         lambda: _valeurs(df)),
    ]
    valide = resultats[0]["ok"]
    for nom, attendu, controle in etapes:
        debut = time.perf_counter()
        if not valide:
            resultats.append(_resultat(nom, attendu, "non évalué : un contrôle précédent a échoué", False, debut))
            continue
        try:
            obtenu, ok = controle()
        except Exception as e:  # noqa: BLE001
            obtenu, ok = f"{type(e).__name__} : {e}"[:200], False
        valide = valide and ok
        resultats.append(_resultat(nom, attendu, obtenu, ok, debut))
    return resultats, (df if valide else None)


def _identifiants(df):
    ids = df["id"]
    if not pd.api.types.is_numeric_dtype(ids) or ids.isna().any():
        return "identifiants non numériques ou manquants", False
    obtenus, attendus = sorted(ids.tolist()), sorted(IDS_TEST)
    if obtenus == attendus:
        return f"les {len(attendus)} identifiants de test.csv", True
    inconnus = len(set(obtenus) - set(attendus))
    absents = len(set(attendus) - set(obtenus))
    return f"{inconnus} identifiant(s) inconnu(s), {absents} absent(s), {len(obtenus) - len(set(obtenus))} doublon(s)", False


def _valeurs(df):
    v = df[COLONNES[1]]
    if pd.api.types.is_bool_dtype(v) or not pd.api.types.is_numeric_dtype(v):
        return f"colonne de type {v.dtype}", False
    if v.isna().any():
        return f"{int(v.isna().sum())} valeur(s) manquante(s)", False
    hors = int((~v.isin([0, 1])).sum())
    return ("0 ou 1" if hors == 0 else f"{hors} valeur(s) autre(s) que 0 ou 1"), hors == 0


def exactitude(df):
    """Proportion d'identifiants de test.csv dont la prédiction est juste (df au format valide)."""
    predit = dict(zip(df["id"].tolist(), df[COLONNES[1]].tolist()))
    return sum(int(predit[i]) == c for i, c in _ETIQUETTES.items()) / len(_ETIQUETTES)


def evaluer(mode, espace):  # espace (variables du notebook) n'est pas utilisé : seul le fichier écrit est jugé
    resultats, df = _controles_format()
    if mode == "visibles":
        return resultats
    debut = time.perf_counter()
    caches = [_resultat("format complet du fichier (les cinq contrôles visibles)", "5/5",
                        f"{sum(r['ok'] for r in resultats)}/5", df is not None, debut)]
    acc = exactitude(df) if df is not None else None
    for seuil in SEUILS:
        debut = time.perf_counter()
        caches.append(_resultat(f"exactitude sur le jeu de test au moins égale à {seuil:.2f}", f">= {seuil:.2f}",
                                "format invalide" if acc is None else f"{acc:.4f}", acc is not None and acc >= seuil, debut))
    return caches


def verifier_fichiers():
    """Les fichiers de données présents dans le dossier sont exactement ceux que le générateur produit."""
    for nom, texte in generer_donnees.fichiers().items():
        assert (DOSSIER / nom).read_bytes() == texte.encode("utf-8"), f"{nom} ne correspond plus au générateur"


def evaluer_script(chemin):
    """Exécute un script de solution dans un dossier temporaire contenant les données, puis rend
    (résultats visibles, résultats cachés, exactitude ou None, variables du script)."""
    ici = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        for nom in ["train.csv", "test.csv", "sample_submission.csv"]:
            shutil.copy(DOSSIER / nom, tmp)
        os.chdir(tmp)
        try:
            variables = runpy.run_path(str(chemin), run_name="__main__")
            visibles, caches = evaluer("visibles", {}), evaluer("caches", {})
            _, df = _controles_format()
            acc = exactitude(df) if df is not None else None
        finally:
            os.chdir(ici)
    return visibles, caches, acc, variables


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "reference":
    verifier_fichiers()
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):   # la correction affiche son inspection des données
        visibles, caches, acc, _ = evaluer_script(DOSSIER / "correction_test9oct.py")
    assert len(visibles) == N_VISIBLES and len(caches) == N_CACHES
    ok = sum(r["ok"] for r in visibles + caches)
    print(f"{ok}/{N_VISIBLES + N_CACHES} ; visibles {N_VISIBLES}, cachés {N_CACHES} ; exactitude de la correction {acc:.4f}")
    sys.exit(0 if ok == N_VISIBLES + N_CACHES else 1)
