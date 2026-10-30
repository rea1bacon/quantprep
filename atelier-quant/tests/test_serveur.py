"""Test de bout en bout du serveur lancer.py : noyau, tests de code, enregistrement des résultats.

Le serveur tourne sur un port libre avec un dossier de suivi temporaire : le vrai suivi/ n'est pas touché.
Usage : python tests/test_serveur.py
"""
import json
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
import lancer  # noqa: E402

echecs = []


def controle(nom, condition, detail=""):
    print(f"[{'OK' if condition else 'ÉCHEC'}] {nom}" + (f" : {detail}" if not condition else ""))
    if not condition:
        echecs.append(nom)


suivi = Path(tempfile.mkdtemp())
for f in ["erreurs.md", "progression.json"]:
    shutil.copy(RACINE / "suivi" / f, suivi / f)
LIGNES_AVANT = (suivi / "erreurs.md").read_text(encoding="utf-8").splitlines()
serveur = lancer.creer_serveur(0, suivi)
port = serveur.server_address[1]
threading.Thread(target=serveur.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{port}"


def post(chemin, corps):
    requete = urllib.request.Request(BASE + chemin, data=json.dumps(corps).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(requete, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def get(chemin):
    with urllib.request.urlopen(BASE + chemin, timeout=30) as r:
        return r.status, r.read().decode("utf-8")


S = {"session": "test", "dossier": "coding/diagnostic"}
executer = lambda code: post("/api/executer", dict(S, code=code, nom="cellule 1"))
tests = lambda mode: post("/api/tests", dict(S, module="coding/diagnostic/tests.py", mode=mode))

try:
    statut, accueil = get("/")
    controle("page d'accueil avec lien vers le diagnostic", statut == 200 and "/examens/diagnostic.html" in accueil)
    publiees = json.loads((RACINE / "examens" / "publiees.json").read_text(encoding="utf-8"))
    controle("page d'accueil : chaque séance publiée listée une fois", accueil.count('<span class="pastille"></span><a href="/examens/') == len(publiees))
    parcours = json.loads((RACINE / "examens" / "parcours.json").read_text(encoding="utf-8"))
    positions = [accueil.index(f'/examens/{i}.html') for jour in parcours for i in jour["ids"]]
    controle("page d'accueil : séances dans l'ordre du parcours", positions == sorted(positions))
    controle("page d'accueil : erreurs à reprendre affichées", "Erreurs à reprendre aujourd'hui" in accueil)
    statut, page = get("/examens/diagnostic.html")
    controle("page du diagnostic servie", statut == 200 and "Test de positionnement" in page)

    v = post("/api/connecter", S)
    controle("premier contact : noyau neuf", v.get("neuf") is True, v)
    v = post("/api/redemarrer", S)
    venv = (RACINE / ".venv-hackerrank").exists()
    controle("noyau démarré avec les versions HackerRank", not venv or (v.get("pandas") == "1.4.2" and v.get("numpy") == "1.21.6"), v)

    r = executer("x = 1 + 1\nprint('bonjour')\nx")
    controle("sortie et valeur de la dernière expression", r["sortie"] == "bonjour\n" and r["valeur"] == "2" and r["erreur"] is None, r)
    r = executer("x * 3")
    controle("variables conservées entre cellules", r["valeur"] == "6", r)
    v = post("/api/connecter", S)
    controle("reconnexion : même noyau, variables conservées", v.get("neuf") is False and executer("x")["valeur"] == "2", v)
    r = executer("y = 1\n1 / 0")
    controle("exception rapportée avec la ligne de la cellule", "ZeroDivisionError" in (r["erreur"] or "") and "cellule 1" in r["erreur"] and "noyau.py" not in r["erreur"], r)
    r = executer("def f(:\n    pass")
    controle("erreur de syntaxe rapportée", "SyntaxError" in (r["erreur"] or ""), r)
    r = executer("import pandas as pd\ndf = pd.read_csv('prix_signaux.csv')\ndf.shape[1]")
    controle("fichier d'exploration lisible depuis le répertoire du notebook", r["valeur"] == "4", r)

    aide = lambda type_, code, ligne, colonne: post("/api/" + type_, dict(S, code=code, ligne=ligne, colonne=colonne))
    r = aide("completer", "import pandas as pd\npd.read_c", 2, 9)
    controle("complétion d'un module : read_csv", "read_csv" in [c["nom"] for c in r.get("completions", [])], r)
    r = aide("completer", "df.grou", 1, 7)
    controle("complétion d'une variable du noyau : df.groupby", "groupby" in [c["nom"] for c in r.get("completions", [])], r)
    r = aide("signatures", "pd.read_csv(", 1, 12)
    controle("signature de pd.read_csv", r.get("signatures") and "filepath_or_buffer" in r["signatures"][0]["texte"], str(r)[:300])
    r = aide("aide", "pd.read_csv", 1, 5)
    controle("documentation de pd.read_csv", r.get("aide") and "CSV" in r["aide"][0]["doc"], str(r)[:300])
    resultats_occupe = {}
    fil = threading.Thread(target=lambda: resultats_occupe.update(executer("import time\ntime.sleep(3)")))
    fil.start()
    time.sleep(1)
    r = aide("completer", "pd.", 1, 3)
    fil.join()
    controle("complétion pendant une exécution : réponse immédiate « occupé », sans attendre", r.get("occupe") is True, r)

    r = tests("visibles")
    controle("tests sans fonction définie : échec explicite", r["reussis"] == 0 and "n'est pas définie" in r["resultats"][0]["erreur"], r)

    executer("def ic_moyen(df):\n    return 0.0")
    r = tests("caches")
    controle("solution fausse : tests cachés partiellement échoués", r["total"] == 16 and r["reussis"] < 16, r)

    correction = (RACINE / "coding" / "diagnostic" / "_correction.py").read_text(encoding="utf-8")
    r = executer(correction)
    controle("la correction s'exécute dans une cellule", r["erreur"] is None, r)
    r = tests("visibles")
    controle("correction : 3/3 tests visibles", (r["reussis"], r["total"]) == (3, 3), r)
    r = tests("caches")
    controle("correction : 16/16 tests cachés", (r["reussis"], r["total"]) == (16, 16), r)

    executer("def ic_moyen(df):\n    df.drop(df.index, inplace=True)\n    return float('nan')")
    r = tests("caches")
    controle("une fonction qui modifie son entrée n'affecte pas les autres tests", r["total"] == 16, r)

    lancer.DELAI_CELLULE_S = 5
    r = executer("import time\ntime.sleep(30)")
    controle("délai dépassé : noyau redémarré", r.get("redemarre") is True, r)
    lancer.DELAI_CELLULE_S = 120  # délai normal rétabli : le noyau neuf doit pouvoir démarrer sur une machine chargée
    r = executer("x")
    controle("après redémarrage, les variables sont perdues", "NameError" in (r["erreur"] or ""), r)

    resultat = {
        "examen": "essai", "debut": "2026-10-04T10:00:00Z", "duree_s": 600, "options": {"retour_arriere": True, "penalite": 0},
        "questions": [
            {"id": "C1", "type": "code", "section": "code", "theme": "pandas", "temps_s": 300, "temps_cible_s": 900, "points": 0.5,
             "soumissions": 1, "caches_reussis": 8, "caches_total": 16, "visibles_reussis": 3, "code_source": "def f():\n    return 1\n"},
            {"id": "M1", "type": "qcm", "section": "math", "theme": "Bayes", "temps_s": 100, "temps_cible_s": 120, "points": 1,
             "reponse": "a", "reponse_texte": "0,162", "bonne_reponse_texte": "0,162"},
            {"id": "M2", "type": "qcm", "section": "math", "theme": "Bayes", "temps_s": 100, "temps_cible_s": 120, "points": 0,
             "reponse": "b", "reponse_texte": "0,950", "bonne_reponse_texte": "0,162"},
            {"id": "M3", "type": "qcm", "section": "math", "theme": "Marches", "temps_s": 100, "temps_cible_s": 120, "points": 0,
             "reponse": "nsp", "reponse_texte": None, "bonne_reponse_texte": "21"},
            {"id": "M4", "type": "qcm", "section": "math", "theme": "Tests", "temps_s": 100, "temps_cible_s": 120, "points": 0,
             "reponse": ["a", "c"], "reponse_texte": "khi-deux ; Shapiro", "bonne_reponse_texte": "khi-deux ; Kolmogorov"},
        ],
    }
    r = post("/api/resultat", resultat)
    controle("résultat enregistré, 4 erreurs ajoutées", r.get("erreurs_ajoutees") == 4 and (suivi / r["fichier"].replace("suivi/", "")).exists(), r)
    journal = (suivi / "erreurs.md").read_text(encoding="utf-8")
    lignes = [l for l in journal.splitlines()[len(LIGNES_AVANT):] if l.startswith("| 20")]
    controle("lignes du journal bien formées (9 colonnes)", len(lignes) == 4 and all(l.count("|") == 10 for l in lignes), lignes)
    controle("cause « je ne sais pas » renseignée", "lacune de cours" in journal)
    controle("réponse à plusieurs options consignée dans le journal", "khi-deux ; Shapiro" in journal)
    progression = json.loads((suivi / "progression.json").read_text(encoding="utf-8"))
    controle("progression mise à jour", progression["sessions"][-1]["sections"]["math"]["points"] == 1 and progression["themes"]["Bayes"]["tentatives"] == 2, str(progression)[-500:])
    controle("code soumis conservé", len(list((suivi / "code").glob("essai-C1-*.py"))) == 1)
finally:
    serveur.shutdown()
    for n in lancer.NOYAUX.values():
        n.arreter()
    shutil.rmtree(suivi, ignore_errors=True)

print(f"{len(echecs)} échec(s)" if echecs else "Tous les tests du serveur passent.")
sys.exit(1 if echecs else 0)
