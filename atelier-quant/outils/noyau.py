"""Noyau d'exécution Python du notebook, lancé par lancer.py.

Lit une commande JSON par ligne sur l'entrée standard et répond une ligne JSON.
Les variables sont conservées d'une cellule à l'autre, comme dans Jupyter.
Commandes :
  {"type": "executer", "code": "...", "nom": "cellule 1"}
  {"type": "tests", "module": "chemin/tests.py", "mode": "visibles" | "caches"}
  {"type": "versions"}
  {"type": "completer" | "signatures" | "aide", "code": "...", "ligne": 1, "colonne": 0}
     (ligne à partir de 1, colonne à partir de 0 ; analyse par jedi avec les variables du noyau)
"""
import ast
import contextlib
import copy
import importlib.util
import io
import json
import linecache
import os
import sys
import threading
import time
import traceback

try:
    import jedi
except ImportError:  # sans jedi, l'éditeur fonctionne sans autocomplétion
    jedi = None

# Le canal de réponse est une copie du descripteur 1 ; le descripteur 1 lui-même est
# redirigé vers le néant pour qu'aucune écriture de bas niveau ne corrompe le protocole.
CANAL = os.fdopen(os.dup(1), "w", encoding="utf-8")
os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
sys.stdout = io.StringIO()

ESPACE = {"__name__": "__main__"}
LONGUEUR_MAX = 20000


def tronquer(texte, n=LONGUEUR_MAX):
    return texte if len(texte) <= n else texte[:n] + f"\n... ({len(texte) - n} caractères coupés)"


def formater_exception(e):
    tb = e.__traceback__
    while tb is not None and tb.tb_frame.f_code.co_filename == __file__:
        tb = tb.tb_next  # retire les cadres du noyau lui-même
    return "".join(traceback.format_exception(type(e), e, tb))


def executer(code, nom):
    sortie = io.StringIO()
    reponse = {"sortie": "", "valeur": None, "erreur": None}
    nom_fichier = f"<{nom}>"
    linecache.cache[nom_fichier] = (len(code), None, code.splitlines(True), nom_fichier)
    debut = time.perf_counter()
    try:
        arbre = ast.parse(code, filename=nom_fichier, mode="exec")
        derniere = None
        if arbre.body and isinstance(arbre.body[-1], ast.Expr):
            derniere = ast.Expression(arbre.body.pop().value)
        with contextlib.redirect_stdout(sortie), contextlib.redirect_stderr(sortie):
            exec(compile(arbre, nom_fichier, "exec"), ESPACE)
            if derniere is not None:
                valeur = eval(compile(derniere, nom_fichier, "eval"), ESPACE)
                if valeur is not None:
                    reponse["valeur"] = tronquer(repr(valeur))
    except SyntaxError as e:
        reponse["erreur"] = "".join(traceback.format_exception_only(type(e), e))
    except BaseException as e:  # noqa: BLE001 : toute erreur de l'utilisateur est rapportée, jamais fatale
        reponse["erreur"] = formater_exception(e)
    reponse["sortie"] = tronquer(sortie.getvalue())
    reponse["duree_s"] = round(time.perf_counter() - debut, 3)
    return reponse


def tests(chemin, mode):
    dossier = os.path.dirname(os.path.abspath(chemin))
    if dossier not in sys.path:
        sys.path.insert(0, dossier)
    spec = importlib.util.spec_from_file_location("tests_question", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "evaluer"):
        # Évaluation libre (par exemple un fichier de prédictions écrit par le notebook) : le module rend lui-même
        # la liste des résultats, chacun {"nom", "attendu", "obtenu", "erreur", "ok", "duree_s"}.
        resultats = module.evaluer(mode, ESPACE)
        return {"mode": mode, "resultats": resultats, "reussis": sum(bool(r["ok"]) for r in resultats), "total": len(resultats)}
    cas = module.cas_visibles() if mode == "visibles" else module.cas_caches()
    fonction = ESPACE.get(module.FONCTION)
    resultats = []
    for c in cas:
        r = {"nom": c["nom"], "attendu": repr(c["attendu"]), "obtenu": None, "erreur": None, "duree_s": 0.0}
        if not callable(fonction):
            r["erreur"] = f"La fonction {module.FONCTION} n'est pas définie : exécute la cellule qui la définit."
            r["ok"] = False
            resultats.append(r)
            continue
        arguments = copy.deepcopy(c["args"])  # la fonction testée ne peut pas modifier les entrées des autres tests
        debut = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                obtenu = fonction(*arguments)
            r["duree_s"] = round(time.perf_counter() - debut, 3)
            r["obtenu"] = tronquer(repr(obtenu), 300)
            r["ok"] = bool(module.comparer(obtenu, c["attendu"])) and r["duree_s"] <= module.LIMITE_S
            if r["duree_s"] > module.LIMITE_S:
                r["erreur"] = f"Temps dépassé : {r['duree_s']} s pour une limite de {module.LIMITE_S} s."
        except BaseException as e:  # noqa: BLE001
            r["duree_s"] = round(time.perf_counter() - debut, 3)
            r["erreur"] = tronquer(formater_exception(e), 3000)
            r["ok"] = False
        resultats.append(r)
    return {"mode": mode, "resultats": resultats, "reussis": sum(r["ok"] for r in resultats), "total": len(resultats)}


VERROU_JEDI = threading.Lock()  # jedi n'est pas prévu pour plusieurs fils d'exécution simultanés


def prechauffer_jedi():
    """La première analyse de pandas par jedi prend plusieurs secondes : on la fait au démarrage."""
    with VERROU_JEDI:
        try:
            jedi.Interpreter("import pandas as pd; import numpy as np; pd.DataFrame().gr", [{}]).complete()
        except Exception:  # noqa: BLE001
            pass


def analyser(type_, code, ligne, colonne):
    if jedi is None:
        return {"completions": [], "signatures": [], "aide": [], "jedi": False}
    lignes = code.split("\n")
    if not 1 <= ligne <= len(lignes):
        return {"completions": [], "signatures": [], "aide": []}
    colonne = max(0, min(colonne, len(lignes[ligne - 1])))  # jedi refuse une colonne hors de la ligne
    with VERROU_JEDI, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        script = jedi.Interpreter(code, [ESPACE])
        if type_ == "completer":
            return {"completions": [{"nom": c.name, "type": c.type, "description": c.description[:120]}
                                    for c in script.complete(ligne, colonne)[:300]]}
        if type_ == "signatures":
            return {"signatures": [{"texte": sig.to_string(), "parametres": [p.to_string() for p in sig.params],
                                    "index": sig.index, "doc": tronquer(sig.docstring(raw=True), 1500)}
                                   for sig in script.get_signatures(ligne, colonne)[:3]]}
        return {"aide": [{"texte": d.description, "doc": tronquer(d.docstring(), 2500)}
                         for d in script.help(ligne, colonne)[:2] if d.docstring()]}


def versions():
    v = {"python": sys.version.split()[0]}
    for nom in ["numpy", "pandas"]:  # scikit-learn et statsmodels, lents à importer, ne sont pas chargés ici
        try:
            v[nom] = importlib.import_module(nom).__version__
        except Exception:  # noqa: BLE001
            v[nom] = "absent"
    return v


if jedi is not None:
    threading.Thread(target=prechauffer_jedi, daemon=True).start()

for ligne in sys.stdin:
    try:
        commande = json.loads(ligne)
        if commande["type"] == "executer":
            reponse = executer(commande["code"], commande.get("nom", "cellule"))
        elif commande["type"] == "tests":
            reponse = tests(commande["module"], commande["mode"])
        elif commande["type"] == "versions":
            reponse = versions()
        elif commande["type"] in ("completer", "signatures", "aide"):
            reponse = analyser(commande["type"], commande["code"], commande["ligne"], commande["colonne"])
        else:
            reponse = {"erreur": f"commande inconnue : {commande['type']}"}
    except BaseException as e:  # noqa: BLE001
        reponse = {"erreur": "Erreur interne du noyau :\n" + formater_exception(e)}
    CANAL.write(json.dumps(reponse, ensure_ascii=False) + "\n")
    CANAL.flush()
