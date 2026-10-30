"""Évalue le code d'un notebook sur une suite de tests de tests_source/, par evaluer(mode, espace) (voir outils/noyau.py).

Les suites s'écrivent comme des tests pytest. Elles importent le code évalué depuis le module _cible, reconstruit ici
à chaque évaluation à partir des variables du notebook : une classe redéfinie dans une cellule est donc bien celle qui
est jugée. Mode "visibles" : les tests listés dans `visibles`. Mode "caches" : tous les tests de la question, les
visibles compris.

Usage autonome de chaque module tests_*.py : python tests_xxx.py reference
"""
import contextlib
import importlib.util
import inspect
import io
import runpy
import sys
import time
import traceback
import types
from pathlib import Path

DOSSIER = Path(__file__).resolve().parent
if str(DOSSIER) not in sys.path:
    sys.path.insert(0, str(DOSSIER))


class _Absent:
    def __init__(self, nom):
        self.nom = nom

    def __call__(self, *args, **kwargs):
        raise NameError(f"{self.nom} n'est pas défini : exécute la cellule qui le définit.")


def _cible(espace):
    module = types.ModuleType("_cible")
    module.__dict__.update(espace)
    module.__getattr__ = lambda nom: _Absent(nom)
    sys.modules["_cible"] = module


def _charger(fichier, espace):
    _cible(espace)
    chemin = DOSSIER / "tests_source" / fichier
    spec = importlib.util.spec_from_file_location("suite_" + chemin.stem, chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _message(e):
    if isinstance(e, AssertionError):
        cadre = traceback.extract_tb(e.__traceback__)[-1]
        return (str(e) or f"assertion fausse : {cadre.line}")[:600]
    return "".join(traceback.format_exception_only(type(e), e)).strip()[:600]


def _cas(module, nom):
    fonction = getattr(module, nom)
    if hasattr(fonction, "_parametres"):
        param, valeurs = fonction._parametres
        return [(f"{nom}[{v}]", fonction, {param: v}) for v in valeurs]
    return [(nom, fonction, {})]


def _lancer(module, nom, fonction, fixes):
    r = {"nom": nom, "attendu": "test réussi", "obtenu": None, "erreur": None, "ok": False, "duree_s": 0.0}
    debut = time.perf_counter()
    try:
        arguments = dict(fixes)
        for p in inspect.signature(fonction).parameters:
            if p not in arguments:
                arguments[p] = getattr(module, p)()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fonction(**arguments)
        r["ok"], r["obtenu"] = True, "test réussi"
    except BaseException as e:  # noqa: BLE001 : un code qui échoue est une note, jamais une erreur du noyau
        r["obtenu"], r["erreur"] = "échec", _message(e)
    r["duree_s"] = round(time.perf_counter() - debut, 3)
    return r


class Suite:
    def __init__(self, fichier, correction, visibles, autres):
        self.fichier, self.correction, self.visibles, self.autres = fichier, correction, list(visibles), list(autres)

    def evaluer(self, mode, espace):
        module = _charger(self.fichier, espace)
        noms = self.visibles if mode == "visibles" else self.visibles + self.autres
        return [_lancer(module, n, f, fixes) for nom in noms for n, f, fixes in _cas(module, nom)]

    def nombres(self):
        module = _charger(self.fichier, {})
        return (sum(len(_cas(module, n)) for n in self.visibles),
                sum(len(_cas(module, n)) for n in self.visibles + self.autres))

    def principal(self):
        """Passe la correction sur tous les tests et affiche le bilan attendu par tests/verifier_depot.py."""
        if len(sys.argv) < 2 or sys.argv[1] != "reference":
            return
        espace = runpy.run_path(str(DOSSIER / "corrections" / self.correction))
        resultats = self.evaluer("visibles", espace) + self.evaluer("caches", espace)
        echecs = [(r["nom"], r["erreur"]) for r in resultats if not r["ok"]]
        n_vis, n_tous = self.nombres()
        print(f"{n_vis + n_tous - len(echecs)}/{n_vis + n_tous} ; visibles {n_vis}, cachés {n_tous}")
        for e in echecs:
            print(e)
        sys.exit(1 if echecs else 0)
