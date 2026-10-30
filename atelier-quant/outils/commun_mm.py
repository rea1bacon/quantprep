"""Fabrique des questions de code des séances de market making et des examens Millennium (coding/millennium/).

Le nombre de tests de chaque question est lu dans coding/millennium/_suites.py, jamais écrit à la main.
"""
import sys
from pathlib import Path

DOSSIER = Path(__file__).resolve().parent.parent / "coding" / "millennium"
sys.path.insert(0, str(DOSSIER))
from _suites import SUITES  # noqa: E402

CHEMIN = "import sys\nsys.path.insert(0, \".\")   # rend importable le paquet fourni, dans le dossier de la question\n"
IMPORTS = CHEMIN + "import numpy as np\nimport pandas as pd"
ABANDON = ("Règle d'abandon : au-delà du double du temps cible d'un exercice, soumettre ce qui existe et passer au suivant.")
AVERTISSEMENT_VERSIONS = ("Le notebook s'exécute sous Python 3.9, pandas 1.4.2 et numpy 1.21.6. Le paquet "
                          "`amplify_quant_trading` fourni dans le dossier est une imitation écrite pour l'entraînement : "
                          "ce n'est pas le code d'un éditeur.")


def _fonctionnement(visibles, tous):
    return (f"\n\n**Fonctionnement.** **Tests visibles** exécute {visibles} test(s). **Soumettre** exécute les {tous} tests de la "
            "question, visibles compris, et la note est la proportion de tests réussis. Le message d'un test échoué dit ce qui "
            "était attendu. Les classes et fonctions jugées sont celles des cellules exécutées en dernier.\n")


def code(ident, section, theme, difficulte, temps_s, enonce, cle, cellules, correction, intro):
    visibles, tous = SUITES[cle].nombres()
    return {
        "id": ident, "type": "code", "section": section, "theme": theme, "difficulte": difficulte, "temps_cible_s": temps_s,
        "enonce": enonce + _fonctionnement(visibles, tous), "dossier": "coding/millennium",
        "module_tests": f"coding/millennium/tests_{cle}.py", "cellules_initiales": cellules,
        "tests_visibles": visibles, "tests_caches": tous,
        "fichier_correction": f"coding/millennium/corrections/{correction}", "intro_correction": intro,
        "verif": {"script": f"coding/millennium/tests_{cle}.py"},
    }
