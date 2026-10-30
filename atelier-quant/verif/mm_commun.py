"""Outils communs aux scripts de vérification des séances de market making (verif/seance_mm_*.py)."""
import json
import runpy
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
MM = RACINE / "coding" / "millennium"
sys.path.insert(0, str(MM))


def correction(nom):
    return runpy.run_path(str(MM / "corrections" / nom))


def ecrire(nom, valeurs, lignes):
    """Sortie déterministe : JSON trié et texte, identiques octet pour octet d'une exécution à l'autre."""
    (RACINE / "verif" / f"{nom}.out.json").write_bytes((json.dumps(valeurs, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    (RACINE / "verif" / f"{nom}.out.txt").write_bytes(("\n".join(lignes) + "\n").encode("utf-8"))
    print("\n".join(lignes))
