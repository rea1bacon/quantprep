"""Défauts de mise en forme visibles dans les textes affichés (énoncés, options, explications, solutions,
leçons, points de contrôle) de toutes les banques publiées.

Usage : python tests/lint_formatage.py      (code de sortie 1 s'il reste un défaut)
Les blocs de code entre triples accents graves et les portions entre \\( \\) sont exclus des contrôles
qui ne les concernent pas.
"""
import json
import re
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
CHAMPS = {"enonce", "solution", "texte", "explication", "contenu", "question", "presentation"}


def textes(objet, chemin=""):
    """Paires (chemin, texte) pour chaque champ affiché."""
    if isinstance(objet, dict):
        for k, v in objet.items():
            if isinstance(v, str) and k in CHAMPS:
                yield f"{chemin}.{k}", v
            elif isinstance(v, (dict, list)):
                yield from textes(v, f"{chemin}.{k}")
    elif isinstance(objet, list):
        for i, v in enumerate(objet):
            yield from textes(v, f"{chemin}[{i}]")


def hors_code(texte):
    """Le texte sans les blocs de code ni les formules."""
    sans_blocs = re.sub(r"```.*?```", " ", texte, flags=re.S)
    return re.sub(r"\\\(.*?\\\)", " ", sans_blocs, flags=re.S)


def defauts(texte):
    d = []
    if texte.count("```") % 2:
        d.append("triple accent grave non apparié")
    t = hors_code(texte)
    sans_code_en_ligne = re.sub(r"`[^`\n]*`", " ", t)
    if re.search(r'\\["\']', t):
        d.append("antislash devant un guillemet : " + re.search(r'.{0,25}\\["\'].{0,25}', t).group(0))
    if re.search(r"\\n", sans_code_en_ligne):
        d.append("\\n littéral")
    for par in re.split(r"\n\s*\n", t):
        if par.count("`") % 2:
            d.append("accent grave non apparié : " + par.strip()[:60])
        if re.sub(r"`[^`]*`", "", par).count("**") % 2:
            d.append("gras non apparié : " + par.strip()[:60])
    for bloc in re.findall(r"(?:^\s*\|.*\n?)+", t, flags=re.M):
        lignes = [l.strip() for l in bloc.strip().split("\n")]
        tailles = {len(l.strip("|").split("|")) for l in lignes}
        if len(tailles) > 1:
            d.append("tableau aux colonnes inégales : " + lignes[0][:60])
    if re.search(r"\\\\\(|\\\\\)", texte):
        d.append("délimiteur mathématique doublement échappé")
    if re.search(r"(?<![\\(])\\\$", t):
        d.append("dollar échappé")
    return d


def main():
    publiees = [e["id"] for e in json.loads((RACINE / "examens" / "publiees.json").read_text(encoding="utf-8"))]
    total = 0
    for ident in publiees:
        banque = json.loads((RACINE / "banque" / f"{ident}.json").read_text(encoding="utf-8"))
        for chemin, texte in textes(banque):
            for defaut in defauts(texte):
                total += 1
                print(f"{ident}{chemin} : {defaut}")
    print(f"{total} défaut(s) de mise en forme")
    return total


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
