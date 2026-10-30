"""Fiche de révision express : les parties « Réflexes en 30 secondes » et « À savoir refaire » de chaque leçon
publiée, rassemblées en une seule page, une partie par séance.

Le contenu est extrait des banques déjà construites (banque/<id>.json), donc sans jeton ni exemple à
exécuter : ce module doit rester le dernier de construire.EXAMENS.
"""
import json
import re
from pathlib import Path

import questions_diagnostic as diag

RACINE = Path(__file__).resolve().parent.parent
ID = "fiche-derniere-heure"
TITRE = "Fiche de révision express : les réflexes de chaque leçon"
VERIF = None
CONVENTIONS = diag.CONVENTIONS
OPTIONS = {"retour_arriere": True, "penalite": 0}
SECTIONS = [{"id": "lecon", "nom": "Fiche", "budget_min": 60}]
PRESENTATION = (
    "Une seule page pour une révision express : pour chaque leçon, le tableau « Réflexes en 30 secondes » (quel indice "
    "de l'énoncé appelle quel outil, et le premier geste à faire) et la liste « À savoir refaire ». Une partie par "
    "séance, dans le sommaire en haut.\n\n"
    "Ordre conseillé si le temps manque : régression linéaire, arbres et XGBoost, probabilités, pandas, algèbre linéaire, "
    "puis le reste. Les preuves sont dans les leçons complètes, accessibles depuis l'accueil."
)
AVERTISSEMENTS = [
    "Rappel : un test ou un entretien se passe seul, sans notes ni outil d'aide. Cette fiche se lit avant, jamais pendant.",
]

# Séances dans l'ordre de lecture conseillé par la présentation.
ORDRE = ["seance-ml-1", "seance-ml-2", "seance-math-1", "seance-code-1", "seance-code-2", "seance-algebre-1", "seance-math-2",
         "seance-math-3", "seance-ml-3", "seance-math-4", "seance-quant-1", "seance-ml-4"]
RUBRIQUES = re.compile(r"^##\s+(Réflexes en 30 secondes|À savoir refaire[^\n]*|8\. Pièges, et ce qu'il faut savoir refaire)\s*$", re.M)


def extraire(ident):
    banque = json.loads((RACINE / "banque" / f"{ident}.json").read_text(encoding="utf-8"))
    lecon = next(q for q in banque["questions"] if q["type"] == "lecon")
    contenu = re.sub(r"^\[\[controle:[\w-]+\]\]\s*$", "", lecon["contenu"], flags=re.M)
    # Garde chaque rubrique de synthèse, de son titre jusqu'au prochain titre de même niveau ou de niveau supérieur.
    lignes = contenu.split("\n")
    garde, k = [], 0
    while k < len(lignes):
        m = re.match(r"^(#{2,3})\s+(.*)$", lignes[k])
        if m and re.search(r"Réflexes en 30 secondes|savoir refaire", m.group(2)):
            niveau, fin = len(m.group(1)), k + 1
            while fin < len(lignes) and not re.match(r"^#{2,%d}\s" % niveau, lignes[fin]):
                fin += 1
            corps = "\n".join(l.replace("### ", "**", 1) + "**" if l.startswith("### ") else l for l in lignes[k + 1:fin]).rstrip()
            garde.append(f"### {re.sub(r'^[0-9]+[.] *', '', m.group(2))}\n{corps}\n")
            k = fin
        else:
            k += 1
    assert garde, f"aucune rubrique de synthèse dans la leçon de {ident}"
    titre_seance = lecon["theme"].replace("Leçon ", "").strip()
    return f"## {titre_seance[0].upper()}{titre_seance[1:]}\n\n" + "\n".join(garde)


def _contenu():
    parties = ["## Mode d'emploi\n\nChaque partie reprend la synthèse d'une leçon. Lis le tableau des réflexes, cache la colonne "
               "« premier geste » et vérifie que tu la retrouves, puis parcours la liste « À savoir refaire » : ce que tu ne "
               "sais pas refaire de tête renvoie à la partie correspondante de la leçon complète."]
    parties += [extraire(i) for i in ORDRE]
    return "\n\n".join(parties) + "\n"


LECONS = [{
    "id": "L-FICHE-FINALE", "type": "lecon", "section": "lecon", "theme": "Fiche de révision express",
    "difficulte": 3, "temps_cible_s": 3600, "contenu": _contenu(), "controles": {},
    "verif": {"script": "outils/questions_fiche_derniere_heure.py"},
}]
CODES = []
QUESTIONS = []
