"""Construit, pour chaque examen ou séance de EXAMENS, banque/<id>.json et examens/<id>.html.

Chaque module source définit : ID, TITRE, VERIF (sortie JSON de son script de vérification), CONVENTIONS,
OPTIONS, SECTIONS, PRESENTATION, AVERTISSEMENTS, CODES (questions de code) et QUESTIONS (choix multiples),
et éventuellement LECONS (voir outils/lecons.py : leurs exemples de code sont exécutés ici, par le noyau
du notebook, et leur vraie sortie est insérée).
Chaque jeton @CLE@, @CLE:3@ ou @CLE:3m@ d'un texte est remplacé par la valeur calculée correspondante,
de sorte qu'aucun nombre calculé ne soit écrit à la main.
Usage, depuis outils/ : python construire.py
"""
import importlib
import json
import re
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
import lancer  # noqa: E402  (noyau du notebook)
# Séances et examens relus et validés, dans l'ordre d'affichage sur la page d'accueil de lancer.py.
EXAMENS = ["questions_diagnostic", "questions_seance_code_1", "questions_seance_math_1", "questions_seance_ml_1",
           "questions_seance_ml_2", "questions_seance_algebre_1", "questions_seance_math_2", "questions_seance_math_3",
           "questions_seance_code_2", "questions_seance_ml_3",
           "questions_seance_math_4", "questions_seance_quant_1",
           "questions_seance_ml_4", "questions_seance_math_5", "questions_seance_ml_5", "questions_seance_code_3",
           "questions_seance_mm_1", "questions_seance_mm_2", "questions_seance_mm_3", "questions_seance_mm_4", "questions_seance_mm_5",
           "questions_oa_millennium", "questions_oa_millennium_desk", "questions_oa_millennium_numpy",
           "questions_examen_blanc_1", "questions_examen_blanc_2", "questions_test_9_octobre",
           "questions_debrief_test_9_octobre", "questions_fiche_derniere_heure"]
# Page d'accueil : les séances rangées par domaine, dans l'ordre conseillé à l'intérieur de chaque bloc : (titre, note, identifiants).
PARCOURS = [
    ("OA de recrutement", "Évaluations en ligne reconstituées, non officielles : examen en conditions réelles, corrigé et entraînements.",
     ["test-9-octobre", "debrief-test-9-octobre", "oa-millennium", "oa-millennium-desk", "oa-millennium-numpy"]),
    ("Probabilités et statistique", "Conditionnement, lois, espérance, chaînes de Markov, martingales, estimation et tests.",
     ["seance-math-1", "seance-math-2", "seance-math-3", "seance-math-4", "seance-math-5"]),
    ("Algèbre linéaire", "Valeurs propres, matrices semi-définies positives, projections, SVD, ACP.", ["seance-algebre-1"]),
    ("Machine learning", "Des fondamentaux (régression, arbres, validation) aux réseaux et aux modèles probabilistes.",
     ["seance-ml-1", "seance-ml-2", "seance-ml-3", "seance-ml-4", "seance-ml-5"]),
    ("Finance quantitative", "Covariance en grande dimension, coefficient d'information, surapprentissage de backtest, ratio de Sharpe.",
     ["seance-quant-1"]),
    ("Market making", "Écart et sélection adverse, inventaire, couverture, arbitrage statistique, nettoyage de données.",
     ["seance-mm-1", "seance-mm-2", "seance-mm-3", "seance-mm-4", "seance-mm-5"]),
    ("Code et données", "pandas, numpy, alignement temporel, backtest sans regard vers le futur, chaîne de classification scikit-learn.",
     ["seance-code-1", "seance-code-2", "seance-code-3"]),
    ("Examens blancs et révision", "En conditions réelles : chronomètre, pas de pause.",
     ["diagnostic", "examen-blanc-1", "examen-blanc-2", "fiche-derniere-heure"]),
]
JETON = re.compile(r"@([A-Za-z0-9_]+)(?::(\d+)(m?))?@")


def nombre(v, decimales, maths):
    if isinstance(v, list):
        return "(" + " ; ".join(nombre(x, decimales, maths) for x in v) + ")"
    if decimales is not None:
        t = f"{v:.{int(decimales)}f}"
    elif float(v) == int(v):
        t = str(int(v))
    else:
        t = f"{v:.10g}"
    return t.replace(".", "{,}" if maths else ",")


def substituer(texte, valeurs, cles):
    def remplace(m):
        cle, decimales, maths = m.group(1), m.group(2), bool(m.group(3))
        if cle not in valeurs:
            raise KeyError(f"valeur de vérification absente : {cle}")
        cles.add(cle)
        return nombre(valeurs[cle], decimales, maths)

    return JETON.sub(remplace, texte)


def construire_qcm(q, valeurs, script):
    cles = set()
    r = dict(q, type="qcm")
    r["enonce"] = substituer(q["enonce"], valeurs, cles)
    r["solution"] = substituer(q["solution"], valeurs, cles)
    # Par défaut l'option "a" est la seule juste. Une question à plusieurs réponses déclare "correctes" (liste
    # d'identifiants, au moins un) : elle est alors notée tout ou rien, et chaque option, juste ou fausse, porte son explication.
    correctes = q.get("correctes", ["a"])
    assert correctes and set(correctes) <= {o["id"] for o in q["options"]}, f"{q['id']} : correctes incohérent"
    r["multiple"] = "correctes" in q
    r["options"] = [{"id": o["id"], "texte": substituer(o["texte"], valeurs, cles), "correcte": o["id"] in correctes,
                     "explication": substituer(o["explication"], valeurs, cles)} for o in q["options"]]
    r["reponse"] = " ; ".join(o["texte"] for o in r["options"] if o["correcte"])
    r["fiche"] = None  # renseigné quand la fiche du thème existera
    r["verif"] = {"script": script, "cles": sorted(cles)}
    return r


def construire_code(q, valeurs):
    correction = (RACINE / q["fichier_correction"]).read_text(encoding="utf-8")
    intro = q.get("intro_correction", "Correction commentée, avec une version pandas et une version sans bibliothèque externe")
    return dict(q, fiche=None, enonce=substituer(q["enonce"], valeurs, set()), solution=(
        f"{intro} (`{q['fichier_correction']}`) :\n\n```python\n{correction}```"))


# Délai d'un exemple de leçon à la construction : distinct du délai d'une cellule du notebook (lancer.DELAI_CELLULE_S),
# certaines expériences (simulations de réseaux, par exemple) dépassant deux minutes sur une machine chargée.
DELAI_CONSTRUCTION_S = 900
BLOC_EXEC = re.compile(r"```python-exec(-erreur)?\n(.*?)```", re.S)
CONTROLE = re.compile(r"^\[\[controle:([\w-]+)\]\]$", re.M)


def construire_lecon(q, valeurs):
    """Exécute les exemples dans l'ordre, dans un noyau neuf, et insère leur sortie réelle."""
    if not BLOC_EXEC.search(q["contenu"]):
        contenu = q["contenu"]
        noyau = None
    else:
        noyau = lancer.Noyau(lancer.dans_racine(q["dossier"]))
    try:
        def remplace(m):
            attendue, code = bool(m.group(1)), m.group(2).rstrip("\n")
            r = noyau.envoyer({"type": "executer", "code": code, "nom": "exemple"}, DELAI_CONSTRUCTION_S)
            if bool(r.get("erreur")) != attendue:
                raise RuntimeError(f"{q['id']} : exemple {'sans' if attendue else 'avec'} erreur inattendue\n{code}\n{r}")
            if attendue:
                sortie = r["erreur"].strip().splitlines()[-1]
            else:
                sortie = (r["sortie"] + (r["valeur"] or "")).rstrip("\n")
            return f"```python\n{code}\n```\n\n```sortie\n{sortie}\n```"

        if noyau is not None:
            contenu = BLOC_EXEC.sub(remplace, q["contenu"])
    finally:
        if noyau is not None:
            noyau.arreter()
    cites = CONTROLE.findall(contenu)
    assert sorted(cites) == sorted(q.get("controles", {})) and len(cites) == len(set(cites)), f"{q['id']} : points de contrôle incohérents"
    controles = {}
    for cid, c in q.get("controles", {}).items():
        controles[cid] = {"question": substituer(c["question"], valeurs, set()), "options": [
            {"id": o["id"], "texte": substituer(o["texte"], valeurs, set()), "correcte": o["id"] == "a",
             "explication": substituer(o["explication"], valeurs, set())} for o in c["options"]]}
    return dict(q, contenu=substituer(contenu, valeurs, set()), controles=controles)


def construire(nom_module):
    src = importlib.import_module(nom_module)
    valeurs = json.loads((RACINE / src.VERIF).read_text(encoding="utf-8")) if src.VERIF else {}
    script = src.VERIF.replace(".out.json", ".py") if src.VERIF else None
    questions = ([construire_lecon(q, valeurs) for q in getattr(src, "LECONS", [])]
                 + [construire_code(q, valeurs) for q in src.CODES] + [construire_qcm(q, valeurs, script) for q in src.QUESTIONS])
    banque = {"id": src.ID, "conventions": src.CONVENTIONS, "questions": questions}
    (RACINE / "banque").mkdir(exist_ok=True)
    (RACINE / "banque" / f"{src.ID}.json").write_text(json.dumps(banque, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    examen = {
        "id": src.ID, "titre": src.TITRE, "duree_minutes": sum(s["budget_min"] for s in src.SECTIONS),
        # Par défaut, pause et « Je ne sais pas » permis ; un examen blanc les interdit dans ses OPTIONS.
        "options": dict({"pause": True, "je_ne_sais_pas": True}, **src.OPTIONS),
        "conventions": src.CONVENTIONS, "presentation": src.PRESENTATION, "avertissements": src.AVERTISSEMENTS,
        "sections": [dict(s, questions=[q for q in questions if q["section"] == s["id"]]) for s in src.SECTIONS],
    }
    donnees = json.dumps(examen, ensure_ascii=False).replace("</", "<\\/")
    gabarit = (RACINE / "outils" / "gabarit_examen.html").read_text(encoding="utf-8")
    assert gabarit.count("/*DONNEES*/") == 1
    (RACINE / "examens").mkdir(exist_ok=True)
    (RACINE / "examens" / f"{src.ID}.html").write_text(gabarit.replace("/*DONNEES*/", donnees), encoding="utf-8", newline="\n")
    print(f"{src.ID} : {len(questions)} questions, {examen['duree_minutes']} min")


if __name__ == "__main__":
    publiees = []
    for nom in EXAMENS:
        construire(nom)
        src = importlib.import_module(nom)
        publiees.append({"id": src.ID, "titre": src.TITRE, "duree_min": sum(s["budget_min"] for s in src.SECTIONS)})
    # Seules ces pages sont proposées par lancer.py : une séance en cours de production n'y figure pas.
    (RACINE / "examens" / "publiees.json").write_text(json.dumps(publiees, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    # L'accueil cite chaque page publiée exactement une fois.
    ids_parcours = [i for _, _, ids in PARCOURS for i in ids]
    ids_publies = [e["id"] for e in publiees]
    assert len(ids_parcours) == len(set(ids_parcours)) and set(ids_parcours) == set(ids_publies), "parcours incohérent"
    parcours = [{"bloc": b, "note": n, "ids": ids} for b, n, ids in PARCOURS]
    (RACINE / "examens" / "parcours.json").write_text(json.dumps(parcours, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
