"""Contrôles automatiques du dépôt. Usage : python tests/verifier_depot.py

1. Réexécute les scripts de verif/ et compare leur sortie à la sortie conservée.
2. Reconstruit la banque et la page du diagnostic et vérifie qu'elles sont à jour.
3. Vérifie la structure des questions (une seule bonne réponse, une explication par option fausse).
4. Vérifie les échappements : aucun caractère de contrôle (antislash LaTeX mangé), aucun jeton
   non remplacé, délimiteurs mathématiques équilibrés, pas de dollar, pas de tiret cadratin,
   esperluettes échappées dans le gabarit, données JSON de la page relisibles à l'identique.
5. Vérifie les renvois internes des fichiers Markdown et des questions.
6. Repasse la correction de la question de code sur ses tests cachés.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
echecs = []


def controle(nom, condition, detail=""):
    print(f"[{'OK' if condition else 'ÉCHEC'}] {nom}" + (f" : {detail}" if detail and not condition else ""))
    if not condition:
        echecs.append(nom)


def lancer(script, *args, cwd=None):
    return subprocess.run([sys.executable, str(script), *args], cwd=cwd or RACINE, env=ENV, capture_output=True, text=True, encoding="utf-8")


def hors_environnement(chemin):
    """Exclut l'environnement virtuel et les caches, qui ne font pas partie du contenu."""
    return not any(partie.startswith(".venv") or partie == "__pycache__" for partie in chemin.relative_to(RACINE).parts)


def fichiers_md():
    return [p for p in RACINE.rglob("*.md") if hors_environnement(p)]


def textes(objet):
    """Toutes les chaînes d'une structure JSON."""
    if isinstance(objet, str):
        yield objet
    elif isinstance(objet, dict):
        for v in objet.values():
            yield from textes(v)
    elif isinstance(objet, list):
        for v in objet:
            yield from textes(v)


# 1. Scripts de vérification -------------------------------------------------
for nom in ["diagnostic", "seance_code_1", "seance_math_1", "seance_ml_1", "seance_ml_2", "seance_algebre_1", "seance_math_3", "seance_math_2", "seance_code_2", "seance_ml_3", "seance_math_4", "seance_quant_1", "seance_ml_4", "examen_blanc_1", "examen_blanc_2", "seance_math_5", "seance_ml_5", "seance_code_3"]:
    avant = (RACINE / "verif" / f"{nom}.out.json").read_text(encoding="utf-8")
    r = lancer(RACINE / "verif" / f"{nom}.py")
    apres = (RACINE / "verif" / f"{nom}.out.json").read_text(encoding="utf-8")
    controle(f"verif/{nom}.py s'exécute sans erreur", r.returncode == 0, r.stderr[-400:])
    controle(f"verif/{nom}.py reproduit la sortie conservée", avant == apres)
venv_python = RACINE / ".venv-hackerrank" / "Scripts" / "python.exe"
for script in ["lecon_pandas.py", "lecon_pandas_2.py"]:
    for interp in [sys.executable, str(venv_python)]:
        r = subprocess.run([interp, str(RACINE / "verif" / script)], cwd=RACINE, env=ENV, capture_output=True, text=True, encoding="utf-8")
        controle(f"verif/{script} : affirmations de la leçon vérifiées ({Path(interp).parent.parent.name})", r.returncode == 0, r.stdout[-300:] + r.stderr[-300:])

# 2. Banques et pages à jour -------------------------------------------------
sys.path.insert(0, str(RACINE / "outils"))
import construire  # noqa: E402

identifiants = [__import__(m).ID for m in construire.EXAMENS]
fichiers = [p for i in identifiants for p in (RACINE / "banque" / f"{i}.json", RACINE / "examens" / f"{i}.html")]
avant = [p.read_text(encoding="utf-8") for p in fichiers]
r = lancer(RACINE / "outils" / "construire.py", cwd=RACINE / "outils")
controle("outils/construire.py s'exécute sans erreur", r.returncode == 0, r.stderr[-400:])
controle("banques et pages à jour", avant == [p.read_text(encoding="utf-8") for p in fichiers])
publiees = [e["id"] for e in json.loads((RACINE / "examens" / "publiees.json").read_text(encoding="utf-8"))]
controle("page d'accueil : exactement les séances validées, dans l'ordre", publiees == identifiants, str(publiees))
# Une séance en cours de production peut avoir sa banque et sa page sans être publiée : on le signale sans échouer.
en_cours = sorted(({p.stem for p in (RACINE / "banque").glob("*.json")} | {p.stem for p in (RACINE / "examens").glob("*.html")}) - set(identifiants))
if en_cours:
    print("[INFO] séances présentes mais non publiées (en production) :", ", ".join(en_cours))

gabarit = (RACINE / "outils" / "gabarit_examen.html").read_text(encoding="utf-8")
html_seul = re.sub(r"<script.*?</script>|<style.*?</style>", "", gabarit, flags=re.S)
controle("esperluettes échappées dans le HTML du gabarit", not re.search(r"&(?!amp;|lt;|gt;|quot;|#\d+;)", html_seul))
controle("MathJax chargé depuis cdnjs", "https://cdnjs.cloudflare.com/ajax/libs/mathjax/" in gabarit)
controle("blocs de code exclus de MathJax", "ignoreHtmlClass: 'sans-math'" in gabarit and "'pre', 'code'" in gabarit)

tous_ids = []
for ident in identifiants:
    # 3. Structure des questions ---------------------------------------------
    banque = json.loads((RACINE / "banque" / f"{ident}.json").read_text(encoding="utf-8"))
    qcm = [q for q in banque["questions"] if q["type"] == "qcm"]
    codes = [q for q in banque["questions"] if q["type"] == "code"]
    lecons = [q for q in banque["questions"] if q["type"] == "lecon"]
    tous_ids += [q["id"] for q in banque["questions"]]
    champs = {"id", "type", "section", "theme", "difficulte", "temps_cible_s", "enonce", "solution", "verif"}
    controle(f"[{ident}] champs obligatoires présents", all(champs <= set(q) for q in qcm + codes))
    champs_lecon = {"id", "type", "section", "theme", "temps_cible_s", "contenu", "controles"}
    controle(f"[{ident}] champs des leçons présents", all(champs_lecon <= set(q) for q in lecons))
    controles_lecons = [c for q in lecons for c in q["controles"].values()]
    controle(f"[{ident}] points de contrôle : 4 options dont une correcte, une explication propre à chacune",
             all(len(c["options"]) == 4 and sum(o["correcte"] for o in c["options"]) == 1 and all(len(o["explication"]) > 30 for o in c["options"])
                 and len({o["explication"] for o in c["options"]}) == 4 for c in controles_lecons))
    controle(f"[{ident}] chaque point de contrôle est placé une fois dans sa leçon",
             all(sorted(re.findall(r"^\[\[controle:([\w-]+)\]\]$", q["contenu"], flags=re.M)) == sorted(q["controles"]) for q in lecons))
    controle(f"[{ident}] aucun exemple de leçon laissé sans exécution", not any("```python-exec" in q["contenu"] for q in lecons))
    controle(f"[{ident}] difficulté entre 1 et 5", all(1 <= q["difficulte"] <= 5 for q in banque["questions"]))
    controle(f"[{ident}] 3 à 5 options ; une seule correcte, ou au moins une si la question est à plusieurs réponses",
             all(3 <= len(q["options"]) <= 5 and (sum(o["correcte"] for o in q["options"]) >= 1 if q.get("multiple") else sum(o["correcte"] for o in q["options"]) == 1) for q in qcm))
    controle(f"[{ident}] textes d'options distincts", all(len({o["texte"] for o in q["options"]}) == len(q["options"]) for q in qcm))
    controle(f"[{ident}] chaque option fausse a sa propre explication (chaque option, pour une question à plusieurs réponses)",
             all(len(o["explication"]) > 40 for q in qcm for o in q["options"] if q.get("multiple") or not o["correcte"])
             and all(len({o["explication"] for o in q["options"]}) == len(q["options"]) for q in qcm))
    controle(f"[{ident}] chaque solution commence par la réponse", all(q["solution"].startswith("Réponse :") for q in qcm))
    if qcm:
        valeurs = json.loads((RACINE / qcm[0]["verif"]["script"].replace(".py", ".out.json")).read_text(encoding="utf-8"))
        controle(f"[{ident}] clés de vérification existantes", all(c in valeurs for q in qcm for c in q["verif"]["cles"]))
        controle(f"[{ident}] chaque question numérique cite au moins une valeur vérifiée", all(q["verif"]["cles"] for q in qcm if q["id"] != "DIAG-M6" and not q.get("conceptuelle")))
    for q in codes:
        for chemin in [q["dossier"], q["module_tests"], q["fichier_correction"], q["verif"]["script"]]:
            controle(f"[{ident}] {q['id']} : {chemin} existe", (RACINE / chemin).exists())

    # 4. Échappements --------------------------------------------------------
    chaines = list(textes(banque))
    controle(f"[{ident}] aucun caractère de contrôle (antislash LaTeX perdu)", not any(re.search(r"[\x00-\x08\x0b-\x1f]", t) for t in chaines))
    controle(f"[{ident}] aucun jeton @...@ non remplacé", not any(re.search(r"@[A-Za-z0-9_]+(:\d+m?)?@", t) for t in chaines))
    controle(f"[{ident}] aucun dollar dans les textes", not any("$" in t for t in chaines))
    controle(f"[{ident}] délimiteurs mathématiques équilibrés", all(t.count(r"\(") == t.count(r"\)") for t in chaines))
    hors_code = [re.sub(r"```.*?```", "", t, flags=re.S) for t in chaines]
    controle(f"[{ident}] commandes LaTeX seulement à l'intérieur des délimiteurs",
             not any(re.search(r"\\[a-zA-Z]", re.sub(r"\\\(.*?\\\)", "", t, flags=re.S)) for t in hors_code))
    page = (RACINE / "examens" / f"{ident}.html").read_text(encoding="utf-8")
    m = re.search(r'<script type="application/json" id="donnees">(.*?)</script>', page, flags=re.S)
    donnees = json.loads(m.group(1)) if m else None
    controle(f"[{ident}] données de la page relisibles", donnees is not None)
    if donnees:
        relues = [q for s in donnees["sections"] for q in s["questions"]]
        controle(f"[{ident}] la page contient exactement les questions de la banque", sorted(relues, key=lambda q: q["id"]) == sorted(banque["questions"], key=lambda q: q["id"]))  # la page range par section
        controle(f"[{ident}] aucune fermeture de balise script dans les données", "</" not in m.group(1))
controle("identifiants de questions uniques dans tout le dépôt", len(tous_ids) == len(set(tous_ids)))
fichiers_texte = [p for p in RACINE.rglob("*") if p.suffix in {".md", ".html", ".json", ".py"} and hors_environnement(p)]
avec_cadratin = [str(p.relative_to(RACINE)) for p in fichiers_texte if chr(0x2014) in p.read_text(encoding="utf-8")]
controle("aucun tiret cadratin dans le dépôt", not avec_cadratin, ", ".join(avec_cadratin))

# 5. Renvois internes --------------------------------------------------------
manquants = []
for md in fichiers_md():
    contenu = md.read_text(encoding="utf-8")
    for cible in re.findall(r"\]\((?!https?://)([^)#]+)", contenu):
        if not (md.parent / cible).exists():
            manquants.append(f"{md.name} -> {cible}")
    if md.name == "JOURNAL_MODIFICATIONS.md":
        continue  # l'historique nomme légitimement des fichiers supprimés
    for cible in re.findall(r"`((?:verif|outils|coding|examens|banque|suivi|tests)/[\w./-]+\.\w+)`", contenu):
        if not (RACINE / cible).exists():
            manquants.append(f"{md.name} -> {cible}")
for q in banque["questions"]:
    if not (RACINE / q["verif"]["script"]).exists():
        manquants.append(f"{q['id']} -> {q['verif']['script']}")
controle("renvois internes valides", not manquants, ", ".join(manquants))

# 6. Question de code et serveur ---------------------------------------------
venv = RACINE / ".venv-hackerrank" / "Scripts" / "python.exe"
modules_tests = {}
for ident in identifiants:
    for q in json.loads((RACINE / "banque" / f"{ident}.json").read_text(encoding="utf-8"))["questions"]:
        if q["type"] == "code":
            modules_tests[q["module_tests"]] = (q["tests_visibles"], q["tests_caches"])
for chemin, (n_vis, n_cach) in modules_tests.items():
    module = RACINE / chemin
    for interp, nom in [(sys.executable, "Python local"), (str(venv), "environnement HackerRank")]:
        r = subprocess.run([interp, module.name, "reference"], cwd=module.parent, env=ENV, capture_output=True, text=True, encoding="utf-8")
        attendu = f"{n_vis + n_cach}/{n_vis + n_cach} ; visibles {n_vis}, cachés {n_cach}"
        controle(f"{chemin} : la correction passe ses {n_vis} + {n_cach} tests ({nom})", r.returncode == 0 and attendu in r.stdout, r.stdout[-300:] + r.stderr[-300:])
controle("fichier d'exploration présent", (RACINE / "coding" / "diagnostic" / "prix_signaux.csv").exists())
r = lancer(RACINE / "tests" / "test_serveur.py")
controle("serveur, noyau et enregistrement des résultats (tests/test_serveur.py)", r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:])
r = lancer(RACINE / "tests" / "lint_formatage.py")
controle("aucun défaut de mise en forme dans les sources (tests/lint_formatage.py)", r.returncode == 0, r.stdout[-1500:])
r = lancer(RACINE / "tests" / "test_rendu.py")
controle("rendu réel de chaque page sans marqueur visible ni formule en erreur (tests/test_rendu.py)", r.returncode == 0, r.stdout[-2000:] + r.stderr[-1000:])
r = lancer(RACINE / "tests" / "test_navigateur.py")
controle("passage complet dans un navigateur (tests/test_navigateur.py)", r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:])

print()
print(f"{len(echecs)} échec(s)" if echecs else "Tous les contrôles passent.")
sys.exit(1 if echecs else 0)
