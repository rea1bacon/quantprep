"""Test de la page d'examen dans un vrai navigateur (Chrome ou Edge sans fenêtre), piloté par le
protocole DevTools. Rejoue un passage : démarrage, notebook (frappe, exécution, tests, soumission),
réponse à une question, fin d'examen, enregistrement du résultat. Échoue sur toute erreur JavaScript.

Le serveur tourne sur un port libre avec un dossier de suivi temporaire. Usage : python tests/test_navigateur.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import websocket  # websocket-client

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
import lancer  # noqa: E402

NAVIGATEURS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]
echecs = []


def controle(nom, condition, detail=""):
    print(f"[{'OK' if condition else 'ÉCHEC'}] {nom}" + (f" : {detail}" if not condition else ""))
    if not condition:
        echecs.append(nom)


class Onglet:
    def __init__(self, url_ws):
        self.ws = websocket.create_connection(url_ws, timeout=60, suppress_origin=True)
        self.n = 0
        self.exceptions = []

    def cmd(self, methode, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": methode, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("method") == "Runtime.exceptionThrown":
                self.exceptions.append(m["params"]["exceptionDetails"])
            if m.get("id") == self.n:
                return m.get("result", {})

    def js(self, expression):
        r = self.cmd("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"])
        return r["result"].get("value")

    def attendre(self, expression, delai=60):
        fin = time.time() + delai
        while time.time() < fin:
            if self.js(expression):
                return True
            time.sleep(0.2)
        return False

    def touche(self, cle, code, vk, shift=False):
        mod = 8 if shift else 0
        texte = {"Enter": "\r"}.get(cle) if not shift else None  # un vrai clavier envoie aussi le caractère tapé
        params = dict(type="keyDown", key=cle, code=code, windowsVirtualKeyCode=vk, modifiers=mod)
        if texte:
            params["text"] = texte
        self.cmd("Input.dispatchKeyEvent", **params)
        self.cmd("Input.dispatchKeyEvent", type="keyUp", key=cle, code=code, windowsVirtualKeyCode=vk, modifiers=mod)


suivi = Path(tempfile.mkdtemp())
profil = Path(tempfile.mkdtemp())
for f in ["erreurs.md", "progression.json"]:
    shutil.copy(RACINE / "suivi" / f, suivi / f)
LIGNES_AVANT = len((suivi / "erreurs.md").read_text(encoding="utf-8").splitlines())
serveur = lancer.creer_serveur(0, suivi)
threading.Thread(target=serveur.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{serveur.server_address[1]}"
executable = next(p for p in NAVIGATEURS if Path(p).exists())
port_debug = 9339
navigateur = subprocess.Popen([executable, "--headless=new", f"--remote-debugging-port={port_debug}", f"--user-data-dir={profil}",
                               "--no-first-run", "--window-size=1400,1000", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    cibles = None
    for _ in range(100):
        try:
            cibles = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port_debug}/json", timeout=2).read())
            if any(c["type"] == "page" for c in cibles):
                break
        except OSError:
            time.sleep(0.2)
    o = Onglet(next(c for c in cibles if c["type"] == "page")["webSocketDebuggerUrl"])
    o.cmd("Runtime.enable")
    o.cmd("Page.enable")
    o.cmd("Page.navigate", url=BASE + "/examens/diagnostic.html")
    controle("page chargée", o.attendre("!!document.getElementById('demarrer')"))
    o.js("window.confirm = () => true; localStorage.clear(); true")

    controle("formules de l'accueil composées par MathJax", o.attendre("!!document.querySelector('mjx-container')", 60))
    o.js("document.getElementById('demarrer').click()")
    controle("première question : notebook à deux colonnes", o.attendre("window.EDITEURS && EDITEURS.length === 3", 60))
    controle("éditeur Monaco chargé", o.js("window.MODE_EDITEUR") == "monaco", o.js("window.MODE_EDITEUR"))
    controle("énoncé rendu avec ses tableaux", o.js("document.querySelectorAll('.enonce table').length") == 2)
    controle("noyau démarré aux versions HackerRank", o.attendre("document.getElementById('versions').textContent.includes('pandas 1.4.2')", 90),
             o.js("document.getElementById('versions').textContent"))
    ed = lambda i: f"EDITEURS[{i}].brut"

    # Frappe au clavier dans la dernière cellule : retrait automatique après « : ».
    o.js(f"EDITEURS[2].setValue(''); EDITEURS[2].focus(); true")
    o.cmd("Input.insertText", text="for i in range(2):")
    o.touche("Escape", "Escape", 27)
    o.touche("Enter", "Enter", 13)
    o.cmd("Input.insertText", text="print(i)")
    o.touche("Escape", "Escape", 27)
    texte = o.js("EDITEURS[2].getValue()")
    controle("retrait automatique après deux-points", texte == "for i in range(2):\n    print(i)", repr(texte))
    o.touche("Enter", "Enter", 13, shift=True)
    controle("Maj+Entrée exécute la cellule", o.attendre("document.querySelectorAll('#cellules pre.sortie')[2].textContent === '0\\n1\\n'"),
             o.js("document.querySelectorAll('#cellules pre.sortie')[2].textContent"))
    controle("Maj+Entrée sur la dernière cellule en crée une nouvelle", o.attendre("EDITEURS.length === 4"))

    # Cellule 1 : lecture du fichier d'exploration (crée la variable df dans le noyau).
    o.js("document.querySelectorAll('#cellules [data-a=executer]')[0].click(); true")
    controle("cellule 1 : affichage du DataFrame", o.attendre("document.querySelectorAll('#cellules pre.sortie')[0].textContent.includes('signal')"))

    # Autocomplétion, aide aux paramètres.
    suggestions = "(document.querySelector('.suggest-widget.visible') || {textContent: ''}).textContent"
    o.js(f"{ed(3)}.setValue('import pandas as pd\\npd.read_c'); {ed(3)}.setPosition({{lineNumber: 2, column: 11}}); {ed(3)}.focus(); {ed(3)}.trigger('t', 'editor.action.triggerSuggest', {{}}); true")
    controle("autocomplétion d'un module : pd.read_c propose read_csv", o.attendre(suggestions + ".includes('read_csv')", 60), o.js(suggestions))
    o.touche("Escape", "Escape", 27)
    o.js(f"{ed(3)}.setValue('df.grou'); {ed(3)}.setPosition({{lineNumber: 1, column: 8}}); {ed(3)}.trigger('t', 'editor.action.triggerSuggest', {{}}); true")
    controle("autocomplétion d'une variable du noyau : df.grou propose groupby", o.attendre(suggestions + ".includes('groupby')", 60), o.js(suggestions))
    o.touche("Enter", "Enter", 13)
    controle("Entrée accepte la suggestion", o.attendre("EDITEURS[3].getValue() === 'df.groupby'"), o.js("EDITEURS[3].getValue()"))
    o.js(f"{ed(3)}.setValue('import pandas as pd\\ndef f(df: pd.DataFrame):\\n    d = df.sort_values([\\'a\\'])\\n    d.grou'); {ed(3)}.setPosition({{lineNumber: 4, column: 11}}); {ed(3)}.trigger('t', 'editor.action.triggerSuggest', {{}}); true")
    controle("autocomplétion dans une fonction au paramètre annoté : d.grou propose groupby", o.attendre(suggestions + ".includes('groupby')", 60), o.js(suggestions))
    o.touche("Escape", "Escape", 27)
    o.js(f"{ed(3)}.setValue('pd.read_csv('); {ed(3)}.setPosition({{lineNumber: 1, column: 13}}); {ed(3)}.trigger('t', 'editor.action.triggerParameterHints', {{}}); true")
    controle("aide aux paramètres de pd.read_csv", o.attendre("(document.querySelector('.parameter-hints-widget') || {textContent: ''}).textContent.includes('filepath_or_buffer')", 60))
    o.touche("Escape", "Escape", 27)

    # Cellule 2 : la correction, puis tests visibles et soumission.
    correction = (RACINE / "coding" / "diagnostic" / "_correction.py").read_text(encoding="utf-8")
    o.js(f"EDITEURS[1].setValue({json.dumps(correction)}); true")
    o.js("document.querySelectorAll('#cellules [data-a=executer]')[1].click(); true")
    o.attendre("document.querySelectorAll('#cellules .numero')[1].textContent !== '[*]'")
    o.js("document.getElementById('testsVisibles').click(); true")
    controle("tests visibles : 3/3", o.attendre("document.getElementById('resultatsTests').textContent.includes('3/3 réussis')"),
             o.js("document.getElementById('resultatsTests').textContent"))
    o.js("document.getElementById('soumettre').click(); true")
    controle("soumission : 16/16 tests cachés", o.attendre("document.getElementById('etatSoumission').textContent.includes('16/16')", 120),
             o.js("document.getElementById('resultatsTests').textContent"))

    # Le code survit à un rechargement de la page.
    o.cmd("Page.reload")
    controle("code conservé après rechargement", o.attendre("window.EDITEURS && EDITEURS.length > 1 && EDITEURS[1].getValue() === " + json.dumps(correction), 60))
    controle("noyau reconnecté après rechargement", o.attendre("document.getElementById('versions').textContent.includes('variables conservées')"),
             o.js("document.getElementById('versions').textContent"))
    o.js("document.getElementById('testsVisibles').click(); true")
    controle("fonction toujours définie après rechargement", o.attendre("document.getElementById('resultatsTests').textContent.includes('3/3 réussis')"))
    o.js("window.confirm = () => true; true")

    # Question suivante : choix multiple.
    o.js("document.getElementById('suiv').click(); true")
    controle("question 2 : quatre options et « je ne sais pas »", o.attendre("document.querySelectorAll('.option').length === 5"))
    o.js("document.querySelector('.option[data-o=a]').click(); true")
    controle("option sélectionnée", o.attendre("!!document.querySelector('.option.choisie[data-o=a]')"))

    o.js("document.getElementById('finir').click(); true")
    controle("bilan affiché", o.attendre("document.getElementById('position').textContent === 'Bilan'"))
    controle("résultat enregistré par le serveur", o.attendre("document.getElementById('enregistrement').textContent.includes('Résultat enregistré')"),
             o.js("document.getElementById('enregistrement').textContent"))
    controle("score de la section code : 1,00 / 1", "1,00 / 1" in o.js("document.querySelector('table').textContent"))
    journal = (suivi / "erreurs.md").read_text(encoding="utf-8")
    controle("17 erreurs ajoutées au journal (questions sans réponse)", sum(l.startswith("| 20") for l in journal.splitlines()[LIGNES_AVANT:]) == 17)
    # Séance de code 1 : plusieurs questions de code qui partagent un noyau.
    o.cmd("Page.navigate", url=BASE + "/examens/seance-code-1.html")
    o.attendre("!!document.getElementById('demarrer')")
    o.js("window.confirm = () => true; true")
    o.js("document.getElementById('demarrer').click(); true")
    # État ancien : sauvegardé par une version de la page qui n'avait pas encore la leçon (cas réel du 6 octobre).
    o.js("{ const e = JSON.parse(localStorage.getItem('atelier-quant:seance-code-1')); delete e.cellules['L-PANDAS-1']; e.ordreControles = {}; "
         "e.debut = Date.now() - 13 * 3600 * 1000; localStorage.setItem('atelier-quant:seance-code-1', JSON.stringify(e)); } true")
    o.cmd("Page.reload")
    controle("état ancien réparé : cellules de la leçon recréées et noyau joint",
             o.attendre("window.EDITEURS && EDITEURS.length === 2 && document.getElementById('versions').textContent.includes('pandas 1.4.2')", 90),
             o.js("(document.getElementById('versions') || {}).textContent"))
    controle("état ancien réparé : points de contrôle affichés", o.attendre("document.querySelectorAll('.controle .option').length === 20"))
    o.js("window.confirm = () => true; document.getElementById('recommencerHaut').click(); true")
    controle("bouton Recommencer : retour à l'accueil de la séance, chronomètre remis à zéro", o.attendre("!!document.getElementById('demarrer')"))
    o.js("document.getElementById('demarrer').click(); true")
    o.attendre("window.EDITEURS && EDITEURS.length === 2", 60)
    controle("séance : bouton Pause visible", o.attendre("!document.getElementById('pause').hidden"))
    o.js("document.getElementById('pause').click(); true")
    controle("pause : contenu masqué", o.attendre("!document.getElementById('voile').hidden"))
    time.sleep(3)
    o.js("document.getElementById('reprendre').click(); true")
    controle("reprise : contenu visible, au moins 2,5 s de pause décomptées",
             o.attendre("document.getElementById('voile').hidden && JSON.parse(localStorage.getItem('atelier-quant:seance-code-1')).pauseTotale >= 2500"),
             o.js("localStorage.getItem('atelier-quant:seance-code-1')")[:300])
    controle("séance 1 : la leçon s'affiche d'abord, avec son notebook", o.attendre("window.EDITEURS && EDITEURS.length === 2 && document.querySelectorAll('.controle').length === 5", 60))
    controle("leçon : sorties des exemples affichées", o.js("document.querySelectorAll('pre.sortie-lecon').length") >= 10)
    o.js("document.querySelector('.controle[data-c=alignement] .option[data-co=a]').click(); true")
    controle("point de contrôle juste : marqué et expliqué", o.attendre("document.querySelector('.controle[data-c=alignement] .titre-controle').textContent.includes('juste') && !!document.querySelector('.controle[data-c=alignement] .explication')"))
    o.js("document.querySelector('.controle[data-c=rang] .option[data-co=b]').click(); true")
    controle("point de contrôle faux : bonne réponse et explication de l'erreur montrées",
             o.attendre("document.querySelector('.controle[data-c=rang] .titre-controle').textContent.includes('à revoir') && document.querySelectorAll('.controle[data-c=rang] .explication').length === 2"))
    o.js("EDITEURS[1].setValue('df.groupby(\"actif\")[\"prix\"].mean()'); document.querySelectorAll('#cellules [data-a=executer]')[0].click(); true")
    o.attendre("document.querySelectorAll('#cellules .numero')[0].textContent !== '[*]'")
    o.js("document.querySelectorAll('#cellules [data-a=executer]')[1].click(); true")
    controle("leçon : le notebook exécute les exemples", o.attendre("document.querySelectorAll('#cellules pre.sortie')[1].textContent.includes('103')"),
             o.js("document.querySelectorAll('#cellules pre.sortie')[1].textContent"))
    o.js("document.getElementById('suiv').click(); true")
    controle("séance 1 : exercice 1 affiché avec ses 3 cellules", o.attendre("window.EDITEURS && EDITEURS.length === 3", 60))
    corrections = (RACINE / "coding" / "seance1" / "_correction.py").read_text(encoding="utf-8")
    o.js(f"EDITEURS[1].setValue({json.dumps(corrections)}); true")
    o.js("document.querySelectorAll('#cellules [data-a=executer]')[1].click(); true")
    o.attendre("document.querySelectorAll('#cellules .numero')[1].textContent !== '[*]'")
    o.js("document.getElementById('soumettre').click(); true")
    controle("séance 1, exercice 1 : 8/8 tests cachés", o.attendre("document.getElementById('etatSoumission').textContent.includes('8/8')", 120),
             o.js("document.getElementById('resultatsTests').textContent"))
    o.js("document.getElementById('suiv').click(); true")
    controle("séance 1 : exercice 2 affiché", o.attendre("window.EDITEURS && EDITEURS.length === 3 && document.querySelector('.enonce h3').textContent.includes('Exercice 2')", 60))
    o.js("document.getElementById('soumettre').click(); true")
    controle("exercice 2 : la fonction définie à l'exercice 1 est dans le même noyau (8/8)", o.attendre("document.getElementById('etatSoumission').textContent.includes('8/8')", 120),
             o.js("document.getElementById('resultatsTests').textContent"))
    o.js("document.getElementById('finir').click(); true")
    controle("séance 1 : bilan enregistré", o.attendre("document.getElementById('enregistrement').textContent.includes('Résultat enregistré')", 60),
             o.js("(document.getElementById('enregistrement') || {}).textContent"))
    controle("bilan : la leçon n'est pas comptée comme question", "/ 4" in o.js("document.querySelector('table').textContent"),
             o.js("document.querySelector('table').textContent"))
    controle("bilan : points de contrôle résumés", "1 sur 5" in o.js("document.body.textContent"))
    controle("lien Accueil visible dans l'en-tête", o.js("!document.getElementById('lienAccueil').hidden"))
    o.js("document.getElementById('retourAccueil').click(); true")
    controle("bilan : le bouton de retour mène à la page d'accueil", o.attendre("document.title === 'Atelier quant' && document.body.textContent.includes('Chaque séance commence par sa leçon')", 30))
    # Quitter une séance en cours par le lien Accueil la met en pause.
    o.cmd("Page.navigate", url=BASE + "/examens/seance-math-1.html")
    o.attendre("!!document.getElementById('demarrer')")
    o.js("document.getElementById('demarrer').click(); true")
    o.attendre("E.phase === 'epreuve'")
    o.js("document.getElementById('lienAccueil').click(); true")
    o.attendre("document.title === 'Atelier quant'", 30)
    controle("quitter une séance en cours par le lien Accueil la met en pause",
             o.js("JSON.parse(localStorage.getItem('atelier-quant:seance-math-1')).pauseDepuis > 0"))

    # Question à plusieurs réponses : cases cumulables, notation tout ou rien.
    o.cmd("Page.navigate", url=BASE + "/examens/seance-ml-5.html")
    o.attendre("!!document.getElementById('demarrer')")
    o.js("window.confirm = () => true; document.getElementById('demarrer').click(); true")
    o.attendre("E.phase === 'epreuve'")
    o.js("window.QM = ITEMS.findIndex(q => q.type === 'qcm' && q.multiple && q.options.filter(x => x.correcte).length >= 2); E.courant = QM; afficher(); true")
    controle("plusieurs réponses : consigne affichée", "Une ou plusieurs réponses" in o.js("document.getElementById('page').textContent"))
    clic = "((id) => { document.querySelector('.option[data-o=\"' + id + '\"]').click(); return true; })"
    o.js("window.BONNES = ITEMS[QM].options.filter(x => x.correcte).map(x => x.id); window.FAUSSE = ITEMS[QM].options.find(x => !x.correcte).id; true")
    o.js(clic + "(BONNES[0])")
    o.js(clic + "(FAUSSE)")
    controle("plusieurs réponses : deux cases cochées à la fois", o.js("document.querySelectorAll('.option.choisie').length") == 2)
    controle("plusieurs réponses : une case fausse cochée donne zéro point", o.js("points(ITEMS[QM]) === 0"))
    o.js(clic + "(FAUSSE)")
    controle("plusieurs réponses : réponse incomplète, zéro point", o.js("BONNES.length < 2 || points(ITEMS[QM]) === 0"))
    o.js("BONNES.slice(1).forEach(id => document.querySelector('.option[data-o=\"' + id + '\"]').click()); true")
    controle("plusieurs réponses : ensemble exact, un point", o.js("points(ITEMS[QM])") == 1, o.js("JSON.stringify(E.reponses[ITEMS[QM].id])"))
    o.js("localStorage.clear(); true")

    # Monaco rejette une promesse « Canceled » quand on ferme un éditeur pendant une tâche en cours
    # (suggestion, survol) : c'est une annulation voulue, pas une erreur de la page.
    vraies = [x for x in o.exceptions if "Canceled" not in x.get("text", "") + json.dumps(x.get("exception", {}))]
    controle("aucune erreur JavaScript", not vraies, json.dumps(vraies, ensure_ascii=False)[:1500])
finally:
    navigateur.kill()
    serveur.shutdown()
    for n in lancer.NOYAUX.values():
        n.arreter()
    time.sleep(0.5)
    shutil.rmtree(suivi, ignore_errors=True)
    shutil.rmtree(profil, ignore_errors=True)

print(f"{len(echecs)} échec(s)" if echecs else "Tous les tests du navigateur passent.")
sys.exit(1 if echecs else 0)
