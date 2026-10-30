"""Rendu réel de chaque page publiée dans Chrome sans fenêtre : aucun marqueur de mise en forme ne doit
rester visible dans le texte affiché (hors blocs de code), ni sur les étapes, ni dans le bilan, où toutes les
solutions et explications sont présentes ; et aucune formule ne doit rester non composée ou en erreur.

Usage : python tests/test_rendu.py [identifiant ...]   (par défaut toutes les pages publiées)
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

import websocket

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
import lancer  # noqa: E402

NAVIGATEUR = next(p for p in [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"] if Path(p).exists())

# Texte visible de la page, sans blocs ni portions de code, sans les formules composées ; puis motifs suspects.
SCRIPT_ANALYSE = r"""(() => {
  const copie = document.getElementById('page').cloneNode(true);
  copie.querySelectorAll('pre, code, mjx-container, textarea, .editeur, script, style').forEach(e => e.remove());
  const texte = copie.textContent;
  const motifs = [[/\\/, 'antislash'], [/\*\*/, 'double astérisque'], [/`/, 'accent grave'], [/\[\[/, 'double crochet'],
                  [/@[A-Za-z]/, 'jeton'], [/&(amp|lt|gt|quot);/, 'entité HTML'], [/\$/, 'dollar']];
  const trouves = [];
  motifs.forEach(([m, nom]) => { const r = texte.match(m); if (r) { const i = r.index; trouves.push(nom + ' : ' + texte.slice(Math.max(0, i - 40), i + 40).replace(/\s+/g, ' ')); } });
  const erreurs = Array.from(document.querySelectorAll('mjx-merror')).map(e => e.getAttribute('data-mjx-error') || e.textContent);
  return JSON.stringify({ id: (ITEMS[E.courant] || {}).id, phase: E.phase, trouves, erreurs });
})()"""


class Onglet:
    def __init__(self, url_ws):
        self.ws = websocket.create_connection(url_ws, timeout=120, suppress_origin=True)
        self.n = 0

    def cmd(self, methode, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": methode, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.n:
                return m.get("result", {})

    def js(self, expression):
        r = self.cmd("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    def attendre(self, expression, delai=60):
        fin = time.time() + delai
        while time.time() < fin:
            if self.js(expression):
                return True
            time.sleep(0.25)
        return False


def composer(o):
    o.js("window.MathJax && MathJax.typesetPromise ? MathJax.typesetPromise().then(() => true) : true")


def main(identifiants):
    suivi = Path(tempfile.mkdtemp())
    profil = Path(tempfile.mkdtemp())
    for f in ["erreurs.md", "progression.json"]:
        shutil.copy(RACINE / "suivi" / f, suivi / f)
    serveur = lancer.creer_serveur(0, suivi)
    threading.Thread(target=serveur.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{serveur.server_address[1]}"
    port = 9390
    nav = subprocess.Popen([NAVIGATEUR, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={profil}",
                            "--window-size=1300,950", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    problemes = 0
    try:
        cibles = None
        for _ in range(150):
            try:
                cibles = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2).read())
                if any(c["type"] == "page" for c in cibles):
                    break
            except OSError:
                time.sleep(0.2)
        o = Onglet(next(c for c in cibles if c["type"] == "page")["webSocketDebuggerUrl"])
        # Auto-contrôle : le détecteur doit signaler un texte piégé injecté dans la page.
        o.cmd("Page.navigate", url=f"{base}/examens/{identifiants[0]}.html")
        o.attendre("!!document.getElementById('demarrer')")
        # L'antislash est fabriqué par son code (92) : aucune couche d'échappement Python ou JavaScript ne peut le perdre.
        o.js("{ const a = String.fromCharCode(92); const p = document.createElement('p'); "
             "p.textContent = 'piège ' + a + '\"prix' + a + '\" **gras** `code`'; document.getElementById('page').appendChild(p); } true")
        piege = json.loads(o.js(SCRIPT_ANALYSE))["trouves"]
        assert {t.split(" :")[0] for t in piege} >= {"antislash", "double astérisque", "accent grave"}, piege
        print("auto-contrôle du détecteur : texte piégé signalé")
        for ident in identifiants:
            o.cmd("Page.navigate", url=f"{base}/examens/{ident}.html")
            o.attendre("!!document.getElementById('demarrer')")
            o.js("localStorage.clear(); window.confirm = () => true; document.getElementById('demarrer').click(); true")
            total = o.js("ITEMS.length")
            vus = []
            for k in range(total):
                o.attendre(f"E.courant === {k}")
                time.sleep(0.4)
                composer(o)
                vus.append(json.loads(o.js(SCRIPT_ANALYSE)))
                if k < total - 1:
                    o.js("document.getElementById('suiv').click(); true")
            o.js("document.getElementById('finir').click(); true")
            o.attendre("E.phase === 'bilan'")
            o.js("document.querySelectorAll('details').forEach(d => d.open = true); true")
            time.sleep(0.5)
            composer(o)
            vus.append(json.loads(o.js(SCRIPT_ANALYSE)))
            defauts = [(v["id"] if v["phase"] == "epreuve" else "bilan", d) for v in vus for d in v["trouves"] + [f"MathJax : {e}" for e in v["erreurs"]]]
            problemes += len(defauts)
            print(f"{ident} : {total} étapes et bilan, " + (f"{len(defauts)} défaut(s)" if defauts else "aucun défaut"))
            for ou, d in defauts:
                print(f"    {ou} : {d}")
    finally:
        nav.kill()
        serveur.shutdown()
        for n in lancer.NOYAUX.values():
            n.arreter()
        time.sleep(0.5)
        shutil.rmtree(suivi, ignore_errors=True)
        shutil.rmtree(profil, ignore_errors=True)
    print(f"{problemes} défaut(s) de rendu")
    return problemes


if __name__ == "__main__":
    ids = sys.argv[1:] or [e["id"] for e in json.loads((RACINE / "examens" / "publiees.json").read_text(encoding="utf-8"))]
    sys.exit(1 if main(ids) else 0)
