"""Point d'entrée unique de la plateforme d'apprentissage. Usage, depuis ce dossier :  python lancer.py

Démarre un serveur local (http://127.0.0.1:8765, accessible depuis cette machine seulement),
ouvre le navigateur sur la liste des examens, et fournit à la page :
  - un noyau Python par question de code, qui exécute les cellules du notebook avec
    l'interpréteur .venv-hackerrank (mêmes versions de bibliothèques que HackerRank) ;
  - l'exécution des tests visibles et cachés ;
  - l'enregistrement des résultats dans suivi/ (résultats, journal d'erreurs, progression, code soumis).
Le code tapé dans la page s'exécute sur cette machine : ne lance ce serveur que pour toi.
"""
import argparse
import datetime as dt
import html
import json
import os
import queue
import re
import subprocess
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RACINE = Path(__file__).resolve().parent
NOYAU = RACINE / "outils" / "noyau.py"
DELAI_CELLULE_S = 120
DELAI_TESTS_S = 180
DELAI_AIDE_S = 60


def interpreteur():
    for chemin in [RACINE / ".venv-hackerrank" / "Scripts" / "python.exe", RACINE / ".venv-hackerrank" / "bin" / "python"]:
        if chemin.exists():
            return str(chemin), True
    return sys.executable, False


def dans_racine(relatif):
    chemin = (RACINE / relatif).resolve()
    if RACINE not in chemin.parents and chemin != RACINE:
        raise ValueError(f"chemin hors du dépôt : {relatif}")
    return chemin


class Noyau:
    """Un processus Python persistant ; redémarré s'il dépasse le délai ou s'il meurt."""

    def __init__(self, dossier):
        self.dossier = dossier
        self.verrou = threading.Lock()
        self.processus = None
        self.demarrer()

    def demarrer(self):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self.processus = subprocess.Popen(
            [interpreteur()[0], str(NOYAU)], cwd=self.dossier, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1,
        )
        self.file = queue.Queue()
        processus, file = self.processus, self.file

        def lire():
            for ligne in processus.stdout:
                file.put(ligne)
            file.put(None)

        threading.Thread(target=lire, daemon=True).start()

    def arreter(self):
        if self.processus and self.processus.poll() is None:
            self.processus.kill()

    def envoyer(self, commande, delai, attendre=True):
        """Envoie une commande ; si attendre est faux et que le noyau est occupé, répond aussitôt « occupé »."""
        if not self.verrou.acquire(blocking=attendre):
            return {"occupe": True}
        try:
            return self._envoyer(commande, delai)
        finally:
            self.verrou.release()

    def _envoyer(self, commande, delai):
        if self.processus.poll() is not None:
            self.demarrer()
        self.processus.stdin.write(json.dumps(commande, ensure_ascii=False) + "\n")
        self.processus.stdin.flush()
        try:
            ligne = self.file.get(timeout=delai)
        except queue.Empty:
            ligne = None
            message = f"Délai de {delai} s dépassé : le noyau a été redémarré et toutes les variables sont perdues."
        else:
            message = "Le noyau s'est arrêté (mémoire épuisée ou sortie forcée) : il a été redémarré et les variables sont perdues."
        if ligne is None:
            self.arreter()
            self.demarrer()
            return {"erreur": message, "redemarre": True}
        return json.loads(ligne)


NOYAUX = {}
VERROU_NOYAUX = threading.Lock()


def noyau(session, dossier, neuf=False):
    """Renvoie (noyau, cree) ; cree vaut True si le noyau vient d'être démarré."""
    cle = (session, dossier)
    with VERROU_NOYAUX:
        if neuf and cle in NOYAUX:
            NOYAUX.pop(cle).arreter()
        cree = cle not in NOYAUX
        if cree:
            NOYAUX[cle] = Noyau(dans_racine(dossier))
        return NOYAUX[cle], cree


NOM_PLATEFORME = "Atelier quant"  # nom affiché sur la page d'accueil et dans le terminal


# ------------------------------------------------------------------ suivi
def enregistrer_resultat(resultat, dossier_suivi):
    """Écrit le résultat, ajoute les erreurs au journal et met à jour la progression."""
    maintenant = dt.datetime.now()
    jour = maintenant.date()
    horodatage = maintenant.strftime("%Y%m%d-%H%M%S")
    (dossier_suivi / "resultats").mkdir(parents=True, exist_ok=True)
    (dossier_suivi / "resultats" / f"{resultat['examen']}-{horodatage}.json").write_text(
        json.dumps(resultat, ensure_ascii=False, indent=1), encoding="utf-8")

    (dossier_suivi / "code").mkdir(parents=True, exist_ok=True)
    for q in resultat["questions"]:
        if q.get("code_source"):
            (dossier_suivi / "code" / f"{resultat['examen']}-{q['id']}-{horodatage}.py").write_text(q["code_source"], encoding="utf-8")

    cellule = lambda t: str(t).replace("|", "/").replace("\n", " ")
    lignes = []
    for q in resultat["questions"]:
        if q["points"] >= 1:
            continue
        if q["type"] == "code":
            donnee = f"{q.get('caches_reussis', 0)}/{q.get('caches_total', '?')} tests cachés" if q.get("soumissions") else "jamais soumis"
            bonne = "tous les tests cachés"
            cause = "à préciser"
        else:
            rep = q["reponse"] if isinstance(q["reponse"], (str, type(None))) else "plusieurs"  # liste : question à plusieurs réponses
            donnee = {"nsp": "je ne sais pas", None: "sans réponse"}.get(rep, q.get("reponse_texte", rep))
            bonne = q.get("bonne_reponse_texte", "")
            cause = {"nsp": "lacune de cours (« je ne sais pas »)", None: "à préciser : manque de temps ou question sautée"}.get(rep, "à préciser")
        lignes.append("| " + " | ".join(cellule(x) for x in [
            jour.isoformat(), f"{resultat['examen']} {q['id']} ({q['theme']})", donnee, bonne, cause,
            f"correction détaillée au bilan de {resultat['examen']}, puis leçon du thème « {q['theme']} »", (jour + dt.timedelta(days=1)).isoformat(),
            (jour + dt.timedelta(days=3)).isoformat(), "ouverte"]) + " |")
    journal = dossier_suivi / "erreurs.md"
    texte = journal.read_text(encoding="utf-8").rstrip("\n") + "\n"
    journal.write_text(texte + "".join(l + "\n" for l in lignes), encoding="utf-8")

    chemin_progression = dossier_suivi / "progression.json"
    progression = json.loads(chemin_progression.read_text(encoding="utf-8"))
    par_section = {}
    for q in resultat["questions"]:
        s = par_section.setdefault(q["section"], {"points": 0.0, "questions": 0, "temps_s": 0})
        s["points"] = round(s["points"] + q["points"], 4)
        s["questions"] += 1
        s["temps_s"] += q["temps_s"]
        t = progression["themes"].setdefault(q["theme"], {"tentatives": 0, "points": 0.0})
        t["tentatives"] += 1
        t["points"] = round(t["points"] + q["points"], 4)
    progression["sessions"].append({"examen": resultat["examen"], "date": maintenant.isoformat(timespec="seconds"),
                                    "duree_s": resultat["duree_s"], "sections": par_section})
    chemin_progression.write_text(json.dumps(progression, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"fichier": f"suivi/resultats/{resultat['examen']}-{horodatage}.json", "erreurs_ajoutees": len(lignes)}


# ------------------------------------------------------------------ serveur
def derniers_resultats(dossier_suivi):
    """Pour chaque séance déjà faite : date et score de la dernière tentative enregistrée."""
    resultats = {}
    for f in sorted((dossier_suivi / "resultats").glob("*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        qs = r.get("questions", [])
        resultats[r.get("examen")] = {"date": f.stem[-15:-7], "points": sum(q.get("points", 0) for q in qs), "total": len(qs)}
    return resultats


def erreurs_a_reprendre(dossier_suivi, aujourdhui):
    """Lignes ouvertes du journal dont une reprise (J+1 ou J+3) tombe aujourd'hui ou est déjà passée."""
    journal = dossier_suivi / "erreurs.md"
    if not journal.exists():
        return []
    a_faire = []
    for ligne in journal.read_text(encoding="utf-8").splitlines():
        cellules = [c.strip().replace("`", "") for c in ligne.strip().strip("|").split("|")]
        if len(cellules) != 9 or not re.match(r"\d{4}-\d{2}-\d{2}$", cellules[0]) or cellules[8] != "ouverte":
            continue
        dues = [d for d in (cellules[6], cellules[7]) if d <= aujourdhui]
        if dues:
            a_faire.append({"question": cellules[1], "fiche": cellules[5], "retard": max(dues) < aujourdhui})
    return a_faire


def page_accueil(dossier_suivi=None):
    dossier_suivi = dossier_suivi or Gestionnaire.dossier_suivi
    aujourdhui = dt.date.today().isoformat()
    lire = lambda nom: json.loads((RACINE / "examens" / nom).read_text(encoding="utf-8")) if (RACINE / "examens" / nom).exists() else None
    publiees = {e["id"]: e for e in (lire("publiees.json") or [])}
    parcours = lire("parcours.json") or [{"bloc": "Séances", "note": "", "ids": list(publiees)}]
    faits = derniers_resultats(dossier_suivi)
    esc = html.escape

    def ligne(ident):
        e = dict({"titre": ident, "duree_min": "?"}, **publiees.get(ident, {}))
        if ident in faits:
            r = faits[ident]
            etat = f'<span class="score">{r["points"]:.2f}'.replace(".", ",").replace(",00", "") + f' / {r["total"]}</span><span class="date">{r["date"][6:8]}/{r["date"][4:6]}</span>'
            classe = "fait"
        else:
            etat, classe = "", ""
        return (f'<li class="{classe}"><span class="pastille"></span><a href="/examens/{esc(ident)}.html">{esc(e["titre"])}</a>'
                f'<span class="droite">{etat}<span class="duree">{e["duree_min"]} min</span></span></li>')

    ordre = [i for b in parcours for i in b["ids"] if i in publiees or not publiees]
    suivante = next((i for i in ordre if i not in faits), None)
    total, n_faits = len(ordre), sum(i in faits for i in ordre)
    reprises = erreurs_a_reprendre(dossier_suivi, aujourdhui)

    cartes = []
    for bloc in parcours:
        ids = [i for i in bloc["ids"] if i in publiees or not publiees]
        if not ids:
            continue
        k = sum(i in faits for i in ids)
        cartes.append(f'<section class="domaine"><div class="domaine-tete"><h2>{esc(bloc["bloc"])}</h2><span class="compte">{k} / {len(ids)}</span></div>'
                      f'<p class="note">{esc(bloc.get("note", ""))}</p><div class="jauge"><span style="width:{100 * k / len(ids):.0f}%"></span></div>'
                      f'<ol>{"".join(ligne(i) for i in ids)}</ol></section>')

    if suivante:
        e = publiees.get(suivante, {"titre": suivante, "duree_min": "?"})
        bloc_suite = (f'<section class="suite"><div><p class="surtitre">Continuer</p><h2>{esc(e["titre"])}</h2>'
                      f'<p class="note">{e["duree_min"]} min · chaque séance commence par sa leçon</p></div>'
                      f'<a class="bouton" href="examens/{esc(suivante)}.html">Commencer</a></section>')
    else:
        bloc_suite = '<section class="suite"><div><p class="surtitre">Parcours</p><h2>Toutes les séances ont été faites une fois.</h2></div></section>'

    if reprises:
        elements = [f'<li><span class="pastille alerte"></span><span>{esc(r["question"])}{" <strong>(en retard)</strong>" if r["retard"] else ""}'
                        f'<br><span class="note">à relire : {esc(r["fiche"])}</span></span></li>' for r in reprises]
        items = "".join(elements[:4]) + (f'<details><summary>Voir les {len(elements) - 4} autres</summary><ul>{"".join(elements[4:])}</ul></details>' if len(elements) > 4 else "")
        bloc_reprises = f'<section class="domaine large"><div class="domaine-tete"><h2>Erreurs à reprendre aujourd\'hui</h2><span class="compte">{len(reprises)}</span></div><ul>{items}</ul></section>'
    else:
        bloc_reprises = '<section class="domaine large"><div class="domaine-tete"><h2>Erreurs à reprendre aujourd\'hui</h2><span class="compte">0</span></div><p class="note">Aucune : rien à revoir aujourd\'hui.</p></section>'

    chemin, conforme = interpreteur()
    avertissement = "" if conforme else '<p class="note">Attention : .venv-hackerrank est absent, le code s\'exécute avec l\'interpréteur courant, dont les versions de bibliothèques diffèrent de celles des leçons.</p>'
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{NOM_PLATEFORME}</title><style>
:root{{--fond:#f5f6f8;--carte:#fff;--texte:#1f2328;--doux:#646b75;--trait:#e3e6ea;--accent:#2f5bd3;--accent-doux:#eef2fc;--bon:#1f7a4d;--alerte:#9a5a08;--ombre:0 1px 2px rgba(16,24,40,.04),0 1px 3px rgba(16,24,40,.06)}}
@media (prefers-color-scheme: dark){{:root{{--fond:#0f1115;--carte:#171a20;--texte:#e6e8eb;--doux:#9aa1ab;--trait:#262b33;--accent:#86a8ff;--accent-doux:#1a2135;--bon:#5cc98f;--alerte:#e2a54f;--ombre:none}}}}
*{{box-sizing:border-box}}
body{{margin:0;font:15.5px/1.55 "Segoe UI Variable Text","Segoe UI",system-ui,sans-serif;background:var(--fond);color:var(--texte);-webkit-font-smoothing:antialiased}}
a{{color:var(--texte);text-decoration:none}} a:hover{{color:var(--accent)}}
.barre{{height:56px;background:var(--carte);border-bottom:1px solid var(--trait);display:flex;align-items:center;padding:0 24px}}
.marque{{display:flex;align-items:center;gap:8px;font-weight:650}}
.marque::before{{content:"";width:22px;height:22px;border-radius:6px;background:linear-gradient(135deg,var(--accent),color-mix(in srgb,var(--accent) 55%,#7b4bd6))}}
main{{max-width:1120px;margin:0 auto;padding:32px 24px 80px}}
h1{{font-size:28px;letter-spacing:-.02em;margin:0 0 4px}} .intro{{color:var(--doux);margin:0 0 24px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:18px}}
.stat{{background:var(--carte);border:1px solid var(--trait);border-radius:12px;padding:14px 18px;box-shadow:var(--ombre)}}
.stat b{{display:block;font-size:26px;letter-spacing:-.02em}} .stat span{{color:var(--doux);font-size:13px}}
.suite{{display:flex;align-items:center;justify-content:space-between;gap:20px;background:var(--carte);border:1px solid var(--trait);border-radius:12px;padding:20px 24px;margin-bottom:18px;box-shadow:var(--ombre)}}
.suite h2{{margin:2px 0;font-size:19px}} .surtitre{{margin:0;font-size:11.5px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--accent)}}
.bouton{{background:var(--accent);color:#fff;font-weight:600;padding:10px 20px;border-radius:8px;white-space:nowrap}} .bouton:hover{{color:#fff;filter:brightness(.92)}}
@media (prefers-color-scheme: dark){{.bouton,.bouton:hover{{color:#0d1220}}}}
.grille{{display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:18px}}
@media (max-width:560px){{.grille{{grid-template-columns:1fr}}}}
.domaine{{background:var(--carte);border:1px solid var(--trait);border-radius:12px;padding:18px 22px;box-shadow:var(--ombre)}}
.domaine.large{{margin-bottom:18px}}
.domaine-tete{{display:flex;align-items:baseline;justify-content:space-between;gap:12px}}
.domaine h2{{font-size:17px;margin:0}} .compte{{font-size:13px;color:var(--doux);font-variant-numeric:tabular-nums}}
.note{{color:var(--doux);font-size:13.5px;margin:4px 0 10px}}
.jauge{{height:4px;border-radius:99px;background:var(--trait);overflow:hidden;margin-bottom:10px}} .jauge span{{display:block;height:100%;background:var(--accent)}}
ol,ul{{list-style:none;margin:0;padding:0}}
li{{display:flex;align-items:flex-start;gap:10px;padding:8px 0;border-top:1px solid var(--trait);font-size:14.5px}}
li:first-child{{border-top:0}} summary{{cursor:pointer;color:var(--accent);font-size:14px;padding:8px 0;border-top:1px solid var(--trait)}} li a{{flex:1;min-width:0}}
.pastille{{flex:none;width:14px;height:14px;margin-top:4px;border-radius:50%;border:1.5px solid var(--trait)}}
li.fait .pastille{{background:var(--bon);border-color:var(--bon)}} .pastille.alerte{{border-color:var(--alerte)}}
.droite{{display:flex;gap:12px;align-items:baseline;white-space:nowrap;font-size:13px;color:var(--doux)}}
.score{{color:var(--bon);font-weight:600}} .duree{{min-width:52px;text-align:right}}
.pied{{color:var(--doux);font-size:12.5px;margin-top:28px}}
</style></head><body>
<div class="barre"><span class="marque">{NOM_PLATEFORME}</span></div>
<main>
<h1>Ton parcours</h1>
<p class="intro">Probabilités, statistique, algèbre linéaire, machine learning, finance quantitative et code. Chaque séance commence par sa leçon, puis enchaîne sur ses exercices ; une question ratée revient le lendemain, puis trois jours plus tard.</p>
<div class="stats"><div class="stat"><b>{n_faits} / {total}</b><span>séances faites</span></div><div class="stat"><b>{len(reprises)}</b><span>erreurs à reprendre aujourd'hui</span></div><div class="stat"><b>{sum(1 for b in parcours for _ in b["ids"]) and len(parcours)}</b><span>domaines</span></div></div>
{bloc_suite}
{bloc_reprises}
<div class="grille">{"".join(cartes)}</div>
{avertissement}
<p class="pied">Interpréteur des notebooks : {esc(chemin)}</p>
</main></body></html>"""


class Gestionnaire(SimpleHTTPRequestHandler):
    dossier_suivi = RACINE / "suivi"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(RACINE), **kwargs)

    def log_message(self, *args):
        pass

    def handle(self):
        # Une page fermée ou rechargée pendant une requête coupe la connexion : ce n'est pas une erreur.
        try:
            super().handle()
        except ConnectionError:
            pass

    def envoyer_json(self, objet, code=200):
        corps = json.dumps(objet, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(corps)))
            self.end_headers()
            self.wfile.write(corps)
        except ConnectionError:
            pass  # la page a été rechargée ou fermée avant la réponse

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            corps = page_accueil().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(corps)))
            self.end_headers()
            self.wfile.write(corps)
        elif self.path == "/api/etat":
            chemin, conforme = interpreteur()
            self.envoyer_json({"interpreteur": chemin, "conforme_hackerrank": conforme})
        else:
            super().do_GET()

    def do_POST(self):
        try:
            requete = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
            if self.path == "/api/executer":
                n, _ = noyau(requete["session"], requete["dossier"])
                self.envoyer_json(n.envoyer({"type": "executer", "code": requete["code"], "nom": requete.get("nom", "cellule")}, DELAI_CELLULE_S))
            elif self.path in ("/api/redemarrer", "/api/connecter"):
                n, cree = noyau(requete["session"], requete["dossier"], neuf=self.path == "/api/redemarrer")
                self.envoyer_json(dict(n.envoyer({"type": "versions"}, DELAI_CELLULE_S), neuf=cree))
            elif self.path in ("/api/completer", "/api/signatures", "/api/aide"):
                # Une aide à la saisie n'attend jamais la fin d'une cellule en cours d'exécution.
                n, _ = noyau(requete["session"], requete["dossier"])
                commande = {"type": self.path.rsplit("/", 1)[1], "code": requete["code"], "ligne": requete["ligne"], "colonne": requete["colonne"]}
                self.envoyer_json(n.envoyer(commande, DELAI_AIDE_S, attendre=False))
            elif self.path == "/api/tests":
                n, _ = noyau(requete["session"], requete["dossier"])
                module = str(dans_racine(requete["module"]))
                self.envoyer_json(n.envoyer({"type": "tests", "module": module, "mode": requete["mode"]}, DELAI_TESTS_S))
            elif self.path == "/api/resultat":
                self.envoyer_json(enregistrer_resultat(requete, self.dossier_suivi))
            else:
                self.envoyer_json({"erreur": "adresse inconnue"}, 404)
        except Exception as e:  # noqa: BLE001
            self.envoyer_json({"erreur": f"{type(e).__name__} : {e}"}, 500)


def creer_serveur(port, dossier_suivi=None):
    if dossier_suivi is not None:
        Gestionnaire.dossier_suivi = Path(dossier_suivi)
    return ThreadingHTTPServer(("127.0.0.1", port), Gestionnaire)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sans-navigateur", action="store_true")
    args = parser.parse_args()
    serveur = creer_serveur(args.port)
    adresse = f"http://127.0.0.1:{args.port}/"
    chemin, conforme = interpreteur()
    print(f"{NOM_PLATEFORME} : {adresse}")
    print(f"Interpréteur des notebooks : {chemin}" + ("" if conforme else "  (ATTENTION : pas l'environnement HackerRank)"))
    print("Ctrl+C pour arrêter.")
    if not args.sans_navigateur:
        webbrowser.open(adresse)
    try:
        serveur.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for n in NOYAUX.values():
            n.arreter()


if __name__ == "__main__":
    main()
