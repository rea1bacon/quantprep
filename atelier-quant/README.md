# Atelier quant

Plateforme locale de cours, d'exercices et d'examens blancs en français : probabilités, statistique, algèbre linéaire, machine learning, finance quantitative et code. Chaque séance commence par une leçon ; les notebooks s'exécutent dans le navigateur.

## Lancer

```
python -m venv .venv-hackerrank
.venv-hackerrank/Scripts/pip install -r requirements.txt
python lancer.py
```

La page d'accueil s'ouvre sur http://127.0.0.1:8765/. Les nombres affichés viennent des scripts de `verif/` ; `python outils/construire.py` régénère `banque/` et `examens/`, et `python tests/verifier_depot.py` lance les contrôles.
