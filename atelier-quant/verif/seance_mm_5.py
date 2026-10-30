"""Valeurs de la séance MM 5 : décompte de ce que le nettoyage retire du fichier sale."""
import numpy as np
import pandas as pd

from mm_commun import MM, correction, ecrire

C = correction("mm_5.py")
V, T = {}, []
chemin = MM / "data_cache" / "messy_trading_data.csv"
brut = pd.read_csv(chemin)
b = brut.copy()
b.columns = [c.strip().lower() for c in b.columns]
n = len(b)
prix = pd.to_numeric(b["price"].astype(str).str.replace(r"[^0-9.-]", "", regex=True), errors="coerce")
vol = pd.to_numeric(b["volume"].astype(str).str.replace(r"[^0-9.-]", "", regex=True), errors="coerce")
sans_prix = int(prix.isna().sum())
sans_vol = int(vol.isna().sum())
neg = int((vol <= 0).sum())
garde = prix.notna() & vol.notna() & (vol > 0)
apres_filtre = int(garde.sum())
propre = C["load_and_clean"](str(chemin))
V.update(N_brut=n, N_sans_prix=sans_prix, N_sans_vol=sans_vol, N_neg=neg, N_filtre=apres_filtre, N_final=len(propre), N_dup=apres_filtre - len(propre), N_grille=180)
assert len(propre) == 144
v = C["realized_vol_by_ticker"](propre)
for t in ["AAPL", "GOOG", "MSFT"]:
    V[f"VOL_{t}"] = round(float(v[t]), 6)
# Le défaut qu'un seul nombre détecte : un prix en milliers avec séparateur, mal lu, perd les trois derniers chiffres
faux = pd.to_numeric(b["price"].astype(str).str.split(",").str[0].str.replace(r"[^0-9.-]", "", regex=True), errors="coerce")
b2 = pd.DataFrame({"ticker": b["ticker"].astype(str).str.strip().str.upper(), "t": pd.to_datetime(b["timestamp"]), "p": faux})[garde].sort_values(["ticker", "t"])
V["VOL_GOOG_faux"] = round(float(b2[b2.ticker == "GOOG"]["p"].pct_change().std()), 4)
T.append(str(V))
ecrire("seance_mm_5", V, T)
