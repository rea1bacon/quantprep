"""Correction de la séance MM 5 : nettoyer un export de transactions."""
import pandas as pd


def _nombre(colonne):
    """Retire tout ce qui n'est ni chiffre, ni point, ni signe moins, puis convertit ; le reste devient NaN."""
    return pd.to_numeric(colonne.astype(str).str.replace(r"[^0-9.-]", "", regex=True), errors="coerce")


def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["price"] = _nombre(df["price"])     # une cellule vide ou N/A devient une chaîne vide après filtrage, puis NaN
    df["volume"] = _nombre(df["volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["price", "volume"])
    df = df[df["volume"] > 0]              # un volume négatif est une donnée fausse : on la retire, on ne la répare pas
    df = df.drop_duplicates()              # après normalisation : deux lignes égales ne diffèrent plus par leur écriture
    return df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)


def realized_vol_by_ticker(df):
    trie = df.sort_values(["ticker", "timestamp"])
    rendements = trie.groupby("ticker")["price"].pct_change()   # le calcul reste à l'intérieur de chaque titre
    return rendements.groupby(trie["ticker"]).std().sort_index()
