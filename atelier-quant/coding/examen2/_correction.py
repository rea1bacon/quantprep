"""Correction de la question de code de l'examen blanc 2 (EB2-C1). À n'ouvrir qu'après l'avoir tentée.

Deux implémentations indépendantes : l'une avec pandas (merge_asof), l'autre sans aucune bibliothèque
externe (tri et recherche dichotomique). Les tests exigent qu'elles soient d'accord.
"""
import bisect
import math

import pandas as pd

TOLERANCE_MS = 5000


def desequilibre(transactions: pd.DataFrame, cotations: pd.DataFrame) -> pd.Series:
    """Solution de référence avec pandas."""
    actifs = sorted(transactions["actif"].unique())
    if len(transactions) == 0:
        return pd.Series(dtype=float)

    # 1. Cotations valides seulement. Une comparaison avec NaN vaut False : les cotations
    #    incomplètes disparaissent avec les cotations croisées ou bloquées (achat >= vente).
    #    Filtrer AVANT la jointure : sinon la dernière cotation trouvée peut être invalide,
    #    alors qu'une cotation valide plus ancienne, à moins de 5 s, existe.
    valides = cotations.loc[cotations["achat"] < cotations["vente"], ["temps", "actif", "achat", "vente"]]
    valides = valides.rename(columns={"temps": "temps_cotation"}).sort_values("temps_cotation")

    # 2. Jointure « au plus tard avant ». merge_asof exige deux cadres triés sur la clé de temps.
    #    by="actif" : jamais la cotation d'un autre actif.
    #    allow_exact_matches=False : une cotation de même temps n'est pas antérieure.
    #    tolerance=5000 : écart au plus 5000 ms, borne incluse (vérifié dans verif/examen_blanc_2.py).
    t = transactions.sort_values("temps")
    m = pd.merge_asof(t, valides, left_on="temps", right_on="temps_cotation", by="actif",
                      allow_exact_matches=False, tolerance=TOLERANCE_MS)

    # 3. Signe par rapport au milieu ; NaN (pas de cotation de référence) compte comme 0.
    milieu = (m["achat"] + m["vente"]) / 2
    achat = m["quantite"].where(m["prix"] > milieu, 0)
    vente = m["quantite"].where(m["prix"] < milieu, 0)

    # 4. Volumes par actif, puis déséquilibre ; NaN quand aucun volume n'est classé.
    v_achat = achat.groupby(m["actif"]).sum()
    v_vente = vente.groupby(m["actif"]).sum()
    total = v_achat + v_vente
    r = (v_achat - v_vente) / total.where(total > 0)
    return r.reindex(actifs).astype(float)


def desequilibre_sans_bibliotheque(transactions, cotations):
    """Même calcul sans pandas ni numpy.

    transactions : liste de (temps, actif, prix, quantite) ; cotations : liste de (temps, actif, achat, vente),
    None ou NaN = manquant. Renvoie une liste de (actif, déséquilibre ou nan), triée par actif.
    """
    manquant = lambda x: x is None or (isinstance(x, float) and math.isnan(x))
    par_actif = {}
    for temps, actif, achat, vente in cotations:
        if manquant(achat) or manquant(vente) or not achat < vente:
            continue
        par_actif.setdefault(actif, []).append((temps, achat, vente))
    temps_par_actif = {}
    for actif, liste in par_actif.items():
        liste.sort()  # au plus une cotation par (actif, temps) : le tri ne compare jamais les prix
        temps_par_actif[actif] = [c[0] for c in liste]

    v_achat, v_vente = {}, {}
    for temps, actif, prix, quantite in transactions:
        v_achat.setdefault(actif, 0)
        v_vente.setdefault(actif, 0)
        if actif not in par_actif:
            continue
        # bisect_left renvoie le nombre de cotations de temps strictement inférieur à `temps`.
        k = bisect.bisect_left(temps_par_actif[actif], temps) - 1
        if k < 0:
            continue
        temps_cotation, achat, vente = par_actif[actif][k]
        if temps - temps_cotation > TOLERANCE_MS:
            continue
        milieu = (achat + vente) / 2
        if prix > milieu:
            v_achat[actif] += quantite
        elif prix < milieu:
            v_vente[actif] += quantite
    sortie = []
    for actif in sorted(v_achat):
        total = v_achat[actif] + v_vente[actif]
        sortie.append((actif, (v_achat[actif] - v_vente[actif]) / total if total > 0 else float("nan")))
    return sortie
