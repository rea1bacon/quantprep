"""Sous-ensemble de pytest utilisé par les suites de tests_source/ (pytest n'est pas installé dans .venv-hackerrank).

Fournit approx, raises, fixture et mark.parametrize, avec la même sémantique que pytest sur les usages de ces suites :
tolérance par défaut relative 1e-6 (et absolue 1e-12) ; si seul abs est donné, seule la tolérance absolue compte.
"""
import math
from collections.abc import Mapping, Sequence

import numpy as np


def _proche(obtenu, attendu, rel, abs_, nan_ok):
    if isinstance(attendu, Mapping):
        return isinstance(obtenu, Mapping) and set(obtenu) == set(attendu) and all(
            _proche(obtenu[k], attendu[k], rel, abs_, nan_ok) for k in attendu)
    if isinstance(attendu, np.ndarray) or (isinstance(attendu, Sequence) and not isinstance(attendu, str)):
        obtenu = np.asarray(obtenu, dtype=float).ravel().tolist() if isinstance(obtenu, np.ndarray) else obtenu
        attendu = np.asarray(attendu, dtype=float).ravel().tolist() if isinstance(attendu, np.ndarray) else attendu
        return len(obtenu) == len(attendu) and all(_proche(o, a, rel, abs_, nan_ok) for o, a in zip(obtenu, attendu))
    o, a = float(obtenu), float(attendu)
    if math.isnan(a) or math.isnan(o):
        return nan_ok and math.isnan(a) and math.isnan(o)
    if abs_ is not None and rel is None:
        tolerance = abs_
    else:
        tolerance = (1e-6 if rel is None else rel) * abs(a)
        tolerance = max(tolerance, 1e-12 if abs_ is None else abs_)
    return abs(o - a) <= tolerance


class approx:
    __array_ufunc__ = None
    __array_priority__ = 100
    __hash__ = None

    def __init__(self, attendu, rel=None, abs=None, nan_ok=False):  # noqa: A002
        self.attendu, self.rel, self.abs, self.nan_ok = attendu, rel, abs, nan_ok

    def __eq__(self, obtenu):
        return _proche(obtenu, self.attendu, self.rel, self.abs, self.nan_ok)

    def __repr__(self):
        return f"{self.attendu!r} à la tolérance près"


class raises:
    def __init__(self, exception, match=None):
        self.exception = exception

    def __enter__(self):
        return self

    def __exit__(self, type_, valeur, trace):
        if type_ is None:
            raise AssertionError(f"une exception {self.exception.__name__} était attendue")
        return issubclass(type_, self.exception)


def fixture(fonction):
    fonction._fixture = True
    return fonction


class _Mark:
    @staticmethod
    def parametrize(nom, valeurs):
        def decorer(fonction):
            fonction._parametres = (nom, list(valeurs))
            return fonction
        return decorer


mark = _Mark()
