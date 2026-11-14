"""Fixture wiring for the iHC GPU tests.

``IHC_TEST_MUTANT=<name>`` patches ``ihc.run_pre`` (and, for the rounding
mutants, ``run_post`` / ``run_head``) with a mutant implementation (see
``_ihc_mutants.py``) before every test, so the same parity suite that must pass
against the real operator must fail against the mutant — the red-run proof
that the tests kill it.
"""

from __future__ import annotations

import os

import pytest

from . import _ihc_mutants

_MUTANT = os.environ.get("IHC_TEST_MUTANT", "").strip().lower()


@pytest.fixture(autouse=True)
def _install_ihc_mutant(monkeypatch):
    if not _MUTANT:
        yield
        return
    from b12x.norm.mhc import ihc

    _ihc_mutants.install(monkeypatch, ihc, _MUTANT)
    yield
