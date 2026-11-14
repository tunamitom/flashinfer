"""Fixture wiring for the iHC GPU tests.

``IHC_TEST_MUTANT=swap|post1`` patches ``ihc.run_pre`` with a mutant
implementation (see ``ihc_mutants.py``) before every test, so the same parity
suite that must pass against the real operator must fail against the mutant —
the red-run proof that the tests kill it.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import ihc_mutants  # noqa: E402

_MUTANT = os.environ.get("IHC_TEST_MUTANT", "").strip().lower()


@pytest.fixture(autouse=True)
def _install_ihc_mutant(monkeypatch):
    if not _MUTANT:
        yield
        return
    from b12x.norm.mhc import ihc

    ihc_mutants.install(monkeypatch, ihc, _MUTANT)
    yield
