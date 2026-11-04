# SPDX-License-Identifier: GPL-3.0-or-later
import pytest

from buzz.telemetry import is_telemetry_enabled


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("BUZZ_ENABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("BUZZ_DISABLE_TELEMETRY", raising=False)


def test_disabled_by_default():
    assert is_telemetry_enabled() is False


def test_opt_in(monkeypatch):
    monkeypatch.setenv("BUZZ_ENABLE_TELEMETRY", "true")
    assert is_telemetry_enabled() is True


def test_disable_wins_over_enable(monkeypatch):
    monkeypatch.setenv("BUZZ_ENABLE_TELEMETRY", "true")
    monkeypatch.setenv("BUZZ_DISABLE_TELEMETRY", "true")
    assert is_telemetry_enabled() is False
