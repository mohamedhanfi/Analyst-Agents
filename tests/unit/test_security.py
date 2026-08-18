"""Unit tests for shared/security — Fernet at-rest encryption."""
from __future__ import annotations

import pytest

from shared import security


@pytest.fixture(autouse=True)
def isolated_key(tmp_path, monkeypatch):
    monkeypatch.setattr(security, "KEY_FILE", tmp_path / ".enc_key")
    monkeypatch.setattr(security, "ENV_KEY", "INSIGHT_FORGE_TEST_KEY")
    monkeypatch.delenv(security.ENV_KEY, raising=False)


def test_encrypt_decrypt_roundtrip(tmp_path):
    src = tmp_path / "sales.csv"
    enc = tmp_path / "sales.csv.enc"
    dec = tmp_path / "sales_dec.csv"
    src.write_bytes(b"order_id,revenue\n1,10\n2,20\n")
    security.encrypt_file(src, enc)
    assert not src.exists()
    assert enc.read_bytes() != b"order_id,revenue\n1,10\n2,20\n"
    security.decrypt_file(enc, dec)
    assert dec.read_bytes() == b"order_id,revenue\n1,10\n2,20\n"
    assert enc.exists()  # ciphertext stays at rest


def test_key_persisted_and_reused(tmp_path):
    k1 = security._load_or_create_key()
    assert security.KEY_FILE.is_file()
    k2 = security._load_or_create_key()
    assert k1 == k2


def test_env_key_wins(tmp_path, monkeypatch):
    import base64
    env_key = base64.urlsafe_b64encode(b"k" * 32).decode("ascii")
    monkeypatch.setenv(security.ENV_KEY, env_key)
    key = security._load_or_create_key()
    assert key == b"k" * 32
    assert not security.KEY_FILE.exists()


def test_encrypt_bytes_roundtrip():
    data = b"raw sensitive payload"
    token = security.encrypt_bytes(data)
    assert token != data
    assert security.decrypt_bytes(token) == data


def test_encryption_enabled_default():
    assert security.encryption_enabled(None) is True
    assert security.encryption_enabled({"retention": {}}) is True
    assert security.encryption_enabled(
        {"retention": {"encrypt_at_rest": False}}) is False