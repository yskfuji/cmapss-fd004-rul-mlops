from __future__ import annotations

import base64

import pytest

from forecasting_api import secrets_provider


def setup_function() -> None:
    secrets_provider.resolve_secret.cache_clear()


def teardown_function() -> None:
    secrets_provider.resolve_secret.cache_clear()


def test_normalize_gcp_key_name_handles_prefixes():
    assert secrets_provider._normalize_gcp_key_name("kms://projects/x") == "projects/x"
    assert secrets_provider._normalize_gcp_key_name("gcp-kms://projects/x") == "projects/x"
    assert secrets_provider._normalize_gcp_key_name("projects/x") == "projects/x"


def test_resolve_secret_prefers_plaintext(monkeypatch):
    monkeypatch.setenv("PLAIN_SECRET", "plain-value")
    monkeypatch.delenv("ENC_SECRET", raising=False)
    assert (
        secrets_provider.resolve_secret(plain_env="PLAIN_SECRET", encrypted_env="ENC_SECRET")
        == "plain-value"
    )


def test_resolve_secret_returns_none_without_ciphertext_or_key(monkeypatch):
    monkeypatch.delenv("PLAIN_SECRET", raising=False)
    monkeypatch.delenv("ENC_SECRET", raising=False)
    monkeypatch.delenv("RULFM_FORECASTING_API_KMS_KEY_URI", raising=False)
    assert (
        secrets_provider.resolve_secret(plain_env="PLAIN_SECRET", encrypted_env="ENC_SECRET")
        is None
    )


def test_resolve_secret_decrypts_with_gcp_kms(monkeypatch):
    monkeypatch.delenv("PLAIN_SECRET", raising=False)
    monkeypatch.setenv("ENC_SECRET", base64.b64encode(b"ciphertext").decode("ascii"))
    monkeypatch.setenv("RULFM_FORECASTING_API_KMS_KEY_URI", "kms://projects/demo/keys/test")
    monkeypatch.setattr(
        secrets_provider,
        "_decrypt_with_gcp_kms",
        lambda ciphertext_b64, *, key_uri: f"decoded:{ciphertext_b64}:{key_uri}",
    )
    value = secrets_provider.resolve_secret(plain_env="PLAIN_SECRET", encrypted_env="ENC_SECRET")
    assert value == (
        f"decoded:{base64.b64encode(b'ciphertext').decode('ascii')}:kms://projects/demo/keys/test"
    )


def test_resolve_secret_rejects_unsupported_kms_uri(monkeypatch):
    monkeypatch.delenv("PLAIN_SECRET", raising=False)
    monkeypatch.setenv("ENC_SECRET", base64.b64encode(b"ciphertext").decode("ascii"))
    monkeypatch.setenv("RULFM_FORECASTING_API_KMS_KEY_URI", "aws-kms://secret")
    with pytest.raises(RuntimeError, match="Unsupported KMS provider URI"):
        secrets_provider.resolve_secret(plain_env="PLAIN_SECRET", encrypted_env="ENC_SECRET")


def test_resolve_secret_is_cached(monkeypatch):
    calls = {"count": 0}
    monkeypatch.delenv("PLAIN_SECRET", raising=False)
    monkeypatch.setenv("ENC_SECRET", base64.b64encode(b"ciphertext").decode("ascii"))
    monkeypatch.setenv("RULFM_FORECASTING_API_KMS_KEY_URI", "kms://projects/demo/keys/test")

    def _decrypt(ciphertext_b64: str, *, key_uri: str) -> str:
        calls["count"] += 1
        return "secret-value"

    monkeypatch.setattr(secrets_provider, "_decrypt_with_gcp_kms", _decrypt)
    assert (
        secrets_provider.resolve_secret(plain_env="PLAIN_SECRET", encrypted_env="ENC_SECRET")
        == "secret-value"
    )
    assert (
        secrets_provider.resolve_secret(plain_env="PLAIN_SECRET", encrypted_env="ENC_SECRET")
        == "secret-value"
    )
    assert calls["count"] == 1
