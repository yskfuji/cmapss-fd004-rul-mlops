from __future__ import annotations

import base64
import os
from functools import lru_cache


def _kms_key_uri() -> str | None:
    value = os.getenv("RULFM_FORECASTING_API_KMS_KEY_URI", "").strip()
    return value or None


def _normalize_gcp_key_name(uri: str) -> str:
    value = str(uri).strip()
    if value.startswith("kms://"):
        return value[len("kms://") :]
    if value.startswith("gcp-kms://"):
        return value[len("gcp-kms://") :]
    return value


def _decrypt_with_gcp_kms(ciphertext_b64: str, *, key_uri: str) -> str:
    from google.cloud import kms_v1

    client = kms_v1.KeyManagementServiceClient()
    key_name = _normalize_gcp_key_name(key_uri)
    ciphertext = base64.b64decode(ciphertext_b64)
    response = client.decrypt(request={"name": key_name, "ciphertext": ciphertext})
    plaintext = bytes(response.plaintext)
    return plaintext.decode("utf-8")


@lru_cache(maxsize=8)
def resolve_secret(*, plain_env: str, encrypted_env: str) -> str | None:
    plain_value = os.getenv(plain_env, "")
    if plain_value:
        return plain_value

    ciphertext_b64 = os.getenv(encrypted_env, "").strip()
    key_uri = _kms_key_uri()
    if not ciphertext_b64 or not key_uri:
        return None

    if key_uri.startswith("kms://") or key_uri.startswith("gcp-kms://"):
        return _decrypt_with_gcp_kms(ciphertext_b64, key_uri=key_uri)

    raise RuntimeError(f"Unsupported KMS provider URI: {key_uri}")