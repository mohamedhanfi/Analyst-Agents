"""At-rest encryption for raw uploads (§5 / audit item G).

Raw user data is the most sensitive artifact of a run. When
`retention.encrypt_at_rest` is true (default), the app stores uploads as
Fernet ciphertext instead of plaintext. A working plaintext copy exists
only while a run executes and is deleted afterwards.

Key management (local-first, no KMS):
- `INSIGHT_FORGE_ENC_KEY` env var (base64 32-byte key) wins when set;
- otherwise a fresh key is generated on first use and persisted to
  `.enc_key` next to the repository root (gitignored).

Fernet (AES-128-CBC + HMAC-SHA256) is stdlib-adjacent via `cryptography`.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
KEY_FILE = ROOT / ".enc_key"
ENV_KEY = "INSIGHT_FORGE_ENC_KEY"


def _load_or_create_key() -> bytes:
    env = os.environ.get(ENV_KEY)
    if env:
        return base64.urlsafe_b64decode(env.encode("ascii"))
    if KEY_FILE.is_file():
        return KEY_FILE.read_bytes().strip()
    key = base64.urlsafe_b64encode(os.urandom(32))
    KEY_FILE.write_bytes(key)
    try:
        import stat
        os.chmod(KEY_FILE, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass  # best-effort on platforms without chmod
    return key


def get_fernet():
    from cryptography.fernet import Fernet
    return Fernet(_load_or_create_key())


def encrypt_file(src: str | Path, dst: str | Path) -> None:
    """Encrypt src bytes into a single Fernet token at dst."""
    f = get_fernet()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = Path(src).read_bytes()
    dst.write_bytes(f.encrypt(data))
    Path(src).unlink(missing_ok=True)


def decrypt_file(src: str | Path, dst: str | Path) -> None:
    """Decrypt a Fernet token file into plaintext at dst. The ciphertext
    at src is left in place (uploads stay encrypted at rest)."""
    f = get_fernet()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = f.decrypt(Path(src).read_bytes())
    dst.write_bytes(data)


def encrypt_bytes(data: bytes) -> bytes:
    from cryptography.fernet import Fernet
    return get_fernet().encrypt(data)


def decrypt_bytes(token: bytes) -> bytes:
    from cryptography.fernet import Fernet
    return get_fernet().decrypt(token)


def encryption_enabled(cfg: Optional[dict] = None) -> bool:
    """True when retention.encrypt_at_rest is true (default: true)."""
    if cfg is None:
        return True
    return bool(cfg.get("retention", {}).get("encrypt_at_rest", True))