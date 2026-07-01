"""
Kalshi RSA-PSS authentication helpers.

Kalshi signs with: RSA-PSS, SHA-256, salt_length=MAX_LENGTH
Message format:    <timestamp_ms><METHOD><path>
e.g.              1719619200000GET/trade-api/rest/v2/portfolio/balance
"""
import base64
import os
import time
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend


def _load_private_key():
    raw = os.environ["KALSHI_PRIVATE_KEY"]
    # Handle \n stored as literal backslash-n, or as spaces
    pem_str = raw.replace("\\n", "\n")
    # If the PEM header/footer lines got spaces instead of newlines, fix them
    if "-----BEGIN" in pem_str and "\n" not in pem_str:
        # all on one line — split on "-----" boundaries
        pem_str = pem_str.replace("-----BEGIN RSA PRIVATE KEY----- ", "-----BEGIN RSA PRIVATE KEY-----\n")
        pem_str = pem_str.replace(" -----END RSA PRIVATE KEY-----", "\n-----END RSA PRIVATE KEY-----")
        pem_str = pem_str.replace("-----BEGIN PRIVATE KEY----- ", "-----BEGIN PRIVATE KEY-----\n")
        pem_str = pem_str.replace(" -----END PRIVATE KEY-----", "\n-----END PRIVATE KEY-----")
    pem = pem_str.encode()
    return serialization.load_pem_private_key(pem, password=None, backend=default_backend())


def _get_timestamp_ms() -> str:
    return str(int(time.time() * 1000))


def build_auth_headers(method: str, path: str) -> dict:
    """
    Returns the three Kalshi auth headers for a request.
    path should be the full URL path, e.g. '/trade-api/rest/v2/portfolio/balance'
    """
    key_id = os.environ["KALSHI_API_KEY_ID"]
    ts = _get_timestamp_ms()

    message = ts + method.upper() + path
    private_key = _load_private_key()
    signature = private_key.sign(
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    sig_b64 = base64.b64encode(signature).decode("utf-8")

    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": sig_b64,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "Content-Type": "application/json",
    }
