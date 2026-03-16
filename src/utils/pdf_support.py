from __future__ import annotations

import warnings
from functools import lru_cache


@lru_cache(maxsize=1)
def get_pdf_reader():
    # Suppress the known ARC4 deprecation warning emitted by pypdf's crypto backend.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"ARC4 has been moved to cryptography\.hazmat\.decrepit\.ciphers\.algorithms\.ARC4.*",
            category=Warning,
            module=r"pypdf\._crypt_providers\._cryptography",
        )
        from pypdf import PdfReader

    return PdfReader
