from __future__ import annotations

from collections.abc import Callable
from typing import Any, NoReturn


def raising_callable(exc: Exception) -> Callable[..., NoReturn]:
    def _raiser(*args: Any, **kwargs: Any) -> NoReturn:
        # Rebuild the exception to avoid reusing a traceback-bearing instance across
        # calls. from None also avoids attaching unrelated exception context when
        # this helper is invoked while another failure is already being handled.
        raise type(exc)(*exc.args) from None

    return _raiser