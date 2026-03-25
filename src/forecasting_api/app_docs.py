from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import openapi_localization


@dataclass(frozen=True)
class ApiResponseDocs:
    request_id_header: dict[str, Any]
    responses_400: dict[int | str, Any]
    responses_401: dict[int | str, Any]
    responses_403: dict[int | str, Any]
    responses_404: dict[int | str, Any]
    responses_409: dict[int | str, Any]
    responses_413: dict[int | str, Any]
    responses_429: dict[int | str, Any]


@dataclass(frozen=True)
class DocsRouteBindings:
    docs_root: Callable[..., Any]
    docs_en: Callable[..., Any]
    docs_ja: Callable[..., Any]
    openapi_en: Callable[..., Any]
    openapi_ja: Callable[..., Any]


def build_api_response_docs(
    *,
    bi: Callable[[str, str], str],
    error_response_model: Any,
) -> ApiResponseDocs:
    request_id_header: dict[str, Any] = {
        "X-Request-Id": {
            "description": bi(
                "Request ID for support correlation.",
                "サポート照合用のリクエストID。",
            ),
            "schema": {"type": "string"},
        }
    }
    return ApiResponseDocs(
        request_id_header=request_id_header,
        responses_400={
            400: {
                "model": error_response_model,
                "description": bi(
                    "Validation/contract error (e.g., V01, V02, V03, V04, V05, COST01, A14).",
                    "検証/契約エラー（例: V01, V02, V03, V04, V05, COST01, A14）。",
                ),
                "headers": request_id_header,
            }
        },
        responses_401={
            401: {
                "model": error_response_model,
                "description": bi("Authentication failed.", "認証失敗。"),
                "headers": request_id_header,
            }
        },
        responses_403={
            403: {
                "model": error_response_model,
                "description": bi(
                    "Policy enforcement denied the request.",
                    "tenant / network policy により拒否されました。",
                ),
                "headers": request_id_header,
            }
        },
        responses_404={
            404: {
                "model": error_response_model,
                "description": bi("Not found (error_code=J01).", "未検出（error_code=J01）。"),
                "headers": request_id_header,
            }
        },
        responses_409={
            409: {
                "model": error_response_model,
                "description": bi(
                    "Job not completed (error_code=J02).",
                    "ジョブ未完了（error_code=J02）。",
                ),
                "headers": request_id_header,
            }
        },
        responses_413={
            413: {
                "model": error_response_model,
                "description": bi(
                    "Payload too large (error_code=S01).",
                    "payloadが大きすぎます（error_code=S01）。",
                ),
                "headers": request_id_header,
            }
        },
        responses_429={
            429: {
                "model": error_response_model,
                "description": bi("Rate/quota exceeded.", "レート/クォータ超過。"),
                "headers": request_id_header,
            }
        },
    )


def configure_openapi_and_docs(app: FastAPI) -> DocsRouteBindings:
    def _openapi_default() -> dict[str, Any]:
        return openapi_localization.default_openapi(app)

    app.openapi = _openapi_default  # type: ignore[assignment]

    def _render_swagger_ui(lang: str) -> HTMLResponse:
        return openapi_localization.render_swagger_ui(app, lang)

    @app.get("/docs", include_in_schema=False)
    async def docs_root(request: Request) -> HTMLResponse:
        preferred = openapi_localization.normalize_lang(
            request.query_params.get("lang")
        ) or openapi_localization.normalize_lang(request.headers.get("accept-language"))
        hint = "日本語" if preferred == "ja" else "English"
        html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{app.title} Docs</title>
    <style>
      body {{
        font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Helvetica, Arial, sans-serif;
        margin: 0;
        background: #f7f8fb;
        color: #0b0c10;
      }}
      .wrap {{
        max-width: 720px;
        margin: 60px auto;
        padding: 0 20px;
      }}
      .card {{
        background: #fff;
        border: 1px solid #e2e2e2;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.08);
      }}
      .actions {{
        display: flex;
        gap: 12px;
        margin-top: 16px;
        flex-wrap: wrap;
      }}
      a.button {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 10px 14px;
        border-radius: 10px;
        text-decoration: none;
        font-weight: 600;
        border: 1px solid #0066ff;
        color: #0066ff;
      }}
      .muted {{
        color: #5f6b7a;
        font-size: 13px;
      }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="card">
        <h1>{app.title} API Docs</h1>
                <p class="muted">
                    Swagger UI has no built-in language toggle. Select a language to view
                    localized descriptions.
                </p>
        <p class="muted">推奨言語: {hint}</p>
        <div class="actions">
          <a class="button" href="/docs/en">English</a>
          <a class="button" href="/docs/ja">日本語</a>
        </div>
      </div>
    </div>
  </body>
</html>"""
        return HTMLResponse(html)

    @app.get("/docs/en", include_in_schema=False)
    async def docs_en() -> HTMLResponse:
        return _render_swagger_ui("en")

    @app.get("/docs/ja", include_in_schema=False)
    async def docs_ja() -> HTMLResponse:
        return _render_swagger_ui("ja")

    @app.get("/openapi.en.json", include_in_schema=False)
    async def openapi_en() -> JSONResponse:
        return JSONResponse(openapi_localization.openapi_with_lang(app, "en"))

    @app.get("/openapi.ja.json", include_in_schema=False)
    async def openapi_ja() -> JSONResponse:
        return JSONResponse(openapi_localization.openapi_with_lang(app, "ja"))

    return DocsRouteBindings(
        docs_root=docs_root,
        docs_en=docs_en,
        docs_ja=docs_ja,
        openapi_en=openapi_en,
        openapi_ja=openapi_ja,
    )


def mount_static_gui(app: FastAPI, *, static_gui_dir: Path) -> Callable[..., Any] | None:
    if not static_gui_dir.is_dir():
        return None

    app.mount(
        "/ui/forecasting",
        StaticFiles(directory=str(static_gui_dir), html=True),
        name="forecasting_gui",
    )

    @app.get("/", include_in_schema=False)
    async def root_redirect_to_gui() -> RedirectResponse:
        return RedirectResponse(url="/ui/forecasting/", status_code=307)

    return root_redirect_to_gui