from __future__ import annotations

import copy
from typing import Any, cast

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse


def normalize_lang(raw: str | None) -> str | None:
    if not raw:
        return None
    token = raw.split(",")[0].strip().lower()
    if token.startswith("ja") or token.startswith("jp"):
        return "ja"
    if token.startswith("en"):
        return "en"
    return None


def filter_lang_text(text: str, lang: str | None) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[EN]"):
            if lang in (None, "en"):
                kept.append(stripped.replace("[EN]", "", 1).lstrip())
            continue
        if stripped.startswith("[JA]"):
            if lang in (None, "ja"):
                kept.append(stripped.replace("[JA]", "", 1).lstrip())
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def filter_lang_obj(obj: Any, lang: str | None) -> Any:
    if isinstance(obj, str):
        return filter_lang_text(obj, lang)
    if isinstance(obj, list):
        return [filter_lang_obj(item, lang) for item in obj]
    if isinstance(obj, dict):
        return {key: filter_lang_obj(value, lang) for key, value in obj.items()}
    return obj


def openapi_base(app: FastAPI) -> dict[str, Any]:
    cached = getattr(app.state, "_openapi_base", None)
    if isinstance(cached, dict):
        return cast(dict[str, Any], cached)
    schema = cast(
        dict[str, Any],
        get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        ),
    )
    app.state._openapi_base = schema
    return schema


def openapi_with_lang(app: FastAPI, lang: str | None) -> dict[str, Any]:
    return cast(dict[str, Any], filter_lang_obj(copy.deepcopy(openapi_base(app)), lang))


def default_openapi(app: FastAPI) -> dict[str, Any]:
    cached = app.openapi_schema
    if isinstance(cached, dict):
        return cast(dict[str, Any], cached)
    localized = cast(dict[str, Any], filter_lang_obj(copy.deepcopy(openapi_base(app)), None))
    app.openapi_schema = localized
    return localized


def render_swagger_ui(app: FastAPI, lang: str) -> HTMLResponse:
    openapi_url = "/openapi.en.json" if lang == "en" else "/openapi.ja.json"
    title = f"{app.title} Docs ({lang.upper()})"
    response = get_swagger_ui_html(
        openapi_url=openapi_url,
        title=title,
        swagger_ui_parameters={
            "docExpansion": "none",
            "supportedSubmitMethods": [],
            "tryItOutEnabled": False,
        },
    )
    body = bytes(response.body).decode("utf-8")
    banner = """
  <style>
    .lang-banner {
      position: sticky;
      top: 0;
      z-index: 1000;
      display: flex;
      gap: 16px;
      align-items: center;
      justify-content: space-between;
      padding: 12px 18px;
      border-bottom: 1px solid #dde2e7;
            background: linear-gradient(
                90deg,
                rgba(0, 102, 255, 0.08),
                rgba(110, 231, 255, 0.08),
                #ffffff
            );
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      font-size: 13px;
    }
    .lang-banner a {
      color: #0066ff;
      text-decoration: none;
      font-weight: 600;
    }
    .lang-banner .muted {
      color: #5f6b7a;
      font-weight: 400;
    }
    .lang-badge {
      display: inline-flex;
      align-items: center;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid rgba(0, 102, 255, 0.3);
      background: rgba(0, 102, 255, 0.08);
      font-size: 12px;
      font-weight: 600;
      color: #0b0c10;
    }
  </style>
  <div class="lang-banner">
        <div class="muted">
            Language: <a href="/docs/en">English</a> /
            <a href="/docs/ja">日本語</a>
        </div>
    <div class="lang-badge">Docs language</div>
    <div><a href="/docs">Language selector</a></div>
  </div>
"""
    body = body.replace("<body>", f"<body>{banner}", 1)
    return HTMLResponse(body)