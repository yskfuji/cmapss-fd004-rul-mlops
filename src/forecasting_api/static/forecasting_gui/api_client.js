function buildHeaders(apiKey) {
  const h = { "Content-Type": "application/json" };
  const key = String(apiKey || "").trim();
  if (key) h["X-API-Key"] = key;
  return h;
}

function normalizeBaseUrl(raw) {
  const base = String(raw || "").trim();
  if (!base) return "";
  return base.replace(/\/+$/, "");
}

function joinUrl(baseUrl, path) {
  const base = normalizeBaseUrl(baseUrl);
  const p = String(path || "").trim();
  if (!base) return p;
  if (!p) return base;
  if (p.startsWith("/")) return `${base}${p}`;
  return `${base}/${p}`;
}

async function parseJsonSafe(resp) {
  return resp.json().catch(() => null);
}

function formatApiErrorMessage(status, body, requestIdHeader, { errorCodeToI18nKey, t } = {}) {
  const code = body?.error_code || body?.error?.error_code || "UNKNOWN";
  const msg = body?.message || body?.error?.message || "";
  const rid = body?.request_id || body?.error?.request_id || requestIdHeader || "";
  const details = body?.details || body?.error?.details;
  const detailsText = details ? JSON.stringify(details, null, 2) : "";
  const i18nKey = typeof errorCodeToI18nKey === "function" ? errorCodeToI18nKey(code) : null;
  const i18nMsg = i18nKey && typeof t === "function" ? t(i18nKey) : "";
  return [`HTTP ${status}`, `error_code=${code}`, msg ? `message=${msg}` : null, rid ? `request_id=${rid}` : null, detailsText]
    .concat(i18nMsg ? [`i18n=${i18nKey}`, `message_i18n=${i18nMsg}`] : [])
    .filter((x) => x && String(x).trim().length > 0)
    .join("\n");
}

export function createApiClient({ getApiKey, getApiBaseUrl, errorCodeToI18nKey, t } = {}) {
  const request = async (path, { method = "GET", body = null, signal, cache } = {}) => {
    const baseUrl = typeof getApiBaseUrl === "function" ? getApiBaseUrl() : "";
    const url = joinUrl(baseUrl, path);
    const resp = await fetch(url, {
      method,
      headers: buildHeaders(typeof getApiKey === "function" ? getApiKey() : ""),
      body: body == null ? undefined : JSON.stringify(body),
      signal,
      cache,
    });
    const requestId = resp.headers.get("x-request-id");
    const parsedBody = await parseJsonSafe(resp);
    if (!resp.ok) {
      throw new Error(formatApiErrorMessage(resp.status, parsedBody, requestId, { errorCodeToI18nKey, t }));
    }
    return { body: parsedBody, requestId };
  };

  return {
    getCmapssFd004Sample(profile, { signal } = {}) {
      return request(`/v1/cmapss/fd004/sample?profile=${encodeURIComponent(String(profile || ""))}`, { method: "GET", signal, cache: "no-store" });
    },
    getCmapssFd004Benchmarks({ signal } = {}) {
      return request("/v1/cmapss/fd004/benchmarks", { method: "GET", signal, cache: "no-store" });
    },
    listModels({ signal } = {}) {
      return request("/v1/models", { method: "GET", signal });
    },
    getDriftBaselineStatus({ signal } = {}) {
      return request("/v1/monitoring/drift/baseline/status", {
        method: "GET",
        signal,
        cache: "no-store",
      });
    },
    persistDriftBaseline(payload, { signal } = {}) {
      return request("/v1/monitoring/drift/baseline", {
        method: "POST",
        body: payload,
        signal,
      });
    },
    checkHealth({ signal } = {}) {
      return request("/health", { method: "GET", signal, cache: "no-store" });
    },
    runTask(task, payload, { signal } = {}) {
      const path =
        task === "drift"
          ? "/v1/monitoring/drift/report"
          : task === "backtest"
            ? "/v1/backtest"
            : task === "train"
              ? "/v1/train"
              : "/v1/forecast";
      return request(path, { method: "POST", body: payload, signal });
    },
    createJob(task, payload, { signal } = {}) {
      return request("/v1/jobs", {
        method: "POST",
        body: { type: task, payload },
        signal,
      });
    },
    getJob(jobId, { signal } = {}) {
      return request(`/v1/jobs/${encodeURIComponent(String(jobId || ""))}`, { method: "GET", signal });
    },
    getJobResult(jobId, { signal } = {}) {
      return request(`/v1/jobs/${encodeURIComponent(String(jobId || ""))}/result`, { method: "GET", signal });
    },
  };
}