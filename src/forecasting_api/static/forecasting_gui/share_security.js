const SHARE_STATE_ALLOWED_KEYS = new Set([
  "lang",
  "density",
  "mode",
  "task",
  "horizon",
  "frequency",
  "missing_policy",
  "quantiles",
  "level",
  "model_id",
  "folds",
  "metric",
  "base_model",
  "train_algo",
  "model_name",
  "training_hours",
]);

const SENSITIVE_KEY_RE = /(api[_-]?key|secret|token|authorization|password|passwd|cookie|session|bearer|x[-_]?api[-_]?key)/i;
const SENSITIVE_VALUE_RE = /(bearer\s+[A-Za-z0-9._\-+=/]+|-----BEGIN\s+[A-Z ]+PRIVATE KEY-----|sk-[A-Za-z0-9]{16,}|x-api-key)/i;

function safePath(base, key) {
  return base ? `${base}.${String(key)}` : String(key);
}

export function findSensitiveInObject(obj, base = "") {
  if (obj === null || obj === undefined) return null;
  if (typeof obj === "string") {
    if (SENSITIVE_VALUE_RE.test(obj)) return { path: base || "value" };
    return null;
  }
  if (Array.isArray(obj)) {
    for (let i = 0; i < obj.length; i++) {
      const hit = findSensitiveInObject(obj[i], `${base}[${i}]`);
      if (hit) return hit;
    }
    return null;
  }
  if (typeof obj === "object") {
    for (const [key, value] of Object.entries(obj)) {
      const path = safePath(base, key);
      if (SENSITIVE_KEY_RE.test(key)) return { path };
      const hit = findSensitiveInObject(value, path);
      if (hit) return hit;
    }
  }
  return null;
}

export function sanitizeShareState(state) {
  const src = state && typeof state === "object" ? state : {};
  const out = {};
  for (const key of SHARE_STATE_ALLOWED_KEYS) {
    if (!(key in src)) continue;
    const v = src[key];
    if (v === null || v === undefined || v === "") {
      out[key] = null;
      continue;
    }
    if (typeof v === "number") {
      out[key] = Number.isFinite(v) ? v : null;
      continue;
    }
    out[key] = String(v);
  }
  return out;
}

export function sanitizeRecordsForSnippet(records, max = 20) {
  const arr = Array.isArray(records) ? records : [];
  const trimmed = arr.slice(0, max);
  return trimmed.map((rec) => {
    const out = {
      series_id: String(rec?.series_id ?? ""),
      timestamp: String(rec?.timestamp ?? ""),
      y: rec?.y,
    };
    if (rec?.x && typeof rec.x === "object" && !Array.isArray(rec.x)) {
      const xSafe = {};
      for (const [k, v] of Object.entries(rec.x)) {
        if (SENSITIVE_KEY_RE.test(k)) continue;
        xSafe[k] = v;
      }
      if (Object.keys(xSafe).length > 0) out.x = xSafe;
    }
    return out;
  });
}