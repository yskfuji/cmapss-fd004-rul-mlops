export function normalizeLang(raw) {
  const s = String(raw || "").trim().toLowerCase();
  if (s === "ja" || s === "jp" || s.startsWith("ja-")) return "ja";
  if (s === "en" || s.startsWith("en-")) return "en";
  return null;
}

function detectDefaultLang() {
  const doc = normalizeLang(document.documentElement.lang);
  if (doc) return doc;
  const nav = normalizeLang(navigator.language);
  return nav || "en";
}

export function getInitialLang({ storageKey }) {
  const fromQuery = normalizeLang(new URLSearchParams(window.location.search).get("lang"));
  if (fromQuery) return fromQuery;

  const fromStorage = normalizeLang(localStorage.getItem(storageKey));
  if (fromStorage) return fromStorage;

  return detectDefaultLang();
}

export function normalizeDensityMode(raw) {
  const s = String(raw || "").trim().toLowerCase();
  if (s === "detailed" || s === "detail" || s === "advanced") return "detailed";
  if (s === "basic" || s === "compact") return "basic";
  return null;
}

export function getInitialDensityMode({ storageKey }) {
  const fromQuery = normalizeDensityMode(new URLSearchParams(window.location.search).get("density"));
  if (fromQuery) return fromQuery;
  const fromStorage = normalizeDensityMode(localStorage.getItem(storageKey));
  if (fromStorage) return fromStorage;
  return "basic";
}