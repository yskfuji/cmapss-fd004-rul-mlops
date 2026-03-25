function bytesToBase64Url(bytes) {
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replaceAll("+", "-").replaceAll("/", "_").replaceAll(/=+$/g, "");
}

function base64UrlToBytes(b64url) {
  const s = String(b64url || "")
    .replaceAll("-", "+")
    .replaceAll("_", "/");
  const padded = s + "===".slice((s.length + 3) % 4);
  const bin = atob(padded);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export function encodeStateToHash(obj) {
  const json = JSON.stringify(obj || {});
  const bytes = new TextEncoder().encode(json);
  const encoded = bytesToBase64Url(bytes);
  return `state=${encoded}`;
}

export function decodeStateFromHash(hash) {
  const raw = String(hash || "").replace(/^#/, "");
  if (!raw) return null;
  const params = new URLSearchParams(raw);
  const s = params.get("state");
  if (!s) return null;
  try {
    const bytes = base64UrlToBytes(s);
    const json = new TextDecoder().decode(bytes);
    const parsed = JSON.parse(json);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}