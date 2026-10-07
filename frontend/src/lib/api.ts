// Keep GUI API calls same-origin so the local browser can reach the backend
// through Vite's development proxy without exposing a second browser origin.
export const API_BASE = "/api";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function fetchWithTimeout(
  input: RequestInfo | URL,
  init: RequestInit | undefined,
  timeoutMs = 10000,
): Promise<Response> {
  const controller = new AbortController();
  const upstreamSignal = init?.signal;
  const abortFromUpstream = () => controller.abort(upstreamSignal?.reason);
  if (upstreamSignal?.aborted) abortFromUpstream();
  else upstreamSignal?.addEventListener("abort", abortFromUpstream, { once: true });
  const timer = setTimeout(() => controller.abort(new Error(`API request timed out after ${timeoutMs}ms`)), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
    upstreamSignal?.removeEventListener("abort", abortFromUpstream);
  }
}

export async function apiGet<T>(path: string, timeoutMs = 10000): Promise<T> {
  const res = await fetchWithTimeout(`${API_BASE}${path}`, undefined, timeoutMs);
  if (!res.ok) {
    const json = await res.json().catch(() => ({}));
    throw new ApiError(
      res.status,
      (json as Record<string, string>)?.detail ?? `GET ${path} ${res.status}`,
    );
  }
  return res.json() as Promise<T>;
}

export async function apiPost<T>(
  path: string,
  body: unknown,
  method: "POST" | "PUT" | "PATCH" | "DELETE" = "POST",
  signal?: AbortSignal,
  timeoutMs = 10000,
): Promise<T> {
  const res = await fetchWithTimeout(`${API_BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  }, timeoutMs);
  const json = (await res.json().catch(() => ({}))) as Record<string, string>;
  if (!res.ok)
    throw new ApiError(
      res.status,
      json?.detail ?? `${method} ${path} ${res.status}`,
    );
  return json as T;
}

export async function apiDelete(path: string): Promise<void> {
  const res = await fetchWithTimeout(`${API_BASE}${path}`, { method: "DELETE" });
  if (!res.ok) {
    const json = await res.json().catch(() => ({}));
    throw new ApiError(res.status, json?.detail ?? `DELETE ${path} ${res.status}`);
  }
}

export async function apiFormPost<T>(
  path: string,
  form: FormData,
): Promise<T> {
  const res = await fetchWithTimeout(`${API_BASE}${path}`, { method: "POST", body: form });
  const json = (await res.json().catch(() => ({}))) as Record<string, string>;
  if (!res.ok)
    throw new ApiError(
      res.status,
      json?.detail ?? `POST ${path} ${res.status}`,
    );
  return json as T;
}
