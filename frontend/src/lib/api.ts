export const API_BASE = "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
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
  method: "POST" | "PUT" | "PATCH" = "POST",
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const json = (await res.json().catch(() => ({}))) as Record<string, string>;
  if (!res.ok)
    throw new ApiError(
      res.status,
      json?.detail ?? `${method} ${path} ${res.status}`,
    );
  return json as T;
}

export async function apiDelete(path: string): Promise<void> {
  const res = await fetch(`${API_BASE}${path}`, { method: "DELETE" });
  if (!res.ok)
    throw new ApiError(res.status, `DELETE ${path} ${res.status}`);
}

export async function apiFormPost<T>(
  path: string,
  form: FormData,
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { method: "POST", body: form });
  const json = (await res.json().catch(() => ({}))) as Record<string, string>;
  if (!res.ok)
    throw new ApiError(
      res.status,
      json?.detail ?? `POST ${path} ${res.status}`,
    );
  return json as T;
}
