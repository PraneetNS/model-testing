/**
 * api.ts — ML Guard Centralized API Client
 * All backend requests go through this module.
 * Auth via X-API-Key header is injected automatically.
 */

const API_BASE = "/api/proxy";

export { API_BASE };

// Helper to get CSRF token from cookies
function getCsrfToken(): string {
  if (typeof document === 'undefined') return "";
  const match = document.cookie.match(/csrf_token=([^;]+)/);
  return match ? match[1] : "";
}

export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const isFormData = init.body instanceof FormData;
  const csrfToken = getCsrfToken();

  const headers: Record<string, string> = {
    ...(isFormData ? {} : { "Content-Type": "application/json" }),
    ...(csrfToken ? { "X-CSRF-Token": csrfToken } : {}),
    "X-API-Key": process.env.NEXT_PUBLIC_API_KEY || "",
    ...(init.headers as Record<string, string> | undefined ?? {}),
  };

  // Ensure path starts with / if not present
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  
  // Strip version if calling proxy, or handle it in proxy. 
  // Let's assume the proxy handles /api/... and targets the backend.
  const res = await fetch(`${API_BASE}${normalizedPath}`, {
    ...init,
    headers,
  });

  if (res.status === 401) {
    if (typeof window !== 'undefined') {
       window.location.href = '/login?expired=true';
    }
  }

  return res;
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await apiFetch(path);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`GET ${path} failed ${res.status}: ${err}`);
  }
  if (res.status === 204) return {} as T;
  const text = await res.text();
  return text ? JSON.parse(text) : ({} as T);
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  const res = await apiFetch(path, {
    method: "POST",
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`POST ${path} failed ${res.status}: ${err}`);
  }
  if (res.status === 204) return {} as T;
  const text = await res.text();
  return text ? JSON.parse(text) : ({} as T);
}

export async function apiDelete<T>(path: string): Promise<T> {
  const res = await apiFetch(path, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`DELETE ${path} failed ${res.status}`);
  if (res.status === 204) return {} as T;
  const text = await res.text();
  return text ? JSON.parse(text) : ({} as T);
}

export async function safeJson<T>(res: Response): Promise<T> {
    if (res.status === 204) return {} as T;
    const text = await res.text();
    return text ? JSON.parse(text) : ({} as T);
}

