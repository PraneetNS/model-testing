/**
 * api.ts — ML Guard Centralized API Client
 * All backend requests go through this module.
 * Auth via X-API-Key header is injected automatically.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";

export { API_BASE };

export const apiHeaders: Record<string, string> = {
  "Content-Type": "application/json",
  "X-API-Key": API_KEY,
};

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: apiHeaders,
    cache: "no-store",
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`GET ${path} failed ${res.status}: ${err}`);
  }
  return res.json();
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: apiHeaders,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`POST ${path} failed ${res.status}: ${err}`);
  }
  return res.json();
}

export async function apiDelete<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "DELETE",
    headers: apiHeaders,
  });
  if (!res.ok) throw new Error(`DELETE ${path} failed ${res.status}`);
  return res.json();
}

/**
 * apiFetch — drop-in authenticated replacement for bare fetch().
 *
 * Usage (replaces):  fetch(`${API_BASE}/api/v1/...`, options)
 * With:              apiFetch(`/api/v1/...`, options)
 *
 * - Always injects X-API-Key header
 * - Skips Content-Type for FormData (browser sets multipart boundary)
 * - Returns the raw Response so callers can call .json(), .text(), etc.
 */
export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const isFormData = init.body instanceof FormData;

  const headers: Record<string, string> = {
    "X-API-Key": API_KEY,
    ...(isFormData ? {} : { "Content-Type": "application/json" }),
    ...(init.headers as Record<string, string> | undefined ?? {}),
  };

  return fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
  });
}

