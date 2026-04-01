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
