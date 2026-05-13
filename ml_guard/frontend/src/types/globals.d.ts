/**
 * globals.d.ts — ML Guard Global Type Declarations
 *
 * `safeJson` is defined in src/lib/api.ts but is used across many components
 * without explicit imports. This ambient declaration makes it globally available
 * so TypeScript doesn't report TS2304 "Cannot find name 'safeJson'" errors.
 * It is polyfilled at runtime via the <Script> in src/app/layout.tsx.
 */

declare function safeJson<T = any>(res: Response): Promise<T>;

// Extend Window so TypeScript accepts window.safeJson at runtime
interface Window {
  safeJson<T = any>(res: Response): Promise<T>;
}

// Override Response.json() to return any instead of unknown
// This fixes TS18046 "data is of type unknown" across all components
// that use pattern: const data = await res.json()
interface Response {
  json(): Promise<any>;
}

