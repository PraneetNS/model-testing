import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  const nonce = Buffer.from(crypto.randomUUID()).toString('base64');
  
  // CSP Header with nonce
  const cspHeader = `
    default-src 'self';
    script-src 'self' 'nonce-${nonce}' 'strict-dynamic' https://apis.google.com https://www.gstatic.com;
    style-src 'self' 'unsafe-inline';
    img-src 'self' data: https://*.googleusercontent.com https://*.githubusercontent.com;
    connect-src 'self' http://localhost:8000 https://*.googleapis.com https://*.firebaseapp.com https://*.firebaseio.com;
    frame-src 'self' https://*.firebaseapp.com;
    frame-ancestors 'none';
    base-uri 'self';
    form-action 'self';
  `.replace(/\s{2,}/g, ' ').trim();

  const requestHeaders = new Headers(request.headers);
  requestHeaders.set('x-nonce', nonce);
  requestHeaders.set('Content-Security-Policy', cspHeader);

  const response = NextResponse.next({
    request: {
      headers: requestHeaders,
    },
  });

  response.headers.set('Content-Security-Policy', cspHeader);

  // CSRF Protection for state-changing internal API routes
  const isApiAction = ['POST', 'PUT', 'DELETE', 'PATCH'].includes(request.method) && request.nextUrl.pathname.startsWith('/api/');
  const isSessionInit = request.nextUrl.pathname === '/api/auth/session';

  if (isApiAction && !isSessionInit) {
    const csrfToken = request.cookies.get('csrf_token')?.value;
    const headerCsrfToken = request.headers.get('x-csrf-token');

    if (!csrfToken || csrfToken !== headerCsrfToken) {
      return new NextResponse(
        JSON.stringify({ error: 'CSRF token mismatch or missing' }),
        { status: 403, headers: { 'content-type': 'application/json' } }
      );
    }
  }

  // Session Management: Auto logout on 401 should be handled client-side or in API wrapper,
  // but we can ensure session exists here for protected routes.
  const sessionToken = request.cookies.get('session_token');
  if (request.nextUrl.pathname.startsWith('/dashboard') && !sessionToken) {
     return NextResponse.redirect(new URL('/login', request.url));
  }

  return response;
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (api routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    {
      source: '/((?!_next/static|_next/image|favicon.ico).*)',
      missing: [
        { type: 'header', key: 'next-router-prefetch' },
        { type: 'header', key: 'purpose', value: 'prefetch' },
      ],
    },
  ],
};
