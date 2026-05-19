import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

async function handleRequest(request: Request, props: { params: Promise<{ path: string[] }> }) {
  const params = await props.params;
  const fullPath = params.path.join('/');
  const searchParams = new URL(request.url).search;
  
  // Handle paths that already include 'api' to avoid doubling /api/api
  const normalizedPath = fullPath.startsWith('api') ? fullPath : `api/${fullPath}`;
  const url = `${BACKEND_URL}/${normalizedPath}${searchParams}`;
  
  const cookieStore = await cookies();
  const apiKey = cookieStore.get('backend_api_key')?.value
    || process.env.BACKEND_API_KEY
    || 'dev-secret-key';
  const sessionToken = cookieStore.get('session_token')?.value;

  if (!apiKey) {
    return NextResponse.json({ error: 'Unauthorized: No API key available' }, { status: 401 });
  }

  const headers = new Headers(request.headers);
  headers.set('X-API-Key', apiKey);
  if (sessionToken) {
    headers.set('Authorization', `Bearer ${sessionToken}`);
  }
  // Remove host header to avoid issues with proxying
  headers.delete('host');

  try {
    const body = request.method !== 'GET' && request.method !== 'HEAD' 
      ? await request.arrayBuffer() 
      : undefined;

    const response = await fetch(url, {
      method: request.method,
      headers: headers,
      body: body,
      cache: 'no-store',
    });

    console.log(`Proxy: ${request.method} ${url} -> ${response.status}`);

    if (response.status === 401) {
      // Clear session cookies if backend says unauthorized
      console.warn("Proxy: 401 Unauthorized from backend.");
      const res = NextResponse.json({ error: 'Session expired' }, { status: 401 });
      res.cookies.delete('session_token');
      res.cookies.delete('backend_api_key');
      res.cookies.delete('csrf_token');
      return res;
    }

    const data = await response.arrayBuffer();
    return new NextResponse(data, {
      status: response.status,
      headers: {
        'Content-Type': response.headers.get('Content-Type') || 'application/json',
      },
    });
  } catch (error) {
    console.error(`Proxy error for ${url}:`, error);
    return NextResponse.json({ error: 'Backend unreachable' }, { status: 502 });
  }
}

export const GET = handleRequest;
export const POST = handleRequest;
export const PUT = handleRequest;
export const DELETE = handleRequest;
export const PATCH = handleRequest;
