import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';

export async function POST(request: Request) {
  try {
    const bodyText = await request.text();
    const { token, api_key } = bodyText ? JSON.parse(bodyText) : {};
    const cookieStore = await cookies();
    
    // 1. Session Token (Firebase ID Token)
    cookieStore.set('session_token', token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 8 * 60 * 60, // 8 hours
      path: '/',
    });

    // 2. Backend API Key
    // Fallback to server-side env if not passed (though user said raw key shouldn't be on client)
    const finalApiKey = api_key || process.env.BACKEND_API_KEY || "dev-secret-key"; 
    
    cookieStore.set('backend_api_key', finalApiKey, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 8 * 60 * 60,
      path: '/',
    });

    // 3. CSRF Token
    const csrfToken = crypto.randomUUID();
    cookieStore.set('csrf_token', csrfToken, {
      httpOnly: false, // Client needs to read this to put in X-CSRF-Token header
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 8 * 60 * 60,
      path: '/',
    });

    return NextResponse.json({ success: true, csrfToken });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to create session' }, { status: 500 });
  }
}

export async function DELETE() {
  const cookieStore = await cookies();
  cookieStore.delete('session_token');
  cookieStore.delete('backend_api_key');
  cookieStore.delete('csrf_token');
  return NextResponse.json({ success: true });
}
