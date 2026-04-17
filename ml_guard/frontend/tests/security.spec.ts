import { test, expect } from '@playwright/test';

test.describe('Security Hardening Checks', () => {
  
  test('Security headers should be present', async ({ request }) => {
    const response = await request.get('/');
    const headers = response.headers();

    expect(headers['content-security-policy']).toBeDefined();
    expect(headers['x-frame-options']).toBe('DENY');
    expect(headers['x-content-type-options']).toBe('nosniff');
    expect(headers['referrer-policy']).toBe('strict-origin-when-cross-origin');
  });

  test('LocalStorage should not contain sensitive tokens', async ({ page }) => {
    await page.goto('/login');
    // Assuming login logic is complex, we check if generic keys are missing or cleaned
    const mlGuardSession = await page.evaluate(() => localStorage.getItem('mlguard_session'));
    const apiKey = await page.evaluate(() => localStorage.getItem('NEXT_PUBLIC_API_KEY'));
    
    expect(mlGuardSession).toBeNull();
    expect(apiKey).toBeNull();
  });

  test('POST requests should fail without a CSRF token', async ({ request }) => {
    // Attempt a POST to an internal API without the X-CSRF-Token header
    const response = await request.post('/api/auth/session', {
      data: { token: 'test' },
      headers: {
        'Content-Type': 'application/json',
        // Intentional skip of X-CSRF-Token
      }
    });

    // Our middleware should return 403
    expect(response.status()).toBe(403);
    const body = await response.json();
    expect(body.error).toContain('CSRF token mismatch');
  });

});
