"""Test for API key endpoints (both legacy and frontend-compatible styles)."""
import httpx, asyncio

API_KEY = 'mlg_PeNfpwQSOtJkWr1Tow62Kr5luLuEugGi'

async def test_apikeys():
    hdr = {'X-API-Key': API_KEY}
    async with httpx.AsyncClient(timeout=10.0) as c:
        # 1. Test GET /auth/apikeys (Frontend style)
        r1 = await c.get('http://localhost:8000/api/v1/auth/apikeys', headers=hdr)
        print(f"DEBUG: GET /auth/apikeys code={r1.status_code}")
        if r1.status_code == 200:
            keys = r1.json()
            print(f"DEBUG: Found {len(keys)} keys.")
        else:
            print(f"DEBUG: ERROR GET: {r1.text}")

        # 2. Test POST /auth/apikey?label=Test (Frontend style)
        r2 = await c.post('http://localhost:8000/api/v1/auth/apikey?label=IntegrationTestKey', headers=hdr)
        print(f"DEBUG: POST /auth/apikey code={r2.status_code}")
        if r2.status_code == 200:
            data = r2.json()
            print(f"DEBUG: Created Key ID: {data['id']}")
        else:
            print(f"DEBUG: ERROR POST: {r2.text}")

        # 3. Verify it shows up in list
        r3 = await c.get('http://localhost:8000/api/v1/auth/keys', headers=hdr)
        if r3.status_code == 200:
            keys = r3.json()
            found = any(k['label'] == 'IntegrationTestKey' for k in keys)
            print(f"Verification in /auth/keys: {'SUCCESS' if found else 'FAILED'}")

asyncio.run(test_apikeys())
