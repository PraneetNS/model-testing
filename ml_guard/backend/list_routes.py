import httpx, asyncio, json

async def main():
    r = await httpx.AsyncClient().get('http://localhost:8000/openapi.json')
    paths = sorted(r.json()['paths'].keys())
    for p in paths:
        print(p)

asyncio.run(main())
